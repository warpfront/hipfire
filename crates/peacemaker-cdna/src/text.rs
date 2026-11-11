// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Text assembler for the typed CDNA instruction model: one instruction line
//! in, one [`Inst`] out ([`parse_line`]).
//!
//! The parser owns no encoding knowledge. It resolves the mnemonic through the
//! opcode tables the encoders already own (`*Op::ALL`, `name()`, `widths()`),
//! builds the typed [`Inst`] variant, and validates it by running the real
//! encoder ([`Inst::words`]), so every legality rule (register alignment,
//! constant-bus limit, modifier support, literal placement) is the encoder's
//! and a line that parses encodes to the bytes `llvm-mc -mcpu=gfx942` produces
//! for it. The text grammar is two things at once:
//!
//! - the canonical spelling every format's `Display` prints (LLVM's text).
//!   LLVM omits the encoding suffix of VOP3 NOP and some ignored DS swizzle
//!   offset bits: use [`Inst::assembly`] for an exact typed round trip; bare
//!   `v_nop` chooses the 32-bit VOP1 form, and
//! - the common LLVM assembler spellings: `_e32` / `_e64` / `_dpp` / `_sdwa`
//!   suffixes or none (the 32-bit form is tried first, the 64-bit VOP3 form
//!   when the operands or modifiers need it), `neg(x)` / `-x` / `abs(x)` /
//!   `|x|` / `sext(x)` source modifiers, `lit(0x…)`, decimal and hex integers,
//!   decimal floats, `v[0:3]` / `a[0:3]` / `s[2:3]` / `ttmp[4:5]` ranges,
//!   `s_waitcnt vmcnt(0) & lgkmcnt(0)` (and `,`/space separators or a plain
//!   integer), `hwreg(...)`, `sendmsg(...)`, `offset:swizzle(...)`, DPP and
//!   SDWA control words, `op_sel:[..]` style lists, and the MFMA mnemonic
//!   aliases `v_mfma_f32_16x16x16bf16_1k` / `v_mfma_f32_16x16x4f32` / ...
//!
//! Constants follow llvm's folding rules per operand type: an integer token in
//! `-16..=64` is the inline integer, a 32-bit (or 16-bit) integer token whose
//! bit pattern equals an inline float constant is that inline float, a float
//! token equal to one of the nine inline floats is that constant, anything else
//! becomes a literal dword (`lit(...)` forces the literal encoding).
//!
//! Branch targets must be numeric dword offsets; symbolic labels are the
//! caller's concern and are reported as a `label operand` error.
//!
//! Errors are strings whose prefix names the failure: `unknown mnemonic`,
//! `operand count`, `bad operand`, `unsupported modifier`, `duplicate
//! modifier`, `label operand`, `unsupported operand`, `encode` (the encoder's
//! own legality message follows), `not an instruction`, `empty instruction`.
//! When a mnemonic has several encodings (`v_add_f32` is VOP2 and VOP3) and
//! none accepts the line, every attempt's reason is listed.
use crate::arch::Arch;
use crate::enc::ds::{Ds, DsOp};
use crate::enc::flat::{Flat, FlatOp, FlatSeg};
use crate::enc::mubuf::{Mubuf, MubufOp};
use crate::enc::smem::{Smem, SmemOffset, SmemOp};
use crate::enc::sop::{
    hwreg_name, waitcnt_imm, Sop1, Sop1Op, Sop2, Sop2Op, Sopc, SopcOp, Sopk, SopkOp, Sopp, SoppOp,
};
use crate::enc::vop::{
    Dpp, DppCtrl, Sdwa, SdwaSel, SdwaUnused, Vop1, Vop1Op, Vop2, Vop2K, Vop2KOp, Vop2Op, VopExt,
    Vopc, VopcOp,
};
use crate::enc::vop3::{Vop3, Vop3Op};
use crate::enc::vop3p::{Mai, MaiOp, Vop3p, Vop3pOp};
use crate::float::f32_to_f16_rne;
use crate::inst::Inst;
use crate::operand::{InlineFloat, Src};
use crate::Result;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Parse one instruction line (trailing `//` and `;` comments allowed) for
/// `arch` into a typed [`Inst`] that [`Inst::encode`] accepts.
///
/// See the module docs for the grammar and the error vocabulary. Symbolic
/// branch labels are not resolved here.
pub fn parse_line(text: &str, arch: Arch) -> Result<Inst> {
    let line = strip_comment(text);
    if line.is_empty() {
        return Err("empty instruction: the line holds no mnemonic".into());
    }
    let (mnemonic, rest) = match line.find(char::is_whitespace) {
        Some(i) => (&line[..i], line[i..].trim()),
        None => (line, ""),
    };
    if mnemonic.ends_with(':') || mnemonic.starts_with('.') {
        return Err(format!(
            "not an instruction: `{mnemonic}` is a label or assembler directive"
        ));
    }
    let lower = if mnemonic.bytes().any(|b| b.is_ascii_uppercase()) {
        Cow::Owned(mnemonic.to_ascii_lowercase())
    } else {
        Cow::Borrowed(mnemonic)
    };
    let (cands, name, suffix) = resolve(&lower, arch)?;
    let ln = Line { name, rest };
    let mut errors: Vec<(&'static str, String)> = Vec::new();
    for cand in cands.iter().copied().filter(|c| suffix.accepts(*c)) {
        let built = build(cand, &ln, suffix, arch).and_then(|inst| {
            inst.words(arch)
                .map(|_| inst)
                .map_err(|e| format!("encode: `{line}`: {e}"))
        });
        match built {
            Ok(inst) => return Ok(inst),
            Err(e) => errors.push((cand.label(), e)),
        }
    }
    if errors.len() == 1 {
        return Err(errors.remove(0).1);
    }
    let joined: Vec<String> = errors.iter().map(|(l, e)| format!("[{l}] {e}")).collect();
    Err(format!(
        "no encoding of {name} accepts `{line}`: {}",
        joined.join("; ")
    ))
}

// ---------------------------------------------------------------------------
// Mnemonic resolution.

/// One table entry a mnemonic can mean.
#[derive(Clone, Copy, Debug)]
enum Cand {
    Sop2(Sop2Op),
    Sop1(Sop1Op),
    Sopk(SopkOp),
    Sopc(SopcOp),
    Sopp(SoppOp),
    Smem(SmemOp),
    Vop1(Vop1Op),
    Vop2(Vop2Op),
    Vop2K(Vop2KOp),
    Vopc(VopcOp),
    Vop3(Vop3Op),
    Vop3p(Vop3pOp),
    Mai(MaiOp),
    Ds(DsOp),
    Flat(FlatOp, FlatSeg),
    Mubuf(MubufOp),
}

impl Cand {
    fn label(self) -> &'static str {
        match self {
            Self::Sop2(_) => "SOP2",
            Self::Sop1(_) => "SOP1",
            Self::Sopk(_) => "SOPK",
            Self::Sopc(_) => "SOPC",
            Self::Sopp(_) => "SOPP",
            Self::Smem(_) => "SMEM",
            Self::Vop1(_) => "VOP1 e32",
            Self::Vop2(_) => "VOP2 e32",
            Self::Vop2K(_) => "VOP2 literal",
            Self::Vopc(_) => "VOPC e32",
            Self::Vop3(_) => "VOP3 e64",
            Self::Vop3p(_) => "VOP3P",
            Self::Mai(_) => "VOP3P-MAI",
            Self::Ds(_) => "DS",
            Self::Flat(..) => "FLAT",
            Self::Mubuf(_) => "MUBUF",
        }
    }
    fn is_e32(self) -> bool {
        matches!(
            self,
            Self::Vop1(_) | Self::Vop2(_) | Self::Vop2K(_) | Self::Vopc(_)
        )
    }
}

/// Explicit encoding suffix of the mnemonic as written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Suffix {
    None,
    E32,
    E64,
    Dpp,
    Sdwa,
}

impl Suffix {
    fn accepts(self, cand: Cand) -> bool {
        match self {
            Self::None => true,
            Self::E32 | Self::Dpp | Self::Sdwa => cand.is_e32(),
            Self::E64 => matches!(cand, Cand::Vop3(_)),
        }
    }
}

type NameIndex = HashMap<&'static str, Vec<Cand>>;

/// Mnemonic -> table entries, per architecture, built once from the opcode
/// tables. The 32-bit forms are listed before the VOP3 form of the same name
/// (that is the order `parse_line` tries them in).
static INDEX: LazyLock<HashMap<Arch, NameIndex>> = LazyLock::new(build_index);

fn add_all<T: Copy>(
    m: &mut NameIndex,
    ops: &'static [T],
    present: impl Fn(T) -> bool,
    name: impl Fn(T) -> &'static str,
    cand: impl Fn(T) -> Cand,
) {
    for &op in ops {
        if present(op) {
            m.entry(name(op)).or_default().push(cand(op));
        }
    }
}

fn build_index() -> HashMap<Arch, NameIndex> {
    let mut all = HashMap::new();
    for &arch in Arch::ALL {
        let mut m: NameIndex = HashMap::new();
        add_all(
            &mut m,
            Sop2Op::ALL,
            |o: Sop2Op| o.opcode(arch).is_some(),
            Sop2Op::name,
            Cand::Sop2,
        );
        add_all(
            &mut m,
            Sop1Op::ALL,
            |o: Sop1Op| o.opcode(arch).is_some(),
            Sop1Op::name,
            Cand::Sop1,
        );
        add_all(
            &mut m,
            SopkOp::ALL,
            |o: SopkOp| o.opcode(arch).is_some(),
            SopkOp::name,
            Cand::Sopk,
        );
        add_all(
            &mut m,
            SopcOp::ALL,
            |o: SopcOp| o.opcode(arch).is_some(),
            SopcOp::name,
            Cand::Sopc,
        );
        add_all(
            &mut m,
            SoppOp::ALL,
            |o: SoppOp| o.opcode(arch).is_some(),
            SoppOp::name,
            Cand::Sopp,
        );
        add_all(
            &mut m,
            SmemOp::ALL,
            |o: SmemOp| o.opcode(arch).is_some(),
            SmemOp::name,
            Cand::Smem,
        );
        add_all(
            &mut m,
            Vop2KOp::ALL,
            |o: Vop2KOp| o.opcode(arch).is_some(),
            Vop2KOp::name,
            Cand::Vop2K,
        );
        add_all(
            &mut m,
            Vop1Op::ALL,
            |o: Vop1Op| o.opcode(arch).is_some(),
            Vop1Op::name,
            Cand::Vop1,
        );
        add_all(
            &mut m,
            Vop2Op::ALL,
            |o: Vop2Op| o.opcode(arch).is_some(),
            Vop2Op::name,
            Cand::Vop2,
        );
        add_all(
            &mut m,
            VopcOp::ALL,
            |o: VopcOp| o.opcode(arch).is_some(),
            VopcOp::name,
            Cand::Vopc,
        );
        for op in Vop3Op::all(arch) {
            m.entry(op.name()).or_default().push(Cand::Vop3(op));
        }
        add_all(
            &mut m,
            Vop3pOp::ALL,
            |o: Vop3pOp| o.opcode(arch).is_some(),
            Vop3pOp::name,
            Cand::Vop3p,
        );
        add_all(
            &mut m,
            MaiOp::ALL,
            |o: MaiOp| o.opcode(arch).is_some(),
            MaiOp::name,
            Cand::Mai,
        );
        add_all(
            &mut m,
            DsOp::ALL,
            |o: DsOp| o.opcode(arch).is_some(),
            DsOp::name,
            Cand::Ds,
        );
        add_all(
            &mut m,
            MubufOp::ALL,
            |o: MubufOp| o.opcode(arch).is_some(),
            MubufOp::name,
            Cand::Mubuf,
        );
        for seg in [FlatSeg::Global, FlatSeg::Flat, FlatSeg::Scratch] {
            for &op in FlatOp::ALL {
                if op.opcode(arch).is_none() {
                    continue;
                }
                if let Some(name) = op.name(seg) {
                    m.entry(name).or_default().push(Cand::Flat(op, seg));
                }
            }
        }
        all.insert(arch, m);
    }
    all
}

/// llvm accepts the pre-gfx940 MFMA spellings (`16x16x16bf16_1k`,
/// `16x16x4f32`, `f64_16x16x4f64`) as aliases of the gfx940+ names that put an
/// underscore between the shape and the type and drop the `_1k` tag.
fn canonical_mnemonic(m: &str) -> Cow<'_, str> {
    let Some(rest) = m.strip_prefix("v_mfma_") else {
        return Cow::Borrowed(m);
    };
    let stripped = rest.strip_suffix("_1k");
    let rest = stripped.unwrap_or(rest);
    let Some((dst, shape)) = rest.split_once('_') else {
        return Cow::Borrowed(m);
    };
    let mut xs = 0;
    let mut end = None;
    for (i, c) in shape.char_indices() {
        match c {
            'x' => xs += 1,
            '0'..='9' => {}
            _ => {
                if xs == 2 {
                    end = Some(i);
                }
                break;
            }
        }
    }
    match end {
        Some(p) if !shape[p..].starts_with('_') => {
            Cow::Owned(format!("v_mfma_{dst}_{}_{}", &shape[..p], &shape[p..]))
        }
        _ if stripped.is_some() => Cow::Owned(format!("v_mfma_{dst}_{shape}")),
        _ => Cow::Borrowed(m),
    }
}

fn resolve(mnemonic: &str, arch: Arch) -> Result<(&'static [Cand], &'static str, Suffix)> {
    let table = INDEX
        .get(&arch)
        .ok_or_else(|| format!("unknown mnemonic: no instruction table for {arch}"))?;
    let (base, suffix) = [
        ("_e32", Suffix::E32),
        ("_e64", Suffix::E64),
        ("_dpp", Suffix::Dpp),
        ("_sdwa", Suffix::Sdwa),
    ]
    .iter()
    .find_map(|&(s, k)| mnemonic.strip_suffix(s).map(|b| (b, k)))
    .unwrap_or((mnemonic, Suffix::None));
    let canon = canonical_mnemonic(base);
    let Some((&name, all)) = table.get_key_value(&*canon) else {
        return Err(format!(
            "unknown mnemonic: `{mnemonic}` is not an instruction of the {arch} PeaceMaker CDNA tables"
        ));
    };
    if !all.iter().copied().any(|cand| suffix.accepts(cand)) {
        return Err(format!(
            "unknown mnemonic: `{mnemonic}`: {name} has no {} encoding",
            match suffix {
                Suffix::E64 => "64-bit (VOP3)",
                Suffix::Dpp => "DPP",
                Suffix::Sdwa => "SDWA",
                _ => "32-bit",
            }
        ));
    }
    Ok((all, name, suffix))
}

// ---------------------------------------------------------------------------
// Error text.

fn err_operand(name: &str, i: usize, text: &str, why: impl std::fmt::Display) -> String {
    format!("bad operand: operand {} `{text}` of {name}: {why}", i + 1)
}

fn err_modifier(name: &str, text: &str, why: impl std::fmt::Display) -> String {
    format!("unsupported modifier: `{text}` on {name}: {why}")
}

// ---------------------------------------------------------------------------
// Lexing.

fn strip_comment(s: &str) -> &str {
    let mut end = s.len();
    for pat in ["//", ";"] {
        if let Some(i) = s.find(pat) {
            end = end.min(i);
        }
    }
    s[..end].trim()
}

/// Split at `sep` characters that sit outside `()` / `[]` / double quotes.
/// Always returns at least one (possibly empty) piece.
fn split_top(s: &str, sep: impl Fn(char) -> bool) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut quote, mut start) = (0i32, false, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '"' => quote = !quote,
            '(' | '[' if !quote => depth += 1,
            ')' | ']' if !quote => depth -= 1,
            c if depth <= 0 && !quote && sep(c) => {
                out.push(&s[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    out.push(&s[start..]);
    out
}

fn parse_int(t: &str) -> Option<i128> {
    let t = t.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let v = if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        if h.is_empty() || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        i128::from_str_radix(h, 16).ok()?
    } else {
        if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        body.parse::<i128>().ok()?
    };
    Some(if neg { -v } else { v })
}

fn parse_float(t: &str) -> Option<f64> {
    let t = t.trim();
    let body = t
        .strip_prefix('-')
        .or_else(|| t.strip_prefix('+'))
        .unwrap_or(t);
    let first = body.chars().next()?;
    if !(first.is_ascii_digit() || first == '.') {
        return None;
    }
    if !body
        .bytes()
        .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return None;
    }
    t.parse::<f64>().ok()
}

fn unwrap_call<'a>(t: &'a str, open: &str) -> Option<&'a str> {
    t.strip_prefix(open)?.strip_suffix(')')
}

// ---------------------------------------------------------------------------
// Registers and operands.

/// A register token: `(register, dwords)`. Dwords is 0 for the read-only
/// `src_*` values, which are one code whatever width the operand has.
fn parse_special(t: &str) -> Option<Result<(Src, u8)>> {
    let ok = |s: Src, n: u8| Some(Ok((s, n)));
    match t {
        "vcc" => ok(Src::VccLo, 2),
        "vcc_lo" => ok(Src::VccLo, 1),
        "vcc_hi" => ok(Src::VccHi, 1),
        "exec" => ok(Src::ExecLo, 2),
        "exec_lo" => ok(Src::ExecLo, 1),
        "exec_hi" => ok(Src::ExecHi, 1),
        "flat_scratch" => ok(Src::FlatScratchLo, 2),
        "flat_scratch_lo" => ok(Src::FlatScratchLo, 1),
        "flat_scratch_hi" => ok(Src::FlatScratchHi, 1),
        "xnack_mask" => ok(Src::XnackMaskLo, 2),
        "xnack_mask_lo" => ok(Src::XnackMaskLo, 1),
        "xnack_mask_hi" => ok(Src::XnackMaskHi, 1),
        "m0" => ok(Src::M0, 1),
        "src_shared_base" => ok(Src::SharedBase, 0),
        "src_shared_limit" => ok(Src::SharedLimit, 0),
        "src_private_base" => ok(Src::PrivateBase, 0),
        "src_private_limit" => ok(Src::PrivateLimit, 0),
        "src_vccz" | "vccz" => ok(Src::Vccz, 0),
        "src_execz" | "execz" => ok(Src::Execz, 0),
        "src_scc" | "scc" => ok(Src::Scc, 0),
        "null" | "src_pops_exiting_wave_id" => Some(Err(format!(
            "unsupported operand: `{t}` has no operand model on gfx942"
        ))),
        _ => None,
    }
}

/// `s5`, `s[4:7]`, `v[0]`, `a[0:3]`, `ttmp[4:5]`.
fn parse_family(t: &str) -> Option<Result<(Src, u8)>> {
    let (prefix, rest) = ["ttmp", "s", "v", "a"]
        .iter()
        .find_map(|&p| t.strip_prefix(p).map(|r| (p, r)))?;
    let first = rest.chars().next()?;
    if !(first.is_ascii_digit() || first == '[') {
        return None;
    }
    let range = || -> Result<(u32, u32)> {
        let num = |s: &str| {
            s.parse::<u32>()
                .map_err(|_| format!("bad register `{t}`: `{s}` is not a register number"))
        };
        if let Some(inner) = rest.strip_prefix('[') {
            let inner = inner
                .strip_suffix(']')
                .ok_or_else(|| format!("bad register `{t}`: missing `]`"))?;
            let compact: String = inner.chars().filter(|c| !c.is_whitespace()).collect();
            match compact.split_once(':') {
                Some((a, b)) => Ok((num(a)?, num(b)?)),
                None => {
                    let n = num(&compact)?;
                    Ok((n, n))
                }
            }
        } else {
            let n = num(rest)?;
            Ok((n, n))
        }
    };
    let (lo, hi) = match range() {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
    };
    let limit = match prefix {
        "s" => 101,
        "ttmp" => 15,
        _ => 255,
    };
    if hi < lo {
        return Some(Err(format!("bad register `{t}`: the range runs backwards")));
    }
    if hi > limit {
        return Some(Err(format!(
            "bad register `{t}`: {prefix}{hi} is beyond the last {prefix} register ({prefix}{limit})"
        )));
    }
    if hi - lo + 1 > 32 {
        return Some(Err(format!(
            "bad register `{t}`: a range spans at most 32 registers"
        )));
    }
    let src = match prefix {
        "s" => Src::Sgpr(lo as u8),
        "ttmp" => Src::Ttmp(lo as u8),
        "v" => Src::Vgpr(lo as u8),
        _ => Src::Agpr(lo as u8),
    };
    Some(Ok((src, (hi - lo + 1) as u8)))
}

fn parse_reg(t: &str) -> Option<Result<(Src, u8)>> {
    parse_special(t).or_else(|| parse_family(t))
}

#[derive(Clone, Copy, Debug)]
enum Body {
    Reg(Src, u8),
    Int(i128),
    Float(f64),
    /// `lit(...)`: the literal encoding was asked for explicitly.
    Lit(u32),
    Off,
}

/// One operand with its source modifiers.
#[derive(Clone, Copy, Debug)]
struct Operand<'a> {
    text: &'a str,
    body: Body,
    neg: bool,
    abs: bool,
    sext: bool,
}

fn parse_operand<'a>(name: &str, i: usize, text: &'a str) -> Result<Operand<'a>> {
    let bad = |why: &str| err_operand(name, i, text, why);
    let mut t = text.trim();
    if t.is_empty() {
        return Err(bad("empty operand"));
    }
    let (mut neg, mut abs, mut sext) = (false, false, false);
    loop {
        if let Some(r) = unwrap_call(t, "neg(") {
            if neg {
                return Err(bad("neg applied twice"));
            }
            neg = true;
            t = r.trim();
        } else if let Some(r) = unwrap_call(t, "abs(") {
            if abs {
                return Err(bad("abs applied twice"));
            }
            abs = true;
            t = r.trim();
        } else if let Some(r) = unwrap_call(t, "sext(") {
            if sext {
                return Err(bad("sext applied twice"));
            }
            sext = true;
            t = r.trim();
        } else if t.len() >= 2 && t.starts_with('|') && t.ends_with('|') {
            if abs {
                return Err(bad("abs applied twice"));
            }
            abs = true;
            t = t[1..t.len() - 1].trim();
        } else if let Some(r) = t.strip_prefix('-') {
            // `-1` / `-0.5` are negative constants; `-v1` / `-|v1|` negate.
            if r.chars()
                .next()
                .is_some_and(|c| c.is_ascii_digit() || c == '.')
            {
                break;
            }
            if neg {
                return Err(bad("neg applied twice"));
            }
            neg = true;
            t = r.trim_start();
        } else {
            break;
        }
    }
    let body = if let Some(inner) = unwrap_call(t, "lit(") {
        if let Some(v) = parse_int(inner) {
            if !(i128::from(i32::MIN)..=i128::from(u32::MAX)).contains(&v) {
                return Err(bad("lit() value does not fit 32 bits"));
            }
            Body::Lit(v as u32)
        } else if let Some(x) = parse_float(inner) {
            Body::Lit((x as f32).to_bits())
        } else {
            return Err(bad("lit() takes an integer or float constant"));
        }
    } else if t == "off" {
        Body::Off
    } else if let Some(r) = parse_reg(t) {
        let (s, n) = r.map_err(|e| {
            if e.starts_with("unsupported operand") {
                e
            } else {
                format!("bad operand: {e} (operand {} of {name})", i + 1)
            }
        })?;
        Body::Reg(s, n)
    } else if let Some(v) = parse_int(t) {
        Body::Int(v)
    } else if let Some(x) = parse_float(t) {
        Body::Float(x)
    } else {
        return Err(bad(
            "not a register, inline constant, literal or modifier expression",
        ));
    };
    Ok(Operand {
        text,
        body,
        neg,
        abs,
        sext,
    })
}

/// How llvm types a constant operand (which values fold to inline
/// constants, and how wide a literal may be).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    B32,
    F16,
    I16,
    I64,
    F64,
}

fn inline_by(f: impl Fn(InlineFloat) -> bool) -> Option<Src> {
    InlineFloat::ALL
        .iter()
        .copied()
        .find(|&x| f(x))
        .map(Src::Float)
}

fn int_src(v: i128, ty: Ty) -> std::result::Result<Src, String> {
    let small = (-16..=64).contains(&v);
    let fits = |lo: i128, hi: i128| (lo..=hi).contains(&v);
    match ty {
        Ty::B32 => {
            if !fits(i128::from(i32::MIN), i128::from(u32::MAX)) {
                return Err(format!("{v} does not fit a 32-bit operand"));
            }
            let bits = v as u32;
            let t = bits as i32;
            if (-16..=64).contains(&t) {
                Ok(Src::Int(t as i8))
            } else {
                Ok(inline_by(|f| f.f32_bits() == bits).unwrap_or(Src::Lit(bits)))
            }
        }
        Ty::F16 => {
            if !fits(-0x8000, 0xffff) {
                return Err(format!("{v} does not fit a 16-bit operand"));
            }
            let bits = v as u16;
            let t = bits as i16;
            if (-16..=64).contains(&t) {
                Ok(Src::Int(t as i8))
            } else {
                Ok(inline_by(|f| f.f16_bits() == bits).unwrap_or(Src::Lit(u32::from(bits))))
            }
        }
        Ty::I16 => {
            if !fits(-0x8000, 0xffff) {
                return Err(format!("{v} does not fit a 16-bit operand"));
            }
            Ok(if small {
                Src::Int(v as i8)
            } else {
                Src::Lit(u32::from(v as u16))
            })
        }
        Ty::I64 => {
            if small {
                Ok(Src::Int(v as i8))
            } else if fits(i128::from(i32::MIN), i128::from(u32::MAX)) {
                Ok(Src::Lit(v as u32))
            } else {
                Err(format!(
                    "{v} does not fit the 32-bit literal of a 64-bit operand"
                ))
            }
        }
        Ty::F64 => {
            if small {
                Ok(Src::Int(v as i8))
            } else if fits(0, i128::from(u32::MAX)) {
                let bits = v as u32;
                Ok(
                    inline_by(|f| f.f64_bits() as u32 == 0 && (f.f64_bits() >> 32) as u32 == bits)
                        .unwrap_or(Src::Lit(bits)),
                )
            } else {
                Err(format!(
                    "{v} does not fit the high-dword literal of a 64-bit float operand"
                ))
            }
        }
    }
}

fn float_src(x: f64, ty: Ty) -> std::result::Result<Src, String> {
    if !x.is_finite() {
        return Err(format!("{x} is not a finite float constant"));
    }
    if x == 0.0 {
        return Ok(Src::Int(0));
    }
    // `Display` prints 1/(2*pi) at f32 precision on every operand width.
    if (x as f32).to_bits() == InlineFloat::InvTwoPi.f32_bits() {
        return Ok(Src::Float(InlineFloat::InvTwoPi));
    }
    match ty {
        Ty::B32 => {
            let bits = (x as f32).to_bits();
            Ok(inline_by(|f| f.f32_bits() == bits).unwrap_or(Src::Lit(bits)))
        }
        Ty::F16 | Ty::I16 => {
            let bits = f32_to_f16_rne(x as f32);
            Ok(inline_by(|f| f.f16_bits() == bits).unwrap_or(Src::Lit(u32::from(bits))))
        }
        Ty::F64 | Ty::I64 => {
            let bits = x.to_bits();
            if let Some(s) = inline_by(|f| f.f64_bits() == bits) {
                return Ok(s);
            }
            if bits as u32 == 0 {
                Ok(Src::Lit((bits >> 32) as u32))
            } else {
                Err(format!(
                    "{x} is not an inline constant and its low dword is nonzero: a 64-bit float literal carries only the high dword"
                ))
            }
        }
    }
}

impl<'a> Operand<'a> {
    fn forbid_mods(&self, name: &str, i: usize) -> Result<()> {
        if self.neg || self.abs || self.sext {
            Err(err_operand(
                name,
                i,
                self.text,
                "source modifiers (neg/abs/sext) are not accepted here",
            ))
        } else {
            Ok(())
        }
    }

    fn check_width(&self, name: &str, i: usize, n: u8, want: u8) -> Result<()> {
        if n != 0 && n != want.max(1) {
            Err(err_operand(
                name,
                i,
                self.text,
                format!("is {n} dword(s) wide but the operand takes {}", want.max(1)),
            ))
        } else {
            Ok(())
        }
    }

    fn to_src(&self, name: &str, i: usize, ty: Ty, want: u8) -> Result<Src> {
        let bad = |why: String| err_operand(name, i, self.text, why);
        match self.body {
            Body::Reg(s, n) => {
                self.check_width(name, i, n, want)?;
                Ok(s)
            }
            Body::Int(v) => int_src(v, ty).map_err(bad),
            Body::Float(x) => float_src(x, ty).map_err(bad),
            Body::Lit(v) => Ok(Src::Lit(v)),
            Body::Off => Err(bad("`off` is not a value operand".into())),
        }
    }
}

// ---------------------------------------------------------------------------
// One line's operands and modifier words.

struct Line<'a> {
    name: &'static str,
    rest: &'a str,
}

struct Ops<'a> {
    name: &'static str,
    list: Vec<&'a str>,
}

impl<'a> Line<'a> {
    /// Comma-separated operands; whitespace-separated modifier words after the
    /// last operand.
    fn split(&self) -> Result<(Ops<'a>, Mods<'a>)> {
        let name = self.name;
        let mut ops: Vec<&'a str> = Vec::new();
        let mut mods: Vec<&'a str> = Vec::new();
        if !self.rest.is_empty() {
            let segs = split_top(self.rest, |c| c == ',');
            let last = segs.len() - 1;
            for (i, &seg) in segs.iter().enumerate() {
                let words: Vec<&'a str> = split_top(seg, char::is_whitespace)
                    .into_iter()
                    .filter(|w| !w.is_empty())
                    .collect();
                if words.is_empty() {
                    return Err(format!(
                        "operand count: operand {} of {name} is empty",
                        i + 1
                    ));
                }
                ops.push(words[0]);
                if words.len() > 1 {
                    if i != last {
                        return Err(format!(
                            "bad operand: `{}` of {name} has tokens after operand {} but is not the last operand",
                            seg.trim(),
                            i + 1
                        ));
                    }
                    mods.extend_from_slice(&words[1..]);
                }
            }
        }
        Ok((Ops { name, list: ops }, Mods::new(name, &mods)?))
    }
}

impl<'a> Ops<'a> {
    fn count(&self, want: usize) -> Result<()> {
        if self.list.len() == want {
            Ok(())
        } else {
            Err(format!(
                "operand count: {} takes {want} operand(s), got {}",
                self.name,
                self.list.len()
            ))
        }
    }
    fn operand(&self, i: usize) -> Result<Operand<'a>> {
        parse_operand(self.name, i, self.list[i])
    }
    /// A register of `want` dwords, no modifiers.
    fn reg(&self, i: usize, want: u8) -> Result<Src> {
        let o = self.operand(i)?;
        o.forbid_mods(self.name, i)?;
        match o.body {
            Body::Reg(s, n) => {
                o.check_width(self.name, i, n, want)?;
                Ok(s)
            }
            _ => Err(err_operand(self.name, i, o.text, "must be a register")),
        }
    }
    /// A register, or `off`.
    fn reg_or_off(&self, i: usize, want: u8) -> Result<Option<Src>> {
        let o = self.operand(i)?;
        o.forbid_mods(self.name, i)?;
        match o.body {
            Body::Off => Ok(None),
            Body::Reg(s, n) => {
                o.check_width(self.name, i, n, want)?;
                Ok(Some(s))
            }
            _ => Err(err_operand(
                self.name,
                i,
                o.text,
                "must be a register or `off`",
            )),
        }
    }
    /// A register or constant, no modifiers.
    fn src(&self, i: usize, ty: Ty, want: u8) -> Result<Src> {
        let o = self.operand(i)?;
        o.forbid_mods(self.name, i)?;
        o.to_src(self.name, i, ty, want)
    }
    /// The carry/select operand of a 32-bit VOP encoding: `vcc`.
    fn vcc(&self, i: usize) -> Result<()> {
        let o = self.operand(i)?;
        o.forbid_mods(self.name, i)?;
        match o.body {
            Body::Reg(Src::VccLo, 2) => Ok(()),
            _ => Err(err_operand(
                self.name,
                i,
                o.text,
                "must be `vcc` in this encoding (the _e64 form takes any SGPR pair)",
            )),
        }
    }
    fn imm(&self, i: usize, lo: i128, hi: i128) -> Result<i128> {
        let t = self.list[i];
        let v = parse_int(t)
            .ok_or_else(|| err_operand(self.name, i, t, "expected an integer immediate"))?;
        if !(lo..=hi).contains(&v) {
            return Err(err_operand(
                self.name,
                i,
                t,
                format!("{v} is outside {lo}..={hi}"),
            ));
        }
        Ok(v)
    }
    /// A mandatory 32-bit literal dword (`v_fmamk_f32` / `v_fmaak_f32`).
    fn literal(&self, i: usize) -> Result<u32> {
        let o = self.operand(i)?;
        o.forbid_mods(self.name, i)?;
        match o.body {
            Body::Int(v) => {
                if !(i128::from(i32::MIN)..=i128::from(u32::MAX)).contains(&v) {
                    return Err(err_operand(
                        self.name,
                        i,
                        o.text,
                        "does not fit a 32-bit literal",
                    ));
                }
                Ok(v as u32)
            }
            Body::Float(x) => Ok((x as f32).to_bits()),
            Body::Lit(v) => Ok(v),
            _ => Err(err_operand(
                self.name,
                i,
                o.text,
                "must be a 32-bit integer or float constant",
            )),
        }
    }
}

struct ModItem<'a> {
    text: &'a str,
    key: &'a str,
    value: Option<&'a str>,
    used: bool,
}

/// The `key[:value]` modifier words of a line; each form takes what it
/// understands and `finish` rejects the rest.
struct Mods<'a> {
    name: &'static str,
    items: Vec<ModItem<'a>>,
}

impl<'a> Mods<'a> {
    fn new(name: &'static str, words: &[&'a str]) -> Result<Self> {
        let mut items: Vec<ModItem<'a>> = Vec::with_capacity(words.len());
        for &w in words {
            let (key, value) = match w.split_once(':') {
                Some((k, v)) => (k, Some(v)),
                None => (w, None),
            };
            if items.iter().any(|i| i.key == key) {
                return Err(format!(
                    "duplicate modifier: `{key}` appears twice on {name}"
                ));
            }
            items.push(ModItem {
                text: w,
                key,
                value,
                used: false,
            });
        }
        Ok(Self { name, items })
    }
    fn has(&self, key: &str) -> bool {
        self.items.iter().any(|i| i.key == key)
    }
    fn flag(&mut self, key: &str) -> Result<bool> {
        let name = self.name;
        match self.items.iter_mut().find(|i| i.key == key) {
            None => Ok(false),
            Some(i) if i.value.is_some() => Err(err_modifier(name, i.text, "takes no value")),
            Some(i) => {
                i.used = true;
                Ok(true)
            }
        }
    }
    fn value(&mut self, key: &str) -> Result<Option<&'a str>> {
        let name = self.name;
        match self.items.iter_mut().find(|i| i.key == key) {
            None => Ok(None),
            Some(i) => match i.value {
                Some(v) => {
                    i.used = true;
                    Ok(Some(v))
                }
                None => Err(err_modifier(name, i.text, "needs a `:value`")),
            },
        }
    }
    fn int(&mut self, key: &str, lo: i128, hi: i128) -> Result<Option<i128>> {
        let name = self.name;
        let Some(v) = self.value(key)? else {
            return Ok(None);
        };
        let n = parse_int(v)
            .ok_or_else(|| err_modifier(name, key, format!("`{v}` is not an integer")))?;
        if !(lo..=hi).contains(&n) {
            return Err(err_modifier(
                name,
                key,
                format!("{n} is outside {lo}..={hi}"),
            ));
        }
        Ok(Some(n))
    }
    /// A `[0,1,..]` list of at most `count` bits: `(bit i = element i, length)`.
    fn bits(&mut self, key: &str, count: usize) -> Result<Option<(u8, usize)>> {
        let name = self.name;
        let Some(v) = self.value(key)? else {
            return Ok(None);
        };
        let inner = v
            .strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| err_modifier(name, key, format!("`{v}` is not a [..] list")))?;
        let mut out = 0u8;
        let mut n = 0usize;
        for (i, part) in inner.split(',').enumerate() {
            if i >= count {
                return Err(err_modifier(
                    name,
                    key,
                    format!("takes at most {count} elements"),
                ));
            }
            match part.trim() {
                "0" => {}
                "1" => out |= 1 << i,
                other => {
                    return Err(err_modifier(
                        name,
                        key,
                        format!("element `{other}` is not 0 or 1"),
                    ))
                }
            }
            n += 1;
        }
        Ok(Some((out, n)))
    }
    fn finish(self) -> Result<()> {
        match self.items.iter().find(|i| !i.used) {
            Some(i) => Err(err_modifier(
                self.name,
                i.text,
                "is not accepted by this instruction form",
            )),
            None => Ok(()),
        }
    }
}

/// `mul:2` / `mul:4` / `div:2` -> OMOD field.
fn omod(m: &mut Mods) -> Result<u8> {
    let name = m.name;
    let mul = m.int("mul", 2, 4)?;
    let div = m.int("div", 2, 2)?;
    match (mul, div) {
        (None, None) => Ok(0),
        (Some(2), None) => Ok(1),
        (Some(4), None) => Ok(2),
        (None, Some(_)) => Ok(3),
        (Some(n), None) => Err(err_modifier(
            name,
            "mul",
            format!("mul:{n} is not mul:2 or mul:4"),
        )),
        (Some(_), Some(_)) => Err(err_modifier(name, "div", "mul and div cannot be combined")),
    }
}

// ---------------------------------------------------------------------------
// Builders. Each turns one table entry plus the line's operands into a typed
// instruction; `parse_line` validates the result with the real encoder.

fn build(cand: Cand, p: &Line, suffix: Suffix, arch: Arch) -> Result<Inst> {
    match cand {
        Cand::Sop2(op) => sop2(op, p),
        Cand::Sop1(op) => sop1(op, p),
        Cand::Sopk(op) => sopk(op, p),
        Cand::Sopc(op) => sopc(op, p),
        Cand::Sopp(op) => sopp(op, p, arch),
        Cand::Smem(op) => smem(op, p),
        Cand::Vop1(op) => vop1(op, p, suffix),
        Cand::Vop2(op) => vop2(op, p, suffix),
        Cand::Vop2K(op) => vop2k(op, p, suffix),
        Cand::Vopc(op) => vopc(op, p, suffix),
        Cand::Vop3(op) => vop3(op, p),
        Cand::Vop3p(op) => vop3p(op, p),
        Cand::Mai(op) => mai(op, p),
        Cand::Ds(op) => ds(op, p),
        Cand::Flat(op, seg) => flat(op, seg, p),
        Cand::Mubuf(op) => mubuf(op, p),
    }
}

fn salu_ty(width: u8) -> Ty {
    if width == 2 {
        Ty::I64
    } else {
        Ty::B32
    }
}

fn sop2(op: Sop2Op, p: &Line) -> Result<Inst> {
    let (o, m) = p.split()?;
    m.finish()?;
    o.count(3)?;
    let (wd, w0, w1) = op.widths();
    Ok(Inst::Sop2(Sop2 {
        op,
        sdst: o.reg(0, wd)?,
        ssrc0: o.src(1, salu_ty(w0), w0)?,
        ssrc1: o.src(2, salu_ty(w1), w1)?,
    }))
}

fn sop1(op: Sop1Op, p: &Line) -> Result<Inst> {
    let (o, m) = p.split()?;
    m.finish()?;
    let (wd, w0) = op.widths();
    o.count(usize::from(wd > 0) + usize::from(w0 > 0))?;
    let mut i = 0;
    let sdst = if wd > 0 {
        i += 1;
        Some(o.reg(0, wd)?)
    } else {
        None
    };
    let ssrc0 = if w0 > 0 {
        Some(o.src(i, salu_ty(w0), w0)?)
    } else {
        None
    };
    Ok(Inst::Sop1(Sop1 { op, sdst, ssrc0 }))
}

fn sopk(op: SopkOp, p: &Line) -> Result<Inst> {
    let (o, m) = p.split()?;
    m.finish()?;
    o.count(2)?;
    let name = op.name();
    let (sdst, simm16) = match op {
        SopkOp::GetregB32 => (
            o.reg(0, 1)?,
            parse_hwreg(o.list[1]).map_err(|e| err_operand(name, 1, o.list[1], e))?,
        ),
        SopkOp::SetregB32 => (
            o.reg(1, 1)?,
            parse_hwreg(o.list[0]).map_err(|e| err_operand(name, 0, o.list[0], e))?,
        ),
        _ => (o.reg(0, 1)?, o.imm(1, -0x8000, 0xffff)? as u16),
    };
    Ok(Inst::Sopk(Sopk { op, sdst, simm16 }))
}

fn sopc(op: SopcOp, p: &Line) -> Result<Inst> {
    let (o, m) = p.split()?;
    m.finish()?;
    o.count(2)?;
    let w = op.width();
    Ok(Inst::Sopc(Sopc {
        op,
        ssrc0: o.src(0, salu_ty(w), w)?,
        ssrc1: o.src(1, salu_ty(w), w)?,
    }))
}

fn sopp(op: SoppOp, p: &Line, arch: Arch) -> Result<Inst> {
    let name = op.name();
    if op == SoppOp::Waitcnt {
        let simm16 = parse_waitcnt(arch, p.rest)
            .map_err(|e| format!("bad operand: operand 1 `{}` of {name}: {e}", p.rest))?;
        return Ok(Inst::Sopp(Sopp { op, simm16 }));
    }
    let (o, m) = p.split()?;
    m.finish()?;
    let simm16 = match op {
        SoppOp::Barrier | SoppOp::IcacheInv => {
            o.count(0)?;
            0
        }
        SoppOp::Endpgm if o.list.is_empty() => 0,
        SoppOp::Endpgm | SoppOp::Nop | SoppOp::Sleep | SoppOp::Setprio | SoppOp::Trap => {
            o.count(1)?;
            o.imm(0, 0, 0xffff)? as u16
        }
        SoppOp::Sendmsg => {
            o.count(1)?;
            parse_sendmsg(o.list[0]).map_err(|e| err_operand(name, 0, o.list[0], e))?
        }
        SoppOp::Branch
        | SoppOp::CbranchScc0
        | SoppOp::CbranchScc1
        | SoppOp::CbranchVccz
        | SoppOp::CbranchVccnz
        | SoppOp::CbranchExecz
        | SoppOp::CbranchExecnz => {
            o.count(1)?;
            let t = o.list[0];
            match parse_int(t) {
                Some(v) if (-0x8000..=0xffff).contains(&v) => v as u16,
                Some(v) => {
                    return Err(err_operand(
                        name,
                        0,
                        t,
                        format!("branch offset {v} is outside the 16-bit dword offset"),
                    ))
                }
                None => {
                    return Err(format!(
                        "label operand: `{t}` of {name} is not a numeric dword offset; symbolic labels are resolved by the caller"
                    ))
                }
            }
        }
        SoppOp::Waitcnt => unreachable!("handled above"),
    };
    Ok(Inst::Sopp(Sopp { op, simm16 }))
}

fn smem(op: SmemOp, p: &Line) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let glc = m.flag("glc")?;
    let offset_mod = m.int("offset", i128::from(i32::MIN), i128::from(i32::MAX))?;
    m.finish()?;
    let wd = op.data_dwords();
    if !op.has_address() {
        o.count(usize::from(wd > 0))?;
        if offset_mod.is_some() {
            return Err(err_modifier(
                name,
                "offset",
                "this instruction has no address",
            ));
        }
        let sdata = if wd > 0 { Some(o.reg(0, wd)?) } else { None };
        return Ok(Inst::Smem(Smem {
            op,
            sdata,
            sbase: None,
            offset: SmemOffset::Imm(0),
            glc,
        }));
    }
    o.count(3)?;
    let sdata = Some(o.reg(0, wd)?);
    let sbase = Some(o.reg(1, if op.buffer() { 4 } else { 2 })?);
    let third = o.operand(2)?;
    third.forbid_mods(name, 2)?;
    let offset = match third.body {
        Body::Int(v) => {
            if offset_mod.is_some() {
                return Err(err_modifier(
                    name,
                    "offset",
                    "the immediate offset is already the third operand",
                ));
            }
            if !(i128::from(i32::MIN)..=i128::from(i32::MAX)).contains(&v) {
                return Err(err_operand(
                    name,
                    2,
                    third.text,
                    "offset does not fit 32 bits",
                ));
            }
            SmemOffset::Imm(v as i32)
        }
        Body::Reg(s, n) => {
            third.check_width(name, 2, n, 1)?;
            match offset_mod {
                Some(i) => SmemOffset::SgprImm(s, i as i32),
                None => SmemOffset::Sgpr(s),
            }
        }
        _ => {
            return Err(err_operand(
                name,
                2,
                third.text,
                "must be an immediate byte offset or an SGPR",
            ))
        }
    };
    Ok(Inst::Smem(Smem {
        op,
        sdata,
        sbase,
        offset,
        glc,
    }))
}

// -- VOP1 / VOP2 / VOPC (32-bit, DPP, SDWA) ----------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ext {
    None,
    Dpp,
    Sdwa,
}

const DPP_KEYS: &[&str] = &[
    "quad_perm",
    "row_shl",
    "row_shr",
    "row_ror",
    "wave_shl",
    "wave_rol",
    "wave_shr",
    "wave_ror",
    "row_mirror",
    "row_half_mirror",
    "row_bcast",
    "row_newbcast",
    "row_mask",
    "bank_mask",
    "bound_ctrl",
];
const SDWA_KEYS: &[&str] = &["dst_sel", "dst_unused", "src0_sel", "src1_sel"];

fn ext_kind(name: &str, m: &Mods, suffix: Suffix) -> Result<Ext> {
    let dpp = suffix == Suffix::Dpp || DPP_KEYS.iter().any(|k| m.has(k));
    let sdwa = suffix == Suffix::Sdwa || SDWA_KEYS.iter().any(|k| m.has(k));
    match (dpp, sdwa) {
        (true, true) => Err(err_modifier(
            name,
            "dpp/sdwa",
            "DPP and SDWA control words cannot be combined",
        )),
        (true, false) if suffix == Suffix::E32 => Err(err_modifier(
            name,
            "dpp",
            "the _e32 spelling takes no DPP control word",
        )),
        (false, true) if suffix == Suffix::E32 => Err(err_modifier(
            name,
            "sdwa",
            "the _e32 spelling takes no SDWA control word",
        )),
        (true, false) => Ok(Ext::Dpp),
        (false, true) => Ok(Ext::Sdwa),
        (false, false) => Ok(Ext::None),
    }
}

fn dpp_ctrl(m: &mut Mods) -> Result<DppCtrl> {
    let name = m.name;
    let mut found: Vec<DppCtrl> = Vec::new();
    if let Some(v) = m.value("quad_perm")? {
        let inner = v
            .strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .ok_or_else(|| err_modifier(name, "quad_perm", format!("`{v}` is not [a,b,c,d]")))?;
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() != 4 {
            return Err(err_modifier(name, "quad_perm", "needs four lane indices"));
        }
        let mut perm = [0u8; 4];
        for (i, s) in parts.iter().enumerate() {
            perm[i] = parse_int(s)
                .filter(|v| (0..=3).contains(v))
                .ok_or_else(|| {
                    err_modifier(
                        name,
                        "quad_perm",
                        format!("`{s}` is not a lane index 0..=3"),
                    )
                })? as u8;
        }
        found.push(DppCtrl::QuadPerm(perm));
    }
    let shifts: [(&str, fn(u8) -> DppCtrl); 3] = [
        ("row_shl", DppCtrl::RowShl),
        ("row_shr", DppCtrl::RowShr),
        ("row_ror", DppCtrl::RowRor),
    ];
    for (key, make) in shifts {
        if let Some(n) = m.int(key, 1, 15)? {
            found.push(make(n as u8));
        }
    }
    for (key, ctrl) in [
        ("wave_shl", DppCtrl::WaveShl1),
        ("wave_rol", DppCtrl::WaveRol1),
        ("wave_shr", DppCtrl::WaveShr1),
        ("wave_ror", DppCtrl::WaveRor1),
    ] {
        if m.int(key, 1, 1)?.is_some() {
            found.push(ctrl);
        }
    }
    if let Some(n) = m.int("row_bcast", 15, 31)? {
        match n {
            15 => found.push(DppCtrl::RowBcast15),
            31 => found.push(DppCtrl::RowBcast31),
            _ => {
                return Err(err_modifier(
                    name,
                    "row_bcast",
                    format!("row_bcast:{n} is not row_bcast:15 or row_bcast:31"),
                ))
            }
        }
    }
    if m.flag("row_mirror")? {
        found.push(DppCtrl::RowMirror);
    }
    if m.flag("row_half_mirror")? {
        found.push(DppCtrl::RowHalfMirror);
    }
    if m.has("row_newbcast") {
        return Err(err_modifier(
            name,
            "row_newbcast",
            "64-bit DPP (row_newbcast) is not modelled",
        ));
    }
    match found.len() {
        0 => Err(err_modifier(
            name,
            "dpp",
            "a DPP form needs one control word (quad_perm, row_shl, row_mirror, ...)",
        )),
        1 => Ok(found[0]),
        _ => Err(err_modifier(name, "dpp", "more than one DPP control word")),
    }
}

fn dpp_fields(m: &mut Mods, s0: &Operand, s1: Option<&Operand>) -> Result<Dpp> {
    let name = m.name;
    let ctrl = dpp_ctrl(m)?;
    let row_mask = m.int("row_mask", 0, 15)?.map_or(15, |v| v as u8);
    let bank_mask = m.int("bank_mask", 0, 15)?.map_or(15, |v| v as u8);
    // llvm accepts both `bound_ctrl:0` (legacy) and `bound_ctrl:1` for the one bit.
    let bound_ctrl = m.int("bound_ctrl", 0, 1)?.is_some();
    if s0.sext || s1.is_some_and(|o| o.sext) {
        return Err(err_modifier(name, "sext", "DPP sources take neg/abs only"));
    }
    Ok(Dpp {
        ctrl,
        row_mask,
        bank_mask,
        bound_ctrl,
        src0_neg: s0.neg,
        src0_abs: s0.abs,
        src1_neg: s1.is_some_and(|o| o.neg),
        src1_abs: s1.is_some_and(|o| o.abs),
    })
}

fn take_sel(m: &mut Mods, key: &str) -> Result<Option<SdwaSel>> {
    let name = m.name;
    let Some(v) = m.value(key)? else {
        return Ok(None);
    };
    Ok(Some(match v {
        "BYTE_0" => SdwaSel::Byte0,
        "BYTE_1" => SdwaSel::Byte1,
        "BYTE_2" => SdwaSel::Byte2,
        "BYTE_3" => SdwaSel::Byte3,
        "WORD_0" => SdwaSel::Word0,
        "WORD_1" => SdwaSel::Word1,
        "DWORD" => SdwaSel::Dword,
        _ => {
            return Err(err_modifier(
                name,
                key,
                format!("`{v}` is not BYTE_0..BYTE_3, WORD_0, WORD_1 or DWORD"),
            ))
        }
    }))
}

struct SdwaMods {
    dst_sel: SdwaSel,
    dst_unused: SdwaUnused,
    clamp: bool,
    omod: u8,
    src0_sel: SdwaSel,
    src1_sel: SdwaSel,
}

/// The SDWA control words. `dst`: `dst_sel` / `dst_unused` exist (not for
/// VOPC or the FP8 converts); `out`: `clamp` / `mul` / `div` exist; `src1`:
/// `src1_sel` exists. llvm defaults: `dst_sel:DWORD`, `dst_unused:
/// UNUSED_PRESERVE`, selects `DWORD`.
fn sdwa_mods(m: &mut Mods, dst: bool, out: bool, src1: bool) -> Result<SdwaMods> {
    let name = m.name;
    let dst_sel = if dst {
        take_sel(m, "dst_sel")?.unwrap_or(SdwaSel::Dword)
    } else {
        SdwaSel::Dword
    };
    let dst_unused = if dst {
        match m.value("dst_unused")? {
            None => SdwaUnused::Preserve,
            Some("UNUSED_PAD") => SdwaUnused::Pad,
            Some("UNUSED_SEXT") => SdwaUnused::Sext,
            Some("UNUSED_PRESERVE") => SdwaUnused::Preserve,
            Some(v) => {
                return Err(err_modifier(
                    name,
                    "dst_unused",
                    format!("`{v}` is not UNUSED_PAD, UNUSED_SEXT or UNUSED_PRESERVE"),
                ))
            }
        }
    } else {
        SdwaUnused::Pad
    };
    let (clamp, omod_bits) = if out {
        (m.flag("clamp")?, omod(m)?)
    } else {
        (false, 0)
    };
    let src0_sel = take_sel(m, "src0_sel")?.unwrap_or(SdwaSel::Dword);
    let src1_sel = if src1 {
        take_sel(m, "src1_sel")?.unwrap_or(SdwaSel::Dword)
    } else {
        SdwaSel::Dword
    };
    Ok(SdwaMods {
        dst_sel,
        dst_unused,
        clamp,
        omod: omod_bits,
        src0_sel,
        src1_sel,
    })
}

fn is_fp8_cvt(op: Vop1Op) -> bool {
    matches!(
        op,
        Vop1Op::CvtF32Fp8 | Vop1Op::CvtF32Bf8 | Vop1Op::CvtPkF32Fp8 | Vop1Op::CvtPkF32Bf8
    )
}

fn vop1_ty(op: Vop1Op) -> Ty {
    match op {
        Vop1Op::MovB64 => Ty::I64,
        Vop1Op::CvtF32F64 => Ty::F64,
        Vop1Op::CvtF32F16 => Ty::F16,
        _ => Ty::B32,
    }
}

fn vop2_ty(op: Vop2Op) -> Ty {
    match op {
        Vop2Op::AddF16 | Vop2Op::SubF16 | Vop2Op::MulF16 | Vop2Op::MaxF16 | Vop2Op::MinF16 => {
            Ty::F16
        }
        Vop2Op::AddU16
        | Vop2Op::SubU16
        | Vop2Op::LshlrevB16
        | Vop2Op::AshrrevI16
        | Vop2Op::MulLoU16 => Ty::I16,
        _ => Ty::B32,
    }
}

fn vop1(op: Vop1Op, p: &Line, suffix: Suffix) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let ext = ext_kind(name, &m, suffix)?;
    if op == Vop1Op::Nop {
        if ext != Ext::None {
            return Err(err_modifier(name, "dpp/sdwa", "v_nop has no DPP/SDWA form"));
        }
        o.count(0)?;
        m.finish()?;
        return Ok(Inst::Vop1(Vop1 {
            op,
            vdst: Src::Vgpr(0),
            src0: Src::Sgpr(0),
            ext: VopExt::None,
        }));
    }
    o.count(2)?;
    let (dw, sw) = op.widths();
    let vdst = o.reg(0, dw)?;
    let ty = vop1_ty(op);
    let (src0, ext) = match ext {
        Ext::None => {
            m.finish()?;
            (o.src(1, ty, sw)?, VopExt::None)
        }
        Ext::Dpp => {
            let s0 = o.operand(1)?;
            let src0 = s0.to_src(name, 1, ty, sw)?;
            let dpp = dpp_fields(&mut m, &s0, None)?;
            m.finish()?;
            (src0, VopExt::Dpp(dpp))
        }
        Ext::Sdwa => {
            let s0 = o.operand(1)?;
            let src0 = s0.to_src(name, 1, ty, sw)?;
            let sm = sdwa_mods(&mut m, !is_fp8_cvt(op), true, false)?;
            m.finish()?;
            let sdwa = Sdwa {
                dst_sel: sm.dst_sel,
                dst_unused: sm.dst_unused,
                clamp: sm.clamp,
                omod: sm.omod,
                src0_sel: sm.src0_sel,
                src0_sext: s0.sext,
                src0_neg: s0.neg,
                src0_abs: s0.abs,
                src1_sel: SdwaSel::Dword,
                src1_sext: false,
                src1_neg: false,
                src1_abs: false,
                sdst: None,
            };
            (src0, VopExt::Sdwa(sdwa))
        }
    };
    Ok(Inst::Vop1(Vop1 {
        op,
        vdst,
        src0,
        ext,
    }))
}

/// `(operand count, src0 index, src1 index, vcc operand indices)` of the text
/// form: `v_cndmask_b32 d, a, b, vcc`; `v_addc_co_u32 d, vcc, a, b, vcc`;
/// `v_add_co_u32 d, vcc, a, b`; `v_add_f32 d, a, b`.
fn vop2_layout(op: Vop2Op) -> (usize, usize, usize, &'static [usize]) {
    if op == Vop2Op::CndmaskB32 {
        (4, 1, 2, &[3])
    } else if op.reads_vcc() {
        (5, 2, 3, &[1, 4])
    } else if op.writes_vcc() {
        (4, 2, 3, &[1])
    } else {
        (3, 1, 2, &[])
    }
}

fn vop2(op: Vop2Op, p: &Line, suffix: Suffix) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let ext = ext_kind(name, &m, suffix)?;
    let (n, i0, i1, vccs) = vop2_layout(op);
    o.count(n)?;
    let vdst = o.reg(0, 1)?;
    for &i in vccs {
        o.vcc(i)?;
    }
    let ty = vop2_ty(op);
    match ext {
        Ext::None => {
            m.finish()?;
            Ok(Inst::Vop2(Vop2 {
                op,
                vdst,
                src0: o.src(i0, ty, 1)?,
                src1: o.src(i1, ty, 1)?,
                ext: VopExt::None,
            }))
        }
        Ext::Dpp => {
            let (s0, s1) = (o.operand(i0)?, o.operand(i1)?);
            let src0 = s0.to_src(name, i0, ty, 1)?;
            let src1 = s1.to_src(name, i1, ty, 1)?;
            let dpp = dpp_fields(&mut m, &s0, Some(&s1))?;
            m.finish()?;
            Ok(Inst::Vop2(Vop2 {
                op,
                vdst,
                src0,
                src1,
                ext: VopExt::Dpp(dpp),
            }))
        }
        Ext::Sdwa => {
            let (s0, s1) = (o.operand(i0)?, o.operand(i1)?);
            let src0 = s0.to_src(name, i0, ty, 1)?;
            let src1 = s1.to_src(name, i1, ty, 1)?;
            let sm = sdwa_mods(&mut m, true, true, true)?;
            m.finish()?;
            Ok(Inst::Vop2(Vop2 {
                op,
                vdst,
                src0,
                src1,
                ext: VopExt::Sdwa(Sdwa {
                    dst_sel: sm.dst_sel,
                    dst_unused: sm.dst_unused,
                    clamp: sm.clamp,
                    omod: sm.omod,
                    src0_sel: sm.src0_sel,
                    src0_sext: s0.sext,
                    src0_neg: s0.neg,
                    src0_abs: s0.abs,
                    src1_sel: sm.src1_sel,
                    src1_sext: s1.sext,
                    src1_neg: s1.neg,
                    src1_abs: s1.abs,
                    sdst: None,
                }),
            }))
        }
    }
}

fn vop2k(op: Vop2KOp, p: &Line, suffix: Suffix) -> Result<Inst> {
    if matches!(suffix, Suffix::Dpp | Suffix::Sdwa) {
        return Err(err_modifier(
            op.name(),
            "dpp/sdwa",
            "the mandatory-literal VOP2 form has no DPP/SDWA encoding",
        ));
    }
    let (o, m) = p.split()?;
    m.finish()?;
    o.count(4)?;
    let vdst = o.reg(0, 1)?;
    let src0 = o.src(1, Ty::B32, 1)?;
    let (src1, literal) = match op {
        Vop2KOp::FmamkF32 => (o.reg(3, 1)?, o.literal(2)?),
        Vop2KOp::FmaakF32 => (o.reg(2, 1)?, o.literal(3)?),
    };
    Ok(Inst::Vop2K(Vop2K {
        op,
        vdst,
        src0,
        src1,
        literal,
    }))
}

fn vopc(op: VopcOp, p: &Line, suffix: Suffix) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let ext = ext_kind(name, &m, suffix)?;
    o.count(3)?;
    let dw = op.src_dwords();
    let ty = if dw == 2 { Ty::I64 } else { Ty::B32 };
    match ext {
        Ext::None => {
            o.vcc(0)?;
            m.finish()?;
            Ok(Inst::Vopc(Vopc {
                op,
                src0: o.src(1, ty, dw)?,
                src1: o.src(2, ty, dw)?,
                ext: VopExt::None,
            }))
        }
        Ext::Dpp => {
            o.vcc(0)?;
            let (s0, s1) = (o.operand(1)?, o.operand(2)?);
            let src0 = s0.to_src(name, 1, ty, dw)?;
            let src1 = s1.to_src(name, 2, ty, dw)?;
            let dpp = dpp_fields(&mut m, &s0, Some(&s1))?;
            m.finish()?;
            Ok(Inst::Vopc(Vopc {
                op,
                src0,
                src1,
                ext: VopExt::Dpp(dpp),
            }))
        }
        Ext::Sdwa => {
            let first = o.operand(0)?;
            first.forbid_mods(name, 0)?;
            let sdst = match first.body {
                Body::Reg(Src::VccLo, 2) => None,
                Body::Reg(s, n) => {
                    first.check_width(name, 0, n, 2)?;
                    Some(s)
                }
                _ => {
                    return Err(err_operand(
                        name,
                        0,
                        first.text,
                        "must be `vcc` or an SGPR pair",
                    ))
                }
            };
            let (s0, s1) = (o.operand(1)?, o.operand(2)?);
            let src0 = s0.to_src(name, 1, ty, dw)?;
            let src1 = s1.to_src(name, 2, ty, dw)?;
            let sm = sdwa_mods(&mut m, false, false, true)?;
            m.finish()?;
            Ok(Inst::Vopc(Vopc {
                op,
                src0,
                src1,
                ext: VopExt::Sdwa(Sdwa {
                    dst_sel: SdwaSel::Dword,
                    dst_unused: SdwaUnused::Pad,
                    clamp: false,
                    omod: 0,
                    src0_sel: sm.src0_sel,
                    src0_sext: s0.sext,
                    src0_neg: s0.neg,
                    src0_abs: s0.abs,
                    src1_sel: sm.src1_sel,
                    src1_sext: s1.sext,
                    src1_neg: s1.neg,
                    src1_abs: s1.abs,
                    sdst,
                }),
            }))
        }
    }
}

// -- VOP3 / VOP3P / MAI --------------------------------------------------------

fn vop3_ty(op: Vop3Op, width: u8) -> Ty {
    if width == 2 {
        if op.name().contains("f64") {
            Ty::F64
        } else {
            Ty::I64
        }
    } else {
        Ty::B32
    }
}

fn vop3(op: Vop3Op, p: &Line) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let (wd, w0, w1, w2) = op.widths();
    let ws = [w0, w1, w2];
    let vop3b = op.is_vop3b();
    o.count(usize::from(wd > 0) + usize::from(vop3b) + ws.iter().filter(|&&w| w > 0).count())?;
    let mut i = 0;
    let dst = if wd > 0 {
        let d = o.reg(0, wd)?;
        i = 1;
        d
    } else {
        Src::Vgpr(0)
    };
    let sdst = if vop3b {
        let s = o.reg(i, 2)?;
        i += 1;
        Some(s)
    } else {
        None
    };
    let mut src: [Option<Src>; 3] = [None; 3];
    let (mut neg, mut abs) = (0u8, 0u8);
    for (j, &w) in ws.iter().enumerate() {
        if w == 0 {
            continue;
        }
        let opd = o.operand(i)?;
        if opd.sext {
            return Err(err_operand(name, i, opd.text, "VOP3 has no sext modifier"));
        }
        src[j] = Some(opd.to_src(name, i, vop3_ty(op, w), w)?);
        if opd.neg {
            neg |= 1 << j;
        }
        if opd.abs {
            abs |= 1 << j;
        }
        i += 1;
    }
    let sel_bits = op.op_sel_bits();
    let mut op_sel = 0u8;
    if let Some((b, _)) = m.bits("op_sel", sel_bits.len())? {
        for (k, &bit) in sel_bits.iter().enumerate() {
            if (b >> k) & 1 != 0 {
                op_sel |= 1 << bit;
            }
        }
    }
    let clamp = m.flag("clamp")?;
    let omod_bits = omod(&mut m)?;
    m.finish()?;
    Ok(Inst::Vop3(Vop3 {
        op,
        dst,
        sdst,
        src,
        abs,
        neg,
        clamp,
        omod: omod_bits,
        op_sel,
    }))
}

fn vop3p(op: Vop3pOp, p: &Line) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let (wd, a, b, c) = op.widths();
    let ws = [a, b, c];
    let n = if c == 0 { 2 } else { 3 };
    o.count(1 + n)?;
    let mix = name.starts_with("v_fma_mix");
    let dst = o.reg(0, wd)?;
    let mut src: [Option<Src>; 3] = [None; 3];
    let mut neg_lo = 0u8;
    for (j, &w) in ws.iter().enumerate() {
        if w == 0 {
            continue;
        }
        let opd = o.operand(1 + j)?;
        if opd.sext || opd.abs {
            return Err(err_operand(
                name,
                1 + j,
                opd.text,
                "packed math takes no sext/abs source modifier",
            ));
        }
        if opd.neg {
            if !mix {
                return Err(err_operand(
                    name,
                    1 + j,
                    opd.text,
                    "use the neg_lo:[..] / neg_hi:[..] modifiers on packed math",
                ));
            }
            neg_lo |= 1 << j;
        }
        src[j] = Some(opd.to_src(name, 1 + j, Ty::B32, w)?);
    }
    let default_hi: u8 = if mix { 0 } else { 7 };
    let hi = match m.bits("op_sel_hi", n)? {
        Some((bits, len)) => {
            let given = (1u8 << len) - 1;
            (bits & given) | (default_hi & !given)
        }
        None => default_hi,
    };
    let op_sel_hi = hi | if c == 0 { 4 } else { 0 };
    let op_sel = m.bits("op_sel", n)?.map_or(0, |(b, _)| b);
    neg_lo |= m.bits("neg_lo", n)?.map_or(0, |(b, _)| b);
    let neg_hi = m.bits("neg_hi", n)?.map_or(0, |(b, _)| b);
    let clamp = m.flag("clamp")?;
    m.finish()?;
    Ok(Inst::Vop3p(Vop3p {
        op,
        dst,
        src,
        op_sel,
        op_sel_hi,
        neg_lo,
        neg_hi,
        clamp,
    }))
}

fn mai(op: MaiOp, p: &Line) -> Result<Inst> {
    let (o, mut m) = p.split()?;
    let Some(shape) = op.shape() else {
        m.finish()?;
        o.count(2)?;
        return Ok(Inst::Mai(Mai {
            op,
            dst: o.reg(0, 1)?,
            src_a: o.src(1, Ty::B32, 1)?,
            src_b: None,
            src_c: None,
            cbsz: 0,
            abid: 0,
            blgp: 0,
        }));
    };
    o.count(4)?;
    let dst = o.reg(0, shape.cd_dwords)?;
    let src_a = o.reg(1, shape.a_dwords)?;
    let src_b = o.reg(2, shape.b_dwords)?;
    let src_c = o.src(3, Ty::B32, shape.cd_dwords)?;
    let cbsz = m.int("cbsz", 0, 7)?.map_or(0, |v| v as u8);
    let abid = m.int("abid", 0, 15)?.map_or(0, |v| v as u8);
    // F64 MFMA reuses the BLGP bits as A/B/C negation (`neg:[a,b,c]`).
    let blgp = if op == MaiOp::MfmaF64_16x16x4F64 {
        m.bits("neg", 3)?.map_or(0, |(b, _)| b)
    } else {
        m.int("blgp", 0, 7)?.map_or(0, |v| v as u8)
    };
    m.finish()?;
    Ok(Inst::Mai(Mai {
        op,
        dst,
        src_a,
        src_b: Some(src_b),
        src_c: Some(src_c),
        cbsz,
        abid,
        blgp,
    }))
}

// -- memory ----------------------------------------------------------------------

fn parse_swizzle(v: &str) -> std::result::Result<u16, String> {
    let inner =
        unwrap_call(v, "swizzle(").ok_or_else(|| format!("`{v}` is not swizzle(MODE, ...)"))?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    let int = |s: &str, lo: i128, hi: i128| {
        parse_int(s)
            .filter(|v| (lo..=hi).contains(v))
            .map(|v| v as u16)
            .ok_or_else(|| format!("`{s}` is not an integer in {lo}..={hi}"))
    };
    let arity = |n: usize| {
        if parts.len() == n {
            Ok(())
        } else {
            Err(format!(
                "swizzle({}) takes {} argument(s), got {}",
                parts[0],
                n - 1,
                parts.len() - 1
            ))
        }
    };
    match parts[0] {
        "QUAD_PERM" => {
            arity(5)?;
            let mut imm = 0x8000u16;
            for (k, s) in parts[1..].iter().enumerate() {
                imm |= int(*s, 0, 3)? << (2 * k);
            }
            Ok(imm)
        }
        "BITMASK_PERM" => {
            arity(2)?;
            let mask = parts[1].trim_matches('"');
            if mask.chars().count() != 5 {
                return Err(format!(
                    "BITMASK_PERM mask `{mask}` needs five characters of 0/1/p/i"
                ));
            }
            let (mut and, mut or, mut xor) = (0u16, 0u16, 0u16);
            for (k, ch) in mask.chars().enumerate() {
                let bit = 1u16 << (4 - k);
                match ch {
                    '0' => {}
                    '1' => or |= bit,
                    'p' => and |= bit,
                    'i' => {
                        and |= bit;
                        xor |= bit;
                    }
                    other => {
                        return Err(format!(
                            "BITMASK_PERM character `{other}` is not 0, 1, p or i"
                        ))
                    }
                }
            }
            Ok(and | or << 5 | xor << 10)
        }
        "SWAP" => {
            arity(2)?;
            let n = int(parts[1], 1, 16)?;
            if !n.is_power_of_two() {
                return Err(format!("SWAP group {n} is not 1, 2, 4, 8 or 16"));
            }
            Ok(0x1f | n << 10)
        }
        "REVERSE" => {
            arity(2)?;
            let n = int(parts[1], 2, 32)?;
            if !n.is_power_of_two() {
                return Err(format!("REVERSE group {n} is not 2, 4, 8, 16 or 32"));
            }
            Ok(0x1f | (n - 1) << 10)
        }
        "BROADCAST" => {
            arity(3)?;
            let group = int(parts[1], 2, 32)?;
            if !group.is_power_of_two() {
                return Err(format!("BROADCAST group {group} is not 2, 4, 8, 16 or 32"));
            }
            let lane = int(parts[2], 0, 31)?;
            if lane >= group {
                return Err(format!(
                    "BROADCAST lane {lane} is outside the group of {group}"
                ));
            }
            Ok((0x20 - group) | lane << 5)
        }
        "ROTATE" => {
            arity(3)?;
            Ok(0xc000 | int(parts[1], 0, 1)? << 10 | int(parts[2], 0, 31)? << 5)
        }
        "FFT" => {
            arity(2)?;
            Ok(0xe000 | int(parts[1], 0, 31)?)
        }
        other => Err(format!("unknown swizzle mode `{other}`")),
    }
}

fn ds(op: DsOp, p: &Line) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let (wv, w0, w1) = op.widths();
    let wa = u8::from(op.has_addr());
    o.count([wv, wa, w0, w1].iter().filter(|&&w| w > 0).count())?;
    let mut i = 0;
    let mut take = |w: u8| -> Result<Option<Src>> {
        if w == 0 {
            return Ok(None);
        }
        let s = o.reg(i, w)?;
        i += 1;
        Ok(Some(s))
    };
    let vdst = take(wv)?;
    let addr = take(wa)?;
    let data0 = take(w0)?;
    let data1 = take(w1)?;
    let (offset0, offset1) = if op.two_addr() {
        (
            m.int("offset0", 0, 255)?.map_or(0, |v| v as u8),
            m.int("offset1", 0, 255)?.map_or(0, |v| v as u8),
        )
    } else {
        let off = match m.value("offset")? {
            None => 0u32,
            Some(v) if op == DsOp::SwizzleB32 && v.starts_with("swizzle(") => {
                u32::from(parse_swizzle(v).map_err(|e| err_modifier(name, "offset", e))?)
            }
            Some(v) => {
                let n = parse_int(v).ok_or_else(|| {
                    err_modifier(name, "offset", format!("`{v}` is not an integer"))
                })?;
                if !(0..=0xffff).contains(&n) {
                    return Err(err_modifier(
                        name,
                        "offset",
                        format!("{n} is outside the 16-bit byte offset 0..=65535"),
                    ));
                }
                n as u32
            }
        };
        Ds::split_offset(off)?
    };
    let gds = m.flag("gds")?;
    m.finish()?;
    Ok(Inst::Ds(Ds {
        op,
        vdst,
        addr,
        data0,
        data1,
        offset0,
        offset1,
        gds,
    }))
}

fn flat(op: FlatOp, seg: FlatSeg, p: &Line) -> Result<Inst> {
    let (o, mut m) = p.split()?;
    let sc0 = m.flag("sc0")?;
    let sc1 = m.flag("sc1")?;
    let nt = m.flag("nt")?;
    let offset = m
        .int("offset", i128::from(i16::MIN), i128::from(i16::MAX))?
        .map_or(0, |v| v as i16);
    m.finish()?;
    let (vw, dw) = op.widths();
    // Loads define vdst, stores read data, atomics read data and return the
    // old value into vdst exactly when sc0 is set.
    let has_vdst = match (vw, dw) {
        (_, 0) => true,
        (0, _) => false,
        _ => sc0,
    };
    let has_saddr = seg != FlatSeg::Flat;
    o.count(usize::from(has_vdst) + 1 + usize::from(dw > 0) + usize::from(has_saddr))?;
    let mut i = 0;
    let vdst = if has_vdst {
        i += 1;
        Some(o.reg(0, vw)?)
    } else {
        None
    };
    let saddr = if has_saddr {
        o.reg_or_off(o.list.len() - 1, if seg == FlatSeg::Global { 2 } else { 1 })?
    } else {
        None
    };
    let addr_dwords: u8 = match seg {
        FlatSeg::Flat => 2,
        FlatSeg::Global if saddr.is_some() => 1,
        FlatSeg::Global => 2,
        FlatSeg::Scratch => 1,
    };
    let addr = o.reg_or_off(i, addr_dwords)?;
    i += 1;
    let data = if dw > 0 { Some(o.reg(i, dw)?) } else { None };
    Ok(Inst::Flat(Flat {
        op,
        seg,
        vdst,
        addr,
        data,
        saddr,
        offset,
        sc0,
        sc1,
        nt,
    }))
}

fn mubuf(op: MubufOp, p: &Line) -> Result<Inst> {
    let name = op.name();
    let (o, mut m) = p.split()?;
    let idxen = m.flag("idxen")?;
    let offen = m.flag("offen")?;
    let sc0 = m.flag("sc0")?;
    let sc1 = m.flag("sc1")?;
    let nt = m.flag("nt")?;
    if m.has("lds") {
        return Err(err_modifier(
            name,
            "lds",
            "MUBUF LDS forms are not modelled",
        ));
    }
    let offset = m.int("offset", 0, 0xfff)?.map_or(0, |v| v as u16);
    m.finish()?;
    o.count(4)?;
    let vdata = o.reg(0, op.data_dwords())?;
    let vaddr = o.reg_or_off(1, if offen && idxen { 2 } else { 1 })?;
    let srsrc = o.reg(2, 4)?;
    let soffset = o.src(3, Ty::B32, 1)?;
    Ok(Inst::Mubuf(Mubuf {
        op,
        vdata,
        vaddr,
        srsrc,
        soffset,
        offset,
        offen,
        idxen,
        sc0,
        sc1,
        nt,
    }))
}

// -- SOPP / SOPK operand syntax ----------------------------------------------------

/// `vmcnt(n)`, `expcnt(n)`, `lgkmcnt(n)` separated by `&`, `,` or space (an
/// unnamed counter is its maximum), or one integer SIMM16.
fn parse_waitcnt(arch: Arch, rest: &str) -> std::result::Result<u16, String> {
    let t = rest.trim();
    if t.is_empty() {
        return Err("s_waitcnt needs counters or an integer".into());
    }
    if let Some(v) = parse_int(t) {
        if !(0..=0xffff).contains(&v) {
            return Err(format!("{v} is outside the 16-bit SIMM16"));
        }
        return Ok(v as u16);
    }
    let caps = arch.caps();
    let (mut vm, mut exp, mut lgkm) = (caps.vmcnt_max, caps.expcnt_max, caps.lgkmcnt_max);
    let mut seen = [false; 3];
    for tok in t.split(|c: char| c == '&' || c == ',' || c.is_whitespace()) {
        if tok.is_empty() {
            continue;
        }
        let (counter, arg) = tok
            .split_once('(')
            .and_then(|(c, a)| a.strip_suffix(')').map(|a| (c, a)))
            .ok_or_else(|| format!("`{tok}` is not vmcnt(n), expcnt(n) or lgkmcnt(n)"))?;
        let n = parse_int(arg)
            .filter(|v| (0..=255).contains(v))
            .ok_or_else(|| format!("`{arg}` is not a counter value"))? as u8;
        let (slot, idx) = match counter {
            "vmcnt" => (&mut vm, 0),
            "expcnt" => (&mut exp, 1),
            "lgkmcnt" => (&mut lgkm, 2),
            other => return Err(format!("unknown wait counter `{other}`")),
        };
        if seen[idx] {
            return Err(format!("{counter} is given twice"));
        }
        seen[idx] = true;
        *slot = n;
    }
    waitcnt_imm(arch, vm, exp, lgkm)
}

/// `hwreg(NAME_or_id[, offset, size])` -> SIMM16 `{size-1, offset, id}`.
fn parse_hwreg(t: &str) -> std::result::Result<u16, String> {
    let inner = unwrap_call(t, "hwreg(")
        .ok_or_else(|| format!("`{t}` is not hwreg(id[, offset, size])"))?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() != 1 && parts.len() != 3 {
        return Err(format!(
            "`{t}`: hwreg takes an id, or an id with offset and size"
        ));
    }
    let id = match (0u16..64).find(|&i| hwreg_name(i) == Some(parts[0])) {
        Some(i) => i,
        None => parse_int(parts[0])
            .filter(|v| (0..64).contains(v))
            .map(|v| v as u16)
            .ok_or_else(|| {
                format!(
                    "`{}` is not a gfx942 hardware register name or an id 0..=63",
                    parts[0]
                )
            })?,
    };
    let (offset, size) = if parts.len() == 3 {
        let off = parse_int(parts[1])
            .filter(|v| (0..=31).contains(v))
            .ok_or_else(|| format!("hwreg offset `{}` is not 0..=31", parts[1]))?;
        let size = parse_int(parts[2])
            .filter(|v| (1..=32).contains(v))
            .ok_or_else(|| format!("hwreg size `{}` is not 1..=32", parts[2]))?;
        (off as u16, size as u16)
    } else {
        (0, 32)
    };
    Ok(id | offset << 6 | (size - 1) << 11)
}

/// `sendmsg(MSG[, OP[, stream]])` -> SIMM16 `{stream[9:8], op[6:4], id[3:0]}`,
/// or one integer SIMM16.
fn parse_sendmsg(t: &str) -> std::result::Result<u16, String> {
    if let Some(v) = parse_int(t) {
        if !(0..=0xffff).contains(&v) {
            return Err(format!("{v} is outside the 16-bit SIMM16"));
        }
        return Ok(v as u16);
    }
    let inner = unwrap_call(t, "sendmsg(")
        .ok_or_else(|| format!("`{t}` is not sendmsg(MSG[, OP[, stream]]) or an integer"))?;
    let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
    if parts.len() > 3 {
        return Err(format!("`{t}`: sendmsg takes at most three arguments"));
    }
    let id = match parts[0] {
        "MSG_INTERRUPT" => 1,
        "MSG_GS" => 2,
        "MSG_GS_DONE" => 3,
        "MSG_SAVEWAVE" => 4,
        "MSG_STALL_WAVE_GEN" => 5,
        "MSG_HALT_WAVES" => 6,
        "MSG_ORDERED_PS_DONE" => 7,
        "MSG_EARLY_PRIM_DEALLOC" => 8,
        "MSG_GS_ALLOC_REQ" => 9,
        "MSG_GET_DOORBELL" => 10,
        "MSG_SYSMSG" => 15,
        other => parse_int(other)
            .filter(|v| (0..=15).contains(v))
            .map(|v| v as u16)
            .ok_or_else(|| format!("`{other}` is not a message name or id 0..=15"))?,
    };
    let op = match parts.get(1).copied() {
        None => 0,
        Some("GS_OP_NOP") => 0,
        Some("GS_OP_CUT") => 1,
        Some("GS_OP_EMIT") => 2,
        Some("GS_OP_EMIT_CUT") => 3,
        Some("SYSMSG_OP_ECC_ERR_INTERRUPT") => 1,
        Some("SYSMSG_OP_REG_RD") => 2,
        Some("SYSMSG_OP_TTRACE_PC") => 4,
        Some(other) => parse_int(other)
            .filter(|v| (0..=7).contains(v))
            .map(|v| v as u16)
            .ok_or_else(|| format!("`{other}` is not a message operation or 0..=7"))?,
    };
    let stream = match parts.get(2).copied() {
        None => 0,
        Some(s) => parse_int(s)
            .filter(|v| (0..=3).contains(v))
            .map(|v| v as u16)
            .ok_or_else(|| format!("`{s}` is not a GS stream 0..=3"))?,
    };
    Ok(id | op << 4 | stream << 8)
}
