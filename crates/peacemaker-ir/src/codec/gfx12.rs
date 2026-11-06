// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Table-gated gfx1201 decoder and encoder. No provenance bytes participate in encoding.
use super::forms::{self, Field};
use crate::{
    inst::{Arch, Form, FormFields, Inst, NamedField},
    isa::{self, FieldClass, OpRow},
    operand::{
        CachePolicy, CacheScope, DelayAluHint, Dpp, Half, ImmField, InlineConst, Modifiers, Msg,
        Omod, Operand, Special, VmemToken,
    },
    provenance::Provenance,
    reg::{Kind, RegRef},
    wait::{Counter, WaitImm},
};
use smallvec::SmallVec;

/// Encoding tables reserve 64-bit lane-mask slots even in wave32. Keep the
/// raw codec wave-neutral, but consumers with kernel metadata must use this
/// width for explicit scalar masks, not for genuine scalar data pairs.
pub fn operand_register(
    arch: Arch,
    wave: crate::inst::Wave,
    inst: &Inst,
    index: usize,
) -> Option<RegRef> {
    let mut reg = match inst.operands.get(index)? {
        Operand::Reg(r) | Operand::Half(r, _) => *r,
        _ => return None,
    };
    if reg.kind == Kind::S && crate::effects::lane_mask_role(arch, inst, index).is_some() {
        reg.len = if wave == crate::inst::Wave::Wave32 {
            1
        } else {
            2
        };
    }
    Some(reg)
}
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("unrecognized machine instruction at byte offset {offset}: {reason}")]
    Rejected { offset: usize, reason: String },
}
fn reject(reason: impl Into<String>) -> DecodeError {
    DecodeError::Rejected {
        offset: 0,
        reason: reason.into(),
    }
}
fn value(field: Option<Field>, words: &[u32]) -> u32 {
    field.map_or(0, |f| f.value(words))
}
fn assign(field: Field, words: &mut [u32], v: u32) -> Result<(), DecodeError> {
    field
        .set(words, v)
        .map_err(|name| reject(format!("{name} does not fit its encoding field")))
}
fn select(
    code: u32,
    width: u8,
    scalar_dest: bool,
    literal: Option<u32>,
) -> Result<Operand, DecodeError> {
    let reg = |kind, base, len| Operand::Reg(RegRef { kind, base, len });
    if code <= 105 {
        return Ok(reg(Kind::S, code as u16, width));
    }
    if code >= 256 && code <= 511 {
        return Ok(reg(Kind::V, (code - 256) as u16, width));
    }
    if scalar_dest && code <= 127 {
        return Ok(match code {
            106 => Operand::Special(Special::VccLo),
            107 => Operand::Special(Special::VccHi),
            124 => Operand::Special(Special::Null),
            125 => Operand::Special(Special::M0),
            126 => Operand::Special(Special::ExecLo),
            127 => Operand::Special(Special::ExecHi),
            _ => return Err(reject(format!("unknown scalar destination {code:#x}"))),
        });
    }
    Ok(match code {
        106 => Operand::Special(Special::VccLo),
        107 => Operand::Special(Special::VccHi),
        108 => Operand::Special(Special::Ttmp(0)),
        109 => Operand::Special(Special::Ttmp(1)),
        110..=123 => Operand::Reg(RegRef {
            kind: Kind::Ttmp,
            base: (code - 108) as u16,
            len: width,
        }),
        // gfx11+ selectors: 124 is `null`, 125 is `m0` (sources and destinations alike).
        124 => Operand::Special(Special::Null),
        125 => Operand::Special(Special::M0),
        126 => Operand::Special(Special::ExecLo),
        127 => Operand::Special(Special::ExecHi),
        128..=192 => Operand::Inline(InlineConst::Integer((code - 128) as i8)),
        193..=208 => Operand::Inline(InlineConst::Integer((192 - code as i32) as i8)),
        // Aperture sources (RDNA3/RDNA4 SSRC encodings).
        235 => Operand::Special(Special::SrcSharedBase),
        236 => Operand::Special(Special::SrcSharedLimit),
        237 => Operand::Special(Special::SrcPrivateBase),
        238 => Operand::Special(Special::SrcPrivateLimit),
        240..=247 => Operand::Inline(InlineConst::FloatBits(match code {
            240 => 0x3f000000,
            241 => 0xbf000000,
            242 => 0x3f800000,
            243 => 0xbf800000,
            244 => 0x40000000,
            245 => 0xc0000000,
            246 => 0x40800000,
            _ => 0xc0800000,
        })),
        248 => Operand::Inline(InlineConst::InvTwoPi),
        255 => Operand::Literal(
            literal.ok_or_else(|| reject("literal selector without literal word"))?,
        ),
        _ => return Err(reject(format!("unknown source selector {code:#x}"))),
    })
}
fn selector(operand: &Operand, width: u8) -> Result<u32, DecodeError> {
    match operand {
        Operand::Reg(r) | Operand::Half(r, _) => {
            if r.len != width {
                return Err(reject(format!(
                    "register width mismatch: {} instead of {width}",
                    r.len
                )));
            }
            Ok(match r.kind {
                Kind::V => u32::from(r.base) + 256,
                Kind::S => u32::from(r.base),
                Kind::Ttmp => u32::from(r.base) + 108,
            })
        }
        Operand::Special(s) => Ok(match s {
            Special::VccLo | Special::Vcc => 106,
            Special::VccHi => 107,
            Special::Ttmp(n) => u32::from(*n) + 108,
            Special::M0 => 125,
            Special::Null => 124,
            Special::ExecLo | Special::Exec => 126,
            Special::ExecHi => 127,
            Special::SrcSharedBase => 235,
            Special::SrcSharedLimit => 236,
            Special::SrcPrivateBase => 237,
            Special::SrcPrivateLimit => 238,
            _ => return Err(reject(format!("unencodable special register {s:?}"))),
        }),
        Operand::Inline(InlineConst::Integer(n)) if (0..=64).contains(n) => Ok((*n as u32) + 128),
        Operand::Inline(InlineConst::Integer(n)) if (-16..=-1).contains(n) => {
            Ok((192 - i32::from(*n)) as u32)
        }
        Operand::Inline(InlineConst::FloatBits(bits)) => [
            0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40800000,
            0xc0800000,
        ]
        .iter()
        .position(|x| x == bits)
        .map(|i| i as u32 + 240)
        .ok_or_else(|| reject("unencodable inline float")),
        Operand::Inline(InlineConst::InvTwoPi) => Ok(248),
        Operand::Literal(_) => Ok(255),
        _ => Err(reject(format!(
            "operand is not a register/source selector: {operand:?}"
        ))),
    }
}
fn reg(kind: Kind, base: u32, bits: u16) -> Operand {
    Operand::Reg(RegRef {
        kind,
        base: base as u16,
        len: (bits / 32).max(1) as u8,
    })
}
fn operand(
    arch: Arch,
    name: &str,
    bits: u16,
    code: u32,
    literal: Option<u32>,
    row: &OpRow,
    words: &[u32],
) -> Result<Operand, DecodeError> {
    let form = row.form;
    let width = (bits / 32).max(1) as u8;
    // A 64-bit scalar-ALU operand at selector 106 is the whole VCC pair (s[106:107]), not VCC_LO: the wave32 lane-mask
    // slots of VALU compares keep `vcc_lo`, but SALU 64-bit data (`s_add_nc_u64 vcc, s[4:5], vcc`) names both halves.
    if bits == 64
        && code == 106
        && row.name.starts_with("s_")
        && matches!(name, "SDST" | "SSRC0" | "SSRC1")
        && matches!(form, Form::Sop1 | Form::Sop2 | Form::Sopc)
    {
        return Ok(Operand::Special(Special::Vcc));
    }
    if name == "SIMM16" {
        if row.name == "s_sendmsg" && code == 3 {
            return Ok(Operand::SendMsg(Msg { id: 3, op: 0 }));
        }
        return Ok(Operand::Imm(if form == Form::Sopk {
            ImmField::Sopk(code as i16)
        } else {
            ImmField::Sopp(code as i16)
        }));
    }
    // `s_sendmsg_rtn_*` carries its message id in the SSRC0 field (not a source selector).
    if name == "SSRC0" && row.name.starts_with("s_sendmsg_rtn") {
        return Ok(Operand::SendMsg(Msg {
            id: code as u8,
            op: 0,
        }));
    }
    if name == "SOFFSET" && form == Form::Smem && code == 124 {
        return Ok(Operand::Imm(ImmField::SmemOffset(
            field_value(arch, "IOFFSET", row, words) as i32,
        )));
    }
    if name == "SBASE" {
        // The pair selector 53 is s[106:107]: VCC (LO/HI), not an SGPR (the SGPR file ends at s105).
        if code * 2 == 106 {
            return Ok(Operand::Special(Special::Vcc));
        }
        return Ok(reg(Kind::S, code * 2, bits));
    }
    if name == "RSRC" {
        return Ok(reg(
            Kind::S,
            code * if arch == Arch::Gfx1201 { 1 } else { 4 },
            bits,
        ));
    }
    if name == "SADDR" && code == 124 {
        return Ok(Operand::Vmem(VmemToken::Off));
    }
    // With SVE clear the VADDR byte is unused; only its canonical zero
    // round-trips through `off`, so any other value is refused.
    if name == "VADDR"
        && form == Form::Vmem(crate::inst::VmemForm::Scratch)
        && field_value(arch, "SVE", row, words) == 0
    {
        if code != 0 {
            return Err(reject("scratch VADDR must be zero when SVE is clear"));
        }
        return Ok(Operand::Vmem(VmemToken::Off));
    }
    // MUBUF with OFFEN and IDXEN both clear does not read VADDR; the byte is
    // unused and only its canonical zero round-trips through `off`.
    if name == "VADDR"
        && form == Form::Vmem(crate::inst::VmemForm::Buffer)
        && field_value(arch, "OFFEN", row, words) == 0
        && field_value(arch, "IDXEN", row, words) == 0
    {
        if code != 0 {
            return Err(reject(
                "buffer VADDR must be zero when OFFEN and IDXEN are clear",
            ));
        }
        return Ok(Operand::Vmem(VmemToken::Off));
    }
    if name == "SRC0" && matches!(form, Form::Vop1Dpp | Form::Vop1Dpp8 | Form::Vop2Dpp) {
        return Ok(reg(Kind::V, code, bits));
    }
    // True16 e32 encodes the half in bit 7 of the VGPR selector. The source
    // selector additionally has bit 8 set to select the VGPR bank.
    if bits == 16
        && matches!(form, Form::Vop1 | Form::Vop2 | Form::Vopc)
        && (name == "VDST" || name == "VSRC1" || name == "SRC0" && code >= 256)
    {
        let r = RegRef {
            kind: Kind::V,
            base: (code & 0x7f) as u16,
            len: 1,
        };
        return Ok(Operand::Half(
            r,
            if code & 0x80 != 0 { Half::Hi } else { Half::Lo },
        ));
    }
    let scalar_dest = name == "SDST"
        || name == "SDATA"
        || name == "VDST"
            && (row.name.starts_with("v_cmp")
                || row.name.starts_with("v_s_")
                || row.name == "v_readlane_b32"
                || row.name == "v_readfirstlane_b32");
    if scalar_dest {
        return select(code, width, true, literal);
    }
    if name == "VDST"
        || name == "VDATA"
        || name == "VSRC"
        || name == "VADDR"
        || name == "ADDR"
        || name.starts_with("DATA")
        || name.starts_with("VDST")
        || name.starts_with("VSRC")
    {
        let r = RegRef {
            kind: Kind::V,
            base: code as u16,
            len: width,
        };
        if bits == 16 && name == "VDST" {
            return Ok(Operand::Half(
                r,
                if field_value(arch, "OPSEL", row, words) & 0x8 != 0 {
                    Half::Hi
                } else {
                    Half::Lo
                },
            ));
        }
        if form == Form::Vmem(crate::inst::VmemForm::Buffer)
            && name == "VADDR"
            && field_value(arch, "IDXEN", row, words) ^ field_value(arch, "OFFEN", row, words) == 1
        {
            return Ok(Operand::Reg(RegRef { len: 1, ..r }));
        }
        // GLOBAL with an SGPR base consumes a 32-bit vector offset on both
        // RDNA3 and RDNA4. With SADDR=off it consumes a 64-bit vector address.
        if form == Form::Vmem(crate::inst::VmemForm::Global)
            && matches!(name, "VADDR" | "ADDR")
            && field_value(arch, "SADDR", row, words) != 124
        {
            return Ok(Operand::Reg(RegRef { len: 1, ..r }));
        }
        if form == Form::Vmem(crate::inst::VmemForm::Scratch) && name == "VADDR" {
            return Ok(Operand::Reg(RegRef { len: 1, ..r }));
        }
        return Ok(Operand::Reg(r));
    }
    // SRC_SCC (253) is a precise 32-bit scalar source; it is never a destination
    // (SDST is 7-bit and the 8-bit VOP3 compare destination is rejected by `select`).
    if code == 253 {
        if matches!(form, Form::Sop1 | Form::Sop2 | Form::Sopc) && bits == 32 {
            return Ok(Operand::Special(Special::Scc));
        }
        return Err(reject(
            "SCC source selector 0xfd is decoded only for 32-bit SOP1/SOP2/SOPC sources",
        ));
    }
    let src = select(code, width, false, literal)?;
    if bits == 16 {
        if let Operand::Reg(r) = &src {
            if r.kind == Kind::V {
                // OPSEL bit n selects the half of source n (SRC0 bit 0, SRC1 bit 1, SRC2 bit 2).
                let bit = match name {
                    "SRC1" => 2,
                    "SRC2" => 4,
                    _ => 1,
                };
                return Ok(Operand::Half(
                    *r,
                    if field_value(arch, "OPSEL", row, words) & bit != 0 {
                        Half::Hi
                    } else {
                        Half::Lo
                    },
                ));
            }
        }
    }
    Ok(src)
}
fn encoded_operand(
    arch: Arch,
    name: &str,
    bits: u16,
    op: &Operand,
    form: Form,
) -> Result<u32, DecodeError> {
    match (name, op) {
        ("SIMM16", Operand::SendMsg(Msg { id: 3, op: 0 })) => Ok(3),
        ("SSRC0", Operand::SendMsg(Msg { id, op: 0 })) if form == Form::Sop1 => Ok(u32::from(*id)),
        ("SIMM16", Operand::Imm(ImmField::Sopp(n) | ImmField::Sopk(n))) => Ok((*n as u16).into()),
        ("SOFFSET", Operand::Imm(ImmField::SmemOffset(_))) if form == Form::Smem => Ok(124),
        ("SBASE", Operand::Reg(r)) if r.kind == Kind::S && r.base % 2 == 0 => {
            Ok(u32::from(r.base / 2))
        }
        ("SBASE", Operand::Special(Special::Vcc | Special::VccLo)) => Ok(53),
        ("SADDR", Operand::Vmem(VmemToken::Off)) => Ok(124),
        ("VADDR", Operand::Vmem(VmemToken::Off))
            if matches!(
                form,
                Form::Vmem(crate::inst::VmemForm::Scratch | crate::inst::VmemForm::Buffer)
            ) =>
        {
            Ok(0)
        }
        ("RSRC", Operand::Reg(r))
            if r.kind == Kind::S && (arch == Arch::Gfx1201 || r.base % 4 == 0) =>
        {
            Ok(u32::from(if arch == Arch::Gfx1201 {
                r.base
            } else {
                r.base / 4
            }))
        }
        ("SRC0", Operand::Reg(r))
            if matches!(form, Form::Vop1Dpp | Form::Vop1Dpp8 | Form::Vop2Dpp)
                && r.kind == Kind::V
                && r.len == 1 =>
        {
            Ok(u32::from(r.base))
        }
        ("VDST" | "VSRC1" | "SRC0", Operand::Half(r, half))
            if bits == 16 && matches!(form, Form::Vop1 | Form::Vop2 | Form::Vopc) =>
        {
            if r.kind != Kind::V || r.base >= 128 || r.len != 1 {
                return Err(reject("true16 e32 half requires a VGPR below v128"));
            }
            Ok(u32::from(r.base)
                + u32::from(name == "SRC0") * 256
                + u32::from(*half == Half::Hi) * 128)
        }
        (_, Operand::Reg(r))
            if name.starts_with('V') || name.starts_with("DATA") || name == "ADDR" =>
        {
            if r.kind != Kind::V && !(name == "VDST" && (form == Form::Vop1 || form == Form::Vop3))
            {
                return Err(reject("vector operand has wrong register bank"));
            }
            let short_address = r.len == 1
                && ((name == "VADDR"
                    && matches!(
                        form,
                        Form::Vmem(crate::inst::VmemForm::Buffer | crate::inst::VmemForm::Scratch)
                    ))
                    || (matches!(name, "VADDR" | "ADDR")
                        && form == Form::Vmem(crate::inst::VmemForm::Global)));
            if r.len != (bits / 32).max(1) as u8 && !short_address {
                return Err(reject(format!("{name} register width mismatch")));
            }
            Ok(u32::from(r.base))
        }
        ("VDST", Operand::Special(s)) if form == Form::Vop3 => {
            selector(&Operand::Special(*s), (bits / 32).max(1) as u8)
        }
        ("VDST" | "SRC0", Operand::Half(r, _)) => Ok(if name == "VDST" {
            u32::from(r.base)
        } else {
            u32::from(r.base) + 256
        }),
        (n, Operand::Special(Special::Scc))
            if n.starts_with("SSRC")
                && bits == 32
                && matches!(form, Form::Sop1 | Form::Sop2 | Form::Sopc) =>
        {
            Ok(253)
        }
        (_, _) => selector(op, (bits / 32).max(1) as u8),
    }
}
fn width(arch: Arch, row: &OpRow, words: &[u32]) -> Result<usize, DecodeError> {
    let base = match row.form {
        Form::Sop1
        | Form::Sop2
        | Form::Sopc
        | Form::Sopk
        | Form::Sopp
        | Form::Vop1
        | Form::Vop2
        | Form::Vopc => 1,
        Form::Vmem(_) if arch == Arch::Gfx1201 => 3,
        _ => 2,
    };
    if words.len() < base {
        return Err(reject(format!(
            "truncated {} instruction (need {base} words)",
            row.name
        )));
    }
    let literal = match row.form {
        Form::Sop1
        | Form::Sop2
        | Form::Sopc
        | Form::Vop1
        | Form::Vop2
        | Form::Vopc
        | Form::Vop3
        | Form::Vop3p
        | Form::Vopd => {
            row.slots(false).any(|(name, _)| {
                let source = name.starts_with("SRC") || name.starts_with("SSRC");
                source
                    && forms::field_for(arch, row.form, name)
                        .is_some_and(|f| value(Some(f), words) == 255)
            }) || row.grammar.contains("literal@last")
                && matches!(
                    row.name,
                    "s_fmaak_f32"
                        | "s_fmamk_f32"
                        | "v_fmaak_f32"
                        | "v_fmamk_f32"
                        | "v_dual_fmaak_f32"
                        | "v_dual_fmamk_f32"
                )
                || row.form == Form::Vopd
                    && (forms::field_for(arch, row.form, "SRCY0")
                        .is_some_and(|f| value(Some(f), words) == 255)
                        || isa::table(arch).iter().any(|y| {
                            y.form == Form::Vopd
                                && y.op.id == ((words[0] >> 17) & 31) as u16
                                && y.grammar.contains("LITERAL:")
                        }))
        }
        _ => false,
    };
    let size = base + usize::from(literal);
    if size > 3 || words.len() < size {
        return Err(reject(format!(
            "truncated or overlong {} instruction (need {size} words)",
            row.name
        )));
    }
    Ok(size)
}
fn field_value(arch: Arch, name: &str, row: &OpRow, words: &[u32]) -> u32 {
    if name == "SDST" && row.form == Form::Vop3 {
        return (words[0] >> 8) & 127;
    }
    value(forms::field_for(arch, row.form, name), words)
}
fn encode_field(
    arch: Arch,
    name: &str,
    row: &OpRow,
    words: &mut [u32],
    value: u32,
) -> Result<(), DecodeError> {
    if name == "SDST" && row.form == Form::Vop3 {
        return assign(Field::new("SDST", 8, 7), words, value);
    }
    assign(
        forms::field_for(arch, row.form, name)
            .ok_or_else(|| reject(format!("unknown bitfield {name} in {}", row.name)))?,
        words,
        value,
    )
}
/// Typed cache policy of an SMEM/VMEM word; default for every other form.
fn cache_policy(arch: Arch, row: &OpRow, words: &[u32]) -> CachePolicy {
    if row.form != Form::Smem && !matches!(row.form, Form::Vmem(_)) {
        return CachePolicy::default();
    }
    if arch == Arch::Gfx1201 {
        CachePolicy {
            th: field_value(arch, "TH", row, words) as u8,
            scope: field_value(arch, "SCOPE", row, words) as u8,
            nv: field_value(arch, "NV", row, words) != 0,
            ..CachePolicy::default()
        }
    } else {
        CachePolicy {
            glc: field_value(arch, "GLC", row, words) != 0,
            slc: field_value(arch, "SLC", row, words) != 0,
            dlc: field_value(arch, "DLC", row, words) != 0,
            ..CachePolicy::default()
        }
    }
}
fn special_mods(arch: Arch, row: &OpRow, words: &[u32], mods: &mut Modifiers) {
    if row.form == Form::Vop3 {
        let carry = row.grammar.contains("SDST:");
        if !carry {
            mods.abs = field_value(arch, "ABS", row, words) as u8;
            mods.op_sel = field_value(arch, "OPSEL", row, words) as u8;
        }
        mods.neg = field_value(arch, "NEG", row, words) as u8;
        mods.clamp = field_value(arch, "CLAMP", row, words) != 0;
        mods.omod = match field_value(arch, "OMOD", row, words) {
            1 => Omod::Mul2,
            2 => Omod::Mul4,
            3 => Omod::Div2,
            _ => Omod::None,
        };
    } else if row.form == Form::Vop3p {
        mods.neg_lo = field_value(arch, "NEG", row, words) as u8;
        mods.neg_hi = field_value(arch, "NEG_HI", row, words) as u8;
        mods.op_sel = field_value(arch, "OPSEL", row, words) as u8;
        mods.op_sel_hi = (field_value(arch, "OPSEL_HI_LO", row, words)
            | field_value(arch, "OPSEL_HI_2", row, words) << 2) as u8;
        mods.clamp = field_value(arch, "CLAMP", row, words) != 0;
    }
    if matches!(row.form, Form::Vop1Dpp | Form::Vop2Dpp) {
        mods.dpp = Some(Dpp {
            ctrl: field_value(arch, "DPP_CTRL", row, words) as u16,
            row_mask: field_value(arch, "ROW_MASK", row, words) as u8,
            bank_mask: field_value(arch, "BANK_MASK", row, words) as u8,
            bound_ctrl: field_value(arch, "BOUND_CTRL", row, words) != 0,
        });
        mods.neg = (field_value(arch, "SRC0_NEG", row, words)
            | field_value(arch, "SRC1_NEG", row, words) << 1) as u8;
        mods.abs = (field_value(arch, "SRC0_ABS", row, words)
            | field_value(arch, "SRC1_ABS", row, words) << 1) as u8;
    }
    mods.cpol = cache_policy(arch, row, words);
    if matches!(arch, Arch::Gfx1100 | Arch::Gfx1151) && row.name.starts_with("s_waitcnt") {
        let bits = words[0] as u16;
        let mut wait = WaitImm::default();
        match row.name {
            "s_waitcnt" => {
                for (counter, count, max) in [
                    (Counter::Vm, (bits >> 10) as u8 & 63, 63),
                    (Counter::Exp, bits as u8 & 7, 7),
                    (Counter::Lgkm, (bits >> 4) as u8 & 63, 63),
                ] {
                    if count != max {
                        wait.per_counter[counter as usize] = Some(count);
                    }
                }
            }
            "s_waitcnt_vscnt" => wait.per_counter[Counter::Vs as usize] = Some(bits as u8 & 63),
            "s_waitcnt_vmcnt" => wait.per_counter[Counter::Vm as usize] = Some(bits as u8 & 63),
            "s_waitcnt_lgkmcnt" => wait.per_counter[Counter::Lgkm as usize] = Some(bits as u8 & 63),
            _ => {}
        }
        mods.wait = Some(wait);
    }
    if arch == Arch::Gfx1201
        && row.form == Form::Sopp
        && row.name.starts_with("s_wait_")
        && row.name != "s_wait_alu"
    {
        let raw = words[0] as u16;
        let mut wait = WaitImm::default();
        if row.name.ends_with("loadcnt_dscnt") || row.name.ends_with("storecnt_dscnt") {
            let counter = if row.name.ends_with("loadcnt_dscnt") {
                Counter::Load
            } else {
                Counter::Store
            };
            wait.per_counter[counter as usize] = Some((raw >> 8) as u8 & 63);
            wait.per_counter[Counter::Ds as usize] = Some(raw as u8 & 63);
        } else {
            let counter = if row.name.ends_with("loadcnt") {
                Counter::Load
            } else if row.name.ends_with("dscnt") {
                Counter::Ds
            } else if row.name.ends_with("storecnt") {
                Counter::Store
            } else if row.name.ends_with("kmcnt") {
                Counter::Km
            } else if row.name.ends_with("samplecnt") {
                Counter::Sample
            } else if row.name.ends_with("bvhcnt") {
                Counter::Bvh
            } else {
                Counter::Exp
            };
            wait.per_counter[counter as usize] = Some(raw as u8);
        }
        mods.wait = Some(wait);
    }
    if row.name == "s_clause" {
        mods.clause = Some(words[0] as u8);
    }
    if row.name == "s_delay_alu" {
        let imm = words[0] as u16;
        mods.delay = Some(DelayAluHint {
            instid0: (imm & 15) as u8,
            instskip: ((imm >> 4) & 7) as u8,
            instid1: ((imm >> 7) & if arch == Arch::Gfx1201 { 0x1ff } else { 0xf }) as u8,
        });
    }
}
fn fields_from(
    arch: Arch,
    row: &OpRow,
    words: &[u32],
    consumed: &mut [u32; 3],
) -> Result<FormFields, DecodeError> {
    let mut ignored = SmallVec::new();
    let mut honored = SmallVec::new();
    if matches!(row.form, Form::Vop3 | Form::Vop3p)
        && !row.grammar.contains("SRC2:")
        && !row.benign_src2.is_empty()
    {
        let src2 = field_value(arch, "SRC2", row, words);
        if src2 == 255 {
            return Err(reject("dangerous-fill src2_unused=0xff (literal selector)"));
        }
        consumed[1] |= forms::field_for(arch, row.form, "SRC2")
            .expect("VOP3 SRC2 layout")
            .mask();
        if row.fields.len() == 1 {
            return Ok(FormFields::Vop3b {
                src2_unused: src2 as u16,
            });
        }
        ignored.push(NamedField {
            name: "src2_unused",
            value: src2,
        });
    }
    for rule in &row.fields {
        if rule.name == "src2_unused" {
            continue;
        }
        if rule.name == "src1_unused" {
            let src1 = field_value(arch, "SRC1", row, words);
            if !rule.allowed.contains(&src1) {
                return Err(reject(format!("unknown don't-care src1_unused={src1:#x}")));
            }
            consumed[1] |= forms::field_for(arch, row.form, "SRC1")
                .expect("VOP3 SRC1")
                .mask();
            ignored.push(NamedField {
                name: rule.name,
                value: src1,
            });
            continue;
        }
        let (wi, raw) = match rule.name {
            "w0_extra" | "smem_len" | "wait_unused" | "vbuffer_tfe" | "vbuffer_nv"
            | "global_nv" => (0, words[0] & rule.mask),
            "w1_extra" | "vsrc_unused" | "vbuffer_format" | "vbuffer_offen" | "vbuffer_idxen"
            | "vbuffer_scope" | "vbuffer_th" | "global_sve" | "global_scope" | "global_th"
            | "dpp_fi" | "dpp8_sel" => (1, words[1] & rule.mask),
            "neg" => (1, (words[1] >> 29) & rule.mask),
            "abs" => (0, (words[0] >> 8) & rule.mask),
            "omod" => (1, (words[1] >> 27) & rule.mask),
            "op_sel" => (0, (words[0] >> 11) & rule.mask),
            "op_sel_hi" => (
                1,
                ((words[1] >> 27) & 3 | (((words[0] >> 14) & 1) << 2)) & rule.mask,
            ),
            other => return Err(reject(format!("unsupported table field {other}"))),
        };
        if matches!(
            rule.name,
            "w0_extra"
                | "w1_extra"
                | "smem_len"
                | "wait_unused"
                | "vsrc_unused"
                | "dpp_fi"
                | "dpp8_sel"
        ) || rule.name.starts_with("vbuffer_")
            || rule.name.starts_with("global_")
        {
            consumed[wi] |= rule.mask;
        }
        if rule.class == FieldClass::Ignored && !rule.allowed.contains(&raw) {
            return Err(reject(format!("unknown don't-care {}={raw:#x}", rule.name)));
        }
        let item = NamedField {
            name: rule.name,
            value: raw,
        };
        if rule.class == FieldClass::Honored {
            honored.push(item);
        } else {
            ignored.push(item);
        }
    }
    if ignored.is_empty() && honored.is_empty() {
        Ok(FormFields::None)
    } else {
        Ok(FormFields::Bits { ignored, honored })
    }
}
fn row_for(arch: Arch, words: &[u32]) -> Result<&'static OpRow, DecodeError> {
    let w0 = *words
        .first()
        .ok_or_else(|| reject("empty instruction stream"))?;
    isa::table(arch)
        .iter()
        .find(|r| {
            let Some((mask, prefix)) = forms::prefix_for(arch, r.form) else {
                return false;
            };
            if w0 & mask != prefix
                || Some(r.op.id as u32) != forms::opcode_for(arch, r.form).map(|f| f.value(words))
            {
                return false;
            }
            // The SRC0 selector of a DPP form names the extension: 0xfa DPP16, 0xe9 DPP8 (0xea, DPP8FI, has no row).
            let selector = w0 & 0x1ff;
            let wanted = match r.form {
                Form::Vop1Dpp | Form::Vop2Dpp => Some(0xfa),
                Form::Vop1Dpp8 => Some(0xe9),
                _ => None,
            };
            if matches!(
                r.form,
                Form::Vop1 | Form::Vop2 | Form::Vop1Dpp | Form::Vop1Dpp8 | Form::Vop2Dpp
            ) && wanted.map_or(matches!(selector, 0xfa | 0xe9 | 0xea), |w| selector != w)
            {
                return false;
            }
            r.form != Form::Vopd
                || words.len() > 1
                    && isa::table(arch)
                        .iter()
                        .any(|y| y.form == Form::Vopd && y.op.id == ((words[0] >> 17) & 31) as u16)
        })
        .ok_or_else(|| reject(format!("undefined {arch:?} opcode/form for {w0:#010x}")))
}
/// Decode one table-declared instruction; caller owns its stream offset.
pub fn decode(words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    decode_for(Arch::Gfx1201, words)
}
/// Decode according to the target's opcode and field-layout tables.
pub fn decode_for(arch: Arch, words: &[u32]) -> Result<(Inst, usize), DecodeError> {
    let row = row_for(arch, words)?;
    let n = width(arch, row, words)?;
    let words = &words[..n];
    let literal = if n > match row.form {
        Form::Sop1
        | Form::Sop2
        | Form::Sopc
        | Form::Sopk
        | Form::Sopp
        | Form::Vop1
        | Form::Vop2
        | Form::Vopc => 1,
        Form::Vmem(_) if arch == Arch::Gfx1201 => 3,
        _ => 2,
    } {
        Some(words[n - 1])
    } else {
        None
    };
    let mut consumed = [0; 3];
    let (mask, _prefix) = forms::prefix_for(arch, row.form).expect("known form");
    consumed[0] |= mask;
    consumed[0] |= forms::opcode_for(arch, row.form)
        .expect("form opcode")
        .mask();
    if matches!(row.form, Form::Vop1Dpp | Form::Vop1Dpp8 | Form::Vop2Dpp) {
        consumed[0] |= 0x1ff;
    }
    let mut operands = SmallVec::new();
    let mut mods = Modifiers::default();
    // A gfx11 VMEM atomic's `VDST@rtn` slot exists only when GLC selects the return.
    let returns = isa::atomic_returns(arch, &cache_policy(arch, row, words));
    for (name, _) in row.slots(returns) {
        if let Some(f) = forms::field_for(arch, row.form, name) {
            consumed[usize::from(f.bit / 32)] |= f.mask();
        }
    }
    if row.form == Form::Vop3 && row.grammar.contains("SDST:") {
        consumed[0] |= 0x7f00;
    }
    if row.form == Form::Vmem(crate::inst::VmemForm::Scratch) {
        let sve = forms::field_for(arch, row.form, "SVE").expect("scratch vector address enable");
        consumed[usize::from(sve.bit / 32)] |= sve.mask();
    }
    if row.form == Form::Vopc && row.name.starts_with("v_cmp_") {
        operands.push(Operand::Special(Special::VccLo));
    }
    for (name, bits) in row.slots(returns) {
        let v = field_value(arch, name, row, words);
        if row.form == Form::Vop3 && name == "VDST" && row.name.starts_with("v_cmpx_") {
            if v != 126 {
                return Err(reject("VOP3 cmpx requires implicit EXEC destination"));
            }
            continue;
        }
        if name == "LITERAL" {
            operands.push(Operand::Literal(
                literal.ok_or_else(|| reject("missing VOPD literal"))?,
            ));
            continue;
        }
        if forms::field_for(arch, row.form, name).is_none() && name != "SDST" {
            return Err(reject(format!("no {} bitfield for {name}", row.name)));
        }
        if name == "SRC2" && v == 255 && literal.is_none() {
            return Err(reject("literal selector without literal word"));
        }
        let parsed = operand(arch, name, bits, v, literal, row, words)?;
        operands.push(parsed);
    }
    if row.form == Form::Smem {
        let offset = field_value(arch, "IOFFSET", row, words);
        if matches!(operands.last(), Some(Operand::Imm(ImmField::SmemOffset(_)))) {
            consumed[1] |= forms::field_for(arch, row.form, "IOFFSET")
                .expect("SMEM IOFFSET")
                .mask();
        } else if offset != 0 {
            operands.push(Operand::Imm(ImmField::SmemDisplacement(
                ((offset << 8) as i32) >> 8,
            )));
            consumed[1] |= forms::field_for(arch, row.form, "IOFFSET")
                .expect("SMEM IOFFSET")
                .mask();
        }
    }
    if row.form == Form::Ds {
        let low = field_value(arch, "OFFSET0", row, words) as u8;
        let high = field_value(arch, "OFFSET1", row, words) as u8;
        if row.name.contains("2addr") {
            if low != 0 {
                operands.push(Operand::Imm(ImmField::DsOffset0(low)));
            }
            if high != 0 {
                operands.push(Operand::Imm(ImmField::DsOffset1(high)));
            }
        } else if low != 0 || high != 0 {
            operands.push(Operand::Imm(ImmField::DsOffset(u16::from_le_bytes([
                low, high,
            ]))));
        }
        consumed[0] |= 0xffff;
    }
    if matches!(row.form, Form::Vmem(_)) {
        let off = field_value(arch, "IOFFSET", row, words);
        if off != 0 {
            // gfx11 MUBUF OFFSET is an unsigned 12-bit byte offset (0..=4095);
            // FLAT/GLOBAL/SCRATCH (13-bit) and gfx12 (24-bit) offsets are signed.
            if arch != Arch::Gfx1201 && row.form == Form::Vmem(crate::inst::VmemForm::Buffer) {
                operands.push(Operand::Imm(ImmField::VmemOffset(off as i32)));
            } else {
                let bits = if arch == Arch::Gfx1201 { 24 } else { 13 };
                operands.push(Operand::Imm(ImmField::VmemOffset(
                    ((off << (32 - bits)) as i32) >> (32 - bits),
                )));
            }
        }
        if row.form == Form::Vmem(crate::inst::VmemForm::Buffer)
            && field_value(arch, "OFFEN", row, words) == 1
        {
            operands.push(Operand::Vmem(VmemToken::Offen));
        }
        if arch != Arch::Gfx1201 && row.form == Form::Vmem(crate::inst::VmemForm::Buffer) {
            for name in ["OFFEN", "IDXEN", "TFE"] {
                let field = forms::field_for(arch, row.form, name).expect("gfx11 buffer flag");
                consumed[usize::from(field.bit / 32)] |= field.mask();
            }
        }
        if arch == Arch::Gfx1201 {
            if row.form == Form::Vmem(crate::inst::VmemForm::Buffer) {
                if let th @ 1..=7 = field_value(arch, "TH", row, words) {
                    operands.push(Operand::CacheTh(th as u8));
                }
                match field_value(arch, "SCOPE", row, words) {
                    0 => {}
                    1 => operands.push(Operand::Scope(CacheScope::Se)),
                    2 => operands.push(Operand::Scope(CacheScope::Dev)),
                    _ => operands.push(Operand::Scope(CacheScope::Sys)),
                }
            } else if matches!(row.name, "global_inv" | "global_wb") {
                match field_value(arch, "SCOPE", row, words) {
                    0 => {}
                    1 => operands.push(Operand::Scope(CacheScope::Se)),
                    2 => operands.push(Operand::Scope(CacheScope::Dev)),
                    _ => operands.push(Operand::Scope(CacheScope::Sys)),
                }
            }
        }
        let field = forms::field_for(arch, row.form, "IOFFSET").expect("VMEM IOFFSET");
        consumed[usize::from(field.bit / 32)] |= field.mask();
    }
    if row.form == Form::Vopd {
        let y_id = ((words[0] >> 17) & 31) as u16;
        let y = isa::table(arch)
            .iter()
            .find(|r| r.form == Form::Vopd && r.op.id == y_id)
            .ok_or_else(|| reject("unknown VOPD Y opcode"))?;
        consumed[0] |= Field::new("OPY", 17, 5).mask();
        let count = operands.len() as u8;
        for (name, bits) in y.slots(false) {
            if name == "LITERAL" {
                operands.push(Operand::Literal(
                    literal.ok_or_else(|| reject("missing VOPD Y literal"))?,
                ));
                continue;
            }
            let mapped = match name {
                "VDSTX" => "VDSTY",
                "SRCX0" => "SRCY0",
                "VSRCX1" => "VSRCY1",
                _ => name,
            };
            let raw = field_value(arch, mapped, y, words);
            let code = if mapped == "VDSTY" {
                (raw << 1) | ((!field_value(arch, "VDSTX", row, words)) & 1)
            } else {
                raw
            };
            operands.push(operand(arch, mapped, bits, code, literal, y, words)?);
            let f = forms::field_for(arch, row.form, mapped).expect("VOPD Y field");
            consumed[usize::from(f.bit / 32)] |= f.mask();
        }
        for i in 0..2 {
            if words[i] & !consumed[i] != 0 {
                return Err(reject(format!(
                    "unknown VOPD bits in word {i}: {:#x}",
                    words[i] & !consumed[i]
                )));
            }
        }
        let fields = FormFields::Vopd {
            y_op: y.op,
            x_operands: count,
        };
        let inst = Inst::from_parts(
            arch,
            row.op,
            row.form,
            fields,
            operands,
            mods,
            literal,
            Provenance::default(),
        )
        .map_err(|e| reject(e.to_string()))?;
        return Ok((inst, n));
    }
    special_mods(arch, row, words, &mut mods);
    if matches!(row.form, Form::Vop3 | Form::Vop3p) {
        for name in if row.form == Form::Vop3 {
            &["ABS", "OPSEL", "CLAMP", "OMOD", "NEG"][..]
        } else {
            &[
                "NEG_HI",
                "OPSEL",
                "OPSEL_HI_LO",
                "OPSEL_HI_2",
                "CLAMP",
                "NEG",
            ][..]
        } {
            let f = forms::field_for(arch, row.form, name).expect("modifier layout");
            consumed[usize::from(f.bit / 32)] |= f.mask();
        }
    }
    if matches!(row.form, Form::Vop1Dpp | Form::Vop2Dpp) {
        for name in [
            "DPP_CTRL",
            "BOUND_CTRL",
            "SRC0_NEG",
            "SRC0_ABS",
            "SRC1_NEG",
            "SRC1_ABS",
            "BANK_MASK",
            "ROW_MASK",
        ] {
            let f = forms::field_for(arch, row.form, name).expect("DPP field");
            consumed[usize::from(f.bit / 32)] |= f.mask();
        }
    }
    if row.form == Form::Smem || matches!(row.form, Form::Vmem(_)) {
        for name in if arch == Arch::Gfx1201 {
            &["TH", "SCOPE", "NV"][..]
        } else {
            &["GLC", "SLC", "DLC"][..]
        } {
            if let Some(f) = forms::field_for(arch, row.form, name) {
                consumed[usize::from(f.bit / 32)] |= f.mask();
            }
        }
    }
    if matches!(row.name, "global_inv" | "global_wb") {
        if field_value(arch, "SADDR", row, words) != 124 {
            return Err(reject(format!("{} SADDR must be off", row.name)));
        }
        consumed[0] |= forms::field_for(arch, row.form, "SADDR")
            .expect("global SADDR")
            .mask();
    }
    let fields = fields_from(arch, row, words, &mut consumed)?;
    for i in 0..n {
        if Some(i) == literal.map(|_| n - 1) {
            continue;
        }
        let unknown = words[i] & !consumed[i];
        if unknown != 0 {
            return Err(reject(format!("unknown bits in word {i}: {unknown:#010x}")));
        }
    }
    let inst = Inst::from_parts(
        arch,
        row.op,
        row.form,
        fields,
        operands,
        mods,
        literal,
        Provenance::default(),
    )
    .map_err(|e| reject(e.to_string()))?;
    Ok((inst, n))
}
/// Rebuild bytes from typed instruction fields. Never reads `inst.prov.bytes`.
pub fn encode(inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    encode_for(Arch::Gfx1201, inst)
}
/// Encode from typed fields without consulting provenance bytes.
pub fn encode_for(arch: Arch, inst: &Inst) -> Result<SmallVec<[u32; 3]>, DecodeError> {
    inst.validate(arch).map_err(|e| reject(e.to_string()))?;
    let row = isa::lookup(arch, inst.op, inst.form).ok_or_else(|| reject("unknown opcode/form"))?;
    if row.form == Form::Vmem(crate::inst::VmemForm::Global) {
        let address = row
            .slots(isa::atomic_returns(arch, &inst.mods.cpol))
            .position(|(name, _)| matches!(name, "VADDR" | "ADDR"));
        let scalar = row
            .slots(isa::atomic_returns(arch, &inst.mods.cpol))
            .position(|(name, _)| name == "SADDR");
        if let (Some(a), Some(s)) = (address, scalar) {
            if let Some(Operand::Reg(r)) = inst.operands.get(a) {
                let expected = if inst.operands.get(s) == Some(&Operand::Vmem(VmemToken::Off)) {
                    2
                } else {
                    1
                };
                if r.len != expected {
                    return Err(reject("GLOBAL VADDR width disagrees with SADDR mode"));
                }
            }
        }
    }
    let mut words = [0u32; 3];
    let (_, prefix) =
        forms::prefix_for(arch, row.form).ok_or_else(|| reject("unsupported form"))?;
    words[0] = prefix;
    assign(
        forms::opcode_for(arch, row.form).expect("form opcode"),
        &mut words,
        inst.op.id.into(),
    )?;
    if matches!(row.form, Form::Vop1Dpp | Form::Vop2Dpp) {
        words[0] |= 0xfa;
    }
    if row.form == Form::Vop1Dpp8 {
        words[0] |= 0xe9;
    }
    let mut operands = inst.operands.iter();
    if row.form == Form::Vopc
        && row.name.starts_with("v_cmp_")
        && operands.next() != Some(&Operand::Special(Special::VccLo))
    {
        return Err(reject("VOPC compare destination must be vcc_lo"));
    }
    for (name, bits) in row.slots(isa::atomic_returns(arch, &inst.mods.cpol)) {
        if row.form == Form::Vop3 && name == "VDST" && row.name.starts_with("v_cmpx_") {
            encode_field(arch, name, row, &mut words, 126)?;
            continue;
        }
        let op = operands
            .next()
            .ok_or_else(|| reject(format!("missing {name} operand")))?;
        if name == "LITERAL" {
            if Some(match op {
                Operand::Literal(value) => *value,
                _ => return Err(reject("VOPD literal operand required")),
            }) != inst.literal
            {
                return Err(reject("VOPD literal and operand differ"));
            }
            continue;
        }
        encode_field(
            arch,
            name,
            row,
            &mut words,
            encoded_operand(arch, name, bits, op, row.form)?,
        )?;
        if row.form == Form::Vmem(crate::inst::VmemForm::Scratch) && name == "VADDR" {
            encode_field(
                arch,
                "SVE",
                row,
                &mut words,
                u32::from(matches!(op, Operand::Reg(_))),
            )?;
        }
        if name == "SOFFSET" && row.form == Form::Smem {
            if let Operand::Imm(ImmField::SmemOffset(off)) = op {
                encode_field(
                    arch,
                    "IOFFSET",
                    row,
                    &mut words,
                    (*off as u32) & 0x00ff_ffff,
                )?;
            }
        }
    }
    if row.form == Form::Vopd {
        if let FormFields::Vopd { y_op, x_operands } = &inst.fields {
            if usize::from(*x_operands) != row.slots(false).count() {
                return Err(reject("VOPD half boundary mismatch"));
            }
            let y = isa::lookup(arch, *y_op, Form::Vopd)
                .ok_or_else(|| reject("unknown VOPD Y opcode"))?;
            assign(Field::new("OPY", 17, 5), &mut words, y.op.id.into())?;
            for (name, bits) in y.slots(false) {
                let mapped = match name {
                    "VDSTX" => "VDSTY",
                    "SRCX0" => "SRCY0",
                    "VSRCX1" => "VSRCY1",
                    _ => name,
                };
                let op = operands
                    .next()
                    .ok_or_else(|| reject(format!("missing Y {mapped} operand")))?;
                if name == "LITERAL" {
                    if Some(match op {
                        Operand::Literal(value) => *value,
                        _ => return Err(reject("VOPD Y literal operand required")),
                    }) != inst.literal
                    {
                        return Err(reject("VOPD Y literal and operand differ"));
                    }
                    continue;
                }
                let value = encoded_operand(arch, mapped, bits, op, y.form)?;
                if mapped == "VDSTY" {
                    let x = field_value(arch, "VDSTX", row, &words);
                    if (value ^ x) & 1 == 0 {
                        return Err(reject("VOPD Y destination parity collides with X"));
                    }
                    encode_field(arch, mapped, y, &mut words, value >> 1)?;
                } else {
                    encode_field(arch, mapped, y, &mut words, value)?;
                }
            }
        } else {
            return Err(reject("missing VOPD Y half"));
        }
    }
    if row.form == Form::Ds {
        for op in operands.by_ref() {
            match op {
                Operand::Imm(ImmField::DsOffset(n)) if !row.name.contains("2addr") => {
                    encode_field(arch, "OFFSET0", row, &mut words, u32::from(*n & 255))?;
                    encode_field(arch, "OFFSET1", row, &mut words, u32::from(*n >> 8))?;
                }
                Operand::Imm(ImmField::DsOffset0(n)) if row.name.contains("2addr") => {
                    encode_field(arch, "OFFSET0", row, &mut words, u32::from(*n))?
                }
                Operand::Imm(ImmField::DsOffset1(n)) if row.name.contains("2addr") => {
                    encode_field(arch, "OFFSET1", row, &mut words, u32::from(*n))?
                }
                _ => return Err(reject("invalid DS offset")),
            }
        }
    }
    if row.form == Form::Smem {
        if let Some(Operand::Imm(ImmField::SmemDisplacement(offset))) = operands.next() {
            encode_field(
                arch,
                "IOFFSET",
                row,
                &mut words,
                (*offset as u32) & 0x00ff_ffff,
            )?;
        }
    }
    if matches!(row.form, Form::Vmem(_)) {
        for op in operands.by_ref() {
            match op {
                Operand::Imm(ImmField::VmemOffset(n)) => {
                    let bits = forms::field_for(arch, row.form, "IOFFSET")
                        .expect("VMEM offset field")
                        .width;
                    encode_field(
                        arch,
                        "IOFFSET",
                        row,
                        &mut words,
                        (*n as u32) & ((1u32 << bits) - 1),
                    )?;
                }
                Operand::Vmem(VmemToken::Offen)
                    if row.form == Form::Vmem(crate::inst::VmemForm::Buffer) =>
                {
                    encode_field(arch, "OFFEN", row, &mut words, 1)?
                }
                Operand::Scope(scope) if row.form == Form::Vmem(crate::inst::VmemForm::Buffer) => {
                    encode_field(
                        arch,
                        "SCOPE",
                        row,
                        &mut words,
                        match scope {
                            CacheScope::Cu => 0,
                            CacheScope::Se => 1,
                            CacheScope::Dev => 2,
                            CacheScope::Sys => 3,
                        },
                    )?
                }
                Operand::CacheTh(th) if row.form == Form::Vmem(crate::inst::VmemForm::Buffer) => {
                    encode_field(arch, "TH", row, &mut words, u32::from(*th))?
                }
                Operand::Scope(scope) if matches!(row.name, "global_inv" | "global_wb") => {
                    encode_field(
                        arch,
                        "SCOPE",
                        row,
                        &mut words,
                        match scope {
                            CacheScope::Cu => 0,
                            CacheScope::Se => 1,
                            CacheScope::Dev => 2,
                            CacheScope::Sys => 3,
                        },
                    )?
                }
                _ => return Err(reject("invalid VMEM modifier operand")),
            }
        }
    }
    if operands.next().is_some() {
        return Err(reject("extra operands not declared by opcode grammar"));
    }
    if matches!(row.form, Form::Vop3 | Form::Vop3p) {
        let m = &inst.mods;
        if row.form == Form::Vop3 {
            if !row.grammar.contains("SDST:") {
                encode_field(arch, "ABS", row, &mut words, m.abs.into())?;
                encode_field(arch, "OPSEL", row, &mut words, m.op_sel.into())?;
            }
            encode_field(
                arch,
                "OMOD",
                row,
                &mut words,
                match m.omod {
                    Omod::None => 0,
                    Omod::Mul2 => 1,
                    Omod::Mul4 => 2,
                    Omod::Div2 => 3,
                },
            )?;
        } else {
            encode_field(arch, "NEG_HI", row, &mut words, m.neg_hi.into())?;
            encode_field(arch, "OPSEL", row, &mut words, m.op_sel.into())?;
            encode_field(
                arch,
                "OPSEL_HI_LO",
                row,
                &mut words,
                (m.op_sel_hi & 3).into(),
            )?;
            encode_field(
                arch,
                "OPSEL_HI_2",
                row,
                &mut words,
                (m.op_sel_hi >> 2).into(),
            )?;
        }
        encode_field(
            arch,
            "NEG",
            row,
            &mut words,
            if row.form == Form::Vop3p {
                m.neg_lo
            } else {
                m.neg
            }
            .into(),
        )?;
        encode_field(arch, "CLAMP", row, &mut words, u32::from(m.clamp))?;
    }
    if matches!(row.form, Form::Vop1Dpp | Form::Vop2Dpp) {
        let dpp = inst
            .mods
            .dpp
            .ok_or_else(|| reject("DPP form requires typed DPP control"))?;
        for (field, v) in [
            ("DPP_CTRL", u32::from(dpp.ctrl)),
            ("ROW_MASK", u32::from(dpp.row_mask)),
            ("BANK_MASK", u32::from(dpp.bank_mask)),
            ("BOUND_CTRL", u32::from(dpp.bound_ctrl)),
            ("SRC0_NEG", u32::from(inst.mods.neg & 1)),
            ("SRC1_NEG", u32::from(inst.mods.neg >> 1 & 1)),
            ("SRC0_ABS", u32::from(inst.mods.abs & 1)),
            ("SRC1_ABS", u32::from(inst.mods.abs >> 1 & 1)),
        ] {
            encode_field(arch, field, row, &mut words, v)?;
        }
    }
    if row.form == Form::Smem || matches!(row.form, Form::Vmem(_)) {
        if arch == Arch::Gfx1201 {
            encode_field(arch, "TH", row, &mut words, inst.mods.cpol.th.into())?;
            encode_field(arch, "SCOPE", row, &mut words, inst.mods.cpol.scope.into())?;
            encode_field(arch, "NV", row, &mut words, u32::from(inst.mods.cpol.nv))?;
            if matches!(row.name, "global_inv" | "global_wb") {
                encode_field(arch, "SADDR", row, &mut words, 124)?;
            }
        } else {
            for (name, bit) in [
                ("GLC", inst.mods.cpol.glc),
                ("SLC", inst.mods.cpol.slc),
                ("DLC", inst.mods.cpol.dlc),
            ] {
                if forms::field_for(arch, row.form, name).is_some() {
                    encode_field(arch, name, row, &mut words, u32::from(bit))?;
                }
            }
        }
    }
    let fields = match &inst.fields {
        FormFields::Vop3b { src2_unused } => {
            encode_field(arch, "SRC2", row, &mut words, u32::from(*src2_unused))?;
            None
        }
        FormFields::Bits { ignored, honored } => Some((ignored, honored)),
        FormFields::None | FormFields::Vopd { .. } => None,
    };
    if let Some((ignored, honored)) = fields {
        for field in ignored.iter().chain(honored) {
            match field.name {
                "src2_unused" => encode_field(arch, "SRC2", row, &mut words, field.value)?,
                "src1_unused" => encode_field(arch, "SRC1", row, &mut words, field.value)?,
                "w0_extra" | "smem_len" | "wait_unused" | "vbuffer_tfe" | "vbuffer_nv"
                | "global_nv" => words[0] |= field.value,
                "w1_extra" | "vsrc_unused" | "vbuffer_format" | "vbuffer_offen"
                | "vbuffer_idxen" | "vbuffer_scope" | "vbuffer_th" | "global_sve"
                | "global_scope" | "global_th" | "dpp_fi" | "dpp8_sel" => words[1] |= field.value,
                "neg" | "abs" | "omod" | "op_sel" | "op_sel_hi" => { /* Encoded by the semantic modifier. */
                }
                other => return Err(reject(format!("unhandled field {other}"))),
            }
        }
    }
    let mut n = match row.form {
        Form::Sop1
        | Form::Sop2
        | Form::Sopc
        | Form::Sopk
        | Form::Sopp
        | Form::Vop1
        | Form::Vop2
        | Form::Vopc => 1,
        Form::Vmem(_) if arch == Arch::Gfx1201 => 3,
        _ => 2,
    };
    if let Some(lit) = inst.literal {
        if n >= 3 {
            return Err(reject("literal exceeds three-word instruction"));
        }
        words[n] = lit;
        n += 1;
    }
    let (decoded, used) = decode_for(arch, &words[..n])?;
    if used != n
        || decoded.op != inst.op
        || decoded.form != inst.form
        || decoded.operands != inst.operands
        || decoded.fields != inst.fields
        || decoded.mods != inst.mods
        || decoded.literal != inst.literal
    {
        return Err(reject(format!(
            "inconsistent typed fields for {}",
            row.name
        )));
    }
    Ok(SmallVec::from_slice(&words[..n]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LLVM gfx1201 `v_min_u32_e64 v72, s92, v24`, used by GDN scan.
    #[test]
    fn gdn_unsigned_min_roundtrip_and_sgpr_use() {
        let words = [0xd513_0048, 0x0202_305c];
        let (inst, consumed) = decode(&words).unwrap();
        assert_eq!(consumed, 2);
        assert_eq!(inst.op.name(Arch::Gfx1201), Some("v_min_u32_e64"));
        assert_eq!(
            inst.effects.defs.as_slice(),
            &[RegRef {
                kind: Kind::V,
                base: 72,
                len: 1
            }]
        );
        assert_eq!(
            inst.effects.uses.as_slice(),
            &[
                RegRef {
                    kind: Kind::S,
                    base: 92,
                    len: 1
                },
                RegRef {
                    kind: Kind::V,
                    base: 24,
                    len: 1
                },
            ]
        );
        assert_eq!(encode(&inst).unwrap().as_slice(), &words);
    }
    #[test]
    fn every_declared_example_roundtrips() {
        let mut failures = Vec::new();
        for row in isa::gfx12() {
            let words: Vec<u32> = row
                .encoding
                .split_whitespace()
                .map(|w| u32::from_str_radix(w, 16).unwrap())
                .collect();
            match decode(&words).and_then(|(inst, len)| {
                if len != words.len() {
                    return Err(reject(format!("width {len} != {}", words.len())));
                }
                let result = encode(&inst)?;
                if &result[..] != words {
                    return Err(reject(format!("words {result:08x?} != {words:08x?}")));
                }
                Ok(())
            }) {
                Ok(()) => (),
                Err(error) => failures.push(format!("{}: {error}", row.name)),
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// One pinned word per row added for the gfx1201/gfx11 JIT caches at
    /// fe77c0837 (`llvm-mc --disassemble`, AMD LLVM 23.0.0git, in the comment).
    /// Each must decode to the LLVM mnemonic and re-encode byte-exact; gfx11
    /// rows also hold on gfx1151, which inherits the gfx1100 table.
    const JIT_CACHE_ROWS: &[(Arch, &str, &[u32])] = &[
        (Arch::Gfx1201, "ds_add_rtn_u32", &[0xd880_0000, 0x0100_0701]), // ds_add_rtn_u32 v1, v1, v7
        (Arch::Gfx1201, "ds_add_u32", &[0xd800_0000, 0x0000_0407]),     // ds_add_u32 v7, v4
        (
            Arch::Gfx1201,
            "ds_load_2addr_b64",
            &[0xd9dc_0100, 0x0000_0037],
        ), // ds_load_2addr_b64 v[0:3], v55 offset1:1
        (Arch::Gfx1201, "ds_load_b96", &[0xdbf8_0512, 0x2200_0025]), // ds_load_b96 v[34:36], v37 offset:1298
        (
            Arch::Gfx1201,
            "ds_load_u16_d16_hi",
            &[0xda9c_0088, 0x0800_002a],
        ), // ds_load_u16_d16_hi v8, v42 offset:136
        (Arch::Gfx1201, "ds_or_b32", &[0xd828_0000, 0x0000_0100]),   // ds_or_b32 v0, v1
        (
            Arch::Gfx1201,
            "flat_load_b128",
            &[0xec05_c07c, 0x0000_0001, 0x0000_1003],
        ), // flat_load_b128 v[1:4], v[3:4] offset:16
        (
            Arch::Gfx1201,
            "flat_load_b32",
            &[0xec05_007c, 0x0000_0001, 0x0000_0006],
        ), // flat_load_b32 v1, v[6:7]
        (
            Arch::Gfx1201,
            "flat_load_b64",
            &[0xec05_407c, 0x0000_0020, 0x0000_0009],
        ), // flat_load_b64 v[32:33], v[9:10]
        (
            Arch::Gfx1201,
            "flat_load_d16_b16",
            &[0xec08_007c, 0x0000_0001, 0x0000_0407],
        ), // flat_load_d16_b16 v1, v[7:8] offset:4
        (
            Arch::Gfx1201,
            "flat_load_d16_hi_b16",
            &[0xec08_c07c, 0x0000_0001, 0x0000_0407],
        ), // flat_load_d16_hi_b16 v1, v[7:8] offset:4
        (
            Arch::Gfx1201,
            "flat_store_b128",
            &[0xec07_407c, 0x0100_0000, 0x0000_1018],
        ), // flat_store_b128 v[24:25], v[2:5] offset:16
        (
            Arch::Gfx1201,
            "flat_store_b32",
            &[0xec06_807c, 0x0100_0000, 0x0000_0000],
        ), // flat_store_b32 v[0:1], v2
        (
            Arch::Gfx1201,
            "global_atomic_add_u32",
            &[0xee0d_4004, 0x0198_0002, 0x0000_0002],
        ), // global_atomic_add_u32 v2, v2, v3, s[4:5] th:TH_ATOMIC_RETURN scope:SCOPE_DEV
        (
            Arch::Gfx1201,
            "global_load_d16_hi_b16",
            &[0xee08_c07c, 0x0000_0000, 0x0000_0006],
        ), // global_load_d16_hi_b16 v0, v[6:7], off
        (Arch::Gfx1201, "s_clz_i32_u32", &[0xbeaf_0a32]),            // s_clz_i32_u32 s47, s50
        (Arch::Gfx1201, "s_max_u32", &[0x8a9e_231e]),                // s_max_u32 s30, s30, s35
        (Arch::Gfx1201, "ds_max_u32", &[0xd820_0040, 0x0000_0107]),  // ds_max_u32 v7, v1 offset:64
        (
            Arch::Gfx1201,
            "v_alignbyte_b32",
            &[0xd617_0006, 0x0686_0702],
        ), // v_alignbyte_b32 v6, v2, v3, v161.l
        (Arch::Gfx1201, "s_bcnt1_i32_b32", &[0xbe80_1805]),          // s_bcnt1_i32_b32 s0, s5
        (Arch::Gfx1201, "s_bfe_i32", &[0x9380_ff75, 0x0001_001d]),   // s_bfe_i32 s0, ttmp9, 0x1001d
        (Arch::Gfx1201, "s_bitcmp0_b32", &[0xbf0c_8018]),            // s_bitcmp0_b32 s24, 0
        (Arch::Gfx1201, "s_cmp_nge_f32", &[0xbf49_8000]),            // s_cmp_nge_f32 s0, 0
        (Arch::Gfx1201, "s_cvt_f32_f16", &[0xbe90_6903]),            // s_cvt_f32_f16 s16, s3
        (Arch::Gfx1201, "s_cvt_hi_f32_f16", &[0xbe80_6a00]),         // s_cvt_hi_f32_f16 s0, s0
        (Arch::Gfx1201, "s_fmaak_f32", &[0xa283_0503, 0x3586_37bd]), // s_fmaak_f32 s3, s3, s5, 0x358637bd
        (Arch::Gfx1201, "s_load_u16", &[0xf401_6b4f, 0xf800_0000]), // s_load_u16 s45, s[30:31], 0x0
        (Arch::Gfx1201, "s_pack_lh_b32_b16", &[0x9980_2780]),       // s_pack_lh_b32_b16 s0, 0, s39
        (
            Arch::Gfx1201,
            "v_add_nc_u32_dpp",
            &[0x4a00_0afa, 0xff09_6205],
        ), // v_add_nc_u32_dpp v0, v5, v5 row_xmask:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
        (Arch::Gfx1201, "v_cmp_eq_u64_e32", &[0x7cb4_0e80]), // v_cmp_eq_u64_e32 vcc_lo, 0, v[7:8]
        (
            Arch::Gfx1201,
            "v_cmp_eq_u64_e64",
            &[0xd45a_0000, 0x0202_3480],
        ), // v_cmp_eq_u64_e64 s0, 0, v[26:27]
        (Arch::Gfx1201, "v_cmp_lt_i64_e32", &[0x7ca2_104a]), // v_cmp_lt_i64_e32 vcc_lo, s[74:75], v[8:9]
        (
            Arch::Gfx1201,
            "v_cmp_ne_u64_e64",
            &[0xd45d_0001, 0x0202_1280],
        ), // v_cmp_ne_u64_e64 s1, 0, v[9:10]
        (
            Arch::Gfx1201,
            "v_cmp_neq_f16_e64",
            &[0xd40d_0000, 0x0202_1680],
        ), // v_cmp_neq_f16_e64 s0, 0, v11.l
        (
            Arch::Gfx1201,
            "v_cmp_nge_f32_e64",
            &[0xd419_0000, 0x0202_32ff, 0x7b80_0000],
        ), // v_cmp_nge_f32_e64 s0, 0x7b800000, v25
        (
            Arch::Gfx1201,
            "v_cmp_nle_f32_e32",
            &[0x7c38_32ff, 0x2680_0000],
        ), // v_cmp_nle_f32_e32 vcc_lo, 0x26800000, v25
        (
            Arch::Gfx1201,
            "v_cmp_nle_f32_e64",
            &[0xd41c_0002, 0x0202_22ff, 0x0d80_0000],
        ), // v_cmp_nle_f32_e64 s2, 0xd800000, v17
        (Arch::Gfx1201, "v_cmpx_le_u64_e32", &[0x7db6_0416]), // v_cmpx_le_u64_e32 s[22:23], v[2:3]
        (Arch::Gfx1201, "v_cmpx_eq_f32_e32", &[0x7d24_0d04]), // v_cmpx_eq_f32_e32 v4, v6
        (Arch::Gfx1201, "v_cmpx_eq_u16_e32", &[0x7d74_1281]), // v_cmpx_eq_u16_e32 1, v9.l
        (
            Arch::Gfx1201,
            "v_cmp_eq_u16_e64",
            &[0xd43a_0001, 0x0202_8081],
        ), // v_cmp_eq_u16_e64 s1, 1, v64.l
        (
            Arch::Gfx1201,
            "v_cmpx_eq_u32_e64",
            &[0xd4ca_007e, 0x0202_1204],
        ), // v_cmpx_eq_u32_e64 s4, v9
        (
            Arch::Gfx1201,
            "v_cmpx_gt_i64_e64",
            &[0xd4d4_007e, 0x0202_0610],
        ), // v_cmpx_gt_i64_e64 s[16:17], v[3:4]
        (Arch::Gfx1201, "v_cmpx_le_f32_e32", &[0x7d26_0880]), // v_cmpx_le_f32_e32 0, v4
        (Arch::Gfx1201, "v_cmpx_lt_u32_e32", &[0x7d92_0087]), // v_cmpx_lt_u32_e32 7, v0
        (Arch::Gfx1201, "v_cmpx_ne_u16_e32", &[0x7d7a_0280]), // v_cmpx_ne_u16_e32 0, v1.l
        (Arch::Gfx1201, "v_cmpx_ngt_f32_e32", &[0x7d36_0104]), // v_cmpx_ngt_f32_e32 v4, v0
        (
            Arch::Gfx1201,
            "v_cmpx_nlg_f32_e32",
            &[0x7d34_02ff, 0xff80_0000],
        ), // v_cmpx_nlg_f32_e32 0xff800000, v1
        (Arch::Gfx1201, "v_cvt_f32_ubyte1_e32", &[0x7e4c_251d]), // v_cvt_f32_ubyte1_e32 v38, v29
        (Arch::Gfx1201, "v_cvt_f32_ubyte2_e32", &[0x7e4e_271d]), // v_cvt_f32_ubyte2_e32 v39, v29
        (Arch::Gfx1201, "v_cvt_f32_ubyte3_e32", &[0x7e14_290a]), // v_cvt_f32_ubyte3_e32 v10, v10
        (Arch::Gfx1201, "v_cvt_f64_i32_e32", &[0x7e08_0904]), // v_cvt_f64_i32_e32 v[4:5], v4
        (Arch::Gfx1201, "v_cvt_f64_u32_e32", &[0x7e08_2d05]), // v_cvt_f64_u32_e32 v[4:5], v5
        (Arch::Gfx1201, "v_cvt_i32_f64_e32", &[0x7e08_0704]), // v_cvt_i32_f64_e32 v4, v[4:5]
        (
            Arch::Gfx1201,
            "v_dual_min_num_f32",
            &[0xcad4_0f06, 0x0606_0f08],
        ), // v_dual_min_num_f32 v6, v6, v7 :: v_dual_max_num_f32 v7, v8, v7
        (Arch::Gfx1201, "v_fmac_f16_e64", &[0xd536_4080, 0x0202_ff80]), // v_fmac_f16_e64 v128.h, v128.l, v127.l op_sel:[0,0,0,1]
        (Arch::Gfx1201, "v_lshlrev_b16", &[0xd738_0000, 0x0202_0088]), // v_lshlrev_b16 v0.l, 8, v0.l
        (
            Arch::Gfx1201,
            "v_maximumminimum_f32",
            &[0xd66d_001c, 0x03fc_1122, 0x4300_0000],
        ), // v_maximumminimum_f32 v28, v34, s8, 0x43000000
        (
            Arch::Gfx1201,
            "v_maxmin_num_f32",
            &[0xd669_0001, 0x03fc_0b01, 0x42fe_0000],
        ), // v_maxmin_num_f32 v1, v1, s5, 0x42fe0000
        (Arch::Gfx1201, "v_min3_i32", &[0xd61a_0001, 0x0404_1208]),    // v_min3_i32 v1, s8, s9, v1
        (Arch::Gfx1201, "v_min3_num_f32", &[0xd629_0006, 0x0432_1706]), // v_min3_num_f32 v6, v6, v11, v12
        (Arch::Gfx1201, "v_min_i32_dpp", &[0x2208_08fa, 0xff09_6104]), // v_min_i32_dpp v4, v4, v4 row_xmask:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        (Arch::Gfx1201, "v_min_num_f32_e32", &[0x2a10_0905]), // v_min_num_f32_e32 v8, v5, v4
        (Arch::Gfx1201, "v_min_num_f64_e32", &[0x1a08_0908]), // v_min_num_f64_e32 v[4:5], v[8:9], v[4:5]
        (Arch::Gfx1201, "v_minimum3_f32", &[0xd62d_024a, 0x0532_754b]), // v_minimum3_f32 v74, v75, |v58|, v76
        (Arch::Gfx1201, "v_minimum_f32", &[0xd765_024a, 0x0202_814a]), // v_minimum_f32 v74, v74, |v64|
        (Arch::Gfx1201, "v_or_b16", &[0xd763_0003, 0x0202_0706]),      // v_or_b16 v3.l, v6.l, v3.l
        (Arch::Gfx1201, "v_or_b32_dpp", &[0x3802_02fa, 0xff09_0101]), // v_or_b32_dpp v1, v1, v1 row_shl:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        (Arch::Gfx1201, "v_xad_u32", &[0xd645_0005, 0x0005_8300]),    // v_xad_u32 v5, v0, -1, s1
        (
            Arch::Gfx1100,
            "scratch_load_d16_b16",
            &[0xdc81_0000, 0x04fc_0002],
        ), // scratch_load_d16_b16 v4, v2, off
        (
            Arch::Gfx1100,
            "v_cmpx_eq_f32_e32",
            &[0x7d24_84ff, 0xff80_0000],
        ), // v_cmpx_eq_f32_e32 0xff800000, v66
        (
            Arch::Gfx1100,
            "v_cvt_f16_u16_e64",
            &[0xd5d0_0095, 0x0201_0195],
        ), // v_cvt_f16_u16_e64 v149.l, v149.l
        (Arch::Gfx1100, "v_fmac_f16_e64", &[0xd536_4086, 0x0202_d586]), // v_fmac_f16_e64 v134.h, v134.l, v106.l op_sel:[0,0,0,1]
        (Arch::Gfx1100, "v_min3_i32", &[0xd61a_0001, 0x0404_0a04]),     // v_min3_i32 v1, s4, s5, v1
        (
            Arch::Gfx1201,
            "global_load_b32",
            &[0xee05_007c, 0x0008_000d, 0x0000_1c0d],
        ), // global_load_b32 v13, v[13:14], off offset:28 scope:SCOPE_DEV
        (
            Arch::Gfx1201,
            "global_inv",
            &[0xee0a_c07c, 0x0008_0000, 0x0000_0000],
        ), // global_inv scope:SCOPE_DEV
        (Arch::Gfx1201, "s_mov_b64", &[0xbe82_01eb]), // s_mov_b64 s[2:3], src_shared_base
    ];

    #[test]
    fn jit_cache_rows_decode_to_llvm_mnemonic_and_reencode_exactly() {
        let mut failures = Vec::new();
        for &(arch, name, words) in JIT_CACHE_ROWS {
            let arches: &[Arch] = if arch == Arch::Gfx1100 {
                &[Arch::Gfx1100, Arch::Gfx1151]
            } else {
                &[arch]
            };
            for &arch in arches {
                let result = decode_for(arch, words).and_then(|(inst, len)| {
                    if len != words.len() {
                        return Err(reject(format!("width {len} != {}", words.len())));
                    }
                    // FLAT/GLOBAL/BUFFER and e32/DPP share opcode ids; the form disambiguates.
                    let decoded = isa::lookup(arch, inst.op, inst.form).map_or("?", |row| row.name);
                    if decoded != name {
                        return Err(reject(format!("decoded as {decoded}")));
                    }
                    let bytes = encode_for(arch, &inst)?;
                    if &bytes[..] != words {
                        return Err(reject(format!("re-encoded {bytes:08x?}")));
                    }
                    Ok(())
                });
                if let Err(error) = result {
                    failures.push(format!("{arch:?} {name}: {error}"));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    #[test]
    fn flat_aperture_and_scoped_invalidate_are_typed() {
        let (flat, _) = decode(&[0xec05_007c, 0x0000_0001, 0x0000_0006]).unwrap();
        let counters = &flat
            .effects
            .mem
            .as_ref()
            .expect("flat load is a memory op")
            .counters;
        assert!(counters.contains(Counter::Load) && counters.contains(Counter::Ds));
        let (mov, _) = decode(&[0xbe82_01eb]).unwrap();
        assert_eq!(
            mov.operands.last(),
            Some(&Operand::Special(Special::SrcSharedBase))
        );
        let (inv, _) = decode(&[0xee0a_c07c, 0x0008_0000, 0x0000_0000]).unwrap();
        assert_eq!(inv.operands.last(), Some(&Operand::Scope(CacheScope::Dev)));
        let (atomic, _) = decode(&[0xee0d_4004, 0x0198_0002, 0x0000_0002]).unwrap();
        assert!(atomic
            .effects
            .mem
            .as_ref()
            .unwrap()
            .counters
            .contains(Counter::Load));
        assert_eq!(atomic.mods.cpol.th & 1, 1);
    }

    /// LLVM gfx1201 `buffer_load_b32 v14, off, s[48:51], s63 offset:80` (OFFEN=IDXEN=0, VADDR byte 0).
    const BUFFER_OFF_WORDS: [u32; 3] = [0xc405_003f, 0x0080_600e, 0x0000_5000];

    #[test]
    fn buffer_off_decodes_vaddr_as_off_and_reads_no_vgpr() {
        let (inst, consumed) = decode(&BUFFER_OFF_WORDS).unwrap();
        assert_eq!(consumed, 3);
        assert_eq!(inst.operands[1], Operand::Vmem(VmemToken::Off));
        assert!(!inst.operands.contains(&Operand::Vmem(VmemToken::Offen)));
        let v = |r: &RegRef| r.kind == Kind::V;
        assert_eq!(inst.effects.uses.iter().filter(|r| v(r)).count(), 0);
        assert_eq!(
            inst.effects.defs.as_slice(),
            &[RegRef {
                kind: Kind::V,
                base: 14,
                len: 1
            }]
        );
        assert_eq!(encode(&inst).unwrap().as_slice(), &BUFFER_OFF_WORDS);
    }

    #[test]
    fn buffer_off_rejects_nonzero_vaddr_byte() {
        let mut words = BUFFER_OFF_WORDS;
        words[2] |= 3; // VADDR byte = 3 with OFFEN=IDXEN=0
        let e = decode(&words).unwrap_err();
        assert!(format!("{e:?}").contains("VADDR must be zero"), "{e:?}");
    }

    #[test]
    fn buffer_offen_keeps_register_vaddr_and_idxen_stays_refused() {
        // OFFEN alone: one VGPR, round-trips.
        let mut words = BUFFER_OFF_WORDS;
        words[1] = (words[1] & 0x00ff_ffff) | (0x40 << 24);
        words[2] = (words[2] & !0xff) | 3;
        let (inst, _) = decode(&words).unwrap();
        assert_eq!(
            inst.operands[1],
            Operand::Reg(RegRef {
                kind: Kind::V,
                base: 3,
                len: 1
            })
        );
        assert_eq!(encode(&inst).unwrap().as_slice(), &words);
        // IDXEN and IDXEN+OFFEN are outside the modeled form and refused by the existing model.
        for flags in [0x80u32, 0xc0] {
            let mut words = BUFFER_OFF_WORDS;
            words[1] = (words[1] & 0x00ff_ffff) | (flags << 24);
            words[2] = (words[2] & !0xff) | 2;
            let e = decode(&words).unwrap_err();
            assert!(
                format!("{e:?}").contains("idxen"),
                "flags {flags:#x}: {e:?}"
            );
        }
    }

    #[test]
    fn buffer_cache_policy_is_typed_and_preserved() {
        let base = [0xc405_005c, 0x4080_50b8, 0x0000_00b6];
        for scope in 0..4u32 {
            for th in 0..8u32 {
                let mut words = base;
                words[1] = (words[1] & !0x7c_0000) | (scope << 18) | (th << 20);
                let (inst, consumed) = decode(&words).unwrap();
                assert_eq!(consumed, 3);
                assert_eq!(inst.mods.cpol.scope, scope as u8);
                assert_eq!(inst.mods.cpol.th, th as u8);
                assert_eq!(
                    inst.operands
                        .iter()
                        .find(|op| matches!(op, Operand::CacheTh(_))),
                    (th != 0).then_some(&Operand::CacheTh(th as u8))
                );
                assert_eq!(encode(&inst).unwrap().as_slice(), &words);
            }
        }
    }

    /// gfx11+ selector 124 is `null` and 125 is `m0` in source operands as
    /// in destinations: a `null` buffer SOFFSET must not read M0.
    #[test]
    fn source_null_and_m0_selectors() {
        let (store, _) = decode(&[0xc407_407c, 0x4080_6800, 0x0000_00b9]).unwrap();
        assert!(store.operands.contains(&Operand::Special(Special::Null)));
        assert!(!store.operands.contains(&Operand::Special(Special::M0)));
        for (word, special) in [(0xbe84_007c, Special::Null), (0xbe84_007d, Special::M0)] {
            let (mov, _) = decode(&[word]).unwrap();
            assert_eq!(mov.operands.last(), Some(&Operand::Special(special)));
            assert_eq!(encode(&mov).unwrap().as_slice(), &[word]);
        }
    }

    #[test]
    fn declared_dont_care_combinations_roundtrip() {
        for row in isa::gfx12() {
            let example: Vec<u32> = row
                .encoding
                .split_whitespace()
                .map(|w| u32::from_str_radix(w, 16).unwrap())
                .collect();
            let mut cases = vec![example];
            for rule in row
                .fields
                .iter()
                .filter(|rule| rule.class == FieldClass::Ignored)
            {
                let (index, shift, mask) = match rule.name {
                    "src1_unused" => (1, 9, rule.mask << 9),
                    "src2_unused" => (1, 18, rule.mask << 18),
                    "omod" => (1, 27, rule.mask << 27),
                    "w0_extra" | "wait_unused" => (0, 0, rule.mask),
                    "w1_extra" | "vsrc_unused" => (1, 0, rule.mask),
                    other => panic!("unexpected ignored field {other} on {}", row.name),
                };
                cases = cases
                    .into_iter()
                    .flat_map(|words| {
                        rule.allowed.iter().map(move |value| {
                            let mut variant = words.clone();
                            variant[index] = (variant[index] & !mask) | (*value << shift);
                            variant
                        })
                    })
                    .collect();
            }
            for words in cases {
                let (inst, len) =
                    decode(&words).unwrap_or_else(|e| panic!("{} {words:08x?}: {e}", row.name));
                assert_eq!(len, words.len(), "{}", row.name);
                assert_eq!(&encode(&inst).unwrap()[..], words, "{}", row.name);
            }
        }
    }

    fn kt48_typed_insts() -> Vec<Inst> {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00; // .text file offset 0x2400 + (kernel VA 0x7f00 - .text VA 0x3400)
        const SIZE: usize = 10_604;
        let words: Vec<_> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut result = Vec::with_capacity(1696);
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = decode(&words[index..])
                .unwrap_or_else(|e| panic!("KT48 byte {:#x}: {e}", index * 4));
            let output =
                encode(&inst).unwrap_or_else(|e| panic!("KT48 byte {:#x}: {e}", index * 4));
            assert_eq!(
                &output[..],
                &words[index..index + count],
                "KT48 byte {:#x}",
                index * 4
            );
            result.push(inst);
            index += count;
        }
        assert_eq!(index * 4, SIZE);
        assert_eq!(result.len(), 1696, "selected KT48 instruction census");
        result
    }

    #[test]
    fn kt48_selected_kernel_reencodes_exactly() {
        kt48_typed_insts();
    }

    #[test]
    fn kt48_all_six_kernel_streams_reencode_exactly() {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        // .text starts at VA 0x3400, file offset 0x2400. Each range is an ELF STT_FUNC.
        const SYMBOLS: [(usize, usize); 6] = [
            (0x3400, 7996),
            (0x5400, 388),
            (0x5600, 1484),
            (0x5c00, 8148),
            (0x7c00, 728),
            (0x7f00, 10604),
        ];
        let mut counts = Vec::new();
        for (va, size) in SYMBOLS {
            let start = 0x2400 + (va - 0x3400);
            let words: Vec<u32> = IMAGE[start..start + size]
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            let mut offset = 0;
            let mut count = 0;
            while offset < words.len() {
                let (inst, n) = decode(&words[offset..])
                    .unwrap_or_else(|e| panic!("KT48 symbol {va:#x}, byte {:#x}: {e}", offset * 4));
                let output = encode(&inst).unwrap();
                assert_eq!(
                    &output[..],
                    &words[offset..offset + n],
                    "KT48 symbol {va:#x}, byte {:#x}",
                    offset * 4
                );
                offset += n;
                count += 1;
            }
            assert_eq!(offset * 4, size, "KT48 symbol {va:#x} length");
            counts.push(count);
        }
        assert_eq!(counts.len(), 6);
        assert_eq!(counts[5], 1696, "selected KT48 kernel census");
        println!("KT48 six symbol instruction counts: {counts:?}");
    }

    #[test]
    fn true16_e32_halves_and_implicit_cmpx_exec_are_typed() {
        let true16 = [0x7f18398d]; // v_mov_b16_e32 v12.h, v13.h
        let (inst, _) = decode(&true16).unwrap();
        assert_eq!(
            inst.operands[0],
            Operand::Half(
                RegRef {
                    kind: Kind::V,
                    base: 12,
                    len: 1
                },
                Half::Hi
            )
        );
        assert_eq!(
            inst.operands[1],
            Operand::Half(
                RegRef {
                    kind: Kind::V,
                    base: 13,
                    len: 1
                },
                Half::Hi
            )
        );
        assert_eq!(&encode(&inst).unwrap()[..], true16);

        let scalar = [0x7f0a3800]; // v_mov_b16_e32 v5.h, s0
        let (inst, _) = decode(&scalar).unwrap();
        assert_eq!(
            inst.operands[0],
            Operand::Half(
                RegRef {
                    kind: Kind::V,
                    base: 5,
                    len: 1
                },
                Half::Hi
            )
        );
        assert_eq!(
            inst.operands[1],
            Operand::Reg(RegRef {
                kind: Kind::S,
                base: 0,
                len: 1
            })
        );
        assert_eq!(&encode(&inst).unwrap()[..], scalar);

        let convert = [0x7e061702]; // v_cvt_f32_f16_e32 v3, v2.l
        let (inst, _) = decode(&convert).unwrap();
        assert_eq!(
            inst.operands[1],
            Operand::Half(
                RegRef {
                    kind: Kind::V,
                    base: 2,
                    len: 1
                },
                Half::Lo
            )
        );
        assert_eq!(&encode(&inst).unwrap()[..], convert);

        let cmpx = [0xd4c4007e, 0x02020807]; // implicit EXEC destination
        let (inst, _) = decode(&cmpx).unwrap();
        assert_eq!(inst.operands.len(), 2);
        assert_eq!(
            inst.text(Arch::Gfx1201).unwrap(),
            "v_cmpx_gt_i32_e64 s7, v4"
        );
        assert_eq!(&encode(&inst).unwrap()[..], cmpx);
        assert!(decode(&[cmpx[0] ^ 1, cmpx[1]])
            .unwrap_err()
            .to_string()
            .contains("implicit EXEC"));
    }

    #[test]
    fn d16_load_preserves_partial_vgpr_write_dependency() {
        let load = [0xee08407c, 0x0000008b, 0x00183002]; // global_load_d16_hi_u8 v139, v[2:3], off offset:6192
        let (inst, _) = decode(&load).unwrap();
        let dest = RegRef {
            kind: Kind::V,
            base: 139,
            len: 1,
        };
        assert!(inst.effects.defs.contains(&dest));
        assert!(
            inst.effects.uses.contains(&dest),
            "upper/lower half must be read before partial write"
        );
        assert_eq!(&encode(&inst).unwrap()[..], load);
    }

    #[test]
    fn dangerous_literal_fill_and_misaligned_smem_are_rejected() {
        // The first two words are the pinned v_add_co_u32 VOP3b example.
        let mut add = [0xd7006a01, 0x0202020e];
        add[1] = (add[1] & !(0x1ff << 18)) | (0xff << 18);
        assert!(decode(&add)
            .unwrap_err()
            .to_string()
            .contains("dangerous-fill"));
        let aligned = [0xf4004100, 0xf8000030]; // s_load_b128 s[4:7], s[0:1], 0x30
        let misaligned = [(aligned[0] & !(0x7f << 6)) | (2 << 6), aligned[1]];
        assert!(decode(&misaligned)
            .unwrap_err()
            .to_string()
            .contains("must be aligned"));
    }

    #[test]
    fn honored_modifiers_and_noncanonical_fill_reencode_without_provenance() {
        let add = [0xd5250001, 0x02020702]; // integer VOP3 with no semantic SRC2
        let changed = [
            add[0] | (5 << 8),
            (add[1] & !(0x7 << 29) & !(0x1ff << 18)) | (3 << 29),
        ];
        let (inst, words) = decode(&changed).unwrap();
        assert_eq!(words, 2);
        assert_eq!(inst.mods.abs, 5);
        assert_eq!(inst.mods.neg, 3);
        assert_eq!(&encode(&inst).unwrap()[..], changed);

        let zero_fill = [add[0], add[1] & !(0x1ff << 18)];
        let (mut lifted, _) = decode(&zero_fill).unwrap();
        lifted.prov.bytes = Some([add[0], add[1], 0]);
        assert_eq!(&encode(&lifted).unwrap()[..], zero_fill);
        assert_ne!(&encode(&lifted).unwrap()[..], add);

        let dot = [0xcc164000, 0x7c0e0501]; // dot op_sel/op_sel_hi are honored
        let dot_modified = [
            (dot[0] & !(0x7 << 11)) | (2 << 11),
            (dot[1] & !(3 << 27)) | (1 << 27),
        ];
        let (inst, _) = decode(&dot_modified).unwrap();
        assert_eq!(inst.mods.op_sel, 2);
        assert_eq!(inst.mods.op_sel_hi, 5);
        assert_eq!(&encode(&inst).unwrap()[..], dot_modified);
    }

    #[test]
    fn packed_neg_lo_and_combined_wait_are_typed() {
        let packed = [0xcc4a4039, 0x7a020347];
        let (inst, _) = decode(&packed).unwrap();
        assert_eq!(inst.mods.neg_lo, 3);
        assert_eq!(inst.mods.neg, 0);
        assert_eq!(&encode(&inst).unwrap()[..], packed);

        let wait = [0xbfc90102];
        let (inst, _) = decode(&wait).unwrap();
        let counters = &inst.mods.wait.as_ref().unwrap().per_counter;
        assert_eq!(counters[Counter::Store as usize], Some(1));
        assert_eq!(counters[Counter::Ds as usize], Some(2));
        assert_eq!(&encode(&inst).unwrap()[..], wait);
    }

    #[test]
    fn swmmac_index_key_is_op_sel_bit_zero_and_renders_before_neg_lo() {
        let (inst, _) = decode(&[0xcc504b00, 0x7c521908]).unwrap();
        assert_eq!(inst.mods.op_sel, 1);
        assert_eq!(
            inst.text(Arch::Gfx1201).unwrap(),
            "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20 \
             index_key:1 neg_lo:[1,1,0] neg_hi:[1,1,0]"
        );
        let (plain, _) = decode(&[0xcc504000, 0x1c521908]).unwrap();
        assert_eq!(
            plain.text(Arch::Gfx1201).unwrap(),
            "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20"
        );
        // Other packed rows keep their op_sel/op_sel_hi spelling.
        let dot = [0xcc164000, 0x7c0e0501];
        let (inst, _) = decode(&[(dot[0] & !(0x7 << 11)) | (2 << 11), dot[1]]).unwrap();
        assert!(inst.text(Arch::Gfx1201).unwrap().contains("op_sel:[0,1,0]"));
    }

    #[test]
    fn kt48_vopd_y_dest_agrees_with_objdump() {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const OBJDUMP: &str =
            include_str!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.objdump.txt");
        let kernel = &IMAGE[0x6f00..0x6f00 + 10_604];
        let words: Vec<_> = kernel
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut count = 0;
        for line in OBJDUMP.lines().filter(|line| line.contains(" :: ")) {
            let (text, location) = line.split_once("//").expect("objdump instruction comment");
            let (address, _) = location.split_once(':').expect("objdump PC");
            let address = usize::from_str_radix(address.trim(), 16).expect("hex objdump PC");
            if !(0x7f00..0x7f00 + 10_604).contains(&address) {
                continue;
            }
            let y_half = text.split_once(" :: ").expect("VOPD Y half").1;
            let y_dest = y_half
                .split_whitespace()
                .nth(1)
                .expect("Y destination")
                .trim_end_matches(',')
                .strip_prefix('v')
                .expect("VGPR destination")
                .parse::<u16>()
                .expect("VGPR index");
            let offset = (address - 0x7f00) / 4;
            let (inst, width) = decode(&words[offset..]).expect("decode VOPD packet");
            assert!((2..=3).contains(&width), "PC {address:#x}");
            let FormFields::Vopd { x_operands, .. } = inst.fields else {
                panic!("PC {address:#x} not VOPD");
            };
            let Operand::Reg(y) = &inst.operands[usize::from(x_operands)] else {
                panic!("PC {address:#x} Y not VGPR");
            };
            assert_eq!(y.base, y_dest, "PC {address:#x}: {text}");
            count += 1;
        }
        assert_eq!(
            count, 342,
            "all KT48 dual-issue packets compared to pinned objdump"
        );
    }

    #[test]
    fn vopd_rejects_destination_parity_collision() {
        let (mut inst, _) = decode(&[0xca1000ff, 0x010200c1, 0x0000c300]).unwrap();
        let FormFields::Vopd { x_operands, .. } = inst.fields else {
            panic!("not VOPD");
        };
        let Operand::Reg(y) = &mut inst.operands[usize::from(x_operands)] else {
            panic!("Y not VGPR");
        };
        y.base ^= 1;
        assert!(encode(&inst)
            .unwrap_err()
            .to_string()
            .contains("parity collides"));
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(512))]
        #[test]
        fn encoded_insts_recover_declared_form(
            row_index in 0usize..isa::gfx12().len(),
            seed in proptest::prelude::any::<u8>(),
        ) {
            let row=&isa::gfx12()[row_index];
            let words:Vec<u32>=row.encoding.split_whitespace()
                .map(|w| u32::from_str_radix(w,16).unwrap()).collect();
            let (mut inst,_) = decode(&words).unwrap();
            if let Some(first) = inst.operands.iter_mut()
                .find(|op| matches!(op,Operand::Reg(RegRef { kind: Kind::V,.. }))) {
                if let Operand::Reg(r)=first {
                    let bank_limit=if inst.form==Form::Vopd {128} else {256};
                    r.base=u16::from(seed) % (bank_limit+1-u16::from(r.len));
                }
            }
            if let FormFields::Vopd { x_operands, .. } = inst.fields {
                let Operand::Reg(x) = &inst.operands[0] else { panic!("X destination not VGPR"); };
                let x_parity = x.base & 1;
                let Operand::Reg(y) = &mut inst.operands[usize::from(x_operands)] else { panic!("Y destination not VGPR"); };
                y.base = (y.base & !1) | (x_parity ^ 1);
            }
            if let Some(literal)=&mut inst.literal {
                let replacement=*literal ^ (u32::from(seed)*0x0101_0101);
                *literal=replacement;
                for op in &mut inst.operands {
                    if let Operand::Literal(value)=op { *value=replacement; }
                }
            }
            if let FormFields::Vop3b { src2_unused }=&mut inst.fields {
                *src2_unused=if seed&1==0 {0} else {128};
            }
            inst=Inst::from_parts(Arch::Gfx1201,inst.op,inst.form,inst.fields.clone(),
                inst.operands.clone(),inst.mods.clone(),inst.literal,Provenance::default()).unwrap();
            let encoded=encode(&inst).unwrap();
            let (decoded,count)=decode(&encoded).unwrap();
            proptest::prop_assert_eq!(count,encoded.len());
            proptest::prop_assert_eq!(decoded,inst);
        }
    }
    #[test]
    fn global_address_width_matches_llvm_disassembly_in_both_modes() {
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        // Oracle test: needs ROCm's llvm-mc. The no-GPU CI runner has no ROCm,
        // so skip there instead of panicking on spawn; ROCm hosts still run it.
        let llvm_mc = crate::rocm_llvm_mc();
        if crate::rocm::tests_disabled() {
            eprintln!("skipping: explicitly disabled ROCm toolchain");
            return;
        }
        for (arch, cpu) in [
            (Arch::Gfx1100, "gfx1100"),
            (Arch::Gfx1151, "gfx1151"),
            (Arch::Gfx1201, "gfx1201"),
        ] {
            for (saddr, width) in [(4u32, 1u8), (124, 2)] {
                let words = if arch == Arch::Gfx1201 {
                    vec![0xee050000 | saddr, 10, 20]
                } else {
                    vec![0xdc520000, 0x0a000014 | saddr << 16]
                };
                let input = words
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .map(|b| format!("0x{b:02x}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                let mut mc = Command::new(&llvm_mc)
                    .args([
                        "-triple=amdgcn-amd-amdhsa",
                        &format!("-mcpu={cpu}"),
                        "-disassemble",
                    ])
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap();
                mc.stdin
                    .take()
                    .unwrap()
                    .write_all(format!("{input}\n").as_bytes())
                    .unwrap();
                let out = mc.wait_with_output().unwrap();
                assert!(
                    out.status.success(),
                    "{}",
                    String::from_utf8_lossy(&out.stderr)
                );
                let text = String::from_utf8_lossy(&out.stdout);
                assert!(
                    text.contains(if width == 1 {
                        "v10, v20, s[4:5]"
                    } else {
                        "v10, v[20:21], off"
                    }),
                    "{text}"
                );
                let (mut inst, _) = decode_for(arch, &words).unwrap();
                let addr = RegRef {
                    kind: Kind::V,
                    base: 20,
                    len: width,
                };
                assert!(inst.effects.uses.contains(&addr));
                assert!(!inst
                    .effects
                    .uses
                    .iter()
                    .any(|r| r.kind == Kind::V && r.base == 20 && r.len != width));
                assert_eq!(
                    encode_for(arch, &inst).unwrap().as_slice(),
                    words.as_slice()
                );
                let slot = inst
                    .operands
                    .iter()
                    .position(|o| *o == Operand::Reg(addr))
                    .unwrap();
                inst.operands[slot] = Operand::Reg(RegRef {
                    len: 3 - width,
                    ..addr
                });
                assert!(encode_for(arch, &inst).is_err());
            }
        }
    }
}

/// MQ4 RADV/ACO gfx1201 binary (full5476/executable5456, SHA d860d885...a680) scalar gaps,
/// disassembled from the raw words by ROCm llvm-mc; opcodes cross-checked against Mesa
/// 60e95b7 `aco_opcodes.py` (s_bfm_b32 gfx11+ = 0x2a, defs U32 only; s_cmp_lg_i32 = 0x01, defs SCC).
#[cfg(test)]
mod aco_pilot_scalar_tests {
    use super::*;
    use crate::effects::ImplicitSet;

    fn pinned(words: &[u32]) -> Inst {
        let (inst, used) = decode(words).unwrap();
        assert_eq!(used, words.len());
        assert_eq!(encode(&inst).unwrap().as_slice(), words);
        // Re-derive from typed fields alone; no decoded provenance participates.
        let rebuilt = Inst::from_parts(
            Arch::Gfx1201,
            inst.op,
            inst.form,
            inst.fields.clone(),
            inst.operands.clone(),
            inst.mods.clone(),
            inst.literal,
            Provenance::default(),
        )
        .unwrap();
        assert_eq!(encode(&rebuilt).unwrap().as_slice(), words);
        inst
    }
    fn s20() -> Operand {
        Operand::Reg(RegRef {
            kind: Kind::S,
            base: 20,
            len: 1,
        })
    }

    /// `s_mov_b32 s20, src_scc` (offset 164): SCC is a read of the implicit SCC bit, not a register.
    #[test]
    fn s_mov_b32_src_scc_is_a_precise_scc_read() {
        let inst = pinned(&[0xbe94_00fd]);
        assert_eq!(inst.op.name(Arch::Gfx1201), Some("s_mov_b32"));
        assert_eq!(
            inst.operands.as_slice(),
            &[s20(), Operand::Special(Special::Scc)]
        );
        assert_eq!(
            inst.effects.defs.as_slice(),
            &[RegRef {
                kind: Kind::S,
                base: 20,
                len: 1
            }]
        );
        assert!(inst.effects.uses.is_empty());
        assert_eq!(inst.effects.implicit.reads, ImplicitSet::SCC);
        assert_eq!(inst.effects.implicit.writes, 0);
        assert_eq!(inst.text(Arch::Gfx1201).unwrap(), "s_mov_b32 s20, scc");
    }
    #[test]
    fn scc_selector_is_a_32_bit_scalar_source_only() {
        // SCC cannot be a destination: the typed operand does not encode as SDST.
        let (mov, _) = decode(&[0xbe94_0080]).unwrap();
        let as_dest = Inst::from_parts(
            Arch::Gfx1201,
            mov.op,
            mov.form,
            FormFields::None,
            smallvec::smallvec![Operand::Special(Special::Scc), s20()],
            Modifiers::default(),
            None,
            Provenance::default(),
        )
        .unwrap();
        assert!(encode(&as_dest).is_err());
        // The 8-bit VOP3 compare destination field never decodes selector 0xfd as SCC.
        let e = decode(&[0xd44a_00fd, 0x0202_0e80]).unwrap_err().to_string();
        assert!(e.contains("unknown source selector 0xfd"), "{e}");
        // Not witnessed outside SOP1/SOP2/SOPC 32-bit sources: refused, not guessed.
        let e = decode(&[0x7e00_02fd]).unwrap_err().to_string(); // v_mov_b32_e32 v0, src_scc
        assert!(e.contains("SCC source selector"), "{e}");
    }
    /// `s_bfm_b32 vcc_lo, 16, 0` (offsets 1044/3168/3916/4648): explicit VCC_LO destination,
    /// no SCC effect (Mesa defines only the U32 result).
    #[test]
    fn s_bfm_b32_vcc_lo_defines_vcc_without_scc() {
        let inst = pinned(&[0x956a_8090]);
        assert_eq!(inst.op.name(Arch::Gfx1201), Some("s_bfm_b32"));
        assert_eq!(inst.form, Form::Sop2);
        assert_eq!(
            inst.operands.as_slice(),
            &[
                Operand::Special(Special::VccLo),
                Operand::Inline(InlineConst::Integer(16)),
                Operand::Inline(InlineConst::Integer(0))
            ]
        );
        assert!(inst.effects.defs.is_empty() && inst.effects.uses.is_empty());
        assert_eq!(inst.effects.implicit.writes, ImplicitSet::VCC);
        assert_eq!(inst.effects.implicit.reads, 0);
        // An ordinary SGPR destination has no implicit effect at all.
        let inst = pinned(&[0x9514_8090]); // s_bfm_b32 s20, 16, 0
        assert_eq!(
            inst.effects.defs.as_slice(),
            &[RegRef {
                kind: Kind::S,
                base: 20,
                len: 1
            }]
        );
        assert_eq!(
            (inst.effects.implicit.reads, inst.effects.implicit.writes),
            (0, 0)
        );
    }
    /// `s_cmp_lg_i32 s20, 0` (offset 5320): SOPC compare writes SCC and reads only its sources.
    #[test]
    fn s_cmp_lg_i32_writes_scc() {
        let inst = pinned(&[0xbf01_8014]);
        assert_eq!(inst.op.name(Arch::Gfx1201), Some("s_cmp_lg_i32"));
        assert_eq!(inst.form, Form::Sopc);
        assert_eq!(
            inst.operands.as_slice(),
            &[s20(), Operand::Inline(InlineConst::Integer(0))]
        );
        assert!(inst.effects.defs.is_empty());
        assert_eq!(
            inst.effects.uses.as_slice(),
            &[RegRef {
                kind: Kind::S,
                base: 20,
                len: 1
            }]
        );
        assert_eq!(inst.effects.implicit.writes, ImplicitSet::SCC);
        assert_eq!(inst.effects.implicit.reads, 0);
    }
    /// The new rows exist only in the gfx1201 table; targets without a row reject table-gated.
    #[test]
    fn new_rows_are_gated_to_the_gfx1201_table() {
        for words in [[0x956a_8090u32], [0xbf01_8014]] {
            for arch in [Arch::Gfx1100, Arch::Gfx1151, Arch::Gfx1030] {
                let e = decode_for(arch, &words).unwrap_err().to_string();
                assert!(e.contains("undefined"), "{arch:?} {words:08x?}: {e}");
            }
        }
    }
}

/// Opcode-fill census for gfx1201: 37 emitted-gap identities plus the Mesa 60e95b7 WMMA, dot and
/// SWMMAC identities (14 added here; (VOP3P,0x16), (VOP3P,0x44) and (VOP3P,0x56) already existed).
/// Samples are pinned ROCm `llvm-mc -mcpu=gfx1201` words; each opcode is cross-checked against
/// Mesa `aco_opcodes.py` (gfx12 column; VOP3-promoted VOPC/VOP2/VOP1 add +0/+0x100/+0x180).
#[cfg(test)]
mod opcode_fill_rows_tests {
    use super::*;
    use crate::effects::ImplicitSet;

    /// (table row, llvm-mc words, llvm text = table sample, Mesa name, Mesa format, Mesa gfx12 opcode)
    type Pinned = (
        &'static str,
        &'static [u32],
        &'static str,
        &'static str,
        &'static str,
        u16,
    );
    const NEW_ROWS: &[Pinned] = &[
        (
            "flat_load_i8",
            &[0xec04_407c, 0x0000_0001, 0x0000_0402],
            "flat_load_i8 v1, v[2:3] offset:4",
            "flat_load_sbyte",
            "FLAT",
            0x11,
        ),
        (
            "flat_load_b96",
            &[0xec05_807c, 0x0000_0001, 0x0000_0804],
            "flat_load_b96 v[1:3], v[4:5] offset:8",
            "flat_load_dwordx3",
            "FLAT",
            0x16,
        ),
        (
            "flat_store_b8",
            &[0xec06_007c, 0x0080_0000, 0x0000_0402],
            "flat_store_b8 v[2:3], v1 offset:4",
            "flat_store_byte",
            "FLAT",
            0x18,
        ),
        (
            "flat_store_b16",
            &[0xec06_407c, 0x0080_0000, 0x0000_0202],
            "flat_store_b16 v[2:3], v1 offset:2",
            "flat_store_short",
            "FLAT",
            0x19,
        ),
        (
            "flat_store_b96",
            &[0xec07_007c, 0x0200_0000, 0x0000_0c02],
            "flat_store_b96 v[2:3], v[4:6] offset:12",
            "flat_store_dwordx3",
            "FLAT",
            0x1c,
        ),
        (
            "flat_load_d16_u8",
            &[0xec07_807c, 0x0000_0001, 0x0000_0102],
            "flat_load_d16_u8 v1, v[2:3] offset:1",
            "flat_load_ubyte_d16",
            "FLAT",
            0x1e,
        ),
        (
            "flat_store_d16_hi_b16",
            &[0xec09_407c, 0x0080_0000, 0x0000_0202],
            "flat_store_d16_hi_b16 v[2:3], v1 offset:2",
            "flat_store_short_d16_hi",
            "FLAT",
            0x25,
        ),
        (
            "s_sext_i32_i8",
            &[0xbe87_0e07],
            "s_sext_i32_i8 s7, s7",
            "s_sext_i32_i8",
            "SOP1",
            0xe,
        ),
        (
            "s_bitset1_b32",
            &[0xbe86_129f],
            "s_bitset1_b32 s6, 31",
            "s_bitset1_b32",
            "SOP1",
            0x12,
        ),
        (
            "s_nor_b32",
            &[0x8f03_0403],
            "s_nor_b32 s3, s3, s4",
            "s_nor_b32",
            "SOP2",
            0x1e,
        ),
        ("v_nop", &[0x7e00_0000], "v_nop", "v_nop", "VOP1", 0x0),
        (
            "v_sqrt_f32_e32",
            &[0x7e02_6702],
            "v_sqrt_f32_e32 v1, v2",
            "v_sqrt_f32",
            "VOP1",
            0x33,
        ),
        (
            "v_cls_i32_e32",
            &[0x7e02_7702],
            "v_cls_i32_e32 v1, v2",
            "v_ffbh_i32",
            "VOP1",
            0x3b,
        ),
        (
            "v_cvt_f16_i16_e32",
            &[0x7e02_a302],
            "v_cvt_f16_i16_e32 v1.l, v2.l",
            "v_cvt_f16_i16",
            "VOP1",
            0x51,
        ),
        (
            "v_cvt_u32_u16_e32",
            &[0x7e02_d702],
            "v_cvt_u32_u16_e32 v1, v2.l",
            "v_cvt_u32_u16",
            "VOP1",
            0x6b,
        ),
        (
            "v_sub_co_ci_u32_e32",
            &[0x4202_0b02],
            "v_sub_co_ci_u32_e32 v1, vcc_lo, v2, v5, vcc_lo",
            "v_subb_co_u32",
            "VOP2",
            0x21,
        ),
        (
            "v_mul_f16_e32",
            &[0x6a02_0702],
            "v_mul_f16_e32 v1.l, v2.l, v3.l",
            "v_mul_f16",
            "VOP2",
            0x35,
        ),
        (
            "v_cmp_nlg_f32_e64",
            &[0xd41a_0000, 0x0202_0501],
            "v_cmp_nlg_f32_e64 s0, v1, v2",
            "v_cmp_nlg_f32",
            "VOPC",
            0x1a,
        ),
        (
            "v_cmp_gt_i16_e64",
            &[0xd434_0000, 0x0202_0501],
            "v_cmp_gt_i16_e64 s0, v1.l, v2.l",
            "v_cmp_gt_i16",
            "VOPC",
            0x34,
        ),
        (
            "v_cmpx_le_u64_e64",
            &[0xd4db_007e, 0x0202_0610],
            "v_cmpx_le_u64_e64 s[16:17], v[3:4]",
            "v_cmpx_le_u64",
            "VOPC",
            0xdb,
        ),
        (
            "v_mul_f16_e64",
            &[0xd535_0001, 0x0202_0702],
            "v_mul_f16_e64 v1.l, v2.l, v3.l",
            "v_mul_f16",
            "VOP2",
            0x35,
        ),
        (
            "v_cvt_f16_i16_e64",
            &[0xd5d1_0001, 0x0201_0102],
            "v_cvt_f16_i16_e64 v1.l, v2.l",
            "v_cvt_f16_i16",
            "VOP1",
            0x51,
        ),
        (
            "v_cvt_u32_u16_e64",
            &[0xd5eb_0001, 0x0201_0102],
            "v_cvt_u32_u16_e64 v1, v2.l",
            "v_cvt_u32_u16",
            "VOP1",
            0x6b,
        ),
        (
            "v_mad_i32_i24",
            &[0xd60a_0001, 0x0412_0702],
            "v_mad_i32_i24 v1, v2, v3, v4",
            "v_mad_i32_i24",
            "VOP3",
            0x20a,
        ),
        (
            "v_subrev_co_u32",
            &[0xd702_0102, 0x0202_0f09],
            "v_subrev_co_u32 v2, s1, v9, v7",
            "v_subrev_co_u32_e64",
            "VOP3",
            0x302,
        ),
        (
            "v_mul_lo_u16",
            &[0xd705_0001, 0x0202_0702],
            "v_mul_lo_u16 v1.l, v2.l, v3.l",
            "v_mul_lo_u16_e64",
            "VOP3",
            0x305,
        ),
        (
            "v_lshrrev_b64",
            &[0xd73d_0009, 0x0202_0709],
            "v_lshrrev_b64 v[9:10], v9, v[3:4]",
            "v_lshrrev_b64",
            "VOP3",
            0x33d,
        ),
        (
            "v_wmma_f32_16x16x16_bf16",
            &[0xcc41_4000, 0x1c02_1908],
            "v_wmma_f32_16x16x16_bf16 v[0:7], v[8:11], v[12:15], v[0:7]",
            "v_wmma_f32_16x16x16_bf16",
            "VOP3P",
            0x41,
        ),
        (
            "v_wmma_bf16_16x16x16_bf16",
            &[0xcc43_4000, 0x1c02_1104],
            "v_wmma_bf16_16x16x16_bf16 v[0:3], v[4:7], v[8:11], v[0:3]",
            "v_wmma_bf16_16x16x16_bf16",
            "VOP3P",
            0x43,
        ),
        (
            "v_swmmac_f32_16x16x32_f16",
            &[0xcc50_4000, 0x1c52_1908],
            "v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20",
            "v_swmmac_f32_16x16x32_f16",
            "VOP3P",
            0x50,
        ),
        (
            "v_swmmac_f32_16x16x32_bf16",
            &[0xcc51_4000, 0x1c52_1908],
            "v_swmmac_f32_16x16x32_bf16 v[0:7], v[8:11], v[12:19], v20",
            "v_swmmac_f32_16x16x32_bf16",
            "VOP3P",
            0x51,
        ),
        (
            "v_swmmac_f16_16x16x32_f16",
            &[0xcc52_4000, 0x1c42_1104],
            "v_swmmac_f16_16x16x32_f16 v[0:3], v[4:7], v[8:15], v16",
            "v_swmmac_f16_16x16x32_f16",
            "VOP3P",
            0x52,
        ),
        (
            "v_swmmac_bf16_16x16x32_bf16",
            &[0xcc53_4000, 0x1c42_1104],
            "v_swmmac_bf16_16x16x32_bf16 v[0:3], v[4:7], v[8:15], v16",
            "v_swmmac_bf16_16x16x32_bf16",
            "VOP3P",
            0x53,
        ),
        (
            "v_swmmac_i32_16x16x32_iu8",
            &[0xcc54_4000, 0x1c3a_1508],
            "v_swmmac_i32_16x16x32_iu8 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_i32_16x16x32_iu8",
            "VOP3P",
            0x54,
        ),
        (
            "v_swmmac_i32_16x16x32_iu4",
            &[0xcc55_4000, 0x1c2e_1308],
            "v_swmmac_i32_16x16x32_iu4 v[0:7], v8, v[9:10], v11",
            "v_swmmac_i32_16x16x32_iu4",
            "VOP3P",
            0x55,
        ),
        (
            "v_swmmac_f32_16x16x32_fp8_fp8",
            &[0xcc57_4000, 0x1c3a_1508],
            "v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_f32_16x16x32_fp8_fp8",
            "VOP3P",
            0x57,
        ),
        (
            "v_swmmac_f32_16x16x32_fp8_bf8",
            &[0xcc58_4000, 0x1c3a_1508],
            "v_swmmac_f32_16x16x32_fp8_bf8 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_f32_16x16x32_fp8_bf8",
            "VOP3P",
            0x58,
        ),
        (
            "v_swmmac_f32_16x16x32_bf8_fp8",
            &[0xcc59_4000, 0x1c3a_1508],
            "v_swmmac_f32_16x16x32_bf8_fp8 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_f32_16x16x32_bf8_fp8",
            "VOP3P",
            0x59,
        ),
        (
            "v_swmmac_f32_16x16x32_bf8_bf8",
            &[0xcc5a_4000, 0x1c3a_1508],
            "v_swmmac_f32_16x16x32_bf8_bf8 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_f32_16x16x32_bf8_bf8",
            "VOP3P",
            0x5a,
        ),
        (
            "v_dot2_f32_f16",
            &[0xcc13_4000, 0x1c0e_0501],
            "v_dot2_f32_f16 v0, v1, v2, v3",
            "v_dot2_f32_f16",
            "VOP3P",
            0x13,
        ),
        (
            "v_dot2_f32_bf16",
            &[0xcc1a_4000, 0x1c0e_0501],
            "v_dot2_f32_bf16 v0, v1, v2, v3",
            "v_dot2_f32_bf16",
            "VOP3P",
            0x1a,
        ),
        (
            "v_dot8_i32_iu4",
            &[0xcc18_4000, 0x7c0e_0501],
            "v_dot8_i32_iu4 v0, v1, v2, v3 neg_lo:[1,1,0]",
            "v_dot8_i32_iu4",
            "VOP3P",
            0x18,
        ),
        (
            "v_cmp_nlg_f32_e32",
            &[0x7c34_0501],
            "v_cmp_nlg_f32_e32 vcc_lo, v1, v2",
            "v_cmp_nlg_f32",
            "VOPC",
            0x1a,
        ),
        (
            "v_cmp_le_u64_e32",
            &[0x7cb6_0602],
            "v_cmp_le_u64_e32 vcc_lo, s[2:3], v[3:4]",
            "v_cmp_le_u64",
            "VOPC",
            0x5b,
        ),
        (
            "v_cmp_gt_u64_e32",
            &[0x7cb8_0602],
            "v_cmp_gt_u64_e32 vcc_lo, s[2:3], v[3:4]",
            "v_cmp_gt_u64",
            "VOPC",
            0x5c,
        ),
        (
            "v_cmp_ne_u64_e32",
            &[0x7cba_0e80],
            "v_cmp_ne_u64_e32 vcc_lo, 0, v[7:8]",
            "v_cmp_lg_u64",
            "VOPC",
            0x5d,
        ),
        (
            "v_cmpx_ge_i32_e32",
            &[0x7d8c_0501],
            "v_cmpx_ge_i32_e32 v1, v2",
            "v_cmpx_ge_i32",
            "VOPC",
            0xc6,
        ),
        (
            "v_cmpx_ge_u32_e32",
            &[0x7d9c_1a0f],
            "v_cmpx_ge_u32_e32 s15, v13",
            "v_cmpx_ge_u32",
            "VOPC",
            0xce,
        ),
        (
            "v_cmpx_lt_i64_e32",
            &[0x7da2_0602],
            "v_cmpx_lt_i64_e32 s[2:3], v[3:4]",
            "v_cmpx_lt_i64",
            "VOPC",
            0xd1,
        ),
        (
            "v_cmpx_eq_u64_e32",
            &[0x7db4_0e80],
            "v_cmpx_eq_u64_e32 0, v[7:8]",
            "v_cmpx_eq_u64",
            "VOPC",
            0xda,
        ),
        (
            "v_cmpx_gt_u64_e32",
            &[0x7db8_0602],
            "v_cmpx_gt_u64_e32 s[2:3], v[3:4]",
            "v_cmpx_gt_u64",
            "VOPC",
            0xdc,
        ),
    ];
    /// Rows that predate the fill. `(VOP3P,0x16)` is `v_dot4_i32_iu8` (canonical ACO/LLVM name; the old
    /// alias `v_dot4_i32_i8` is iu8 with neg_lo:[1,1,0]); operand signedness comes from neg_lo.
    const EXISTING_USEFUL: &[Pinned] = &[
        (
            "v_dot4_i32_iu8",
            &[0xcc16_4000, 0x7c0e_0501],
            "v_dot4_i32_iu8 v0, v1, v2, v3 neg_lo:[1,1,0]",
            "v_dot4_i32_iu8",
            "VOP3P",
            0x16,
        ),
        (
            "v_dot4_i32_iu8",
            &[0xcc16_4000, 0x1c0e_0501],
            "v_dot4_i32_iu8 v0, v1, v2, v3",
            "v_dot4_i32_iu8",
            "VOP3P",
            0x16,
        ),
        (
            "v_wmma_i32_16x16x16_iu8",
            &[0xcc44_4008, 0x7c22_0900],
            "v_wmma_i32_16x16x16_iu8 v[8:15], v[0:1], v[4:5], v[8:15] neg_lo:[1,1,0]",
            "v_wmma_i32_16x16x16_iu8",
            "VOP3P",
            0x44,
        ),
        (
            "v_swmmac_i32_16x16x64_iu4",
            &[0xcc56_4000, 0x1c3a_1508],
            "v_swmmac_i32_16x16x64_iu4 v[0:7], v[8:9], v[10:13], v14",
            "v_swmmac_i32_16x16x64_iu4",
            "VOP3P",
            0x56,
        ),
    ];

    fn mesa_family(form: Form) -> &'static [&'static str] {
        match form {
            Form::Sop1 => &["SOP1"],
            Form::Sop2 => &["SOP2"],
            Form::Vop1 => &["VOP1"],
            Form::Vop2 => &["VOP2"],
            Form::Vopc => &["VOPC"],
            Form::Vop3p => &["VOP3P"],
            Form::Vmem(crate::inst::VmemForm::Flat) => &["FLAT"],
            Form::Vop3 => &["VOP3", "VOPC", "VOP2", "VOP1"],
            other => panic!("unexpected form {other:?}"),
        }
    }
    fn mesa_offset(mesa_format: &str, form: Form) -> u16 {
        match (mesa_format, form) {
            ("VOP2", Form::Vop3) => 0x100,
            ("VOP1", Form::Vop3) => 0x180,
            _ => 0,
        }
    }
    fn v(base: u16, len: u8) -> RegRef {
        RegRef {
            kind: Kind::V,
            base,
            len,
        }
    }
    fn s(base: u16, len: u8) -> RegRef {
        RegRef {
            kind: Kind::S,
            base,
            len,
        }
    }
    fn inst(words: &[u32]) -> Inst {
        decode(words).unwrap().0
    }

    #[test]
    fn every_filled_row_matches_mesa_table_and_llvm_words() {
        let mut failures = Vec::new();
        for (pinned, check_table) in NEW_ROWS
            .iter()
            .map(|p| (p, true))
            .chain(EXISTING_USEFUL.iter().map(|p| (p, false)))
        {
            let &(name, words, text, mesa_name, mesa_format, mesa_op) = pinned;
            let result = (|| -> Result<(), String> {
                let (inst, used) = decode(words).map_err(|e| e.to_string())?;
                if used != words.len() {
                    return Err(format!("width {used} != {}", words.len()));
                }
                if inst.op.name(Arch::Gfx1201) != Some(name) {
                    return Err(format!("decoded as {:?}", inst.op.name(Arch::Gfx1201)));
                }
                let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).ok_or("no table row")?;
                if check_table {
                    let table: Vec<u32> = row
                        .encoding
                        .split_whitespace()
                        .map(|w| u32::from_str_radix(w, 16).unwrap())
                        .collect();
                    if row.sample != text || table != words {
                        return Err(format!(
                            "table sample {:?} {:?} != pinned {text:?} {words:08x?}",
                            row.sample, row.encoding
                        ));
                    }
                }
                if !mesa_family(inst.form).contains(&mesa_format) {
                    return Err(format!(
                        "{mesa_name} is Mesa {mesa_format}, row is {:?}",
                        inst.form
                    ));
                }
                if inst.op.id != mesa_op + mesa_offset(mesa_format, inst.form) {
                    return Err(format!(
                        "{mesa_name} gfx12 opcode {mesa_op:#x} does not map to row opcode {:#x}",
                        inst.op.id
                    ));
                }
                if encode(&inst).map_err(|e| e.to_string())?.as_slice() != words {
                    return Err("re-encode differs".into());
                }
                // Typed fields alone (no decoded provenance) rebuild the same words.
                let rebuilt = Inst::from_parts(
                    Arch::Gfx1201,
                    inst.op,
                    inst.form,
                    inst.fields.clone(),
                    inst.operands.clone(),
                    inst.mods.clone(),
                    inst.literal,
                    Provenance::default(),
                )
                .map_err(|e| e.to_string())?;
                if encode(&rebuilt).map_err(|e| e.to_string())?.as_slice() != words {
                    return Err("typed rebuild differs".into());
                }
                if decode(&encode(&inst).unwrap())
                    .map_err(|e| e.to_string())?
                    .0
                    != inst
                {
                    return Err("decode(encode(i)) != i".into());
                }
                Ok(())
            })();
            if let Err(error) = result {
                failures.push(format!("{name} {words:08x?}: {error}"));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert_eq!(NEW_ROWS.len(), 51);
    }

    #[test]
    fn filled_rows_carry_source_verified_effects() {
        // SWMMAC reads and writes its accumulator VDST and reads the A, B and sparse-index registers.
        let swmmac = inst(&[0xcc50_4000, 0x1c52_1908]); // v_swmmac_f32_16x16x32_f16 v[0:7], v[8:11], v[12:19], v20
        assert_eq!(swmmac.effects.defs.as_slice(), &[v(0, 8)]);
        for reg in [v(0, 8), v(8, 4), v(12, 8), v(20, 1)] {
            assert!(swmmac.effects.uses.contains(&reg), "{reg:?}");
        }
        let wmma = inst(&[0xcc41_4000, 0x1c02_1908]); // v_wmma_f32_16x16x16_bf16 v[0:7], v[8:11], v[12:15], v[0:7]
        assert_eq!(wmma.effects.defs.as_slice(), &[v(0, 8)]);
        for reg in [v(8, 4), v(12, 4), v(0, 8)] {
            assert!(wmma.effects.uses.contains(&reg), "{reg:?}");
        }
        let dot = inst(&[0xcc13_4000, 0x1c0e_0501]); // v_dot2_f32_f16 v0, v1, v2, v3
        assert_eq!(dot.effects.defs.as_slice(), &[v(0, 1)]);
        assert_eq!(dot.effects.uses.as_slice(), &[v(1, 1), v(2, 1), v(3, 1)]);
        // FLAT: Load/Ds and Store/Ds counters; D16 loads preserve the other half of VDST.
        let load = inst(&[0xec07_807c, 0x0000_0001, 0x0000_0102]); // flat_load_d16_u8 v1, v[2:3] offset:1
        assert_eq!(load.effects.defs.as_slice(), &[v(1, 1)]);
        assert!(load.effects.uses.contains(&v(1, 1)) && load.effects.uses.contains(&v(2, 2)));
        let counters = &load.effects.mem.as_ref().unwrap().counters;
        assert!(counters.contains(Counter::Load) && counters.contains(Counter::Ds));
        let store = inst(&[0xec07_007c, 0x0200_0000, 0x0000_0c02]); // flat_store_b96 v[2:3], v[4:6] offset:12
        assert!(store.effects.defs.is_empty());
        assert!(store.effects.uses.contains(&v(2, 2)) && store.effects.uses.contains(&v(4, 3)));
        let counters = &store.effects.mem.as_ref().unwrap().counters;
        assert!(counters.contains(Counter::Store) && counters.contains(Counter::Ds));
        // SALU.
        let nor = inst(&[0x8f03_0403]); // s_nor_b32 s3, s3, s4: writes SCC
        assert_eq!(
            (nor.effects.implicit.reads, nor.effects.implicit.writes),
            (0, ImplicitSet::SCC)
        );
        let bitset = inst(&[0xbe86_129f]); // s_bitset1_b32 s6, 31: read-modify-write of SDST, SCC untouched
        assert_eq!(bitset.effects.defs.as_slice(), &[s(6, 1)]);
        assert!(bitset.effects.uses.contains(&s(6, 1)));
        assert_eq!(
            (
                bitset.effects.implicit.reads,
                bitset.effects.implicit.writes
            ),
            (0, 0)
        );
        // VALU implicit state.
        let nop = inst(&[0x7e00_0000]);
        assert!(
            nop.effects.defs.is_empty() && nop.effects.uses.is_empty() && nop.operands.is_empty()
        );
        assert_eq!(
            (nop.effects.implicit.reads, nop.effects.implicit.writes),
            (0, 0)
        );
        let subb = inst(&[0x4202_0b02]); // v_sub_co_ci_u32_e32 v1, vcc_lo, v2, v5, vcc_lo
        assert_eq!(
            (subb.effects.implicit.reads, subb.effects.implicit.writes),
            (ImplicitSet::VCC, ImplicitSet::VCC)
        );
        let cmp = inst(&[0x7cb6_0602]); // v_cmp_le_u64_e32 vcc_lo, s[2:3], v[3:4]
        assert_eq!(
            (cmp.effects.implicit.reads, cmp.effects.implicit.writes),
            (0, ImplicitSet::VCC)
        );
        assert!(cmp.effects.defs.is_empty());
        assert!(cmp.effects.uses.contains(&s(2, 2)) && cmp.effects.uses.contains(&v(3, 2)));
        for words in [&[0x7d8c_0501u32][..], &[0xd4db_007e, 0x0202_0610]] {
            // v_cmpx_ge_i32_e32, v_cmpx_le_u64_e64
            let cmpx = inst(words);
            assert_eq!(
                (cmpx.effects.implicit.reads, cmpx.effects.implicit.writes),
                (0, ImplicitSet::EXEC)
            );
            assert!(cmpx.effects.defs.is_empty());
        }
        let subrev = inst(&[0xd702_0102, 0x0202_0f09]); // v_subrev_co_u32 v2, s1, v9, v7
        assert!(subrev.effects.defs.contains(&v(2, 1)) && subrev.effects.defs.contains(&s(1, 2)));
        // 16-bit destinations write one half and therefore read the old VGPR.
        for words in [
            &[0x7e02_a302u32][..],
            &[0x6a02_0702],
            &[0xd535_0001, 0x0202_0702],
            &[0xd5d1_0001, 0x0201_0102],
            &[0xd705_0001, 0x0202_0702],
        ] {
            let half = inst(words);
            assert_eq!(half.effects.defs.as_slice(), &[v(1, 1)], "{words:08x?}");
            assert!(half.effects.uses.contains(&v(1, 1)), "{words:08x?}");
        }
    }

    /// Every declared don't-care / literal combination of the new rows roundtrips on its own sample.
    #[test]
    fn filled_rows_roundtrip_all_declared_dont_care_values() {
        for &(name, words, ..) in NEW_ROWS {
            let (inst, _) = decode(words).unwrap();
            let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
            for rule in row.fields.iter().filter(|r| r.class == FieldClass::Ignored) {
                let Some(placement) = (match rule.name {
                    "w0_extra" => Some((0, 0)),
                    "w1_extra" | "vsrc_unused" => Some((1, 0)),
                    "src2_unused" => Some((1, 18)),
                    "src1_unused" => Some((1, 9)),
                    _ => None,
                }) else {
                    continue;
                };
                for &allowed in &rule.allowed {
                    let mut variant = words.to_vec();
                    let mask = rule.mask << placement.1;
                    variant[placement.0] =
                        (variant[placement.0] & !mask) | (allowed << placement.1);
                    let (decoded, used) =
                        decode(&variant).unwrap_or_else(|e| panic!("{name} {variant:08x?}: {e}"));
                    assert_eq!(used, variant.len(), "{name}");
                    assert_eq!(
                        encode(&decoded).unwrap().as_slice(),
                        variant.as_slice(),
                        "{name} {variant:08x?}"
                    );
                }
            }
        }
    }
}

/// gfx1201 forms found missing while lifting the rows-6 decode-layer hipcc bodies (`standalone/*/sa.co`). Every word
/// is what `llvm-mc --arch=amdgcn --mcpu=gfx1201 --show-encoding` assembles from the text beside it.
#[cfg(test)]
mod rows6_gap_tests {
    use super::*;

    /// (table row, llvm-mc words, llvm text = table sample)
    const NEW_ROWS: &[(&str, &[u32], &str)] = &[
        (
            "buffer_load_i8",
            &[0xc404_4010, 0x4080_6801, 0x0000_0230],
            "buffer_load_i8 v1, v48, s[52:55], s16 offen offset:2",
        ),
        (
            "v_bcnt_u32_b32",
            &[0xd71e_0002, 0x0201_0102],
            "v_bcnt_u32_b32 v2, v2, 0",
        ),
        (
            "v_cmp_ge_u64_e32",
            &[0x7cbc_0504],
            "v_cmp_ge_u64_e32 vcc_lo, v[4:5], v[2:3]",
        ),
        (
            "v_cmp_gt_u16_e64",
            &[0xd43c_0000, 0x0202_0494],
            "v_cmp_gt_u16_e64 s0, 20, v2.l",
        ),
        (
            "v_cmpx_ge_u64_e32",
            &[0x7dbc_0505],
            "v_cmpx_ge_u64_e32 v[5:6], v[2:3]",
        ),
        (
            "v_cmpx_gt_u16_e32",
            &[0x7d78_0894],
            "v_cmpx_gt_u16_e32 20, v4.l",
        ),
        (
            "v_cmpx_le_u32_e32",
            &[0x7d96_0506],
            "v_cmpx_le_u32_e32 v6, v2",
        ),
        (
            "v_cmpx_lt_u64_e32",
            &[0x7db2_ceff, 0x0000_270f],
            "v_cmpx_lt_u64_e32 0x270f, v[103:104]",
        ),
        (
            "v_mul_hi_i32_i24_e32",
            &[0x143a_32ff, 0x0000_5000],
            "v_mul_hi_i32_i24_e32 v29, 0x5000, v25",
        ),
        (
            "v_mov_b32_dpp",
            &[0x7e06_02e9, 0x688f_ac03],
            "v_mov_b32_dpp v3, v3 dpp8:[4,5,6,7,0,1,2,3]",
        ),
    ];

    /// Existing rows whose cache-policy field list widened: `scope:SCOPE_SYS` on FLAT and `TH_LOAD_LU` on SCRATCH.
    const WIDENED: &[(&str, &[u32], &str)] = &[
        (
            "flat_load_b32",
            &[0xec05_007c, 0x000c_0003, 0x0000_0001],
            "flat_load_b32 v3, v[1:2] scope:SCOPE_SYS",
        ),
        (
            "scratch_load_b32",
            &[0xed05_007c, 0x0030_0003, 0x0000_0000],
            "scratch_load_b32 v3, off, off th:TH_LOAD_LU",
        ),
    ];

    fn v(base: u16) -> Operand {
        Operand::Reg(RegRef {
            kind: Kind::V,
            base,
            len: 1,
        })
    }

    fn roundtrip(name: &str, words: &[u32]) -> Inst {
        let (inst, used) = decode(words).unwrap_or_else(|e| panic!("{name} {words:08x?}: {e}"));
        assert_eq!(used, words.len(), "{name}");
        let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).expect("table row");
        assert_eq!(row.name, name, "{words:08x?}");
        assert_eq!(encode(&inst).unwrap().as_slice(), words, "{name}");
        // Typed fields alone (no decoded provenance) rebuild the same words.
        let rebuilt = Inst::from_parts(
            Arch::Gfx1201,
            inst.op,
            inst.form,
            inst.fields.clone(),
            inst.operands.clone(),
            inst.mods.clone(),
            inst.literal,
            Provenance::default(),
        )
        .unwrap();
        assert_eq!(encode(&rebuilt).unwrap().as_slice(), words, "{name} typed");
        assert_eq!(decode(&encode(&inst).unwrap()).unwrap().0, inst, "{name}");
        inst
    }

    #[test]
    fn new_rows_match_their_llvm_words_and_table_samples() {
        for &(name, words, text) in NEW_ROWS {
            let inst = roundtrip(name, words);
            let row = isa::lookup(Arch::Gfx1201, inst.op, inst.form).unwrap();
            let table: Vec<u32> = row
                .encoding
                .split_whitespace()
                .map(|w| u32::from_str_radix(w, 16).unwrap())
                .collect();
            assert_eq!((row.sample, table.as_slice()), (text, words), "{name}");
        }
    }

    #[test]
    fn widened_cache_policy_fields_roundtrip() {
        for &(name, words, _) in WIDENED {
            roundtrip(name, words);
        }
        let (flat, _) = decode(WIDENED[0].1).unwrap();
        let scope = flat
            .fields
            .honored()
            .iter()
            .find(|f| f.name == "global_scope");
        assert_eq!(scope.map(|f| f.value), Some(0xc0000));
        let (scratch, _) = decode(WIDENED[1].1).unwrap();
        let th = scratch
            .fields
            .honored()
            .iter()
            .find(|f| f.name == "global_th");
        assert_eq!(th.map(|f| f.value), Some(0x30_0000));
    }

    #[test]
    fn compare_rows_keep_their_implicit_mask_effects() {
        // The new e32 compares write VCC (cmp) or EXEC (cmpx) implicitly, exactly like their existing siblings.
        let (cmp, _) = decode(&[0x7cbc_0504]).unwrap();
        assert_ne!(
            cmp.effects.implicit.writes & crate::effects::ImplicitSet::VCC,
            0
        );
        let (cmpx, _) = decode(&[0x7dbc_0505]).unwrap();
        assert_ne!(
            cmpx.effects.implicit.writes & crate::effects::ImplicitSet::EXEC,
            0
        );
        assert_eq!(
            cmpx.effects.implicit.writes & crate::effects::ImplicitSet::VCC,
            0
        );
    }

    #[test]
    fn dpp8_is_its_own_form_with_typed_lane_selects() {
        let words = [0x7e06_02e9, 0x688f_ac03];
        let (inst, used) = decode(&words).unwrap();
        assert_eq!(used, 2);
        assert_eq!(inst.form, Form::Vop1Dpp8);
        assert_eq!(inst.operands.as_slice(), &[v(3), v(3)]);
        assert!(inst.mods.dpp.is_none());
        let select = inst.fields.honored().iter().find(|f| f.name == "dpp8_sel");
        assert_eq!(select.map(|f| f.value), Some(0x688f_ac00));
        // Other lane selects encode through the typed field alone.
        let identity = Inst::from_parts(
            Arch::Gfx1201,
            inst.op,
            inst.form,
            FormFields::Bits {
                ignored: SmallVec::new(),
                honored: smallvec::smallvec![NamedField {
                    name: "dpp8_sel",
                    value: 0xfac6_8800
                }],
            },
            inst.operands.clone(),
            inst.mods.clone(),
            None,
            Provenance::default(),
        )
        .unwrap();
        assert_eq!(
            encode(&identity).unwrap().as_slice(),
            &[0x7e06_02e9, 0xfac6_8803]
        );
        // DPP16 shares the opcode and still decodes as DPP16; the fetch-inactive selector (0xea) has no row.
        assert_eq!(
            decode(&[0x7e04_02fa, 0xff09_0101]).unwrap().0.form,
            Form::Vop1Dpp
        );
        assert!(decode(&[0x7e06_02ea, 0x688f_ac03]).is_err());
    }
}
