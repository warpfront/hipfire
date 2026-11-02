// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed gfx1201 instructions for edit scripts, built from the opcode table.
//!
//! [`inst`] takes a mnemonic and typed operands. Encoding don't-cares come from the
//! row's pinned `llvm-mc` example (the assembler's own fill, e.g. VOP3 unused `src2`
//! 0x80), the honored cache-policy fields follow the typed `mods`, and the result must
//! encode (labels aside) — so a script instruction is exactly what the assembler would
//! emit for the same text.

use peacemaker_ir::cfg::BlockId;
use peacemaker_ir::codec::gfx12;
use peacemaker_ir::inst::{Arch, FormFields, Inst};
use peacemaker_ir::operand::{ImmField, InlineConst, Modifiers, Msg, Operand, Special};
use peacemaker_ir::provenance::Provenance;
use peacemaker_ir::reg::{Kind, RegRef};

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("no gfx1201 table row named {0}")]
    UnknownMnemonic(String),
    #[error("{name}: {reason}")]
    Invalid { name: String, reason: String },
}

/// `name operands` with `mods`; see the module docs.
pub fn inst(name: &str, operands: Vec<Operand>, mods: Modifiers) -> Result<Inst, BuildError> {
    let row = peacemaker_ir::isa::gfx12()
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| BuildError::UnknownMnemonic(name.into()))?;
    let bad = |reason: String| BuildError::Invalid {
        name: name.into(),
        reason,
    };
    let example: Vec<u32> = row
        .encoding
        .split_whitespace()
        .map(|w| u32::from_str_radix(w, 16).expect("table dword"))
        .collect();
    let (template, _) = gfx12::decode(&example).map_err(|e| bad(format!("table example: {e}")))?;
    let mut fields = template.fields;
    if let FormFields::Bits { honored, .. } = &mut fields {
        for field in honored.iter_mut() {
            match field.name {
                "global_th" => field.value = u32::from(mods.cpol.th) << 20,
                "global_scope" => field.value = u32::from(mods.cpol.scope) << 18,
                "global_nv" => field.value = u32::from(mods.cpol.nv) << 7,
                _ => {}
            }
        }
    }
    let literal = operands.iter().find_map(|op| match op {
        Operand::Literal(v) => Some(*v),
        _ => None,
    });
    let built = Inst::from_parts(
        Arch::Gfx1201,
        row.op,
        row.form,
        fields,
        operands.into(),
        mods,
        literal,
        Provenance::default(),
    )
    .map_err(|e| bad(e.to_string()))?;
    let mut probe = built.clone();
    for op in probe.operands.iter_mut() {
        if matches!(op, Operand::Label(_)) {
            *op = Operand::Imm(ImmField::Sopp(0));
        }
    }
    gfx12::encode(&probe).map_err(|e| bad(e.to_string()))?;
    Ok(built)
}

/// `name operands` without modifiers.
pub fn op(name: &str, operands: Vec<Operand>) -> Result<Inst, BuildError> {
    inst(name, operands, Modifiers::default())
}

pub fn s(n: u16) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::S,
        base: n,
        len: 1,
    })
}
pub fn s2(n: u16) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::S,
        base: n,
        len: 2,
    })
}
pub fn s4(n: u16) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::S,
        base: n,
        len: 4,
    })
}
pub fn v(n: u16) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::V,
        base: n,
        len: 1,
    })
}
pub fn ttmp(n: u16) -> Operand {
    Operand::Reg(RegRef {
        kind: Kind::Ttmp,
        base: n,
        len: 1,
    })
}
pub fn exec_lo() -> Operand {
    Operand::Special(Special::ExecLo)
}
pub fn label(block: BlockId) -> Operand {
    Operand::Label(block)
}

/// An immediate as the assembler encodes it: an inline constant in -16..=64, else a
/// 32-bit literal.
pub fn imm(value: i64) -> Operand {
    match i8::try_from(value) {
        Ok(n) if (-16..=64).contains(&n) => Operand::Inline(InlineConst::Integer(n)),
        _ => Operand::Literal(value as u32),
    }
}

/// SOPP simm16 (`s_setprio`, `s_wait_alu`, `s_wait_*cnt` raw fields).
pub fn simm16(value: u16) -> Operand {
    Operand::Imm(ImmField::Sopp(value as i16))
}

/// SMEM immediate byte offset.
pub fn smem_offset(offset: i32) -> Operand {
    Operand::Imm(ImmField::SmemOffset(offset))
}

/// VMEM immediate byte offset (`offset:-8`).
pub fn vmem_offset(offset: i32) -> Operand {
    Operand::Imm(ImmField::VmemOffset(offset))
}

/// `hwreg(id, offset, size)` as the SOPK simm16 the codec types it as.
pub fn hwreg(id: u16, offset: u16, size: u16) -> Operand {
    Operand::Imm(ImmField::Sopk((id | offset << 6 | (size - 1) << 11) as i16))
}

/// `s_sendmsg_rtn` message id (SSRC0 field).
pub fn sendmsg_rtn(id: u8) -> Operand {
    Operand::SendMsg(Msg { id, op: 0 })
}

/// Modifiers of a `s_wait_*cnt` on one counter (the replay reads them).
pub fn wait_mods(counter: peacemaker_ir::wait::Counter, count: u8) -> Modifiers {
    let mut wait = peacemaker_ir::wait::WaitImm::default();
    wait.per_counter[counter as usize] = Some(count);
    Modifiers {
        wait: Some(wait),
        ..Modifiers::default()
    }
}

/// gfx12 `HW_REG_*` ids used by the M3 clients.
pub mod hw {
    pub const HW_ID1: u16 = 23;
    pub const HW_ID2: u16 = 24;
    pub const SHADER_CYCLES_LO: u16 = 29;
    pub const SHADER_CYCLES_HI: u16 = 30;
}

/// `MSG_RTN_GET_REALTIME`.
pub const MSG_RTN_GET_REALTIME: u8 = 0x83;

/// `s_wait_alu` depctr encodings (unmentioned counters at their no-wait maximum).
pub mod depctr {
    pub const SA_SDST_0: u16 = 0xff9e;
    pub const SA_SDST_0_VM_VSRC_0: u16 = 0xff82;
    pub const VA_SDST_0: u16 = 0xf19f;
}
