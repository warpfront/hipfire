// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Double-precision, 64-bit integer and carry-out vector opcodes (wave32 and wave64).
//!
//! Pseudocode is the RDNA3/RDNA4 ISA text. f64 arithmetic is host IEEE binary64 (RNE; the launch mode checked by
//! `State::new` keeps f64/f16 denormals), bit-exact for every non-NaN operand and result. No NaN behaviour of the
//! double-precision opcodes is measured on any architecture, so a NaN operand or a NaN result (`inf - inf`) is a hard
//! error. `v_cvt_i32_f64` documents its NaN result (0) and saturation, which Rust's cast implements exactly.
//! Carry/borrow masks are one dword (wave32) or an SGPR pair (wave64) whose inactive lanes are zero (ISA: "Inactive lanes write zero into this
//! carry-out"). DPP, SDWA, OMOD and OP_SEL have no meaning here and are errors.
use super::{Result, State, Wide};
use crate::{
    operand::{Omod, Operand, Special},
    Arch, Inst,
};

fn stem(name: &str) -> &str {
    let n = name
        .strip_suffix("_e32")
        .or_else(|| name.strip_suffix("_e64"))
        .unwrap_or(name);
    // The gfx12 spelling of the carry-out 64-bit MAD (`_co_`) is the gfx11 opcode's twin.
    match n {
        "v_mad_co_u64_u32" => "v_mad_u64_u32",
        "v_mad_co_i64_i32" => "v_mad_i64_i32",
        n => n,
    }
}

pub(super) fn handles(name: &str) -> bool {
    matches!(
        stem(name),
        "v_cvt_f64_i32"
            | "v_cvt_f64_u32"
            | "v_cvt_f64_f32"
            | "v_cvt_i32_f64"
            | "v_cvt_f32_f64"
            | "v_add_f64"
            | "v_min_f64"
            | "v_lshrrev_b64"
            | "v_ashrrev_i64"
            | "v_lshlrev_b64"
            | "v_subrev_co_u32"
            | "v_sub_co_ci_u32"
            | "v_sub_co_u32"
            | "v_subrev_co_ci_u32"
            | "v_mad_u64_u32"
            | "v_mad_i64_i32"
    )
}

fn clamp01(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

/// Source modifiers of a double: absolute value first, negate after.
fn modified(bits: u64, abs: bool, neg: bool) -> u64 {
    let bits = if abs { bits & !(1 << 63) } else { bits };
    if neg {
        bits ^ (1 << 63)
    } else {
        bits
    }
}

pub(super) fn execute(_arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let m = &i.mods;
    let o = &i.operands;
    let op = stem(name);
    if m.dpp.is_some() || m.sdwa.is_some() {
        return Err(format!("{name}: DPP/SDWA is not modeled"));
    }
    if m.op_sel != 0 || m.omod != Omod::None {
        return Err(format!("{name}: op_sel/omod have no modeled meaning"));
    }
    let float_src = matches!(
        op,
        "v_cvt_f64_f32" | "v_cvt_i32_f64" | "v_cvt_f32_f64" | "v_add_f64" | "v_min_f64"
    );
    if !float_src && m.neg | m.abs != 0 {
        return Err(format!("{name}: neg/abs on an integer source"));
    }
    if m.clamp
        && !matches!(
            op,
            "v_cvt_i32_f64" | "v_cvt_f32_f64" | "v_add_f64" | "v_min_f64"
        )
    {
        return Err(format!("{name}: clamp is not modeled"));
    }
    let src = |idx: usize| {
        o.get(idx)
            .ok_or_else(|| format!("{name}: missing operand {idx}"))
    };
    let bit = |n: u8, k: u8| n >> k & 1 != 0;
    let mut vals = [0u64; 64];
    let mut carry = 0u64;
    let explicit_carry = matches!(op, "v_sub_co_ci_u32" | "v_subrev_co_ci_u32") && o.len() >= 5
        || matches!(op, "v_subrev_co_u32" | "v_sub_co_u32");
    let sdst_null = matches!(o.get(1), Some(Operand::Special(Special::Null)));
    if op == "v_mad_i64_i32" && !sdst_null {
        return Err(format!(
            "{name}: the signed form's carry/overflow output is not specified; SDST must be null"
        ));
    }
    for lane in 0..s.lanes {
        if s.exec >> lane & 1 == 0 {
            continue;
        }
        let f64_src = |idx: usize, k: u8| -> Result<f64> {
            let x = f64::from_bits(modified(
                s.read_wide(src(idx)?, lane, Wide::Float)?,
                bit(m.abs, k),
                bit(m.neg, k),
            ));
            if x.is_nan() {
                return Err(format!(
                    "{name}: NaN double operand: propagation is not measured"
                ));
            }
            Ok(x)
        };
        vals[lane] = match op {
            "v_cvt_f64_i32" => f64::from(s.read(src(1)?, lane, 0)? as i32).to_bits(),
            "v_cvt_f64_u32" => f64::from(s.read(src(1)?, lane, 0)?).to_bits(),
            "v_cvt_f64_f32" => {
                let bits = (modified(
                    u64::from(s.read(src(1)?, lane, 0)?) << 32,
                    bit(m.abs, 0),
                    bit(m.neg, 0),
                ) >> 32) as u32;
                let x = f32::from_bits(bits);
                if x.is_nan() {
                    return Err(format!(
                        "{name}: NaN f32 operand: quieting/payload is not measured"
                    ));
                }
                f64::from(x).to_bits()
            }
            // Truncate toward zero, saturate (infinity included), NaN -> 0: Rust's `as` is exactly that.
            "v_cvt_i32_f64" => {
                let x = f64::from_bits(modified(
                    s.read_wide(src(1)?, lane, Wide::Float)?,
                    bit(m.abs, 0),
                    bit(m.neg, 0),
                ));
                u64::from(x as i32 as u32)
            }
            "v_cvt_f32_f64" => {
                let y = f64_src(1, 0)? as f32;
                let y = if m.clamp {
                    clamp01(f64::from(y)) as f32
                } else {
                    y
                };
                u64::from(y.to_bits())
            }
            "v_add_f64" => {
                let r = f64_src(1, 0)? + f64_src(2, 1)?;
                if r.is_nan() {
                    return Err(format!(
                        "{name}: NaN double result (inf - inf): default NaN is not measured"
                    ));
                }
                (if m.clamp { clamp01(r) } else { r }).to_bits()
            }
            "v_min_f64" => {
                let (a, b) = (f64_src(1, 0)?, f64_src(2, 1)?);
                let r = if a < b
                    || (a == 0.0 && b == 0.0 && a.is_sign_negative() && !b.is_sign_negative())
                {
                    a
                } else {
                    b
                };
                (if m.clamp { clamp01(r) } else { r }).to_bits()
            }
            // D0.u64 = S1.u64 >> S0[5:0]
            "v_lshrrev_b64" => {
                s.read_wide(src(2)?, lane, Wide::Unsigned)? >> (s.read(src(1)?, lane, 0)? & 63)
            }
            // D0.u64 = S1.u64 << S0[5:0]
            "v_lshlrev_b64" => {
                s.read_wide(src(2)?, lane, Wide::Unsigned)? << (s.read(src(1)?, lane, 0)? & 63)
            }
            // D0.i64 = (S1.i64 >> S0[5:0]), arithmetic (sign-filling) shift
            "v_ashrrev_i64" => {
                ((s.read_wide(src(2)?, lane, Wide::Signed)? as i64)
                    >> (s.read(src(1)?, lane, 0)? & 63)) as u64
            }
            // { carry, D0.u64 } = 65'(S0.u32) * 65'(S1.u32) + 65'(S2.u64); the carry bit goes to the SDST lane mask.
            "v_mad_u64_u32" => {
                let (a, b) = (s.read(src(2)?, lane, 0)?, s.read(src(3)?, lane, 0)?);
                let c = s.read_wide(src(4)?, lane, Wide::Unsigned)?;
                let r = u128::from(a) * u128::from(b) + u128::from(c);
                carry |= u64::from(r >> 64 != 0) << lane;
                r as u64
            }
            // D0.i64 = S0.i32 * S1.i32 + S2.i64 (wrapping). The ISA's carry/overflow output of the signed form is not
            // specified by the pseudocode, so a non-null SDST is refused below.
            "v_mad_i64_i32" => {
                let (a, b) = (
                    s.read(src(2)?, lane, 0)? as i32,
                    s.read(src(3)?, lane, 0)? as i32,
                );
                let c = s.read_wide(src(4)?, lane, Wide::Signed)? as i64;
                (i64::from(a) * i64::from(b)).wrapping_add(c) as u64
            }
            // tmp = S1 - S0; carry-out = S0 > S1 (unsigned borrow).
            "v_subrev_co_u32" => {
                let (a, b) = (s.read(src(2)?, lane, 0)?, s.read(src(3)?, lane, 0)?);
                carry |= u64::from(a > b) << lane;
                u64::from(b.wrapping_sub(a))
            }
            // tmp = S0 - S1; carry-out = S1 > S0 (unsigned borrow); VOP3, the carry-out lands in SDST.
            "v_sub_co_u32" => {
                let (a, b) = (s.read(src(2)?, lane, 0)?, s.read(src(3)?, lane, 0)?);
                carry |= u64::from(b > a) << lane;
                u64::from(a.wrapping_sub(b))
            }
            // tmp = S1 - S0 - carry_in; carry-out = S0 + carry_in > S1 (unsigned borrow). The RDNA3 pseudocode prints the
            // `v_sub_co_ci_u32` carry formula (S1 + carry_in > S0) under this opcode too; the borrow of S1 - S0 - carry_in is
            // S0 + carry_in > S1, which is what the RDNA4 pseudocode states.
            "v_subrev_co_ci_u32" => {
                let (a_idx, b_idx) = if o.len() >= 5 { (2, 3) } else { (1, 2) };
                let (a, b) = (s.read(src(a_idx)?, lane, 0)?, s.read(src(b_idx)?, lane, 0)?);
                let cin = if o.len() >= 5 {
                    s.read_mask(src(4)?)?
                } else {
                    s.vcc
                } >> lane
                    & 1;
                carry |= u64::from(u64::from(a) + cin > u64::from(b)) << lane;
                u64::from(b.wrapping_sub(a).wrapping_sub(cin as u32))
            }
            // tmp = S0 - S1 - carry_in; carry-out = S1 + carry_in > S0 (unsigned).
            "v_sub_co_ci_u32" => {
                let (a_idx, b_idx) = if o.len() >= 5 { (2, 3) } else { (1, 2) };
                let (a, b) = (s.read(src(a_idx)?, lane, 0)?, s.read(src(b_idx)?, lane, 0)?);
                let cin = if o.len() >= 5 {
                    s.read_mask(src(4)?)?
                } else {
                    s.vcc
                } >> lane
                    & 1;
                carry |= u64::from(u64::from(b) + cin > u64::from(a)) << lane;
                u64::from(a.wrapping_sub(b).wrapping_sub(cin as u32))
            }
            _ => return Err(format!("unsupported wide opcode {name}")),
        };
    }
    for lane in 0..s.lanes {
        if s.exec >> lane & 1 == 0 {
            continue;
        }
        if matches!(
            op,
            "v_cvt_f64_i32"
                | "v_cvt_f64_u32"
                | "v_cvt_f64_f32"
                | "v_add_f64"
                | "v_min_f64"
                | "v_lshrrev_b64"
                | "v_ashrrev_i64"
                | "v_lshlrev_b64"
                | "v_mad_u64_u32"
                | "v_mad_i64_i32"
        ) {
            s.put_wide(&o[0], lane, vals[lane])?;
        } else {
            s.put(&o[0], lane, 0, vals[lane] as u32)?;
        }
    }
    match op {
        "v_subrev_co_u32" | "v_sub_co_u32" | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32"
            if explicit_carry =>
        {
            s.put_mask(src(1)?, carry)?
        }
        "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => s.vcc = carry,
        "v_mad_u64_u32" if !sdst_null => s.put_mask(src(1)?, carry)?,
        _ => {}
    }
    Ok(())
}
