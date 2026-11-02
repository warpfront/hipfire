// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Scalar ALU semantics: SOP1, SOP2, SOPK, SOPC and SOPP other than
//! `s_waitcnt` / `s_barrier` / `s_endpgm` (handled in `sim/mod.rs`)
//! (CDNA3 ISA §5, §12.1-12.5).
//!
//! Every semantics block cites the pseudocode of CDNA3 ISA §12.1 (SOP2,
//! p.96-108), §12.2 (SOPK, p.109-114), §12.3 (SOP1, p.106-124), §12.4 (SOPC,
//! p.124-129) and §12.5 (SOPP, p.129-137). Operands are read through
//! [`Wave::read_s`] / [`Wave::read_s64`] and written through
//! [`Wave::write_s`] / [`Wave::write_s64`], which enforce the in-flight
//! register checks; 32-bit and 64-bit scalar operands are typed `B32` / `B64`
//! (inline integers sign-extend, literals are the raw dword).
//!
//! Not modelled (each returns [`SimError::Unsupported`](super::SimError) naming
//! the instruction): `s_getpc_b64` / `s_setpc_b64` (the wave carries an
//! instruction index, not a byte PC), `s_sendmsg`, `s_trap`, `s_getreg_b32` /
//! `s_setreg_b32` on any hardware register but MODE, or on MODE bits other
//! than FP_ROUND, FP_DENORM, DX10_CLAMP, IEEE and FP16_OVFL.
use super::{fault, unsupported, Flow, Mode, SimResult, Ty, Wave};
use crate::enc::sop::{Sop1Op, Sop2Op, SopcOp, SopkOp, SoppOp};
use crate::inst::Inst;
use crate::operand::Src;

/// Execute one scalar instruction. Branches return `Flow::Branch(offset)`
/// when taken, `Flow::Next` otherwise. Every opcode of the SOP tables is
/// implemented or returns `SimError::Unsupported` naming it.
pub fn exec(w: &mut Wave, inst: &Inst) -> SimResult<Flow> {
    match inst {
        Inst::Sop2(i) => sop2(w, i.op, i.sdst, i.ssrc0, i.ssrc1)?,
        Inst::Sop1(i) => sop1(w, i.op, i.sdst, i.ssrc0)?,
        Inst::Sopk(i) => sopk(w, i.op, i.sdst, i.simm16)?,
        Inst::Sopc(i) => sopc(w, i.op, i.ssrc0, i.ssrc1)?,
        Inst::Sopp(i) => return sopp(w, i.op, i.simm16),
        other => return unsupported(format!("{other:?} is not a scalar ALU instruction")),
    }
    Ok(Flow::Next)
}

/// 32-bit scalar source.
fn r32(w: &Wave, s: Src) -> SimResult<u32> {
    w.read_s(s, Ty::B32)
}
/// 64-bit scalar source.
fn r64(w: &Wave, s: Src) -> SimResult<u64> {
    w.read_s64(s, Ty::B64)
}

/// `(1 << width) - 1` with `width` already clamped to 0..=32.
fn mask32(width: u32) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}
/// `(1 << width) - 1` with `width` already clamped to 0..=64.
fn mask64(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `signext_from_bit(x, width)`: sign-extend `x` from bit `width - 1`; width 0 gives 0.
fn sext_from_bit32(x: u32, width: u32) -> u32 {
    if width == 0 {
        0
    } else if width >= 32 {
        x
    } else {
        (((x << (32 - width)) as i32) >> (32 - width)) as u32
    }
}

fn sop2(w: &mut Wave, op: Sop2Op, d: Src, s0: Src, s1: Src) -> SimResult<()> {
    use Sop2Op::*;
    // Result helpers: `v32(.., scc)` writes D and, when given, SCC.
    match op {
        // CDNA3 ISA §12.1 S_ADD_U32 p.97, S_ADDC_U32 p.98: SCC = carry out.
        AddU32 | AddcU32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let cin = if op == AddcU32 { u64::from(w.scc) } else { 0 };
            let t = u64::from(a) + u64::from(b) + cin;
            w.write_s(d, t as u32)?;
            w.scc = t >> 32 != 0;
        }
        // S_SUB_U32 p.97, S_SUBB_U32 p.98: SCC = borrow out.
        SubU32 | SubbU32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let bin = if op == SubbU32 { u32::from(w.scc) } else { 0 };
            let t = a.wrapping_sub(b).wrapping_sub(bin);
            w.write_s(d, t)?;
            w.scc = u64::from(b) + u64::from(bin) > u64::from(a);
        }
        // S_ADD_I32 p.97: SCC = signed overflow.
        AddI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let t = a.wrapping_add(b);
            w.write_s(d, t)?;
            w.scc = (a >> 31 == b >> 31) && (a >> 31 != t >> 31);
        }
        // S_SUB_I32 p.97.
        SubI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let t = a.wrapping_sub(b);
            w.write_s(d, t)?;
            w.scc = (a >> 31 != b >> 31) && (a >> 31 != t >> 31);
        }
        // S_MIN_I32 .. S_MAX_U32 p.98-99: SCC = S0 was selected.
        MinI32 | MaxI32 | MinU32 | MaxU32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let signed = matches!(op, MinI32 | MaxI32);
            let (ord_lt, ord_ge) = if signed {
                ((a as i32) < (b as i32), (a as i32) >= (b as i32))
            } else {
                (a < b, a >= b)
            };
            let pick0 = if matches!(op, MinI32 | MinU32) {
                ord_lt
            } else {
                ord_ge
            };
            w.scc = pick0;
            w.write_s(d, if pick0 { a } else { b })?;
        }
        // S_CSELECT_B32/B64 p.99: D = SCC ? S0 : S1; SCC unchanged.
        CselectB32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            w.write_s(d, if w.scc { a } else { b })?;
        }
        CselectB64 => {
            let (a, b) = (r64(w, s0)?, r64(w, s1)?);
            w.write_s64(d, if w.scc { a } else { b })?;
        }
        // S_AND/OR/XOR/ANDN2/ORN2_B32 p.99-101: SCC = result != 0.
        AndB32 | OrB32 | XorB32 | Andn2B32 | Orn2B32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let r = match op {
                AndB32 => a & b,
                OrB32 => a | b,
                XorB32 => a ^ b,
                Andn2B32 => a & !b,
                _ => a | !b,
            };
            w.write_s(d, r)?;
            w.scc = r != 0;
        }
        AndB64 | OrB64 | XorB64 | Andn2B64 | Orn2B64 => {
            let (a, b) = (r64(w, s0)?, r64(w, s1)?);
            let r = match op {
                AndB64 => a & b,
                OrB64 => a | b,
                XorB64 => a ^ b,
                Andn2B64 => a & !b,
                _ => a | !b,
            };
            w.write_s64(d, r)?;
            w.scc = r != 0;
        }
        // S_LSHL/LSHR/ASHR_B32/I32 p.103-104: the count is S1[4:0]; SCC = result != 0.
        LshlB32 | LshrB32 | AshrI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)? & 31);
            let r = match op {
                LshlB32 => a << b,
                LshrB32 => a >> b,
                _ => ((a as i32) >> b) as u32,
            };
            w.write_s(d, r)?;
            w.scc = r != 0;
        }
        // 64-bit forms: the count is S1[5:0] (S1 is a 32-bit operand).
        LshlB64 | LshrB64 | AshrI64 => {
            let (a, b) = (r64(w, s0)?, r32(w, s1)? & 63);
            let r = match op {
                LshlB64 => a << b,
                LshrB64 => a >> b,
                _ => ((a as i64) >> b) as u64,
            };
            w.write_s64(d, r)?;
            w.scc = r != 0;
        }
        // S_BFM_B32 p.104: ((1 << S0[4:0]) - 1) << S1[4:0]; SCC unchanged.
        BfmB32 => {
            let (a, b) = (r32(w, s0)? & 31, r32(w, s1)? & 31);
            w.write_s(d, ((1u32 << a) - 1) << b)?;
        }
        // S_MUL_I32 p.104: low 32 bits.
        MulI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            w.write_s(d, a.wrapping_mul(b))?;
        }
        // S_BFE_U32 p.105: offset S1[4:0], width S1[22:16] (7 bits). A width of 32
        // or more keeps every bit (the pseudocode's `1 << width` is undefined
        // there). [UNVERIFIED: confirm on MI300X: width > 32 on S_BFE_U32/I32.]
        BfeU32 => {
            let (a, c) = (r32(w, s0)?, r32(w, s1)?);
            let r = (a >> (c & 31)) & mask32((c >> 16) & 0x7f);
            w.write_s(d, r)?;
            w.scc = r != 0;
        }
        // S_BFE_I32 p.105: the field is extracted from the arithmetic shift of S0
        // (so bits past bit 31 are the sign), masked, then sign-extended from
        // its top bit; width 0 gives 0.
        BfeI32 => {
            let (a, c) = (r32(w, s0)?, r32(w, s1)?);
            let width = ((c >> 16) & 0x7f).min(32);
            let tmp = (((a as i32) >> (c & 31)) as u32) & mask32(width);
            let r = sext_from_bit32(tmp, width);
            w.write_s(d, r)?;
            w.scc = r != 0;
        }
        // S_ABSDIFF_I32 p.106: |S0 - S1| with wraparound (0x80000000 stays negative).
        AbsdiffI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let mut r = a.wrapping_sub(b) as i32;
            if r < 0 {
                r = r.wrapping_neg()
            }
            w.write_s(d, r as u32)?;
            w.scc = r != 0;
        }
        // S_MUL_HI_U32/I32 p.106-107.
        MulHiU32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            w.write_s(d, ((u64::from(a) * u64::from(b)) >> 32) as u32)?;
        }
        MulHiI32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            w.write_s(
                d,
                ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
            )?;
        }
        // S_LSHL{1..4}_ADD_U32 p.107-108: SCC = unsigned overflow of the sum.
        Lshl1AddU32 | Lshl2AddU32 | Lshl3AddU32 | Lshl4AddU32 => {
            let sh = match op {
                Lshl1AddU32 => 1,
                Lshl2AddU32 => 2,
                Lshl3AddU32 => 3,
                _ => 4,
            };
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let t = (u64::from(a) << sh) + u64::from(b);
            w.write_s(d, t as u32)?;
            w.scc = t >> 32 != 0;
        }
        // S_PACK_{LL,LH,HH}_B32_B16 p.108: D = { S1.half, S0.half }; SCC unchanged.
        PackLlB32B16 | PackLhB32B16 | PackHhB32B16 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            let lo = if op == PackHhB32B16 {
                a >> 16
            } else {
                a & 0xffff
            };
            let hi = if op == PackLlB32B16 {
                b & 0xffff
            } else {
                b >> 16
            };
            w.write_s(d, (hi << 16) | lo)?;
        }
    }
    Ok(())
}

fn sop1(w: &mut Wave, op: Sop1Op, d: Option<Src>, s0: Option<Src>) -> SimResult<()> {
    use Sop1Op::*;
    let name = format!("{op:?}");
    let src =
        || s0.ok_or_else(|| super::SimError::Fault(format!("{name}: missing source operand")));
    let dst =
        || d.ok_or_else(|| super::SimError::Fault(format!("{name}: missing destination operand")));
    match op {
        // S_MOV_B32/B64 p.107 ff.
        MovB32 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, a)?;
        }
        MovB64 => {
            let a = r64(w, src()?)?;
            w.write_s64(dst()?, a)?;
        }
        // S_CMOV_B32/B64: only when SCC is set (the source is still read).
        CmovB32 => {
            let a = r32(w, src()?)?;
            let d = dst()?;
            if w.scc {
                w.write_s(d, a)?
            }
        }
        CmovB64 => {
            let a = r64(w, src()?)?;
            let d = dst()?;
            if w.scc {
                w.write_s64(d, a)?
            }
        }
        // S_NOT_B32/B64: SCC = result != 0.
        NotB32 => {
            let r = !r32(w, src()?)?;
            w.write_s(dst()?, r)?;
            w.scc = r != 0;
        }
        NotB64 => {
            let r = !r64(w, src()?)?;
            w.write_s64(dst()?, r)?;
            w.scc = r != 0;
        }
        // S_BREV_B32: bit reversal; SCC unchanged.
        BrevB32 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, a.reverse_bits())?;
        }
        // S_BCNT1_I32_B32/B64: SCC = count != 0 (equivalently S0 != 0).
        Bcnt1I32B32 => {
            let n = r32(w, src()?)?.count_ones();
            w.write_s(dst()?, n)?;
            w.scc = n != 0;
        }
        Bcnt1I32B64 => {
            let n = r64(w, src()?)?.count_ones();
            w.write_s(dst()?, n)?;
            w.scc = n != 0;
        }
        // S_FF1_I32_B32/B64: index of the lowest set bit, -1 if none; SCC unchanged.
        Ff1I32B32 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, if a == 0 { u32::MAX } else { a.trailing_zeros() })?;
        }
        Ff1I32B64 => {
            let a = r64(w, src()?)?;
            w.write_s(dst()?, if a == 0 { u32::MAX } else { a.trailing_zeros() })?;
        }
        // S_FLBIT_I32_B32: leading zeros before the first one, -1 if none.
        FlbitI32B32 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, if a == 0 { u32::MAX } else { a.leading_zeros() })?;
        }
        // S_SEXT_I32_I8 / I16.
        SextI32I8 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, i32::from(a as u8 as i8) as u32)?;
        }
        SextI32I16 => {
            let a = r32(w, src()?)?;
            w.write_s(dst()?, i32::from(a as u16 as i16) as u32)?;
        }
        // S_{AND,OR,XOR,ANDN2}_SAVEEXEC_B64 p.119 ff.: D = EXEC (saved first),
        // EXEC = S0 <op> EXEC (ANDN2: S0 & ~EXEC, as the ISA text gives it),
        // SCC = new EXEC != 0. [UNVERIFIED: confirm on MI300X: the ANDN2 operand
        // that is negated.]
        AndSaveexecB64 | OrSaveexecB64 | XorSaveexecB64 | Andn2SaveexecB64 => {
            let a = r64(w, src()?)?;
            let d = dst()?;
            let saved = r64(w, Src::ExecLo)?;
            let new = match op {
                AndSaveexecB64 => a & saved,
                OrSaveexecB64 => a | saved,
                XorSaveexecB64 => a ^ saved,
                _ => a & !saved,
            };
            w.write_s64(d, saved)?;
            w.write_s64(Src::ExecLo, new)?;
            w.scc = new != 0;
        }
        // S_ABS_I32: |S0| with wraparound (0x80000000 stays negative); SCC = result != 0.
        AbsI32 => {
            let a = r32(w, src()?)? as i32;
            let r = if a < 0 { a.wrapping_neg() } else { a };
            w.write_s(dst()?, r as u32)?;
            w.scc = r != 0;
        }
        // S_BITSET0/1_B32: D[S0[4:0]] = 0/1; SCC unchanged.
        Bitset0B32 | Bitset1B32 => {
            let bit = r32(w, src()?)? & 31;
            let d = dst()?;
            let cur = r32(w, d)?;
            w.write_s(
                d,
                if op == Bitset0B32 {
                    cur & !(1 << bit)
                } else {
                    cur | (1 << bit)
                },
            )?;
        }
        // The wave carries an instruction index, not a byte PC: no real byte
        // address exists to return or to jump to.
        GetpcB64 | SetpcB64 => {
            return unsupported(format!("{name}: the wave has no byte-addressed PC"))
        }
    }
    Ok(())
}

/// MODE register bits the emulator models (CDNA3 ISA §3.5 Table 4 p.9-10):
/// FP_ROUND [3:0], FP_DENORM [7:4], DX10_CLAMP [8], IEEE [9], FP16_OVFL [23].
const MODE_MODELLED: u32 = 0x0000_03ff | (1 << 23);

fn mode_bits(m: &Mode) -> u32 {
    u32::from(m.round32 & 3)
        | u32::from(m.round16_64 & 3) << 2
        | u32::from(m.denorm32 & 3) << 4
        | u32::from(m.denorm16_64 & 3) << 6
        | u32::from(m.dx10_clamp) << 8
        | u32::from(m.ieee) << 9
        | u32::from(m.fp16_ovfl) << 23
}

fn set_mode_bits(m: &mut Mode, v: u32) {
    m.round32 = (v & 3) as u8;
    m.round16_64 = (v >> 2 & 3) as u8;
    m.denorm32 = (v >> 4 & 3) as u8;
    m.denorm16_64 = (v >> 6 & 3) as u8;
    m.dx10_clamp = v >> 8 & 1 != 0;
    m.ieee = v >> 9 & 1 != 0;
    m.fp16_ovfl = v >> 23 & 1 != 0;
}

/// Decode a `hwreg(id, offset, size)` SIMM16 (CDNA3 ISA §12.2 S_GETREG_B32
/// p.112): `(id, offset, size, field mask in place)`.
fn hwreg(simm16: u16, name: &str) -> SimResult<(u32, u32, u32)> {
    let id = u32::from(simm16 & 0x3f);
    let off = u32::from(simm16 >> 6 & 0x1f);
    let size = u32::from(simm16 >> 11) + 1;
    if id != 1 {
        return unsupported(format!(
            "{name}: hardware register id {id} (only MODE, id 1, is modelled)"
        ));
    }
    let mask = mask32(size) << off;
    if mask & !MODE_MODELLED != 0 {
        return unsupported(format!("{name}: MODE bits {:#x} are outside the modelled FP_ROUND/FP_DENORM/DX10_CLAMP/IEEE/FP16_OVFL fields", mask & !MODE_MODELLED));
    }
    Ok((id, off, size))
}

fn sopk(w: &mut Wave, op: SopkOp, d: Src, simm16: u16) -> SimResult<()> {
    use SopkOp::*;
    let sx = i32::from(simm16 as i16) as u32;
    let zx = u32::from(simm16);
    match op {
        // S_MOVK_I32 / S_CMOVK_I32 p.109: D = signext(simm16) (CMOVK iff SCC).
        MovkI32 => w.write_s(d, sx)?,
        CmovkI32 => {
            if w.scc {
                w.write_s(d, sx)?
            }
        }
        // S_CMPK_*_I32 p.109-110: signed compare with signext(simm16).
        CmpkEqI32 | CmpkLgI32 | CmpkGtI32 | CmpkGeI32 | CmpkLtI32 | CmpkLeI32 => {
            let a = r32(w, d)? as i32;
            let b = sx as i32;
            w.scc = match op {
                CmpkEqI32 => a == b,
                CmpkLgI32 => a != b,
                CmpkGtI32 => a > b,
                CmpkGeI32 => a >= b,
                CmpkLtI32 => a < b,
                _ => a <= b,
            };
        }
        // S_CMPK_*_U32 p.110-111: unsigned compare with zext(simm16).
        CmpkEqU32 | CmpkLgU32 | CmpkGtU32 | CmpkGeU32 | CmpkLtU32 | CmpkLeU32 => {
            let a = r32(w, d)?;
            w.scc = match op {
                CmpkEqU32 => a == zx,
                CmpkLgU32 => a != zx,
                CmpkGtU32 => a > zx,
                CmpkGeU32 => a >= zx,
                CmpkLtU32 => a < zx,
                _ => a <= zx,
            };
        }
        // S_ADDK_I32 p.111: D += signext(simm16), SCC = signed overflow.
        AddkI32 => {
            let a = r32(w, d)?;
            let r = a.wrapping_add(sx);
            w.write_s(d, r)?;
            w.scc = (a >> 31 == sx >> 31) && (a >> 31 != r >> 31);
        }
        // S_MULK_I32 p.111: low 32 bits of D * signext(simm16).
        MulkI32 => {
            let a = r32(w, d)?;
            w.write_s(d, a.wrapping_mul(sx))?;
        }
        // S_GETREG_B32 p.112: D = (MODE >> offset) & ((1 << size) - 1).
        GetregB32 => {
            let (_, off, size) = hwreg(simm16, "s_getreg_b32")?;
            let v = (mode_bits(&w.mode) >> off) & mask32(size);
            w.write_s(d, v)?;
        }
        // S_SETREG_B32 p.112-113: the field of MODE becomes the low bits of D;
        // the rest of MODE is kept. (Every modelled bit is writable: the
        // privileged-bit write mask does not apply to them.)
        SetregB32 => {
            let (_, off, size) = hwreg(simm16, "s_setreg_b32")?;
            let a = r32(w, d)?;
            let mask = mask32(size) << off;
            let old = mode_bits(&w.mode);
            set_mode_bits(&mut w.mode, ((a << off) & mask) | (old & !mask));
        }
    }
    Ok(())
}

fn sopc(w: &mut Wave, op: SopcOp, s0: Src, s1: Src) -> SimResult<()> {
    use SopcOp::*;
    w.scc = match op {
        // S_CMP_*_I32 / _U32 p.124-127.
        CmpEqI32 | CmpLgI32 | CmpGtI32 | CmpGeI32 | CmpLtI32 | CmpLeI32 => {
            let (a, b) = (r32(w, s0)? as i32, r32(w, s1)? as i32);
            match op {
                CmpEqI32 => a == b,
                CmpLgI32 => a != b,
                CmpGtI32 => a > b,
                CmpGeI32 => a >= b,
                CmpLtI32 => a < b,
                _ => a <= b,
            }
        }
        CmpEqU32 | CmpLgU32 | CmpGtU32 | CmpGeU32 | CmpLtU32 | CmpLeU32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)?);
            match op {
                CmpEqU32 => a == b,
                CmpLgU32 => a != b,
                CmpGtU32 => a > b,
                CmpGeU32 => a >= b,
                CmpLtU32 => a < b,
                _ => a <= b,
            }
        }
        // S_BITCMP0/1_B32 p.127: SCC = S0[S1[4:0]] == 0 / 1.
        Bitcmp0B32 | Bitcmp1B32 => {
            let (a, b) = (r32(w, s0)?, r32(w, s1)? & 31);
            let bit = a >> b & 1 != 0;
            if op == Bitcmp1B32 {
                bit
            } else {
                !bit
            }
        }
        // S_CMP_EQ_U64 / S_CMP_LG_U64 p.128.
        CmpEqU64 | CmpLgU64 => {
            let (a, b) = (r64(w, s0)?, r64(w, s1)?);
            if op == CmpEqU64 {
                a == b
            } else {
                a != b
            }
        }
    };
    Ok(())
}

fn sopp(w: &mut Wave, op: SoppOp, simm16: u16) -> SimResult<Flow> {
    use SoppOp::*;
    let taken = |c: bool| {
        if c {
            Flow::Branch(simm16 as i16)
        } else {
            Flow::Next
        }
    };
    Ok(match op {
        // S_NOP p.130, S_SLEEP, S_SETPRIO, S_ICACHE_INV: no architectural
        // effect in a functional model (wait states and hazards are not
        // modelled; code memory is immutable).
        Nop | Sleep | Setprio | IcacheInv => Flow::Next,
        // S_BRANCH / S_CBRANCH_* §12.5: the offset is a signed dword count
        // from the next instruction. VCCZ/EXECZ are VCC == 0 / EXEC == 0.
        Branch => taken(true),
        CbranchScc0 => taken(!w.scc),
        CbranchScc1 => taken(w.scc),
        CbranchVccz => taken(w.vcc == 0),
        CbranchVccnz => taken(w.vcc != 0),
        CbranchExecz => taken(w.exec == 0),
        CbranchExecnz => taken(w.exec != 0),
        Sendmsg | Trap => {
            return unsupported(format!(
                "{op:?}: messages and the trap handler are not modelled"
            ))
        }
        Waitcnt | Barrier | Endpgm => {
            return fault(format!("{op:?} is handled by exec_one, not the scalar ALU"))
        }
    })
}
