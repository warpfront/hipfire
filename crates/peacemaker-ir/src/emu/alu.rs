//! Exact CPU semantics for scalar and vector ALU opcodes (wave32).
//!
//! Every opcode is keyed by its exact ISA name; an unknown name is an error, never a
//! best guess. Float arithmetic uses host IEEE-754 binary32 (`+ - * fma`, round to
//! nearest even), which is bit-exact for every non-NaN operand and every non-NaN result. Whatever the
//! ISA documents only loosely is a hard error rather than a guess.
//!
//! Special values come from the exhaustive raw hardware sweeps, one per architecture: gfx1151
//! (`pm-r2/probe/capture-v2-gfx1151.manifest.json`) and gfx1201 (`pm-r2/probe/edge-capture-v2-gfx1201.manifest.json`),
//! both reconstructed losslessly with `edge_recon.py` (every table file fingerprint verified). Every rule below
//! agrees with the tables for all 4 VCC/SCC input states and leaves VCC/SCC untouched unless stated. The two
//! sweeps use the same operand sets and are bit-identical for gids 0..=28, 32, 33 and 40..=92 and for the VOPD pair
//! contexts and extra tuples, so those rules are shared through [`edge_qualified`]; gfx1201 differs from gfx1151 in
//! exactly the places named below, each of which keeps an architecture-specific branch:
//!
//! * NaN operands of `v_add_f32`, `v_sub_f32`, `v_mul_f32`, `v_fma_f32`, `v_fmac_f32`, `v_fmamk_f32`,
//!   `v_div_fmas_f32` (VCC clear) and the VOPD `add`/`mul`/`fmac`: the first NaN operand in source order
//!   (`S0`, `S1`, `S2`; fmamk: `S0`, `K`, `S1`) wins, keeps sign and payload and gets bit 22 set. `v_sub_f32`
//!   negates `S1` before the pick (a NaN `S1` comes out with its sign flipped). A fused multiply-add whose
//!   product is `inf * 0` returns the default NaN `0xffc00000` even when the addend is a NaN; the default NaN
//!   is also the result of every invalid operation on non-NaN operands (`inf - inf`, `inf * 0`).
//!   Source `neg` on a NaN operand is measured only for `v_fma_f32` `S0`/`S2`; any other `abs`/`neg` on a
//!   NaN operand is an error.
//! * `v_max_f32`/`v_min_f32`/`v_max3_f32`/`v_min3_f32` (`((S0,S1),S2)`) exist on gfx1151 only: the first signalling
//!   NaN is quieted and returned, otherwise two quiet NaNs give `S0`, one quiet NaN loses to the number. The gfx12
//!   `v_max_num_f32`/`v_min_num_f32`/`v_max3_num_f32`/`v_min3_num_f32` (gfx1201 gids 34..=38) follow [`num_select`]:
//!   a NaN loses to a number and two NaNs give a quieted `S0`.
//! * `v_rndne_f32` and `v_ldexp_f32` quiet a NaN input (sign and payload kept); `v_cvt_f16_f32` converts NaN
//!   to `sign | 0x7c00 | 0x200 | payload >> 13`; `v_cvt_f32_f16` quiets and widens it. `v_rndne_f32` of a negative
//!   input that rounds to zero gives `-0` (`0x80000000`; measured for `-0.25, -0.5, -minsub, -0` on both arches), so
//!   host `round_ties_even` is exact. The ISA pseudocode (`+0`) is wrong for both.
//! * `v_div_scale_f32`: the exponent-field chain of [`div_scale_chain`] on both arches (gids 29/30/31, all 262144
//!   tuples x 4 states), with exact scalar-destination, NULL and VCC behavior. The v2 per-lane readback of the SGPR
//!   predicate (gid 29 `D1`, values {0,1}) equals the VCC bit of gid 31 in every tuple-state of both arches, and the
//!   VCC/SCC flags are untouched for the SGPR and NULL destinations (it is not an independent whole-wave capture). gfx1201
//!   differs: every signalling NaN that would be returned as `D` (`S0` passed through or scaled; 30752 of 262144
//!   tuples) comes out quieted (bit 22 set).
//! * `v_div_fixup_f32`: NaN quotient gives `sign | inf`, an underflowing quotient `sign | 0`.
//! * `v_div_fmas_f32`: VCC clear is the plain fused multiply-add. A set VCC lane with a NaN/Inf operand gives the same
//!   exceptional result (all 164808 nonfinite tuples of gid 32 agree in VCC states 2/3). A set VCC lane with finite
//!   operands is qualified on gfx1201 only, by the finite fused-rounding discriminator capture
//!   (`pm-r2/probe/fmas-gfx1201.raw.tsv`: 96 cases x 4 VCC/SCC states x 32 lanes, with the subnormal tie, the
//!   pre-rounding overflow and high-`S2` cancellation cases) together with the full generic gid-32 corpus: the exact
//!   fused sum is scaled by 2**64 when the biased exponent of `S2` exceeds 127, otherwise by 2**-64, and rounded to f32
//!   **once** ([`fmas_f32`]). gfx1151 (Halo) has no such discriminators and keeps the hard error for finite operands,
//!   as does any set VCC lane on an unqualified arch.
//! * `v_fma_mix_f32`/`v_fma_mixlo_f16`/`v_fma_mixhi_f16` (gfx1151 gids 93..=101, gfx1201 gids 95..=103; the f32/f16
//!   source variants with immediate zero addend): NaN operands follow the fma rule above, a half source widens
//!   exactly (NaN payload kept) before it. A half result is rounded **once** from the exact fused result
//!   ([`fma_f16`]); rounding to f32 first double-rounds in about 0.2% of the tuples. `mixlo`/`mixhi` keep the other
//!   destination half. A NaN operand is qualified only in the swept layouts (`op_sel:[0,0,0]`, only `B` possibly a
//!   half); source `neg`/`abs` on a NaN operand and `clamp` of a NaN result are not in the sweep and stay errors.
//! * `v_cvt_pk_f32_fp8` (gfx12 only, gfx1201 gids 93/94, all 4096 packed pairs): both FP8 NaN encodings of either
//!   element give `0xffc00000` whatever the sign; every finite value is the OCP E4M3 value.
//!
//! NaN operands of `v_subrev_f32`, `v_dual_sub_f32`, `v_dual_subrev_f32`, `v_dual_fmamk_f32`, `v_fmaak_f32` and
//! `v_dual_fmaak_f32` are not in either sweep and stay errors; the invalid operations (NaN result from non-NaN
//! operands) of `v_subrev_f32`/`v_dual_sub_f32`/`v_dual_subrev_f32` give the default NaN on gfx1151 only (gfx1201
//! keeps the hard error). Every other arch (gfx1100 included) keeps the hard errors for NaN operands, NaN results,
//! the documented-but-unmeasured `v_div_scale_f32`/`v_div_fixup_f32` corner cases and:
//! * `v_rndne_f32` of a negative input rounding to zero (documented pseudocode gives `+0`, IEEE `-0`).
//! * OMOD results that would flush a non-zero denormal (sign of the flushed zero).
//! * DPP with FI=1 and an out-of-range source (the BC/FI table disagrees with its prose).
//!
//! Not qualified on any arch: `v_med3_num_f32` (not modeled), the `v_div_scale_f32` exponent domains the sweeps leave
//! open, BF16/FP8 encoders. Finite operands of `v_div_fmas_f32` under VCC are unqualified on every arch but gfx1201.
//!
//! Descriptor float mode must be round to nearest even, allow input/output denormals,
//! use FP16_OVFL=0, and on gfx1151 use IEEE_MODE=1/DX10_CLAMP=1; `State::new` checks it.
//! F8_MODE has no descriptor field: FP8 conversion assumes the normal launch default
//! F8_MODE=0. No opcode in the semantic table changes MODE.
use super::{convert, Result, State};
use crate::{
    operand::{Dpp, InlineConst, Omod, Operand},
    Arch, FormFields, Inst,
};

const NAN_IN: &str = "NaN operand: the ISA document does not specify NaN propagation (selection/payload/sign) for this opcode";
const NAN_OUT: &str = "NaN result: the ISA document does not specify the default NaN encoding for this opcode";
const UNSUPPORTED: &str = "unsupported vector opcode";
const INF: u32 = 0x7f80_0000;

/// Where the first source operand of a lane comes from (DPP routing).
#[derive(Clone, Copy)]
enum Src0 {
    Plain,
    Lane(usize),
    Zero,
}

/// How a computed dword lands in the destination register.
#[derive(Clone, Copy)]
enum Val {
    Dword(u32),
    Low16(u32),
    High16(u32),
}

/// Modifier policy of an opcode's result.
#[derive(Clone, Copy)]
enum Class {
    /// Integer/bitwise: no source modifiers, clamp or omod.
    Int,
    /// Bit select: source neg/abs only.
    Select,
    /// Integer result from a float source: source neg/abs and (inert) clamp.
    FloatToInt,
    /// f32 result: source modifiers, omod, clamp.
    F32,
    /// Handled by the opcode itself.
    Own,
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > INF
}

fn num(bits: u32) -> Result<f32> {
    if is_nan(bits) {
        Err(NAN_IN.into())
    } else {
        Ok(f32::from_bits(bits))
    }
}

fn fin(x: f32) -> Result<u32> {
    if x.is_nan() {
        Err(NAN_OUT.into())
    } else {
        Ok(x.to_bits())
    }
}

/// Default NaN of an invalid operation (`inf - inf`, `inf * 0`, inf-minus-inf in a fused multiply-add), measured on
/// gfx1151 and gfx1201 for add/sub/mul/fma/fmac/fmamk/div_fmas.
const DEFAULT_NAN: u32 = 0xffc0_0000;

/// Result of `v_subrev_f32`/`v_dual_sub_f32`/`v_dual_subrev_f32`, whose NaN operands are not measured
/// (`num` errors on them): a NaN result is then an invalid operation. These opcodes are in neither sweep; only
/// gfx1151 takes the default NaN (as its measured `v_sub_f32` sibling), every other arch keeps the hard error.
fn fin_invalid(arch: Arch, x: f32) -> Result<u32> {
    if x.is_nan() && arch == Arch::Gfx1151 {
        Ok(DEFAULT_NAN)
    } else {
        fin(x)
    }
}

const QUIET: u32 = 0x0040_0000;
const SIGN: u32 = 0x8000_0000;
const NAN_MOD: &str = "abs/neg source modifier on a NaN operand: not measured for this opcode operand";

/// Architectures whose raw v2 edge tables prove the NaN/default-NaN/signed-zero/divide rules shared by the helpers
/// below (see the module comment). Anything else keeps the documented hard errors.
fn edge_qualified(arch: Arch) -> bool {
    matches!(arch, Arch::Gfx1151 | Arch::Gfx1201)
}

fn is_snan(bits: u32) -> bool {
    is_nan(bits) && bits & QUIET == 0
}

fn is_zero(bits: u32) -> bool {
    bits & 0x7fff_ffff == 0
}

fn is_inf(bits: u32) -> bool {
    bits & 0x7fff_ffff == INF
}

/// A non-NaN host result is exact; a NaN one is an invalid operation of non-NaN operands.
fn or_default_nan(x: f32) -> u32 {
    if x.is_nan() {
        DEFAULT_NAN
    } else {
        x.to_bits()
    }
}

/// The first NaN operand in source order, quieted (bit 22 set, sign and payload kept).
fn first_nan(ops: &[u32]) -> Option<u32> {
    ops.iter().find(|&&x| is_nan(x)).map(|&x| x | QUIET)
}

/// `a + b` of post-modifier operand words (qualified arches: measured NaN rules; others: NaN is an error).
fn add_bits(arch: Arch, a: u32, b: u32) -> Result<u32> {
    if !edge_qualified(arch) {
        return fin(num(a)? + num(b)?);
    }
    Ok(first_nan(&[a, b]).unwrap_or_else(|| or_default_nan(f32::from_bits(a) + f32::from_bits(b))))
}

fn mul_bits(arch: Arch, a: u32, b: u32) -> Result<u32> {
    if !edge_qualified(arch) {
        return fin(num(a)? * num(b)?);
    }
    Ok(first_nan(&[a, b]).unwrap_or_else(|| or_default_nan(f32::from_bits(a) * f32::from_bits(b))))
}

/// `fma(a, b, c)`: an invalid product (`inf * 0`) beats a NaN addend, which in turn comes after the factors.
fn fma_bits(arch: Arch, a: u32, b: u32, c: u32) -> Result<u32> {
    if !edge_qualified(arch) {
        return fin(num(a)?.mul_add(num(b)?, num(c)?));
    }
    if let Some(n) = first_nan(&[a, b]) {
        return Ok(n);
    }
    if is_inf(a) && is_zero(b) || is_zero(a) && is_inf(b) {
        return Ok(DEFAULT_NAN);
    }
    if is_nan(c) {
        return Ok(c | QUIET);
    }
    Ok(or_default_nan(f32::from_bits(a).mul_add(f32::from_bits(b), f32::from_bits(c))))
}

/// Source modifiers: absolute value first, negate after.
fn sgn(v: u32, abs: bool, neg: bool) -> u32 {
    let v = if abs { v & 0x7fff_ffff } else { v };
    if neg {
        v ^ 0x8000_0000
    } else {
        v
    }
}

fn active(s: &State) -> impl Iterator<Item = usize> {
    let exec = s.exec;
    (0..32).filter(move |lane| exec & (1 << lane) != 0)
}

pub(super) fn execute(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    if let FormFields::Vopd { y_op, x_operands } = i.fields {
        let y = y_op.name(arch).ok_or("unknown VOPD Y opcode")?;
        return vopd(arch, name, y, usize::from(x_operands), i, s);
    }
    if name.starts_with("s_") {
        return scalar(name, i, s);
    }
    match name {
        "v_readfirstlane_b32" => readfirstlane(i, s),
        _ if name.starts_with("v_wmma") => super::wmma::execute(arch,name,i,s),
        "v_permlane16_b32" | "v_permlanex16_b32" => permlane(name, i, s),
        "v_div_scale_f32" => div_scale(arch, i, s),
        "v_cvt_pk_f32_fp8_e32" | "v_cvt_pk_f32_fp8_e64" => cvt_pk_f32_fp8(arch, i, s),
        _ if name.starts_with("v_cmp") => compare(name, i, s),
        _ => lanes(arch, name, i, s),
    }
}

fn store(s: &mut State, dst: &Operand, lane: usize, v: Val) -> Result<()> {
    match v {
        Val::Dword(x) => s.put(dst, lane, 0, x),
        Val::Low16(x) => {
            let old = s.read(dst, lane, 0)?;
            s.put(dst, lane, 0, (old & 0xffff_0000) | (x & 0xffff))
        }
        Val::High16(x) => {
            let old = s.read(dst, lane, 0)?;
            s.put(dst, lane, 0, (old & 0xffff) | (x << 16))
        }
    }
}

fn dword(v: Val) -> Result<u32> {
    match v {
        Val::Dword(x) => Ok(x),
        _ => Err("VOPD half-dword destination unsupported".into()),
    }
}

/// VOPD: both halves read the pre-instruction state; both writes land afterwards.
fn vopd(arch: Arch, x: &str, y: &str, split: usize, i: &Inst, s: &mut State) -> Result<()> {
    let (ox, oy) = i.operands.split_at(split);
    if s.exec == 0 {
        // Still reject unknown opcodes when no lane is active.
        for (name, ops) in [(x, ox), (y, oy)] {
            if let Err(e) = vector(arch, name, ops, i, s, 0, Src0::Plain) {
                if e.starts_with(UNSUPPORTED) {
                    return Err(e);
                }
            }
        }
        return Ok(());
    }
    let mut out = [[0u32; 2]; 32];
    for lane in active(s) {
        out[lane] = [
            dword(vector(arch, x, ox, i, s, lane, Src0::Plain)?)?,
            dword(vector(arch, y, oy, i, s, lane, Src0::Plain)?)?,
        ];
    }
    for lane in active(s) {
        s.put(&ox[0], lane, 0, out[lane][0])?;
        s.put(&oy[0], lane, 0, out[lane][1])?;
    }
    Ok(())
}

fn readfirstlane(i: &Inst, s: &mut State) -> Result<()> {
    // EXEC == 0 forces lane 0.
    let lane = if s.exec == 0 { 0 } else { s.exec.trailing_zeros() as usize };
    let v = s.read(&i.operands[1], lane, 0)?;
    s.put(&i.operands[0], 0, 0, v)
}

fn dpp_fi(i: &Inst) -> bool {
    match &i.fields {
        FormFields::Bits { honored, ignored } => honored.iter().chain(ignored.iter()).any(|f| f.name == "dpp_fi" && f.value != 0),
        _ => false,
    }
}

/// Source lane for a DPP16 control word (wave32 layout); `None` is out of range.
fn dpp_lane(ctrl: u16, lane: usize) -> Result<Option<usize>> {
    let row = lane & !15;
    let col = lane & 15;
    let n = usize::from(ctrl & 15);
    Ok(match ctrl {
        0x000..=0x0ff => Some((lane & !3) | usize::from((ctrl >> (2 * (lane & 3))) & 3)),
        0x101..=0x10f => (col + n < 16).then_some(lane + n),
        0x111..=0x11f => col.checked_sub(n).map(|c| row + c),
        0x121..=0x12f => Some(row + ((col + 16 - n) & 15)),
        0x140 => Some(row + 15 - col),
        0x141 => Some((lane & !7) + 7 - (lane & 7)),
        0x150..=0x15f => Some(row + n),
        0x160..=0x16f => Some(row + (col ^ n)),
        _ => return Err(format!("DPP control {ctrl:#x} is not a valid RDNA DPP16 control")),
    })
}

/// DPP routing for one lane: `None` disables the destination write.
///
/// BC/FI table (FI=0): out-of-range or disabled source lane -> write disabled when
/// BC=0, source reads zero when BC=1. FI=1 reads a disabled in-range lane normally.
fn dpp_plan(d: &Dpp, fi: bool, s: &State, lane: usize) -> Result<Option<Src0>> {
    if d.row_mask & (1 << (lane / 16)) == 0 || d.bank_mask & (1 << ((lane / 4) & 3)) == 0 {
        return Ok(None);
    }
    let invalid = if d.bound_ctrl { Some(Src0::Zero) } else { None };
    Ok(match dpp_lane(d.ctrl, lane)? {
        None if fi => return Err("DPP FI=1 with an out-of-range source lane: the BC/FI table and its prose disagree".into()),
        None => invalid,
        Some(src) if fi || s.exec & (1 << src) != 0 => Some(Src0::Lane(src)),
        Some(_) => invalid,
    })
}

fn read_src(s: &State, op: &Operand, lane: usize, d: Src0) -> Result<u32> {
    match d {
        Src0::Plain => s.read(op, lane, 0),
        Src0::Lane(src) => s.read(op, src, 0),
        Src0::Zero => Ok(0),
    }
}

fn stem(name: &str) -> &str {
    name.strip_suffix("_e32")
        .or_else(|| name.strip_suffix("_e64"))
        .or_else(|| name.strip_suffix("_dpp"))
        .unwrap_or(name)
}

fn lanes(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let dpp = i.mods.dpp;
    if dpp.is_some() != name.ends_with("_dpp") {
        return Err(format!("{name}: DPP control and opcode form disagree"));
    }
    if s.exec == 0 {
        if let Err(e) = vector(arch, name, o, i, s, 0, Src0::Plain) {
            if e.starts_with(UNSUPPORTED) {
                return Err(e);
            }
        }
        return Ok(());
    }
    let fi = dpp_fi(i);
    let mut vals: [Option<Val>; 32] = [None; 32];
    for lane in active(s) {
        let d = match dpp {
            Some(d) => match dpp_plan(&d, fi, s, lane)? {
                Some(d) => d,
                None => continue,
            },
            None => Src0::Plain,
        };
        vals[lane] = Some(vector(arch, name, o, i, s, lane, d)?);
    }
    for (lane, v) in vals.into_iter().enumerate() {
        if let Some(v) = v {
            store(s, &o[0], lane, v)?;
        }
    }
    Ok(())
}

fn plain(i: &Inst) -> Result<()> {
    let m = &i.mods;
    if m.neg | m.abs != 0 || m.omod != Omod::None || m.clamp {
        Err("neg/abs/omod/clamp on an integer-result opcode is not modeled".into())
    } else {
        Ok(())
    }
}

/// Output modifiers for an f32 result: OMOD, then CLAMP (NaN clamps to +0, -0 to +0).
fn out_f32(arch: Arch, bits: u32, i: &Inst) -> Result<u32> {
    let m = &i.mods;
    let mut x = f32::from_bits(bits);
    if m.omod != Omod::None {
        match arch {
            // RDNA3: OMOD is ignored when output denormals are enabled.
            Arch::Gfx1100 | Arch::Gfx1151 => {}
            Arch::Gfx1201 => {
                if x.is_nan() {
                    return Err("OMOD on a NaN result: payload not specified".into());
                }
                let y = match m.omod {
                    Omod::Mul2 => x * 2.0,
                    Omod::Mul4 => x * 4.0,
                    Omod::Div2 => x * 0.5,
                    Omod::None => x,
                };
                if y != 0.0 && y.abs() < f32::MIN_POSITIVE {
                    return Err("OMOD flushes a non-zero denormal: sign of the flushed zero is not specified".into());
                }
                // -0 * OMOD = +0.
                x = if y == 0.0 { 0.0 } else { y };
            }
            _ => return Err("OMOD semantics for this architecture are not modeled".into()),
        }
    }
    if m.clamp {
        x = if x.is_nan() || x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            x
        };
    }
    Ok(x.to_bits())
}

fn max_gt(a: f32, b: f32) -> bool {
    a > b || (a == 0.0 && b == 0.0 && !a.is_sign_negative() && b.is_sign_negative())
}

fn min_lt(a: f32, b: f32) -> bool {
    a < b || (a == 0.0 && b == 0.0 && a.is_sign_negative() && !b.is_sign_negative())
}

/// maxNum/minNum rules of the ISA (`v_max_num_f32`): a NaN operand loses to a number.
fn num_select(a: u32, b: u32, max: bool) -> u32 {
    match (is_nan(a), is_nan(b)) {
        (true, true) => a | 0x0040_0000,
        (true, false) => b,
        (false, true) => a,
        (false, false) => {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            if if max { max_gt(fa, fb) } else { min_lt(fa, fb) } {
                a
            } else {
                b
            }
        }
    }
}

/// Legacy `v_max_f32`/`v_min_f32`. gfx1151 (measured): a signalling NaN wins and is returned quieted (first one
/// in source order), two quiet NaNs give `a`, one quiet NaN loses to the number. Other arches: NaN is an error
/// (the behavior depends on MODE.IEEE and is not measured there).
fn legacy_select(arch: Arch, a: u32, b: u32, max: bool) -> Result<u32> {
    if arch != Arch::Gfx1151 {
        let (fa, fb) = (num(a)?, num(b)?);
        return Ok(if if max { max_gt(fa, fb) } else { min_lt(fa, fb) } { a } else { b });
    }
    Ok(if is_snan(a) {
        a | QUIET
    } else if is_snan(b) {
        b | QUIET
    } else if is_nan(a) {
        if is_nan(b) { a } else { b }
    } else if is_nan(b) {
        a
    } else if if max { max_gt(f32::from_bits(a), f32::from_bits(b)) } else { min_lt(f32::from_bits(a), f32::from_bits(b)) } {
        a
    } else {
        b
    })
}

fn f16_in(h: u16) -> Result<u32> {
    if h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 {
        return Err("NaN half operand: payload handling not specified".into());
    }
    Ok(convert::f16_to_f32(h))
}

/// A 16-bit source: true16 operands already carry their half; SGPR/literal operands use
/// OP_SEL; float inline constants are f16 values.
fn half_src(s: &State, op: &Operand, i: &Inst, lane: usize, n: usize) -> Result<u16> {
    let v = match op {
        Operand::Half(..) => s.read(op, lane, 0)?,
        Operand::Inline(InlineConst::FloatBits(b)) => u32::from(convert::f32_to_f16(*b)),
        Operand::Inline(InlineConst::InvTwoPi) => 0x3118,
        _ => {
            let w = s.read(op, lane, 0)?;
            if i.mods.op_sel & (1 << n) != 0 {
                w >> 16
            } else {
                w
            }
        }
    };
    Ok(v as u16)
}

/// Fused multiply-add inputs for `v_fma_mix*` as post-modifier f32 words: `{OPSEL_HI,OPSEL}` choose f32/lo/hi
/// half (widened exactly, a NaN half keeps sign and payload), NEG_HI is an absolute value, NEG a negation. A NaN
/// operand needs a qualified arch and no abs/neg on it.
fn mix_in(arch: Arch, s: &State, o: &[Operand], i: &Inst, lane: usize) -> Result<[u32; 3]> {
    let m = &i.mods;
    let mut out = [0u32; 3];
    for n in 0..3 {
        let op = o.get(n + 1).ok_or("missing vector operand")?;
        let bits = if m.op_sel_hi & (1 << n) == 0 {
            s.read(op, lane, 0)?
        } else if let Operand::Inline(InlineConst::FloatBits(b)) = op {
            convert::f16_to_f32(convert::f32_to_f16(*b))
        } else {
            let w = s.read(op, lane, 0)?;
            convert::f16_to_f32((if m.op_sel & (1 << n) != 0 { w >> 16 } else { w }) as u16)
        };
        let (abs, neg) = (m.neg_hi & (1 << n) != 0, m.neg_lo & (1 << n) != 0);
        if is_nan(bits) {
            if !edge_qualified(arch) {
                return Err(NAN_IN.into());
            }
            if abs || neg {
                return Err(NAN_MOD.into());
            }
            if m.op_sel != 0 || m.op_sel_hi & 0b101 != 0 {
                return Err("NaN operand of v_fma_mix with op_sel or a half A/C source: only op_sel:[0,0,0] with B as the half source is measured".into());
            }
        }
        out[n] = sgn(bits, abs, neg);
    }
    Ok(out)
}

/// Exact `a * b + c` of finite f32 words as an f64 that rounds to odd. The f64 product is exact (24 x 24 bit
/// significands); the f64 sum plus its exact TwoSum error rounds to odd, which keeps the sticky information a later
/// nearest-even rounding to at most 24 bits needs.
fn fma_wide_odd(a: u32, b: u32, c: u32) -> f64 {
    let (a, b, c) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)), f64::from(f32::from_bits(c)));
    let p = a * b;
    let s = p + c;
    let bb = s - p;
    let err = (p - (s - bb)) + (c - bb);
    if err != 0.0 && s.to_bits() & 1 == 0 {
        // `s` has the even neighbor's parity: step to the odd neighbor on the side of the error.
        let away = (err > 0.0) == (s > 0.0);
        f64::from_bits(if away { s.to_bits() + 1 } else { s.to_bits() - 1 })
    } else {
        s
    }
}

/// `a * b + c` of finite f32 words, rounded once to binary16 (nearest even, overflow to infinity), as the
/// `v_fma_mix{lo,hi}_f16` tables require.
fn fma_f16(a: u32, b: u32, c: u32) -> u16 {
    f64_to_f16(fma_wide_odd(a, b, c))
}

/// `v_div_fmas_f32` with a set VCC lane and finite operands (gfx1201): the exact fused sum scaled by 2**64 if the
/// biased exponent of `c` is above 127, else by 2**-64, then one nearest-even rounding to f32. The scale is a power of
/// two that keeps every finite f32 term in the normal f64 range, so it is exact; rounding happens only in the final
/// conversion, which also covers subnormal and overflowing f32 results.
fn fmas_f32(a: u32, b: u32, c: u32) -> u32 {
    const UP: f64 = f64::from_bits((1023 + 64) << 52);
    const DOWN: f64 = f64::from_bits((1023 - 64) << 52);
    let scale = if c >> 23 & 0xff > 127 { UP } else { DOWN };
    ((fma_wide_odd(a, b, c) * scale) as f32).to_bits()
}

fn rne64(x: u64, shift: u32) -> u64 {
    if shift >= 64 {
        return 0;
    }
    let (q, r, half) = (x >> shift, x & ((1u64 << shift) - 1), 1u64 << (shift - 1));
    q + u64::from(r > half || r == half && q & 1 != 0)
}

/// binary64 to binary16, nearest even; `x` is finite and, if non-zero, a normal f64.
fn f64_to_f16(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u32;
    if x == 0.0 {
        return sign as u16;
    }
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    if e > 15 {
        return (sign | 0x7c00) as u16;
    }
    let m = (bits & ((1u64 << 52) - 1)) | 1u64 << 52;
    let h = if e >= -14 {
        (((e + 15) as u32) << 10) + rne64(m, 42) as u32 - 1024
    } else {
        rne64(m, 42 + (-14 - e) as u32) as u32
    };
    (sign | h.min(0x7c00)) as u16
}

fn clamp01(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

fn vector(arch: Arch, name: &str, o: &[Operand], i: &Inst, s: &State, lane: usize, d: Src0) -> Result<Val> {
    let m = &i.mods;
    let raw = |n: usize| -> Result<u32> {
        let op = o.get(n + 1).ok_or("missing vector operand")?;
        read_src(s, op, lane, if n == 0 { d } else { Src0::Plain })
    };
    let f = |n: usize| -> Result<u32> {
        let v = raw(n)?;
        if n == 0 && matches!(d, Src0::Zero) && (m.abs | m.neg) & 1 != 0 {
            return Err("DPP bound-control zero with src0 modifiers: order not specified".into());
        }
        Ok(sgn(v, m.abs >> n & 1 != 0, m.neg >> n & 1 != 0))
    };
    let fnum = |n: usize| -> Result<f32> { num(f(n)?) };
    // Post-modifier operand word for NaN-propagating opcodes: `abs`, and `neg` where it was not measured, on a
    // NaN operand is an error (its effect on sign/payload/priority is not measured).
    let g = |n: usize, neg_ok: bool| -> Result<u32> {
        let v = f(n)?;
        if is_nan(v) && (m.abs >> n & 1 != 0 || (m.neg >> n & 1 != 0 && !neg_ok)) {
            return Err(NAN_MOD.into());
        }
        Ok(v)
    };
    // Literal constant of fmamk/fmaak: VOP2 keeps it in `Inst::literal`, VOPD as an operand.
    let lit = |pos: usize| -> Result<u32> {
        if o.len() > 3 {
            raw(pos)
        } else {
            i.literal.ok_or_else(|| "missing literal constant".into())
        }
    };
    let (class, value) = match stem(name) {
        "v_mov_b32" | "v_dual_mov_b32" => (Class::Int, raw(0)?),
        "v_cndmask_b32" | "v_dual_cndmask_b32" => {
            let mask = if o.len() > 3 { s.read(&o[3], lane, 0)? } else { s.vcc };
            let pick = if mask >> lane & 1 != 0 { 1 } else { 0 };
            (Class::Select, f(pick)?)
        }
        "v_add_nc_u32" | "v_dual_add_nc_u32" => (Class::Int, raw(0)?.wrapping_add(raw(1)?)),
        "v_sub_nc_u32" => (Class::Int, raw(0)?.wrapping_sub(raw(1)?)),
        "v_subrev_nc_u32" => (Class::Int, raw(1)?.wrapping_sub(raw(0)?)),
        "v_add3_u32" => (Class::Int, raw(0)?.wrapping_add(raw(1)?).wrapping_add(raw(2)?)),
        "v_and_b32" | "v_dual_and_b32" => (Class::Int, raw(0)? & raw(1)?),
        "v_or_b32" => (Class::Int, raw(0)? | raw(1)?),
        "v_xor_b32" => (Class::Int, raw(0)? ^ raw(1)?),
        "v_lshlrev_b32" | "v_dual_lshlrev_b32" => (Class::Int, raw(1)? << (raw(0)? & 31)),
        "v_lshrrev_b32" => (Class::Int, raw(1)? >> (raw(0)? & 31)),
        "v_ashrrev_i32" => (Class::Int, ((raw(1)? as i32) >> (raw(0)? & 31)) as u32),
        "v_lshl_add_u32" => (Class::Int, (raw(0)? << (raw(1)? & 31)).wrapping_add(raw(2)?)),
        "v_lshl_or_b32" => (Class::Int, (raw(0)? << (raw(1)? & 31)) | raw(2)?),
        "v_mul_lo_u32" => (Class::Int, raw(0)?.wrapping_mul(raw(1)?)),
        "v_mul_hi_u32" => (Class::Int, (u64::from(raw(0)?) * u64::from(raw(1)?) >> 32) as u32),
        "v_mul_hi_i32" => (Class::Int, ((i64::from(raw(0)? as i32) * i64::from(raw(1)? as i32)) >> 32) as u32),
        "v_mul_u32_u24" => (Class::Int, (raw(0)? & 0xff_ffff).wrapping_mul(raw(1)? & 0xff_ffff)),
        "v_mad_u32_u24" => (Class::Int, (raw(0)? & 0xff_ffff).wrapping_mul(raw(1)? & 0xff_ffff).wrapping_add(raw(2)?)),
        "v_min_i32" => (Class::Int, (raw(0)? as i32).min(raw(1)? as i32) as u32),
        "v_max_i32" => (Class::Int, (raw(0)? as i32).max(raw(1)? as i32) as u32),
        "v_min_u32" => (Class::Int, raw(0)?.min(raw(1)?)),
        "v_max_u32" => (Class::Int, raw(0)?.max(raw(1)?)),
        "v_bfe_u32" => {
            let width = raw(2)? & 31;
            (Class::Int, if width == 0 { 0 } else { (raw(0)? >> (raw(1)? & 31)) & ((1u32 << width) - 1) })
        }
        "v_perm_b32" => {
            // BYTE_PERMUTE({S0,S1}, sel): 0..7 bytes, 8..11 sign of bytes 1/3/5/7, 12 zero, >=13 0xff.
            let (a, b, c) = (raw(0)?, raw(1)?, raw(2)?);
            let bytes = (u64::from(b) | (u64::from(a) << 32)).to_le_bytes();
            let sign = |byte: u8| if byte & 0x80 != 0 { 0xff } else { 0 };
            let mut out = 0u32;
            for n in 0..4 {
                let byte = match (c >> (8 * n)) & 255 {
                    sel @ 0..=7 => bytes[sel as usize],
                    8 => sign(bytes[1]),
                    9 => sign(bytes[3]),
                    10 => sign(bytes[5]),
                    11 => sign(bytes[7]),
                    12 => 0,
                    _ => 0xff,
                };
                out |= u32::from(byte) << (8 * n);
            }
            (Class::Int, out)
        }
        "v_cvt_f32_u32" => (Class::F32, (raw(0)? as f32).to_bits()),
        "v_cvt_f32_i32" => (Class::F32, (raw(0)? as i32 as f32).to_bits()),
        // Saturating, truncating, NaN -> 0 (ISA: "NAN is converted to 0").
        "v_cvt_i32_f32" => (Class::FloatToInt, convert::f32_to_i32(f(0)?)),
        "v_cvt_u32_f32" => (Class::FloatToInt, convert::f32_to_u32(f(0)?)),
        "v_rndne_f32" => {
            let bits = g(0, false)?;
            if is_nan(bits) {
                if !edge_qualified(arch) {
                    return Err(NAN_IN.into());
                }
                (Class::F32, bits | QUIET)
            } else {
                let r = f32::from_bits(bits).round_ties_even();
                if bits >> 31 != 0 && r == 0.0 && !edge_qualified(arch) {
                    return Err("v_rndne_f32 of a negative input rounding to zero: pseudocode gives +0, IEEE gives -0; only gfx1151 and gfx1201 are measured (-0)".into());
                }
                (Class::F32, r.to_bits())
            }
        }
        "v_cvt_f32_f16" => {
            let h = half_src(s, o.get(1).ok_or("missing vector operand")?, i, lane, 0)?;
            if h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 {
                if !edge_qualified(arch) {
                    return Err("NaN half operand: payload handling not specified".into());
                }
                if (m.abs | m.neg) & 1 != 0 {
                    return Err(NAN_MOD.into());
                }
                (Class::F32, convert::f16_to_f32(h) | QUIET)
            } else {
                let h = sgn(u32::from(h) << 16, m.abs & 1 != 0, m.neg & 1 != 0) >> 16;
                (Class::F32, f16_in(h as u16)?)
            }
        }
        "v_cvt_f16_f32" => {
            plain(i)?;
            if !matches!(o[0], Operand::Half(..)) {
                return Err("16-bit destination without a true16 half selector: upper-half policy not specified".into());
            }
            let x = f(0)?;
            if is_nan(x) && !edge_qualified(arch) {
                return Err(NAN_IN.into());
            }
            (Class::Own, u32::from(convert::f32_to_f16(x)))
        }
        "v_ldexp_f32" => {
            let x = g(0, false)?;
            if is_nan(x) && !edge_qualified(arch) {
                return Err(NAN_IN.into());
            }
            (Class::F32, convert::ldexp_f32(x, raw(1)? as i32))
        }
        "v_add_f32" | "v_dual_add_f32" => (Class::F32, add_bits(arch, g(0, false)?, g(1, false)?)?),
        // v_sub_f32 negates S1 before the NaN pick (measured: a NaN S1 returns with its sign flipped).
        "v_sub_f32" => (Class::F32, add_bits(arch, g(0, false)?, g(1, false)? ^ SIGN)?),
        "v_dual_sub_f32" => (Class::F32, fin_invalid(arch, fnum(0)? - fnum(1)?)?),
        "v_subrev_f32" | "v_dual_subrev_f32" => (Class::F32, fin_invalid(arch, fnum(1)? - fnum(0)?)?),
        "v_mul_f32" | "v_dual_mul_f32" => (Class::F32, mul_bits(arch, g(0, false)?, g(1, false)?)?),
        // Source `neg` on a NaN is measured for S0 and S2 only.
        "v_fma_f32" => (Class::F32, fma_bits(arch, g(0, true)?, g(1, false)?, g(2, true)?)?),
        "v_fmac_f32" | "v_dual_fmac_f32" => {
            let acc = s.read(&o[0], lane, 0)?;
            (Class::F32, fma_bits(arch, g(0, false)?, g(1, false)?, acc)?)
        }
        // fmamk: D = S0 * K + S1; fmaak: D = S0 * S1 + K.
        "v_fmamk_f32" => (Class::F32, fma_bits(arch, g(0, false)?, lit(1)?, g(1, false)?)?),
        "v_dual_fmamk_f32" => (Class::F32, fin(fnum(0)?.mul_add(num(lit(1)?)?, fnum(2)?))?),
        "v_fmaak_f32" | "v_dual_fmaak_f32" => (Class::F32, fin(fnum(0)?.mul_add(fnum(1)?, num(lit(2)?)?))?),
        "v_max_f32" => (Class::F32, legacy_select(arch, g(0, false)?, g(1, false)?, true)?),
        "v_min_f32" => (Class::F32, legacy_select(arch, g(0, false)?, g(1, false)?, false)?),
        "v_max3_f32" => (Class::F32, legacy_select(arch, legacy_select(arch, g(0, false)?, g(1, false)?, true)?, g(2, false)?, true)?),
        "v_min3_f32" => (Class::F32, legacy_select(arch, legacy_select(arch, g(0, false)?, g(1, false)?, false)?, g(2, false)?, false)?),
        "v_max_num_f32" | "v_dual_max_num_f32" => (Class::F32, num_select(f(0)?, f(1)?, true)),
        "v_min_num_f32" | "v_dual_min_num_f32" => (Class::F32, num_select(f(0)?, f(1)?, false)),
        "v_max3_num_f32" => (Class::F32, num_select(num_select(f(0)?, f(1)?, true), f(2)?, true)),
        "v_min3_num_f32" => (Class::F32, num_select(num_select(f(0)?, f(1)?, false), f(2)?, false)),
        "v_div_fmas_f32" => {
            let vcc = s.vcc & (1 << lane) != 0;
            if vcc && !edge_qualified(arch) {
                return Err("v_div_fmas_f32 with VCC set on an unqualified architecture: ISA says 2.0F**32 while v_div_scale_f32 scales by 2**64; the sweep lacks the fused-rounding discriminators, so the scaled result is not qualified".into());
            }
            let (a, b, c) = (g(0, false)?, g(1, false)?, g(2, false)?);
            if vcc && [a, b, c].iter().all(|&x| x & 0x7f80_0000 != 0x7f80_0000) {
                // VCC set scales the result. Only gfx1201 has the finite fused-rounding discriminators; for a
                // NaN/Inf operand the result (priority/sign/payload) does not depend on the scale on any qualified arch.
                if arch != Arch::Gfx1201 {
                    return Err("v_div_fmas_f32 with VCC set and finite operands: ISA says 2.0F**32 while v_div_scale_f32 scales by 2**64; the gfx1151 sweep lacks the fused-rounding discriminators, so the scaled result is not qualified".into());
                }
                (Class::F32, fmas_f32(a, b, c))
            } else {
                (Class::F32, fma_bits(arch, a, b, c)?)
            }
        }
        "v_div_fixup_f32" => (Class::F32, div_fixup(arch, g(0, false)?, g(1, false)?, g(2, false)?)?),
        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
            let [a, b, c] = mix_in(arch, s, o, i, lane)?;
            let half_result = stem(name) != "v_fma_mix_f32";
            let finite = [a, b, c].iter().all(|&x| x & 0x7f80_0000 != 0x7f80_0000);
            if half_result && finite && edge_qualified(arch) {
                // Fused, then rounded once straight to f16.
                let mut h = fma_f16(a, b, c);
                if m.clamp {
                    h = convert::f32_to_f16(clamp01(f32::from_bits(convert::f16_to_f32(h))).to_bits());
                }
                (Class::Own, u32::from(h))
            } else {
                let mut r = fma_bits(arch, a, b, c)?;
                if m.clamp {
                    if is_nan(r) {
                        return Err("clamp of a NaN v_fma_mix result is not measured".into());
                    }
                    r = clamp01(f32::from_bits(r)).to_bits();
                }
                (Class::Own, if half_result { u32::from(convert::f32_to_f16(r)) } else { r })
            }
        }
        "v_exp_f32" | "v_rcp_f32" | "v_rcp_iflag_f32" | "v_log_f32" | "v_rsq_f32" | "v_sqrt_f32" => {
            if !matches!(arch,Arch::Gfx1151|Arch::Gfx1201) {return Err(format!("unqualified numerical architecture {arch:?}"));}
            let model=match stem(name) {
                "v_exp_f32"=>super::trans::exp_f32,"v_rcp_f32"=>super::trans::rcp_f32,
                "v_rcp_iflag_f32"=>super::trans::rcp_iflag_f32,"v_log_f32"=>super::trans::log_f32,
                "v_rsq_f32"=>super::trans::rsq_f32,_=>super::trans::sqrt_f32,
            };
            (Class::F32,model(arch,f(0)?))
        },
        _ => return Err(format!("{UNSUPPORTED} {name}")),
    };
    let value = match class {
        Class::Int => {
            plain(i)?;
            value
        }
        Class::Select => {
            if m.omod != Omod::None || m.clamp {
                return Err("omod/clamp on v_cndmask_b32 is not modeled".into());
            }
            value
        }
        Class::FloatToInt => {
            if m.omod != Omod::None {
                return Err("omod on an integer-result opcode is not modeled".into());
            }
            value
        }
        Class::F32 => out_f32(arch, value, i)?,
        Class::Own => value,
    };
    Ok(match stem(name) {
        "v_fma_mixlo_f16" => Val::Low16(value),
        "v_fma_mixhi_f16" => Val::High16(value),
        _ => Val::Dword(value),
    })
}

fn cvt_pk_f32_fp8(arch: Arch, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let mut out = [None; 32];
    for lane in active(s) {
        let h = half_src(s, &o[1], i, lane, 0)?;
        let mut pair = [0u32; 2];
        for (n, slot) in pair.iter_mut().enumerate() {
            let byte = (h >> (8 * n)) as u8;
            *slot = if byte & 0x7f != 0x7f {
                convert::fp8_to_f32(byte)
            } else if arch == Arch::Gfx1201 {
                // Both NaN encodings of either element give the default NaN (gfx1201 gids 93/94, all 4096 pairs).
                DEFAULT_NAN
            } else {
                return Err("FP8 NaN to f32: NaN payload not specified".into());
            };
        }
        out[lane] = Some(pair);
    }
    for (lane, pair) in out.into_iter().enumerate() {
        if let Some(pair) = pair {
            s.put(&o[0], lane, 0, pair[0])?;
            s.put(&o[0], lane, 1, pair[1])?;
        }
    }
    Ok(())
}

/// `v_permlane16_b32` / `v_permlanex16_b32`. OPSEL[0] is fetch-inactive, OPSEL[1] bound
/// control: an inactive source lane disables the write, reads zero (BC) or is read (FI).
fn permlane(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let (o, m) = (&i.operands, &i.mods);
    if m.abs | m.neg != 0 || m.omod != Omod::None || m.clamp || m.op_sel & 0b1100 != 0 {
        return Err(format!("{name}: abs/neg/omod/clamp and OPSEL[2]/OPSEL[3] are not modeled (only OPSEL[0]=FI, OPSEL[1]=BC)"));
    }
    let sel = u64::from(s.read(&o[2], 0, 0)?) | (u64::from(s.read(&o[3], 0, 0)?) << 32);
    let cross = name == "v_permlanex16_b32";
    let mut vals = [None; 32];
    for lane in active(s) {
        let row = if cross { (lane ^ 16) & 16 } else { lane & 16 };
        let src = row | ((sel >> (4 * (lane & 15))) & 15) as usize;
        vals[lane] = if i.mods.op_sel & 1 != 0 || s.exec & (1 << src) != 0 {
            Some(s.read(&o[1], src, 0)?)
        } else if i.mods.op_sel & 2 != 0 {
            Some(0)
        } else {
            None
        };
    }
    for (lane, v) in vals.into_iter().enumerate() {
        if let Some(v) = v {
            s.put(&o[0], lane, 0, v)?;
        }
    }
    Ok(())
}

/// `v_div_scale_f32` on gfx1151 and gfx1201, from the hardware sweeps (spec gids 29/30/31, all 262144 operand tuples
/// of the 64-member f32 set including zeros, denormals, infinities and NaNs, all 4 VCC/SCC states). Operands are
/// `S0` (the value to scale), `S1` (denominator), `S2` (numerator), post-modifier words; everything is decided
/// by the raw biased exponent fields (`e1`, `e2`, so NaN/infinity count as 255), `d = e2 - e1`:
///
/// * VCC lane predicate = `d >= 96 || d <= -96`, also for zero operands; the ISA document's "denormal
///   quotient" test is not what the hardware does.
/// * `S1 == 0 || S2 == 0`: `D = 0xffc00000` (S0 ignored, even a NaN).
/// * `d >= 96`: `D = S0 == S1 ? S0 * 2^64 : S0`.
/// * `e1 == 0` (denormal denominator): `D = S0 * 2^64`.
/// * `e1 >= 254` (reciprocal denormal): `D = S0 * 2^-64`, except `D = S0` when the predicate is set and
///   `S0 != S1`. The hardware subtracts 64 from the exponent field: `S0` with exponent field <= 63 (denormals
///   included, zero excluded) gives `+-inf`, exponent fields >= 91 are exact.
/// * `d <= -96`: `D = S0 == S1 ? S0 : S0 * 2^64` (the numerator is scaled).
/// * `e2 <= 23`: `D = S0 * 2^64`; otherwise `D = S0`.
///
/// `S0 * 2^64` leaves NaN and infinity untouched (a signalling NaN stays signalling); `S0 == S1` is a bitwise
/// comparison. Domains the sweep leaves open (`e1` 228..=253, `d == -96` with `e1 != 254`, `e2` 24..=26 where the
/// scale decision is reached, the `2^-64` of exponent fields 64..=90) are errors.
fn div_scale_chain(s0: u32, s1: u32, s2: u32) -> Result<(u32, bool)> {
    let exp = |x: u32| (x >> 23 & 255) as i32;
    let (e0, e1, e2) = (exp(s0), exp(s1), exp(s2));
    let d = e2 - e1;
    if d == -96 && e1 != 254 {
        return Err("v_div_scale_f32: exponent difference -96 is measured only for denominator exponent 254".into());
    }
    let flag = d >= 96 || d <= -96;
    if is_zero(s1) || is_zero(s2) {
        return Ok((0xffc0_0000, flag));
    }
    let up = |x: u32| if x >> 23 & 255 == 255 { x } else { convert::ldexp_f32(x, 64) };
    let down = || -> Result<u32> {
        Ok(match e0 {
            255 => s0,
            _ if is_zero(s0) => s0,
            0..=63 => (s0 & SIGN) | INF,
            64..=90 => return Err("v_div_scale_f32: S0 * 2^-64 with exponent field 64..=90 is not measured".into()),
            _ => convert::ldexp_f32(s0, -64),
        })
    };
    let scaled = if d >= 96 {
        if s0 == s1 { up(s0) } else { s0 }
    } else if e1 == 0 {
        up(s0)
    } else if (228..=253).contains(&e1) {
        return Err("v_div_scale_f32: denominator exponent 228..=253 is not measured".into());
    } else if e1 >= 254 {
        if flag && s0 != s1 { s0 } else { down()? }
    } else if d <= -96 {
        if s0 == s1 { s0 } else { up(s0) }
    } else if e2 <= 23 {
        up(s0)
    } else if e2 <= 26 {
        return Err("v_div_scale_f32: numerator exponent 24..=26 is not measured".into());
    } else {
        s0
    };
    Ok((scaled, flag))
}

/// Qualified-arch `v_div_scale_f32`: [`div_scale_chain`], plus gfx1201 quieting every signalling NaN it would
/// return as `D` (30752 of the 262144 tuples, none left unquieted); gfx1151 returns it untouched.
fn div_scale_edge(arch: Arch, s0: u32, s1: u32, s2: u32) -> Result<(u32, bool)> {
    let (d, flag) = div_scale_chain(s0, s1, s2)?;
    Ok((if arch == Arch::Gfx1201 && is_snan(d) { d | QUIET } else { d }, flag))
}

/// `v_div_scale_f32` per the RDNA4 pseudocode, restricted to its proven domain (every arch without a
/// hardware-measured chain, i.e. not gfx1151/gfx1201, where the pseudocode does not match the hardware).
fn div_scale_isa(s0: u32, s1: u32, s2: u32) -> Result<(u32, bool)> {
    let ldexp64 = |x: u32| -> Result<u32> {
        if is_nan(x) {
            return Err(NAN_IN.into());
        }
        Ok(convert::ldexp_f32(x, 64))
    };
    let (e1, e2) = ((s1 >> 23 & 255) as i32, (s2 >> 23 & 255) as i32);
    if s1 & 0x7fff_ffff == 0 || s2 & 0x7fff_ffff == 0 {
        return Err("v_div_scale_f32 with a zero numerator/denominator yields NAN.f32: encoding not specified".into());
    }
    if is_nan(s1) || is_nan(s2) {
        return Err(NAN_IN.into());
    }
    if e1 >= 253 {
        return Err("v_div_scale_f32: `1.0/S1 == DENORM.f64` is ambiguous when |S1| >= 2^126".into());
    }
    let mut flag = false;
    let d = if e2 - e1 >= 96 {
        flag = true;
        if s0 == s1 { ldexp64(s0)? } else { s0 }
    } else if e1 == 0 {
        ldexp64(s0)?
    } else {
        let qf = f32::from_bits(s2) / f32::from_bits(s1);
        let qd = f64::from(f32::from_bits(s2)) / f64::from(f32::from_bits(s1));
        let denorm32 = qf != 0.0 && qf.abs() < f32::MIN_POSITIVE;
        let denorm64 = qd.abs() < f64::from(f32::MIN_POSITIVE);
        if denorm32 != denorm64 || qd.abs() == f64::from(f32::MIN_POSITIVE) {
            return Err("v_div_scale_f32: quotient lies on the denormal boundary; hardware rounding not specified".into());
        }
        if denorm32 {
            flag = true;
            if f32::from_bits(s0) == f32::from_bits(s2) { ldexp64(s0)? } else { s0 }
        } else if e2 <= 23 {
            ldexp64(s0)?
        } else {
            s0
        }
    };
    Ok((d, flag))
}

/// `v_div_scale_f32 D, SDST, S0, S1, S2`. The predicate lands in SDST: VCC is replaced as a whole (SCC and every
/// other state untouched), a NULL destination discards it, so VCC is then preserved.
fn div_scale(arch: Arch, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let m = &i.mods;
    let mut out = [None; 32];
    let mut vcc = 0u32;
    for lane in active(s) {
        let src = |n: usize| -> Result<u32> {
            let v = sgn(s.read(&o[2 + n], lane, 0)?, m.abs >> n & 1 != 0, m.neg >> n & 1 != 0);
            if edge_qualified(arch) && is_nan(v) && (m.abs | m.neg) >> n & 1 != 0 {
                return Err(NAN_MOD.into());
            }
            Ok(v)
        };
        let (s0, s1, s2) = (src(0)?, src(1)?, src(2)?);
        let (d, flag) = if edge_qualified(arch) { div_scale_edge(arch, s0, s1, s2)? } else { div_scale_isa(s0, s1, s2)? };
        if flag {
            vcc |= 1 << lane;
        }
        out[lane] = Some(out_f32(arch, d, i)?);
    }
    for (lane, v) in out.into_iter().enumerate() {
        if let Some(v) = v {
            s.put(&o[0], lane, 0, v)?;
        }
    }
    s.put(&o[1], 0, 0, vcc)
}

/// `v_div_fixup_f32 D, S0 (quotient), S1 (denominator), S2 (numerator)`. The special-case chain is the ISA's
/// (`cvtToQuietNAN` of `S2`, else `S1`; `0xffc00000` for 0/0 and inf/inf; signed inf/zero). Measured on gfx1151 and
/// gfx1201 (gid 33, all 262144 tuples each): a NaN quotient reaching the final branch gives `sign | inf` and an
/// underflowing quotient (`exponent(S2) - exponent(S1) < -150`) gives the signed zero.
fn div_fixup(arch: Arch, q: u32, den: u32, nu: u32) -> Result<u32> {
    let (da, na) = (den & 0x7fff_ffff, nu & 0x7fff_ffff);
    let sign = (den ^ nu) & SIGN;
    Ok(if na > INF {
        nu | QUIET
    } else if da > INF {
        den | QUIET
    } else if (da == 0 && na == 0) || (da == INF && na == INF) {
        0xffc0_0000
    } else if da == 0 || na == INF {
        sign | INF
    } else if da == INF || na == 0 {
        sign
    } else if ((na >> 23) as i32) - ((da >> 23) as i32) < -150 {
        if !edge_qualified(arch) {
            return Err("v_div_fixup_f32 underflow branch: UNDERFLOW_F32 is not defined by the ISA".into());
        }
        sign
    } else if edge_qualified(arch) && is_nan(q) {
        sign | INF
    } else {
        // exponent(S1) == 255 is unreachable here: NaN and infinity were handled above.
        sign | (q & 0x7fff_ffff)
    })
}

#[derive(Clone, Copy)]
enum Pred {
    F,
    Tru,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Lg,
    O,
    U,
    Nge,
    Nlg,
    Ngt,
    Nle,
    Neq,
    Nlt,
    Class,
}

fn pred(p: &str) -> Option<Pred> {
    Some(match p {
        "f" => Pred::F,
        "tru" => Pred::Tru,
        "eq" => Pred::Eq,
        "ne" => Pred::Ne,
        "lt" => Pred::Lt,
        "le" => Pred::Le,
        "gt" => Pred::Gt,
        "ge" => Pred::Ge,
        "lg" => Pred::Lg,
        "o" => Pred::O,
        "u" => Pred::U,
        "nge" => Pred::Nge,
        "nlg" => Pred::Nlg,
        "ngt" => Pred::Ngt,
        "nle" => Pred::Nle,
        "neq" => Pred::Neq,
        "nlt" => Pred::Nlt,
        "class" => Pred::Class,
        _ => return None,
    })
}

fn ord_test<T: Ord>(p: Pred, a: T, b: T) -> Option<bool> {
    Some(match p {
        Pred::F => false,
        Pred::Tru => true,
        Pred::Eq => a == b,
        Pred::Ne => a != b,
        Pred::Lt => a < b,
        Pred::Le => a <= b,
        Pred::Gt => a > b,
        Pred::Ge => a >= b,
        _ => return None,
    })
}

fn float_test(p: Pred, a: f32, b: f32) -> Option<bool> {
    Some(match p {
        Pred::F => false,
        Pred::Tru => true,
        Pred::Eq => a == b,
        Pred::Lt => a < b,
        Pred::Le => a <= b,
        Pred::Gt => a > b,
        Pred::Ge => a >= b,
        Pred::Lg => a < b || a > b,
        Pred::O => !a.is_nan() && !b.is_nan(),
        Pred::U => a.is_nan() || b.is_nan(),
        Pred::Nge => !(a >= b),
        Pred::Nlg => !(a < b || a > b),
        Pred::Ngt => !(a > b),
        Pred::Nle => !(a <= b),
        Pred::Neq => !(a == b),
        Pred::Nlt => !(a < b),
        _ => return None,
    })
}

/// `v_cmp_*` / `v_cmpx_*` over u32/i32/f32 and `class_f32`. Lanes that are inactive or
/// disabled by DPP write a zero bit.
fn compare(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let m = &i.mods;
    let cmpx = name.starts_with("v_cmpx_");
    let rest = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")).ok_or("not a compare")?;
    let rest = rest.strip_suffix("_e64").or_else(|| rest.strip_suffix("_e32")).or_else(|| rest.strip_suffix("_dpp")).unwrap_or(rest);
    let (p, ty) = rest.rsplit_once('_').ok_or_else(|| format!("unsupported compare {name}"))?;
    let p = pred(p).ok_or_else(|| format!("unsupported compare {name}"))?;
    let float = ty == "f32";
    if !float && ty != "u32" && ty != "i32" {
        return Err(format!("unsupported compare {name}"));
    }
    if float != matches!(p, Pred::Lg | Pred::O | Pred::U | Pred::Nge | Pred::Nlg | Pred::Ngt | Pred::Nle | Pred::Neq | Pred::Nlt | Pred::Class)
        && !matches!(p, Pred::F | Pred::Tru | Pred::Eq | Pred::Lt | Pred::Le | Pred::Gt | Pred::Ge | Pred::Ne)
        || matches!(p, Pred::Ne) && float
        || matches!(p, Pred::Class) && !float
    {
        return Err(format!("unsupported compare {name}"));
    }
    if m.omod != Omod::None || (!float && m.neg | m.abs != 0) {
        return Err("omod, or neg/abs on an integer compare, is not modeled".into());
    }
    if o.len() < 2 {
        return Err("missing compare operand".into());
    }
    let (a_op, b_op) = (&o[o.len() - 2], &o[o.len() - 1]);
    let dpp = m.dpp;
    let fi = dpp_fi(i);
    let mut mask = 0u32;
    for lane in active(s) {
        let d = match dpp {
            Some(d) => match dpp_plan(&d, fi, s, lane)? {
                Some(d) => d,
                None => continue,
            },
            None => Src0::Plain,
        };
        let a = read_src(s, a_op, lane, d)?;
        let b = s.read(b_op, lane, 0)?;
        let yes = if let Pred::Class = p {
            // Modifiers apply to S0 only; S1 is the integer class mask.
            let class = f32_class(sgn(a, m.abs & 1 != 0, m.neg & 1 != 0));
            Some(b & (1 << class) != 0)
        } else if float {
            let x = f32::from_bits(sgn(a, m.abs & 1 != 0, m.neg & 1 != 0));
            let y = f32::from_bits(sgn(b, m.abs & 2 != 0, m.neg & 2 != 0));
            float_test(p, x, y)
        } else if ty == "i32" {
            ord_test(p, a as i32, b as i32)
        } else {
            ord_test(p, a, b)
        };
        if yes.ok_or_else(|| format!("unsupported compare {name}"))? {
            mask |= 1 << lane;
        }
    }
    if cmpx {
        s.exec = mask;
        Ok(())
    } else if o.len() >= 3 {
        s.put(&o[0], 0, 0, mask)
    } else {
        s.vcc = mask;
        Ok(())
    }
}

/// `v_cmp_class_f32` class index: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denorm, 5 -0,
/// 6 +0, 7 +denorm, 8 +normal, 9 +inf.
fn f32_class(a: u32) -> u32 {
    let abs = a & 0x7fff_ffff;
    let neg = a >> 31 != 0;
    if abs > INF {
        u32::from(a & 0x0040_0000 != 0)
    } else if abs == INF {
        if neg { 2 } else { 9 }
    } else if abs == 0 {
        if neg { 5 } else { 6 }
    } else if abs < 0x0080_0000 {
        if neg { 4 } else { 7 }
    } else if neg {
        3
    } else {
        8
    }
}

fn scalar(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let rd = |s: &State, idx: usize| -> Result<u32> { s.read(o.get(idx).ok_or("missing scalar operand")?, 0, 0) };
    if let Some(rest) = name.strip_prefix("s_cmp_") {
        let (p, ty) = rest.split_once('_').ok_or_else(|| format!("unsupported scalar compare {name}"))?;
        let p = match p {
            "eq" => Pred::Eq,
            "lg" => Pred::Ne,
            "gt" => Pred::Gt,
            "ge" => Pred::Ge,
            "lt" => Pred::Lt,
            "le" => Pred::Le,
            _ => return Err(format!("unsupported scalar compare {name}")),
        };
        let (a, b) = (rd(s, 0)?, rd(s, 1)?);
        s.scc = match ty {
            "u32" => ord_test(p, a, b),
            "i32" => ord_test(p, a as i32, b as i32),
            _ => return Err(format!("unsupported scalar compare {name}")),
        }
        .ok_or_else(|| format!("unsupported scalar compare {name}"))?;
        return Ok(());
    }
    if name == "s_bitcmp1_b32" || name == "s_bitcmp0_b32" {
        let bit = rd(s, 0)? >> (rd(s, 1)? & 31) & 1;
        s.scc = (bit == 1) == (name == "s_bitcmp1_b32");
        return Ok(());
    }
    let a = rd(s, 1)?;
    let b = if o.len() > 2 { rd(s, 2)? } else { 0 };
    let nonzero = |v: u32| (v, Some(v != 0));
    let (value, flag) = match name {
        "s_mov_b32" => (a, None),
        "s_cselect_b32" => (if s.scc { a } else { b }, None),
        "s_add_u32" | "s_add_co_u32" | "s_addc_u32" | "s_add_co_ci_u32" => {
            let carry = u64::from((name == "s_addc_u32" || name == "s_add_co_ci_u32") && s.scc);
            let sum = u64::from(a) + u64::from(b) + carry;
            (sum as u32, Some(sum >> 32 != 0))
        }
        "s_sub_u32" | "s_sub_co_u32" | "s_subb_u32" | "s_sub_co_ci_u32" => {
            let borrow = u64::from((name == "s_subb_u32" || name == "s_sub_co_ci_u32") && s.scc);
            (a.wrapping_sub(b).wrapping_sub(borrow as u32), Some(u64::from(b) + borrow > u64::from(a)))
        }
        "s_add_i32" | "s_add_co_i32" => {
            let (v, f) = (a as i32).overflowing_add(b as i32);
            (v as u32, Some(f))
        }
        "s_sub_i32" | "s_sub_co_i32" => {
            let (v, f) = (a as i32).overflowing_sub(b as i32);
            (v as u32, Some(f))
        }
        "s_mul_i32" => (a.wrapping_mul(b), None),
        "s_mulk_i32" => (s.read(&o[0], 0, 0)?.wrapping_mul(a), None),
        "s_mul_hi_u32" => ((u64::from(a) * u64::from(b) >> 32) as u32, None),
        "s_and_b32" => nonzero(a & b),
        "s_or_b32" => nonzero(a | b),
        "s_xor_b32" => nonzero(a ^ b),
        "s_andn2_b32" => nonzero(a & !b),
        "s_orn2_b32" => nonzero(a | !b),
        "s_not_b32" => nonzero(!a),
        "s_lshl_b32" => nonzero(a << (b & 31)),
        "s_lshr_b32" => nonzero(a >> (b & 31)),
        "s_ashr_i32" => nonzero(((a as i32) >> (b & 31)) as u32),
        "s_min_i32" => ((a as i32).min(b as i32) as u32, Some((a as i32) < (b as i32))),
        "s_max_i32" => ((a as i32).max(b as i32) as u32, Some((a as i32) >= (b as i32))),
        "s_min_u32" => (a.min(b), Some(a < b)),
        "s_max_u32" => (a.max(b), Some(a >= b)),
        "s_ctz_i32_b32" => (if a == 0 { u32::MAX } else { a.trailing_zeros() }, None),
        _ => return Err(format!("unsupported scalar opcode {name}")),
    };
    s.put(&o[0], 0, 0, value)?;
    if let Some(f) = flag {
        s.scc = f;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        operand::{Half, Special},
        reg::{Kind, RegRef},
    };

    // Update together with `State` in mod.rs when fields change.
    fn state() -> State {
        State { s: [0; 106], t: [0; 16], v: Box::new([[0; 32]; 256]), exec: u32::MAX, vcc: 0, scc: false, m0: 0, pc: 0, ended: false, barrier: None, signals: 0 }
    }
    fn reg(kind: Kind, base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind, base, len })
    }
    fn v(base: u16) -> Operand {
        reg(Kind::V, base, 1)
    }
    fn half(base: u16, h: Half) -> Operand {
        Operand::Half(RegRef { kind: Kind::V, base, len: 1 }, h)
    }
    fn lit(bits: u32) -> Operand {
        Operand::Literal(bits)
    }
    fn p2(k: i32) -> u32 {
        ((127 + k) as u32) << 23
    }
    fn insn(name: &str, ops: Vec<Operand>) -> Inst {
        let r = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(Arch::Gfx1201, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap()
    }
    fn run(name: &str, i: &Inst, s: &mut State) -> Result<()> {
        execute(Arch::Gfx1201, name, i, s)
    }

    // DPP16: row_shl/quad_perm/row_share/xmask/mirror routing, FI=0 inactive sources, BC, row/bank masks.
    #[test]
    fn dpp_routes_lanes_and_applies_masks_bound_ctrl_and_inactive_sources() {
        let mut i = insn("v_mov_b32_dpp", vec![v(1), v(0)]);
        let dpp = |ctrl, row_mask, bank_mask, bound_ctrl| Some(Dpp { ctrl, row_mask, bank_mask, bound_ctrl });
        let fresh = || {
            let mut st = state();
            for l in 0..32 {
                st.v[0][l] = 100 + l as u32;
                st.v[1][l] = 7;
            }
            st
        };
        // row_shl:1 with lane 5 inactive: lanes 15/31 are out of range, lane 4 reads inactive lane 5.
        i.mods.dpp = dpp(0x101, 0xf, 0xf, false);
        let mut st = fresh();
        st.exec = !(1 << 5);
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let keep = l == 5 || l % 16 == 15 || l == 4;
            assert_eq!(st.v[1][l], if keep { 7 } else { 101 + l as u32 }, "bc=0 lane {l}");
        }
        i.mods.dpp = dpp(0x101, 0xf, 0xf, true);
        let mut st = fresh();
        st.exec = !(1 << 5);
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let want = if l == 5 { 7 } else if l % 16 == 15 || l == 4 { 0 } else { 101 + l as u32 };
            assert_eq!(st.v[1][l], want, "bc=1 lane {l}");
        }
        // row_mask keeps row 0, bank_mask 0b1101 keeps lanes 4..7 and 20..23.
        i.mods.dpp = dpp(0x101, 0b10, 0b1101, true);
        let mut st = fresh();
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let keep = l < 16 || (l / 4) % 4 == 1;
            let want = if keep { 7 } else if l % 16 == 15 { 0 } else { 101 + l as u32 };
            assert_eq!(st.v[1][l], want, "masked lane {l}");
        }
        // Controls that never leave the row: source lane of each destination lane.
        let cases: [(u16, fn(usize) -> usize); 5] = [
            (0x1b, |l| (l & !3) | (3 - (l & 3))),   // quad_perm [3,2,1,0]
            (0x153, |l| (l & !15) + 3),             // row_share:3
            (0x161, |l| (l & !15) + ((l & 15) ^ 1)), // row_xmask:1
            (0x140, |l| (l & !15) + 15 - (l & 15)),  // row_mirror
            (0x123, |l| (l & !15) + (((l & 15) + 13) & 15)), // row_ror:3
        ];
        for (ctrl, src) in cases {
            i.mods.dpp = dpp(ctrl, 0xf, 0xf, false);
            let mut st = fresh();
            run("v_mov_b32_dpp", &i, &mut st).unwrap();
            for l in 0..32 {
                assert_eq!(st.v[1][l], 100 + src(l) as u32, "ctrl {ctrl:#x} lane {l}");
            }
        }
        i.mods.dpp = dpp(0x130, 0xf, 0xf, false);
        assert!(run("v_mov_b32_dpp", &i, &mut fresh()).is_err(), "wave_shl is not an RDNA control");
    }

    // VOPD reads every input before either half writes.
    #[test]
    fn vopd_halves_read_old_registers() {
        let (x, y) = {
            let t = crate::isa::gfx12();
            (t.iter().find(|r| r.name == "v_dual_mul_f32").unwrap(), t.iter().find(|r| r.name == "v_dual_add_f32").unwrap())
        };
        let ops = vec![v(0), v(1), v(2), v(1), v(0), v(3)];
        let i = Inst::from_parts(Arch::Gfx1201, x.op, x.form, FormFields::Vopd { y_op: y.op, x_operands: 3 }, ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap();
        let mut st = state();
        st.exec = 0b01;
        for l in 0..2 {
            st.v[0][l] = 2f32.to_bits();
            st.v[1][l] = 3f32.to_bits();
            st.v[2][l] = 4f32.to_bits();
            st.v[3][l] = 0.5f32.to_bits();
        }
        run("v_dual_mul_f32", &i, &mut st).unwrap();
        assert_eq!((st.v[0][0], st.v[1][0]), (12f32.to_bits(), 2.5f32.to_bits()), "X=v1*v2 and Y=old v0+v3");
        assert_eq!((st.v[0][1], st.v[1][1]), (2f32.to_bits(), 3f32.to_bits()), "inactive lane untouched");
    }

    // Permlane supports only FI=OPSEL[0] and BC=OPSEL[1]; every other modifier errors, even with EXEC=0.
    #[test]
    fn permlane_rejects_unsupported_modifiers_and_reserved_opsel() {
        // gfx12 has no v_permlane16_b32: build it from the gfx1100 table, with SGPR selectors.
        for (name, arch) in [("v_permlane16_b32", Arch::Gfx1100), ("v_permlanex16_b32", Arch::Gfx1201)] {
            let (table, ops) = if arch == Arch::Gfx1100 {
                (crate::isa::gfx1100(), vec![v(0), v(1), reg(Kind::S, 0, 1), reg(Kind::S, 1, 1)])
            } else {
                (crate::isa::gfx12(), vec![v(0), v(1), lit(0x76543210), lit(0xfedcba98)])
            };
            let r = table.iter().find(|r| r.name == name).unwrap();
            let base = Inst::from_parts(arch, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap();
            let fresh = || {
                let mut st = state();
                st.s[0] = 0x7654_3210;
                st.s[1] = 0xfedc_ba98;
                for l in 0..32 {
                    st.v[1][l] = 100 + l as u32;
                }
                st
            };
            let run = |i: &Inst, s: &mut State| execute(arch, name, i, s);
            let mut legal=base.clone();legal.mods.op_sel=3;
            let mut st=fresh();run(&legal,&mut st).unwrap();
            for lane in 0..32 {
                let source=if name=="v_permlanex16_b32"{lane^16}else{lane};
                assert_eq!(st.v[0][lane],100+source as u32,"{name} FI+BC lane {lane}");
            }
            let bad: [(&str, fn(&mut Inst)); 8] = [
                ("abs", |i| i.mods.abs = 1),
                ("neg", |i| i.mods.neg = 0b100),
                ("omod", |i| i.mods.omod = Omod::Mul2),
                ("clamp", |i| i.mods.clamp = true),
                ("op_sel[2]", |i| i.mods.op_sel = 0b0100),
                ("op_sel[3]", |i| i.mods.op_sel = 0b1000),
                ("op_sel[3]+FI", |i| i.mods.op_sel = 0b1001),
                ("op_sel[2]+BC", |i| i.mods.op_sel = 0b0110),
            ];
            for (what, apply) in bad {
                for exec in [u32::MAX, 0] {
                    let mut i = base.clone();
                    apply(&mut i);
                    let mut st = fresh();
                    st.exec = exec;
                    let before = st.v[0];
                    assert!(run(&i, &mut st).is_err(), "{name}: {what} exec {exec:#x} must error");
                    assert_eq!(st.v[0], before, "{name}: {what} must not write");
                }
            }
        }
    }

    // Input neg/abs apply abs first; OMOD then CLAMP; -0 clamps to +0 and -0*OMOD = +0.
    #[test]
    fn float_modifiers_order_and_zero_signs() {
        let mut i = insn("v_fma_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = (-3f32).to_bits();
        st.v[3][0] = 10f32.to_bits();
        i.mods.neg = 0b001;
        i.mods.abs = 0b010;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 4.0, "-2 * |-3| + 10");
        i.mods.omod = Omod::Mul2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 8.0);
        i.mods.omod = Omod::Div2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 2.0);
        i.mods.clamp = true;
        i.mods.omod = Omod::Mul2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 1.0, "clamp after omod");
        // fma(1, -0, -0) = -0: clamp and omod both give +0.
        let mut j = insn("v_fma_f32", vec![v(0), v(1), v(2), v(3)]);
        st.v[1][0] = 1f32.to_bits();
        st.v[2][0] = (-0f32).to_bits();
        st.v[3][0] = (-0f32).to_bits();
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0x8000_0000);
        j.mods.clamp = true;
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
        j.mods.clamp = false;
        j.mods.omod = Omod::Mul2;
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
        // A scaled result that flushes a non-zero denormal has an unspecified zero sign.
        st.v[1][0] = p2(-126);
        st.v[2][0] = 1f32.to_bits();
        st.v[3][0] = 0;
        j.mods.omod = Omod::Div2;
        assert!(run("v_fma_f32", &j, &mut st).is_err());
    }

    // gfx12 maxNum/minNum rules (gfx1201 gids 34..=38) and the NaN arithmetic the gfx1201 tables qualify; Gfx1100
    // keeps the hard error for arithmetic NaN.
    #[test]
    fn nan_rules_and_signed_zero_in_max_num() {
        let i = insn("v_max_num_f32_e32", vec![v(0), v(1), v(2)]);
        let mut st = state();
        let cases = [
            (0x7fc0_0001, 3f32.to_bits(), 3f32.to_bits()),
            (3f32.to_bits(), 0x7fc0_0001, 3f32.to_bits()),
            (0x7f80_0001, 0x7fc0_0002, 0x7fc0_0001),
            ((-0f32).to_bits(), 0f32.to_bits(), 0),
            (0f32.to_bits(), (-0f32).to_bits(), 0),
            (1f32.to_bits(), 2f32.to_bits(), 2f32.to_bits()),
        ];
        for (l, (a, b, _)) in cases.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
        }
        run("v_max_num_f32_e32", &i, &mut st).unwrap();
        for (l, (_, _, want)) in cases.iter().enumerate() {
            assert_eq!(st.v[0][l], *want, "lane {l}");
        }
        let (one, two, half_) = (1f32.to_bits(), 2f32.to_bits(), 0.5f32.to_bits());
        let (qn, qn2, qn3, sn) = (0x7fc0_0001u32, 0xffc0_0002u32, 0x7fc0_0003u32, 0x7f80_0001u32);
        // v_max3_num/v_min3_num are ((S0, S1), S2) over the same rule (gfx1201 gids 36..=38).
        let (max3, min3) = (insn("v_max3_num_f32", vec![v(0), v(1), v(2), v(3)]), insn("v_min3_num_f32", vec![v(0), v(1), v(2), v(3)]));
        let mut st = state();
        st.exec = 1;
        for (name, i, srcs, want) in [
            ("v_max3_num_f32", &max3, [sn, one, two], two),         // stage 1 drops the NaN, stage 2 orders numbers
            ("v_max3_num_f32", &max3, [one, two, qn], two),         // a NaN S2 loses to the number
            ("v_max3_num_f32", &max3, [qn, qn2, one], one),         // stage 1: two NaNs give quieted S0; it loses to S2
            ("v_max3_num_f32", &max3, [qn, qn2, qn3], qn),          // NaN everywhere: quieted S0
            ("v_max3_num_f32", &max3, [sn, qn2, one], one),         // stage 1 gives the quieted sNaN S0, which still loses
            ("v_min3_num_f32", &min3, [one, two, half_], half_),
        ] {
            (st.v[1][0], st.v[2][0], st.v[3][0]) = (srcs[0], srcs[1], srcs[2]);
            run(name, i, &mut st).unwrap();
            assert_eq!(st.v[0][0], want, "{name} {srcs:x?}");
        }
        // Arithmetic NaN: the first NaN operand is quieted, inf - inf is the default NaN (gfx1201 gids 0 and 2).
        let add = insn("v_add_f32_e32", vec![v(0), v(1), v(2)]);
        (st.v[1][0], st.v[2][0]) = (0x7f80_0001, one);
        run("v_add_f32_e32", &add, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0x7fc0_0001);
        (st.v[1][0], st.v[2][0]) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
        run("v_add_f32_e32", &add, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xffc0_0000);
        // Gfx1100 has no measurement: NaN operands and results are hard errors.
        let add = insn_on(Arch::Gfx1100, "v_add_f32_e32", vec![v(0), v(1), v(2)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 0x7fc0_0000;
        let e = execute(Arch::Gfx1100, "v_add_f32_e32", &add, &mut st).unwrap_err();
        assert!(e.contains("NaN"), "{e}");
        (st.v[1][0], st.v[2][0]) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
        assert!(execute(Arch::Gfx1100, "v_add_f32_e32", &add, &mut st).is_err(), "inf - inf default NaN encoding is unspecified");
    }

    // On arches without a measured chain (Gfx1100) v_div_scale_f32 flag/scale decisions follow the RDNA4
    // pseudocode chain in order.
    #[test]
    fn div_scale_isa_chain_and_vcc_mask_without_a_measured_chain() {
        let i = insn_on(Arch::Gfx1100, "v_div_scale_f32", vec![v(0), Operand::Special(Special::VccLo), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 0b11111;
        // (S0, S1 = denominator, S2 = numerator) per lane.
        let lanes = [
            (3f32.to_bits(), 3f32.to_bits(), 1f32.to_bits()),            // ordinary: pass S0, no flag
            (p2(-110), 3f32.to_bits(), p2(-110)),                        // exponent(S2) <= 23: scale S0 by 2^64
            (p2(-120), p2(10), p2(-120)),                                // quotient 2^-130 denormal, S0 = numerator
            (p2(10), p2(10), p2(-120)),                                  // same, S0 = denominator: flag only
            (p2(-100), p2(-100), p2(30)),                                // exponent gap 130 >= 96, S0 = denominator
        ];
        for (l, (a, b, c)) in lanes.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
            st.v[3][l] = *c;
        }
        execute(Arch::Gfx1100, "v_div_scale_f32", &i, &mut st).unwrap();
        let want = [3f32.to_bits(), p2(-46), p2(-56), p2(10), p2(-36)];
        for l in 0..5 {
            assert_eq!(st.v[0][l], want[l], "lane {l}");
        }
        assert_eq!(st.vcc, 0b11100);
        // Unspecified or ambiguous domains are errors, not guesses.
        for (a, b, c) in [(1f32.to_bits(), 0, 1f32.to_bits()), (1f32.to_bits(), p2(127), 1f32.to_bits()), (1f32.to_bits(), 1f32.to_bits(), 0x7fc0_0000)] {
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = a;
            st.v[2][0] = b;
            st.v[3][0] = c;
            assert!(execute(Arch::Gfx1100, "v_div_scale_f32", &i, &mut st).is_err(), "{a:#x} {b:#x} {c:#x}");
        }
    }

    #[test]
    fn div_fmas_and_fixup_exceptional_cases() {
        let fmas = insn("v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = 3f32.to_bits();
        st.v[3][0] = 1f32.to_bits();
        run("v_div_fmas_f32", &fmas, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 7.0);
        st.vcc = 1;
        st.scc = true;
        run("v_div_fmas_f32", &fmas, &mut st).unwrap();
        // gfx1201 finite VCC: (2 * 3 + 1) scaled by 2**-64 (the exponent of S2 = 1.0 is not above 127).
        assert_eq!(st.v[0][0], (7f32 * f32::from_bits(p2(-64))).to_bits());
        assert_eq!((st.vcc, st.scc, st.exec), (1, true, 1));

        let fix = insn("v_div_fixup_f32", vec![v(0), v(1), v(2), v(3)]);
        let q = (1f32 / 3f32).to_bits();
        let inf = f32::INFINITY.to_bits();
        // (quotient, denominator, numerator) -> result
        let cases: [(u32, u32, u32, u32); 9] = [
            (q, 3f32.to_bits(), 1f32.to_bits(), q),
            (q, (-3f32).to_bits(), 1f32.to_bits(), q | 0x8000_0000),
            (q, 0, 0, 0xffc0_0000),                                  // 0/0
            (q, inf, inf, 0xffc0_0000),                              // inf/inf
            (q, inf, 1f32.to_bits(), 0),                             // x/inf
            (q, inf | 0x8000_0000, 1f32.to_bits(), 0x8000_0000),     // x/-inf
            (q, 0, 1f32.to_bits(), inf),                             // x/0
            (q, 3f32.to_bits(), inf, inf),                           // inf/y
            (q, 3f32.to_bits(), 0x7f80_0001, 0x7fc0_0001),           // NaN numerator is quieted
        ];
        let mut st = state();
        st.exec = (1 << cases.len()) - 1;
        for (l, (a, b, c, _)) in cases.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
            st.v[3][l] = *c;
        }
        run("v_div_fixup_f32", &fix, &mut st).unwrap();
        for (l, (_, _, _, want)) in cases.iter().enumerate() {
            assert_eq!(st.v[0][l], *want, "lane {l}");
        }
        // exponent(S2) - exponent(S1) < -150 selects the ISA's undefined UNDERFLOW_F32: the gfx1201 table measures
        // the signed zero, Gfx1100 keeps the hard error.
        st.exec = 1;
        st.v[2][0] = p2(100);
        st.v[3][0] = p2(-60);
        run("v_div_fixup_f32", &fix, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
        let fix100 = insn_on(Arch::Gfx1100, "v_div_fixup_f32", vec![v(0), v(1), v(2), v(3)]);
        assert!(execute(Arch::Gfx1100, "v_div_fixup_f32", &fix100, &mut st).is_err());
    }

    // BYTE_PERMUTE selectors 8..11 replicate the sign of bytes 1/3/5/7, 12 is 0, >= 13 is 0xff.
    #[test]
    fn perm_b32_selectors() {
        let i = insn("v_perm_b32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 0xaabb_ccdd; // S0: bytes 4..7
        st.v[2][0] = 0x1122_3380; // S1: bytes 0..3
        for (sel, want) in [(0x0c0d_0a09u32, 0x00ff_ff00u32), (0x0b08_0700, 0xff00_aa80), (0x0d0c_0302, 0xff00_1122)] {
            st.v[3][0] = sel;
            run("v_perm_b32", &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], want, "sel {sel:#x}");
        }
    }

    // True16 sources: the half selector rides in the operand and OP_SEL is not applied twice.
    #[test]
    fn cvt_f32_f16_half_selector_neg_and_nan() {
        let mut i = insn("v_cvt_f32_f16_e64", vec![v(0), half(1, Half::Hi)]);
        i.mods.op_sel = 1;
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 0x3c00_4000;
        run("v_cvt_f32_f16_e64", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 1.0);
        i.mods.neg = 1;
        run("v_cvt_f32_f16_e64", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), -1.0);
        st.v[1][0] = 0x7e00_0000;
        assert!(run("v_cvt_f32_f16_e64", &i, &mut st).is_err(), "neg on a NaN half is not measured");
        i.mods.neg = 0;
        run("v_cvt_f32_f16_e64", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0x7fc0_0000, "a NaN half widens quieted (gfx1201 gids 91/92)");
        // v_cvt_f16_f32 only rewrites the selected half.
        let o = insn("v_cvt_f16_f32_e32", vec![half(2, Half::Hi), v(1)]);
        st.v[1][0] = 0.5f32.to_bits();
        st.v[2][0] = 0xdead_beef;
        run("v_cvt_f16_f32_e32", &o, &mut st).unwrap();
        assert_eq!(st.v[2][0], 0x3800_beef);
    }

    #[test]
    fn f32_to_integer_conversions_saturate_and_zero_nan() {
        let vals = [f32::NAN.to_bits(), f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 3e9f32.to_bits(), (-1.9f32).to_bits(), 2.5f32.to_bits()];
        let (si, su) = (insn("v_cvt_i32_f32_e32", vec![v(0), v(1)]), insn("v_cvt_u32_f32_e32", vec![v(2), v(1)]));
        let mut st = state();
        st.exec = 0b111111;
        for (l, x) in vals.iter().enumerate() {
            st.v[1][l] = *x;
        }
        run("v_cvt_i32_f32_e32", &si, &mut st).unwrap();
        run("v_cvt_u32_f32_e32", &su, &mut st).unwrap();
        let want_i = [0, i32::MAX as u32, i32::MIN as u32, i32::MAX as u32, (-1i32) as u32, 2];
        let want_u = [0, u32::MAX, 0, 3_000_000_000, 0, 2];
        for l in 0..6 {
            assert_eq!((st.v[0][l], st.v[2][l]), (want_i[l], want_u[l]), "lane {l}");
        }
    }

    // fma_mix: NEG_HI is |x|, NEG negates after it, OPSEL_HI selects f16, OPSEL the high half.
    #[test]
    fn fma_mixlo_modifiers_and_destination_merge() {
        let mut i = insn("v_fma_mixlo_f16", vec![v(0), v(1), v(2), Operand::Inline(InlineConst::Integer(0))]);
        let mut st = state();
        st.exec = 1;
        st.v[0][0] = 0xdead_beef;
        st.v[1][0] = (-3f32).to_bits();
        st.v[2][0] = 0x3c00_4000; // low half 2.0, high half 1.0
        i.mods.op_sel_hi = 0b010;
        i.mods.neg_hi = 0b001;
        i.mods.neg_lo = 0b001;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_c600, "-|-3| * 2 = -6, high half preserved");
        i.mods.op_sel = 0b010;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_c200, "-|-3| * 1 = -3");
        i.mods.clamp = true;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_0000, "clamp of a negative result is +0");
    }

    #[test]
    fn cvt_pk_f32_fp8_selects_half_and_maps_nan_to_the_default_nan_on_gfx1201_only() {
        let i = insn("v_cvt_pk_f32_fp8_e32", vec![reg(Kind::V, 10, 2), half(4, Half::Hi)]);
        let mut st = state();
        st.exec = 1;
        st.v[4][0] = 0xb838_0000; // high half: 0x38 = 1.0, 0xb8 = -1.0
        run("v_cvt_pk_f32_fp8_e32", &i, &mut st).unwrap();
        assert_eq!((st.v[10][0], st.v[11][0]), (1f32.to_bits(), (-1f32).to_bits()));
        // Both NaN encodings of either element give 0xffc00000 whatever their sign (gfx1201 gids 93/94); 0x3f is
        // the largest-mantissa 1.875.
        for (word, want) in [
            (0x007f_0000u32, (0xffc0_0000, 0)),
            (0x3fff_0000, (0xffc0_0000, 1.875f32.to_bits())),
            (0xffff_0000, (0xffc0_0000, 0xffc0_0000)),
            (0x7f7f_0000, (0xffc0_0000, 0xffc0_0000)),
        ] {
            st.v[4][0] = word;
            run("v_cvt_pk_f32_fp8_e32", &i, &mut st).unwrap();
            assert_eq!((st.v[10][0], st.v[11][0]), want, "{word:#010x}");
        }
        // Any other arch keeps the hard error and writes nothing.
        for arch in [Arch::Gfx1100, Arch::Gfx1151] {
            st.v[4][0] = 0x007f_0000;
            (st.v[10][0], st.v[11][0]) = (7, 8);
            assert!(execute(arch, "v_cvt_pk_f32_fp8_e32", &i, &mut st).is_err(), "{arch:?}");
            assert_eq!((st.v[10][0], st.v[11][0]), (7, 8));
        }
    }

    // Source modifiers reach the class test; NaN compares follow the ISA predicates; inactive lanes are 0.
    #[test]
    fn compares_apply_modifiers_and_zero_inactive_lanes() {
        let mut class = insn("v_cmp_class_f32_e64", vec![reg(Kind::S, 10, 2), v(1), lit(0x200)]);
        let mut st = state();
        st.exec = 0b0111;
        st.v[1][0] = f32::NEG_INFINITY.to_bits();
        st.v[1][1] = f32::INFINITY.to_bits();
        st.v[1][2] = 1f32.to_bits();
        st.v[1][3] = f32::INFINITY.to_bits(); // inactive
        run("v_cmp_class_f32_e64", &class, &mut st).unwrap();
        assert_eq!(st.s[10], 0b010);
        class.mods.abs = 1;
        run("v_cmp_class_f32_e64", &class, &mut st).unwrap();
        assert_eq!(st.s[10], 0b011, "|-inf| is +inf");
        let nan = 0x7fc0_0000;
        for (name, want) in [("v_cmp_neq_f32_e64", 0b01), ("v_cmp_nlt_f32_e64", 0b11), ("v_cmp_lt_f32_e64", 0b00)] {
            let c = insn(name, vec![reg(Kind::S, 12, 2), v(1), v(2)]);
            st.v[1][0] = nan;
            st.v[1][1] = 1f32.to_bits();
            st.v[2][0] = 1f32.to_bits();
            st.v[2][1] = 1f32.to_bits();
            st.exec = 0b11;
            run(name, &c, &mut st).unwrap();
            assert_eq!(st.s[12], want, "{name}");
        }
    }

    // VOP2 fmamk/fmaak keep K in the instruction's literal word: fmamk = S0*K + V1, fmaak = S0*V1 + K.
    #[test]
    fn fmamk_and_fmaak_use_the_literal_constant() {
        let build = |name: &str| {
            let r = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
            Inst::from_parts(Arch::Gfx1201, r.op, r.form, Default::default(), vec![v(0), v(1), v(2)].into_iter().collect(), Default::default(), Some(3f32.to_bits()), Default::default()).unwrap()
        };
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = 1f32.to_bits();
        run("v_fmamk_f32", &build("v_fmamk_f32"), &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 7.0);
        run("v_fmaak_f32", &build("v_fmaak_f32"), &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 5.0);
    }

    #[test]
    fn readfirstlane_forces_lane_zero_when_exec_is_empty() {
        let i = insn("v_readfirstlane_b32", vec![reg(Kind::S, 5, 1), v(0)]);
        let mut st = state();
        st.v[0][0] = 42;
        st.v[0][2] = 9;
        st.exec = 0;
        run("v_readfirstlane_b32", &i, &mut st).unwrap();
        assert_eq!(st.s[5], 42);
        st.exec = 0b100;
        run("v_readfirstlane_b32", &i, &mut st).unwrap();
        assert_eq!(st.s[5], 9);
    }

    /// Architectures whose raw v2 edge tables qualify the shared special-value rules; `Gfx1100` is the unqualified guard.
    const EDGE: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];

    fn table(arch: Arch) -> &'static [crate::isa::OpRow] {
        match arch {
            Arch::Gfx1100 => crate::isa::gfx1100(),
            Arch::Gfx1151 => crate::isa::gfx1151(),
            _ => crate::isa::gfx12(),
        }
    }

    fn insn_on(arch: Arch, name: &str, ops: Vec<Operand>) -> Inst {
        let r = table(arch).iter().find(|r| r.name == name).unwrap_or_else(|| panic!("{name} is not in the {arch:?} table"));
        Inst::from_parts(arch, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap()
    }

    /// One active lane of `name` on `arch`: sources in `v1..`, destination `v0` preset to `dst`.
    fn lane_on(arch: Arch, name: &str, srcs: &[u32], dst: u32, tweak: impl Fn(&mut Inst)) -> std::result::Result<u32, String> {
        // The decode tables lack v_min_f32/v_min3_f32; they share the operand shape of their max twins.
        let row = match name {
            "v_min_f32_e32" => "v_max_f32_e32",
            "v_min3_f32" => "v_max3_f32",
            n => n,
        };
        let mut i = insn_on(arch, row, (0..=srcs.len() as u16).map(v).collect());
        tweak(&mut i);
        let mut st = state();
        st.exec = 1;
        st.v[0][0] = dst;
        for (n, x) in srcs.iter().enumerate() {
            st.v[n + 1][0] = *x;
        }
        execute(arch, name, &i, &mut st).map(|_| st.v[0][0])
    }

    // Measured v_rndne_f32 (gid 23 of both the gfx1151 and gfx1201 sweeps): negative inputs rounding to zero give -0,
    // positive ones +0, normal ties are even. Gfx1100 keeps ties-even and the hard error for the signed zero.
    #[test]
    fn rndne_signed_zero_and_ties_even_on_both_measured_arches() {
        let name = "v_rndne_f32_e32";
        let neg_zero = 0x8000_0000;
        for arch in EDGE {
            let i = insn_on(arch, name, vec![v(0), v(1)]);
            let mut st = state();
            st.exec = 1;
            for (x, want) in [
                ((-0.25f32).to_bits(), neg_zero),
                ((-0.5f32).to_bits(), neg_zero),
                (0x8000_0001, neg_zero),
                (0x8000_0000, neg_zero),
                (0.25f32.to_bits(), 0),
                (0.5f32.to_bits(), 0),
                (0x0000_0001, 0),
                (0, 0),
                (2.5f32.to_bits(), 2.0f32.to_bits()),
                (3.5f32.to_bits(), 4.0f32.to_bits()),
                ((-2.5f32).to_bits(), (-2.0f32).to_bits()),
                ((-1.5f32).to_bits(), (-2.0f32).to_bits()),
            ] {
                st.v[1][0] = x;
                execute(arch, name, &i, &mut st).unwrap();
                assert_eq!(st.v[0][0], want, "{arch:?} rndne({x:#010x})");
            }
        }
        let i = insn_on(Arch::Gfx1100, name, vec![v(0), v(1)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2.5f32.to_bits();
        execute(Arch::Gfx1100, name, &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 2.0f32.to_bits());
        st.v[1][0] = (-0.3f32).to_bits();
        assert!(execute(Arch::Gfx1100, name, &i, &mut st).is_err());
    }

    // Measured default NaN 0xffc00000 for invalid operations of non-NaN operands (gfx1151 and gfx1201 gids 0/2/3);
    // valid arithmetic is unchanged. v_subrev_f32 is in neither sweep: gfx1151 only takes the default NaN for it.
    // Gfx1100 keeps the hard errors.
    #[test]
    fn invalid_operations_give_the_default_nan_on_measured_arches_and_valid_arithmetic_is_exact() {
        let ninf = f32::NEG_INFINITY.to_bits();
        let pinf = f32::INFINITY.to_bits();
        let zero = 0f32.to_bits();
        let nan_default = 0xffc0_0000u32;
        // (opcode, src0, src1, expected) -- invalid: inf-inf (same sign), inf*0 either order, inf+(-inf).
        let cases = [
            ("v_sub_f32_e32", ninf, ninf, nan_default),
            ("v_sub_f32_e32", pinf, pinf, nan_default),
            ("v_add_f32_e32", pinf, ninf, nan_default),
            ("v_mul_f32_e32", ninf, zero, nan_default),
            ("v_mul_f32_e32", zero, pinf, nan_default),
            ("v_sub_f32_e32", 5f32.to_bits(), 2f32.to_bits(), 3f32.to_bits()),
            ("v_sub_f32_e32", pinf, ninf, pinf),
            ("v_mul_f32_e32", ninf, 2f32.to_bits(), ninf),
            ("v_mul_f32_e32", (-0f32).to_bits(), 3f32.to_bits(), (-0f32).to_bits()),
        ];
        for arch in EDGE {
            for (name, a, b, want) in cases {
                assert_eq!(lane_on(arch, name, &[a, b], 0, |_| {}), Ok(want), "{arch:?} {name} {a:#010x} {b:#010x}");
            }
        }
        assert_eq!(lane_on(Arch::Gfx1151, "v_subrev_f32_e32", &[ninf, ninf], 0, |_| {}), Ok(nan_default));
        assert_eq!(lane_on(Arch::Gfx1151, "v_subrev_f32_e32", &[5f32.to_bits(), 2f32.to_bits()], 0, |_| {}), Ok((-3f32).to_bits()));
        assert!(lane_on(Arch::Gfx1201, "v_subrev_f32_e32", &[ninf, ninf], 0, |_| {}).is_err(), "subrev is not in the gfx1201 sweep");
        assert_eq!(lane_on(Arch::Gfx1201, "v_subrev_f32_e32", &[5f32.to_bits(), 2f32.to_bits()], 0, |_| {}), Ok((-3f32).to_bits()));
        for (name, a, b) in [("v_sub_f32_e32", ninf, ninf), ("v_mul_f32_e32", ninf, zero), ("v_add_f32_e32", pinf, ninf)] {
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = a;
            st.v[2][0] = b;
            let i = insn_on(Arch::Gfx1100, name, vec![v(0), v(1), v(2)]);
            assert!(execute(Arch::Gfx1100, name, &i, &mut st).is_err(), "{name} Gfx1100 default NaN is not measured");
            assert_eq!(st.v[0][0], 0, "{name} must not write on error");
        }
    }

    // Sweep gids 0/2/3/8/11/12/13/15 of both arches (all 4 VCC/SCC states): the first NaN operand in source order
    // wins, keeps sign and payload and is quieted; `v_sub_f32` negates S1 first; fmamk orders S0, K, S1; an `inf * 0`
    // product beats a NaN addend; the pure default NaN is 0xffc00000.
    #[test]
    fn nan_operands_select_first_nan_quiet_it_and_keep_sign_and_payload() {
        let (qn, qn2, sn) = (0x7fc1_2345u32, 0xffc0_abcdu32, 0x7f80_0001u32);
        let one = 1f32.to_bits();
        for arch in EDGE {
            let ok = |name: &str, srcs: &[u32], want: u32| assert_eq!(lane_on(arch, name, srcs, 0, |_| {}), Ok(want), "{arch:?} {name} {srcs:x?}");
            ok("v_mul_f32_e32", &[one, 0xffc0_0002], 0xffc0_0002);
            ok("v_mul_f32_e32", &[qn, qn2], qn);
            ok("v_mul_f32_e32", &[sn, qn2], 0x7fc0_0001); // a signalling S0 is quieted, not skipped
            ok("v_mul_f32_e32", &[qn2, sn], qn2); // a quiet S0 beats a signalling S1
            ok("v_mul_f32_e32", &[one, sn], 0x7fc0_0001);
            ok("v_add_f32_e32", &[0x7fc0_0000, one], 0x7fc0_0000);
            ok("v_add_f32_e32", &[one, 0xff80_0002], 0xffc0_0002);
            ok("v_sub_f32_e32", &[one, qn2], 0x7fc0_abcd); // S1 is negated: sign flips
            ok("v_sub_f32_e32", &[one, sn], 0xffc0_0001);
            ok("v_sub_f32_e32", &[qn, qn2], qn);
            ok("v_fma_f32", &[qn, qn2, 0x7fc0_0001], qn);
            ok("v_fma_f32", &[sn, qn2, one], 0x7fc0_0001);
            ok("v_fma_f32", &[one, one, 0xff80_0002], 0xffc0_0002);
            // The product's invalid operation (inf * 0) has priority over a NaN addend; so has inf + (-inf).
            ok("v_fma_f32", &[0, INF, qn], 0xffc0_0000);
            ok("v_fma_f32", &[INF, 0, sn], 0xffc0_0000);
            ok("v_fma_f32", &[one, 0xff80_0000, INF], 0xffc0_0000);
            ok("v_fma_f32", &[one, one, sn], 0x7fc0_0001);
            assert_eq!(lane_on(arch, "v_fmac_f32_e32", &[one, one], sn, |_| {}), Ok(0x7fc0_0001), "{arch:?}: accumulator is the destination");
            // A NaN S0 wins over a NaN K even when K comes first in the encoding.
            let r = table(arch).iter().find(|r| r.name == "v_fmamk_f32").unwrap();
            let i = Inst::from_parts(arch, r.op, r.form, Default::default(), vec![v(0), v(1), v(2)].into_iter().collect(), Default::default(), Some(qn2), Default::default()).unwrap();
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = qn;
            st.v[2][0] = one;
            execute(arch, "v_fmamk_f32", &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], qn, "{arch:?} fmamk: S0 before K");
            st.v[1][0] = one;
            st.v[2][0] = sn;
            execute(arch, "v_fmamk_f32", &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], 0xffc0_abcd, "{arch:?} fmamk: K before S1");
            // The VOPD halves behave like their VOP2 forms: X = v0 = v1 * v2, Y = v1 = v0 + v3.
            let t = table(arch);
            let (x, y) = (t.iter().find(|r| r.name == "v_dual_mul_f32").unwrap(), t.iter().find(|r| r.name == "v_dual_add_f32").unwrap());
            let i = Inst::from_parts(arch, x.op, x.form, FormFields::Vopd { y_op: y.op, x_operands: 3 }, vec![v(0), v(1), v(2), v(1), v(0), v(3)].into_iter().collect(), Default::default(), None, Default::default()).unwrap();
            let mut st = state();
            st.exec = 1;
            (st.v[0][0], st.v[1][0], st.v[2][0], st.v[3][0]) = (one, sn, qn, qn2);
            execute(arch, "v_dual_mul_f32", &i, &mut st).unwrap();
            assert_eq!((st.v[0][0], st.v[1][0]), (0x7fc0_0001, qn2), "{arch:?}");
        }
        // Gfx1100 has no measurement: every NaN operand is a hard error.
        for (name, srcs) in [("v_mul_f32_e32", vec![one, sn]), ("v_add_f32_e32", vec![qn, one]), ("v_fma_f32", vec![one, one, sn])] {
            assert!(lane_on(Arch::Gfx1100, name, &srcs, 0, |_| {}).is_err(), "{name} NaN on Gfx1100 stays unqualified");
        }
    }

    // Source modifiers on NaN operands: `neg` is measured for fma S0 and S2 (gids 13..16), nothing else.
    #[test]
    fn nan_operands_with_source_modifiers_are_qualified_only_where_measured() {
        let (qn, qn2, sn) = (0x7fc1_2345u32, 0xffc0_abcdu32, 0x7f80_0001u32);
        let one = 1f32.to_bits();
        fn neg(bits: u8) -> impl Fn(&mut Inst) {
            move |i| i.mods.neg = bits
        }
        for arch in EDGE {
            assert_eq!(lane_on(arch, "v_fma_f32", &[qn2, one, one], 0, neg(0b001)), Ok(0x7fc0_abcd), "{arch:?} -S0 flips the NaN sign");
            assert_eq!(lane_on(arch, "v_fma_f32", &[one, one, qn], 0, neg(0b100)), Ok(0xffc1_2345), "{arch:?} -S2");
            assert_eq!(lane_on(arch, "v_fma_f32", &[one, one, sn], 0, neg(0b100)), Ok(0xffc0_0001));
            assert_eq!(lane_on(arch, "v_fma_f32", &[0, INF, qn], 0, neg(0b100)), Ok(0xffc0_0000), "invalid product beats -C NaN");
            assert_eq!(lane_on(arch, "v_fma_f32", &[2f32.to_bits(), 3f32.to_bits(), 10f32.to_bits()], 0, neg(0b001)), Ok(4f32.to_bits()), "-2 * 3 + 10");
            assert!(lane_on(arch, "v_fma_f32", &[one, qn, one], 0, neg(0b010)).is_err(), "{arch:?} -S1 on a NaN is not measured");
            assert!(lane_on(arch, "v_fma_f32", &[qn, one, one], 0, |i| i.mods.abs = 0b001).is_err(), "{arch:?} abs on a NaN is not measured");
            assert!(lane_on(arch, "v_mul_f32_e64", &[qn, one], 0, neg(0b001)).is_err(), "{arch:?} -S0 on a mul NaN is not measured");
            assert_eq!(lane_on(arch, "v_mul_f32_e64", &[(-1f32).to_bits(), 2f32.to_bits()], 0, |i| i.mods.abs = 0b001), Ok(2f32.to_bits()), "abs on numbers is unchanged");
        }
    }

    // gfx1151 sweep (gids 34..38, exhaustive): signalling NaN first (quieted), two quiet NaNs give S0, one quiet NaN
    // loses to a number; max3/min3 are ((S0,S1),S2); signed zeros order -0 < +0. The legacy opcodes do not exist on
    // gfx12 (its `_num` forms are covered by `nan_rules_and_signed_zero_in_max_num`) and are an error on Gfx1100.
    #[test]
    fn legacy_max_min_nan_rules_are_gfx1151_only() {
        let (qn, qn2, sn) = (0x7fc1_2345u32, 0xffc0_abcdu32, 0x7f80_0001u32);
        let (one, two) = (1f32.to_bits(), 2f32.to_bits());
        let ok = |name: &str, srcs: &[u32], want: u32| assert_eq!(lane_on(Arch::Gfx1151, name, srcs, 0, |_| {}), Ok(want), "{name} {srcs:x?}");
        ok("v_max_f32_e32", &[sn, one], 0x7fc0_0001);
        ok("v_max_f32_e32", &[0x7fc0_0000, one], one);
        ok("v_max_f32_e32", &[0x7fc0_0000, sn], 0x7fc0_0001);
        ok("v_max_f32_e32", &[one, qn2], one);
        ok("v_max_f32_e32", &[qn, qn2], qn);
        ok("v_min_f32_e32", &[two, sn], 0x7fc0_0001);
        ok("v_max_f32_e32", &[0x8000_0000, 0], 0);
        ok("v_max_f32_e32", &[0, 0x8000_0000], 0);
        ok("v_min_f32_e32", &[0x8000_0000, 0], 0x8000_0000);
        ok("v_min_f32_e32", &[0, 0x8000_0000], 0x8000_0000);
        ok("v_max3_f32", &[one, sn, two], two); // stage 1 yields a quiet NaN, which loses to S2
        ok("v_max3_f32", &[0x7fc0_0001, one, sn], 0x7fc0_0001); // a signalling S2 wins stage 2
        ok("v_min3_f32", &[0x7fc0_0000, one, qn], one);
        assert!(lane_on(Arch::Gfx1151, "v_max_f32_e64", &[qn, one], 0, |i| i.mods.abs = 0b001).is_err(), "abs on a NaN is not measured");
        for (name, srcs) in [("v_max_f32_e32", vec![qn, one]), ("v_max3_f32", vec![one, sn, two])] {
            assert!(lane_on(Arch::Gfx1100, name, &srcs, 0, |_| {}).is_err(), "{name} NaN on Gfx1100 stays unqualified");
        }
    }

    // Sweep gids 23, 28, 89..92 of both arches: NaN inputs keep sign and payload; f32->f16 sets the quiet bit and
    // keeps payload bits 22..13; f16->f32 quiets and widens; everything else is unchanged.
    #[test]
    fn rndne_ldexp_and_f16_conversions_handle_nan_inputs_on_measured_arches() {
        let (qn2, sn) = (0xffc0_abcdu32, 0x7f80_0001u32);
        for arch in EDGE {
            assert_eq!(lane_on(arch, "v_rndne_f32_e32", &[sn], 0, |_| {}), Ok(0x7fc0_0001));
            assert_eq!(lane_on(arch, "v_rndne_f32_e32", &[qn2], 0, |_| {}), Ok(qn2));
            assert_eq!(lane_on(arch, "v_ldexp_f32", &[sn, 3], 0, |_| {}), Ok(0x7fc0_0001));
            assert_eq!(lane_on(arch, "v_ldexp_f32", &[0xff80_0002, 1], 0, |_| {}), Ok(0xffc0_0002));
            assert_eq!(lane_on(arch, "v_ldexp_f32", &[1f32.to_bits(), 3], 0, |_| {}), Ok(8f32.to_bits()));
            // f32 -> f16 into either half; the other half is preserved.
            for (src, h) in [(0x7f80_0001u32, 0x7e00u32), (0xffe5_a5a5, 0xff2d), (0xff80_0001, 0xfe00)] {
                let lo = insn_on(arch, "v_cvt_f16_f32_e32", vec![half(0, Half::Lo), v(1)]);
                let hi = insn_on(arch, "v_cvt_f16_f32_e32", vec![half(0, Half::Hi), v(1)]);
                for (i, want) in [(lo, 0xdead_0000 | h), (hi, (h << 16) | 0xbeef)] {
                    let mut st = state();
                    st.exec = 1;
                    st.v[1][0] = src;
                    st.v[0][0] = 0xdead_beef;
                    execute(arch, "v_cvt_f16_f32_e32", &i, &mut st).unwrap();
                    assert_eq!(st.v[0][0], want, "{arch:?} {src:#x}");
                }
            }
            let lo = insn_on(arch, "v_cvt_f32_f16_e64", vec![v(0), half(1, Half::Lo)]);
            let hi = insn_on(arch, "v_cvt_f32_f16_e64", vec![v(0), half(1, Half::Hi)]);
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = 0x7dff_7c01;
            execute(arch, "v_cvt_f32_f16_e64", &lo, &mut st).unwrap();
            assert_eq!(st.v[0][0], 0x7fc0_2000);
            execute(arch, "v_cvt_f32_f16_e64", &hi, &mut st).unwrap();
            assert_eq!(st.v[0][0], 0x7fff_e000);
            let mut neg = lo;
            neg.mods.neg = 1;
            assert!(execute(arch, "v_cvt_f32_f16_e64", &neg, &mut st).is_err(), "{arch:?} neg on a NaN half is not measured");
        }
        // Gfx1100 has no measurement: the same NaN inputs are hard errors.
        for (name, srcs) in [("v_rndne_f32_e32", vec![sn]), ("v_ldexp_f32", vec![sn, 3])] {
            assert!(lane_on(Arch::Gfx1100, name, &srcs, 0, |_| {}).is_err(), "{name} NaN on Gfx1100 stays unqualified");
        }
        let cvt = insn_on(Arch::Gfx1100, "v_cvt_f16_f32_e32", vec![half(0, Half::Lo), v(1)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = sn;
        assert!(execute(Arch::Gfx1100, "v_cvt_f16_f32_e32", &cvt, &mut st).is_err());
        let cvt = insn_on(Arch::Gfx1100, "v_cvt_f32_f16_e64", vec![v(0), half(1, Half::Lo)]);
        st.v[1][0] = 0x0000_7c01;
        assert!(execute(Arch::Gfx1100, "v_cvt_f32_f16_e64", &cvt, &mut st).is_err());
    }

    // Sweep gids 29/30/31 of both arches (all 262144 tuples x 4 states): D0, the predicate per exponent fields, and
    // the scalar/NULL/VCC destinations. gfx1201 additionally quiets a signalling NaN returned as D (see
    // `div_scale_gfx1201_quiets_every_signalling_nan_gfx1151_returns_untouched`).
    #[test]
    fn div_scale_measured_chain_flags_and_destinations_on_both_arches() {
        let one = 1f32.to_bits();
        // (S0, S1 denominator, S2 numerator) -> (D0, predicate lane bit)
        let rows: [([u32; 3], u32, bool); 17] = [
            ([one, 0, one], 0xffc0_0000, true),                       // zero denominator: default NaN, flag from d = 127
            ([0, 0, 0], 0xffc0_0000, false),                          // 0/0
            ([0x7fc0_0001, one, one], 0x7fc0_0001, false),            // quiet NaN S0 passes through
            ([0x1f80_0000, 0x1f80_0000, 0x7f00_0000], one, true),     // d >= 96, S0 == S1: scaled by 2^64
            ([0x0b80_0000, 0x7180_0000, 0x0b80_0000], 0x2b80_0000, true), // d <= -96: numerator scaled although the quotient is normal
            ([0x7180_0000, 0x7180_0000, 0x0b80_0000], 0x7180_0000, true), // d <= -96, S0 == S1: unscaled
            ([0x7f00_0000, 0x7f00_0000, one], 0x5f00_0000, true),     // e1 = 254: denominator scaled by 2^-64
            ([0x1f80_0000, 0x7f00_0000, 0x7f00_0000], 0x7f80_0000, false), // e1 = 254, S0 exponent <= 63: +inf
            ([one, 1, 0x0080_0000], 0x5f80_0000, false),              // denormal denominator
            ([one, 0x0b80_0000, 0x0b80_0000], 0x5f80_0000, false),    // e2 <= 23
            ([one, one, 0x0b80_0000], one, true),                     // d = -104, S0 == S1
            ([one, one, 0x1500_0000], one, false),                    // d = -85: nothing to do
            ([0x7fc1_2345, 0x7f7f_ffff, 0x7fc1_2345], 0x7fc1_2345, false),
            ([0x8000_0000, 0x7f00_0000, 0x7f00_0000], 0x8000_0000, false), // zero stays zero
            ([0x5f80_0000, 0x7f00_0000, one], 0x5f80_0000, true),     // e1 = 254 and flagged: S0 != S1 is unscaled
            ([1, 0x7f00_0000, 0x7f00_0000], 0x7f80_0000, false),      // denormal S0 * 2^-64 = +inf
            ([0xbf80_0000, 0x7f00_0000, 0x7f00_0000], 0x9f80_0000, false), // sign kept
        ];
        let mask = rows.iter().enumerate().filter(|(_, r)| r.2).fold(0u32, |m, (l, _)| m | 1 << l);
        let load = |st: &mut State| {
            st.exec = 0x1_ffff | 1 << 20;
            for (l, (src, ..)) in rows.iter().enumerate() {
                (st.v[1][l], st.v[2][l], st.v[3][l]) = (src[0], src[1], src[2]);
            }
            st.v[0][20] = 0x1234_5678;
            st.v[1][20] = one;
            (st.v[2][20], st.v[3][20]) = (one, one);
            st.v[0][25] = 0xdead_beef;
        };
        for arch in EDGE {
            // VCC destination: replaced as a whole (inactive lanes cleared), SCC untouched, inactive lane 25 untouched.
            let i = insn_on(arch, "v_div_scale_f32", vec![v(0), Operand::Special(Special::VccLo), v(1), v(2), v(3)]);
            let mut st = state();
            load(&mut st);
            st.exec &= !(1 << 16); // lane 16 inactive: its D0 and VCC bit stay
            st.v[0][16] = 0xcafe_f00d;
            st.vcc = u32::MAX;
            st.scc = true;
            execute(arch, "v_div_scale_f32", &i, &mut st).unwrap();
            for (l, (_, want, _)) in rows.iter().enumerate() {
                assert_eq!(st.v[0][l], if l == 16 { 0xcafe_f00d } else { *want }, "{arch:?} lane {l}");
            }
            assert_eq!(st.v[0][20], one, "lane 20: 1/1 passes S0");
            assert_eq!(st.v[0][25], 0xdead_beef);
            assert_eq!(st.vcc, mask & !(1 << 16), "{arch:?} VCC is overwritten by the predicate, inactive lanes read 0");
            assert!(st.scc, "SCC is untouched");
            // NULL destination: VCC and SCC both preserved.
            let i = insn_on(arch, "v_div_scale_f32", vec![v(0), Operand::Special(Special::Null), v(1), v(2), v(3)]);
            let mut st = state();
            load(&mut st);
            st.vcc = 0xa5a5_5a5a;
            st.scc = true;
            execute(arch, "v_div_scale_f32", &i, &mut st).unwrap();
            assert_eq!(st.v[0][3], one);
            assert_eq!((st.vcc, st.scc), (0xa5a5_5a5a, true));
            // SGPR destination receives the predicate; VCC is untouched.
            let i = insn_on(arch, "v_div_scale_f32", vec![v(0), reg(Kind::S, 10, 1), v(1), v(2), v(3)]);
            let mut st = state();
            load(&mut st);
            st.vcc = 0xa5a5_5a5a;
            execute(arch, "v_div_scale_f32", &i, &mut st).unwrap();
            assert_eq!((st.s[10], st.vcc), (mask, 0xa5a5_5a5a), "{arch:?}");
            // Domains the sweep leaves open stay errors: e1 228..=253, d = -96 with e1 != 254, e2 24..=26, S0 * 2^-64 with
            // exponent field 64..=90.
            for src in [
                [one, 0x7800_0000, one],
                [one, 0x3200_0000, 0x0200_0000],
                [one, 0x0c80_0000, 0x0c80_0000],
                [0x2300_0000, 0x7f00_0000, 0x7f00_0000],
            ] {
                let mut st = state();
                st.exec = 1;
                (st.v[1][0], st.v[2][0], st.v[3][0]) = (src[0], src[1], src[2]);
                st.v[0][0] = 7;
                assert!(execute(arch, "v_div_scale_f32", &i, &mut st).is_err(), "{arch:?} {src:x?}");
                assert_eq!(st.v[0][0], 7, "an error must not write");
            }
            // A NaN operand with abs/neg is not measured.
            let mut st = state();
            st.exec = 1;
            (st.v[1][0], st.v[2][0], st.v[3][0]) = (one, one, 0x7fc0_0000);
            let mut j = i.clone();
            j.mods.neg = 0b100;
            assert!(execute(arch, "v_div_scale_f32", &j, &mut st).is_err(), "{arch:?}");
        }
    }

    // gfx1201 sweep gids 29/30/31: every signalling NaN the chain would return as D (S0 passed through, or "scaled"
    // since scaling leaves a NaN alone) comes out with bit 22 set; gfx1151 returns it untouched. Quiet NaNs and the
    // default NaN of a zero operand are identical on both.
    #[test]
    fn div_scale_gfx1201_quiets_every_signalling_nan_gfx1151_returns_untouched() {
        let one = 1f32.to_bits();
        // (S0, S1, S2), D0 on gfx1151, predicate
        let cases: [([u32; 3], u32, bool); 5] = [
            ([0x7f80_0001, one, one], 0x7f80_0001, false),
            ([0xff80_0002, one, 0x0b80_0000], 0xff80_0002, true), // d <= -96: the numerator "scaling" keeps the NaN
            ([0x7fa0_beef, 1, 0x0080_0000], 0x7fa0_beef, false),  // denormal denominator: S0 * 2^64 keeps the NaN
            ([0xffc0_abcd, one, one], 0xffc0_abcd, false),        // a quiet NaN is unchanged
            ([0x7f80_0001, 0, one], 0xffc0_0000, true),           // zero denominator: default NaN ignores S0
        ];
        for arch in EDGE {
            for ([s0, s1, s2], d1151, flag) in cases {
                let want = if arch == Arch::Gfx1201 && is_snan(d1151) { d1151 | QUIET } else { d1151 };
                let i = insn_on(arch, "v_div_scale_f32", vec![v(0), Operand::Special(Special::VccLo), v(1), v(2), v(3)]);
                let mut st = state();
                st.exec = 1;
                (st.v[1][0], st.v[2][0], st.v[3][0]) = (s0, s1, s2);
                execute(arch, "v_div_scale_f32", &i, &mut st).unwrap();
                assert_eq!((st.v[0][0], st.vcc), (want, u32::from(flag)), "{arch:?} {s0:#x} {s1:#x} {s2:#x}");
            }
        }
    }

    // Sweep gid 33 of both arches: ISA special-case chain plus NaN quotient -> sign|inf and underflow -> signed zero;
    // gid 32 is a plain fma including NaN/Inf rules, with VCC clear or (NaN/Inf operand present) set.
    #[test]
    fn div_fixup_and_div_fmas_special_values_on_measured_arches() {
        let one = 1f32.to_bits();
        let two = 2f32.to_bits();
        for arch in EDGE {
            let ok = |name: &str, srcs: &[u32], want: u32| assert_eq!(lane_on(arch, name, srcs, 0, |_| {}), Ok(want), "{arch:?} {name} {srcs:x?}");
            ok("v_div_fixup_f32", &[0x7fc0_0000, one, one], 0x7f80_0000);
            ok("v_div_fixup_f32", &[0x7fc0_0000, 0xbf80_0000, one], 0xff80_0000);
            ok("v_div_fixup_f32", &[0x7f80_0001, one, one], 0x7f80_0000);
            ok("v_div_fixup_f32", &[one, 0x7f7f_ffff, 0x0080_0000], 0); // exponent(S2) - exponent(S1) < -150
            ok("v_div_fixup_f32", &[one, 0xff7f_ffff, 0x0080_0000], 0x8000_0000);
            ok("v_div_fixup_f32", &[one, 0xffc0_abcd, 0x7f80_0001], 0x7fc0_0001); // S2 first, quieted
            ok("v_div_fixup_f32", &[one, 0xffc0_abcd, one], 0xffc0_abcd);
            ok("v_div_fixup_f32", &[one, 0x7f80_0001, 0x7fc1_2345], 0x7fc1_2345);
            ok("v_div_fixup_f32", &[one, 0, 0], 0xffc0_0000);
            ok("v_div_fixup_f32", &[0x7fc0_0001, 0, one], 0x7f80_0000);
            ok("v_div_fixup_f32", &[one, 0x7f80_0000, 0x7f80_0000], 0xffc0_0000);
            ok("v_div_fmas_f32", &[0x7fc1_2345, one, one], 0x7fc1_2345);
            ok("v_div_fmas_f32", &[0, 0x7f80_0000, 0x7fc1_2345], 0xffc0_0000);
            ok("v_div_fmas_f32", &[0x7f80_0001, 0xffc0_abcd, one], 0x7fc0_0001);
            ok("v_div_fmas_f32", &[one, one, two], 3f32.to_bits());
            // VCC set with a NaN/Inf operand is the plain exceptional result (sweep gid 32 states 2/3 equal state 0/1
            // for all 164808 nonfinite tuples), lane by lane; finite operands under VCC are gfx1201-only.
            let i = insn_on(arch, "v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
            let lanes_in = [
                [0x7f80_0000, 0, 0x7fc1_2345], // inf*0 beats the NaN addend: default NaN
                [one, one, two],               // VCC clear: plain finite fma
                [0xff80_0000, two, 5f32.to_bits()], // -inf product, finite addend
                [one, 0x7f80_0001, 0xffc0_0005], // S1 sNaN quieted, ahead of the addend
            ];
            let mut st = state();
            st.exec = 0b1111;
            st.vcc = 0b1101;
            for (l, [a, b, c]) in lanes_in.into_iter().enumerate() {
                (st.v[1][l], st.v[2][l], st.v[3][l]) = (a, b, c);
            }
            execute(arch, "v_div_fmas_f32", &i, &mut st).unwrap();
            assert_eq!(&st.v[0][..4], &[0xffc0_0000, 3f32.to_bits(), 0xff80_0000, 0x7fc0_0001], "{arch:?}");
            // Finite operands with VCC set: only gfx1201 qualifies them (its lane 1 is 3.0 scaled by 2**64, as
            // the exponent of S2 = 2.0 is above 127); elsewhere the whole instruction fails and nothing is written.
            st.vcc = 0b1111;
            st.v[0][..4].fill(0x1234);
            let big = f32::MAX.to_bits();
            if arch == Arch::Gfx1201 {
                execute(arch, "v_div_fmas_f32", &i, &mut st).unwrap();
                assert_eq!(&st.v[0][..4], &[0xffc0_0000, (3f32 * f32::from_bits(p2(64))).to_bits(), 0xff80_0000, 0x7fc0_0001]);
                // 3 * f32::MAX scaled by 2**64 overflows the single f32 rounding.
                let mut st = state();
                st.exec = 1;
                st.vcc = 1;
                (st.v[1][0], st.v[2][0], st.v[3][0]) = (big, two, big);
                execute(arch, "v_div_fmas_f32", &i, &mut st).unwrap();
                assert_eq!(st.v[0][0], 0x7f80_0000);
            } else {
                let err = execute(arch, "v_div_fmas_f32", &i, &mut st).unwrap_err();
                assert!(err.contains("VCC") && err.contains("finite"), "{err}");
                assert_eq!(&st.v[0][..4], &[0x1234; 4]);
                // Finite operands whose fma would overflow are just as unqualified.
                let mut st = state();
                st.exec = 1;
                st.vcc = 1;
                (st.v[1][0], st.v[2][0], st.v[3][0]) = (big, two, big);
                assert!(execute(arch, "v_div_fmas_f32", &i, &mut st).unwrap_err().contains("VCC"), "{arch:?}");
            }
        }
        // Gfx1100 is not measured: VCC set errors even for NaN/Inf operands.
        let i = insn_on(Arch::Gfx1100, "v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 1;
        st.vcc = 1;
        (st.v[1][0], st.v[2][0], st.v[3][0]) = (0x7f80_0000, one, one);
        assert!(execute(Arch::Gfx1100, "v_div_fmas_f32", &i, &mut st).unwrap_err().contains("VCC"));
        assert_eq!(st.v[0][0], 0);
        // Unmeasured arches keep the documented chain's hard errors.
        assert!(lane_on(Arch::Gfx1100, "v_div_fixup_f32", &[one, 0x7f7f_ffff, 0x0080_0000], 0, |_| {}).is_err());
    }

    // gfx1201 finite VCC (`fmas-gfx1201.raw.tsv`): one F32 rounding of (a * b + c) * 2**(+-64), upscale iff the biased
    // exponent of S2 is above 127. The first tie row separates single rounding from f32-first double rounding.
    #[test]
    fn gfx1201_div_fmas_finite_vcc_rounds_the_scaled_fused_sum_once() {
        let i = insn_on(Arch::Gfx1201, "v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
        let fmas = |vcc: bool, [a, b, c]: [u32; 3]| {
            let mut st = state();
            st.exec = 1;
            st.vcc = u32::from(vcc);
            (st.v[1][0], st.v[2][0], st.v[3][0]) = (a, b, c);
            execute(Arch::Gfx1201, "v_div_fmas_f32", &i, &mut st).unwrap();
            st.v[0][0]
        };
        // Exact sum 2**-86 * (1 + 2**-24) is an f32 tie that rounds to 2**-86 when VCC is clear; scaled by 2**-64 it is
        // just above half the smallest subnormal, so one rounding gives 1 (f32-first would give 0).
        assert_eq!(fmas(false, [0x1480_0000, 0x3f80_0001, 0x8880_0000]), 0x1480_0000);
        assert_eq!(fmas(true, [0x1480_0000, 0x3f80_0001, 0x8880_0000]), 0x0000_0001);
        assert_eq!(fmas(true, [0x9480_0000, 0x3f80_0001, 0x0880_0000]), 0x8000_0001, "signed tie");
        // The product alone overflows f32; the scaled result does not.
        assert_eq!(fmas(false, [0x6780_0000, 0x6780_0000, 0]), 0x7f80_0000);
        assert_eq!(fmas(true, [0x6780_0000, 0x6780_0000, 0]), 0x6f80_0000);
        assert_eq!(fmas(true, [0xe780_0000, 0x6780_0000, 0]), 0xef80_0000);
        // Exponent of S2 selects the scale: 1.0 (biased 127) scales down, 2.0 (128) up, whatever its sign.
        let (one, two) = (1f32.to_bits(), 2f32.to_bits());
        assert_eq!(fmas(true, [one, one, one]), p2(-63));
        assert_eq!(fmas(true, [one, one, two]), (3f32 * f32::from_bits(p2(64))).to_bits());
        assert_eq!(fmas(true, [one, one, (-2f32).to_bits()]), 0xdf80_0000);
        assert_eq!(fmas(true, [one, one, 128f32.to_bits()]), (129f32 * f32::from_bits(p2(64))).to_bits());
        // Exact cancellation stays a positive zero.
        assert_eq!(fmas(true, [one, one, (-1f32).to_bits()]), 0);
    }

    // VCC picks the scaled path per lane; the instruction leaves EXEC, VCC and SCC alone and skips inactive lanes.
    #[test]
    fn gfx1201_div_fmas_vcc_is_lane_selective_and_preserves_flags() {
        let i = insn_on(Arch::Gfx1201, "v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 0b0111;
        st.vcc = 0b1101;
        st.scc = true;
        let tie = [0x1480_0000, 0x3f80_0001, 0x8880_0000];
        let lanes = [tie, tie, [0x6780_0000, 0x6780_0000, 0], tie];
        for (l, [a, b, c]) in lanes.into_iter().enumerate() {
            (st.v[1][l], st.v[2][l], st.v[3][l]) = (a, b, c);
        }
        st.v[0][..4].fill(0x1234);
        execute(Arch::Gfx1201, "v_div_fmas_f32", &i, &mut st).unwrap();
        // lane 0 VCC set, lane 1 VCC clear, lane 2 VCC set, lane 3 VCC set but inactive.
        assert_eq!(&st.v[0][..4], &[0x0000_0001, 0x1480_0000, 0x6f80_0000, 0x1234]);
        assert_eq!((st.exec, st.vcc, st.scc), (0b0111, 0b1101, true));
    }

    // Sweeps gfx1151 gids 93..=101 / gfx1201 gids 95..=103: NaN operands follow the fma rule (a half source widens with
    // its payload before it), half results are rounded once from the exact fused value, `mixlo`/`mixhi` keep the
    // other destination half. The single-rounding tuples are table rows whose double rounding (f32 first) differs.
    #[test]
    fn fma_mix_nan_rules_and_single_rounded_half_results_on_measured_arches() {
        let (one, two, qn, sn) = (1f32.to_bits(), 2f32.to_bits(), 0x7fc1_2345u32, 0x7f80_0001u32);
        for arch in EDGE {
            let mix = |name: &str, srcs: [u32; 3], sel_hi: u8, dst: u32| {
                let mut i = insn_on(arch, name, vec![v(0), v(1), v(2), v(3)]);
                i.mods.op_sel_hi = sel_hi;
                let mut st = state();
                st.exec = 1;
                st.v[0][0] = dst;
                (st.v[1][0], st.v[2][0], st.v[3][0]) = (srcs[0], srcs[1], srcs[2]);
                execute(arch, name, &i, &mut st).map(|_| st.v[0][0])
            };
            assert_eq!(mix("v_fma_mix_f32", [qn, one, one], 0, 0), Ok(qn), "{arch:?}");
            assert_eq!(mix("v_fma_mix_f32", [one, sn, qn], 0, 0), Ok(0x7fc0_0001));
            assert_eq!(mix("v_fma_mix_f32", [0, INF, qn], 0, 0), Ok(0xffc0_0000), "invalid product beats a NaN addend");
            // A half source widens exactly: the f16 signalling NaN 0x7c01 becomes f32 0x7f802000, then is quieted.
            assert_eq!(mix("v_fma_mix_f32", [one, 0x7e00_7c01, one], 0b010, 0), Ok(0x7fc0_2000), "{arch:?}");
            assert!(mix("v_fma_mix_f32", [0x0000_7c01, one, one], 0b001, 0).is_err(), "{arch:?} NaN in a half A source is not swept");
            // Half results: f32 NaN -> sign | 0x7c00 | 0x200 | payload >> 13, the other half is kept.
            assert_eq!(mix("v_fma_mixlo_f16", [qn, one, one], 0, 0xdead_beef), Ok(0xdead_7e09), "{arch:?}");
            assert_eq!(mix("v_fma_mixhi_f16", [qn, one, one], 0, 0xdead_beef), Ok(0x7e09_beef), "{arch:?}");
            assert_eq!(mix("v_fma_mixlo_f16", [INF, two, one], 0, 0xdead_beef), Ok(0xdead_7c00));
            assert_eq!(mix("v_fma_mixlo_f16", [0, INF, qn], 0, 0xdead_beef), Ok(0xdead_fe00), "default NaN as a half");
            // The exact sum is just below the f16 overflow tie 65520: fused rounding gives 65504, f32-first gives inf.
            assert_eq!(mix("v_fma_mixlo_f16", [1, 0x007f_ffff, 0xc77f_f000], 0, 0xdead_beef), Ok(0xdead_fbff), "{arch:?}");
            assert_eq!(mix("v_fma_mixhi_f16", [1, 0x807f_ffff, 0x477f_f000], 0, 0xdead_beef), Ok(0x7bff_beef), "{arch:?}");
            // A product that is an f32 tie only after f32 rounding (f16 source, immediate zero addend).
            let mut i = insn_on(arch, "v_fma_mixlo_f16", vec![v(0), v(1), v(2), Operand::Inline(InlineConst::Integer(0))]);
            i.mods.op_sel_hi = 0b010;
            let mut st = state();
            st.exec = 1;
            st.v[0][0] = 0xdead_beef;
            (st.v[1][0], st.v[2][0]) = (0x3dcc_cccd, 0x7e00_0401);
            execute(arch, "v_fma_mixlo_f16", &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], 0xdead_0067, "{arch:?}");
            // Source modifiers on a NaN operand and clamp of a NaN result are not in the sweep.
            let mut i = insn_on(arch, "v_fma_mix_f32", vec![v(0), v(1), v(2), v(3)]);
            i.mods.neg_lo = 0b001;
            let mut st = state();
            st.exec = 1;
            (st.v[1][0], st.v[2][0], st.v[3][0]) = (qn, one, one);
            assert!(execute(arch, "v_fma_mix_f32", &i, &mut st).is_err(), "{arch:?} -S0 on a NaN");
            let mut i = insn_on(arch, "v_fma_mixlo_f16", vec![v(0), v(1), v(2), v(3)]);
            i.mods.clamp = true;
            assert!(execute(arch, "v_fma_mixlo_f16", &i, &mut st).is_err(), "{arch:?} clamp of a NaN result");
        }
        // Gfx1100 has no measurement: NaN operands (f32 or half sources) are hard errors.
        for (name, srcs, sel_hi) in [("v_fma_mix_f32", [qn, one, one], 0u8), ("v_fma_mix_f32", [one, 0x0000_7c01, one], 0b010)] {
            let mut i = insn_on(Arch::Gfx1100, name, vec![v(0), v(1), v(2), v(3)]);
            i.mods.op_sel_hi = sel_hi;
            let mut st = state();
            st.exec = 1;
            (st.v[1][0], st.v[2][0], st.v[3][0]) = (srcs[0], srcs[1], srcs[2]);
            assert!(execute(Arch::Gfx1100, name, &i, &mut st).is_err(), "{srcs:x?}");
        }
    }

    #[test]
    fn unknown_vector_opcode_errors_even_without_active_lanes() {
        let i = insn("v_add_f32_e32", vec![v(0), v(1), v(2)]);
        let mut st = state();
        st.exec = 0;
        assert!(run("v_madeup_f32_e32", &i, &mut st).is_err());
    }
}
