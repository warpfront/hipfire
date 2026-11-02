// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The exact k4 `hc_act` epilogue of the BF16 wide verify GEMV, one item
//! (`m`, `r`) at a time, on wave-uniform scalar inputs.
//!
//! Source of truth: the permanent receipt
//! `kernels/bf16_rows_wide_pm_gfx1151.decode.isa`, symbol
//! `gemv_bf16_xf32_k4` (entry at line 801), the epilogue from the fold
//! `s_add_f32 s0, s0, s1` / `s_add_f32 s1, s4, s5` / `s_add_f32 s0, s0, s1`
//! through the activation to the final BF16 round trip (receipt lines 1202-1332,
//! addresses `0xBE78..0xC104`). Every floating-point operation below is the
//! decoded singleton operation with the decoded constants and operand order;
//! only the bit-exact BF16 rounding selects are restructured (see below).
//! This immutable decode snapshot and its `.decode.json` / `.identity.json`
//! retain the historical singleton object's identity; they do not certify the
//! current same-stem rowtile image. Its emission evidence lives in the current
//! `.proof.json`, `.m7.json` and `.shape.d` artifacts.
//!
//! # Contract
//!
//! `out = (q0 + q1) + (q2 + q3)` (`s_add_f32`, the decoded fold order). When
//! `act == 0` that is the result. Otherwise, with `rt` the BF16 round trip:
//!
//! ```text
//! x = rt(sum);  x = rt(scale * x);  y = x / (1 + exp(-x))  (decoded expansion);  out = rt(y)
//! ```
//!
//! `rt(u)`: finite `u` (exponent field != 0xff): `(u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000`;
//! Inf/NaN (exponent field 0xff): `low16(u) == 0 ? u : u | 0x10000`, then
//! `& 0xffff0000`. That is the decoded integer sequence: a pure Inf passes
//! through, a NaN keeps its payload with bit 16 forced when only the low half
//! carried it. No other NaN quieting happens anywhere.
//!
//! The decoded `1/(1+e)` division is the full `v_div_scale` / `v_rcp` /
//! FMA-refine / `v_div_fmas` / `v_div_fixup` sequence, and `exp(-x)` is the
//! decoded range reduction (`-x * log2(e)` split into `rndne` + hi/lo FMA
//! remainder, `v_exp_f32`, `v_ldexp_f32`, then the `x > 0x42ce8ed0` -> +Inf and
//! `x < 0xc2b17218` -> 0 `v_cndmask` clamps).
//!
//! Inputs and outputs are SGPRs. The result is in `out`: an f32 whose low half is
//! zero (the BF16 value) when `act != 0`, the raw fold otherwise. VGPR
//! `v_tmp[0]` holds the unrounded quotient afterwards; every other scratch is
//! dead.
//!
//! # Deviations from the decoded stream (bit-identical results)
//!
//!  * The compiler's structured `if` around each BF16 round trip is one
//!    `s_and_b32 exec` / `s_cselect` / `s_cmp` / `s_cbranch` dance on a uniform
//!    condition (`exec_lo != 0`). Here each round trip computes the finite and
//!    the Inf/NaN result and `s_cselect`s between them, so no EXEC arithmetic and
//!    no extra labels are needed, and the wave's EXEC is neither read nor changed.
//!    All VALU inputs below are wave-uniform (SGPR-derived), so any non-empty EXEC
//!    produces the decoded lane-0 value, and `v_readfirstlane_b32` reads it.
//!  * The decoded final exponent test (`v_and`/`v_cmp_eq_u32`/`s_cbranch_vccnz`
//!    on the quotient) is the same SALU select on the read-back scalar.
//!  * `s_delay_alu` is a scheduling hint (the hardware interlocks every SALU and
//!    VALU dependency here): the decoded SALU hints are kept where the SALU
//!    sequence is unchanged; the VALU/TRANS hints are inserted by the Builder
//!    (`enable_delay_alu`) before each dependent VALU, so none are written by
//!    hand and no two hints are ever adjacent.
//!
//! gfx1151 only: the scalar FP ops (`s_add_f32`, `s_mul_f32`, `s_sub_f32`,
//! `s_rndne_f32`, `s_fmamk_f32`, `s_cmp_{ngt,nlt}_f32`, `s_cvt_i32_f32`) are
//! RDNA3.5 (`peacemaker-ir/isa/gfx1151.tbl`); the VALU ops are in the gfx1100
//! table that gfx1151 inherits.
use super::common::{op, s, s_add_i32, sop, v};
use crate::{Arch, Builder};

type R = Result<(), String>;

/// Registers of one [`emit_value`] call (all numbers are plain register
/// indices; every one must be declared and live in the builder's plan).
#[derive(Clone, Copy, Debug)]
pub(super) struct Regs {
    /// S: f32 bits of q0..q3, read-only.
    pub q: [u8; 4],
    /// S: f32 bits of the activation input scale, read-only.
    pub scale: u8,
    /// S: `hc_act`; non-zero means activate, read-only.
    pub act: u8,
    /// S: result bits (see the module contract).
    pub out: u8,
    /// S scratch; the first three are used.
    pub s_tmp: [u8; 11],
    /// V scratch; the first six are used.
    pub v_tmp: [u8; 16],
}

/// A VALU instruction over VGPR defs/uses plus scalar (SGPR) sources.
fn vop(b: &mut Builder, text: String, vdefs: &[u8], vuses: &[u8], suses: &[u8]) -> R {
    let defs: Vec<_> = vdefs.iter().map(|&n| v(n)).collect();
    let uses: Vec<_> = vuses
        .iter()
        .map(|&n| v(n))
        .chain(suses.iter().map(|&n| s(n)))
        .collect();
    op(b, text, &defs, &uses)
}

/// One decoded SALU scheduling hint (never a result-affecting instruction).
fn hint(b: &mut Builder, text: &str) -> R {
    op(b, format!("s_delay_alu {text}"), &[], &[])
}

/// In-place scalar BF16 round trip of `val` (see the module contract) on three
/// scratch SGPRs `t`. `lead` is an optional hint for the first instruction.
fn round_trip(b: &mut Builder, val: u8, t: [u8; 3], lead: Option<&str>) -> R {
    let [t0, t1, t2] = t;
    let add = s_add_i32(b.spec.arch);
    if let Some(lead) = lead {
        hint(b, lead)?;
    }
    // Finite: u + ((u >> 16) & 1) + 0x7fff.
    sop(
        b,
        format!("s_bfe_u32 s{t0}, s{val}, 0x10010"),
        &[t0],
        &[val],
    )?;
    hint(b, "instid0(SALU_CYCLE_1)")?;
    sop(b, format!("{add} s{t0}, s{val}, s{t0}"), &[t0], &[val, t0])?;
    sop(b, format!("s_addk_i32 s{t0}, 0x7fff"), &[t0], &[t0])?;
    // Inf/NaN: low16 == 0 ? u : u | 0x10000.
    sop(b, format!("s_and_b32 s{t1}, s{val}, 0xffff"), &[t1], &[val])?;
    sop(b, format!("s_or_b32 s{t2}, s{val}, 0x10000"), &[t2], &[val])?;
    sop(b, format!("s_cmp_eq_u32 s{t1}, 0"), &[], &[t1])?;
    sop(
        b,
        format!("s_cselect_b32 s{t1}, s{val}, s{t2}"),
        &[t1],
        &[val, t2],
    )?;
    // Exponent field all ones selects the Inf/NaN result.
    sop(
        b,
        format!("s_and_b32 s{t2}, s{val}, 0x7f800000"),
        &[t2],
        &[val],
    )?;
    sop(b, format!("s_cmp_eq_u32 s{t2}, 0x7f800000"), &[], &[t2])?;
    sop(
        b,
        format!("s_cselect_b32 s{t0}, s{t1}, s{t0}"),
        &[t0],
        &[t1, t0],
    )?;
    hint(b, "instid0(SALU_CYCLE_1)")?;
    sop(
        b,
        format!("s_and_b32 s{val}, s{t0}, 0xffff0000"),
        &[val],
        &[t0],
    )
}

/// The exact k4 `hc_act` epilogue for one item. `label` is a unique prefix
/// (the branch target `{label}_done` is placed here). Clobbers only
/// `r.out`, `r.s_tmp[..3]`, `r.v_tmp[..6]`, VCC and SCC; EXEC is untouched.
pub(super) fn emit_value(b: &mut Builder, label: &str, r: &Regs) -> R {
    if b.spec.arch != Arch::Gfx1151 {
        return Err("qwen4_bf16_hc_epilogue: scalar FP epilogue is gfx1151 only".into());
    }
    let out = r.out;
    let t = [r.s_tmp[0], r.s_tmp[1], r.s_tmp[2]];
    let w: [u8; 6] = [
        r.v_tmp[0], r.v_tmp[1], r.v_tmp[2], r.v_tmp[3], r.v_tmp[4], r.v_tmp[5],
    ];
    let used_s: Vec<u8> =
        r.q.iter()
            .copied()
            .chain([r.scale, r.act, out])
            .chain(t)
            .collect();
    for (i, a) in used_s.iter().enumerate() {
        if used_s[..i].contains(a) {
            return Err(format!("qwen4_bf16_hc_epilogue: SGPR s{a} used twice"));
        }
    }
    for (i, a) in w.iter().enumerate() {
        if w[..i].contains(a) {
            return Err(format!("qwen4_bf16_hc_epilogue: VGPR v{a} used twice"));
        }
    }
    let [t0, t1, t2] = t;
    let [w0, w1, w2, w3, w4, w5] = w;
    let [q0, q1, q2, q3] = r.q;

    // Fold: (q0 + q1) + (q2 + q3), receipt 0xBE88..0xBE94.
    sop(
        b,
        format!("s_add_f32 s{t0}, s{q0}, s{q1}"),
        &[t0],
        &[q0, q1],
    )?;
    sop(
        b,
        format!("s_add_f32 s{t1}, s{q2}, s{q3}"),
        &[t1],
        &[q2, q3],
    )?;
    hint(b, "instid0(SALU_CYCLE_3)")?;
    sop(
        b,
        format!("s_add_f32 s{out}, s{t0}, s{t1}"),
        &[out],
        &[t0, t1],
    )?;
    // hc_act == 0: the raw fold is the result.
    let done = format!("{label}_done");
    sop(b, format!("s_cmp_eq_u32 s{}, 0", r.act), &[], &[r.act])?;
    op(b, format!("s_cbranch_scc1 {done}"), &[], &[])?;

    // x = rt(sum); x = rt(scale * x)   (0xBEA0..0xBF98).
    round_trip(b, out, t, Some("instid0(SALU_CYCLE_3)"))?;
    sop(
        b,
        format!("s_mul_f32 s{out}, s{}, s{out}", r.scale),
        &[out],
        &[r.scale, out],
    )?;
    round_trip(b, out, t, Some("instid0(SALU_CYCLE_3)"))?;

    // exp(-x): decoded range reduction (0xBFA0..0xC010).
    sop(
        b,
        format!("s_mul_f32 s{t0}, s{out}, 0xbfb8aa3b"),
        &[t0],
        &[out],
    )?;
    hint(
        b,
        "instid0(SALU_CYCLE_3) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)",
    )?;
    sop(
        b,
        format!("s_xor_b32 s{t1}, s{t0}, 0x80000000"),
        &[t1],
        &[t0],
    )?;
    sop(b, format!("s_rndne_f32 s{t2}, s{t0}"), &[t2], &[t0])?;
    sop(
        b,
        format!("s_fmamk_f32 s{t1}, s{out}, 0xbfb8aa3b, s{t1}"),
        &[t1],
        &[out, t1],
    )?;
    sop(b, format!("s_cmp_ngt_f32 s{out}, 0x42ce8ed0"), &[], &[out])?;
    sop(
        b,
        format!("s_sub_f32 s{t0}, s{t0}, s{t2}"),
        &[t0],
        &[t0, t2],
    )?;
    hint(
        b,
        "instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)",
    )?;
    sop(
        b,
        format!("s_fmamk_f32 s{t1}, s{out}, 0xb2a5705f, s{t1}"),
        &[t1],
        &[out, t1],
    )?;
    // SCC of the `ngt` compare, captured before the `nlt` compare overwrites it.
    sop(b, "s_cselect_b32 vcc_lo, -1, 0".to_string(), &[], &[])?;
    sop(b, format!("s_cmp_nlt_f32 s{out}, 0xc2b17218"), &[], &[out])?;
    sop(
        b,
        format!("s_add_f32 s{t0}, s{t0}, s{t1}"),
        &[t0],
        &[t0, t1],
    )?;
    hint(b, "instid0(SALU_CYCLE_3)")?;
    vop(b, format!("v_exp_f32_e32 v{w0}, s{t0}"), &[w0], &[], &[t0])?;
    sop(b, format!("s_cvt_i32_f32 s{t0}, s{t2}"), &[t0], &[t2])?;
    vop(
        b,
        format!("v_ldexp_f32 v{w0}, v{w0}, s{t0}"),
        &[w0],
        &[w0],
        &[t0],
    )?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{w0}, 0, v{w0}, vcc_lo"),
        &[w0],
        &[w0],
        &[],
    )?;
    sop(b, "s_cselect_b32 vcc_lo, -1, 0".to_string(), &[], &[])?;
    vop(
        b,
        format!("v_cndmask_b32_e32 v{w0}, 0x7f800000, v{w0}, vcc_lo"),
        &[w0],
        &[w0],
        &[],
    )?;

    // y = x / (1 + exp(-x)): full division expansion (0xC01C..0xC07C).
    vop(
        b,
        format!("v_add_f32_e32 v{w0}, 1.0, v{w0}"),
        &[w0],
        &[w0],
        &[],
    )?;
    vop(
        b,
        format!("v_div_scale_f32 v{w1}, null, v{w0}, v{w0}, s{out}"),
        &[w1],
        &[w0],
        &[out],
    )?;
    vop(b, format!("v_rcp_f32_e32 v{w2}, v{w1}"), &[w2], &[w1], &[])?;
    vop(
        b,
        format!("v_xor_b32_e32 v{w1}, 0x80000000, v{w1}"),
        &[w1],
        &[w1],
        &[],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{w3}, v{w1}, v{w2}, 1.0"),
        &[w3],
        &[w1, w2],
        &[],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{w2}, v{w3}, v{w2}"),
        &[w2],
        &[w2, w3],
        &[],
    )?;
    vop(
        b,
        format!("v_div_scale_f32 v{w3}, vcc_lo, s{out}, v{w0}, s{out}"),
        &[w3],
        &[w0],
        &[out],
    )?;
    vop(
        b,
        format!("v_mul_f32_e32 v{w4}, v{w3}, v{w2}"),
        &[w4],
        &[w3, w2],
        &[],
    )?;
    vop(
        b,
        format!("v_fma_f32 v{w5}, v{w1}, v{w4}, v{w3}"),
        &[w5],
        &[w1, w4, w3],
        &[],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{w4}, v{w5}, v{w2}"),
        &[w4],
        &[w4, w5, w2],
        &[],
    )?;
    vop(
        b,
        format!("v_fmac_f32_e32 v{w3}, v{w1}, v{w4}"),
        &[w3],
        &[w3, w1, w4],
        &[],
    )?;
    vop(
        b,
        format!("v_div_fmas_f32 v{w1}, v{w3}, v{w2}, v{w4}"),
        &[w1],
        &[w3, w2, w4],
        &[],
    )?;
    vop(
        b,
        format!("v_div_fixup_f32 v{w0}, v{w1}, v{w0}, s{out}"),
        &[w0],
        &[w1, w0],
        &[out],
    )?;

    // out = rt(y): read the quotient back, final BF16 round trip.
    op(
        b,
        format!("v_readfirstlane_b32 s{out}, v{w0}"),
        &[s(out)],
        &[v(w0)],
    )?;
    round_trip(b, out, t, None)?;
    b.label(&done)
}
