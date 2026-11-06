// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent projected-read half of the decomposed reference of the fused HC8 read-up entries
//! (`hc8_read_up_e4m3_n{1..4}`, `tests/hc8_read_up_gfx1201.rs`). The WMMA half is the product `set` GEMM entry and
//! the independent `hc8_ref_kw1_k{K}_set` reference (`support/hc8_reference.rs`); this file is the
//! `hyper_read_projected_f32` arithmetic that follows it, in two forms that share nothing with the fused generator:
//!
//! 1. **`emit_projected_reference`**: a separately emitted Builder kernel (one wave, one thread per `(token, h)`)
//!    that reads the projected gate logits `up [rows][4*hidden]` (the `set` GEMM output) and
//!    `normalized [rows][4*hidden]`, and writes `mixed [rows][hidden]`. Its instruction sequence is the DAG of the
//!    gfx1201 `hyper_read_projected_f32` object as disassembled in `hc8-read-anchor.md` (cached object
//!    `tensor_ops.04fd36c457f465b3.hsaco`, kernel @0x82f00), one branch after the other, not the fused kernel's
//!    LDS fold: `value = +0`; per branch `b = 0..3`: `x = rt(up)`, `e = expf(-x)` (`0xbfb8aa3b` / `0xb2a5705f`
//!    range reduction, `v_exp_f32`, `v_ldexp_f32`, unordered `nlt` / `ngt` clamps), `den = 1.0 + e`,
//!    `gate = rt(1.0 / den)` with the full `v_div_scale / v_rcp / v_fma / v_fmac / v_div_scale / v_mul / v_fma /
//!    v_fmac / v_fma / v_div_fmas / v_div_fixup` expansion, `normed = rt(normalized)`, `term = rt(gate * normed)`,
//!    `value += term`; then `mixed = rt(value / 4.0)` with the same expansion (denominator the SGPR constant `4.0`,
//!    as the object divides by `(float)branches`). `rt` is the HIP BF16 round (`+ ((w >> 16) & 1) + 0x7fff` unless the
//!    exponent field is all ones; there a payload-carrying NaN is OR-ed with `0x10000` and Inf is kept; then mask
//!    `0xffff0000`). It runs on the CPU interpreter, so `v_exp_f32`, `v_rcp_f32`, `v_div_scale_f32`,
//!    `v_div_fmas_f32`, `v_div_fixup_f32`, `v_ldexp_f32` are the emulator's models, exactly as for the fused kernel.
//! 2. **`host_read_word`**: a pure-Rust model of the same DAG on one `(token, h)` word using the emulator's raw-bit
//!    `v_exp_f32` and `v_ldexp_f32` models (never the host `exp`) with host IEEE `+ * fma` and a correctly rounded
//!    host division for `1.0 / den` and `value / 4.0`. It returns `None` for every word outside the domain where host
//!    arithmetic and the interpreter are expected to agree (non-finite or huge `x`, subnormal intermediates), so a
//!    pass is a finite-normal-domain cross-check of the DAG constants and fold order, not an exceptional-value
//!    check.
//!
//! LIMITS (honest): there is no equivalence receipt between this transcription and the hipcc object beyond the
//! disassembly it was copied from; the emulator's `v_exp_f32` / `v_rcp_f32` are documented candidate models, not
//! exhaustively hardware-qualified (`peacemaker-ir/src/emu/trans.rs`); the division/NaN domains the interpreter
//! refuses are counted by the test and never reported as a pass. The hardware byte oracle is the receipt.
//!
//! Kernarg (32 B): `up@0 normalized@8 mixed@16` (u64), `hidden@24 rows@28` (u32). Block 32 (one wave: lanes 16..31
//! repeat lanes 0..15 and store the same words), grid `projected_grid(hidden, rows) = (hidden / 16, rows)`.
#![allow(dead_code)]

use peacemaker_isa::insn::{Instruction, MemoryClass};
use peacemaker_isa::kernels::common::{mem as mem_op, op, s, smem, sop, sr, v};
use peacemaker_isa::ledger::Counter;
use peacemaker_isa::plan::Access;
use peacemaker_isa::reg::{Kind, Live};
use peacemaker_isa::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;

pub const PROJECTED_SYMBOL: &str = "hc8_read_up_projected_ref";
pub const PROJECTED_BLOCK: u32 = 32;
pub const PROJECTED_KERNARG_BYTES: usize = 32;

/// `(hidden / 16, rows)`: grid X is the 16-column hidden tile, grid Y the token.
pub fn projected_grid(hidden: u32, rows: u32) -> (u32, u32) {
    (hidden / 16, rows)
}

/// The reference module source: the emitted entry and the assembler text for
/// `peacemaker_isa::native::assemble(&text, Arch::Gfx1201)`.
pub fn projected_module() -> Result<(Emitted, String), String> {
    let e = emit_projected_reference()?;
    let (text, _) = peacemaker_isa::kernels::iu4_gemm::module(
        std::slice::from_ref(&e),
        "hc8_read_up_projected_module",
    )?;
    Ok((e, text))
}

// ------------------------------------------------------------------------------------------------ host model

/// Zero (either sign) or an exponent field in `32..=220`: far from the subnormal, overflow and `v_div_scale` scaling
/// domains, where host IEEE arithmetic and the interpreter's ops are expected to agree bit for bit.
fn safe(w: u32) -> bool {
    w & 0x7fff_ffff == 0 || (32..=220).contains(&((w >> 23) & 0xff))
}

/// HIP `bf16_round` followed by `bf16_to_f32` on raw f32 bits, exactly the object's sequence: `(w + ((w >> 16) & 1) +
/// 0x7fff)` unless the exponent field is all ones; there `low16 != 0 ? w | 0x10000 : w`; then `& 0xffff0000`.
pub fn bf16_rt_bits(w: u32) -> u32 {
    if w & 0x7f80_0000 == 0x7f80_0000 {
        let t = if w & 0xffff != 0 { w | 0x1_0000 } else { w };
        t & 0xffff_0000
    } else {
        w.wrapping_add(((w >> 16) & 1) + 0x7fff) & 0xffff_0000
    }
}

/// Host model of `gate = rt(1.0 / (1.0 + expf(-rt(u))))` on the emulator's `v_exp_f32` / `v_ldexp_f32`; `None`
/// outside the finite normal domain (`|x| > 60`, non-finite, subnormal intermediates).
fn host_gate(u: u32) -> Option<u32> {
    let x = f32::from_bits(bf16_rt_bits(u));
    if !x.is_finite() || x.abs() > 60.0 || (x != 0.0 && x.abs() < 1e-18) {
        return None;
    }
    let c1 = f32::from_bits(0xbfb8_aa3b);
    let c2 = f32::from_bits(0xb2a5_705f);
    let t0 = x * c1;
    let t1 = c1.mul_add(x, -t0);
    let t2 = t0.round_ties_even();
    let t0 = t0 - t2;
    let t1 = c2.mul_add(x, t1);
    let t0 = t0 + t1;
    let n = t2 as i32;
    let e = peacemaker_ir::emu::trans::exp_f32(peacemaker_ir::Arch::Gfx1201, t0.to_bits());
    let e = peacemaker_ir::emu::convert::ldexp_f32(e, n);
    if !safe(e) {
        return None;
    }
    let den = 1.0f32 + f32::from_bits(e);
    if !safe(den.to_bits()) {
        return None;
    }
    let gate = 1.0f32 / den;
    if !safe(gate.to_bits()) {
        return None;
    }
    Some(bf16_rt_bits(gate.to_bits()))
}

/// Host model of one output word `mixed[token][h]` from the four projected logits and the four normalized words of
/// the branches (`b = 0..3`, raw f32 bits), or `None` when the word is outside the domain the model covers.
pub fn host_read_word(ups: [u32; 4], norms: [u32; 4]) -> Option<u32> {
    let mut value = 0.0f32;
    for b in 0..4 {
        let gate = f32::from_bits(host_gate(ups[b])?);
        if (norms[b] >> 23) & 0xff == 0xff {
            return None;
        }
        let normed = f32::from_bits(bf16_rt_bits(norms[b]));
        let prod = gate * normed;
        if !safe(prod.to_bits()) {
            return None;
        }
        let term = f32::from_bits(bf16_rt_bits(prod.to_bits()));
        value += term;
        if !safe(value.to_bits()) {
            return None;
        }
    }
    let q = value / 4.0f32;
    if !safe(q.to_bits()) {
        return None;
    }
    Some(bf16_rt_bits(q.to_bits()))
}

// ------------------------------------------------------------------------------------------------ emitter

const VGPRS: u8 = 24;
const SGPR_END: u8 = 32;

fn plan() -> Result<RegPlan, String> {
    let mut p = RegPlan::new(u16::from(VGPRS), 104)?;
    p.s::<2>("kernarg_ptr", 0, Live::Whole)?;
    for base in (8u8..SGPR_END).step_by(8) {
        p.add_range(&format!("s_blk{base}"), Kind::S, base, 8, Live::Whole)?;
    }
    for base in (0u8..VGPRS).step_by(8) {
        p.add_range(&format!("v_blk{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    Ok(p)
}

// SGPRs.
const S_UP: u8 = 8; // s8:9
const S_NORM: u8 = 10; // s10:11
const S_MIX: u8 = 12; // s12:13
const S_HID: u8 = 14;
const S_ROWS: u8 = 15;
const S_WGX: u8 = 16;
const S_WGY: u8 = 17;
const S_UPB: u8 = 18; // token * 4*hidden * 4
const S_MIXB: u8 = 19; // token * hidden * 4
const S_B1: u8 = 20; // hidden * 4
const S_B2: u8 = 21; // hidden * 8
const S_B3: u8 = 22; // hidden * 12
const S_FOUR: u8 = 23; // 4.0f
const S_H16: u8 = 24; // wgx * 16
const MASK0: u8 = 28; // s28:29
const MASK1: u8 = 30; // s30:31

// VGPRs.
const V_TID: u8 = 0;
const V_H4: u8 = 1; // h * 4
const V_OFF: u8 = 3; // v3..v6: byte offset of branch 0..3 inside up / normalized
const V_MIXOFF: u8 = 7;
const V_SUM: u8 = 8;
const V_X: u8 = 9;
const V_N: u8 = 10;
const V_TERM: u8 = 11;
const W: [u8; 5] = [12, 13, 14, 15, 16];

fn valu(b: &mut Builder, text: String, defs: &[u8], uses: &[u8]) -> R {
    op(
        b,
        text,
        &defs.iter().map(|&n| v(n)).collect::<Vec<_>>(),
        &uses.iter().map(|&n| v(n)).collect::<Vec<_>>(),
    )
}

fn alu_wait(b: &mut Builder) -> R {
    b.push(Instruction::new(
        "s_wait_alu depctr_va_vcc(0)",
        vec![],
        vec![],
    ))
}

/// The object's BF16 round trip of the f32 in `v{val}` on three scratch VGPRs.
fn round_trip(b: &mut Builder, val: u8, t: [u8; 3]) -> R {
    let [t0, t1, t2] = t;
    valu(b, format!("v_bfe_u32 v{t0}, v{val}, 16, 1"), &[t0], &[val])?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{t0}, v{val}, v{t0}"),
        &[t0],
        &[val, t0],
    )?;
    valu(
        b,
        format!("v_add_nc_u32_e32 v{t0}, 0x7fff, v{t0}"),
        &[t0],
        &[t0],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{t1}, 0xffff, v{val}"),
        &[t1],
        &[val],
    )?;
    valu(
        b,
        format!("v_or_b32_e32 v{t2}, 0x10000, v{val}"),
        &[t2],
        &[val],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK0}, 0, v{t1}"),
        &[sr(MASK0, 2)],
        &[v(t1)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t1}, v{t2}, v{val}, s{MASK0}"),
        &[v(t1)],
        &[v(t2), v(val), sr(MASK0, 2)],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{t2}, 0x7f800000, v{val}"),
        &[t2],
        &[val],
    )?;
    op(
        b,
        format!("v_cmp_eq_u32_e64 s{MASK1}, 0x7f800000, v{t2}"),
        &[sr(MASK1, 2)],
        &[v(t2)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{t0}, v{t0}, v{t1}, s{MASK1}"),
        &[v(t0)],
        &[v(t0), v(t1), sr(MASK1, 2)],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{val}, 0xffff0000, v{t0}"),
        &[val],
        &[t0],
    )
}

/// `v{x} = 1.0 + expf(-v{x})`: the object's `exp(-x)` at 0x83060 and the `v_add_f32 v5, 1.0, v5` after it.
fn exp_den(b: &mut Builder, x: u8, t: [u8; 3]) -> R {
    let [ta, tb, tc] = t;
    valu(
        b,
        format!("v_mul_f32_e32 v{ta}, 0xbfb8aa3b, v{x}"),
        &[ta],
        &[x],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{tb}, 0xbfb8aa3b, v{x}, -v{ta}"),
        &[tb],
        &[x, ta],
    )?;
    valu(b, format!("v_rndne_f32_e32 v{tc}, v{ta}"), &[tc], &[ta])?;
    valu(
        b,
        format!("v_sub_f32_e32 v{ta}, v{ta}, v{tc}"),
        &[ta],
        &[ta, tc],
    )?;
    // keep e when !(x > 0x42ce8ed0); +Inf when x < 0xc2b17218 (both unordered-true predicates).
    op(
        b,
        format!("v_cmp_nlt_f32_e64 s{MASK0}, 0x42ce8ed0, v{x}"),
        &[sr(MASK0, 2)],
        &[v(x)],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{tb}, 0xb2a5705f, v{x}"),
        &[tb],
        &[tb, x],
    )?;
    valu(
        b,
        format!("v_add_f32_e32 v{ta}, v{ta}, v{tb}"),
        &[ta],
        &[ta, tb],
    )?;
    valu(b, format!("v_cvt_i32_f32_e32 v{tb}, v{tc}"), &[tb], &[tc])?;
    valu(b, format!("v_exp_f32_e32 v{ta}, v{ta}"), &[ta], &[ta])?;
    valu(
        b,
        format!("v_ldexp_f32 v{ta}, v{ta}, v{tb}"),
        &[ta],
        &[ta, tb],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{ta}, 0, v{ta}, s{MASK0}"),
        &[v(ta)],
        &[v(ta), sr(MASK0, 2)],
    )?;
    op(
        b,
        format!("v_cmp_ngt_f32_e64 s{MASK1}, 0xc2b17218, v{x}"),
        &[sr(MASK1, 2)],
        &[v(x)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e64 v{x}, 0x7f800000, v{ta}, s{MASK1}"),
        &[v(x)],
        &[v(ta), sr(MASK1, 2)],
    )?;
    valu(b, format!("v_add_f32_e32 v{x}, 1.0, v{x}"), &[x], &[x])
}

/// `v{w[0]} = 1.0 / v{x}`: the object's gate division at 0x830d4 (denominator `v{x}`). Result in `w[0]`.
fn recip(b: &mut Builder, x: u8, w: [u8; 5]) -> R {
    let [w0, w1, w2, w3, w4] = w;
    valu(
        b,
        format!("v_div_scale_f32 v{w0}, null, v{x}, v{x}, 1.0"),
        &[w0],
        &[x],
    )?;
    valu(b, format!("v_rcp_f32_e32 v{w1}, v{w0}"), &[w1], &[w0])?;
    valu(
        b,
        format!("v_fma_f32 v{w2}, -v{w0}, v{w1}, 1.0"),
        &[w2],
        &[w0, w1],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w1}, v{w2}, v{w1}"),
        &[w1],
        &[w1, w2],
    )?;
    valu(
        b,
        format!("v_div_scale_f32 v{w2}, vcc_lo, 1.0, v{x}, 1.0"),
        &[w2],
        &[x],
    )?;
    valu(
        b,
        format!("v_mul_f32_e32 v{w3}, v{w2}, v{w1}"),
        &[w3],
        &[w2, w1],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w4}, -v{w0}, v{w3}, v{w2}"),
        &[w4],
        &[w0, w3, w2],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w3}, v{w4}, v{w1}"),
        &[w3],
        &[w3, w4, w1],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w0}, -v{w0}, v{w3}, v{w2}"),
        &[w0],
        &[w0, w3, w2],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_div_fmas_f32 v{w0}, v{w0}, v{w1}, v{w3}"),
        &[w0],
        &[w0, w1, w3],
    )?;
    valu(
        b,
        format!("v_div_fixup_f32 v{w0}, v{w0}, v{x}, 1.0"),
        &[w0],
        &[x],
    )
}

/// `v{w[1]} = v{num} / s{den}`: the object's `value / (float)branches` at 0x832b8 (denominator the SGPR `den`).
fn div_by_sgpr(b: &mut Builder, num: u8, den: u8, w: [u8; 5]) -> R {
    let [w0, w1, w2, w3, w4] = w;
    op(
        b,
        format!("v_div_scale_f32 v{w0}, null, s{den}, s{den}, v{num}"),
        &[v(w0)],
        &[s(den), v(num)],
    )?;
    valu(b, format!("v_rcp_f32_e32 v{w1}, v{w0}"), &[w1], &[w0])?;
    valu(
        b,
        format!("v_fma_f32 v{w2}, -v{w0}, v{w1}, 1.0"),
        &[w2],
        &[w0, w1],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w1}, v{w2}, v{w1}"),
        &[w1],
        &[w1, w2],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{w2}, vcc_lo, v{num}, s{den}, v{num}"),
        &[v(w2)],
        &[v(num), s(den)],
    )?;
    valu(
        b,
        format!("v_mul_f32_e32 v{w3}, v{w2}, v{w1}"),
        &[w3],
        &[w2, w1],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w4}, -v{w0}, v{w3}, v{w2}"),
        &[w4],
        &[w0, w3, w2],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{w3}, v{w4}, v{w1}"),
        &[w3],
        &[w3, w4, w1],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{w0}, -v{w0}, v{w3}, v{w2}"),
        &[w0],
        &[w0, w3, w2],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_div_fmas_f32 v{w0}, v{w0}, v{w1}, v{w3}"),
        &[w0],
        &[w0, w1, w3],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{w1}, v{w0}, s{den}, v{num}"),
        &[v(w1)],
        &[v(w0), s(den), v(num)],
    )
}

fn spec() -> KernelSpec {
    let kernargs = KernargLayout::new(32)
        .pointer_access("up", 0, Access::ReadOnly)
        .pointer_access("normalized", 8, Access::ReadOnly)
        .pointer_access("mixed", 16, Access::WriteOnly)
        .hidden("hidden", 24, 4, "by_value")
        .hidden("rows", 28, 4, "by_value");
    KernelSpec {
        kernel_id: "hc8_read_up_reference".into(),
        variant: "projected".into(),
        arch: Arch::Gfx1201,
        symbol: PROJECTED_SYMBOL.into(),
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: true,
        workgroup_size: PROJECTED_BLOCK as u16,
        group_segment_fixed_size: 0,
        wave32: true,
        cu_mode: false,
    }
}

/// The projected-read reference entry; see the module doc for the ABI, launch and DAG.
pub fn emit_projected_reference() -> Result<Emitted, String> {
    let mut bld = Builder::new(spec(), plan()?);
    bld.enable_delay_alu();
    let b = &mut bld;

    // Kernarg: up, normalized, mixed -> s8..s13; hidden, rows -> s14, s15.
    smem(b, S_UP, 8, 0, 0)?;
    b.wait_all()?;
    sop(b, format!("s_mov_b32 s{S_WGX}, ttmp9"), &[S_WGX], &[])?;
    sop(
        b,
        format!("s_and_b32 s{S_WGY}, ttmp7, 0xffff"),
        &[S_WGY],
        &[],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_UPB}, s{S_WGY}, s{S_HID}"),
        &[S_UPB],
        &[S_WGY, S_HID],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_UPB}, s{S_UPB}, 4"),
        &[S_UPB],
        &[S_UPB],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_MIXB}, s{S_WGY}, s{S_HID}"),
        &[S_MIXB],
        &[S_WGY, S_HID],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_MIXB}, s{S_MIXB}, 2"),
        &[S_MIXB],
        &[S_MIXB],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_B1}, s{S_HID}, 2"),
        &[S_B1],
        &[S_HID],
    )?;
    sop(
        b,
        format!("s_lshl_b32 s{S_B2}, s{S_HID}, 3"),
        &[S_B2],
        &[S_HID],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{S_B3}, s{S_HID}, 12"),
        &[S_B3],
        &[S_HID],
    )?;
    sop(b, format!("s_mov_b32 s{S_FOUR}, 4.0"), &[S_FOUR], &[])?;
    sop(
        b,
        format!("s_lshl_b32 s{S_H16}, s{S_WGX}, 4"),
        &[S_H16],
        &[S_WGX],
    )?;
    // High words of the wave32 masks are never written.
    sop(b, format!("s_mov_b32 s{}, 0", MASK0 + 1), &[MASK0 + 1], &[])?;
    sop(b, format!("s_mov_b32 s{}, 0", MASK1 + 1), &[MASK1 + 1], &[])?;

    // Lane geometry: h = wgx*16 + (lane & 15); lanes 16..31 repeat lanes 0..15.
    valu(
        b,
        format!("v_and_b32_e32 v{V_H4}, 15, v{V_TID}"),
        &[V_H4],
        &[V_TID],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_H4}, s{S_H16}, v{V_H4}"),
        &[v(V_H4)],
        &[s(S_H16), v(V_H4)],
    )?;
    valu(
        b,
        format!("v_lshlrev_b32_e32 v{V_H4}, 2, v{V_H4}"),
        &[V_H4],
        &[V_H4],
    )?;
    // Branch byte offsets inside one token's `[4*hidden]` row of up / normalized, plus the token base.
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_OFF}, s{S_UPB}, v{V_H4}"),
        &[v(V_OFF)],
        &[s(S_UPB), v(V_H4)],
    )?;
    for (br, sg) in [(1u8, S_B1), (2, S_B2), (3, S_B3)] {
        op(
            b,
            format!("v_add_nc_u32_e32 v{}, s{sg}, v{V_OFF}", V_OFF + br),
            &[v(V_OFF + br)],
            &[s(sg), v(V_OFF)],
        )?;
    }
    op(
        b,
        format!("v_add_nc_u32_e32 v{V_MIXOFF}, s{S_MIXB}, v{V_H4}"),
        &[v(V_MIXOFF)],
        &[s(S_MIXB), v(V_H4)],
    )?;
    valu(b, format!("v_mov_b32_e32 v{V_SUM}, 0"), &[V_SUM], &[])?;

    let rt_scratch = [W[1], W[2], W[3]];
    for br in 0..4u8 {
        let off = V_OFF + br;
        mem_op(
            b,
            format!(
                "global_load_b32 v{V_X}, v{off}, s[{S_UP}:{}]",
                S_UP + 1
            ),
            &[v(V_X)],
            &[v(off), sr(S_UP, 2)],
            MemoryClass::VmemLoad,
        )?;
        mem_op(
            b,
            format!(
                "global_load_b32 v{V_N}, v{off}, s[{S_NORM}:{}]",
                S_NORM + 1
            ),
            &[v(V_N)],
            &[v(off), sr(S_NORM, 2)],
            MemoryClass::VmemLoad,
        )?;
        b.wait(Counter::Load, 0)?;
        // gate input rt(up); den = 1 + expf(-x); gate = rt(1 / den).
        round_trip(b, V_X, [W[0], W[1], W[2]])?;
        exp_den(b, V_X, [W[0], W[1], W[2]])?;
        recip(b, V_X, W)?;
        round_trip(b, W[0], rt_scratch)?;
        // normed = rt(normalized); term = rt(gate * normed); value += term.
        round_trip(b, V_N, rt_scratch)?;
        valu(
            b,
            format!("v_mul_f32_e32 v{V_TERM}, v{}, v{V_N}", W[0]),
            &[V_TERM],
            &[W[0], V_N],
        )?;
        round_trip(b, V_TERM, rt_scratch)?;
        valu(
            b,
            format!("v_add_f32_e32 v{V_SUM}, v{V_SUM}, v{V_TERM}"),
            &[V_SUM],
            &[V_SUM, V_TERM],
        )?;
    }
    // mixed = rt(value / 4.0), the quotient lands in W[1]; its round trip uses W[2..4] as scratch.
    div_by_sgpr(b, V_SUM, S_FOUR, W)?;
    round_trip(b, W[1], [W[2], W[3], W[4]])?;
    mem_op(
        b,
        format!(
            "global_store_b32 v{V_MIXOFF}, v{}, s[{S_MIX}:{}]",
            W[1],
            S_MIX + 1
        ),
        &[],
        &[v(V_MIXOFF), v(W[1]), sr(S_MIX, 2)],
        MemoryClass::VmemStore,
    )?;
    b.wait_all()?;
    op(b, "s_endpgm", &[], &[])?;
    bld.finish()
}
