// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Fused HC8 read-up: `hc8_read_up_e4m3_n{1..4}`, in module `hc8_wmma_pm_gfx1201`.
//!
//! It is `hc8_gemm_e4m3_mw4_kw1_n{NT}_set` (same decoder, K loop, register plan) with
//! the projected `[rows, 4*hidden]` activations never leaving the chip: the epilogue
//! is the product `hyper_read_projected_f32` (branches = 4), bit for bit, rows 1..=64.
//!
//! # ABI (56 B)
//! `q@0 s@8 low@16 normalized@24 mixed@32` (u64), `hidden@40 low_rank@44 rows@48`
//! (u32), `pad@52`. Block 128 (4 waves), grid `hidden/16`. Wave `b` owns M tile
//! `b*(hidden/16) + wg` (so `M = 4*hidden`, branch `b`, hidden columns
//! `wg*16..wg*16+16`) over the full K (`K = low_rank`, KW = 1). Inputs: `low` f32
//! `[rows, K]` (BF16-exact), `normalized` f32 `[rows, 4*hidden]`; output f32
//! `[rows, hidden]`. `rows` outside `1..=16*NT` returns before any barrier or store.
//!
//! # Epilogue DAG per `(token, h)` (the observed `hyper_read_projected_f32` object)
//! Wave `b`: `u = acc * s[m]` (one `v_mul_f32`); `x = rt(u)`; `e = expf(-x)` (range
//! reduction with `0xbfb8aa3b`/`0xb2a5705f`, `v_exp_f32`, `v_ldexp_f32`, unordered
//! `nlt`/`ngt` clamps); `den = 1.0 + e`; `gate = rt(1.0 / den)` with the full
//! `v_div_scale/v_rcp/v_fma/v_div_fmas/v_div_fixup` sequence; `normed =
//! rt(normalized[token*4*hidden + b*hidden + h])`; `term_b = rt(gate * normed)`. The
//! four terms fold through LDS: `value = +0; value += term_0; ...; value += term_3`
//! (`v_add_f32`, ascending `b`), `mixed = rt(value / 4.0)` with the full division
//! sequence (`v_div_scale/v_rcp/.../v_div_fixup`, denominator `4.0`). `rt` is the HIP
//! BF16 round: round-to-nearest-even bias unless the exponent is all ones, in which
//! case a payload-carrying NaN is OR-ed with `0x10000` and Inf is kept; then mask
//! `0xffff0000`.
//!
//! LDS holds the four waves' terms at `((wave*NR + j)*32 + lane)*4`, `NR = 8*NT`;
//! wave `w` folds and stores the accumulator indices `j` with `j % 4 == w`.

use super::{
    exec_all, exec_mask, plan, prologue, vload128, zero_regs, Fmt, Geo, Lay, ACC, KARG, LANE,
    LB, LR, SM, SO, SP, TMP, TMP2, VT, VT2, WAVE, WG, YA, YP, k_loops, branch,
};
use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{add64, lit, op, s, smem, sop, v};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec};

type R = Result<(), String>;

/// Workgroup threads: four waves, one per HC branch.
pub const READ_UP_BLOCK: u32 = 128;
/// Kernarg segment bytes.
pub const READ_UP_KERNARG_BYTES: u32 = 56;

/// Normalized-base SGPR pair (`normalized + (b*hidden + wg*16)*4`).
const NP: u8 = 24;
/// Output pointer SGPR pair.
const OP: u8 = 46;
const BRANCHES: u32 = 4;

/// Symbol of the `nt`-column-tile export.
pub fn read_up_symbol(nt: u32) -> String {
    format!("hc8_read_up_e4m3_n{nt}")
}

fn geo(nt: u32) -> Geo {
    Geo {
        fmt: Fmt::E4M3,
        mw: BRANCHES,
        kw: 1,
        nt,
        act: false,
    }
}

fn lds_bytes(nt: u32) -> u32 {
    BRANCHES * 8 * nt * 128
}

fn valu(b: &mut Builder, text: String, defs: &[u8], uses: &[u8]) -> R {
    b.push(Instruction::new(
        text,
        defs.iter().map(|&r| v(r)).collect(),
        uses.iter().map(|&r| v(r)).collect(),
    ))
}

fn alu_wait(b: &mut Builder) -> R {
    b.push(Instruction::new(
        "s_wait_alu depctr_va_vcc(0)",
        vec![],
        vec![],
    ))
}

/// HIP `bf16_to_f32(bf16_round(x))` in place on `val`; `t` scratch.
fn round_trip(b: &mut Builder, val: u8, t: [u8; 3]) -> R {
    let [t0, t1, t2] = t;
    valu(
        b,
        format!("v_bfe_u32 v{t0}, v{val}, 16, 1"),
        &[t0],
        &[val],
    )?;
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
    valu(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{t1}"),
        &[],
        &[t1],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_cndmask_b32_e32 v{t1}, v{t2}, v{val}, vcc_lo"),
        &[t1],
        &[t2, val],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{t2}, 0x7f800000, v{val}"),
        &[t2],
        &[val],
    )?;
    valu(
        b,
        format!("v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v{t2}"),
        &[],
        &[t2],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_cndmask_b32_e32 v{t0}, v{t0}, v{t1}, vcc_lo"),
        &[t0],
        &[t0, t1],
    )?;
    valu(
        b,
        format!("v_and_b32_e32 v{val}, 0xffff0000, v{t0}"),
        &[val],
        &[t0],
    )
}

/// Full IEEE division `num / den` exactly as compiled in the product object. Each
/// operand is an inline constant (`Err(text)`) or a VGPR (`Ok(reg)`). Result in
/// `w[0]`; clobbers `w[1..6]` (an operand VGPR may be `w[0]`).
fn divide(b: &mut Builder, num: Result<u8, &str>, den: Result<u8, &str>, w: [u8; 6]) -> R {
    let [w0, w1, w2, w3, w4, w5] = w;
    let text = |o: Result<u8, &str>| match o {
        Ok(r) => format!("v{r}"),
        Err(c) => c.to_string(),
    };
    let (n, d) = (text(num), text(den));
    let ext: Vec<u8> = {
        let mut u: Vec<u8> = [den.ok(), num.ok()].into_iter().flatten().collect();
        u.dedup();
        u
    };
    let with = |extra: &[u8]| -> Vec<u8> {
        let mut u = ext.clone();
        u.extend_from_slice(extra);
        u
    };
    valu(b, format!("v_div_scale_f32 v{w1}, null, {d}, {d}, {n}"), &[w1], &ext)?;
    valu(b, format!("v_rcp_f32_e32 v{w2}, v{w1}"), &[w2], &[w1])?;
    valu(b, format!("v_fma_f32 v{w3}, -v{w1}, v{w2}, 1.0"), &[w3], &[w1, w2])?;
    valu(b, format!("v_fmac_f32_e32 v{w2}, v{w3}, v{w2}"), &[w2], &[w2, w3])?;
    valu(b, format!("v_div_scale_f32 v{w3}, vcc_lo, {n}, {d}, {n}"), &[w3], &ext)?;
    valu(b, format!("v_mul_f32_e32 v{w4}, v{w3}, v{w2}"), &[w4], &[w3, w2])?;
    valu(b, format!("v_fma_f32 v{w5}, -v{w1}, v{w4}, v{w3}"), &[w5], &[w1, w4, w3])?;
    valu(b, format!("v_fmac_f32_e32 v{w4}, v{w5}, v{w2}"), &[w4], &[w4, w5, w2])?;
    valu(b, format!("v_fma_f32 v{w1}, -v{w1}, v{w4}, v{w3}"), &[w1], &[w1, w4, w3])?;
    alu_wait(b)?;
    valu(b, format!("v_div_fmas_f32 v{w1}, v{w1}, v{w2}, v{w4}"), &[w1], &[w1, w2, w4])?;
    valu(b, format!("v_div_fixup_f32 v{w0}, v{w1}, {d}, {n}"), &[w0], &with(&[w1]))
}

/// `gate_pre = 1.0 / (1.0 + expf(-x))` (not yet BF16-rounded) in `t0`; `x` preserved.
fn sigmoid(b: &mut Builder, x: u8, t: [u8; 6]) -> R {
    let [t0, t1, t2, t3, ..] = t;
    valu(
        b,
        format!("v_mul_f32_e32 v{t0}, 0xbfb8aa3b, v{x}"),
        &[t0],
        &[x],
    )?;
    valu(
        b,
        format!("v_fma_f32 v{t1}, 0xbfb8aa3b, v{x}, -v{t0}"),
        &[t1],
        &[x, t0],
    )?;
    valu(
        b,
        format!("v_rndne_f32_e32 v{t2}, v{t0}"),
        &[t2],
        &[t0],
    )?;
    valu(
        b,
        format!("v_sub_f32_e32 v{t0}, v{t0}, v{t2}"),
        &[t0],
        &[t0, t2],
    )?;
    valu(
        b,
        format!("v_fmac_f32_e32 v{t1}, 0xb2a5705f, v{x}"),
        &[t1],
        &[t1, x],
    )?;
    valu(
        b,
        format!("v_add_f32_e32 v{t0}, v{t0}, v{t1}"),
        &[t0],
        &[t0, t1],
    )?;
    valu(
        b,
        format!("v_cvt_i32_f32_e32 v{t3}, v{t2}"),
        &[t3],
        &[t2],
    )?;
    valu(b, format!("v_exp_f32_e32 v{t0}, v{t0}"), &[t0], &[t0])?;
    valu(
        b,
        format!("v_ldexp_f32 v{t0}, v{t0}, v{t3}"),
        &[t0],
        &[t0, t3],
    )?;
    valu(
        b,
        format!("v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v{x}"),
        &[],
        &[x],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_cndmask_b32_e32 v{t0}, 0, v{t0}, vcc_lo"),
        &[t0],
        &[t0],
    )?;
    valu(
        b,
        format!("v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v{x}"),
        &[],
        &[x],
    )?;
    alu_wait(b)?;
    valu(
        b,
        format!("v_cndmask_b32_e32 v{t0}, 0x7f800000, v{t0}, vcc_lo"),
        &[t0],
        &[t0],
    )?;
    valu(
        b,
        format!("v_add_f32_e32 v{t0}, 1.0, v{t0}"),
        &[t0],
        &[t0],
    )?;
    // t0 = 1.0 / t0.
    divide(b, Err("1.0"), Ok(t0), t)
}

fn store_out(b: &mut Builder, c: u32, r: u32, val: u8) -> R {
    exec_mask(b, c)?;
    b.push(
        Instruction::new(
            format!(
                "global_store_b32 v{}, v{val}, s[{OP}:{}] offset:{}",
                YA + c as u8,
                OP + 1,
                4 * r
            ),
            vec![],
            vec![v(YA + c as u8), v(val), s(OP), s(OP + 1)],
        )
        .memory(MemoryClass::VmemStore),
    )?;
    exec_all(b)?;
    b.release_store_sources()
}

fn epilogue(b: &mut Builder, g: &Geo, l: &Lay, slot: usize, end: &str) -> R {
    let nt = g.nt;
    let nr = 8 * nt;
    let sv = l.xb;
    let nrm = sv + 8;
    let t0 = nrm + nr as u8;
    let t: [u8; 6] = std::array::from_fn(|i| t0 + i as u8);
    let val = t0 + 6;
    let p = t0 + 7;
    let noff = p + 4;

    smem(b, OP, 2, KARG, 32)?;
    // Normalized pointer: normalized + (wave*hidden + wg*16)*4.
    sop(b, format!("s_mul_i32 s{TMP}, s{SM}, s{WAVE}"), &[TMP], &[SM, WAVE])?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{TMP}, 2"), &[TMP], &[TMP])?;
    sop(b, format!("s_lshl_b32 s{TMP2}, s{WG}, 6"), &[TMP2], &[WG])?;
    let a = b.spec.arch;
    sop(
        b,
        format!("{} s{TMP}, s{TMP}, s{TMP2}", crate::kernels::common::s_add_i32(a)),
        &[TMP],
        &[TMP, TMP2],
    )?;
    add64(b, NP, YP, TMP)?;
    // Normalized byte offsets of column tile c: r*16*hidden + h*32 + c*256*hidden.
    sop(b, format!("s_lshl_b32 s{TMP}, s{SM}, 4"), &[TMP], &[SM])?;
    op(
        b,
        format!("v_and_b32_e32 v{VT}, 15, v{LANE}"),
        &[v(VT)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_mul_lo_u32 v{noff}, s{TMP}, v{VT}"),
        &[v(noff)],
        &[v(VT), s(TMP)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{VT2}, 4, v{LANE}"),
        &[v(VT2)],
        &[v(LANE)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{VT2}, 5, v{VT2}"),
        &[v(VT2)],
        &[v(VT2)],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{noff}, v{noff}, v{VT2}"),
        &[v(noff)],
        &[v(noff), v(VT2)],
    )?;
    for c in 1..nt {
        sop(
            b,
            format!("s_mul_i32 s{TMP2}, s{TMP}, {}", 16 * c),
            &[TMP2],
            &[TMP],
        )?;
        op(
            b,
            format!("v_add_nc_u32_e32 v{}, s{TMP2}, v{noff}", noff + c as u8),
            &[v(noff + c as u8)],
            &[v(noff), s(TMP2)],
        )?;
    }
    // Scales of this lane's eight rows, then the normalized values of live columns.
    vload128(b, sv, SO, SP, 0)?;
    vload128(b, sv + 4, SO, SP, 16)?;
    for c in 0..nt {
        exec_mask(b, c)?;
        vload128(b, nrm + 8 * c as u8, noff + c as u8, NP, 0)?;
        vload128(b, nrm + 8 * c as u8 + 4, noff + c as u8, NP, 16)?;
    }
    exec_all(b)?;
    // LDS store base: ((wave*NR)*32 + lane)*4; fold read base: lane*4.
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LB}, 2, v{LANE}"),
        &[v(LB)],
        &[v(LANE)],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{WAVE}, {}", lit(nr * 128)),
        &[TMP],
        &[WAVE],
    )?;
    op(
        b,
        format!("v_add_nc_u32_e32 v{LB}, s{TMP}, v{LB}"),
        &[v(LB)],
        &[v(LB), s(TMP)],
    )?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{LR}, 2, v{LANE}"),
        &[v(LR)],
        &[v(LANE)],
    )?;
    for j in 0..nr {
        let (c, r) = ((j / 8) as u8, (j % 8) as u8);
        let n = nrm + 8 * c + r;
        op(
            b,
            format!("v_mul_f32_e32 v{val}, v{}, v{}", ACC + j as u8, sv + r),
            &[v(val)],
            &[v(ACC + j as u8), v(sv + r)],
        )?;
        round_trip(b, val, [t[1], t[2], t[3]])?;
        sigmoid(b, val, t)?;
        // gate = rt(1/den) in t0; normed = rt(normalized).
        round_trip(b, t[0], [t[1], t[2], t[3]])?;
        round_trip(b, n, [t[1], t[2], t[3]])?;
        op(
            b,
            format!("v_mul_f32_e32 v{val}, v{}, v{n}", t[0]),
            &[v(val)],
            &[v(t[0]), v(n)],
        )?;
        round_trip(b, val, [t[1], t[2], t[3]])?;
        b.ds_store(
            slot,
            Instruction::new(
                format!("ds_store_b32 v{LB}, v{val} offset:{}", j * 128),
                vec![],
                vec![v(LB), v(val)],
            )
            .memory(MemoryClass::DsStore),
        )?;
    }
    b.wait_all()?;
    b.barrier(&[Transition::Ready(slot)])?;
    // Wave w folds j % 4 == w: value = +0 + term0 + term1 + term2 + term3, / 4.0.
    for w in 0..BRANCHES {
        let next = format!(".Lhc8_{}_fold{w}", read_up_symbol(nt));
        sop(b, format!("s_cmp_eq_u32 s{WAVE}, {w}"), &[], &[WAVE])?;
        branch(b, &format!("s_cbranch_scc0 {next}"))?;
        for j in (0..nr).filter(|j| j % BRANCHES == w) {
            for k in 0..BRANCHES {
                b.ds_load(
                    slot,
                    Instruction::new(
                        format!(
                            "ds_load_b32 v{}, v{LR} offset:{}",
                            p + k as u8,
                            (k * nr + j) * 128
                        ),
                        vec![v(p + k as u8)],
                        vec![v(LR)],
                    )
                    .memory(MemoryClass::DsLoad),
                )?;
            }
            op(
                b,
                format!("v_add_f32_e32 v{val}, 0, v{p}"),
                &[v(val)],
                &[v(p)],
            )?;
            for k in 1..BRANCHES {
                op(
                    b,
                    format!("v_add_f32_e32 v{val}, v{val}, v{}", p + k as u8),
                    &[v(val)],
                    &[v(val), v(p + k as u8)],
                )?;
            }
            divide(b, Ok(val), Err("4.0"), t)?;
            round_trip(b, t[0], [t[1], t[2], t[3]])?;
            store_out(b, j / 8, j % 8, t[0])?;
        }
        b.wait_all()?;
        branch(b, &format!("s_branch {end}"))?;
        b.label(&next)?;
    }
    b.wait_all()
}

/// Build the `nt`-column-tile read-up export.
pub fn emit_read_up(nt: u32) -> Result<Emitted, String> {
    if !(1..=4).contains(&nt) {
        return Err(format!("hc8_read_up: NT must be 1..=4, got {nt}"));
    }
    let g = geo(nt);
    g.validate()?;
    let l = Lay::new(&g)?;
    let symbol = read_up_symbol(nt);
    let kernargs = KernargLayout::new(READ_UP_KERNARG_BYTES)
        .pointer_access("q", 0, Access::ReadOnly)
        .pointer_access("s", 8, Access::ReadOnly)
        .pointer_access("low", 16, Access::ReadOnly)
        .pointer_access("normalized", 24, Access::ReadOnly)
        .pointer_access("mixed", 32, Access::WriteOnly)
        .hidden("hidden", 40, 4, "by_value")
        .hidden("low_rank", 44, 4, "by_value")
        .hidden("rows", 48, 4, "by_value")
        .hidden("pad", 52, 4, "by_value");
    let spec = KernelSpec {
        kernel_id: "hc8_read_up".into(),
        variant: format!("e4m3_n{nt}"),
        arch: Arch::Gfx1201,
        symbol,
        kernargs,
        user_sgpr_count: 2,
        system_sgpr_workgroup_id_y: false,
        workgroup_size: READ_UP_BLOCK as u16,
        group_segment_fixed_size: lds_bytes(nt),
        wave32: true,
        cu_mode: false,
    };
    let mut regs = plan(&g, &l)?;
    regs.add_range("norm_base", Kind::S, NP, 2, Live::Whole)?;
    regs.add_range("out_ptr", Kind::S, OP, 2, Live::Whole)?;
    let mut b = Builder::new(spec, regs);
    b.enable_delay_alu();
    let slot = b.lds_slot("terms", 0, lds_bytes(nt))?;
    let tag = read_up_symbol(nt);
    let end = format!(".Lhc8_{tag}_end");
    prologue(&mut b, &g, &end, true)?;
    zero_regs(&mut b, ACC, g.nr())?;
    zero_regs(&mut b, l.xb, 16 * l.u * g.nt)?;
    k_loops(&mut b, &g, &l, &tag)?;
    epilogue(&mut b, &g, &l, slot, &end)?;
    b.wait_all()?;
    b.label(&end)?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

/// All four exports, `n1..n4`, in [`read_up_symbol`] order.
pub fn emit_read_ups() -> Result<Vec<Emitted>, String> {
    (1..=4)
        .map(|nt| emit_read_up(nt).map_err(|e| format!("{}: {}: {e}", super::MODULE, read_up_symbol(nt))))
        .collect()
}
