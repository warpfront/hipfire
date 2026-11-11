// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Independent gfx1100 BF16 fixed-eight-row emitters, total logical R=1..64.
//!
//! Arithmetic references in `xtx-singleton-reference/` (source transcription,
//! NOT hardware byte-equality certification): `bf16-scalar.co`, export
//! `gemv_bf16_xf32`, SHA256
//! `b217ac560c63db3027eda1d83711c8d6399cb23df692ffd6deb0637acbd7bd64`;
//! `bf16-hc.co`, export `gemv_bf16_xf32_k4`, SHA256
//! `3b7b89d5f8ff79609a04c949cd9cb7deec3e8d7ef758fce492cc303a3279c9bb`.
//! Matching source SHAs respectively:
//! `80ca9c63f6d74d061213fa4013f6a1da6f87c024ba60d145113f00d6c17bef42`,
//! `995b0bb6355dedc7edca0660c95cf9a39ede8441d2d31855f39f5a7614faed25`.
//! PLAN sections 4 and 13 freeze the interfaces and compiled-object DAGs.
//! Only S1 addressing/scheduling conventions are reused from Halo, not its
//! scalar-float epilogue. This module has no dependency on the Halo emitter.
//!
//! Ordinary: x4 ABI96, grid [sum Mi, ceil(R/8), 1], block32, LDS0. K%8!=0
//! selects the ALL-scalar path l+32*g into a0. Otherwise lane l visits
//! 8*l+256*g+j, j=0..7, with a[j&3]=fma(Xj,Wj,a[j&3]). Tails execute only
//! wholly valid chunks under EXEC; no padding or vector-prefix scalar tail.
//! Local compiled fold is (a3+a2)+(a1+a0), then own+other 16/8/4/2/1.
//!
//! HC: W0/X8/Y16/M24/K28/scale32/act36/R40, logical44/segment48;
//! grid [M, ceil(R/8), 1], block128, LDS128. K%32==0 fixes four contiguous
//! K/4 quarters, each with the same four-chain DAG. The final quarter fold
//! is VALU (p0+p1)+(p2+p3), before scalar read/INTEGER BF16 RN. Activation
//! is RN(SiLU(RN(scale * RN(sum)))), with scale as the FIRST VALU operand.
//! SiLU transcribes 0xbaa0..0xbb6c, including full exp/ldexp/div refinement
//! and exact constants. Numerator div_scale owns VCC until div_fmas.
//! hc_act=0 bypasses all RN, scale and activation nodes.
//!
//! Descriptor pins: scalar KD@0x680 RSRC1=0xe0af0003; HC KD@0x90c0
//! RSRC1=0xe0af000d. Both round32/16_64=0/0, denorm32/16_64=3/3,
//! DX10_CLAMP=1, IEEE_MODE=1, FP16_OVFL=0, WGP=1, wave32. Existing PM
//! descriptor defaults match these modes; candidate resources differ.
//! Portable CPU exp2/rcp/divide and exceptional FP domains are NOT a
//! gfx1100 byte oracle: actual intrinsic/hardware gates own those claims.
//!
//! Eight rows use v40..71 regardless of total R. Budgets: 80 VGPR/80 SGPR;
//! LDS0 ordinary, LDS128 HC. All invalid-domain exits precede the barrier.
//! Inactive rows execute no loads, arithmetic or stores. Addressing retains
//! the existing host-checked 32-bit device-offset restriction; weight-row
//! bases use full 64-bit product/carry, not truncated pointer arithmetic.

use crate::insn::{Instruction, MemoryClass};
use crate::kernels::common::{
    lit, op, s, s_add_i32, s_add_u32, s_addc_u32, smem, sop, v, SRD_WORD3,
};
use crate::lds::Transition;
use crate::plan::Access;
use crate::reg::{Kind, Live};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};

type R = Result<(), String>;
const KARG: u8 = 0;
const SRD: u8 = 4;
/// Kernarg pointer/scalar block s[PA:PA+7] (A0..A3, or the HC W X Y M K). gfx1100
/// `FeatureUserSGPRInit16Bug`: user + system SGPRs >= 16 declares 14 user SGPRs, so the workgroup
/// ids are s14/s15 and the block lives clear of them.
const PA: u8 = 72;
const PXY: u8 = 16;
const Y3M: u8 = 24;
const M2KB: u8 = 28;
const XP: u8 = 16;
const Y0: u8 = 18;
const M0: u8 = 26;
const SK: u8 = 30;
const SR: u8 = 31;
const SY: u8 = 32;
const SM: u8 = 34;
const COUNT: u8 = 40;
const BASE: u8 = 41;
const PART: u8 = 42;
const QUARTER: u8 = 43;
const SCALE: u8 = 44;
const TMP: u8 = 45;
const WRL: u8 = 46;
const WRH: u8 = 47;
const EM: u8 = 48;
const KREM: u8 = 49;
const GOFF: u8 = 50;
const XG: u8 = 51;
const GI: u8 = 52;
const KB: u8 = 53;
const MB: u8 = 54;
const STMP: u8 = 55;
const QX: u8 = 57;
const SPATH: u8 = 58;
const NG: u8 = 59;
const COL: u8 = 60;
const ACT: u8 = 61;
const SOUT: u8 = 64;
const TID: u8 = 0;
const HOFF: u8 = 1;
const XOFF: u8 = 3;
const SHUF: u8 = 4;
const LANE: u8 = 5;
const LDST: u8 = 9;
const VT: u8 = 13;
const WS: u8 = 16;
const XS: u8 = 24;
const ACC0: u16 = 40;
const REG_BUDGET: u16 = 80;

fn acc(row: u32, stream: u32) -> u8 {
    let index = u32::from(ACC0) + 4 * row + stream;
    u8::try_from(index).expect("fixed-eight accumulator index fits v255")
}
fn label(hc: bool, name: &str) -> String {
    format!(".Lbf16xtx_t8_{}_{name}", if hc { "hc" } else { "x4" })
}
fn jump(b: &mut Builder, hc: bool, cond: &str, name: &str) -> R {
    branch(b, &format!("s_{cond} {}", label(hc, name)))
}
fn mov(b: &mut Builder, dst: u8, src: u8) -> R {
    sop(b, format!("s_mov_b32 s{dst}, s{src}"), &[dst], &[src])
}

/// Emit the ordinary x4-ABI96 S1 or HC-ABI48 four-quarter gfx1100 twin.
pub fn emit_rowtiles(hc: bool) -> Result<Emitted, String> {
    let arch = Arch::Gfx1100;
    let args = if hc {
        KernargLayout::new(48)
            .pointer_access("W", 0, Access::ReadOnly)
            .pointer_access("X", 8, Access::ReadOnly)
            .pointer_access("Y", 16, Access::WriteOnly)
            .hidden("M", 24, 4, "by_value")
            .hidden("K", 28, 4, "by_value")
            .hidden("scale", 32, 4, "by_value")
            .hidden("hc_act", 36, 4, "by_value")
            .hidden("R", 40, 4, "by_value")
    } else {
        KernargLayout::new(96)
            .pointer_access("A0", 0, Access::ReadOnly)
            .pointer_access("A1", 8, Access::ReadOnly)
            .pointer_access("A2", 16, Access::ReadOnly)
            .pointer_access("A3", 24, Access::ReadOnly)
            .pointer_access("x", 32, Access::ReadOnly)
            .pointer_access("y0", 40, Access::WriteOnly)
            .pointer_access("y1", 48, Access::WriteOnly)
            .pointer_access("y2", 56, Access::WriteOnly)
            .pointer_access("y3", 64, Access::WriteOnly)
            .hidden("M0", 72, 4, "by_value")
            .hidden("M1", 76, 4, "by_value")
            .hidden("M2", 80, 4, "by_value")
            .hidden("M3", 84, 4, "by_value")
            .hidden("K", 88, 4, "by_value")
            .hidden("R", 92, 4, "by_value")
    };
    let high = usize::from(ACC0) + 4 * 8;
    if high > usize::from(REG_BUDGET) || high > 256 || SOUT + 3 >= REG_BUDGET as u8 {
        return Err("gfx1100 BF16 fixed-eight register map exceeds budget".into());
    }
    let mut regs = RegPlan::new(REG_BUDGET, REG_BUDGET)?;
    for base in (0..REG_BUDGET as u8).step_by(8) {
        regs.add_range(&format!("s{base}"), Kind::S, base, 8, Live::Whole)?;
        regs.add_range(&format!("v{base}"), Kind::V, base, 8, Live::Whole)?;
    }
    let mut b = Builder::new(
        KernelSpec {
            kernel_id: "qwen4_bf16_rows_gfx1100".into(),
            variant: if hc { "t8_hc_k4_s1" } else { "t8_x4_s1" }.into(),
            arch,
            symbol: format!(
                "gemv_bf16_xf32_{}_t8_pm_gfx1100",
                if hc { "k4_rows" } else { "rows_x4" }
            ),
            kernargs: args,
            user_sgpr_count: 2,
            system_sgpr_workgroup_id_y: true,
            workgroup_size: if hc { 128 } else { 32 },
            group_segment_fixed_size: if hc { 128 } else { 0 },
            wave32: true,
            cu_mode: false,
        },
        regs,
    );
    b.enable_delay_alu();
    let partials = if hc {
        Some(b.lds_slot("quarters", 0, 128)?)
    } else {
        None
    };
    prologue(&mut b, hc)?;
    for r in 0..8 {
        sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
        jump(&mut b, hc, "cbranch_scc1", "initialized")?;
        for st in 0..4 {
            let a = acc(r, st);
            valu(&mut b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
        }
    }
    b.label(&label(hc, "initialized"))?;
    sop(&mut b, format!("s_cmp_lg_u32 s{SPATH}, 0"), &[], &[SPATH])?;
    jump(&mut b, hc, "cbranch_scc1", "scalar")?;
    sweep(&mut b, hc, false)?;
    jump(&mut b, hc, "branch", "reduce")?;
    b.label(&label(hc, "scalar"))?;
    sweep(&mut b, hc, true)?;
    b.label(&label(hc, "reduce"))?;
    for r in 0..8 {
        sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
        jump(&mut b, hc, "cbranch_scc1", "reduced")?;
        reduce(&mut b, r)?;
        if let Some(slot) = partials {
            sop(
                &mut b,
                format!("s_lshl_b32 s{TMP}, s{QUARTER}, 2"),
                &[TMP],
                &[QUARTER],
            )?;
            valu(
                &mut b,
                &format!("v_mov_b32_e32 v{LDST}, s{TMP}"),
                &[LDST],
                &[],
                &[TMP],
            )?;
            exec_lane0(&mut b)?;
            b.ds_store(
                slot,
                Instruction::new(
                    format!("ds_store_b32 v{LDST}, v{}{}", acc(r, 3), off(16 * r)),
                    vec![],
                    vec![v(LDST), v(acc(r, 3))],
                )
                .memory(MemoryClass::DsStore),
            )?;
            b.wait_all()?;
            exec_all(&mut b)?;
        } else {
            store(&mut b, r, acc(r, 3))?;
        }
    }
    b.label(&label(hc, "reduced"))?;
    if let Some(slot) = partials {
        b.barrier(&[Transition::Ready(slot)])?;
        sop(
            &mut b,
            format!("s_cmp_lg_u32 s{QUARTER}, 0"),
            &[],
            &[QUARTER],
        )?;
        jump(&mut b, hc, "cbranch_scc1", "end")?;
        for r in 0..8 {
            sop(&mut b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
            jump(&mut b, hc, "cbranch_scc1", "end")?;
            valu(
                &mut b,
                &format!("v_mov_b32_e32 v{LDST}, {}", lit(16 * r)),
                &[LDST],
                &[],
                &[],
            )?;
            b.ds_load(
                slot,
                Instruction::new(
                    format!("ds_load_b128 v[{WS}:{}], v{LDST}", WS + 3),
                    (0..4).map(|i| v(WS + i)).collect(),
                    vec![v(LDST)],
                )
                .memory(MemoryClass::DsLoad),
            )?;
            // b978/b984: VALU quarter fold, before scalar read/RN.
            valu(&mut b, "v_add_f32_e32 v16, v16, v17", &[16], &[16, 17], &[])?;
            valu(&mut b, "v_add_f32_e32 v18, v18, v19", &[18], &[18, 19], &[])?;
            valu(&mut b, "v_add_f32_e32 v16, v16, v18", &[16], &[16, 18], &[])?;
            hc_value(&mut b, r)?;
            store(&mut b, r, XS)?;
        }
    }
    b.label(&label(hc, "end"))?;
    branch(&mut b, "s_endpgm")?;
    b.finish()
}

fn prologue(b: &mut Builder, hc: bool) -> R {
    let ids = b.spec.workgroup_ids(false)?;
    let (crate::arch::WgIdSource::Sgpr(wgx), crate::arch::WgIdSource::Sgpr(wgy)) = (ids.x, ids.y)
    else {
        return Err("gfx1100 BF16 rowtiles require SGPR workgroup IDs".into());
    };
    if hc {
        smem(b, PA, 8, KARG, 0)?;
        smem(b, 16, 4, KARG, 32)?;
        b.wait_all()?;
        for (d, x) in [
            (SCALE, 16),
            (ACT, 17),
            (SR, 18),
            (XP, PA + 2),
            (XP + 1, PA + 3),
            (SY, PA + 4),
            (SY + 1, PA + 5),
            (SM, PA + 6),
            (SK, PA + 7),
        ] {
            mov(b, d, x)?;
        }
        mov(b, COL, wgx)?;
        valu(
            b,
            &format!("v_lshrrev_b32_e32 v{VT}, 5, v{TID}"),
            &[VT],
            &[TID],
            &[],
        )?;
        op(
            b,
            format!("v_readfirstlane_b32 s{QUARTER}, v{VT}"),
            &[s(QUARTER)],
            &[v(VT)],
        )?;
        mov(b, SRD, PA)?;
        mov(b, SRD + 1, PA + 1)?;
    } else {
        smem(b, PA, 8, KARG, 0)?;
        smem(b, PXY, 8, KARG, 32)?;
        smem(b, Y3M, 4, KARG, 64)?;
        smem(b, M2KB, 4, KARG, 80)?;
        b.wait_all()?;
        mov(b, COL, wgx)?;
        for i in 0..4u8 {
            sop(
                b,
                format!("s_cmp_lt_u32 s{COL}, s{}", M0 + i),
                &[],
                &[COL, M0 + i],
            )?;
            jump(b, hc, "cbranch_scc1", &format!("matrix{i}"))?;
            sop(
                b,
                format!("s_sub_u32 s{COL}, s{COL}, s{}", M0 + i),
                &[COL],
                &[COL, M0 + i],
            )?;
        }
        jump(b, hc, "branch", "end")?;
        for i in 0..4u8 {
            b.label(&label(hc, &format!("matrix{i}")))?;
            for (d, x) in [
                (SRD, PA + 2 * i),
                (SRD + 1, PA + 2 * i + 1),
                (SY, Y0 + 2 * i),
                (SY + 1, Y0 + 2 * i + 1),
                (SM, M0 + i),
            ] {
                mov(b, d, x)?;
            }
            jump(b, hc, "branch", "selected")?;
        }
        b.label(&label(hc, "selected"))?;
        sop(b, format!("s_mov_b32 s{QUARTER}, 0"), &[QUARTER], &[])?;
    }
    for (text, uses) in [
        (format!("s_cmp_ge_u32 s{COL}, s{SM}"), vec![COL, SM]),
        (format!("s_cmp_eq_u32 s{SR}, 0"), vec![SR]),
        (format!("s_cmp_gt_u32 s{SR}, 64"), vec![SR]),
        (format!("s_cmp_le_i32 s{SK}, 0"), vec![SK]),
    ] {
        sop(b, text, &[], &uses)?;
        jump(b, hc, "cbranch_scc1", "end")?;
    }
    sop(b, format!("s_lshl_b32 s{BASE}, s{wgy}, 3"), &[BASE], &[wgy])?;
    sop(b, format!("s_cmp_ge_u32 s{BASE}, s{SR}"), &[], &[BASE, SR])?;
    jump(b, hc, "cbranch_scc1", "end")?;
    // Single-definition clamp: COUNT's ONE def (the min) dominates every
    // `s_cmp_le_u32 COUNT, r` user; TMP is dead here on both paths and
    // redefined before any later read (cf. qwen4_bf16_rows_wide).
    sop(
        b,
        format!("s_sub_u32 s{TMP}, s{SR}, s{BASE}"),
        &[TMP],
        &[SR, BASE],
    )?;
    sop(
        b,
        format!("s_min_u32 s{COUNT}, s{TMP}, 8"),
        &[COUNT],
        &[TMP],
    )?;
    if hc {
        sop(b, format!("s_and_b32 s{TMP}, s{SK}, 31"), &[TMP], &[SK])?;
        sop(b, format!("s_cmp_lg_u32 s{TMP}, 0"), &[], &[TMP])?;
        jump(b, hc, "cbranch_scc1", "end")?;
        sop(b, format!("s_lshr_b32 s{PART}, s{SK}, 2"), &[PART], &[SK])?;
    } else {
        mov(b, PART, SK)?;
    }
    sop(b, format!("s_lshl_b32 s{KB}, s{SK}, 2"), &[KB], &[SK])?;
    sop(b, format!("s_lshl_b32 s{MB}, s{SM}, 2"), &[MB], &[SM])?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{SK}, 1"), &[TMP], &[SK])?;
    sop(
        b,
        format!("s_mul_i32 s{WRL}, s{COL}, s{TMP}"),
        &[WRL],
        &[COL, TMP],
    )?;
    sop(
        b,
        format!("s_mul_hi_u32 s{WRH}, s{COL}, s{TMP}"),
        &[WRH],
        &[COL, TMP],
    )?;
    sop(
        b,
        format!("{} s{SRD}, s{SRD}, s{WRL}", s_add_u32(b.spec.arch)),
        &[SRD],
        &[SRD, WRL],
    )?;
    sop(
        b,
        format!(
            "{} s{}, s{}, s{WRH}",
            s_addc_u32(b.spec.arch),
            SRD + 1,
            SRD + 1
        ),
        &[SRD + 1],
        &[SRD + 1, WRH],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{QX}, s{QUARTER}, s{PART}"),
        &[QX],
        &[QUARTER, PART],
    )?;
    sop(b, format!("s_lshl_b32 s{TMP}, s{QX}, 1"), &[TMP], &[QX])?;
    sop(
        b,
        format!("{} s{SRD}, s{SRD}, s{TMP}", s_add_u32(b.spec.arch)),
        &[SRD],
        &[SRD, TMP],
    )?;
    sop(
        b,
        format!("{} s{}, s{}, 0", s_addc_u32(b.spec.arch), SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1],
    )?;
    sop(b, format!("s_lshl_b32 s{QX}, s{QX}, 2"), &[QX], &[QX])?;
    sop(
        b,
        format!("s_and_b32 s{}, s{}, 0xffff", SRD + 1, SRD + 1),
        &[SRD + 1],
        &[SRD + 1],
    )?;
    sop(b, format!("s_mov_b32 s{}, -1", SRD + 2), &[SRD + 2], &[])?;
    sop(
        b,
        format!("s_mov_b32 s{}, {}", SRD + 3, lit(SRD_WORD3)),
        &[SRD + 3],
        &[],
    )?;
    sop(
        b,
        format!("s_and_b32 s{SPATH}, s{PART}, 7"),
        &[SPATH],
        &[PART],
    )?;
    valu(
        b,
        &format!("v_and_b32_e32 v{LANE}, 31, v{TID}"),
        &[LANE],
        &[TID],
        &[],
    )
}

fn sweep(b: &mut Builder, hc: bool, scalar: bool) -> R {
    let (shift, size, n) = if scalar { (5, 32, 1) } else { (8, 256, 8) };
    let tag = if scalar { "scalar" } else { "fast" };
    sop(
        b,
        format!(
            "{} s{NG}, s{PART}, {}",
            s_add_i32(b.spec.arch),
            lit(size - 1)
        ),
        &[NG],
        &[PART],
    )?;
    sop(b, format!("s_lshr_b32 s{NG}, s{NG}, {shift}"), &[NG], &[NG])?;
    sop(b, format!("s_mov_b32 s{GI}, 0"), &[GI], &[])?;
    b.loop_(&label(hc, &format!("{tag}_loop")), |b| {
        sop(
            b,
            format!("s_lshl_b32 s{TMP}, s{GI}, {shift}"),
            &[TMP],
            &[GI],
        )?;
        sop(
            b,
            format!("s_sub_u32 s{KREM}, s{PART}, s{TMP}"),
            &[KREM],
            &[PART, TMP],
        )?;
        if scalar {
            mov(b, EM, KREM)?;
        } else {
            sop(b, format!("s_lshr_b32 s{EM}, s{KREM}, 3"), &[EM], &[KREM])?;
        }
        sop(b, format!("s_lshl_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
        sop(b, format!("s_not_b32 s{EM}, s{EM}"), &[EM], &[EM])?;
        sop(
            b,
            format!("s_cmp_ge_u32 s{KREM}, {}", lit(size)),
            &[],
            &[KREM],
        )?;
        sop(b, format!("s_cselect_b32 s{EM}, -1, s{EM}"), &[EM], &[EM])?;
        sop(b, format!("s_lshl_b32 s{GOFF}, s{TMP}, 1"), &[GOFF], &[TMP])?;
        sop(b, format!("s_lshl_b32 s{XG}, s{TMP}, 2"), &[XG], &[TMP])?;
        valu(
            b,
            &format!(
                "v_lshlrev_b32_e32 v{HOFF}, {}, v{LANE}",
                if scalar { 1 } else { 4 }
            ),
            &[HOFF],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_nc_u32_e32 v{HOFF}, s{GOFF}, v{HOFF}"),
            &[HOFF],
            &[HOFF],
            &[GOFF],
        )?;
        exec_set(b)?;
        let load = if scalar {
            format!(
                "buffer_load_u16 v{WS}, v{HOFF}, s[{SRD}:{}], 0 offen",
                SRD + 3
            )
        } else {
            format!(
                "buffer_load_b128 v[{WS}:{}], v{HOFF}, s[{SRD}:{}], 0 offen",
                WS + 3,
                SRD + 3
            )
        };
        let defs: Vec<u8> = (0..if scalar { 1 } else { 4 }).map(|i| WS + i).collect();
        vmem(
            b,
            &load,
            &defs,
            &[HOFF],
            &[SRD, SRD + 1, SRD + 2, SRD + 3],
            false,
        )?;
        b.wait_all()?;
        if scalar {
            valu(
                b,
                &format!("v_lshlrev_b32_e32 v{WS}, 16, v{WS}"),
                &[WS],
                &[WS],
                &[],
            )?;
        } else {
            widen_weights(b)?;
        }
        exec_all(b)?;
        for r in 0..8 {
            sop(b, format!("s_cmp_le_u32 s{COUNT}, {r}"), &[], &[COUNT])?;
            jump(b, hc, "cbranch_scc1", &format!("{tag}_advanced"))?;
            sop(
                b,
                format!("{} s{TMP}, s{BASE}, {r}", s_add_i32(b.spec.arch)),
                &[TMP],
                &[BASE],
            )?;
            sop(
                b,
                format!("s_mul_i32 s{TMP}, s{TMP}, s{KB}"),
                &[TMP],
                &[TMP, KB],
            )?;
            sop(
                b,
                format!("{} s{TMP}, s{TMP}, s{QX}", s_add_i32(b.spec.arch)),
                &[TMP],
                &[TMP, QX],
            )?;
            sop(
                b,
                format!("{} s{TMP}, s{TMP}, s{XG}", s_add_i32(b.spec.arch)),
                &[TMP],
                &[TMP, XG],
            )?;
            valu(
                b,
                &format!(
                    "v_lshlrev_b32_e32 v{XOFF}, {}, v{LANE}",
                    if scalar { 2 } else { 5 }
                ),
                &[XOFF],
                &[LANE],
                &[],
            )?;
            valu(
                b,
                &format!("v_add_nc_u32_e32 v{XOFF}, s{TMP}, v{XOFF}"),
                &[XOFF],
                &[XOFF],
                &[TMP],
            )?;
            exec_set(b)?;
            for j in 0..n {
                vmem(
                    b,
                    &format!(
                        "global_load_b32 v{}, v{XOFF}, s[{XP}:{}]{}",
                        XS + j as u8,
                        XP + 1,
                        off(4 * j)
                    ),
                    &[XS + j as u8],
                    &[XOFF],
                    &[XP, XP + 1],
                    false,
                )?;
            }
            b.wait_all()?;
            for j in 0..n {
                let a = acc(r, j & 3);
                valu(
                    b,
                    &format!("v_fmac_f32_e32 v{a}, v{}, v{}", XS + j as u8, WS + j as u8),
                    &[a],
                    &[a, XS + j as u8, WS + j as u8],
                    &[],
                )?;
            }
            exec_all(b)?;
        }
        b.label(&label(hc, &format!("{tag}_advanced")))?;
        sop(
            b,
            format!("{} s{GI}, s{GI}, 1", s_add_i32(b.spec.arch)),
            &[GI],
            &[GI],
        )?;
        sop(b, format!("s_cmp_lt_u32 s{GI}, s{NG}"), &[], &[GI, NG])?;
        jump(b, hc, "cbranch_scc1", &format!("{tag}_loop"))
    })
}

fn reduce(b: &mut Builder, row: u32) -> R {
    // Compiled scalar 18d8/19f0/1a00; HC b71c/b874/b884.
    for (d, x) in [(1, 0), (3, 2), (3, 1)] {
        let (d, x) = (acc(row, d), acc(row, x));
        valu(
            b,
            &format!("v_add_f32_e32 v{d}, v{d}, v{x}"),
            &[d],
            &[d, x],
            &[],
        )?;
    }
    let a = acc(row, 3);
    b.ds_crosslane(
        Instruction::new(
            format!("ds_swizzle_b32 v{XS}, v{a} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            vec![v(XS)],
            vec![v(a)],
        )
        .memory(MemoryClass::DsLoad),
    )?;
    valu(
        b,
        &format!("v_add_f32_e32 v{a}, v{a}, v{XS}"),
        &[a],
        &[a, XS],
        &[],
    )?;
    for d in [8, 4, 2, 1] {
        valu(
            b,
            &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v{LANE}", 32 - d),
            &[],
            &[LANE],
            &[],
        )?;
        valu(
            b,
            &format!("v_cndmask_b32_e64 v{SHUF}, 0, {d}, vcc_lo"),
            &[SHUF],
            &[],
            &[],
        )?;
        valu(
            b,
            &format!("v_add_lshl_u32 v{SHUF}, v{SHUF}, v{LANE}, 2"),
            &[SHUF],
            &[SHUF, LANE],
            &[],
        )?;
        b.ds_crosslane(
            Instruction::new(
                format!("ds_bpermute_b32 v{XS}, v{SHUF}, v{a}"),
                vec![v(XS)],
                vec![v(SHUF), v(a)],
            )
            .memory(MemoryClass::DsLoad),
        )?;
        valu(
            b,
            &format!("v_add_f32_e32 v{a}, v{a}, v{XS}"),
            &[a],
            &[a, XS],
            &[],
        )?;
    }
    Ok(())
}

fn store(b: &mut Builder, row: u32, value: u8) -> R {
    sop(
        b,
        format!("{} s{TMP}, s{BASE}, {row}", s_add_i32(b.spec.arch)),
        &[TMP],
        &[BASE],
    )?;
    sop(
        b,
        format!("s_mul_i32 s{TMP}, s{TMP}, s{MB}"),
        &[TMP],
        &[TMP, MB],
    )?;
    sop(b, format!("s_lshl_b32 s{STMP}, s{COL}, 2"), &[STMP], &[COL])?;
    sop(
        b,
        format!("{} s{TMP}, s{TMP}, s{STMP}", s_add_i32(b.spec.arch)),
        &[TMP],
        &[TMP, STMP],
    )?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{XOFF}, s{TMP}"),
        &[XOFF],
        &[],
        &[TMP],
    )?;
    exec_lane0(b)?;
    vmem(
        b,
        &format!("global_store_b32 v{XOFF}, v{value}, s[{SY}:{}]", SY + 1),
        &[],
        &[XOFF, value],
        &[SY, SY + 1],
        true,
    )?;
    b.wait_all()?;
    exec_all(b)
}

// Exact scalar INTEGER BF16 RN bit function, b990..ba08 and later rounds.
fn bf16_round(b: &mut Builder, row: u32, stage: u32) -> R {
    let tag = format!("rn{row}_{stage}");
    sop(
        b,
        format!("s_and_b32 s65, s{SOUT}, 0x7f800000"),
        &[65],
        &[SOUT],
    )?;
    sop(b, "s_cmp_eq_u32 s65, 0x7f800000", &[], &[65])?;
    jump(b, true, "cbranch_scc1", &format!("{tag}_special"))?;
    sop(
        b,
        format!("s_bfe_u32 s65, s{SOUT}, 0x10010"),
        &[65],
        &[SOUT],
    )?;
    sop(
        b,
        format!("s_add_u32 s65, s{SOUT}, s65"),
        &[65],
        &[SOUT, 65],
    )?;
    sop(b, "s_add_u32 s65, s65, 0x7fff", &[65], &[65])?;
    jump(b, true, "branch", &format!("{tag}_mask"))?;
    b.label(&label(true, &format!("{tag}_special")))?;
    sop(b, format!("s_and_b32 s66, s{SOUT}, 0xffff"), &[66], &[SOUT])?;
    sop(b, format!("s_or_b32 s67, s{SOUT}, 0x10000"), &[67], &[SOUT])?;
    sop(b, "s_cmp_eq_u32 s66, 0", &[], &[66])?;
    sop(
        b,
        format!("s_cselect_b32 s65, s{SOUT}, s67"),
        &[65],
        &[SOUT, 67],
    )?;
    b.label(&label(true, &format!("{tag}_mask")))?;
    sop(
        b,
        format!("s_and_b32 s{SOUT}, s65, 0xffff0000"),
        &[SOUT],
        &[65],
    )
}
fn read_value(b: &mut Builder, val: u8) -> R {
    op(
        b,
        format!("v_readfirstlane_b32 s{SOUT}, v{val}"),
        &[s(SOUT)],
        &[v(val)],
    )
}
fn hc_value(b: &mut Builder, row: u32) -> R {
    let done = format!("act{row}_done");
    valu(b, &format!("v_mov_b32_e32 v{XS}, v{WS}"), &[XS], &[WS], &[])?;
    sop(b, format!("s_cmp_eq_u32 s{ACT}, 0"), &[], &[ACT])?;
    jump(b, true, "cbranch_scc1", &done)?;
    read_value(b, XS)?;
    bf16_round(b, row, 0)?;
    valu(
        b,
        &format!("v_mul_f32_e64 v{XS}, s{SCALE}, s{SOUT}"),
        &[XS],
        &[],
        &[SCALE, SOUT],
    )?;
    read_value(b, XS)?;
    bf16_round(b, row, 1)?;
    silu(b)?;
    read_value(b, XS)?;
    bf16_round(b, row, 2)?;
    valu(
        b,
        &format!("v_mov_b32_e32 v{XS}, s{SOUT}"),
        &[XS],
        &[],
        &[SOUT],
    )?;
    b.label(&label(true, &done))
}

// Pinned HC baa0..bb6c. s64 remains the immutable rounded numerator;
// v24=h/t/exp/denominator/output, v25=e/ds, v26=j/rcp,
// v27=scaled numerator, v28=quotient, v29=remainder.
fn silu(b: &mut Builder) -> R {
    valu(b, "v_mul_f32_e64 v24, 0xbfb8aa3b, s64", &[24], &[], &[64])?;
    valu(
        b,
        "v_cmp_nlt_f32_e64 vcc_lo, 0x42ce8ed0, s64",
        &[],
        &[],
        &[64],
    )?;
    valu(
        b,
        "v_fma_f32 v25, 0xbfb8aa3b, s64, -v24",
        &[25],
        &[24],
        &[64],
    )?;
    valu(b, "v_rndne_f32_e32 v26, v24", &[26], &[24], &[])?;
    // Source VOPD fmamk/sub are independent nodes, not a reassociation.
    valu(
        b,
        "v_fmamk_f32 v25, s64, 0xb2a5705f, v25",
        &[25],
        &[25],
        &[64],
    )?;
    valu(b, "v_sub_f32_e32 v24, v24, v26", &[24], &[24, 26], &[])?;
    valu(b, "v_add_f32_e32 v24, v24, v25", &[24], &[24, 25], &[])?;
    valu(b, "v_cvt_i32_f32_e32 v25, v26", &[25], &[26], &[])?;
    valu(b, "v_exp_f32_e32 v24, v24", &[24], &[24], &[])?;
    valu(b, "v_ldexp_f32 v24, v24, v25", &[24], &[24, 25], &[])?;
    valu(
        b,
        "v_cndmask_b32_e32 v24, 0, v24, vcc_lo",
        &[24],
        &[24],
        &[],
    )?;
    valu(
        b,
        "v_cmp_ngt_f32_e64 vcc_lo, 0xc2b17218, s64",
        &[],
        &[],
        &[64],
    )?;
    valu(
        b,
        "v_cndmask_b32_e32 v24, 0x7f800000, v24, vcc_lo",
        &[24],
        &[24],
        &[],
    )?;
    valu(b, "v_add_f32_e32 v24, 1.0, v24", &[24], &[24], &[])?;
    valu(
        b,
        "v_div_scale_f32 v25, null, v24, v24, s64",
        &[25],
        &[24],
        &[64],
    )?;
    valu(b, "v_rcp_f32_e32 v26, v25", &[26], &[25], &[])?;
    valu(b, "v_fma_f32 v27, -v25, v26, 1.0", &[27], &[25, 26], &[])?;
    valu(b, "v_fmac_f32_e32 v26, v27, v26", &[26], &[26, 27], &[])?;
    // VCC carry ownership: no comparison or interleaved row before div_fmas.
    valu(
        b,
        "v_div_scale_f32 v27, vcc_lo, s64, v24, s64",
        &[27],
        &[24],
        &[64],
    )?;
    valu(b, "v_mul_f32_e32 v28, v27, v26", &[28], &[27, 26], &[])?;
    valu(
        b,
        "v_fma_f32 v29, -v25, v28, v27",
        &[29],
        &[25, 28, 27],
        &[],
    )?;
    valu(b, "v_fmac_f32_e32 v28, v29, v26", &[28], &[28, 29, 26], &[])?;
    valu(
        b,
        "v_fma_f32 v25, -v25, v28, v27",
        &[25],
        &[25, 28, 27],
        &[],
    )?;
    valu(
        b,
        "v_div_fmas_f32 v25, v25, v26, v28",
        &[25],
        &[25, 26, 28],
        &[],
    )?;
    valu(
        b,
        "v_div_fixup_f32 v24, v25, v24, s64",
        &[24],
        &[25, 24],
        &[64],
    )
}

fn off(imm: u32) -> String {
    if imm == 0 {
        String::new()
    } else {
        format!(" offset:{imm}")
    }
}
fn exec_set(b: &mut Builder) -> R {
    op(b, format!("s_mov_b32 exec_lo, s{EM}"), &[], &[s(EM)])
}
fn exec_all(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}
fn exec_lane0(b: &mut Builder) -> R {
    op(b, "s_mov_b32 exec_lo, 1", &[], &[])
}
fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices
        .iter()
        .map(|&base| crate::reg::RegRef { kind, base, len: 1 })
        .collect()
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), inputs))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> R {
    let mut inputs = refs(Kind::V, uses);
    inputs.extend(refs(Kind::S, scalar));
    b.push(
        Instruction::new(text, refs(Kind::V, defs), inputs).memory(if store {
            MemoryClass::VmemStore
        } else {
            MemoryClass::VmemLoad
        }),
    )
}
fn branch(b: &mut Builder, text: &str) -> R {
    b.control(Instruction::new(text, vec![], vec![]))
}
fn widen_weights(b: &mut Builder) -> R {
    // Backwards unpack keeps every original packed dword live until read.
    for d in (0..4u8).rev() {
        let (lo, hi, packed) = (WS + 2 * d, WS + 2 * d + 1, WS + d);
        valu(
            b,
            &format!("v_and_b32_e32 v{hi}, 0xffff0000, v{packed}"),
            &[hi],
            &[packed],
            &[],
        )?;
        valu(
            b,
            &format!("v_lshlrev_b32_e32 v{lo}, 16, v{packed}"),
            &[lo],
            &[packed],
            &[],
        )?;
    }
    Ok(())
}
