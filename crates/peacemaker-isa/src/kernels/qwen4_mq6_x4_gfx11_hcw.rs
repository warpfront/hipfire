// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact gfx1151 HC-write arithmetic from the U3 hipcc epilogue.
//! Float operations follow `u3_b8_r8_p2_hcw.s` lines 3864, 3883–3917,
//! 3935, 4282–4320 and the corresponding F32-state branch. BF16 conversion
//! follows ROCm's `hip/amd_detail/amd_hip_bfloat16.h::float_to_bfloat16`:
//! finite RNE; nonfinite values retain their high half and set bit 16 when
//! any discarded bit is nonzero (preserving a signaling NaN).
//! Callers use FP denorm modes 3/3. No helper changes EXEC or FP modes.
use crate::kernels::common::{op, v};
use crate::{Arch, Builder};

type R = Result<(), String>;
/// Scratch VGPRs starting at `t`, disjoint from every operand.
pub const TEMPS: u8 = 9;
/// No ordinary SGPRs are clobbered. VCC is clobbered by comparisons and division.
/// Like `gdn_scan`, VCC is implicit: RegPlan's SGPR budget ends at 104 and
/// cannot represent the architectural VCC register (106) as a live RegRef.
pub const SGPRS: &[u8] = &[];

fn arch(b: &Builder) -> R {
    if b.spec.arch != Arch::Gfx1151 {
        return Err("HCW arithmetic is exact-gfx1151 only".into());
    }
    Ok(())
}

/// Round F32 to BF16 bits in the low half; upper half is zero. `dst == src` is allowed.
pub fn bf16_bits(b: &mut Builder, dst: u8, src: u8, t: u8) -> R {
    arch(b)?;
    let a = t + 6;
    let c = t + 7;
    let n = t + 8;
    op(
        b,
        format!("v_bfe_u32 v{a}, v{src}, 16, 1"),
        &[v(a)],
        &[v(src)],
    )?;
    op(
        b,
        format!("v_add3_u32 v{a}, v{src}, v{a}, 0x7fff"),
        &[v(a)],
        &[v(src), v(a)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{c}, 0xffff, v{src}"),
        &[v(c)],
        &[v(src)],
    )?;
    op(
        b,
        format!("v_or_b32_e32 v{n}, 0x10000, v{src}"),
        &[v(n)],
        &[v(src)],
    )?;
    op(b, format!("v_cmp_eq_u32_e32 vcc_lo, 0, v{c}"), &[], &[v(c)])?;
    op(
        b,
        format!("v_cndmask_b32_e32 v{n}, v{n}, v{src}, vcc_lo"),
        &[v(n)],
        &[v(n), v(src)],
    )?;
    op(
        b,
        format!("v_and_b32_e32 v{c}, 0x7f800000, v{src}"),
        &[v(c)],
        &[v(src)],
    )?;
    op(
        b,
        format!("v_cmp_ne_u32_e32 vcc_lo, 0x7f800000, v{c}"),
        &[],
        &[v(c)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e32 v{a}, v{n}, v{a}, vcc_lo"),
        &[v(a)],
        &[v(n), v(a)],
    )?;
    op(
        b,
        format!("v_lshrrev_b32_e32 v{dst}, 16, v{a}"),
        &[v(dst)],
        &[v(a)],
    )
}

fn bf16f(b: &mut Builder, dst: u8, src: u8, t: u8) -> R {
    bf16_bits(b, dst, src, t)?;
    op(
        b,
        format!("v_lshlrev_b32_e32 v{dst}, 16, v{dst}"),
        &[v(dst)],
        &[v(dst)],
    )
}

/// `hcw_gate(logit)` as F32 bits, including every hipcc exp/div instruction.
/// `dst == logit` is allowed.
pub fn gate(b: &mut Builder, dst: u8, logit: u8, t: u8) -> R {
    arch(b)?;
    let x = t;
    let sc = t + 1;
    let r = t + 2;
    let e = t + 3;
    let q = t + 4;
    let residual = t + 5;
    bf16f(b, x, logit, t)?;
    op(
        b,
        format!("v_mul_f32_e32 v{x}, 0x3e800000, v{x}"),
        &[v(x)],
        &[v(x)],
    )?;
    bf16f(b, x, x, t)?;
    // expf(-x): compensated base-2 reduction, exp2, ldexp and range guards.
    op(
        b,
        format!("v_mul_f32_e32 v{sc}, 0xbfb8aa3b, v{x}"),
        &[v(sc)],
        &[v(x)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{r}, 0xbfb8aa3b, v{x}, -v{sc}"),
        &[v(r)],
        &[v(x), v(sc)],
    )?;
    op(b, format!("v_rndne_f32_e32 v{e}, v{sc}"), &[v(e)], &[v(sc)])?;
    op(
        b,
        format!("v_sub_f32_e32 v{sc}, v{sc}, v{e}"),
        &[v(sc)],
        &[v(sc), v(e)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{r}, 0xb2a5705f, v{x}"),
        &[v(r)],
        &[v(r), v(x)],
    )?;
    op(
        b,
        format!("v_cmp_nlt_f32_e32 vcc_lo, 0x42ce8ed0, v{x}"),
        &[],
        &[v(x)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{sc}, v{sc}, v{r}"),
        &[v(sc)],
        &[v(sc), v(r)],
    )?;
    op(b, format!("v_cvt_i32_f32_e32 v{r}, v{e}"), &[v(r)], &[v(e)])?;
    op(b, format!("v_exp_f32_e32 v{sc}, v{sc}"), &[v(sc)], &[v(sc)])?;
    op(
        b,
        format!("v_ldexp_f32 v{sc}, v{sc}, v{r}"),
        &[v(sc)],
        &[v(sc), v(r)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e32 v{sc}, 0, v{sc}, vcc_lo"),
        &[v(sc)],
        &[v(sc)],
    )?;
    op(
        b,
        format!("v_cmp_ngt_f32_e32 vcc_lo, 0xc2b17218, v{x}"),
        &[],
        &[v(x)],
    )?;
    op(
        b,
        format!("v_cndmask_b32_e32 v{x}, 0x7f800000, v{sc}, vcc_lo"),
        &[v(x)],
        &[v(sc)],
    )?;
    op(
        b,
        format!("v_add_f32_e32 v{x}, 1.0, v{x}"),
        &[v(x)],
        &[v(x)],
    )?;
    // Correctly rounded 1/d, in hipcc operand order.
    op(
        b,
        format!("v_div_scale_f32 v{sc}, null, v{x}, v{x}, 1.0"),
        &[v(sc)],
        &[v(x)],
    )?;
    op(b, format!("v_rcp_f32_e32 v{r}, v{sc}"), &[v(r)], &[v(sc)])?;
    op(
        b,
        format!("v_fma_f32 v{e}, -v{sc}, v{r}, 1.0"),
        &[v(e)],
        &[v(sc), v(r)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{r}, v{e}, v{r}"),
        &[v(r)],
        &[v(r), v(e)],
    )?;
    op(
        b,
        format!("v_div_scale_f32 v{e}, vcc_lo, 1.0, v{x}, 1.0"),
        &[v(e)],
        &[v(x)],
    )?;
    op(
        b,
        format!("v_mul_f32_e32 v{q}, v{e}, v{r}"),
        &[v(q)],
        &[v(e), v(r)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{residual}, -v{sc}, v{q}, v{e}"),
        &[v(residual)],
        &[v(sc), v(q), v(e)],
    )?;
    op(
        b,
        format!("v_fmac_f32_e32 v{q}, v{residual}, v{r}"),
        &[v(q)],
        &[v(q), v(residual), v(r)],
    )?;
    op(
        b,
        format!("v_fma_f32 v{sc}, -v{sc}, v{q}, v{e}"),
        &[v(sc)],
        &[v(sc), v(q), v(e)],
    )?;
    op(
        b,
        format!("v_div_fmas_f32 v{sc}, v{sc}, v{r}, v{q}"),
        &[v(sc)],
        &[v(sc), v(r), v(q)],
    )?;
    op(
        b,
        format!("v_div_fixup_f32 v{x}, v{sc}, v{x}, 1.0"),
        &[v(x)],
        &[v(sc), v(x)],
    )?;
    bf16f(b, x, x, t)?;
    // hipcc implements 2 * rounded_reciprocal by addition, not multiplication.
    op(
        b,
        format!("v_add_f32_e32 v{x}, v{x}, v{x}"),
        &[v(x)],
        &[v(x)],
    )?;
    bf16f(b, dst, x, t)
}

fn unpack(b: &mut Builder, dst: u8, packed: u8, high: bool) -> R {
    if high {
        op(
            b,
            format!("v_and_b32_e32 v{dst}, 0xffff0000, v{packed}"),
            &[v(dst)],
            &[v(packed)],
        )
    } else {
        op(
            b,
            format!("v_lshlrev_b32_e32 v{dst}, 16, v{packed}"),
            &[v(dst)],
            &[v(packed)],
        )
    }
}

/// Apply one gate to eight channels of BF16 state. `out == stream` is allowed.
/// `mq` is four parked BF16x2 dwords; `g` already contains `gate`'s F32 result.
pub fn apply_bf16(b: &mut Builder, out: u8, stream: u8, mq: u8, g: u8, t: u8) -> R {
    arch(b)?;
    for p in 0..4 {
        for h in 0..2 {
            unpack(b, t, mq + p, h != 0)?;
            op(
                b,
                format!("v_mul_f32_e32 v{t}, v{g}, v{t}"),
                &[v(t)],
                &[v(g), v(t)],
            )?;
            bf16f(b, t, t, t)?;
            unpack(b, t + 1, stream + p, h != 0)?;
            op(
                b,
                format!("v_add_f32_e32 v{t}, v{}, v{t}", t + 1),
                &[v(t)],
                &[v(t + 1), v(t)],
            )?;
            bf16_bits(b, t + 2 + h, t, t)?;
        }
        op(
            b,
            format!("v_lshlrev_b32_e32 v{}, 16, v{}", t + 3, t + 3),
            &[v(t + 3)],
            &[v(t + 3)],
        )?;
        op(
            b,
            format!("v_or_b32_e32 v{}, v{}, v{}", out + p, t + 2, t + 3),
            &[v(out + p)],
            &[v(t + 2), v(t + 3)],
        )?;
    }
    Ok(())
}

/// Apply one gate to eight F32 channels, reproducing the F32-state BF16 round trips.
/// `out == stream` is allowed; result dwords contain BF16 values expanded to F32.
pub fn apply_f32(b: &mut Builder, out: u8, stream: u8, mq: u8, g: u8, t: u8) -> R {
    arch(b)?;
    for i in 0..8 {
        unpack(b, t, mq + i / 2, i % 2 != 0)?;
        op(
            b,
            format!("v_mul_f32_e32 v{t}, v{g}, v{t}"),
            &[v(t)],
            &[v(g), v(t)],
        )?;
        bf16f(b, t, t, t)?;
        bf16f(b, t + 1, stream + i, t)?;
        op(
            b,
            format!("v_add_f32_e32 v{t}, v{t}, v{}", t + 1),
            &[v(t)],
            &[v(t), v(t + 1)],
        )?;
        bf16f(b, out + i, t, t)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reg::Live, KernargLayout, KernelSpec, RegPlan};

    fn emission() -> Vec<String> {
        let mut regs = RegPlan::new(48, 0).unwrap();
        for n in 0..48 {
            regs.v::<1>("hcw_test", n, Live::Whole).unwrap();
        }
        let mut b = Builder::new(
            KernelSpec {
                kernel_id: "hcw_test".into(),
                variant: "test".into(),
                arch: Arch::Gfx1151,
                symbol: "hcw_test".into(),
                kernargs: KernargLayout::new(0),
                user_sgpr_count: 0,
                system_sgpr_workgroup_id_y: false,
                workgroup_size: 32,
                group_segment_fixed_size: 0,
                wave32: true,
                cu_mode: false,
            },
            regs,
        );
        gate(&mut b, 0, 0, 32).unwrap();
        bf16_bits(&mut b, 1, 1, 32).unwrap();
        apply_bf16(&mut b, 4, 4, 16, 0, 32).unwrap();
        apply_f32(&mut b, 8, 8, 16, 0, 32).unwrap();
        b.program()
            .instructions
            .iter()
            .map(|i| i.text.clone())
            .collect()
    }

    #[test]
    fn gfx1151_emission_is_deterministic_without_exec_or_gfx12_waits() {
        let a = emission();
        assert_eq!(a, emission());
        assert!(a
            .iter()
            .all(|i| !i.contains("s_wait_alu") && !i.contains("exec") && !i.contains("s_denorm")));
        assert!(a.iter().any(|i| i.starts_with("v_div_fmas_f32")));
        assert!(a.iter().any(|i| i.starts_with("v_exp_f32_e32")));
    }
}
