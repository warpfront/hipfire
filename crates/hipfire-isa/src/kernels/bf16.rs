//! f32 -> BF16 rounding, as two named contracts. Both round finite values to
//! nearest-even; they differ on non-finite input, so the bytes a kernel
//! stores for a NaN depend on which one it names, and its oracle must use the
//! same contract.
use super::common::{op, v};
use crate::Builder;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Bf16 {
    /// `hip_bfloat16::round_to_bfloat16`: RNE on finite values; infinities
    /// pass through and a NaN keeps its payload, with a quiet bit forced when
    /// only the low half carried it. Result: the BF16 bits in the low half.
    HipBfloat16,
    /// `(u + 0x7fff + ((u >> 16) & 1))` on finite values, every non-finite
    /// value passed through unchanged (the fn-moe-sym `sym_bf16_rt`/`_bits`
    /// pair). Result: an f32 whose high half is the BF16 bits.
    RneFinitePassthrough,
}

impl Bf16 {
    /// [`Bf16::HipBfloat16`] of `v{src}` into the low half of `v{dst}`;
    /// `scratch` is four dead VGPRs, VCC is clobbered.
    pub fn hip_bfloat16(b: &mut Builder, src: u8, dst: u8, scratch: [u8; 4]) -> Result<(), String> {
        let [t0, t1, t2, t3] = scratch;
        op(b, format!("v_lshrrev_b32_e32 v{t0}, 16, v{src}"), &[v(t0)], &[v(src)])?;
        op(b, format!("v_and_b32_e32 v{t0}, 1, v{t0}"), &[v(t0)], &[v(t0)])?;
        op(b, format!("v_add_nc_u32_e32 v{t0}, 0x7fff, v{t0}"), &[v(t0)], &[v(t0)])?;
        op(b, format!("v_add_nc_u32_e32 v{t1}, v{src}, v{t0}"), &[v(t1)], &[v(src), v(t0)])?;
        op(b, format!("v_and_b32_e32 v{t2}, 0xffff, v{src}"), &[v(t2)], &[v(src)])?;
        op(b, format!("v_cmp_ne_u32_e32 vcc_lo, 0, v{t2}"), &[], &[v(t2)])?;
        op(b, format!("v_cndmask_b32_e64 v{t2}, 0, 0x10000, vcc_lo"), &[v(t2)], &[])?;
        op(b, format!("v_or_b32_e32 v{t2}, v{src}, v{t2}"), &[v(t2)], &[v(src), v(t2)])?;
        op(b, format!("v_and_b32_e32 v{t3}, 0x7f800000, v{src}"), &[v(t3)], &[v(src)])?;
        op(b, format!("v_cmp_eq_u32_e32 vcc_lo, 0x7f800000, v{t3}"), &[], &[v(t3)])?;
        op(b, format!("v_cndmask_b32_e32 v{dst}, v{t1}, v{t2}, vcc_lo"), &[v(dst)], &[v(t1), v(t2)])?;
        op(b, format!("v_lshrrev_b32_e32 v{dst}, 16, v{dst}"), &[v(dst)], &[v(dst)])
    }

    /// [`Bf16::RneFinitePassthrough`] of `v{x}` in place; with `clear_low`
    /// the low half of a rounded value is zeroed (the BF16 value as f32),
    /// else only the high half is defined. `tmp` is a dead VGPR, `mask` a
    /// dead lane-mask SGPR pair.
    pub fn rne_finite_passthrough(b: &mut Builder, x: u8, tmp: u8, mask: u8, clear_low: bool) -> Result<(), String> {
        use super::common::s;
        op(b, format!("v_bfe_u32 v{tmp}, v{x}, 16, 1"), &[v(tmp)], &[v(x)])?;
        op(b, format!("v_add3_u32 v{tmp}, v{x}, 0x7fff, v{tmp}"), &[v(tmp)], &[v(x), v(tmp)])?;
        if clear_low { op(b, format!("v_and_b32_e32 v{tmp}, 0xffff0000, v{tmp}"), &[v(tmp)], &[v(tmp)])?; }
        op(b, format!("v_cmp_class_f32_e64 s{mask}, v{x}, 0x1f8"), &[s(mask)], &[v(x)])?;
        op(b, format!("v_cndmask_b32_e64 v{x}, v{x}, v{tmp}, s{mask}"), &[v(x)], &[v(x), v(tmp), s(mask)])
    }
}
