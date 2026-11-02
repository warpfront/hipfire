// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::{
    kernels::{
        bf16::Bf16,
        common::{op, s, v},
        iu4_gemm::region::{emit_interleaved, Binding, Region},
    },
    reg::RegRef,
    Builder,
};

/// Temporary VGPRs [`Epilogue::silu_dense`] consumes per element.
pub const DENSE_SILU_TEMPS: u8 = 6;

/// Numerator of [`Epilogue::silu_dense_with`]'s IEEE divide.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DivNum {
    Gate,
    One,
}
/// The op [`Epilogue::silu_dense_with`] ends with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Finish {
    MulUp,
    MulUpFirst,
    Quotient,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epilogue {
    Set,
    Add,
    SiLU,
    Bf16Rne,
}
impl Epilogue {
    /// SET copies the sum; ADD keeps the existing output as the left operand.
    pub fn apply(self, b: &mut Builder, sum: u8, output: u8) -> Result<(), String> {
        match self {
            Self::Set if sum == output => Ok(()),
            Self::Set => op(
                b,
                format!("v_mov_b32_e32 v{output}, v{sum}"),
                &[v(output)],
                &[v(sum)],
            ),
            Self::Add => op(
                b,
                format!("v_add_f32_e32 v{output}, v{output}, v{sum}"),
                &[v(output)],
                &[v(output), v(sum)],
            ),
            _ => Err("SiLU/BF16 need explicit temporary bindings".into()),
        }
    }
    /// Instantiate the imported hipcc region, preserving its per-value DAG.
    pub fn silu(b: &mut Builder, bindings: &[Binding]) -> Result<(), String> {
        let region = if b.spec.arch.gfx12() {
            Region::silu()?
        } else {
            Region::silu_gfx1100()?
        };
        emit_interleaved(b, &region, bindings)
    }
    /// `g[k] = g[k] / (1 + expf(-g[k])) * u[k]` for k = 0..n-1, the exact op
    /// DAG hipcc emits for the dense V2B SiLU multiply (LLVM's f32 `exp`
    /// lowering with range checks and the IEEE `fdiv` expansion with f32
    /// denormals enabled) over `gate..gate+n` / `up..up+n`, with h written
    /// over the gate values. The `n` elements are interleaved op by op: one
    /// SiLU emitter for gfx11 and gfx12. Element `k` uses the temporaries
    /// `tmp + DENSE_SILU_TEMPS*k ..+DENSE_SILU_TEMPS` and the lane-mask
    /// pairs `mask + 2k` (underflow), `mask + 2(n+k)` (overflow),
    /// `mask + 2(2n+k)` (numerator scale).
    pub fn silu_dense(
        b: &mut Builder,
        gate: u8,
        up: u8,
        tmp: u8,
        mask: u8,
        n: u8,
    ) -> Result<(), String> {
        Self::silu_dense_with(b, gate, up, tmp, mask, n, DivNum::Gate, Finish::MulUp)
    }
    /// [`Self::silu_dense`] generalised over the IEEE divide's numerator and
    /// the op after it. `DivNum::Gate` is `g / (1 + expf(-g))`; `DivNum::One`
    /// is hipcc's `1.0f / (1.0f + expf(-g))` (the divide with the constant
    /// numerator: `v_div_scale_f32 .., 1.0`, `v_div_fixup_f32 .., 1.0`).
    /// `Finish::MulUp` writes `q * u` over the gate values (`v_mul_f32 g, q,
    /// u`), `Finish::MulUpFirst` writes `u * q` (`v_mul_f32 g, u, q`: the
    /// operand order of hipcc's `silu_mul_bf16_rt(up) * ...` product),
    /// `Finish::Quotient` leaves the quotient `q` in the element's fifth
    /// temporary (`tmp + DENSE_SILU_TEMPS*k + 5`) and writes nothing else
    /// (`up` is not read).
    #[allow(clippy::too_many_arguments)]
    pub fn silu_dense_with(
        b: &mut Builder,
        gate: u8,
        up: u8,
        tmp: u8,
        mask: u8,
        n: u8,
        num: DivNum,
        fin: Finish,
    ) -> Result<(), String> {
        let t = |k: u8, i: u8| tmp + DENSE_SILU_TEMPS * k + i;
        let (m_under, m_over, m_num) = (
            |k: u8| mask + 2 * k,
            |k: u8| mask + 2 * (n + k),
            |k: u8| mask + 2 * (2 * n + k),
        );
        let nt = |k: u8| match num {
            DivNum::Gate => format!("v{}", gate + k),
            DivNum::One => "1.0".to_string(),
        };
        let nu = |k: u8| match num {
            DivNum::Gate => vec![v(gate + k)],
            DivNum::One => vec![],
        };
        let cat = |mut a: Vec<RegRef>, c: Vec<RegRef>| {
            a.extend(c);
            a
        };
        type Step<'a> = &'a dyn Fn(&mut Builder, u8) -> Result<(), String>;
        let steps: [Step; 26] = [
            // ph = RN(-log2e * g); pl = fma(-log2e, g, -ph); pl = fma(-log2e_lo, g, pl)
            &|b, k| {
                op(
                    b,
                    format!("v_mul_f32_e32 v{}, 0xbfb8aa3b, v{}", t(k, 0), gate + k),
                    &[v(t(k, 0))],
                    &[v(gate + k)],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_fma_f32 v{}, 0xbfb8aa3b, v{}, -v{}",
                        t(k, 1),
                        gate + k,
                        t(k, 0)
                    ),
                    &[v(t(k, 1))],
                    &[v(gate + k), v(t(k, 0))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_fmac_f32_e32 v{}, 0xb2a5705f, v{}", t(k, 1), gate + k),
                    &[v(t(k, 1))],
                    &[v(t(k, 1)), v(gate + k)],
                )
            },
            // e = rndne(ph); a = (ph - e) + pl; r = ldexp(exp2(a), int(e))
            &|b, k| {
                op(
                    b,
                    format!("v_rndne_f32_e32 v{}, v{}", t(k, 2), t(k, 0)),
                    &[v(t(k, 2))],
                    &[v(t(k, 0))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_sub_f32_e32 v{0}, v{0}, v{1}", t(k, 0), t(k, 2)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0)), v(t(k, 2))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_add_f32_e32 v{0}, v{0}, v{1}", t(k, 0), t(k, 1)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0)), v(t(k, 1))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_exp_f32_e32 v{0}, v{0}", t(k, 0)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_cvt_i32_f32_e32 v{0}, v{0}", t(k, 2)),
                    &[v(t(k, 2))],
                    &[v(t(k, 2))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_ldexp_f32 v{0}, v{0}, v{1}", t(k, 0), t(k, 2)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0)), v(t(k, 2))],
                )
            },
            // r = -g < -103.28 ? 0 : r; r = -g > 88.72 ? +inf : r
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_cmp_nlt_f32_e64 s{}, 0x42ce8ed0, v{}",
                        m_under(k),
                        gate + k
                    ),
                    &[s(m_under(k))],
                    &[v(gate + k)],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{1}", t(k, 0), m_under(k)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0)), s(m_under(k))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_cmp_ngt_f32_e64 s{}, 0xc2b17218, v{}",
                        m_over(k),
                        gate + k
                    ),
                    &[s(m_over(k))],
                    &[v(gate + k)],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_cndmask_b32_e64 v{0}, 0x7f800000, v{0}, s{1}",
                        t(k, 0),
                        m_over(k)
                    ),
                    &[v(t(k, 0))],
                    &[v(t(k, 0)), s(m_over(k))],
                )
            },
            // d = 1 + r; q = g / d (div_scale, rcp, three fma refinements, fmas, fixup)
            &|b, k| {
                op(
                    b,
                    format!("v_add_f32_e32 v{0}, 1.0, v{0}", t(k, 0)),
                    &[v(t(k, 0))],
                    &[v(t(k, 0))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_div_scale_f32 v{0}, null, v{1}, v{1}, {2}",
                        t(k, 1),
                        t(k, 0),
                        nt(k)
                    ),
                    &[v(t(k, 1))],
                    &cat(vec![v(t(k, 0))], nu(k)),
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_div_scale_f32 v{0}, s{1}, {2}, v{3}, {2}",
                        t(k, 2),
                        m_num(k),
                        nt(k),
                        t(k, 0)
                    ),
                    &[v(t(k, 2)), s(m_num(k))],
                    &cat(vec![v(t(k, 0))], nu(k)),
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_rcp_f32_e32 v{}, v{}", t(k, 3), t(k, 1)),
                    &[v(t(k, 3))],
                    &[v(t(k, 1))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_fma_f32 v{}, -v{}, v{}, 1.0", t(k, 4), t(k, 1), t(k, 3)),
                    &[v(t(k, 4))],
                    &[v(t(k, 1)), v(t(k, 3))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_fmac_f32_e32 v{0}, v{1}, v{0}", t(k, 3), t(k, 4)),
                    &[v(t(k, 3))],
                    &[v(t(k, 3)), v(t(k, 4))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_mul_f32_e32 v{}, v{}, v{}", t(k, 4), t(k, 2), t(k, 3)),
                    &[v(t(k, 4))],
                    &[v(t(k, 2)), v(t(k, 3))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_fma_f32 v{}, -v{}, v{}, v{}",
                        t(k, 5),
                        t(k, 1),
                        t(k, 4),
                        t(k, 2)
                    ),
                    &[v(t(k, 5))],
                    &[v(t(k, 1)), v(t(k, 4)), v(t(k, 2))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("v_fmac_f32_e32 v{}, v{}, v{}", t(k, 4), t(k, 5), t(k, 3)),
                    &[v(t(k, 4))],
                    &[v(t(k, 4)), v(t(k, 5)), v(t(k, 3))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_fma_f32 v{}, -v{}, v{}, v{}",
                        t(k, 5),
                        t(k, 1),
                        t(k, 4),
                        t(k, 2)
                    ),
                    &[v(t(k, 5))],
                    &[v(t(k, 1)), v(t(k, 4)), v(t(k, 2))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!("s_mov_b32 vcc_lo, s{}", m_num(k)),
                    &[],
                    &[s(m_num(k))],
                )?;
                op(
                    b,
                    format!(
                        "v_div_fmas_f32 v{0}, v{0}, v{1}, v{2}",
                        t(k, 5),
                        t(k, 3),
                        t(k, 4)
                    ),
                    &[v(t(k, 5))],
                    &[v(t(k, 5)), v(t(k, 3)), v(t(k, 4))],
                )
            },
            &|b, k| {
                op(
                    b,
                    format!(
                        "v_div_fixup_f32 v{0}, v{0}, v{1}, {2}",
                        t(k, 5),
                        t(k, 0),
                        nt(k)
                    ),
                    &[v(t(k, 5))],
                    &cat(vec![v(t(k, 5)), v(t(k, 0))], nu(k)),
                )
            },
            // h = q * u (or u * q), or the quotient itself
            &|b, k| match fin {
                Finish::MulUp => op(
                    b,
                    format!("v_mul_f32_e32 v{}, v{}, v{}", gate + k, t(k, 5), up + k),
                    &[v(gate + k)],
                    &[v(t(k, 5)), v(up + k)],
                ),
                Finish::MulUpFirst => op(
                    b,
                    format!("v_mul_f32_e32 v{}, v{}, v{}", gate + k, up + k, t(k, 5)),
                    &[v(gate + k)],
                    &[v(up + k), v(t(k, 5))],
                ),
                Finish::Quotient => Ok(()),
            },
        ];
        for step in steps {
            for k in 0..n {
                step(b, k)?;
            }
        }
        Ok(())
    }
    pub fn bf16_rne(b: &mut Builder, src: u8, dst: u8, scratch: [u8; 4]) -> Result<(), String> {
        Bf16::hip_bfloat16(b, src, dst, scratch)
    }
}
