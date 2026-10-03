// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Integer EXP/LOG/RCP/RSQ/SQRT models ported from ROCm/rocm-systems at pinned
// commit 3866a6c80b985d1fd770d197ebaab97bf2397eea (emulation/rocjitsu):
//   lib/util/include/util/amdgpu_exp.h, amdgpu_log.h, amdgpu_rcp.h,
//   amdgpu_rsq.h, amdgpu_sqrt.h
//   lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/transcendental.h
//   tests/transcendental_test.cpp (published captured words, see `tests`)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Raw-bit V_EXP/V_LOG/V_RCP/V_RSQ/V_SQRT/V_RCP_IFLAG `F32` value models.
//!
//! Every function is pure integer arithmetic (no host `f32`, no libm, no
//! allocation) over IEEE-754 binary32 words. The models are *empirical staged
//! fixed-point cubics* fitted upstream to physical RDNA3/RDNA4 captures, with
//! sparse one-ULP residual tables for RCP, RSQ and SQRT. They are candidate
//! models, not a claim about the hardware implementation: **no function here
//! may be reported bit-exact on a card until a local exhaustive `2^32`
//! comparison has recorded zero mismatches**; counts quoted by upstream are not
//! this repository's measurements.
//!
//! Mode and exception conventions (the contract, taken from upstream's
//! `execute_v_*_f32_vop1` bodies at the pinned commit):
//!
//! * FP32 `MODE` is round-to-nearest with denormals preserved. These models are
//!   independent of `MODE` rounding and of denormal mode: every subnormal input
//!   and subnormal result is mapped explicitly by the model (RCP/RSQ/SQRT/LOG
//!   treat any subnormal input as signed zero; EXP maps the small-input interval
//!   to one; RCP flushes sub-normal-range results to signed zero).
//! * Signaling NaNs are quieted (`quiet_snan == true`) and keep sign and payload.
//!   Upstream selects quieting per instruction from `MODE.IEEE` / architecture
//!   (`fp_mode::quiets_nan`); this API fixes it to `true`, matching the kernel
//!   `MODE` in use. A `quiet_snan == false` variant is intentionally not exposed.
//! * `V_RCP_IFLAG_F32` has the same *value* as `V_RCP_F32` (upstream calls the
//!   same `transcendental::rcp_f32` with the default quieting `true` for both
//!   its scalar and SIMD paths). Its only difference is architectural state
//!   outside the value API: it sets TRAPSTS ALU cause bit 6 (`1 << 6`) when an
//!   active lane's source compares equal to `0.0` (either sign) and, unlike
//!   `V_RCP_F32`, does not use the `FLUSH_NEAREST` SDWA OMOD policy. Neither is
//!   modelled here.
//!
//! Only `Gfx1151` and `Gfx1201` are supported by this value API; every other
//! [`Arch`] panics. Hardware qualification requires the own exhaustive gate above.

use crate::Arch;

mod tables {
    //! Coefficient and residual tables transcribed from the pinned upstream
    //! headers; see `isa/emu/trans_tables.rs` for the provenance header.
    include!("../../isa/emu/trans_tables.rs");
}

use tables::{
    RCP, RCP_CORRECTIONS, RCP_CORRECTION_OFFSETS, RSQ, RSQ_CORRECTIONS, RSQ_CORRECTION_OFFSETS,
    SQRT, SQRT_CORRECTIONS,
};

#[inline]
fn qualify(arch: Arch, func: &str) {
    match arch {
        Arch::Gfx1151 | Arch::Gfx1201 => {}
        other => panic!("emu::trans::{func}: {other:?} is not qualified (only Gfx1151 and Gfx1201)"),
    }
}

/// Round-half-to-even right shift (`shift >= 1`).
#[inline]
fn round_even(value: u64, shift: u32) -> u64 {
    let whole = value >> shift;
    let rest = value & ((1u64 << shift) - 1);
    let midpoint = 1u64 << (shift - 1);
    whole + u64::from(rest > midpoint || (rest == midpoint && (whole & 1) != 0))
}

/// `product >= 2^47` keeps 24 significant bits (round at bit 24, scale by two),
/// otherwise rounds at bit 23; shared by the EXP/LOG quadratic stages.
#[inline]
fn round_product_q36(product: u64) -> u64 {
    if product >= (1u64 << 47) { round_even(product, 24) * 2 } else { round_even(product, 23) }
}

/// Binary search a residual bucket for `key`; returns its entry on a hit.
#[inline]
fn bucket_find(entries: &[u16], first: usize, last: usize, key: u16) -> Option<u16> {
    let bucket = &entries[first..last];
    let at = bucket.partition_point(|&e| (e & 0x7fff) < key);
    match bucket.get(at) {
        Some(&e) if (e & 0x7fff) == key => Some(e),
        _ => None,
    }
}

#[inline]
fn apply_residual(result: u32, entry: Option<u16>) -> u32 {
    match entry {
        Some(e) if e & 0x8000 != 0 => result.wrapping_add(1),
        Some(_) => result.wrapping_sub(1),
        None => result,
    }
}

// ---------------------------------------------------------------- EXP

mod exp {
    use super::{round_even, round_product_q36, tables::EXP};

    /// Constant/linear coefficients Q28, quadratic Q35, cubic Q24; 32 intervals
    /// of the fractional part. Negative arguments use a ones-complement
    /// reduction (exact negative integers approach the previous interval).
    pub(super) fn evaluate(input: u32, quiet_snan: bool) -> u32 {
        let mag = input & 0x7fff_ffff;
        if mag >= 0x7f80_0000 {
            if mag > 0x7f80_0000 {
                return input | if quiet_snan { 0x0040_0000 } else { 0 };
            }
            return if input >> 31 != 0 { 0 } else { 0x7f80_0000 };
        }
        // The small-input interval returns one, including negative inputs.
        if mag < 0x3380_0000 {
            return 0x3f80_0000;
        }
        let negative = input >> 31 != 0;
        if !negative && mag >= 0x4300_0000 {
            return 0x7f80_0000;
        }
        if negative && mag > 0x42fc_0000 {
            return 0;
        }
        // 29 fractional bits retained.
        let exponent = (mag >> 23) as i32 - 127;
        let mantissa = u64::from((mag & 0x7f_ffff) | 0x80_0000);
        let shift = exponent + 6;
        let absolute = if shift >= 0 { mantissa << shift } else { mantissa >> -shift };
        let phase: i64 = if negative { -(absolute as i64) - 1 } else { absolute as i64 };
        let fraction = (phase as u32) & 0xff_ffff;
        let c = EXP[(((phase as u64) >> 24) & 31) as usize];
        // Linear product rounded to Q29 (capped at Q28 for large products),
        // then its low guard bit is dropped before accumulation.
        let lp = u64::from(c.linear) * u64::from(fraction);
        let linear = if lp >= (1u64 << 47) { round_even(lp, 24) } else { round_even(lp, 23) >> 1 };
        // Quadratic stage: 18 coordinate bits, Q24 square, inner coefficient
        // rounded to Q35 before the multiply.
        let coordinate = u64::from(fraction >> 6);
        let square = (coordinate * coordinate) >> 12;
        let inner =
            round_even((u64::from(c.quadratic) << 7) + u64::from(c.cubic) * coordinate, 7);
        let quad = round_product_q36(inner * square);
        // Combine in Q36 and round once to the FP32 significand.
        let sum = ((u64::from(c.constant) + linear) << 8) + quad;
        (((phase >> 29) + 126) * 0x80_0000 + round_even(sum, 13) as i64) as u32
    }
}

// ---------------------------------------------------------------- LOG

mod log {
    use super::{round_even, round_product_q36, tables::LOG};

    /// Below one: `k` is the distance in units of 2^-24, normalised in
    /// four-bit groups; the second coefficient row starts at 1/128 from one.
    /// Q50 guard bits are kept and the result magnitude rounds upward.
    fn near_one_negative(bits: u32) -> u32 {
        let k = 0x3f80_0000u32 - bits;
        let upper = k >= 0x2_0000;
        let c1: u64 = if upper { 12_102_221 } else { 12_102_203 };
        let c2: u64 = if upper { 12_097_323 } else { 12_101_979 };
        let c3: u64 = if upper { 2_088_428 } else { 2_035_248 };
        let width = 32 - k.leading_zeros();
        let normalization = if width < 18 { 4 * ((18 - width) / 4) } else { 0 };
        let linear_shift = if normalization < 14 { 14 - normalization } else { 0 };
        let square_shift = if normalization < 12 { 12 - normalization } else { 0 };
        let k64 = u64::from(k);
        let linear = ((2 * c1 * k64) >> linear_shift) << linear_shift;
        let square = ((k64 * k64) >> square_shift) << square_shift;
        let inner = round_even((c2 << 22) + c3 * (k64 - 1), 22);
        let product_shift = 38 - normalization;
        let quadratic = (inner * square + ((1u64 << product_shift) - 1)) >> product_shift;
        let fixed = (linear << 2) + (quadratic << (16 - normalization))
            - if upper { 100u64 << 16 } else { 0 };
        let mut leading = 63 - fixed.leading_zeros();
        let shift = leading - 23;
        let mut mantissa = (fixed + ((1u64 << shift) - 1)) >> shift;
        if mantissa == 1u64 << 24 {
            mantissa >>= 1;
            leading += 1;
        }
        0x8000_0000 | ((leading + 127 - 50) << 23) | (mantissa as u32 & 0x7f_ffff)
    }

    /// Above one: `i` is the distance in units of 2^-23. Constants Q35; the
    /// zero-origin row gains precision through four-bit normalisation. The
    /// packed significand is the truncated positive one advanced by one,
    /// including exact grid hits.
    fn near_one_positive(bits: u32) -> u32 {
        let i = bits - 0x3f80_0000;
        let row = if i < 0x1_0000 { 0 } else if i < 0x2_0000 { 1 } else { 2 };
        const CONSTANTS: [u64; 3] = [0, 192, 2856];
        const LINEARS: [u64; 3] = [12_102_203, 12_102_186, 12_102_072];
        const QUADRATICS: [u64; 3] = [12_102_094, 12_097_614, 12_084_310];
        const CUBICS: [u64; 3] = [1_999_052, 1_948_887, 1_883_467];
        let width = 32 - i.leading_zeros();
        let normalization = if row != 0 { 0 } else { 4 * ((18 - width) / 4) };
        let wide = u64::from(i);
        let normalized = wide << normalization;
        let linear = ((LINEARS[row] * normalized) >> 13) << 2;
        let square = (normalized * wide) >> 12;
        let inner = round_even((QUADRATICS[row] << 22) - CUBICS[row] * (2 * wide + 1), 22);
        let quadratic = round_product_q36(inner * square);
        let fixed = CONSTANTS[row] + linear - quadratic;
        let mut leading = 63 - fixed.leading_zeros();
        let shift = leading - 23;
        let mut mantissa = (fixed >> shift) + 1;
        if mantissa == 1u64 << 24 {
            mantissa >>= 1;
            leading += 1;
        }
        ((leading + 127 - 35 - normalization) << 23) | (mantissa as u32 & 0x7f_ffff)
    }

    /// Ordinary path: the extracted exponent is kept until the final signed
    /// accumulation; the quadratic product is rounded to Q35 (capped at 24
    /// bits); the final conversion rounds the magnitude downward, including
    /// negative results.
    fn ordinary(bits: u32) -> u32 {
        let fraction = u64::from(bits & 0x3_ffff);
        let mut index = ((bits >> 18) & 31) as usize;
        if index == 1 && fraction >= 0x2_0000 {
            index = 32;
        }
        let c = LOG[index];
        let linear = (u64::from(c.linear) * fraction) >> 16;
        let inner = round_even(
            (u64::from(c.quadratic) << 22) - u64::from(c.cubic) * (2 * fraction + 1),
            22,
        );
        let square = (fraction * fraction) >> 12;
        let quadratic = round_product_q36(inner * square);
        let exponent = (bits >> 23) as i64 - 127;
        // The u64 difference may transiently wrap; the signed reinterpretation
        // below is the intended arithmetic (as upstream).
        let tail = ((u64::from(c.constant) + linear) << 5).wrapping_sub(quadratic) as i64;
        let fixed = exponent * (1i64 << 35) + tail;
        let negative = fixed < 0;
        let magnitude = fixed.unsigned_abs();
        let mut leading = 63 - magnitude.leading_zeros();
        let shift = leading - 23;
        let mut mantissa =
            (magnitude + if negative { (1u64 << shift) - 1 } else { 0 }) >> shift;
        if mantissa == 1u64 << 24 {
            mantissa >>= 1;
            leading += 1;
        }
        (if negative { 0x8000_0000 } else { 0 })
            | ((leading + 127 - 35) << 23)
            | (mantissa as u32 & 0x7f_ffff)
    }

    pub(super) fn evaluate(bits: u32, quiet_snan: bool) -> u32 {
        let magnitude = bits & 0x7fff_ffff;
        if magnitude > 0x7f80_0000 {
            return bits | if quiet_snan { 0x0040_0000 } else { 0 };
        }
        // LOG flushes either sign of subnormal input independently of MODE.
        if magnitude < 0x0080_0000 {
            return 0xff80_0000;
        }
        if bits & 0x8000_0000 != 0 {
            return 0xffc0_0000;
        }
        if magnitude == 0x7f80_0000 {
            return 0x7f80_0000;
        }
        if bits == 0x3f80_0000 {
            return 0;
        }
        if (0x3f7c_0000..0x3f80_0000).contains(&bits) {
            return near_one_negative(bits);
        }
        if bits > 0x3f80_0000 && bits < 0x3f84_0000 {
            return near_one_positive(bits);
        }
        ordinary(bits)
    }
}

// ---------------------------------------------------------------- RCP

mod rcp {
    use super::{
        apply_residual, bucket_find, round_even, RCP, RCP_CORRECTIONS, RCP_CORRECTION_OFFSETS,
    };

    /// Quarter-ULP fixed-point cubic over the 23-bit mantissa.
    fn polynomial(mantissa: u32) -> u32 {
        let c = RCP[(mantissa >> 18) as usize];
        let fraction = u64::from(mantissa & 0x3_ffff);
        let linear_product = u64::from(c.linear) * fraction;
        let shift = if linear_product >= (1u64 << 38) { 15 } else { 14 };
        let linear = round_even(linear_product, shift) << (shift - 14);
        let inner = (u64::from(c.quadratic) << 18) - u64::from(c.cubic) * fraction;
        let square_input = u128::from(fraction & !1);
        let quadratic = ((u128::from(inner) * square_input * square_input) >> 51) as u64;
        let value = (u64::from(c.constant) * 16)
            .wrapping_sub(linear)
            .wrapping_add(quadratic * 2 + 32)
            >> 6;
        (0x3e80_0000u64 + value) as u32
    }

    fn normalized_bits(mantissa: u32) -> u32 {
        let result = polynomial(mantissa);
        let bucket = (mantissa >> 15) as usize;
        let first = usize::from(RCP_CORRECTION_OFFSETS[bucket]);
        let last = usize::from(RCP_CORRECTION_OFFSETS[bucket + 1]);
        let entry = bucket_find(&RCP_CORRECTIONS, first, last, (mantissa & 0x7fff) as u16);
        apply_residual(result, entry)
    }

    pub(super) fn evaluate(bits: u32, quiet_snan: bool) -> u32 {
        let sign = bits & 0x8000_0000;
        let magnitude = bits & 0x7fff_ffff;
        if magnitude > 0x7f80_0000 {
            return bits | if quiet_snan { 0x0040_0000 } else { 0 };
        }
        // Subnormal input (either sign of zero too) -> signed infinity.
        if magnitude < 0x0080_0000 {
            return sign | 0x7f80_0000;
        }
        if magnitude == 0x7f80_0000 {
            return sign;
        }
        let normalized = normalized_bits(magnitude & 0x7f_ffff);
        let exponent = (normalized >> 23) as i32 + 127 - (magnitude >> 23) as i32;
        if exponent <= 0 {
            return sign;
        }
        if exponent >= 255 {
            return sign | 0x7f80_0000;
        }
        sign | ((exponent as u32) << 23) | (normalized & 0x7f_ffff)
    }
}

// ---------------------------------------------------------------- RSQ

mod rsq {
    use super::{
        apply_residual, bucket_find, round_even, RSQ, RSQ_CORRECTIONS, RSQ_CORRECTION_OFFSETS,
    };

    /// Quarter-ULP fixed-point cubic; `index` = exponent parity (bit 23) and
    /// the 23-bit input mantissa.
    fn polynomial(index: u32) -> u32 {
        let c = RSQ[(index >> 18) as usize];
        let fraction = u64::from(index & 0x3_ffff);
        let product = u64::from(c.linear) * fraction;
        let shift = if product >= (1u64 << 37) { 14 } else { 13 };
        let linear = round_even(product, shift) << (shift - 13);
        let inner = (u64::from(c.quadratic) << 18) - u64::from(c.cubic) * fraction;
        let square_input = u128::from(fraction & !1);
        let quadratic = ((u128::from(inner) * square_input * square_input) >> 51) as u64;
        0x3e80_0000
            + ((u64::from(c.constant) * 32)
                .wrapping_sub(linear)
                .wrapping_add(quadratic * 4 + 64)
                >> 7) as u32
    }

    fn normalized_bits(index: u32) -> u32 {
        let result = polynomial(index);
        let bucket = (index >> 15) as usize;
        let first = usize::from(RSQ_CORRECTION_OFFSETS[bucket]);
        let last = usize::from(RSQ_CORRECTION_OFFSETS[bucket + 1]);
        let entry = bucket_find(&RSQ_CORRECTIONS, first, last, (index & 0x7fff) as u16);
        apply_residual(result, entry)
    }

    pub(super) fn evaluate(input: u32, quiet_snan: bool) -> u32 {
        let magnitude = input & 0x7fff_ffff;
        let sign = input & 0x8000_0000;
        if magnitude > 0x7f80_0000 {
            return input | if quiet_snan { 0x0040_0000 } else { 0 };
        }
        // Signed zero and flushed subnormals -> signed infinity.
        if magnitude < 0x0080_0000 {
            return sign | 0x7f80_0000;
        }
        if sign != 0 {
            return 0xffc0_0000;
        }
        if magnitude == 0x7f80_0000 {
            return 0;
        }
        let exponent = (magnitude >> 23) as i32 - 127;
        let parity = (exponent & 1) as u32;
        let normalized = normalized_bits((parity << 23) | (magnitude & 0x7f_ffff));
        // `exponent - parity` is even, so the division is exact.
        let result_exponent = (normalized >> 23) as i32 - (exponent - parity as i32) / 2;
        ((result_exponent as u32) << 23) | (normalized & 0x7f_ffff)
    }
}

// ---------------------------------------------------------------- SQRT

mod sqrt {
    use super::{SQRT, SQRT_CORRECTIONS};

    /// Sparse +-1 ULP residual lookup keyed by `index << 1`.
    fn correct(index: u32, result: u32) -> u32 {
        let key = index << 1;
        let at = SQRT_CORRECTIONS.partition_point(|&e| (e & !1) < key);
        match SQRT_CORRECTIONS.get(at) {
            Some(&e) if (e & !1) == key => {
                if e & 1 != 0 { result.wrapping_add(1) } else { result.wrapping_sub(1) }
            }
            _ => result,
        }
    }

    /// Staged cubic over exponent parity and the 23-bit mantissa (24-bit `index`).
    fn normalized_bits(index: u32) -> u32 {
        let c = SQRT[(index >> 19) as usize];
        let fraction = i64::from(index & 0x7_ffff);
        let product = u64::from(c.linear) * fraction as u64;
        let bias = if product >= (1u64 << 40) { 1u64 << 16 } else { 1u64 << 15 };
        let linear = ((product + bias) >> 17) as i64;
        let square_input = fraction & !3;
        let square = (square_input * square_input) >> 14;
        let inner = (i64::from(c.quadratic) << 8) + ((256 - i64::from(c.cubic) * fraction) >> 10);
        let quadratic = inner * square;
        let negated = if quadratic >= (1i64 << 47) {
            (((1i64 << 23) - quadratic) >> 24) * 2
        } else {
            ((1i64 << 22) - quadratic) >> 23
        };
        let sum = (i64::from(c.constant) << 22) + (linear << 19) + negated * 4096;
        let quotient = sum >> 24;
        let remainder = sum & ((1i64 << 24) - 1);
        let midpoint = 1i64 << 23;
        let mantissa =
            quotient + i64::from(remainder > midpoint || (remainder == midpoint && quotient & 1 != 0));
        let result = 0x3f80_0000u32.wrapping_add(mantissa as u32);
        // Exhaustive normalised captures bound every residual to within 1/2048
        // ULP of a rounding midpoint; other inputs need no table lookup.
        if remainder < midpoint - 8192 || remainder > midpoint + 8192 {
            return result;
        }
        correct(index, result)
    }

    pub(super) fn evaluate(input: u32, quiet_snan: bool) -> u32 {
        let magnitude = input & 0x7fff_ffff;
        let sign = input & 0x8000_0000;
        if magnitude > 0x7f80_0000 {
            return input | if quiet_snan { 0x0040_0000 } else { 0 };
        }
        // Subnormal input flushes to signed zero independently of MODE.
        if magnitude < 0x0080_0000 {
            return sign;
        }
        if sign != 0 {
            return 0xffc0_0000;
        }
        if magnitude == 0x7f80_0000 {
            return input;
        }
        let exponent = (magnitude >> 23) as i32 - 127;
        let parity = (exponent & 1) as u32;
        let normalized = normalized_bits((parity << 23) | (magnitude & 0x7f_ffff));
        // `exponent - parity` is even, so the division is exact.
        let result_exponent = (normalized >> 23) as i32 + (exponent - parity as i32) / 2;
        ((result_exponent as u32) << 23) | (normalized & 0x7f_ffff)
    }
}

// ---------------------------------------------------------------- API

/// `V_EXP_F32` raw bits (`2^x`). Candidate model: see module docs.
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn exp_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "exp_f32");
    exp::evaluate(x, true)
}

/// `V_RCP_F32` raw bits. Candidate model: see module docs.
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn rcp_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "rcp_f32");
    rcp::evaluate(x, true)
}

/// `V_RCP_IFLAG_F32` raw bits: identical to [`rcp_f32`] (see module docs for
/// the TRAPSTS cause bit that this value API does not model).
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn rcp_iflag_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "rcp_iflag_f32");
    rcp::evaluate(x, true)
}

/// `V_LOG_F32` raw bits (`log2(x)`). Candidate model: see module docs.
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn log_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "log_f32");
    log::evaluate(x, true)
}

/// `V_RSQ_F32` raw bits. Candidate model: see module docs.
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn rsq_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "rsq_f32");
    rsq::evaluate(x, true)
}

/// `V_SQRT_F32` raw bits. Candidate model: see module docs.
///
/// # Panics
/// For any `arch` other than [`Arch::Gfx1151`] / [`Arch::Gfx1201`].
pub fn sqrt_f32(arch: Arch, x: u32) -> u32 {
    qualify(arch, "sqrt_f32");
    sqrt::evaluate(x, true)
}

#[cfg(test)]
mod tests {
    //! Fixtures are the raw words published in upstream's
    //! `tests/transcendental_test.cpp` at the pinned commit (captured by AMD on
    //! physical gfx1100 and gfx1201). They check the port against upstream; they
    //! are **not** this repository's own hardware measurements.
    use super::*;

    const ARCHES: [Arch; 2] = [Arch::Gfx1151, Arch::Gfx1201];
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;

    fn fnv(digest: u64, word: u32) -> u64 {
        (digest ^ u64::from(word)).wrapping_mul(FNV_PRIME)
    }

    #[test]
    fn rcp_published_words() {
        let cases: [(u32, u32); 16] = [
            (0x3f80_0000, 0x3f80_0000), (0x3f80_0001, 0x3f7f_fffe), (0x3f81_ffff, 0x3f7c_0fc3),
            (0x3f82_0000, 0x3f7c_0fc1), (0x3f83_ffff, 0x3f78_3e12), (0x3f84_0000, 0x3f78_3e10),
            (0x3fc0_0000, 0x3f2a_aaaa), (0x3fff_ffff, 0x3f00_0001), (0x3f80_2922, 0x3f7f_add6),
            (0x3f93_2edd, 0x3f5e_a262), (0x3fb0_333c, 0x3f39_f868), (0x3fff_f486, 0x3f00_05be),
            (0x0080_0000, 0x7e80_0000), (0x7f00_0000, 0x0000_0000), (0x0000_0001, 0x7f80_0000),
            (0x7fa1_2345, 0x7fe1_2345),
        ];
        for arch in ARCHES {
            for (input, want) in cases {
                for sign in [0u32, 0x8000_0000] {
                    assert_eq!(rcp_f32(arch, input | sign), want | sign, "{arch:?} {input:#x}");
                    assert_eq!(rcp_iflag_f32(arch, input | sign), want | sign);
                }
            }
        }
    }

    #[test]
    fn rsq_published_words() {
        let cases: [(u32, u32); 25] = [
            (0x3f80_0000, 0x3f80_0000), (0x3f80_0001, 0x3f7f_ffff), (0x3f83_ffff, 0x3f7c_1765),
            (0x3f84_0000, 0x3f7c_1764), (0x3f80_40c4, 0x3f7f_bf55), (0x3f80_cab3, 0x3f7f_363d),
            (0x3fbf_ffff, 0x3f51_05ec), (0x3fc0_0000, 0x3f51_05ec), (0x3fff_ffff, 0x3f35_04f3),
            (0x4000_0000, 0x3f35_04f3), (0x4000_0001, 0x3f35_04f2), (0x4003_ffff, 0x3f32_416a),
            (0x4004_0000, 0x3f32_416a), (0x403f_ffff, 0x3f13_cd3b), (0x4040_0000, 0x3f13_cd3a),
            (0x407f_ffff, 0x3f00_0000), (0x0000_0000, 0x7f80_0000), (0x8000_0000, 0xff80_0000),
            (0x0000_0001, 0x7f80_0000), (0x8000_0001, 0xff80_0000), (0xbf80_0000, 0xffc0_0000),
            (0xff80_0000, 0xffc0_0000), (0x7f80_0000, 0x0000_0000), (0x7fa1_2345, 0x7fe1_2345),
            (0xffa1_2345, 0xffe1_2345),
        ];
        for arch in ARCHES {
            for (input, want) in cases {
                assert_eq!(rsq_f32(arch, input), want, "{arch:?} {input:#x}");
            }
        }
    }

    #[test]
    fn sqrt_published_words() {
        let cases: [(u32, u32); 26] = [
            (0x3f7a_2707, 0x3f7d_0f30), (0x3f80_0001, 0x3f80_0000), (0x3f80_0005, 0x3f80_0002),
            (0x3f80_005f, 0x3f80_0030), (0x3f80_15ee, 0x3f80_0af7), (0x4005_a8f7, 0x3fb8_fa71),
            (0x3f82_3902, 0x3f81_1b48), (0x4006_e699, 0x3fb9_d5ba), (0x3f83_39a8, 0x3f81_9a42),
            (0x4006_59f1, 0x3fb9_74bf), (0x0080_0001, 0x2000_0000), (0x00ff_ffff, 0x2035_04f3),
            (0x0100_0001, 0x2035_04f4), (0x3fff_ffff, 0x3fb5_04f3), (0x407f_ffff, 0x3fff_ffff),
            (0x7e80_0001, 0x5f00_0000), (0x7f7f_ffff, 0x5f7f_ffff), (0x0000_0000, 0x0000_0000),
            (0x8000_0000, 0x8000_0000), (0x0000_0001, 0x0000_0000), (0x8000_0001, 0x8000_0000),
            (0xbf80_0000, 0xffc0_0000), (0xff80_0000, 0xffc0_0000), (0x7f80_0000, 0x7f80_0000),
            (0x7fa1_2345, 0x7fe1_2345), (0xffa1_2345, 0xffe1_2345),
        ];
        for arch in ARCHES {
            for (input, want) in cases {
                assert_eq!(sqrt_f32(arch, input), want, "{arch:?} {input:#x}");
            }
        }
    }

    #[test]
    fn log_published_words() {
        let cases: [(u32, u32); 31] = [
            (0x3f00_0000, 0xbf80_0000), (0x3f7b_ffff, 0xbcba_1fa3), (0x3f7c_0000, 0xbcba_1f74),
            (0x3f7c_0001, 0xbcba_1f46), (0x3f7d_ffec, 0xbc39_6b23), (0x3f7d_ffff, 0xbc39_643a),
            (0x3f7e_0000, 0xbc39_63dd), (0x3f7e_0001, 0xbc39_6380), (0x3f7f_f5cb, 0xb96b_a0e4),
            (0x3f7f_fffd, 0xb48a_7fae), (0x3f7f_fffe, 0xb438_aa3c), (0x3f7f_ffff, 0xb3b8_aa3c),
            (0x3f80_0000, 0x0000_0000), (0x3f80_0001, 0x3438_aa3b), (0x3f80_0005, 0x3566_d4c6),
            (0x3f80_0006, 0x358a_7faa), (0x3f80_0043, 0x3741_5204), (0x3f80_0c30, 0x3a0c_a2fa),
            (0x3f80_ffff, 0x3c37_f1ce), (0x3f81_0000, 0x3c37_f286), (0x3f81_0001, 0x3c37_f33d),
            (0x3f81_ffff, 0x3cb7_3c5a), (0x3f82_0000, 0x3cb7_3cb4), (0x3f82_0001, 0x3cb7_3d0f),
            (0x3f83_5d48, 0x3d19_508b), (0x3f83_ffff, 0x3d35_d66f), (0x3f84_0000, 0x3d35_d69c),
            (0x3f85_ffff, 0x3d87_59af), (0x3f86_0000, 0x3d87_59c5), (0x3f86_0001, 0x3d87_59db),
            (0x3fff_ffff, 0x3f7f_ffff),
        ];
        for arch in ARCHES {
            for (input, want) in cases {
                assert_eq!(log_f32(arch, input), want, "{arch:?} {input:#x}");
            }
            // Special values from the model's own early-outs (upstream source,
            // not separately captured): zero/subnormal -> -inf, negative -> NaN.
            assert_eq!(log_f32(arch, 0), 0xff80_0000);
            assert_eq!(log_f32(arch, 0x8000_0000), 0xff80_0000);
            assert_eq!(log_f32(arch, 0xbf80_0000), 0xffc0_0000);
            assert_eq!(log_f32(arch, 0x7f80_0000), 0x7f80_0000);
            assert_eq!(log_f32(arch, 0x7fa1_2345), 0x7fe1_2345);
        }
    }

    #[test]
    fn exp_published_words() {
        let cases: [(u32, u32); 20] = [
            (0x337f_ffff, 0x3f80_0000), (0x3380_0000, 0x3f80_0000), (0x3380_0001, 0x3f80_0000),
            (0xb37f_ffff, 0x3f80_0000), (0xb380_0000, 0x3f7f_ffff), (0xb380_0001, 0x3f7f_ffff),
            (0x42ff_ffff, 0x7f7f_ffa7), (0x4300_0000, 0x7f80_0000), (0xc2fc_0000, 0x0080_0000),
            (0xc2fc_0001, 0x0000_0000), (0x3f05_67ec, 0x3fb7_b03d), (0xc114_ed44, 0x3ace_cc1e),
            (0x3550_0000, 0x3f80_0004), (0x3e00_0090, 0x3f8b_95d0), (0x3e80_001e, 0x3f98_37f6),
            (0x3ec0_001a, 0x3fa5_fedc), (0x3f00_0013, 0x3fb5_04fc), (0x3f20_002a, 0x3fc5_6740),
            (0x3f40_006f, 0x3fd7_453e), (0x3f60_0022, 0x3fea_c0dc),
        ];
        for arch in ARCHES {
            for (input, want) in cases {
                assert_eq!(exp_f32(arch, input), want, "{arch:?} {input:#x}");
            }
            assert_eq!(exp_f32(arch, 0xff80_0000), 0);
            assert_eq!(exp_f32(arch, 0x7f80_0000), 0x7f80_0000);
            assert_eq!(exp_f32(arch, 0x3f80_0000), 0x4000_0000);
            assert_eq!(exp_f32(arch, 0x7fa1_2345), 0x7fe1_2345);
        }
    }

    #[test]
    fn unqualified_arch_panics() {
        for arch in [Arch::Gfx1010, Arch::Gfx1030, Arch::Gfx1100] {
            for f in [exp_f32, rcp_f32, rcp_iflag_f32, log_f32, rsq_f32, sqrt_f32] {
                let outcome = std::panic::catch_unwind(|| f(arch, 0x3f80_0000));
                assert!(outcome.is_err(), "{arch:?} must panic");
            }
        }
    }

    // The upstream FNV digests below sweep 2^23..2^24 inputs per function.
    // Run with: cargo test -p peacemaker-ir --release trans:: -- --ignored

    #[test]
    #[ignore = "16M-input sweep; run with --release"]
    fn rcp_upstream_digest() {
        let mut digest = FNV_OFFSET;
        for mantissa in 0..(1u32 << 23) {
            digest = fnv(digest, rcp_f32(Arch::Gfx1201, 0x3f80_0000 | mantissa));
        }
        assert_eq!(digest, 0xd54e_c249_9257_2df9);
    }

    #[test]
    #[ignore = "16M-input sweep; run with --release"]
    fn rsq_upstream_digest() {
        let mut digest = FNV_OFFSET;
        for index in 0..(1u32 << 24) {
            digest = fnv(digest, rsq_f32(Arch::Gfx1201, 0x3f80_0000 + index));
        }
        assert_eq!(digest, 0x010b_c79e_b6e4_8caf);
    }

    #[test]
    #[ignore = "16M-input sweep; run with --release"]
    fn sqrt_upstream_digest() {
        let mut digest = FNV_OFFSET;
        for index in 0..(1u32 << 24) {
            digest = fnv(digest, sqrt_f32(Arch::Gfx1201, 0x3f80_0000 + index));
        }
        assert_eq!(digest, 0x514d_49d3_1f12_176b);
    }

    #[test]
    #[ignore = "16M-input sweep; run with --release"]
    fn log_upstream_digest() {
        let mut digest = FNV_OFFSET;
        for bits in 0x3f00_0000u32..0x4000_0000 {
            digest = fnv(digest, log_f32(Arch::Gfx1201, bits));
        }
        assert_eq!(digest, 0xc33a_54ef_bae0_98be);
    }

    #[test]
    #[ignore = "2x16M-input sweep; run with --release"]
    fn exp_upstream_digests() {
        let expected = [0x6895_9bb6_84f4_2e5fu64, 0x88b7_5684_494d_b8d3];
        for (sign, want) in expected.into_iter().enumerate() {
            let mut digest = FNV_OFFSET;
            for index in 0..(1u32 << 24) {
                let positive = (index as f32) * f32::from_bits(0x3380_0000);
                digest = fnv(digest, exp_f32(Arch::Gfx1201, positive.to_bits() | ((sign as u32) << 31)));
            }
            assert_eq!(digest, want, "sign={sign}");
        }
    }
}
