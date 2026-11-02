// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Integer DOT arithmetic adapted from ROCm/rocm-systems PR #12120:
// https://github.com/ROCm/rocm-systems/pull/12120
// gfx11_dot2.h and gfx12_dot.h, reviewed commit 4075f62177da.
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

//! Per-output WMMA arithmetic; fragment/lane mapping belongs to the interpreter.
//! Unlike a host FP32 dot, GFX11 uses C's sign frame and one's-complement opposing
//! terms (even signed zero). GFX12 aligns product pairs independently with
//! arithmetic floor. Both round at each internal DOT step, not once at the end.
//! Raw-bit decoding preserves the measured NaN precedence and diagnostic payload.

use crate::Arch;

const EMPTY_EXP: i32 = -1024;
const FACTOR_NAN: u32 = 0xffc0_0a3d;
const INVALID_NAN: u32 = 0xffc0_0000;

#[derive(Clone, Copy, Default)]
struct Term {
    significand: u64,
    exponent: i32,
    negative: bool,
}

impl Term {
    fn leading(self) -> i32 {
        if self.significand == 0 {
            EMPTY_EXP
        } else {
            self.exponent + (63 - self.significand.leading_zeros()) as i32
        }
    }

    fn magnitude(self, grid: i32) -> u64 {
        let shift = self.exponent - grid;
        if self.significand == 0 || shift <= -64 {
            0
        } else if shift < 0 {
            self.significand >> -shift
        } else {
            self.significand << shift
        }
    }

    /// Floor-align a term to `grid`. A negative term with a discarded tail decrements by one unit. For the FP32
    /// accumulator C that decrement applies iff the tail is non-zero and `grid - exponent <= 23` (measured on gfx1201,
    /// all 497,794 unique BF16 operand/C evaluations and the F16 capture): equal to "the aligned magnitude is
    /// non-zero" for a normal C, but a negative subnormal C whose grid sits far below it still floors to -1 unit.
    fn align_floor(self, grid: i32, accumulator: bool) -> i64 {
        let magnitude = self.magnitude(grid);
        if !self.negative {
            return magnitude as i64;
        }
        let shift = self.exponent - grid;
        let tail = self.significand != 0
            && shift < 0
            && (shift <= -64 || self.significand & ((1u64 << -shift) - 1) != 0);
        -(magnitude as i64) - i64::from(tail && (!accumulator || -shift <= 23))
    }
}

#[derive(Clone, Copy)]
struct Factor {
    significand: u64,
    alignment: i32,
    negative: bool,
    nan: bool,
    infinite: bool,
}

fn factor(bits: u16, bf16: bool, gfx11: bool) -> Factor {
    let (fraction_bits, bias, infinity) = if bf16 {
        (7, 127, 0x7f80)
    } else {
        (10, 15, 0x7c00)
    };
    let magnitude = bits & 0x7fff;
    let exponent = magnitude >> fraction_bits;
    let fraction = magnitude & ((1 << fraction_bits) - 1);
    Factor {
        significand: if exponent != 0 {
            u64::from(fraction | (1 << fraction_bits))
        } else if bf16 && gfx11 {
            0
        } else {
            u64::from(fraction)
        },
        alignment: i32::from(exponent.max(1)) - bias,
        negative: bits & 0x8000 != 0,
        nan: magnitude > infinity,
        infinite: magnitude == infinity,
    }
}

fn round_even(value: u64, shift: i32) -> u64 {
    if shift <= 0 {
        return value << -shift;
    }
    if shift >= 64 {
        return 0;
    }
    let tail = value & ((1u64 << shift) - 1);
    let halfway = 1u64 << (shift - 1);
    let head = value >> shift;
    head + u64::from(tail > halfway || (tail == halfway && head & 1 != 0))
}

fn pack(units: i64, grid: i32, frame_negative: bool, gfx11: bool) -> u32 {
    if units == 0 {
        return 0;
    }
    let sign = u32::from((units < 0) != frame_negative) << 31;
    let magnitude = units.unsigned_abs();
    let top = (63 - magnitude.leading_zeros()) as i32;
    let mut exponent = top + grid;
    if !gfx11 && exponent < -126 {
        let fraction = round_even(magnitude, -149 - grid);
        return if fraction == 0 {
            0
        } else {
            sign | fraction as u32
        };
    }
    let mut significand = round_even(magnitude, top - 23);
    if significand == 0x100_0000 {
        significand >>= 1;
        exponent += 1;
    }
    // GFX11 normalizes/rounds before deciding to flush, rather than performing
    // IEEE gradual-underflow rounding and then flushing the converted result.
    if exponent < -126 {
        return 0;
    }
    if exponent > 127 {
        return sign | 0x7f80_0000;
    }
    sign | ((exponent + 127) as u32) << 23 | (significand as u32 & 0x7f_ffff)
}

fn dot_step(a: &[u16], b: &[u16], acc: u32, bf16: bool, gfx11: bool) -> u32 {
    let mut products = [Term::default(); 4];
    let mut grids = [EMPTY_EXP; 4];
    let mut positive_inf = false;
    let mut negative_inf = false;
    let mut invalid = false;
    let fraction_bits = if bf16 { 7 } else { 10 };
    for i in 0..a.len() {
        let left = factor(a[i], bf16, gfx11);
        let right = factor(b[i], bf16, gfx11);
        // Check all factors before invalid products/C: a later factor NaN has
        // precedence over an earlier infinity-times-zero.
        if left.nan || right.nan {
            return FACTOR_NAN;
        }
        invalid |=
            (left.infinite && right.significand == 0) || (right.infinite && left.significand == 0);
        if left.infinite || right.infinite {
            if left.negative != right.negative {
                negative_inf = true;
            } else {
                positive_inf = true;
            }
        }
        let significand = left.significand * right.significand;
        products[i] = Term {
            significand,
            exponent: left.alignment + right.alignment - 2 * fraction_bits,
            negative: left.negative != right.negative,
        };
        if significand != 0 {
            grids[i] = left.alignment + right.alignment - 24;
        }
    }
    if invalid || (positive_inf && negative_inf) {
        return INVALID_NAN;
    }
    if acc & 0x7fff_ffff > 0x7f80_0000 {
        return acc | 0x0040_0000;
    }
    positive_inf |= acc == 0x7f80_0000;
    negative_inf |= acc == 0xff80_0000;
    if positive_inf || negative_inf {
        return if positive_inf && negative_inf {
            INVALID_NAN
        } else if negative_inf {
            0xff80_0000
        } else {
            0x7f80_0000
        };
    }
    let acc_exponent = ((acc >> 23) & 255) as i32;
    let c = Term {
        significand: if acc_exponent != 0 {
            u64::from((acc & 0x7f_ffff) | 0x80_0000)
        } else if gfx11 {
            0
        } else {
            u64::from(acc & 0x7f_ffff)
        },
        exponent: if gfx11 {
            acc_exponent - 150
        } else {
            acc_exponent.max(1) - 150
        },
        negative: acc >> 31 != 0,
    };
    let product_grid = *grids[..a.len()].iter().max().unwrap();
    if gfx11 {
        if c.significand | products[0].significand | products[1].significand == 0 {
            return 0;
        }
        let leading = c
            .leading()
            .max(products[0].leading())
            .max(products[1].leading());
        let grid = product_grid.max(leading - 26);
        let mut total = 0;
        for term in [c, products[0], products[1]] {
            let magnitude = term.magnitude(grid) as i64;
            total += if term.negative != c.negative {
                -magnitude - 1
            } else {
                magnitude
            };
        }
        pack(total, grid, c.negative, true)
    } else {
        let grid = product_grid.max(c.leading().max(-126) - 26);
        let mut total = c.align_floor(grid, true);
        for i in (0..a.len()).step_by(2) {
            let pair_grid = grids[i].max(grids[i + 1]);
            let pair = products[i].align_floor(pair_grid, false)
                + products[i + 1].align_floor(pair_grid, false);
            total += Term {
                significand: pair.unsigned_abs(),
                exponent: pair_grid,
                negative: pair < 0,
            }
            .align_floor(grid, false);
        }
        pack(total, grid, false, false)
    }
}

fn floating_dot(arch: Arch, a: &[u16], b: &[u16], mut c: u32, bf16: bool) -> u32 {
    assert_eq!(a.len(), 16, "F16/BF16 WMMA K must be 16");
    assert_eq!(b.len(), 16, "F16/BF16 WMMA K must be 16");
    let step = match arch {
        Arch::Gfx1151 => 2,
        Arch::Gfx1201 => 4,
        _ => panic!("WMMA numerical model is not qualified for {arch:?}"),
    };
    for k in (0..16).step_by(step) {
        c = dot_step(&a[k..k + step], &b[k..k + step], c, bf16, step == 2);
    }
    c
}

/// One F16-input/F32-output WMMA dot in increasing logical K order.
/// GFX1151 executes eight DOT2 steps (flush FP32 C; one's-complement sign frame);
/// GFX1201 executes four DOT4 steps (pair grids, gradual underflow).
/// The caller supplies logical K, not VGPR order: gfx12 wave32 swaps physical
/// K bits 2 and 3. Evidence and independently checked counts are recorded in
/// `/home/kaden/qcal/release-0.4.1/pm-r2/probe/report.md`.
pub fn wmma_f32_f16(arch: Arch, a: &[u16], b: &[u16], c: u32) -> u32 {
    floating_dot(arch, a, b, c, false)
}

/// BF16-input/F32-output WMMA, with raw NaNs and per-step rounding.
/// GFX1151 flushes BF16 factors; GFX1201 preserves them. Factor NaN yields
/// `0xffc00a3d`; invalid products yield `0xffc00000`; C NaN is quieted with its
/// sign/payload preserved unless a product takes precedence.
/// Evidence: `/home/kaden/qcal/release-0.4.1/pm-r2/probe/report.md`.
pub fn wmma_f32_bf16(arch: Arch, a: &[u16], b: &[u16], c: u32) -> u32 {
    floating_dot(arch, a, b, c, true)
}

fn integer_dot<T: Copy + Into<i64>>(a: &[T], b: &[T], c: i32, clamp: bool) -> i32 {
    assert_eq!(a.len(), b.len(), "WMMA operand K mismatch");
    let mut sum = i64::from(c);
    for (&left, &right) in a.iter().zip(b) {
        sum += Into::<i64>::into(left) * Into::<i64>::into(right);
    }
    if clamp {
        sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    } else {
        sum as i32
    }
}

/// Signed/sign-decoded IU4 dot: exact integer products and one final addition
/// modulo 2^32, or final signed saturation when CLAMP is set. K=16 on GFX1151;
/// GFX1201 supports K=16 and K=32. No host signed-overflow assumptions.
/// Each nibble must already be decoded according to its own NEG_LO sign flag;
/// unsigned nibbles 0..15 also fit this API.
/// Evidence: `/home/kaden/qcal/release-0.4.1/pm-r2/probe/report.md`.
pub fn wmma_i32_iu4(arch: Arch, a: &[i8], b: &[i8], c: i32, clamp: bool) -> i32 {
    match arch {
        Arch::Gfx1151 => assert_eq!(a.len(), 16, "GFX1151 IU4 K must be 16"),
        Arch::Gfx1201 => assert!(matches!(a.len(), 16 | 32), "GFX1201 IU4 K must be 16 or 32"),
        _ => panic!("WMMA numerical model is not qualified for {arch:?}"),
    }
    integer_dot(a, b, c, clamp)
}

/// IU8 WMMA dot over decoded bytes (signed -128..127 or unsigned 0..255 according to each operand's own NEG_LO flag,
/// hence 16-bit carriers), K = 16, in physical K order. Without CLAMP the exact sum wraps modulo 2^32. GFX1201
/// receipt `rcpt/gfx1201-C-r1` (0 mismatches over 295,680 words per clamp variant): CLAMP saturates to i32 after
/// `C + products(K 0..8)` and AGAIN after adding `products(K 8..16)` (a single final saturation has 4,864
/// mismatches, per-4-K 372, per-product 1,291). GFX1151 CLAMP has no receipt and is refused.
pub fn wmma_i32_iu8(arch: Arch, a: &[i16], b: &[i16], c: i32, clamp: bool) -> Result<i32, String> {
    if !matches!(arch, Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(format!(
            "IU8 WMMA numerical model is not qualified for {arch:?}"
        ));
    }
    assert_eq!(a.len(), 16, "IU8 K must be 16");
    if !clamp {
        return Ok(integer_dot(a, b, c, false));
    }
    if arch != Arch::Gfx1201 {
        return Err(format!("IU8 WMMA CLAMP is not measured on {arch:?}"));
    }
    let half = integer_dot(&a[..8], &b[..8], c, true);
    Ok(integer_dot(&a[8..], &b[8..], half, true))
}

/// `v_dot2_f32_f16` / `v_dot2_f32_bf16` on GFX1201: exactly one two-term pair step of the WMMA FP32 accumulation
/// (`dot_step` with pair grids and the threshold-23 accumulator rule). Receipt
/// `pm-opcode-fill-numerics/rcpt/gfx1201-C-r1`: 0 mismatches over 1,321,984 full-set words x 8 MODE settings per
/// opcode (NaN payloads, infinities, denormals in and out, -0) and 99,742 words x 15 modifier variants; round and
/// denorm MODE and CLAMP have no effect. GFX1151/GFX1100 have no receipt and are refused.
pub fn dot2_f32(arch: Arch, a: [u16; 2], b: [u16; 2], c: u32, bf16: bool) -> Result<u32, String> {
    if arch != Arch::Gfx1201 {
        return Err(format!("floating-point dot2 is not qualified on {arch:?}"));
    }
    Ok(dot_step(&a, &b, c, bf16, false))
}

/// `v_dot4_i32_iu8` / `v_dot8_i32_iu4`: the ISA pseudocode sums every decoded product and `S2.i32` in a signed
/// accumulator and wraps the dword. GFX1201 probe (0 mismatches over 402,304 full-grid evaluations plus 72,304
/// per modifier set): the sum is exact and CLAMP is ONE final signed i32 saturation. GFX1151/GFX1100 have no probe
/// receipt yet, so CLAMP there is refused rather than assumed.
pub fn dot_i32(arch: Arch, a: &[i32], b: &[i32], c: i32, clamp: bool) -> Result<i32, String> {
    if !matches!(arch, Arch::Gfx1100 | Arch::Gfx1151 | Arch::Gfx1201) {
        return Err(format!("integer dot is not defined for {arch:?}"));
    }
    if clamp && arch != Arch::Gfx1201 {
        return Err(format!(
            "CLAMP on an integer dot product is not measured on {arch:?}"
        ));
    }
    Ok(integer_dot(a, b, c, clamp))
}

/// Integer SWMMAC dot over the compressed slots of one output element, in slot order (16 slots for IU8/IU4 x32, 32
/// for IU4 x64; each slot's A value already paired with its selected B value). GFX1201 receipt `rcpt/gfx1201-C-r2`
/// (0 mismatches / 38,400 words per kind): exact products; without CLAMP one wrap modulo 2^32; with CLAMP the
/// accumulator saturates to i32 after `C + products(first half of the slots)` and again after the second half
/// (groups of 1, 2, 4 and the other half-splits have 600-3,600 mismatches).
pub fn swmmac_i32(arch: Arch, a: &[i16], b: &[i16], c: i32, clamp: bool) -> i32 {
    assert_eq!(arch, Arch::Gfx1201, "SWMMAC exists on GFX1201 only");
    assert!(
        matches!(a.len(), 16 | 32),
        "integer SWMMAC has 16 or 32 compressed slots"
    );
    if !clamp {
        return integer_dot(a, b, c, false);
    }
    let half = a.len() / 2;
    let first = integer_dot(&a[..half], &b[..half], c, true);
    integer_dot(&a[half..], &b[half..], first, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Raw gfx1151 silicon words published in ROCm/rocm-systems#12056,
    // not expectations computed by this model or host IEEE arithmetic.
    #[test]
    fn published_gfx1151_cancellation_keeps_complement_deficit() {
        assert_eq!(
            dot_step(
                &[0x3c00, 0x3c00],
                &[0x3c00, 0x3c00],
                0xc000_0000,
                false,
                true
            ),
            0x3400_0000
        );
    }

    #[test]
    fn published_gfx1151_opposing_small_integer_sum_is_not_ieee() {
        assert_eq!(
            dot_step(
                &[0x4200, 0x4800],
                &[0x4800, 0x4400],
                0xbf80_0000,
                false,
                true
            ),
            0x425c_0001
        );
    }

    #[test]
    fn published_gfx1151_signed_zero_accumulator_changes_result() {
        assert_eq!(
            dot_step(
                &[0x3c00, 0x3c00],
                &[0x0000, 0x3c00],
                0x8000_0000,
                false,
                true
            ),
            0x3f80_0001
        );
        assert_eq!(
            dot_step(
                &[0x3c00, 0x3c00],
                &[0x0000, 0x3c00],
                0x0000_0000,
                false,
                true
            ),
            0x3f80_0000
        );
    }

    // Raw gfx1201 BF16 WMMA discriminator (wmma-capture-gfx1201-v2.bf16, op 0x41): A = 16 x bf16 subnormal 0x0001,
    // B = [0x4040, 0x4100, 0...], negative subnormal C. The product grid lies far below C, and hardware still floors
    // the discarded C tail (-1 unit): 0x000afffc, not the 0x000b0000 of a model that drops it.
    #[test]
    fn gfx1201_bf16_negative_subnormal_c_is_floored_when_the_grid_is_far_below() {
        let a = [0x0001u16; 16];
        let mut b = [0u16; 16];
        b[0] = 0x4040;
        b[1] = 0x4100;
        assert_eq!(
            wmma_f32_bf16(Arch::Gfx1201, &a, &b, 0x8000_0001),
            0x000a_fffc
        );
    }
}
