// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! CPU arithmetic oracle; no GPU/NPU access. Production contract uses finite
//! nonnegative scales and K <= 17408. Integer arithmetic uses i128 to expose
//! overflow before admitting a signed64 device state.

/// Exact widening, including binary16 subnormals.
pub fn half(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exponent = (h >> 10) & 31;
    let mantissa = (h & 1023) as u32;
    let value = match exponent {
        0 => (mantissa as f64 * 2f64.powi(-24)) as f32,
        31 => f32::from_bits(0x7f800000 | (mantissa << 13)),
        _ => f32::from_bits(((exponent as u32 + 112) << 23) | (mantissa << 13)),
    };
    f32::from_bits(value.to_bits() | sign)
}

/// BF16 nearest-even, returned exactly widened to f32.
pub fn bf(x: f32) -> f32 {
    assert!(x.is_finite());
    f32::from_bits(x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) & 0xffff0000)
}

/// Smallest power-of-two grid exponent with RNE mantissas <= 2^bits - 1.
/// Exponent bits avoid log2 boundary errors; f64 shifts/rounding are exact for
/// these f32 inputs and <=16-bit mantissas. Signed inputs are supported by this
/// primitive, although the shared device scale contract is nonnegative.
pub fn grid(x: &[f32], bits: u32) -> (Vec<i64>, i32) {
    assert!((1..=16).contains(&bits));
    assert!(x.iter().all(|v| v.is_finite()));
    let max = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
    if max == 0.0 {
        return (vec![0; x.len()], 0);
    }
    let raw = max.to_bits();
    let encoded_exponent = (raw >> 23) & 255;
    let top = if encoded_exponent == 0 {
        31 - (raw & 0x7fffff).leading_zeros() as i32 - 149
    } else {
        encoded_exponent as i32 - 127
    };
    let mut exponent = top - (bits as i32 - 1);
    let limit = (1i64 << bits) - 1;
    while (max as f64 * 2f64.powi(-exponent)).round_ties_even() > limit as f64 {
        exponent += 1;
    }
    let mantissas = x
        .iter()
        .map(|v| (*v as f64 * 2f64.powi(-exponent)).round_ties_even() as i64)
        .collect();
    (mantissas, exponent)
}

/// One exact RNE to binary32, including subnormal, carry and signed underflow.
/// No int64 -> f64 -> f32 double rounding. Integer zero is canonically +0.
pub fn round_scaled(x: i128, exponent: i32) -> f32 {
    if x == 0 {
        return 0.0;
    }
    let sign = if x < 0 { 1u32 << 31 } else { 0 };
    let magnitude = x.unsigned_abs();
    let top = 127 - magnitude.leading_zeros() as i32;
    let mut result_exponent = top + exponent;
    if result_exponent > 127 {
        return f32::from_bits(sign | 0x7f800000);
    }
    let shift = if result_exponent >= -126 {
        top - 23
    } else {
        -149 - exponent
    };
    let mut mantissa = if shift <= 0 {
        magnitude << (-shift as u32)
    } else if shift >= 128 {
        0
    } else {
        let quotient = magnitude >> shift;
        let remainder = magnitude - (quotient << shift);
        let midpoint = 1u128 << (shift - 1);
        quotient + u128::from(remainder > midpoint || (remainder == midpoint && quotient & 1 != 0))
    };
    if result_exponent < -126 {
        return f32::from_bits(sign | mantissa as u32);
    }
    if mantissa == 1 << 24 {
        mantissa >>= 1;
        result_exponent += 1;
    }
    if result_exponent > 127 {
        return f32::from_bits(sign | 0x7f800000);
    }
    f32::from_bits(sign | ((result_exponent + 127) as u32) << 23 | (mantissa as u32 & 0x7fffff))
}

pub fn fixed(sc: &[f32], d: &[f32], c: &[i32], bits: u32) -> f32 {
    assert_eq!(sc.len(), d.len());
    assert_eq!(sc.len(), c.len());
    let (s, a) = grid(sc, bits);
    let (v, b) = grid(d, bits);
    let sum: i128 = s
        .iter()
        .zip(v)
        .zip(c)
        .map(|((&s, d), &c)| s as i128 * d as i128 * c as i128)
        .sum();
    assert!(sum.abs() < (1i128 << 63));
    round_scaled(sum, a + b)
}

pub fn current(sc: &[f32], d: &[f32], c: &[i32]) -> f32 {
    assert_eq!(sc.len(), d.len());
    assert_eq!(sc.len(), c.len());
    sc.iter()
        .zip(d)
        .zip(c)
        .fold(0f32, |sum, ((&w, &a), &c)| (w * a).mul_add(c as f32, sum))
}

pub fn reference(sc: &[f32], d: &[f32], c: &[i32]) -> f64 {
    assert_eq!(sc.len(), d.len());
    assert_eq!(sc.len(), c.len());
    sc.iter()
        .zip(d)
        .zip(c)
        .map(|((&w, &a), &c)| w as f64 * a as f64 * c as f64)
        .sum()
}

/// Proposed native BF16 operands, ascending f32 FMA. AIE accumulator RNE
/// equivalence is a silicon gate, not assumed proved by this CPU function.
pub fn bf_fold(sc: &[f32], d: &[f32], c: &[i32]) -> f32 {
    sc.iter().zip(d).zip(c).fold(0f32, |sum, ((&w, &a), &c)| {
        bf(w * a).mul_add(bf(c as f32), sum)
    })
}

/// Defer scale means separately across four epochs, unchanged integer codes.
pub fn group4(sc: &[f32], d: &[f32], c: &[i32]) -> f32 {
    let mut sum = 0f32;
    for ((s, a), c) in sc.chunks(4).zip(d.chunks(4)).zip(c.chunks(4)) {
        let w = (s.iter().map(|x| *x as f64).sum::<f64>() / s.len() as f64) as f32;
        let d = (a.iter().map(|x| *x as f64).sum::<f64>() / a.len() as f64) as f32;
        sum = (w * d).mul_add(c.iter().sum::<i32>() as f32, sum);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounding_boundaries() {
        assert_eq!(
            round_scaled((1 << 24) + 1, 0).to_bits(),
            ((1 << 24) as f32).to_bits()
        );
        assert_eq!(
            round_scaled((1 << 24) + 3, 0).to_bits(),
            (((1 << 24) + 4) as f32).to_bits()
        );
        assert_eq!(round_scaled(1, -150).to_bits(), 0);
        assert_eq!(round_scaled(3, -150).to_bits(), 2);
        assert_eq!(round_scaled(-1, -150).to_bits(), 1 << 31);
        assert_eq!(round_scaled((1 << 24) - 1, -150).to_bits(), 1 << 23);
        assert_eq!(round_scaled(0, 99).to_bits(), 0);
        assert_eq!(
            round_scaled((1 << 25) - 1, 103).to_bits(),
            f32::INFINITY.to_bits()
        );
    }
    #[test]
    fn split_order_invariance() {
        let terms = [
            32767i128 * 32767 * 8192,
            -32767i128 * 32767 * 7168,
            65535,
            -19,
        ];
        let a: i128 = terms.iter().sum();
        let b: i128 = terms.iter().rev().sum();
        assert_eq!(
            round_scaled(a, -30).to_bits(),
            round_scaled(b, -30).to_bits()
        );
    }
    #[test]
    fn integer_f32_conversion_oracle() {
        let mut state = 1u64;
        for _ in 0..100000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = (state as i64) >> 10;
            assert_eq!(
                round_scaled(value as i128, 0).to_bits(),
                (value as f32).to_bits()
            );
        }
    }
    #[test]
    fn fixed_product_is_signed_i32() {
        let (s, _) = grid(&[1.0, 0.0001, -1.0], 15);
        assert!(s.iter().all(|x| x.abs() <= 32767));
        assert!(32767i64 * 32767 < i32::MAX as i64);
    }
    #[test]
    fn smallest_exponent_handles_subnormal_and_rounding_carry() {
        assert_eq!(grid(&[f32::from_bits(1)], 15), (vec![16384], -163));
        assert_eq!(grid(&[32767.0], 15), (vec![32767], 0));
        assert_eq!(grid(&[32767.5], 15), (vec![16384], 1));
    }
}
