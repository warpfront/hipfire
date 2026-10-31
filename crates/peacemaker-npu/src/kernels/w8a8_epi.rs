// SPDX-License-Identifier: Apache-2.0
//! W8A8 direct-BF16 contract: grid15-up scales and one integer RNE.

/// Round a finite nonnegative binary32 scale upward to its 15-bit grid.
/// Zero (including negative zero) has the canonical representation `(0, 0)`.
/// Panics for negative, infinite or NaN inputs.
pub fn grid15_up(v: f32) -> (u16, i16) {
    assert!(
        v.is_finite() && v >= 0.0,
        "grid15_up requires a finite nonnegative scale"
    );
    if v == 0.0 {
        return (0, 0);
    }
    let bits = v.to_bits();
    let exp = (bits >> 23) as i32;
    let (sig, power) = if exp == 0 {
        (bits & 0x7fffff, -149)
    } else {
        ((bits & 0x7fffff) | 0x800000, exp - 150)
    };
    let l = 31 - sig.leading_zeros() as i32;
    let mut e = power + l - 14;
    let mut m = if l > 14 {
        let shift = (l - 14) as u32;
        (sig >> shift) + u32::from(sig & ((1 << shift) - 1) != 0)
    } else {
        sig << (14 - l)
    };
    if m == 32768 {
        m = 16384;
        e += 1;
    }
    (m as u16, e as i16)
}

/// Quantize one decoded weight row for the direct-BF16 contract: the grid15-up row scale
/// `ws >= absmax / 127` (NaN-ignoring absmax), then `w8 = clamp(rne(w / ws), -127, 127)` with an
/// `f32` division; a zero row gives zero codes. Returns `ws` and its `(Sm, Se)` decomposition.
pub fn w8_direct_row(w: &[f32], out: &mut [i8]) -> (f32, (u16, i16)) {
    let absmax = w.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let grid = grid15_up(((absmax as f64) / 127.0) as f32);
    let ws = (f64::from(grid.0) * 2f64.powi(i32::from(grid.1))) as f32;
    if ws == 0.0 {
        out.fill(0);
    } else {
        for (o, &x) in out.iter_mut().zip(w) {
            // Saturating, NaN-to-zero float→int like the device conversion, then clamp.
            *o = ((x / ws).round_ties_even() as i32).clamp(-127, 127) as i8;
        }
    }
    (ws, grid)
}

/// Exact `i * sm * dm * 2^(se+de)`, rounded once to IEEE BF16.
/// Includes gradual underflow, signed nonzero underflow, carries and infinity.
/// A zero integer product always returns positive zero.
pub fn bf16_direct(i: i32, sm: u16, se: i16, dm: u16, de: i16) -> u16 {
    let j = i64::from(i) * i64::from(sm) * i64::from(dm);
    round_bf16(j, i32::from(se) + i32::from(de))
}

pub(crate) fn round_bf16(j: i64, power: i32) -> u16 {
    if j == 0 {
        return 0;
    }
    let sign = if j < 0 { 0x8000 } else { 0 };
    let m = j.unsigned_abs();
    let l = 63 - m.leading_zeros() as i32;
    let exponent = l + power;
    if exponent > 127 {
        return sign | 0x7f80;
    }
    let shift = (l - 7).max(-133 - power);
    let q = if shift <= 0 {
        m << (-shift) as u32
    } else if shift >= 64 {
        0
    } else {
        let shift = shift as u32;
        let q = m >> shift;
        let rem = m & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        q + u64::from(rem > half || (rem == half && q & 1 != 0))
    };
    // The leading significand bit supplies the extra exponent unit; adding it
    // also handles q=256 (normal carry) and q=128 (subnormal -> minimum normal).
    let biased_minus_one = (shift + power + 133).max(0) as u64;
    sign | ((biased_minus_one << 7) + q).min(0x7f80) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent dyadic-distance oracle. Search adjacent positive BF16 values
    // and compare exact u128 distances on their common grid; no float rounding.
    fn oracle(i: i32, sm: u16, se: i16, dm: u16, de: i16) -> u16 {
        let j = i64::from(i) * i64::from(sm) * i64::from(dm);
        if j == 0 {
            return 0;
        }
        let sign = if j < 0 { 0x8000 } else { 0 };
        let m = u128::from(j.unsigned_abs());
        let p = i32::from(se) + i32::from(de);
        let l = 127 - m.leading_zeros() as i32;
        if l + p < -134 {
            return sign;
        }
        if l + p > 127 {
            return sign | 0x7f80;
        }
        let dyadic = |bits: u16| -> (u128, i32) {
            let e = i32::from(bits >> 7);
            if e == 0 {
                (u128::from(bits), -133)
            } else {
                (u128::from(128 + (bits & 127)), e - 134)
            }
        };
        let cmp = |bits: u16| {
            let (n, e) = dyadic(bits);
            if n == 0 {
                return std::cmp::Ordering::Less;
            }
            let nl = 127 - n.leading_zeros() as i32;
            if nl + e != l + p {
                return (nl + e).cmp(&(l + p));
            }
            let base = p.min(e);
            (n << (e - base) as u32).cmp(&(m << (p - base) as u32))
        };
        let (mut lo, mut hi) = (0u16, 0x7f80u16);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if cmp(mid).is_le() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let (a, ap) = dyadic(lo);
        // 0x7f80 is the finite carry boundary 2^128 in this distance oracle.
        let (b, bp) = if hi == 0x7f80 { (128, 121) } else { dyadic(hi) };
        let base = p.min(ap).min(bp);
        let value = m << (p - base) as u32;
        let lower = a << (ap - base) as u32;
        let upper = b << (bp - base) as u32;
        let result = match (value - lower).cmp(&(upper - value)) {
            std::cmp::Ordering::Less => lo,
            std::cmp::Ordering::Greater => hi,
            std::cmp::Ordering::Equal => {
                if lo & 1 == 0 {
                    lo
                } else {
                    hi
                }
            }
        };
        sign | result
    }

    #[test]
    fn grid_boundaries() {
        assert_eq!(grid15_up(0.0), (0, 0));
        assert_eq!(grid15_up(-0.0), (0, 0));
        assert_eq!(grid15_up(1.0), (16384, -14));
        assert_eq!(grid15_up(f32::from_bits(1)), (16384, -163));
        assert_eq!(grid15_up(f32::from_bits(0x3fffffff)), (16384, -13));
        assert_eq!(grid15_up(f32::MAX), (16384, 114));
        let mut state = 0x341d729au32;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let v = f32::from_bits(state & 0x7fffffff);
            if !v.is_finite() || v == 0.0 {
                continue;
            }
            let (m, e) = grid15_up(v);
            let s = f64::from(m) * 2f64.powi(i32::from(e));
            assert!((16384..32768).contains(&m));
            assert!(s >= f64::from(v));
            assert!(s - f64::from(v) < 2f64.powi(i32::from(e)));
        }
    }

    #[test]
    fn ties_carries_signs_and_classes() {
        for (j, p, want) in [
            (0, 0, 0),
            (257, -8, 0x3f80),
            (259, -8, 0x3f82),
            (511, -8, 0x4000),
            (-257, -8, 0xbf80),
            (1, -134, 0),
            (-1, -134, 0x8000),
            (3, -134, 2),
            (255, -134, 0x0080),
            (1, -126, 0x0080),
            (509, 119, 0x7f7e),
            (511, 119, 0x7f80),
            (-511, 119, 0xff80),
        ] {
            assert_eq!(round_bf16(j, p), want, "j={j} p={p}");
        }
        assert_eq!(
            bf16_direct(i32::MIN, 65535, i16::MAX, 65535, i16::MAX),
            0xff80
        );
        assert_eq!(bf16_direct(-1, 0, 100, 16384, 100), 0);
        assert_eq!(bf16_direct(-1, 16384, i16::MIN, 16384, i16::MIN), 0x8000);
    }

    #[test]
    fn random_exact_distance_crosscheck() {
        let mut state = 0x127ac865u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..100_000 {
            let i = (next() as i32) >> 6;
            let sm = (16384 + (next() & 16383)) as u16;
            let dm = (16384 + (next() & 16383)) as u16;
            let se = (next() % 201) as i16 - 200;
            let de = (next() % 201) as i16 - 100;
            assert_eq!(
                bf16_direct(i, sm, se, dm, de),
                oracle(i, sm, se, dm, de),
                "i={i} sm={sm} se={se} dm={dm} de={de}"
            );
        }
    }
}
