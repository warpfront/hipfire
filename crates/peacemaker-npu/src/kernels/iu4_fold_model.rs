// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! Lane-level models of hipfire's IU4 f32 fold on the AIE2P vector unit (analysis only; no program is emitted).
//!
//! Per output and K128 epoch the GPU does two IEEE binary32 operations (`iu4`): `t = RN(f32(sc) * d)` and
//! `sum = fma(t, f32(C), sum)`. Both models below compute exactly that, element by element, using only lane-wise
//! primitives that exist on AIE2P, and charge every primitive to the VLIW slot that issues it
//! (`crates/peacemaker-npu/src/isa/gen.rs` encodings):
//! * **Mv** (one per bundle, 512-bit = 16 x i32 lanes): `vadd.32`, `vsub.32`, `vband`, `vbor`, `vlt/vge/veqz.32`
//!   (mask), `vsel.32`, `vabs_gtz.32`, `vmax/vmin`, `vmov` (vector <-> accumulator bits), `vups` (vector -> accumulator
//!   with a scalar left shift), `vconv.fp32.bf16`. There is **no** lane-variable shift, no vector clz and no xor.
//! * **St** (one per bundle): `vsrs` (accumulator -> vector, scalar right shift, rounding mode, 16 lanes),
//!   `vconv.bf16.fp32`, `vfloor.s32.bf16`.
//! * **Vec** (one per bundle, shared with the GEMM's VMAC): integer `vmul` into 64-bit accumulator lanes (16 lanes),
//!   integer accumulator `vadd/vsub`, and the fp32-accumulator `vadd.f/vsub.f/vmul.f` (bf16 inputs only).
//!
//! Costs are counted per element as `ops / lanes`; integer ops use 16 lanes, fp32 accumulator ops are reported for
//! both 16 and 32 lanes (the lane count of an fp32 `cm` operation is not established here). Elements whose operands or
//! results leave the window a fast path handles (subnormal/huge exponents, the alignment window of the integer path,
//! an fp32 intermediate that is subnormal) are **flagged** and finished by a scalar fallback; the models count them,
//! and the result of every element (fast or fallback) is compared bit for bit with `f32` / `f32::mul_add`.
//!
//! [`fold_fa`] (float-assisted) executes every primitive: integer `t`, exact integer product `Q = Mt * C`, an exact
//! two-float split of `Q * 2^et` via the magic-number conversion the GPU itself uses, then the correctly rounded
//! three-term sum `RN(hi + lo + s)` of Boldo & Melquiond (2Sum, 2Sum, round-to-odd sum, final RN add), round-to-odd
//! made from 2Sum plus an integer fix of the last bit. It ASSUMES the AIE2P fp32 accumulator add is IEEE binary32
//! round-to-nearest-even for normal results (unverified on silicon; subnormal results are flagged).
//! [`fold_int`] (integer only, no fp hardware assumption) computes the same exact value with an integer
//! align / add / normalize / round; it executes on i128 for clarity and charges each step the cost of its AIE2P
//! decomposition (variable shifts and clz as 6-stage select ladders on two-word values; see the constants).
use std::fmt;

/// Per-element slot usage (sum of `1/lanes` per issued vector op).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cost {
    pub mv: f64,
    pub st: f64,
    pub vec_int: f64,
    pub vec_fp: f64,
}

impl Cost {
    fn mv(&mut self, n: u32) {
        self.mv += f64::from(n) / 16.0;
    }
    fn st(&mut self, n: u32) {
        self.st += f64::from(n) / 16.0;
    }
    fn vi(&mut self, n: u32) {
        self.vec_int += f64::from(n) / 16.0;
    }
    /// fp32 accumulator ops, charged at 16 lanes (`vec_fp`); the 32-lane figure is `vec_fp / 2`.
    fn vf(&mut self, n: u32) {
        self.vec_fp += f64::from(n) / 16.0;
    }
    /// Bundles per element if the three slots issue in parallel and fp32 accumulator ops have `fp_lanes` lanes.
    pub fn cycles(&self, fp_lanes: u32) -> f64 {
        let vec = self.vec_int + self.vec_fp * 16.0 / f64::from(fp_lanes);
        self.mv.max(self.st).max(vec)
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mv {:.0} St {:.0} Vec-int {:.0} Vec-fp {:.0} ops/16 lanes -> {:.3} (fp 16 lanes) / {:.3} (fp 32 lanes) cycles per element",
            self.mv * 16.0, self.st * 16.0, self.vec_int * 16.0, self.vec_fp * 16.0, self.cycles(16), self.cycles(32))
    }
}

/// Amortized operand decode: `sc = S * 2^a` (`|S|` in `[2^10, 2^11)`), `d = D * 2^b` (`D` in `[2^23, 2^24)`).
/// `None` for zero / subnormal / non-finite operands (whole row or token goes to the fallback).
pub fn decode_sc(sc: u16) -> Option<(i32, i32)> {
    let (e, m) = (i32::from((sc >> 10) & 0x1f), i32::from(sc & 0x3ff));
    if e == 0 || e == 0x1f {
        return None;
    }
    let s = m | 0x400;
    Some((if sc >> 15 == 1 { -s } else { s }, e - 15 - 10))
}
pub fn decode_d(d: f32) -> Option<(i32, i32)> {
    let bits = d.to_bits();
    let e = ((bits >> 23) & 0xff) as i32;
    if e == 0 || e == 0xff || bits >> 31 == 1 {
        return None;
    }
    Some(((bits & 0x7f_ffff | 0x80_0000) as i32, e - 127 - 23))
}

/// Round-half-to-even `x / 2^sh` (the `vsrs` conv-even mode).
fn srs_rne(x: i64, sh: u32) -> i64 {
    let q = x >> sh;
    let r = x - (q << sh);
    let half = 1i64 << (sh - 1);
    if r > half || (r == half && q & 1 == 1) {
        q + 1
    } else {
        q
    }
}

/// `t = RN(sc * d)` on the integer path: `(Mt, et)` with `t = Mt * 2^et`, `|Mt|` in `[2^23, 2^25)` (the `m` case is
/// folded into the mantissa as `2 * r11`, so `et = a + b + 10` needs no per-element select).
fn t_int(c: &mut Cost, (s, a): (i32, i32), (d, b): (i32, i32)) -> (i32, i32) {
    c.vi(1); // vmul 32x16: D (i32) x S (i16) -> 64-bit accumulator lanes
    let p = i64::from(d) * i64::from(s);
    c.st(2); // two vsrs (conv-even) with shifts 10 and 11
    let (r10, r11) = (srs_rne(p, 10), srs_rne(p, 11));
    c.mv(4); // vabs, vge (mask), vadd (2*r11), vsel
    let mt = if r11.abs() >= 1 << 23 { 2 * r11 } else { r10 };
    (i32::try_from(mt).unwrap(), a + b + 10)
}

/// The GPU result for one element: `fma(RN(sc*d), C, s)`.
pub fn reference(sc: u16, d: f32, cc: i32, s: f32) -> f32 {
    (super::iu4::f16_to_f32(sc) * d).mul_add(cc as f32, s)
}

fn is_fast_f32(x: f32) -> bool {
    x == 0.0 || x.is_normal()
}

fn two_sum(c: &mut Cost, a: f32, b: f32) -> (f32, f32) {
    c.vf(6);
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// Round-to-odd `x + y`: 2Sum, then when the error is nonzero and the rounded sum's last bit is even, step its bit
/// pattern one unit toward the error (`vmov` both to vectors, `veqz`, `vband`+`veqz` for parity, two sign compares,
/// `vsel` for +-1, `vsel`+`vadd` to apply, `vmov` back: 10 Mv).
fn ro_sum(c: &mut Cost, x: f32, y: f32) -> (f32, bool) {
    let (r, e) = two_sum(c, x, y);
    c.mv(10);
    if e == 0.0 || r.to_bits() & 1 == 1 {
        return (r, is_fast_f32(r) && is_fast_f32(e));
    }
    let up = (e > 0.0) == (r > 0.0);
    let bits = if up { r.to_bits() + 1 } else { r.to_bits() - 1 };
    (f32::from_bits(bits), is_fast_f32(r) && is_fast_f32(e))
}

/// Float-assisted fold of one element. Returns `(result, fast)`; `fast = false` means the element must be recomputed
/// by the scalar fallback (the returned value is then the exact reference).
pub fn fold_fa(c: &mut Cost, sc: u16, d: f32, cc: i32, s: f32) -> (f32, bool) {
    let exact = reference(sc, d, cc, s);
    let (Some(sd), Some(dd)) = (decode_sc(sc), decode_d(d)) else {
        return (exact, false);
    };
    // C == 0 makes t*C a signed zero whose sign the split below loses (fma(t, 0, -0) = -0 for negative t): flag it
    // (veqz + mask OR, 1 Mv).
    c.mv(1);
    if cc == 0 {
        return (exact, false);
    }
    let (mt, et) = t_int(c, sd, dd);
    c.vi(1); // vmul 32x16: Mt (i32) x C (i16)
    let q = i64::from(mt) * i64::from(cc);
    c.st(2);
    c.mv(1); // vsrs floor 19 (qh), vsrs 0 no-sat (low word) + vband (ql)
    let qh = (q >> 19) as i32;
    let ql = (q & 0x7ffff) as i32;
    // Magic constants 1.5 * 2^(E + 23) for E = et + 19 (hi) and E = et (lo): vmul.f of the amortized bf16 factors
    // 2^(a+10) (per row) and 1.5 * 2^b (per token), times 2^42 / 2^23 folded into those factors.
    c.vf(2);
    let (eh, el) = (et + 19 + 23 + 127, et + 23 + 127);
    if !(1..=254).contains(&eh) || !(1..=254).contains(&el) {
        return (exact, false);
    }
    let (mh, ml) = ((eh as u32) << 23 | 0x40_0000, (el as u32) << 23 | 0x40_0000);
    c.mv(6); // vmov x2 (magic bits to vectors), vadd.32 x2, vmov x2 (back to accumulators)
    let xh = f32::from_bits((mh as i32).wrapping_add(qh) as u32);
    let xl = f32::from_bits((ml as i32).wrapping_add(ql) as u32);
    c.vf(2);
    let (hi, lo) = (xh - f32::from_bits(mh), xl - f32::from_bits(ml));
    debug_assert_eq!(f64::from(hi) + f64::from(lo), q as f64 * 2f64.powi(et));
    // Correctly rounded RN(hi + lo + s) (Boldo & Melquiond, round-to-odd).
    let (uh, ul) = two_sum(c, lo, s);
    let (th, tl) = two_sum(c, hi, uh);
    let (v, ro_fast) = ro_sum(c, tl, ul);
    c.vf(1);
    let z = th + v;
    let fast =
        ro_fast && [hi, lo, uh, ul, th, tl, z].iter().all(|&x| is_fast_f32(x)) && is_fast_f32(s);
    (if fast { z } else { exact }, fast)
}

/// Integer-path costs of the steps that are not a single AIE2P primitive (see the module doc).
/// One stage of a lane-variable shift ladder on a two-word (64-bit) value: bit test (`vband`, `veqz`), constant shift
/// of both words with the carry word (`vups`+`vsrs` per word, `vbor` to merge), select of both words.
const LADDER_STAGE: (u32, u32) = (7, 2); // (Mv, St)
/// One clz stage: compare the high word against the stage threshold, select-shift both words (as above) and select
/// the running count.
const CLZ_STAGE: (u32, u32) = (9, 2);
/// Fast-path alignment window `delta = es - et`.
pub const INT_DELTA: std::ops::RangeInclusive<i32> = -8..=24;

/// Integer-only fold of one element (see the module doc). `s` is the running f32 sum; the state a real kernel keeps
/// is `(ms, es)`, unpacked once per output, so the unpack/pack is amortized and not charged.
pub fn fold_int(c: &mut Cost, sc: u16, d: f32, cc: i32, s: f32) -> (f32, bool) {
    let exact = reference(sc, d, cc, s);
    let (Some(sd), Some(dd)) = (decode_sc(sc), decode_d(d)) else {
        return (exact, false);
    };
    if !is_fast_f32(s) {
        return (exact, false);
    }
    let (mt, et) = t_int(c, sd, dd);
    c.vi(1); // Q = Mt * C
    let q = i128::from(mt) * i128::from(cc);
    // s = ms * 2^es (ms 24-bit signed, zero handled as ms = 0, es = et: veqz + vsel x2).
    let (ms, es) = if s == 0.0 {
        (0i128, et)
    } else {
        let bits = s.to_bits();
        let m = i128::from(bits & 0x7f_ffff | 0x80_0000);
        (
            if bits >> 31 == 1 { -m } else { m },
            ((bits >> 23) & 0xff) as i32 - 150,
        )
    };
    c.mv(3);
    c.mv(3); // delta = es - et, bias add + unsigned compare (window flag)
    let delta = es - et;
    if !INT_DELTA.contains(&delta) {
        return (exact, false);
    }
    // Frame L = et - 8: z * 2^-L = Q * 2^8 + ms * 2^(delta + 8), the second a lane-variable left shift (6 stages).
    c.mv(2);
    c.st(2); // constant shift of Q by 8 (two words)
    c.mv(6 * LADDER_STAGE.0);
    c.st(6 * LADDER_STAGE.1);
    let l = et - 8;
    let z = (q << 8) + (ms << (delta + 8));
    c.mv(4); // two-word add with carry
    if z == 0 {
        c.mv(2);
        // Exact zero: RN(+0 + -0) etc. follows IEEE: x + (-x) = +0 under RNE; reference already encodes the sign.
        return (exact, false);
    }
    // Normalize |z| so its MSB lands at bit 62, then one constant RNE shift to 24 bits.
    c.mv(4); // two-word abs
    c.mv(6 * CLZ_STAGE.0);
    c.st(6 * CLZ_STAGE.1);
    let mag = z.unsigned_abs();
    let len = 128 - mag.leading_zeros() as i32; // bit length
    if len > 63 {
        return (exact, false);
    }
    let norm = (mag << (63 - len)) as i64; // MSB at bit 62
    c.st(1); // vsrs conv-even by 39
    let mut m = srs_rne(norm, 39);
    let mut e = l + len - 24;
    c.mv(3); // mantissa overflow 2^24 -> 2^23, exponent + 1
    if m == 1 << 24 {
        m >>= 1;
        e += 1;
    }
    c.mv(3); // exponent update, sign re-apply
    let biased = e + 23 + 127;
    if !(1..=254).contains(&biased) {
        return (exact, false);
    }
    let bits =
        (if z < 0 { 1u32 << 31 } else { 0 }) | (biased as u32) << 23 | (m as u32 & 0x7f_ffff);
    (f32::from_bits(bits), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn int(&mut self, lo: i32, hi: i32) -> i32 {
            lo + (self.next() % (hi - lo + 1) as u64) as i32
        }
    }

    /// Realistic hipfire operands: f16 scale in [2^-10, 2^-3), d in [2^-17, 2^2), C from 128 int4 products.
    fn realistic(r: &mut Rng) -> (u16, f32, i32) {
        let sc = ((r.int(5, 11) as u16) << 10)
            | (r.next() as u16 & 0x3ff)
            | if r.next() % 8 == 0 { 0x8000 } else { 0 };
        let d = f32::from_bits(((r.int(110, 129) as u32) << 23) | (r.next() as u32 & 0x7f_ffff));
        let cc: i32 = (0..128).map(|_| (r.int(0, 15) - 8) * r.int(-8, 7)).sum();
        (sc, d, cc)
    }

    /// Runs `fold` over whole outputs (40 epochs, the 27B K = 5120 chain) and over adversarial single steps; every
    /// element must equal the GPU reference bit for bit. Returns (cost per element over fast elements, fallback rate).
    fn exercise(name: &str, fold: fn(&mut Cost, u16, f32, i32, f32) -> (f32, bool)) -> (Cost, f64) {
        let mut r = Rng(0x9e37_79b9_7f4a_7c15);
        let (mut cost, mut fast, mut total) = (Cost::default(), 0u64, 0u64);
        for _ in 0..20_000 {
            let mut s = 0.0f32;
            for _ in 0..40 {
                let (sc, d, cc) = realistic(&mut r);
                let mut c = Cost::default();
                let (z, f) = fold(&mut c, sc, d, cc, s);
                let want = reference(sc, d, cc, s);
                assert_eq!(
                    z.to_bits(),
                    want.to_bits(),
                    "{name}: sc {sc:#06x} d {d:e} C {cc} s {s:e}"
                );
                total += 1;
                if f {
                    fast += 1;
                    cost.mv += c.mv;
                    cost.st += c.st;
                    cost.vec_int += c.vec_int;
                    cost.vec_fp += c.vec_fp;
                }
                s = want;
            }
        }
        // Adversarial single steps: cancellation (s = -t*C nearly), ties, extreme C, s near t*C ulps, subnormals,
        // huge and tiny scales, zero.
        let (mut adversarial, mut adversarial_fast) = (0u64, 0u64);
        for i in 0..400_000u64 {
            let (sc, d, cc) = realistic(&mut r);
            let t = super::super::iu4::f16_to_f32(sc) * d;
            let p = f64::from(t) * f64::from(cc);
            let s = match i % 8 {
                0 => -(p as f32),
                1 => {
                    -(p as f32)
                        + f32::from_bits(r.next() as u32 & 0x007f_ffff)
                            * if r.next() & 1 == 0 { 1.0 } else { -1.0 }
                }
                2 => (p as f32) * f32::from_bits(0x3f80_0000 + (r.next() as u32 % 64)),
                3 => f32::from_bits(r.next() as u32 & 0x807f_ffff), // subnormal / zero
                4 => f32::from_bits(r.next() as u32 & !0x4000_0000), // any finite magnitude
                5 => -(p as f32) * 0.5f32.powi(r.int(0, 30)),
                6 => 0.0,
                _ => (p as f32) * 2f32.powi(r.int(-40, 40)),
            };
            let cc = match i % 5 {
                0 => 8192,
                1 => -7168,
                2 => r.int(-3, 3),
                _ => cc,
            };
            let sc = if i % 97 == 0 {
                0x0001 + (r.next() as u16 & 0x3ff)
            } else {
                sc
            }; // subnormal f16 scale
            let d = if i % 89 == 0 {
                f32::from_bits(r.next() as u32 & 0x007f_ffff)
            } else {
                d
            }; // subnormal d
            if !s.is_finite() {
                continue;
            }
            let mut c = Cost::default();
            let (z, f) = fold(&mut c, sc, d, cc, s);
            let want = reference(sc, d, cc, s);
            assert_eq!(
                z.to_bits(),
                want.to_bits(),
                "{name} adversarial {i}: sc {sc:#06x} d {d:e} C {cc} s {s:e}"
            );
            adversarial += 1;
            adversarial_fast += u64::from(f);
        }
        // Exact ties of the final rounding: t = 1 + j*2^-23 (sc = 1, d = t), small C, s = +-2^(24+n), so t*C + s
        // falls on or next to a midpoint of the binade of s; both parities and signs.
        let mut ties = 0u64;
        for j in 0..64u32 {
            for cc in [1, 2, 3, 5, -1, -2, -3, 4097, -4097] {
                for n in 0..12 {
                    for sign in [1.0f32, -1.0] {
                        let d = f32::from_bits(0x3f80_0000 + j);
                        let s = sign * 2f32.powi(24 + n)
                            + if j % 2 == 0 {
                                0.0
                            } else {
                                sign * 2f32.powi(n + 1)
                            };
                        let mut c = Cost::default();
                        let (z, f) = fold(&mut c, 0x3c00, d, cc, s);
                        let want = reference(0x3c00, d, cc, s);
                        assert_eq!(
                            z.to_bits(),
                            want.to_bits(),
                            "{name} tie: d {d:e} C {cc} s {s:e}"
                        );
                        ties += u64::from(f);
                    }
                }
            }
        }
        let n = fast as f64;
        let per = Cost {
            mv: cost.mv / n,
            st: cost.st / n,
            vec_int: cost.vec_int / n,
            vec_fp: cost.vec_fp / n,
        };
        let rate = 1.0 - fast as f64 / total as f64;
        println!("{name}: {total} chained ({fast} fast path) + {adversarial} adversarial ({adversarial_fast} fast path) + \
            {ties} fast-path tie elements bit-exact; realistic fallback rate {:.4}%; {per}", rate * 100.0);
        (per, rate)
    }

    #[test]
    fn float_assisted_fold_is_bit_exact() {
        // The realistic fallback is C == 0 (about 1 in 600 random int4 epochs).
        let (c, rate) = exercise("fold_fa", fold_fa);
        assert!(rate < 1e-2, "fallback rate {rate}");
        assert!(c.mv > 0.0);
    }

    #[test]
    fn integer_fold_is_bit_exact() {
        // Exactness only: the fallback rate is a measured property of the alignment window and is reported.
        let (c, _) = exercise("fold_int", fold_int);
        assert!(c.mv > 0.0);
    }
}
