// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Matrix and packed-math semantics (CDNA3 ISA §6.7, §7.1, §7.3).
//!
//! MFMA layouts follow §7.1.4 p.46–48, including H=1 for F64 (§7.1.3.4).
//! Accumulation models are explicit numerical contracts, not a claim that an
//! arbitrary CPU reduction reproduces the undocumented matrix-core arithmetic.
//! [UNVERIFIED: confirm on MI300X] NaN payloads/signs and signed-zero results.
//!
//! Non-RNE packed VALU modes, undefined MFMA broadcasts, MFMA FP16_OVFL,
//! DOT OPSEL, and packed32 inline sources (whose two-item materialization is
//! not specified here) return Unsupported. MFMA ignores EXEC and round MODE
//! (CDNA3 ISA §7.1.2 Table 28 p.42); no inactive-lane reduction is invented.
use super::{unsupported, Config, Flow, SimError, SimResult, Ty, Wave};
use crate::arch::Arch;
use crate::enc::vop3p::{Mai, MaiOp, MfmaShape, MfmaType, Vop3p, Vop3pOp};
use crate::float::{self, F8};
use crate::inst::Inst;
use crate::operand::Src;

/// Explicit F32/F64 reductions, with RNE at each model's output boundary.
///
/// CDNA3 ISA §7.1.4 p.46 specifies the mathematical product, not its reduction
/// tree. AMD's [matrix-core notes](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
/// describe the layout but do not pin arithmetic. [arXiv:2602.10262 §5](https://arxiv.org/html/2602.10262v1)
/// measures throughput/latency, NOT rounding order. Numerical characterization
/// [arXiv:2609.14845 §4.3](https://arxiv.org/html/2609.14845v1) and
/// [MMA-Sim §IV](https://arxiv.org/html/2511.10909v1) report type-dependent fused
/// groups and limited-precision, asymmetric alignment; none of the three ideal
/// RNE models below is a documented hardware guarantee.
/// [UNVERIFIED: confirm on MI300X] Probe cancellation, half-ULP ties, group
/// boundaries, odd/even FP8 products, subnormals and overflow for each opcode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AccumModel {
    /// Type-dependent CDNA3 candidate derived from MMA-Sim `amd/sim.py`,
    /// `arithmetic/fdpa.py::{tr_fdpa,gtr_fdpa}` and `arithmetic/fma.py` at
    /// [commit ce75698](https://github.com/microsoft/MMA-Sim/blob/ce75698d88a48e999a7284dd614994aa8a2caa19/src/mmasim/mmasim/arithmetic/fdpa.py).
    /// F32/F64: serial FMA. F16/BF16: groups of EIGHT (not four).
    /// FP8/BF8: groups of sixteen with separate even/odd partial sums.
    /// Products truncate to 24 fractional alignment bits; partial sums
    /// realign with RD24; C aligns with RD24, the product sum with RD32.
    /// Normalize (subnormal-aware), RD31, then RNE to F32.
    /// FP8 discards C completely when its exponent gap exceeds 25.
    /// See arXiv:2609.14845 §4.3.1/4.3.4, Algorithms 1/2; these details are
    /// independently pinned by the associated MMA-Sim source. Default because
    /// it has opcode/type-specific numerical evidence, unlike ideal RNE sums.
    /// [UNVERIFIED: confirm on MI300X] Confirm groups, alignment exponents,
    /// truncation versus floor, C-discard threshold, RD31 and special values.
    #[default]
    Cdna3Fused,
    /// One RNE of the exact sum, with no intermediate overflow/underflow.
    Exact,
    /// Ascending k, one correctly rounded fused multiply-add per product.
    Serial,
    /// Exact C plus each consecutive group, rounded once per group.
    /// Zero denotes a one-product group (exec rejects it as an invalid config).
    Grouped(u8),
}

// Kulisch-style positive/negative fixed-point sums. F32 needs 558 bits (including
// 32 products); F64 needs 4202. The least bit is the smallest possible product,
// so neither exponent gaps nor cancellation discard ANY finite input bits.
// Sparse insertion touches at most three words plus carry; no heap allocation.
struct Wide<const N: usize, const BASE: i32> {
    pos: [u64; N],
    neg: [u64; N],
    nan: bool,
    inf: u8,
    all_negative_zero: bool,
    nonzero: bool,
}
type Wide32 = Wide<10, -298>;
type Wide64 = Wide<68, -2148>;

fn parts(x: f64) -> (bool, u64, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 2047) as i32;
    let mut mant = bits & ((1u64 << 52) - 1);
    let mut power = if exp == 0 { -1074 } else { exp - 1023 - 52 };
    if exp != 0 {
        mant |= 1u64 << 52;
    }
    if mant != 0 {
        let z = mant.trailing_zeros();
        mant >>= z;
        power += z as i32;
    }
    (bits >> 63 != 0, mant, power)
}

impl<const N: usize, const BASE: i32> Wide<N, BASE> {
    fn new() -> Self {
        Self {
            pos: [0; N],
            neg: [0; N],
            nan: false,
            inf: 0,
            all_negative_zero: true,
            nonzero: false,
        }
    }
    fn add_word(words: &mut [u64; N], mut at: usize, mut value: u64) {
        while value != 0 {
            let (v, carry) = words[at].overflowing_add(value);
            words[at] = v;
            value = u64::from(carry);
            at += 1;
        }
    }
    fn term(&mut self, negative: bool, mant: u128, power: i32) {
        if mant == 0 {
            self.all_negative_zero &= negative;
            return;
        }
        self.nonzero = true;
        let shift = (power - BASE) as usize;
        let index = shift / 64;
        let bit = shift % 64;
        let words = if negative {
            &mut self.neg
        } else {
            &mut self.pos
        };
        let lo = mant as u64;
        let hi = (mant >> 64) as u64;
        Self::add_word(words, index, lo << bit);
        if bit == 0 {
            Self::add_word(words, index + 1, hi);
        } else {
            Self::add_word(words, index + 1, (lo >> (64 - bit)) | (hi << bit));
            Self::add_word(words, index + 2, hi >> (64 - bit));
        }
    }
    fn value(&mut self, x: f64) {
        if x.is_nan() {
            self.nan = true;
        } else if x.is_infinite() {
            self.inf |= if x.is_sign_negative() { 2 } else { 1 };
        } else {
            let (s, m, e) = parts(x);
            self.term(s, m as u128, e);
        }
    }
    fn product(&mut self, a: f64, b: f64) {
        if a.is_nan()
            || b.is_nan()
            || (a == 0.0 && b.is_infinite())
            || (b == 0.0 && a.is_infinite())
        {
            self.nan = true;
        } else if a.is_infinite() || b.is_infinite() {
            self.inf |= if a.is_sign_negative() ^ b.is_sign_negative() {
                2
            } else {
                1
            };
        } else {
            let (sa, ma, ea) = parts(a);
            let (sb, mb, eb) = parts(b);
            self.term(sa ^ sb, u128::from(ma) * u128::from(mb), ea + eb);
        }
    }
    /// IEEE encoding at the requested precision, retaining a guard bit and
    /// sticky OR of every discarded bit, including subnormal halfway cases.
    fn rounded(mut self, precision: usize, emin: i32, emax: i32, ebits: usize) -> u64 {
        let fraction = precision - 1;
        let sign_bit = fraction + ebits;
        let inf = ((1u64 << ebits) - 1) << fraction;
        if self.nan || self.inf == 3 {
            return inf | (1u64 << (fraction - 1));
        }
        if self.inf != 0 {
            return inf | (u64::from(self.inf == 2) << sign_bit);
        }
        let order = (0..N)
            .rev()
            .find_map(|i| (self.pos[i] != self.neg[i]).then(|| self.pos[i].cmp(&self.neg[i])));
        let Some(order) = order else {
            return u64::from(!self.nonzero && self.all_negative_zero) << sign_bit;
        };
        let negative = order == std::cmp::Ordering::Less;
        if negative {
            std::mem::swap(&mut self.pos, &mut self.neg);
        }
        let mut borrow = false;
        for i in 0..N {
            let (v, b1) = self.pos[i].overflowing_sub(self.neg[i]);
            let (v, b2) = v.overflowing_sub(u64::from(borrow));
            self.pos[i] = v;
            borrow = b1 || b2;
        }
        let top_word = (0..N).rev().find(|&i| self.pos[i] != 0).unwrap();
        let top = top_word * 64 + 63 - self.pos[top_word].leading_zeros() as usize;
        let shift = (top as i32 - fraction as i32).max(emin - fraction as i32 - BASE) as usize;
        let i = shift / 64;
        let bit = shift % 64;
        let mut significand = self.pos[i] >> bit;
        if bit != 0 && i + 1 < N {
            significand |= self.pos[i + 1] << (64 - bit);
        }
        if shift != 0 {
            let guard = shift - 1;
            let sticky = self.pos[..guard / 64].iter().any(|&v| v != 0)
                || (self.pos[guard / 64] & ((1u64 << (guard % 64)) - 1)) != 0;
            let half = (self.pos[guard / 64] >> (guard % 64)) & 1 != 0;
            if half && (sticky || significand & 1 != 0) {
                significand += 1;
            }
        }
        let mut exponent = BASE + shift as i32 + fraction as i32;
        if significand >= 1u64 << precision {
            significand >>= 1;
            exponent += 1;
        }
        let sign = u64::from(negative) << sign_bit;
        if exponent > emax {
            return sign | inf;
        }
        if significand < 1u64 << fraction {
            return sign | significand;
        }
        sign | (((exponent - emin + 1) as u64) << fraction)
            | (significand & ((1u64 << fraction) - 1))
    }
    fn f32(self) -> f32 {
        f32::from_bits(self.rounded(24, -126, 127, 8) as u32)
    }
    fn f64(self) -> f64 {
        f64::from_bits(self.rounded(53, -1022, 1023, 11))
    }
    fn f16(self) -> u16 {
        self.rounded(11, -14, 15, 5) as u16
    }
}

/// Pure per-element F32 reduction reused by MFMA and kernel CPU references.
/// `input_type` describes A/B precision, not C/D. Operands must already be
/// decoded; BF16/F16/FP8 conversions are exact. Lengths must agree.
/// [UNVERIFIED: confirm on MI300X] Every model's hardware correspondence.
pub fn accumulate(model: AccumModel, input_type: MfmaType, c: f32, a: &[f32], b: &[f32]) -> f32 {
    accumulate_typed(model, input_type, input_type, c, a, b)
}

/// Typed mixed-precision counterpart of [`accumulate`]. Both operand types
/// matter for the candidate's raw subnormal exponents (FP8 -7, BF8 -15).
pub fn accumulate_typed(
    model: AccumModel,
    a_type: MfmaType,
    b_type: MfmaType,
    c: f32,
    a: &[f32],
    b: &[f32],
) -> f32 {
    assert_eq!(a.len(), b.len(), "MFMA reduction operand lengths differ");
    if model == AccumModel::Serial || (model == AccumModel::Cdna3Fused && a_type == MfmaType::F32) {
        return a.iter().zip(b).fold(c, |sum, (&x, &y)| x.mul_add(y, sum));
    }
    if model == AccumModel::Cdna3Fused {
        let fp8 = matches!(a_type, MfmaType::Fp8 | MfmaType::Bf8);
        let group = if fp8 { 16 } else { 8 };
        return a
            .chunks(group)
            .zip(b.chunks(group))
            .fold(c, |sum, (a, b)| cdna_group(a_type, b_type, sum, a, b, fp8));
    }
    let group = match model {
        AccumModel::Grouped(g) => usize::from(g.max(1)),
        _ => a.len().max(1),
    };
    let mut sum = c;
    for (a, b) in a.chunks(group).zip(b.chunks(group)) {
        let mut wide = Wide32::new();
        wide.value(f64::from(sum));
        for (&a, &b) in a.iter().zip(b) {
            wide.product(f64::from(a), f64::from(b));
        }
        sum = wide.f32();
    }
    sum
}

/// Pure F64 reduction; the Exact model also retains all 106 product bits.
/// CDNA3 ISA §7.3 p.52 fixes F64 to RNE with input/output subnormals enabled.
/// [UNVERIFIED: confirm on MI300X] Reduction order and NaN bit patterns.
pub fn accumulate_f64(model: AccumModel, c: f64, a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "MFMA reduction operand lengths differ");
    if matches!(model, AccumModel::Serial | AccumModel::Cdna3Fused) {
        return a.iter().zip(b).fold(c, |sum, (&x, &y)| x.mul_add(y, sum));
    }
    let group = match model {
        AccumModel::Grouped(g) => usize::from(g.max(1)),
        _ => a.len().max(1),
    };
    let mut sum = c;
    for (a, b) in a.chunks(group).zip(b.chunks(group)) {
        let mut wide = Wide64::new();
        wide.value(sum);
        for (&a, &b) in a.iter().zip(b) {
            wide.product(a, b);
        }
        sum = wide.f64();
    }
    sum
}

fn minimum_exponent(ty: MfmaType) -> i32 {
    match ty {
        MfmaType::F16 => -14,
        MfmaType::Fp8 => -7,
        MfmaType::Bf8 => -15,
        _ => -126,
    }
}
fn raw_exponent(value: f64, minimum: i32) -> i32 {
    if value == 0.0 {
        minimum
    } else {
        (((value.to_bits() >> 52) & 2047) as i32 - 1023).max(minimum)
    }
}
fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}
fn quantize(value: f64, exponent: i32, floor: bool) -> f64 {
    if value == 0.0 {
        return value;
    }
    let scaled = value * pow2(-exponent);
    (if floor {
        scaled.floor()
    } else {
        scaled.trunc()
    }) * pow2(exponent)
}
/// Exact integer-quantum implementation of MMA-Sim TR/GTR-FDPA. F64 is used
/// only as a container: decoded products have <=22 bits, each aligned sum
/// <=29 bits, and the final RD32/RD24 sum <=38 bits. Thus no host F64 operation
/// loses a finite bit. This is NOT the Exact model (which needs a wide sum).
/// Sources: arXiv:2609.14845 §4.3 Algorithms 1/2 and MMA-Sim
/// `src/mmasim/mmasim/arithmetic/fdpa.py`, `helper.py::frexp_and_normalize`.
/// [UNVERIFIED: confirm on MI300X] Including canonical NaN = 0x7fffffff.
fn cdna_group(a_type: MfmaType, b_type: MfmaType, c: f32, a: &[f32], b: &[f32], fp8: bool) -> f32 {
    let mut products = [0f64; 16];
    let mut exponents = [-999i32; 16];
    let mut nonfinite = !c.is_finite();
    for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
        let (a, b) = (f64::from(a), f64::from(b));
        let mut product = a * b;
        // TR-FDPA product overflow is tested on magnitude, before any
        // cancellation. Partial-sum overflow within a group is allowed.
        if !fp8 && product.abs() >= pow2(128) {
            product = if product.is_sign_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        nonfinite |= !product.is_finite();
        products[index] = product;
        if product != 0.0 {
            exponents[index] = raw_exponent(a, minimum_exponent(a_type))
                + raw_exponent(b, minimum_exponent(b_type));
        }
    }
    if nonfinite {
        let mut special = Wide32::new();
        special.value(f64::from(c));
        for &product in &products[..a.len()] {
            special.value(product);
        }
        let value = special.f32();
        return if value.is_nan() {
            f32::from_bits(0x7fff_ffff)
        } else {
            value
        };
    }
    let groups = if fp8 { 2 } else { 1 };
    let mut exp = [-999; 2];
    let mut partial = [0f64; 2];
    for index in 0..a.len() {
        exp[index % groups] = exp[index % groups].max(exponents[index]);
    }
    for index in 0..a.len() {
        let group = index % groups;
        partial[group] += quantize(products[index], exp[group] - 24, false);
    }
    let dot_exp = exp[..groups].iter().copied().max().unwrap();
    let dot = if fp8 {
        quantize(partial[0], dot_exp - 24, true) + quantize(partial[1], dot_exp - 24, true)
    } else {
        partial[0]
    };
    let c_exp = raw_exponent(f64::from(c), -126);
    let maximum = dot_exp.max(c_exp);
    let dot = quantize(dot, maximum - 32, true);
    let c = if fp8 && c_exp < maximum - 25 {
        0.0
    } else {
        quantize(f64::from(c), maximum - 24, true)
    };
    let sum = dot + c;
    let normalized_exp = raw_exponent(sum, -126);
    quantize(sum, normalized_exp - 31, true) as f32
}

fn item_bits(regs: &[[u32; 64]], ty: MfmaType, item: usize, lane: usize) -> u64 {
    match ty {
        MfmaType::F64 => {
            u64::from(regs[item * 2][lane]) | (u64::from(regs[item * 2 + 1][lane]) << 32)
        }
        MfmaType::F32 | MfmaType::I32 => u64::from(regs[item][lane]),
        MfmaType::F16 | MfmaType::Bf16 => {
            u64::from((regs[item / 2][lane] >> (16 * (item % 2))) & 65535)
        }
        _ => u64::from((regs[item / 4][lane] >> (8 * (item % 4))) & 255),
    }
}
fn decoded(arch: Arch, ty: MfmaType, bits: u64) -> f64 {
    match ty {
        MfmaType::F64 => f64::from_bits(bits),
        MfmaType::F32 => f64::from(f32::from_bits(bits as u32)),
        MfmaType::F16 => f64::from(float::f16_to_f32(bits as u16)),
        MfmaType::Bf16 => f64::from(float::bf16_to_f32(bits as u16)),
        MfmaType::Fp8 | MfmaType::Bf8 => f64::from(float::f8_to_f32(
            arch.caps().fp8,
            if ty == MfmaType::Fp8 {
                F8::Fp8
            } else {
                F8::Bf8
            },
            bits as u8,
        )),
        MfmaType::I8 => f64::from(bits as u8 as i8),
        MfmaType::I32 => f64::from(bits as u32 as i32),
    }
}
fn b_lane(lane: usize, blgp: u8) -> usize {
    match blgp {
        0 => lane,
        1 => lane % 32,
        2 => lane % 32 + 32,
        3 => (lane + 16) % 64,
        4..=7 => lane % 16 + usize::from(blgp - 4) * 16,
        _ => lane,
    }
}
/// Output (item, lane), CDNA3 ISA §7.1.4.2 p.48. H=1 gives the F64
/// contiguous-row layout of §7.1.3.4 p.45–46, with two dwords per item.
fn output(s: MfmaShape, block: usize, row: usize, col: usize) -> (usize, usize) {
    let (m, n) = (usize::from(s.m), usize::from(s.n));
    let h = if s.cd == MfmaType::F64 { 1 } else { 4 };
    let bi = 64usize.div_ceil(n * m / h);
    let mi = (64 / bi) / n;
    let g = m / (h * mi);
    (
        row % h + h * (row / (h * mi) + g * (block / bi)),
        col + n * ((row / h) % mi + mi * (block % bi)),
    )
}

/// Pure gfx942 matrix reference over register/lane arrays.
///
/// Inputs and output use CDNA3 ISA §7.1.4 p.46–48. Broadcasting uses
/// §7.1.5 p.48–50; F64 instead uses BLGP as A/B/C sign toggles. This function
/// applies the lane formula even for CBSZ that is undefined on the opcode;
/// `exec` rejects such forms. Caller supplies correctly sized arrays and
/// fields (CBSZ <=4 unless F64, ABID <2^CBSZ unless F64, BLGP <=7). Sources preserve denormals;
/// `exec` handles F32 A/B MODE input flushing before calling it (§7.3 p.52).
#[allow(clippy::too_many_arguments)]
pub fn mfma_reference(
    op: MaiOp,
    a: &[[u32; 64]],
    b: &[[u32; 64]],
    c: &[[u32; 64]],
    cbsz: u8,
    abid: u8,
    blgp: u8,
    model: AccumModel,
) -> Vec<[u32; 64]> {
    reference(Arch::Gfx942, op, a, b, c, cbsz, abid, blgp, model)
}
#[allow(clippy::too_many_arguments)]
fn reference(
    arch: Arch,
    op: MaiOp,
    a: &[[u32; 64]],
    b: &[[u32; 64]],
    c: &[[u32; 64]],
    cbsz: u8,
    abid: u8,
    blgp: u8,
    model: AccumModel,
) -> Vec<[u32; 64]> {
    let Some(s) = op.shape() else {
        return Vec::new();
    };
    let (m, n, k, blocks) = (
        usize::from(s.m),
        usize::from(s.n),
        usize::from(s.k),
        usize::from(s.blocks),
    );
    let kl = k / (64 / (m * blocks));
    let mut result = vec![[0; 64]; usize::from(s.cd_dwords)];
    let mut av = [0f32; 32];
    let mut bv = [0f32; 32];
    let mut a64 = [0f64; 32];
    let mut b64 = [0f64; 32];
    // Each input item participates in many outputs; decode it once. All
    // frozen shapes have K_L <=8, including packed FP8/BF8 and I8.
    let mut decoded_a = [[0f64; 64]; 8];
    let mut decoded_b = [[0f64; 64]; 8];
    for item in 0..kl {
        for lane in 0..64 {
            decoded_a[item][lane] = decoded(arch, s.a, item_bits(a, s.a, item, lane));
            decoded_b[item][lane] = decoded(arch, s.b, item_bits(b, s.b, item, lane));
        }
    }
    for block in 0..blocks {
        for row in 0..m {
            for col in 0..n {
                let (item, lane) = output(s, block, row, col);
                let cbits = item_bits(c, s.cd, item, lane);
                let mut integer = cbits as u32;
                for ki in 0..k {
                    let mut la = row + m * (block + blocks * (ki / kl));
                    let mut lb = col + n * (block + blocks * (ki / kl));
                    if s.cd != MfmaType::F64 {
                        let size = 64 >> cbsz;
                        la = la % size + size * usize::from(abid);
                        lb = b_lane(lb, blgp);
                    }
                    let aa = decoded_a[ki % kl][la];
                    let bb = decoded_b[ki % kl][lb];
                    if s.cd == MfmaType::I32 {
                        // §7.3 p.52: signed I8 × signed I8 -> signed16 -> signed32.
                        let product = (aa as i16 * bb as i16) as i32;
                        integer = integer.wrapping_add(product as u32);
                    } else if s.cd == MfmaType::F64 {
                        a64[ki] = if blgp & 1 != 0 { -aa } else { aa };
                        b64[ki] = if blgp & 2 != 0 { -bb } else { bb };
                    } else {
                        av[ki] = aa as f32;
                        bv[ki] = bb as f32;
                    }
                }
                if s.cd == MfmaType::F64 {
                    let cc = f64::from_bits(cbits) * if blgp & 4 != 0 { -1.0 } else { 1.0 };
                    let bits = accumulate_f64(model, cc, &a64[..k], &b64[..k]).to_bits();
                    result[item * 2][lane] = bits as u32;
                    result[item * 2 + 1][lane] = (bits >> 32) as u32;
                } else {
                    result[item][lane] = if s.cd == MfmaType::I32 {
                        integer
                    } else {
                        accumulate_typed(
                            model,
                            s.a,
                            s.b,
                            f32::from_bits(cbits as u32),
                            &av[..k],
                            &bv[..k],
                        )
                        .to_bits()
                    };
                }
            }
        }
    }
    result
}
fn offset(src: Src, amount: u8) -> SimResult<Src> {
    let n = |v: u8| {
        v.checked_add(amount)
            .ok_or_else(|| SimError::Fault("register range crosses 255".into()))
    };
    Ok(match src {
        Src::Vgpr(v) => Src::Vgpr(n(v)?),
        Src::Agpr(v) => Src::Agpr(n(v)?),
        Src::Sgpr(v) => Src::Sgpr(n(v)?),
        Src::Ttmp(v) => Src::Ttmp(n(v)?),
        _ if amount == 0 => src,
        _ => return unsupported(format!("multi-register operand {src:?}")),
    })
}
fn canonical_source(src: Src) -> bool {
    src.is_agpr() || Src::from_code(src.code(), None).ok() == Some(src)
}
fn matrix_registers(w: &Wave, src: Src, width: u8) -> SimResult<&[[u32; 64]]> {
    // One read per register validates allocation and pending-load hazards;
    // return a borrowed range instead of copying all sixty-four lanes.
    for reg in 0..width {
        w.read_v(offset(src, reg)?, 0, Ty::B32)?;
    }
    let (file, base) = match src {
        Src::Vgpr(n) => (&w.v, usize::from(n)),
        Src::Agpr(n) => (&w.a, usize::from(n)),
        _ => return unsupported(format!("matrix register source {src:?}")),
    };
    Ok(&file[base..base + usize::from(width)])
}
/// MFMA forces execution for all threads, including EXEC=0, and RNE:
/// CDNA3 ISA §7.1.2 Table 28 p.42. AccVGPR moves remain lane-masked.
fn exec_mai(w: &mut Wave, i: &Mai, cfg: &Config) -> SimResult<()> {
    if i.op.opcode(w.arch).is_none() {
        return unsupported(i.op.name());
    }
    if matches!(i.op, MaiOp::AccvgprReadB32 | MaiOp::AccvgprWriteB32) {
        if i.src_b.is_some() || i.src_c.is_some() || i.cbsz != 0 || i.abid != 0 || i.blgp != 0 {
            return unsupported(format!("{} malformed move", i.op.name()));
        }
        let legal = match i.op {
            MaiOp::AccvgprReadB32 => i.dst.is_vgpr() && i.src_a.is_agpr(),
            _ => i.dst.is_agpr() && !i.src_a.is_agpr() && canonical_source(i.src_a),
        };
        if !legal {
            return unsupported(format!("{} register-file form", i.op.name()));
        }
        for lane in 0..64 {
            if w.lane_active(lane) {
                let value = w.read_v(i.src_a, lane, Ty::B32)?;
                w.write_v(i.dst, lane, value)?;
            }
        }
        return Ok(());
    }
    let s =
        i.op.shape()
            .ok_or_else(|| SimError::Unsupported(i.op.name().into()))?;
    if cfg.accum == AccumModel::Grouped(0) {
        return unsupported(format!("{} Grouped(0)", i.op.name()));
    }
    if i.blgp > 7 || i.cbsz > 7 || i.abid > 15 {
        return unsupported(format!("{} broadcast field out of range", i.op.name()));
    }
    if s.cd != MfmaType::F64
        && ((1u16 << i.cbsz) > u16::from(s.blocks) || u16::from(i.abid) >= (1u16 << i.cbsz))
    {
        return unsupported(format!(
            "{} undefined CBSZ/ABID (CDNA3 ISA §7.1.5.1 p.49)",
            i.op.name()
        ));
    }
    if s.cd != MfmaType::F64 && w.mode.fp16_ovfl {
        // Table 28 specifies saturation, but the exact point at which a
        // grouped, limited-precision reduction saturates is not pinned by
        // the cited numerical model. Do not silently compute unsaturated.
        return unsupported(format!("{} FP16_OVFL saturation timing", i.op.name()));
    }
    let b = i
        .src_b
        .ok_or_else(|| SimError::Unsupported(format!("{} missing B", i.op.name())))?;
    let c = i
        .src_c
        .ok_or_else(|| SimError::Unsupported(format!("{} missing C", i.op.name())))?;
    if !(i.src_a.is_vgpr() || i.src_a.is_agpr())
        || !(b.is_vgpr() || b.is_agpr())
        || !(i.dst.is_vgpr() || i.dst.is_agpr())
        || !(c.is_inline() || (c.is_agpr() == i.dst.is_agpr() && (c.is_vgpr() || c.is_agpr())))
    {
        return unsupported(format!("{} register-file form", i.op.name()));
    }
    if !canonical_source(c) {
        return unsupported(format!("{} illegal inline C", i.op.name()));
    }
    for (src, width) in [
        (i.dst, s.cd_dwords),
        (i.src_a, s.a_dwords),
        (b, s.b_dwords),
        (c, s.cd_dwords),
    ] {
        if width > 1 && matches!(src, Src::Vgpr(n) | Src::Agpr(n) if n & 1 != 0) {
            return unsupported(format!("{} misaligned register tuple", i.op.name()));
        }
    }
    let mut inline = [[0; 64]; 16];
    let c = if c.is_inline() {
        if s.cd == MfmaType::F64 {
            let bits = w.read_s64(c, Ty::F64)?;
            for pair in inline[..usize::from(s.cd_dwords)].chunks_exact_mut(2) {
                pair[0].fill(bits as u32);
                pair[1].fill((bits >> 32) as u32);
            }
        } else {
            let bits = w.read_s(c, Ty::B32)?;
            for register in &mut inline[..usize::from(s.cd_dwords)] {
                register.fill(bits);
            }
        }
        &inline[..usize::from(s.cd_dwords)]
    } else {
        matrix_registers(w, c, s.cd_dwords)?
    };
    let a = matrix_registers(w, i.src_a, s.a_dwords)?;
    let b = matrix_registers(w, b, s.b_dwords)?;
    matrix_registers(w, i.dst, s.cd_dwords)?;
    // Only F32 A/B obey input flushing. No copy in the preserve/default case.
    let mut flushed_a = [[0; 64]; 2];
    let mut flushed_b = [[0; 64]; 2];
    let (a, b) = if s.a == MfmaType::F32 && w.mode.denorm32 & 1 == 0 {
        flushed_a[..a.len()].copy_from_slice(a);
        flushed_b[..b.len()].copy_from_slice(b);
        for register in flushed_a.iter_mut().chain(flushed_b.iter_mut()) {
            for bits in register {
                if f32::from_bits(*bits).is_subnormal() {
                    *bits &= 0x8000_0000;
                }
            }
        }
        (&flushed_a[..a.len()], &flushed_b[..b.len()])
    } else {
        (a, b)
    };
    let out = reference(w.arch, i.op, a, b, c, i.cbsz, i.abid, i.blgp, cfg.accum);
    for (reg, values) in out.iter().enumerate() {
        for (lane, &value) in values.iter().enumerate() {
            w.write_v(offset(i.dst, reg as u8)?, lane, value)?;
        }
    }
    Ok(())
}

/// Execute one VOP3P or MAI instruction.
pub fn exec(w: &mut Wave, inst: &Inst, cfg: &Config) -> SimResult<Flow> {
    match inst {
        Inst::Mai(i) => exec_mai(w, i, cfg)?,
        Inst::Vop3p(i) => exec_packed(w, i)?,
        _ => return unsupported(format!("MFMA executor for {inst:?}")),
    }
    Ok(Flow::Next)
}

/// Packed VOP3P, CDNA3 ISA §6.7 p.38–39 and §12.10 p.254–263.
fn exec_packed(w: &mut Wave, i: &Vop3p) -> SimResult<()> {
    use Vop3pOp::*;
    if i.op.opcode(w.arch).is_none() {
        return unsupported(i.op.name());
    }
    if i.op_sel > 7 || i.op_sel_hi > 7 || i.neg_lo > 7 || i.neg_hi > 7 {
        return unsupported(format!("{} modifier field out of range", i.op.name()));
    }
    if !i.dst.is_vgpr()
        || i.src
            .iter()
            .flatten()
            .any(|src| src.is_agpr() || !canonical_source(*src))
    {
        return unsupported(format!(
            "{} illegal packed register/literal form",
            i.op.name()
        ));
    }
    let wide = matches!(i.op, PkFmaF32 | PkMulF32 | PkAddF32 | PkMovB32);
    let mix = matches!(i.op, FmaMixF32 | FmaMixloF16 | FmaMixhiF16);
    let dot = matches!(
        i.op,
        Dot2F32F16 | Dot2I32I16 | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4 | Dot8U32U4
    );
    let fp16 = matches!(i.op, PkFmaF16 | PkAddF16 | PkMulF16 | PkMinF16 | PkMaxF16);
    let float = wide && i.op != PkMovB32 || mix || fp16;
    if dot && (i.op_sel != 0 || i.op_sel_hi != 7) {
        return unsupported(format!(
            "{} OPSEL forbidden (CDNA3 ISA §6.2 p.32)",
            i.op.name()
        ));
    }
    if dot && i.op != Dot2F32F16 && (i.neg_lo != 0 || i.neg_hi != 0) {
        return unsupported(format!("{} integer dot source modifiers", i.op.name()));
    }
    if i.op == Dot2F32F16 && i.neg_hi & 4 != 0 {
        return unsupported(format!(
            "{} unmodelled NEG_HI[2] for single-dword C",
            i.op.name()
        ));
    }
    if i.op == PkMovB32 && (i.neg_lo != 0 || i.neg_hi != 0) {
        return unsupported(format!("{} unmodelled bitwise NEG fields", i.op.name()));
    }
    if !float && !dot && i.op != PkMovB32 && (i.neg_lo != 0 || i.neg_hi != 0) {
        return unsupported(format!(
            "{} unmodelled packed-integer NEG fields",
            i.op.name()
        ));
    }
    if float
        && ((wide || mix) && w.mode.round32 != 0
            || (fp16 || matches!(i.op, FmaMixloF16 | FmaMixhiF16)) && w.mode.round16_64 != 0)
    {
        return unsupported(format!("{} non-RNE MODE", i.op.name()));
    }
    // Packed32 explicitly has no output modifiers (§6.7 p.39): the CLAMP
    // encoding bit is ignored. [UNVERIFIED: confirm on MI300X]
    if wide {
        if i.src.iter().flatten().any(|src| src.is_inline()) {
            return unsupported(format!(
                "{} unmodelled packed32 inline pair materialization",
                i.op.name()
            ));
        }
        for src in std::iter::once(i.dst).chain(i.src.iter().flatten().copied()) {
            if matches!(src, Src::Vgpr(n) | Src::Sgpr(n) | Src::Agpr(n) | Src::Ttmp(n) if n & 1 != 0)
            {
                return unsupported(format!("{} misaligned packed32 pair", i.op.name()));
            }
        }
    }
    let needed = match i.op {
        PkMovB32 | PkAddF16 | PkMulF16 | PkMinF16 | PkMaxF16 | PkAddU16 | PkSubU16 | PkMulLoU16
        | PkLshlrevB16 | PkLshrrevB16 | PkMaxI16 | PkMinI16 | PkMulF32 | PkAddF32 => 2,
        _ => 3,
    };
    if i.src[..needed].iter().any(Option::is_none) || i.src[needed..].iter().any(Option::is_some) {
        return unsupported(format!("{} missing or excess source", i.op.name()));
    }
    for lane in 0..64 {
        if w.lane_active(lane) {
            let mut src = [0u64; 3];
            for (index, operand) in i.src.iter().enumerate() {
                if let Some(operand) = operand {
                    src[index] = if wide {
                        w.read_v64(*operand, lane, Ty::B64)?
                    } else {
                        u64::from(w.read_v(*operand, lane, if fp16 { Ty::F16 } else { Ty::B32 })?)
                    };
                }
            }
            if i.op == PkMovB32 {
                let lo = (src[0] >> (32 * (i.op_sel & 1))) as u32;
                let hi = (src[1] >> (32 * ((i.op_sel >> 1) & 1))) as u32;
                w.write_v64(i.dst, lane, u64::from(lo) | (u64::from(hi) << 32))?;
            } else if mix {
                let mut f = [0f32; 3];
                for index in 0..3 {
                    f[index] = if i.op_sel_hi >> index & 1 == 0 {
                        input32(src[index] as u32, w.mode.denorm32)
                    } else {
                        input16(
                            (src[index] >> (16 * ((i.op_sel >> index) & 1))) as u16,
                            w.mode.denorm16_64,
                        )
                    };
                    if i.neg_hi >> index & 1 != 0 {
                        f[index] = f[index].abs();
                    }
                    if i.neg_lo >> index & 1 != 0 {
                        f[index] = -f[index];
                    }
                }
                let value = f[0].mul_add(f[1], f[2]);
                let value = clamp_float(value, i.clamp, w.mode.dx10_clamp);
                if i.op == FmaMixF32 {
                    w.write_v(i.dst, lane, output32(value, w.mode.denorm32).to_bits())?;
                } else {
                    let bits = half_result(
                        float::f32_to_f16_rne(value),
                        w,
                        f.iter().all(|v| v.is_finite()),
                    );
                    let old = w.read_v(i.dst, lane, Ty::B32)?;
                    w.write_v(
                        i.dst,
                        lane,
                        if i.op == FmaMixloF16 {
                            old & 0xffff_0000 | u32::from(bits)
                        } else {
                            old & 0x0000_ffff | (u32::from(bits) << 16)
                        },
                    )?;
                }
            } else if dot {
                let value = dot_result(w, i, src)?;
                w.write_v(i.dst, lane, value)?;
            } else {
                let mut result = 0u64;
                for half in 0..2 {
                    let sel = if half == 0 { i.op_sel } else { i.op_sel_hi };
                    let neg = if half == 0 { i.neg_lo } else { i.neg_hi };
                    let mut v = [0u32; 3];
                    for index in 0..needed {
                        v[index] = (src[index]
                            >> ((if wide { 32 } else { 16 }) * ((sel >> index) & 1)))
                            as u32;
                        if !wide {
                            v[index] &= 65535;
                        }
                        if neg >> index & 1 != 0 {
                            v[index] ^= if wide { 0x8000_0000 } else { 0x8000 };
                        }
                    }
                    let value = if wide {
                        let a = input32(v[0], w.mode.denorm32);
                        let b = input32(v[1], w.mode.denorm32);
                        let c = input32(v[2], w.mode.denorm32);
                        let value = match i.op {
                            PkFmaF32 => a.mul_add(b, c),
                            PkMulF32 => a * b,
                            PkAddF32 => a + b,
                            _ => unreachable!(),
                        };
                        u64::from(output32(value, w.mode.denorm32).to_bits())
                    } else if fp16 {
                        let bits = if matches!(i.op, PkMinF16 | PkMaxF16) {
                            minmax_half(v[0] as u16, v[1] as u16, i.op == PkMaxF16, w)
                        } else {
                            let a = f64::from(input16(v[0] as u16, w.mode.denorm16_64));
                            let b = f64::from(input16(v[1] as u16, w.mode.denorm16_64));
                            let c = f64::from(input16(v[2] as u16, w.mode.denorm16_64));
                            let mut sum = Wide32::new();
                            match i.op {
                                PkFmaF16 => {
                                    sum.product(a, b);
                                    sum.value(c);
                                }
                                PkMulF16 => sum.product(a, b),
                                PkAddF16 => {
                                    sum.value(a);
                                    sum.value(b);
                                }
                                _ => unreachable!(),
                            }
                            sum.f16()
                        };
                        let bits = if i.clamp {
                            float::f32_to_f16_rne(clamp_float(
                                float::f16_to_f32(bits),
                                true,
                                w.mode.dx10_clamp,
                            ))
                        } else {
                            bits
                        };
                        let finite = v[..needed].iter().all(|bits| bits & 0x7c00 != 0x7c00);
                        u64::from(half_result(bits, w, finite))
                    } else {
                        u64::from(integer_half(i.op, v[0] as u16, v[1] as u16, i.clamp))
                    };
                    result |= value << (half * if wide { 32 } else { 16 });
                }
                if wide {
                    w.write_v64(i.dst, lane, result)?;
                } else {
                    w.write_v(i.dst, lane, result as u32)?;
                }
            }
        }
    }
    Ok(())
}
fn input32(bits: u32, denorm: u8) -> f32 {
    let value = f32::from_bits(bits);
    if denorm & 1 == 0 && value.is_subnormal() {
        f32::from_bits(bits & 0x8000_0000)
    } else {
        value
    }
}
fn output32(value: f32, denorm: u8) -> f32 {
    if denorm & 2 == 0 && value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}
fn input16(bits: u16, denorm: u8) -> f32 {
    let bits = if denorm & 1 == 0 && bits & 0x7c00 == 0 {
        bits & 0x8000
    } else {
        bits
    };
    float::f16_to_f32(bits)
}

/// `v_pk_min/max_f16` delegates to `v_min/max_f16`: CDNA3 ISA §12.10
/// p.258–259 and §12.7 p.172–173. IEEE mode propagates a signaling NaN,
/// quiet NaNs select the other operand, and zeros have an explicit sign rule.
fn minmax_half(mut a: u16, mut b: u16, maximum: bool, w: &Wave) -> u16 {
    if w.mode.denorm16_64 & 1 == 0 {
        if a & 0x7c00 == 0 {
            a &= 0x8000;
        }
        if b & 0x7c00 == 0 {
            b &= 0x8000;
        }
    }
    let nan = |v: u16| v & 0x7c00 == 0x7c00 && v & 0x03ff != 0;
    if w.mode.ieee {
        if nan(a) && a & 0x0200 == 0 {
            return a | 0x0200;
        }
        if nan(b) && b & 0x0200 == 0 {
            return b | 0x0200;
        }
    }
    if nan(a) {
        return b;
    }
    if nan(b) {
        return a;
    }
    if a & 0x7fff == 0 && b & 0x7fff == 0 {
        return if maximum {
            (a & b) & 0x8000
        } else {
            (a | b) & 0x8000
        };
    }
    let (af, bf) = (float::f16_to_f32(a), float::f16_to_f32(b));
    let choose_a = if maximum {
        if w.mode.ieee {
            af >= bf
        } else {
            af > bf
        }
    } else {
        af < bf
    };
    if choose_a {
        a
    } else {
        b
    }
}
/// FP16_OVFL saturates finite overflow, but preserves true infinities:
/// CDNA3 ISA §3.5 Table 4 p.11.
fn half_result(mut bits: u16, w: &Wave, finite_inputs: bool) -> u16 {
    if finite_inputs && w.mode.fp16_ovfl && bits & 0x7fff == 0x7c00 {
        bits = bits & 0x8000 | 0x7bff;
    }
    if w.mode.denorm16_64 & 2 == 0 && bits & 0x7c00 == 0 {
        bits &= 0x8000;
    }
    bits
}
/// MODE.DX10_CLAMP controls NaN-to-zero, CDNA3 ISA §3.5 Table 4 p.10.
fn clamp_float(value: f32, clamp: bool, dx10_clamp: bool) -> f32 {
    if !clamp {
        value
    } else if value.is_nan() {
        if dx10_clamp {
            0.0
        } else {
            value
        }
    } else {
        value.max(0.0).min(1.0)
    }
}
fn integer_half(op: Vop3pOp, a: u16, b: u16, clamp: bool) -> u16 {
    use Vop3pOp::*;
    match op {
        PkAddU16 => {
            if clamp {
                a.saturating_add(b)
            } else {
                a.wrapping_add(b)
            }
        }
        PkSubU16 => {
            if clamp {
                a.saturating_sub(b)
            } else {
                a.wrapping_sub(b)
            }
        }
        PkMulLoU16 => {
            if clamp {
                a.saturating_mul(b)
            } else {
                a.wrapping_mul(b)
            }
        }
        PkLshlrevB16 => b.wrapping_shl(u32::from(a & 15)),
        PkLshrrevB16 => b >> (a & 15),
        PkMaxI16 => (a as i16).max(b as i16) as u16,
        PkMinI16 => (a as i16).min(b as i16) as u16,
        _ => unreachable!(),
    }
}
/// DOT2 floating products are exact in F32; §12.10 p.261 specifies low then
/// high addition, rounding after each +=. §6.4 p.36 mandates input/output
/// denormal flushing, independent of MODE. [UNVERIFIED: confirm on MI300X]
/// Whether real hardware fuses the two additions instead of this pseudocode.
fn dot_result(w: &Wave, i: &Vop3p, src: [u64; 3]) -> SimResult<u32> {
    use Vop3pOp::*;
    if i.op == Dot2F32F16 {
        let mut sum = input32(src[2] as u32, 0);
        if i.neg_lo & 4 != 0 {
            sum = -sum;
        }
        for half in 0..2 {
            let sel = if half == 0 { i.op_sel } else { i.op_sel_hi };
            let neg = if half == 0 { i.neg_lo } else { i.neg_hi };
            let mut a = input16((src[0] >> (16 * (sel & 1))) as u16, 0);
            let mut b = input16((src[1] >> (16 * ((sel >> 1) & 1))) as u16, 0);
            if neg & 1 != 0 {
                a = -a;
            }
            if neg & 2 != 0 {
                b = -b;
            }
            sum += a * b;
        }
        return Ok(output32(clamp_float(sum, i.clamp, w.mode.dx10_clamp), 0).to_bits());
    }
    let (bits, signed) = match i.op {
        Dot2I32I16 => (16, true),
        Dot2U32U16 => (16, false),
        Dot4I32I8 => (8, true),
        Dot4U32U8 => (8, false),
        Dot8I32I4 => (4, true),
        Dot8U32U4 => (4, false),
        _ => unreachable!(),
    };
    let mask = (1u32 << bits) - 1;
    let mut sum = if signed {
        src[2] as u32 as i32 as i64
    } else {
        src[2] as u32 as i64
    };
    for index in 0..32 / bits {
        let field = |s: u64| {
            let value = (s as u32 >> (index * bits)) & mask;
            if signed {
                ((value << (32 - bits)) as i32 >> (32 - bits)) as i64
            } else {
                i64::from(value)
            }
        };
        sum += field(src[0]) * field(src[1]);
    }
    let _ = w;
    Ok(if i.clamp {
        if signed {
            sum.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as u32
        } else {
            sum.clamp(0, i64::from(u32::MAX)) as u32
        }
    } else {
        sum as u32
    })
}
