// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `sim::mfma`: VOP3P-MAI (MFMA, `v_accvgpr_*`) and VOP3P packed math.
//!
//! Every MFMA expectation comes from an INDEPENDENT matrix reference written
//! in this file: operands are placed in registers with the CDNA3 ISA §7.1.4.1
//! input layout (`A[i,k]` -> item `k % K_L` of lane `i + M*(k / K_L)`; `B[k,j]`
//! -> item `k % K_L` of lane `j + N*(k / K_L)`, p.46-47), results are read back
//! with the §7.1.4.2 output layout (`D[i,j]` -> item `(i%H) + H*(i/(H*M_I) +
//! G*(b/B_I))` of lane `j + N*((i/H)%M_I + M_I*(b%B_I))`, H=1 for F64 else 4,
//! p.47-48) and compared with a plain `D = C + sum_k A*B` computed on exact
//! rationals (`Fx`, an i128 mantissa/exponent pair) rounded by this file's own
//! IEEE rounder. The accumulation models are re-implemented here from their
//! definitions (Exact / Serial / Grouped); `Cdna3Fused` is checked on inputs
//! whose sums are exact in every model (small integers; F32/F64 where it is
//! a serial FMA) and on hand-derived probes. `mfma_reference` is only used as
//! a secondary cross-check.
//!
//! Broadcast: CDNA3 ISA §7.1.5.1 p.49 (`p_a(l) = l % S + S*ABID`, `S = 64 >>
//! CBSZ`; CBSZ with `1 << CBSZ` above the block count is undefined), BLGP
//! Table 29 p.49-50, F64 BLGP/NEG §7.1.5.3 p.50, denormals §7.3 p.52, packed
//! math §6.7 p.38-39 and §12.10 p.254-263, DOT must not use OPSEL §6.2 p.32.
use peacemaker_cdna::arch::Fp8Encoding;
use peacemaker_cdna::enc::vop3p::{Mai, MaiOp, MfmaShape, MfmaType, Vop3p, Vop3pOp};
use peacemaker_cdna::float::{f8_to_f32, F8};
use peacemaker_cdna::inst::Inst;
use peacemaker_cdna::operand::{InlineFloat, Src};
use peacemaker_cdna::sim::mfma::{accumulate_typed, mfma_reference, AccumModel};
use peacemaker_cdna::sim::{exec_one, Config, Memory, Mode, SimError, SimResult, Wave};
use peacemaker_cdna::Arch;
use std::sync::LazyLock;

use MfmaType::*;

type Regs = Vec<[u32; 64]>;

const POISON: u32 = 0xdead_beef;
const MODELS: [AccumModel; 8] = [
    AccumModel::Cdna3Fused,
    AccumModel::Exact,
    AccumModel::Serial,
    AccumModel::Grouped(1),
    AccumModel::Grouped(2),
    AccumModel::Grouped(3),
    AccumModel::Grouped(4),
    AccumModel::Grouped(8),
];

// ---------------------------------------------------------------------------
// Deterministic RNG

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    /// Inclusive.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
    fn chance(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }
    fn sign(&mut self) -> u64 {
        self.below(2)
    }
}

// ---------------------------------------------------------------------------
// Exact rationals and an independent IEEE rounder

/// `m * 2^e`.
#[derive(Clone, Copy, Debug)]
struct Fx {
    m: i128,
    e: i32,
}

fn bitlen(m: u128) -> i32 {
    128 - m.leading_zeros() as i32
}

impl Fx {
    const ZERO: Fx = Fx { m: 0, e: 0 };
    fn trim(self) -> Fx {
        if self.m == 0 {
            return Fx::ZERO;
        }
        let tz = self.m.trailing_zeros();
        Fx {
            m: self.m >> tz,
            e: self.e + tz as i32,
        }
    }
    fn neg(self) -> Fx {
        Fx {
            m: -self.m,
            e: self.e,
        }
    }
    fn mul(self, o: Fx) -> Fx {
        Fx {
            m: self.m.checked_mul(o.m).expect("Fx product overflows i128"),
            e: self.e + o.e,
        }
        .trim()
    }
    fn aligned(self, e: i32) -> i128 {
        if self.m == 0 {
            return 0;
        }
        let sh = (self.e - e) as u32;
        assert!(
            bitlen(self.m.unsigned_abs()) as u32 + sh <= 124,
            "exact span too wide for the i128 reference"
        );
        self.m << sh
    }
    fn add(self, o: Fx) -> Fx {
        if self.m == 0 {
            return o;
        }
        if o.m == 0 {
            return self;
        }
        let e = self.e.min(o.e);
        Fx {
            m: self.aligned(e) + o.aligned(e),
            e,
        }
        .trim()
    }
}
fn fx_lt(a: Fx, b: Fx) -> bool {
    a.add(b.neg()).m < 0
}

#[derive(Clone, Copy)]
struct Fmt {
    p: i32,
    emin: i32,
    emax: i32,
    bits: u32,
}
const F16F: Fmt = Fmt {
    p: 11,
    emin: -14,
    emax: 15,
    bits: 16,
};
const BF16F: Fmt = Fmt {
    p: 8,
    emin: -126,
    emax: 127,
    bits: 16,
};
const F32F: Fmt = Fmt {
    p: 24,
    emin: -126,
    emax: 127,
    bits: 32,
};
const F64F: Fmt = Fmt {
    p: 53,
    emin: -1022,
    emax: 1023,
    bits: 64,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rm {
    Rne,
    Rtp,
    Rtn,
    Rtz,
}

/// Finite IEEE bits of format `f` -> exact value.
fn fx_from_bits(bits: u64, f: Fmt) -> Fx {
    let ebits = f.bits - f.p as u32;
    let mbits = (f.p - 1) as u32;
    let neg = (bits >> (f.bits - 1)) & 1 != 0;
    let ex = ((bits >> mbits) & ((1u64 << ebits) - 1)) as i32;
    let fr = (bits & ((1u64 << mbits) - 1)) as i128;
    let bias = (1i32 << (ebits - 1)) - 1;
    assert!(
        ex != (1i32 << ebits) - 1,
        "non-finite operand {bits:#x} in an exact reference"
    );
    let (m, e) = if ex == 0 {
        (fr, 1 - bias - mbits as i32)
    } else {
        (fr | (1i128 << mbits), ex - bias - mbits as i32)
    };
    Fx {
        m: if neg { -m } else { m },
        e,
    }
    .trim()
}

/// Round an exact value to format `f` (RNE or a directed mode); exact zero is
/// `+0`, an underflowing nonzero keeps its sign.
fn round_fx(x: Fx, f: Fmt, rm: Rm) -> u64 {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let ebits = f.bits - f.p as u32;
    let frac_bits = (f.p - 1) as u32;
    let sign = |neg: bool| u64::from(neg) << (f.bits - 1);
    if x.m == 0 {
        return 0;
    }
    let neg = x.m < 0;
    let mag = x.m.unsigned_abs();
    let top = bitlen(mag) - 1 + x.e;
    let mut q = top.max(f.emin) - (f.p - 1);
    let mut kept: u128;
    if q <= x.e {
        kept = mag << (x.e - q) as u32;
    } else {
        let sh = (q - x.e) as u32;
        let (k, cmp, inexact) = if sh > 127 {
            (0u128, Less, true)
        } else {
            let rem = mag & ((1u128 << sh) - 1);
            let half = 1u128 << (sh - 1);
            (mag >> sh, rem.cmp(&half), rem != 0)
        };
        let odd = k & 1 == 1;
        let up = match rm {
            Rm::Rne => cmp == Greater || (cmp == Equal && odd),
            Rm::Rtp => !neg && inexact,
            Rm::Rtn => neg && inexact,
            Rm::Rtz => false,
        };
        kept = k + u128::from(up);
    }
    while bitlen(kept) > f.p {
        kept >>= 1;
        q += 1;
    }
    if kept == 0 {
        return sign(neg);
    }
    let kb = bitlen(kept);
    let ev = kb - 1 + q;
    if ev > f.emax {
        let inf = ((1u64 << ebits) - 1) << frac_bits;
        let max = (((1u64 << ebits) - 2) << frac_bits) | ((1u64 << frac_bits) - 1);
        let to_inf = match rm {
            Rm::Rne => true,
            Rm::Rtp => !neg,
            Rm::Rtn => neg,
            Rm::Rtz => false,
        };
        return sign(neg) | if to_inf { inf } else { max };
    }
    if kb < f.p {
        return sign(neg) | kept as u64;
    }
    let ef = (ev - f.emin + 1) as u64;
    sign(neg) | (ef << frac_bits) | (kept as u64 - (1u64 << frac_bits))
}

fn p2(e: i32) -> f64 {
    if e >= -1022 {
        assert!(e <= 1023);
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        assert!(e >= -1074);
        f64::from_bits(1u64 << (e + 1074) as u32)
    }
}

// ---------------------------------------------------------------------------
// Accumulation models, re-implemented from their definitions

#[derive(Clone, Copy)]
enum St {
    Fin(Fx),
    Inf(bool),
}

fn st_round(x: Fx, f: Fmt, rm: Rm) -> St {
    let bits = round_fx(x, f, rm);
    let ebits = f.bits - f.p as u32;
    let inf = ((1u64 << ebits) - 1) << (f.p - 1) as u32;
    let mag = bits & !(1u64 << (f.bits - 1));
    if mag == inf {
        St::Inf(bits >> (f.bits - 1) != 0)
    } else {
        St::Fin(fx_from_bits(bits, f))
    }
}
fn st_bits(st: St, f: Fmt) -> u64 {
    match st {
        St::Fin(x) => round_fx(x, f, Rm::Rne),
        St::Inf(n) => {
            (((1u64 << (f.bits - f.p as u32)) - 1) << (f.p - 1) as u32)
                | (u64::from(n) << (f.bits - 1))
        }
    }
}

/// `Exact`: one rounding of `c + sum(terms)`. `Serial`: one rounding after each
/// term. `Grouped(g)`: exact `acc + sum(group)` rounded once per group of `g`
/// consecutive terms, ascending. (`Cdna3Fused` is never passed here.)
fn accumulate(model: AccumModel, f: Fmt, rm: Rm, c: Fx, terms: &[Fx]) -> u64 {
    let group = match model {
        AccumModel::Exact => terms.len().max(1),
        AccumModel::Serial => 1,
        AccumModel::Grouped(g) => usize::from(g),
        AccumModel::Cdna3Fused => panic!("Cdna3Fused has no generic reference"),
    };
    let mut st = St::Fin(c);
    for chunk in terms.chunks(group) {
        let sum = chunk.iter().fold(Fx::ZERO, |a, &t| a.add(t));
        st = match st {
            St::Inf(n) => St::Inf(n),
            St::Fin(x) => st_round(x.add(sum), f, rm),
        };
    }
    st_bits(st, f)
}

// ---------------------------------------------------------------------------
// Element coding

fn ew(t: MfmaType) -> u32 {
    match t {
        F64 => 64,
        F32 | I32 => 32,
        F16 | Bf16 => 16,
        Fp8 | Bf8 | I8 => 8,
    }
}
fn f8_kind(t: MfmaType) -> F8 {
    if t == Fp8 {
        F8::Fp8
    } else {
        F8::Bf8
    }
}
fn f8_dec(t: MfmaType, code: u8) -> f32 {
    f8_to_f32(Fp8Encoding::Fnuz, f8_kind(t), code)
}
fn f8_code(t: MfmaType, v: f64) -> u8 {
    (0..=255u8)
        .filter(|c| *c != 0x80)
        .find(|c| f8_dec(t, *c) as f64 == v)
        .unwrap_or_else(|| panic!("{v} is not an {t:?} value"))
}
fn fx_of_f32(x: f32) -> Fx {
    assert!(x.is_finite());
    fx_from_bits(u64::from(x.to_bits()), F32F)
}
/// Decode one operand element exactly. `flush32` flushes F32 denormals
/// (CDNA3 ISA §7.3 p.52, A/B only).
fn dec(t: MfmaType, bits: u64, flush32: bool) -> Fx {
    match t {
        F32 => {
            let mut b = bits as u32;
            if flush32 && (b >> 23) & 0xff == 0 {
                b &= 0x8000_0000;
            }
            fx_from_bits(u64::from(b), F32F)
        }
        F64 => fx_from_bits(bits, F64F),
        F16 => fx_from_bits(bits, F16F),
        Bf16 => fx_from_bits(bits, BF16F),
        Fp8 | Bf8 => fx_of_f32(f8_dec(t, bits as u8)),
        I8 | I32 => panic!("integer type has no float decode"),
    }
}
fn exact_enc(v: f64, f: Fmt) -> u64 {
    let fx = fx_from_bits(v.to_bits(), F64F);
    let bits = round_fx(fx, f, Rm::Rne);
    assert!(
        fx_from_bits(bits, f).add(fx.neg()).m == 0,
        "{v} is not exactly representable"
    );
    bits
}
/// Encode an exactly representable value.
fn enc(t: MfmaType, v: f64) -> u64 {
    match t {
        F64 => v.to_bits(),
        F32 => exact_enc(v, F32F),
        F16 => exact_enc(v, F16F),
        Bf16 => exact_enc(v, BF16F),
        Fp8 | Bf8 => u64::from(f8_code(t, v)),
        I8 => u64::from(v as i8 as u8),
        I32 => u64::from(v as i32 as u32),
    }
}
fn dec64(t: MfmaType, bits: u64) -> f64 {
    match t {
        F32 => f64::from(f32::from_bits(bits as u32)),
        F16 => f64::from(peacemaker_cdna::float::f16_to_f32(bits as u16)),
        Bf16 => f64::from(f32::from_bits((bits as u32) << 16)),
        Fp8 | Bf8 => f64::from(f8_dec(t, bits as u8)),
        _ => panic!("no plain f64 decode for {t:?}"),
    }
}

// ---------------------------------------------------------------------------
// Register layout (CDNA3 ISA §7.1.4)

fn kl(s: &MfmaShape) -> usize {
    assert_eq!(s.blocks, 1);
    s.k as usize / (64 / (s.m as usize * s.blocks as usize))
}
fn put(regs: &mut [[u32; 64]], t: MfmaType, lane: usize, item: usize, v: u64) {
    match ew(t) {
        64 => {
            regs[2 * item][lane] = v as u32;
            regs[2 * item + 1][lane] = (v >> 32) as u32;
        }
        32 => regs[item][lane] = v as u32,
        16 => {
            let sh = 16 * (item % 2);
            regs[item / 2][lane] =
                (regs[item / 2][lane] & !(0xffffu32 << sh)) | (((v & 0xffff) as u32) << sh);
        }
        8 => {
            let sh = 8 * (item % 4);
            regs[item / 4][lane] =
                (regs[item / 4][lane] & !(0xffu32 << sh)) | (((v & 0xff) as u32) << sh);
        }
        w => panic!("width {w}"),
    }
}
fn get(regs: &[[u32; 64]], t: MfmaType, lane: usize, item: usize) -> u64 {
    match ew(t) {
        64 => u64::from(regs[2 * item][lane]) | (u64::from(regs[2 * item + 1][lane]) << 32),
        32 => u64::from(regs[item][lane]),
        16 => u64::from((regs[item / 2][lane] >> (16 * (item % 2))) & 0xffff),
        8 => u64::from((regs[item / 4][lane] >> (8 * (item % 4))) & 0xff),
        w => panic!("width {w}"),
    }
}
fn pack_a(s: &MfmaShape, a: &[u64]) -> Regs {
    let (m, k, l) = (s.m as usize, s.k as usize, kl(s));
    let mut r = vec![[0u32; 64]; s.a_dwords as usize];
    for i in 0..m {
        for kk in 0..k {
            put(&mut r, s.a, i + m * (kk / l), kk % l, a[i * k + kk]);
        }
    }
    r
}
fn pack_b(s: &MfmaShape, b: &[u64]) -> Regs {
    let (n, k, l) = (s.n as usize, s.k as usize, kl(s));
    let mut r = vec![[0u32; 64]; s.b_dwords as usize];
    for j in 0..n {
        for kk in 0..k {
            put(&mut r, s.b, j + n * (kk / l), kk % l, b[kk * n + j]);
        }
    }
    r
}
fn unpack_a(s: &MfmaShape, r: &[[u32; 64]]) -> Vec<u64> {
    let (m, k, l) = (s.m as usize, s.k as usize, kl(s));
    let mut a = vec![0u64; m * k];
    for i in 0..m {
        for kk in 0..k {
            a[i * k + kk] = get(r, s.a, i + m * (kk / l), kk % l);
        }
    }
    a
}
fn unpack_b(s: &MfmaShape, r: &[[u32; 64]]) -> Vec<u64> {
    let (n, k, l) = (s.n as usize, s.k as usize, kl(s));
    let mut b = vec![0u64; k * n];
    for j in 0..n {
        for kk in 0..k {
            b[kk * n + j] = get(r, s.b, j + n * (kk / l), kk % l);
        }
    }
    b
}
/// `(item, lane)` of `D[i,j]` (block 0), §7.1.4.2 p.48.
fn out_pos(s: &MfmaShape, i: usize, j: usize) -> (usize, usize) {
    let (m, n) = (s.m as usize, s.n as usize);
    let h = if s.cd == F64 { 1 } else { 4 };
    let bi = 64usize.div_ceil(n * m / h);
    let mi = (64 / bi) / n;
    let g = m / (h * mi);
    assert_eq!(
        h * g * (ew(s.cd) as usize / 32),
        s.cd_dwords as usize,
        "G*H items must fill the C/D registers"
    );
    ((i % h) + h * (i / (h * mi)), j + n * ((i / h) % mi))
}
fn pack_cd(s: &MfmaShape, c: &[u64]) -> Regs {
    let (m, n) = (s.m as usize, s.n as usize);
    let mut r = vec![[0u32; 64]; s.cd_dwords as usize];
    for i in 0..m {
        for j in 0..n {
            let (item, lane) = out_pos(s, i, j);
            put(&mut r, s.cd, lane, item, c[i * n + j]);
        }
    }
    r
}
fn unpack_cd(s: &MfmaShape, r: &[[u32; 64]]) -> Vec<u64> {
    let (m, n) = (s.m as usize, s.n as usize);
    let mut c = vec![0u64; m * n];
    for i in 0..m {
        for j in 0..n {
            let (item, lane) = out_pos(s, i, j);
            c[i * n + j] = get(r, s.cd, lane, item);
        }
    }
    c
}

/// BLGP, Table 29 p.49-50.
fn blgp_perm(v: u8, l: usize) -> usize {
    match v {
        0 => l,
        1 => l % 32,
        2 => l % 32 + 32,
        3 => (l + 16) % 64,
        4 => l % 16,
        5 => l % 16 + 16,
        6 => l % 16 + 32,
        7 => l % 16 + 48,
        _ => panic!("BLGP {v}"),
    }
}

#[derive(Clone, Copy, Debug)]
struct Ctl {
    cbsz: u8,
    abid: u8,
    blgp: u8,
}
const NO_CTL: Ctl = Ctl {
    cbsz: 0,
    abid: 0,
    blgp: 0,
};

/// The logical A and B the hardware sees after the lane permutations of
/// §7.1.5 (A: CBSZ/ABID; B: BLGP). F64 uses neither (§7.1.5.3).
fn effective(op: MaiOp, a: &[u64], b: &[u64], ctl: Ctl) -> (Vec<u64>, Vec<u64>) {
    let s = op.shape().unwrap();
    if s.cd == F64 {
        return (a.to_vec(), b.to_vec());
    }
    let (a0, b0) = (pack_a(&s, a), pack_b(&s, b));
    let (mut ar, mut br) = (a0.clone(), b0.clone());
    let size = 64usize >> ctl.cbsz;
    for r in 0..ar.len() {
        for l in 0..64 {
            ar[r][l] = a0[r][l % size + size * usize::from(ctl.abid)];
        }
    }
    for r in 0..br.len() {
        for l in 0..64 {
            br[r][l] = b0[r][blgp_perm(ctl.blgp, l)];
        }
    }
    (unpack_a(&s, &ar), unpack_b(&s, &br))
}

// ---------------------------------------------------------------------------
// Matrix reference

#[derive(Clone, Copy, Debug)]
struct RefOpts {
    rm: Rm,
    denorm32: u8,
    /// Inputs are small integers: every model, `Cdna3Fused` included, is exact.
    invariant: bool,
    /// F64 BLGP negation bits (A, B, C), §7.1.5.3.
    neg: u8,
}
const REF_DEFAULT: RefOpts = RefOpts {
    rm: Rm::Rne,
    denorm32: 3,
    invariant: false,
    neg: 0,
};

/// `D[i,j] = C[i,j] + sum_k A[i,k]*B[k,j]` on logical row-major bit matrices.
fn ref_mfma(
    op: MaiOp,
    a: &[u64],
    b: &[u64],
    c: &[u64],
    model: AccumModel,
    o: &RefOpts,
) -> Vec<u64> {
    let s = op.shape().unwrap();
    let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
    let mut out = Vec::with_capacity(m * n);
    if s.cd == I32 {
        // §7.3 p.52: I8 products sign-extend; the sum wraps at 32 bits.
        for i in 0..m {
            for j in 0..n {
                let mut acc = c[i * n + j] as u32 as i32;
                for kk in 0..k {
                    acc = acc.wrapping_add(
                        i32::from(a[i * k + kk] as u8 as i8) * i32::from(b[kk * n + j] as u8 as i8),
                    );
                }
                out.push(u64::from(acc as u32));
            }
        }
        return out;
    }
    let f = if s.cd == F64 { F64F } else { F32F };
    let flush = s.a == F32 && o.denorm32 & 1 == 0;
    let fused = model == AccumModel::Cdna3Fused && !matches!(s.a, F32 | F64);
    let model = match model {
        AccumModel::Cdna3Fused if !fused => AccumModel::Serial,
        other => other,
    };
    let (na, nb, nc) = if s.cd == F64 {
        (o.neg & 1 != 0, o.neg & 2 != 0, o.neg & 4 != 0)
    } else {
        (false, false, false)
    };
    for i in 0..m {
        for j in 0..n {
            let mut xs = Vec::with_capacity(k);
            let mut ys = Vec::with_capacity(k);
            for kk in 0..k {
                let mut x = dec(s.a, a[i * k + kk], flush);
                let mut y = dec(s.b, b[kk * n + j], flush);
                if na {
                    x = x.neg();
                }
                if nb {
                    y = y.neg();
                }
                xs.push(x);
                ys.push(y);
            }
            let mut cx = dec(s.cd, c[i * n + j], false);
            if nc {
                cx = cx.neg();
            }
            if fused {
                out.push(fused_reduce(s.a, s.b, cx, &xs, &ys));
            } else {
                let terms: Vec<Fx> = xs.iter().zip(&ys).map(|(x, y)| x.mul(*y)).collect();
                out.push(accumulate(model, f, o.rm, cx, &terms));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Independent model of the `Cdna3Fused` candidate for F16/BF16/FP8/BF8
// (MMA-Sim TR/GTR-FDPA as summarized in arXiv:2609.14845 §4.3 and in the
// `AccumModel::Cdna3Fused` documentation; [UNVERIFIED: confirm on MI300X]).
// Per group of 8 (F16/BF16) or 16 (FP8/BF8) consecutive k, with acc = running
// F32 sum starting at C:
//   products p_i exact; e_i = rawexp(a_i) + rawexp(b_i) (zero products skipped),
//   rawexp(v) = max(floor(log2 |v|), min exponent of the type: F16 -14, FP8 -7, BF8 -15, else -126);
//   FP8/BF8 split even/odd indices into two partial sums, others one;
//   exp[g] = max e_i of partial g; partial[g] = sum trunc_toward_zero(p_i, 2^(exp[g]-24));
//   dot_exp = max exp[g]; FP8: dot = floor(partial0, 2^(dot_exp-24)) + floor(partial1, same), else partial0;
//   maximum = max(dot_exp, rawexp(acc, -126)); dot = floor(dot, 2^(maximum-32));
//   acc' = floor(acc, 2^(maximum-24)), or 0 for FP8/BF8 when rawexp(acc) < maximum-25;
//   sum = dot + acc'; result = floor(sum, 2^(rawexp(sum)-31)) rounded to F32 (RNE).
// floor = toward -infinity ("RD"). Finite inputs only.

/// Quantize to a multiple of `2^q` toward zero (`floor == false`) or -inf.
fn quant(x: Fx, q: i32, floor: bool) -> Fx {
    if x.m == 0 || x.e >= q {
        return x;
    }
    let sh = (q - x.e) as u32;
    let m = if sh >= 127 {
        if floor && x.m < 0 {
            -1
        } else {
            0
        }
    } else if floor || x.m >= 0 {
        x.m >> sh
    } else {
        -((-x.m) >> sh)
    };
    Fx { m, e: q }.trim()
}
fn raw_exp(x: Fx, min: i32) -> i32 {
    if x.m == 0 {
        min
    } else {
        (bitlen(x.m.unsigned_abs()) - 1 + x.e).max(min)
    }
}
fn min_exp(t: MfmaType) -> i32 {
    match t {
        F16 => -14,
        Fp8 => -7,
        Bf8 => -15,
        _ => -126,
    }
}

fn fused_group(at: MfmaType, bt: MfmaType, c: Fx, a: &[Fx], b: &[Fx], fp8: bool) -> Fx {
    let prods: Vec<Fx> = a.iter().zip(b).map(|(x, y)| x.mul(*y)).collect();
    let mut exps = vec![-999i32; prods.len()];
    for (i, p) in prods.iter().enumerate() {
        if p.m != 0 {
            exps[i] = raw_exp(a[i], min_exp(at)) + raw_exp(b[i], min_exp(bt));
        }
    }
    let groups = if fp8 { 2 } else { 1 };
    let mut exp = [-999i32; 2];
    for i in 0..prods.len() {
        exp[i % groups] = exp[i % groups].max(exps[i]);
    }
    let mut partial = [Fx::ZERO; 2];
    for i in 0..prods.len() {
        let g = i % groups;
        partial[g] = partial[g].add(quant(prods[i], exp[g] - 24, false));
    }
    let dot_exp = exp[..groups].iter().copied().max().unwrap();
    let dot = if fp8 {
        quant(partial[0], dot_exp - 24, true).add(quant(partial[1], dot_exp - 24, true))
    } else {
        partial[0]
    };
    let c_exp = raw_exp(c, -126);
    let maximum = dot_exp.max(c_exp);
    let dot = quant(dot, maximum - 32, true);
    let c2 = if fp8 && c_exp < maximum - 25 {
        Fx::ZERO
    } else {
        quant(c, maximum - 24, true)
    };
    let sum = dot.add(c2);
    quant(sum, raw_exp(sum, -126) - 31, true)
}

/// F32 result bits of the whole reduction (all groups).
fn fused_reduce(at: MfmaType, bt: MfmaType, c: Fx, a: &[Fx], b: &[Fx]) -> u64 {
    let fp8 = matches!(at, Fp8 | Bf8);
    let group = if fp8 { 16 } else { 8 };
    let mut acc = c;
    for (ca, cb) in a.chunks(group).zip(b.chunks(group)) {
        let r = fused_group(at, bt, acc, ca, cb, fp8);
        acc = fx_from_bits(round_fx(r, F32F, Rm::Rne), F32F);
    }
    round_fx(acc, F32F, Rm::Rne)
}

// ---------------------------------------------------------------------------
// Operand generators

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gen {
    /// Wide exponent spread: rounding differs between models.
    Wide,
    /// Few significant bits, narrow exponents: every sum is exact in f64.
    Bounded,
    /// Integers in [-6, 6] (FP8/BF8: [-4, 4]): exact in every model.
    SmallInt,
}

static FP8_SMALL: LazyLock<Vec<u8>> = LazyLock::new(|| small_codes_of(Fp8));
static BF8_SMALL: LazyLock<Vec<u8>> = LazyLock::new(|| small_codes_of(Bf8));
fn small_codes_of(t: MfmaType) -> Vec<u8> {
    (0..=255u8)
        .filter(|c| *c != 0x80)
        .filter(|c| {
            let v = f8_dec(t, *c);
            v.fract() == 0.0 && v.abs() <= 4.0
        })
        .collect()
}
fn small_codes(t: MfmaType) -> &'static [u8] {
    if t == Fp8 {
        FP8_SMALL.as_slice()
    } else {
        BF8_SMALL.as_slice()
    }
}

fn gen_elem(rng: &mut Rng, t: MfmaType, g: Gen) -> u64 {
    let sign = rng.sign();
    match (t, g) {
        (I8, _) => {
            if rng.chance(4) {
                [0x80u64, 0x81, 0xff, 0x00, 0x01, 0x7f, 0x7e][rng.below(7) as usize]
            } else {
                rng.below(256)
            }
        }
        (F32, Gen::SmallInt) => enc(F32, rng.range(-6, 6) as f64),
        (F16, Gen::SmallInt) => enc(F16, rng.range(-6, 6) as f64),
        (Bf16, Gen::SmallInt) => enc(Bf16, rng.range(-6, 6) as f64),
        (F64, Gen::SmallInt) => enc(F64, rng.range(-6, 6) as f64),
        (Fp8 | Bf8, Gen::SmallInt) => {
            let c = small_codes(t);
            u64::from(c[rng.below(c.len() as u64) as usize])
        }
        (F32, Gen::Wide) => {
            if rng.chance(16) {
                0
            } else {
                (sign << 31) | ((121 + rng.below(13)) << 23) | rng.below(1 << 23)
            }
        }
        (F32, Gen::Bounded) => {
            let v = rng.range(1, 2048) as f64 * p2(rng.range(-6, 0) as i32);
            enc(F32, if sign == 1 { -v } else { v })
        }
        (F16, Gen::Wide) => {
            if rng.chance(16) {
                (sign << 15) | rng.below(1024)
            } else {
                (sign << 15) | ((1 + rng.below(30)) << 10) | rng.below(1024)
            }
        }
        (F16, Gen::Bounded) => (sign << 15) | ((11 + rng.below(9)) << 10) | rng.below(1024),
        (Bf16, Gen::Wide) => {
            if rng.chance(16) {
                0
            } else {
                (sign << 15) | ((115 + rng.below(25)) << 7) | rng.below(128)
            }
        }
        (Bf16, Gen::Bounded) => (sign << 15) | ((120 + rng.below(15)) << 7) | rng.below(128),
        (Fp8, Gen::Wide | Gen::Bounded) => {
            let c = rng.below(255) as u8;
            u64::from(if c >= 0x80 { c + 1 } else { c })
        }
        (Bf8, Gen::Wide) => {
            let c = rng.below(255) as u8;
            u64::from(if c >= 0x80 { c + 1 } else { c })
        }
        (Bf8, Gen::Bounded) => (sign << 7) | ((10 + rng.below(13)) << 2) | rng.below(4),
        (F64, _) => {
            let v = (1 + rng.below(1 << 30)) as f64 * p2(rng.range(-14, 2) as i32);
            (if sign == 1 { -v } else { v }).to_bits()
        }
        (I32, _) => unreachable!(),
    }
}

fn gen_c(rng: &mut Rng, cd: MfmaType, g: Gen) -> u64 {
    let sign = rng.sign();
    match cd {
        I32 => {
            if rng.chance(4) {
                u64::from(
                    [i32::MAX, i32::MIN, -1, 0, 1, i32::MAX - 1][rng.below(6) as usize] as u32,
                )
            } else {
                rng.below(1 << 32)
            }
        }
        F32 => match g {
            Gen::SmallInt => enc(F32, rng.range(-100, 100) as f64),
            Gen::Bounded => {
                let v = rng.range(1, 4096) as f64 * p2(rng.range(-12, 6) as i32);
                enc(F32, if sign == 1 { -v } else { v })
            }
            Gen::Wide => {
                if rng.chance(16) {
                    0
                } else {
                    (sign << 31) | ((107 + rng.below(41)) << 23) | rng.below(1 << 23)
                }
            }
        },
        F64 => match g {
            Gen::SmallInt => enc(F64, rng.range(-100, 100) as f64),
            _ => {
                let v = (1 + rng.below(1 << 50)) as f64 * p2(rng.range(-40, -10) as i32);
                (if sign == 1 { -v } else { v }).to_bits()
            }
        },
        _ => unreachable!(),
    }
}

fn rand_mats(rng: &mut Rng, s: &MfmaShape, g: Gen) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
    let a: Vec<u64> = (0..m * k).map(|_| gen_elem(rng, s.a, g)).collect();
    let b: Vec<u64> = (0..k * n).map(|_| gen_elem(rng, s.b, g)).collect();
    let c: Vec<u64> = (0..m * n).map(|_| gen_c(rng, s.cd, g)).collect();
    (a, b, c)
}

// ---------------------------------------------------------------------------
// Execution harness

const A0: u8 = 0;
const B0: u8 = 8;
const C0: u8 = 16;
const D0: u8 = 48;

#[derive(Clone, Copy, Debug)]
struct Form {
    a_acc: bool,
    b_acc: bool,
    cd_acc: bool,
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    form: Form,
    c_zero: bool,
    in_place: bool,
    ctl: Ctl,
    model: AccumModel,
    mode: Mode,
}
impl Setup {
    fn new(model: AccumModel) -> Self {
        Setup {
            form: Form {
                a_acc: false,
                b_acc: false,
                cd_acc: false,
            },
            c_zero: false,
            in_place: true,
            ctl: NO_CTL,
            model,
            mode: Mode::default(),
        }
    }
}

fn forms() -> Vec<Form> {
    (0..8)
        .map(|b| Form {
            a_acc: b & 1 != 0,
            b_acc: b & 2 != 0,
            cd_acc: b & 4 != 0,
        })
        .collect()
}
fn mfma_ops() -> Vec<MaiOp> {
    MaiOp::ALL
        .iter()
        .copied()
        .filter(|o| o.shape().is_some())
        .collect()
}
fn reg(acc: bool, n: u8) -> Src {
    if acc {
        Src::Agpr(n)
    } else {
        Src::Vgpr(n)
    }
}
fn load(w: &mut Wave, acc: bool, base: u8, regs: &[[u32; 64]]) {
    let file = if acc { &mut w.a } else { &mut w.v };
    for (n, r) in regs.iter().enumerate() {
        file[usize::from(base) + n] = *r;
    }
}
fn run_inst(w: &mut Wave, inst: &Inst, accum: AccumModel) -> SimResult<()> {
    let cfg = Config {
        accum,
        ..Config::default()
    };
    exec_one(w, inst, &mut [], &mut Memory::default(), &cfg).map(|_| ())
}

fn build(op: MaiOp, a: &Regs, b: &Regs, c: &Regs, su: &Setup) -> (Wave, Inst, u8) {
    let mut w = Wave::new(Arch::Gfx942, 0, POISON);
    w.mode = su.mode;
    load(&mut w, su.form.a_acc, A0, a);
    load(&mut w, su.form.b_acc, B0, b);
    load(&mut w, su.form.cd_acc, C0, c);
    let d = if su.in_place && !su.c_zero { C0 } else { D0 };
    let inst = Inst::Mai(Mai {
        op,
        dst: reg(su.form.cd_acc, d),
        src_a: reg(su.form.a_acc, A0),
        src_b: Some(reg(su.form.b_acc, B0)),
        src_c: Some(if su.c_zero {
            Src::Int(0)
        } else {
            reg(su.form.cd_acc, C0)
        }),
        cbsz: su.ctl.cbsz,
        abid: su.ctl.abid,
        blgp: su.ctl.blgp,
    });
    (w, inst, d)
}

/// Execute one MFMA; on success also checks that nothing outside D changed
/// (A, B, and C when `D != C`).
fn exec_mfma(op: MaiOp, a: &Regs, b: &Regs, c: &Regs, su: &Setup) -> SimResult<Regs> {
    let s = op.shape().unwrap();
    let (mut w, inst, d) = build(op, a, b, c, su);
    let (before_v, before_a) = (w.v.clone(), w.a.clone());
    run_inst(&mut w, &inst, su.model)?;
    let n = usize::from(s.cd_dwords);
    for (acc, before, after) in [(false, &before_v, &w.v), (true, &before_a, &w.a)] {
        for r in 0..before.len() {
            let in_d = acc == su.form.cd_acc && (usize::from(d)..usize::from(d) + n).contains(&r);
            if !in_d {
                assert!(
                    before[r] == after[r],
                    "{} wrote {}{r} outside D ({su:?})",
                    op.name(),
                    if acc { 'a' } else { 'v' }
                );
            }
        }
    }
    let file = if su.form.cd_acc { &w.a } else { &w.v };
    Ok(file[usize::from(d)..usize::from(d) + n].to_vec())
}

fn assert_matrix(ctx: &str, s: &MfmaShape, got: &[u64], want: &[u64]) {
    let n = s.n as usize;
    assert_eq!(got.len(), want.len());
    for (idx, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g == w,
            "{ctx}: D[{}][{}] = {g:#x}, expected {w:#x}",
            idx / n,
            idx % n
        );
    }
}

/// Run `op` on logical matrices and compare with the independent reference.
fn run_and_compare(op: MaiOp, a: &[u64], b: &[u64], c: &[u64], su: &Setup, invariant: bool) {
    let s = op.shape().unwrap();
    let ctx = format!("{} {su:?}", op.name());
    let got = exec_mfma(op, &pack_a(&s, a), &pack_b(&s, b), &pack_cd(&s, c), su)
        .unwrap_or_else(|e| panic!("{ctx}: {e}"));
    let (ae, be) = effective(op, a, b, su.ctl);
    let opts = RefOpts {
        rm: Rm::Rne,
        denorm32: su.mode.denorm32,
        invariant,
        neg: su.ctl.blgp,
    };
    let want = ref_mfma(op, &ae, &be, c, su.model, &opts);
    assert_matrix(&ctx, &s, &unpack_cd(&s, &got), &want);
}

// `Cdna3Fused` has an independent reference for every float type: serial FMA for F32/F64 inputs,
// `fused_reduce` for F16/BF16/FP8/BF8.

// ---------------------------------------------------------------------------
// Tests: layout

/// Literal ISA layout, written per shape without the generic formulas: A has
/// the single column `k = kk` equal to `i + 1`, B the single row `k = kk`
/// equal to `j + 1`, so `D[i,j] = (i+1)(j+1)` and the output registers can be
/// listed from the §7.1.4.2 formula solved for `(i, j)`.
#[test]
fn layout_spot_checks_with_literal_formulas() {
    // (op, M, N, K, K_L, kind): kind 0 = 16x16 H=4, 1 = 32x32 H=4, 2 = F64 16x16 H=1.
    let table: [(MaiOp, usize, usize, usize, usize, u8); 9] = [
        (MaiOp::MfmaF32_16x16x16F16, 16, 16, 16, 4, 0),
        (MaiOp::MfmaF32_16x16x16Bf16, 16, 16, 16, 4, 0),
        (MaiOp::MfmaF32_32x32x8F16, 32, 32, 8, 4, 1),
        (MaiOp::MfmaF32_32x32x8Bf16, 32, 32, 8, 4, 1),
        (MaiOp::MfmaF32_16x16x4F32, 16, 16, 4, 1, 0),
        (MaiOp::MfmaF32_32x32x2F32, 32, 32, 2, 1, 1),
        (MaiOp::MfmaF64_16x16x4F64, 16, 16, 4, 1, 2),
        (MaiOp::MfmaI32_16x16x32I8, 16, 16, 32, 8, 0),
        (MaiOp::MfmaI32_32x32x16I8, 32, 32, 16, 8, 1),
    ];
    for (op, m, n, k, kl_lit, kind) in table {
        let s = op.shape().unwrap();
        assert_eq!(
            (s.m as usize, s.n as usize, s.k as usize, kl(&s)),
            (m, n, k, kl_lit),
            "{}",
            op.name()
        );
        for kk in 0..k {
            let mut a = vec![[0u32; 64]; s.a_dwords as usize];
            let mut b = vec![[0u32; 64]; s.b_dwords as usize];
            for lane in 0..64 {
                if lane / m == kk / kl_lit {
                    put(
                        &mut a,
                        s.a,
                        lane,
                        kk % kl_lit,
                        enc(s.a, (lane % m + 1) as f64),
                    );
                }
                if lane / n == kk / kl_lit {
                    put(
                        &mut b,
                        s.b,
                        lane,
                        kk % kl_lit,
                        enc(s.b, (lane % n + 1) as f64),
                    );
                }
            }
            let c = vec![[0u32; 64]; s.cd_dwords as usize];
            let got = exec_mfma(op, &a, &b, &c, &Setup::new(AccumModel::Exact)).unwrap();
            let mut want = vec![[0u32; 64]; s.cd_dwords as usize];
            let items = if kind == 1 { 16 } else { 4 };
            for t in 0..items {
                for lane in 0..64 {
                    let (i, j) = match kind {
                        0 => (4 * (lane / 16) + t, lane % 16),
                        1 => (8 * (t / 4) + 4 * (lane / 32) + t % 4, lane % 32),
                        _ => (4 * t + lane / 16, lane % 16),
                    };
                    let v = enc(s.cd, ((i + 1) * (j + 1)) as f64);
                    if s.cd == F64 {
                        want[2 * t][lane] = v as u32;
                        want[2 * t + 1][lane] = (v >> 32) as u32;
                    } else {
                        want[t][lane] = v as u32;
                    }
                }
            }
            assert!(
                got == want,
                "{} kk={kk}: output registers differ from the §7.1.4 literal layout",
                op.name()
            );
        }
    }
}

#[test]
fn blgp_table_29_literals() {
    // Table 29 p.49-50 evaluated by hand for lanes 37 and 60.
    assert_eq!(
        (0..8).map(|v| blgp_perm(v, 37)).collect::<Vec<_>>(),
        [37, 5, 37, 53, 5, 21, 37, 53]
    );
    assert_eq!(
        (0..8).map(|v| blgp_perm(v, 60)).collect::<Vec<_>>(),
        [60, 28, 60, 12, 12, 28, 44, 60]
    );
    // §7.1.5.1 p.49 example: CBSZ=2 -> S=16, ABID=1 -> lanes 16..31 are broadcast.
    assert_eq!(37 % (64 >> 2) + (64 >> 2) * 1, 21);
}

/// BLGP on the instruction: `D[i,j]` reads B lane `blgp_perm(j + 16*kk)` of a
/// register in which lane `l` holds `100 + l`.
#[test]
fn blgp_all_values_literal_f32() {
    let op = MaiOp::MfmaF32_16x16x4F32;
    let s = op.shape().unwrap();
    for blgp in 0..8u8 {
        for kk in 0..4usize {
            let mut a = vec![[0u32; 64]; 1];
            let mut b = vec![[0u32; 64]; 1];
            for lane in 0..64 {
                if lane / 16 == kk {
                    a[0][lane] = 1.0f32.to_bits();
                }
                b[0][lane] = (100.0f32 + lane as f32).to_bits();
            }
            let c = vec![[0u32; 64]; 4];
            let mut su = Setup::new(AccumModel::Exact);
            su.ctl.blgp = blgp;
            let got = exec_mfma(op, &a, &b, &c, &su).unwrap();
            for t in 0..4 {
                for lane in 0..64 {
                    let j = lane % 16;
                    let src_lane = match blgp {
                        0 => j + 16 * kk,
                        1 => (j + 16 * kk) % 32,
                        2 => (j + 16 * kk) % 32 + 32,
                        3 => (j + 16 * kk + 16) % 64,
                        4 => j,
                        5 => j + 16,
                        6 => j + 32,
                        _ => j + 48,
                    };
                    assert_eq!(
                        got[t][lane],
                        (100.0f32 + src_lane as f32).to_bits(),
                        "BLGP {blgp} kk {kk} item {t} lane {lane}"
                    );
                }
            }
            let _ = s;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: random matrices, every shape / form / model

#[test]
fn every_shape_all_forms_and_models_match_the_independent_matrix_reference() {
    let mut rng = Rng::new(0xC0FF_EE01);
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        for model in MODELS {
            for (n, form) in forms().into_iter().enumerate() {
                let g = if n % 3 == 2 { Gen::SmallInt } else { Gen::Wide };
                let (a, b, c) = rand_mats(&mut rng, &s, g);
                let mut su = Setup::new(model);
                su.form = form;
                su.in_place = n % 2 == 0;
                run_and_compare(op, &a, &b, &c, &su, g == Gen::SmallInt);
            }
            // Inline-zero C (`src_c = 0`): D = A*B, in both register files.
            for cd_acc in [false, true] {
                let g = Gen::Wide;
                let (a, b, _) = rand_mats(&mut rng, &s, g);
                let zero = vec![0u64; s.m as usize * s.n as usize];
                let mut su = Setup::new(model);
                su.form = Form {
                    a_acc: cd_acc,
                    b_acc: !cd_acc,
                    cd_acc,
                };
                su.c_zero = true;
                su.in_place = false;
                run_and_compare(op, &a, &b, &zero, &su, g == Gen::SmallInt);
            }
        }
    }
}

/// With D != C the C registers stay intact and D is fully written (the
/// untouched-register assertions of `exec_mfma` cover every other register).
#[test]
fn d_distinct_from_c_leaves_c_untouched() {
    let mut rng = Rng::new(7);
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
        let mut su = Setup::new(AccumModel::Exact);
        su.in_place = false;
        let (mut w, inst, d) = build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
        run_inst(&mut w, &inst, su.model).unwrap();
        let n = usize::from(s.cd_dwords);
        assert!(
            w.v[usize::from(C0)..usize::from(C0) + n] == pack_cd(&s, &c)[..],
            "{}: C changed",
            op.name()
        );
        assert_ne!(d, C0);
        let got = unpack_cd(&s, &w.v[usize::from(d)..usize::from(d) + n]);
        let opts = RefOpts {
            invariant: true,
            ..REF_DEFAULT
        };
        assert_matrix(
            op.name(),
            &s,
            &got,
            &ref_mfma(op, &a, &b, &c, AccumModel::Exact, &opts),
        );
    }
}

/// `Exact` equals plain f64 arithmetic where that is exact (bounded
/// operands), cross-checking both the instruction and this file's `Fx`
/// reference.
#[test]
fn exact_model_matches_plain_f64_reference_on_bounded_operands() {
    let mut rng = Rng::new(0xF64);
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        if s.cd != F32 {
            continue;
        }
        let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
        for round in 0..4 {
            let (a, b, c) = rand_mats(&mut rng, &s, Gen::Bounded);
            let mut want = Vec::new();
            for i in 0..m {
                for j in 0..n {
                    let mut sum = f64::from(f32::from_bits(c[i * n + j] as u32));
                    for kk in 0..k {
                        sum += dec64(s.a, a[i * k + kk]) * dec64(s.b, b[kk * n + j]);
                    }
                    want.push(u64::from((sum as f32).to_bits()));
                }
            }
            let fx = ref_mfma(op, &a, &b, &c, AccumModel::Exact, &REF_DEFAULT);
            assert_matrix(
                &format!("{} Fx-vs-f64 round {round}", op.name()),
                &s,
                &fx,
                &want,
            );
            let mut su = Setup::new(AccumModel::Exact);
            su.form = forms()[round * 2 % 8];
            let got =
                exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su).unwrap();
            assert_matrix(
                &format!("{} exec-vs-f64 round {round}", op.name()),
                &s,
                &unpack_cd(&s, &got),
                &want,
            );
        }
    }
}

/// Small-integer F64 operands are exact in every model: plain f64 reference.
#[test]
fn f64_small_integers_match_plain_f64_in_every_model() {
    let op = MaiOp::MfmaF64_16x16x4F64;
    let s = op.shape().unwrap();
    let mut rng = Rng::new(64);
    let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
    let mut want = Vec::new();
    for i in 0..16 {
        for j in 0..16 {
            let mut sum = f64::from_bits(c[i * 16 + j]);
            for kk in 0..4 {
                sum += f64::from_bits(a[i * 4 + kk]) * f64::from_bits(b[kk * 16 + j]);
            }
            want.push(sum.to_bits());
        }
    }
    for model in MODELS {
        for form in forms() {
            let mut su = Setup::new(model);
            su.form = form;
            let got =
                exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su).unwrap();
            assert_matrix(
                &format!("{model:?} {form:?}"),
                &s,
                &unpack_cd(&s, &got),
                &want,
            );
        }
    }
}

/// `Cdna3Fused` on F32 and F64 inputs is a serial FMA (the default model's
/// documented behavior): equal to this file's Serial reference even where
/// Exact differs.
#[test]
fn cdna3_fused_is_serial_fma_for_f32_and_f64_inputs() {
    let mut rng = Rng::new(0x5E12);
    for op in [
        MaiOp::MfmaF32_16x16x4F32,
        MaiOp::MfmaF32_32x32x2F32,
        MaiOp::MfmaF64_16x16x4F64,
    ] {
        let s = op.shape().unwrap();
        let mut differs = false;
        for _ in 0..4 {
            let (a, b, c) = rand_mats(&mut rng, &s, Gen::Wide);
            let serial = ref_mfma(op, &a, &b, &c, AccumModel::Serial, &REF_DEFAULT);
            let exact = ref_mfma(op, &a, &b, &c, AccumModel::Exact, &REF_DEFAULT);
            differs |= serial != exact;
            let got = exec_mfma(
                op,
                &pack_a(&s, &a),
                &pack_b(&s, &b),
                &pack_cd(&s, &c),
                &Setup::new(AccumModel::Cdna3Fused),
            )
            .unwrap();
            assert_matrix(op.name(), &s, &unpack_cd(&s, &got), &serial);
        }
        assert!(
            differs,
            "{}: random data never separated Serial from Exact",
            op.name()
        );
    }
}

/// Wide-gen data separates the explicit models (the reference itself is
/// exercised on rounding-sensitive inputs, not just exact ones).
#[test]
fn wide_data_separates_exact_serial_and_grouped() {
    let mut rng = Rng::new(0xD1FF);
    for op in [
        MaiOp::MfmaF32_16x16x16F16,
        MaiOp::MfmaF32_32x32x8Bf16,
        MaiOp::MfmaF32_16x16x32Bf8Bf8,
    ] {
        let s = op.shape().unwrap();
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::Wide);
        let e = ref_mfma(op, &a, &b, &c, AccumModel::Exact, &REF_DEFAULT);
        let r = ref_mfma(op, &a, &b, &c, AccumModel::Serial, &REF_DEFAULT);
        let g = ref_mfma(op, &a, &b, &c, AccumModel::Grouped(2), &REF_DEFAULT);
        assert!(
            e != r,
            "{}: Wide data never separated Exact from Serial",
            op.name()
        );
        let _ = g;
    }
}

// ---------------------------------------------------------------------------
// Tests: FP8 / BF8

#[test]
fn fp8_bf8_known_codes_and_nan() {
    // CDNA3 ISA §7.2 Table 30 p.50: FP8 max 240 (0x7f), min normal 2^-7, min denorm 2^-10;
    // BF8 max 57344 (0x7f), min normal 2^-15, min denorm 2^-17; 0x80 = NaN.
    assert_eq!(f8_dec(Fp8, 0x7f), 240.0);
    assert_eq!(f8_dec(Fp8, 0xff), -240.0);
    assert_eq!(f8_dec(Fp8, 0x08), 2.0f32.powi(-7));
    assert_eq!(f8_dec(Fp8, 0x01), 2.0f32.powi(-10));
    assert_eq!(f8_dec(Bf8, 0x7f), 57344.0);
    assert_eq!(f8_dec(Bf8, 0x04), 2.0f32.powi(-15));
    assert_eq!(f8_dec(Bf8, 0x01), 2.0f32.powi(-17));
    assert_eq!(f8_dec(Fp8, 0x00), 0.0);
    assert!(f8_dec(Fp8, 0x80).is_nan() && f8_dec(Bf8, 0x80).is_nan());
}

/// All 255 non-NaN codes of every A x B pairing, both shapes, 4 mixes: each
/// output holds exactly one nonzero product `A[i,k0]*B[k0,j]` (everything else
/// +0), so the result must equal the helper-decoded values' product (exact in
/// f32 for these formats). Signed zero of an all-zero sum is not asserted
/// (`[UNVERIFIED]` hardware behavior): expected zeros accept either sign.
#[test]
fn fp8_bf8_every_code_pair_both_shapes_all_mixes() {
    let codes: Vec<u8> = (0..=255u8).filter(|c| *c != 0x80).collect();
    assert_eq!(codes.len(), 255);
    let ops = [
        MaiOp::MfmaF32_16x16x32Fp8Fp8,
        MaiOp::MfmaF32_16x16x32Fp8Bf8,
        MaiOp::MfmaF32_16x16x32Bf8Fp8,
        MaiOp::MfmaF32_16x16x32Bf8Bf8,
        MaiOp::MfmaF32_32x32x16Fp8Fp8,
        MaiOp::MfmaF32_32x32x16Fp8Bf8,
        MaiOp::MfmaF32_32x32x16Bf8Fp8,
        MaiOp::MfmaF32_32x32x16Bf8Bf8,
    ];
    for op in ops {
        let s = op.shape().unwrap();
        let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
        let chunk = |size: usize, idx: usize| -> Vec<u8> {
            (0..size)
                .map(|i| codes[(idx * size + i) % codes.len()])
                .collect()
        };
        let mut run = 0usize;
        let mut seen = std::collections::HashSet::new();
        for ca in 0..codes.len().div_ceil(m) {
            for cb in 0..codes.len().div_ceil(n) {
                let (av, bv) = (chunk(m, ca), chunk(n, cb));
                let k0 = run % k;
                run += 1;
                let mut a = vec![0u64; m * k];
                let mut b = vec![0u64; k * n];
                for i in 0..m {
                    a[i * k + k0] = u64::from(av[i]);
                }
                for j in 0..n {
                    b[k0 * n + j] = u64::from(bv[j]);
                }
                let c = vec![0u64; m * n];
                for model in [AccumModel::Cdna3Fused, AccumModel::Exact] {
                    let mut su = Setup::new(model);
                    su.form = forms()[run % 8];
                    su.in_place = run % 2 == 0;
                    let got = unpack_cd(
                        &s,
                        &exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su)
                            .unwrap_or_else(|e| panic!("{}: {e}", op.name())),
                    );
                    for i in 0..m {
                        for j in 0..n {
                            seen.insert((av[i], bv[j]));
                            let want = (f64::from(f8_dec(s.a, av[i]))
                                * f64::from(f8_dec(s.b, bv[j]))
                                + 0.0) as f32;
                            let g = f32::from_bits(got[i * n + j] as u32);
                            let ok = if want == 0.0 {
                                g == 0.0
                            } else {
                                g.to_bits() == want.to_bits()
                            };
                            assert!(
                                ok,
                                "{} {model:?} A={:#04x} B={:#04x}: got {g:e}, want {want:e}",
                                op.name(),
                                av[i],
                                bv[j]
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            seen.len(),
            255 * 255,
            "{}: not every code pair was exercised",
            op.name()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: I8

#[test]
fn i8_random_and_signed_extremes_all_forms_wrap_exactly() {
    let mut rng = Rng::new(0x18);
    for op in [MaiOp::MfmaI32_16x16x32I8, MaiOp::MfmaI32_32x32x16I8] {
        let s = op.shape().unwrap();
        for form in forms() {
            let (a, b, c) = rand_mats(&mut rng, &s, Gen::Wide);
            let mut su = Setup::new(AccumModel::Cdna3Fused);
            su.form = form;
            run_and_compare(op, &a, &b, &c, &su, false);
        }
        let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
        // (-128)*(-128) = 16384 per term; K terms; C near i32::MAX wraps.
        for (av, bv, cv, per_term) in [
            (0x80u64, 0x80u64, i32::MAX - 100, 16384i32),
            (0x80, 0x7f, i32::MIN + 5, -16256),
            (0x7f, 0x7f, i32::MAX, 16129),
            (0x80, 0xff, i32::MIN, 128),
        ] {
            let a = vec![av; m * k];
            let b = vec![bv; k * n];
            let c = vec![u64::from(cv as u32); m * n];
            let want = cv.wrapping_add(per_term * k as i32);
            let got = exec_mfma(
                op,
                &pack_a(&s, &a),
                &pack_b(&s, &b),
                &pack_cd(&s, &c),
                &Setup::new(AccumModel::Serial),
            )
            .unwrap();
            assert!(
                unpack_cd(&s, &got)
                    .iter()
                    .all(|x| *x == u64::from(want as u32)),
                "{}: A={av:#x} B={bv:#x} C={cv}: want {want}",
                op.name()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: BLGP / CBSZ / ABID / F64 modifiers

#[test]
fn blgp_all_values_every_non_f64_shape() {
    let mut rng = Rng::new(0xB16);
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        if s.cd == F64 {
            continue;
        }
        for blgp in 0..8u8 {
            let g = Gen::Wide;
            let (a, b, c) = rand_mats(&mut rng, &s, g);
            let mut su = Setup::new(AccumModel::Cdna3Fused);
            su.ctl.blgp = blgp;
            su.form = forms()[usize::from(blgp)];
            run_and_compare(op, &a, &b, &c, &su, g == Gen::SmallInt);
        }
    }
}

#[test]
fn cbsz_abid_undefined_for_one_block_shapes_is_rejected() {
    // §7.1.5.1 p.49: (1 << CBSZ) above the block count is undefined; ABID >= 1 << CBSZ names no block.
    let mut rng = Rng::new(0xCB);
    let bad = [
        (1u8, 0u8),
        (2, 0),
        (3, 0),
        (4, 0),
        (1, 1),
        (2, 3),
        (0, 1),
        (0, 2),
        (0, 15),
        (4, 15),
    ];
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        if s.cd == F64 {
            continue;
        }
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
        for (cbsz, abid) in bad {
            let mut su = Setup::new(AccumModel::Exact);
            su.ctl = Ctl {
                cbsz,
                abid,
                blgp: 0,
            };
            let (mut w, inst, _) =
                build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
            let (v0, a0) = (w.v.clone(), w.a.clone());
            let r = run_inst(&mut w, &inst, su.model);
            assert!(
                matches!(r, Err(SimError::Unsupported(_))),
                "{} CBSZ={cbsz} ABID={abid}: {r:?}",
                op.name()
            );
            assert!(
                w.v == v0 && w.a == a0,
                "{}: rejected MFMA modified registers",
                op.name()
            );
        }
    }
}

#[test]
fn f64_blgp_negates_a_b_c_and_ignores_cbsz_abid() {
    let op = MaiOp::MfmaF64_16x16x4F64;
    let s = op.shape().unwrap();
    let mut rng = Rng::new(0xF640);
    let (a, b, c) = rand_mats(&mut rng, &s, Gen::Wide);
    for blgp in 0..8u8 {
        // §7.1.5.3 p.50: BLGP[0]/[1]/[2] negate A/B/C; CBSZ and ABID are ignored.
        for (cbsz, abid) in [(0u8, 0u8), (1, 1), (4, 15), (3, 9), (7, 15)] {
            for model in [
                AccumModel::Exact,
                AccumModel::Serial,
                AccumModel::Cdna3Fused,
            ] {
                let mut su = Setup::new(model);
                su.ctl = Ctl { cbsz, abid, blgp };
                su.form = forms()[usize::from(blgp)];
                run_and_compare(op, &a, &b, &c, &su, false);
            }
        }
        // Independent expectation of the negation: D = (-1)^c C + (-1)^(a+b) A.B on small integers.
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
        let mut su = Setup::new(AccumModel::Exact);
        su.ctl = Ctl {
            cbsz: 0,
            abid: 0,
            blgp,
        };
        let got = unpack_cd(
            &s,
            &exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su).unwrap(),
        );
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = if blgp & 4 != 0 {
                    -f64::from_bits(c[i * 16 + j])
                } else {
                    f64::from_bits(c[i * 16 + j])
                };
                for kk in 0..4 {
                    let (mut x, mut y) = (
                        f64::from_bits(a[i * 4 + kk]),
                        f64::from_bits(b[kk * 16 + j]),
                    );
                    if blgp & 1 != 0 {
                        x = -x;
                    }
                    if blgp & 2 != 0 {
                        y = -y;
                    }
                    sum += x * y;
                }
                assert_eq!(
                    f64::from_bits(got[i * 16 + j]),
                    sum,
                    "BLGP {blgp} D[{i}][{j}]"
                );
            }
        }
    }
}

/// `mfma_reference` applies the §7.1.5 formulas even for CBSZ/ABID the
/// instruction rejects; compared with this file's lane-permutation model.
#[test]
fn pure_reference_matches_independent_reference_including_broadcast_formulas() {
    let mut rng = Rng::new(0x9E);
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        let (m, n) = (s.m as usize, s.n as usize);
        let _ = (m, n);
        for model in [
            AccumModel::Exact,
            AccumModel::Serial,
            AccumModel::Grouped(4),
            AccumModel::Cdna3Fused,
        ] {
            let g = Gen::Wide;
            let (a, b, c) = rand_mats(&mut rng, &s, g);
            let (ar, br, cr) = (pack_a(&s, &a), pack_b(&s, &b), pack_cd(&s, &c));
            let mut ctls = vec![NO_CTL];
            if s.cd == F64 {
                for blgp in 0..8u8 {
                    ctls.push(Ctl {
                        cbsz: 3,
                        abid: 5,
                        blgp,
                    });
                }
            } else {
                for cbsz in 0..=4u8 {
                    for abid in 0..(1u8 << cbsz) {
                        ctls.push(Ctl {
                            cbsz,
                            abid,
                            blgp: (abid + cbsz) % 8,
                        });
                    }
                }
            }
            for ctl in ctls {
                let (ae, be) = effective(op, &a, &b, ctl);
                let opts = RefOpts {
                    rm: Rm::Rne,
                    denorm32: 3,
                    invariant: g == Gen::SmallInt,
                    neg: ctl.blgp,
                };
                let want = pack_cd(&s, &ref_mfma(op, &ae, &be, &c, model, &opts));
                let got = mfma_reference(op, &ar, &br, &cr, ctl.cbsz, ctl.abid, ctl.blgp, model);
                assert!(
                    got == want,
                    "{} {model:?} {ctl:?}: mfma_reference differs from the independent reference",
                    op.name()
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: denormals and modes (§7.3 p.52)

fn single_cell(op: MaiOp, a00: u64, b00: u64, c00: u64, mode: Mode, model: AccumModel) -> Vec<u64> {
    let s = op.shape().unwrap();
    let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
    let mut a = vec![0u64; m * k];
    let mut b = vec![0u64; k * n];
    let mut c = vec![0u64; m * n];
    a[0] = a00;
    b[0] = b00;
    c[0] = c00;
    let mut su = Setup::new(model);
    su.mode = mode;
    unpack_cd(
        &s,
        &exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su)
            .unwrap_or_else(|e| panic!("{}: {e}", op.name())),
    )
}

#[test]
fn f32_inputs_obey_mode_denorm_but_c_and_d_do_not() {
    // §7.3 p.52: F32 A/B honor MODE.denorm (input flush when bit 0 is clear); C and D never flush.
    for op in [MaiOp::MfmaF32_16x16x4F32, MaiOp::MfmaF32_32x32x2F32] {
        for d in 0..4u8 {
            let mode = Mode {
                denorm32: d,
                ..Mode::default()
            };
            let flush_in = d & 1 == 0;
            for model in [
                AccumModel::Cdna3Fused,
                AccumModel::Exact,
                AccumModel::Serial,
            ] {
                let big = enc(F32, p2(100));
                let denorm = 0x0000_4000u64; // 2^14 * 2^-149 = 2^-135
                let want = if flush_in { 0 } else { enc(F32, p2(-35)) };
                let d00 = single_cell(op, denorm, big, 0, mode, model);
                assert_eq!(
                    d00[0],
                    want,
                    "{} denorm32={d} {model:?} denormal A",
                    op.name()
                );
                let d00 = single_cell(op, big, denorm, 0, mode, model);
                assert_eq!(
                    d00[0],
                    want,
                    "{} denorm32={d} {model:?} denormal B",
                    op.name()
                );
                // Denormal result of normal inputs: 2^-100 * 2^-40 = 2^-140 = bits 0x200, kept.
                let d00 = single_cell(op, enc(F32, p2(-100)), enc(F32, p2(-40)), 0, mode, model);
                assert_eq!(
                    d00[0],
                    0x200,
                    "{} denorm32={d} {model:?} denormal D",
                    op.name()
                );
                // Denormal C passes through unchanged.
                let d00 = single_cell(op, 0, 0, 3, mode, model);
                assert_eq!(d00[0], 3, "{} denorm32={d} {model:?} denormal C", op.name());
            }
        }
    }
}

#[test]
fn sub_32_bit_inputs_and_f64_ignore_mode_denorm() {
    // §7.3 p.52: F16/BF16/FP8/BF8 ignore MODE.denorm; F64 ignores MODE and rounds to nearest even;
    // §7.1.2 Table 28 p.42: MFMA RoundMode ignores MODE and forces RNE.
    let mode = Mode {
        denorm32: 0,
        denorm16_64: 0,
        round32: 1,
        round16_64: 1,
        ..Mode::default()
    };
    for op in mfma_ops() {
        let s = op.shape().unwrap();
        match s.a {
            F16 | Bf16 | Fp8 | Bf8 => {
                let (a, b) = match s.a {
                    F16 => (0x0200u64, enc(F16, p2(14))),
                    Bf16 => (0x0040, enc(Bf16, p2(120))),
                    _ => (1, 0),
                };
                let (a, b) = if matches!(s.a, Fp8 | Bf8) {
                    (1, enc(s.b, if s.b == Fp8 { 128.0 } else { 32768.0 }))
                } else {
                    (a, b)
                };
                let want = (dec64(s.a, a) * dec64(s.b, b)) as f32;
                assert!(want != 0.0);
                for model in [
                    AccumModel::Cdna3Fused,
                    AccumModel::Exact,
                    AccumModel::Serial,
                ] {
                    let d00 = single_cell(op, a, b, 0, mode, model);
                    assert_eq!(
                        d00[0],
                        u64::from(want.to_bits()),
                        "{} {model:?}: denormal operand flushed",
                        op.name()
                    );
                }
            }
            F64 => {
                for model in [
                    AccumModel::Cdna3Fused,
                    AccumModel::Exact,
                    AccumModel::Serial,
                ] {
                    // 2^-1074 * 2^1000 = 2^-74; 2^-600 * 2^-474 = 2^-1074 (denormal D kept).
                    let d00 =
                        single_cell(op, p2(-1074).to_bits(), p2(1000).to_bits(), 0, mode, model);
                    assert_eq!(d00[0], p2(-74).to_bits(), "{model:?} F64 denormal input");
                    let d00 =
                        single_cell(op, p2(-600).to_bits(), p2(-474).to_bits(), 0, mode, model);
                    assert_eq!(d00[0], 1, "{model:?} F64 denormal output");
                    // RTP would give 1 + 2^-52; MODE is ignored, RNE gives 1.0.
                    let d00 = single_cell(
                        op,
                        p2(-27).to_bits(),
                        p2(-27).to_bits(),
                        1.0f64.to_bits(),
                        mode,
                        model,
                    );
                    assert_eq!(
                        d00[0],
                        1.0f64.to_bits(),
                        "{model:?} F64 must ignore MODE.round"
                    );
                }
            }
            _ => {}
        }
    }
}

/// §7.1.2 Table 28 p.42: the MFMA rounding mode ignores MODE and is forced to RNE: every
/// `MODE.round32` / `round16_64` value gives the nearest-even result, in every model.
#[test]
fn mfma_ignores_mode_rounding_and_always_rounds_to_nearest_even() {
    // F16/BF16/F32: C = 1.0 + a half-ulp product 2^-24 ties to even (1.0; RTP would give 1 + 2^-23).
    // FP8/BF8: C = 2^20 + product 2^-4 (half-ulp tie) -> 2^20 (RTP would give 2^20 + 2^-3).
    for op in f32_ops() {
        let s = op.shape().unwrap();
        let (p, rne) = if matches!(s.a, Fp8 | Bf8) {
            (
                Probe {
                    a: vec![p2(-2)],
                    b: vec![p2(-2)],
                    c: p2(20),
                },
                0x4980_0000u64,
            )
        } else {
            (
                Probe {
                    a: vec![p2(-12)],
                    b: vec![p2(-12)],
                    c: 1.0,
                },
                0x3f80_0000,
            )
        };
        for r32 in 0..4u8 {
            for r16 in 0..4u8 {
                let mode = Mode {
                    round32: r32,
                    round16_64: r16,
                    ..Mode::default()
                };
                for model in [
                    AccumModel::Exact,
                    AccumModel::Serial,
                    AccumModel::Cdna3Fused,
                ] {
                    let d = run_probe_res(op, &p, model, mode).unwrap_or_else(|e| {
                        panic!("{} round32={r32} round16_64={r16}: {e}", op.name())
                    });
                    assert_eq!(
                        d,
                        rne,
                        "{} {model:?} round32={r32} round16_64={r16}",
                        op.name()
                    );
                }
            }
        }
    }
}

/// §3.5 Table 4 p.10-11 (FP16_OVFL) is not modelled for MFMA: the flag is rejected, not ignored.
#[test]
fn fp16_ovfl_with_mfma_is_unsupported() {
    let mut rng = Rng::new(0x0F);
    for op in f32_ops() {
        let s = op.shape().unwrap();
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
        let mut su = Setup::new(AccumModel::Exact);
        su.mode = Mode {
            fp16_ovfl: true,
            ..Mode::default()
        };
        let (mut w, inst, _) = build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
        let (v0, a0) = (w.v.clone(), w.a.clone());
        let r = run_inst(&mut w, &inst, su.model);
        assert!(
            matches!(r, Err(SimError::Unsupported(_))),
            "{} FP16_OVFL=1: {r:?}",
            op.name()
        );
        assert!(
            w.v == v0 && w.a == a0,
            "{}: rejected MFMA wrote registers",
            op.name()
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: adversarial expected-value probes (hand-derived, beyond f64)

struct Probe {
    a: Vec<f64>,
    b: Vec<f64>,
    c: f64,
}

fn probe_mats(s: &MfmaShape, p: &Probe) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let (m, n, k) = (s.m as usize, s.n as usize, s.k as usize);
    assert!(p.a.len() <= k && p.b.len() <= k);
    let ea: Vec<u64> = p.a.iter().map(|v| enc(s.a, *v)).collect();
    let eb: Vec<u64> = p.b.iter().map(|v| enc(s.b, *v)).collect();
    let mut a = vec![0u64; m * k];
    let mut b = vec![0u64; k * n];
    for i in 0..m {
        for (kk, v) in ea.iter().enumerate() {
            a[i * k + kk] = *v;
        }
    }
    for j in 0..n {
        for (kk, v) in eb.iter().enumerate() {
            b[kk * n + j] = *v;
        }
    }
    (a, b, vec![enc(s.cd, p.c); m * n])
}
fn run_probe_res(op: MaiOp, p: &Probe, model: AccumModel, mode: Mode) -> SimResult<u64> {
    let s = op.shape().unwrap();
    let (a, b, c) = probe_mats(&s, p);
    let mut su = Setup::new(model);
    su.mode = mode;
    su.form = Form {
        a_acc: true,
        b_acc: false,
        cd_acc: false,
    };
    let d = unpack_cd(
        &s,
        &exec_mfma(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su)?,
    );
    assert!(
        d.iter().all(|x| *x == d[0]),
        "{}: probe output is not uniform",
        op.name()
    );
    Ok(d[0])
}
fn run_probe(op: MaiOp, p: &Probe, model: AccumModel) -> u64 {
    run_probe_res(op, p, model, Mode::default())
        .unwrap_or_else(|e| panic!("{} {model:?}: {e}", op.name()))
}
fn f32_ops() -> Vec<MaiOp> {
    mfma_ops()
        .into_iter()
        .filter(|o| o.shape().unwrap().cd == F32)
        .collect()
}

/// `(role 0 = A, role 1 = B)` operand values whose product is exactly the
/// largest the format pair supports in this test.
fn big(t: MfmaType, role: usize) -> f64 {
    match t {
        F32 | Bf16 => {
            if role == 0 {
                p2(100)
            } else {
                p2(27)
            }
        }
        F16 => p2(15),
        Fp8 => 240.0,
        Bf8 => 57344.0,
        _ => unreachable!(),
    }
}

fn g_bits(model: AccumModel, g1: u64, g2: u64) -> u64 {
    // Result when every group of this model contains both terms (g2) or splits them (g1).
    match model {
        AccumModel::Grouped(1) | AccumModel::Serial => g1,
        _ => g2,
    }
}

#[test]
fn wide_cancellation_beyond_f64_keeps_the_denormal_c() {
    // C = 2^-149 (smallest F32 denormal), T0 = +big, T1 = -big. The exact sum is
    // 2^-149; anything that adds C to the huge partial sum (f64 or serial F32) loses it.
    for op in f32_ops() {
        let s = op.shape().unwrap();
        if s.k < 2 {
            continue;
        }
        let p = Probe {
            a: vec![big(s.a, 0), -big(s.a, 0)],
            b: vec![big(s.b, 1), big(s.b, 1)],
            c: p2(-149),
        };
        for model in MODELS {
            let want = match model {
                AccumModel::Exact | AccumModel::Grouped(2..) => 1,
                AccumModel::Serial | AccumModel::Grouped(1) | AccumModel::Grouped(0) => 0,
                // F32 inputs: serial FMA. Narrower inputs: the truncating candidate drops C (floor / discard).
                AccumModel::Cdna3Fused => 0,
            };
            assert_eq!(run_probe(op, &p, model), want, "{} {model:?}", op.name());
        }
    }
}

#[test]
fn half_ulp_tie_probe_separates_models() {
    // C = 2^-13, T0 = 2^14 (k=0), T1 = 2^-10 (k=1). ulp(2^14) = 2^-9, so 2^-10 is
    // exactly a half-ulp tie that only the extra 2^-13 breaks upward.
    // Exact: 2^14 + 2^-9 (next float up). Serial: 2^14 + 2^-13 -> 2^14, then tie -> even 2^14.
    // Grouped(g>=2): one group holds both -> next up. Cdna3Fused: serial FMA (F32) or
    // truncated alignment dropping C (narrower inputs): 2^14.
    let (base, next) = (0x4680_0000u64, 0x4680_0001u64);
    for op in f32_ops() {
        let s = op.shape().unwrap();
        let (x, y) = match s.a {
            F32 | Bf16 | F16 => (p2(7), p2(-5)),
            _ => (p2(7), p2(-5)),
        };
        let p = Probe {
            a: vec![x, y],
            b: vec![x, y],
            c: p2(-13),
        };
        for model in MODELS {
            let want = match model {
                AccumModel::Exact => next,
                AccumModel::Cdna3Fused => base,
                m => g_bits(m, base, next),
            };
            assert_eq!(run_probe(op, &p, model), want, "{} {model:?}", op.name());
        }
    }
}

#[test]
fn group_boundary_probe_matches_each_models_group_size() {
    // T0 = 2^14 (k=0), tiny = 2^-10 at k=1 and at k=G where G is the CDNA3 fused
    // group (8 for F16/BF16, 16 for FP8/BF8). Same group -> exact sum rounds up;
    // split groups -> each tie rounds to even, staying at 2^14.
    let (base, next) = (0x4680_0000u64, 0x4680_0001u64);
    for op in f32_ops() {
        let s = op.shape().unwrap();
        let g = match s.a {
            F16 | Bf16 => 8,
            Fp8 | Bf8 => 16,
            _ => continue,
        };
        let k = s.k as usize;
        let tiny_at = if k > g { g } else { 2 };
        let mut a = vec![0.0; tiny_at + 1];
        let mut b = vec![0.0; tiny_at + 1];
        a[0] = p2(7);
        b[0] = p2(7);
        for idx in [1, tiny_at] {
            a[idx] = p2(-5);
            b[idx] = p2(-5);
        }
        let p = Probe { a, b, c: 0.0 };
        for model in MODELS {
            let want = match model {
                AccumModel::Exact => next,
                AccumModel::Serial | AccumModel::Grouped(1) => base,
                AccumModel::Grouped(gs) => {
                    if usize::from(gs) > tiny_at {
                        next
                    } else {
                        base
                    }
                }
                // Two groups (k > G): each tie rounds down. One group (k <= G): sum is exact.
                AccumModel::Cdna3Fused => {
                    if k > g {
                        base
                    } else {
                        next
                    }
                }
            };
            assert_eq!(
                run_probe(op, &p, model),
                want,
                "{} {model:?} (tiny products at k=1 and k={tiny_at})",
                op.name()
            );
        }
    }
}

#[test]
fn asymmetric_round_down_of_c_versus_fp8_discard() {
    // T0 = 2^14, C = -2^-13. Exact/Serial/Grouped: 2^14 - 2^-13 rounds to 2^14.
    // Cdna3Fused (MMA-Sim): F32 serial FMA -> 2^14; F16/BF16 align C to 2^-10 with
    // round-DOWN -> 2^14 - 2^-10 = the float just below 2^14; FP8/BF8 discard a C
    // more than 25 exponents below the products -> 2^14.
    let (base, prev) = (0x4680_0000u64, 0x467f_ffffu64);
    for op in f32_ops() {
        let s = op.shape().unwrap();
        let p = Probe {
            a: vec![p2(7)],
            b: vec![p2(7)],
            c: -p2(-13),
        };
        for model in MODELS {
            let want = match (model, s.a) {
                (AccumModel::Cdna3Fused, F16 | Bf16) => prev,
                _ => base,
            };
            assert_eq!(run_probe(op, &p, model), want, "{} {model:?}", op.name());
        }
    }
}

#[test]
fn fp8_c_discard_breaks_a_half_ulp_tie_only_in_exact_models() {
    // FP8/BF8: T0 = 2^14, T1 = 2^-10 (an exact half-ulp tie), C = 2^-13.
    // Exact keeps C as a sticky bit (rounds up); Cdna3Fused discards C (gap > 25) and ties to even.
    for op in f32_ops() {
        let s = op.shape().unwrap();
        if !matches!(s.a, Fp8 | Bf8) {
            continue;
        }
        let p = Probe {
            a: vec![p2(7), p2(-5)],
            b: vec![p2(7), p2(-5)],
            c: p2(-13),
        };
        assert_eq!(
            run_probe(op, &p, AccumModel::Exact),
            0x4680_0001,
            "{}",
            op.name()
        );
        assert_eq!(
            run_probe(op, &p, AccumModel::Cdna3Fused),
            0x4680_0000,
            "{}",
            op.name()
        );
    }
}

#[test]
fn overflow_probes_for_wide_exponent_inputs() {
    for op in f32_ops() {
        let s = op.shape().unwrap();
        if !matches!(s.a, F32 | Bf16) {
            continue;
        }
        let f32_serial = s.a == F32;
        // (1) C = 1, T0 = +2^128, T1 = -2^128: exact sum 1.0; serial overflows to +inf and stays there.
        let p = Probe {
            a: vec![p2(127), -p2(127)],
            b: vec![2.0, 2.0],
            c: 1.0,
        };
        for model in MODELS {
            let want = match model {
                AccumModel::Exact | AccumModel::Grouped(2..) => 0x3f80_0000,
                AccumModel::Serial | AccumModel::Grouped(1) | AccumModel::Grouped(0) => 0x7f80_0000,
                // BF16 under Cdna3Fused turns the opposite infinities into a NaN: [UNVERIFIED] bit pattern, not asserted.
                AccumModel::Cdna3Fused => {
                    if !f32_serial {
                        continue;
                    }
                    0x7f80_0000
                }
            };
            assert_eq!(
                run_probe(op, &p, model),
                want,
                "{} {model:?} cancel-overflow",
                op.name()
            );
        }
        // (2) A single product of 2^128 overflows in every model.
        let p = Probe {
            a: vec![p2(127)],
            b: vec![2.0],
            c: 0.0,
        };
        for model in MODELS {
            assert_eq!(
                run_probe(op, &p, model),
                0x7f80_0000,
                "{} {model:?} product overflow",
                op.name()
            );
        }
        // (3) C = F32::MAX: adding exactly half an ulp (2^103) ties to even = overflow; 2^102 does not.
        let p = Probe {
            a: vec![p2(51)],
            b: vec![p2(52)],
            c: f64::from(f32::MAX),
        };
        for model in MODELS {
            assert_eq!(
                run_probe(op, &p, model),
                0x7f80_0000,
                "{} {model:?} tie overflow",
                op.name()
            );
        }
        let p = Probe {
            a: vec![p2(50)],
            b: vec![p2(52)],
            c: f64::from(f32::MAX),
        };
        for model in MODELS {
            assert_eq!(
                run_probe(op, &p, model),
                0x7f7f_ffff,
                "{} {model:?} below-tie",
                op.name()
            );
        }
    }
}

#[test]
fn f64_wide_probes_cancellation_tie_and_overflow() {
    let op = MaiOp::MfmaF64_16x16x4F64;
    // Cancellation beyond double: T0 = 2^1000, T1 = -2^1000, C = 2^-1000.
    let p = Probe {
        a: vec![p2(500), -p2(500)],
        b: vec![p2(500), p2(500)],
        c: p2(-1000),
    };
    for model in MODELS {
        let want = match model {
            AccumModel::Exact | AccumModel::Grouped(2..) => p2(-1000).to_bits(),
            _ => 0,
        };
        assert_eq!(run_probe(op, &p, model), want, "{model:?} cancellation");
    }
    // Tie: C = 1, two products of 2^-53 (a half-ulp each).
    let p = Probe {
        a: vec![p2(-27), p2(-27)],
        b: vec![p2(-26), p2(-26)],
        c: 1.0,
    };
    for model in MODELS {
        let want = match model {
            AccumModel::Exact | AccumModel::Grouped(2..) => 0x3ff0_0000_0000_0001,
            _ => 1.0f64.to_bits(),
        };
        assert_eq!(run_probe(op, &p, model), want, "{model:?} tie");
    }
    // Overflow: C = 1, +2^1024, -2^1024.
    let p = Probe {
        a: vec![p2(1023), -p2(1023)],
        b: vec![2.0, 2.0],
        c: 1.0,
    };
    for model in MODELS {
        let want = match model {
            AccumModel::Exact | AccumModel::Grouped(2..) => 1.0f64.to_bits(),
            _ => f64::INFINITY.to_bits(),
        };
        assert_eq!(run_probe(op, &p, model), want, "{model:?} overflow");
    }
}

// ---------------------------------------------------------------------------
// Tests: errors

/// §7.1.2 Table 28 p.42: the MFMA execution mask "forces 1 for all threads": EXEC (partial or
/// zero) neither suppresses nor selects lanes. Every register file entry must equal the full-EXEC
/// run, including all 64 lanes of D.
#[test]
fn mfma_forces_all_lanes_regardless_of_exec() {
    let mut rng = Rng::new(0xE7);
    for op in [
        MaiOp::MfmaF32_16x16x16F16,
        MaiOp::MfmaI32_32x32x16I8,
        MaiOp::MfmaF64_16x16x4F64,
        MaiOp::MfmaF32_16x16x4F32,
        MaiOp::MfmaF32_32x32x16Bf8Fp8,
    ] {
        let s = op.shape().unwrap();
        let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
        let mut su = Setup::new(AccumModel::Exact);
        su.in_place = false;
        run_and_compare(op, &a, &b, &c, &su, true);
        let (mut full, inst, d) =
            build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
        run_inst(&mut full, &inst, su.model).unwrap();
        for exec in [
            0u64,
            1,
            !1,
            0xffff_ffff,
            0xaaaa_aaaa_aaaa_aaaa,
            0x8000_0000_0000_0000,
        ] {
            let (mut w, inst, _) =
                build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
            w.exec = exec;
            run_inst(&mut w, &inst, su.model)
                .unwrap_or_else(|e| panic!("{} EXEC={exec:#x}: {e}", op.name()));
            let n = usize::from(s.cd_dwords);
            assert!(
                w.v[usize::from(d)..usize::from(d) + n]
                    == full.v[usize::from(d)..usize::from(d) + n],
                "{} EXEC={exec:#x}: D differs from the full-wave result",
                op.name()
            );
            assert!(
                w.v == full.v && w.a == full.a,
                "{} EXEC={exec:#x}: register state differs",
                op.name()
            );
        }
    }
}

#[test]
fn invalid_operand_forms_and_configs_are_unsupported() {
    let op = MaiOp::MfmaF32_16x16x16F16;
    let s = op.shape().unwrap();
    let mut rng = Rng::new(1);
    let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
    let su = Setup::new(AccumModel::Exact);
    let (w0, inst, _) = build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
    let Inst::Mai(mai) = inst else { unreachable!() };
    let variants: Vec<(&str, Mai)> = vec![
        (
            "ACC_CD mismatch: dst VGPR, C AGPR",
            Mai {
                src_c: Some(Src::Agpr(C0)),
                ..mai.clone()
            },
        ),
        (
            "ACC_CD mismatch: dst AGPR, C VGPR",
            Mai {
                dst: Src::Agpr(D0),
                ..mai.clone()
            },
        ),
        (
            "inline-constant A",
            Mai {
                src_a: Src::Int(1),
                ..mai.clone()
            },
        ),
        (
            "inline-constant B",
            Mai {
                src_b: Some(Src::Int(1)),
                ..mai.clone()
            },
        ),
        (
            "missing B",
            Mai {
                src_b: None,
                ..mai.clone()
            },
        ),
        (
            "missing C",
            Mai {
                src_c: None,
                ..mai.clone()
            },
        ),
        (
            "BLGP out of range",
            Mai {
                blgp: 8,
                ..mai.clone()
            },
        ),
    ];
    for (what, v) in variants {
        let mut w = Wave::new(Arch::Gfx942, 0, POISON);
        w.v = w0.v.clone();
        w.a = w0.a.clone();
        let r = run_inst(&mut w, &Inst::Mai(v), AccumModel::Exact);
        assert!(matches!(r, Err(SimError::Unsupported(_))), "{what}: {r:?}");
    }
    let mut w = Wave::new(Arch::Gfx942, 0, POISON);
    w.v = w0.v.clone();
    let r = run_inst(&mut w, &Inst::Mai(mai), AccumModel::Grouped(0));
    assert!(
        matches!(r, Err(SimError::Unsupported(_))),
        "Grouped(0): {r:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests: v_accvgpr_read_b32 / v_accvgpr_write_b32

fn mov(op: MaiOp, dst: Src, src: Src) -> Inst {
    Inst::Mai(Mai {
        op,
        dst,
        src_a: src,
        src_b: None,
        src_c: None,
        cbsz: 0,
        abid: 0,
        blgp: 0,
    })
}

#[test]
fn accvgpr_write_and_read_move_whole_registers() {
    let mut w = Wave::new(Arch::Gfx942, 0, POISON);
    for l in 0..64 {
        w.v[5][l] = 0x1000_0000 + (l as u32) * 0x0101;
    }
    run_inst(
        &mut w,
        &mov(MaiOp::AccvgprWriteB32, Src::Agpr(7), Src::Vgpr(5)),
        AccumModel::Exact,
    )
    .unwrap();
    assert!(w.a[7] == w.v[5]);
    run_inst(
        &mut w,
        &mov(MaiOp::AccvgprReadB32, Src::Vgpr(9), Src::Agpr(7)),
        AccumModel::Exact,
    )
    .unwrap();
    assert!(w.v[9] == w.v[5]);
    // Unrelated registers keep the poison.
    assert!(
        w.a[6] == [POISON; 64]
            && w.a[8] == [POISON; 64]
            && w.v[8] == [POISON; 64]
            && w.v[10] == [POISON; 64]
    );
}

#[test]
fn accvgpr_write_constant_and_scalar_sources() {
    let cases: [(Src, u32); 5] = [
        (Src::Int(-5), 0xffff_fffb),
        (Src::Int(64), 64),
        (Src::Int(0), 0),
        (Src::Float(InlineFloat::One), 0x3f80_0000),
        (Src::Float(InlineFloat::NegTwo), 0xc000_0000),
    ];
    for (n, (src, want)) in cases.into_iter().enumerate() {
        let mut w = Wave::new(Arch::Gfx942, 0, POISON);
        run_inst(
            &mut w,
            &mov(MaiOp::AccvgprWriteB32, Src::Agpr(n as u8), src),
            AccumModel::Exact,
        )
        .unwrap();
        assert!(w.a[n] == [want; 64], "{src:?}");
    }
    // SGPR broadcast to every lane.
    let mut w = Wave::new(Arch::Gfx942, 0, POISON);
    w.s[3] = 0x1234_5678;
    run_inst(
        &mut w,
        &mov(MaiOp::AccvgprWriteB32, Src::Agpr(2), Src::Sgpr(3)),
        AccumModel::Exact,
    )
    .unwrap();
    assert!(w.a[2] == [0x1234_5678; 64]);
}

#[test]
fn accvgpr_moves_honor_exec_per_lane() {
    for exec in [
        0x0f0f_0f0f_0f0f_0f0fu64,
        1,
        0,
        0x8000_0000_0000_0000,
        0xffff_ffff_0000_0000,
    ] {
        let mut w = Wave::new(Arch::Gfx942, 0, POISON);
        for l in 0..64 {
            w.v[5][l] = 0x4000_0000 + l as u32;
            w.a[7][l] = 0x7000_0000 + l as u32;
        }
        w.exec = exec;
        run_inst(
            &mut w,
            &mov(MaiOp::AccvgprWriteB32, Src::Agpr(11), Src::Vgpr(5)),
            AccumModel::Exact,
        )
        .unwrap();
        run_inst(
            &mut w,
            &mov(MaiOp::AccvgprReadB32, Src::Vgpr(12), Src::Agpr(7)),
            AccumModel::Exact,
        )
        .unwrap();
        for l in 0..64 {
            let on = exec >> l & 1 != 0;
            assert_eq!(
                w.a[11][l],
                if on { 0x4000_0000 + l as u32 } else { POISON },
                "write EXEC={exec:#x} lane {l}"
            );
            assert_eq!(
                w.v[12][l],
                if on { 0x7000_0000 + l as u32 } else { POISON },
                "read EXEC={exec:#x} lane {l}"
            );
        }
    }
}

#[test]
fn accvgpr_results_of_mfma_read_back_through_accvgpr_read() {
    let op = MaiOp::MfmaF32_16x16x16F16;
    let s = op.shape().unwrap();
    let mut rng = Rng::new(3);
    let (a, b, c) = rand_mats(&mut rng, &s, Gen::SmallInt);
    let mut su = Setup::new(AccumModel::Exact);
    su.form = Form {
        a_acc: false,
        b_acc: true,
        cd_acc: true,
    };
    su.in_place = false;
    let (mut w, inst, d) = build(op, &pack_a(&s, &a), &pack_b(&s, &b), &pack_cd(&s, &c), &su);
    run_inst(&mut w, &inst, su.model).unwrap();
    for r in 0..4u8 {
        run_inst(
            &mut w,
            &mov(MaiOp::AccvgprReadB32, Src::Vgpr(100 + r), Src::Agpr(d + r)),
            AccumModel::Exact,
        )
        .unwrap();
    }
    let got = unpack_cd(&s, &w.v[100..104]);
    let opts = RefOpts {
        invariant: true,
        ..REF_DEFAULT
    };
    assert_matrix(
        "accvgpr_read of D",
        &s,
        &got,
        &ref_mfma(op, &a, &b, &c, AccumModel::Exact, &opts),
    );
}

// ---------------------------------------------------------------------------
// VOP3P helpers

fn vop3p(
    op: Vop3pOp,
    dst: Src,
    srcs: &[Src],
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) -> Inst {
    let mut src = [None; 3];
    for (i, s) in srcs.iter().enumerate() {
        src[i] = Some(*s);
    }
    Inst::Vop3p(Vop3p {
        op,
        dst,
        src,
        op_sel,
        op_sel_hi,
        neg_lo,
        neg_hi,
        clamp,
    })
}
fn new_wave() -> Wave {
    Wave::new(Arch::Gfx942, 0, POISON)
}
fn vexec(w: &mut Wave, inst: Inst) -> SimResult<()> {
    run_inst(w, &inst, AccumModel::default())
}

fn gen_h(rng: &mut Rng, lo: u64, hi: u64) -> u16 {
    let sign = rng.sign() as u16;
    // Subnormals only when the caller's exponent window reaches the bottom of the format.
    if lo < 12 && rng.chance(8) {
        (sign << 15) | (1 + rng.below(1023)) as u16
    } else {
        (sign << 15) | (((lo + rng.below(hi - lo + 1)) as u16) << 10) | rng.below(1024) as u16
    }
}
fn gen_f32(rng: &mut Rng, lo: u64, hi: u64) -> u32 {
    ((rng.sign() as u32) << 31)
        | (((lo + rng.below(hi - lo + 1)) as u32) << 23)
        | rng.below(1 << 23) as u32
}
fn is_nan16(b: u16) -> bool {
    b & 0x7c00 == 0x7c00 && b & 0x3ff != 0
}

fn f16_op(op: Vop3pOp, x: [u16; 3]) -> u16 {
    let f = |b: u16| fx_from_bits(u64::from(b), F16F);
    let r = |v: Fx| round_fx(v, F16F, Rm::Rne) as u16;
    match op {
        Vop3pOp::PkFmaF16 => r(f(x[0]).mul(f(x[1])).add(f(x[2]))),
        Vop3pOp::PkAddF16 => r(f(x[0]).add(f(x[1]))),
        Vop3pOp::PkMulF16 => r(f(x[0]).mul(f(x[1]))),
        Vop3pOp::PkMinF16 => {
            if fx_lt(f(x[1]), f(x[0])) {
                x[1]
            } else {
                x[0]
            }
        }
        Vop3pOp::PkMaxF16 => {
            if fx_lt(f(x[0]), f(x[1])) {
                x[1]
            } else {
                x[0]
            }
        }
        _ => unreachable!(),
    }
}
/// clamp to [0, 1]; NaN -> 0 (MODE.DX10_CLAMP, §3.? MODE table).
fn clamp16(b: u16) -> u16 {
    if is_nan16(b) || b & 0x8000 != 0 {
        0
    } else if b >= 0x3c00 {
        0x3c00
    } else {
        b
    }
}
fn clamp32(b: u32) -> u32 {
    if (b & 0x7f80_0000 == 0x7f80_0000 && b & 0x7f_ffff != 0) || b & 0x8000_0000 != 0 {
        0
    } else if b >= 0x3f80_0000 {
        0x3f80_0000
    } else {
        b
    }
}

/// Packed 16-bit float op: lo input = `src[i].half(op_sel[i])`, hi input =
/// `src[i].half(op_sel_hi[i])`, NEG_LO / NEG_HI flip the sign of the lo / hi
/// input (CDNA3 ISA §6.7 p.38 "op_sel and neg fields for both the low and high operands").
fn pk16_ref(
    op: Vop3pOp,
    s: [u32; 3],
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) -> u32 {
    let mut out = 0u32;
    for half in 0..2u32 {
        let sel = if half == 0 { op_sel } else { op_sel_hi };
        let neg = if half == 0 { neg_lo } else { neg_hi };
        let mut x = [0u16; 3];
        for i in 0..3 {
            x[i] = (s[i] >> (16 * u32::from((sel >> i) & 1))) as u16;
            if (neg >> i) & 1 != 0 {
                x[i] ^= 0x8000;
            }
        }
        let r = f16_op(op, x);
        let r = if clamp { clamp16(r) } else { r };
        out |= u32::from(r) << (16 * half);
    }
    out
}

fn pk16_case(
    rng: &mut Rng,
    op: Vop3pOp,
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
    ef: (u64, u64),
) {
    let mut w = new_wave();
    let n_src = if op == Vop3pOp::PkFmaF16 { 3 } else { 2 };
    let mut s = [[0u32; 64]; 3];
    for i in 0..n_src {
        for l in 0..64 {
            s[i][l] = u32::from(gen_h(rng, ef.0, ef.1)) | (u32::from(gen_h(rng, ef.0, ef.1)) << 16);
        }
    }
    for i in 0..3 {
        w.v[1 + i] = s[i];
    }
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(
            op,
            Src::Vgpr(4),
            &srcs[..n_src],
            op_sel,
            op_sel_hi,
            neg_lo,
            neg_hi,
            clamp,
        ),
    )
    .unwrap();
    for l in 0..64 {
        let want = pk16_ref(
            op,
            [s[0][l], s[1][l], s[2][l]],
            op_sel,
            op_sel_hi,
            neg_lo,
            neg_hi,
            clamp,
        );
        assert_eq!(w.v[4][l], want, "{} op_sel={op_sel:#05b} op_sel_hi={op_sel_hi:#05b} neg_lo={neg_lo:#05b} neg_hi={neg_hi:#05b} clamp={clamp} lane {l}: srcs {:#010x} {:#010x} {:#010x}", op.name(), s[0][l], s[1][l], s[2][l]);
    }
}

const PK_F16: [Vop3pOp; 5] = [
    Vop3pOp::PkFmaF16,
    Vop3pOp::PkAddF16,
    Vop3pOp::PkMulF16,
    Vop3pOp::PkMinF16,
    Vop3pOp::PkMaxF16,
];

#[test]
fn pk_f16_op_sel_and_op_sel_hi_select_halves_independently() {
    let mut rng = Rng::new(0x16);
    for op in PK_F16 {
        for op_sel in 0..8u8 {
            for op_sel_hi in 0..8u8 {
                pk16_case(&mut rng, op, op_sel, op_sel_hi, 0, 0, false, (9, 21));
            }
        }
    }
}

#[test]
fn pk_f16_neg_lo_and_neg_hi_flip_each_half() {
    let mut rng = Rng::new(0x17);
    for op in PK_F16 {
        for neg_lo in 0..8u8 {
            for neg_hi in 0..8u8 {
                let (os, osh) = (rng.below(8) as u8, rng.below(8) as u8);
                pk16_case(&mut rng, op, os, osh, neg_lo, neg_hi, false, (9, 21));
            }
        }
    }
}

#[test]
fn pk_f16_clamp_saturates_to_unit_interval() {
    let mut rng = Rng::new(0x18);
    for op in PK_F16 {
        for _ in 0..24 {
            let (os, osh) = (rng.below(8) as u8, rng.below(8) as u8);
            let (nl, nh) = (rng.below(8) as u8, rng.below(8) as u8);
            pk16_case(&mut rng, op, os, osh, nl, nh, true, (12, 16));
        }
    }
    // Literal: 0.75 + 0.75 = 1.5 -> 1.0; -1.0 + 0.25 -> -0.75 -> 0; 0.25 + 0.5 = 0.75 stays.
    let mut w = new_wave();
    w.v[1] = [0x3a00 | (0xbc00 << 16); 64];
    w.v[2] = [0x3a00 | (0x3400 << 16); 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            true,
        ),
    )
    .unwrap();
    assert_eq!(
        w.v[3][0],
        0x3c00 | (0 << 16),
        "clamp(0.75+0.75)=1.0 low, clamp(-1.0+0.25)=0 high"
    );
    w.v[1] = [0x3400 | (0x3800 << 16); 64];
    w.v[2] = [0x3800 | (0x3400 << 16); 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            true,
        ),
    )
    .unwrap();
    assert_eq!(w.v[3][0], 0x3a00 | (0x3a00 << 16));
}

#[test]
fn pk_f16_literal_values_and_nan_clamp() {
    // v_pk_add_f16: (1.0, 2.0) + (3.0, -0.5): lo 4.0, hi 1.5.
    let mut w = new_wave();
    w.v[1] = [0x3c00 | (0x4000 << 16); 64];
    w.v[2] = [0x4200 | (0xb800 << 16); 64];
    w.v[3] = [0x4000 | (0x4000 << 16); 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][5], 0x4400 | (0x3e00 << 16));
    // v_pk_mul_f16: (1.0, 2.0) * (3.0, -0.5) = (3.0, -1.0).
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMulF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][5], 0x4200 | (0xbc00 << 16));
    // v_pk_fma_f16: (1.0, 2.0) * (3.0, -0.5) + 2.0 = (5.0, 1.0).
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkFmaF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)],
            0,
            7,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][5], 0x4500 | (0x3c00 << 16));
    // v_pk_min_f16 / v_pk_max_f16 of (1.0, 2.0), (3.0, -0.5).
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMinF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][5], 0x3c00 | (0xb800 << 16));
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMaxF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][5], 0x4200 | (0x4000 << 16));
    // NaN: passes through unclamped, clamps to 0 with MODE.DX10_CLAMP (default on).
    w.v[1] = [0x7e00 | (0x3c00 << 16); 64];
    w.v[2] = [0x3c00 | (0x7e00 << 16); 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert!(is_nan16(w.v[4][0] as u16) && is_nan16((w.v[4][0] >> 16) as u16));
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(4),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            true,
        ),
    )
    .unwrap();
    assert_eq!(w.v[4][0], 0);
}

fn pk_int_ref(op: Vop3pOp, s: [u32; 2], op_sel: u8, op_sel_hi: u8, clamp: bool) -> u32 {
    let mut out = 0u32;
    for half in 0..2u32 {
        let sel = if half == 0 { op_sel } else { op_sel_hi };
        let x = (s[0] >> (16 * u32::from(sel & 1))) as u16;
        let y = (s[1] >> (16 * u32::from((sel >> 1) & 1))) as u16;
        let r: u16 = match op {
            Vop3pOp::PkAddU16 => {
                if clamp {
                    x.saturating_add(y)
                } else {
                    x.wrapping_add(y)
                }
            }
            Vop3pOp::PkSubU16 => {
                if clamp {
                    x.saturating_sub(y)
                } else {
                    x.wrapping_sub(y)
                }
            }
            Vop3pOp::PkMulLoU16 => x.wrapping_mul(y),
            Vop3pOp::PkLshlrevB16 => y << (x & 15),
            Vop3pOp::PkLshrrevB16 => y >> (x & 15),
            Vop3pOp::PkMaxI16 => (x as i16).max(y as i16) as u16,
            Vop3pOp::PkMinI16 => (x as i16).min(y as i16) as u16,
            _ => unreachable!(),
        };
        out |= u32::from(r) << (16 * half);
    }
    out
}

#[test]
fn pk_integer_ops_match_u16_i16_formulas_for_every_op_sel_combination() {
    let ops = [
        Vop3pOp::PkAddU16,
        Vop3pOp::PkSubU16,
        Vop3pOp::PkMulLoU16,
        Vop3pOp::PkLshlrevB16,
        Vop3pOp::PkLshrrevB16,
        Vop3pOp::PkMaxI16,
        Vop3pOp::PkMinI16,
    ];
    let mut rng = Rng::new(0x1A);
    for op in ops {
        for op_sel in 0..4u8 {
            for op_sel_hi in 0..4u8 {
                for clamp in [false, true] {
                    if clamp && !matches!(op, Vop3pOp::PkAddU16 | Vop3pOp::PkSubU16) {
                        continue;
                    }
                    let mut w = new_wave();
                    let mut s = [[0u32; 64]; 2];
                    for i in 0..2 {
                        for l in 0..64 {
                            s[i][l] = match rng.below(4) {
                                0 => [
                                    0x0000_0000,
                                    0xffff_ffff,
                                    0x8000_8000,
                                    0x7fff_7fff,
                                    0x0001_ffff,
                                ][rng.below(5) as usize],
                                _ => rng.next() as u32,
                            };
                        }
                        w.v[1 + i] = s[i];
                    }
                    vexec(
                        &mut w,
                        vop3p(
                            op,
                            Src::Vgpr(3),
                            &[Src::Vgpr(1), Src::Vgpr(2)],
                            op_sel,
                            op_sel_hi,
                            0,
                            0,
                            clamp,
                        ),
                    )
                    .unwrap();
                    for l in 0..64 {
                        assert_eq!(w.v[3][l], pk_int_ref(op, [s[0][l], s[1][l]], op_sel, op_sel_hi, clamp),
                        "{} op_sel={op_sel} op_sel_hi={op_sel_hi} clamp={clamp} srcs {:#010x} {:#010x}", op.name(), s[0][l], s[1][l]);
                    }
                }
            }
        }
    }
    // Literals: add 0xfff0+0x0020 wraps to 0x0010 / saturates to 0xffff; shifts use the low nibble.
    let mut w = new_wave();
    w.v[1] = [0xfff0_fff0; 64];
    w.v[2] = [0x0020_0020; 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddU16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[3][0], 0x0010_0010);
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddU16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            true,
        ),
    )
    .unwrap();
    assert_eq!(w.v[3][0], 0xffff_ffff);
    w.v[1] = [0x0013_0004; 64]; // shift counts 4 and 3 (19 & 15)
    w.v[2] = [0x8001_8001; 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkLshlrevB16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[3][0], 0x0008_0010);
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkLshrrevB16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(w.v[3][0], 0x1000_0800);
}

// ---- fma_mix

fn mix_case(
    rng: &mut Rng,
    op: Vop3pOp,
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) {
    let mut w = new_wave();
    let mut src = [[0u32; 64]; 3];
    for i in 0..3 {
        let as_f32 = (op_sel_hi >> i) & 1 == 0;
        for l in 0..64 {
            src[i][l] = if as_f32 {
                gen_f32(rng, 120, 134)
            } else {
                u32::from(gen_h(rng, 10, 20)) | (u32::from(gen_h(rng, 10, 20)) << 16)
            };
        }
        w.v[1 + i] = src[i];
    }
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(
            op,
            Src::Vgpr(4),
            &srcs,
            op_sel,
            op_sel_hi,
            neg_lo,
            neg_hi,
            clamp,
        ),
    )
    .unwrap();
    for l in 0..64 {
        // CDNA3 ISA §12.10 V_MAD_MIX_F32 p.259: select = {OPSEL_HI, OPSEL}: 0/1 = f32, 2 = low f16, 3 = high f16;
        // NEG_HI is an absolute value, NEG_LO a negation.
        let mut x = [Fx::ZERO; 3];
        for i in 0..3 {
            let sel = ((op_sel_hi >> i) & 1) * 2 + ((op_sel >> i) & 1);
            let mut v = match sel {
                0 | 1 => fx_from_bits(u64::from(src[i][l]), F32F),
                2 => fx_from_bits(u64::from(src[i][l] & 0xffff), F16F),
                _ => fx_from_bits(u64::from(src[i][l] >> 16), F16F),
            };
            if (neg_hi >> i) & 1 != 0 {
                v.m = v.m.abs();
            }
            if (neg_lo >> i) & 1 != 0 {
                v = v.neg();
            }
            x[i] = v;
        }
        let mut r32 = round_fx(x[0].mul(x[1]).add(x[2]), F32F, Rm::Rne) as u32;
        if clamp {
            r32 = clamp32(r32);
        }
        let h = round_fx(fx_from_bits(u64::from(r32), F32F), F16F, Rm::Rne) as u32;
        let want = match op {
            Vop3pOp::FmaMixF32 => r32,
            Vop3pOp::FmaMixloF16 => (POISON & 0xffff_0000) | h,
            _ => (POISON & 0x0000_ffff) | (h << 16),
        };
        assert_eq!(w.v[4][l], want, "{} op_sel={op_sel:#05b} op_sel_hi={op_sel_hi:#05b} neg_lo={neg_lo:#05b} neg_hi={neg_hi:#05b} clamp={clamp} lane {l}", op.name());
    }
}

const MIX: [Vop3pOp; 3] = [
    Vop3pOp::FmaMixF32,
    Vop3pOp::FmaMixloF16,
    Vop3pOp::FmaMixhiF16,
];

#[test]
fn fma_mix_selects_f32_or_f16_halves_and_preserves_the_other_half() {
    let mut rng = Rng::new(0x31);
    for op in MIX {
        for op_sel in 0..8u8 {
            for op_sel_hi in 0..8u8 {
                mix_case(&mut rng, op, op_sel, op_sel_hi, 0, 0, false);
            }
        }
    }
    // Literal: mixlo keeps dst[31:16] = 0xdead, mixhi keeps dst[15:0] = 0xbeef.
    let mut w = new_wave();
    w.v[1] = [0x3c00; 64]; // 1.0 as f16 (low)
    w.v[2] = [0x4000; 64]; // 2.0
    w.v[3] = [0x3800; 64]; // 0.5
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixloF16, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(
        w.v[4][9],
        0xdead_0000 | 0x4100,
        "1*2+0.5 = 2.5 (0x4100) in the low half"
    );
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixhiF16, Src::Vgpr(5), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[5][9], 0x4100_beef);
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixF32, Src::Vgpr(6), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[6][9], 2.5f32.to_bits());
}

#[test]
fn fma_mix_neg_is_negate_and_neg_hi_is_abs() {
    let mut rng = Rng::new(0x32);
    for op in MIX {
        for neg_lo in 0..8u8 {
            for neg_hi in 0..8u8 {
                let (os, osh) = (rng.below(8) as u8, rng.below(8) as u8);
                mix_case(&mut rng, op, os, osh, neg_lo, neg_hi, false);
            }
        }
        for _ in 0..16 {
            let (os, osh, nl, nh) = (
                rng.below(8) as u8,
                rng.below(8) as u8,
                rng.below(8) as u8,
                rng.below(8) as u8,
            );
            mix_case(&mut rng, op, os, osh, nl, nh, true);
        }
    }
    // Literal: src0 = -3.0 (f32), abs -> 3.0, negate -> -3.0; fma(-3.0, 2.0, 1.0) = -5.0, with abs only 7.0.
    let mut w = new_wave();
    w.v[1] = [(-3.0f32).to_bits(); 64];
    w.v[2] = [2.0f32.to_bits(); 64];
    w.v[3] = [1.0f32.to_bits(); 64];
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixF32, Src::Vgpr(4), &srcs, 0, 0, 0, 0, false),
    )
    .unwrap();
    assert_eq!(f32::from_bits(w.v[4][0]), -5.0);
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixF32, Src::Vgpr(4), &srcs, 0, 0, 0, 1, false),
    )
    .unwrap();
    assert_eq!(f32::from_bits(w.v[4][0]), 7.0);
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixF32, Src::Vgpr(4), &srcs, 0, 0, 1, 1, false),
    )
    .unwrap();
    assert_eq!(
        f32::from_bits(w.v[4][0]),
        -5.0,
        "abs first (3.0), then negate (-3.0)"
    );
    vexec(
        &mut w,
        vop3p(Vop3pOp::FmaMixF32, Src::Vgpr(4), &srcs, 0, 0, 0, 0, true),
    )
    .unwrap();
    assert_eq!(f32::from_bits(w.v[4][0]), 0.0, "clamp(-5.0) = 0");
}

// ---- packed F32

fn pk32_case(
    rng: &mut Rng,
    op: Vop3pOp,
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) {
    let mut w = new_wave();
    let n_src = if op == Vop3pOp::PkFmaF32 { 3 } else { 2 };
    let mut s = [[[0u32; 64]; 2]; 3];
    for i in 0..n_src {
        for d in 0..2 {
            for l in 0..64 {
                s[i][d][l] = gen_f32(rng, 120, 134);
                w.v[2 + 2 * i + d][l] = s[i][d][l];
            }
        }
    }
    let srcs = [Src::Vgpr(2), Src::Vgpr(4), Src::Vgpr(6)];
    vexec(
        &mut w,
        vop3p(
            op,
            Src::Vgpr(8),
            &srcs[..n_src],
            op_sel,
            op_sel_hi,
            neg_lo,
            neg_hi,
            clamp,
        ),
    )
    .unwrap();
    for l in 0..64 {
        for half in 0..2usize {
            // §6.7 p.39: OPSEL / OPSEL_HI choose the first or second DWORD of each 64-bit source.
            let sel = if half == 0 { op_sel } else { op_sel_hi };
            let neg = if half == 0 { neg_lo } else { neg_hi };
            let mut x = [Fx::ZERO; 3];
            for i in 0..n_src {
                let mut v = fx_from_bits(u64::from(s[i][usize::from((sel >> i) & 1)][l]), F32F);
                if (neg >> i) & 1 != 0 {
                    v = v.neg();
                }
                x[i] = v;
            }
            let r = match op {
                Vop3pOp::PkFmaF32 => x[0].mul(x[1]).add(x[2]),
                Vop3pOp::PkMulF32 => x[0].mul(x[1]),
                _ => x[0].add(x[1]),
            };
            // Clamp is encoded by LLVM but §6.7 p.39 gives packed-32 no output modifiers: ignored.
            let _ = clamp;
            assert_eq!(w.v[8 + half][l], round_fx(r, F32F, Rm::Rne) as u32,
                "{} op_sel={op_sel:#05b} op_sel_hi={op_sel_hi:#05b} neg_lo={neg_lo:#05b} neg_hi={neg_hi:#05b} half {half} lane {l}", op.name());
        }
    }
}

const PK_F32: [Vop3pOp; 3] = [Vop3pOp::PkFmaF32, Vop3pOp::PkMulF32, Vop3pOp::PkAddF32];

#[test]
fn pk_f32_pairs_select_dwords_negate_and_ignore_clamp() {
    let mut rng = Rng::new(0x32F);
    for op in PK_F32 {
        for op_sel in 0..8u8 {
            for op_sel_hi in 0..8u8 {
                pk32_case(&mut rng, op, op_sel, op_sel_hi, 0, 0, false);
            }
        }
        for neg_lo in 0..8u8 {
            for neg_hi in 0..8u8 {
                let (os, osh) = (rng.below(8) as u8, rng.below(8) as u8);
                pk32_case(&mut rng, op, os, osh, neg_lo, neg_hi, false);
            }
        }
        for _ in 0..8 {
            pk32_case(&mut rng, op, 0, 7, 0, 0, true);
        }
    }
    // Literal with clamp set (ignored): (2.0, 3.0) * (4.0, 5.0) + (1.0, 1.0) = (9.0, 16.0), not clamped to 1.0.
    let mut w = new_wave();
    for (reg, v) in [
        (2, 2.0f32),
        (3, 3.0),
        (4, 4.0),
        (5, 5.0),
        (6, 1.0),
        (7, 1.0),
    ] {
        w.v[reg] = [v.to_bits(); 64];
    }
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkFmaF32,
            Src::Vgpr(8),
            &[Src::Vgpr(2), Src::Vgpr(4), Src::Vgpr(6)],
            0,
            7,
            0,
            0,
            true,
        ),
    )
    .unwrap();
    assert_eq!(
        (f32::from_bits(w.v[8][0]), f32::from_bits(w.v[9][0])),
        (9.0, 16.0)
    );
    // v_pk_mul_f32 and v_pk_add_f32 literals.
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMulF32,
            Src::Vgpr(8),
            &[Src::Vgpr(2), Src::Vgpr(4)],
            0,
            7,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(
        (f32::from_bits(w.v[8][0]), f32::from_bits(w.v[9][0])),
        (8.0, 15.0)
    );
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF32,
            Src::Vgpr(8),
            &[Src::Vgpr(2), Src::Vgpr(4)],
            0,
            7,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!(
        (f32::from_bits(w.v[8][0]), f32::from_bits(w.v[9][0])),
        (6.0, 8.0)
    );
}

#[test]
fn pk_f32_rejects_odd_register_pairs() {
    let mut w = new_wave();
    let r = vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF32,
            Src::Vgpr(8),
            &[Src::Vgpr(3), Src::Vgpr(4)],
            0,
            7,
            0,
            0,
            false,
        ),
    );
    assert!(matches!(r, Err(SimError::Unsupported(_))), "{r:?}");
}

// ---- v_pk_mov_b32

#[test]
fn pk_mov_b32_is_a_dword_gather_driven_by_op_sel_only() {
    // CDNA3 ISA §12.10 V_PK_MOV_B32 p.263: D[31:0] = S0.dword[OPSEL[0]], D[63:32] = S1.dword[OPSEL[1]];
    // OP_SEL_HI is ignored.
    for op_sel in 0..4u8 {
        for op_sel_hi in 0..8u8 {
            let mut w = new_wave();
            for l in 0..64 {
                w.v[2][l] = 0x1000 + l as u32;
                w.v[3][l] = 0x2000 + l as u32;
                w.v[4][l] = 0x3000 + l as u32;
                w.v[5][l] = 0x4000 + l as u32;
            }
            vexec(
                &mut w,
                vop3p(
                    Vop3pOp::PkMovB32,
                    Src::Vgpr(0),
                    &[Src::Vgpr(2), Src::Vgpr(4)],
                    op_sel,
                    op_sel_hi,
                    0,
                    0,
                    false,
                ),
            )
            .unwrap();
            for l in 0..64 {
                let lo = if op_sel & 1 == 0 { 0x1000 } else { 0x2000 } + l as u32;
                let hi = if op_sel & 2 == 0 { 0x3000 } else { 0x4000 } + l as u32;
                assert_eq!(
                    (w.v[0][l], w.v[1][l]),
                    (lo, hi),
                    "op_sel={op_sel} op_sel_hi={op_sel_hi} lane {l}"
                );
            }
        }
    }
    // ISA example: `v_pk_mov_b32 v0, v2, v4 op_sel:[0,1]` -> v0 = v2, v1 = v5.
    let mut w = new_wave();
    w.v[2] = [0xa2; 64];
    w.v[3] = [0xa3; 64];
    w.v[4] = [0xb4; 64];
    w.v[5] = [0xb5; 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMovB32,
            Src::Vgpr(0),
            &[Src::Vgpr(2), Src::Vgpr(4)],
            0b10,
            0,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!((w.v[0][0], w.v[1][0]), (0xa2, 0xb5));
    // `v_pk_mov_b32 v0, s6, s6 op_sel:[0,1]`: 64-bit move from s[6:7].
    let mut w = new_wave();
    w.s[6] = 0x6666_0006;
    w.s[7] = 0x7777_0007;
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMovB32,
            Src::Vgpr(0),
            &[Src::Sgpr(6), Src::Sgpr(6)],
            0b10,
            0,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert_eq!((w.v[0][33], w.v[1][33]), (0x6666_0006, 0x7777_0007));
}

// ---- dots

fn h2f(bits: u16) -> f32 {
    let x = fx_from_bits(u64::from(bits), F16F);
    (x.m as f64 * p2(x.e)) as f32
}

/// Few-significant-bit halves and a small-integer/1024 accumulator keep every
/// partial sum exact in f32, so sequential, fused and pairwise additions agree.
fn dot2f_case(rng: &mut Rng, neg_lo: u8, neg_hi: u8) {
    let mut w = new_wave();
    let mut s = [[0u32; 64]; 3];
    for l in 0..64 {
        for i in 0..2 {
            let mut half = || {
                (((rng.sign() as u16) << 15)
                    | ((13 + rng.below(5) as u16) << 10)
                    | ((rng.below(32) as u16) << 5)) as u32
            };
            s[i][l] = half() | (half() << 16);
        }
        s[2][l] = ((rng.range(-(1 << 14), 1 << 14) as f32) / 1024.0).to_bits();
    }
    for i in 0..3 {
        w.v[1 + i] = s[i];
    }
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::Dot2F32F16,
            Src::Vgpr(4),
            &srcs,
            0,
            7,
            neg_lo,
            neg_hi,
            false,
        ),
    )
    .unwrap();
    for l in 0..64 {
        let mut t = f32::from_bits(s[2][l]);
        if neg_lo & 4 != 0 {
            t = -t;
        }
        for half in 0..2 {
            let neg = if half == 0 { neg_lo } else { neg_hi };
            let mut a = h2f((s[0][l] >> (16 * half)) as u16);
            let mut b = h2f((s[1][l] >> (16 * half)) as u16);
            if neg & 1 != 0 {
                a = -a;
            }
            if neg & 2 != 0 {
                b = -b;
            }
            t += a * b;
        }
        assert_eq!(
            w.v[4][l],
            t.to_bits(),
            "dot2_f32_f16 neg_lo={neg_lo:#05b} neg_hi={neg_hi:#05b} lane {l}"
        );
    }
}

#[test]
fn dot2_f32_f16_matches_the_pseudocode_with_neg_fields() {
    let mut rng = Rng::new(0xD2);
    for neg_lo in 0..8u8 {
        for neg_hi in 0..4u8 {
            dot2f_case(&mut rng, neg_lo, neg_hi);
        }
    }
    // Literal (§12.10 V_DOT2_F32_F16 p.260-261): (1.5, -2.0) . (4.0, 0.25) + 10.0 = 10 + 6 - 0.5 = 15.5.
    let mut w = new_wave();
    w.v[1] = [0x3e00 | (0xc000 << 16); 64];
    w.v[2] = [0x4400 | (0x3400 << 16); 64];
    w.v[3] = [10.0f32.to_bits(); 64];
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot2F32F16, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(f32::from_bits(w.v[4][0]), 15.5);
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::Dot2F32F16,
            Src::Vgpr(4),
            &srcs,
            0,
            7,
            0b001,
            0b010,
            false,
        ),
    )
    .unwrap();
    // lo: -1.5*4 = -6; hi: -2.0 * -0.25 = +0.5 -> 10 - 6 + 0.5 = 4.5
    assert_eq!(f32::from_bits(w.v[4][0]), 4.5);
}

#[test]
fn dot2_f32_f16_rejects_op_sel_and_ambiguous_neg_hi() {
    // §6.2 p.32: DOT instructions must not use OPSEL; the default is OP_SEL=0, OP_SEL_HI=7.
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    for (os, osh, nh) in [
        (1u8, 7u8, 0u8),
        (2, 7, 0),
        (4, 7, 0),
        (0, 3, 0),
        (0, 0, 0),
        (0, 6, 0),
        (0, 7, 4),
    ] {
        let mut w = new_wave();
        for r in 1..4 {
            w.v[r] = [0x3c00_3c00; 64];
        }
        let r = vexec(
            &mut w,
            vop3p(
                Vop3pOp::Dot2F32F16,
                Src::Vgpr(4),
                &srcs,
                os,
                osh,
                0,
                nh,
                false,
            ),
        );
        assert!(
            matches!(r, Err(SimError::Unsupported(_))),
            "op_sel={os} op_sel_hi={osh} neg_hi={nh}: {r:?}"
        );
        assert!(w.v[4] == [POISON; 64]);
    }
}

fn sext(x: u32, bits: u32) -> i64 {
    i64::from(((x << (32 - bits)) as i32) >> (32 - bits))
}
/// §12.10 V_DOT{2,4,8}_{I,U}32_*: `tmp = S2; tmp += elem(S0,e)*elem(S1,e)` over every element,
/// wrapping at 32 bits.
fn dot_int_ref(bits: u32, signed: bool, s0: u32, s1: u32, s2: u32) -> u32 {
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut acc = s2;
    for e in 0..32 / bits {
        let (x, y) = ((s0 >> (e * bits)) & mask, (s1 >> (e * bits)) & mask);
        let p = if signed {
            sext(x, bits) * sext(y, bits)
        } else {
            i64::from(x) * i64::from(y)
        };
        acc = acc.wrapping_add(p as u32);
    }
    acc
}

const DOT_INT: [(Vop3pOp, u32, bool); 6] = [
    (Vop3pOp::Dot2I32I16, 16, true),
    (Vop3pOp::Dot2U32U16, 16, false),
    (Vop3pOp::Dot4I32I8, 8, true),
    (Vop3pOp::Dot4U32U8, 8, false),
    (Vop3pOp::Dot8I32I4, 4, true),
    (Vop3pOp::Dot8U32U4, 4, false),
];

fn gen_dot_dword(rng: &mut Rng, bits: u32) -> u32 {
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut v = 0u32;
    for e in 0..32 / bits {
        let pick = match rng.below(5) {
            0 => 1u32 << (bits - 1),       // signed minimum / unsigned 2^(b-1)
            1 => (1u32 << (bits - 1)) - 1, // signed maximum
            2 => mask,                     // -1 / unsigned maximum
            3 => 0,
            _ => rng.next() as u32 & mask,
        };
        v |= (pick & mask) << (e * bits);
    }
    v
}

#[test]
fn integer_dots_signed_and_unsigned_wrap_like_the_pseudocode() {
    let mut rng = Rng::new(0xD0);
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    for (op, bits, signed) in DOT_INT {
        let mut w = new_wave();
        let mut s = [[0u32; 64]; 3];
        for i in 0..3 {
            for l in 0..64 {
                s[i][l] = if i == 2 && rng.chance(3) {
                    [i32::MAX as u32, i32::MIN as u32, u32::MAX, 0][rng.below(4) as usize]
                } else {
                    gen_dot_dword(&mut rng, if i == 2 { 32 } else { bits })
                };
            }
            w.v[1 + i] = s[i];
        }
        vexec(&mut w, vop3p(op, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false)).unwrap();
        for l in 0..64 {
            assert_eq!(
                w.v[4][l],
                dot_int_ref(bits, signed, s[0][l], s[1][l], s[2][l]),
                "{} lane {l}: {:#010x} . {:#010x} + {:#010x}",
                op.name(),
                s[0][l],
                s[1][l],
                s[2][l]
            );
        }
    }
    // Literals. dot4_i32_i8: (-128,127,-1,1).(-128,127,-1,1) + 5 = 16384+16129+1+1+5.
    let mut w = new_wave();
    w.v[1] = [0x01ff_7f80; 64];
    w.v[2] = [0x01ff_7f80; 64];
    w.v[3] = [5; 64];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot4I32I8, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[4][0], 16384 + 16129 + 1 + 1 + 5);
    // dot4_u32_u8: (128,127,255,1).(128,127,255,1) + 5.
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot4U32U8, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[4][0], 16384 + 16129 + 65025 + 1 + 5);
    // dot2_i32_i16: (-32768, 32767)^2 + S2 = 2^30 + 32767^2 + 7.
    w.v[1] = [0x7fff_8000; 64];
    w.v[2] = [0x7fff_8000; 64];
    w.v[3] = [7; 64];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot2I32I16, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[4][0], (1u32 << 30) + 32767 * 32767 + 7);
    // dot2_u32_u16 wraps: (0xffff, 0xffff)^2 = 2 * 4294836225 mod 2^32.
    w.v[1] = [0xffff_ffff; 64];
    w.v[2] = [0xffff_ffff; 64];
    w.v[3] = [3; 64];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot2U32U16, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(
        w.v[4][0],
        3u32.wrapping_add(2u32.wrapping_mul(0xffff * 0xffff))
    );
    // dot8_i32_i4: eight nibbles -8 (0x8) squared = 8 * 64 = 512; dot8_u32_u4: eight nibbles 15 -> 8 * 225.
    w.v[1] = [0x8888_8888; 64];
    w.v[2] = [0x8888_8888; 64];
    w.v[3] = [0; 64];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot8I32I4, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[4][0], 512);
    w.v[1] = [0xffff_ffff; 64];
    w.v[2] = [0xffff_ffff; 64];
    vexec(
        &mut w,
        vop3p(Vop3pOp::Dot8U32U4, Src::Vgpr(4), &srcs, 0, 7, 0, 0, false),
    )
    .unwrap();
    assert_eq!(w.v[4][0], 8 * 225);
}

#[test]
fn integer_dots_reject_every_non_default_source_modifier() {
    let srcs = [Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)];
    for (op, _, _) in DOT_INT {
        for (os, osh, nl, nh) in [
            (1u8, 7u8, 0u8, 0u8),
            (0, 3, 0, 0),
            (0, 7, 1, 0),
            (0, 7, 0, 1),
            (0, 0, 0, 0),
            (4, 7, 0, 0),
        ] {
            let mut w = new_wave();
            for r in 1..4 {
                w.v[r] = [0x0101_0101; 64];
            }
            let r = vexec(
                &mut w,
                vop3p(op, Src::Vgpr(4), &srcs, os, osh, nl, nh, false),
            );
            assert!(
                matches!(r, Err(SimError::Unsupported(_))),
                "{} op_sel={os} op_sel_hi={osh} neg_lo={nl} neg_hi={nh}: {r:?}",
                op.name()
            );
            assert!(
                w.v[4] == [POISON; 64],
                "{}: rejected dot wrote its destination",
                op.name()
            );
        }
    }
}

// ---- modes and EXEC for packed math

#[test]
fn non_rne_packed_rounding_is_unsupported_or_correctly_rounded() {
    // 1.0 + 2^-12 (f16): RNE gives 0x3c00; round toward +inf gives 0x3c01.
    let mut w = new_wave();
    w.mode = Mode {
        round16_64: 1,
        ..Mode::default()
    };
    w.v[1] = [0x3c00_3c00; 64];
    w.v[2] = [0x0c00_0c00; 64];
    let r = vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    );
    match r {
        Err(SimError::Unsupported(_)) => {}
        Ok(()) => assert_eq!(
            w.v[3][0], 0x3c01_3c01,
            "round16_64=RTP silently produced RNE"
        ),
        Err(e) => panic!("{e}"),
    }
    // 1.0f + 2^-24 (f32): RNE 1.0, RTP 0x3f800001.
    let mut w = new_wave();
    w.mode = Mode {
        round32: 1,
        ..Mode::default()
    };
    for r in [2, 4] {
        w.v[r] = [1.0f32.to_bits(); 64];
    }
    w.v[3] = [0; 64];
    w.v[5] = [0; 64];
    w.v[6] = [2.0f32.powi(-24).to_bits(); 64];
    w.v[7] = [0; 64];
    let r = vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF32,
            Src::Vgpr(8),
            &[Src::Vgpr(2), Src::Vgpr(6)],
            0,
            7,
            0,
            0,
            false,
        ),
    );
    match r {
        Err(SimError::Unsupported(_)) => {}
        Ok(()) => assert_eq!(w.v[8][0], 0x3f80_0001, "round32=RTP silently produced RNE"),
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn packed_math_writes_only_active_lanes() {
    let mut w = new_wave();
    w.exec = 0xaaaa_aaaa_aaaa_aaaa;
    w.v[1] = [0x3c00_3c00; 64];
    w.v[2] = [0x4000_4000; 64];
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkAddF16,
            Src::Vgpr(3),
            &[Src::Vgpr(1), Src::Vgpr(2)],
            0,
            3,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    for l in 0..64 {
        assert_eq!(
            w.v[3][l],
            if l % 2 == 1 { 0x4200_4200 } else { POISON },
            "lane {l}"
        );
    }
    w.exec = 0;
    vexec(
        &mut w,
        vop3p(
            Vop3pOp::PkMovB32,
            Src::Vgpr(4),
            &[Src::Vgpr(0), Src::Vgpr(0)],
            0,
            0,
            0,
            0,
            false,
        ),
    )
    .unwrap();
    assert!(w.v[4] == [POISON; 64] && w.v[5] == [POISON; 64]);
}

#[test]
fn every_vop3p_opcode_is_covered_by_a_test_above() {
    // Exhaustive match: adding a Vop3pOp variant fails to compile here until a test covers it.
    fn covered_by(op: Vop3pOp) -> &'static str {
        use Vop3pOp::*;
        match op {
            PkFmaF16 | PkAddF16 | PkMulF16 | PkMinF16 | PkMaxF16 => "pk_f16_*",
            PkAddU16 | PkSubU16 | PkMulLoU16 | PkLshlrevB16 | PkLshrrevB16 | PkMaxI16
            | PkMinI16 => "pk_integer_ops_*",
            FmaMixF32 | FmaMixloF16 | FmaMixhiF16 => "fma_mix_*",
            Dot2F32F16 => "dot2_f32_f16_*",
            Dot2I32I16 | Dot2U32U16 | Dot4I32I8 | Dot4U32U8 | Dot8I32I4 | Dot8U32U4 => {
                "integer_dots_*"
            }
            PkFmaF32 | PkMulF32 | PkAddF32 => "pk_f32_*",
            PkMovB32 => "pk_mov_b32_*",
        }
    }
    assert_eq!(Vop3pOp::ALL.len(), 26);
    for op in Vop3pOp::ALL {
        assert!(!covered_by(*op).is_empty());
    }
    assert_eq!(mfma_ops().len(), 17);
}

// ---------------------------------------------------------------------------
// Tests: `accumulate_typed` crafted probes and the independent Cdna3Fused model

fn pw(e: i32) -> f32 {
    2.0f32.powi(e)
}

/// Hand-derived from the documented candidate (TR/GTR-FDPA): C aligned to the product maximum
/// exponent with round-down; FP8 discards a C more than 25 exponents under the products.
#[test]
fn accumulate_typed_crafted_cdna3_fused_probes() {
    let f = AccumModel::Cdna3Fused;
    // BF16: C = 2^10 vs cancelling products +-2^35: the product maximum exponent (35) still
    // aligns C to 2^11 and rounds it down to 0 (exact arithmetic would keep 2^10).
    assert_eq!(
        accumulate_typed(f, Bf16, Bf16, pw(10), &[pw(35), -pw(35)], &[1.0, 1.0]),
        0.0
    );
    assert_eq!(
        accumulate_typed(
            AccumModel::Exact,
            Bf16,
            Bf16,
            pw(10),
            &[pw(35), -pw(35)],
            &[1.0, 1.0]
        ),
        pw(10)
    );
    // FP8: C = 2^-20, products [1, 32, 0, -32]: the odd partial sum cancels to 0 but its max
    // exponent (5) still pushes C (exponent -20, gap 25 is not > 25 but the floor quantum is 2^-19) to 0 -> 1.0.
    assert_eq!(
        accumulate_typed(f, Fp8, Fp8, pw(-20), &[1.0, 32.0, 0.0, -32.0], &[1.0; 4]),
        1.0
    );
    // F16/BF16 C rounds DOWN: product -1, C = -(2^-24 + 2^-25) -> C floors to -2^-23 -> -(1 + 2^-23);
    // product +1, C = +(2^-24 + 2^-25) floors to 2^-24 -> tie -> 1.0.
    let c = pw(-24) + pw(-25);
    for t in [F16, Bf16] {
        assert_eq!(
            accumulate_typed(f, t, t, -c, &[-1.0], &[1.0]),
            -(1.0 + pw(-23)),
            "{t:?} negative C bias"
        );
        assert_eq!(
            accumulate_typed(f, t, t, c, &[1.0], &[1.0]),
            1.0,
            "{t:?} positive C"
        );
    }
    // Raw-exponent dependence: product 2.25 from 1.5 * 1.5 (exponent sum 0, C kept at 2^-24 quantum:
    // 2.25 + 3*2^-24 rounds up to 2.25 + 2^-22) vs 2.25 * 1 (exponent sum 1, C floors to 2^-23, a tie -> 2.25).
    let c = 3.0 * pw(-24);
    assert_eq!(
        accumulate_typed(f, F16, F16, c, &[1.5], &[1.5]),
        2.25 + pw(-22)
    );
    assert_eq!(accumulate_typed(f, F16, F16, c, &[2.25], &[1.0]), 2.25);
    // BF16 product >= 2^128 is +inf; opposite overflowing products give NaN (inf - inf).
    assert_eq!(
        accumulate_typed(f, Bf16, Bf16, 0.0, &[pw(127)], &[2.0]),
        f32::INFINITY
    );
    assert_eq!(
        accumulate_typed(f, Bf16, Bf16, 0.0, &[-pw(127)], &[2.0]),
        f32::NEG_INFINITY
    );
    assert!(accumulate_typed(f, Bf16, Bf16, 0.0, &[pw(127), -pw(127)], &[2.0, 2.0]).is_nan());
}

/// `accumulate_typed` (all type mixes, K = one and two groups) against this file's integer-quantum
/// implementation of the same candidate: the exec path is covered by the matrix tests above.
#[test]
fn cdna3_fused_accumulate_matches_the_independent_quantization_reference() {
    let mut rng = Rng::new(0xF05E);
    let mixes: [(MfmaType, MfmaType, usize); 8] = [
        (F16, F16, 8),
        (F16, F16, 16),
        (Bf16, Bf16, 8),
        (Bf16, Bf16, 16),
        (Fp8, Fp8, 16),
        (Fp8, Bf8, 32),
        (Bf8, Fp8, 32),
        (Bf8, Bf8, 16),
    ];
    for (at, bt, k) in mixes {
        for round in 0..300 {
            let g = if round % 5 == 0 {
                Gen::SmallInt
            } else {
                Gen::Wide
            };
            let a: Vec<u64> = (0..k).map(|_| gen_elem(&mut rng, at, g)).collect();
            let b: Vec<u64> = (0..k).map(|_| gen_elem(&mut rng, bt, g)).collect();
            let c = gen_c(&mut rng, F32, g);
            let af: Vec<f32> = a.iter().map(|x| dec64(at, *x) as f32).collect();
            let bf: Vec<f32> = b.iter().map(|x| dec64(bt, *x) as f32).collect();
            let ax: Vec<Fx> = a.iter().map(|x| dec(at, *x, false)).collect();
            let bx: Vec<Fx> = b.iter().map(|x| dec(bt, *x, false)).collect();
            let want = fused_reduce(at, bt, dec(F32, c, false), &ax, &bx);
            let got = accumulate_typed(
                AccumModel::Cdna3Fused,
                at,
                bt,
                f32::from_bits(c as u32),
                &af,
                &bf,
            );
            assert_eq!(
                u64::from(got.to_bits()),
                want,
                "{at:?}x{bt:?} k={k} round {round}: A={a:x?} B={b:x?} C={c:#x}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: more packed-math rules

#[test]
fn integer_pk_ops_reject_neg_fields() {
    let ops = [
        Vop3pOp::PkAddU16,
        Vop3pOp::PkSubU16,
        Vop3pOp::PkMulLoU16,
        Vop3pOp::PkLshlrevB16,
        Vop3pOp::PkLshrrevB16,
        Vop3pOp::PkMaxI16,
        Vop3pOp::PkMinI16,
    ];
    for op in ops {
        for (nl, nh) in [(1u8, 0u8), (0, 1), (3, 3)] {
            let mut w = new_wave();
            w.v[1] = [0x0001_0001; 64];
            w.v[2] = [0x0002_0002; 64];
            let r = vexec(
                &mut w,
                vop3p(
                    op,
                    Src::Vgpr(3),
                    &[Src::Vgpr(1), Src::Vgpr(2)],
                    0,
                    3,
                    nl,
                    nh,
                    false,
                ),
            );
            assert!(
                matches!(r, Err(SimError::Unsupported(_))),
                "{} neg_lo={nl} neg_hi={nh}: {r:?}",
                op.name()
            );
            assert!(w.v[3] == [POISON; 64]);
        }
    }
}

#[test]
fn clamp_of_nan_follows_mode_dx10_clamp() {
    // §3.5 Table 4 p.10: DX10_CLAMP set -> NaN clamps to 0; clear -> NaN passes through.
    for op in [Vop3pOp::PkAddF16, Vop3pOp::PkMulF16] {
        for dx10 in [true, false] {
            let mut w = new_wave();
            w.mode.dx10_clamp = dx10;
            w.v[1] = [0x7e00 | (0x7e00 << 16); 64];
            w.v[2] = [0x3c00 | (0x3c00 << 16); 64];
            vexec(
                &mut w,
                vop3p(
                    op,
                    Src::Vgpr(3),
                    &[Src::Vgpr(1), Src::Vgpr(2)],
                    0,
                    3,
                    0,
                    0,
                    true,
                ),
            )
            .unwrap();
            if dx10 {
                assert_eq!(w.v[3][0], 0, "{} DX10_CLAMP=1", op.name());
            } else {
                assert!(
                    is_nan16(w.v[3][0] as u16) && is_nan16((w.v[3][0] >> 16) as u16),
                    "{} DX10_CLAMP=0: {:#x}",
                    op.name(),
                    w.v[3][0]
                );
            }
        }
    }
    // fma_mix: NaN f32 source with clamp.
    for dx10 in [true, false] {
        let mut w = new_wave();
        w.mode.dx10_clamp = dx10;
        w.v[1] = [0x7fc0_0000; 64];
        w.v[2] = [1.0f32.to_bits(); 64];
        w.v[3] = [1.0f32.to_bits(); 64];
        vexec(
            &mut w,
            vop3p(
                Vop3pOp::FmaMixF32,
                Src::Vgpr(4),
                &[Src::Vgpr(1), Src::Vgpr(2), Src::Vgpr(3)],
                0,
                0,
                0,
                0,
                true,
            ),
        )
        .unwrap();
        if dx10 {
            assert_eq!(w.v[4][0], 0);
        } else {
            assert!(f32::from_bits(w.v[4][0]).is_nan());
        }
    }
}

/// §12.7 p.172-173 (V_MIN_F16 / V_MAX_F16 special cases) as used by v_pk_min/max_f16.
#[test]
fn pk_min_max_f16_nan_and_zero_special_cases() {
    let rep = |h: u16| u32::from(h) | (u32::from(h) << 16);
    let run = |ieee: bool, op: Vop3pOp, x: u16, y: u16| -> u32 {
        let mut w = new_wave();
        w.mode.ieee = ieee;
        w.v[1] = [rep(x); 64];
        w.v[2] = [rep(y); 64];
        vexec(
            &mut w,
            vop3p(
                op,
                Src::Vgpr(3),
                &[Src::Vgpr(1), Src::Vgpr(2)],
                0,
                3,
                0,
                0,
                false,
            ),
        )
        .unwrap();
        assert_eq!(w.v[3][0], w.v[3][63]);
        w.v[3][0]
    };
    for op in [Vop3pOp::PkMinF16, Vop3pOp::PkMaxF16] {
        for (x, y) in [(0x7c01u16, 0x3c00u16), (0x3c00, 0x7c01)] {
            // IEEE=1: a signaling NaN is quieted and propagated.
            assert_eq!(
                run(true, op, x, y),
                rep(0x7e01),
                "{} IEEE=1 sNaN {x:#x},{y:#x}",
                op.name()
            );
            // IEEE=0: an sNaN is treated like a qNaN: the other input wins.
            assert_eq!(
                run(false, op, x, y),
                rep(0x3c00),
                "{} IEEE=0 sNaN {x:#x},{y:#x}",
                op.name()
            );
        }
        for (x, y) in [(0x7e00u16, 0x3c00u16), (0x3c00, 0x7e00)] {
            assert_eq!(
                run(true, op, x, y),
                rep(0x3c00),
                "{} IEEE=1 qNaN {x:#x},{y:#x}",
                op.name()
            );
            assert_eq!(
                run(false, op, x, y),
                rep(0x3c00),
                "{} IEEE=0 qNaN {x:#x},{y:#x}",
                op.name()
            );
        }
    }
    for (x, y) in [(0x0000u16, 0x8000u16), (0x8000, 0x0000)] {
        assert_eq!(
            run(true, Vop3pOp::PkMinF16, x, y),
            rep(0x8000),
            "min(+0,-0) = -0"
        );
        assert_eq!(
            run(true, Vop3pOp::PkMaxF16, x, y),
            rep(0x0000),
            "max(+0,-0) = +0"
        );
    }
}

/// §3.5 Table 4 p.11: with FP16_OVFL set, finite F16 overflow saturates to +-max while a true Inf
/// operand stays Inf.
#[test]
fn pk_f16_fp16_ovfl_saturates_finite_overflow_but_keeps_true_inf() {
    let rep = |h: u16| u32::from(h) | (u32::from(h) << 16);
    for (ovfl, big, neg_big) in [(false, 0x7c00u16, 0xfc00u16), (true, 0x7bff, 0xfbff)] {
        let mut w = new_wave();
        w.mode.fp16_ovfl = ovfl;
        w.v[1] = [rep(0x7bff); 64];
        w.v[2] = [rep(0x7bff); 64];
        w.v[3] = [rep(0xfbff); 64];
        vexec(
            &mut w,
            vop3p(
                Vop3pOp::PkAddF16,
                Src::Vgpr(4),
                &[Src::Vgpr(1), Src::Vgpr(2)],
                0,
                3,
                0,
                0,
                false,
            ),
        )
        .unwrap();
        assert_eq!(w.v[4][0], rep(big), "FP16_OVFL={ovfl}: 65504 + 65504");
        vexec(
            &mut w,
            vop3p(
                Vop3pOp::PkAddF16,
                Src::Vgpr(4),
                &[Src::Vgpr(3), Src::Vgpr(3)],
                0,
                3,
                0,
                0,
                false,
            ),
        )
        .unwrap();
        assert_eq!(w.v[4][0], rep(neg_big), "FP16_OVFL={ovfl}: -65504 - 65504");
        // A real infinity operand stays infinite either way.
        w.v[5] = [rep(0x7c00); 64];
        w.v[6] = [rep(0x3c00); 64];
        vexec(
            &mut w,
            vop3p(
                Vop3pOp::PkAddF16,
                Src::Vgpr(4),
                &[Src::Vgpr(5), Src::Vgpr(6)],
                0,
                3,
                0,
                0,
                false,
            ),
        )
        .unwrap();
        assert_eq!(w.v[4][0], rep(0x7c00), "FP16_OVFL={ovfl}: inf + 1");
    }
}

#[test]
fn accvgpr_write_accepts_readable_scalar_aliases_and_rejects_malformed_ttmp() {
    let mut w = new_wave();
    w.m0 = 0x0a0b_0c0d;
    w.vcc = 0x1111_2222_3333_4444;
    w.exec = u64::MAX;
    w.scc = true;
    let cases: [(Src, u32); 4] = [
        (Src::M0, 0x0a0b_0c0d),
        (Src::VccLo, 0x3333_4444),
        (Src::VccHi, 0x1111_2222),
        (Src::Scc, 1),
    ];
    for (n, (src, want)) in cases.into_iter().enumerate() {
        run_inst(
            &mut w,
            &mov(MaiOp::AccvgprWriteB32, Src::Agpr(20 + n as u8), src),
            AccumModel::Exact,
        )
        .unwrap();
        assert!(w.a[20 + n] == [want; 64], "{src:?}");
    }
    // ttmp has 16 entries: Ttmp(16) is malformed and must be rejected, not index out of range.
    let r = run_inst(
        &mut w,
        &mov(MaiOp::AccvgprWriteB32, Src::Agpr(30), Src::Ttmp(16)),
        AccumModel::Exact,
    );
    assert!(r.is_err(), "Ttmp(16): {r:?}");
}
