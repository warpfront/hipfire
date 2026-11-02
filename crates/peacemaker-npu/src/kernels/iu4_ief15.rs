// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! IEF15 host side (fold-contract §3, §8 M0; design `docs/ief15-n0.md`): CPU oracle, scale encoders, host packing, the
//! array design around [`iu4_ief15_core`] and a CPU emulator of the core contract that proves the packing.
//!
//! # Arithmetic (frozen, fold-contract §3)
//! Per output `(t, r)`: `I = sum_e S[r,e] * D[t,e] * C[t,r,e]` exactly (`C` the int32 K128 partial of the unchanged QT44
//! nibbles and A4 codes, `[-7168, 8192]`), `Y = round_scaled(I, a_r + b_t)` with one RNE to binary32. `S`, `D` are
//! `u16 <= 32767`, `a`, `b` are `i16` ([`encode_weight_scales`], [`encode_act_scales`]). Nonfinite or negative scales are
//! rejected before slot publication, so they can never reach the encoders (the encoders assert it).
//!
//! # Wave and packing (`docs/ief15-n0.md` §2)
//! A wave is 256 tokens x 256 features, `waves = MW * NW <= 256` in the order `w = mw * NW + nw`. Core `(column c, core
//! row rho)` of wave `(mw, nw)` computes tokens `mw*256 + (2*(rho/2) + c%2)*64 ..+64` x features
//! `nw*256 + (c/2)*64 + (rho%2)*32 ..+32`. Per core and epoch one A half chunk and one B chunk of 2176 B arrive (int4 codes
//! `2048 B`, then D / S `64 B`, then b / a `64 B`); [`Ief15Layout::pack_in`] writes them in the V8 segment structure:
//! arg0 = 8 injecting-column A segments, arg1 = 8 column B segments (each `(wave, epoch, 2176 B)`), arg2 = the V8 int8 C
//! layout (`Geometry::pair_c_offset`, 8192 B tiles, f32 `[mb 0..8][nb 0..4]` blocks of 8 x 8).
use super::gemm_array::{self as array, ArrayDesign, Geometry, Variant};
use super::gemm_core::{Control, CoreVariant, PairRole};
use super::gemm_i8::{ArgKind, ArgSpec};
use super::iu4::{Acts, Weights};
use super::iu4_ief15_core as ief;
use crate::dma::Bd;

/// K of one epoch.
pub const EPOCH_K: usize = ief::EPOCH_K;
/// Largest epoch count.
pub const MAX_EPOCHS: usize = ief::MAX_EPOCHS;
/// Tokens / features of one wave.
pub const WAVE: usize = array::IEF15_WAVE;
/// Largest S / D mantissa.
pub const MANT_MAX: u32 = 32767;
/// Host bound on `|a|`, `|b|`: the core's 16-bit `a_r + b_t` cannot wrap (it clamps the sum to `+-300`, exact because beyond it
/// every result is already 0 / -0 / +-inf for `|I| < 2^51`).
pub const EXP_HOST_MAX: i16 = 16383;
const CHUNK: usize = ief::A_HALF_BYTES;
const COLS: usize = array::COLS;
const ROWS: usize = array::ROWS;
const _: () = assert!(
    ief::A_HALF_BYTES == ief::B_BYTES && WAVE == 256 && ief::TM * 4 == WAVE && ief::TN * 8 == WAVE
);

// ---------------------------------------------------------------------------------------------------------------------
// CPU oracle: ported bit-exactly from `tools/npu/fold-model/src/lib.rs::{grid, round_scaled}` (not a dependency of this crate).

/// Smallest power-of-two grid exponent with RNE mantissas `<= 2^bits - 1`, and the mantissas (port of
/// `tools/npu/fold-model/src/lib.rs::grid`). Exponent bits avoid log2 boundary errors; f64 shifts / rounding are exact for these
/// f32 inputs and `<= 16`-bit mantissas. All-zero input returns exponent 0.
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

/// [`grid`] with the IEF15 15-bit mantissa (`<= 32767`).
pub fn grid15(x: &[f32]) -> (Vec<i64>, i32) {
    grid(x, 15)
}

/// One exact RNE to binary32 of `x * 2^exponent`, including subnormal, carry and signed underflow (port of
/// `tools/npu/fold-model/src/lib.rs::round_scaled`). No int64 -> f64 -> f32 double rounding. Integer zero is canonically +0.
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

/// Weight sidecars from the exact f16 -> f32 epoch scales `w.scale(r, e)`: `S` `[feature][K/128]` and `a` `[feature]`, one
/// [`grid15`] per weight row over all its epochs. Scales must be finite and non-negative (the production contract rejects
/// anything else before slot publication; this function asserts it).
pub fn encode_weight_scales(w: &Weights) -> (Vec<u16>, Vec<i16>) {
    let epochs = w.k / EPOCH_K;
    let (mut s, mut a) = (
        Vec::with_capacity(w.rows * epochs),
        Vec::with_capacity(w.rows),
    );
    for r in 0..w.rows {
        let v: Vec<f32> = (0..epochs).map(|e| w.scale(r, e)).collect();
        assert!(
            v.iter().all(|x| x.is_finite() && *x >= 0.0),
            "weight row {r}: scales must be finite and non-negative"
        );
        let (m, x) = grid15(&v);
        s.extend(m.iter().map(|&m| {
            u16::try_from(m)
                .ok()
                .filter(|&m| u32::from(m) <= MANT_MAX)
                .expect("S mantissa")
        }));
        a.push(i16::try_from(x).expect("a exponent"));
    }
    (s, a)
}

/// Activation sidecars from `x.d(t, e)`: `D` epoch-major `[K/128][token]` and `b` `[token]`, one [`grid15`] per token over all
/// its epochs. Same precondition as [`encode_weight_scales`].
pub fn encode_act_scales(x: &Acts) -> (Vec<u16>, Vec<i16>) {
    let epochs = x.k / EPOCH_K;
    let (mut d, mut b) = (vec![0u16; epochs * x.tokens], Vec::with_capacity(x.tokens));
    for t in 0..x.tokens {
        let v: Vec<f32> = (0..epochs).map(|e| x.d(t, e)).collect();
        assert!(
            v.iter().all(|x| x.is_finite() && *x >= 0.0),
            "token {t}: scales must be finite and non-negative"
        );
        let (m, e) = grid15(&v);
        for (ep, &m) in m.iter().enumerate() {
            d[ep * x.tokens + t] = u16::try_from(m)
                .ok()
                .filter(|&m| u32::from(m) <= MANT_MAX)
                .expect("D mantissa");
        }
        b.push(i16::try_from(e).expect("b exponent"));
    }
    (d, b)
}

/// Operand and sidecar validation shared by packing and the reference: shapes, `S, D <= 32767`, `|a|, |b| <= 16383`.
fn check_operands(
    w: &Weights,
    x: &Acts,
    s: &[u16],
    a: &[i16],
    d: &[u16],
    b: &[i16],
) -> Result<(), String> {
    let (tokens, features, k) = (x.tokens, w.rows, x.k);
    if w.k != k {
        return Err(format!("weights K {} != activation K {k}", w.k));
    }
    if k == 0 || k % EPOCH_K != 0 {
        return Err(format!("K {k} is not a positive multiple of {EPOCH_K}"));
    }
    let e = k / EPOCH_K;
    for (name, got, want) in [
        ("S", s.len(), features * e),
        ("a", a.len(), features),
        ("D", d.len(), e * tokens),
        ("b", b.len(), tokens),
    ] {
        if got != want {
            return Err(format!("{name} has {got} entries, expected {want}"));
        }
    }
    if let Some(i) = s.iter().position(|&v| u32::from(v) > MANT_MAX) {
        return Err(format!("S[{i}] = {} exceeds {MANT_MAX}", s[i]));
    }
    if let Some(i) = d.iter().position(|&v| u32::from(v) > MANT_MAX) {
        return Err(format!("D[{i}] = {} exceeds {MANT_MAX}", d[i]));
    }
    let bad = |v: &i16| !(-EXP_HOST_MAX..=EXP_HOST_MAX).contains(v);
    if let Some(i) = a.iter().position(bad) {
        return Err(format!("a[{i}] = {} outside +-{EXP_HOST_MAX}", a[i]));
    }
    if let Some(i) = b.iter().position(bad) {
        return Err(format!("b[{i}] = {} outside +-{EXP_HOST_MAX}", b[i]));
    }
    Ok(())
}

/// CPU oracle `Y[token][feature]` (`f32`, row-major `tokens x features`): `I` exact in i128 (asserted `|I| < 2^63`), one
/// [`round_scaled`] with exponent `a_r + b_t`. Panics on malformed operands (see [`Ief15Layout::pack_in`] for the `Err` form).
pub fn reference_values(
    w: &Weights,
    x: &Acts,
    s: &[u16],
    a: &[i16],
    d: &[u16],
    b: &[i16],
) -> Vec<f32> {
    check_operands(w, x, s, a, d, b).unwrap_or_else(|e| panic!("IEF15 reference: {e}"));
    let (tokens, features, k) = (x.tokens, w.rows, x.k);
    let e = k / EPOCH_K;
    let w8: Vec<i8> = (0..features * k).map(|i| w.value(i / k, i % k)).collect();
    let x8: Vec<i8> = (0..tokens * k).map(|i| x.value(i / k, i % k)).collect();
    let mut out = vec![0f32; tokens * features];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(tokens);
    let per = tokens.div_ceil(threads);
    std::thread::scope(|sc| {
        for (part, chunk) in out.chunks_mut(per * features).enumerate() {
            let (w8, x8) = (&w8, &x8);
            sc.spawn(move || {
                for (i, row) in chunk.chunks_mut(features).enumerate() {
                    let t = part * per + i;
                    for (r, y) in row.iter_mut().enumerate() {
                        let mut acc = 0i128;
                        for ep in 0..e {
                            let xr = &x8[t * k + ep * EPOCH_K..][..EPOCH_K];
                            let wr = &w8[r * k + ep * EPOCH_K..][..EPOCH_K];
                            let c: i32 = xr
                                .iter()
                                .zip(wr)
                                .map(|(&p, &q)| i32::from(p) * i32::from(q))
                                .sum();
                            acc += i128::from(s[r * e + ep])
                                * i128::from(d[ep * tokens + t])
                                * i128::from(c);
                        }
                        assert!(
                            acc.abs() < 1i128 << 63,
                            "|I| = {} >= 2^63 at token {t} feature {r}",
                            acc.abs()
                        );
                        *y = round_scaled(acc, i32::from(a[r]) + i32::from(b[t]));
                    }
                }
            });
        }
    });
    out
}

fn le_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|y| y.to_le_bytes()).collect()
}

/// Deterministic native operands with the full `C` range in every epoch: `x` is all `-8` (every A4 block), feature 0 all
/// `u = 0` (`C = +8192`), feature 1 all `u = 15` (`C = -7168`), the other features random. Scales as
/// [`iu4::random_operands`](super::iu4::random_operands).
pub fn extreme_operands(tokens: usize, features: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
    let (mut w, mut x) = super::iu4::random_operands(tokens, features, k, seed, false);
    for blk in x.chunks_mut(super::iu4::XBLK_BYTES) {
        blk[8..].fill(0x88);
        blk[4..8].copy_from_slice(&(-8 * 128i32).to_le_bytes());
    }
    let row = k.div_ceil(256) * super::iu4::GROUP_BYTES;
    for (r, fill) in [(0usize, 0x00u8), (1, 0xff)] {
        if r < features {
            for g in w[r * row..(r + 1) * row].chunks_mut(super::iu4::GROUP_BYTES) {
                g[8..].fill(fill);
            }
        }
    }
    (w, x)
}

// ---------------------------------------------------------------------------------------------------------------------
// Layout: packing, unpacking, reference and the core-contract emulator. No array design (and hence no core) is needed.

/// Column that injects the A half of core row `row` of column `c` (V8: column `c0 = 2*row + c%2`).
fn a_column(c: usize, row: usize) -> usize {
    2 * row + c % 2
}
/// A block (64 tokens of a wave) of column `c`'s cores, the half (`h`) injected by column `c`.
fn inject_block(c: usize) -> (usize, usize) {
    (2 * (c / 4) + c % 2, (c / 2) % 2)
}
/// B column that injects the B half `row % 2` of B block `c / 2` for core row `row` of column `c`.
fn b_column(c: usize, row: usize) -> usize {
    2 * (c / 2) + row % 2
}

/// Byte offset of block `(blk, kb)` of an A half chunk (`mbl`) / B chunk (`kb`, `nb`).
fn a_block_at(mbl: usize, kb: usize) -> usize {
    (mbl * 16 + kb) * 32
}
fn b_block_at(kb: usize, nb: usize) -> usize {
    (kb * 4 + nb) * 32
}

fn set_nibble(buf: &mut [u8], x: usize, nibble: u8) {
    buf[x / 2] |= (nibble & 15) << (4 * (x & 1));
}
fn get_nibble(buf: &[u8], x: usize) -> i8 {
    ((buf[x / 2] >> (4 * (x & 1))) << 4) as i8 >> 4
}

/// Problem geometry of an IEF15 GEMM: everything except the array design.
#[derive(Clone, Copy, Debug)]
pub struct Ief15Layout {
    geo: Geometry,
}

impl Ief15Layout {
    /// `K` a positive multiple of 128 with `1..=136` epochs, `waves = ceil(tokens/256) * ceil(features/256) <= 256`.
    pub fn new(tokens: usize, features: usize, k: usize, ctl: Control) -> Result<Self, String> {
        if tokens == 0 || features == 0 {
            return Err("empty GEMM".into());
        }
        if k == 0 || k % EPOCH_K != 0 {
            return Err(format!("K {k} is not a positive multiple of {EPOCH_K}"));
        }
        if !(1..=MAX_EPOCHS).contains(&(k / EPOCH_K)) {
            return Err(format!("{} epochs outside 1..={MAX_EPOCHS}", k / EPOCH_K));
        }
        let waves = tokens.div_ceil(WAVE) * features.div_ceil(WAVE);
        if waves > array::MAX_WAVES {
            return Err(format!("{waves} waves exceed {}", array::MAX_WAVES));
        }
        Ok(Self {
            geo: Geometry::ief15(tokens, features, k, ctl),
        })
    }
    pub fn tokens(&self) -> usize {
        self.geo.m
    }
    pub fn features(&self) -> usize {
        self.geo.n
    }
    pub fn k(&self) -> usize {
        self.geo.k
    }
    pub fn epochs(&self) -> usize {
        self.geo.kc
    }
    pub fn waves(&self) -> usize {
        self.geo.waves
    }
    /// `(MW, NW)`.
    pub fn wave_grid(&self) -> (usize, usize) {
        (self.geo.mw, self.geo.nw)
    }
    /// Real operations `2 * tokens * features * k`.
    pub fn useful_ops(&self) -> u64 {
        2 * self.geo.m as u64 * self.geo.n as u64 * self.geo.k as u64
    }
    /// Byte sizes of arg0, arg1, arg2.
    pub fn arg_bytes(&self) -> [usize; 3] {
        [self.geo.a_bytes(), self.geo.b_bytes(), self.geo.c_bytes()]
    }

    /// Token of row `i` of the natural 8-token block `mb` of A block `blk` in M-wave `mw`.
    fn token(&self, mw: usize, blk: usize, mb: usize, i: usize) -> usize {
        mw * WAVE + blk * ief::TM + mb * 8 + i
    }
    /// Feature `f` (`nb*8 + j`) of core `(c, h)` in N-wave `nw`.
    fn feature(&self, nw: usize, c: usize, h: usize, f: usize) -> usize {
        nw * WAVE + (c / 2) * ief::TM + h * ief::TN + f
    }

    fn check(
        &self,
        w: &Weights,
        x: &Acts,
        s: &[u16],
        a: &[i16],
        d: &[u16],
        b: &[i16],
    ) -> Result<(), String> {
        if (x.tokens, w.rows, x.k) != (self.tokens(), self.features(), self.k()) {
            return Err(format!(
                "operands are {}x{}x{}, design is {}x{}x{}",
                x.tokens,
                w.rows,
                x.k,
                self.tokens(),
                self.features(),
                self.k()
            ));
        }
        check_operands(w, x, s, a, d, b)
    }

    /// The A segment tile of injecting column `c` for M-wave `mw`: `epochs` chunks of 2176 B (spec §2.1).
    fn a_tile(&self, x: &Acts, d: &[u16], b: &[i16], c: usize, mw: usize) -> Vec<u8> {
        let (tokens, e) = (self.tokens(), self.epochs());
        let (blk, hh) = inject_block(c);
        let mut tile = vec![0u8; e * CHUNK];
        for ep in 0..e {
            let chunk = &mut tile[ep * CHUNK..(ep + 1) * CHUNK];
            for mbl in 0..4 {
                for i in 0..8 {
                    let t = self.token(mw, blk, 2 * mbl + hh, i);
                    if t >= tokens {
                        continue;
                    }
                    let blk72 = &x.bytes[(ep * tokens + t) * super::iu4::XBLK_BYTES..]
                        [..super::iu4::XBLK_BYTES];
                    for kb in 0..16 {
                        chunk[a_block_at(mbl, kb) + i * 4..][..4]
                            .copy_from_slice(&blk72[8 + kb * 4..8 + kb * 4 + 4]);
                    }
                    let lane = mbl * 8 + i;
                    chunk[ief::SCALE_OFF + 2 * lane..][..2]
                        .copy_from_slice(&d[ep * tokens + t].to_le_bytes());
                    chunk[ief::EXP_OFF + 2 * lane..][..2].copy_from_slice(&b[t].to_le_bytes());
                }
            }
        }
        tile
    }

    /// The B segment tile of injecting column `c = 2j + h` for N-wave `nw`: `epochs` chunks of 2176 B (spec §2.2).
    fn b_tile(&self, w: &Weights, s: &[u16], a: &[i16], c: usize, nw: usize) -> Vec<u8> {
        let (features, e) = (self.features(), self.epochs());
        let mut tile = vec![0u8; e * CHUNK];
        for ep in 0..e {
            let chunk = &mut tile[ep * CHUNK..(ep + 1) * CHUNK];
            for f in 0..ief::TN {
                let r = self.feature(nw, c / 2 * 2, c % 2, f);
                if r >= features {
                    continue;
                }
                let (nb, j) = (f / 8, f % 8);
                for kb in 0..16 {
                    for kk in 0..8 {
                        let u_minus_8 = w.value(r, ep * EPOCH_K + kb * 8 + kk);
                        set_nibble(
                            &mut chunk[b_block_at(kb, nb)..][..32],
                            kk * 8 + j,
                            u_minus_8 as u8,
                        );
                    }
                }
                chunk[ief::SCALE_OFF + 2 * f..][..2].copy_from_slice(&s[r * e + ep].to_le_bytes());
                chunk[ief::EXP_OFF + 2 * f..][..2].copy_from_slice(&a[r].to_le_bytes());
            }
        }
        tile
    }

    /// Packed `[arg0, arg1]` (spec §2.3). Errs on shape mismatches, `S` / `D` above 32767 and `a` / `b` outside
    /// `[-16383, 16383]`. Padding tokens / features are all zero (codes, scales, exponents).
    pub fn pack_in(
        &self,
        w: &Weights,
        x: &Acts,
        s: &[u16],
        a: &[i16],
        d: &[u16],
        b: &[i16],
    ) -> Result<[Vec<u8>; 2], String> {
        self.check(w, x, s, a, d, b)?;
        let (mut arg0, mut arg1) = (
            Vec::with_capacity(self.geo.a_bytes()),
            Vec::with_capacity(self.geo.b_bytes()),
        );
        for c in 0..COLS {
            let tiles: Vec<Vec<u8>> = (0..self.geo.mw)
                .map(|mw| self.a_tile(x, d, b, c, mw))
                .collect();
            for wv in 0..self.waves() {
                arg0.extend_from_slice(&tiles[self.geo.wave_coords(wv).0]);
            }
        }
        for c in 0..COLS {
            let tiles: Vec<Vec<u8>> = (0..self.geo.nw)
                .map(|nw| self.b_tile(w, s, a, c, nw))
                .collect();
            for wv in 0..self.waves() {
                arg1.extend_from_slice(&tiles[self.geo.wave_coords(wv).1]);
            }
        }
        debug_assert_eq!(
            (arg0.len(), arg1.len()),
            (self.geo.a_bytes(), self.geo.b_bytes())
        );
        Ok([arg0, arg1])
    }

    /// LE f32 `[token][feature]` from the C argument (padding dropped).
    pub fn unpack_out(&self, c: &[u8]) -> Vec<u8> {
        assert_eq!(c.len(), self.geo.c_bytes(), "C stream size");
        let (tokens, features) = (self.tokens(), self.features());
        let mut out = vec![0u8; tokens * features * 4];
        for col in 0..COLS {
            for wv in 0..self.waves() {
                for row in 0..ROWS {
                    let (mw, nw) = self.geo.wave_coords(wv);
                    let (blk, h) = (2 * (row / 2) + col % 2, row % 2);
                    let tile = &c[self.geo.pair_c_offset(col, wv, row)..][..ief::COUT_BYTES];
                    for mb in 0..8 {
                        for nb in 0..4 {
                            for i in 0..8 {
                                let t = self.token(mw, blk, mb, i);
                                if t >= tokens {
                                    continue;
                                }
                                for j in 0..8 {
                                    let r = self.feature(nw, col, h, nb * 8 + j);
                                    if r >= features {
                                        continue;
                                    }
                                    let at = ((mb * 4 + nb) * 64 + i * 8 + j) * 4;
                                    out[(t * features + r) * 4..][..4]
                                        .copy_from_slice(&tile[at..at + 4]);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The CPU oracle in LE bytes (`reference_values`).
    pub fn reference(
        &self,
        w: &Weights,
        x: &Acts,
        s: &[u16],
        a: &[i16],
        d: &[u16],
        b: &[i16],
    ) -> Vec<u8> {
        self.check(w, x, s, a, d, b)
            .unwrap_or_else(|e| panic!("IEF15 reference: {e}"));
        le_bytes(&reference_values(w, x, s, a, d, b))
    }

    /// CPU emulation of the core contract on the packed arguments: for every core and wave exactly what `docs/ief15-n0.md`
    /// says the core does (own / peer half chunk pairing of a vertical pair, int4 unpack, `C` per epoch, `I = sum S*D*C`,
    /// one `pack(I, a_r + b_t)` per output, core rows rho 0..8 written to their natural `[mb][nb]` blocks), returning arg2
    /// (every byte written; the buffer starts as 0xA5 poison). `pack` maps `(I, a_r + b_t)` to the f32 bits;
    /// [`Ief15Layout::emulate`] uses [`round_scaled`].
    pub fn emulate_with(&self, args: [&[u8]; 2], pack: &dyn Fn(i64, i32) -> u32) -> Vec<u8> {
        assert_eq!(
            [args[0].len(), args[1].len()],
            [self.geo.a_bytes(), self.geo.b_bytes()],
            "argument sizes"
        );
        let (e, waves) = (self.epochs(), self.waves());
        let (a_seg, b_seg) = (self.geo.a_segment_bytes(), self.geo.b_segment_bytes());
        let mut out = vec![0xa5u8; self.geo.c_bytes()];
        let unpack_a = |chunk: &[u8]| -> Vec<i8> {
            let mut v = vec![0i8; 32 * EPOCH_K];
            for mbl in 0..4 {
                for kb in 0..16 {
                    for i in 0..8 {
                        for kk in 0..8 {
                            v[(mbl * 8 + i) * EPOCH_K + kb * 8 + kk] =
                                get_nibble(&chunk[a_block_at(mbl, kb)..], i * 8 + kk);
                        }
                    }
                }
            }
            v
        };
        let unpack_b = |chunk: &[u8]| -> Vec<i8> {
            let mut v = vec![0i8; 32 * EPOCH_K];
            for kb in 0..16 {
                for nb in 0..4 {
                    for kk in 0..8 {
                        for j in 0..8 {
                            v[(nb * 8 + j) * EPOCH_K + kb * 8 + kk] =
                                get_nibble(&chunk[b_block_at(kb, nb)..], kk * 8 + j);
                        }
                    }
                }
            }
            v
        };
        let u16_at = |chunk: &[u8], off: usize, lane: usize| {
            u16::from_le_bytes([chunk[off + 2 * lane], chunk[off + 2 * lane + 1]])
        };
        for c in 0..COLS {
            for row in 0..ROWS {
                for wv in 0..waves {
                    let h = row % 2;
                    let own = |ep: usize| {
                        &args[0][a_column(c, row) * a_seg + (wv * e + ep) * CHUNK..][..CHUNK]
                    };
                    let peer = |ep: usize| {
                        &args[0][a_column(c, row ^ 1) * a_seg + (wv * e + ep) * CHUNK..][..CHUNK]
                    };
                    let bch = |ep: usize| {
                        &args[1][b_column(c, row) * b_seg + (wv * e + ep) * CHUNK..][..CHUNK]
                    };
                    let mut acc = vec![0i128; 8 * 8 * 32];
                    for ep in 0..e {
                        let (ao, ap, bb) =
                            (unpack_a(own(ep)), unpack_a(peer(ep)), unpack_b(bch(ep)));
                        for rho in 0..8 {
                            let (chunk, ints) = if rho < 4 {
                                (own(ep), &ao)
                            } else {
                                (peer(ep), &ap)
                            };
                            for i in 0..8 {
                                let lane = (rho % 4) * 8 + i;
                                let dv = i128::from(u16_at(chunk, ief::SCALE_OFF, lane));
                                let xr = &ints[lane * EPOCH_K..][..EPOCH_K];
                                for f in 0..32 {
                                    let wr = &bb[f * EPOCH_K..][..EPOCH_K];
                                    let cc: i32 = xr
                                        .iter()
                                        .zip(wr)
                                        .map(|(&p, &q)| i32::from(p) * i32::from(q))
                                        .sum();
                                    acc[(rho * 8 + i) * 32 + f] +=
                                        i128::from(u16_at(bch(ep), ief::SCALE_OFF, f))
                                            * dv
                                            * i128::from(cc);
                                }
                            }
                        }
                    }
                    // a and b are identical in every epoch (checked on the first and last chunk).
                    assert!(
                        own(0)[ief::EXP_OFF..] == own(e - 1)[ief::EXP_OFF..]
                            && peer(0)[ief::EXP_OFF..] == peer(e - 1)[ief::EXP_OFF..]
                            && bch(0)[ief::EXP_OFF..] == bch(e - 1)[ief::EXP_OFF..],
                        "a / b differ between epochs"
                    );
                    let base = self.geo.pair_c_offset(c, wv, row);
                    for rho in 0..8 {
                        let (chunk, mb) = if rho < 4 {
                            (own(e - 1), 2 * rho + h)
                        } else {
                            (peer(e - 1), 2 * (rho - 4) + 1 - h)
                        };
                        for i in 0..8 {
                            for f in 0..32 {
                                let bx = i32::from(
                                    u16_at(chunk, ief::EXP_OFF, (rho % 4) * 8 + i) as i16
                                ) + i32::from(u16_at(bch(e - 1), ief::EXP_OFF, f) as i16);
                                let i_acc = i64::try_from(acc[(rho * 8 + i) * 32 + f])
                                    .expect("|I| >= 2^63");
                                let at = base + (((mb * 4 + f / 8) * 64 + i * 8 + f % 8) * 4);
                                out[at..at + 4].copy_from_slice(&pack(i_acc, bx).to_le_bytes());
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// [`Ief15Layout::emulate_with`] with the exact [`round_scaled`].
    pub fn emulate(&self, args: [&[u8]; 2]) -> Vec<u8> {
        self.emulate_with(args, &|i, x| round_scaled(i128::from(i), x).to_bits())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Array design

/// IEF15 memtile descriptors of a column: the V8 streaming-pair map (BD ids, locks, chains, slot offsets) with A / B ring
/// BD lengths of one 2176 B chunk and the C tile of 8192 B (V8 int8 `COUT`).
pub(crate) fn memtile_descriptors(col: u32) -> Vec<(u32, Bd)> {
    let (a, b) = array::pair_ab_ring_bds();
    let mut v = array::v8_pair_memtile_descriptors(col);
    for (id, bd) in &mut v {
        if a.contains(id) {
            bd.len_words = (ief::A_HALF_BYTES / 4) as u32;
        } else if b.contains(id) {
            bd.len_words = (ief::B_BYTES / 4) as u32;
        } else {
            assert_eq!(
                bd.len_words as usize * 4,
                ief::COUT_BYTES,
                "memtile BD {id}: not an A / B / C descriptor"
            );
        }
    }
    v
}

/// The array design of `geo` around explicit core programs (`lower` for the even core rows, `upper` for the odd ones).
pub(crate) fn design_with_programs(
    geo: Geometry,
    ctl: Control,
    lower: Vec<u8>,
    upper: Vec<u8>,
) -> Result<ArrayDesign, String> {
    for (name, p) in [("Lower", &lower), ("Upper", &upper)] {
        if p.len() > 16 * 1024 {
            return Err(format!(
                "IEF15 {name} core program {} B exceeds 16 KiB",
                p.len()
            ));
        }
    }
    let topo = geo.topo;
    let insts = array::build_txn_dynamic(&geo, false);
    Ok(ArrayDesign {
        pdi: crate::pdi::build(&array::build_cdo(topo, [&lower, &upper], false).to_words()),
        insts: insts.to_bytes(),
        args: vec![
            ArgSpec {
                bytes: geo.a_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.b_bytes(),
                kind: ArgKind::In,
            },
            ArgSpec {
                bytes: geo.c_bytes(),
                kind: ArgKind::Out,
            },
        ],
        geo,
        variant: Variant::Ief15,
        program: lower,
        upper_program: upper,
        core: if ctl.is_slow() {
            CoreVariant::FastSlowCtl
        } else {
            CoreVariant::Fast
        },
        v8: None,
        shim_axi: crate::dma::ShimAxi::default(),
    })
}

/// An IEF15 GEMM on the whole array: layout plus [`ArrayDesign`] (programs from
/// `iu4_ief15_core::program_pair(epochs, waves, role, ctl)`). Submit with the design's `pdi` / `insts` (or the lean body)
/// and `[arg0, arg1, arg2]`, `arg2` poisonable (the core writes every byte).
pub struct Ief15Gemm {
    layout: Ief15Layout,
    design: ArrayDesign,
}

impl Ief15Gemm {
    /// Build the design (see [`Ief15Layout::new`] for the domain).
    pub fn new(tokens: usize, features: usize, k: usize, ctl: Control) -> Result<Self, String> {
        let layout = Ief15Layout::new(tokens, features, k, ctl)?;
        let program = |role| ief::program_pair(layout.epochs(), layout.waves(), role, ctl).finish();
        let design = design_with_programs(
            layout.geo,
            ctl,
            program(PairRole::Lower),
            program(PairRole::Upper),
        )?;
        Ok(Self { layout, design })
    }
    /// The same design around explicit core programs (`lower`, `upper`; each at most 16 KiB): tests and probes of the data
    /// movement (descriptors, routes, TXN) with a stand-in core, e.g. the V8 `no_compute` pair program that runs the lock
    /// protocol without touching the data. The C stream is then not a GEMM.
    pub fn with_programs(
        tokens: usize,
        features: usize,
        k: usize,
        ctl: Control,
        lower: Vec<u8>,
        upper: Vec<u8>,
    ) -> Result<Self, String> {
        let layout = Ief15Layout::new(tokens, features, k, ctl)?;
        let design = design_with_programs(layout.geo, ctl, lower, upper)?;
        Ok(Self { layout, design })
    }
    pub fn layout(&self) -> &Ief15Layout {
        &self.layout
    }
    pub fn design(&self) -> &ArrayDesign {
        &self.design
    }
    pub fn epochs(&self) -> usize {
        self.layout.epochs()
    }
    pub fn useful_ops(&self) -> u64 {
        self.layout.useful_ops()
    }
    /// See [`Ief15Layout::pack_in`].
    pub fn pack_in(
        &self,
        w: &Weights,
        x: &Acts,
        s: &[u16],
        a: &[i16],
        d: &[u16],
        b: &[i16],
    ) -> Result<[Vec<u8>; 2], String> {
        self.layout.pack_in(w, x, s, a, d, b)
    }
    /// See [`Ief15Layout::reference`].
    pub fn reference(
        &self,
        w: &Weights,
        x: &Acts,
        s: &[u16],
        a: &[i16],
        d: &[u16],
        b: &[i16],
    ) -> Vec<u8> {
        self.layout.reference(w, x, s, a, d, b)
    }
    /// See [`Ief15Layout::unpack_out`].
    pub fn unpack_out(&self, c: &[u8]) -> Vec<u8> {
        self.layout.unpack_out(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::iu4;
    use std::collections::HashSet;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Random sidecars with S / D extremes (0, 1, 32767) and a / b that reach subnormal and overflow results.
    fn sidecars(
        tokens: usize,
        features: usize,
        e: usize,
        seed: u64,
    ) -> (Vec<u16>, Vec<i16>, Vec<u16>, Vec<i16>) {
        let mut r = Rng(seed | 1);
        let mant = |r: &mut Rng| match r.next() % 6 {
            0 => 0,
            1 => 1,
            2 => 32767,
            _ => (r.next() % 32768) as u16,
        };
        let exp = |r: &mut Rng| match r.next() % 8 {
            0 => -180,
            1 => 120,
            2 => -140,
            3 => 100,
            _ => (r.next() % 61) as i16 - 30,
        };
        let s = (0..features * e).map(|_| mant(&mut r)).collect();
        let a = (0..features).map(|_| exp(&mut r)).collect();
        let d = (0..e * tokens).map(|_| mant(&mut r)).collect();
        let b = (0..tokens).map(|_| exp(&mut r)).collect();
        (s, a, d, b)
    }

    fn emulate_roundtrip(tokens: usize, features: usize, k: usize, seed: u64, extreme: bool) {
        let layout = Ief15Layout::new(tokens, features, k, Control::Fast).unwrap();
        let e = k / EPOCH_K;
        let (wb, xb) = if extreme {
            extreme_operands(tokens, features, k, seed)
        } else {
            iu4::random_operands(tokens, features, k, seed, true)
        };
        let (w, x) = (Weights::new(features, k, &wb), Acts::new(tokens, k, &xb));
        let (s, a, d, b) = sidecars(tokens, features, e, seed ^ 0x9e37);
        let [a0, a1] = layout.pack_in(&w, &x, &s, &a, &d, &b).unwrap();
        assert_eq!(
            [a0.len(), a1.len()],
            [layout.arg_bytes()[0], layout.arg_bytes()[1]]
        );
        let want = layout.reference(&w, &x, &s, &a, &d, &b);
        let c = layout.emulate([&a0, &a1]);
        assert_eq!(c.len(), layout.arg_bytes()[2]);
        let got = layout.unpack_out(&c);
        assert_eq!(got.len(), want.len());
        if let Some(i) =
            (0..got.len() / 4).find(|&i| got[i * 4..i * 4 + 4] != want[i * 4..i * 4 + 4])
        {
            panic!(
                "{tokens}x{features}x{k}: token {} feature {}: got {:#x} want {:#x}",
                i / features,
                i % features,
                u32::from_le_bytes(got[i * 4..i * 4 + 4].try_into().unwrap()),
                u32::from_le_bytes(want[i * 4..i * 4 + 4].try_into().unwrap())
            );
        }
        // The model of the core's final pack gives the same bytes when |I| < 2^51 (always, E <= 136).
        let normal = (0..got.len() / 4)
            .filter(|&i| {
                let ex =
                    (u32::from_le_bytes(got[i * 4..i * 4 + 4].try_into().unwrap()) >> 23) & 255;
                (1..255).contains(&ex)
            })
            .count();
        assert!(
            tokens * features < 64 || normal > got.len() / 4 / 10,
            "operands must produce mostly normal nonzero results ({normal})"
        );
        let c2 = layout.emulate_with([&a0, &a1], &ief::pack_lane_model);
        assert_eq!(c2, c, "pack_lane_model differs from round_scaled");
        // Padding outputs are +0: every padded token / feature slot of the C stream holds +0.
        for col in 0..COLS {
            for wv in 0..layout.waves() {
                for row in 0..ROWS {
                    let (mw, nw) = layout.geo.wave_coords(wv);
                    let (blk, h) = (2 * (row / 2) + col % 2, row % 2);
                    let tile = &c[layout.geo.pair_c_offset(col, wv, row)..][..ief::COUT_BYTES];
                    for mb in 0..8 {
                        for nb in 0..4 {
                            for i in 0..8 {
                                for j in 0..8 {
                                    if layout.token(mw, blk, mb, i) >= tokens
                                        || layout.feature(nw, col, h, nb * 8 + j) >= features
                                    {
                                        let at = ((mb * 4 + nb) * 64 + i * 8 + j) * 4;
                                        assert_eq!(
                                            tile[at..at + 4],
                                            [0; 4],
                                            "padding output must be +0"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn round_scaled_is_single_rounding() {
        // Exact oracle for |x| < 2^53: f64 holds x and x * 2^e exactly, `as f32` rounds once.
        let mut r = Rng(0x1234_5678_9abc_def1);
        let check = |x: i128, e: i32| {
            let want = (x as f64 * 2f64.powi(e)) as f32;
            let got = round_scaled(x, e);
            assert_eq!(got.to_bits(), want.to_bits(), "x {x} e {e}");
        };
        for _ in 0..200_000 {
            let bits = (r.next() % 52) as u32;
            let x = ((r.next() as i128) & ((1i128 << bits) - 1))
                * if r.next() & 1 == 0 { 1 } else { -1 };
            check(x, (r.next() % 400) as i32 - 330);
        }
        // Ties: odd multiples of half an ulp at several magnitudes and in the subnormal range.
        for top in [24u32, 25, 30, 40, 50] {
            for tie in [
                (1i128 << top) + (1 << (top - 24)),
                (3i128 << (top - 1)) + (1 << (top - 24)),
                (1i128 << top) + 3 * (1 << (top - 24)),
            ] {
                for e in [-170, -149, -126, -100, 0, 90, 103] {
                    check(tie, e);
                    check(-tie, e);
                }
            }
        }
        assert_eq!(round_scaled(0, 5).to_bits(), 0);
        assert_eq!(
            round_scaled(-1, -400).to_bits(),
            0x8000_0000,
            "negative underflow is -0"
        );
        assert_eq!(round_scaled(1, 200).to_bits(), 0x7f80_0000);
        assert_eq!(round_scaled(-1, 200).to_bits(), 0xff80_0000);
        assert_eq!(
            round_scaled((1 << 25) - 1, 0),
            33554432.0,
            "carry into the exponent"
        );
        assert_eq!(
            round_scaled((1 << 24) - 1, 104).to_bits(),
            f32::MAX.to_bits()
        );
        assert_eq!(
            round_scaled((1 << 25) - 1, 103).to_bits(),
            0x7f80_0000,
            "rounds up to 2^128: inf"
        );
    }

    #[test]
    fn grid_matches_fold_contract() {
        assert_eq!(grid15(&[0.0, 0.0]), (vec![0, 0], 0));
        assert_eq!(grid15(&[1.0, 0.5, 0.25]), (vec![16384, 8192, 4096], -14));
        // 65535 first rounds to 32768 (tie to even) > 32767 and moves to the next exponent.
        assert_eq!(grid15(&[65535.0]), (vec![16384], 2));
        let (m, a) = grid15(&[f32::from_bits(1), 0.0]);
        assert!(
            (16384..=32767).contains(&m[0]) && a == -149 - 14,
            "smallest subnormal: m {} a {a}",
            m[0]
        );
    }

    #[test]
    fn encoders_reproduce_the_f32_fold() {
        let (tokens, features, k) = (9, 11, 512);
        let (wb, xb) = iu4::random_operands(tokens, features, k, 41, true);
        let (w, x) = (Weights::new(features, k, &wb), Acts::new(tokens, k, &xb));
        let (s, a) = encode_weight_scales(&w);
        let (d, b) = encode_act_scales(&x);
        let e = k / EPOCH_K;
        assert!(s.iter().chain(&d).all(|&v| u32::from(v) <= MANT_MAX));
        assert_eq!(
            (s.len(), a.len(), d.len(), b.len()),
            (features * e, features, e * tokens, tokens)
        );
        // Each row / token uses the full 15 bits for its largest scale.
        for r in 0..features {
            assert!((16384..=32767).contains(s[r * e..(r + 1) * e].iter().max().unwrap()));
        }
        let y = reference_values(&w, &x, &s, &a, &d, &b);
        let parts = iu4::partials(&w, &x);
        let fold = iu4::fold_set(&w, &x, &parts);
        for t in 0..tokens {
            for r in 0..features {
                // Projection error of every term: |dS| <= 2^(a-1), |dD| <= 2^(b-1), plus the f32 fold's own rounding.
                let (mut bound, mut mag) = (0f64, 0f64);
                for ep in 0..e {
                    let (sc, dd) = (f64::from(w.scale(r, ep)), f64::from(x.d(t, ep)));
                    let c = f64::from(parts[(ep * tokens + t) * features + r]).abs();
                    let (ds, dd_) = (
                        2f64.powi(i32::from(a[r]) - 1),
                        2f64.powi(i32::from(b[t]) - 1),
                    );
                    bound += c * (ds * dd + sc * dd_ + ds * dd_);
                    mag += c * sc * dd;
                }
                let diff =
                    (f64::from(y[t * features + r]) - f64::from(fold[t * features + r])).abs();
                assert!(
                    diff <= bound + mag * 2f64.powi(-20),
                    "token {t} feature {r}: {diff} > {bound}"
                );
            }
        }
        // The oracle's C equals the GPU partials: I recomputed from `partials` matches `reference_values` exactly.
        for t in 0..tokens {
            for r in 0..features {
                let i: i128 = (0..e)
                    .map(|ep| {
                        i128::from(s[r * e + ep])
                            * i128::from(d[ep * tokens + t])
                            * i128::from(parts[(ep * tokens + t) * features + r])
                    })
                    .sum();
                assert_eq!(
                    y[t * features + r].to_bits(),
                    round_scaled(i, i32::from(a[r]) + i32::from(b[t])).to_bits()
                );
            }
        }
    }

    #[test]
    fn pack_in_validates() {
        let (tokens, features, k) = (3, 5, 256);
        let layout = Ief15Layout::new(tokens, features, k, Control::Fast).unwrap();
        let (wb, xb) = iu4::random_operands(tokens, features, k, 5, false);
        let (w, x) = (Weights::new(features, k, &wb), Acts::new(tokens, k, &xb));
        let (s, a, d, b) = sidecars(tokens, features, 2, 6);
        assert!(layout.pack_in(&w, &x, &s, &a, &d, &b).is_ok());
        let err = |r: Result<[Vec<u8>; 2], String>| r.unwrap_err();
        let mut bad = s.clone();
        bad[3] = 32768;
        assert!(err(layout.pack_in(&w, &x, &bad, &a, &d, &b)).contains("S[3]"));
        let mut bad = d.clone();
        bad[1] = 40000;
        assert!(err(layout.pack_in(&w, &x, &s, &a, &bad, &b)).contains("D[1]"));
        let mut bad = a.clone();
        bad[2] = 16384;
        assert!(err(layout.pack_in(&w, &x, &s, &bad, &d, &b)).contains("a[2]"));
        let mut bad = b.clone();
        bad[0] = -16384;
        assert!(err(layout.pack_in(&w, &x, &s, &a, &d, &bad)).contains("b[0]"));
        let (mut ok_a, mut ok_b) = (a.clone(), b.clone());
        ok_a[0] = 16383;
        ok_b[0] = -16383;
        assert!(layout.pack_in(&w, &x, &s, &ok_a, &d, &ok_b).is_ok());
        assert!(err(layout.pack_in(&w, &x, &s[1..], &a, &d, &b)).contains("S has"));
        assert!(err(layout.pack_in(&w, &x, &s, &a, &d, &b[1..])).contains("b has"));
        let other = Ief15Layout::new(tokens + 1, features, k, Control::Fast).unwrap();
        assert!(err(other.pack_in(&w, &x, &s, &a, &d, &b)).contains("operands are"));
        for (t, f, kk) in [
            (0, 1, 128),
            (1, 0, 128),
            (1, 1, 0),
            (1, 1, 100),
            (1, 1, 128 * 137),
            (256 * 257, 256, 128),
        ] {
            assert!(
                Ief15Layout::new(t, f, kk, Control::Fast).is_err(),
                "{t}x{f}x{kk}"
            );
        }
        assert!(Ief15Layout::new(256 * 16, 256 * 16, 128, Control::Fast).is_ok());
        assert!(Ief15Layout::new(256 * 16 + 1, 256 * 16, 128, Control::Fast).is_err());
    }

    #[test]
    #[should_panic(expected = "exceeds 32767")]
    fn reference_rejects_oversized_scales() {
        let (wb, xb) = iu4::random_operands(1, 1, 128 * 2, 5, false);
        let _ = reference_values(
            &Weights::new(1, 256, &wb),
            &Acts::new(1, 256, &xb),
            &[1, 32768],
            &[0],
            &[1, 1],
            &[0],
        );
    }

    #[test]
    fn packing_smallest_one_epoch() {
        emulate_roundtrip(256, 256, 128, 1, true)
    }
    #[test]
    fn packing_padding_300_700_e5() {
        emulate_roundtrip(300, 700, 640, 2, false)
    }
    #[test]
    fn packing_tiny_shapes() {
        emulate_roundtrip(1, 1, 128, 3, false);
        emulate_roundtrip(65, 33, 384, 4, true);
    }
    #[test]
    fn packing_e20_e40_e48_small() {
        for (e, seed) in [(20usize, 7u64), (40, 8), (48, 9)] {
            emulate_roundtrip(70, 90, e * EPOCH_K, seed, e == 40)
        }
    }
    #[test]
    fn packing_e136() {
        emulate_roundtrip(256, 256, 136 * EPOCH_K, 10, true)
    }
    #[test]
    fn packing_extreme_c_every_epoch() {
        emulate_roundtrip(128, 96, 20 * EPOCH_K, 11, true)
    }
    #[test]
    fn packing_multi_wave_order() {
        emulate_roundtrip(257, 513, 256, 12, false)
    }

    #[test]
    fn extreme_operands_reach_full_c_range() {
        let (wb, xb) = extreme_operands(4, 3, 640, 1);
        let p = iu4::partials(&Weights::new(3, 640, &wb), &Acts::new(4, 640, &xb));
        for e in 0..5 {
            for t in 0..4 {
                assert_eq!(&p[(e * 4 + t) * 3..][..2], &[8192, -7168]);
            }
        }
    }

    #[test]
    fn geometry_and_descriptors() {
        let layout = Ief15Layout::new(300, 700, 640, Control::Fast).unwrap();
        let geo = layout.geo;
        assert_eq!((geo.mw, geo.nw, geo.waves, geo.kc), (2, 3, 6, 5));
        let unit = geo.waves * geo.kc * CHUNK;
        assert_eq!(
            layout.arg_bytes(),
            [8 * unit, 8 * unit, 8 * geo.waves * 4 * 8192]
        );
        assert_eq!(geo.pair_c_offset(0, 0, 0), 0);
        assert_eq!(geo.pair_c_offset(1, 0, 0), geo.waves * 4 * 8192);
        assert_eq!(geo.pair_c_offset(0, 0, 2), geo.waves * 2 * 8192);
        // Descriptors: V8 ids and offsets, A / B ring lengths of one chunk, C 8192 B.
        let (a, b) = array::pair_ab_ring_bds();
        let v8 = array::v8_pair_memtile_descriptors(0);
        let ief = memtile_descriptors(0);
        assert_eq!(v8.len(), ief.len());
        for ((i8, v), (i15, f)) in v8.iter().zip(&ief) {
            assert_eq!(i8, i15);
            assert_eq!(
                (v.addr, v.locks, v.next, v.iteration),
                (f.addr, f.locks, f.next, f.iteration)
            );
            let want = if a.contains(i8) {
                544
            } else if b.contains(i8) {
                544
            } else {
                2048
            };
            assert_eq!(f.len_words, want, "BD {i8}");
        }
        assert_eq!(a.iter().chain(&b).collect::<HashSet<_>>().len(), 8);
        // Switch circuits are V8's, the core BDs and locks are the core module's.
        let topo = geo.topo;
        assert!(
            topo.is_pair()
                && topo.is_ief15()
                && topo.c_tile_bytes() == 8192
                && topo.wave_m() == 256
                && topo.wave_n() == 256
        );
        // CDO and dynamic TXN build around placeholder programs (the real core is not needed for the descriptors).
        let d = design_with_programs(geo, Control::Fast, vec![0; 64], vec![0; 64]).unwrap();
        assert_eq!(d.variant(), Variant::Ief15);
        assert_eq!(
            d.args.iter().map(|a| a.bytes).collect::<Vec<_>>(),
            layout.arg_bytes()
        );
        assert_eq!(d.waves(), 6);
        assert!(!d.insts.is_empty() && !d.pdi.is_empty());
        let lean = d.lean_insts().unwrap();
        assert!(lean.len() < d.insts.len());
        assert_eq!(d.useful_ops(), 2 * 300 * 700 * 640);
        assert!(design_with_programs(geo, Control::Fast, vec![0; 16385], vec![]).is_err());
    }

    #[test]
    fn lean_body_grows_with_ring_parity() {
        // Odd J = epochs * waves implies odd waves: none < waves odd (C rings) < both odd (A / B rings as well).
        let mk = |t: usize, k: usize| {
            let g = Geometry::ief15(t, 256, k, Control::Fast);
            design_with_programs(g, Control::Fast, vec![0; 64], vec![0; 64])
                .unwrap()
                .lean_insts()
                .unwrap()
                .len()
        };
        let (none, waves_odd, both_odd) = (mk(512, 256), mk(256, 256), mk(256, 128 * 3));
        assert!(
            none < waves_odd && waves_odd < both_odd,
            "{none} {waves_odd} {both_odd}"
        );
    }
}
