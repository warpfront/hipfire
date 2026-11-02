// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! hipfire's dense IU4 prefill GEMM contract (symmetric MQ4G256V2 weights x `block_i4_128` activations, gfx1151
//! `gemm_mq4g256v2_*_iu4_pm_v2b_*`) and the NPU deployment that produces its per-K128-epoch int32 partials.
//!
//! Native operand formats, read unchanged (hipfire `crates/peacemaker-isa/src/kernels/iu4_fold.rs`,
//! `kernels/src/gemm_mq4g256v2_residual_iu4_v2b.gfx11.hip`, `kernels/src/block_i4_128_quant.hip`):
//! * Weights `W`: `rows` output features, each `K/256` groups of [`GROUP_BYTES`] = `[f16 sc0, f16 zp0, f16 sc1,
//!   f16 zp1]` + 128 nibble bytes. K half `h` of the group (`K` epoch `e = 2*group + h`) is the 64 bytes at
//!   `8 + 64*h`; byte `j` holds K element `2j` in its low and `2j+1` in its high nibble. Nibbles are unsigned `u`
//!   with the symmetric contract `zp == -8*sc`; the GPU rebiases with `XOR 0x88888888`, so the integer operand is
//!   `u - 8` in `[-8, 7]`. The epoch scale is `sc_e` (bytes `4*h..4*h+2` of the header); `zp` is unused.
//! * Activations `X`: `block_i4_128[e*tokens + t]` of [`XBLK_BYTES`] = `{f32 d, i32 s, u8 qs[64]}`; `qs` packs K
//!   elements `128e + 2j` (low nibble) and `+1` (high) as two's complement int4 in `[-8, 7]`.
//!
//! GPU numerics per output `(t, r)` (token, feature) and ascending epoch `e`:
//! `C_e = sum_{j<128} (u_{r,128e+j} - 8) * q_{t,128e+j}` exactly (int32 WMMA chain; `-7168 <= C_e <= 8192`), then
//! `sum = fma(RN(f32(sc_e) * d_e), f32(C_e), sum)` from `sum = +0`. SET stores `sum`; ADD stores `RN(old + sum)`;
//! the F1-lite gate/up entry stores `g / (1 + expf(-g)) * u` with hipcc's `expf`/`fdiv` expansion (not modelled here).
//!
//! The NPU deployment ([`EpochGemm`]) computes the `C_e` bit-exactly and returns them; the f32 fold stays on the GPU.
//! It is the streaming V8 array design with `Epilogue::I32` and K = 128, run as a batched GEMM whose M-waves are
//! `(epoch, token block)` pairs: wave rows `e*Mpad + t` carry `X` epoch `e`, and the B tile of every wave of epoch
//! `e` is `W` epoch `e` ([`gemm_array::ArrayDesign::pack_in_per_mw`]). The core programs, routes, descriptors and
//! TXN are V8's unchanged; only the host packing differs. Operands are expanded on the host from the native
//! nibbles to the int8 the V8 core consumes (the same integers; no precision change).
use super::gemm_array::{self, ArrayDesign};
use super::gemm_core::{Control, Epilogue};

/// MQ4G256V2 group bytes (`[sc0 zp0 sc1 zp1]` f16 + 128 nibble bytes).
pub const GROUP_BYTES: usize = 136;
/// `block_i4_128` bytes (`f32 d, i32 s, 64 nibble bytes`).
pub const XBLK_BYTES: usize = 72;
/// K elements per fold epoch.
pub const EPOCH_K: usize = 128;

/// Exact IEEE binary16 -> binary32 (subnormals, infinities and NaN payloads preserved).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: value m * 2^-24, normalize into binary32.
            let shift = m.leading_zeros() - 21; // brings the top set bit to bit 10
            let m = (m << shift) & 0x3ff;
            sign | ((127 - 15 + 1 - shift) << 23) | (m << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Symmetric MQ4G256V2 weights: `rows` features x `k` (row-major groups).
#[derive(Clone, Copy)]
pub struct Weights<'a> {
    pub rows: usize,
    pub k: usize,
    pub bytes: &'a [u8],
}

impl<'a> Weights<'a> {
    /// `k` is a positive multiple of 128: whole MQ4G256V2 groups, plus (IEF15 odd epoch counts, tests) one tail group of which
    /// only the first K half (epoch `k/128 - 1`) is addressed; its second half is never read.
    pub fn new(rows: usize, k: usize, bytes: &'a [u8]) -> Self {
        assert!(
            k % EPOCH_K == 0 && k > 0,
            "MQ4G256V2 K must be a positive multiple of 128 (256 for whole groups)"
        );
        assert_eq!(
            bytes.len(),
            rows * k.div_ceil(256) * GROUP_BYTES,
            "weight bytes"
        );
        Self { rows, k, bytes }
    }
    fn group(&self, r: usize, g: usize) -> &'a [u8] {
        let at = (r * self.k.div_ceil(256) + g) * GROUP_BYTES;
        &self.bytes[at..at + GROUP_BYTES]
    }
    /// Rebiased integer operand `u - 8` of feature `r`, K index `kk`.
    pub fn value(&self, r: usize, kk: usize) -> i8 {
        let e = kk / EPOCH_K;
        let j = kk % EPOCH_K;
        let byte = self.group(r, e / 2)[8 + 64 * (e % 2) + j / 2];
        let u = if j % 2 == 0 { byte & 15 } else { byte >> 4 };
        u as i8 - 8
    }
    /// Epoch scale `sc_e` as the GPU widens it (f16 -> f32, exact).
    pub fn scale(&self, r: usize, e: usize) -> f32 {
        let hdr = self.group(r, e / 2);
        let at = 4 * (e % 2);
        f16_to_f32(u16::from_le_bytes([hdr[at], hdr[at + 1]]))
    }
}

/// `block_i4_128` activations: `tokens` x `k`, block `e*tokens + t`.
#[derive(Clone, Copy)]
pub struct Acts<'a> {
    pub tokens: usize,
    pub k: usize,
    pub bytes: &'a [u8],
}

impl<'a> Acts<'a> {
    pub fn new(tokens: usize, k: usize, bytes: &'a [u8]) -> Self {
        assert!(
            k % EPOCH_K == 0 && k > 0,
            "block_i4_128 K must be a positive multiple of 128"
        );
        assert_eq!(
            bytes.len(),
            (k / EPOCH_K) * tokens * XBLK_BYTES,
            "activation bytes"
        );
        Self { tokens, k, bytes }
    }
    fn block(&self, t: usize, e: usize) -> &'a [u8] {
        let at = (e * self.tokens + t) * XBLK_BYTES;
        &self.bytes[at..at + XBLK_BYTES]
    }
    /// Signed int4 operand of token `t`, K index `kk`.
    pub fn value(&self, t: usize, kk: usize) -> i8 {
        let j = kk % EPOCH_K;
        let byte = self.block(t, kk / EPOCH_K)[8 + j / 2];
        let q = if j % 2 == 0 { byte << 4 } else { byte & 0xf0 };
        (q as i8) >> 4
    }
    /// Epoch scale `d_e` (f32).
    pub fn d(&self, t: usize, e: usize) -> f32 {
        let b = self.block(t, e);
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}

/// Integer operands of epoch `e`: activations `tokens x 128` (row-major) into `xa`, weights `128 x rows`
/// (row-major, i.e. `W^T`) into `wb`.
fn epoch_operands(w: &Weights, x: &Acts, e: usize, xa: &mut [i8], wb: &mut [i8]) {
    for t in 0..x.tokens {
        for j in 0..EPOCH_K {
            xa[t * EPOCH_K + j] = x.value(t, e * EPOCH_K + j);
        }
    }
    for r in 0..w.rows {
        for j in 0..EPOCH_K {
            wb[j * w.rows + r] = w.value(r, e * EPOCH_K + j);
        }
    }
}

/// CPU reference of the GPU's int32 epoch partials: `[e][t][r]` (epoch, token, feature), `C_e` exactly.
pub fn partials(w: &Weights, x: &Acts) -> Vec<i32> {
    assert_eq!(w.k, x.k, "K mismatch");
    let (tokens, rows, epochs) = (x.tokens, w.rows, w.k / EPOCH_K);
    let mut out = vec![0i32; epochs * tokens * rows];
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(epochs.max(1));
    std::thread::scope(|s| {
        for (part, chunk) in out
            .chunks_mut(tokens * rows * epochs.div_ceil(threads))
            .enumerate()
        {
            let e0 = part * epochs.div_ceil(threads);
            s.spawn(move || {
                let mut xa = vec![0i8; tokens * EPOCH_K];
                let mut wt = vec![0i8; rows * EPOCH_K];
                for (i, c) in chunk.chunks_mut(tokens * rows).enumerate() {
                    let e = e0 + i;
                    // Feature-major copy of W epoch e so the inner loop is two contiguous 128-element rows.
                    for t in 0..tokens {
                        for j in 0..EPOCH_K {
                            xa[t * EPOCH_K + j] = x.value(t, e * EPOCH_K + j);
                        }
                    }
                    for r in 0..rows {
                        for j in 0..EPOCH_K {
                            wt[r * EPOCH_K + j] = w.value(r, e * EPOCH_K + j);
                        }
                    }
                    for t in 0..tokens {
                        let xr = &xa[t * EPOCH_K..(t + 1) * EPOCH_K];
                        for r in 0..rows {
                            let wr = &wt[r * EPOCH_K..(r + 1) * EPOCH_K];
                            c[t * rows + r] = xr
                                .iter()
                                .zip(wr)
                                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                                .sum();
                        }
                    }
                }
            });
        }
    });
    out
}

/// The GPU's f32 SET fold of `partials` (`[e][t][r]`): `[t][r]`, `sum = fma(RN(sc_e * d_e), C_e, sum)` from `+0`
/// in ascending `e`. ADD is `RN(old + sum)` on this result.
pub fn fold_set(w: &Weights, x: &Acts, partials: &[i32]) -> Vec<f32> {
    let (tokens, rows, epochs) = (x.tokens, w.rows, w.k / EPOCH_K);
    assert_eq!(partials.len(), epochs * tokens * rows, "partials length");
    let mut sum = vec![0.0f32; tokens * rows];
    for e in 0..epochs {
        let sc: Vec<f32> = (0..rows).map(|r| w.scale(r, e)).collect();
        for t in 0..tokens {
            let d = x.d(t, e);
            for r in 0..rows {
                let i = t * rows + r;
                sum[i] = (sc[r] * d).mul_add(partials[(e * tokens + t) * rows + r] as f32, sum[i]);
            }
        }
    }
    sum
}

/// Deterministic native operands for tests and the hardware harness: random nibbles, positive normal f16 scales
/// in `[2^-10, 2^-3)`, `d` positive normal f32 and `s` the exact nibble sum. `extremes` forces the all-`-8`
/// activation block 0 of every token and features 0/1 to all-`u = 0` / all-`u = 15`, so `C_0` reaches
/// `+8192` and `-7168` (the full GPU range).
pub fn random_operands(
    tokens: usize,
    rows: usize,
    k: usize,
    seed: u64,
    extremes: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut w = vec![0u8; rows * k.div_ceil(256) * GROUP_BYTES];
    for g in w.chunks_mut(GROUP_BYTES) {
        for h in 0..2 {
            let sc: u16 = (((next() % 7) as u16 + 5) << 10) | (next() as u16 & 0x3ff);
            let zp = sc ^ 0x8000; // -sc: a placeholder, unused by the symmetric fold
            g[4 * h..4 * h + 2].copy_from_slice(&sc.to_le_bytes());
            g[4 * h + 2..4 * h + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for b in &mut g[8..] {
            *b = next() as u8;
        }
    }
    let mut x = vec![0u8; (k / EPOCH_K) * tokens * XBLK_BYTES];
    for blk in x.chunks_mut(XBLK_BYTES) {
        let d = f32::from_bits(((next() % 20) as u32 + 110) << 23 | (next() as u32 & 0x7f_ffff));
        for b in &mut blk[8..] {
            *b = next() as u8;
        }
        let s: i32 = blk[8..]
            .iter()
            .map(|&b| i32::from(((b << 4) as i8) >> 4) + i32::from((b as i8) >> 4))
            .sum();
        blk[0..4].copy_from_slice(&d.to_le_bytes());
        blk[4..8].copy_from_slice(&s.to_le_bytes());
    }
    if extremes {
        for t in 0..tokens {
            let blk = &mut x[t * XBLK_BYTES..(t + 1) * XBLK_BYTES];
            blk[8..].fill(0x88);
            blk[4..8].copy_from_slice(&(-8 * 128i32).to_le_bytes());
        }
        for (r, fill) in [(0usize, 0x00u8), (1, 0xff)] {
            if r < rows {
                w[r * k.div_ceil(256) * GROUP_BYTES + 8..r * k.div_ceil(256) * GROUP_BYTES + 72]
                    .fill(fill);
            }
        }
    }
    (w, x)
}

/// Epoch-batched V8 deployment producing the `C_e` partials of a `tokens x rows x k` IU4 GEMM.
pub struct EpochGemm {
    pub design: ArrayDesign,
    pub tokens: usize,
    pub rows: usize,
    pub k: usize,
    /// Token rows per epoch block (tokens padded to the 512-row wave).
    pub mpad: usize,
}

impl EpochGemm {
    /// `k` a multiple of 256 (MQ4G256V2), `epochs * ceil(tokens/512) * ceil(rows/512) <= 256` waves.
    pub fn new(tokens: usize, rows: usize, k: usize, ctl: Control) -> Self {
        assert!(
            k % 256 == 0 && k > 0,
            "K must be a positive multiple of 256"
        );
        let mpad = tokens.div_ceil(gemm_array::WAVE_M) * gemm_array::WAVE_M;
        let design = gemm_array::design_v8((k / EPOCH_K) * mpad, rows, EPOCH_K, Epilogue::I32, ctl);
        assert_eq!(design.wave_m(), gemm_array::WAVE_M);
        Self {
            design,
            tokens,
            rows,
            k,
            mpad,
        }
    }
    pub fn epochs(&self) -> usize {
        self.k / EPOCH_K
    }
    /// Real useful ops `2*tokens*rows*k`.
    pub fn useful_ops(&self) -> u64 {
        2 * (self.tokens * self.rows * self.k) as u64
    }
    /// Packed `[arg0, arg1]` from the native operands.
    pub fn pack_in(&self, w: &Weights, x: &Acts) -> [Vec<u8>; 2] {
        assert_eq!(
            (w.rows, w.k, x.tokens, x.k),
            (self.rows, self.k, self.tokens, self.k),
            "operand shapes"
        );
        let epochs = self.epochs();
        let mut a = vec![0i8; epochs * self.mpad * EPOCH_K];
        let mut b = vec![0i8; epochs * EPOCH_K * self.rows];
        let mut xa = vec![0i8; self.tokens * EPOCH_K];
        for e in 0..epochs {
            let wb = &mut b[e * EPOCH_K * self.rows..(e + 1) * EPOCH_K * self.rows];
            epoch_operands(w, x, e, &mut xa, wb);
            a[e * self.mpad * EPOCH_K..][..self.tokens * EPOCH_K].copy_from_slice(&xa);
        }
        let waves_per_epoch = self.mpad / gemm_array::WAVE_M;
        let per = EPOCH_K * self.rows;
        let b_of_mw = |mw: usize| {
            let e = mw / waves_per_epoch;
            &b[e * per..(e + 1) * per]
        };
        self.design.pack_in_per_mw(&a, &b_of_mw)
    }
    /// `[e][t][r]` partials from the packed C argument.
    pub fn unpack_out(&self, out: &[u8]) -> Vec<i32> {
        let full = self.design.unpack_out(out);
        let mut p = Vec::with_capacity(self.epochs() * self.tokens * self.rows);
        for e in 0..self.epochs() {
            p.extend_from_slice(&full[e * self.mpad * self.rows..][..self.tokens * self.rows]);
        }
        p
    }
    /// The CPU partials in [`ArrayDesign::unpack_out`] form (`epochs*mpad x rows`, padded tokens zero).
    pub fn padded(&self, partials: &[i32]) -> Vec<i32> {
        let mut full = vec![0i32; self.epochs() * self.mpad * self.rows];
        for e in 0..self.epochs() {
            full[e * self.mpad * self.rows..][..self.tokens * self.rows].copy_from_slice(
                &partials[e * self.tokens * self.rows..(e + 1) * self.tokens * self.rows],
            );
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_widening_is_exact() {
        for (h, want) in [
            (0x3c00u16, 1.0f32),
            (0xc000, -2.0),
            (0x0001, 2f32.powi(-24)),
            (0x03ff, 1023.0 * 2f32.powi(-24)),
            (0x0400, 2f32.powi(-14)),
            (0x7bff, 65504.0),
            (0x8000, -0.0),
            (0x7c00, f32::INFINITY),
        ] {
            assert_eq!(f16_to_f32(h).to_bits(), want.to_bits(), "{h:#06x}");
        }
        assert!(f16_to_f32(0x7e00).is_nan());
        // Every finite half round-trips through f64 arithmetic of its definition.
        for h in 0..0x7c00u16 {
            let (e, m) = (i32::from(h >> 10), f64::from(h & 0x3ff));
            let v = if e == 0 {
                m * 2f64.powi(-24)
            } else {
                (1.0 + m / 1024.0) * 2f64.powi(e - 15)
            };
            assert_eq!(f64::from(f16_to_f32(h)), v, "{h:#06x}");
        }
    }

    /// Nibble position == K index on both sides, the rebias, and the scale bytes per epoch.
    #[test]
    fn native_operand_decoding() {
        let k = 512;
        let mut w = vec![0u8; k / 256 * GROUP_BYTES];
        // Group 1, half 1 (epoch 3), K element 3*128 + 5 = byte 2 high nibble of that half.
        w[GROUP_BYTES + 8 + 64 + 2] = 0xf0;
        w[GROUP_BYTES + 4..GROUP_BYTES + 6].copy_from_slice(&0x4000u16.to_le_bytes()); // sc1 of group 1 = 2.0
        let ww = Weights::new(1, k, &w);
        assert_eq!(ww.value(0, 3 * 128 + 5), 7);
        assert_eq!(ww.value(0, 3 * 128 + 4), -8);
        assert_eq!(ww.scale(0, 3), 2.0);
        assert_eq!(ww.scale(0, 2), 0.0);
        let mut x = vec![0u8; (k / 128) * 2 * XBLK_BYTES];
        let blk = (2 * 2 + 1) * XBLK_BYTES; // epoch 2, token 1
        x[blk + 8 + 10] = 0x9f; // K 2*128+20 low nibble -1, K 2*128+21 high nibble -7
        x[blk..blk + 4].copy_from_slice(&0.5f32.to_le_bytes());
        let xx = Acts::new(2, k, &x);
        assert_eq!(
            (
                xx.value(1, 256 + 20),
                xx.value(1, 256 + 21),
                xx.value(0, 256 + 20)
            ),
            (-1, -7, 0)
        );
        assert_eq!(xx.d(1, 2), 0.5);
    }

    /// The fold is the GPU's single-rounded fma chain, not multiply-round-then-add.
    #[test]
    fn fold_order_and_rounding() {
        let k = 256;
        let mut w = vec![0u8; GROUP_BYTES];
        w[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // sc0 = 1
        w[4..6].copy_from_slice(&0x3c00u16.to_le_bytes()); // sc1 = 1
        let mut x = vec![0u8; 2 * XBLK_BYTES];
        let (d0, d1) = (4.0f32 - 2f32.powi(-21), 1.0f32 - 2f32.powi(-24));
        x[0..4].copy_from_slice(&d0.to_le_bytes());
        x[XBLK_BYTES..XBLK_BYTES + 4].copy_from_slice(&d1.to_le_bytes());
        let (ww, xx) = (Weights::new(1, k, &w), Acts::new(1, k, &x));
        // sum_0 = 4 - 2^-21; exact sum_0 + 3*d1 = 7 - 1.375 ulp(7): fused rounds to 7 - 2^-21, while rounding the
        // product first (3 - 2^-22) makes a tie that goes to the even 7 - 2^-20.
        let got = fold_set(&ww, &xx, &[1, 3])[0];
        assert_eq!(got, 7.0 - 2f32.powi(-21));
        assert_eq!(d0 + d1 * 3.0, 7.0 - 2f32.powi(-20));
    }

    #[test]
    fn extreme_operands_reach_the_gpu_range() {
        let (tokens, rows, k) = (2, 3, 256);
        let (w, x) = random_operands(tokens, rows, k, 7, true);
        let p = partials(&Weights::new(rows, k, &w), &Acts::new(tokens, k, &x));
        assert_eq!(&p[0..3][..2], &[8192, -7168]);
    }
}
