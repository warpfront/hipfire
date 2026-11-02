// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Q4LM: the 4-bit Lloyd-Max row encoding of the Qwen4 PLE n-gram table.
//!
//! A Q4LM row of `width` values is `width / 32` blocks of 18 bytes
//! ([`Q4LM_BLOCK_BYTES`]):
//!
//! ```text
//! block (18 B)  [0..2)   f16 scale bits, little endian
//!               [2..18)  16 bytes of 4-bit labels; byte i = label[2i] | (label[2i+1] << 4)
//! ```
//!
//! A 160-wide PLE row is 5 blocks = 90 bytes.  Label `l` selects
//! [`LM4_CODEBOOK`]`[l]`, the 16 Lloyd-Max levels of a unit normal.
//!
//! # Encode (from a Q8F16 block of `[f16 scale][32 x i8]`)
//!
//! The encoder reproduces, operation for operation, the diagnostic emulation
//! `lm4` PLE emulation (branch `diag/ple-lm-emulate`, commit
//! `233cb86ac`, `external_rows.rs::lut_requant_block`):
//!
//! 1. `v[i] = f16_to_f32(scale_q8) * (q8[i] as i8 as f32)` (the Q8F16 decode).
//! 2. `rms = (sum_i f64(v[i])^2 / 32).sqrt() as f32`, the f64 sum taken
//!    sequentially in index order.
//! 3. For each `ratio` of [`LM4_SCALE_GRID`], in grid order:
//!    `scale = f16::from_f32(rms * ratio).to_f32()`;
//!    `safe = if scale == 0.0 { 1.0 } else { scale }`; per value `t = v / safe`,
//!    `label` = the number of codebook midpoints `(book[k] + book[k + 1]) / 2.0`
//!    (f32) strictly below `t` (a NaN `t` gives label 0), `y = book[label] * scale`,
//!    and `err += (f64(y) - f64(v))^2` sequentially in index order.
//! 4. The first ratio with `err < best_err` (`best_err` starts at `+inf`) wins; its
//!    `f16::from_f32(rms * ratio)` bits and labels are stored.
//!
//! The emulation keeps the *unquantised* values when no ratio ever beats
//! `+inf` (every ratio's error is infinite or NaN: an f16-overflowing scale or
//! a non-finite input).  A Q4LM block cannot hold that, so
//! [`encode_q4lm_row_from_q8`] returns an error for such a block instead.
//! Real Q8F16 rows never reach it (it needs `rms * 0.65 > 65504`).
//!
//! # Decode
//!
//! `out_bf16[j] = f32_to_bf16_rne(LM4_CODEBOOK[label_j] * f16_to_f32(scale_bits))`,
//! the same RNE narrowing `external_rows` uses for every row encoding.  The f32
//! product is the emulation's `y` exactly, so a decoded row equals the
//! emulation's BF16 row bit for bit.

use crate::external_rows::f32_to_bf16_bits;
use crate::llama::f16_to_f32;

/// Values per Q4LM block (one f16 scale each).
pub const Q4LM_BLOCK_WEIGHTS: usize = 32;
/// Bytes of one Q4LM block: f16 scale + 16 bytes of packed 4-bit labels.
pub const Q4LM_BLOCK_BYTES: usize = 18;

/// Bytes of one Q8F16 block: f16 scale + 32 signed bytes.
const Q8_BLOCK_BYTES: usize = 34;

/// Lloyd-Max levels of a unit normal for 16 levels (4 bits), ascending.
pub const LM4_CODEBOOK: [f32; 16] = [
    -2.7325894832611084,
    -2.069017171859741,
    -1.6180464029312134,
    -1.2562311887741089,
    -0.9423404335975647,
    -0.6567591428756714,
    -0.38804829120635986,
    -0.12839503586292267,
    0.12839503586292267,
    0.38804829120635986,
    0.6567591428756714,
    0.9423404335975647,
    1.2562311887741089,
    1.6180464029312134,
    2.069017171859741,
    2.7325894832611084,
];

/// RMS multipliers the per-block scale search tries, in order.
pub const LM4_SCALE_GRID: [f32; 9] = [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.35];

/// IEEE bits of [`LM4_CODEBOOK`] (what a sidecar header records).
pub fn lm4_codebook_bits() -> [u32; 16] {
    LM4_CODEBOOK.map(f32::to_bits)
}

/// IEEE bits of [`LM4_SCALE_GRID`] (what a sidecar header records).
pub fn lm4_scale_grid_bits() -> [u32; 9] {
    LM4_SCALE_GRID.map(f32::to_bits)
}

/// Bytes of one `width`-value Q4LM row, or `None` unless `width` is a
/// positive whole number of 32-value blocks.
pub fn q4lm_row_bytes(width: usize) -> Option<usize> {
    (width > 0 && width % Q4LM_BLOCK_WEIGHTS == 0)
        .then(|| width / Q4LM_BLOCK_WEIGHTS * Q4LM_BLOCK_BYTES)
}

/// The 15 codebook midpoints, with the emulation's f32 expression.
#[inline]
fn codebook_midpoints() -> [f32; 15] {
    let mut mid = [0f32; 15];
    for (slot, pair) in mid.iter_mut().zip(LM4_CODEBOOK.windows(2)) {
        *slot = (pair[0] + pair[1]) / 2.0;
    }
    mid
}

/// Number of `mid` entries strictly below `t`: equals
/// `book.windows(2).filter(|w| (w[0] + w[1]) / 2.0 < t).count()` (NaN gives 0).
#[inline(always)]
fn label_of(mid: &[f32; 15], t: f32) -> u8 {
    let mut count = 0u8;
    for &m in mid {
        count += (m < t) as u8;
    }
    count
}

/// Encode one block of 32 decoded values; `None` when no ratio of the grid
/// has a finite error (see the module docs).
fn encode_block(values: &[f32; Q4LM_BLOCK_WEIGHTS], mid: &[f32; 15]) -> Option<(u16, [u8; 32])> {
    let sum: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    let rms = (sum / Q4LM_BLOCK_WEIGHTS as f64).sqrt() as f32;
    let mut best: Option<(u16, [u8; 32])> = None;
    let mut best_err = f64::INFINITY;
    for ratio in LM4_SCALE_GRID {
        let scale_half = half::f16::from_f32(rms * ratio);
        let scale = scale_half.to_f32();
        let safe = if scale == 0.0 { 1.0 } else { scale };
        let mut labels = [0u8; Q4LM_BLOCK_WEIGHTS];
        for (label, &v) in labels.iter_mut().zip(values.iter()) {
            *label = label_of(mid, v / safe);
        }
        let mut err = 0f64;
        for (&label, &v) in labels.iter().zip(values.iter()) {
            let y = LM4_CODEBOOK[usize::from(label)] * scale;
            let d = f64::from(y) - f64::from(v);
            err += d * d;
        }
        if err < best_err {
            best = Some((scale_half.to_bits(), labels));
            best_err = err;
        }
    }
    best
}

/// Encode one Q8F16 row (`width / 32` blocks of `[f16 scale][32 x i8]`,
/// `q8_row`) into a Q4LM row (`width / 32` blocks of 18 bytes, `out`).
///
/// The result decodes (see [`decode_q4lm_row_bf16`]) to exactly the BF16 row
/// the `lm4` emulation produces from the same Q8F16 row.  Errors on a length
/// mismatch, or on a block the emulation would leave unquantised (module docs).
pub fn encode_q4lm_row_from_q8(q8_row: &[u8], out: &mut [u8]) -> Result<(), String> {
    if q8_row.is_empty() || q8_row.len() % Q8_BLOCK_BYTES != 0 {
        return Err(format!(
            "Q8F16 row of {} bytes is not a positive multiple of {Q8_BLOCK_BYTES}",
            q8_row.len()
        ));
    }
    let blocks = q8_row.len() / Q8_BLOCK_BYTES;
    if out.len() != blocks * Q4LM_BLOCK_BYTES {
        return Err(format!(
            "Q4LM output of {} bytes does not fit {blocks} blocks ({} bytes)",
            out.len(),
            blocks * Q4LM_BLOCK_BYTES
        ));
    }
    let mid = codebook_midpoints();
    for (block, (src, dst)) in q8_row
        .chunks_exact(Q8_BLOCK_BYTES)
        .zip(out.chunks_exact_mut(Q4LM_BLOCK_BYTES))
        .enumerate()
    {
        let scale = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
        let mut values = [0f32; Q4LM_BLOCK_WEIGHTS];
        for (value, &q) in values.iter_mut().zip(&src[2..]) {
            *value = scale * (q as i8) as f32;
        }
        let (bits, labels) = encode_block(&values, &mid).ok_or_else(|| {
            format!(
                "block {block}: no scale of the grid gives a finite error (Q8F16 scale {scale}); \
                 the block cannot be represented as Q4LM"
            )
        })?;
        dst[..2].copy_from_slice(&bits.to_le_bytes());
        for (byte, pair) in dst[2..].iter_mut().zip(labels.chunks_exact(2)) {
            *byte = pair[0] | (pair[1] << 4);
        }
    }
    Ok(())
}

/// Decode one Q4LM row (`q4_row`, `width / 32` blocks of 18 bytes) into its
/// little-endian BF16 form (`out_bf16`, `2 * width` bytes).
pub fn decode_q4lm_row_bf16(q4_row: &[u8], out_bf16: &mut [u8]) -> Result<(), String> {
    if q4_row.is_empty() || q4_row.len() % Q4LM_BLOCK_BYTES != 0 {
        return Err(format!(
            "Q4LM row of {} bytes is not a positive multiple of {Q4LM_BLOCK_BYTES}",
            q4_row.len()
        ));
    }
    let blocks = q4_row.len() / Q4LM_BLOCK_BYTES;
    if out_bf16.len() != blocks * Q4LM_BLOCK_WEIGHTS * 2 {
        return Err(format!(
            "BF16 output of {} bytes does not fit {blocks} Q4LM blocks ({} bytes)",
            out_bf16.len(),
            blocks * Q4LM_BLOCK_WEIGHTS * 2
        ));
    }
    for (src, dst) in q4_row
        .chunks_exact(Q4LM_BLOCK_BYTES)
        .zip(out_bf16.chunks_exact_mut(Q4LM_BLOCK_WEIGHTS * 2))
    {
        let scale = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
        for (cells, &packed) in dst.chunks_exact_mut(4).zip(&src[2..]) {
            let lo = LM4_CODEBOOK[usize::from(packed & 0x0f)] * scale;
            let hi = LM4_CODEBOOK[usize::from(packed >> 4)] * scale;
            cells[..2].copy_from_slice(&f32_to_bf16_bits(lo).to_le_bytes());
            cells[2..].copy_from_slice(&f32_to_bf16_bits(hi).to_le_bytes());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- the emulation, copied verbatim from `233cb86ac` (the oracle) ------

    const Q8_BLOCK_WEIGHTS: usize = 32;
    const LM4: [f32; 16] = [
        -2.7325894832611084,
        -2.069017171859741,
        -1.6180464029312134,
        -1.2562311887741089,
        -0.9423404335975647,
        -0.6567591428756714,
        -0.38804829120635986,
        -0.12839503586292267,
        0.12839503586292267,
        0.38804829120635986,
        0.6567591428756714,
        0.9423404335975647,
        1.2562311887741089,
        1.6180464029312134,
        2.069017171859741,
        2.7325894832611084,
    ];
    const LM_GRID: [f32; 9] = [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25, 1.35];

    fn lut_requant_block(values: &mut [f32; Q8_BLOCK_WEIGHTS], book: &[f32]) {
        let sum: f64 = values.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let rms = (sum / Q8_BLOCK_WEIGHTS as f64).sqrt() as f32;
        let mut best = *values;
        let mut best_err = f64::INFINITY;
        for ratio in LM_GRID {
            let scale = half::f16::from_f32(rms * ratio).to_f32();
            let safe = if scale == 0.0 { 1.0 } else { scale };
            let mut y = [0f32; Q8_BLOCK_WEIGHTS];
            let mut err = 0f64;
            for (index, &v) in values.iter().enumerate() {
                let t = v / safe;
                // np.searchsorted(midpoints, t): count of midpoints strictly below t.
                let label = book.windows(2).filter(|w| (w[0] + w[1]) / 2.0 < t).count();
                y[index] = book[label] * scale;
                let d = f64::from(y[index]) - f64::from(v);
                err += d * d;
            }
            if err < best_err {
                best = y;
                best_err = err;
            }
        }
        *values = best;
    }

    /// The emulation's decode of one Q8F16 block: `(plain, requantised)` BF16
    /// bits of its 32 values (`Q8F16::decode_row` with the `lm4` book active).
    fn oracle_block(block: &[u8]) -> ([u16; 32], [u16; 32]) {
        assert_eq!(block.len(), Q8_BLOCK_BYTES);
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let mut values = [0f32; Q8_BLOCK_WEIGHTS];
        for index in 0..Q8_BLOCK_WEIGHTS {
            values[index] = scale * (block[2 + index] as i8) as f32;
        }
        let plain = values.map(f32_to_bf16_bits);
        lut_requant_block(&mut values, &LM4);
        (plain, values.map(f32_to_bf16_bits))
    }

    fn bf16_cells(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }

    // ---- deterministic generators ------------------------------------------

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        /// Uniform in [-1, 1).
        fn unit(&mut self) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        }
    }

    fn block_with(scale_bits: u16, mut ints: impl FnMut(usize) -> i8) -> [u8; Q8_BLOCK_BYTES] {
        let mut block = [0u8; Q8_BLOCK_BYTES];
        block[..2].copy_from_slice(&scale_bits.to_le_bytes());
        for index in 0..Q8_BLOCK_WEIGHTS {
            block[2 + index] = ints(index) as u8;
        }
        block
    }

    /// A random Q8F16 block covering the scale and integer regimes the real
    /// table and the edge cases exercise.
    fn random_block(rng: &mut Rng) -> [u8; Q8_BLOCK_BYTES] {
        let scale_bits: u16 = match rng.below(10) {
            // typical: small positive normals
            0..=3 => crate::llama::f32_to_f16(1e-4 + rng.unit().abs() * 0.05),
            // any normal-range positive scale
            4 => crate::llama::f32_to_f16(rng.unit().abs() * 4.0),
            // subnormal f16
            5 => 1 + rng.below(0x3ff) as u16,
            // negative scales
            6 => 0x8000 | crate::llama::f32_to_f16(rng.unit().abs() * 0.1),
            // zero scale (with nonzero ints below)
            7 => 0,
            // near the f16 maximum (65504) and large
            8 => 0x7bff - rng.below(64) as u16,
            // fully random finite bits
            _ => (rng.next() as u16) & 0x7bff | ((rng.next() as u16) & 0x8000),
        };
        match rng.below(8) {
            // all-zero ints
            0 => block_with(scale_bits, |_| 0),
            // +-127 extremes
            1 => {
                let r = rng.next();
                block_with(scale_bits, |i| if (r >> i) & 1 == 1 { 127 } else { -127 })
            }
            // a lone spike
            2 => {
                let at = rng.below(32) as usize;
                block_with(scale_bits, |i| if i == at { 127 } else { 0 })
            }
            // small ints
            3 => {
                let r = rng.next();
                block_with(scale_bits, |i| ((r >> (2 * (i % 32))) & 3) as i8 - 1)
            }
            // uniform full range, including -128
            4 => {
                let r = rng.next();
                let mut inner = Rng(r | 1);
                block_with(scale_bits, move |_| (inner.next() >> 56) as u8 as i8)
            }
            // roughly normal: sum of three uniforms
            _ => {
                let mut inner = Rng(rng.next() | 1);
                block_with(scale_bits, move |_| {
                    let v = (inner.unit() + inner.unit() + inner.unit()) * 38.0;
                    v.round().clamp(-127.0, 127.0) as i8
                })
            }
        }
    }

    // ---- tests --------------------------------------------------------------

    #[test]
    fn midpoint_label_count_matches_the_windows_filter() {
        let mid = codebook_midpoints();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut probes = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        for pair in LM4.windows(2) {
            let m = (pair[0] + pair[1]) / 2.0;
            probes.extend([
                m,
                f32::from_bits(m.to_bits().wrapping_add(1)),
                f32::from_bits(m.to_bits().wrapping_sub(1)),
            ]);
        }
        for _ in 0..200_000 {
            probes.push(rng.unit() * 4.0);
        }
        for t in probes {
            let want = LM4.windows(2).filter(|w| (w[0] + w[1]) / 2.0 < t).count();
            assert_eq!(usize::from(label_of(&mid, t)), want, "t = {t:e}");
        }
        assert_eq!(label_of(&mid, f32::NAN), 0);
    }

    #[test]
    fn constants_match_the_emulation_and_headers() {
        assert_eq!(LM4_CODEBOOK, LM4);
        assert_eq!(LM4_SCALE_GRID, LM_GRID);
        assert_eq!(lm4_codebook_bits(), LM4.map(f32::to_bits));
        assert_eq!(lm4_scale_grid_bits(), LM_GRID.map(f32::to_bits));
        assert_eq!(q4lm_row_bytes(160), Some(90));
        assert_eq!(q4lm_row_bytes(32), Some(18));
        assert_eq!(q4lm_row_bytes(0), None);
        assert_eq!(q4lm_row_bytes(33), None);
    }

    /// Check one Q8F16 block (as a one-block row) against the oracle.  Returns
    /// whether the block was representable.
    fn check_block(block: &[u8; Q8_BLOCK_BYTES]) -> bool {
        let (plain, want) = oracle_block(block);
        let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
        match encode_q4lm_row_from_q8(block, &mut q4) {
            Ok(()) => {
                let mut out = [0u8; 64];
                decode_q4lm_row_bf16(&q4, &mut out).unwrap();
                assert_eq!(bf16_cells(&out), want, "block {block:02x?}");
                true
            }
            Err(error) => {
                // The emulation keeps the plain Q8F16 values for a block no
                // ratio can quantise; Q4LM refuses it.
                assert_eq!(plain, want, "block {block:02x?}: {error}");
                false
            }
        }
    }

    #[test]
    fn decode_of_encode_equals_the_emulation_bit_for_bit() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut ok, mut refused) = (0usize, 0usize);
        // Edge blocks first.
        let mut edge: Vec<[u8; Q8_BLOCK_BYTES]> = Vec::new();
        let f16_max = 0x7bffu16;
        for scale in [
            0u16, 0x8000, 1, 0x3ff, 0x0400, 0x8001, 0x83ff, 0x3c00, 0xbc00, f16_max, 0xfbff, 0x5640,
        ] {
            edge.push(block_with(scale, |_| 0));
            edge.push(block_with(scale, |_| 127));
            edge.push(block_with(scale, |_| -127));
            edge.push(block_with(scale, |_| -128));
            edge.push(block_with(scale, |i| if i % 2 == 0 { 127 } else { -127 }));
            edge.push(block_with(scale, |i| i as i8 - 16));
            edge.push(block_with(scale, |i| if i == 31 { 1 } else { 0 }));
        }
        for block in &edge {
            if check_block(block) {
                ok += 1;
            } else {
                refused += 1;
            }
        }
        for _ in 0..60_000 {
            let block = random_block(&mut rng);
            if check_block(&block) {
                ok += 1;
            } else {
                refused += 1;
            }
        }
        assert!(ok > 40_000, "only {ok} representable blocks");
        // The huge-scale edge blocks are the unrepresentable ones.
        assert!(refused > 0, "no unrepresentable block was exercised");
    }

    #[test]
    fn multi_block_rows_encode_each_block_independently() {
        let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
        let mut rows = 0;
        while rows < 3000 {
            let mut q8 = Vec::new();
            for _ in 0..5 {
                q8.extend_from_slice(&random_block(&mut rng));
            }
            let mut q4 = vec![0u8; 90];
            if encode_q4lm_row_from_q8(&q8, &mut q4).is_err() {
                continue;
            }
            rows += 1;
            let mut out = vec![0u8; 320];
            decode_q4lm_row_bf16(&q4, &mut out).unwrap();
            let want: Vec<u16> = q8
                .chunks_exact(Q8_BLOCK_BYTES)
                .flat_map(|block| oracle_block(block).1)
                .collect();
            assert_eq!(bf16_cells(&out), want);
        }
    }

    #[test]
    fn non_finite_scales_are_refused_not_encoded() {
        for scale in [0x7c00u16, 0xfc00, 0x7e00] {
            let block = block_with(scale, |i| i as i8);
            let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
            assert!(
                encode_q4lm_row_from_q8(&block, &mut q4).is_err(),
                "{scale:#x}"
            );
            assert!(!check_block(&block));
        }
        // rms * 0.65 > 65504: every grid scale overflows f16.
        let block = block_with(0x7bff, |_| 127);
        assert!(!check_block(&block));
    }

    #[test]
    fn layout_is_scale_then_low_nibble_first_labels() {
        let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
        q4[..2].copy_from_slice(&crate::llama::f32_to_f16(0.5).to_le_bytes());
        for i in 0..16 {
            q4[2 + i] = ((2 * i) % 16) as u8 | ((((2 * i + 1) % 16) as u8) << 4);
        }
        let mut out = [0u8; 64];
        decode_q4lm_row_bf16(&q4, &mut out).unwrap();
        for (j, cell) in bf16_cells(&out).into_iter().enumerate() {
            assert_eq!(cell, f32_to_bf16_bits(LM4[j % 16] * 0.5), "value {j}");
        }
    }

    #[test]
    fn row_length_mismatches_are_errors() {
        let q8 = [0u8; Q8_BLOCK_BYTES];
        let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
        assert!(encode_q4lm_row_from_q8(&[], &mut q4).is_err());
        assert!(encode_q4lm_row_from_q8(&q8[..33], &mut q4).is_err());
        assert!(encode_q4lm_row_from_q8(&q8, &mut q4[..17]).is_err());
        let mut out = [0u8; 64];
        assert!(decode_q4lm_row_bf16(&[], &mut out).is_err());
        assert!(decode_q4lm_row_bf16(&q4[..17], &mut out).is_err());
        assert!(decode_q4lm_row_bf16(&q4, &mut out[..62]).is_err());
        assert!(decode_q4lm_row_bf16(&q4, &mut [0u8; 66]).is_err());
        // An all-zero block is representable and decodes to (signed) zeros.
        encode_q4lm_row_from_q8(&q8, &mut q4).unwrap();
        decode_q4lm_row_bf16(&q4, &mut out).unwrap();
        assert!(bf16_cells(&out).iter().all(|&c| c & 0x7fff == 0));
    }

    // ---- property tests -----------------------------------------------------
    //
    // Same semantics as `check_block` / `decode_of_encode_equals_the_emulation_bit_for_bit`:
    // an encoded block must decode to the emulation's requantised BF16 bits; a
    // refused block must be one the emulation leaves at its plain Q8F16 values.

    mod props {
        use super::*;
        use proptest::prelude::*;

        /// Q8F16 scale bit patterns: the edge values the deterministic test
        /// uses, typical small normals, subnormals, negatives, near-max, and
        /// arbitrary finite bits.
        fn scale_bits() -> impl Strategy<Value = u16> {
            prop_oneof![
                3 => Just(0u16).boxed(),
                3 => prop::sample::select(vec![
                    0x8000u16, 1, 0x3ff, 0x0400, 0x8001, 0x83ff, 0x3c00, 0xbc00, 0x7bff,
                    0xfbff, 0x5640,
                ]).boxed(),
                // typical: small positive normals
                8 => (1e-4f32..0.05f32).prop_map(crate::llama::f32_to_f16).boxed(),
                // any normal-range positive scale
                2 => (0.0f32..4.0f32).prop_map(crate::llama::f32_to_f16).boxed(),
                // subnormal f16
                2 => (1u16..0x400).boxed(),
                // negative scales
                2 => (0.0f32..0.1f32).prop_map(|v| 0x8000 | crate::llama::f32_to_f16(v)).boxed(),
                // near the f16 maximum
                2 => (0x7bc0u16..=0x7bff).boxed(),
                // fully random finite bits
                4 => (any::<u16>()).prop_map(|b| b & 0xfbff).boxed(),
            ]
        }

        /// 32 Q8 integers: all-zero, +-127 extremes, lone spike, small, uniform
        /// full range (incl. -128), and roughly-normal.
        fn ints() -> impl Strategy<Value = Vec<i8>> {
            prop_oneof![
                1 => Just(vec![0i8; 32]).boxed(),
                1 => prop::collection::vec(prop::bool::ANY.prop_map(|b| if b { 127i8 } else { -127 }), 32).boxed(),
                1 => (0usize..32).prop_map(|at| (0..32).map(|i| if i == at { 127i8 } else { 0 }).collect()).boxed(),
                1 => prop::collection::vec(-1i8..=2, 32).boxed(),
                2 => prop::collection::vec(any::<i8>(), 32).boxed(),
                2 => prop::collection::vec(-127i8..=127, 32)
                    .prop_map(|v| v.into_iter().map(|x| (x / 3).clamp(-127, 127)).collect())
                    .boxed(),
            ]
        }

        fn block() -> impl Strategy<Value = [u8; Q8_BLOCK_BYTES]> {
            (scale_bits(), ints()).prop_map(|(scale, ints)| block_with(scale, |i| ints[i]))
        }

        /// 1..=40 blocks; 5 blocks is the real 160-wide PLE row, 32 blocks a
        /// 1024-wide row.
        fn row() -> impl Strategy<Value = Vec<[u8; Q8_BLOCK_BYTES]>> {
            prop_oneof![
                3 => Just(5usize).boxed(),
                1 => Just(32usize).boxed(),
                4 => (1usize..=40).boxed(),
            ]
            .prop_flat_map(|n| prop::collection::vec(block(), n))
        }

        /// Sorted, deduplicated interior cut points in `1..len`, plus both ends.
        fn boundaries(raw: &[usize], len: usize) -> Vec<usize> {
            let mut b: Vec<usize> = raw
                .iter()
                .filter(|_| len > 1)
                .map(|c| 1 + c % (len - 1))
                .collect();
            b.push(0);
            b.push(len);
            b.sort_unstable();
            b.dedup();
            b
        }

        fn flat(blocks: &[[u8; Q8_BLOCK_BYTES]]) -> Vec<u8> {
            blocks.iter().flatten().copied().collect()
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(256))]

            /// (1) + (3): per block, encode succeeds => decode(encode) is the
            /// emulation's requantised BF16 bits; encode refuses => the emulation
            /// left the block at its plain Q8F16 values. A row is refused iff some
            /// block is, and a row that encodes decodes to the emulation row.
            #[test]
            fn prop_decode_of_encode_equals_the_emulation(blocks in row()) {
                let q8 = flat(&blocks);
                let width = blocks.len() * Q8_BLOCK_WEIGHTS;
                let mut any_refused = false;
                for (index, b) in blocks.iter().enumerate() {
                    let (plain, want) = oracle_block(b);
                    let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
                    match encode_q4lm_row_from_q8(b, &mut q4) {
                        Ok(()) => {
                            let mut out = [0u8; 64];
                            decode_q4lm_row_bf16(&q4, &mut out).unwrap();
                            prop_assert_eq!(bf16_cells(&out), want.to_vec(), "block {} {:02x?}", index, b);
                        }
                        Err(error) => {
                            any_refused = true;
                            prop_assert_eq!(plain, want, "block {} {:02x?}: {}", index, b, error);
                        }
                    }
                }
                let mut q4 = vec![0u8; q4lm_row_bytes(width).unwrap()];
                let encoded = encode_q4lm_row_from_q8(&q8, &mut q4);
                prop_assert_eq!(encoded.is_err(), any_refused, "row refusal must equal any-block refusal");
                if encoded.is_ok() {
                    let mut out = vec![0u8; width * 2];
                    decode_q4lm_row_bf16(&q4, &mut out).unwrap();
                    let want: Vec<u16> = blocks.iter().flat_map(|b| oracle_block(b).1).collect();
                    prop_assert_eq!(bf16_cells(&out), want);
                }
            }

            /// (2) Row partition invariance: encoding a row equals the
            /// concatenation of encoding each block independently, and equals the
            /// concatenation of encoding random contiguous chunks; decode is
            /// likewise chunk-invariant.
            #[test]
            fn prop_row_encode_is_partition_invariant(
                blocks in row(),
                cuts in prop::collection::vec(any::<usize>(), 0..5),
            ) {
                let q8 = flat(&blocks);
                let n = blocks.len();
                let mut full = vec![0u8; n * Q4LM_BLOCK_BYTES];
                let full_res = encode_q4lm_row_from_q8(&q8, &mut full);

                let mut per_block = Vec::with_capacity(full.len());
                let mut all_ok = true;
                for b in &blocks {
                    let mut q4 = [0u8; Q4LM_BLOCK_BYTES];
                    match encode_q4lm_row_from_q8(b, &mut q4) {
                        Ok(()) => per_block.extend_from_slice(&q4),
                        Err(_) => all_ok = false,
                    }
                }
                prop_assert_eq!(full_res.is_ok(), all_ok);

                let bounds = boundaries(&cuts, n);
                let mut chunked = Vec::with_capacity(full.len());
                let mut chunks_ok = true;
                for w in bounds.windows(2) {
                    let (lo, hi) = (w[0], w[1]);
                    let mut part = vec![0u8; (hi - lo) * Q4LM_BLOCK_BYTES];
                    match encode_q4lm_row_from_q8(&q8[lo * Q8_BLOCK_BYTES..hi * Q8_BLOCK_BYTES], &mut part) {
                        Ok(()) => chunked.extend_from_slice(&part),
                        Err(_) => chunks_ok = false,
                    }
                }
                prop_assert_eq!(full_res.is_ok(), chunks_ok, "partition {:?}", bounds);

                if full_res.is_ok() {
                    prop_assert_eq!(&full, &per_block);
                    prop_assert_eq!(&full, &chunked, "partition {:?}", bounds);
                    // Decode partition invariance.
                    let mut whole = vec![0u8; n * Q4LM_BLOCK_WEIGHTS * 2];
                    decode_q4lm_row_bf16(&full, &mut whole).unwrap();
                    let mut stitched = Vec::with_capacity(whole.len());
                    for w in bounds.windows(2) {
                        let (lo, hi) = (w[0], w[1]);
                        let mut part = vec![0u8; (hi - lo) * Q4LM_BLOCK_WEIGHTS * 2];
                        decode_q4lm_row_bf16(&full[lo * Q4LM_BLOCK_BYTES..hi * Q4LM_BLOCK_BYTES], &mut part).unwrap();
                        stitched.extend_from_slice(&part);
                    }
                    prop_assert_eq!(whole, stitched, "decode partition {:?}", bounds);
                }
            }
        }
    }

    /// Real rows against the oracle.
    ///
    ///   HIPFIRE_PROBE_MODEL=/path/to/model.mq4-xts [HIPFIRE_PROBE_ROWS=100000] \
    ///     cargo test -p hipfire-runtime --lib -- ple_q4::tests::real_rows_match --ignored --nocapture
    ///
    /// Reads the first `HIPFIRE_PROBE_ROWS` rows (default 100000) of PLE shard
    /// 0 of an HFQM model with embedded Q8F16 PLE rows.
    #[test]
    #[ignore = "needs HIPFIRE_PROBE_MODEL: an HFQM with embedded Q8F16 PLE shards"]
    fn real_rows_match_the_emulation() {
        use std::os::unix::fs::FileExt;
        let path = std::env::var("HIPFIRE_PROBE_MODEL").expect("HIPFIRE_PROBE_MODEL");
        let want_rows: u64 = std::env::var("HIPFIRE_PROBE_ROWS")
            .ok()
            .map(|v| v.parse().expect("HIPFIRE_PROBE_ROWS"))
            .unwrap_or(100_000);
        let hfq = crate::hfq::HfqFile::open(std::path::Path::new(&path)).expect("open model");
        let name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight";
        let range = hfq
            .tensor_range(name)
            .expect("tensor range")
            .unwrap_or_else(|| panic!("{name} is not in {path}"));
        assert_eq!(range.dtype(), "Q8_0");
        let width = range.logical_shape()[1];
        let row_q8 = width / 32 * Q8_BLOCK_BYTES;
        let row_q4 = q4lm_row_bytes(width).expect("row width");
        let rows = want_rows.min(range.length / row_q8 as u64);
        let file = std::fs::File::open(&path).unwrap();
        let mut q8 = vec![0u8; row_q8];
        let mut q4 = vec![0u8; row_q4];
        let mut out = vec![0u8; width * 2];
        let mut refused = 0u64;
        for row in 0..rows {
            file.read_exact_at(&mut q8, range.offset + row * row_q8 as u64)
                .unwrap();
            if encode_q4lm_row_from_q8(&q8, &mut q4).is_err() {
                refused += 1;
                continue;
            }
            decode_q4lm_row_bf16(&q4, &mut out).unwrap();
            let want: Vec<u16> = q8
                .chunks_exact(Q8_BLOCK_BYTES)
                .flat_map(|block| oracle_block(block).1)
                .collect();
            assert_eq!(bf16_cells(&out), want, "row {row}");
        }
        eprintln!("ple_q4 probe: {rows} rows of {path} match the emulation ({refused} refused)");
    }
}
