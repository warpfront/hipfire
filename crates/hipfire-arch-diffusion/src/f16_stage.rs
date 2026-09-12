// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-side f16 staging for the streaming weight upload (the host memory
//! diet).
//!
//! The old upload path decoded a whole checkpoint to f32 host `Vec`s
//! (transformer ~47 GB, T5-XXL ~18.5 GB), uploaded each tensor f32, and cast
//! it to f16 **on the device**. That is three copies of every weight — the
//! host f32 table, the transient f32 device scratch, and the f16 result — for
//! a model that only ever needs the last one. On a 128 GB unified-memory box
//! with other services resident it pushed the host into zram swap, and since
//! "VRAM" is system RAM on an iGPU the GPU allocations paged too: the VAE
//! decode went from 3.96 s to 359 s.
//!
//! This module is the replacement: convert ONE tensor at a time out of the
//! mmapped checkpoint bytes into a reusable [`F16Stage`] buffer, upload it
//! straight into an `F16` device tensor, and move on. Peak host cost is the
//! largest single tensor (FLUX's `single_blocks.*.linear1.weight`, 21504×3072
//! → 132 MB of f16 words), not the model.
//!
//! **Bit-exactness with the device cast it replaces.** The device did
//! `(_Float16)x`, i.e. IEEE round-to-nearest-even (`kernels/src/
//! cast_f32_to_f16.hip`). [`f32_to_f16_rne`] implements the same rounding, so
//! a streamed weight is bit-identical to the same weight uploaded f32 and cast
//! on the GPU, and the block-parity gates see no change. This is deliberately
//! NOT `hipfire_runtime::llama::f32_to_f16`, which truncates — that function
//! encodes HFQ bytes, where the historical truncation is load-bearing.
//!
//! BF16 → f16 goes through f32 rather than by direct field surgery, because
//! that is exactly what the path it replaces did: `decode_dtype` widened the
//! bf16 bits into f32 (an exact shift — bf16 is the high half of an f32) and
//! the device rounded f32 → f16. Composing the two is bit-identical and
//! obviously so. BF16 carries 7 mantissa bits against f16's 10, so the
//! mantissa is exact and only the exponent range can lose: |x| > 65504
//! saturates to inf and |x| < 2^-24 flushes to zero. FLUX/T5 weights live far
//! inside that window (max |w| is order 1), so neither happens in practice —
//! but the conversion is defined for both rather than silently wrong.

use rayon::prelude::*;

/// f32 → f16 bits with IEEE round-to-nearest-even, matching the device
/// `(_Float16)` conversion in `cast_f32_to_f16.hip`.
///
/// Overflow (including a value that rounds up past the largest half, 65504)
/// yields ±inf; a value below half the smallest subnormal flushes to ±0.
/// NaN yields a quiet NaN with the sign preserved — the payload is not,
/// which no weight path depends on.
#[inline]
pub fn f32_to_f16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mant = bits & 0x007F_FFFF;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;

    if exp == 128 {
        // Inf / NaN.
        return sign | if mant != 0 { 0x7E00 } else { 0x7C00 };
    }
    if exp >= 16 {
        return sign | 0x7C00; // magnitude >= 2^16 — no half can hold it
    }
    if exp >= -14 {
        // Normal half. Keep 10 mantissa bits; round the 13 dropped ones.
        // The `+ round` is allowed to carry into the exponent field: that is
        // how 0x7BFF (65504) becomes inf, and how the largest subnormal
        // becomes the smallest normal below.
        let half = (((exp + 15) as u32) << 10) | (mant >> 13);
        let rem = mant & 0x1FFF;
        let up = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
        return sign | (half + up as u32) as u16;
    }
    if exp < -25 {
        return sign; // below 2^-25: rounds to zero either way
    }
    // Subnormal half: restore f32's implicit leading 1, then drop
    // `13 + shift` bits with the same round-to-nearest-even rule.
    let full = mant | 0x0080_0000;
    let drop = 13 + (-exp - 14) as u32; // 14 ..= 24
    let sub = full >> drop;
    let rem = full & ((1u32 << drop) - 1);
    let halfway = 1u32 << (drop - 1);
    let up = rem > halfway || (rem == halfway && sub & 1 == 1);
    sign | (sub + up as u32) as u16
}

/// A reusable host buffer that converts safetensors tensor bytes to f16 words.
///
/// One instance is threaded through a whole model upload, so the allocation
/// grows to the largest tensor and is then reused — the point of the type is
/// that nothing model-sized is ever live.
#[derive(Debug, Default)]
pub struct F16Stage {
    buf: Vec<u16>,
}

impl F16Stage {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Drop the staged words, keeping the allocation for the next tensor.
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// The words staged since the last [`clear`](Self::clear).
    pub fn words(&self) -> &[u16] {
        &self.buf
    }

    /// Bytes currently allocated by the staging buffer (for RSS accounting).
    pub fn capacity_bytes(&self) -> usize {
        self.buf.capacity() * 2
    }

    /// Convert one safetensors tensor and APPEND it to the staged words.
    ///
    /// Appending rather than replacing is what makes a row-concatenated key
    /// (diffusers splits FLUX's fused `qkv` / `linear1` into three or four
    /// tensors) a single staged upload: push each part in order, then upload
    /// [`words`](Self::words) once.
    ///
    /// Returns the number of words appended.
    pub fn push(&mut self, dtype: &str, bytes: &[u8]) -> Result<usize, String> {
        let elem = match dtype {
            "F32" => 4usize,
            "BF16" | "F16" => 2,
            other => {
                return Err(format!(
                    "unsupported tensor dtype `{other}` (F32/BF16/F16 only)"
                ))
            }
        };
        if bytes.len() % elem != 0 {
            return Err(format!(
                "{dtype} tensor has {} bytes, not a multiple of {elem}",
                bytes.len()
            ));
        }
        let n = bytes.len() / elem;
        let start = self.buf.len();
        self.buf.resize(start + n, 0);
        let dst = &mut self.buf[start..];
        match dtype {
            // F16 → F16 is a byte copy. The path this replaces widened to f32
            // and let the device round back, which is the identity for every
            // finite half (widening is exact, so RNE returns the same word);
            // only a NaN payload could differ, and no weight is NaN.
            "F16" => dst
                .par_iter_mut()
                .zip(bytes.par_chunks_exact(2))
                .for_each(|(o, c)| *o = u16::from_le_bytes([c[0], c[1]])),
            "BF16" => dst
                .par_iter_mut()
                .zip(bytes.par_chunks_exact(2))
                .for_each(|(o, c)| {
                    let wide = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                    *o = f32_to_f16_rne(wide);
                }),
            _ => dst
                .par_iter_mut()
                .zip(bytes.par_chunks_exact(4))
                .for_each(|(o, c)| {
                    *o = f32_to_f16_rne(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }),
        }
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// `half::f16::from_f32` is the reference RNE conversion; the device
    /// `(_Float16)` cast rounds the same way.
    fn want(v: f32) -> u16 {
        f16::from_f32(v).to_bits()
    }

    #[test]
    fn rne_matches_reference_for_every_bf16_pattern() {
        // 65 536 patterns sweep the whole f32 exponent range with a 7-bit
        // mantissa: normals, subnormal-half territory, overflow, ±0, ±inf.
        for bits in 0u16..=u16::MAX {
            let v = f32::from_bits((bits as u32) << 16);
            if v.is_nan() {
                assert!(
                    f16::from_bits(f32_to_f16_rne(v)).is_nan(),
                    "bf16 {bits:#06x}"
                );
                continue;
            }
            assert_eq!(f32_to_f16_rne(v), want(v), "bf16 {bits:#06x} = {v:e}");
        }
    }

    #[test]
    fn rne_round_trips_every_finite_half() {
        for bits in 0u16..=u16::MAX {
            let h = f16::from_bits(bits);
            if h.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16_rne(h.to_f32()), bits, "half {bits:#06x}");
        }
    }

    #[test]
    fn rne_matches_reference_around_every_rounding_boundary() {
        // For each half, probe the f32 neighbourhood of the tie point between
        // it and its successor — the only place a rounding rule can differ.
        for bits in 0u16..0x7C00u16 {
            let lo = f16::from_bits(bits).to_f32();
            let hi = f16::from_bits(bits + 1).to_f32();
            let mid = 0.5f32 * (lo + hi);
            for v in [
                mid,
                f32::from_bits(mid.to_bits() - 1),
                f32::from_bits(mid.to_bits() + 1),
            ] {
                assert_eq!(f32_to_f16_rne(v), want(v), "near half {bits:#06x}: {v:e}");
                let n = -v;
                assert_eq!(f32_to_f16_rne(n), want(n), "near half -{bits:#06x}: {n:e}");
            }
        }
    }

    #[test]
    fn rne_matches_reference_on_a_pseudo_random_sweep() {
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = f32::from_bits((x >> 32) as u32);
            if v.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16_rne(v), want(v), "{v:e}");
        }
    }

    #[test]
    fn saturation_and_flush_are_defined() {
        assert_eq!(f32_to_f16_rne(65504.0), 0x7BFF); // largest half
        assert_eq!(f32_to_f16_rne(65520.0), 0x7C00); // ties-to-even -> inf
        assert_eq!(f32_to_f16_rne(-65520.0), 0xFC00);
        assert_eq!(f32_to_f16_rne(1e30), 0x7C00);
        assert_eq!(f32_to_f16_rne(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16_rne(-0.0), 0x8000);
        assert_eq!(f32_to_f16_rne(2f32.powi(-24)), 0x0001); // min subnormal
        assert_eq!(f32_to_f16_rne(2f32.powi(-25)), 0x0000); // exact tie -> even
        assert_eq!(f32_to_f16_rne(-2f32.powi(-26)), 0x8000);
    }

    #[test]
    fn push_appends_each_dtype_and_reuses_the_buffer() {
        let vals = [1.0f32, -2.5, 0.0, 6.1e-5, 1234.5];
        let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16_bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let f16_bytes: Vec<u8> = vals
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
            .collect();

        let mut stage = F16Stage::new();
        assert_eq!(stage.push("F32", &f32_bytes).unwrap(), 5);
        assert_eq!(stage.push("BF16", &bf16_bytes).unwrap(), 5);
        assert_eq!(stage.push("F16", &f16_bytes).unwrap(), 5);
        assert_eq!(stage.words().len(), 15);
        for (i, v) in vals.iter().enumerate() {
            assert_eq!(stage.words()[i], want(*v), "f32 slot {i}");
            let wide = f32::from_bits(v.to_bits() & 0xFFFF_0000);
            assert_eq!(stage.words()[5 + i], want(wide), "bf16 slot {i}");
            assert_eq!(stage.words()[10 + i], want(*v), "f16 slot {i}");
        }
        let cap = stage.capacity_bytes();
        stage.clear();
        assert!(stage.words().is_empty());
        assert_eq!(
            stage.capacity_bytes(),
            cap,
            "clear must keep the allocation"
        );
    }

    #[test]
    fn push_rejects_a_dtype_the_gpu_path_cannot_take() {
        let mut stage = F16Stage::new();
        let err = stage.push("F64", &[0u8; 8]).unwrap_err();
        assert!(err.contains("F64"), "{err}");
        let err = stage.push("BF16", &[0u8; 3]).unwrap_err();
        assert!(err.contains("multiple of 2"), "{err}");
    }
}
