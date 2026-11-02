// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ3-I4 / MQ2-I4 planar tensor codec (HFQ qt=131..=134, DESIGN F1/F2).
//!
//! CPU-only, GPU-free. This module is the single home of the wire format for
//! the sub-4-bit Flash-Next expert tensors; packer, loader and CPU references
//! import it and never fork it.
//!
//! # Tensor bytes (`[M, K]`, `K % 128 == 0`, `K % R == 0`)
//!
//! ```text
//! [codes plane: M*K*b/8 bytes][scales plane: M*(K/128) little-endian f16]
//! ```
//!
//! * Codes are a row-major little-endian bitstream: within a row, code `j`
//!   occupies bits `[b*j, b*(j+1))` of the row's bitstream (bit 0 = LSB of the
//!   row's first byte). Row `r` starts at byte `r*K*b/8`. W3 packs eight codes
//!   in three bytes, W2 four codes in one byte. Because `K % 128 == 0` every
//!   row is byte aligned, so rows never share a byte.
//! * The scale for row `r`, group `g` (128 consecutive columns) is the LE f16
//!   at `codes_bytes + 2*(r*(K/128) + g)`.
//! * Dequantised value in the rotated basis: `float(d[r,g]) * (2*c - L)` with
//!   `L = 2^b - 1`.
//! * Import validation: u8 codes `> L` are rejected; scales must be finite
//!   and non-negative (sign bit clear, so `-0.0` is rejected); a group whose
//!   scale is `+0` must hold only the midpoint code `2^(b-1)`; the byte length
//!   must be exact.
//!
//! The rotation width `R` (256 or 128) is a property of the profile and is
//! never inferred from `K`: an `R128` down matrix with `K % 256 == 0` is legal
//! and stays `R128`.

use std::fmt;

/// Scale group width in columns (all profiles).
pub const GROUP: usize = 128;

/// Code width / grid of a MQ-I4 tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MqI4Codec {
    /// 3-bit codes, grid `±{1,3,5,7}` (`2c - 7`).
    W3,
    /// 2-bit codes, grid `±{1,3}` (`2c - 3`).
    W2,
}

impl MqI4Codec {
    /// Bits per code.
    pub const fn bits(self) -> usize {
        match self {
            Self::W3 => 3,
            Self::W2 => 2,
        }
    }

    /// `L = 2^b - 1`, the largest legal code.
    pub const fn levels(self) -> u8 {
        match self {
            Self::W3 => 7,
            Self::W2 => 3,
        }
    }

    /// Canonical code of an all-zero group: `2^(b-1)`.
    pub const fn midpoint(self) -> u8 {
        match self {
            Self::W3 => 4,
            Self::W2 => 2,
        }
    }
}

/// Activation/weight FWHT rotation width of a MQ-I4 profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MqI4Rotation {
    /// 256-wide FWHT (gate/up), sign seeds 42 / 1042.
    R256,
    /// 128-wide row-local FWHT (down), sign seeds 43 / 1043.
    R128,
}

impl MqI4Rotation {
    /// Rotation width in columns.
    pub const fn width(self) -> usize {
        match self {
            Self::R256 => 256,
            Self::R128 => 128,
        }
    }

    /// `(sign seed, permutation seed)` handed to `gen_fwht_signs`.
    pub const fn seeds(self) -> (u32, u32) {
        match self {
            Self::R256 => (42, 1042),
            Self::R128 => (43, 1043),
        }
    }
}

/// One of the four wire types qt=131..=134.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MqI4Profile {
    pub codec: MqI4Codec,
    pub rotation: MqI4Rotation,
}

impl MqI4Profile {
    /// qt=131 `MQ3I4G128R256` (FN gate_up, W3).
    pub const MQ3_R256: Self = Self {
        codec: MqI4Codec::W3,
        rotation: MqI4Rotation::R256,
    };
    /// qt=132 `MQ3I4G128R128` (FN down, W3).
    pub const MQ3_R128: Self = Self {
        codec: MqI4Codec::W3,
        rotation: MqI4Rotation::R128,
    };
    /// qt=133 `MQ2I4G128R256` (FN gate_up, W2).
    pub const MQ2_R256: Self = Self {
        codec: MqI4Codec::W2,
        rotation: MqI4Rotation::R256,
    };
    /// qt=134 `MQ2I4G128R128` (FN down, W2).
    pub const MQ2_R128: Self = Self {
        codec: MqI4Codec::W2,
        rotation: MqI4Rotation::R128,
    };
    /// All four profiles in qt order (131, 132, 133, 134).
    pub const ALL: [Self; 4] = [
        Self::MQ3_R256,
        Self::MQ3_R128,
        Self::MQ2_R256,
        Self::MQ2_R128,
    ];

    /// HFQ `quant_type` byte → profile; `None` for every other byte.
    pub const fn from_quant_type(qt: u8) -> Option<Self> {
        match qt {
            131 => Some(Self::MQ3_R256),
            132 => Some(Self::MQ3_R128),
            133 => Some(Self::MQ2_R256),
            134 => Some(Self::MQ2_R128),
            _ => None,
        }
    }

    /// HFQ `quant_type` byte of this profile.
    pub const fn quant_type(self) -> u8 {
        match (self.codec, self.rotation) {
            (MqI4Codec::W3, MqI4Rotation::R256) => 131,
            (MqI4Codec::W3, MqI4Rotation::R128) => 132,
            (MqI4Codec::W2, MqI4Rotation::R256) => 133,
            (MqI4Codec::W2, MqI4Rotation::R128) => 134,
        }
    }

    /// Canonical name, identical in every registry (HFQ header, `DType`).
    pub const fn name(self) -> &'static str {
        match (self.codec, self.rotation) {
            (MqI4Codec::W3, MqI4Rotation::R256) => "MQ3I4G128R256",
            (MqI4Codec::W3, MqI4Rotation::R128) => "MQ3I4G128R128",
            (MqI4Codec::W2, MqI4Rotation::R256) => "MQ2I4G128R256",
            (MqI4Codec::W2, MqI4Rotation::R128) => "MQ2I4G128R128",
        }
    }
}

/// Typed MQ-I4 codec failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MqI4Error {
    /// `M` or `K` is zero.
    ZeroDimension { m: usize, k: usize },
    /// `K` is not a multiple of the 128-column scale group.
    KNotGroupAligned { k: usize },
    /// `K` is not a multiple of the profile's rotation width.
    KNotRotationAligned { k: usize, rotation_width: usize },
    /// A size computation overflowed `usize`.
    SizeOverflow { m: usize, k: usize },
    /// A buffer has the wrong length.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A code exceeds `L = 2^b - 1`.
    CodeOutOfRange {
        row: usize,
        col: usize,
        code: u8,
        max: u8,
    },
    /// A scale is NaN or infinite.
    ScaleNotFinite { index: usize, bits: u16 },
    /// A scale has its sign bit set (negative values and `-0.0`).
    ScaleNegative { index: usize, bits: u16 },
    /// A group with scale `+0` holds a non-midpoint code.
    ZeroScaleNonMidpoint {
        row: usize,
        group: usize,
        col: usize,
        code: u8,
        midpoint: u8,
    },
    /// An f32 scale is not exactly representable as a finite non-negative f16.
    ScaleNotExactF16 { bits: u32 },
}

impl fmt::Display for MqI4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroDimension { m, k } => {
                write!(f, "MQ-I4 tensor has a zero dimension: M={m} K={k}")
            }
            Self::KNotGroupAligned { k } => {
                write!(f, "MQ-I4 requires K % {GROUP} == 0, got K={k}")
            }
            Self::KNotRotationAligned { k, rotation_width } => write!(
                f,
                "MQ-I4 profile requires K % {rotation_width} == 0 (rotation width), got K={k}"
            ),
            Self::SizeOverflow { m, k } => {
                write!(f, "MQ-I4 byte size overflows usize: M={m} K={k}")
            }
            Self::LengthMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "MQ-I4 {what} length mismatch: expected {expected}, got {got}"
            ),
            Self::CodeOutOfRange {
                row,
                col,
                code,
                max,
            } => write!(
                f,
                "MQ-I4 code {code} at row {row} col {col} exceeds the grid maximum {max}"
            ),
            Self::ScaleNotFinite { index, bits } => {
                write!(
                    f,
                    "MQ-I4 scale #{index} (f16 bits {bits:#06x}) is not finite"
                )
            }
            Self::ScaleNegative { index, bits } => write!(
                f,
                "MQ-I4 scale #{index} (f16 bits {bits:#06x}) has its sign bit set \
                 (negative or -0.0)"
            ),
            Self::ZeroScaleNonMidpoint {
                row,
                group,
                col,
                code,
                midpoint,
            } => write!(
                f,
                "MQ-I4 row {row} group {group} has scale +0 but col {col} holds code {code}; \
                 an all-zero group must hold the midpoint code {midpoint}"
            ),
            Self::ScaleNotExactF16 { bits } => write!(
                f,
                "MQ-I4 scale f32 bits {bits:#010x} is not an exactly representable finite \
                 non-negative f16"
            ),
        }
    }
}

impl std::error::Error for MqI4Error {}

/// A decoded MQ-I4 tensor: one `u8` code per weight (row-major `[M, K]`) and
/// the f16 bit pattern of each `[M, K/128]` scale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MqI4Tensor {
    pub codes: Vec<u8>,
    pub scale_bits: Vec<u16>,
}

/// Bytes of the codes plane of an `[M, K]` tensor: `M*K*b/8`.
///
/// Saturates on overflow; [`tensor_bytes`] and the pack/unpack entry points
/// perform the checked, validated computation.
pub fn codes_bytes(codec: MqI4Codec, m: usize, k: usize) -> usize {
    m.saturating_mul(k).saturating_mul(codec.bits()) / 8
}

/// Number of f16 scales of an `[M, K]` tensor: `M*(K/128)`.
pub fn scale_count(m: usize, k: usize) -> usize {
    m.saturating_mul(k / GROUP)
}

/// Validate `[M, K]` against the 128-column group; returns `(m*k, codes_bytes)`.
fn check_shape(codec: MqI4Codec, m: usize, k: usize) -> Result<(usize, usize), MqI4Error> {
    if m == 0 || k == 0 {
        return Err(MqI4Error::ZeroDimension { m, k });
    }
    if k % GROUP != 0 {
        return Err(MqI4Error::KNotGroupAligned { k });
    }
    let overflow = MqI4Error::SizeOverflow { m, k };
    let n = m.checked_mul(k).ok_or(overflow.clone())?;
    let bytes = n.checked_mul(codec.bits()).ok_or(overflow)? / 8;
    Ok((n, bytes))
}

/// [`check_shape`] plus the profile's rotation-width divisibility.
fn check_profile_shape(
    profile: MqI4Profile,
    m: usize,
    k: usize,
) -> Result<(usize, usize), MqI4Error> {
    let shape = check_shape(profile.codec, m, k)?;
    let width = profile.rotation.width();
    if k % width != 0 {
        return Err(MqI4Error::KNotRotationAligned {
            k,
            rotation_width: width,
        });
    }
    Ok(shape)
}

/// Exact byte size of an `[M, K]` tensor of `profile`: codes plane + scales
/// plane. Errors unless `K % 128 == 0` and `K % R == 0`.
pub fn tensor_bytes(profile: MqI4Profile, m: usize, k: usize) -> Result<usize, MqI4Error> {
    let (_, codes) = check_profile_shape(profile, m, k)?;
    let scales = m
        .checked_mul(k / GROUP)
        .and_then(|s| s.checked_mul(2))
        .ok_or(MqI4Error::SizeOverflow { m, k })?;
    codes
        .checked_add(scales)
        .ok_or(MqI4Error::SizeOverflow { m, k })
}

/// Pack one `u8` code per weight (`m*k`, row-major) into the planar codes
/// plane. Errors on any code `> L`, wrong length, or `K % 128 != 0`.
pub fn pack_codes(
    codec: MqI4Codec,
    codes: &[u8],
    m: usize,
    k: usize,
) -> Result<Vec<u8>, MqI4Error> {
    let (n, bytes) = check_shape(codec, m, k)?;
    if codes.len() != n {
        return Err(MqI4Error::LengthMismatch {
            what: "codes",
            expected: n,
            got: codes.len(),
        });
    }
    let mut out = vec![0u8; bytes];
    pack_into(codec, codes, &mut out, k)?;
    Ok(out)
}

/// Pack `codes` (a whole number of rows) into `out`. Every row is a multiple
/// of 8 codes, hence of 3 (W3) or 1 (W2) whole bytes, so walking the flat
/// buffer in fixed chunks is identical to walking it row by row.
fn pack_into(codec: MqI4Codec, codes: &[u8], out: &mut [u8], k: usize) -> Result<(), MqI4Error> {
    let max = codec.levels();
    let oob = |index: usize, code: u8| MqI4Error::CodeOutOfRange {
        row: index / k,
        col: index % k,
        code,
        max,
    };
    match codec {
        MqI4Codec::W3 => {
            for (ci, (src, dst)) in codes
                .chunks_exact(8)
                .zip(out.chunks_exact_mut(3))
                .enumerate()
            {
                let mut acc = 0u32;
                for (j, &c) in src.iter().enumerate() {
                    if c > max {
                        return Err(oob(ci * 8 + j, c));
                    }
                    acc |= u32::from(c) << (3 * j);
                }
                dst[0] = acc as u8;
                dst[1] = (acc >> 8) as u8;
                dst[2] = (acc >> 16) as u8;
            }
        }
        MqI4Codec::W2 => {
            for (ci, (src, dst)) in codes.chunks_exact(4).zip(out.iter_mut()).enumerate() {
                let mut acc = 0u8;
                for (j, &c) in src.iter().enumerate() {
                    if c > max {
                        return Err(oob(ci * 4 + j, c));
                    }
                    acc |= c << (2 * j);
                }
                *dst = acc;
            }
        }
    }
    Ok(())
}

/// Unpack the planar codes plane into one `u8` code per weight. `bytes` must
/// be exactly `M*K*b/8` long. Every b-bit pattern is a legal code.
pub fn unpack_codes(
    codec: MqI4Codec,
    bytes: &[u8],
    m: usize,
    k: usize,
) -> Result<Vec<u8>, MqI4Error> {
    let (n, expected) = check_shape(codec, m, k)?;
    if bytes.len() != expected {
        return Err(MqI4Error::LengthMismatch {
            what: "codes plane",
            expected,
            got: bytes.len(),
        });
    }
    let mut out = vec![0u8; n];
    match codec {
        MqI4Codec::W3 => {
            for (src, dst) in bytes.chunks_exact(3).zip(out.chunks_exact_mut(8)) {
                let acc = u32::from(src[0]) | (u32::from(src[1]) << 8) | (u32::from(src[2]) << 16);
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = ((acc >> (3 * j)) & 7) as u8;
                }
            }
        }
        MqI4Codec::W2 => {
            for (&src, dst) in bytes.iter().zip(out.chunks_exact_mut(4)) {
                for (j, d) in dst.iter_mut().enumerate() {
                    *d = (src >> (2 * j)) & 3;
                }
            }
        }
    }
    Ok(out)
}

/// Convert an f16-valued f32 (as stored in the calibration NPZ, which keeps
/// f16 values in F32 arrays) to its f16 bit pattern.
///
/// Exact or error, never rounding: the value must be finite, have a clear
/// sign bit (so `-0.0` is rejected), and survive an f32 → f16 → f32 round trip
/// bit-for-bit.
pub fn scale_bits_from_f16_valued_f32(value: f32) -> Result<u16, MqI4Error> {
    let bits = value.to_bits();
    if bits >> 31 != 0 || !value.is_finite() {
        return Err(MqI4Error::ScaleNotExactF16 { bits });
    }
    let h = half::f16::from_f32(value);
    if h.to_f32().to_bits() != bits {
        return Err(MqI4Error::ScaleNotExactF16 { bits });
    }
    Ok(h.to_bits())
}

/// Validate scales (finite, non-negative) and the zero-scale ⇒ midpoint rule.
/// `codes` is one `u8` per weight, already known to be `<= L`.
fn validate_scales(
    codec: MqI4Codec,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
) -> Result<(), MqI4Error> {
    let groups_per_row = k / GROUP;
    let mid = codec.midpoint();
    for (index, &bits) in scale_bits.iter().enumerate() {
        if bits & 0x8000 != 0 {
            return Err(MqI4Error::ScaleNegative { index, bits });
        }
        if bits & 0x7C00 == 0x7C00 {
            return Err(MqI4Error::ScaleNotFinite { index, bits });
        }
        if bits == 0 {
            let row = index / groups_per_row;
            let group = index % groups_per_row;
            let base = row * k + group * GROUP;
            if let Some((off, &code)) = codes[base..base + GROUP]
                .iter()
                .enumerate()
                .find(|&(_, &c)| c != mid)
            {
                return Err(MqI4Error::ZeroScaleNonMidpoint {
                    row,
                    group,
                    col: group * GROUP + off,
                    code,
                    midpoint: mid,
                });
            }
        }
    }
    Ok(())
}

/// Encode an `[M, K]` tensor of `profile` from one `u8` code per weight and
/// the f16 bit pattern of each `[M, K/128]` scale. Applies the full import
/// validation (see the module docs); the result is exactly
/// [`tensor_bytes`]`(profile, m, k)` long.
pub fn encode_tensor(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    codes: &[u8],
    scale_bits: &[u16],
) -> Result<Vec<u8>, MqI4Error> {
    let (n, codes_len) = check_profile_shape(profile, m, k)?;
    let groups = scale_count(m, k);
    if codes.len() != n {
        return Err(MqI4Error::LengthMismatch {
            what: "codes",
            expected: n,
            got: codes.len(),
        });
    }
    if scale_bits.len() != groups {
        return Err(MqI4Error::LengthMismatch {
            what: "scales",
            expected: groups,
            got: scale_bits.len(),
        });
    }
    let total = tensor_bytes(profile, m, k)?;
    let mut out = vec![0u8; total];
    let (plane, scales) = out.split_at_mut(codes_len);
    pack_into(profile.codec, codes, plane, k)?;
    validate_scales(profile.codec, k, codes, scale_bits)?;
    for (dst, &bits) in scales.chunks_exact_mut(2).zip(scale_bits) {
        dst.copy_from_slice(&bits.to_le_bytes());
    }
    Ok(out)
}

/// Decode and fully validate an `[M, K]` tensor of `profile`. `bytes` must be
/// exactly [`tensor_bytes`]`(profile, m, k)` long.
pub fn decode_tensor(
    profile: MqI4Profile,
    m: usize,
    k: usize,
    bytes: &[u8],
) -> Result<MqI4Tensor, MqI4Error> {
    let total = tensor_bytes(profile, m, k)?;
    if bytes.len() != total {
        return Err(MqI4Error::LengthMismatch {
            what: "tensor",
            expected: total,
            got: bytes.len(),
        });
    }
    let (_, codes_len) = check_profile_shape(profile, m, k)?;
    let (plane, scales) = bytes.split_at(codes_len);
    let codes = unpack_codes(profile.codec, plane, m, k)?;
    let scale_bits: Vec<u16> = scales
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    validate_scales(profile.codec, k, &codes, &scale_bits)?;
    Ok(MqI4Tensor { codes, scale_bits })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift for reproducible pseudo-random fixtures.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn random_tensor(codec: MqI4Codec, m: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u16>) {
        let mut rng = Rng(seed);
        let levels = u64::from(codec.levels()) + 1;
        let codes: Vec<u8> = (0..m * k).map(|_| (rng.next() % levels) as u8).collect();
        let scales: Vec<u16> = (0..scale_count(m, k))
            .map(|_| {
                // A finite, strictly positive f16 (exponent 1..=30).
                let exp = 1 + (rng.next() % 30) as u16;
                let man = (rng.next() % 1024) as u16;
                (exp << 10) | man
            })
            .collect();
        (codes, scales)
    }

    #[test]
    fn fn_expert_byte_sizes_match_the_contract() {
        // gate_up [1280,2560], down [2560,640] (DESIGN F1).
        assert_eq!(codes_bytes(MqI4Codec::W3, 1280, 2560), 1_228_800);
        assert_eq!(scale_count(1280, 2560) * 2, 51_200);
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ3_R256, 1280, 2560).unwrap(),
            1_280_000
        );
        assert_eq!(codes_bytes(MqI4Codec::W3, 2560, 640), 614_400);
        assert_eq!(scale_count(2560, 640) * 2, 25_600);
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ3_R128, 2560, 640).unwrap(),
            640_000
        );
        assert_eq!(1_280_000 + 640_000, 1_920_000);

        assert_eq!(
            tensor_bytes(MqI4Profile::MQ2_R256, 1280, 2560).unwrap(),
            870_400
        );
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ2_R128, 2560, 640).unwrap(),
            435_200
        );
        assert_eq!(870_400 + 435_200, 1_305_600);
    }

    #[test]
    fn w3_pack_known_answer() {
        // Codes 0..=7 -> bits LSB-first 000 100 010 110 001 101 011 111.
        let codes: Vec<u8> = (0..8).collect();
        let mut row = codes.clone();
        // Pad the row out to K=128 by repeating the pattern; the first 3
        // bytes must stay the known answer and the pattern must repeat.
        row.resize(128, 0);
        for j in 8..128 {
            row[j] = row[j - 8];
        }
        let packed = pack_codes(MqI4Codec::W3, &row, 1, 128).unwrap();
        assert_eq!(packed.len(), 48);
        assert_eq!(&packed[..3], &[0x88, 0xC6, 0xFA]);
        for chunk in packed.chunks_exact(3) {
            assert_eq!(chunk, &[0x88, 0xC6, 0xFA]);
        }
        assert_eq!(unpack_codes(MqI4Codec::W3, &packed, 1, 128).unwrap(), row);
    }

    #[test]
    fn w2_pack_known_answer() {
        let mut row = vec![0u8; 128];
        for (j, c) in row.iter_mut().enumerate() {
            *c = (j % 4) as u8;
        }
        let packed = pack_codes(MqI4Codec::W2, &row, 1, 128).unwrap();
        assert_eq!(packed.len(), 32);
        assert!(packed.iter().all(|&b| b == 0xE4));
        assert_eq!(unpack_codes(MqI4Codec::W2, &packed, 1, 128).unwrap(), row);
    }

    #[test]
    fn rows_are_independent_bitstreams() {
        // Row 1 must start at byte K*b/8, not mid-byte of row 0.
        let mut codes = vec![0u8; 2 * 128];
        codes[128] = 7; // row 1, col 0
        let packed = pack_codes(MqI4Codec::W3, &codes, 2, 128).unwrap();
        assert_eq!(packed.len(), 96);
        assert!(packed[..48].iter().all(|&b| b == 0));
        assert_eq!(packed[48], 0x07);
        assert!(packed[49..].iter().all(|&b| b == 0));
        let mut codes = vec![0u8; 2 * 128];
        codes[128 + 127] = 3; // row 1, last col -> top bits of row 1's last byte
        let packed = pack_codes(MqI4Codec::W2, &codes, 2, 128).unwrap();
        assert_eq!(packed.len(), 64);
        assert_eq!(packed[32 + 31], 0xC0);
    }

    #[test]
    fn random_round_trip_both_codecs() {
        for (codec, profile) in [
            (MqI4Codec::W3, MqI4Profile::MQ3_R256),
            (MqI4Codec::W2, MqI4Profile::MQ2_R256),
            (MqI4Codec::W3, MqI4Profile::MQ3_R128),
            (MqI4Codec::W2, MqI4Profile::MQ2_R128),
        ] {
            let (m, k) = (3, 256);
            let (codes, scales) = random_tensor(codec, m, k, 0x9E37_79B9_7F4A_7C15);
            let plane = pack_codes(codec, &codes, m, k).unwrap();
            assert_eq!(unpack_codes(codec, &plane, m, k).unwrap(), codes);

            let bytes = encode_tensor(profile, m, k, &codes, &scales).unwrap();
            assert_eq!(bytes.len(), tensor_bytes(profile, m, k).unwrap());
            assert_eq!(&bytes[..plane.len()], &plane[..]);
            // Scale plane layout: LE f16 at codes_bytes + 2*(r*K/128 + g).
            let base = plane.len();
            for r in 0..m {
                for g in 0..k / GROUP {
                    let at = base + 2 * (r * (k / GROUP) + g);
                    assert_eq!(
                        u16::from_le_bytes([bytes[at], bytes[at + 1]]),
                        scales[r * (k / GROUP) + g]
                    );
                }
            }
            let decoded = decode_tensor(profile, m, k, &bytes).unwrap();
            assert_eq!(decoded.codes, codes);
            assert_eq!(decoded.scale_bits, scales);
        }
    }

    #[test]
    fn pack_rejects_out_of_range_codes_and_bad_lengths() {
        let mut codes = vec![0u8; 128];
        codes[5] = 8;
        assert_eq!(
            pack_codes(MqI4Codec::W3, &codes, 1, 128),
            Err(MqI4Error::CodeOutOfRange {
                row: 0,
                col: 5,
                code: 8,
                max: 7
            })
        );
        codes[5] = 4;
        assert_eq!(
            pack_codes(MqI4Codec::W2, &codes, 1, 128),
            Err(MqI4Error::CodeOutOfRange {
                row: 0,
                col: 5,
                code: 4,
                max: 3
            })
        );
        // Code error positions are reported per row.
        let mut two_rows = vec![0u8; 256];
        two_rows[128 + 9] = 9;
        assert_eq!(
            pack_codes(MqI4Codec::W3, &two_rows, 2, 128),
            Err(MqI4Error::CodeOutOfRange {
                row: 1,
                col: 9,
                code: 9,
                max: 7
            })
        );
        assert!(matches!(
            pack_codes(MqI4Codec::W3, &[0u8; 127], 1, 128),
            Err(MqI4Error::LengthMismatch { what: "codes", .. })
        ));
        assert!(matches!(
            unpack_codes(MqI4Codec::W3, &[0u8; 47], 1, 128),
            Err(MqI4Error::LengthMismatch { .. })
        ));
        assert!(matches!(
            unpack_codes(MqI4Codec::W3, &[0u8; 49], 1, 128),
            Err(MqI4Error::LengthMismatch { .. })
        ));
        assert_eq!(
            pack_codes(MqI4Codec::W3, &[], 0, 128),
            Err(MqI4Error::ZeroDimension { m: 0, k: 128 })
        );
    }

    #[test]
    fn k_must_be_a_multiple_of_group_and_rotation() {
        // Not a multiple of 128.
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ3_R128, 1, 192),
            Err(MqI4Error::KNotGroupAligned { k: 192 })
        );
        assert!(matches!(
            pack_codes(MqI4Codec::W3, &[0u8; 64], 1, 64),
            Err(MqI4Error::KNotGroupAligned { k: 64 })
        ));
        // R256 with K=640 fails (640 = 5*128), R128 with K=640 is fine.
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ3_R256, 2, 640),
            Err(MqI4Error::KNotRotationAligned {
                k: 640,
                rotation_width: 256
            })
        );
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ2_R256, 2, 640),
            Err(MqI4Error::KNotRotationAligned {
                k: 640,
                rotation_width: 256
            })
        );
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ3_R128, 2, 640).unwrap(),
            2 * 240 + 2 * 5 * 2
        );
        assert_eq!(
            tensor_bytes(MqI4Profile::MQ2_R128, 2, 640).unwrap(),
            2 * 160 + 2 * 5 * 2
        );
        // encode/decode apply the same rule.
        let (codes, scales) = random_tensor(MqI4Codec::W3, 2, 640, 7);
        assert!(matches!(
            encode_tensor(MqI4Profile::MQ3_R256, 2, 640, &codes, &scales),
            Err(MqI4Error::KNotRotationAligned { .. })
        ));
        let ok = encode_tensor(MqI4Profile::MQ3_R128, 2, 640, &codes, &scales).unwrap();
        assert!(matches!(
            decode_tensor(MqI4Profile::MQ3_R256, 2, 640, &ok),
            Err(MqI4Error::KNotRotationAligned { .. })
        ));
        assert_eq!(
            decode_tensor(MqI4Profile::MQ3_R128, 2, 640, &ok)
                .unwrap()
                .codes,
            codes
        );
        // R128 is a profile property, not inferred from K: K=256 is legal for it.
        assert!(tensor_bytes(MqI4Profile::MQ3_R128, 2, 256).is_ok());
    }

    #[test]
    fn bad_scales_are_rejected_on_encode_and_decode() {
        let profile = MqI4Profile::MQ3_R256;
        let (m, k) = (1, 256);
        let (codes, good) = random_tensor(MqI4Codec::W3, m, k, 11);
        for bits in [
            0x7E00u16, // NaN
            0x7C00,    // +inf
            0xFC00,    // -inf (sign set: reported as negative)
            0x8001,    // negative subnormal
            0xBC00,    // -1.0
            0x8000,    // -0.0
        ] {
            let mut scales = good.clone();
            scales[1] = bits;
            let err = encode_tensor(profile, m, k, &codes, &scales).unwrap_err();
            if bits & 0x8000 != 0 {
                assert_eq!(err, MqI4Error::ScaleNegative { index: 1, bits });
            } else {
                assert_eq!(err, MqI4Error::ScaleNotFinite { index: 1, bits });
            }

            // Same bytes through decode.
            let mut bytes = encode_tensor(profile, m, k, &codes, &good).unwrap();
            let at = codes_bytes(MqI4Codec::W3, m, k) + 2;
            bytes[at..at + 2].copy_from_slice(&bits.to_le_bytes());
            assert_eq!(decode_tensor(profile, m, k, &bytes).unwrap_err(), err);
        }
    }

    #[test]
    fn zero_scale_requires_the_midpoint_code() {
        for (codec, profile) in [
            (MqI4Codec::W3, MqI4Profile::MQ3_R256),
            (MqI4Codec::W2, MqI4Profile::MQ2_R256),
        ] {
            let (m, k) = (1, 256);
            let mid = codec.midpoint();
            let mut codes = vec![mid; m * k];
            let scales = vec![0u16, 0x3C00]; // group 0 zero, group 1 one
                                             // All-midpoint zero group is legal (group 1 may hold anything).
            codes[200] = 0;
            let bytes = encode_tensor(profile, m, k, &codes, &scales).unwrap();
            assert_eq!(decode_tensor(profile, m, k, &bytes).unwrap().codes, codes);
            // A non-midpoint code inside the zero group is rejected.
            codes[17] = mid ^ 1;
            let want = MqI4Error::ZeroScaleNonMidpoint {
                row: 0,
                group: 0,
                col: 17,
                code: mid ^ 1,
                midpoint: mid,
            };
            assert_eq!(
                encode_tensor(profile, m, k, &codes, &scales).unwrap_err(),
                want
            );
            // Decode: forge the bytes by encoding with a nonzero scale first.
            let forged_scales = vec![0x3C00u16, 0x3C00];
            let mut forged = encode_tensor(profile, m, k, &codes, &forged_scales).unwrap();
            let at = codes_bytes(codec, m, k);
            forged[at..at + 2].copy_from_slice(&0u16.to_le_bytes());
            assert_eq!(decode_tensor(profile, m, k, &forged).unwrap_err(), want);
        }
        assert_eq!(MqI4Codec::W3.midpoint(), 4);
        assert_eq!(MqI4Codec::W2.midpoint(), 2);
    }

    #[test]
    fn encode_decode_reject_wrong_lengths() {
        let profile = MqI4Profile::MQ2_R128;
        let (m, k) = (2, 128);
        let (codes, scales) = random_tensor(MqI4Codec::W2, m, k, 3);
        assert!(matches!(
            encode_tensor(profile, m, k, &codes[1..], &scales),
            Err(MqI4Error::LengthMismatch { what: "codes", .. })
        ));
        assert!(matches!(
            encode_tensor(profile, m, k, &codes, &scales[1..]),
            Err(MqI4Error::LengthMismatch { what: "scales", .. })
        ));
        let bytes = encode_tensor(profile, m, k, &codes, &scales).unwrap();
        assert!(matches!(
            decode_tensor(profile, m, k, &bytes[1..]),
            Err(MqI4Error::LengthMismatch { what: "tensor", .. })
        ));
        let mut long = bytes.clone();
        long.push(0);
        assert!(matches!(
            decode_tensor(profile, m, k, &long),
            Err(MqI4Error::LengthMismatch { what: "tensor", .. })
        ));
        assert!(decode_tensor(profile, m, k, &bytes).is_ok());
    }

    #[test]
    fn scale_conversion_is_exact_or_error() {
        assert_eq!(scale_bits_from_f16_valued_f32(0.0), Ok(0));
        assert_eq!(scale_bits_from_f16_valued_f32(1.0), Ok(0x3C00));
        assert_eq!(scale_bits_from_f16_valued_f32(0.5), Ok(0x3800));
        assert_eq!(scale_bits_from_f16_valued_f32(65504.0), Ok(0x7BFF));
        // Smallest positive f16 subnormal is exact in f32.
        assert_eq!(scale_bits_from_f16_valued_f32(2f32.powi(-24)), Ok(0x0001));
        // Every positive finite f16 round-trips.
        for bits in 0u16..0x7C00 {
            let v = half::f16::from_bits(bits).to_f32();
            assert_eq!(scale_bits_from_f16_valued_f32(v), Ok(bits));
        }
        // Not f16-representable: needs rounding.
        for v in [
            0.1f32,
            1.0 + 2f32.powi(-12),
            65520.0,
            2f32.powi(-25),
            1e-10,
            1e10,
        ] {
            assert!(
                matches!(
                    scale_bits_from_f16_valued_f32(v),
                    Err(MqI4Error::ScaleNotExactF16 { .. })
                ),
                "{v} must be rejected"
            );
        }
        // Non-finite and negative values, including -0.0.
        for v in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            -2f32.powi(-24),
        ] {
            assert!(
                matches!(
                    scale_bits_from_f16_valued_f32(v),
                    Err(MqI4Error::ScaleNotExactF16 { .. })
                ),
                "{v} must be rejected"
            );
        }
    }

    #[test]
    fn profile_quant_type_round_trip() {
        let expected = [
            (131u8, MqI4Codec::W3, MqI4Rotation::R256, "MQ3I4G128R256"),
            (132, MqI4Codec::W3, MqI4Rotation::R128, "MQ3I4G128R128"),
            (133, MqI4Codec::W2, MqI4Rotation::R256, "MQ2I4G128R256"),
            (134, MqI4Codec::W2, MqI4Rotation::R128, "MQ2I4G128R128"),
        ];
        for (qt, codec, rotation, name) in expected {
            let p = MqI4Profile::from_quant_type(qt).expect("profile");
            assert_eq!(p, MqI4Profile { codec, rotation });
            assert_eq!(p.quant_type(), qt);
            assert_eq!(p.name(), name);
        }
        assert_eq!(
            MqI4Profile::ALL.map(MqI4Profile::quant_type),
            [131, 132, 133, 134]
        );
        for qt in [0u8, 44, 53, 54, 129, 130, 135, 255] {
            assert_eq!(MqI4Profile::from_quant_type(qt), None, "qt={qt}");
        }
    }

    #[test]
    fn codec_and_rotation_constants() {
        assert_eq!((MqI4Codec::W3.bits(), MqI4Codec::W3.levels()), (3, 7));
        assert_eq!((MqI4Codec::W2.bits(), MqI4Codec::W2.levels()), (2, 3));
        assert_eq!(
            (MqI4Rotation::R256.width(), MqI4Rotation::R256.seeds()),
            (256, (42, 1042))
        );
        assert_eq!(
            (MqI4Rotation::R128.width(), MqI4Rotation::R128.seeds()),
            (128, (43, 1043))
        );
    }
}
