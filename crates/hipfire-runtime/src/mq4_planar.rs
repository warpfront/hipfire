// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! MQ4 "drop-zero" planar expert tensors (`MQ4G256V2P` gate/up, `MQ4G128V2P`
//! down; DESIGN F1, `HIPFIRE_QWEN4_EXPERT_DROP_ZERO`).
//!
//! CPU-only, GPU-free. The single home of the layout: the load-time repack
//! (`weight_store`), the CPU reference/expander and the kernel byte oracles
//! import it and never fork it.
//!
//! The shipped Flash-Next routed experts are symmetric MQ4-XTS: every group
//! half stores `zero = f16(-8 * scale)` (the loader's symmetric-header check,
//! `zp == -8 * sc` in f32, both finite). The zero field is therefore redundant
//! and the repack drops it. Every code and every scale stays bit-identical, so
//! `w = sc * (q - 8)` is unchanged (`sc` is f16 and `q - 8` a small integer: the
//! f32 FMA `fma(sc, q, -8 * sc)` is exact, and `-8 * sc` widened from f16 is the
//! f32 product exactly because x8 only moves the exponent).
//!
//! # Tensor bytes (`[M, K]`, `K % 256 == 0` gate/up, `K % 128 == 0` down)
//!
//! ```text
//! [codes plane: M*K/2 bytes][scales plane: M*(K/128) little-endian f16]
//! ```
//!
//! * Codes keep the shipped nibble order (low nibble = even weight); row `r`
//!   starts at byte `r*K/2` and is the concatenation of the shipped groups'
//!   nibble bytes (128 B per 256-wide gate/up group, 64 B per 128-wide down
//!   group).
//! * The scale for row `r`, 128-column group `g` is the LE f16 at
//!   `codes_bytes + 2*(r*(K/128) + g)`. For gate/up the two halves of a shipped
//!   256-group are consecutive scales (`scale_0`, `scale_1`).
//!
//! Per expert (gate_up `[1280, 2560]` + down `[2560, 640]`): 1,689,600 +
//! 844,800 = 2,534,400 B against 1,740,800 + 870,400 = 2,611,200 shipped
//! (-76,800 B, -2.94%); both planar sizes are multiples of 256 B.

use std::fmt;

/// Scale group width in columns.
pub const GROUP: usize = 128;
/// HFQ `quant_type` of the planar gate/up tensor.
pub const QT_GATE_UP: u8 = 137;
/// HFQ `quant_type` of the planar down tensor.
pub const QT_DOWN: u8 = 138;
/// Canonical dtype name of the planar gate/up tensor.
pub const NAME_GATE_UP: &str = "MQ4G256V2P";
/// Canonical dtype name of the planar down tensor.
pub const NAME_DOWN: &str = "MQ4G128V2P";
/// Flash-Next routed expert gate/up tensor `[rows, K]` (gate rows then up rows).
pub const FLASH_NEXT_GATE_UP: (usize, usize) = (1280, 2560);
/// Flash-Next routed expert down tensor `[rows, K]`.
pub const FLASH_NEXT_DOWN: (usize, usize) = (2560, 640);

/// Which projection of an expert a tensor is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mq4PlanarKind {
    /// Shipped `MQ4G256V2` (qt44): 136 B per 256-wide group
    /// (`sc0 zp0 sc1 zp1` f16 + 128 nibble bytes).
    GateUp,
    /// Shipped `MQ4G128V2` (qt53): 68 B per 128-wide group
    /// (`sc zp` f16 + 64 nibble bytes).
    Down,
}

impl Mq4PlanarKind {
    /// Columns of one shipped group.
    pub const fn shipped_group_cols(self) -> usize {
        match self {
            Self::GateUp => 256,
            Self::Down => 128,
        }
    }

    /// Bytes of one shipped group.
    pub const fn shipped_group_bytes(self) -> usize {
        match self {
            Self::GateUp => 136,
            Self::Down => 68,
        }
    }

    /// Header bytes of one shipped group (`scale zero` per 128 columns).
    const fn shipped_header_bytes(self) -> usize {
        match self {
            Self::GateUp => 8,
            Self::Down => 4,
        }
    }

    /// Nibble bytes of one shipped group.
    const fn code_bytes_per_group(self) -> usize {
        self.shipped_group_bytes() - self.shipped_header_bytes()
    }

    /// 128-column scale groups of one shipped group.
    const fn halves(self) -> usize {
        self.shipped_group_cols() / GROUP
    }

    /// The `quant_type` of the planar tensor.
    pub const fn quant_type(self) -> u8 {
        match self {
            Self::GateUp => QT_GATE_UP,
            Self::Down => QT_DOWN,
        }
    }

    /// Canonical dtype name of the planar tensor.
    pub const fn name(self) -> &'static str {
        match self {
            Self::GateUp => NAME_GATE_UP,
            Self::Down => NAME_DOWN,
        }
    }
}

/// Typed repack failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mq4PlanarError {
    /// `M` or `K` is zero, or `K` is not a whole number of shipped groups.
    BadShape { m: usize, k: usize },
    /// A size computation overflowed `usize`.
    SizeOverflow { m: usize, k: usize },
    /// A buffer has the wrong length.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A shipped header is not on the symmetric grid (`zp == -8 * sc`, both
    /// finite): the tensor cannot drop its zero plane.
    Asymmetric {
        row: usize,
        half: usize,
        scale_bits: u16,
        zero_bits: u16,
    },
}

impl fmt::Display for Mq4PlanarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::BadShape { m, k } => write!(f, "MQ4 planar tensor shape [{m}, {k}] is illegal"),
            Self::SizeOverflow { m, k } => {
                write!(f, "MQ4 planar byte size overflows usize: M={m} K={k}")
            }
            Self::LengthMismatch {
                what,
                expected,
                got,
            } => write!(
                f,
                "MQ4 planar {what} length mismatch: expected {expected}, got {got}"
            ),
            Self::Asymmetric {
                row,
                half,
                scale_bits,
                zero_bits,
            } => write!(
                f,
                "MQ4 planar repack refused: row {row} half {half} header is not symmetric \
                 (scale f16 {scale_bits:#06x}, zero f16 {zero_bits:#06x}; need zero == -8 * scale)"
            ),
        }
    }
}

impl std::error::Error for Mq4PlanarError {}

/// Decode an IEEE binary16 bit pattern exactly (the GPU checker's
/// `(float)__builtin_bit_cast(_Float16, ..)`).
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// The symmetric-grid predicate of `qwen4_moe_sym_check.gfx1201.hip`: both
/// finite and `zp == -8 * sc` in f32. Identical to the loader's
/// `routed_headers_symmetric` predicate.
pub fn header_symmetric(scale_bits: u16, zero_bits: u16) -> bool {
    let scale = f16_bits_to_f32(scale_bits);
    let zero = f16_bits_to_f32(zero_bits);
    scale.is_finite() && zero.is_finite() && zero == -8.0f32 * scale
}

/// The shipped zero bits the expander writes for `scale_bits`: `f16(-8 * sc)`
/// (what the quantizer stores).
pub fn zero_bits_for_scale(scale_bits: u16) -> u16 {
    half::f16::from_f32(-8.0f32 * f16_bits_to_f32(scale_bits)).to_bits()
}

fn check_shape(kind: Mq4PlanarKind, m: usize, k: usize) -> Result<(), Mq4PlanarError> {
    if m == 0 || k == 0 || k % kind.shipped_group_cols() != 0 {
        return Err(Mq4PlanarError::BadShape { m, k });
    }
    Ok(())
}

/// Bytes of the codes plane of `[m, k]`: `m * k / 2`.
pub fn codes_bytes(m: usize, k: usize) -> Option<usize> {
    m.checked_mul(k).map(|n| n / 2)
}

/// Number of f16 scales of `[m, k]`: `m * (k / 128)`.
pub fn scale_count(m: usize, k: usize) -> Option<usize> {
    m.checked_mul(k / GROUP)
}

/// Exact planar byte size of an `[m, k]` tensor.
pub fn planar_bytes(kind: Mq4PlanarKind, m: usize, k: usize) -> Result<usize, Mq4PlanarError> {
    check_shape(kind, m, k)?;
    let overflow = Mq4PlanarError::SizeOverflow { m, k };
    let codes = codes_bytes(m, k).ok_or(overflow.clone())?;
    let scales = scale_count(m, k)
        .and_then(|n| n.checked_mul(2))
        .ok_or(overflow.clone())?;
    codes.checked_add(scales).ok_or(overflow)
}

/// Exact shipped (qt44 / qt53) byte size of an `[m, k]` tensor.
pub fn shipped_bytes(kind: Mq4PlanarKind, m: usize, k: usize) -> Result<usize, Mq4PlanarError> {
    check_shape(kind, m, k)?;
    let overflow = Mq4PlanarError::SizeOverflow { m, k };
    (k / kind.shipped_group_cols())
        .checked_mul(kind.shipped_group_bytes())
        .and_then(|row| row.checked_mul(m))
        .ok_or(overflow)
}

/// Byte offset of the scale plane inside a planar `[m, k]` tensor.
pub fn scale_plane_offset(m: usize, k: usize) -> usize {
    m * k / 2
}

/// Repack one shipped tensor `src` (qt44 for [`Mq4PlanarKind::GateUp`], qt53 for
/// [`Mq4PlanarKind::Down`]) of shape `[m, k]` into planar `dst`. Every header
/// is checked symmetric; the first asymmetric one fails the call and leaves
/// `dst` partially written (callers discard it).
pub fn repack_tensor(
    kind: Mq4PlanarKind,
    m: usize,
    k: usize,
    src: &[u8],
    dst: &mut [u8],
) -> Result<(), Mq4PlanarError> {
    let want_src = shipped_bytes(kind, m, k)?;
    let want_dst = planar_bytes(kind, m, k)?;
    if src.len() != want_src {
        return Err(Mq4PlanarError::LengthMismatch {
            what: "shipped source",
            expected: want_src,
            got: src.len(),
        });
    }
    if dst.len() != want_dst {
        return Err(Mq4PlanarError::LengthMismatch {
            what: "planar destination",
            expected: want_dst,
            got: dst.len(),
        });
    }
    let groups_per_row = k / kind.shipped_group_cols();
    let halves = kind.halves();
    let code_bytes = kind.code_bytes_per_group();
    let header = kind.shipped_header_bytes();
    let (codes, scales) = dst.split_at_mut(m * k / 2);
    let mut group = 0usize;
    for (row, row_src) in src
        .chunks_exact(groups_per_row * kind.shipped_group_bytes())
        .enumerate()
    {
        for src_group in row_src.chunks_exact(kind.shipped_group_bytes()) {
            for half in 0..halves {
                let at = 4 * half;
                let scale_bits = u16::from_le_bytes([src_group[at], src_group[at + 1]]);
                let zero_bits = u16::from_le_bytes([src_group[at + 2], src_group[at + 3]]);
                if !header_symmetric(scale_bits, zero_bits) {
                    return Err(Mq4PlanarError::Asymmetric {
                        row,
                        half,
                        scale_bits,
                        zero_bits,
                    });
                }
                let s = 2 * (group * halves + half);
                scales[s..s + 2].copy_from_slice(&scale_bits.to_le_bytes());
            }
            codes[group * code_bytes..(group + 1) * code_bytes]
                .copy_from_slice(&src_group[header..]);
            group += 1;
        }
    }
    Ok(())
}

/// Inverse of [`repack_tensor`]: rebuild the shipped bytes (`zero =
/// f16(-8 * scale)`) from planar `src`. For quantizer-written files (zero
/// written as `f16(-8 * scale)`) this reproduces the shipped bytes exactly.
pub fn expand_tensor(
    kind: Mq4PlanarKind,
    m: usize,
    k: usize,
    src: &[u8],
    dst: &mut [u8],
) -> Result<(), Mq4PlanarError> {
    let want_src = planar_bytes(kind, m, k)?;
    let want_dst = shipped_bytes(kind, m, k)?;
    if src.len() != want_src {
        return Err(Mq4PlanarError::LengthMismatch {
            what: "planar source",
            expected: want_src,
            got: src.len(),
        });
    }
    if dst.len() != want_dst {
        return Err(Mq4PlanarError::LengthMismatch {
            what: "shipped destination",
            expected: want_dst,
            got: dst.len(),
        });
    }
    let halves = kind.halves();
    let code_bytes = kind.code_bytes_per_group();
    let header = kind.shipped_header_bytes();
    let (codes, scales) = src.split_at(m * k / 2);
    for (group, dst_group) in dst.chunks_exact_mut(kind.shipped_group_bytes()).enumerate() {
        for half in 0..halves {
            let s = 2 * (group * halves + half);
            let scale_bits = u16::from_le_bytes([scales[s], scales[s + 1]]);
            let at = 4 * half;
            dst_group[at..at + 2].copy_from_slice(&scale_bits.to_le_bytes());
            dst_group[at + 2..at + 4]
                .copy_from_slice(&zero_bits_for_scale(scale_bits).to_le_bytes());
        }
        dst_group[header..].copy_from_slice(&codes[group * code_bytes..(group + 1) * code_bytes]);
    }
    Ok(())
}

/// CPU dequant of one weight of a planar tensor: `sc * (q - 8)` (what every
/// kernel computes as `fma(sc, q, -8 * sc)`), at row `r`, column `c`.
pub fn dequant_planar(m: usize, k: usize, planar: &[u8], r: usize, c: usize) -> f32 {
    debug_assert!(r < m && c < k);
    let byte = planar[r * (k / 2) + c / 2];
    let q = if c % 2 == 0 { byte & 0xf } else { byte >> 4 };
    let s = m * k / 2 + 2 * (r * (k / GROUP) + c / GROUP);
    let sc = f16_bits_to_f32(u16::from_le_bytes([planar[s], planar[s + 1]]));
    f32::mul_add(sc, f32::from(q), -8.0f32 * sc)
}

/// CPU dequant of one weight of a SHIPPED tensor: `sc * q + zp` (what the
/// shipped kernels compute), at row `r`, column `c`.
pub fn dequant_shipped(kind: Mq4PlanarKind, k: usize, shipped: &[u8], r: usize, c: usize) -> f32 {
    let gc = kind.shipped_group_cols();
    let groups_per_row = k / gc;
    let group = r * groups_per_row + c / gc;
    let base = group * kind.shipped_group_bytes();
    let half = (c % gc) / GROUP;
    let h = base + 4 * half;
    let sc = f16_bits_to_f32(u16::from_le_bytes([shipped[h], shipped[h + 1]]));
    let zp = f16_bits_to_f32(u16::from_le_bytes([shipped[h + 2], shipped[h + 3]]));
    let within = c % gc;
    let byte = shipped[base + kind.shipped_header_bytes() + within / 2];
    let q = if within % 2 == 0 {
        byte & 0xf
    } else {
        byte >> 4
    };
    f32::mul_add(sc, f32::from(q), zp)
}

/// Planar payload bytes of a stacked aggregate `[experts, m, k]` tensor.
pub fn stack_planar_bytes(
    kind: Mq4PlanarKind,
    experts: usize,
    m: usize,
    k: usize,
) -> Result<usize, Mq4PlanarError> {
    planar_bytes(kind, m, k)?
        .checked_mul(experts)
        .ok_or(Mq4PlanarError::SizeOverflow { m, k })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GU: (usize, usize) = (1280, 2560);
    const DN: (usize, usize) = (2560, 640);

    /// Deterministic xorshift for the synthetic tensors.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// A shipped tensor the way the quantizer writes it: `zero = f16(-8 * scale)`.
    fn synth_shipped(kind: Mq4PlanarKind, m: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut next = rng(seed);
        let groups = m * (k / kind.shipped_group_cols());
        let mut out = vec![0u8; groups * kind.shipped_group_bytes()];
        for group in out.chunks_exact_mut(kind.shipped_group_bytes()) {
            for half in 0..kind.halves() {
                // Finite f16 scales across magnitudes, including +0 and subnormals.
                let scale_bits = match next() % 11 {
                    0 => 0u16,
                    1 => (next() % 0x3ff) as u16,
                    // Below 0x6c00 so that -8 * scale stays a finite f16 (a symmetric header).
                    _ => (next() % 0x6c00) as u16,
                };
                group[4 * half..4 * half + 2].copy_from_slice(&scale_bits.to_le_bytes());
                group[4 * half + 2..4 * half + 4]
                    .copy_from_slice(&zero_bits_for_scale(scale_bits).to_le_bytes());
            }
            for b in &mut group[kind.shipped_header_bytes()..] {
                *b = next() as u8;
            }
        }
        out
    }

    #[test]
    fn sizes_match_the_spec() {
        let gu = planar_bytes(Mq4PlanarKind::GateUp, GU.0, GU.1).unwrap();
        let dn = planar_bytes(Mq4PlanarKind::Down, DN.0, DN.1).unwrap();
        assert_eq!((gu, dn), (1_689_600, 844_800));
        assert_eq!(gu + dn, 2_534_400);
        assert_eq!(gu % 256, 0);
        assert_eq!(dn % 256, 0);
        assert_eq!(
            shipped_bytes(Mq4PlanarKind::GateUp, GU.0, GU.1).unwrap(),
            1_740_800
        );
        assert_eq!(
            shipped_bytes(Mq4PlanarKind::Down, DN.0, DN.1).unwrap(),
            870_400
        );
        assert_eq!(scale_plane_offset(GU.0, GU.1), 1_638_400);
        assert_eq!(scale_plane_offset(DN.0, DN.1), 819_200);
    }

    #[test]
    fn repack_round_trips_byte_exact_and_dequant_is_bit_equal() {
        for (kind, (m, k)) in [
            (Mq4PlanarKind::GateUp, (48, 2560)),
            (Mq4PlanarKind::Down, (64, 640)),
        ] {
            let shipped = synth_shipped(kind, m, k, 0x1234_5678 + m as u64);
            let mut planar = vec![0u8; planar_bytes(kind, m, k).unwrap()];
            repack_tensor(kind, m, k, &shipped, &mut planar).unwrap();
            let mut back = vec![0u8; shipped.len()];
            expand_tensor(kind, m, k, &planar, &mut back).unwrap();
            assert_eq!(back, shipped, "{kind:?} expand(repack(x)) != x");
            for r in 0..m {
                for c in 0..k {
                    let a = dequant_shipped(kind, k, &shipped, r, c);
                    let b = dequant_planar(m, k, &planar, r, c);
                    assert_eq!(a.to_bits(), b.to_bits(), "{kind:?} r{r} c{c}");
                }
            }
        }
    }

    #[test]
    fn repack_places_planes_as_documented() {
        let (m, k) = (3, 512);
        let kind = Mq4PlanarKind::GateUp;
        let shipped = synth_shipped(kind, m, k, 7);
        let mut planar = vec![0u8; planar_bytes(kind, m, k).unwrap()];
        repack_tensor(kind, m, k, &shipped, &mut planar).unwrap();
        // Shipped group index 3 (row 1, group 1): codes and both scales.
        let g = 3usize;
        assert_eq!(
            &planar[g * 128..(g + 1) * 128],
            &shipped[g * 136 + 8..(g + 1) * 136]
        );
        let scales = m * k / 2;
        let s = scales + 2 * (g * 2);
        assert_eq!(&planar[s..s + 2], &shipped[g * 136..g * 136 + 2]);
        assert_eq!(&planar[s + 2..s + 4], &shipped[g * 136 + 4..g * 136 + 6]);
    }

    #[test]
    fn repack_refuses_an_asymmetric_group_and_bad_lengths() {
        let (m, k) = (2, 256);
        let kind = Mq4PlanarKind::GateUp;
        let mut shipped = synth_shipped(kind, m, k, 9);
        let mut planar = vec![0u8; planar_bytes(kind, m, k).unwrap()];
        // Flip a zero bit on row 1, half 1.
        shipped[136 + 6] ^= 1;
        let err = repack_tensor(kind, m, k, &shipped, &mut planar).unwrap_err();
        assert!(
            matches!(
                err,
                Mq4PlanarError::Asymmetric {
                    row: 1,
                    half: 1,
                    ..
                }
            ),
            "{err}"
        );
        assert!(repack_tensor(kind, m, k, &shipped[1..], &mut planar).is_err());
        assert!(repack_tensor(kind, m, k, &shipped, &mut planar[1..]).is_err());
        // Non-finite scales are asymmetric too.
        let mut inf = synth_shipped(kind, m, k, 9);
        inf[0..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(repack_tensor(kind, m, k, &inf, &mut planar).is_err());
        assert!(planar_bytes(kind, 0, k).is_err());
        assert!(planar_bytes(kind, 1, 192).is_err());
    }

    #[test]
    fn zero_sign_of_a_plus_zero_header_never_changes_the_dequantised_value() {
        // A legacy +0 zero beside scale +0 passes the f32 check (0 == -0); the
        // planar dequant derives -0 and still yields +0 for every code.
        let (m, k) = (1, 256);
        let kind = Mq4PlanarKind::GateUp;
        let mut shipped = vec![0u8; shipped_bytes(kind, m, k).unwrap()];
        for (i, b) in shipped[8..].iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut planar = vec![0u8; planar_bytes(kind, m, k).unwrap()];
        repack_tensor(kind, m, k, &shipped, &mut planar).unwrap();
        for c in 0..k {
            let a = dequant_shipped(kind, k, &shipped, 0, c);
            let b = dequant_planar(m, k, &planar, 0, c);
            assert_eq!(a.to_bits(), b.to_bits(), "col {c}");
        }
    }

    #[test]
    fn minus_eight_scale_widens_exactly_for_every_finite_f16_scale() {
        // The kernels derive zp = -8.0f32 * sc; the shipped zero is the f16 of the
        // same product. Exhaustive over all 65,536 patterns whose product is in
        // f16 range (larger products are non-finite zeros: asymmetric).
        for bits in 0..=u16::MAX {
            if !f16_bits_to_f32(bits).is_finite() {
                continue;
            }
            let derived = -8.0f32 * f16_bits_to_f32(bits);
            if derived.abs() <= 65504.0 {
                let z = f16_bits_to_f32(zero_bits_for_scale(bits));
                assert_eq!(derived.to_bits(), z.to_bits(), "scale bits {bits:#06x}");
            }
        }
    }
}
