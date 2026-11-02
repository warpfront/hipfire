//! CPU decode, scan and layout helpers for the OCP FP8 E4M3 weight formats.
//!
//! * qt=55 `FP8E4M3B128` (`DType::FP8E4M3B128`): 128x128 block scales (`s_inv`,
//!   bf16 bits, a *multiplier*), expert-blocked payload.
//! * qt=56 `FP8E4M3PT`: one bf16 scale per tensor in a 16-byte header.
//!
//! Layouts, decode formula and the FNUZ fold theorem are specified in
//! `docs/quant-formats/fp8-e4m3.md`. This module is the CPU twin of
//! `subq4/calib2/fp8_teacher.py:dequant_e4m3_block` and of the gfx942 hardware
//! path (`v_cvt_f32_fp8` reads FNUZ). It lives in `rdna-compute` because
//! `hipfire-quantize`, `hipfire-runtime` and the `rdna-compute` examples can all
//! depend on it without a cycle.
//!
//! Numerical contract (frozen):
//! `W_bf16 = bf16_rne(f32(e4m3fn(q)) * f32(bf16 s_inv))` per element. The F32
//! product is exact for every finite code and every scale whose product stays in
//! the F32 normal range (4 mantissa bits x 8 mantissa bits); anything else is
//! refused rather than silently rounded twice.

/// Edge of the square scale block: one `s_inv` per 128 rows x 128 columns.
pub const B128_BLOCK: usize = 128;
/// Expert-block payload alignment (bytes).
pub const B128_ALIGN: usize = 256;
/// qt=56 header bytes: bf16 scale bits (LE) at byte 0, bytes 2..16 zero.
pub const PT_HEADER_BYTES: usize = 16;

/// Refusal reasons shared by the decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Error {
    /// `0x7F` / `0xFF` (e4m3fn NaN) at the given flat code index.
    NanCode { index: usize },
    /// A scale is NaN/inf at the given scale index.
    NonFiniteScale { index: usize },
    /// The F32 product `e4m3 * scale` is not exact (F32 subnormal range) at the
    /// given flat code index.
    InexactProduct { index: usize },
    /// The bf16 result overflowed to inf at the given flat code index.
    Overflow { index: usize },
    /// Slice lengths disagree with `rows`/`k` (or the geometry overflows).
    Shape,
}

impl std::fmt::Display for Fp8Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Fp8Error::NanCode { index } => write!(f, "e4m3 NaN code at index {index}"),
            Fp8Error::NonFiniteScale { index } => write!(f, "non-finite scale at index {index}"),
            Fp8Error::InexactProduct { index } => {
                write!(f, "f32 product e4m3*scale not exact at index {index}")
            }
            Fp8Error::Overflow { index } => write!(f, "bf16 overflow to inf at index {index}"),
            Fp8Error::Shape => write!(f, "fp8 e4m3 slice length / geometry mismatch"),
        }
    }
}

impl std::error::Error for Fp8Error {}

// ----------------------------------------------------------------- layout helpers

/// `(ceil(rows/128), ceil(k/128))`: the `s_inv` grid of one `[rows, k]` expert.
pub const fn b128_scale_dims(rows: usize, k: usize) -> (usize, usize) {
    (rows.div_ceil(B128_BLOCK), k.div_ceil(B128_BLOCK))
}

/// Number of bf16 scales of one `[rows, k]` expert block.
pub fn b128_scale_count(rows: usize, k: usize) -> Option<usize> {
    let (a, b) = b128_scale_dims(rows, k);
    a.checked_mul(b)
}

/// Code bytes of one expert block (`rows * k`).
pub fn b128_code_bytes(rows: usize, k: usize) -> Option<usize> {
    rows.checked_mul(k)
}

/// `S(R,K) = round_up(R*K + 2*ceil(R/128)*ceil(K/128), 256)`: byte stride of one
/// expert block. Codes at `e*S`, scales at `e*S + R*K`.
pub fn b128_block_stride(rows: usize, k: usize) -> Option<usize> {
    let used = b128_code_bytes(rows, k)?.checked_add(b128_scale_count(rows, k)?.checked_mul(2)?)?;
    Some(used.checked_add(B128_ALIGN - 1)? / B128_ALIGN * B128_ALIGN)
}

/// Payload bytes of a qt=55 tensor with logical shape `[experts, rows, k]`.
pub fn b128_tensor_bytes(experts: usize, rows: usize, k: usize) -> Option<usize> {
    experts.checked_mul(b128_block_stride(rows, k)?)
}

/// Byte offset of expert `e`'s scale array within the tensor payload.
pub fn b128_scales_offset(e: usize, rows: usize, k: usize) -> Option<usize> {
    e.checked_mul(b128_block_stride(rows, k)?)?
        .checked_add(b128_code_bytes(rows, k)?)
}

/// Payload bytes of a qt=56 tensor `[rows, k]`: `16 + rows*k`.
pub fn pt_tensor_bytes(rows: usize, k: usize) -> Option<usize> {
    PT_HEADER_BYTES.checked_add(rows.checked_mul(k)?)
}

/// Byte offset of row `r` of a qt=56 tensor with row width `k`: `16 + r*k`.
pub fn pt_row_offset(r: usize, k: usize) -> Option<usize> {
    PT_HEADER_BYTES.checked_add(r.checked_mul(k)?)
}

/// Build a qt=56 header for the given scale bits.
pub fn pt_header(scale_bf16: u16) -> [u8; PT_HEADER_BYTES] {
    let mut h = [0u8; PT_HEADER_BYTES];
    h[..2].copy_from_slice(&scale_bf16.to_le_bytes());
    h
}

/// Read the scale of a qt=56 payload (first 16 bytes must be the header with
/// bytes 2..16 zero).
pub fn pt_header_scale(payload: &[u8]) -> Result<u16, Fp8Error> {
    let h = payload.get(..PT_HEADER_BYTES).ok_or(Fp8Error::Shape)?;
    if h[2..].iter().any(|&b| b != 0) {
        return Err(Fp8Error::Shape);
    }
    Ok(u16::from_le_bytes([h[0], h[1]]))
}

// ----------------------------------------------------------------- scalar decode

/// OCP e4m3fn code -> exact F32. `None` for the NaN codes `0x7F` / `0xFF`.
/// bias 7; `E==0` -> `+-m * 2^-9`; else `+-(1+m/8) * 2^(E-7)`; `0x80` is `-0.0`.
pub const fn e4m3fn_to_f32(b: u8) -> Option<f32> {
    let e = ((b >> 3) & 15) as u32;
    let m = (b & 7) as u32;
    if e == 15 && m == 7 {
        return None;
    }
    let sign = ((b >> 7) as u32) << 31;
    let mag = if e == 0 {
        // m * 2^-9 : normal in F32; m == 0 -> +0.
        if m == 0 {
            0.0f32
        } else {
            // m in 1..=7: build exactly as (1 + frac) * 2^(ilog2(m) - 9).
            let lz = 31 - m.leading_zeros(); // position of the top set bit, 0..=2
            let frac = (m << (23 - lz)) & 0x007F_FFFF;
            f32::from_bits(((lz + 127 - 9) << 23) | frac)
        }
    } else {
        f32::from_bits(((e + 127 - 7) << 23) | (m << 20))
    };
    Some(f32::from_bits(mag.to_bits() | sign))
}

/// gfx942 `v_cvt_f32_fp8` model: the byte is read as OCP-FNUZ (e4m3fnuz, bias 8,
/// subnormal `m * 2^-10`, `0x80` = NaN). `None` only for `0x80`. Applied to an
/// e4m3fn byte it yields exactly `fn(b)/2` for every byte except `0x80` and the
/// fn NaN codes (`0x7F`/`0xFF`, which are finite 240 / -240 under FNUZ).
pub const fn e4m3fnuz_to_f32(b: u8) -> Option<f32> {
    if b == 0x80 {
        return None;
    }
    let e = ((b >> 3) & 15) as u32;
    let m = (b & 7) as u32;
    let sign = ((b >> 7) as u32) << 31;
    let mag = if e == 0 {
        if m == 0 {
            0.0f32
        } else {
            let lz = 31 - m.leading_zeros();
            let frac = (m << (23 - lz)) & 0x007F_FFFF;
            f32::from_bits(((lz + 127 - 10) << 23) | frac)
        }
    } else {
        f32::from_bits(((e + 127 - 8) << 23) | (m << 20))
    };
    Some(f32::from_bits(mag.to_bits() | sign))
}

/// F32 -> BF16 bits, round-to-nearest-even on the bit pattern, exactly the
/// teacher formula `(u + 0x7FFF + ((u >> 16) & 1)) >> 16`. Defined for finite
/// inputs; callers refuse non-finite values and bf16 overflow first
/// ([`bf16_rne_checked`]).
#[inline]
pub const fn bf16_rne(x: f32) -> u16 {
    let u = x.to_bits();
    (u.wrapping_add(0x7FFF).wrapping_add((u >> 16) & 1) >> 16) as u16
}

/// [`bf16_rne`] that refuses non-finite input and results that round to inf
/// (the teacher raises in both cases).
#[inline]
pub fn bf16_rne_checked(x: f32) -> Option<u16> {
    if !x.is_finite() {
        return None;
    }
    let r = bf16_rne(x);
    ((r & 0x7F80) != 0x7F80).then_some(r)
}

/// bf16 bits -> F32 (high-16-bit shift).
#[inline]
pub const fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

const fn build_fn_lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = match e4m3fn_to_f32(i as u8) {
            Some(v) => v,
            None => f32::NAN,
        };
        i += 1;
    }
    t
}

/// 256-entry e4m3fn value table (NaN at `0x7F`/`0xFF`).
pub static E4M3FN_F32: [f32; 256] = build_fn_lut();

#[inline]
fn is_nan_code(b: u8) -> bool {
    b & 0x7F == 0x7F
}

/// Decode one element: `bf16_rne(f32(e4m3fn(b)) * s)` with the teacher's refusals.
#[inline]
fn decode_elem(b: u8, s: f32, index: usize) -> Result<u16, Fp8Error> {
    let v = E4M3FN_F32[b as usize];
    if v.is_nan() {
        return Err(Fp8Error::NanCode { index });
    }
    let p = v * s;
    if (p as f64) != (v as f64) * (s as f64) {
        return Err(Fp8Error::InexactProduct { index });
    }
    bf16_rne_checked(p).ok_or(Fp8Error::Overflow { index })
}

/// qt=56 row decode: every element uses the single scale. `out.len()` must equal
/// `codes.len()`. Same formula and refusals as [`dequant_b128_block`].
pub fn dequant_row_pt(codes: &[u8], s_bf16: u16, out: &mut [u16]) -> Result<(), Fp8Error> {
    if codes.len() != out.len() {
        return Err(Fp8Error::Shape);
    }
    let s = bf16_to_f32(s_bf16);
    if !s.is_finite() {
        return Err(Fp8Error::NonFiniteScale { index: 0 });
    }
    for (i, (&b, o)) in codes.iter().zip(out.iter_mut()).enumerate() {
        *o = decode_elem(b, s, i)?;
    }
    Ok(())
}

/// Teacher twin of `dequant_e4m3_block` with the 128x128 block: `codes` is
/// `u8[rows*k]` row-major, `s_inv` is bf16 bits `[ceil(rows/128) * ceil(k/128)]`
/// row-major. Returns bf16 bits `[rows*k]`. Refuses NaN codes, non-finite scales,
/// inexact products and bf16 overflow (error index = flat code / scale index).
pub fn dequant_b128_block(
    codes: &[u8],
    s_inv: &[u16],
    rows: usize,
    k: usize,
) -> Result<Vec<u16>, Fp8Error> {
    if Some(codes.len()) != b128_code_bytes(rows, k)
        || Some(s_inv.len()) != b128_scale_count(rows, k)
    {
        return Err(Fp8Error::Shape);
    }
    let (_, sc_cols) = b128_scale_dims(rows, k);
    for (i, &s) in s_inv.iter().enumerate() {
        if !bf16_to_f32(s).is_finite() {
            return Err(Fp8Error::NonFiniteScale { index: i });
        }
    }
    let mut out = vec![0u16; codes.len()];
    for r in 0..rows {
        let srow = (r / B128_BLOCK) * sc_cols;
        for c in 0..k {
            let idx = r * k + c;
            let s = bf16_to_f32(s_inv[srow + c / B128_BLOCK]);
            out[idx] = decode_elem(codes[idx], s, idx)?;
        }
    }
    Ok(out)
}

/// Decode one qt=55 expert block laid out as in the payload (`codes ‖ s_inv LE ‖
/// zero pad`). `block` must be at least `rows*k + 2*scales` bytes.
pub fn dequant_b128_expert_bytes(
    block: &[u8],
    rows: usize,
    k: usize,
) -> Result<Vec<u16>, Fp8Error> {
    let nc = b128_code_bytes(rows, k).ok_or(Fp8Error::Shape)?;
    let ns = b128_scale_count(rows, k).ok_or(Fp8Error::Shape)?;
    let end = ns
        .checked_mul(2)
        .and_then(|b| b.checked_add(nc))
        .ok_or(Fp8Error::Shape)?;
    let body = block.get(..end).ok_or(Fp8Error::Shape)?;
    let s_inv: Vec<u16> = body[nc..]
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    dequant_b128_block(&body[..nc], &s_inv, rows, k)
}

/// gfx942 fold decode of one element, modelling the hardware path exactly:
/// reinterpret the OCP byte as FNUZ (`v_cvt_f32_fp8`, value = `fn(b)/2`), with
/// `0x80` (FNUZ NaN) handled explicitly as `-0.0` so the BF16 sign of zero
/// matches the teacher, multiply by `f32(2 * s)`, then [`bf16_rne`].
///
/// `None` when the teacher would refuse: NaN codes `0x7F`/`0xFF` (FNUZ would
/// silently read them as +-240), non-finite or overflowing `2*s`, inexact
/// product, or bf16 overflow.
pub fn fnuz_fold_decode(b: u8, s_bf16: u16) -> Option<u16> {
    if is_nan_code(b) {
        return None;
    }
    let s = bf16_to_f32(s_bf16);
    if !s.is_finite() {
        return None;
    }
    let s2 = s * 2.0;
    if !s2.is_finite() {
        return None;
    }
    // 0x80: FNUZ NaN on hardware; the kernel decodes it explicitly as -0.0.
    let v = match e4m3fnuz_to_f32(b) {
        Some(v) => v,
        None => -0.0f32,
    };
    let p = v * s2;
    if (p as f64) != (v as f64) * (s2 as f64) {
        return None;
    }
    bf16_rne_checked(p)
}

/// Result of [`scan_codes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodeScan {
    /// Count of NaN codes (`0x7F`, `0xFF`).
    pub nan: u64,
    /// Count of negative zeros (`0x80`).
    pub neg_zero: u64,
}

/// Count NaN codes and `-0` codes in a code stream.
pub fn scan_codes(codes: &[u8]) -> CodeScan {
    let mut s = CodeScan::default();
    for &b in codes {
        if is_nan_code(b) {
            s.nan += 1;
        } else if b == 0x80 {
            s.neg_zero += 1;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scale sweep fixture (see `tests/fixtures/fp8_e4m3/gen_golden.py`):
    /// little-endian records `{u16 scale_bits, u16[254] teacher_bf16}` for the 254
    /// finite codes in ascending byte order (0x7F and 0xFF skipped).
    const GOLDEN: &[u8] = include_bytes!("../tests/fixtures/fp8_e4m3/golden.bin");
    const REC: usize = 2 + 254 * 2;

    fn finite_codes() -> Vec<u8> {
        (0u16..256)
            .map(|b| b as u8)
            .filter(|&b| !is_nan_code(b))
            .collect()
    }

    fn records() -> Vec<(u16, Vec<u16>)> {
        assert_eq!(GOLDEN.len() % REC, 0, "fixture size");
        GOLDEN
            .chunks_exact(REC)
            .map(|r| {
                let s = u16::from_le_bytes([r[0], r[1]]);
                let v = r[2..]
                    .chunks_exact(2)
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect();
                (s, v)
            })
            .collect()
    }

    #[test]
    fn fp8_fixture_covers_the_sweep() {
        let recs = records();
        assert!(recs.len() >= 40, "fixture has {} scales", recs.len());
        let scales: Vec<u16> = recs.iter().map(|r| r.0).collect();
        for want in [1.0f32, 2f32.powi(-20), 2f32.powi(-9), 0.0017, 2f32.powi(10)] {
            assert!(
                scales.contains(&bf16_rne(want)),
                "fixture missing scale {want}"
            );
        }
    }

    #[test]
    fn fp8_e4m3fn_value_table() {
        assert_eq!(e4m3fn_to_f32(0x7F), None);
        assert_eq!(e4m3fn_to_f32(0xFF), None);
        assert_eq!(e4m3fn_to_f32(0x00).unwrap().to_bits(), 0);
        assert_eq!(e4m3fn_to_f32(0x80).unwrap().to_bits(), 0x8000_0000);
        assert_eq!(e4m3fn_to_f32(0x01).unwrap(), 2f32.powi(-9));
        assert_eq!(e4m3fn_to_f32(0x07).unwrap(), 7.0 * 2f32.powi(-9));
        assert_eq!(e4m3fn_to_f32(0x08).unwrap(), 2f32.powi(-6));
        assert_eq!(e4m3fn_to_f32(0x38).unwrap(), 1.0);
        assert_eq!(e4m3fn_to_f32(0x7E).unwrap(), 448.0);
        assert_eq!(e4m3fn_to_f32(0xFE).unwrap(), -448.0);
        // fnuz read of the same byte is exactly half, except 0x80 / the fn NaNs.
        for b in 0u16..=255 {
            let b = b as u8;
            if b == 0x80 || is_nan_code(b) {
                continue;
            }
            let fnv = e4m3fn_to_f32(b).unwrap();
            let fz = e4m3fnuz_to_f32(b).unwrap();
            assert_eq!(
                fz.to_bits() << 1,
                (fnv * 0.5).to_bits() << 1,
                "byte {b:#04x}"
            );
            assert_eq!(fz, fnv * 0.5, "byte {b:#04x}");
        }
        assert_eq!(e4m3fnuz_to_f32(0x80), None);
        // The fn NaN codes are finite +-240 under FNUZ: why they are refused.
        assert_eq!(e4m3fnuz_to_f32(0x7F), Some(240.0));
        assert_eq!(e4m3fnuz_to_f32(0xFF), Some(-240.0));
    }

    #[test]
    fn fp8_bf16_rne_ties_to_even() {
        assert_eq!(bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80); // tie, even stays
        assert_eq!(bf16_rne(f32::from_bits(0x3F81_8000)), 0x3F82); // tie, odd rounds up
        assert_eq!(bf16_rne(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(bf16_rne(-0.0), 0x8000);
    }

    #[test]
    fn fp8_block_decode_matches_teacher_bits() {
        let codes = finite_codes();
        assert_eq!(codes.len(), 254);
        for (s, want) in records() {
            // k=254 spans two 128-wide scale blocks; both carry the same scale.
            let got = dequant_b128_block(&codes, &[s, s], 1, 254)
                .unwrap_or_else(|e| panic!("scale {s:#06x}: {e}"));
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(g, w, "scale {s:#06x} code {:#04x}", codes[i]);
            }
        }
    }

    #[test]
    fn fp8_pt_row_decode_matches_teacher_bits() {
        let codes = finite_codes();
        for (s, want) in records() {
            let mut out = vec![0u16; codes.len()];
            dequant_row_pt(&codes, s, &mut out).unwrap();
            assert_eq!(out, want, "scale {s:#06x}");
        }
    }

    #[test]
    fn fp8_fnuz_fold_matches_teacher_bits() {
        let codes = finite_codes();
        for (s, want) in records() {
            for (i, &b) in codes.iter().enumerate() {
                assert_eq!(
                    fnuz_fold_decode(b, s),
                    Some(want[i]),
                    "scale {s:#06x} code {b:#04x}"
                );
            }
        }
    }

    #[test]
    fn fp8_neg_zero_keeps_sign_bit() {
        // -0 * (+s) = -0 -> bf16 0x8000 in teacher and fold; a naive 0x80->0x00 remap gives 0x0000.
        let one = 0x3F80u16;
        assert_eq!(fnuz_fold_decode(0x80, one), Some(0x8000));
        assert_eq!(fnuz_fold_decode(0x00, one), Some(0x0000));
        let got = dequant_b128_block(&[0x80], &[one], 1, 1).unwrap();
        assert_eq!(got, vec![0x8000]);
    }

    #[test]
    fn fp8_nan_codes_refused() {
        let one = 0x3F80u16;
        for b in [0x7Fu8, 0xFF] {
            assert_eq!(fnuz_fold_decode(b, one), None);
            assert_eq!(
                dequant_b128_block(&[0x38, b], &[one], 1, 2),
                Err(Fp8Error::NanCode { index: 1 })
            );
            let mut out = [0u16; 1];
            assert_eq!(
                dequant_row_pt(&[b], one, &mut out),
                Err(Fp8Error::NanCode { index: 0 })
            );
            assert_eq!(e4m3fn_to_f32(b), None);
        }
        assert_eq!(
            scan_codes(&[0x7F, 0xFF, 0x80, 0x80, 0x80, 0x00, 0x38]),
            CodeScan {
                nan: 2,
                neg_zero: 3
            }
        );
    }

    #[test]
    fn fp8_nonfinite_scale_refused() {
        for s in [0x7F80u16, 0xFF80, 0x7FC0] {
            assert_eq!(
                dequant_b128_block(&[0x38], &[s], 1, 1),
                Err(Fp8Error::NonFiniteScale { index: 0 })
            );
            assert_eq!(fnuz_fold_decode(0x38, s), None);
        }
    }

    #[test]
    fn fp8_layout_geometry() {
        // Qwen4 routed experts (contract): gate_up R=1280 K=2560, down R=2560 K=640.
        assert_eq!(b128_block_stride(1280, 2560), Some(3_277_312));
        assert_eq!(b128_block_stride(2560, 640), Some(1_638_656));
        assert_eq!(b128_scale_dims(1280, 2560), (10, 20));
        assert_eq!(b128_tensor_bytes(3, 1280, 2560), Some(3 * 3_277_312));
        assert_eq!(
            b128_scales_offset(2, 1280, 2560),
            Some(2 * 3_277_312 + 1280 * 2560)
        );
        // ragged geometry: 130 x 130 -> 2x2 scales, 16900 + 8 = 16908 -> 17152.
        assert_eq!(b128_block_stride(130, 130), Some(17_152));
        assert_eq!(b128_block_stride(1, 1), Some(256));
        assert_eq!(b128_block_stride(0, 0), Some(0));
        assert_eq!(b128_tensor_bytes(usize::MAX, 2, 2), None);
        assert_eq!(pt_tensor_bytes(2_500_012, 160), Some(16 + 2_500_012 * 160));
        assert_eq!(pt_row_offset(3, 160), Some(16 + 480));
        let h = pt_header(0x3F80);
        assert_eq!(&h[..2], &[0x80, 0x3F]);
        assert_eq!(pt_header_scale(&h), Ok(0x3F80));
        let mut bad = h;
        bad[9] = 1;
        assert_eq!(pt_header_scale(&bad), Err(Fp8Error::Shape));
    }

    #[test]
    fn fp8_expert_bytes_roundtrip_ragged_blocks() {
        // 130 x 131: 2x2 scale grid with distinct scales; payload padded to stride.
        let (rows, k) = (130usize, 131usize);
        let codes: Vec<u8> = (0..rows * k)
            .map(|i| finite_codes()[(i * 7) % 254])
            .collect();
        let s_inv: Vec<u16> = vec![0x3F80, 0x3E80, 0x4000, 0x3C00];
        let stride = b128_block_stride(rows, k).unwrap();
        let mut blob = vec![0u8; stride];
        blob[..codes.len()].copy_from_slice(&codes);
        for (i, s) in s_inv.iter().enumerate() {
            blob[codes.len() + 2 * i..codes.len() + 2 * i + 2].copy_from_slice(&s.to_le_bytes());
        }
        let want = dequant_b128_block(&codes, &s_inv, rows, k).unwrap();
        assert_eq!(dequant_b128_expert_bytes(&blob, rows, k).unwrap(), want);
        // block boundaries pick the right scale: row 129 col 130 uses s_inv[3].
        let idx = 129 * k + 130;
        assert_eq!(
            want[idx],
            decode_elem(codes[idx], bf16_to_f32(s_inv[3]), idx).unwrap()
        );
        assert_eq!(want[idx], fnuz_fold_decode(codes[idx], s_inv[3]).unwrap());
        assert_eq!(
            dequant_b128_block(&codes, &s_inv[..3], rows, k),
            Err(Fp8Error::Shape)
        );
    }
}
