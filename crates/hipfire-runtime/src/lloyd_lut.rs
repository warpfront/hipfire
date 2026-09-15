// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
//! MQ4G256V2-Lloyd (qt=52) per-tensor codebook LUT builders (host side).
//!
//! Wire contract: qt=52 is byte-identical to MQ4G256V2 on the wire (136 B/group:
//! dual fp16 half-grids + 128 B nibbles). Nibble `q` decodes through a PER-TENSOR
//! 16-level CENTERED codebook:
//!
//! ```text
//! w = sc·C[q] + zp'      with  C[q] = L[q] − 7.5,  zp' = fp16(zp + 7.5·sc)
//! ```
//!
//! * `L` (f32[16] in [0,15] units, E4M3-snapped by the constrained Lloyd fit)
//!   arrives in the `<stem>.lloyd_levels.weight` F32 sidecar.
//! * The loader centers `C = L − 7.5` (bit-exact in f32 for on-grid sidecars)
//!   and builds two kernel-arg LUTs: [`lloyd_luts_from_levels`].
//! * The loader rewrites the weight headers `zp → zp'` BEFORE upload
//!   ([`apply_lloyd_centering`]); `sc` bytes are untouched. The file keeps the
//!   uncentered zp, so file md5 ≠ GPU bytes for qt=52 — intentional, logged.
//!
//! Why centered: E4M3 (the gfx12 FP8 WMMA operand format) has only integer steps
//! at magnitudes ≥ 8, so an uncentered codebook (levels up to 14.7, mass peak
//! near 7.5) rounds with up to 0.44 absolute error and even collapses two levels
//! onto one byte — measured −3.5% MSE vs uniform on real Qwen3.8-27B tensors.
//! Centered magnitudes (≤ 7.5, mass peak at E4M3-zero where steps are 2^-9)
//! recover +13.9% with the constrained (E4M3-snapped + polished) fit.
//! A per-tensor power-of-two fold was measured and rejected: the E4M3 grid is
//! logarithmic, so folds are scale-invariant (all k give identical error).
//!
//! E4M3 convention here is bias-7 (matches the gfx12 FP8 prefill kernels'
//! hardcoded `C0..C3` nibble table: q=8 → 0x50 = 2^(10−7)·1). The loader's
//! [`e4m3_bias7_encode_rn`] reproduces that table exactly for integers 0..15.

/// Centering constant: codebook levels live in [0,15], centered to [-7.5,7.5].
pub const LLOYD_CENTER: f32 = 7.5;

/// Decode one E4M3 (bias-7) byte to f32. `e == 15` codes (NaN/Inf) are never
/// produced by [`e4m3_bias7_encode_rn`] and decode here as +240·(mantissa sign
/// ignored) — callers must not feed them.
pub fn e4m3_bias7_decode(b: u8) -> f32 {
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    let mag = if e == 0 {
        m * 2.0f32.powi(-9)
    } else {
        2.0f32.powi(e - 7) * (1.0 + m / 8.0)
    };
    if b & 0x80 == 0 { mag } else { -mag }
}

/// Encode f32 to E4M3 (bias-7) with round-to-nearest. Exhaustive 256-code
/// search: ~256 comparisons per level, 16 levels per tensor — negligible at
/// load. `e == 15` codes are skipped (NaN/Inf must never reach FP8 WMMA).
/// Non-positive inputs map to +0.0 (code 0x00); negative centered levels use
/// the sign bit (codes 0x80+). Ties resolve to the lowest code; sidecar levels
/// are on-grid so ties never occur on the load path (round-trip exact).
pub fn e4m3_bias7_encode_rn(v: f32) -> u8 {
    if !(v > 0.0) {
        // Covers v == 0.0 and negatives: negatives are handled with the sign
        // bit below; this arm is +0.0 / NaN-guard only.
        if v == 0.0 {
            return 0x00;
        }
    }
    let sign = if v < 0.0 { 0x80u8 } else { 0x00 };
    let mag = v.abs();
    let mut best: u8 = 0x00;
    let mut best_err = f32::INFINITY;
    for code in 0u16..256 {
        let c = code as u8;
        if (c >> 3) & 0xF == 15 {
            continue; // NaN/Inf — never a LUT entry
        }
        let d = e4m3_bias7_decode(c & 0x7F);
        let err = (d - mag).abs();
        if err < best_err {
            best_err = err;
            best = c & 0x7F;
        }
    }
    best | sign
}

/// Build the two kernel-arg LUTs from sidecar levels (f32[16], [0,15] units).
///
/// * `e4m3`: `[u32;4]` — centered levels `L−7.5` rounded to E4M3 bytes, 4 bytes
///   packed little-endian per dword (byte 4j+k = level 4j+k). Replaces the
///   `C0..C3` immediates in the `HIPFIRE_FP8_LUT_ARG` prefill kernels.
/// * `f16`: `[u32;8]` — centered levels as f16 bits, 2 per dword little-endian
///   (dword j = level 2j | level 2j+1 << 16). Feeds the GEMV/decode kernels.
///   f16 is exact for every E4M3-representable value, so with an on-grid
///   (constrained-fit) sidecar both LUTs decode to bit-identical f32 values.
pub fn lloyd_luts_from_levels(levels: &[f32; 16]) -> ([u32; 4], [u32; 8]) {
    let mut e4 = [0u32; 4];
    let mut h = [0u32; 8];
    for (i, &l) in levels.iter().enumerate() {
        let c = l - LLOYD_CENTER;
        let b = e4m3_bias7_encode_rn(c);
        e4[i / 4] |= (b as u32) << ((i % 4) * 8);
        // RN-even via the half crate (NOT the truncating llama::f32_to_f16:
        // the zp' error budget was measured with RN and every basis point of
        // codebook precision shows up in KLD).
        let bits = half::f16::from_f32(c).to_bits();
        h[i / 2] |= (bits as u32) << ((i % 2) * 16);
    }
    (e4, h)
}

/// Validate + parse a `lloyd_levels` sidecar: F32 (qt=2), shape [16], 64 bytes.
/// Returns the 16 levels in [0,15] units. Anything else is a hard error —
/// a malformed codebook must fail the load, never silently decode uniform.
pub fn lloyd_levels_from_sidecar(
    quant_type: u8,
    shape: &[u32],
    data: &[u8],
) -> Result<[f32; 16], String> {
    if quant_type != 2 {
        return Err(format!(
            "lloyd_levels sidecar has quant_type={quant_type} (expected 2=F32)"
        ));
    }
    if shape != [16] {
        return Err(format!(
            "lloyd_levels sidecar shape mismatch ({shape:?} vs expected [16])"
        ));
    }
    if data.len() != 64 {
        return Err(format!(
            "lloyd_levels sidecar size mismatch ({} vs expected 64)",
            data.len()
        ));
    }
    let mut levels = [0f32; 16];
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        levels[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(levels)
}

/// Sidecar name for a weight tensor: strip trailing `.weight`, append
/// `.lloyd_levels.weight`. Same strip rule as the AWQ sidecar.
pub fn lloyd_sidecar_name(weight_name: &str) -> String {
    match weight_name.strip_suffix(".weight") {
        Some(stem) => format!("{stem}.lloyd_levels.weight"),
        None => format!("{weight_name}.lloyd_levels.weight"),
    }
}

/// Rewrite qt=52 weight headers in place: `zp → fp16(zp + 7.5·sc)` per half,
/// `sc` bytes untouched. `data` must be exactly `m·(k/256)·136` bytes —
/// anything else (notably the stale +32B-prefix prototype layout) is rejected.
/// Layout per 136 B group: `[0..2)` sc0 fp16, `[2..4)` zp0 fp16, `[4..6)` sc1,
/// `[6..8)` zp1, `[8..136)` nibbles.
pub fn apply_lloyd_centering(data: &mut [u8], m: usize, k: usize) -> Result<(), String> {
    if k % 256 != 0 {
        return Err(format!(
            "MQ4G256V2Lloyd has K={k} but requires K%256==0"
        ));
    }
    let expected = m * (k / 256) * 136;
    if data.len() != expected {
        return Err(format!(
            "MQ4G256V2Lloyd blob length mismatch: expected {expected}, got {} \
             (stale +32B-prefix artifacts are rejected: re-quantize to pure 136 B/group)",
            data.len()
        ));
    }
    for group in data.chunks_exact_mut(136) {
        for half in 0..2 {
            let o = half * 4;
            let sc = half::f16::from_bits(u16::from_le_bytes([group[o], group[o + 1]])).to_f32();
            let zp = half::f16::from_bits(u16::from_le_bytes([group[o + 2], group[o + 3]])).to_f32();
            let zp_prime = zp + LLOYD_CENTER * sc;
            let bits = half::f16::from_f32(zp_prime).to_bits().to_le_bytes();
            group[o + 2] = bits[0];
            group[o + 3] = bits[1];
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel's hardcoded C0..C3 nibble table, byte per integer level.
    const KERNEL_TABLE: [u8; 16] = [
        0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E, 0x50, 0x51, 0x52, 0x53,
        0x54, 0x55, 0x56, 0x57,
    ];

    #[test]
    fn e4m3_encode_reproduces_kernel_integer_table() {
        // Integers 0..15 must encode to the exact bytes the uniform FP8 kernel
        // bakes in as C0..C3 — the LUT-arg kernel decodes through the same
        // hardware path, so any divergence here is a silent 2x-scale-class bug.
        for q in 0..16 {
            let b = e4m3_bias7_encode_rn(q as f32);
            assert_eq!(b, KERNEL_TABLE[q], "integer level {q}");
            assert_eq!(e4m3_bias7_decode(b), q as f32, "integer decode {q}");
        }
    }

    #[test]
    fn uniform_codebook_round_trips_through_centered_luts() {
        let levels: [f32; 16] = core::array::from_fn(|i| i as f32);
        let (e4, h) = lloyd_luts_from_levels(&levels);
        for i in 0..16 {
            let c = i as f32 - LLOYD_CENTER;
            let b = (e4[i / 4] >> ((i % 4) * 8)) as u8;
            assert_eq!(e4m3_bias7_decode(b), c, "e4 level {i}");
            let bits = (h[i / 2] >> ((i % 2) * 16)) as u16;
            assert_eq!(half::f16::from_bits(bits).to_f32(), c, "f16 level {i}");
        }
    }

    #[test]
    fn constrained_codebook_round_trips_byte_exact() {
        // Polished constrained codebook (measured §co-design): every level is
        // E4M3-representable centered, so encode∘decode is the identity and
        // the f16 LUT agrees with the E4M3 LUT bit-for-bit in f32.
        let levels: [f32; 16] = [
            0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.25, 7.09375, 7.90625, 8.75, 9.5,
            10.25, 11.25, 12.5, 13.5, 15.0,
        ];
        let (e4, h) = lloyd_luts_from_levels(&levels);
        for i in 0..16 {
            let c = levels[i] - LLOYD_CENTER;
            let b = (e4[i / 4] >> ((i % 4) * 8)) as u8;
            assert_eq!(e4m3_bias7_decode(b), c, "constrained e4 level {i}");
            let bits = (h[i / 2] >> ((i % 2) * 16)) as u16;
            assert_eq!(
                half::f16::from_bits(bits).to_f32(),
                c,
                "constrained f16 level {i}"
            );
        }
    }

    #[test]
    fn centering_rewrites_zp_and_keeps_sc() {
        // One group: sc0=0.5/zp0=-3.0, sc1=2.0/zp1=1.0.
        let mut data = vec![0u8; 136];
        data[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        data[2..4].copy_from_slice(&half::f16::from_f32(-3.0).to_bits().to_le_bytes());
        data[4..6].copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes());
        data[6..8].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        apply_lloyd_centering(&mut data, 1, 256).unwrap();
        let sc0 = half::f16::from_bits(u16::from_le_bytes([data[0], data[1]])).to_f32();
        let zp0 = half::f16::from_bits(u16::from_le_bytes([data[2], data[3]])).to_f32();
        let sc1 = half::f16::from_bits(u16::from_le_bytes([data[4], data[5]])).to_f32();
        let zp1 = half::f16::from_bits(u16::from_le_bytes([data[6], data[7]])).to_f32();
        assert_eq!(sc0, 0.5);
        assert_eq!(sc1, 2.0);
        assert_eq!(zp0, half::f16::from_f32(-3.0 + 7.5 * 0.5).to_f32());
        assert_eq!(zp1, half::f16::from_f32(1.0 + 7.5 * 2.0).to_f32());
    }

    #[test]
    fn centering_rejects_bad_layouts() {
        // Stale +32B-prefix layout and K%256 violations fail closed.
        let mut data = vec![0u8; 136 + 32];
        assert!(apply_lloyd_centering(&mut data, 1, 256).is_err());
        let mut data = vec![0u8; 136];
        assert!(apply_lloyd_centering(&mut data, 1, 128).is_err());
    }

    #[test]
    fn sidecar_validation_rejects_malformed() {
        let good = vec![0u8; 64];
        assert!(lloyd_levels_from_sidecar(2, &[16], &good).is_ok());
        assert!(lloyd_levels_from_sidecar(1, &[16], &good).is_err());
        assert!(lloyd_levels_from_sidecar(2, &[8], &good).is_err());
        assert!(lloyd_levels_from_sidecar(2, &[16], &vec![0u8; 32]).is_err());
    }

    #[test]
    fn sidecar_name_mirrors_awq_strip_rule() {
        assert_eq!(
            lloyd_sidecar_name("model.layers.0.q_proj.weight"),
            "model.layers.0.q_proj.lloyd_levels.weight"
        );
        assert_eq!(
            lloyd_sidecar_name("bare"),
            "bare.lloyd_levels.weight"
        );
    }
}
