// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

use std::collections::HashMap;

use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32};
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

pub(crate) fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => data
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "BF16" => data
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        "F32" => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        other => panic!("unsupported dtype: {other}"),
    }
}

// ─── FP8 E4M3 + UE8M0-scale dequant (DeepSeek V4 Flash) ─────────────────────
//
// DeepSeek V4 ships its quantized weights as paired safetensors entries:
//   <name>.weight  : I8 raw bytes, each byte one FP8 E4M3 value
//   <name>.scale   : F8_E8M0 raw bytes, each byte one UE8M0 exponent
//
// The block shape on DeepSeek V4-shipped checkpoints is [1, 16] (per-row, 16-col
// groups) — i.e. scale shape `[R, C/16]` for weight shape `[R, C]` — even
// though the `quantization_config.weight_block_size` in `config.json`
// reads `[128, 128]`. We verify the implied block from the actual scale
// shape rather than the config to avoid being misled.
//
// E4M3 format (1 sign + 4 exp + 3 mant, bias=7):
//   - exp=0, mant=0      → ±0
//   - exp=0, mant!=0     → denormal: (-1)^s · 2^-6 · (mant/8)
//   - exp=15, mant=7     → NaN (only one NaN code in E4M3)
//   - otherwise normal:  (-1)^s · 2^(exp-7) · (1 + mant/8)
//
// UE8M0 format (8-bit unsigned exponent only, no sign, no mantissa):
//   scale = 2^(byte - 127)
//
// Returns f32 in row-major order matching `weight_shape`.

pub(crate) fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0xf && mant == 7.0 {
        // E4M3's single NaN code — treat as 0 for quant purposes (clean
        // bytes flagged elsewhere; downstream MQ-family quant has no
        // NaN handling and would emit garbage).
        return 0.0;
    }
    if exp == 0 {
        if mant == 0.0 {
            return 0.0;
        }
        return sign * (2.0f32.powi(-6)) * (mant / 8.0);
    }
    sign * (2.0f32.powi(exp - 7)) * (1.0 + mant / 8.0)
}

#[inline]
pub(crate) fn ue8m0_to_scale(byte: u8) -> f32 {
    // 2^(exp - 127). Cheap: shift into f32's exponent field directly.
    // byte=127 → 1.0, byte=0 → 2^-127 (subnormal range — fine, we return 0
    // implicitly through f32 rounding), byte=255 → +inf (won't appear on
    // well-formed checkpoints; if it does we propagate inf and the
    // downstream MQ quant will produce extreme outputs detectable in QA).
    2.0f32.powi(byte as i32 - 127)
}

/// Helper for the main quantize loop: convert one tensor's raw bytes to
/// f32, transparently handling DeepSeek V4's FP8 E4M3 + UE8M0-scale pairs.
///
/// If `meta.dtype == "I8"` and a scale sibling is registered in
/// `fp8_scale_for[weight_name]`, dequant the pair. Otherwise fall back
/// to `to_f32(data, dtype)`.
pub(crate) fn tensor_to_f32_with_optional_fp8_scale(
    name: &str,
    raw_data: &[u8],
    meta: &TensorMeta,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
) -> Vec<f32> {
    // FP8 E4M3 + UE8M0 paired storage (DeepSeek V4). The dtype tag is either
    // `I8` (older safetensors writer) or `F8_E4M3` (newer); both
    // store identical E4M3 bytes, so the dequant math is the same.
    if (meta.dtype == "I8" || meta.dtype == "F8_E4M3") && fp8_scale_for.contains_key(name) {
        let (sfi, sname) = &fp8_scale_for[name];
        let (smeta, sbytes) = st_files[*sfi]
            .tensor_data(sname)
            .unwrap_or_else(|| panic!("FP8 scale tensor missing: {sname}"));
        if smeta.dtype == "F8_E8M0" {
            return dequantize_e4m3_ue8m0_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else if smeta.dtype == "U8"
            && meta.dtype == "F8_E4M3"
            && smeta.shape.len() == 2
            && meta.shape.len() == 2
            && smeta.shape[0] == meta.shape[0]
            && smeta.shape[1] * 32 == meta.shape[1]
        {
            // MXFP8 (`quantization_config.weight_block_size == [1,32]`): the
            // `.weight_scale_inv` U8 tensor is RAW E8M0 exponent bytes, not an
            // F32 multiplier. Shape-implied [1,32] block; strict validation.
            return dequantize_mxfp8_e8m0_strict(raw_data, &meta.shape, sbytes, &smeta.shape, 32)
                .unwrap_or_else(|e| panic!("MXFP8 dequant of {name}: {e}"));
        } else if smeta.dtype == "F32" {
            // MiniMax-M2: e4m3 + F32 block-[128,128] weight_scale_inv (multiply).
            return dequantize_e4m3_f32scale_to_f32(raw_data, &meta.shape, sbytes, &smeta.shape);
        } else {
            panic!(
                "expected F8_E8M0 or F32 scale for {name}, got {}",
                smeta.dtype
            );
        }
    }
    if meta.dtype == "I8" {
        panic!(
            "tensor {name} has dtype I8 but no .scale sibling registered \
                — unexpected on a non-DeepSeek V4 checkpoint."
        );
    }
    to_f32(raw_data, &meta.dtype)
}

/// Convert one E2M1 nibble (4-bit FP: 1 sign + 2 exp + 1 mantissa, bias=1) to f32.
///
/// E2M1 codes (signed magnitude on the 3 low bits, high bit is sign):
///   nibble & 0x7 → magnitude  → value
///   0  → 0          → 0.0
///   1  → denorm 0.5 → 0.5
///   2  → normal 1.0 → 1.0
///   3  → normal 1.5 → 1.5
///   4  → normal 2.0 → 2.0
///   5  → normal 3.0 → 3.0
///   6  → normal 4.0 → 4.0
///   7  → normal 6.0 → 6.0
/// Sign bit: bit 3 (0x8).
///
/// Total range: ±6.0. Per OCP MX spec (FP4 E2M1).
#[inline]
pub(crate) fn e2m1_to_f32(nibble: u8) -> f32 {
    // Lookup table for the 8 magnitude codes; sign is applied after.
    pub(crate) const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n = (nibble & 0x0f) as usize;
    let mag = MAG[n & 0x7];
    if (n & 0x8) != 0 {
        -mag
    } else {
        mag
    }
}

/// Dequantize a paired E2M1 weight + UE8M0 scale tensor to f32.
///
/// `storage_shape` is the byte-shape from safetensors: [rows, cols_stored]
/// where `cols_stored = logical_cols / 2` (two E2M1 nibbles per byte; low
/// nibble is the even logical column, high nibble is the odd column).
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size in
/// logical-element units is [rows / scale_rows, logical_cols / scale_cols].
/// Per DeepSeek V4 spec (model.py:132-137): block 32 along logical K → scale_cols
/// = logical_cols / 32.
///
/// Returns row-major f32 of LOGICAL shape, length = rows * cols_stored * 2.
pub(crate) fn dequantize_e2m1_ue8m0_to_f32(
    weight_bytes: &[u8],
    storage_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(
        storage_shape.len(),
        2,
        "expected 2D storage shape, got {:?}",
        storage_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale shape, got {:?}",
        scale_shape
    );
    let (rows, cols_stored) = (storage_shape[0], storage_shape[1]);
    let logical_cols = cols_stored * 2;
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols_stored,
        "FP4 weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "FP4 scale byte count mismatch");
    assert!(
        rows % sr == 0 && logical_cols % sc == 0,
        "FP4 scale shape {:?} doesn't tile logical weight shape [{}, {}]",
        scale_shape,
        rows,
        logical_cols
    );
    let block_rows = rows / sr;
    let block_cols_logical = logical_cols / sc;

    let mut out = vec![0.0f32; rows * logical_cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols_logical {
                    let c = sc_j * block_cols_logical + dj;
                    // c is the LOGICAL column. Byte storing it sits at
                    // (c / 2); low nibble for even c, high nibble for odd.
                    let byte = weight_bytes[r * cols_stored + (c / 2)];
                    let nibble = if (c & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                    out[r * logical_cols + c] = e2m1_to_f32(nibble) * scale;
                }
            }
        }
    }
    (out, vec![rows, logical_cols])
}

/// Dequantize a paired E4M3 weight + UE8M0 scale tensor to f32.
///
/// `weight_shape` is the LOGICAL [rows, cols] of the weight matrix.
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size is
/// [weight_rows / scale_rows, weight_cols / scale_cols].
///
/// Returns row-major f32, length = rows * cols.
pub(crate) fn dequantize_e4m3_ue8m0_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale,  got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(scale_bytes.len(), sr * sc, "scale  byte count mismatch");
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;

    let mut out = vec![0.0f32; rows * cols];
    // Each (sr_i, sc_j) scale governs the block weight[sr_i*block_rows .. (sr_i+1)*block_rows,
    //                                                  sc_j*block_cols .. (sc_j+1)*block_cols].
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    let b = weight_bytes[r * cols + c];
                    out[r * cols + c] = e4m3_to_f32(b) * scale;
                }
            }
        }
    }
    out
}

/// Strict E4M3 byte → f32. The single NaN code (`S.1111.111`) is an error:
/// unlike [`e4m3_to_f32`] it never decays to 0.
pub(crate) fn e4m3_to_f32_strict(byte: u8) -> Result<f32, String> {
    if (byte & 0x7f) == 0x7f {
        return Err(format!("E4M3 NaN code 0x{byte:02x}"));
    }
    Ok(e4m3_to_f32(byte))
}

/// Exact `2^(byte-127)` for an E8M0 exponent byte (`byte == 0` is the f32
/// subnormal `2^-127`). `0xFF` is the E8M0 NaN code and is refused.
pub(crate) fn e8m0_to_scale_strict(byte: u8) -> Result<f32, String> {
    match byte {
        255 => Err("E8M0 exponent byte 255 (NaN code)".to_string()),
        0 => Ok(f32::from_bits(0x0040_0000)),
        b => Ok(f32::from_bits((b as u32) << 23)),
    }
}

/// Dequantize an MXFP8 `[N,K]` E4M3 weight with raw E8M0 exponent bytes
/// `[N,K/block_k]` (OCP MX; checkpoint `quantization_config.weight_block_size`
/// `[1, block_k]`): `value = float32(E4M3) * 2^(u8 - 127)`.
///
/// The scale tensor is raw exponent bytes (`U8` or `F8_E8M0` storage), never
/// an F32 multiplier and never a block-128 grid. Every code is validated:
/// E4M3 NaN, E8M0 255 and any non-finite product are errors (no NaN→0).
pub(crate) fn dequantize_mxfp8_e8m0_strict(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
    block_k: usize,
) -> Result<Vec<f32>, String> {
    if weight_shape.len() != 2 || scale_shape.len() != 2 || block_k == 0 {
        return Err(format!(
            "MXFP8 needs 2-D weight/scale, got {weight_shape:?} / {scale_shape:?}"
        ));
    }
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    if cols % block_k != 0 || scale_shape != [rows, cols / block_k] {
        return Err(format!(
            "MXFP8 scale shape {scale_shape:?} is not [{rows}, {}] for weight {weight_shape:?} \
             with block [1,{block_k}]",
            cols / block_k.max(1)
        ));
    }
    if weight_bytes.len() != rows * cols || scale_bytes.len() != scale_shape[0] * scale_shape[1] {
        return Err("MXFP8 byte count does not match shape".to_string());
    }
    let sc = cols / block_k;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for j in 0..sc {
            let scale = e8m0_to_scale_strict(scale_bytes[r * sc + j])
                .map_err(|e| format!("scale[{r},{j}]: {e}"))?;
            for c in j * block_k..(j + 1) * block_k {
                let v = e4m3_to_f32_strict(weight_bytes[r * cols + c])
                    .map_err(|e| format!("weight[{r},{c}]: {e}"))? as f64
                    * scale as f64;
                let v32 = v as f32;
                if !v32.is_finite() {
                    return Err(format!("weight[{r},{c}]: non-finite dequantized value"));
                }
                out[r * cols + c] = v32;
            }
        }
    }
    Ok(out)
}

/// Dequantize FP8 E4M3 weights paired with an F32 block-[128,128]
/// `weight_scale_inv` (MiniMax-M2 / DeepSeek-V3 fp8 block quant). Dequant is
/// MULTIPLY: `out = e4m3_to_f32(b) * scale` (the stored scale ≈ amax/448 per
/// block, verified ~5e-4 on the real checkpoint). Scale tile is [rows/sr, cols/sc]
/// = [128, 128] on MiniMax.
pub(crate) fn dequantize_e4m3_f32scale_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(
        weight_shape.len(),
        2,
        "expected 2D weight, got {:?}",
        weight_shape
    );
    assert_eq!(
        scale_shape.len(),
        2,
        "expected 2D scale, got {:?}",
        scale_shape
    );
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(
        weight_bytes.len(),
        rows * cols,
        "weight byte count mismatch"
    );
    assert_eq!(
        scale_bytes.len(),
        sr * sc * 4,
        "f32 scale byte count mismatch"
    );
    assert!(
        rows % sr == 0 && cols % sc == 0,
        "scale shape {:?} doesn't tile weight shape {:?}",
        scale_shape,
        weight_shape
    );
    let block_rows = rows / sr;
    let block_cols = cols / sc;
    let mut out = vec![0.0f32; rows * cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let so = (sr_i * sc + sc_j) * 4;
            let scale = f32::from_le_bytes([
                scale_bytes[so],
                scale_bytes[so + 1],
                scale_bytes[so + 2],
                scale_bytes[so + 3],
            ]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    out[r * cols + c] = e4m3_to_f32(weight_bytes[r * cols + c]) * scale;
                }
            }
        }
    }
    out
}

/// Dequantize an Inkling-2 NVFP4 expert tensor: 1x16 blocks of E2M1 weights
/// with per-block E4M3 scales, then per-expert `amax / denominator`. Mirrors
/// `_dequant_expert_chunk` in the merged llama.cpp `conversion/inkling2.py`.
///
/// `weight_shape` is the packed storage shape `[experts, rows, cols/2]`;
/// the returned logical shape is `[experts, rows, cols]`. `scale_shape` is
/// `[experts, rows, cols/16]` of raw E4M3 codes. `amax` is one F32 per expert,
/// `denominator` one F32; both must be finite with amax >= 0 and denominator > 0.
pub(crate) fn dequantize_nvfp4_amax_denominator(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
    amax_bytes: &[u8],
    denominator_bytes: &[u8],
) -> Result<Vec<f32>, String> {
    if weight_shape.len() != 3 || scale_shape.len() != 3 {
        return Err(format!(
            "NVFP4 needs 3-D weight/scale, got {weight_shape:?} / {scale_shape:?}"
        ));
    }
    let (experts, rows, cols_packed) = (weight_shape[0], weight_shape[1], weight_shape[2]);
    let cols = cols_packed
        .checked_mul(2)
        .ok_or("NVFP4 column count overflow")?;
    if cols % 16 != 0 {
        return Err(format!("NVFP4 logical cols {cols} not divisible by 16"));
    }
    let blocks = cols / 16;
    if scale_shape != [experts, rows, blocks] {
        return Err(format!(
            "NVFP4 scale shape {scale_shape:?} != [{experts}, {rows}, {blocks}]"
        ));
    }
    if weight_bytes.len() != experts * rows * cols_packed
        || scale_bytes.len() != experts * rows * blocks
    {
        return Err("NVFP4 weight/scale byte count mismatch".to_string());
    }
    if amax_bytes.len() != experts * 4 || denominator_bytes.len() != 4 {
        return Err(format!(
            "NVFP4 expected {} amax bytes and 4 denominator bytes, got {} / {}",
            experts * 4,
            amax_bytes.len(),
            denominator_bytes.len()
        ));
    }
    let denominator = f32::from_le_bytes(denominator_bytes[..4].try_into().unwrap());
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(format!(
            "NVFP4 denominator {denominator} must be finite and > 0"
        ));
    }

    let mut out = Vec::with_capacity(experts * rows * cols);
    for e in 0..experts {
        let amax = f32::from_le_bytes(amax_bytes[e * 4..e * 4 + 4].try_into().unwrap());
        if !amax.is_finite() || amax < 0.0 {
            return Err(format!("NVFP4 amax[{e}] {amax} must be finite and >= 0"));
        }
        let outer = amax / denominator;
        for r in 0..rows {
            let row_base = (e * rows + r) * blocks;
            for b in 0..blocks {
                let scale = e4m3_to_f32_strict(scale_bytes[row_base + b])?;
                let packed = (e * rows + r) * cols_packed + b * 8;
                for j in 0..16 {
                    let byte = weight_bytes[packed + j / 2];
                    let nibble = if j % 2 == 0 { byte & 0x0f } else { byte >> 4 };
                    let v = e2m1_to_f32(nibble) * scale * outer;
                    if !v.is_finite() {
                        return Err(format!(
                            "NVFP4 non-finite dequant at expert {e} row {r} col {}",
                            b * 16 + j
                        ));
                    }
                    out.push(v);
                }
            }
        }
    }
    Ok(out)
}

pub(crate) fn dequantize_hfq_q8f16(data: &[u8], n_elements: usize) -> Result<Vec<f32>, String> {
    let n_blocks = n_elements.div_ceil(32);
    let expected = n_blocks * 34;
    if data.len() != expected {
        return Err(format!(
            "Q8F16 byte size {} != {expected} for {n_elements} elements",
            data.len()
        ));
    }
    let mut out = vec![0.0f32; n_elements];
    for b in 0..n_blocks {
        let off = b * 34;
        let scale = f16_to_f32(u16::from_le_bytes([data[off], data[off + 1]]));
        let start = b * 32;
        let end = (start + 32).min(n_elements);
        for i in start..end {
            out[i] = (data[off + 2 + i - start] as i8) as f32 * scale;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32le(v: f32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    #[test]
    fn nvfp4_all_e2m1_codes_low_nibble_first() {
        // One expert, one row, 16 logical columns; logical column j is nibble j.
        let mut packed = vec![0u8; 8];
        for j in 0..16 {
            if j % 2 == 0 {
                packed[j / 2] |= j as u8;
            } else {
                packed[j / 2] |= (j as u8) << 4;
            }
        }
        let out = dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[56], // E4M3 1.0
            &[1, 1, 1],
            &f32le(1.0),
            &f32le(1.0),
        )
        .unwrap();
        let expected = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        assert_eq!(out.len(), 16);
        for (got, want) in out.iter().zip(expected.iter()) {
            assert_eq!(got.to_bits(), want.to_bits(), "E2M1 nibble order/table");
        }
    }

    #[test]
    fn nvfp4_block_and_outer_scale() {
        // 16 logical columns = 8 packed bytes, one 1x16 scale block.
        // Column 0: 0.5 (E2M1 code 1) * 2.0 (E4M3 0x40) * (6 / 3) = 2.0.
        let mut packed = [0u8; 8];
        packed[0] = 0x01; // low nibble 1, high nibble 0
        let out = dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[64],
            &[1, 1, 1],
            &f32le(6.0),
            &f32le(3.0),
        )
        .unwrap();
        assert_eq!(out.len(), 16);
        assert_eq!(out[0], 2.0);
        assert!(out[1..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn nvfp4_refuses_nan_scale_and_bad_companions() {
        let packed = [0u8; 8];
        assert!(dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[0x7f], // E4M3 NaN code
            &[1, 1, 1],
            &f32le(1.0),
            &f32le(1.0),
        )
        .is_err());
        assert!(dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[56],
            &[1, 1, 1],
            &f32le(-1.0),
            &f32le(1.0),
        )
        .is_err());
        assert!(dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[56],
            &[1, 1, 1],
            &f32le(1.0),
            &f32le(0.0),
        )
        .is_err());
    }

    #[test]
    fn nvfp4_refuses_nonfinite_product() {
        // 16 logical columns; column 0: 6.0 (E2M1 7) * 448.0 (E4M3 max 0x7e)
        // * 3e38 -> +inf. Must fail the finite-product check, not the shape check.
        let mut packed = [0u8; 8];
        packed[0] = 0x07;
        let err = dequantize_nvfp4_amax_denominator(
            &packed,
            &[1, 1, 8],
            &[0x7e],
            &[1, 1, 1],
            &f32le(3e38),
            &f32le(1.0),
        )
        .unwrap_err();
        assert!(
            err.contains("non-finite"),
            "expected non-finite refusal, got {err}"
        );
    }

    #[test]
    #[ignore = "requires the downloaded Inkling-2 checkpoint under D:/hipfire-models"]
    fn nvfp4_real_checkpoint_expert_inspect() {
        use hipfire_quantize::safetensors_file::SafetensorsFile;
        let path = std::path::Path::new(
            r"D:\hipfire-models\Inkling-2-Mini\model-00001-of-00001.safetensors",
        );
        let st = SafetensorsFile::open(path).expect("open checkpoint");
        let base = "model.llm.layers.0.mlp.experts.w13_weight";
        let names = [
            base.to_string(),
            format!("{base}_scale"),
            format!("{base}_amax"),
            format!("{base}_outer_scale_denominator"),
        ];
        for n in &names {
            let (m, b) = st.tensor_data(n).unwrap_or_else(|| panic!("missing {n}"));
            eprintln!(
                "{n}: dtype={} shape={:?} bytes={}",
                m.dtype,
                m.shape,
                b.len()
            );
        }
        let (wm, wb_all) = st.tensor_data(&names[0]).unwrap();
        let (sm, sb_all) = st.tensor_data(&names[1]).unwrap();
        let (_, ab_all) = st.tensor_data(&names[2]).unwrap();
        let (_, db) = st.tensor_data(&names[3]).unwrap();
        let rows = wm.shape[1];
        let packed = wm.shape[2];
        let per_expert_w = rows * packed;
        let per_expert_s = sm.shape[1] * sm.shape[2];
        let wb = &wb_all[..per_expert_w];
        let sb = &sb_all[..per_expert_s];
        let ab = &ab_all[..4];
        let out = dequantize_nvfp4_amax_denominator(
            wb,
            &[1, rows, packed],
            sb,
            &[1, sm.shape[1], sm.shape[2]],
            ab,
            db,
        )
        .expect("dequant expert 0");
        let finite = out.iter().all(|v| v.is_finite());
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "expert 0: {} values, finite={finite}, min={min}, max={max}",
            out.len()
        );
        eprintln!("first 16: {:?}", &out[..16]);
        assert!(finite);
        assert_eq!(out.len(), rows * packed * 2);
    }

    #[test]
    #[ignore = "requires the downloaded Inkling-2 checkpoint under D:/hipfire-models"]
    fn mxfp8_real_checkpoint_expert_inspect() {
        use hipfire_quantize::safetensors_file::SafetensorsFile;
        let path = std::path::Path::new(
            r"D:\hipfire-models\Inkling-2-Mini\model-00001-of-00001.safetensors",
        );
        let st = SafetensorsFile::open(path).expect("open checkpoint");
        let base = "model.llm.layers.20.mlp.experts.w13_weight";
        let scale_name = format!("{base}_scale");
        for n in [base, scale_name.as_str()] {
            let (m, b) = st.tensor_data(n).unwrap_or_else(|| panic!("missing {n}"));
            eprintln!(
                "{n}: dtype={} shape={:?} bytes={}",
                m.dtype,
                m.shape,
                b.len()
            );
        }
        let (wm, wb_all) = st.tensor_data(base).unwrap();
        let (sm, sb_all) = st.tensor_data(&scale_name).unwrap();
        let rows = wm.shape[1];
        let cols = wm.shape[2];
        let sc = sm.shape[2];
        assert_eq!(sc * 32, cols, "MXFP8 stored block must be [1,32]");
        let wb = &wb_all[..rows * cols];
        let sb = &sb_all[..rows * sc];
        let out = dequantize_mxfp8_e8m0_strict(wb, &[rows, cols], sb, &[rows, sc], 32)
            .expect("dequant MXFP8 expert 0");
        let finite = out.iter().all(|v| v.is_finite());
        let max = out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = out.iter().copied().fold(f32::INFINITY, f32::min);
        eprintln!(
            "MXFP8 expert 0: {} values, finite={finite}, min={min}, max={max}",
            out.len()
        );
        assert!(finite);
        assert_eq!(out.len(), rows * cols);
    }
}
