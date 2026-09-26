// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    non_snake_case,
    clippy::all
)]
use crate::quant_fwht::{cpu_fwht_128, cpu_fwht_256, gen_fwht_signs};
use crate::quant_hfp4::{e2m1_round, e4m3_scale_decode, e4m3_scale_encode_roundup, E2M1_LUT};

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::e8;
use crate::e8::*;
use crate::e8_gptq;
use crate::e8_gptq::*;
use crate::gguf_input;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

/// Quantize one row of K FP32 weights to mfp4-E8 byte format.
/// Same E4M3 scale as mfp4+P; per-32-weight-block data = 4 E8 codewords (u32 each).
/// Returns 16-B header + (K/32) x 17-B blocks. Byte-identical footprint to mfp4+P.
pub(crate) fn quantize_mfp4g32_e8_row(row: &[f32]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfp4-E8 requires K%32==0, got K={}",
        row.len()
    );
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };
    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05; // FWHT flag, identical to mfp4+P
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let block_scale_factor = e4m3_scale_decode(scale_byte);
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * 17;
        out[payload_off] = scale_byte;
        // 4 E8 codewords, 8 weights each, 32 bits little-endian.
        for g in 0..4 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = block[g * 8 + i] * inv_row_scale * inv_block_scale; // => [-6,6]
            }
            let idx = e8::quantize8(&v, e8::QUANT_STEP);
            out[payload_off + 1 + g * 4..payload_off + 1 + g * 4 + 4]
                .copy_from_slice(&idx.to_le_bytes());
        }
    }
    out
}

/// E8 row encoder with a post-encode least-squares row-scale correction.
///
/// The ordinary encoder chooses row/block scales from maxima so no lattice
/// coordinate clips. Once the E8 codewords and block scales are fixed, the
/// remaining row-scale scalar has a closed-form MSE optimum:
/// `alpha = dot(w, q) / dot(q, q)`. Folding alpha into the existing f16 row
/// scale changes neither the wire format nor the gfx1151 decoder. This variant
/// is kept as an explicit overlay tier until full-corpus quality decides
/// whether it should repair the MQ2R router/head buckets.
pub(crate) fn quantize_mfp4g32_e8_row_lsq(row: &[f32]) -> Vec<u8> {
    let mut out = quantize_mfp4g32_e8_row(row);
    let row_scale = f16_to_f32(u16::from_le_bytes([out[0], out[1]]));
    if !(row_scale > 0.0) {
        return out;
    }

    let n_blocks = row.len() / 32;
    let mut dot_wq = 0.0f64;
    let mut dot_qq = 0.0f64;
    for b in 0..n_blocks {
        let payload_off = 16 + b * 17;
        let block_scale = e4m3_scale_decode(out[payload_off]);
        let scale = row_scale * block_scale;
        for g in 0..4 {
            let off = payload_off + 1 + g * 4;
            let idx = u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            let decoded = e8::dequantize8(idx, e8::QUANT_STEP);
            for i in 0..8 {
                let q = scale * decoded[i];
                let w = row[b * 32 + g * 8 + i];
                dot_wq += (w as f64) * (q as f64);
                dot_qq += (q as f64) * (q as f64);
            }
        }
    }
    if dot_qq > 0.0 {
        let alpha = (dot_wq / dot_qq) as f32;
        if alpha.is_finite() && alpha > 0.0 {
            let corrected = f32_to_f16(row_scale * alpha);
            let corrected_scale = f16_to_f32(corrected);
            let beta = (corrected_scale / row_scale) as f64;
            // Compare the representable f16 candidate against the original
            // scale. The dot(w,w) term cancels:
            //   ΔSSE = -2(β-1)dot(w,q) + (β²-1)dot(q,q).
            // This guards against f16 rounding moving the closed-form optimum
            // to the wrong side of the original scale.
            let delta_sse = -2.0 * (beta - 1.0) * dot_wq + (beta * beta - 1.0) * dot_qq;
            if corrected_scale > 0.0 && delta_sse <= 0.0 {
                out[0..2].copy_from_slice(&corrected.to_le_bytes());
            }
        }
    }
    out
}

/// E8 row-scale correction weighted by a diagonal activation Hessian.
///
/// `importance[i] = Σ_t x[t,i]²` in the same FWHT-rotated domain as `row`.
/// For fixed codewords/block scales, this minimizes the diagonal-Hessian
/// approximation to output error while preserving the E8 wire format.
pub(crate) fn quantize_mfp4g32_e8_row_awls(row: &[f32], importance: &[f64]) -> Vec<u8> {
    assert_eq!(row.len(), importance.len());
    let mut out = quantize_mfp4g32_e8_row(row);
    let row_scale = f16_to_f32(u16::from_le_bytes([out[0], out[1]]));
    if !(row_scale > 0.0) {
        return out;
    }

    let n_blocks = row.len() / 32;
    let mut dot_wq = 0.0f64;
    let mut dot_qq = 0.0f64;
    for b in 0..n_blocks {
        let payload_off = 16 + b * 17;
        let block_scale = e4m3_scale_decode(out[payload_off]);
        let scale = row_scale * block_scale;
        for g in 0..4 {
            let off = payload_off + 1 + g * 4;
            let idx = u32::from_le_bytes([out[off], out[off + 1], out[off + 2], out[off + 3]]);
            let decoded = e8::dequantize8(idx, e8::QUANT_STEP);
            for i in 0..8 {
                let column = b * 32 + g * 8 + i;
                let h = importance[column];
                let q = scale * decoded[i];
                let w = row[column];
                dot_wq += h * (w as f64) * (q as f64);
                dot_qq += h * (q as f64) * (q as f64);
            }
        }
    }
    if dot_qq > 0.0 {
        let alpha = (dot_wq / dot_qq) as f32;
        if alpha.is_finite() && alpha > 0.0 {
            let corrected = f32_to_f16(row_scale * alpha);
            let corrected_scale = f16_to_f32(corrected);
            let beta = (corrected_scale / row_scale) as f64;
            let delta_error = -2.0 * (beta - 1.0) * dot_wq + (beta * beta - 1.0) * dot_qq;
            if corrected_scale > 0.0 && delta_error <= 0.0 {
                out[0..2].copy_from_slice(&corrected.to_le_bytes());
            }
        }
    }
    out
}

/// Generic mfpN-E8 row encoder (n=2 or n=3). Same outer container as mfp4-E8
/// (16 B row header + (K/32) blocks) but each block is 1 + 4*n bytes:
///   - 1 B: E4M3 block scale
///   - 4 × n B: codewords (low 8n bits of u32, LE)
/// Row bytes = 16 + (1 + 4*n) * (K/32). Callers use the thin wrappers below.
pub(crate) fn quantize_mfpn_e8_row(row: &[f32], n: u32, quant_step: f32) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfpN-E8 requires K%32==0, got K={}",
        row.len()
    );
    assert!(n == 2 || n == 3, "only n=2 or n=3 supported by this helper");
    let k = row.len();
    let n_blocks = k / 32;
    let block_bytes = 1 + 4 * n as usize;
    let row_bytes = 16 + n_blocks * block_bytes;
    let mut out = vec![0u8; row_bytes];
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };
    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05; // same FWHT flag as mfp4-E8
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let s = if block_max_normalized > 0.0 {
            block_max_normalized / 6.0
        } else {
            0.0
        };
        let scale_byte = e4m3_scale_encode_roundup(s);
        let block_scale_factor = e4m3_scale_decode(scale_byte);
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * block_bytes;
        out[payload_off] = scale_byte;
        // 4 codewords per block, 8 weights each, packed into low 8n bits (LE).
        for g in 0..4 {
            let mut v = [0.0f32; 8];
            for i in 0..8 {
                v[i] = block[g * 8 + i] * inv_row_scale * inv_block_scale;
            }
            let idx = e8::quantize8_n(&v, quant_step, n);
            // Write only the low n bytes (upper bits of u32 are guaranteed zero
            // by encode_index_n — see e8.rs high-bit-zero invariant).
            let bytes = idx.to_le_bytes();
            let cw_off = payload_off + 1 + g * n as usize;
            out[cw_off..cw_off + n as usize].copy_from_slice(&bytes[..n as usize]);
        }
    }
    out
}

pub(crate) fn quantize_mfp3g32_e8_row(row: &[f32]) -> Vec<u8> {
    quantize_mfpn_e8_row(row, 3, e8::QUANT_STEP_MFP3)
}

pub(crate) fn quantize_mfp2g32_e8_row(row: &[f32]) -> Vec<u8> {
    quantize_mfpn_e8_row(row, 2, e8::QUANT_STEP_MFP2)
}

/// mfpN-E8 2D: FWHT-rotate (same signs as mfp4-E8), then per-row encode.
pub(crate) fn quantize_mfpn_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    n: u32,
    quant_step: f32,
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfpN-E8 requires k%256==0, got k={}", k);
    let block_bytes = 1 + 4 * n as usize;
    let row_bytes = 16 + block_bytes * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        out.extend_from_slice(&quantize_mfpn_e8_row(&row_buf, n, quant_step));
    }
    out
}

pub fn quantize_mfp3g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mfpn_e8_2d(f32_data, m, k, signs1, signs2, 3, e8::QUANT_STEP_MFP3)
}

pub fn quantize_mfp2g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mfpn_e8_2d(f32_data, m, k, signs1, signs2, 2, e8::QUANT_STEP_MFP2)
}

/// CPU reference dequant for mfp3-E8. Returns row-major f32 in the ROTATED domain.
/// Mirrors the kernel mfp3_decode_index decode exactly (3-bit nibbles, center 3,
/// coset bit 23, e7_high 2b at bit 21).
#[allow(dead_code)]
pub(crate) fn dequant_mfp3g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let block_bytes = 13usize; // 1 + 4*3
    let row_bytes = 16 + block_bytes * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp3-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * block_bytes;
            let scale_byte = packed[po];
            let scale = row_scale_a * e4m3_scale_decode(scale_byte) * e8::QUANT_STEP_MFP3;
            for g in 0..4usize {
                let cw_off = po + 1 + g * 3;
                // 3-byte LE narrow read (safe — block is 13 B, max cw_off = po+1+3*3=po+10, reads to po+12)
                let idx: u32 = (packed[cw_off] as u32)
                    | ((packed[cw_off + 1] as u32) << 8)
                    | ((packed[cw_off + 2] as u32) << 16);
                // mfp3_decode_index: 3-bit nibbles, center 3, coset bit 23, e7_high @21 (2b)
                let coset = (idx >> 23) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7 {
                    e[i] = (idx >> (3 * i as u32)) & 0x7;
                    sl += e[i];
                }
                let e7_high = (idx >> 21) & 0x3;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                for i in 0..8usize {
                    let c = (e[i] as i32 - 3) as f32;
                    let coord = if coset == 1 { c + 0.5 } else { c };
                    out[r * k + b * 32 + g * 8 + i] = scale * coord;
                }
            }
        }
    }
    out
}

/// CPU reference dequant for mfp2-E8. Returns row-major f32 in the ROTATED domain.
/// Mirrors the kernel mfp2_decode_index decode exactly (2-bit nibbles, center 1,
/// coset bit 15, e7_high 1b at bit 14).
#[allow(dead_code)]
pub(crate) fn dequant_mfp2g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let block_bytes = 9usize; // 1 + 4*2
    let row_bytes = 16 + block_bytes * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp2-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * block_bytes;
            let scale_byte = packed[po];
            let scale = row_scale_a * e4m3_scale_decode(scale_byte) * e8::QUANT_STEP_MFP2;
            for g in 0..4usize {
                let cw_off = po + 1 + g * 2;
                // 2-byte LE narrow read (safe — block is 9 B, max cw_off = po+1+3*2=po+7, reads to po+8)
                let idx: u32 = (packed[cw_off] as u32) | ((packed[cw_off + 1] as u32) << 8);
                // mfp2_decode_index: 2-bit nibbles, center 1, coset bit 15, e7_high @14 (1b)
                let coset = (idx >> 15) & 1;
                let mut e = [0u32; 8];
                let mut sl: u32 = 0;
                for i in 0..7 {
                    e[i] = (idx >> (2 * i as u32)) & 0x3;
                    sl += e[i];
                }
                let e7_high = (idx >> 14) & 0x1;
                let p7 = e7_high << 1;
                e[7] = p7 | ((sl + p7) & 1);
                for i in 0..8usize {
                    let c = (e[i] as i32 - 1) as f32;
                    let coord = if coset == 1 { c + 0.5 } else { c };
                    out[r * k + b * 32 + g * 8 + i] = scale * coord;
                }
            }
        }
    }
    out
}

/// mfp4-E8 2D: FWHT-rotate the tensor (same signs as mfp4+P), then per-row
/// quantize_mfp4g32_e8_row. Byte layout identical to mfp4+P (NO prefix).
/// On-disk per-256-block Hessian magic ("E8H1").
pub(crate) const E8_HESSIAN_MAGIC: u32 = 0x45_38_48_31;

/// Sanitize a full safetensors tensor name into a filesystem-safe key.
pub(crate) fn hessian_key(tensor_name: &str) -> String {
    tensor_name.replace(['/', '\\'], "_").replace("..", "_")
}

/// Read per-256-block Hessians for one tensor from `<dir>/<key>.hblk`.
/// File layout: [u32 magic][u32 n_blocks][u32 k][f32 ... n_blocks*256*256].
/// Returns empty Vec if the file is missing (-> caller falls back to RTN).
pub(crate) fn load_hessian_blocks(dir: &Path, tensor_name: &str) -> Vec<e8_gptq::HBlock> {
    let path = dir.join(format!("{}.hblk", hessian_key(tensor_name)));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if bytes.len() < 12 {
        return Vec::new();
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != E8_HESSIAN_MAGIC {
        eprintln!("warning: {} bad Hessian magic; ignoring", path.display());
        return Vec::new();
    }
    let n_blocks = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let stored_k = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    if stored_k != n_blocks * 256 {
        eprintln!(
            "warning: {} Hessian K={} disagrees with {} blocks; ignoring",
            path.display(),
            stored_k,
            n_blocks
        );
        return Vec::new();
    }
    let want = 12 + n_blocks * 256 * 256 * 4;
    if bytes.len() < want {
        eprintln!(
            "warning: {} truncated Hessian ({} < {}); ignoring",
            path.display(),
            bytes.len(),
            want
        );
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_blocks);
    let mut off = 12usize;
    for _ in 0..n_blocks {
        let mut blk = vec![0.0f32; 256 * 256];
        for v in blk.iter_mut() {
            *v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            off += 4;
        }
        out.push(blk);
    }
    out
}

/// GPTQ-on-E8 fired/fallback telemetry. `fired` = a non-empty `.hblk` was
/// loaded for this (tensor,expert) (Hessian-aware LDLQ ran); `fallback` =
/// empty/missing Hessian -> silent RTN E8. ~0 fired with --hessian-dir set
/// means a KEY-MISMATCH BUG (filenames != hessian_key), not a flat result.
pub(crate) static GPTQ_E8_FIRED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static GPTQ_E8_FALLBACK: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// GPTQ-E8 wrapper that wires the production helpers into the e8_gptq module.
/// `h_blocks` empty -> RTN fallback (byte-identical to quantize_mfp4g32_e8_2d).
pub(crate) fn quantize_mfp4g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp4g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

/// GPTQ-mfp3-E8 wrapper. `h_blocks` empty -> RTN fallback (byte-identical to
/// quantize_mfp3g32_e8_2d). Output layout: 16-byte header + 13 B per 32-block.
pub(crate) fn quantize_mfp3g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp3g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

/// GPTQ-mfp2-E8 wrapper. `h_blocks` empty -> RTN fallback (byte-identical to
/// quantize_mfp2g32_e8_2d). Output layout: 16-byte header + 9 B per 32-block.
pub(crate) fn quantize_mfp2g32_e8_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    e8_gptq::quantize_mfp2g32_e8_gptq_2d(
        f32_data,
        m,
        k,
        signs1,
        signs2,
        h_blocks,
        &cpu_fwht_256,
        &e4m3_scale_encode_roundup,
        &e4m3_scale_decode,
        &f32_to_f16,
    )
}

pub(crate) fn quantize_mfp4g32_e8_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfp4-E8 requires k%256==0, got k={}", k);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        out.extend_from_slice(&quantize_mfp4g32_e8_row(&row_buf));
    }
    out
}

/// CPU reference dequant for mfp4-E8. Returns row-major f32 [m*k] in the ROTATED domain.
/// Bit-exact mirror of the gfx942 dequantize_mfp4g32_e8_to_f16 kernel decode.
#[allow(dead_code)]
pub(crate) fn dequant_mfp4g32_e8(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    assert_eq!(packed.len(), m * row_bytes, "mfp4-E8 size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let scale = row_scale_a * e4m3_scale_decode(packed[po]);
            for g in 0..4 {
                let idx = u32::from_le_bytes([
                    packed[po + 1 + g * 4],
                    packed[po + 2 + g * 4],
                    packed[po + 3 + g * 4],
                    packed[po + 4 + g * 4],
                ]);
                let vd = e8::dequantize8(idx, e8::QUANT_STEP);
                for i in 0..8 {
                    out[r * k + b * 32 + g * 8 + i] = scale * vd[i];
                }
            }
        }
    }
    out
}

/// Convert an AoS-packed mfp4-E8 row to SoA layout.
/// AoS layout: [16B hdr] + n_blocks * [1B E4M3 scale + 16B codewords]
/// SoA layout: [16B hdr] + [n_blocks B scales, pad to 16B] + [n_blocks * 16B codewords]
/// The header is the same except byte[6] (flag) is set to 0x06 (was 0x05).
pub(crate) fn aos_to_soa_row(aos: &[u8], n_blocks: usize) -> Vec<u8> {
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_len = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; soa_len];
    // Copy header, change flag byte
    out[..16].copy_from_slice(&aos[..16]);
    out[6] = 0x06; // SoA flag
                   // Gather scales
    for b in 0..n_blocks {
        out[16 + b] = aos[16 + b * 17]; // scale byte at start of each 17B AoS block
    }
    // Gather codewords (4 x u32 = 16B per block)
    let cw_start = 16 + scale_padded;
    for b in 0..n_blocks {
        let src = 16 + b * 17 + 1; // skip the 1B scale
        let dst = cw_start + b * 16;
        out[dst..dst + 16].copy_from_slice(&aos[src..src + 16]);
    }
    out
}

/// GPTQ-on-E8 with the qt=35 structure-of-arrays wire layout.
///
/// LDLQ changes only the E8 codewords. The scale/header and AoS-to-SoA
/// permutation remain identical to the ordinary qt=35 encoder.
pub(crate) fn quantize_mfp4g32_e8_soa_gptq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    h_blocks: &[e8_gptq::HBlock],
) -> Vec<u8> {
    use rayon::prelude::*;

    let aos = quantize_mfp4g32_e8_gptq_2d(f32_data, m, k, signs1, signs2, h_blocks);
    let n_blocks = k / 32;
    let aos_row_bytes = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row_bytes];
    out.par_chunks_mut(soa_row_bytes)
        .zip(aos.par_chunks(aos_row_bytes))
        .for_each(|(dst, src)| dst.copy_from_slice(&aos_to_soa_row(src, n_blocks)));
    out
}

/// mfp4-E8 SoA quantizer: same E8 encoding as quantize_mfp4g32_e8_2d, then permuted to SoA.
pub(crate) fn quantize_mfp4g32_e8_soa_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "mfp4-E8-SoA requires k%256==0, got k={}", k);
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row_bytes];
    out.par_chunks_mut(soa_row_bytes)
        .enumerate()
        .for_each(|(r, dst)| {
            let mut row_buf = f32_data[r * k..(r + 1) * k].to_vec();
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            let aos_row = quantize_mfp4g32_e8_row(&row_buf);
            let soa_row = aos_to_soa_row(&aos_row, n_blocks);
            dst.copy_from_slice(&soa_row);
        });
    out
}

/// 128-wide E8-SoA encoder.
///
/// Same E8 lattice codec, same SoA wire layout, and the same byte geometry as
/// [`quantize_mfp4g32_e8_soa_2d`]: `n_blocks = k/32`, scales padded to 16B,
/// then `n_blocks * 16B` of codewords. The ONLY difference is the FWHT
/// segmentation — 128 elements instead of 256 — which is what lets a
/// reduction axis that is not a multiple of 256 be rotated at all. qwen4's
/// routed-expert `down_proj` is exactly that case: K = 640 = 5 x 128.
///
/// The rotation is orthogonal and is applied to both weights and activations,
/// so this changes the *distribution* the lattice sees, never the decode math.
/// The activation side must use the matching G128 basis (`rotate_x_mq_128`,
/// seeds 43/1043 — the same tables MQ4G128V2 already consumes).
pub(crate) fn quantize_mfp4g32_e8_soa128_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(f32_data.len(), m * k);
    assert!(
        k % 128 == 0,
        "mfp4-E8-SoA-128 requires k%128==0, got k={}",
        k
    );
    assert!(signs1.len() == 128 && signs2.len() == 128);
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row_bytes];
    out.par_chunks_mut(soa_row_bytes)
        .enumerate()
        .for_each(|(r, dst)| {
            let mut row_buf = f32_data[r * k..(r + 1) * k].to_vec();
            for seg in 0..(k / 128) {
                cpu_fwht_128(&mut row_buf[seg * 128..(seg + 1) * 128], signs1, signs2);
            }
            let aos_row = quantize_mfp4g32_e8_row(&row_buf);
            let soa_row = aos_to_soa_row(&aos_row, n_blocks);
            dst.copy_from_slice(&soa_row);
        });
    out
}

/// 128-wide AoS E8 encoder for reduction axes that 256 cannot tile.
///
/// Same E8 lattice codec, same AoS wire layout and the same byte geometry as
/// [`quantize_mfp4g32_e8_2d`]: `[16 B header][n_blocks x (1 B E4M3 scale + 16 B
/// codewords)]` with `n_blocks = k/32`. The ONLY difference is the FWHT
/// segmentation — 128 elements instead of 256 — which is what lets a reduction
/// axis that 256 cannot tile be rotated at all. qwen4's routed-expert
/// `down_proj` is exactly that case: K = 640 = 5 x 128.
///
/// Because the row layout is unchanged, this format is exactly iso-size with
/// `MFP4G32E8`/`MFP4G32E8SOA` per 32-weight block, and unlike the SoA sibling it
/// carries no scale padding. It is NOT header-free: the 16 B row header holds
/// the f16 row scale, which is 16 B/row beyond what the shipping MQ4 V2 tiers
/// spend — 4.7% of a K=640 row, 1.2% of a K=2560 row.
///
/// The activation side must use the matching G128 basis (`rotate_x_mq_128`,
/// sign seeds 43/1043) — the same tables `MQ4G128V2` already consumes.
pub(crate) fn quantize_mfp4g32_e8_g128_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(f32_data.len(), m * k);
    assert!(k % 128 == 0, "mfp4-E8-G128 requires k%128==0, got k={}", k);
    assert!(signs1.len() == 128 && signs2.len() == 128);
    let row_bytes = mfp4g32_e8_g128_row_bytes(k);
    let mut out = vec![0u8; m * row_bytes];
    out.par_chunks_mut(row_bytes)
        .enumerate()
        .for_each(|(r, dst)| {
            let mut row_buf = f32_data[r * k..(r + 1) * k].to_vec();
            for seg in 0..(k / 128) {
                cpu_fwht_128(&mut row_buf[seg * 128..(seg + 1) * 128], signs1, signs2);
            }
            let aos_row = quantize_mfp4g32_e8_row(&row_buf);
            dst.copy_from_slice(&aos_row);
        });
    out
}

/// Row stride of the qt=42 AoS layout: `[16 B header][n_blocks x 17 B]`.
///
/// Byte-identical to `MFP4G32E8` (qt=34), because the rotation width is not in
/// the bytes — it is only in which 128-element segments share a FWHT. This is
/// the loader/GEMM row-stride rule for the new tag; `k % 32 == 0` is the wire
/// requirement, while the *encoder* additionally needs `k % 128 == 0` to place
/// whole rotation segments.
pub(crate) fn mfp4g32_e8_g128_row_bytes(k: usize) -> usize {
    assert!(k % 32 == 0, "mfp4g32-E8-G128 requires k%32==0, got k={}", k);
    16 + 17 * (k / 32)
}

/// SoA E8 with the explicit least-squares row-scale repair above.
pub(crate) fn quantize_mfp4g32_e8_soa_lsq_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(f32_data.len(), m * k);
    assert!(
        k % 256 == 0,
        "mfp4-E8-SoA-LSQ requires k%256==0, got k={}",
        k
    );
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row_bytes];
    out.par_chunks_mut(soa_row_bytes)
        .enumerate()
        .for_each(|(r, dst)| {
            let mut row_buf = f32_data[r * k..(r + 1) * k].to_vec();
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            let aos_row = quantize_mfp4g32_e8_row_lsq(&row_buf);
            let soa_row = aos_to_soa_row(&aos_row, n_blocks);
            dst.copy_from_slice(&soa_row);
        });
    out
}

/// SoA E8 with activation-weighted least-squares row scales.
pub(crate) fn quantize_mfp4g32_e8_soa_awls_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    importance: &[f64],
) -> Vec<u8> {
    use rayon::prelude::*;

    assert_eq!(f32_data.len(), m * k);
    assert_eq!(importance.len(), k);
    assert!(
        k % 256 == 0,
        "mfp4-E8-SoA-AWLS requires k%256==0, got k={}",
        k
    );
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; m * soa_row_bytes];
    out.par_chunks_mut(soa_row_bytes)
        .enumerate()
        .for_each(|(r, dst)| {
            let mut row_buf = f32_data[r * k..(r + 1) * k].to_vec();
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
            }
            let aos_row = quantize_mfp4g32_e8_row_awls(&row_buf, importance);
            let soa_row = aos_to_soa_row(&aos_row, n_blocks);
            dst.copy_from_slice(&soa_row);
        });
    out
}

/// Load and sum one or more `DS4HIM01` diagonal head-imatrix files.
///
/// Multiple paths are separated by `:` in `HIPFIRE_E8_IMATRIX`; raw sums are
/// additive, so corpora with different row counts receive proportional weight.
pub(crate) fn load_ds4_head_importance(k: usize) -> Result<Vec<f64>, String> {
    let spec = hipfire_config::developer_var("HIPFIRE_E8_IMATRIX")
        .map_err(|_| "mfp4e8soa-awls requires HIPFIRE_E8_IMATRIX".to_string())?;
    let mut total = vec![0.0f64; k];
    let mut files = 0usize;
    for path in spec.split(':').filter(|path| !path.is_empty()) {
        let bytes = std::fs::read(path).map_err(|e| format!("read E8 imatrix {path}: {e}"))?;
        let expected = 16 + k * 8;
        if bytes.len() != expected || &bytes[..8] != b"DS4HIM01" {
            return Err(format!(
                "E8 imatrix {path}: invalid format/size {} (expected {expected})",
                bytes.len()
            ));
        }
        let hidden = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let rows = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if hidden != k || rows == 0 {
            return Err(format!(
                "E8 imatrix {path}: hidden={hidden} rows={rows}, expected hidden={k} and rows>0"
            ));
        }
        for (column, value) in total.iter_mut().enumerate() {
            let offset = 16 + column * 8;
            *value += f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        }
        files += 1;
    }
    if files == 0 || total.iter().all(|value| *value == 0.0) {
        return Err("E8 imatrix set is empty or all-zero".to_string());
    }
    eprintln!("loaded {files} E8 head-imatrix file(s), hidden={k}");
    Ok(total)
}

/// CPU reference dequant for mfp4-E8 SoA. Returns row-major f32 [m*k] in ROTATED domain.
/// Bit-exact with dequant_mfp4g32_e8 for the same weight data.
#[allow(dead_code)]
pub(crate) fn dequant_mfp4g32_e8_soa(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let n_blocks = k / 32;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    assert_eq!(packed.len(), m * soa_row_bytes, "mfp4-E8-SoA size mismatch");
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = r * soa_row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        let scale_arr = &packed[base + 16..base + 16 + n_blocks];
        let cw_arr = &packed[base + 16 + scale_padded..base + 16 + scale_padded + n_blocks * 16];
        for b in 0..n_blocks {
            let scale = row_scale_a * e4m3_scale_decode(scale_arr[b]);
            for g in 0..4 {
                let co = b * 16 + g * 4;
                let idx = u32::from_le_bytes([
                    cw_arr[co],
                    cw_arr[co + 1],
                    cw_arr[co + 2],
                    cw_arr[co + 3],
                ]);
                let vd = e8::dequantize8(idx, e8::QUANT_STEP);
                for i in 0..8 {
                    out[r * k + b * 32 + g * 8 + i] = scale * vd[i];
                }
            }
        }
    }
    out
}

/// Fit ONE 16-entry fp16 Lloyd-Max codebook over ALL normalized values of a
/// FWHT-rotated tensor, in the same ~[-6,6] domain that feeds e2m1_round in
/// mfp4. `vals` are pre-collected: for every element, value*inv_row_scale*
/// inv_block_scale (i.e. exactly what mfp4 hands to e2m1_round). Mirrors the
/// percentile-init + 8 k-means iters + sort-ascending recipe of
/// quantize_mq4g256_lloyd, but tensor-global instead of per-block.
pub(crate) fn fit_mfp4_lloyd_codebook(vals: &[f32]) -> [f32; 16] {
    let mut sorted: Vec<f32> = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mut cb = [0.0f32; 16];
    if n == 0 {
        for k in 0..16 {
            cb[k] = E2M1_LUT[k];
        }
        return cb;
    }
    for k in 0..16 {
        let frac = (2 * k + 1) as f32 / 32.0;
        let idx = ((frac * (n as f32 - 1.0)).round() as usize).min(n - 1);
        cb[k] = sorted[idx];
    }
    let range = sorted[n - 1] - sorted[0];
    if range > 0.0 {
        let mut _it = 0usize;
        loop {
            let mut sums = [0.0f64; 16];
            let mut counts = [0u64; 16];
            for &w in vals {
                let mut best = 0usize;
                let mut best_d = (w - cb[0]).abs();
                for k in 1..16 {
                    let d = (w - cb[k]).abs();
                    if d < best_d {
                        best_d = d;
                        best = k;
                    }
                }
                sums[best] += w as f64;
                counts[best] += 1;
            }
            let mut moved = false;
            for k in 0..16 {
                if counts[k] > 0 {
                    let c = (sums[k] / counts[k] as f64) as f32;
                    if c != cb[k] {
                        moved = true;
                    }
                    cb[k] = c;
                }
            }
            _it += 1;
            if !moved || _it >= 8 {
                break;
            }
        }
    }
    cb.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    cb
}

/// Nearest-codebook index for mfp4L. The codebook is fp16-rounded BEFORE search
/// so quantize matches kernel recon (kernel reads fp16 cb).
pub(crate) fn nearest_cb_idx(x: f32, cb: &[f32; 16]) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &c) in cb.iter().enumerate() {
        let cf = f16_to_f32(f32_to_f16(c));
        let e = (cf - x).abs();
        if e < best_err {
            best_err = e;
            best = i as u8;
        }
    }
    best
}

/// Quantize one mfp4-normalized row to 17-B-block bytes using a fixed
/// per-tensor codebook (nibble = nearest-codebook-index).
pub(crate) fn quantize_mfp4g32_lloyd_row(row: &[f32], cb: &[f32; 16]) -> Vec<u8> {
    assert!(
        row.len() % 32 == 0,
        "mfp4L requires K%32==0, got K={}",
        row.len()
    );
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 {
        row_max_abs / 6.0
    } else {
        1.0
    };
    let inv_row_scale = if row_max_abs > 0.0 {
        1.0 / row_scale_a
    } else {
        0.0
    };
    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes());
    out[6] = 0x05;
    out[7] = 0u8;
    for b in 0..n_blocks {
        let block = &row[b * 32..b * 32 + 32];
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;
        let block_e: u8 = if block_max_normalized > 0.0 {
            let e_signed = (block_max_normalized / 6.0).log2().ceil() as i32 + 127;
            e_signed.clamp(0, 254) as u8
        } else {
            0u8
        };
        let block_scale_factor = ((block_e as i32 - 127) as f32).exp2();
        let inv_block_scale = if block_scale_factor > 0.0 {
            1.0 / block_scale_factor
        } else {
            0.0
        };
        let payload_off = 16 + b * 17;
        out[payload_off] = block_e;
        for i in 0..16 {
            let lo = block[2 * i] * inv_row_scale * inv_block_scale;
            let hi = block[2 * i + 1] * inv_row_scale * inv_block_scale;
            out[payload_off + 1 + i] =
                (nearest_cb_idx(lo, cb) & 0x0F) | ((nearest_cb_idx(hi, cb) & 0x0F) << 4);
        }
    }
    out
}

/// MFP4G32-Lloyd 2D: FWHT-rotate the tensor, fit ONE per-tensor 16-entry codebook,
/// then emit [32-B fp16 codebook][M rows].
pub(crate) fn quantize_mfp4g32_lloyd_2d(
    f32_data: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k);
    assert!(k % 256 == 0, "MFP4G32Lloyd requires k%256==0, got k={}", k);
    let mut rotated = vec![0.0f32; m * k];
    let mut norm_vals: Vec<f32> = Vec::with_capacity(m * k);
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        rotated[r * k..(r + 1) * k].copy_from_slice(&row_buf);
        let row_max_abs = row_buf.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
        let inv_row_scale = if row_max_abs > 0.0 {
            6.0 / row_max_abs
        } else {
            0.0
        };
        for b in 0..(k / 32) {
            let block = &row_buf[b * 32..b * 32 + 32];
            let bmax = block.iter().cloned().fold(0.0f32, |a, v| a.max(v.abs()));
            let bmn = bmax * inv_row_scale;
            let block_e: i32 = if bmn > 0.0 {
                (bmn / 6.0).log2().ceil() as i32 + 127
            } else {
                0
            };
            let inv_block_scale = (-(block_e.clamp(0, 254) - 127) as f32).exp2();
            for &v in block {
                norm_vals.push(v * inv_row_scale * inv_block_scale);
            }
        }
    }
    let cb = fit_mfp4_lloyd_codebook(&norm_vals);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(32 + m * row_bytes);
    for k16 in 0..16usize {
        out.extend_from_slice(&f32_to_f16(cb[k16]).to_le_bytes());
    }
    for r in 0..m {
        out.extend_from_slice(&quantize_mfp4g32_lloyd_row(
            &rotated[r * k..(r + 1) * k],
            &cb,
        ));
    }
    debug_assert_eq!(out.len(), 32 + m * row_bytes);
    out
}

/// CPU reference dequant for mfp4L.
/// Returns row-major f32 [m*k] in the ROTATED domain.
#[allow(dead_code)]
pub(crate) fn dequant_mfp4g32_lloyd(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
    let row_bytes = 16 + 17 * (k / 32);
    assert_eq!(packed.len(), 32 + m * row_bytes, "mfp4L size mismatch");
    let mut cb = [0.0f32; 16];
    for i in 0..16 {
        cb[i] = f16_to_f32(u16::from_le_bytes([packed[2 * i], packed[2 * i + 1]]));
    }
    let mut out = vec![0.0f32; m * k];
    for r in 0..m {
        let base = 32 + r * row_bytes;
        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
        for b in 0..(k / 32) {
            let po = base + 16 + b * 17;
            let block_e = packed[po] as i32;
            let scale = row_scale_a * ((block_e - 127) as f32).exp2();
            for i in 0..16 {
                let byte = packed[po + 1 + i];
                out[r * k + b * 32 + 2 * i] = scale * cb[(byte & 0x0F) as usize];
                out[r * k + b * 32 + 2 * i + 1] = scale * cb[((byte >> 4) & 0x0F) as usize];
            }
        }
    }
    out
}

/// CPU reference dequantization for HFP4G32 — bit-exact mirror of `gemv_hfp4g32.hip`'s dequant.
/// Returns the K reconstructed FP32 weights for one row.
#[allow(dead_code)] // used by tests + future round-trip diagnostics
pub(crate) fn dequant_hfp4g32_row(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 32 == 0, "HFP4G32 requires K%32 == 0");
    let n_blocks = k / 32;
    assert_eq!(
        packed.len(),
        16 + n_blocks * 17,
        "HFP4G32 row size mismatch"
    );

    let row_scale_a_bits = u16::from_le_bytes([packed[0], packed[1]]);
    let row_scale_a = f16_to_f32(row_scale_a_bits);

    let mut out = vec![0.0f32; k];
    for b in 0..n_blocks {
        let payload_off = 16 + b * 17;
        let block_e = packed[payload_off] as i32;
        let block_scale = (block_e - 127) as f32;
        let block_scale_factor = block_scale.exp2();
        let scale = row_scale_a * block_scale_factor;

        for i in 0..16 {
            let byte = packed[payload_off + 1 + i];
            let lo_nibble = (byte & 0x0F) as usize;
            let hi_nibble = ((byte >> 4) & 0x0F) as usize;
            out[b * 32 + 2 * i] = scale * E2M1_LUT[lo_nibble];
            out[b * 32 + 2 * i + 1] = scale * E2M1_LUT[hi_nibble];
        }
    }
    out
}

#[cfg(test)]
mod awq_tests {
    use super::*;
    use crate::calibration::{awq_pre_scale_weights, compute_awq_scales};
    use crate::dequant::{dequantize_e2m1_ue8m0_to_f32, e2m1_to_f32};
    use crate::model_filter::{is_q8_tensor, q8_class_of, should_quantize};
    use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
    use crate::quant_hfp4::{quantize_hfp4g32_row, quantize_mfp4g32_2d};

    /// Verify geometric mean of computed AWQ scales is ~1.0 — the
    /// normalization in compute_awq_scales should center the scale
    /// vector so downstream min-max quantization isn't perturbed.
    #[test]
    pub(crate) fn awq_scales_geomean_is_one() {
        // Realistic-ish imatrix: log-normal-ish per-channel statistics
        let in_sum2: Vec<f32> = (0..256)
            .map(|j| (1.0 + 10.0 * (j as f32 / 256.0)).exp()) // 1.0 → e^11
            .collect();
        for &alpha in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let s = compute_awq_scales(&in_sum2, alpha);
            assert_eq!(s.len(), in_sum2.len());
            // Geometric mean = exp(mean(log(s)))
            let log_mean = s.iter().map(|&v| (v as f64).ln()).sum::<f64>() / s.len() as f64;
            let geo_mean = log_mean.exp();
            assert!(
                (geo_mean - 1.0).abs() < 1e-4,
                "alpha={alpha}: geo_mean={geo_mean} (want 1.0)"
            );
        }
    }

    /// Alpha = 0 should produce all-ones scales (AWQ disabled at layer level).
    #[test]
    pub(crate) fn awq_scales_alpha_zero_is_identity() {
        let in_sum2: Vec<f32> = (1..=128).map(|j| j as f32).collect();
        let s = compute_awq_scales(&in_sum2, 0.0);
        for &v in &s {
            assert!((v - 1.0).abs() < 1e-5, "alpha=0 scale {v} should be 1.0");
        }
    }

    /// Larger imatrix values should produce larger scales for alpha > 0.
    /// Monotonicity check.
    #[test]
    pub(crate) fn awq_scales_monotonic_in_imatrix() {
        let in_sum2 = vec![1.0_f32, 4.0, 16.0, 64.0, 256.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for w in s.windows(2) {
            assert!(w[1] > w[0], "scales not monotonic: {} -> {}", w[0], w[1]);
        }
    }

    /// AWQ math identity: `(W · diag(s)) · (x / s) == W · x` at infinite
    /// precision. With fp32 weights + fp32 activations, error should be
    /// at floating-point rounding precision (~1e-5 relative).
    #[test]
    pub(crate) fn awq_math_identity_holds() {
        // Tiny test: 4 output × 8 input matmul
        let m = 4;
        let k = 8;
        // Random-ish weights and activations
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let x: Vec<f32> = (0..k).map(|j| (j as f32 + 1.0) * 0.5).collect();

        // Reference: y = W * x
        let mut y_ref = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_ref[i] += w[i * k + j] * x[j];
            }
        }

        // AWQ-scaled: pre-scale W, pre-divide x
        let in_sum2: Vec<f32> = (1..=k).map(|j| j as f32 * 10.0).collect();
        let s = compute_awq_scales(&in_sum2, 0.5);
        let mut w_scaled = w.clone();
        awq_pre_scale_weights(&mut w_scaled, m, k, &s);
        let x_div: Vec<f32> = x.iter().zip(&s).map(|(&xv, &sv)| xv / sv).collect();

        // y' = (W * diag(s)) * (x / s)
        let mut y_awq = vec![0.0_f32; m];
        for i in 0..m {
            for j in 0..k {
                y_awq[i] += w_scaled[i * k + j] * x_div[j];
            }
        }

        // Compare
        for i in 0..m {
            let rel = (y_awq[i] - y_ref[i]).abs() / y_ref[i].abs().max(1e-6);
            assert!(
                rel < 1e-5,
                "row {i}: AWQ y={} ref y={} rel_err={}",
                y_awq[i],
                y_ref[i],
                rel
            );
        }
    }

    /// Edge case: zero imatrix entries should produce finite scales
    /// (clamped via 1e-12 floor in compute_awq_scales).
    #[test]
    pub(crate) fn awq_handles_zero_imatrix() {
        let in_sum2 = vec![0.0_f32, 1.0, 4.0, 0.0];
        let s = compute_awq_scales(&in_sum2, 0.5);
        for &v in &s {
            assert!(
                v.is_finite() && v > 0.0,
                "scale {v} should be finite + positive"
            );
        }
    }
}

#[cfg(test)]
mod hfp4_tests {
    use super::*;
    use crate::dequant::{dequantize_e2m1_ue8m0_to_f32, e2m1_to_f32};
    use crate::quant_fwht::{cpu_fwht_256, gen_fwht_signs};
    use crate::quant_hfp4::{quantize_hfp4g32_row, quantize_mfp4g32_2d};

    #[test]
    pub(crate) fn e2m1_round_matches_lattice() {
        // Each lattice value should round to its own code.
        for (i, &val) in E2M1_LUT.iter().enumerate() {
            let nibble = e2m1_round(val);
            // +0 and -0 are both at value 0.0; either nibble is acceptable.
            if val.abs() < 1e-6 {
                assert!(
                    nibble == 0 || nibble == 8,
                    "zero rounds to nibble {}",
                    nibble
                );
            } else {
                assert_eq!(
                    nibble, i as u8,
                    "code {} rounded to nibble {} not {}",
                    i, nibble, i
                );
            }
        }
    }

    #[test]
    pub(crate) fn e2m1_round_midpoint() {
        // Halfway between +1.0 and +1.5 → either is acceptable (tie).
        let n = e2m1_round(1.25);
        assert!(n == 2 || n == 3, "midpoint rounded to {}", n);
        // Halfway between +4.0 and +6.0 (= 5.0) → either is acceptable.
        let n = e2m1_round(5.0);
        assert!(n == 6 || n == 7, "5.0 rounded to {}", n);
    }

    #[test]
    pub(crate) fn round_trip_constant_row() {
        // All-1.0 row: row_scale_a = 1/6, every block_e ≈ 127 + log2(1) = 127, every nibble = 2 (=1.0).
        let row = vec![1.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        for (i, &v) in recovered.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-2, "elem {} recovered to {}", i, v);
        }
    }

    #[test]
    pub(crate) fn round_trip_mixed_magnitudes() {
        // Row with mixed positive/negative E2M1 magnitudes — should round-trip exactly.
        let row: Vec<f32> = (0..64)
            .map(|i| {
                let v = E2M1_LUT[i % 16];
                v * 6.0 // scale up so row_scale_a sees max abs at 6 * 6 = 36, brings code lattice back to [-6, 6]
            })
            .collect();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        // Bound: |recovered - input| ≤ row_scale * 2^(block_e - 127) * 0.5 (half min E2M1 step).
        // With row_scale_a = 36/6 = 6, and block_max_normalized = 6, block_e = 127 → step ≈ 0.5 → tol = 3.0.
        // Actual tolerance should be much tighter for exact lattice values; allow some headroom.
        for (i, (&got, &want)) in recovered.iter().zip(row.iter()).enumerate() {
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel_err < 0.1,
                "elem {}: got {} want {} rel_err {}",
                i,
                got,
                want,
                rel_err
            );
        }
    }

    #[test]
    pub(crate) fn round_trip_per_block_error_bound() {
        // Mathematical guarantee: for every element, |recovered - original| must be ≤
        //   row_scale_a * 2^(block_e - 127) * (max_E2M1_step / 2)
        // = effective_block_scale * 1.0  (max E2M1 step is 2.0, half = 1.0)
        //
        // This is the format's correctness contract; if this fails we have a real bug.
        // NRMSE quality on raw weights is a downstream concern (MXFP4 family is documented
        // as needing rotation+smoothing for production accuracy — that's MFP4G32 in v1.5).
        let mut rng_state: u64 = 0xdead_beef_dead_beef;
        let mut next_uniform = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32).max(1e-7)
        };
        // Box-Muller Gaussian std=0.5.
        let row: Vec<f32> = (0..512)
            .flat_map(|_| {
                let u1 = next_uniform();
                let u2 = next_uniform();
                let r = (-2.0 * u1.ln()).sqrt();
                let t = 2.0 * std::f32::consts::PI * u2;
                [r * t.cos() * 0.5, r * t.sin() * 0.5]
            })
            .collect();

        let k = row.len();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, k);

        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));

        // Per-block half-max-step bound. Allow 1% slack for FP16 row-scale rounding.
        for b in 0..(k / 32) {
            let payload_off = 16 + b * 17;
            let block_e = packed[payload_off] as i32;
            let block_scale = ((block_e - 127) as f32).exp2();
            // Max E2M1 step is 2.0 (between 4 and 6); half = 1.0. Round-trip element error must
            // be ≤ effective block scale × 1.0 × (1 + slack). Slack absorbs FP16 row-scale rounding.
            let bound = row_scale_a * block_scale * 1.0 * 1.01 + 1e-5;
            for i in 0..32 {
                let idx = b * 32 + i;
                let err = (recovered[idx] - row[idx]).abs();
                assert!(
                    err <= bound,
                    "block {} elem {} err {} exceeds bound {} (block_e={}, row_scale_a={}, block_scale={})",
                    b,
                    i,
                    err,
                    bound,
                    block_e,
                    row_scale_a,
                    block_scale
                );
            }
        }
    }

    #[test]
    pub(crate) fn header_layout_matches_spec() {
        // 64 elements = 2 blocks. Row size: 16 + 2*17 = 50 bytes.
        let row = vec![3.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        assert_eq!(packed.len(), 50);
        // Block count == 2.
        let bc = u16::from_le_bytes([packed[4], packed[5]]);
        assert_eq!(bc, 2);
        // Format flags: rotation off, no row_scale_b.
        assert_eq!(packed[6] & 0x0F, 0);
        // First block UE8M0 byte at offset 16.
        // Last block payload ends at 16 + 2*17 = 50 (= total).
        // Sanity: row_scale_a > 0 (FP16 bits non-zero).
        let rs_bits = u16::from_le_bytes([packed[0], packed[1]]);
        assert_ne!(rs_bits, 0);
    }

    #[test]
    pub(crate) fn mfp4_stamps_rotation_flag() {
        // MFP4G32 must stamp format_flags = 0x05 (bit 0 + bits 2-3 = 01) in every row
        // header so loaders/tooling can detect the offline-FWHT variant. Byte length must
        // match HFP4G32 (only the flag byte and the rotated weight content differ).
        let m = 3;
        let k = 256;
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let f32_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(packed.len(), m * row_bytes, "MFP4G32 byte length mismatch");
        for r in 0..m {
            let off = r * row_bytes;
            assert_eq!(
                packed[off + 6],
                0x05,
                "row {} format_flags expected 0x05, got {:#x}",
                r,
                packed[off + 6]
            );
            // block_count must equal k/32.
            let bc = u16::from_le_bytes([packed[off + 4], packed[off + 5]]);
            assert_eq!(bc as usize, k / 32);
        }
    }

    // Orthogonality of the FWHT (`dot(R(W), R(x)) ≈ dot(W, x)`) is the load-bearing
    // correctness property and is empirically validated by `examples/test_gemv_mfp4g32.rs`
    // across K = {512, 1024, 1280, 1536, 1792, 2048} on real GPU hardware (max-abs error
    // ≤ 1.14e-5 vs 5e-3 tolerance — three orders of magnitude under). A CPU-only unit test
    // can't tighten that further without duplicating the GPU's CPU-reference path.

    #[test]
    pub(crate) fn mfp4_lloyd_round_trip_cpu() {
        let (m, k) = (64usize, 256usize);
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        // Deterministic pseudo-random weights (xorshift64).
        let mut s: u64 = 0x1234_5678_9abc_def0;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s & 0xFF_FFFF) as f32 / 0xFF_FFFF_u32 as f32) * 2.0 - 1.0
        };
        let data: Vec<f32> = (0..m * k).map(|_| 0.5 * rnd()).collect();
        let packed = quantize_mfp4g32_lloyd_2d(&data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(
            packed.len(),
            32 + m * row_bytes,
            "byte size incl 32-B prefix"
        );
        // Build reference: FWHT-rotated original (same as what dequant returns).
        let mut rotated_ref = vec![0.0f32; m * k];
        let mut row_buf = vec![0.0f32; k];
        for r in 0..m {
            row_buf.copy_from_slice(&data[r * k..(r + 1) * k]);
            for seg in 0..(k / 256) {
                cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], &signs1, &signs2);
            }
            rotated_ref[r * k..(r + 1) * k].copy_from_slice(&row_buf);
        }
        // Compare dequant output (rotated domain) to the ROTATED reference.
        let recon_rot = dequant_mfp4g32_lloyd(&packed, m, k);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..m * k {
            let e = (recon_rot[i] - rotated_ref[i]) as f64;
            num += e * e;
            den += (rotated_ref[i] as f64).powi(2);
        }
        let nrmse = (num / den).sqrt();
        // FWHT homogenizes magnitudes; 4-bit Lloyd error in rotated domain should be small.
        assert!(
            nrmse < 0.15,
            "mfp4L NRMSE {} too high (layout/codebook bug)",
            nrmse
        );
    }
}

/// K=640 feasibility probe for the 128-wide E8-SoA tier.
///
/// qwen4's routed-expert `down_proj` reduces over K = 640 = 5x128, which is not
/// a multiple of 256 — so the 256-wide E8-SoA segmentation cannot rotate it and
/// the 256-wide SoA GEMV group loop cannot cover it. This module measures both
/// halves of that claim on the CPU: the encoder's round trip at K=640, and the
/// exact block coverage of each group geometry.
#[cfg(test)]
mod e8_soa128_tests {
    use super::*;
    use crate::quant_fwht::{
        cpu_fwht_128, gen_fwht_signs, quantize_mq4g128v2, quantize_mq4g256v2, quantize_mq6g256v2,
    };

    /// qwen4 routed-expert `down_proj` reduction axis: 5 x 128.
    const K_DOWN: usize = 640;
    /// qwen4 routed-expert `gate_up` reduction axis: 10 x 256 = 20 x 128.
    const K_GATE_UP: usize = 2560;
    const M: usize = 8;

    /// Deterministic small-magnitude weights (same shape of data as real rows).
    fn probe_weights(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..m * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let unit = ((state >> 33) & 0xffff) as f32 / 65536.0;
                (unit - 0.5) * 0.04
            })
            .collect()
    }

    /// Activations are multiples of 1/128 so every value is exact in f16 (the
    /// prefill WMMA arm stages its input through f16); that rounding must not be
    /// part of the error being measured.
    fn probe_activations(rows: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..rows * k)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let value = ((state >> 40) % 257) as i32 - 128;
                value as f32 / 128.0
            })
            .collect()
    }

    /// Same 128-element rotation the runtime applies to the activation
    /// (`rotate_x_mq_128`, seeds 43/1043).
    fn rotate128(values: &[f32]) -> Vec<f32> {
        let signs1 = gen_fwht_signs(43, 128);
        let signs2 = gen_fwht_signs(1043, 128);
        let mut rotated = values.to_vec();
        for segment in rotated.chunks_mut(128) {
            cpu_fwht_128(segment, &signs1, &signs2);
        }
        rotated
    }

    fn rotate256(values: &[f32]) -> Vec<f32> {
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let mut rotated = values.to_vec();
        for segment in rotated.chunks_mut(256) {
            cpu_fwht_256(segment, &signs1, &signs2);
        }
        rotated
    }

    fn rel_l2(actual: &[f32], expected: &[f32]) -> f32 {
        assert_eq!(actual.len(), expected.len(), "probe length mismatch");
        let num: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = expected.iter().map(|b| (*b as f64).powi(2)).sum();
        (num / den.max(f64::MIN_POSITIVE)).sqrt() as f32
    }

    /// Blocks the SoA GEMV group loop visits for one row, given a group that
    /// spans `group_elems` K-elements (= `group_elems/32` 32-weight blocks).
    ///
    /// Mirrors `gemv_mfp4g32_e8_soa_gfx1151` exactly: 32 threads, 8 weights per
    /// thread per group, `pairs = groups/2` then a single `tail` group.
    fn blocks_covered(k: usize, group_elems: usize) -> Vec<usize> {
        let blocks_per_group = group_elems / 32;
        let groups = k / group_elems;
        let pairs = groups >> 1;
        let tail = groups & 1;
        let mut out = Vec::new();
        for p in 0..pairs {
            let g = p << 1;
            out.extend(g * blocks_per_group..(g + 1) * blocks_per_group);
            out.extend((g + 1) * blocks_per_group..(g + 2) * blocks_per_group);
        }
        if tail == 1 {
            let g = pairs << 1;
            out.extend(g * blocks_per_group..(g + 1) * blocks_per_group);
        }
        out
    }

    /// The blocking fact: at K=640 the shipped 256-element group geometry stops
    /// after 512 elements. This is silent in the kernel (no assert, no fault) —
    /// the last 128 K elements are simply never multiplied.
    #[test]
    fn e8_soa_256_wide_group_undercovers_k640() {
        let covered = blocks_covered(K_DOWN, 256);
        let missing: Vec<usize> = (0..K_DOWN / 32).filter(|b| !covered.contains(b)).collect();
        assert_eq!(
            missing,
            vec![16, 17, 18, 19],
            "256-wide groups must miss the last 128 of K=640"
        );
        assert_eq!(covered.len(), 16, "512 of 640 elements covered");
    }

    /// The fix's shape: a 128-element group (16 threads x 8 weights) covers
    /// K=640 and K=2560 exactly, each block once, with the shipped
    /// `pairs`/`tail` loop skeleton unchanged.
    #[test]
    fn e8_soa_128_wide_group_covers_expert_k_axes_exactly() {
        for k in [K_DOWN, K_GATE_UP] {
            let covered = blocks_covered(k, 128);
            let mut sorted = covered.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..k / 32).collect::<Vec<usize>>(),
                "128-wide groups must cover every block of K={k} exactly once"
            );
            // gate_up keeps working at 256 too, so only down_proj forces 128.
            if k == K_GATE_UP {
                assert_eq!(blocks_covered(k, 256).len(), k / 32);
            }
        }
    }

    /// K=640 end-to-end: encode with the 128-wide SoA encoder, decode through
    /// the CPU SoA decoder (bit-identical to the kernel's lattice decode), and
    /// compare both the reconstructed weights and the projected
    /// `down_proj` reduction against the exact f32 reference.
    #[test]
    fn e8_soa_128_wide_k640_down_proj_matches_f32_reference() {
        let weights = probe_weights(M, K_DOWN, 0x5eed_1234_abcd_0001);
        let signs1 = gen_fwht_signs(43, 128);
        let signs2 = gen_fwht_signs(1043, 128);
        let packed = quantize_mfp4g32_e8_soa128_2d(&weights, M, K_DOWN, &signs1, &signs2);

        // Geometry: 20 blocks, scales padded to 32 B, 16 B header.
        let row_bytes = 16 + 32 + (K_DOWN / 32) * 16;
        assert_eq!(row_bytes, 368);
        assert_eq!(packed.len(), M * row_bytes);

        let decoded_rot = dequant_mfp4g32_e8_soa(&packed, M, K_DOWN);
        let reference_rot = rotate128(&weights);

        // Wiring: the driver must be exactly "rotate in 128-wide segments,
        // then the shipped E8 row encoder, then the shipped SoA permutation".
        // Any deviation in segmentation or permutation breaks byte identity.
        let n_blocks = K_DOWN / 32;
        for r in 0..M {
            let mut row_buf = weights[r * K_DOWN..(r + 1) * K_DOWN].to_vec();
            for seg in 0..(K_DOWN / 128) {
                cpu_fwht_128(&mut row_buf[seg * 128..(seg + 1) * 128], &signs1, &signs2);
            }
            let expected = aos_to_soa_row(&quantize_mfp4g32_e8_row(&row_buf), n_blocks);
            assert_eq!(
                &packed[r * row_bytes..(r + 1) * row_bytes],
                expected.as_slice(),
                "row {r}: 128-wide SoA bytes must be the shipped row encoder, permuted"
            );
        }

        let weight_rel_l2 = rel_l2(&decoded_rot, &reference_rot);
        let worst_row = (0..M)
            .map(|r| {
                rel_l2(
                    &decoded_rot[r * K_DOWN..(r + 1) * K_DOWN],
                    &reference_rot[r * K_DOWN..(r + 1) * K_DOWN],
                )
            })
            .fold(0.0f32, f32::max);

        // The FWHT is orthonormal, so `dot(decoded_rot, x_rot) == dot(decoded, x)`
        // and the gap to the exact natural-basis projection is the tier's own
        // contribution — exactly what MoE `down_proj` computes.
        let x = probe_activations(1, K_DOWN, 0x0abc_def0_0000_0002);
        let x_rot = rotate128(&x);
        let projected: Vec<f32> = (0..M)
            .map(|r| {
                decoded_rot[r * K_DOWN..(r + 1) * K_DOWN]
                    .iter()
                    .zip(&x_rot)
                    .map(|(w, xv)| w * xv)
                    .sum::<f32>()
            })
            .collect();
        let exact: Vec<f32> = (0..M)
            .map(|r| {
                weights[r * K_DOWN..(r + 1) * K_DOWN]
                    .iter()
                    .zip(&x)
                    .map(|(w, xv)| w * xv)
                    .sum::<f32>()
            })
            .collect();
        let projection_rel_l2 = rel_l2(&projected, &exact);
        let worst_projection_row = (0..M)
            .map(|r| ((projected[r] - exact[r]) / exact[r].abs().max(f32::MIN_POSITIVE)).abs())
            .fold(0.0f32, f32::max);

        // Same projection over many more rows: the 8-row figure above is a
        // small-sample estimate whose denominator can nearly cancel in one row
        // (hence the large "worst row relative"), so report the converged value
        // alongside it rather than reading a trend into 8 samples.
        const WIDE: usize = 64;
        let wide_weights = probe_weights(WIDE, K_DOWN, 0x5eed_1234_abcd_0004);
        let wide_packed =
            quantize_mfp4g32_e8_soa128_2d(&wide_weights, WIDE, K_DOWN, &signs1, &signs2);
        let wide_decoded = dequant_mfp4g32_e8_soa(&wide_packed, WIDE, K_DOWN);
        let wide_projected: Vec<f32> = (0..WIDE)
            .map(|r| {
                wide_decoded[r * K_DOWN..(r + 1) * K_DOWN]
                    .iter()
                    .zip(&x_rot)
                    .map(|(w, xv)| w * xv)
                    .sum::<f32>()
            })
            .collect();
        let wide_exact: Vec<f32> = (0..WIDE)
            .map(|r| {
                wide_weights[r * K_DOWN..(r + 1) * K_DOWN]
                    .iter()
                    .zip(&x)
                    .map(|(w, xv)| w * xv)
                    .sum::<f32>()
            })
            .collect();
        let wide_projection_rel_l2 = rel_l2(&wide_projected, &wide_exact);

        println!(
            "E8-SoA-128 K={K_DOWN} (M={M}): weight rel_l2 = {weight_rel_l2:.5} \
             (worst row {worst_row:.5}); down_proj projection rel_l2 = \
             {projection_rel_l2:.5} (worst row relative {worst_projection_row:.5}); \
             projection rel_l2 over M={WIDE} = {wide_projection_rel_l2:.5}"
        );
        // Tripwire only, matching the "layout/codebook bug" NRMSE ceiling this
        // file already uses for mfp4L. The tier's real quality number is the
        // reported figure plus the shared-K control below, not this bound.
        assert!(
            weight_rel_l2 < 0.15,
            "128-wide E8-SoA reconstruction too coarse: {weight_rel_l2}"
        );
    }

    /// Control: at a K both segmentations can rotate, the 128-wide tier is not
    /// worse than the shipped 256-wide one. This is what makes 128-wide a
    /// format choice rather than a quality regression.
    #[test]
    fn e8_soa_128_wide_quality_matches_256_wide_at_shared_k() {
        const K: usize = 512;
        let weights = probe_weights(M, K, 0x5eed_1234_abcd_0003);
        let packed128 = quantize_mfp4g32_e8_soa128_2d(
            &weights,
            M,
            K,
            &gen_fwht_signs(43, 128),
            &gen_fwht_signs(1043, 128),
        );
        let packed256 = quantize_mfp4g32_e8_soa_2d(
            &weights,
            M,
            K,
            &gen_fwht_signs(42, 256),
            &gen_fwht_signs(1042, 256),
        );
        let rel128 = rel_l2(
            &dequant_mfp4g32_e8_soa(&packed128, M, K),
            &rotate128(&weights),
        );
        let rel256 = rel_l2(
            &dequant_mfp4g32_e8_soa(&packed256, M, K),
            &rotate256(&weights),
        );
        println!(
            "E8-SoA control K={K}: 128-wide rel_l2 = {rel128:.5}, 256-wide rel_l2 = {rel256:.5}"
        );
        assert!(
            rel128 < rel256 * 1.10,
            "128-wide tier is materially worse than 256-wide ({rel128} vs {rel256})"
        );
    }

    // ── Incumbent MQ-V2 layouts, for the iso-size comparison ────────────────
    //
    // CPU mirrors of the shipped decoders. No CPU-side decoder for qt44/qt47
    // or qt53 exists in the tree, so their layout sanity is asserted rather
    // than assumed: a wrong nibble order or header offset decodes to noise
    // (rel_l2 near 1.0), not to a 4-bit tier.

    /// qt53 MQ4G128V2: row-local, `ceil(K/128)` groups of 68 B =
    /// `[f16 scale][f16 zero][64 B nibbles]`, byte i = q[2i] | q[2i+1] << 4.
    fn decode_mq4g128v2(packed: &[u8], m: usize, k: usize) -> Vec<f32> {
        const GROUP: usize = 128;
        const BYTES: usize = 68;
        let gpr = k.div_ceil(GROUP);
        assert_eq!(packed.len(), m * gpr * BYTES, "qt53 byte extent");
        let mut out = vec![0.0f32; m * k];
        for row in 0..m {
            for g in 0..gpr {
                let base = (row * gpr + g) * BYTES;
                let scale = f16_to_f32(u16::from_le_bytes([packed[base], packed[base + 1]]));
                let zero = f16_to_f32(u16::from_le_bytes([packed[base + 2], packed[base + 3]]));
                for i in 0..64usize {
                    let byte = packed[base + 4 + i];
                    let at = g * GROUP + 2 * i;
                    if at < k {
                        out[row * k + at] = zero + scale * ((byte & 0xF) as f32);
                    }
                    if at + 1 < k {
                        out[row * k + at + 1] = zero + scale * (((byte >> 4) & 0xF) as f32);
                    }
                }
            }
        }
        out
    }

    /// qt44 MQ4G256V2 (`bits`=4) and qt47 MQ6G256V2 (`bits`=6): flat 256-weight
    /// groups, 8 B header holding two 128-weight halves, payload 128 B or 192 B.
    /// Both encoders flatten the tensor, so K % 256 == 0 is required for a group
    /// to stay inside one row — which is exactly why neither can express
    /// K=640, and why the K=640 table below has only two rows.
    fn decode_mq256(packed: &[u8], m: usize, k: usize, bits: u32) -> Vec<f32> {
        assert_eq!(k % 256, 0, "flat-G256 layouts need K%256==0");
        let group_bytes = if bits == 4 { 136 } else { 200 };
        assert_eq!(
            packed.len(),
            m * (k / 256) * group_bytes,
            "G256 byte extent"
        );
        let mut out = vec![0.0f32; m * k];
        for b in 0..m * (k / 256) {
            let base = b * group_bytes;
            for h in 0..2usize {
                let scale = f16_to_f32(u16::from_le_bytes([
                    packed[base + h * 4],
                    packed[base + h * 4 + 1],
                ]));
                let zero = f16_to_f32(u16::from_le_bytes([
                    packed[base + h * 4 + 2],
                    packed[base + h * 4 + 3],
                ]));
                for i in 0..128usize {
                    let at = h * 128 + i;
                    let q = if bits == 4 {
                        let byte = packed[base + 8 + at / 2];
                        (if at % 2 == 0 { byte & 0xF } else { byte >> 4 }) as u32
                    } else {
                        let bo = base + 8 + (at / 4) * 3;
                        let b0 = packed[bo] as u32;
                        let b1 = packed[bo + 1] as u32;
                        let b2 = packed[bo + 2] as u32;
                        match at % 4 {
                            0 => b0 & 63,
                            1 => ((b0 >> 6) | (b1 << 2)) & 63,
                            2 => ((b1 >> 4) | (b2 << 4)) & 63,
                            _ => (b2 >> 2) & 63,
                        }
                    };
                    out[b * 256 + at] = zero + scale * (q as f32);
                }
            }
        }
        out
    }

    /// Gaussian sample, sd = 0.02. Real expert weights are far closer to this
    /// than to a uniform source, and the difference is decisive for a lattice
    /// codec: a lattice's shaping gain comes from codeword *density*, which a
    /// uniform source cannot reward and a peaked source can.
    fn probe_weights_gaussian(m: usize, k: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..m * k)
            .map(|_| {
                // Irwin-Hall(12): sum of 12 uniforms, mean 6, variance 1.
                let sum: f32 = (0..12).map(|_| lcg_unit(&mut state)).sum();
                (sum - 6.0) * 0.02
            })
            .collect()
    }

    fn lcg_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) & 0xffff) as f32 / 65536.0
    }

    /// One tier's comparison row: bytes/row, weight reconstruction error in the
    /// tier's own (orthogonal) basis, and the projection against the exact f32
    /// reference, averaged over independently drawn activations.
    ///
    /// The activation average is load-bearing. With a single `x` the projection
    /// figure is a one-sample estimate whose denominator can nearly cancel in one
    /// row; for white error the expected ratio is exactly `weight_rel_l2`, so a
    /// tier that lands far below its own weight error on one `x` is reporting
    /// sampling luck, not quality. Returns (weight rel_l2, mean projection rel_l2).
    #[allow(clippy::too_many_arguments)]
    fn tier_row(
        name: &str,
        bytes_per_row: usize,
        weights: &[f32],
        rotated_reference: &[f32],
        decoded_rot: &[f32],
        activations: &[f32],
        activations_rot: &[f32],
        m: usize,
        k: usize,
    ) -> (f32, f32) {
        assert_eq!(activations.len() % k, 0, "activation buffer must tile at K");
        assert_eq!(activations.len(), activations_rot.len());
        let n_act = activations.len() / k;
        let weight_rel_l2 = rel_l2(decoded_rot, rotated_reference);
        let mut projected = vec![0.0f32; m];
        let mut exact = vec![0.0f32; m];
        let mut sum = 0.0f64;
        let mut worst = 0.0f32;
        for a in 0..n_act {
            let x = &activations[a * k..(a + 1) * k];
            let x_rot = &activations_rot[a * k..(a + 1) * k];
            for r in 0..m {
                projected[r] = decoded_rot[r * k..(r + 1) * k]
                    .iter()
                    .zip(x_rot)
                    .map(|(w, v)| w * v)
                    .sum::<f32>();
                exact[r] = weights[r * k..(r + 1) * k]
                    .iter()
                    .zip(x)
                    .map(|(w, v)| w * v)
                    .sum::<f32>();
            }
            let rel = rel_l2(&projected, &exact);
            sum += rel as f64;
            worst = worst.max(rel);
        }
        let projection_rel_l2 = (sum / n_act as f64) as f32;

        // Diagnostics that explain a weight-MSE / projection-MSE split: the
        // per-row least-squares gain (a systematic gain error is NOT averaged
        // away by the K-term dot product the way white error is), and the cosine
        // between the error and the reference (|cos| = 1 fully systematic).
        let gain: f64 = (0..m)
            .map(|r| {
                let row = &decoded_rot[r * k..(r + 1) * k];
                let reference = &rotated_reference[r * k..(r + 1) * k];
                let dot: f64 = row
                    .iter()
                    .zip(reference)
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
                dot / norm.max(f64::MIN_POSITIVE)
            })
            .sum::<f64>()
            / m as f64;
        let mut dot_er = 0.0f64;
        let mut norm_e = 0.0f64;
        let mut norm_r = 0.0f64;
        for (a, b) in decoded_rot.iter().zip(rotated_reference) {
            let error = (*a - *b) as f64;
            dot_er += error * (*b as f64);
            norm_e += error * error;
            norm_r += (*b as f64).powi(2);
        }
        let alignment = dot_er / (norm_e.sqrt() * norm_r.sqrt()).max(f64::MIN_POSITIVE);
        println!(
            "  {name:<12} {bytes_per_row:>5} B/row ({:.5} B/w)  weight rel_l2={weight_rel_l2:.5}  \
             proj rel_l2 mean={projection_rel_l2:.5} worst={worst:.5}  gain={gain:.5}  \
             err·ref cos={alignment:+.4}",
            bytes_per_row as f32 / k as f32
        );
        (weight_rel_l2, projection_rel_l2)
    }

    /// E8-SoA `[16 B header][scales padded to 16 B][n_blocks * 16 B codewords]`.
    fn e8_soa_row_bytes(k: usize) -> usize {
        let n_blocks = k / 32;
        16 + (((n_blocks + 15) >> 4) << 4) + n_blocks * 16
    }

    /// Every tier representable at `k`, on identical weights and activations.
    /// qt44/qt47 drop out when K % 256 != 0: their groups are flat and must stay
    /// inside one row.
    fn compare_tiers(label: &str, k: usize, m: usize, act: usize, weights: &[f32], seed: u64) {
        let y = probe_activations(act, k, seed);
        let y_rot128 = rotate128(&y);
        let y_rot256 = rotate256(&y);
        let rotated128 = rotate128(weights);
        let rotated256 = rotate256(weights);
        let s1_128 = gen_fwht_signs(43, 128);
        let s2_128 = gen_fwht_signs(1043, 128);
        let s1_256 = gen_fwht_signs(42, 256);
        let s2_256 = gen_fwht_signs(1042, 256);
        println!("{label}: K={k}, M={m}, {act} activations");

        let packed_e8 = quantize_mfp4g32_e8_soa128_2d(weights, m, k, &s1_128, &s2_128);
        assert_eq!(packed_e8.len(), m * e8_soa_row_bytes(k));
        let e8 = tier_row(
            "E8-SoA-128",
            e8_soa_row_bytes(k),
            weights,
            &rotated128,
            &dequant_mfp4g32_e8_soa(&packed_e8, m, k),
            &y,
            &y_rot128,
            m,
            k,
        );

        let packed_g128 = quantize_mfp4g32_e8_g128_2d(weights, m, k, &s1_128, &s2_128);
        assert_eq!(packed_g128.len(), m * mfp4g32_e8_g128_row_bytes(k));
        let e8a = tier_row(
            "E8-G128-AoS",
            mfp4g32_e8_g128_row_bytes(k),
            weights,
            &rotated128,
            &dequant_mfp4g32_e8(&packed_g128, m, k),
            &y,
            &y_rot128,
            m,
            k,
        );

        let packed53 = quantize_mq4g128v2(weights, m, k, &s1_128, &s2_128).expect("qt53 encode");
        assert_eq!(packed53.len(), m * (k / 128) * 68);
        let mq4g128 = tier_row(
            "MQ4G128V2",
            (k / 128) * 68,
            weights,
            &rotated128,
            &decode_mq4g128v2(&packed53, m, k),
            &y,
            &y_rot128,
            m,
            k,
        );
        // Layout gate: a wrong nibble order or header offset decodes to noise
        // (rel_l2 near 1.0) instead of a 4-bit tier.
        assert!(
            mq4g128.0 > 0.01 && mq4g128.0 < 0.25,
            "qt53 decoder layout is wrong: weight rel_l2 {}",
            mq4g128.0
        );

        if k % 256 == 0 {
            let packed44 = quantize_mq4g256v2(weights, m, k, &s1_256, &s2_256);
            let mq4g256 = tier_row(
                "MQ4G256V2",
                (k / 256) * 136,
                weights,
                &rotated256,
                &decode_mq256(&packed44, m, k, 4),
                &y,
                &y_rot256,
                m,
                k,
            );
            let packed47 = quantize_mq6g256v2(weights, m, k, &s1_256, &s2_256);
            let mq6 = tier_row(
                "MQ6G256V2",
                (k / 256) * 200,
                weights,
                &rotated256,
                &decode_mq256(&packed47, m, k, 6),
                &y,
                &y_rot256,
                m,
                k,
            );
            assert!(
                mq4g256.0 > 0.01 && mq4g256.0 < 0.25,
                "qt44 decoder layout is wrong: weight rel_l2 {}",
                mq4g256.0
            );
            assert!(
                mq6.0 > 0.005 && mq6.0 < 0.15,
                "qt47 decoder layout is wrong: weight rel_l2 {}",
                mq6.0
            );
        }

        assert!(
            e8.1 < 0.25,
            "128-wide E8-SoA projection error out of family: {}",
            e8.1
        );
        assert!(
            e8a.1 < 0.25,
            "128-wide E8-AoS projection error out of family: {}",
            e8a.1
        );
        // The AoS G128 row is the shipped E8 AoS row, byte for byte: the
        // rotation width is not in the wire format. Pinned here so a future
        // "optimisation" of the row stride cannot pass unnoticed.
        assert_eq!(
            mfp4g32_e8_g128_row_bytes(k),
            16 + 17 * (k / 32),
            "qt=42 row stride must equal qt=34's at K={k}"
        );
    }

    /// The tier's acceptance question: at qwen4's two expert reduction axes, on
    /// identical weights and activations, does 128-wide E8-SoA beat the MQ-V2
    /// tiers qwen4 ships today (qt53 down_proj, qt44 gate_up)?

    /// Per repo convention a tier's quality is reported from probe data, never
    /// asserted from it — the layout gates in `compare_tiers` are the tripwires.
    /// Both weight sources are run because the answer is distribution-dependent:
    /// a lattice's shaping gain needs a peaked source, which a uniform one
    /// cannot provide.
    #[test]
    fn e8_soa_128_wide_versus_incumbent_tiers_on_uniform_weights() {
        const M: usize = 64;
        const ACT: usize = 32;
        for k in [K_DOWN, K_GATE_UP] {
            compare_tiers(
                "uniform probe weights",
                k,
                M,
                ACT,
                &probe_weights(M, k, 0x5eed_1234_abcd_0004 + k as u64),
                0x0abc_def0_0000_0002 + k as u64,
            );
        }
    }

    #[test]
    fn e8_soa_128_wide_versus_incumbent_tiers_on_gaussian_weights() {
        const M: usize = 64;
        const ACT: usize = 32;
        for k in [K_DOWN, K_GATE_UP] {
            compare_tiers(
                "gaussian probe weights",
                k,
                M,
                ACT,
                &probe_weights_gaussian(M, k, 0x5eed_1234_abcd_0007 + k as u64),
                0x0abc_def0_0000_0008 + k as u64,
            );
        }
    }
    // ── Device-backed coverage + numerics (gfx1151) ─────────────────────────

    fn as_bytes_f32(values: &[f32]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
        }
    }

    fn download_f32(gpu: &rdna_compute::Gpu, t: &rdna_compute::GpuTensor, n: usize) -> Vec<f32> {
        let mut host = vec![0f32; n];
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                host.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(&host[..]),
            )
        };
        gpu.hip.memcpy_dtoh(bytes, &t.buf).expect("dtoh");
        host
    }

    /// qt=42 on device, at both of qwen4's expert reduction axes and through the
    /// MoE kernels the tier actually needs.
    ///
    /// Three things are ASSERTED rather than inferred, because all three failed
    /// silently elsewhere in this family:
    ///  1. coverage — `y` is poisoned before the launch and every element must be
    ///     overwritten. The grid under-launch of 0e0ad47b4 produced exactly the
    ///     opposite (a stale tail read back as weights-times-activation), and a
    ///     `groups_per_row = K/256` walk on K=640 produces it again. An unwritten
    ///     count is the test: `assert_eq!(unwritten, 0)`.
    ///  2. basis — the device G128 rotation must equal the CPU `cpu_fwht_128`
    ///     tables the encoder used (seeds 43/1043). A rotation-width mismatch
    ///     between weights and activation is fluent garbage, not a fault.
    ///  3. wiring — the kernel must reproduce the CPU decode GEMV, which is the
    ///     same E8 lattice body the shipped 256-wide kernels run.
    ///
    /// The tier's own error against the exact f32 product is printed, not
    /// asserted: per repo convention that number is settled on a real artifact.
    #[test]
    #[ignore = "device-backed: run `cargo test --release -p hipfire-quantize --bin hipfire-quantize -- --ignored e8g128_device --nocapture`"]
    fn e8g128_device_coverage_and_numerics() {
        const POISON: f32 = 12345.625;
        const M: usize = 64;
        let mut gpu = rdna_compute::Gpu::init().expect("device init");
        let signs1 = gen_fwht_signs(43, 128);
        let signs2 = gen_fwht_signs(1043, 128);

        for k in [K_DOWN, K_GATE_UP] {
            let weights = probe_weights(M, k, 0x5eed_1234_abcd_0009 + k as u64);
            let packed = quantize_mfp4g32_e8_g128_2d(&weights, M, k, &signs1, &signs2);
            let x = probe_activations(1, k, 0x0abc_def0_0000_000a + k as u64);
            let x_rot = rotate128(&x);

            let decoded = dequant_mfp4g32_e8(&packed, M, k);
            let reference: Vec<f32> = (0..M)
                .map(|r| {
                    decoded[r * k..(r + 1) * k]
                        .iter()
                        .zip(&x_rot)
                        .map(|(w, v)| w * v)
                        .sum()
                })
                .collect();
            let exact: Vec<f32> = (0..M)
                .map(|r| {
                    weights[r * k..(r + 1) * k]
                        .iter()
                        .zip(&x)
                        .map(|(w, v)| w * v)
                        .sum()
                })
                .collect();

            let w_gpu = gpu
                .alloc_tensor(&[packed.len()], rdna_compute::DType::Raw)
                .expect("alloc w");
            gpu.hip.memcpy_htod(&w_gpu.buf, &packed).expect("htod w");
            let x_gpu = gpu
                .alloc_tensor(&[k], rdna_compute::DType::F32)
                .expect("alloc x");
            gpu.hip
                .memcpy_htod(&x_gpu.buf, as_bytes_f32(&x))
                .expect("htod x");
            let rot_gpu = gpu
                .alloc_tensor(&[k], rdna_compute::DType::F32)
                .expect("alloc rot");
            let y_gpu = gpu
                .alloc_tensor(&[M], rdna_compute::DType::F32)
                .expect("alloc y");
            gpu.hip
                .memcpy_htod(&y_gpu.buf, as_bytes_f32(&vec![POISON; M]))
                .expect("poison y");

            gpu.rotate_x_mq_128(&x_gpu, &rot_gpu, k)
                .expect("rotate_x_mq_128");
            gpu.gemv_mfp4g32_e8g128(&w_gpu, &rot_gpu, &y_gpu, M, k)
                .expect("gemv_mfp4g32_e8g128");
            gpu.hip.device_synchronize().expect("sync");

            let rot_device = download_f32(&gpu, &rot_gpu, k);
            let basis_rel = rel_l2(&rot_device, &x_rot);
            let y = download_f32(&gpu, &y_gpu, M);
            let unwritten = y.iter().filter(|v| **v == POISON).count();
            let wiring = rel_l2(&y, &reference);
            let format = rel_l2(&y, &exact);
            println!(
                "qt=42 dense K={k}: unwritten={unwritten}/{M} basis rel_l2={basis_rel:.3e} \
                 wiring rel_l2={wiring:.3e} format rel_l2={format:.5}"
            );
            assert_eq!(
                unwritten, 0,
                "K={k}: {unwritten} of {M} outputs never written"
            );
            assert!(
                basis_rel < 1e-5,
                "K={k}: device G128 rotation disagrees with the encoder's basis: {basis_rel}"
            );
            assert!(
                wiring < 1e-3,
                "K={k}: kernel disagrees with the CPU decode GEMV: {wiring}"
            );
        }

        // MoE down (the projection whose reduction axis is K=640) and MoE gate_up.
        // These are the kernels the tier needs for the k8-indexed decode path.
        for k in [K_DOWN, K_GATE_UP] {
            // k_top = 8 is the production decode width, so the wrapper the decode
            // call site uses is exercised at its real rank count rather than
            // through a batch-of-one shortcut.
            const K_TOP: usize = 8;
            let weights = probe_weights(M, k, 0x5eed_1234_abcd_000b + k as u64);
            let packed = quantize_mfp4g32_e8_g128_2d(&weights, M, k, &signs1, &signs2);
            let x = probe_activations(1, k, 0x0abc_def0_0000_000c + k as u64);
            let x_rot = rotate128(&x);
            let decoded = dequant_mfp4g32_e8(&packed, M, k);
            let reference: Vec<f32> = (0..M)
                .map(|r| {
                    decoded[r * k..(r + 1) * k]
                        .iter()
                        .zip(&x_rot)
                        .map(|(w, v)| w * v)
                        .sum()
                })
                .collect();

            let w_gpu = gpu
                .alloc_tensor(&[packed.len()], rdna_compute::DType::Raw)
                .expect("alloc w");
            gpu.hip.memcpy_htod(&w_gpu.buf, &packed).expect("htod w");
            let ptrs = gpu
                .alloc_tensor(&[8], rdna_compute::DType::Raw)
                .expect("alloc ptrs");
            let expert_ptr = [w_gpu.buf.as_ptr() as u64];
            let ptr_bytes = unsafe {
                std::slice::from_raw_parts(
                    expert_ptr.as_ptr() as *const u8,
                    std::mem::size_of_val(&expert_ptr),
                )
            };
            gpu.hip
                .memcpy_htod(&ptrs.buf, ptr_bytes)
                .expect("htod ptrs");
            let topk = gpu
                .alloc_tensor(&[K_TOP * 4], rdna_compute::DType::Raw)
                .expect("alloc topk");
            // All 8 ranks route to the single expert under test, so every rank's
            // output must equal the same reference.
            let topk_idx: [i32; K_TOP] = [0; K_TOP];
            let topk_bytes = unsafe {
                std::slice::from_raw_parts(
                    topk_idx.as_ptr() as *const u8,
                    std::mem::size_of_val(&topk_idx),
                )
            };
            gpu.hip
                .memcpy_htod(&topk.buf, topk_bytes)
                .expect("htod topk");
            let rot_batch = gpu
                .alloc_tensor(&[K_TOP * k], rdna_compute::DType::F32)
                .expect("alloc rot_batch");
            let mut rot_all = Vec::with_capacity(K_TOP * k);
            for _ in 0..K_TOP {
                rot_all.extend_from_slice(&x_rot);
            }
            gpu.hip
                .memcpy_htod(&rot_batch.buf, as_bytes_f32(&rot_all))
                .expect("htod rot_batch");
            let out = gpu
                .alloc_tensor(&[K_TOP * M], rdna_compute::DType::F32)
                .expect("alloc out");
            gpu.hip
                .memcpy_htod(&out.buf, as_bytes_f32(&vec![POISON; K_TOP * M]))
                .expect("poison out");

            gpu.gemv_mfp4g32_e8g128_moe_down_k8_indexed_batched_expanded(
                &ptrs, &topk, &rot_batch, &out, M, k, K_TOP, 1,
            )
            .expect("moe down g128");
            gpu.hip.device_synchronize().expect("sync");
            let y = download_f32(&gpu, &out, K_TOP * M);
            let unwritten = y.iter().filter(|v| **v == POISON).count();
            let worst_rank = (0..K_TOP)
                .map(|r| rel_l2(&y[r * M..(r + 1) * M], &reference))
                .fold(0.0f32, f32::max);
            println!(
                "qt=42 MoE down K={k}: unwritten={unwritten}/{} wiring rel_l2 worst-of-{K_TOP}={worst_rank:.3e}",
                K_TOP * M
            );
            assert_eq!(
                unwritten, 0,
                "MoE down K={k}: {unwritten} outputs never written"
            );
            assert!(
                worst_rank < 1e-3,
                "MoE down K={k}: kernel disagrees: {worst_rank}"
            );

            // gate_up is the gate||up concatenation: the operand has 2M rows and
            // row < M is gate, row >= M is up. Encoding only M rows here would
            // make the kernel read past the tensor for the `up` half.
            let full = 2 * M;
            let gu_weights = probe_weights(full, k, 0x5eed_1234_abcd_000d + k as u64);
            let gu_packed = quantize_mfp4g32_e8_g128_2d(&gu_weights, full, k, &signs1, &signs2);
            let gu_decoded = dequant_mfp4g32_e8(&gu_packed, full, k);
            let gu_ref: Vec<f32> = (0..full)
                .map(|r| {
                    gu_decoded[r * k..(r + 1) * k]
                        .iter()
                        .zip(&x_rot)
                        .map(|(w, v)| w * v)
                        .sum()
                })
                .collect();
            let gu_gpu = gpu
                .alloc_tensor(&[gu_packed.len()], rdna_compute::DType::Raw)
                .expect("alloc gu");
            gpu.hip
                .memcpy_htod(&gu_gpu.buf, &gu_packed)
                .expect("htod gu");
            let gu_ptrs = gpu
                .alloc_tensor(&[8], rdna_compute::DType::Raw)
                .expect("alloc gu ptrs");
            let gu_ptr = [gu_gpu.buf.as_ptr() as u64];
            let gu_ptr_bytes = unsafe {
                std::slice::from_raw_parts(
                    gu_ptr.as_ptr() as *const u8,
                    std::mem::size_of_val(&gu_ptr),
                )
            };
            gpu.hip
                .memcpy_htod(&gu_ptrs.buf, gu_ptr_bytes)
                .expect("htod gu ptrs");

            let gate = gpu
                .alloc_tensor(&[K_TOP * M], rdna_compute::DType::F32)
                .expect("alloc gate");
            let up = gpu
                .alloc_tensor(&[K_TOP * M], rdna_compute::DType::F32)
                .expect("alloc up");
            gpu.hip
                .memcpy_htod(&gate.buf, as_bytes_f32(&vec![POISON; K_TOP * M]))
                .expect("poison gate");
            gpu.hip
                .memcpy_htod(&up.buf, as_bytes_f32(&vec![POISON; K_TOP * M]))
                .expect("poison up");
            // The single-token wrapper the decode call site actually uses.
            gpu.gemv_mfp4g32_e8g128_moe_gate_up_k8_indexed(
                &gu_ptrs, &topk, &rot_batch, &gate, &up, full, k,
            )
            .expect("moe gate_up g128 (single-token wrapper)");
            gpu.hip.device_synchronize().expect("sync");
            let gate_v = download_f32(&gpu, &gate, K_TOP * M);
            let up_v = download_f32(&gpu, &up, K_TOP * M);
            let unwritten_gate = gate_v.iter().filter(|v| **v == POISON).count();
            let unwritten_up = up_v.iter().filter(|v| **v == POISON).count();
            let worst_gate = (0..K_TOP)
                .map(|r| rel_l2(&gate_v[r * M..(r + 1) * M], &gu_ref[..M]))
                .fold(0.0f32, f32::max);
            let worst_up = (0..K_TOP)
                .map(|r| rel_l2(&up_v[r * M..(r + 1) * M], &gu_ref[M..]))
                .fold(0.0f32, f32::max);
            println!(
                "qt=42 MoE gate_up K={k}: unwritten={}/{} gate rel_l2 worst-of-{K_TOP}={worst_gate:.3e} \
                 up rel_l2 worst-of-{K_TOP}={worst_up:.3e}",
                unwritten_gate + unwritten_up,
                2 * K_TOP * M
            );
            assert_eq!(
                unwritten_gate + unwritten_up,
                0,
                "MoE gate_up K={k}: outputs never written"
            );
            assert!(
                worst_gate < 1e-3 && worst_up < 1e-3,
                "MoE gate_up K={k}: kernel disagrees: gate {worst_gate}, up {worst_up}"
            );
        }
    }
    /// qt=42 grouped-WMMA prefill on device. The kernel's K loop was flattened
    /// from `group -> tile` to a single 16-value tile loop, so this asserts the
    /// two properties that rewrite could break: every output written (poisoned
    /// buffer, counted) and the flattened order still reproducing the reference.
    ///
    /// The reference mirrors the kernel's own operand rounding — the A operand is
    /// E8-decoded then rounded to f16, the B operand is the activation staged
    /// through f16 by `ensure_fp16_x` — so the residual is accumulation drift, not
    /// representation error. One expert, one 16-slot tile, `x_row_div = 1`.
    #[test]
    #[ignore = "device-backed: run `cargo test --release -p hipfire-quantize --bin hipfire-quantize -- --ignored e8g128_device_grouped --nocapture`"]
    fn e8g128_device_grouped_wmma_coverage_and_numerics() {
        const POISON: f32 = 12345.625;
        const M: usize = 16;
        const M_TOTAL: usize = 16;
        let mut gpu = rdna_compute::Gpu::init().expect("device init");
        let signs1 = gen_fwht_signs(43, 128);
        let signs2 = gen_fwht_signs(1043, 128);

        for k in [K_DOWN, K_GATE_UP] {
            let weights = probe_weights(M, k, 0x5eed_1234_abcd_000e + k as u64);
            let packed = quantize_mfp4g32_e8_g128_2d(&weights, M, k, &signs1, &signs2);
            let decoded = dequant_mfp4g32_e8(&packed, M, k);
            let x = probe_activations(M_TOTAL, k, 0x0abc_def0_0000_000f + k as u64);
            let x_rot = rotate128(&x);

            // Kernel-side rounding: A goes through f16 after the scale multiply,
            // B through f16 via ensure_fp16_x.
            let a_reg: Vec<f32> = decoded.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
            let x_f16: Vec<f32> = x_rot.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect();
            let mut reference = vec![0f32; M_TOTAL * M];
            for slot in 0..M_TOTAL {
                for m in 0..M {
                    reference[slot * M + m] =
                        (0..k).map(|i| a_reg[m * k + i] * x_f16[slot * k + i]).sum();
                }
            }

            let w_gpu = gpu
                .alloc_tensor(&[packed.len()], rdna_compute::DType::Raw)
                .expect("alloc w");
            gpu.hip.memcpy_htod(&w_gpu.buf, &packed).expect("htod w");
            let ptrs = gpu
                .alloc_tensor(&[8], rdna_compute::DType::Raw)
                .expect("alloc ptrs");
            let expert_ptr = [w_gpu.buf.as_ptr() as u64];
            let ptr_bytes = unsafe {
                std::slice::from_raw_parts(
                    expert_ptr.as_ptr() as *const u8,
                    std::mem::size_of_val(&expert_ptr),
                )
            };
            gpu.hip
                .memcpy_htod(&ptrs.buf, ptr_bytes)
                .expect("htod ptrs");

            let tile_ids: [i32; 1] = [0; 1];
            let tile_gpu = gpu
                .alloc_tensor(&[4], rdna_compute::DType::Raw)
                .expect("alloc tile ids");
            let tile_bytes = unsafe {
                std::slice::from_raw_parts(
                    tile_ids.as_ptr() as *const u8,
                    std::mem::size_of_val(&tile_ids),
                )
            };
            gpu.hip
                .memcpy_htod(&tile_gpu.buf, tile_bytes)
                .expect("htod tile ids");

            let sorted: Vec<i32> = (0..M_TOTAL as i32).collect();
            let sorted_gpu = gpu
                .alloc_tensor(&[M_TOTAL * 4], rdna_compute::DType::Raw)
                .expect("alloc sorted");
            let sorted_bytes = unsafe {
                std::slice::from_raw_parts(
                    sorted.as_ptr() as *const u8,
                    std::mem::size_of_val(&sorted[..]),
                )
            };
            gpu.hip
                .memcpy_htod(&sorted_gpu.buf, sorted_bytes)
                .expect("htod sorted");

            let x_gpu = gpu
                .alloc_tensor(&[M_TOTAL * k], rdna_compute::DType::F32)
                .expect("alloc x");
            gpu.hip
                .memcpy_htod(&x_gpu.buf, as_bytes_f32(&x_rot))
                .expect("htod x");
            let y_gpu = gpu
                .alloc_tensor(&[M_TOTAL * M], rdna_compute::DType::F32)
                .expect("alloc y");
            gpu.hip
                .memcpy_htod(&y_gpu.buf, as_bytes_f32(&vec![POISON; M_TOTAL * M]))
                .expect("poison y");

            gpu.gemm_mfp4g32_e8g128_moe_grouped_wmma(
                &ptrs,
                &tile_gpu,
                &sorted_gpu,
                &x_gpu,
                &y_gpu,
                M,
                k,
                1,
                M_TOTAL,
                M_TOTAL,
            )
            .expect("grouped wmma g128");
            gpu.hip.device_synchronize().expect("sync");

            let y = download_f32(&gpu, &y_gpu, M_TOTAL * M);
            let unwritten = y.iter().filter(|v| **v == POISON).count();
            let wiring = rel_l2(&y, &reference);
            println!(
                "qt=42 grouped WMMA K={k}: unwritten={unwritten}/{} wiring rel_l2={wiring:.3e}",
                M_TOTAL * M
            );
            assert_eq!(
                unwritten, 0,
                "grouped WMMA K={k}: {unwritten} outputs never written"
            );
            assert!(
                wiring < 5e-3,
                "grouped WMMA K={k}: flattened tile loop disagrees with the reference: {wiring}"
            );
        }
    }
}
