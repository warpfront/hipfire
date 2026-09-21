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

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use crate::e8;
use crate::e8_gptq;
use crate::gguf_input;
use crate::reap_overlay;
use clap::Parser;
use hipfire_quantize::float16::{bf16_to_f32, f16_to_f32, f32_to_f16};
use hipfire_quantize::hessian_io;
use hipfire_quantize::safetensors_file::{SafetensorsFile, TensorMeta};

// ─── HFQ4-G256 Quantization ─────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights (0.531 B/w).
/// 18 VGPRs, 100% occupancy on RDNA1. Beats Q4_K at all matrix sizes.
/// CPU-side FWHT (Walsh-Hadamard Transform) on a 256-element group.
/// Matches the GPU-side fwht_forward_256 in turbo_common: signs1 → butterfly → scale → signs2.
pub(crate) fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 {
        x[i] *= signs1[i];
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 {
        x[i] *= scale * signs2[i];
    }
}

/// Generate FWHT sign table (matches engine's gen_fwht_signs).
pub(crate) fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// MagnumQuant HFQ4-G256: FWHT-rotated 4-bit quantization.
/// Same binary format as HFQ4-G256 (136 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
pub(crate) fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

pub(crate) const MQ4V2_GROUP_BYTES: usize = 136;
pub(crate) const MQ4C_GROUP_BYTES: usize = 136;
pub(crate) const MQ6V2_GROUP_BYTES: usize = 200;
pub(crate) const MQ5V2_GROUP_BYTES: usize = 168;

/// MQ4CG256 encoder — single-scale per-256 asymmetric, fp16 header, 136 B/group (pad layout).
///
/// For each 256-weight group after FWHT:
///   lo = min(group); hi = max(group)
///   step_f32 = (hi - lo) / 15; scale=f16(step); zero=f16(lo)
///   st=f32(scale); z=f32(zero); // round-tripped
///   for i in 0..256: q[i]=clamp(rint((w[i]-z)/st),0,15)
/// Degenerate group (hi==lo): scale=0, zero=f16(lo), all q=0.
///
/// Pad layout (per weight tensor with `m` rows, `gpr = K/256` groups/row, `n = m*gpr`):
///   per group, 136 B stride: `[0..4)` fp16 header, `[4..8)` zero padding, `[8..136)` 128 B nibbles.
pub(crate) fn quantize_mq4cg256(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    let group_size = 256;
    let n = w.len();
    assert!(k % 256 == 0, "MQ4CG256 requires K % 256 == 0, got K={k}");
    let gpr = k / 256;
    let total_groups = m * gpr;
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let expected_len = total_groups * MQ4C_GROUP_BYTES;
    assert_eq!(expected_len, m * gpr * 136, "total bytes must be m*gpr*136");
    assert_eq!(expected_len, total_groups * 136);
    let mut output = vec![0u8; expected_len];
    for b in 0..total_groups {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&w[start..end]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let lo = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
        let sc_bits = if hi == lo { 0u16 } else { f32_to_f16(step_f32) };
        let z_bits = f32_to_f16(lo);
        let st = f16_to_f32(sc_bits);
        let z = f16_to_f32(z_bits);
        let degenerate = hi == lo || step_f32 == 0.0 || st == 0.0;
        let base = b * 136;
        output[base..base + 2].copy_from_slice(&sc_bits.to_le_bytes());
        output[base + 2..base + 4].copy_from_slice(&z_bits.to_le_bytes());
        if degenerate {
        } else {
            let inv = 1.0 / st;
            let mut q = [0u8; 256];
            for i in 0..256 {
                let qq = ((group[i] - z) * inv + 0.5).floor().clamp(0.0, 15.0) as u8;
                q[i] = qq;
            }
            for i in 0..128 {
                let lo_q = q[2 * i];
                let hi_q = q[2 * i + 1];
                output[base + 8 + i] = (lo_q & 0xF) | ((hi_q & 0xF) << 4);
            }
        }
    }
    assert_eq!(output.len(), m * gpr * 136);
    output
}

/// MQ4G256 v2 encoder — per-128 asymmetric, fp16 header, byte-identical nibble payload to v1.
///
/// See `docs/quant-formats/mq4-v2.md` §3. For each 256-weight group after FWHT:
/// for each half `h` in {0,1} over weights `128h..128h+127`:
///   lo = min(w in half); hi = max(w in half)
///   step_f32 = (hi-lo)/15; scale[h]=f16(step); zero[h]=f16(lo)
///   st=f32(scale[h]); z=f32(zero[h]); // round-tripped
///   for i in half: q[i]=clamp(rint((w[i]-z)/st),0,15)
/// Degenerate half (hi==lo): scale=0, zero=f16(lo), all q=0.
/// Header: [0..2) fp16 scale h0, [2..4) fp16 zero h0, [4..6) fp16 scale h1, [6..8) fp16 zero h1, [8..136) nibbles.
fn write_mq4g256v2_group(
    output: &mut [u8],
    scales: [u16; 2],
    zeros: [u16; 2],
    codes: &[u8],
) {
    debug_assert_eq!(output.len(), MQ4V2_GROUP_BYTES);
    debug_assert_eq!(codes.len(), 256);
    output[..2].copy_from_slice(&scales[0].to_le_bytes());
    output[2..4].copy_from_slice(&zeros[0].to_le_bytes());
    output[4..6].copy_from_slice(&scales[1].to_le_bytes());
    output[6..8].copy_from_slice(&zeros[1].to_le_bytes());
    for i in 0..128 {
        output[8 + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
    }
}

/// Pack already-selected MQ4V2 codes without applying FWHT or rounding again.
///
/// `headers` is one `[scale0, zero0, scale1, zero1]` F16-bit tuple per
/// 256-weight group. `codes` is row-major and contains one unpacked uint4 code
/// per weight. This is the export boundary used by learned-rounding tools.
pub(crate) fn pack_mq4g256v2_from_codes(
    m: usize,
    k: usize,
    headers: &[[u16; 4]],
    codes: &[u8],
) -> Result<Vec<u8>, String> {
    if k % 256 != 0 {
        return Err(format!("MQ4V2 learned codes require K % 256 == 0, got K={k}"));
    }
    let expected_codes = m
        .checked_mul(k)
        .ok_or_else(|| format!("MQ4V2 shape overflows: {m}x{k}"))?;
    if codes.len() != expected_codes {
        return Err(format!(
            "MQ4V2 learned code count {} != M*K {}",
            codes.len(),
            expected_codes
        ));
    }
    let group_count = expected_codes / 256;
    if headers.len() != group_count {
        return Err(format!(
            "MQ4V2 learned header count {} != M*K/256 {}",
            headers.len(),
            group_count
        ));
    }
    if let Some((index, code)) = codes.iter().enumerate().find(|(_, code)| **code > 15) {
        return Err(format!("MQ4V2 code {code} at index {index} exceeds uint4"));
    }

    let mut output = vec![0u8; group_count * MQ4V2_GROUP_BYTES];
    for group in 0..group_count {
        let header = headers[group];
        let group_codes = &codes[group * 256..(group + 1) * 256];
        for half in 0..2 {
            if header[half * 2] == 0
                && group_codes[half * 128..(half + 1) * 128]
                    .iter()
                    .any(|&code| code != 0)
            {
                return Err(format!(
                    "MQ4V2 group {group} half {half} has zero scale with nonzero codes"
                ));
            }
        }
        write_mq4g256v2_group(
            &mut output[group * MQ4V2_GROUP_BYTES..(group + 1) * MQ4V2_GROUP_BYTES],
            [header[0], header[2]],
            [header[1], header[3]],
            group_codes,
        );
    }
    Ok(output)
}

/// Pack the frozen C3 `[M,K/256,2,2]` F16 `(d,z)` grid and unpacked U8
/// codes directly. This function only serializes: it never applies FWHT or
/// selects a grid/code, so trained codes cross the export boundary exactly once.
pub(crate) fn pack_mq4g256v2_from_f16_grid(
    m: usize,
    k: usize,
    d_z_f16: &[u8],
    codes: &[u8],
) -> Result<Vec<u8>, String> {
    if k % 256 != 0 {
        return Err(format!("MQ4V2 final codes require K % 256 == 0, got K={k}"));
    }
    let group_count = m
        .checked_mul(k)
        .ok_or_else(|| format!("MQ4V2 shape overflows: {m}x{k}"))?
        / 256;
    let expected_grid_bytes = group_count
        .checked_mul(8)
        .ok_or_else(|| format!("MQ4V2 grid byte count overflows: {m}x{k}"))?;
    if d_z_f16.len() != expected_grid_bytes {
        return Err(format!(
            "MQ4V2 d_z_f16 byte count {} != M*K/256*2*2*2 {}",
            d_z_f16.len(),
            expected_grid_bytes
        ));
    }
    let headers: Vec<[u16; 4]> = d_z_f16
        .chunks_exact(8)
        .map(|bytes| {
            [
                u16::from_le_bytes([bytes[0], bytes[1]]),
                u16::from_le_bytes([bytes[2], bytes[3]]),
                u16::from_le_bytes([bytes[4], bytes[5]]),
                u16::from_le_bytes([bytes[6], bytes[7]]),
            ]
        })
        .collect();
    pack_mq4g256v2_from_codes(m, k, &headers, codes)
}

/// Round-to-nearest-even into the fp8 E4M3 grid (1 sign, 4 exponent bits
/// bias 8, 3 mantissa bits), returned as f32. Models what a fold-free fp8
/// kernel computes for the mantissa product: `e4m3((code - 8) * m) * 2^e`.
/// Steps: |x| in [8,16) step 1, [4,8) step 0.5, [2,4) step 0.25, etc.
/// Normal-range grid; our folded operands (|x| <= 12 or exact 0) never hit
/// the subnormal or saturation paths, which clamp defensively.
fn e4m3_rne(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let neg = x < 0.0;
    let ax = x.abs();
    let mut exp = ax.log2().floor() as i32;
    exp = exp.clamp(-6, 8);
    let step = 2f32.powi(exp - 3);
    let q = ax / step;
    let lo = q.floor();
    let frac = q - lo;
    let round_up = frac > 0.5 || (frac == 0.5 && (lo as i64 & 1) == 1);
    let mut n = if round_up { lo + 1.0 } else { lo };
    let mut e = exp;
    if n >= 16.0 {
        n = 8.0;
        e += 1;
    }
    let mut v = n * 2f32.powi(e - 3);
    if v > 448.0 {
        v = 448.0;
    }
    if neg { -v } else { v }
}

pub(crate) fn quantize_mq4g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mq4g256v2_impl(w, m, k, signs1, signs2, false, false, false)
}

/// MQ4V2 with a symmetric per-128 grid in the existing affine wire format.
///
/// The stored zero point is exactly `-8*d`, so code 8 reconstructs zero and
/// shipped decoders remain byte-format compatible. The four scale candidates
/// mirror the runtime A4 producer ladder, shifted to the 15-interval midpoint
/// grid; selection includes fp16 round-trip error.
pub(crate) fn quantize_mq4g256v2_symmetric(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mq4g256v2_impl(w, m, k, signs1, signs2, true, false, false)
}

/// MQ4V2 symmetric variant with the per-128 scale `d` restricted to an exact
/// power of two (zero f16 mantissa), zero = f16(-8d). Wire format unchanged.
/// Candidates bracket base = amax/7.5: {2^floor(log2 base), 2^ceil(log2 base)}
/// as exact f16 powers of two (0/subnormal skipped). Candidate MSE and final
/// code choice are fold-honest: reconstruction is e4m3((code-8) * m) * 2^e,
/// what a fold-free fp8 kernel computes (identity here since m = 1 keeps
/// (code-8) exactly on the e4m3 grid).
pub(crate) fn quantize_mq4g256v2_symmetric_pow2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mq4g256v2_impl(w, m, k, signs1, signs2, true, true, false)
}

/// MQ4V2 symmetric half-pow2 variant: `d = m * 2^e` with m in {1, 1.5}
/// (f16 mantissa 0x000 or 0x200), zero = f16(-8d). Four candidates bracket
/// base = amax/7.5 (floor/ceil octave x {1, 1.5}); fold-honest MSE and code
/// choice as above (1.5x products round into e4m3, e.g. 10.5 -> 10).
pub(crate) fn quantize_mq4g256v2_symmetric_pow2_half(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    quantize_mq4g256v2_impl(w, m, k, signs1, signs2, true, true, true)
}

fn quantize_mq4g256v2_impl(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
    symmetric: bool,
    pow2_scale: bool,
    pow2_half: bool,
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ4V2 requires K % 256 == 0, got K={k}");
    assert_eq!(w.len(), m * k, "w.len() {} != m*k {}*{}={}", w.len(), m, k, m * k);
    let group_count = w.len() / 256;
    let mut output = vec![0u8; group_count * MQ4V2_GROUP_BYTES];
    for group_index in 0..group_count {
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&w[group_index * 256..(group_index + 1) * 256]);
        cpu_fwht_256(&mut group, signs1, signs2);

        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut codes = [0u8; 256];
        for half in 0..2 {
            let offset = half * 128;
            let values = &group[offset..offset + 128];
            if symmetric {
                let amax = values.iter().fold(0.0f32, |acc, &value| acc.max(value.abs()));
                if amax == 0.0 {
                    scales[half] = 0;
                    zeros[half] = f32_to_f16(-0.0);
                    codes[offset..offset + 128].fill(8);
                    continue;
                }

                if pow2_scale {
                    // Fold-honest scale search: d = m * 2^e with m = 1, or
                    // m in {1, 1.5} for the half variant (f16 mantissa 0x000
                    // or 0x200). Octaves bracket base = amax/7.5 (up to 4
                    // candidates: floor/ceil x {1, 1.5}); 0/subnormal skipped.
                    // A fold-free fp8 kernel reconstructs e4m3((code-8) * m)
                    // * 2^e, so both candidate MSE and the final per-code
                    // choice search all 16 codes under that reconstruction.
                    let base = amax / 7.5;
                    let log2b = base.log2();
                    let floor_e = log2b.floor() as i32;
                    let ceil_e = log2b.ceil() as i32;
                    let mut exponents = [floor_e, ceil_e];
                    if floor_e == ceil_e {
                        exponents[1] = i32::MIN;
                    }
                    let mantissas: &[f32] = if pow2_half { &[1.0, 1.5] } else { &[1.0] };
                    // (f16 scale bits, mantissa m, octave e)
                    let mut candidates = [(0u16, 1.0f32, 0i32); 4];
                    let mut candidate_count = 0;
                    for &mantissa in mantissas {
                        for &exponent in &exponents {
                            if exponent == i32::MIN {
                                continue;
                            }
                            let biased = exponent + 15;
                            if !(1..=30).contains(&biased) {
                                continue;
                            }
                            let mut bits = (biased as u16) << 10;
                            if mantissa == 1.5 {
                                bits |= 0x0200;
                            }
                            candidates[candidate_count] = (bits, mantissa, exponent);
                            candidate_count += 1;
                        }
                    }
                    if candidate_count == 0 {
                        let clamped = (floor_e + 15).clamp(1, 30);
                        candidates[0] = ((clamped as u16) << 10, 1.0, clamped - 15);
                        candidate_count = 1;
                    }
                    let mut best_mse = f64::INFINITY;
                    let mut best_codes = [0u8; 128];
                    for i in 0..candidate_count {
                        let (scale_bits, mantissa, exponent) = candidates[i];
                        let two_e = 2f32.powi(exponent);
                        let mut recon = [0.0f32; 16];
                        for code in 0..16 {
                            recon[code] = e4m3_rne((code as f32 - 8.0) * mantissa) * two_e;
                        }
                        let mut mse = 0.0f64;
                        let mut trial = [0u8; 128];
                        for (j, &value) in values.iter().enumerate() {
                            let mut best_code = 0usize;
                            let mut best_err = f64::INFINITY;
                            for code in 0..16 {
                                let err = (value - recon[code]) as f64;
                                let sq = err * err;
                                if sq < best_err {
                                    best_err = sq;
                                    best_code = code;
                                }
                            }
                            trial[j] = best_code as u8;
                            mse += best_err;
                        }
                        if mse < best_mse {
                            best_mse = mse;
                            scales[half] = scale_bits;
                            zeros[half] = f32_to_f16(-8.0 * f16_to_f32(scale_bits));
                            best_codes = trial;
                        }
                    }
                    codes[offset..offset + 128].copy_from_slice(&best_codes);
                } else {
                    let candidate_base = (amax / 7.5) * 0.5;
                    let multipliers = [
                        1.0f32,
                        f32::from_bits(0x3fa4_9249),
                        f32::from_bits(0x3fdb_6db7),
                        2.0f32,
                    ];
                    let mut best_mse = f64::INFINITY;
                    for multiplier in multipliers {
                        let scale_bits = f32_to_f16(candidate_base * multiplier);
                        let scale = f16_to_f32(scale_bits);
                        if scale == 0.0 {
                            continue;
                        }
                        let zero_bits = f32_to_f16(-8.0 * scale);
                        let zero = f16_to_f32(zero_bits);
                        let inverse = 1.0 / scale;
                        let mut mse = 0.0f64;
                        for &value in values {
                            let code = ((value - zero) * inverse + 0.5).floor().clamp(0.0, 15.0);
                            let error = value - code.mul_add(scale, zero);
                            mse += (error as f64) * (error as f64);
                        }
                        if mse < best_mse {
                            best_mse = mse;
                            scales[half] = scale_bits;
                            zeros[half] = zero_bits;
                        }
                    }
                    let scale = f16_to_f32(scales[half]);
                    let zero = f16_to_f32(zeros[half]);
                    let inverse = 1.0 / scale;
                    for i in 0..128 {
                        codes[offset + i] =
                            ((group[offset + i] - zero) * inverse + 0.5).floor().clamp(0.0, 15.0)
                                as u8;
                    }
                }
            } else {
                let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let step_f32 = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                scales[half] = if hi == lo { 0 } else { f32_to_f16(step_f32) };
                zeros[half] = f32_to_f16(lo);
                let scale = f16_to_f32(scales[half]);
                let zero = f16_to_f32(zeros[half]);
                if hi != lo && step_f32 != 0.0 && scale != 0.0 {
                    let inverse = 1.0 / scale;
                    for i in 0..128 {
                        codes[offset + i] =
                            ((group[offset + i] - zero) * inverse + 0.5).floor().clamp(0.0, 15.0)
                                as u8;
                    }
                }
            }
        }
        write_mq4g256v2_group(
            &mut output[group_index * MQ4V2_GROUP_BYTES
                ..(group_index + 1) * MQ4V2_GROUP_BYTES],
            scales,
            zeros,
            &codes,
        );
    }
    output
}
/// MQ6G256V2 encoder — per-128 asymmetric fp16 header, neutral-size GEMM/GEMV.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..200) 192B packed 6-bit`.
/// Payload unchanged from MQ6G256: 4 values per 3 bytes (q0 | q1<<6, etc).
/// Half 0 covers q[0..128), half 1 q[128..256); reconstruction q*f32(s[h])+f32(z[h]).
/// Encoder round-trips s/z through fp16 before selecting q; degenerate half: scale=0, zero=f16(lo), q=0.
/// K%256 required.
pub(crate) fn quantize_mq6g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ6G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ6V2_GROUP_BYTES;
    let mut output = vec![0u8; total_groups * block_bytes];
    for b in 0..total_groups {
        let start = b * 256;
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&w[start..start + 256]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut sts = [0.0f32; 2];
        let mut zs = [0.0f32; 2];
        let mut degenerate = [false; 2];
        for h in 0..2 {
            let off = h * 128;
            let slice = &group[off..off + 128];
            let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step_f32 = if hi > lo { (hi - lo) / 63.0 } else { 0.0 };
            let mut sc_bits = f32_to_f16(step_f32);
            if hi == lo {
                sc_bits = 0u16;
            }
            let z_bits = f32_to_f16(lo);
            let st = f16_to_f32(sc_bits);
            let z = f16_to_f32(z_bits);
            scales[h] = sc_bits;
            zeros[h] = z_bits;
            sts[h] = st;
            zs[h] = z;
            degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
        }
        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&scales[0].to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&zeros[0].to_le_bytes());
        output[out_off + 4..out_off + 6].copy_from_slice(&scales[1].to_le_bytes());
        output[out_off + 6..out_off + 8].copy_from_slice(&zeros[1].to_le_bytes());
        let mut q = [0u8; 256];
        for h in 0..2 {
            let off = h * 128;
            if degenerate[h] {
                for i in 0..128 {
                    q[off + i] = 0;
                }
            } else {
                let st = sts[h];
                let z = zs[h];
                let inv = 1.0 / st;
                for i in 0..128 {
                    let v = group[off + i];
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 63.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in (0..256).step_by(4) {
            let bo = out_off + 8 + (i / 4) * 3;
            let q0 = q[i] & 63;
            let q1 = q[i + 1] & 63;
            let q2 = q[i + 2] & 63;
            let q3 = q[i + 3] & 63;
            output[bo] = q0 | (q1 << 6);
            output[bo + 1] = (q1 >> 2) | (q2 << 4);
            output[bo + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}
/// MQ5G256V2 encoder — per-128 asymmetric fp16 header, neutral-size.
///
/// Layout per 256-weight group: `[0..2) fp16 s0,[2..4) fp16 z0,[4..6) fp16 s1,[6..8) fp16 z1,[8..168) 160B packed 5-bit`.
/// Payload unchanged from MQ5G256: 8 values per 5 bytes.
pub(crate) fn quantize_mq5g256v2(
    w: &[f32],
    m: usize,
    k: usize,
    signs1: &[f32],
    signs2: &[f32],
) -> Vec<u8> {
    assert!(k % 256 == 0, "MQ5G256V2 requires K % 256 == 0, got K={k}");
    let n = w.len();
    assert_eq!(n, m * k, "w.len() {} != m*k {}*{}={}", n, m, k, m * k);
    let gpr = k / 256;
    let total_groups = m * gpr;
    let block_bytes = MQ5V2_GROUP_BYTES;
    let mut output = vec![0u8; total_groups * block_bytes];
    for b in 0..total_groups {
        let start = b * 256;
        let mut group = [0.0f32; 256];
        group.copy_from_slice(&w[start..start + 256]);
        cpu_fwht_256(&mut group, signs1, signs2);
        let mut scales = [0u16; 2];
        let mut zeros = [0u16; 2];
        let mut sts = [0.0f32; 2];
        let mut zs = [0.0f32; 2];
        let mut degenerate = [false; 2];
        for h in 0..2 {
            let off = h * 128;
            let slice = &group[off..off + 128];
            let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let step_f32 = if hi > lo { (hi - lo) / 31.0 } else { 0.0 };
            let mut sc_bits = f32_to_f16(step_f32);
            if hi == lo {
                sc_bits = 0u16;
            }
            let z_bits = f32_to_f16(lo);
            let st = f16_to_f32(sc_bits);
            let z = f16_to_f32(z_bits);
            scales[h] = sc_bits;
            zeros[h] = z_bits;
            sts[h] = st;
            zs[h] = z;
            degenerate[h] = hi == lo || step_f32 == 0.0 || st == 0.0;
        }
        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&scales[0].to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&zeros[0].to_le_bytes());
        output[out_off + 4..out_off + 6].copy_from_slice(&scales[1].to_le_bytes());
        output[out_off + 6..out_off + 8].copy_from_slice(&zeros[1].to_le_bytes());
        let mut q = [0u8; 256];
        for h in 0..2 {
            let off = h * 128;
            if degenerate[h] {
                for i in 0..128 {
                    q[off + i] = 0;
                }
            } else {
                let st = sts[h];
                let z = zs[h];
                let inv = 1.0 / st;
                for i in 0..128 {
                    let v = group[off + i];
                    let qq = ((v - z) * inv + 0.5).floor().clamp(0.0, 31.0) as u8;
                    q[off + i] = qq;
                }
            }
        }
        for i in (0..256).step_by(8) {
            let bo = out_off + 8 + (i / 8) * 5;
            let q0 = q[i] & 31;
            let q1 = q[i + 1] & 31;
            let q2 = q[i + 2] & 31;
            let q3 = q[i + 3] & 31;
            let q4 = q[i + 4] & 31;
            let q5 = q[i + 5] & 31;
            let q6 = q[i + 6] & 31;
            let q7 = q[i + 7] & 31;
            output[bo] = q0 | (q1 << 5);
            output[bo + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
            output[bo + 2] = (q3 >> 1) | (q4 << 4);
            output[bo + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
            output[bo + 4] = (q6 >> 2) | (q7 << 3);
        }
    }
    output
}
/// MagnumQuant MQ6-G256: FWHT-rotated 6-bit quantization.
/// Same binary format as HFQ6-G256 (200 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
pub(crate) fn quantize_mq6g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = ((group[i] - min_val) * inv_scale + 0.5) as u8;
            let q1 = ((group[i + 1] - min_val) * inv_scale + 0.5) as u8;
            let q2 = ((group[i + 2] - min_val) * inv_scale + 0.5) as u8;
            let q3 = ((group[i + 3] - min_val) * inv_scale + 0.5) as u8;
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off] = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }

    output
}

/// MagnumQuant MQ5-G256: FWHT-rotated 5-bit quantization.
/// 168 bytes/group = 8 B affine header (f32 scale + f32 min) + 160 B payload
/// (5 bits x 256 weights = 1280 bits). 5.25 bpw. Sits between MQ4 (136 B/group)
/// and MQ6 (200 B/group). The 5-bit codes cross byte boundaries: 8 values pack
/// into 5 bytes (8*5 = 40 bits). The rotation is baked into the weights; the
/// GEMV kernel rotates x instead of inverse-rotating w.
///
/// Pack layout (q0..q7 each 5-bit, clamped to 31), per 5-byte chunk:
///   b0 = q0        | (q1 << 5)              // q0[4:0], q1[2:0]
///   b1 = (q1 >> 3) | (q2 << 2) | (q3 << 7)  // q1[4:3], q2[4:0], q3[0]
///   b2 = (q3 >> 1) | (q4 << 4)              // q3[4:1], q4[3:0]
///   b3 = (q4 >> 4) | (q5 << 1) | (q6 << 6)  // q4[4], q5[4:0], q6[1:0]
///   b4 = (q6 >> 2) | (q7 << 3)              // q6[4:2], q7[4:0]
/// The HIP unpacker reverses this (read 5 bytes -> 8 codes), then
///   val = q * scale + min.
pub(crate) fn quantize_mq5g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 168; // 8 (scale+min) + 160 (packed 5-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 31.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 8 values per 5 bytes (8*5 = 40 bits). 256/8 = 32 chunks ->
        // 32*5 = 160 payload bytes. byte_off = 8 + (i/8)*5.
        for i in (0..256).step_by(8) {
            let q0 = (((group[i] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q1 = (((group[i + 1] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q2 = (((group[i + 2] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q3 = (((group[i + 3] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q4 = (((group[i + 4] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q5 = (((group[i + 5] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q6 = (((group[i + 6] - min_val) * inv_scale + 0.5) as u8).min(31);
            let q7 = (((group[i + 7] - min_val) * inv_scale + 0.5) as u8).min(31);

            let byte_off = 8 + (i / 8) * 5;
            output[out_off + byte_off] = q0 | (q1 << 5);
            output[out_off + byte_off + 1] = (q1 >> 3) | (q2 << 2) | (q3 << 7);
            output[out_off + byte_off + 2] = (q3 >> 1) | (q4 << 4);
            output[out_off + byte_off + 3] = (q4 >> 4) | (q5 << 1) | (q6 << 6);
            output[out_off + byte_off + 4] = (q6 >> 2) | (q7 << 3);
        }
    }

    output
}

/// MagnumQuant MQ8-G256: FWHT-rotated symmetric INT8 quantization.
/// Format: [f16 scale][int8 × 256] = 258 bytes per 256 weights (1.008 B/w).
/// Symmetric: scale = max(abs(group)) / 127, q = round(val / scale), no zero-point.
/// Target: dp4a (v_dot4_i32_iu8) on gfx1100 for 4x VALU throughput.
pub(crate) fn quantize_mq8g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 258; // 2 (f16 scale) + 256 (int8 values)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation
        cpu_fwht_256(&mut group, signs1, signs2);

        // Symmetric quantization: scale = max(|val|) / 127
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv_scale = if amax > 0.0 { 127.0 / amax } else { 0.0 };

        let out_off = b * block_bytes;
        // Store scale as f16 (2 bytes)
        let scale_f16 = f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;

        // Quantize to signed INT8
        for i in 0..256 {
            let q = (group[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

pub(crate) fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 128 bytes of nibbles
        // byte[i] = weight[2*i] (lo nibble) | weight[2*i+1] (hi nibble)
        for i in 0..128 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len {
                group[idx_lo]
            } else {
                min_val
            };
            let hi_val = if idx_hi < actual_len {
                group[idx_hi]
            } else {
                min_val
            };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mq4g256v2_roundtrip_shape() {
        let m = 2usize;
        let k = 256usize;
        let w: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01).sin()).collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let blob = quantize_mq4g256v2(&w, m, k, &s1, &s2);
        let n_blocks = (w.len() + 255) / 256;
        assert_eq!(blob.len(), n_blocks * MQ4V2_GROUP_BYTES);
        assert_eq!(MQ4V2_GROUP_BYTES, 136);
        // header 8B + 128B payload per block, nibbles present
        assert!(blob.len() >= 136);
    }

    #[test]
    fn mq4g256v2_learned_code_export_is_byte_exact() {
        let m = 2usize;
        let k = 512usize;
        let w: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32 * 0.03125).sin() * 3.0) + (i % 11) as f32)
            .collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let encoded = quantize_mq4g256v2(&w, m, k, &s1, &s2);
        let mut headers = Vec::new();
        let mut codes = Vec::new();
        for block in encoded.chunks_exact(MQ4V2_GROUP_BYTES) {
            headers.push([
                u16::from_le_bytes([block[0], block[1]]),
                u16::from_le_bytes([block[2], block[3]]),
                u16::from_le_bytes([block[4], block[5]]),
                u16::from_le_bytes([block[6], block[7]]),
            ]);
            for &packed in &block[8..] {
                codes.push(packed & 0x0f);
                codes.push(packed >> 4);
            }
        }
        let repacked = pack_mq4g256v2_from_codes(m, k, &headers, &codes).unwrap();
        assert_eq!(repacked, encoded);
        let d_z_f16: Vec<u8> = headers
            .iter()
            .flat_map(|header| header.iter().flat_map(|bits| bits.to_le_bytes()))
            .collect();
        let repacked_from_grid =
            pack_mq4g256v2_from_f16_grid(m, k, &d_z_f16, &codes).unwrap();
        assert_eq!(repacked_from_grid, encoded);
    }

    #[test]
    fn mq4cg256_pad_layout_136() {
        let m = 2usize;
        let k = 512usize;
        let w: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.5).collect();
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        let blob = quantize_mq4cg256(&w, m, k, &s1, &s2);
        let gpr = k / 256;
        assert_eq!(blob.len(), m * gpr * MQ4C_GROUP_BYTES);
        assert_eq!(MQ4C_GROUP_BYTES, 136);
        // pad bytes [4..8) must be zero
        for g in 0..m * gpr {
            let base = g * 136;
            assert_eq!(
                &blob[base + 4..base + 8],
                &[0u8; 4],
                "pad bytes must be zero at group {g}"
            );
            // header scale/zero are fp16 values — at least not all zero for non-degenerate
            // (first group should have non-zero scale for this synthetic data)
        }
    }

    #[test]
    fn mq4_flags_map_to_quant_types() {
        // codec mapping reachability: labels map via QuantType and GgufFormat
        use crate::hfq::QuantType;
        use crate::pipeline_gguf::GgufFormat;
        assert_eq!(GgufFormat::from_flag("mq4"), Some(GgufFormat::Mq4V2));
        assert_eq!(GgufFormat::from_flag("mq4v2"), Some(GgufFormat::Mq4V2));
        assert_eq!(
            GgufFormat::from_flag("mq4v2-lloyd"),
            Some(GgufFormat::Mq4V2Lloyd)
        );
        assert_eq!(GgufFormat::from_flag("mq4l"), Some(GgufFormat::Mq4V2Lloyd));
        assert_eq!(GgufFormat::from_flag("mq4c"), Some(GgufFormat::Mq4C));
        assert_eq!(GgufFormat::from_flag("mq4v1"), Some(GgufFormat::Mq4));
        assert_eq!(GgufFormat::from_flag("mq4g256v2"), Some(GgufFormat::Mq4V2));
        assert_eq!(GgufFormat::from_flag("mq4cg256"), Some(GgufFormat::Mq4C));
        assert_eq!(QuantType::from_u8(44), Some(QuantType::MQ4G256V2));
        assert_eq!(QuantType::from_u8(45), Some(QuantType::MQ4CG256));
        assert_eq!(QuantType::MQ4G256V2 as u8, 44);
        assert_eq!(QuantType::MQ4CG256 as u8, 45);
        // new V2 family
        assert_eq!(GgufFormat::from_flag("mq6v2"), Some(GgufFormat::Mq6V2));
        assert_eq!(GgufFormat::from_flag("mq5v2"), Some(GgufFormat::Mq5V2));
        assert_eq!(GgufFormat::from_flag("mq3v2"), Some(GgufFormat::Mq3V2));
        assert_eq!(GgufFormat::from_flag("mq2v2"), Some(GgufFormat::Mq2V2));
        assert_eq!(GgufFormat::from_flag("mq6g256v2"), Some(GgufFormat::Mq6V2));
        assert_eq!(GgufFormat::from_flag("mq2g256v2"), Some(GgufFormat::Mq2V2));
        assert_eq!(QuantType::from_u8(47), Some(QuantType::MQ6G256V2));
        assert_eq!(QuantType::from_u8(48), Some(QuantType::MQ5G256V2));
        assert_eq!(QuantType::from_u8(49), Some(QuantType::MQ3G256V2));
        assert_eq!(QuantType::from_u8(50), Some(QuantType::MQ2G256V2));
        assert_eq!(QuantType::MQ6G256V2 as u8, 47);
        assert_eq!(QuantType::MQ5G256V2 as u8, 48);
        assert_eq!(QuantType::MQ3G256V2 as u8, 49);
        assert_eq!(QuantType::MQ2G256V2 as u8, 50);
        // 51 = MQ2G256LloydU, the unrotated MQ2-Lloyd sibling (Maple native
        // ternary), claimed 2026-08-22. This line previously pinned 51 as free.
        assert_eq!(QuantType::from_u8(51), Some(QuantType::MQ2G256LloydU));
        assert_eq!(QuantType::MQ2G256LloydU as u8, 51);
        // 52 = MQ4G256V2L (MQ4v2 + per-tensor Lloyd codebook). Claimed with this PR.
        assert_eq!(QuantType::from_u8(52), Some(QuantType::MQ4G256V2L));
        assert_eq!(QuantType::MQ4G256V2L as u8, 52);
        // unknown remains rejected — 46 is the next genuinely free id
        assert_eq!(QuantType::from_u8(46), None);
        assert_eq!(QuantType::from_u8(255), None);
    }

    #[test]
    fn mq6v2_mq5v2_wire_layout() {
        let s1 = gen_fwht_signs(42, 256);
        let s2 = gen_fwht_signs(1042, 256);
        // byte count: m*ceil(k/256)*B
        for (m, k, b) in [
            (1usize, 256usize, MQ6V2_GROUP_BYTES),
            (2, 512, MQ6V2_GROUP_BYTES),
            (4, 1024, MQ6V2_GROUP_BYTES),
        ] {
            let w = vec![0.1f32; m * k];
            let blob = quantize_mq6g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq6v2 bytes m={m} k={k}");
            assert_eq!(b, 200);
        }
        for (m, k, b) in [
            (1usize, 256usize, MQ5V2_GROUP_BYTES),
            (3, 768, MQ5V2_GROUP_BYTES),
        ] {
            let w = vec![-0.2f32; m * k];
            let blob = quantize_mq5g256v2(&w, m, k, &s1, &s2);
            assert_eq!(blob.len(), m * (k / 256) * b, "mq5v2 bytes m={m} k={k}");
            assert_eq!(b, 168);
        }
        // fp16 header positions, payload offset, degenerate halves. FWHT sign
        // multiplication can turn +0.0 into -0.0, and the wire contract stores
        // f16(lo), so either signed-zero encoding is valid.
        let m = 2;
        let k = 256;
        let w_zero = vec![0.0f32; m * k];
        for (name, group_bytes, blob) in [
            (
                "mq6v2",
                MQ6V2_GROUP_BYTES,
                quantize_mq6g256v2(&w_zero, m, k, &s1, &s2),
            ),
            (
                "mq5v2",
                MQ5V2_GROUP_BYTES,
                quantize_mq5g256v2(&w_zero, m, k, &s1, &s2),
            ),
        ] {
            for g in 0..m * (k / 256) {
                let base = g * group_bytes;
                let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
                let z0 = u16::from_le_bytes([blob[base + 2], blob[base + 3]]);
                let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
                let z1 = u16::from_le_bytes([blob[base + 6], blob[base + 7]]);
                assert_eq!(s0, 0, "{name} degenerate scale half0 must be 0");
                assert_eq!(s1b, 0, "{name} degenerate scale half1 must be 0");
                assert_eq!(z0 & 0x7fff, 0, "{name} half0 zero must be signed f16 zero");
                assert_eq!(z1 & 0x7fff, 0, "{name} half1 zero must be signed f16 zero");
                assert!(
                    blob[base + 8..base + group_bytes]
                        .iter()
                        .all(|&byte| byte == 0),
                    "{name} degenerate payload must be 0"
                );
            }
        }
        // non-degenerate: ensure header bytes are valid fp16 and payload non-zero
        let w_rand: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.007).sin() * 2.0).collect();
        let blob = quantize_mq5g256v2(&w_rand, m, k, &s1, &s2);
        for g in 0..m * (k / 256) {
            let base = g * MQ5V2_GROUP_BYTES;
            let s0 = u16::from_le_bytes([blob[base], blob[base + 1]]);
            let s1b = u16::from_le_bytes([blob[base + 4], blob[base + 5]]);
            // At least one half must be non-degenerate for this synthetic data
            assert!(s0 != 0 || s1b != 0, "expected non-degenerate header");
            // payload after 8 must contain packed codes (not all zero)
            // 5-bit payload is 160B, check not all zero
            let payload = &blob[base + 8..base + MQ5V2_GROUP_BYTES];
            assert_eq!(payload.len(), MQ5V2_GROUP_BYTES - 8);
            assert!(payload.iter().any(|&b| b != 0));
        }
    }
}
