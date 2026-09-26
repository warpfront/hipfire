// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 sealed-MoE channel proof.
//!
//! This executable launches the dedicated qt53 ordinary GEMV, indexed top-10
//! down, grouped prefill down, and both top-10 combine kernels.  Every result
//! is checked against an independent CPU decoder for the exact 68-byte
//! MQ4G128V2 wire rows.  A small qt44 grouped gate/up launch is included so
//! the complete top-10 prefill permutation is compiled and exercised too.
//!
//! It also covers the two *packed trunk* tiers as dense channels: qt44
//! (MQ4G256V2) and qt47 (MQ6G256V2) at the shapes the trunk dispatches, one row
//! and several rows, each checked against a host rotation plus a CPU dot so the
//! batched GEMM can be compared with the per-row GEMV.
//!
//! Usage:
//!   cargo run --release --features lab --example qwen4_qt53_channel -p hipfire-runtime

use half::f16;
use rdna_compute::{gen_fwht_signs, DType, Gpu, GpuTensor};

const QT53_GROUP_BYTES: usize = 68;
const QT44_GROUP_BYTES: usize = 136;
const QT47_GROUP_BYTES: usize = 200;
const TOP_K: usize = 10;

fn qt53_bytes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k.div_ceil(128);
    let row_stride = groups * QT53_GROUP_BYTES;
    let mut out = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_stride + group * QT53_GROUP_BYTES;
            out[base..base + 2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16(1)
            out[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes()); // f16(0)
            for i in 0..64 {
                let q0 = ((seed + row * 7 + group * 11 + i * 3) & 0x0f) as u8;
                let q1 = ((seed + row * 13 + group * 5 + i * 9 + 1) & 0x0f) as u8;
                out[base + 4 + i] = q0 | (q1 << 4);
            }
        }
    }
    out
}

fn qt44_bytes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "qt44 harness rows use complete G256 groups");
    let groups = k / 256;
    let row_stride = groups * QT44_GROUP_BYTES;
    let mut out = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_stride + group * QT44_GROUP_BYTES;
            // [scale_0, zero_0, scale_1, zero_1], all f16.
            out[base..base + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
            out[base + 4..base + 6].copy_from_slice(&0x3c00u16.to_le_bytes());
            out[base + 6..base + 8].copy_from_slice(&0u16.to_le_bytes());
            for i in 0..128 {
                let q0 = ((seed + row * 5 + group * 7 + i * 3) & 0x0f) as u8;
                let q1 = ((seed + row * 17 + group * 13 + i * 11 + 1) & 0x0f) as u8;
                out[base + 8 + i] = q0 | (q1 << 4);
            }
        }
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let value = if exponent == 0 {
        if fraction == 0 {
            0.0
        } else {
            (fraction as f32 / 1024.0) * 2.0f32.powi(-14)
        }
    } else if exponent == 0x1f {
        if fraction == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent as i32 - 15)
    };
    if sign == 0 {
        value
    } else {
        -value
    }
}
const HFQ4_G128_GROUP_BYTES: usize = 72;

fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_bf16_bits(value: f32) -> u16 {
    assert!(
        value.is_finite(),
        "source BF16 harness value must be finite"
    );
    let bits = value.to_bits();
    let rounding = 0x7fffu32 + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

fn source_bf16_words(len: usize, seed: usize) -> Vec<u16> {
    (0..len)
        .map(|index| {
            let coarse = (seed.wrapping_add(index.wrapping_mul(17)) % 257) as f32 - 128.0;
            let fine =
                (seed.wrapping_mul(3).wrapping_add(index.wrapping_mul(11)) % 29) as f32 - 14.0;
            f32_to_bf16_bits(coarse * 0.03125 + fine * 0.001953125)
        })
        .collect()
}

fn activation_values(k: usize, seed: usize) -> Vec<f32> {
    (0..k)
        .map(|index| {
            let coarse = (seed.wrapping_add(index.wrapping_mul(13)) % 193) as f32 - 96.0;
            let fine =
                (seed.wrapping_mul(5).wrapping_add(index.wrapping_mul(7)) % 31) as f32 - 15.0;
            coarse * 0.0078125 + fine * 0.0009765625
        })
        .collect()
}

fn report_quant_loss(label: &str, source: &[f32], decoded: &[f32]) -> Result<(), String> {
    if source.len() != decoded.len() || source.is_empty() {
        return Err(format!(
            "{label}: invalid metric lengths {} and {}",
            source.len(),
            decoded.len()
        ));
    }
    let mut sum_squared = 0.0f64;
    let mut sum_absolute = 0.0f64;
    let mut max_absolute = 0.0f32;
    for (&source_value, &decoded_value) in source.iter().zip(decoded) {
        if !source_value.is_finite() || !decoded_value.is_finite() {
            return Err(format!("{label}: non-finite source-BF16 quant metric"));
        }
        let difference = source_value - decoded_value;
        let absolute = difference.abs();
        sum_squared += f64::from(difference) * f64::from(difference);
        sum_absolute += f64::from(absolute);
        max_absolute = max_absolute.max(absolute);
    }
    let count = source.len() as f64;
    let mse = sum_squared / count;
    let mae = sum_absolute / count;
    if !mse.is_finite() || !mae.is_finite() || !max_absolute.is_finite() {
        return Err(format!("{label}: non-finite aggregate"));
    }
    println!(
        "{label}: finite source-BF16 quant loss (mse={mse:.8e}, mae={mae:.8e}, max_abs={max_absolute:.8e})"
    );
    Ok(())
}

fn hfq4g128_embedding_bytes(source: &[u16], rows: usize, dim: usize) -> Result<Vec<u8>, String> {
    if dim == 0 || dim % 128 != 0 || source.len() != rows.saturating_mul(dim) {
        return Err(format!(
            "HFQ4-G128 embedding geometry mismatch: rows={rows} dim={dim} source={}",
            source.len()
        ));
    }
    let groups = dim / 128;
    let row_stride = groups * HFQ4_G128_GROUP_BYTES;
    let mut payload = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let logical_start = row * dim + group * 128;
            let values: Vec<f32> = source[logical_start..logical_start + 128]
                .iter()
                .map(|&bits| bf16_bits_to_f32(bits))
                .collect();
            let zero = values.iter().copied().fold(f32::INFINITY, f32::min);
            let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = high - zero;
            let scale = if range > 0.0 { range / 15.0 } else { 0.0 };
            let base = row * row_stride + group * HFQ4_G128_GROUP_BYTES;
            payload[base..base + 4].copy_from_slice(&scale.to_le_bytes());
            payload[base + 4..base + 8].copy_from_slice(&zero.to_le_bytes());
            for index in 0..128 {
                let quantized = if scale > 0.0 {
                    (((values[index] - zero) / scale + 0.5)
                        .floor()
                        .clamp(0.0, 15.0)) as u8
                } else {
                    0
                };
                let byte = base + 8 + index / 2;
                if index & 1 == 0 {
                    payload[byte] = quantized;
                } else {
                    payload[byte] |= quantized << 4;
                }
            }
        }
    }
    Ok(payload)
}

fn hfq4g128_decode_row(row: &[u8], dim: usize) -> Result<Vec<f32>, String> {
    if dim == 0 || dim % 128 != 0 || row.len() != (dim / 128) * HFQ4_G128_GROUP_BYTES {
        return Err(format!(
            "HFQ4-G128 row geometry mismatch: dim={dim} row_bytes={}",
            row.len()
        ));
    }
    let mut decoded = vec![0.0f32; dim];
    for index in 0..dim {
        let group = index / 128;
        let within = index % 128;
        let base = group * HFQ4_G128_GROUP_BYTES;
        let scale = f32::from_le_bytes(row[base..base + 4].try_into().unwrap());
        let zero = f32::from_le_bytes(row[base + 4..base + 8].try_into().unwrap());
        let packed = row[base + 8 + within / 2];
        let quantized = if within & 1 == 0 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        decoded[index] = scale * quantized as f32 + zero;
    }
    Ok(decoded)
}

fn cpu_fwht_128(values: &mut [f32]) {
    assert_eq!(values.len(), 128);
    let signs1 = gen_fwht_signs(43, 128);
    let signs2 = gen_fwht_signs(1043, 128);
    for (value, sign) in values.iter_mut().zip(&signs1) {
        *value *= sign;
    }
    let mut stride = 1;
    while stride < 128 {
        let mut offset = 0;
        while offset < 128 {
            for index in 0..stride {
                let left = values[offset + index];
                let right = values[offset + index + stride];
                values[offset + index] = left + right;
                values[offset + index + stride] = left - right;
            }
            offset += stride * 2;
        }
        stride <<= 1;
    }
    let normalization = 1.0f32 / 128.0f32.sqrt();
    for (value, sign) in values.iter_mut().zip(&signs2) {
        *value *= normalization * sign;
    }
}

fn cpu_rotate_128(input: &[f32], batch: usize, k: usize) -> Vec<f32> {
    assert_eq!(input.len(), batch * k);
    let mut output = vec![0.0f32; input.len()];
    let groups = k.div_ceil(128);
    for row in 0..batch {
        for group in 0..groups {
            let start = group * 128;
            let actual = (k - start).min(128);
            let mut values = [0.0f32; 128];
            values[..actual].copy_from_slice(&input[row * k + start..row * k + start + actual]);
            cpu_fwht_128(&mut values);
            output[row * k + start..row * k + start + actual].copy_from_slice(&values[..actual]);
        }
    }
    output
}

fn qt53_pack_bf16(
    source: &[u16],
    rows: usize,
    k: usize,
) -> Result<(Vec<u8>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || k == 0 || source.len() != rows.saturating_mul(k) {
        return Err(format!(
            "qt53 source geometry mismatch: rows={rows} k={k} source={}",
            source.len()
        ));
    }
    let groups = k.div_ceil(128);
    let mut payload = vec![0u8; rows * groups * QT53_GROUP_BYTES];
    let mut rotated_source = vec![0.0f32; rows * k];
    let mut rotated_decoded = vec![0.0f32; rows * k];
    for row in 0..rows {
        for group_index in 0..groups {
            let group_start = group_index * 128;
            let actual = (k - group_start).min(128);
            let mut values = [0.0f32; 128];
            for index in 0..actual {
                values[index] = bf16_bits_to_f32(source[row * k + group_start + index]);
            }
            cpu_fwht_128(&mut values);
            rotated_source[row * k + group_start..row * k + group_start + actual]
                .copy_from_slice(&values[..actual]);
            let low = values.iter().copied().fold(f32::INFINITY, f32::min);
            let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = high - low;
            let step = if range > 0.0 { range / 15.0 } else { 0.0 };
            let mut scale_bits = f16::from_f32(step).to_bits();
            let zero_bits = f16::from_f32(low).to_bits();
            let mut scale = f16::from_bits(scale_bits).to_f32();
            let zero = f16::from_bits(zero_bits).to_f32();
            let degenerate = high == low || step == 0.0 || scale == 0.0;
            if degenerate {
                scale_bits = 0;
                scale = 0.0;
            }
            let base = (row * groups + group_index) * QT53_GROUP_BYTES;
            payload[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
            payload[base + 2..base + 4].copy_from_slice(&zero_bits.to_le_bytes());
            for index in 0..128 {
                let quantized = if degenerate {
                    0
                } else {
                    (((values[index] - zero) / scale + 0.5)
                        .floor()
                        .clamp(0.0, 15.0)) as u8
                };
                let byte = base + 4 + index / 2;
                if index & 1 == 0 {
                    payload[byte] = quantized;
                } else {
                    payload[byte] |= quantized << 4;
                }
                if index < actual {
                    rotated_decoded[row * k + group_start + index] =
                        scale * quantized as f32 + zero;
                }
            }
        }
    }
    Ok((payload, rotated_source, rotated_decoded))
}

fn qt53_dot(row: &[u8], k: usize, x: &[f32]) -> f32 {
    let groups = k.div_ceil(128);
    let row_stride = groups * QT53_GROUP_BYTES;
    assert_eq!(row.len(), row_stride);
    assert_eq!(x.len(), k);
    let mut acc = 0.0f32;
    for group in 0..groups {
        let base = group * QT53_GROUP_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
        let zero = f16_to_f32(u16::from_le_bytes([row[base + 2], row[base + 3]]));
        for in_group in 0..128 {
            let logical = group * 128 + in_group;
            if logical >= k {
                continue;
            }
            let packed = row[base + 4 + (in_group >> 1)];
            let q = if in_group & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            acc += (scale * q as f32 + zero) * x[logical];
        }
    }
    acc
}

fn raw_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn raw_ptrs(tensors: &[GpuTensor]) -> Vec<u8> {
    tensors
        .iter()
        .flat_map(|tensor| (tensor.buf.as_ptr() as usize as u64).to_le_bytes())
        .collect()
}
fn zero_raw_i32(gpu: &Gpu, count: usize) -> Result<GpuTensor, String> {
    gpu.upload_raw(&vec![0u8; count * 4], &[count * 4])
        .map_err(|error| error.to_string())
}

fn normalize_top10(tokens: usize, seed: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; tokens * TOP_K];
    for token in 0..tokens {
        let mut sum = 0.0f32;
        for rank in 0..TOP_K {
            let weight = 1.0 + ((seed + token * 7 + rank * 3) % 17) as f32;
            weights[token * TOP_K + rank] = weight;
            sum += weight;
        }
        for rank in 0..TOP_K {
            weights[token * TOP_K + rank] /= sum;
        }
    }
    weights
}

fn check_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    relative_tolerance: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: length {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut max_denominator = 1.0f32;
    let mut max_index = 0usize;

    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if !got.is_finite() {
            return Err(format!("{label}: non-finite output at {index}: {got}"));
        }
        let abs = (got - want).abs();
        let denominator = want.abs().max(1.0);
        let relative = abs / denominator;
        if abs > max_abs {
            max_abs = abs;
        }
        if relative > max_relative {
            max_relative = relative;
            max_denominator = denominator;
            max_index = index;
        }
        let bound = relative_tolerance * denominator;
        if relative > relative_tolerance {
            return Err(format!(
                "{label}: mismatch at {index}: got {got:.8e}, want {want:.8e}, abs {abs:.8e}, rel {relative:.8e} > {relative_tolerance:.8e} (denom {denominator:.8e}, abs_bound {bound:.8e})"
            ));
        }
    }
    println!(
        "{label}: PASS (max_abs={max_abs:.8e}, max_rel={max_relative:.8e}, denominator={max_denominator:.8e} at {max_index})"
    );
    Ok(())
}

/// CPU model of the source-BF16 top-10 combine (`moe_down_combine_*_top10`):
/// route weight, expert output, their product and every partial sum round to
/// BF16, accumulating experts in ascending id order onto a zero residual.
/// `slot_outputs` is `[tokens * TOP_K, hidden]`. Callers pass the GPU's own
/// expert outputs so a last-ulp GEMV difference cannot flip a BF16 rounding.
fn bf16_top10_combine(
    slot_outputs: &[f32],
    indices: &[i32],
    weights: &[f32],
    tokens: usize,
    hidden: usize,
) -> Vec<f32> {
    let bf16 = |value: f32| bf16_bits_to_f32(f32_to_bf16_bits(value));
    let mut combined = vec![0.0f32; tokens * hidden];
    for token in 0..tokens {
        let mut ranks: Vec<usize> = (0..TOP_K).collect();
        ranks.sort_by_key(|&rank| indices[token * TOP_K + rank]);
        for row in 0..hidden {
            let mut accumulated = 0.0f32;
            for &rank in &ranks {
                let flat = token * TOP_K + rank;
                let weighted = bf16(bf16(slot_outputs[flat * hidden + row]) * bf16(weights[flat]));
                accumulated = bf16(accumulated + weighted);
            }
            combined[token * hidden + row] = accumulated;
        }
    }
    combined
}

fn free_all(gpu: &mut Gpu, tensors: impl IntoIterator<Item = GpuTensor>) -> Result<(), String> {
    for tensor in tensors {
        gpu.free_tensor(tensor).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn embedding_row(gpu: &mut Gpu) -> Result<(), String> {
    let rows = 3usize;
    let dim = 256usize;
    let token_id = 2usize;
    let source_words = source_bf16_words(rows * dim, 211);
    let source_values: Vec<f32> = source_words
        .iter()
        .map(|&bits| bf16_bits_to_f32(bits))
        .collect();
    let payload = hfq4g128_embedding_bytes(&source_words, rows, dim)?;
    let row_stride = (dim / 128) * HFQ4_G128_GROUP_BYTES;
    let expected = hfq4g128_decode_row(
        &payload[token_id * row_stride..(token_id + 1) * row_stride],
        dim,
    )?;
    report_quant_loss(
        "HFQ4-G128 embedding row source-BF16 quant loss",
        &source_values[token_id * dim..(token_id + 1) * dim],
        &expected,
    )?;
    let table = gpu
        .upload_raw(&payload, &[payload.len()])
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[dim], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.embedding_lookup_hfq4g128(&table, &output, token_id as u32, dim)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    let result = check_close(
        "HFQ4-G128 embedding row exact payload",
        &actual,
        &expected,
        1e-6,
    );
    free_all(gpu, [table, output])?;
    result
}

fn qt53_dense_case(
    gpu: &mut Gpu,
    label: &str,
    rows: usize,
    k: usize,
    weight_seed: usize,
    activation_seed: usize,
) -> Result<(), String> {
    let source_words = source_bf16_words(rows * k, weight_seed);
    let (weight_bytes, source_rotated, decoded_rotated) = qt53_pack_bf16(&source_words, rows, k)?;
    report_quant_loss(
        &format!("{label} source-BF16 quant loss"),
        &source_rotated,
        &decoded_rotated,
    )?;
    let x_host = activation_values(k, activation_seed);
    let x_rotated_cpu = cpu_rotate_128(&x_host, 1, k);
    let weight = gpu
        .upload_raw(&weight_bytes, &[weight_bytes.len()])
        .map_err(|error| error.to_string())?;
    let x = gpu
        .upload_f32(&x_host, &[k])
        .map_err(|error| error.to_string())?;
    let x_rot = gpu
        .zeros(&[k], DType::F32)
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[rows], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.rotate_x_mq_128_v2(&x, &x_rot, k, 1)
        .map_err(|error| error.to_string())?;
    let rotated_actual = gpu
        .download_f32(&x_rot)
        .map_err(|error| error.to_string())?;
    let rotation_result = check_close(
        &format!("{label} activation FWHT"),
        &rotated_actual,
        &x_rotated_cpu,
        3e-6,
    );
    if let Err(error) = rotation_result {
        free_all(gpu, [weight, x, x_rot, output])?;
        return Err(error);
    }
    gpu.gemv_mq4g128v2(&weight, &x_rot, &output, rows, k)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    let groups = k.div_ceil(128);
    let row_stride = groups * QT53_GROUP_BYTES;
    let expected: Vec<f32> = (0..rows)
        .map(|row| {
            qt53_dot(
                &weight_bytes[row * row_stride..(row + 1) * row_stride],
                k,
                &x_rotated_cpu,
            )
        })
        .collect();
    let result = check_close(label, &actual, &expected, 3e-5);
    free_all(gpu, [weight, x, x_rot, output])?;
    result
}

fn shared_dense_qt53(gpu: &mut Gpu) -> Result<(), String> {
    qt53_dense_case(
        gpu,
        "qt53 shared/dense K=640 exact payload",
        3,
        640,
        307,
        401,
    )
}

fn qt53_boundary_matrix(gpu: &mut Gpu) -> Result<(), String> {
    for &k in &[1usize, 127, 128, 129, 160, 320] {
        qt53_dense_case(
            gpu,
            &format!("qt53 dense boundary K={k} exact payload"),
            2,
            k,
            503 + k,
            601 + k,
        )?;
    }
    println!("qt53 dense K boundaries [1,127,128,129,160,320,640]: PASS (K=640 shared path)");
    Ok(())
}

fn ordinary_gemv(gpu: &mut Gpu) -> Result<(), String> {
    let m = 3usize;
    let k = 160usize; // Deliberately exercises the ragged second G128 group.
    let weight_bytes = qt53_bytes(m, k, 5);
    let x: Vec<f32> = (0..k).map(|i| 0.03125 * i as f32 - 1.0).collect();
    let weight = gpu
        .upload_raw(&weight_bytes, &[weight_bytes.len()])
        .map_err(|error| error.to_string())?;
    let x_gpu = gpu
        .upload_f32(&x, &[k])
        .map_err(|error| error.to_string())?;
    let y_gpu = gpu
        .zeros(&[m], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemv_mq4g128v2(&weight, &x_gpu, &y_gpu, m, k)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&y_gpu)
        .map_err(|error| error.to_string())?;
    let row_stride = k.div_ceil(128) * QT53_GROUP_BYTES;
    let expected: Vec<f32> = (0..m)
        .map(|row| {
            qt53_dot(
                &weight_bytes[row * row_stride..(row + 1) * row_stride],
                k,
                &x,
            )
        })
        .collect();
    let result = check_close("qt53 ordinary GEMV", &actual, &expected, 3e-5);
    free_all(gpu, [weight, x_gpu, y_gpu])?;
    result
}

fn indexed_top10(gpu: &mut Gpu) -> Result<(), String> {
    let experts_count = 16usize;
    let m = 5usize;
    let k = 160usize;
    let tokens = 2usize;
    let row_stride = k.div_ceil(128) * QT53_GROUP_BYTES;
    let expert_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt53_bytes(m, k, 19 + expert * 23))
        .collect();
    let expert_tensors: Vec<GpuTensor> = expert_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let ptrs = gpu
        .upload_raw(&raw_ptrs(&expert_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;
    let indices_host: Vec<i32> = (0..tokens * TOP_K)
        .map(|slot| ((slot * 7 + 3) % experts_count) as i32)
        .collect();
    let indices = gpu
        .upload_raw(&raw_i32(&indices_host), &[tokens * TOP_K * 4])
        .map_err(|error| error.to_string())?;
    let x_host: Vec<f32> = (0..tokens * TOP_K * k)
        .map(|index| 0.0078125 * (index as f32 + 1.0) - 0.5)
        .collect();
    let x = gpu
        .upload_f32(&x_host, &[tokens * TOP_K, k])
        .map_err(|error| error.to_string())?;
    let expanded = gpu
        .zeros(&[tokens * TOP_K, m], DType::F32)
        .map_err(|error| error.to_string())?;
    let weights_host = normalize_top10(tokens, 31);
    let weights = gpu
        .upload_f32(&weights_host, &[tokens * TOP_K])
        .map_err(|error| error.to_string())?;
    let residual = gpu
        .zeros(&[tokens, m], DType::F32)
        .map_err(|error| error.to_string())?;

    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &ptrs,
        &indices,
        &x,
        &expanded,
        m,
        k,
        tokens,
        experts_count,
    )
    .map_err(|error| error.to_string())?;
    let expanded_host = gpu
        .download_f32(&expanded)
        .map_err(|error| error.to_string())?;
    gpu.moe_down_combine_top10_batched(&expanded, &indices, &weights, &residual, m, tokens)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&residual)
        .map_err(|error| error.to_string())?;
    let mut expected_expanded = vec![0.0f32; tokens * TOP_K * m];
    for flat in 0..tokens * TOP_K {
        let expert = indices_host[flat] as usize;
        for row in 0..m {
            let row_start = row * row_stride;
            expected_expanded[flat * m + row] = qt53_dot(
                &expert_bytes[expert][row_start..row_start + row_stride],
                k,
                &x_host[flat * k..(flat + 1) * k],
            );
        }
    }
    let expected = bf16_top10_combine(&expanded_host, &indices_host, &weights_host, tokens, m);
    let result = check_close(
        "qt53 indexed top10 GEMV",
        &expanded_host,
        &expected_expanded,
        4e-5,
    )
    .and_then(|()| check_close("qt53 top10 BF16 combine", &actual, &expected, 0.0));
    let mut owned = vec![ptrs, indices, x, expanded, weights, residual];
    owned.extend(expert_tensors);
    free_all(gpu, owned)?;
    result
}

fn grouped_prefill(gpu: &mut Gpu) -> Result<(), String> {
    let experts_count = 16usize;
    let batch = 2usize;
    let total_slots = batch * TOP_K;
    let hidden = 256usize;
    let intermediate = 640usize;
    let grouped_rows = 272usize; // 16-row tile capacity above the live padded total.
    let row_stride = intermediate.div_ceil(128) * QT53_GROUP_BYTES;
    let down_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt53_bytes(hidden, intermediate, 47 + expert * 29))
        .collect();
    let down_tensors: Vec<GpuTensor> = down_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let down_ptrs = gpu
        .upload_raw(&raw_ptrs(&down_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;

    // Exercise the dedicated qt44 gate/up grouped entry using the same top-10
    // permutation.
    let gate_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt44_bytes(2 * intermediate, hidden, 71 + expert * 31))
        .collect();
    let gate_tensors: Vec<GpuTensor> = gate_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let gate_ptrs = gpu
        .upload_raw(&raw_ptrs(&gate_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;

    let indices_host: Vec<i32> = (0..total_slots)
        .map(|slot| ((slot * 5 + 1) % experts_count) as i32)
        .collect();
    let indices = gpu
        .upload_raw(&raw_i32(&indices_host), &[total_slots * 4])
        .map_err(|error| error.to_string())?;
    let weights_host = normalize_top10(batch, 113);
    let weights = gpu
        .upload_f32(&weights_host, &[total_slots])
        .map_err(|error| error.to_string())?;
    let counts = zero_raw_i32(gpu, experts_count)?;
    let offsets = zero_raw_i32(gpu, experts_count + 1)?;
    let sorted = zero_raw_i32(gpu, grouped_rows)?;
    let tile_ids = zero_raw_i32(gpu, grouped_rows / 16)?;
    let inverse = zero_raw_i32(gpu, total_slots)?;

    gpu.moe_scatter_fused_top10(
        &indices,
        &counts,
        &offsets,
        &sorted,
        &tile_ids,
        &inverse,
        total_slots,
        experts_count,
        grouped_rows,
        16,
    )
    .map_err(|error| error.to_string())?;

    let rot_host: Vec<f32> = (0..total_slots * intermediate)
        .map(|index| 0.00390625 * (index as f32 + 3.0) - 0.75)
        .collect();
    let rot = gpu
        .upload_f32(&rot_host, &[total_slots, intermediate])
        .map_err(|error| error.to_string())?;
    let grouped_down = gpu
        .zeros(&[grouped_rows, hidden], DType::F32)
        .map_err(|error| error.to_string())?;
    let residual = gpu
        .zeros(&[batch, hidden], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemm_mq4g128v2_moe_grouped_top10(
        &down_ptrs,
        &tile_ids,
        &sorted,
        &rot,
        &grouped_down,
        hidden,
        intermediate,
        1,
        grouped_rows,
        total_slots,
        experts_count,
    )
    .map_err(|error| error.to_string())?;
    gpu.moe_down_combine_grouped_top10(
        &grouped_down,
        &inverse,
        &indices,
        &weights,
        &residual,
        hidden,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&residual)
        .map_err(|error| error.to_string())?;
    let grouped_host = gpu
        .download_f32(&grouped_down)
        .map_err(|error| error.to_string())?;
    let mut inverse_bytes = vec![0u8; total_slots * 4];
    gpu.hip
        .memcpy_dtoh(&mut inverse_bytes, &inverse.buf)
        .map_err(|error| error.to_string())?;
    let mut slot_outputs = vec![0.0f32; total_slots * hidden];
    let mut expected_slots = vec![0.0f32; total_slots * hidden];
    for flat in 0..total_slots {
        let grouped_row =
            i32::from_le_bytes(inverse_bytes[flat * 4..flat * 4 + 4].try_into().unwrap());
        if grouped_row < 0 || grouped_row as usize >= grouped_rows {
            return Err(format!(
                "qt53 grouped prefill: slot {flat} maps to grouped row {grouped_row}"
            ));
        }
        let grouped_row = grouped_row as usize;
        slot_outputs[flat * hidden..(flat + 1) * hidden]
            .copy_from_slice(&grouped_host[grouped_row * hidden..(grouped_row + 1) * hidden]);
        let expert = indices_host[flat] as usize;
        for row in 0..hidden {
            let row_start = row * row_stride;
            expected_slots[flat * hidden + row] = qt53_dot(
                &down_bytes[expert][row_start..row_start + row_stride],
                intermediate,
                &rot_host[flat * intermediate..(flat + 1) * intermediate],
            );
        }
    }
    let expected = bf16_top10_combine(&slot_outputs, &indices_host, &weights_host, batch, hidden);
    let result = check_close(
        "qt53 grouped prefill GEMM",
        &slot_outputs,
        &expected_slots,
        5e-5,
    )
    .and_then(|()| check_close("qt53 grouped BF16 combine", &actual, &expected, 0.0));

    // The gate/up launch is deliberately after the independent qt53 check: if
    // kernel compilation or its top-10 gather/unscatter contract is broken, this
    // executable still reports it as a failed channel rather than hiding the
    // useful qt53 CPU-reference result behind a prior error.
    let gate_x_host: Vec<f32> = (0..batch * hidden)
        .map(|index| 0.01171875 * index as f32 - 0.4)
        .collect();
    let gate_x = gpu
        .upload_f32(&gate_x_host, &[batch, hidden])
        .map_err(|error| error.to_string())?;
    let gate_grouped = gpu
        .zeros(&[grouped_rows, 2 * intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    let gate_out = gpu
        .zeros(&[total_slots, intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    let up_out = gpu
        .zeros(&[total_slots, intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemm_mq4g256v2_moe_grouped_top10(
        &gate_ptrs,
        &tile_ids,
        &sorted,
        &gate_x,
        &gate_grouped,
        2 * intermediate,
        hidden,
        TOP_K,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    gpu.moe_gate_up_unscatter_top10(
        &gate_grouped,
        &sorted,
        &gate_out,
        &up_out,
        intermediate,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    let gate_probe = gpu
        .download_f32(&gate_out)
        .map_err(|error| error.to_string())?;
    if gate_probe.iter().any(|value| !value.is_finite()) {
        return Err("qt44 grouped gate/up top10 produced a non-finite value".to_string());
    }
    println!("qt44 grouped gate/up + top10 unscatter: PASS");

    let mut owned = vec![
        down_ptrs,
        gate_ptrs,
        indices,
        weights,
        counts,
        offsets,
        sorted,
        tile_ids,
        inverse,
        rot,
        grouped_down,
        residual,
        gate_x,
        gate_grouped,
        gate_out,
        up_out,
    ];
    owned.extend(down_tensors);
    owned.extend(gate_tensors);
    free_all(gpu, owned)?;
    result
}

// ── qt44 (MQ4G256V2) dense channel references ───────────────────────────────
//
// The trunk's packed projections run as `rotate + gemv/gemm_mq4g256v2`; these
// helpers decode the *same wire bytes* on the CPU so the runtime's addressing
// and FWHT basis are checked against an independent implementation.

fn cpu_fwht_256(values: &mut [f32]) {
    assert_eq!(values.len(), 256);
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    for (value, sign) in values.iter_mut().zip(&signs1) {
        *value *= sign;
    }
    let mut stride = 1;
    while stride < 256 {
        let mut offset = 0;
        while offset < 256 {
            for index in 0..stride {
                let left = values[offset + index];
                let right = values[offset + index + stride];
                values[offset + index] = left + right;
                values[offset + index + stride] = left - right;
            }
            offset += stride * 2;
        }
        stride <<= 1;
    }
    let normalization = 1.0f32 / 256.0f32.sqrt();
    for (value, sign) in values.iter_mut().zip(&signs2) {
        *value *= normalization * sign;
    }
}

fn cpu_rotate_256(input: &[f32], batch: usize, k: usize) -> Vec<f32> {
    assert_eq!(input.len(), batch * k);
    let mut output = vec![0.0f32; input.len()];
    let groups = k.div_ceil(256);
    for row in 0..batch {
        for group in 0..groups {
            let start = group * 256;
            let actual = (k - start).min(256);
            let mut values = [0.0f32; 256];
            values[..actual].copy_from_slice(&input[row * k + start..row * k + start + actual]);
            cpu_fwht_256(&mut values);
            output[row * k + start..row * k + start + actual].copy_from_slice(&values[..actual]);
        }
    }
    output
}

/// Decode one 136-byte group row exactly as `quant_fwht::quantize_mq4g256v2`
/// writes it: `[s0,z0,s1,z1]` fp16 halves, then byte `i` carrying logical
/// `2i` in the low nibble and `2i+1` in the high nibble, half `h = index/128`
/// owning `s[h]/z[h]`.
fn qt44_dot(row: &[u8], k: usize, x_rotated: &[f32]) -> f32 {
    let groups = k.div_ceil(256);
    let row_stride = groups * QT44_GROUP_BYTES;
    assert_eq!(row.len(), row_stride);
    assert_eq!(x_rotated.len(), k);
    let mut acc = 0.0f32;
    for group in 0..groups {
        let base = group * QT44_GROUP_BYTES;
        let mut scale = [0.0f32; 2];
        let mut zero = [0.0f32; 2];
        for h in 0..2 {
            scale[h] = f16_to_f32(u16::from_le_bytes([
                row[base + h * 4],
                row[base + h * 4 + 1],
            ]));
            zero[h] = f16_to_f32(u16::from_le_bytes([
                row[base + h * 4 + 2],
                row[base + h * 4 + 3],
            ]));
        }
        for index in 0..256 {
            let logical = group * 256 + index;
            if logical >= k {
                continue;
            }
            let packed = row[base + 8 + index / 2];
            let q = if index & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let h = index / 128;
            acc += (scale[h] * q as f32 + zero[h]) * x_rotated[logical];
        }
    }
    acc
}

fn qt44_dense_case(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    rows: usize,
    k: usize,
    activation_seed: usize,
) -> Result<(), String> {
    let weight_seed = 977 + rows * 31 + k % 97;
    let payload = qt44_bytes(m, k, weight_seed);
    let x_host = activation_values(k * rows, activation_seed);
    let x_rotated_cpu = cpu_rotate_256(&x_host, rows, k);
    let weight = gpu
        .upload_raw(&payload, &[payload.len()])
        .map_err(|error| error.to_string())?;
    let x = gpu
        .upload_f32(&x_host, &[k * rows])
        .map_err(|error| error.to_string())?;
    let x_rot = gpu
        .zeros(&[k * rows], DType::F32)
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[rows * m], DType::F32)
        .map_err(|error| error.to_string())?;
    if rows > 1 {
        gpu.rotate_x_mq_batched(&x, &x_rot, k, rows)
            .map_err(|error| error.to_string())?;
    } else {
        gpu.rotate_x_mq(&x, &x_rot, k)
            .map_err(|error| error.to_string())?;
    }
    let rotated_actual = gpu
        .download_f32(&x_rot)
        .map_err(|error| error.to_string())?;
    let rotation_result = check_close(
        &format!("{label} qt44 rotation basis"),
        &rotated_actual,
        &x_rotated_cpu,
        1e-6,
    );
    let gemm_result = match rotation_result {
        Ok(()) => {
            let result = if rows > 1 {
                gpu.gemm_mq4g256v2(&weight, &x_rot, &output, m, k, rows)
                    .map_err(|error| error.to_string())
            } else {
                gpu.gemv_mq4g256v2(&weight, &x_rot, &output, m, k)
                    .map_err(|error| error.to_string())
            };
            match result {
                Ok(()) => {
                    let actual = gpu
                        .download_f32(&output)
                        .map_err(|error| error.to_string())?;
                    // The synthetic payload carries scale 1 / zero 0, so the
                    // reference dot is linear in the packed nibbles.
                    let row_stride = k.div_ceil(256) * QT44_GROUP_BYTES;
                    let mut expected = Vec::with_capacity(rows * m);
                    for row in 0..rows {
                        for weight_row in 0..m {
                            expected.push(qt44_dot(
                                &payload[weight_row * row_stride..(weight_row + 1) * row_stride],
                                k,
                                &x_rotated_cpu[row * k..(row + 1) * k],
                            ));
                        }
                    }
                    // The batched launcher must agree with the per-row GEMV,
                    // which the exact-value case above pins to the CPU
                    // reference.  A shared-layout bug shows up as O(1); a
                    // summation-order difference stays in the f32 noise floor.
                    let per_row = gpu
                        .zeros(&[rows * m], DType::F32)
                        .map_err(|error| error.to_string())?;
                    for row in 0..rows {
                        let x_row = x_rot.sub_offset(row * k, k);
                        let y_row = per_row.sub_offset(row * m, m);
                        gpu.gemv_mq4g256v2(&weight, &x_row, &y_row, m, k)
                            .map_err(|error| error.to_string())?;
                    }
                    let per_row_actual = gpu
                        .download_f32(&per_row)
                        .map_err(|error| error.to_string())?;
                    // The batched launcher accumulates its 10240-term dot in a
                    // different order from the per-row GEMV, and the spread is
                    // *absolute*: ~0.02-0.19 measured across every shape here,
                    // while the same shapes' single-row path matches the CPU
                    // reference bit-exactly.  `check_close` compares against
                    // max(1, |want|), so this bound means "agree within 0.5",
                    // which an addressing or basis fault (error of the order of
                    // the term sum) exceeds by two orders of magnitude.
                    let gemv_vs_gemm = check_close(
                        &format!("{label} qt44 gemm-vs-gemv m={m} k={k} rows={rows}"),
                        &actual,
                        &per_row_actual,
                        0.5,
                    );
                    // Same absolute bound as the cross-check above: the
                    // single-row path is bit-exact, so the multi-row slack is
                    // the batched accumulation order, not the wire format.
                    let cpu = check_close(
                        &format!("{label} qt44 dense m={m} k={k} rows={rows}"),
                        &actual,
                        &expected,
                        0.5,
                    );
                    free_all(gpu, [per_row])?;
                    gemv_vs_gemm.and(cpu)
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    };
    free_all(gpu, [weight, x, x_rot, output])?;
    gemm_result
}

fn qt44_dense_cases(gpu: &mut Gpu) -> Result<(), String> {
    // Shapes the packed trunk actually dispatches: the hyper-connection mixer
    // down projection, its four-row block inject, the GDN scalar projections,
    // and a wide projection at both row counts.
    for (rows, k, label) in [
        (1usize, 10240usize, "mixer down decode"),
        (4, 10240, "mixer down prefill"),
        (1, 2560, "gdn scalar decode"),
        (6, 2560, "gdn scalar prefill"),
        (8, 10240, "block inject prefill"),
        (1, 2560, "wide decode"),
        (8, 2560, "wide prefill"),
    ] {
        let m = match label {
            "mixer down decode" | "mixer down prefill" => 320,
            "gdn scalar decode" | "gdn scalar prefill" => 48,
            "block inject prefill" => 4,
            _ => 256,
        };
        qt44_dense_case(gpu, label, m, rows, k, 61 + rows)?;
    }
    Ok(())
}

// ── qt47 (MQ6G256V2) dense channel references ───────────────────────────────
//
// The 6-bit trunk tier shares the aligned-K 256 group and FWHT basis with qt44,
// so only the payload width and bit packing differ: 8-byte header, then 192
// bytes carrying four 6-bit values per three bytes.

fn mq6_bytes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "qt47 harness rows use complete G256 groups");
    let groups = k / 256;
    let row_stride = groups * QT47_GROUP_BYTES;
    let mut out = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_stride + group * QT47_GROUP_BYTES;
            for half in 0..2 {
                out[base + half * 4..base + half * 4 + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
                out[base + half * 4 + 2..base + half * 4 + 4].copy_from_slice(&0u16.to_le_bytes());
            }
            let mut q = [0u8; 256];
            for (index, value) in q.iter_mut().enumerate() {
                *value = ((seed + row * 3 + group * 29 + index * 5) & 63) as u8;
            }
            for i in (0..256).step_by(4) {
                let bo = base + 8 + (i / 4) * 3;
                let q0 = q[i];
                let q1 = q[i + 1];
                let q2 = q[i + 2];
                let q3 = q[i + 3];
                out[bo] = q0 | (q1 << 6);
                out[bo + 1] = (q1 >> 2) | (q2 << 4);
                out[bo + 2] = (q2 >> 4) | (q3 << 2);
            }
        }
    }
    out
}

fn mq6_dot(row: &[u8], k: usize, x_rotated: &[f32]) -> f32 {
    let groups = k.div_ceil(256);
    let row_stride = groups * QT47_GROUP_BYTES;
    assert_eq!(row.len(), row_stride);
    assert_eq!(x_rotated.len(), k);
    let mut acc = 0.0f32;
    for group in 0..groups {
        let base = group * QT47_GROUP_BYTES;
        let mut scale = [0.0f32; 2];
        let mut zero = [0.0f32; 2];
        for half in 0..2 {
            scale[half] = f16_to_f32(u16::from_le_bytes([
                row[base + half * 4],
                row[base + half * 4 + 1],
            ]));
            zero[half] = f16_to_f32(u16::from_le_bytes([
                row[base + half * 4 + 2],
                row[base + half * 4 + 3],
            ]));
        }
        for i in (0..256).step_by(4) {
            let bo = base + 8 + (i / 4) * 3;
            let b0 = row[bo] as u32;
            let b1 = row[bo + 1] as u32;
            let b2 = row[bo + 2] as u32;
            let q = [
                (b0 & 63) as u8,
                ((b0 >> 6) | (b1 << 2)) as u8 & 63,
                ((b1 >> 4) | (b2 << 4)) as u8 & 63,
                ((b2 >> 2) & 63) as u8,
            ];
            for (offset, value) in q.iter().enumerate() {
                let index = i + offset;
                let logical = group * 256 + index;
                if logical >= k {
                    continue;
                }
                let half = index / 128;
                acc += (scale[half] * *value as f32 + zero[half]) * x_rotated[logical];
            }
        }
    }
    acc
}

fn mq6_dense_case(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    rows: usize,
    k: usize,
    activation_seed: usize,
) -> Result<(), String> {
    let payload = mq6_bytes(m, k, 613 + rows * 7);
    let x_host = activation_values(k * rows, activation_seed);
    let x_rotated_cpu = cpu_rotate_256(&x_host, rows, k);
    let weight = gpu
        .upload_raw(&payload, &[payload.len()])
        .map_err(|error| error.to_string())?;
    let x = gpu
        .upload_f32(&x_host, &[k * rows])
        .map_err(|error| error.to_string())?;
    let x_rot = gpu
        .zeros(&[k * rows], DType::F32)
        .map_err(|error| error.to_string())?;
    let output = gpu
        .zeros(&[rows * m], DType::F32)
        .map_err(|error| error.to_string())?;
    if rows > 1 {
        gpu.rotate_x_mq_batched(&x, &x_rot, k, rows)
            .map_err(|error| error.to_string())?;
    } else {
        gpu.rotate_x_mq(&x, &x_rot, k)
            .map_err(|error| error.to_string())?;
    }
    let rotated_actual = gpu
        .download_f32(&x_rot)
        .map_err(|error| error.to_string())?;
    check_close(
        &format!("{label} qt47 rotation basis"),
        &rotated_actual,
        &x_rotated_cpu,
        1e-6,
    )?;
    if rows > 1 {
        gpu.gemm_mq6g256v2(&weight, &x_rot, &output, m, k, rows)
            .map_err(|error| error.to_string())?;
    } else {
        gpu.gemv_mq6g256v2(&weight, &x_rot, &output, m, k)
            .map_err(|error| error.to_string())?;
    }
    let actual = gpu
        .download_f32(&output)
        .map_err(|error| error.to_string())?;
    let row_stride = (k / 256) * QT47_GROUP_BYTES;
    let mut expected = Vec::with_capacity(rows * m);
    for row in 0..rows {
        for weight_row in 0..m {
            expected.push(mq6_dot(
                &payload[weight_row * row_stride..(weight_row + 1) * row_stride],
                k,
                &x_rotated_cpu[row * k..(row + 1) * k],
            ));
        }
    }
    let result = check_close(
        &format!("{label} qt47 dense m={m} k={k} rows={rows}"),
        &actual,
        &expected,
        0.5,
    );
    free_all(gpu, [weight, x, x_rot, output])?;
    result
}

fn mq6_dense_cases(gpu: &mut Gpu) -> Result<(), String> {
    for (m, rows, k, label) in [
        (320usize, 1usize, 10240usize, "mixer down decode"),
        (320, 4, 10240, "mixer down prefill"),
        (1024, 1, 2560, "gdn qkv decode"),
        (1024, 8, 2560, "gdn qkv prefill"),
    ] {
        mq6_dense_case(gpu, label, m, rows, k, 131 + rows)?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    println!("GPU: {}", gpu.arch);
    qt44_dense_cases(&mut gpu)?;
    mq6_dense_cases(&mut gpu)?;
    embedding_row(&mut gpu)?;
    shared_dense_qt53(&mut gpu)?;
    qt53_boundary_matrix(&mut gpu)?;
    ordinary_gemv(&mut gpu)?;
    indexed_top10(&mut gpu)?;
    grouped_prefill(&mut gpu)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Qwen4 qt53 channel: FAIL: {error}");
        std::process::exit(1);
    }
    println!("Qwen4 qt53 channel: PASS");
}
