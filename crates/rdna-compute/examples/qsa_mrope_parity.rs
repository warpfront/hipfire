// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Small GPU/reference oracle for image QSA norms and pooled-key mRoPE.
//! No model weights. Includes rectangular image positions, post-image decode,
//! gate-strided Q, contiguous K/index Q, both index-cache storage dtypes, and
//! incremental pooling across an image boundary.
//! Run: cargo run -p rdna-compute --example qsa_mrope_parity --release

use half::bf16;
use rdna_compute::qsa_mrope::{norm_mrope, pool_mrope, QsaMrope};
use rdna_compute::rope::{Rope, RopeScaling};
use rdna_compute::tensor_ops::{
    IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope, QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn b(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn coordinate(p: usize, f: usize, positions: &[[i32; 3]], delta: i32) -> f32 {
    if p >= positions.len() {
        return (p as i32 + delta) as f32;
    }
    let axis = if f % 3 == 1 && f < 33 {
        1
    } else if f % 3 == 2 && f < 30 {
        2
    } else {
        0
    };
    positions[p][axis] as f32
}
fn rotate(x: &[f32], p: usize, positions: &[[i32; 3]], delta: i32) -> Vec<f32> {
    let mut y = x.to_vec();
    for d in 0..32 {
        let angle =
            coordinate(p, d, positions, delta) * 10_000_000.0f32.powf(-((2 * d) as f32) / 64.0);
        let (sin, cos) = angle.sin_cos();
        y[d] = x[d] * cos - x[d + 32] * sin;
        y[d + 32] = x[d] * sin + x[d + 32] * cos;
    }
    y
}
fn reference_norm(
    x: &[f32],
    norm: &[f32],
    p: usize,
    positions: &[[i32; 3]],
    delta: i32,
) -> Vec<f32> {
    let sum = x.iter().fold(0.0f32, |sum, &x| sum + x * x);
    let inv = (sum / x.len() as f32 + 1e-6).sqrt().recip();
    let y: Vec<_> = x
        .iter()
        .zip(norm)
        .map(|(&x, &n)| x * inv * (1.0 + n))
        .collect();
    rotate(&y, p, positions, delta)
}
fn reference_pool(
    raw: &[f32],
    norm: &[f32],
    block: usize,
    dim: usize,
    positions: &[[i32; 3]],
    delta: i32,
) -> Vec<f32> {
    let mean: Vec<f32> = (0..dim)
        .map(|d| b((0..4).fold(0.0, |sum, r| sum + raw[(block * 4 + r) * dim + d] / 4.0)))
        .collect();
    let mut sum = vec![0.0f32; 256];
    for d in 0..dim {
        sum[d] = b(mean[d] * mean[d]);
    }
    for stride in [128, 64, 32, 16, 8, 4, 2, 1] {
        for d in 0..stride {
            sum[d] += sum[d + stride];
        }
    }
    let inv = (sum[0] / dim as f32 + 1e-6).sqrt().recip();
    let y: Vec<_> = mean
        .iter()
        .zip(norm)
        .map(|(&x, &n)| b(x * inv * (1.0 + n)))
        .collect();
    rotate(&y, block * 4, positions, delta)
        .into_iter()
        .map(b)
        .collect()
}
fn upload_bf16(gpu: &mut Gpu, values: &[f32]) -> Result<GpuTensor> {
    let mut t = gpu.upload_f16_bits(
        &values
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect::<Vec<_>>(),
        &[values.len()],
    )?;
    t.dtype = DType::BF16;
    Ok(t)
}
fn read(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<f32>> {
    if t.dtype == DType::F32 {
        return Ok(gpu.download_f32(t)?);
    }
    let mut bytes = vec![0; t.numel() * 2];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|b| bf16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32())
        .collect())
}
fn error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}
fn view(t: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: t.buf.byte_view(offset * 4, len * 4),
        shape: vec![len],
        dtype: DType::F32,
    }
}
fn main() -> Result<()> {
    let mut gpu = Gpu::init()?;
    let positions = vec![
        [0; 3],
        [1; 3],
        [2, 2, 2],
        [2, 2, 3],
        [2, 2, 4],
        [2, 3, 2],
        [2, 3, 3],
        [2, 3, 4],
        [5; 3],
        [6; 3],
        [7; 3],
    ];
    let delta = -3;
    let bytes: Vec<_> = positions
        .iter()
        .flatten()
        .flat_map(|x: &i32| x.to_ne_bytes())
        .collect();
    let pos_gpu = gpu.upload_raw(&bytes, &[bytes.len()])?;
    let m = QsaMrope {
        positions: &pos_gpu,
        prompt_rows: positions.len(),
        delta,
        section: [11, 11, 10],
    };
    let yarn = Rope {
        theta: 10_000_000.0,
        scaling: RopeScaling::None,
    }
    .yarn(64);
    let mut reports = Vec::new();
    for (dim, heads, gate) in [(256, 2, true), (256, 2, false), (128, 4, false)] {
        let head_stride = dim * if gate { 2 } else { 1 };
        let stride = head_stride * heads;
        let rows = 16;
        let input: Vec<f32> = (0..rows * stride)
            .map(|i| ((i * 19 % 257) as f32 - 128.0) / 128.0)
            .collect();
        let norm: Vec<f32> = (0..dim).map(|d| b((d % 13) as f32 / 64.0)).collect();
        let norm_gpu = upload_bf16(&mut gpu, &norm)?;
        let all = gpu.upload_f32(&input, &[input.len()])?;
        let split = gpu.upload_f32(&input, &[input.len()])?;
        let make = |values, rows, start| IndexedAttentionNormRopeBatch {
            values,
            norm: &norm_gpu,
            rows,
            row_stride: stride,
            heads,
            head_dim: dim,
            head_stride,
            position_start: start,
            rotary_dim: 64,
            yarn,
        };
        norm_mrope(&mut gpu, &make(&all, rows, 0), m)?;
        let ranges = [(0, 5), (5, 6), (11, 1), (12, 4)];
        let chunks: Vec<_> = ranges
            .iter()
            .map(|&(start, n)| view(&split, start * stride, n * stride))
            .collect();
        for (&(start, n), chunk) in ranges.iter().zip(&chunks) {
            norm_mrope(&mut gpu, &make(chunk, n, start), m)?;
        }
        let actual = read(&gpu, &all)?;
        let divided = read(&gpu, &split)?;
        let mut reference = input.clone();
        for row in 0..rows {
            for h in 0..heads {
                let off = row * stride + h * head_stride;
                reference[off..off + dim].copy_from_slice(&reference_norm(
                    &input[off..off + dim],
                    &norm,
                    row,
                    &positions,
                    delta,
                ));
            }
        }
        let max_error = error(&actual, &reference);
        let split_error = error(&actual, &divided);
        if max_error > 3e-5 || split_error != 0.0 {
            return Err(format!(
                "norm parity dim={dim} gate={gate}: reference={max_error} chunk={split_error}"
            )
            .into());
        }
        reports.push(json!({"op":"norm_mrope","dim":dim,"heads":heads,"gate_stride":gate,"max_abs_error":max_error,"chunk_error":split_error}));
        gpu.free_tensor(all)?;
        gpu.free_tensor(split)?;
        gpu.free_tensor(norm_gpu)?;
    }
    let dim = 128;
    let raw: Vec<f32> = (0..16 * dim)
        .map(|i| b(((i * 31 % 241) as f32 - 120.0) / 128.0))
        .collect();
    let norm: Vec<f32> = (0..dim).map(|d| b((d % 7) as f32 / 128.0)).collect();
    let norm_gpu = upload_bf16(&mut gpu, &norm)?;
    let reference: Vec<_> = (0..4)
        .flat_map(|block| reference_pool(&raw, &norm, block, dim, &positions, delta))
        .collect();
    for dtype in [DType::F32, DType::BF16] {
        let raw_gpu = if dtype == DType::F32 {
            gpu.upload_f32(&raw, &[raw.len()])?
        } else {
            upload_bf16(&mut gpu, &raw)?
        };
        let all = gpu.alloc_tensor(&[4 * dim], dtype)?;
        let split = gpu.alloc_tensor(&[4 * dim], dtype)?;
        let make = |pooled, start, rows| IndexedAttentionPoolRope {
            raw_keys: &raw_gpu,
            pooled,
            norm: Some(&norm_gpu),
            block_count: (start + rows) / 4,
            compress: 4,
            index_dim: dim,
            position: Some(QsaPositionBinding {
                position_start: start,
                rows,
            }),
            grid_bound: 4,
            yarn,
        };
        pool_mrope(&mut gpu, &make(&all, 0, 16), m)?;
        pool_mrope(&mut gpu, &make(&split, 0, 7), m)?;
        pool_mrope(&mut gpu, &make(&split, 7, 9), m)?;
        let actual = read(&gpu, &all)?;
        let divided = read(&gpu, &split)?;
        let max_error = error(&actual, &reference);
        let split_error = error(&actual, &divided);
        if max_error > 0.016 || split_error != 0.0 {
            return Err(format!(
                "pool parity {dtype:?}: reference={max_error} incremental={split_error}"
            )
            .into());
        }
        reports.push(json!({"op":"pool_mrope","dtype":format!("{dtype:?}"),"max_abs_error":max_error,"incremental_error":split_error}));
        gpu.free_tensor(raw_gpu)?;
        gpu.free_tensor(all)?;
        gpu.free_tensor(split)?;
    }
    gpu.free_tensor(norm_gpu)?;
    gpu.free_tensor(pos_gpu)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"status":"pass","arch":gpu.arch,"cases":reports}))?
    );
    Ok(())
}
