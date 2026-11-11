// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Eager QSA rotations for image-conditioned Flash-Next requests.
//! The prompt retains three coordinates per token; decode and pooled keys
//! resolve their coordinates from the same owner, independently of KV offsets.

use crate::tensor_ops::{IndexedAttentionNormRopeBatch, IndexedAttentionPoolRope};
use crate::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult, KernargBlob};

#[derive(Clone, Copy)]
pub struct QsaMrope<'a> {
    /// Interleaved signed i32 [token, (t,h,w)], stored as Raw bytes.
    pub positions: &'a GpuTensor,
    pub prompt_rows: usize,
    pub delta: i32,
    pub section: [usize; 3],
}

fn invalid(message: &str) -> HipError {
    HipError::new(0, message)
}

fn int(value: usize) -> HipResult<i32> {
    i32::try_from(value).map_err(|_| invalid("QSA mRoPE dimension exceeds i32"))
}

impl QsaMrope<'_> {
    pub fn validate(&self, rotary_dim: usize, end: usize) -> HipResult<()> {
        if self.prompt_rows == 0
            || self.prompt_rows > i32::MAX as usize / 12
            || self.positions.dtype != DType::Raw
            || self.positions.numel() != self.prompt_rows * 12
            || !(2..=64).contains(&rotary_dim)
            || rotary_dim % 2 != 0
            || self.section.iter().any(|&s| s > rotary_dim / 2)
            || self.section.iter().sum::<usize>() != rotary_dim / 2
            || end > i32::MAX as usize
            || self.prompt_rows as i64 + (self.delta as i64) < 0
            || end as i64 + (self.delta as i64) > i32::MAX as i64
        {
            return Err(invalid("invalid QSA mRoPE coordinates or extent"));
        }
        Ok(())
    }

    fn push(&self, args: &mut KernargBlob) -> HipResult<()> {
        args.push_i32(int(self.prompt_rows)?);
        args.push_i32(self.delta);
        args.push_i32(int(self.section[1])?);
        args.push_i32(int(self.section[2])?);
        Ok(())
    }
}

pub fn ensure_eager(gpu: &Gpu) -> HipResult<()> {
    let stream_capture = match gpu.active_stream.as_ref() {
        Some(stream) => gpu.hip.stream_is_capturing(stream)?,
        None => false,
    };
    if stream_capture
        || gpu.graphs.capture_mode
        || gpu.graphs.verify.capturing.is_some()
        || gpu.graphs.replay.capturing.is_some()
        || gpu.replay.is_recording()
        || gpu.replay.retained_body_active()
        || gpu.replay.should_route_aql()
        || gpu.replay.should_route_pm4()
    {
        return Err(invalid("QSA image mRoPE requires eager HIP"));
    }
    Ok(())
}

pub fn norm_mrope(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRopeBatch<'_>,
    m: QsaMrope<'_>,
) -> HipResult<()> {
    ensure_eager(gpu)?;
    let end = p
        .position_start
        .checked_add(p.rows)
        .ok_or_else(|| invalid("QSA mRoPE row overflow"))?;
    m.validate(p.rotary_dim, end)?;
    if p.rows == 0
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim > p.head_dim
        || p.norm.dtype != DType::BF16
        || p.norm.numel() != p.head_dim
        || p.values.dtype != DType::F32
    {
        return Err(invalid("invalid QSA mRoPE norm geometry"));
    }
    let span = (p.heads - 1)
        .checked_mul(p.head_stride)
        .and_then(|s| s.checked_add(p.head_dim))
        .ok_or_else(|| invalid("QSA mRoPE head overflow"))?;
    let extent = (p.rows - 1)
        .checked_mul(p.row_stride)
        .and_then(|s| s.checked_add(span))
        .ok_or_else(|| invalid("QSA mRoPE extent overflow"))?;
    if p.row_stride < span || p.values.numel() < extent || extent > i32::MAX as usize {
        return Err(invalid("QSA mRoPE values extent mismatch"));
    }
    let kernel = "indexed_attention_norm_mrope_f32_batched";
    gpu.ensure_kernel_public("qsa_mrope", crate::kernels::QSA_MROPE_SRC, kernel)?;
    let mut args = KernargBlob::new();
    for tensor in [p.values, p.norm, m.positions] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        p.rows,
        p.row_stride,
        p.heads,
        p.head_dim,
        p.head_stride,
        p.position_start,
        p.rotary_dim,
    ] {
        args.push_i32(int(value)?);
    }
    m.push(&mut args)?;
    p.yarn.push(&mut args);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [p.heads as u32, 1, p.rows as u32],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

pub fn pool_mrope(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
    m: QsaMrope<'_>,
) -> HipResult<()> {
    ensure_eager(gpu)?;
    let position = p
        .position
        .ok_or_else(|| invalid("QSA mRoPE pooling requires position"))?;
    let end = position
        .position_start
        .checked_add(position.rows)
        .ok_or_else(|| invalid("QSA mRoPE pool position overflow"))?;
    m.validate(p.index_dim.min(64), end)?;
    if p.compress == 0
        || p.index_dim == 0
        || p.index_dim > 256
        || p.index_dim % 2 != 0
        || p.block_count != end / p.compress
        || p.grid_bound < p.block_count
    {
        return Err(invalid("invalid QSA mRoPE pooling geometry"));
    }
    let elements = p
        .block_count
        .checked_mul(p.index_dim)
        .ok_or_else(|| invalid("QSA mRoPE pool extent overflow"))?;
    let raw = elements
        .checked_mul(p.compress)
        .ok_or_else(|| invalid("QSA mRoPE raw extent overflow"))?;
    if p.raw_keys.numel() < raw
        || raw > i32::MAX as usize
        || p.pooled.numel() < elements
        || p.norm
            .is_some_and(|n| n.dtype != DType::BF16 || n.numel() != p.index_dim)
    {
        return Err(invalid("QSA mRoPE pool storage mismatch"));
    }
    let kernel = match (p.raw_keys.dtype, p.pooled.dtype) {
        (DType::F32, DType::F32) => "indexed_attention_pool_mrope_f32",
        (DType::BF16, DType::BF16) => "indexed_attention_pool_mrope_bf16",
        _ => return Err(invalid("QSA mRoPE raw and pooled dtype mismatch")),
    };
    let first = position.position_start / p.compress;
    if first >= p.block_count {
        return Ok(());
    }
    gpu.ensure_kernel_public("qsa_mrope", crate::kernels::QSA_MROPE_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_ptr(p.norm.map_or(std::ptr::null_mut(), |n| n.buf.as_ptr()));
    args.push_ptr(m.positions.buf.as_ptr());
    for value in [p.block_count, p.compress, p.index_dim, first] {
        args.push_i32(int(value)?);
    }
    m.push(&mut args)?;
    p.yarn.push(&mut args);
    args.pad_to(16);
    gpu.launch_blob_recorded(
        kernel,
        [(p.block_count - first) as u32, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}
