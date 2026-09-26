// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-side embedding lookup dispatch methods (HFQ4G256, HFQ4G128, Q8, Q4K).

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use crate::kernels;
use hip_bridge::{HipError, HipResult};

impl Gpu {
    /// GPU-side embedding lookup: copy row `token_id` from embedding table to output.
    /// Avoids downloading the entire embedding table to CPU.
    pub fn embedding_lookup(
        &self,
        table: &GpuTensor,  // [vocab_size * dim] F32
        output: &GpuTensor, // [dim] F32
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let byte_offset = (token_id as usize) * dim * 4;
        let byte_size = dim * 4;
        self.hip
            .memcpy_dtod_offset(&output.buf, &table.buf, byte_offset, byte_size)
    }

    /// Q4_LUT GEMV: 4-bit with LDS codebook lookup. 48 bytes per 32 elements.

    /// Wave-cooperative Q4 GEMV (Q4_F16_G32 format, 0.625 B/w). Shuffle-based nibble distribution.

    /// Q4-as-Q8 GEMV: 4-bit precision stored in Q8_0 format (1.0625 B/w). Gets Q8 occupancy.

    /// Q8_0 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_q8(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q8", kernels::EMBEDDING_Q8_SRC, "embedding_q8")?;
        let func = &self.functions["embedding_q8"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip
                .launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// Q8 embedding lookup driven by a device token-id scalar, broadcasting
    /// the dequantized row into `copies` contiguous output rows. Typed for
    /// retained replay so the token id remains dynamic across PM4 launches.
    pub fn embedding_lookup_q8_buf_broadcast(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id_buf: &GpuTensor,
        dim: usize,
        copies: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let logical_name = "embedding_q8_buf_broadcast";
        self.ensure_kernel(logical_name, kernels::EMBEDDING_Q8_SRC, logical_name)?;
        let tp = table.buf.as_ptr();
        let op = output.buf.as_ptr();
        let idp = token_id_buf.buf.as_ptr();
        let mut d = dim as i32;
        let mut n = copies as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &idp as *const _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(logical_name, [1, 1, 1], [256, 1, 1], 0, &mut params, || {
            let mut b = hip_bridge::KernargBlob::new();
            b.push_ptr(tp);
            b.push_ptr(op);
            b.push_ptr(idp);
            b.push_i32(d);
            b.push_i32(n);
            b
        })
    }

    /// Q4_K embedding lookup: dequantize one row on GPU, output F32.
    /// table is raw Q4_K bytes on GPU, output is [dim] F32.
    pub fn embedding_lookup_q4k(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel("embedding_q4k", kernels::EMBEDDING_Q4K_SRC, "embedding_q4k")?;
        let func = &self.functions["embedding_q4k"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip
                .launch_kernel(func, [1, 1, 1], [256, 1, 1], 0, None, &mut params)
        }
    }

    /// HFQ4-G256 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g256(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g256",
            kernels::EMBEDDING_HFQ4G256_SRC,
            "embedding_hfq4g256",
        )?;
        let func = &self.functions["embedding_hfq4g256"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        let bytes = crate::profile::embedding_hfq4g256_bytes(dim);
        let timer =
            crate::profile::begin_timer(&self.hip, "embedding", "embedding_lookup_hfq4g256", bytes);
        let result = unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        };
        if let Some(t) = timer {
            t.finish(&self.hip);
        }
        result
    }

    /// Batched Q8_0 embedding lookup. Same hipGraph-captureable pattern as
    /// the HFQ4G256 variant. `output` shape: `[n × dim]` row-major.
    pub fn embedding_lookup_q8_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_q8_batched",
            kernels::EMBEDDING_Q8_BATCHED_SRC,
            "embedding_q8_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_q8_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(op);
                b.push_ptr(tidp);
                b.push_i32(d);
                b
            },
        )
    }

    /// Batched F16 embedding lookup. Copies N rows of an F16 table into
    /// `output[n × dim]` (F32), reading token ids from a device buffer so the
    /// caller's chain stays GPU-resident. The F16→F32 widening is exact, so the
    /// result is byte-identical to a host `f16_to_f32` per-element conversion.
    ///
    /// `output` shape: `[n × dim]` row-major. `token_ids` shape: `[n]` i32.
    pub fn embedding_lookup_f16_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_f16_batched",
            kernels::EMBEDDING_F16_BATCHED_SRC,
            "embedding_f16_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_f16_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(op);
                b.push_ptr(tidp);
                b.push_i32(d);
                b
            },
        )
    }
    /// Batched BF16 embedding lookup. Resident BF16 rows are widened to F32
    /// on device; token IDs are read from a persistent device buffer.
    pub fn embedding_lookup_bf16_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_bf16_batched",
            kernels::EMBEDDING_BF16_BATCHED_SRC,
            "embedding_bf16_batched",
        )?;
        if table.dtype != DType::BF16
            || output.dtype != DType::F32
            || token_ids.dtype != DType::Raw
            || output.numel() < n * dim
            || token_ids.numel() < n
        {
            return Err(HipError::new(0, "embedding BF16 shape/dtype mismatch"));
        }
        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "embedding_bf16_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(op);
                b.push_ptr(tidp);
                b.push_i32(d);
                b
            },
        )
    }
    /// Batched packed MQv2 embedding lookup. The decoder writes each row in
    /// the quantizer's rotated basis; the matching orthogonal FWHT is then
    /// applied on device so `output` is a natural F32 embedding row. No row
    /// bytes or token ids cross back to the host.
    pub fn embedding_lookup_mq4v2_batched(
        &mut self,
        table: &GpuTensor,
        rotated: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        if !matches!(table.dtype, DType::MQ4G256V2 | DType::MQ4G128V2)
            || rotated.dtype != DType::F32
            || output.dtype != DType::F32
            || token_ids.dtype != DType::Raw
            || table.shape.last().copied() != Some(dim)
            || rotated.numel() < n.saturating_mul(dim)
            || output.numel() < n.saturating_mul(dim)
            || token_ids.buf.size() < n.saturating_mul(std::mem::size_of::<i32>())
        {
            return Err(HipError::new(0, "embedding MQv2 shape/dtype mismatch"));
        }
        let format = if table.dtype == DType::MQ4G256V2 { 44usize } else { 53usize };
        let groups = if format == 44 {
            dim.checked_div(256)
                .ok_or_else(|| HipError::new(0, "qt44 embedding width is not 256-aligned"))?
        } else {
            dim.div_ceil(128)
        };
        let row_bytes = groups
            .checked_mul(if format == 44 { 136 } else { 68 })
            .ok_or_else(|| HipError::new(0, "embedding MQv2 row stride overflow"))?;
        let rows = table.shape[..table.shape.len() - 1]
            .iter()
            .try_fold(1usize, |acc, &v| acc.checked_mul(v))
            .ok_or_else(|| HipError::new(0, "embedding MQv2 row count overflow"))?;
        if table.buf.size() < rows.saturating_mul(row_bytes) {
            return Err(HipError::new(0, "embedding MQv2 packed table is truncated"));
        }
        self.ensure_kernel(
            "embedding_mq4v2_batched",
            kernels::EMBEDDING_MQ4V2_BATCHED_SRC,
            "embedding_mq4v2_batched",
        )?;
        let tp = table.buf.as_ptr();
        let rp = rotated.buf.as_ptr();
        let ids = token_ids.buf.as_ptr();
        let mut format_i = format as i32;
        let mut n_i = n as i32;
        let mut dim_i = dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &tp as *const _ as *mut c_void,
            &rp as *const _ as *mut c_void,
            &ids as *const _ as *mut c_void,
            &mut n_i as *mut _ as *mut c_void,
            &mut dim_i as *mut _ as *mut c_void,
            &mut format_i as *mut _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "embedding_mq4v2_batched",
            [n as u32, groups as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(rp);
                b.push_ptr(ids);
                b.push_i32(n_i);
                b.push_i32(dim_i);
                b.push_i32(format_i);
                b
            },
        )?;
        if format == 44 {
            self.rotate_x_mq_batched(rotated, output, dim, n)
        } else {
            self.rotate_x_mq_128_v2(rotated, output, dim, n)
        }
    }


    /// Batched HFQ4-G256 embedding lookup. Dequantizes N rows in a single
    /// launch, reading token ids from a device buffer. hipGraph-capture-safe:
    /// callers update `token_ids` between replays and replay the same graph.
    ///
    /// `output` shape: `[n × dim]` row-major. `token_ids` shape: `[n]` i32.
    pub fn embedding_lookup_hfq4g256_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g256_batched",
            kernels::EMBEDDING_HFQ4G256_BATCHED_SRC,
            "embedding_hfq4g256_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_hfq4g256_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(op);
                b.push_ptr(tidp);
                b.push_i32(d);
                b
            },
        )
    }

    /// HFQ4-G128 embedding lookup: dequantize one row on GPU, output F32.
    pub fn embedding_lookup_hfq4g128(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_id: u32,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g128",
            kernels::EMBEDDING_HFQ4G128_SRC,
            "embedding_hfq4g128",
        )?;
        let func = &self.functions["embedding_hfq4g128"];

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tid = token_id as i32;
        let mut d = dim as i32;

        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tid as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        unsafe {
            self.hip.launch_kernel(
                func,
                [1, 1, 1],
                [256, 1, 1],
                0,
                self.stream_ref(),
                &mut params,
            )
        }
    }

    /// Batched HFQ4-G128 embedding lookup. `output` is `[n × dim]` row-major
    /// and `token_ids` is a device-resident i32-compatible buffer.
    pub fn embedding_lookup_hfq4g128_batched(
        &mut self,
        table: &GpuTensor,
        output: &GpuTensor,
        token_ids: &GpuTensor,
        n: usize,
        dim: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        self.ensure_kernel(
            "embedding_hfq4g128_batched",
            kernels::EMBEDDING_HFQ4G128_BATCHED_SRC,
            "embedding_hfq4g128_batched",
        )?;

        let mut tp = table.buf.as_ptr();
        let mut op = output.buf.as_ptr();
        let mut tidp = token_ids.buf.as_ptr();
        let mut d = dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &mut tp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut tidp as *mut _ as *mut c_void,
            &mut d as *mut _ as *mut c_void,
        ];

        self.launch_maybe_blob(
            "embedding_hfq4g128_batched",
            [n as u32, 1, 1],
            [256, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(tp);
                b.push_ptr(op);
                b.push_ptr(tidp);
                b.push_i32(d);
                b
            },
        )
    }
}
