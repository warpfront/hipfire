// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 embedding rows decoded on the device.
//!
//! The packed `embed.weight` (qt44, 246 MB) stays on the host; a prefill block
//! gathers its rows' packed bytes (1,224 B per 2304-wide row) into one staging
//! buffer and [`Gpu::inkling2_embed_qt44_rows`] decodes them to F32 rows.
//!
//! The decode is **bitwise** the host decoder (`hipfire_cpu::quant::dequant_group`
//! for `Mq4G256V2`, behind `EmbedPacked::row_f32`): per element
//! `scale * code + zero` rounded as a product then a sum (contraction off), then
//! the inverse FWHT-256 in the host's exact operation order: `* signs2`, the
//! radix-2 butterfly with strides 1, 2, 4, …, 128 (`a + b`, `a - b`), then
//! `* (0.0625 * signs1)`. The sign tables are the device's G256 MQ tables
//! (seeds 42 / 1042, [`Gpu::ensure_mq_signs`]), the host's `SIGNS1` / `SIGNS2`.

use std::ffi::c_void;

use crate::dispatch::{DType, Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};

/// Runtime module name of [`INKLING2_EMBED_HIP_SRC`].
pub const INKLING2_EMBED_MODULE: &str = "inkling2_embed";

/// Bytes of one packed qt44 group: two `(scale, zero)` F16 pairs, 128 code bytes.
pub const INKLING2_EMBED_QT44_GROUP_BYTES: usize = 136;

/// One workgroup (32 lanes x 8 elements) per `(256-group, row)`. Lane `l` holds
/// elements `8l .. 8l + 7`: butterfly strides 1, 2, 4 stay in the lane, strides
/// 8 .. 128 pair lane `l` with `l ^ (stride / 8)`.
pub const INKLING2_EMBED_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ __launch_bounds__(32)
void inkling2_embed_qt44_rows(
    const unsigned char* __restrict__ packed,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ out,
    int dim)
{
#pragma clang fp contract(off)
    const int group = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    const int groups = dim >> 8;
    const unsigned char* gp = packed + ((size_t)row * groups + group) * 136;
    const unsigned int h = *(const unsigned int*)(gp + (lane < 16 ? 0 : 4));
    const float scale = __half2float(__ushort_as_half((unsigned short)(h & 0xffffu)));
    const float zero = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
    const unsigned int codes = *(const unsigned int*)(gp + 8 + lane * 4);
    const int i0 = lane * 8;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float c = (float)((codes >> (4 * j)) & 0xfu);
        const float d = scale * c + zero;
        v[j] = d * signs2[i0 + j];
    }
#pragma unroll
    for (int s = 1; s < 8; s <<= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (!(j & s)) {
                const float a = v[j];
                const float b = v[j + s];
                v[j] = a + b;
                v[j + s] = a - b;
            }
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float p = __shfl_xor(v[j], m, 32);
            v[j] = upper ? p - v[j] : v[j] + p;
        }
    }
    float* o = out + (size_t)row * dim + group * 256 + i0;
    float r[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = v[j] * (0.0625f * signs1[i0 + j]);
    *(float4*)(o) = make_float4(r[0], r[1], r[2], r[3]);
    *(float4*)(o + 4) = make_float4(r[4], r[5], r[6], r[7]);
}
"#;

impl Gpu {
    /// Decode `rows` packed qt44 embedding rows (`packed`, `Raw`, rows of
    /// `dim / 256 * 136` bytes back to back) into `out [rows, dim]` F32, bitwise
    /// the host qt44 decoder (module docs). `dim % 256 == 0`.
    pub fn inkling2_embed_qt44_rows(
        &mut self,
        packed: &GpuTensor,
        out: &GpuTensor,
        rows: usize,
        dim: usize,
    ) -> HipResult<()> {
        let bad = |m: String| Err(HipError::new(0, &format!("inkling2_embed_qt44_rows: {m}")));
        if rows == 0 {
            return Ok(());
        }
        if dim == 0 || dim % 256 != 0 {
            return bad(format!("dim {dim} is not a positive multiple of 256"));
        }
        let row_bytes = dim / 256 * INKLING2_EMBED_QT44_GROUP_BYTES;
        if packed.dtype != DType::Raw || packed.buf.size() < rows * row_bytes {
            return bad(format!(
                "packed {:?} of {} bytes < {rows} x {row_bytes}",
                packed.dtype,
                packed.buf.size()
            ));
        }
        if out.dtype != DType::F32 || out.buf.size() < rows * dim * 4 {
            return bad(format!(
                "out {:?} of {} bytes < [{rows}, {dim}] F32",
                out.dtype,
                out.buf.size()
            ));
        }
        if (out.buf.as_ptr() as usize) % 16 != 0 {
            return bad("out not 16-byte aligned".into());
        }
        self.ensure_mq_signs()?;
        self.bind_thread()?;
        self.ensure_kernel(
            INKLING2_EMBED_MODULE,
            INKLING2_EMBED_HIP_SRC,
            "inkling2_embed_qt44_rows",
        )?;
        let sign = |t: &Option<GpuTensor>| {
            t.as_ref()
                .map(|t| t.buf.as_ptr())
                .ok_or_else(|| HipError::new(0, "inkling2_embed_qt44_rows: no MQ sign tables"))
        };
        let s1 = sign(&self.scratch.mq_signs1)?;
        let s2 = sign(&self.scratch.mq_signs2)?;
        let pp = packed.buf.as_ptr();
        let op = out.buf.as_ptr();
        let di = dim as i32;
        let mut params: Vec<*mut c_void> = vec![
            &pp as *const _ as *mut c_void,
            &s1 as *const _ as *mut c_void,
            &s2 as *const _ as *mut c_void,
            &op as *const _ as *mut c_void,
            &di as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_embed_qt44_rows",
            [(dim / 256) as u32, rows as u32, 1],
            [32, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(pp);
                b.push_ptr(s1);
                b.push_ptr(s2);
                b.push_ptr(op);
                b.push_i32(di);
                b
            },
        )
    }
}
