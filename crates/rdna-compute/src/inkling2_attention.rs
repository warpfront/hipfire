// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Inkling-2 singleton decode attention, split-KV FA2 (gfx1201 only).
//!
//! Drop-in numerical replacement for the scalar `inkling2_gf_attn_decode_f32`
//! (`hipfire-arch-inkling2/src/gpu_forward.rs`) on the same operands:
//! `q [24,128]` F32, eight F16 `[rows,128]` K and V head planes, the beta
//! biased scores `bias [24,n_kv]` (`rel - sink`, from
//! [`Gpu::inkling2_rel_bias_gather_f32`], clamp unchanged) and the ring
//! `slots [n_kv]` (logical key `k` lives in physical row `slots[k]`).
//!
//! Derived from the shared gfx1201 Q8 FA2 partial+merge family
//! (`kernels/src/attention_q8_0_fa2_gqa.gfx1201.hip`): fragment-major KT64
//! LDS planes, swapped-operand WMMA, CK-style LSE + normalized-O split
//! records, stable max-LSE merge. Shape: H24/KV8/D128 (GQA3); one 16-row
//! tile per KV head carries the three live Q heads in rows 0..2. Q and P
//! enter the WMMA as hi/lo f16 pairs (`x = f16(x) + f16(x - f16(x))`; f16 x
//! f16 products are exact in the f32 accumulator), so the remaining named
//! differences vs the scalar kernel are the dot-product order and the online
//! split merge. The sink is beta's zero-score, zero-value key, added ONCE in
//! the merge; empty partitions write `LSE=-inf, O=0`.
//!
//! Evidence (R9700, `release-0.5.0/inkling/speed/attn-replay`): bitwise
//! ring/linear identity, max |FA2 - scalar| 8.0e-6 against a beta-calibrated
//! budget of 3.5e-5, f64-reference error 4.2e-6 vs beta's 5.8e-6, planted
//! faults 38/38. Not a serving route: no dispatch or route selects it.
//!
//! Scratch is caller-owned F32: `lse [splits,24]`, `oacc [splits,24,128]`.

pub const INKLING2_ATTN_FA2_MODULE: &str = "inkling2_attention_fa2";

pub const INKLING2_ATTN_FA2_HIP_SRC: &str = r#"
#include <hip/hip_runtime.h>

#define IH 24
#define IKV 8
#define ID 128
#define IGQA 3

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

struct KvPlanes {
    const unsigned short* p[8];
};

// lane = 16*kg + ml:
//   Kfrag(sub,dc,lane)[j] = K[key=sub*16+ml][dim=dc*16+kg*8+j]
//   Vfrag(sub,dc,lane)[j] = V[key=sub*16+kg*8+j][dim=dc*16+ml]
__device__ __forceinline__ int ik_kfrag8(int sub, int dc, int lane)
{
    return (dc * 4 + sub) * 32 + lane;
}
__device__ __forceinline__ int ik_vfrag8(int sub, int dc, int lane)
{
    return (sub * 8 + dc) * 32 + lane;
}

// Grid (n_splits, 8), block 128 (four wave32s; every wave fills, wave w owns
// KT16 subtile w of each KT64 tile).
extern "C" __global__ __launch_bounds__(128, 1) void inkling2_attn_fa2_partial_gfx1201(
    const float* __restrict__ q,
    KvPlanes kp,
    KvPlanes vp,
    const float* __restrict__ bias,
    const int* __restrict__ slots,
    float* __restrict__ lse,
    float* __restrict__ oacc,
    int n_kv,
    int n_splits,
    float inv_sqrt_d)
{
    const int split = (int)blockIdx.x;
    const int kv_h = (int)blockIdx.y;
    if (split >= n_splits || kv_h >= IKV)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;

    __shared__ half8_t KV8[2048];          // K frags [0,1024), V frags [1024,2048)
    __shared__ float red_m[4][IGQA];
    __shared__ float red_l[4][IGQA];
    _Float16* V16 = (_Float16*)(KV8 + 1024);

    const unsigned short* kb = kp.p[kv_h];
    const unsigned short* vb = vp.p[kv_h];

    const bool live = ml < IGQA;
    const int h = IGQA * kv_h + (live ? ml : 0);
    const float* brow = bias + (size_t)h * (size_t)n_kv;

    half8_t qhi[8];
    half8_t qlo[8];
#pragma unroll
    for (int dc = 0; dc < 8; ++dc) {
        half8_t f = {0, 0, 0, 0, 0, 0, 0, 0};
        half8_t g = {0, 0, 0, 0, 0, 0, 0, 0};
        if (live) {
            const float* qr = q + (size_t)h * ID + dc * 16 + kg * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                f[j] = (_Float16)qr[j];
                g[j] = (_Float16)(qr[j] - (float)f[j]);
            }
        }
        qhi[dc] = f;
        qlo[dc] = g;
    }

    const int tiles_total = (n_kv + 63) / 64;
    const int per = (tiles_total + n_splits - 1) / n_splits;
    const int t_begin = split * per;
    const int t_end = min(t_begin + per, tiles_total);
    const int key_end = min(n_kv, t_end * 64);

    float8_t Ofr[8];
#pragma unroll
    for (int dc = 0; dc < 8; ++dc)
        Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float m_old = -INFINITY;
    float l_val = 0.0f;

    for (int tile = t_begin; tile < t_end; ++tile) {
        const int kt = tile * 64;
#pragma unroll 2
        for (int t = 0; t < 8; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 4;
            const int c = idx & 15;
            const int gk = kt + k;
            half8_t kv = {0, 0, 0, 0, 0, 0, 0, 0};
            half8_t vv = {0, 0, 0, 0, 0, 0, 0, 0};
            if (gk < key_end) {
                const size_t row = (size_t)slots[gk] * ID + (size_t)c * 8;
                kv = *(const half8_t*)(kb + row);
                vv = *(const half8_t*)(vb + row);
            }
            const int sub = k >> 4;
            KV8[ik_kfrag8(sub, c >> 1, 16 * (c & 1) + (k & 15))] = kv;
            const int vkg = (k >> 3) & 1;
            const int vj = k & 7;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = c * 8 + i;
                V16[ik_vfrag8(sub, d >> 4, 16 * vkg + (d & 15)) * 8 + vj] = vv[i];
            }
        }
        __syncthreads();

        const int ks = kt + wave * 16;
        if (ks < key_end) {
            float8_t sacc = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int dc = 0; dc < 8; ++dc) {
                const half8_t kf = KV8[ik_kfrag8(wave, dc, lane)];
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf, qhi[dc], sacc);
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf, qlo[dc], sacc);
            }
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int key = ks + 8 * kg + j;
                const float x = (live && key < key_end)
                    ? sacc[j] * inv_sqrt_d + brow[key]
                    : -INFINITY;
                sacc[j] = x;
                mx = fmaxf(mx, x);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
            const float m_new = fmaxf(m_old, mx);
            const float alpha = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            m_old = m_new;
            const float origin = (m_new == -INFINITY) ? 0.0f : m_new;
            half8_t phi, plo;
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float e = expf(sacc[j] - origin);
                phi[j] = (_Float16)e;
                plo[j] = (_Float16)(e - (float)phi[j]);
                sum += e;
            }
            sum += __shfl_xor(sum, 16, 32);
            l_val = l_val * alpha + sum;
#pragma unroll
            for (int dc = 0; dc < 8; ++dc) {
                const half8_t vf = KV8[1024 + ik_vfrag8(wave, dc, lane)];
                Ofr[dc] *= alpha;
                Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, phi, Ofr[dc]);
                Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, plo, Ofr[dc]);
            }
        }
        __syncthreads();
    }

    float* ost = (float*)KV8;  // [4][3][128]
    if (live) {
        if (kg == 0) {
            red_m[wave][ml] = m_old;
            red_l[wave][ml] = l_val;
        }
#pragma unroll
        for (int dc = 0; dc < 8; ++dc)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                ost[(wave * IGQA + ml) * ID + dc * 16 + kg * 8 + j] = Ofr[dc][j];
    }
    __syncthreads();
    const int d = tid;
#pragma unroll
    for (int r = 0; r < IGQA; ++r) {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < 4; ++w)
            M = fmaxf(M, red_m[w][r]);
        float L = 0.0f, O = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float mw = red_m[w][r];
            const float wt = (mw == -INFINITY) ? 0.0f : expf(mw - M);
            L += wt * red_l[w][r];
            O += wt * ost[(w * IGQA + r) * ID + d];
        }
        const int rec = split * IH + IGQA * kv_h + r;
        if (d == 0)
            lse[rec] = L > 0.0f ? M + logf(L) : -INFINITY;
        oacc[(size_t)rec * ID + d] = L > 0.0f ? O / L : 0.0f;
    }
}

// Grid 24, block 128. The sink is one synthetic key, score 0 and value 0.
extern "C" __global__ __launch_bounds__(128) void inkling2_attn_fa2_merge_gfx1201(
    const float* __restrict__ lse,
    const float* __restrict__ oacc,
    float* __restrict__ out,
    int n_splits)
{
    const int h = (int)blockIdx.x;
    const int d = (int)threadIdx.x;
    if (h >= IH)
        return;
    float m = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float v = lse[s * IH + h];
        if (isfinite(v))
            m = fmaxf(m, v);
    }
    float den = expf(0.0f - m);
    float num = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float v = lse[s * IH + h];
        if (!isfinite(v))
            continue;
        const float w = expf(v - m);
        den += w;
        num += w * oacc[((size_t)s * IH + h) * ID + d];
    }
    out[(size_t)h * ID + d] = num / den;
}
"#;

use std::ffi::c_void;

use crate::dispatch::{Gpu, GpuTensor};
use hip_bridge::{HipError, HipResult};

pub const INKLING2_ATTN_HEADS: usize = 24;
pub const INKLING2_ATTN_KV_HEADS: usize = 8;
pub const INKLING2_ATTN_HEAD_DIM: usize = 128;
pub const INKLING2_ATTN_MAX_SPLITS: usize = 32;

/// Partition count measured fastest on R9700 per key count (replay receipts
/// `attn-replay/out/receipts-v2.jsonl`).
pub fn inkling2_attn_fa2_splits(n_kv: usize) -> usize {
    match n_kv {
        0..=128 => 2,
        129..=192 => 4,
        193..=2048 => 8,
        2049..=4096 => 16,
        _ => 8,
    }
}

#[repr(C)]
struct KvPlanes([*const c_void; 8]);

impl Gpu {
    /// Split-KV FA2 decode attention for one Inkling-2 query row.
    /// `out [24,128]` F32; `lse`/`oacc` are caller-owned scratch of at least
    /// `splits*24` / `splits*24*128` F32. Fails closed off gfx1201.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_decode_fa2(
        &mut self,
        q: &GpuTensor,
        k_cache: &[GpuTensor],
        v_cache: &[GpuTensor],
        bias: &GpuTensor,
        slots: &GpuTensor,
        lse: &GpuTensor,
        oacc: &GpuTensor,
        out: &GpuTensor,
        n_kv: usize,
        splits: usize,
    ) -> HipResult<()> {
        self.bind_thread()?;
        let (h, kvh, d) = (
            INKLING2_ATTN_HEADS,
            INKLING2_ATTN_KV_HEADS,
            INKLING2_ATTN_HEAD_DIM,
        );
        if self.arch != "gfx1201" {
            return Err(HipError::new(
                0,
                &format!("inkling2_attn_decode_fa2: gfx1201 only, not {}", self.arch),
            ));
        }
        if k_cache.len() != kvh || v_cache.len() != kvh {
            return Err(HipError::new(
                0,
                "inkling2_attn_decode_fa2: need 8 K and 8 V head planes",
            ));
        }
        if splits == 0 || splits > INKLING2_ATTN_MAX_SPLITS || n_kv == 0 {
            return Err(HipError::new(
                0,
                &format!("inkling2_attn_decode_fa2: splits {splits}, n_kv {n_kv}"),
            ));
        }
        let f32_len = |t: &GpuTensor| t.buf.size() / 4;
        if f32_len(q) < h * d
            || f32_len(out) < h * d
            || f32_len(bias) < h * n_kv
            || slots.buf.size() < n_kv * 4
            || f32_len(lse) < splits * h
            || f32_len(oacc) < splits * h * d
        {
            return Err(HipError::new(
                0,
                "inkling2_attn_decode_fa2: operand too small",
            ));
        }
        self.ensure_kernel(
            INKLING2_ATTN_FA2_MODULE,
            INKLING2_ATTN_FA2_HIP_SRC,
            "inkling2_attn_fa2_partial_gfx1201",
        )?;
        self.ensure_kernel(
            INKLING2_ATTN_FA2_MODULE,
            INKLING2_ATTN_FA2_HIP_SRC,
            "inkling2_attn_fa2_merge_gfx1201",
        )?;

        let q_ptr = q.buf.as_ptr();
        let kp = KvPlanes(std::array::from_fn(|i| {
            k_cache[i].buf.as_ptr() as *const c_void
        }));
        let vp = KvPlanes(std::array::from_fn(|i| {
            v_cache[i].buf.as_ptr() as *const c_void
        }));
        let bias_ptr = bias.buf.as_ptr();
        let slots_ptr = slots.buf.as_ptr();
        let lse_ptr = lse.buf.as_ptr();
        let oacc_ptr = oacc.buf.as_ptr();
        let out_ptr = out.buf.as_ptr();
        let n_kv_val = n_kv as i32;
        let splits_val = splits as i32;
        let isd = 1.0f32 / (d as f32).sqrt();
        let mut params: Vec<*mut c_void> = vec![
            &q_ptr as *const _ as *mut c_void,
            &kp as *const _ as *mut c_void,
            &vp as *const _ as *mut c_void,
            &bias_ptr as *const _ as *mut c_void,
            &slots_ptr as *const _ as *mut c_void,
            &lse_ptr as *const _ as *mut c_void,
            &oacc_ptr as *const _ as *mut c_void,
            &n_kv_val as *const _ as *mut c_void,
            &splits_val as *const _ as *mut c_void,
            &isd as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_attn_fa2_partial_gfx1201",
            [splits as u32, kvh as u32, 1],
            [128, 1, 1],
            0,
            &mut params,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(q_ptr);
                for p in kp.0.iter().chain(vp.0.iter()) {
                    b.push_ptr(*p);
                }
                b.push_ptr(bias_ptr);
                b.push_ptr(slots_ptr);
                b.push_ptr(lse_ptr);
                b.push_ptr(oacc_ptr);
                b.push_i32(n_kv_val);
                b.push_i32(splits_val);
                b.push_f32(isd);
                b
            },
        )?;
        let mut mparams: Vec<*mut c_void> = vec![
            &lse_ptr as *const _ as *mut c_void,
            &oacc_ptr as *const _ as *mut c_void,
            &out_ptr as *const _ as *mut c_void,
            &splits_val as *const _ as *mut c_void,
        ];
        self.launch_maybe_blob(
            "inkling2_attn_fa2_merge_gfx1201",
            [h as u32, 1, 1],
            [d as u32, 1, 1],
            0,
            &mut mparams,
            || {
                let mut b = hip_bridge::KernargBlob::new();
                b.push_ptr(lse_ptr);
                b.push_ptr(oacc_ptr);
                b.push_ptr(out_ptr);
                b.push_i32(splits_val);
                b
            },
        )
    }
}
