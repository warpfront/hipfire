// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if HIPFIRE_RMSNORM_EMIT_Q81
#include <hip/hip_fp16.h>
#include <stdint.h>
#define HIPFIRE_RMSNORM_F32_KERNEL rmsnorm_f32_q8_1
struct block_q8_1_mmq { half2 ds4[4]; int8_t qs[128]; };
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
#elif HIPFIRE_RMSNORM_WAVE_REDUCE
#define HIPFIRE_RMSNORM_F32_KERNEL rmsnorm_f32_wave_reduce
#else
#define HIPFIRE_RMSNORM_F32_KERNEL rmsnorm_f32
#endif

extern "C" __global__ void HIPFIRE_RMSNORM_F32_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
#if HIPFIRE_RMSNORM_EMIT_Q81
    block_q8_1_mmq* __restrict__ out_q8,
#endif
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
#if HIPFIRE_RMSNORM_WAVE_REDUCE
    // Preserve the stock power-of-two reduction order down through 32 live
    // lanes, then finish that identical tree with wave shuffles.  This removes
    // five LDS read/write + workgroup-barrier stages without reassociation.
    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float wave_sum = threadIdx.x < 32 ? sdata[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset >>= 1)
            wave_sum += __shfl_down(wave_sum, offset, 32);
        if (threadIdx.x == 0) sdata[0] = wave_sum;
    }
    __syncthreads();
#else
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
#endif
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
#if HIPFIRE_RMSNORM_EMIT_Q81
    // Re-evaluate four contiguous normalized values per lane so the Q8_1
    // reduction has exactly the standalone quantizer's 8-lane/32-value tree.
    for (int chunk = 0; chunk < n; chunk += blockDim.x * 4) {
        const int i0 = chunk + threadIdx.x * 4;
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int batch_base = blockIdx.x * n;
        if (i0 + 3 < n) {
            float4 xv = *reinterpret_cast<const float4*>(x + batch_base + i0);
            float4 wv = *reinterpret_cast<const float4*>(weight + i0);
            v = make_float4(xv.x * wv.x * rms, xv.y * wv.y * rms,
                            xv.z * wv.z * rms, xv.w * wv.w * rms);
        }
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        float sum = v.x + v.y + v.z + v.w;
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
            sum += __shfl_xor(sum, offset, 32);
        }
        const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
#define Q81(a) ((char)lrintf(fmaxf(-127.0f, fminf(127.0f, (a) * id))))
        const char4 q = make_char4(Q81(v.x), Q81(v.y), Q81(v.z), Q81(v.w));
#undef Q81
        if (i0 < n) {
            const int kblock = i0 / 128;
            const int iqs = i0 & 127;
            block_q8_1_mmq* qb = out_q8 + (long long)blockIdx.x * (n / 128) + kblock;
            reinterpret_cast<char4*>(qb->qs)[iqs / 4] = q;
            if ((iqs & 31) == 0) qb->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
        }
    }
#endif
}

#undef HIPFIRE_RMSNORM_F32_KERNEL
