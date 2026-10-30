// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Latent-attention split ops (generic latent/tail geometry).
//
//  * latent_attention_split_kv_bf16: one projected row [latent || tail]
//    -> RMSNorm over the latent channels ONLY (fp32 statistics, plain gamma,
//    no 1+gamma, no bias), rounded once to BF16, written together with the
//    untouched tail channels into row `*pos_ptr + i` of a row-major
//    [cache_rows, latent+tail] BF16 cache. The tail is never normalized and
//    never rotated. The write position is device-resident (graph-capture safe).
//  * latent_attention_rms_rows_bf16: row RMSNorm, BF16 result kept as an F32
//    plane (input and output may alias).
//
// Reduction order is fixed (per-thread strided partials, then a binary tree),
// so results are run-to-run deterministic.

#define LA_BLOCK 256

static __device__ __forceinline__ uint16_t la_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float la_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}
static __device__ __forceinline__ float la_load_gamma(const void* g, int gamma_bf16, int i) {
    return gamma_bf16 ? la_bf16_to_f32(((const uint16_t*)g)[i]) : ((const float*)g)[i];
}

static __device__ __forceinline__ float la_block_sum(float v, float* red) {
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s = LA_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float total = red[0];
    __syncthreads();
    return total;
}

extern "C" __global__ __launch_bounds__(LA_BLOCK) void latent_attention_split_kv_bf16(
    const float* __restrict__ kv_a,   // [n_rows, latent+tail], BF16-valued F32
    const void* __restrict__ gamma,   // [latent] F32 or BF16
    uint16_t* __restrict__ cache,     // [cache_rows, latent+tail] BF16
    const int* __restrict__ pos_ptr,  // device scalar: rows already in cache
    int n_rows, int latent, int tail, float eps, int gamma_bf16, int cache_rows
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    const int width = latent + tail;
    const int dst_row = *pos_ptr + row;
    if (dst_row < 0 || dst_row >= cache_rows) return;  // block-uniform
    __shared__ float red[LA_BLOCK];
    const float* x = kv_a + (size_t)row * width;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < latent; c += LA_BLOCK) {
        float v = x[c];
        acc += v * v;
    }
    float total = la_block_sum(acc, red);
    float r = 1.0f / sqrtf(total / (float)latent + eps);
    uint16_t* out = cache + (size_t)dst_row * width;
    for (int c = threadIdx.x; c < latent; c += LA_BLOCK) {
        float v = x[c] * r;
        out[c] = la_f32_to_bf16(v * la_load_gamma(gamma, gamma_bf16, c));
    }
    for (int c = threadIdx.x; c < tail; c += LA_BLOCK) {
        out[latent + c] = la_f32_to_bf16(x[latent + c]);
    }
}

extern "C" __global__ __launch_bounds__(LA_BLOCK) void latent_attention_rms_rows_bf16(
    const float* x,                   // [n_rows, width]
    const void* __restrict__ gamma,   // [width] F32 or BF16
    float* out,                       // [n_rows, width] BF16-valued F32; may alias x
    int n_rows, int width, float eps, int gamma_bf16
) {
    const int row = blockIdx.x;
    if (row >= n_rows) return;
    __shared__ float red[LA_BLOCK];
    const float* xr = x + (size_t)row * width;
    float acc = 0.0f;
    for (int c = threadIdx.x; c < width; c += LA_BLOCK) {
        float v = xr[c];
        acc += v * v;
    }
    float total = la_block_sum(acc, red);
    float r = 1.0f / sqrtf(total / (float)width + eps);
    float* o = out + (size_t)row * width;
    for (int c = threadIdx.x; c < width; c += LA_BLOCK) {
        float v = xr[c] * r;
        o[c] = __uint_as_float(((uint32_t)la_f32_to_bf16(v * la_load_gamma(gamma, gamma_bf16, c))) << 16);
    }
}
