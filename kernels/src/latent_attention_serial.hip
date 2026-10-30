// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Serial reference latent attention (deterministic FP32 softmax), independent
// QK and V widths:
//     key_h(t)   = [ knope_h(t)[nope]  ||  tail(t)[tail] ]   (tail shared by heads)
//     score      = (q_h . key_h(t)) * scale
//     y_h        = sum_t softmax_t(score) * v_h(t)[dv]
// with causal visibility j <= *pos_ptr + i for query row i. No positional
// rotation anywhere; the tail channels are plain extra key/query channels.
//
// Inputs:  q [n_rows, H, nope+tail] (BF16-valued F32), tail read straight from
// the latent cache row (columns latent..latent+tail), K_nope / V from the
// token-major BF16 mirrors written by latent_attention_decompress_*.
// Output:  y [n_rows, H, dv] BF16-valued F32 (one BF16 rounding, as the
// reference's BF16 attention output).
//
// One block per (head, query row); keys are processed in tiles of LA_KT, one
// key per thread (dot product in fixed ascending channel order), tile max and
// tile sum by fixed binary trees, value accumulation in ascending key order.
// Rows at or beyond the device length are never read. No atomics.

#define LA_KT 128
#define LA_DVQ 4   // value accumulators per thread => dv <= LA_KT * LA_DVQ

static __device__ __forceinline__ uint16_t la_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float la_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

// acc = fma(q[d], bf16(row[d]), acc) for d in [0, n) ascending.
static __device__ __forceinline__ float la_dot_row(
    const float* __restrict__ qs, const uint16_t* __restrict__ row, int n, int vec8, float acc
) {
    if (vec8) {
        const uint4* r4 = (const uint4*)row;
        for (int d = 0; d < n; d += 8) {
            const uint4 u = r4[d >> 3];
            acc = fmaf(qs[d + 0], __uint_as_float(u.x << 16), acc);
            acc = fmaf(qs[d + 1], __uint_as_float(u.x & 0xffff0000u), acc);
            acc = fmaf(qs[d + 2], __uint_as_float(u.y << 16), acc);
            acc = fmaf(qs[d + 3], __uint_as_float(u.y & 0xffff0000u), acc);
            acc = fmaf(qs[d + 4], __uint_as_float(u.z << 16), acc);
            acc = fmaf(qs[d + 5], __uint_as_float(u.z & 0xffff0000u), acc);
            acc = fmaf(qs[d + 6], __uint_as_float(u.w << 16), acc);
            acc = fmaf(qs[d + 7], __uint_as_float(u.w & 0xffff0000u), acc);
        }
    } else {
        for (int d = 0; d < n; ++d) acc = fmaf(qs[d], la_bf16_to_f32(row[d]), acc);
    }
    return acc;
}

extern "C" __global__ __launch_bounds__(LA_KT) void latent_attention_serial_bf16(
    const float* __restrict__ q,         // [n_rows, H, nope+tail]
    const uint16_t* __restrict__ cache,  // [cap, latent+tail] BF16
    const uint16_t* __restrict__ knope,  // [mcap, H, nope] BF16
    const uint16_t* __restrict__ vmir,   // [mcap, H, dv] BF16
    float* __restrict__ y,               // [n_rows, H, dv]
    const int* __restrict__ pos_ptr,     // device scalar: rows cached before this call
    int n_rows, int n_heads, int nope, int tail, int dv, int latent, int cap,
    float scale, int vec8, int mcap
) {
    extern __shared__ float smem[];
    const int h = blockIdx.x;
    const int i = blockIdx.y;
    const int tid = threadIdx.x;
    const int dqk = nope + tail;
    const int width = latent + tail;
    float* qs = smem;
    float* ps = qs + dqk;
    float* red = ps + LA_KT;

    int len = *pos_ptr + i + 1;
    if (len > cap) len = cap;
    if (len > mcap) len = mcap;
    if (len < 0) len = 0;

    const float* qr = q + ((size_t)i * n_heads + h) * dqk;
    for (int d = tid; d < dqk; d += LA_KT) qs[d] = qr[d];
    __syncthreads();

    float acc[LA_DVQ];
    #pragma unroll
    for (int r = 0; r < LA_DVQ; ++r) acc[r] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int kt = 0; kt < len; kt += LA_KT) {
        const int j = kt + tid;
        const bool valid = j < len;
        float s = -INFINITY;
        if (valid) {
            float a = la_dot_row(qs, knope + ((size_t)j * n_heads + h) * nope, nope, vec8, 0.0f);
            a = la_dot_row(qs + nope, cache + (size_t)j * width + latent, tail, vec8, a);
            s = a * scale;
        }
        red[tid] = s;
        __syncthreads();
        for (int st = LA_KT / 2; st > 0; st >>= 1) {
            if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
            __syncthreads();
        }
        const float tmax = red[0];
        __syncthreads();
        const float m_new = fmaxf(m, tmax);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - m_new);
        const float p = valid ? expf(s - m_new) : 0.0f;
        ps[tid] = p;
        red[tid] = p;
        __syncthreads();
        for (int st = LA_KT / 2; st > 0; st >>= 1) {
            if (tid < st) red[tid] += red[tid + st];
            __syncthreads();
        }
        const float psum = red[0];
        l = l * corr + psum;
        m = m_new;
        const int cnt = (len - kt < LA_KT) ? (len - kt) : LA_KT;
        const uint16_t* vbase = vmir + ((size_t)kt * n_heads + h) * dv;
        const size_t vstride = (size_t)n_heads * dv;
        #pragma unroll
        for (int r = 0; r < LA_DVQ; ++r) {
            const int d = tid + r * LA_KT;
            if (d < dv) {
                float a = acc[r] * corr;
                for (int jj = 0; jj < cnt; ++jj)
                    a = fmaf(ps[jj], la_bf16_to_f32(vbase[(size_t)jj * vstride + d]), a);
                acc[r] = a;
            }
        }
        __syncthreads();
    }

    float* yr = y + ((size_t)i * n_heads + h) * dv;
    #pragma unroll
    for (int r = 0; r < LA_DVQ; ++r) {
        const int d = tid + r * LA_KT;
        if (d < dv) {
            const float o = (len > 0) ? acc[r] / l : 0.0f;
            yr[d] = la_bf16_to_f32(la_f32_to_bf16(o));
        }
    }
}
