// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Stream routing with the bounded Helmert chart, one workgroup per row.
//
// Inputs are the OLD streams x[r] (flattened, n*width values) and the raw
// selector projection z_raw[r] = x[r] . phi^T ([zdim] values, produced by the
// ordinary projection GEMM, zdim = 2n + (n-1)^2). Per row:
//
//   inv  = rsqrt(mean(x^2) + selector_eps)                 (weightless RMS)
//   z    = round(z_raw) * inv                              (round: model dtype)
//   pre  = sigmoid(alpha_pre  * z[0:n]        + b_pre)
//   post = post_factor * sigmoid(alpha_post * z[n:2n] + b_post)
//   Z    = reshape(alpha_z * z[2n:] + b_z, n-1, n-1)
//   E    = Q Z Q^T                       Q = Helmert basis [n, n-1], Q^T Q = I
//   B    = (I + 11^T / n) / 2
//   rho  = max(0, max_ij(-E_ij / B_ij))
//   Phi  = B + E / (1 + rho)
//   g    = sigmoid(eta)
//   res  = (1 - g) I + g Phi                                [n, n] dest/source
//
// `params` is one F32 vector: [alpha_pre, alpha_post, alpha_z, eta,
// b_pre[n], b_post[n], b_z[(n-1)^2]]; its bias tail is index-aligned with z.
// Outputs pre/post/res are F32. Supported 2 <= n <= 8. All outputs of a row
// depend only on that row's old streams.
//
// Grid:  [rows, 1, 1]
// Block: [256, 1, 1]

#include <hip/hip_runtime.h>

#define SM_THREADS 256
#define SM_MAX_N 8
#define SM_MAX_ZDIM (2 * SM_MAX_N + (SM_MAX_N - 1) * (SM_MAX_N - 1))

__device__ __forceinline__ float sm_round_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

__device__ __forceinline__ float sm_sigmoid(float v) {
    return 1.0f / (1.0f + expf(-v));
}

// Orthonormal Helmert column kk (0-based): entries 1 for rows < k, -k at row
// k, zero below, scaled by 1/sqrt(k(k+1)), with k = kk + 1.
__device__ __forceinline__ float sm_helmert(int i, int kk) {
    const int k = kk + 1;
    const float s = 1.0f / sqrtf((float)(k * (k + 1)));
    if (i < k) return s;
    if (i == k) return -(float)k * s;
    return 0.0f;
}

extern "C" __launch_bounds__(SM_THREADS)
__global__ void stream_mix_route_helmert(
    const float* __restrict__ x,       // [rows, x_dim] old streams
    const float* __restrict__ z_raw,   // [rows, zdim]
    const float* __restrict__ params,  // [4 + zdim]
    float*       __restrict__ pre,     // [rows, n]
    float*       __restrict__ post,    // [rows, n]
    float*       __restrict__ res,     // [rows, n, n]
    int n,
    int x_dim,
    float post_factor,
    float selector_eps,
    int bf16
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nm = n - 1;
    const int m2 = nm * nm;
    const int zdim = 2 * n + m2;

    __shared__ float red[SM_THREADS];
    __shared__ float tz[SM_MAX_ZDIM];
    __shared__ float ratio_s[SM_MAX_N * SM_MAX_N];

    // Weightless inverse RMS over the whole stream stack, fixed reduction tree.
    const float* xr = x + (size_t)row * x_dim;
    float acc = 0.0f;
    for (int d = tid; d < x_dim; d += SM_THREADS) {
        const float v = xr[d];
        acc += v * v;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)x_dim + selector_eps);

    const float alpha_pre = params[0];
    const float alpha_post = params[1];
    const float alpha_z = params[2];
    const float eta = params[3];
    const float* bias = params + 4;

    if (tid < zdim) {
        float zv = z_raw[(size_t)row * zdim + tid];
        if (bf16) zv = sm_round_bf16(zv);
        zv *= inv;
        if (tid < n) {
            pre[(size_t)row * n + tid] = sm_sigmoid(alpha_pre * zv + bias[tid]);
        } else if (tid < 2 * n) {
            post[(size_t)row * n + (tid - n)] =
                post_factor * sm_sigmoid(alpha_post * zv + bias[tid]);
        } else {
            tz[tid - 2 * n] = alpha_z * zv + bias[tid];
        }
    }
    __syncthreads();

    // E_ij = sum_{m,k} Q[i,m] Z[m,k] Q[j,k]
    const float inv_n = 1.0f / (float)n;
    float e = 0.0f;
    float bij = 0.0f;
    if (tid < n * n) {
        const int i = tid / n;
        const int j = tid - i * n;
        for (int k = 0; k < nm; ++k) {
            float u = 0.0f;
            for (int m = 0; m < nm; ++m) u += sm_helmert(i, m) * tz[m * nm + k];
            e += u * sm_helmert(j, k);
        }
        bij = 0.5f * ((i == j ? 1.0f : 0.0f) + inv_n);
        ratio_s[tid] = -e / bij;
    }
    __syncthreads();

    if (tid < n * n) {
        float rho = 0.0f;
        for (int t = 0; t < n * n; ++t) rho = fmaxf(rho, ratio_s[t]);
        const int i = tid / n;
        const int j = tid - i * n;
        const float phi = bij + e / (1.0f + rho);
        const float g = sm_sigmoid(eta);
        res[(size_t)row * n * n + tid] = (1.0f - g) * (i == j ? 1.0f : 0.0f) + g * phi;
    }
}
