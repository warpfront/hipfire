// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Fused Helmert-chart route finalize + stream aggregate (+ plain-gamma RMSNorm),
// one workgroup per row, the old streams read from HBM ONCE.
//
// The unfused chain reads the whole [rows, n, width] stream stack three times
// before the branch: the selector GEMM, `stream_mix_route_helmert` (weightless
// RMS over every stream) and `stream_mix_aggregate`. This kernel replaces the
// last two. Each thread loads its n*KPER elements d = tid + 256*j (j ascending,
// stream-major) into registers exactly once. Because width = KPER*256, element
// j = s*KPER + k of thread tid is x[s, c] with c = tid + 256*k, i.e. the same
// element the aggregate sum of column c needs, so the registers serve the RMS
// and the aggregate:
//
//   rms_acc   = sum_{j ascending} x_j^2           (route_helmert's per-thread order)
//   inv       = rsqrt(tree(rms_acc)/x_dim + selector_eps)
//   pre/post/res  exactly as stream_mix_route_helmert (written to global)
//   a[c]      = round(sum_s pre[s] * x[s, c])     (aggregate's per-column order)
//   out[c]    = round(a[c] * rsqrt(tree(sum a^2)/width + eps) * gamma[c])
//
// Every arithmetic expression, reduction order and reduction tree is copied
// from stream_mix_route_helmert.hip / stream_mix_aggregate.hip, so pre, post, res
// and out are bit-identical to running the two kernels in sequence. `pre` is
// consumed from LDS instead of being re-read from global (same F32 values).
//
// Streams are element type XT: float (F32 planes carrying BF16 values) or
// unsigned short (BF16 bits, widened by the high-half shift, exact).
//
// Instantiated for (n, KPER) pairs listed at the bottom; the host admits only
// those geometries. Supported 2 <= n <= 8.
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

// Orthonormal Helmert column kk (0-based), see stream_mix_route_helmert.hip.
__device__ __forceinline__ float sm_helmert(int i, int kk) {
    const int k = kk + 1;
    const float s = 1.0f / sqrtf((float)(k * (k + 1)));
    if (i < k) return s;
    if (i == k) return -(float)k * s;
    return 0.0f;
}

__device__ __forceinline__ float sm_ld(const float* p) { return *p; }
__device__ __forceinline__ float sm_ld(const unsigned short* p) {
    return __uint_as_float(((unsigned int)*p) << 16);
}

template <typename XT, int N, int KPER>
__device__ __forceinline__ void stream_mix_route_aggregate_body(
    const XT* __restrict__ x,          // [rows, N*width] old streams
    const float* __restrict__ z_raw,   // [rows, zdim]
    const float* __restrict__ params,  // [4 + zdim]
    float* __restrict__ pre,           // [rows, N]
    float* __restrict__ post,          // [rows, N]
    float* __restrict__ res,           // [rows, N, N]
    const void* __restrict__ gamma,    // [width] or null
    float* __restrict__ out,           // [rows, width]
    float post_factor,
    float selector_eps,
    float eps,
    int bf16,
    int has_norm,
    int gamma_kind,
    int n_rt
) {
    constexpr int WIDTH = KPER * SM_THREADS;
    constexpr int X_DIM = N * WIDTH;
    constexpr int NM = N - 1;
    constexpr int ZDIM = 2 * N + NM * NM;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float red[SM_THREADS];
    __shared__ float tz[SM_MAX_ZDIM];
    __shared__ float ratio_s[SM_MAX_N * SM_MAX_N];
    __shared__ float pre_s[SM_MAX_N];

    // One global read of the row; the weightless RMS accumulates in
    // route_helmert's order (d = tid + 256*j ascending).
    const XT* xr = x + (size_t)row * X_DIM;
    float v[N * KPER];
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < N * KPER; ++j) {
        const float t = sm_ld(xr + (size_t)j * SM_THREADS + tid);
        v[j] = t;
        acc += t * t;
    }
    red[tid] = acc;
    __syncthreads();
    for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)X_DIM + selector_eps);

    const float alpha_pre = params[0];
    const float alpha_post = params[1];
    const float alpha_z = params[2];
    const float eta = params[3];
    const float* bias = params + 4;

    if (tid < ZDIM) {
        float zv = z_raw[(size_t)row * ZDIM + tid];
        if (bf16) zv = sm_round_bf16(zv);
        zv *= inv;
        if (tid < N) {
            const float p = sm_sigmoid(alpha_pre * zv + bias[tid]);
            pre[(size_t)row * N + tid] = p;
            pre_s[tid] = p;
        } else if (tid < 2 * N) {
            post[(size_t)row * N + (tid - N)] =
                post_factor * sm_sigmoid(alpha_post * zv + bias[tid]);
        } else {
            tz[tid - 2 * N] = alpha_z * zv + bias[tid];
        }
    }
    __syncthreads();

    // E_ij = sum_{m,k} Q[i,m] Z[m,k] Q[j,k]
    // Written with the RUNTIME stream count (n_rt == N, a kernel argument), as in
    // stream_mix_route_helmert.hip: with N a compile-time constant the compiler
    // constant-folds the Helmert scales and re-associates/contracts this chain
    // differently, which changes `res` by an ulp.
    const int n = n_rt;
    const int nm = n - 1;
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

    // Aggregate: column c = tid + 256*k, sum over streams in stream order.
    // (route_helmert's tail leaves no pending reads of `red`, and the barrier
    // after the controls stage above ordered every read of red[0].)
    float agg[KPER];
    float ss = 0.0f;
#pragma unroll
    for (int k = 0; k < KPER; ++k) {
        float a = 0.0f;
#pragma unroll
        for (int s = 0; s < N; ++s) a += pre_s[s] * v[s * KPER + k];
        if (bf16) a = sm_round_bf16(a);
        agg[k] = a;
        ss += a * a;
    }

    float r = 1.0f;
    if (has_norm) {
        red[tid] = ss;
        __syncthreads();
        for (int off = SM_THREADS / 2; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        r = rsqrtf(red[0] / (float)WIDTH + eps);
    }

    float* o = out + (size_t)row * WIDTH;
#pragma unroll
    for (int k = 0; k < KPER; ++k) {
        const int c = tid + k * SM_THREADS;
        float val = agg[k];
        if (has_norm) {
            float g;
            if (gamma_kind == 1) {
                g = __uint_as_float(((unsigned int)((const unsigned short*)gamma)[c]) << 16);
            } else {
                g = ((const float*)gamma)[c];
            }
            val = val * r * g;
            if (bf16) val = sm_round_bf16(val);
        }
        o[c] = val;
    }
}

#define SM_ROUTE_AGG_KERNEL(NAME, XT, N, KPER)                                        \
    extern "C" __launch_bounds__(SM_THREADS) __global__ void NAME(                    \
        const XT* __restrict__ x, const float* __restrict__ z_raw,                    \
        const float* __restrict__ params, float* __restrict__ pre,                    \
        float* __restrict__ post, float* __restrict__ res,                            \
        const void* __restrict__ gamma, float* __restrict__ out, float post_factor,   \
        float selector_eps, float eps, int bf16, int has_norm, int gamma_kind,        \
        int n_rt) {                                                                   \
        stream_mix_route_aggregate_body<XT, N, KPER>(                                 \
            x, z_raw, params, pre, post, res, gamma, out, post_factor, selector_eps,  \
            eps, bf16, has_norm, gamma_kind, n_rt);                                   \
    }

// Raon-1-mini: 4 streams x 2560 (10 x 256).
SM_ROUTE_AGG_KERNEL(stream_mix_route_aggregate_n4_k10, float, 4, 10)
SM_ROUTE_AGG_KERNEL(stream_mix_route_aggregate_n4_k10_bf16, unsigned short, 4, 10)
