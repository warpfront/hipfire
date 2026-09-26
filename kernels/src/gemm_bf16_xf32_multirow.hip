// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact BF16-weight × F32-input multirow prefill GEMM.

#include <hip/hip_runtime.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// Layouts:
//   W: [M, K] BF16, row-major
//   X: [N, K] F32, row-major
//   Y: [N, M] F32, row-major
//
// One wave owns one output-channel row and four adjacent token rows.  The
// per-token lane accumulators intentionally match gemv_bf16_xf32: each lane
// loads eight BF16 values with uint4, distributes them over four accumulators,
// reduces with (a0+a1)+(a2+a3), then performs the same wave shuffle.  The
// widened weights are reused across the four token rows, but no arithmetic
// association is changed from the batch-1 reference path.
//
// Grid: [M, ceil(N / 4), 1].  A partial final token tile aliases its last
// valid input row for masked lanes, so no out-of-range activation is read.

// Lane 0 of `v += __shfl_down(v, off, 32)` for off = 16, 8, 4, 2, 1: the
// balanced tree v[i] + v[i+off] over lanes i < off.  Only lanes below `off`
// feed lane 0, so on gfx11/gfx12 the partner comes from a cross-row permlane
// (off 16) and in-row DPP shifts folded into the adds (off < 16) instead of
// LDS bpermutes; additions, operands and order are unchanged.  Only lane 0's
// result is defined.
static __device__ __forceinline__ float hf_lane0_tree_sum(float v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
#else
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
#endif
    return v;
}
__launch_bounds__(32, 16)
extern "C" __global__ void gemm_bf16_xf32_multirow(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int batch_base = blockIdx.y * 4;
    const int lane = threadIdx.x;
    if (row >= M || lane >= 32 || batch_base >= N) return;

    const unsigned short* w_row = W + (size_t)row * K;
    const int b0 = batch_base;
    const int b1 = (batch_base + 1 < N) ? batch_base + 1 : b0;
    const int b2 = (batch_base + 2 < N) ? batch_base + 2 : b0;
    const int b3 = (batch_base + 3 < N) ? batch_base + 3 : b0;
    const float* x0 = X + (size_t)b0 * K;
    const float* x1 = X + (size_t)b1 * K;
    const float* x2 = X + (size_t)b2 * K;
    const float* x3 = X + (size_t)b3 * K;

    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

    // Match gemv_bf16_xf32's wide K traversal and association exactly:
    // each lane handles eight consecutive BF16 values, with four accumulators
    // reused for positions 0/4, 1/5, 2/6, and 3/7 in that eight-value group.
    if ((K & 7) == 0) {
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa0 = *(const float4*)(x0 + i);
            const float4 xb0 = *(const float4*)(x0 + i + 4);
            const float4 xa1 = *(const float4*)(x1 + i);
            const float4 xb1 = *(const float4*)(x1 + i + 4);
            const float4 xa2 = *(const float4*)(x2 + i);
            const float4 xb2 = *(const float4*)(x2 + i + 4);
            const float4 xa3 = *(const float4*)(x3 + i);
            const float4 xb3 = *(const float4*)(x3 + i + 4);

            const float w0 = HF_BF16_LO(wv.x);
            const float w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y);
            const float w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z);
            const float w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w);
            const float w7 = HF_BF16_HI(wv.w);

            a00 += w0 * xa0.x;
            a01 += w1 * xa0.y;
            a02 += w2 * xa0.z;
            a03 += w3 * xa0.w;
            a00 += w4 * xb0.x;
            a01 += w5 * xb0.y;
            a02 += w6 * xb0.z;
            a03 += w7 * xb0.w;

            a10 += w0 * xa1.x;
            a11 += w1 * xa1.y;
            a12 += w2 * xa1.z;
            a13 += w3 * xa1.w;
            a10 += w4 * xb1.x;
            a11 += w5 * xb1.y;
            a12 += w6 * xb1.z;
            a13 += w7 * xb1.w;

            a20 += w0 * xa2.x;
            a21 += w1 * xa2.y;
            a22 += w2 * xa2.z;
            a23 += w3 * xa2.w;
            a20 += w4 * xb2.x;
            a21 += w5 * xb2.y;
            a22 += w6 * xb2.z;
            a23 += w7 * xb2.w;

            a30 += w0 * xa3.x;
            a31 += w1 * xa3.y;
            a32 += w2 * xa3.z;
            a33 += w3 * xa3.w;
            a30 += w4 * xb3.x;
            a31 += w5 * xb3.y;
            a32 += w6 * xb3.z;
            a33 += w7 * xb3.w;
        }
    }

    // Match gemv_bf16_xf32's scalar tail for K % 8 != 0 (and its complete
    // traversal when the wide path is unavailable).
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + lane; i < K; i += 32) {
        const float weight = __uint_as_float(((unsigned int)w_row[i]) << 16);
        a00 += weight * x0[i];
        a10 += weight * x1[i];
        a20 += weight * x2[i];
        a30 += weight * x3[i];
    }

    float s0 = (a00 + a01) + (a02 + a03);
    float s1 = (a10 + a11) + (a12 + a13);
    float s2 = (a20 + a21) + (a22 + a23);
    float s3 = (a30 + a31) + (a32 + a33);
    s0 = hf_lane0_tree_sum(s0);
    s1 = hf_lane0_tree_sum(s1);
    s2 = hf_lane0_tree_sum(s2);
    s3 = hf_lane0_tree_sum(s3);

    if (lane == 0) {
        Y[(size_t)b0 * M + row] = s0;
        if (batch_base + 1 < N) Y[(size_t)b1 * M + row] = s1;
        if (batch_base + 2 < N) Y[(size_t)b2 * M + row] = s2;
        if (batch_base + 3 < N) Y[(size_t)b3 * M + row] = s3;
    }
}


// gfx11/gfx12 only (permlanex16 / row DPP); dispatched on gfx1151.
#if defined(__GFX11__) || defined(__GFX12__)
// Two lane-0 trees in one pass: lanes 0-15 carry a, lanes 16-31 carry b.
// After the cross-row step lane i holds a[i] + a[i+16] and lane 16+i holds
// b[16+i] + b[i] (the same sum, IEEE add commutes); the in-row DPP steps then
// reduce both rows with the reference offsets, so a ends in lane 0 and b in
// lane 16 bit-identical to two hf_lane0_tree_sum calls.  a is returned in
// lane 0; b is broadcast (readlane 16).
static __device__ __forceinline__ void hf_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}

// Rows-loop sibling for small K (K % 8 == 0, ceil(K / 256) == ITERS): one wave
// keeps its four token rows' X in registers and walks HF_ROWS_R output rows,
// prefetching the next row's weights, instead of one wave per row re-loading
// X.  Per (row, token) the lane traversal, accumulator chains, (a0+a1)+(a2+a3)
// fold and shuffle tree are the kernel above's, so results are bitwise equal.
// Grid: [ceil(M / HF_ROWS_R), ceil(N / 4), 1], block 32.  Row groups vary
// fastest: with token tiles fastest, concurrent waves write Y lines of many
// token rows 4*M bytes apart and the kernel degrades ~8x once N > 512.
#define HF_ROWS_R 16
template <int ITERS>
__device__ __forceinline__ void bf16_rows_loop(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int N) {
    const int lane = threadIdx.x;
    const int batch_base = blockIdx.y * 4;
    const int row_base = blockIdx.x * HF_ROWS_R;
    if (row_base >= M || lane >= 32 || batch_base >= N) return;
    const int b0 = batch_base;
    const int bt[4] = {b0, (b0 + 1 < N) ? b0 + 1 : b0, (b0 + 2 < N) ? b0 + 2 : b0,
                       (b0 + 3 < N) ? b0 + 3 : b0};
    bool ok[ITERS];
    float4 xa[ITERS][4], xb[ITERS][4];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        const int i = lane * 8 + it * 256;
        ok[it] = i + 8 <= K;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float* xr = X + (size_t)bt[t] * K + (ok[it] ? i : 0);
            xa[it][t] = *(const float4*)(xr);
            xb[it][t] = *(const float4*)(xr + 4);
        }
    }
    const int rows = (M - row_base < HF_ROWS_R) ? M - row_base : HF_ROWS_R;
    uint4 wn[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it)
        wn[it] = *(const uint4*)(W + (size_t)row_base * K + (ok[it] ? lane * 8 + it * 256 : 0));
    for (int r = 0; r < rows; ++r) {
        uint4 wv[ITERS];
#pragma unroll
        for (int it = 0; it < ITERS; ++it) wv[it] = wn[it];
        const int rn = (r + 1 < rows) ? r + 1 : r;
#pragma unroll
        for (int it = 0; it < ITERS; ++it)
            wn[it] = *(const uint4*)(W + (size_t)(row_base + rn) * K
                                     + (ok[it] ? lane * 8 + it * 256 : 0));
        float a[4][4];
#pragma unroll
        for (int t = 0; t < 4; ++t) a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
#pragma unroll
        for (int it = 0; it < ITERS; ++it) {
            if (!ok[it]) continue;
            const float w0 = HF_BF16_LO(wv[it].x), w1 = HF_BF16_HI(wv[it].x);
            const float w2 = HF_BF16_LO(wv[it].y), w3 = HF_BF16_HI(wv[it].y);
            const float w4 = HF_BF16_LO(wv[it].z), w5 = HF_BF16_HI(wv[it].z);
            const float w6 = HF_BF16_LO(wv[it].w), w7 = HF_BF16_HI(wv[it].w);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                a[t][0] += w0 * xa[it][t].x;
                a[t][1] += w1 * xa[it][t].y;
                a[t][2] += w2 * xa[it][t].z;
                a[t][3] += w3 * xa[it][t].w;
                a[t][0] += w4 * xb[it][t].x;
                a[t][1] += w5 * xb[it][t].y;
                a[t][2] += w6 * xb[it][t].z;
                a[t][3] += w7 * xb[it][t].w;
            }
        }
        float s[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) s[t] = (a[t][0] + a[t][1]) + (a[t][2] + a[t][3]);
        hf_lane0_tree_sum2(s[0], s[1]);
        hf_lane0_tree_sum2(s[2], s[3]);
        if (lane == 0) {
            const int row = row_base + r;
#pragma unroll
            for (int t = 0; t < 4; ++t)
                if (batch_base + t < N) Y[(size_t)(batch_base + t) * M + row] = s[t];
        }
    }
}


__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_xf32_multirow_rows3(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int N) {
    bf16_rows_loop<3>(W, X, Y, M, K, N);
}

// Per-thread-output sibling for 256 < K <= 512 (K % 8 == 0): the four-row
// kernel above spends a lane per 8 K values, so K = 320 leaves 24 of 32
// lanes idle in its second step and pays a shuffle tree per output.  Here
// lane = output row (32 rows per block, weights staged in LDS with an odd
// dword stride) and each thread replays the reference's 32 lane partials
// serially — same (a0..a3) chains, same (a0+a1)+(a2+a3) fold — then the
// 16/8/4/2/1 tree in registers, so every output is bitwise identical.  Eight
// waves take TPW tokens each, TGROUPS times.  The asm pins keep the unrolled
// lane pairs from being hoisted into one another (spills otherwise).
// Grid: [ceil(M / 32), ceil(N / (8 * TPW * TGROUPS)), 1], block 256,
// dynamic LDS 32 * (K / 2 + 1) * 4 bytes.
template <int ITERS, int TPW, int TGROUPS>
__device__ __forceinline__ void bf16_per_thread(const unsigned short* __restrict__ W,
    const float* __restrict__ X, float* __restrict__ Y, int M, int K, int N) {
    extern __shared__ unsigned int ws[];
    const int stride = K / 2 + 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int row0 = blockIdx.x * 32;
    const int vpr = K / 8;  // uint4 per row
    for (int v = tid; v < 32 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = row0 + r < M ? row0 + r : M - 1;
        const uint4 u = *(const uint4*)(W + (size_t)row * K + c * 8);
        unsigned int* d = ws + r * stride + c * 4;
        d[0] = u.x; d[1] = u.y; d[2] = u.z; d[3] = u.w;
    }
    __syncthreads();
    const unsigned int* wr = ws + lane * stride;
    for (int tg = 0; tg < TGROUPS; ++tg) {
        const int tbase = ((blockIdx.y * TGROUPS + tg) * 8 + wave) * TPW;
        if (tbase >= N) break;
        int tok[TPW];
#pragma unroll
        for (int t = 0; t < TPW; ++t) tok[t] = tbase + t < N ? tbase + t : tbase;
        // Level-16 pairs first: vs[l] = s[l] + s[l + 16] is the first tree
        // step, so each iteration retires two virtual lanes.
        float h[TPW][16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float sp[2][TPW];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int vl = l + 16 * p;
                float a[TPW][4];
#pragma unroll
                for (int t = 0; t < TPW; ++t) a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
#pragma unroll
                for (int it = 0; it < ITERS; ++it) {
                    const int i = vl * 8 + it * 256;
                    if (i + 8 > K) break;
                    const unsigned int u0 = wr[i / 2], u1 = wr[i / 2 + 1], u2 = wr[i / 2 + 2], u3 = wr[i / 2 + 3];
                    const float w0 = HF_BF16_LO(u0), w1 = HF_BF16_HI(u0);
                    const float w2 = HF_BF16_LO(u1), w3 = HF_BF16_HI(u1);
                    const float w4 = HF_BF16_LO(u2), w5 = HF_BF16_HI(u2);
                    const float w6 = HF_BF16_LO(u3), w7 = HF_BF16_HI(u3);
#pragma unroll
                    for (int t = 0; t < TPW; ++t) {
                        const float* xr = X + (size_t)tok[t] * K + i;
                        a[t][0] += w0 * xr[0];
                        a[t][1] += w1 * xr[1];
                        a[t][2] += w2 * xr[2];
                        a[t][3] += w3 * xr[3];
                        a[t][0] += w4 * xr[4];
                        a[t][1] += w5 * xr[5];
                        a[t][2] += w6 * xr[6];
                        a[t][3] += w7 * xr[7];
                    }
                }
#pragma unroll
                for (int t = 0; t < TPW; ++t) sp[p][t] = (a[t][0] + a[t][1]) + (a[t][2] + a[t][3]);
            }
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                h[t][l] = sp[0][t] + sp[1][t];
                asm volatile("" : "+v"(h[t][l]) :: "memory");
            }
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
#pragma unroll
                for (int l = 0; l < off; ++l) h[t][l] = h[t][l] + h[t][l + off];
            if (row0 + lane < M && tbase + t < N) Y[(size_t)(tbase + t) * M + row0 + lane] = h[t][0];
        }
    }
}
__launch_bounds__(256, 1)
extern "C" __global__ void gemm_bf16_xf32_multirow_pto2(
    const unsigned short* __restrict__ W, const float* __restrict__ X,
    float* __restrict__ Y, int M, int K, int N) {
    bf16_per_thread<2, 2, 4>(W, X, Y, M, K, N);
}

#endif  // __GFX11__ || __GFX12__

// BF16 -> F16 (RNE via the F32 widening, which is exact).  Model-lifetime
// weight shadows for the gfx1151 F16 WMMA prefill route.
extern "C" __global__ void convert_bf16_to_f16(
    const unsigned short* __restrict__ in, _Float16* __restrict__ out, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = (_Float16)__uint_as_float((unsigned int)in[i] << 16);
}
