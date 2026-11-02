// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// One warp per weight row, R activation rows per warp. x is pre-rotated F32
// [R,K], y is F32 [R,M]. Each group's header and packed bytes are read and
// decoded once; each activation row keeps the scalar GEMV's own four sums.
template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_group(
    const char* gp, int boff, int tid, const float* x, int K, int base,
    float (&acc)[R]
) {
    const unsigned int hA = *(const unsigned int*)(gp);
    const unsigned int hB = *(const unsigned int*)(gp + 4);
    const unsigned int hs = (tid < 16) ? hA : hB;
    float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
    float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
    const unsigned char* d = (const unsigned char*)(gp + 8 + boff);
    unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2];
    unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5];
    int q0 = b0 & 63;
    int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
    int q2 = (b1 >> 4) | ((b2 & 3) << 4);
    int q3 = b2 >> 2;
    int q4 = b3 & 63;
    int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
    int q6 = (b4 >> 4) | ((b5 & 3) << 4);
    int q7 = b5 >> 2;
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float* xr = x + (long long)r * K;
        acc[r] += (sc * (float)q0 + zp) * xr[base]
                + (sc * (float)q1 + zp) * xr[base + 1]
                + (sc * (float)q2 + zp) * xr[base + 2]
                + (sc * (float)q3 + zp) * xr[base + 3]
                + (sc * (float)q4 + zp) * xr[base + 4]
                + (sc * (float)q5 + zp) * xr[base + 5]
                + (sc * (float)q6 + zp) * xr[base + 6]
                + (sc * (float)q7 + zp) * xr[base + 7];
    }
}

template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_row(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row
) {
    const int tid = threadIdx.x;
    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 200;
    const int boff = tid * 6;
    float acc0[R], acc1[R], acc2[R], acc3[R];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        acc0[r] = acc1[r] = acc2[r] = acc3[r] = 0.0f;

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;
    for (int q = 0; q < quads; ++q) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 200;
        const int base = g * 256 + tid * 8;
        mq6g256v2_f32_rows_group<R>(gp0,       boff, tid, x, K, base,       acc0);
        mq6g256v2_f32_rows_group<R>(gp0 + 200, boff, tid, x, K, base + 256, acc1);
        mq6g256v2_f32_rows_group<R>(gp0 + 400, boff, tid, x, K, base + 512, acc2);
        mq6g256v2_f32_rows_group<R>(gp0 + 600, boff, tid, x, K, base + 768, acc3);
    }
    const int g = quads << 2;
    const char* gp = row_ptr + g * 200;
    const int base = g * 256 + tid * 8;
    if (tail >= 1)
        mq6g256v2_f32_rows_group<R>(gp,       boff, tid, x, K, base,       acc0);
    if (tail >= 2)
        mq6g256v2_f32_rows_group<R>(gp + 200, boff, tid, x, K, base + 256, acc1);
    if (tail >= 3)
        mq6g256v2_f32_rows_group<R>(gp + 400, boff, tid, x, K, base + 512, acc2);

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        float acc = (acc0[r] + acc1[r]) + (acc2[r] + acc3[r]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) y[(long long)r * M + row] = acc;
    }
}

template<int R>
__device__ __forceinline__ void mq6g256v2_f32_rows_body(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    mq6g256v2_f32_rows_row<R>(A, x, y, M, K, row);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r2(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<2>(A, x, y, M, K); }

__launch_bounds__(32, 12)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r3(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<3>(A, x, y, M, K); }

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r4(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<4>(A, x, y, M, K); }

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r5(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<5>(A, x, y, M, K); }

__launch_bounds__(32, 6)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r6(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<6>(A, x, y, M, K); }

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r7(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<7>(A, x, y, M, K); }

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq6g256v2_f32_rows_r8(
    const char* A, const float* x, float* y, int M, int K
) { mq6g256v2_f32_rows_body<8>(A, x, y, M, K); }

// Four MQ6G256V2 matrices reading the same R rotated x rows (same K) in one
// launch, blocks routed as gemv_mq6g256v2_x4 routes them; each output row is
// the one-matrix rows kernel's (bitwise). Unused matrices have M = 0.
#define MQ6_ROWS_X4(R, OCC)                                                          \
    __launch_bounds__(32, OCC) extern "C" __global__ void gemm_mq6g256v2_f32_rows_x4_r##R( \
        const char* A0, const char* A1, const char* A2, const char* A3, const float* x, \
        float* y0, float* y1, float* y2, float* y3, int M0, int M1, int M2, int M3, int K) { \
        int row = blockIdx.x;                                                        \
        const char* A = A0; float* y = y0; int M = M0;                               \
        if (row >= M0) {                                                             \
            row -= M0; A = A1; y = y1; M = M1;                                       \
            if (row >= M1) {                                                         \
                row -= M1; A = A2; y = y2; M = M2;                                   \
                if (row >= M2) {                                                     \
                    row -= M2; A = A3; y = y3; M = M3;                               \
                    if (row >= M3) return;                                           \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        mq6g256v2_f32_rows_row<R>(A, x, y, M, K, row);                               \
    }
MQ6_ROWS_X4(2, 16)
MQ6_ROWS_X4(3, 12)
MQ6_ROWS_X4(4, 8)
MQ6_ROWS_X4(5, 6)
MQ6_ROWS_X4(6, 6)
MQ6_ROWS_X4(7, 4)
MQ6_ROWS_X4(8, 4)
#undef MQ6_ROWS_X4

// ---------------------------------------------------------------------------
// Stream-split few-row family (3..8 rows): gemm_mq6g256v2_f32_rows_split_r{R}_w{W}_s{S}
// and the x4 form. Output is bitwise equal to the kernels above and to the
// one-row decode singleton.
//
// DAG (identical per output word): a group's 8 weights are w_i = fma(sc, q_i, zp);
// d = x1*w1; d = fma(x0,w0,d); then fma x2..x7 in order; acc_stream += d. A lane
// keeps four accumulation streams (group g feeds stream g % 4), folds them as
// (a0 + a1) + (a2 + a3), then reduces across the 32 lanes with shfl_down 16..1;
// lane 0 stores. Splitting only changes WHO computes each stream, never the order
// within a stream or the fold tree, so the bits cannot change.
//
// W = weight rows per workgroup (one x load feeds W rows).
// S = waves per workgroup. S == 4: wave s owns stream s of all W rows and the
//     four partials are folded from LDS as (p0 + p1) + (p2 + p3). S == 1: one
//     wave walks all four streams itself. A workgroup never spans two matrices
//     in the x4 form; grid = sum ceil(Mi / W). Rows past M clamp their pointer
//     to row0 and skip the store.
// ---------------------------------------------------------------------------
template<int R, int W>
__device__ __forceinline__ void mq6_split_group(
    const char* const (&gp)[W], int boff, int tid, const float* x, int K, int base,
    float (&acc)[W][R]
) {
    float w[W][8];
    #pragma unroll
    for (int v = 0; v < W; ++v) {
        const unsigned int hA = *(const unsigned int*)(gp[v]);
        const unsigned int hB = *(const unsigned int*)(gp[v] + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        const unsigned char* d = (const unsigned char*)(gp[v] + 8 + boff);
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2];
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5];
        int q[8];
        q[0] = b0 & 63;
        q[1] = (b0 >> 6) | ((b1 & 0xF) << 2);
        q[2] = (b1 >> 4) | ((b2 & 3) << 4);
        q[3] = b2 >> 2;
        q[4] = b3 & 63;
        q[5] = (b3 >> 6) | ((b4 & 0xF) << 2);
        q[6] = (b4 >> 4) | ((b5 & 3) << 4);
        q[7] = b5 >> 2;
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[v][i] = __builtin_fmaf(sc, (float)q[i], zp);
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float4 xa = *(const float4*)(x + (long long)r * K + base);
        const float4 xb = *(const float4*)(x + (long long)r * K + base + 4);
        #pragma unroll
        for (int v = 0; v < W; ++v) {
            float dd = xa.y * w[v][1];
            dd = __builtin_fmaf(xa.x, w[v][0], dd);
            dd = __builtin_fmaf(xa.z, w[v][2], dd);
            dd = __builtin_fmaf(xa.w, w[v][3], dd);
            dd = __builtin_fmaf(xb.x, w[v][4], dd);
            dd = __builtin_fmaf(xb.y, w[v][5], dd);
            dd = __builtin_fmaf(xb.z, w[v][6], dd);
            dd = __builtin_fmaf(xb.w, w[v][7], dd);
            acc[v][r] = acc[v][r] + dd;
        }
    }
}

template<int R, int W>
__device__ __forceinline__ void mq6_split_stream(
    const char* const (&rp)[W], int boff, int lane, const float* x, int K, int gpr, int s,
    float (&acc)[W][R]
) {
    #pragma unroll
    for (int v = 0; v < W; ++v)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[v][r] = 0.0f;
    for (int g = s; g < gpr; g += 4) {
        const char* gp[W];
        #pragma unroll
        for (int v = 0; v < W; ++v) gp[v] = rp[v] + g * 200;
        mq6_split_group<R, W>(gp, boff, lane, x, K, g * 256 + lane * 8, acc);
    }
}

__device__ __forceinline__ void mq6_split_store(
    float a, int lane, int v, int r, float* y, int M, int row0
) {
    for (int offset = 16; offset > 0; offset >>= 1)
        a += __shfl_down(a, offset);
    if (lane == 0 && row0 + v < M) y[(long long)r * M + row0 + v] = a;
}

template<int R, int W, int S>
__device__ __forceinline__ void mq6_split_rows(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row0, float* lds
) {
    const int lane = threadIdx.x & 31;
    const int wv = threadIdx.x >> 5;
    const int gpr = K / 256;
    const int boff = lane * 6;
    const char* rp[W];
    #pragma unroll
    for (int v = 0; v < W; ++v) {
        const int row = (row0 + v < M) ? row0 + v : row0;
        rp[v] = A + (long long)row * gpr * 200;
    }
    if constexpr (S == 1) {
        float a0[W][R], a1[W][R], a2[W][R], a3[W][R];
        mq6_split_stream<R, W>(rp, boff, lane, x, K, gpr, 0, a0);
        mq6_split_stream<R, W>(rp, boff, lane, x, K, gpr, 1, a1);
        mq6_split_stream<R, W>(rp, boff, lane, x, K, gpr, 2, a2);
        mq6_split_stream<R, W>(rp, boff, lane, x, K, gpr, 3, a3);
        #pragma unroll
        for (int v = 0; v < W; ++v)
            #pragma unroll
            for (int r = 0; r < R; ++r)
                mq6_split_store((a0[v][r] + a1[v][r]) + (a2[v][r] + a3[v][r]), lane, v, r, y, M, row0);
    } else {
        float a[W][R];
        mq6_split_stream<R, W>(rp, boff, lane, x, K, gpr, wv, a);
        #pragma unroll
        for (int v = 0; v < W; ++v)
            #pragma unroll
            for (int r = 0; r < R; ++r)
                lds[((v * R + r) * 4 + wv) * 32 + lane] = a[v][r];
        __syncthreads();
        for (int o = wv; o < W * R; o += 4) {
            const float* p = lds + o * 4 * 32 + lane;
            mq6_split_store((p[0] + p[32]) + (p[64] + p[96]), lane, o / R, o % R, y, M, row0);
        }
    }
}

#define MQ6_SPLIT(R, W, S, OCC)                                                          \
    __launch_bounds__(32 * S, OCC) extern "C" __global__                                \
    void gemm_mq6g256v2_f32_rows_split_r##R##_w##W##_s##S(                              \
        const char* A, const float* x, float* y, int M, int K) {                         \
        __shared__ float lds[(S == 1 ? 1 : W * R * 4) * 32];                             \
        mq6_split_rows<R, W, S>(A, x, y, M, K, blockIdx.x * W, lds); }                   \
    __launch_bounds__(32 * S, OCC) extern "C" __global__                                \
    void gemm_mq6g256v2_f32_rows_x4_split_r##R##_w##W##_s##S(                           \
        const char* A0, const char* A1, const char* A2, const char* A3, const float* x, \
        float* y0, float* y1, float* y2, float* y3, int M0, int M1, int M2, int M3, int K) { \
        __shared__ float lds[(S == 1 ? 1 : W * R * 4) * 32];                             \
        int b = blockIdx.x; const char* A = A0; float* y = y0; int M = M0;              \
        int n0 = (M0 + W - 1) / W, n1 = (M1 + W - 1) / W, n2 = (M2 + W - 1) / W,        \
            n3 = (M3 + W - 1) / W;                                                       \
        if (b >= n0) { b -= n0; A = A1; y = y1; M = M1;                                  \
            if (b >= n1) { b -= n1; A = A2; y = y2; M = M2;                              \
                if (b >= n2) { b -= n2; A = A3; y = y3; M = M3; if (b >= n3) return; } } } \
        mq6_split_rows<R, W, S>(A, x, y, M, K, b * W, lds); }

#define MQ6_SPLIT_R(R) MQ6_SPLIT(R, 4, 1, 2) MQ6_SPLIT(R, 4, 4, 4) MQ6_SPLIT(R, 8, 4, 2)
MQ6_SPLIT_R(3) MQ6_SPLIT_R(4) MQ6_SPLIT_R(5) MQ6_SPLIT_R(6) MQ6_SPLIT_R(7) MQ6_SPLIT_R(8)
#undef MQ6_SPLIT_R
#undef MQ6_SPLIT
