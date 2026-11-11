// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Packed-epilogue stream-split few-row MQ6G256V2 kernels (3..8 rows), opt-in
// (HIPFIRE_QWEN4_MQ6_R8_PK=1, default off):
//   gemm_mq6g256v2_f32_rows_pk_r{R}_w{W}_s{S}      one matrix
//   gemm_mq6g256v2_f32_rows_x4_pk_r{R}_w{W}_s{S}   four matrices, one launch
//
// Bitwise equal to gemm_mq6g256v2_f32_rows[_x4]_split_r{R}_w{W}_s{S} (and so to
// the one-row decode singleton): the main loop is that kernel's, group for
// group, so every per-lane stream (x1*w1, then fmaf x0..x7 in order, then
// stream += d) is unchanged, and the four streams still fold as
// (a0 + a1) + (a2 + a3). Only the cross-lane reduction is restructured.
//
// Incumbent: every output runs its own shfl_down 16, 8, 4, 2, 1 chain (five
// ds_bpermute / swizzle steps, one output at a time), ~1000 instructions per
// wave for 32 outputs. Here the 32-lane tree of every output is the same tree
// (level d adds lane l and lane l ^ d of the previous level, d = 16, 8, 4, 2,
// 1; IEEE addition is commutative so which operand is "mine" cannot change a
// bit), but two outputs share each register per level: lanes with bit d clear
// keep output U and receive U from the partner lane, lanes with bit d set keep
// V and receive V. One exchange then reduces two outputs, so N outputs cost
// ~N exchanges instead of 5N. After the packing levels lane l holds the output
// whose index is sum_i bit_(4-i)(l) << i over the packing levels i.
//
// Padding registers (outputs >= N) are never added into a live output: a pad
// partner only ever lands in lanes whose output is itself a pad.

// Lane l receives the value of lane l ^ D (D in 16, 8, 4, 2, 1).
template<int D>
__device__ __forceinline__ float mq6_pk_xchg(float v) {
    const int iv = __float_as_int(v);
    int r;
    if constexpr (D == 16) {
        r = __builtin_amdgcn_permlanex16(iv, iv, 0x76543210u, 0xFEDCBA98u, false, false);
    } else if constexpr (D == 8) {
        r = __builtin_amdgcn_update_dpp(0, iv, 0x168, 0xF, 0xF, false);  // row_xmask:8
    } else if constexpr (D == 4) {
        r = __builtin_amdgcn_update_dpp(0, iv, 0x164, 0xF, 0xF, false);  // row_xmask:4
    } else if constexpr (D == 2) {
        r = __builtin_amdgcn_update_dpp(0, iv, 0x162, 0xF, 0xF, false);  // row_xmask:2
    } else {
        r = __builtin_amdgcn_update_dpp(0, iv, 0x161, 0xF, 0xF, false);  // row_xmask:1
    }
    return __int_as_float(r);
}

// v: NR (power of two, <= 32) per-lane partials of NR outputs. Returns the
// reduced value of the output this lane owns (see the header comment).
template<int D, int NR>
__device__ __forceinline__ float mq6_pk_reduce(const float (&v)[NR], int lane) {
    if constexpr (D == 0) {
        return v[0];
    } else if constexpr (NR == 1) {
        const float t[1] = { v[0] + mq6_pk_xchg<D>(v[0]) };
        return mq6_pk_reduce<D / 2, 1>(t, lane);
    } else {
        const bool hi = (lane & D) != 0;
        float o[NR / 2];
        #pragma unroll
        for (int i = 0; i < NR / 2; ++i) {
            const float u = v[2 * i], w = v[2 * i + 1];
            const float keep = hi ? w : u;
            const float send = hi ? u : w;
            o[i] = keep + mq6_pk_xchg<D>(send);
        }
        return mq6_pk_reduce<D / 2, NR / 2>(o, lane);
    }
}

constexpr int mq6_pk_pow2(int n) { return n <= 1 ? 1 : (n <= 2 ? 2 : (n <= 4 ? 4 : (n <= 8 ? 8 : (n <= 16 ? 16 : 32)))); }
constexpr int mq6_pk_log2(int n) { return n <= 1 ? 0 : (n <= 2 ? 1 : (n <= 4 ? 2 : (n <= 8 ? 3 : (n <= 16 ? 4 : 5)))); }

// Output index (0..NR-1) owned by `lane` after mq6_pk_reduce with NR registers.
template<int NR>
__device__ __forceinline__ int mq6_pk_owner(int lane) {
    constexpr int L = mq6_pk_log2(NR);
    int j = 0;
    #pragma unroll
    for (int i = 0; i < L; ++i) j |= ((lane >> (4 - i)) & 1) << i;
    return j;
}

// ---- main loop: verbatim from gemm_mq6g256v2_f32_rows.hip (split family) ----
template<int R, int W>
__device__ __forceinline__ void mq6_pk_group(
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
__device__ __forceinline__ void mq6_pk_stream(
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
        mq6_pk_group<R, W>(gp, boff, lane, x, K, g * 256 + lane * 8, acc);
    }
}

template<int R, int W, int S>
__device__ __forceinline__ void mq6_pk_rows(
    const char* __restrict__ A, const float* __restrict__ x,
    float* __restrict__ y, int M, int K, int row0, float* lds
) {
    constexpr int N = W * R;  // outputs per workgroup, o = v * R + r
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
        constexpr int NR = mq6_pk_pow2(N);
        constexpr int L = mq6_pk_log2(NR);
        float a0[W][R], a1[W][R], a2[W][R], a3[W][R];
        mq6_pk_stream<R, W>(rp, boff, lane, x, K, gpr, 0, a0);
        mq6_pk_stream<R, W>(rp, boff, lane, x, K, gpr, 1, a1);
        mq6_pk_stream<R, W>(rp, boff, lane, x, K, gpr, 2, a2);
        mq6_pk_stream<R, W>(rp, boff, lane, x, K, gpr, 3, a3);
        float f[NR];
        #pragma unroll
        for (int o = 0; o < NR; ++o) {
            if (o < N) f[o] = (a0[o / R][o % R] + a1[o / R][o % R]) + (a2[o / R][o % R] + a3[o / R][o % R]);
            else f[o] = 0.0f;
        }
        const float res = mq6_pk_reduce<16, NR>(f, lane);
        const int o = mq6_pk_owner<NR>(lane);
        if ((lane & ((1 << (5 - L)) - 1)) == 0 && o < N && row0 + o / R < M)
            y[(long long)(o % R) * M + row0 + o / R] = res;
    } else {
        float a[W][R];
        mq6_pk_stream<R, W>(rp, boff, lane, x, K, gpr, wv, a);
        #pragma unroll
        for (int v = 0; v < W; ++v)
            #pragma unroll
            for (int r = 0; r < R; ++r)
                lds[((v * R + r) * 4 + wv) * 32 + lane] = a[v][r];
        __syncthreads();
        // wave wv folds outputs o = wv + 4 j, j = 0..NJ-1, from the four streams
        constexpr int NJ = (N + 3) / 4;
        constexpr int NR = mq6_pk_pow2(NJ);
        constexpr int L = mq6_pk_log2(NR);
        float f[NR];
        #pragma unroll
        for (int j = 0; j < NR; ++j) {
            const int o = wv + 4 * j;
            if (j < NJ && o < N) {
                const float* p = lds + o * 4 * 32 + lane;
                f[j] = (p[0] + p[32]) + (p[64] + p[96]);
            } else f[j] = 0.0f;
        }
        const float res = mq6_pk_reduce<16, NR>(f, lane);
        const int j = mq6_pk_owner<NR>(lane);
        const int o = wv + 4 * j;
        if ((lane & ((1 << (5 - L)) - 1)) == 0 && j < NJ && o < N && row0 + o / R < M)
            y[(long long)(o % R) * M + row0 + o / R] = res;
    }
}

#define MQ6_PK(R, W, S, OCC)                                                             \
    __launch_bounds__(32 * S, OCC) extern "C" __global__                                \
    void gemm_mq6g256v2_f32_rows_pk_r##R##_w##W##_s##S(                                 \
        const char* A, const float* x, float* y, int M, int K) {                         \
        __shared__ float lds[(S == 1 ? 1 : W * R * 4) * 32];                             \
        mq6_pk_rows<R, W, S>(A, x, y, M, K, blockIdx.x * W, lds); }                      \
    __launch_bounds__(32 * S, OCC) extern "C" __global__                                \
    void gemm_mq6g256v2_f32_rows_x4_pk_r##R##_w##W##_s##S(                              \
        const char* A0, const char* A1, const char* A2, const char* A3, const float* x, \
        float* y0, float* y1, float* y2, float* y3, int M0, int M1, int M2, int M3, int K) { \
        __shared__ float lds[(S == 1 ? 1 : W * R * 4) * 32];                             \
        int b = blockIdx.x; const char* A = A0; float* y = y0; int M = M0;              \
        int n0 = (M0 + W - 1) / W, n1 = (M1 + W - 1) / W, n2 = (M2 + W - 1) / W,        \
            n3 = (M3 + W - 1) / W;                                                       \
        if (b >= n0) { b -= n0; A = A1; y = y1; M = M1;                                  \
            if (b >= n1) { b -= n1; A = A2; y = y2; M = M2;                              \
                if (b >= n2) { b -= n2; A = A3; y = y3; M = M3; if (b >= n3) return; } } } \
        mq6_pk_rows<R, W, S>(A, x, y, M, K, b * W, lds); }

#define MQ6_PK_R(R) MQ6_PK(R, 4, 1, 2) MQ6_PK(R, 4, 4, 4) MQ6_PK(R, 8, 4, 2)
MQ6_PK_R(3) MQ6_PK_R(4) MQ6_PK_R(5) MQ6_PK_R(6) MQ6_PK_R(7) MQ6_PK_R(8)
#undef MQ6_PK_R
#undef MQ6_PK
