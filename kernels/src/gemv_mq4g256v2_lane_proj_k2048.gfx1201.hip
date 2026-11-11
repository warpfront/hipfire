// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Lane-batched multi-weight MQ4G256V2 GEMV for the A3B (`qwen3_5_moe`) decode
// projections, exact gfx1201, K = 2048 (8 groups of 136 B per row).
//
// One launch computes up to four weight segments (A_s: rows m_s, output
// y_s[b * m_s + row]) against NC activation columns x[b][2048]:
//
//   DeltaNet  QKVZA : wqkv 8192 + wz 4096 + w_beta 32 + w_alpha 32
//   FullAttn  QKV   : wq 8192 + wk 512 + wv 512            (m3 = 0)
//   MoE quartet     : router 256 + selector 1 + shared gate 512 + shared up 512
//
// Per (row, column) the arithmetic is the singleton `fused_qkvza_mq4g256v2` /
// `fused_qkv_mq4g256v2` one-row-per-wave32 kernel's, term for term. A row is
// owned by a 32-lane half of a wave64 (lane `l` = threadIdx.x & 31): lane `l`
// owns elements [g*256 + l*8, +8) of every group g; group g accumulates into
// acc[g & 3] (g and g + 4 in that order); the group sum is
// fma(u7,x7, ... fma(u2,x2, fma(u0,x0, u1*x1))) with u = fma(sc, q, zp) and
// the contraction spelled out exactly as the singleton compiles it; the row
// dot is (acc0 + acc1) + (acc2 + acc3) followed by the 32-lane tree
// a[i] + a[i + 16], then + [i + 8], [i + 4], [i + 2], [i + 1] (lane 0 holds
// the result; the singleton's __shfl_down ladder 16, 8, 4, 2, 1).
//
// What differs from the singleton is amortisation only: each weight row is
// streamed once for all columns, `u = fma(sc, q, zp)` is computed once per row
// instead of once per column (identical value), the activation columns are
// staged once per workgroup in LDS, and the two halves of a wave64 run two
// independent rows per instruction (wave64 issues a full-width FP32 op per
// clock where wave32 issues half).

#ifndef HIPFIRE_LANE_PROJ_ROWS
#define HIPFIRE_LANE_PROJ_ROWS 3          // rows per half-wave
#endif
#ifndef HIPFIRE_LANE_PROJ_WAVES
#define HIPFIRE_LANE_PROJ_WAVES 8         // wave64s per workgroup
#endif
#ifndef HIPFIRE_LANE_PROJ_OCC
#define HIPFIRE_LANE_PROJ_OCC 4
#endif
#ifndef HIPFIRE_LANE_PROJ_COLGROUP
#define HIPFIRE_LANE_PROJ_COLGROUP 2
#endif
#ifdef HIPFIRE_LANE_PROJ_KERNEL
// Lab tuning variant: entry names are pasted from the module name below.
#define LP_VARIANT_NAMES 1
#endif
#ifndef HIPFIRE_LANE_PROJ_KERNEL
#define HIPFIRE_LANE_PROJ_KERNEL gemv_mq4g256v2_lane_proj_k2048_gfx1201
#endif

#define LP_R HIPFIRE_LANE_PROJ_ROWS
#define LP_GROUPS 8
#define LP_K 2048
#define LP_THREADS (HIPFIRE_LANE_PROJ_WAVES * 64)
#define LP_FENCE() do { asm volatile("" ::: "memory"); __builtin_amdgcn_sched_barrier(0); } while (0)

#define LP_LD(p) (*(const unsigned int*)(p))

extern __shared__ float4 xs4[];   // [NC][2048] f32

#define LP_DPP_SHL(v, n) \
    __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x100 + (n), 0xF, 0xF, false))

// Sum of the 32 lanes of `a` in the singleton's tree order; lane 0 of each
// 32-lane half holds the result.
__device__ __forceinline__ float lp_reduce32(float a) {
    a += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(a), 0x401F));   // lane ^ 16
    a += LP_DPP_SHL(a, 8);
    a += LP_DPP_SHL(a, 4);
    a += LP_DPP_SHL(a, 2);
    a += LP_DPP_SHL(a, 1);
    return a;
}

template <int NC>
__device__ __forceinline__ void lane_proj_body(
    const char* __restrict__ A0, const char* __restrict__ A1,
    const char* __restrict__ A2, const char* __restrict__ A3,
    float* __restrict__ y0, float* __restrict__ y1,
    float* __restrict__ y2, float* __restrict__ y3,
    const float* __restrict__ x,   // [NC][2048]
    int m0, int m1, int m2, int m3
) {
    {
        const int n4 = NC * (LP_K / 4);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        // All global loads are issued before the first LDS store (one memory
        // round trip, not one per iteration). Transposed so a lane-contiguous
        // float4 read is conflict-free: the 16-byte chunk
        // c = b*512 + g*64 + l*2 + h (lane l, elements 4h..4h+3 of group g)
        // lands at b*512 + g*64 + h*32 + l.
        constexpr int ITERS = (NC * (LP_K / 4) + LP_THREADS - 1) / LP_THREADS;
        constexpr bool EXACT = (NC * (LP_K / 4)) % LP_THREADS == 0;
        float4 v[ITERS];
        #pragma unroll
        for (int it = 0; it < ITERS; it++) {
            const int i = threadIdx.x + it * LP_THREADS;
            if (EXACT || i < n4) v[it] = x4[i];
        }
        #pragma unroll
        for (int it = 0; it < ITERS; it++) {
            const int i = threadIdx.x + it * LP_THREADS;
            const int c = i & 511;
            if (EXACT || i < n4) xs4[(i & ~511) + (c & ~63) + ((c & 1) << 5) + ((c & 63) >> 1)] = v[it];
        }
    }
    __syncthreads();

    const int tid = threadIdx.x & 31;
    const int half = (threadIdx.x >> 5) & 1;
    const int wave = threadIdx.x >> 6;
    const int total_m = m0 + m1 + m2 + m3;
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;
    // Every half-wave owns one contiguous run of rows (equal up to the last),
    // walked in batches of LP_R rows; the result does not depend on the split.
    const int hw = (blockIdx.x * HIPFIRE_LANE_PROJ_WAVES + wave) * 2 + half;
    const int n_hw = gridDim.x * HIPFIRE_LANE_PROJ_WAVES * 2;
    const int rows_per = (total_m + n_hw - 1) / n_hw;
    const int row_start = hw * rows_per;
    const int row_end = min(total_m, row_start + rows_per);

    #pragma unroll 1
    for (int base = row_start; base < row_end; base += LP_R) {
        // The activation columns in LDS are loop-invariant: without this the
        // compiler hoists all of them out of the row loop.
        asm volatile("" ::: "memory");
        const char* rp[LP_R];
        float* yp[LP_R];
        int rowi[LP_R];
        int mseg[LP_R];
        bool valid[LP_R];
        #pragma unroll
        for (int r = 0; r < LP_R; r++) {
            int gid = base + r;
            valid[r] = gid < row_end;
            if (!valid[r]) gid = row_end - 1;
            const char* A; float* y; int row; int m;
            if (gid < m0) {
                A = A0; y = y0; row = gid; m = m0;
            } else if (gid < m0 + m1) {
                A = A1; y = y1; row = gid - m0; m = m1;
            } else if (gid < m0 + m1 + m2) {
                A = A2; y = y2; row = gid - (m0 + m1); m = m2;
            } else {
                A = A3; y = y3; row = gid - (m0 + m1 + m2); m = m3;
            }
            rp[r] = A + (long long)row * (LP_GROUPS * 136);
            yp[r] = y; rowi[r] = row; mseg[r] = m;
        }

        // Weights of group pair (p, p + 4) for every row, loaded one pair
        // ahead of the compute that consumes them.
        unsigned int hdr[2][LP_R][2];
        unsigned int pk[2][LP_R][2];
#define LP_LOAD_PAIR(slot, p_)                                                             \
        _Pragma("unroll")                                                                  \
        for (int r = 0; r < LP_R; r++) {                                                   \
            _Pragma("unroll")                                                              \
            for (int h = 0; h < 2; h++) {                                                  \
                const int g_ = (p_) + 4 * h;                                               \
                hdr[slot][r][h] = LP_LD(rp[r] + g_ * 136 + hoff);                          \
                pk[slot][r][h] = LP_LD(rp[r] + g_ * 136 + 8 + boff);                       \
            }                                                                              \
        }
        LP_LOAD_PAIR(0, 0)

        float acc[LP_R][4][NC];
        #pragma unroll
        for (int r = 0; r < LP_R; r++)
            #pragma unroll
            for (int p = 0; p < 4; p++)
                #pragma unroll
                for (int b = 0; b < NC; b++) acc[r][p][b] = 0.0f;

        #pragma unroll
        for (int p = 0; p < 4; p++) {
            const int cur = p & 1;
            if (p < 3) {
                LP_LOAD_PAIR(cur ^ 1, p + 1)
            }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int g = p + 4 * h;
                float u[LP_R][8];
                #pragma unroll
                for (int r = 0; r < LP_R; r++) {
                    const unsigned int hs = hdr[cur][r][h];
                    const unsigned int pv = pk[cur][r][h];
                    const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                    const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
                    u[r][0] = __builtin_fmaf(sc, (float)((pv) & 0xFu), zp);
                    u[r][1] = __builtin_fmaf(sc, (float)(((pv) >> 4) & 0xFu), zp);
                    u[r][2] = __builtin_fmaf(sc, (float)(((pv) >> 8) & 0xFu), zp);
                    u[r][3] = __builtin_fmaf(sc, (float)(((pv) >> 12) & 0xFu), zp);
                    u[r][4] = __builtin_fmaf(sc, (float)(((pv) >> 16) & 0xFu), zp);
                    u[r][5] = __builtin_fmaf(sc, (float)(((pv) >> 20) & 0xFu), zp);
                    u[r][6] = __builtin_fmaf(sc, (float)(((pv) >> 24) & 0xFu), zp);
                    u[r][7] = __builtin_fmaf(sc, (float)(((pv) >> 28) & 0xFu), zp);
                }
                #pragma unroll
                for (int b = 0; b < NC; b++) {
                    const float4 xa = xs4[b * 512 + g * 64 + tid];
                    const float4 xb = xs4[b * 512 + g * 64 + 32 + tid];
                    #pragma unroll
                    for (int r = 0; r < LP_R; r++) {
                        // The singleton's per-group sum, contraction spelled out:
                        // t0 + t1 -> fma(u0, x0, u1 * x1), then each further term
                        // as fma(u_k, x_k, sum); the group sum is added to acc[g & 3].
                        float t1 = u[r][1] * xa.y;
                        asm("" : "+v"(t1));   // not volatile: chains may interleave
                        float sum = __builtin_fmaf(u[r][0], xa.x, t1);
                        sum = __builtin_fmaf(u[r][2], xa.z, sum);
                        sum = __builtin_fmaf(u[r][3], xa.w, sum);
                        sum = __builtin_fmaf(u[r][4], xb.x, sum);
                        sum = __builtin_fmaf(u[r][5], xb.y, sum);
                        sum = __builtin_fmaf(u[r][6], xb.z, sum);
                        sum = __builtin_fmaf(u[r][7], xb.w, sum);
                        acc[r][p][b] += sum;
                        // Pin the update here: without it the compiler sinks the
                        // pure arithmetic to the reduction and keeps every x and
                        // u value live until then (spills at 3+ columns).
                        asm volatile("" : "+v"(acc[r][p][b]));
                    }
                    if ((b + 1) % HIPFIRE_LANE_PROJ_COLGROUP == 0) LP_FENCE();
                }
            }
            LP_FENCE();
        }
#undef LP_LOAD_PAIR

        #pragma unroll
        for (int r = 0; r < LP_R; r++) {
            #pragma unroll
            for (int b = 0; b < NC; b++) {
                const float a = lp_reduce32((acc[r][0][b] + acc[r][1][b]) + (acc[r][2][b] + acc[r][3][b]));
                if (tid == 0 && valid[r]) yp[r][(long long)b * mseg[r] + rowi[r]] = a;
            }
        }
    }
}

#ifdef LP_VARIANT_NAMES
#define LP_CAT3_(a, b, c) a##b##c
#define LP_CAT3(a, b, c) LP_CAT3_(a, b, c)

#define LP_ENTRY(NCV)                                                                   \
__launch_bounds__(LP_THREADS)                                                           \
__attribute__((amdgpu_waves_per_eu(HIPFIRE_LANE_PROJ_OCC, HIPFIRE_LANE_PROJ_OCC)))     \
extern "C" __global__ void LP_CAT3(HIPFIRE_LANE_PROJ_KERNEL, _n, NCV)(                  \
    const char* __restrict__ A0, const char* __restrict__ A1,                           \
    const char* __restrict__ A2, const char* __restrict__ A3,                           \
    float* __restrict__ y0, float* __restrict__ y1,                                     \
    float* __restrict__ y2, float* __restrict__ y3,                                     \
    const float* __restrict__ x,                                                        \
    int m0, int m1, int m2, int m3)                                                     \
{                                                                                       \
    lane_proj_body<NCV>(A0, A1, A2, A3, y0, y1, y2, y3, x, m0, m1, m2, m3);             \
}

LP_ENTRY(1)
LP_ENTRY(2)
LP_ENTRY(3)
LP_ENTRY(4)
LP_ENTRY(5)
LP_ENTRY(6)
LP_ENTRY(7)
LP_ENTRY(8)
#else
// Product module: literal entry names (the registry census reads them).
#define LP_ENTRY(NAME, NCV)                                                             \
__launch_bounds__(LP_THREADS)                                                           \
__attribute__((amdgpu_waves_per_eu(HIPFIRE_LANE_PROJ_OCC, HIPFIRE_LANE_PROJ_OCC)))     \
extern "C" __global__ void NAME(                                                        \
    const char* __restrict__ A0, const char* __restrict__ A1,                           \
    const char* __restrict__ A2, const char* __restrict__ A3,                           \
    float* __restrict__ y0, float* __restrict__ y1,                                     \
    float* __restrict__ y2, float* __restrict__ y3,                                     \
    const float* __restrict__ x,                                                        \
    int m0, int m1, int m2, int m3)                                                     \
{                                                                                       \
    lane_proj_body<NCV>(A0, A1, A2, A3, y0, y1, y2, y3, x, m0, m1, m2, m3);             \
}

LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n1, 1)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n2, 2)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n3, 3)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n4, 4)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n5, 5)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n6, 6)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n7, 7)
LP_ENTRY(gemv_mq4g256v2_lane_proj_k2048_gfx1201_n8, 8)
#endif
