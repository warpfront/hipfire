// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Nine-path fused MoE activation + down + weighted combine for MQ4G128V2
// (qt=53) routed down weights (routed k=8, decode T=1, down_k=512).
//
// Replaces the four-launch sequence
//   silu_mul_f32 -> mq_rotate_x_128_v2 ->
//   gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded ->
//   moe_down_combine_k8_batched
// with one launch. One 256-thread CTA owns RPB output rows for all 8 routed
// experts: warp w is routed slot w (expert topk_indices[w]).
//
//   1. Activation (per warp, in registers). Lane l forms silu(gate)*up for
//      elements 4l..4l+3 of each of the four 128-wide groups of slot w and
//      runs the qt53 FWHT128 (sign1 -> radix-2 butterflies -> 1/sqrt(128) ->
//      sign2) exactly as mq_rotate_x_128_v2 does. The lane that produces
//      elements 4l..4l+3 is the lane that dots them, so the rotated
//      activation never leaves the register file (no LDS, no barrier).
//   2. Down. For each of the RPB rows the warp dots its expert's 68-byte-group
//      row (4 groups) against the rotated activation; the 32 lane partials of
//      every row are parked in LDS.
//   3. Reduce. One thread per (warp, row) folds the 32 lane partials in the
//      shfl_down tree order (16, 8, 4, 2, 1) of the expanded kernel.
//   4. Fold. Warp 0 combines the 8 slot partials with topk_weights in
//      ascending slot order and accumulates into `out` (single owner per row,
//      no atomics).
//
// BYTE-EXACT with the four-kernel sequence. The compiler-contracted FMA
// orientation of each replaced kernel is pinned explicitly (verified against
// the gfx1201 ISA of the originals, see moe_down_mq4g128v2_ninepath_hw):
//   * silu_mul_f32:     round(silu_div * up), precise division and expf.
//   * mq_rotate_x_128_v2 first level: p = x1*s1 (rounded); v0 = fma(x0,s0,p);
//     v1 = fma(x0,s0,-p); likewise (x2,x3). Remaining levels are plain adds.
//   * expanded down:    d_i = fma(scale, q_i, zero); per group
//     s = d1*x1 (rounded); s = fma(d0,x0,s); s = fma(d2,x2,s);
//     s = fma(d3,x3,s); lane acc += s over groups 0..3 ascending.
//   * combine:          a = 0; a = fma(w[k], part[k], a) for k = 0..7;
//     out += a.
//
// Requirements: down_k == 512 (4 groups, 272 B rows), down_m % RPB == 0,
// blockDim == 256 (8 warps x 32 lanes, wave32), top-k == 8.
// Mirrors gemv_mq4g256v2_moe_ninepath_d4's frozen launch contract: grid
// down_m / RPB, block 256, no dynamic LDS.

#ifndef HIPFIRE_MQ4G128V2_NINEPATH_KERNEL
#define HIPFIRE_MQ4G128V2_NINEPATH_KERNEL gemv_mq4g128v2_moe_ninepath_d4
#endif
#ifndef HIPFIRE_MQ4G128V2_NINEPATH_RPB
#define HIPFIRE_MQ4G128V2_NINEPATH_RPB 16
#endif

static __device__ __forceinline__ const char* mq4g128v2_np_uniform_ptr(const char* p) {
    const unsigned long long v = (unsigned long long)p;
    const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)v);
    const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(v >> 32));
    return (const char*)(((unsigned long long)hi << 32) | lo);
}

__launch_bounds__(256, 4)
extern "C" __global__ void HIPFIRE_MQ4G128V2_NINEPATH_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [8]
    const float* __restrict__ topk_weights,              // [8]
    const float* __restrict__ gate,                      // [8 x 512] gate rows
    const float* __restrict__ up,                        // [8 x 512] up rows
    const float* __restrict__ signs1,                    // [128] qt53 sign1 (seed 43)
    const float* __restrict__ signs2,                    // [128] qt53 sign2 (seed 1043)
    float* __restrict__ out,                             // [down_m] accumulated
    int down_m, int down_k, int n_experts
) {
    constexpr int RPB = HIPFIRE_MQ4G128V2_NINEPATH_RPB;
    constexpr int ROW_BYTES = 4 * 68;
    // lane partials: [warp][row][lane], row pitch 33 so the strided reduce
    // reads are bank-conflict free.
    __shared__ float lane_part[8 * RPB * 33];
    __shared__ float part[8][RPB];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = __builtin_amdgcn_readfirstlane(tid >> 5); // 0..7 = krank
    const int d0 = lane * 4;

    // ---- 1. activation: silu(gate)*up -> FWHT128 (qt53), in registers ----
    float xs[4][4];
    {
        const float4 s1v = *reinterpret_cast<const float4*>(signs1 + d0);
        const float4 s2v = *reinterpret_cast<const float4*>(signs2 + d0);
        const float sg1[4] = {s1v.x, s1v.y, s1v.z, s1v.w};
        const float sg2[4] = {s2v.x, s2v.y, s2v.z, s2v.w};
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            const int base = warp * 512 + g * 128 + d0;
            const float4 gv = *reinterpret_cast<const float4*>(gate + base);
            const float4 uv = *reinterpret_cast<const float4*>(up + base);
            const float gg[4] = {gv.x, gv.y, gv.z, gv.w};
            const float uu[4] = {uv.x, uv.y, uv.z, uv.w};
            float a[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const float v = gg[i];
                a[i] = __fmul_rn(v / (1.0f + expf(-v)), uu[i]);
            }
            // sign1 + radix-4 in-lane butterflies (product x1*s1 rounded,
            // x0*s0 fused into both outputs; likewise x3*s3 / x2*s2).
            const float p1 = __fmul_rn(a[1], sg1[1]);
            float v0 = __fmaf_rn(a[0], sg1[0], p1);
            float v1 = __fmaf_rn(a[0], sg1[0], -p1);
            const float p3 = __fmul_rn(a[3], sg1[3]);
            float v2 = __fmaf_rn(a[2], sg1[2], p3);
            float v3 = __fmaf_rn(a[2], sg1[2], -p3);
            float t;
            t = v0; v0 = v0 + v2; v2 = t - v2;
            t = v1; v1 = v1 + v3; v3 = t - v3;
#define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define HBFLY4(pat, str) \
    HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); HBFLY(v3, pat, str)
            HBFLY4(0x041F, 1);
            HBFLY4(0x081F, 2);
            HBFLY4(0x101F, 4);
            HBFLY4(0x201F, 8);
            HBFLY4(0x401F, 16);
#undef HBFLY4
#undef HBFLY
            const float scale = 0.0883883476f;
            xs[g][0] = __fmul_rn(__fmul_rn(v0, scale), sg2[0]);
            xs[g][1] = __fmul_rn(__fmul_rn(v1, scale), sg2[1]);
            xs[g][2] = __fmul_rn(__fmul_rn(v2, scale), sg2[2]);
            xs[g][3] = __fmul_rn(__fmul_rn(v3, scale), sg2[3]);
        }
    }

    // ---- 2. down: RPB rows of this warp's expert ----
    const int expert_id = __builtin_amdgcn_readfirstlane(topk_indices[warp]);
    const bool expert_valid = (unsigned)expert_id < (unsigned)n_experts;
    const char* __restrict__ A = expert_valid
        ? mq4g128v2_np_uniform_ptr(reinterpret_cast<const char*>(expert_ptrs[expert_id]))
        : nullptr;
    const int r0 = blockIdx.x * RPB;
    float* __restrict__ lp = lane_part + (warp * RPB) * 33;

    #pragma unroll 4
    for (int r = 0; r < RPB; r++) {
        float acc = 0.0f;
        if (expert_valid) {
            const char* row_ptr = A + (long long)(r0 + r) * ROW_BYTES;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const char* gp = row_ptr + g * 68;
                const unsigned int hdr = *reinterpret_cast<const unsigned int*>(gp);
                const float scale = __half2float(__ushort_as_half((unsigned short)(hdr & 0xFFFFu)));
                const float zero = __half2float(__ushort_as_half((unsigned short)(hdr >> 16)));
                const unsigned int pk = *reinterpret_cast<const unsigned short*>(gp + 4 + lane * 2);
                const float d0f = __fmaf_rn(scale, (float)(pk & 0xFu), zero);
                const float d1f = __fmaf_rn(scale, (float)((pk >> 4) & 0xFu), zero);
                const float d2f = __fmaf_rn(scale, (float)((pk >> 8) & 0xFu), zero);
                const float d3f = __fmaf_rn(scale, (float)((pk >> 12) & 0xFu), zero);
                float s = __fmul_rn(d1f, xs[g][1]);
                s = __fmaf_rn(d0f, xs[g][0], s);
                s = __fmaf_rn(d2f, xs[g][2], s);
                s = __fmaf_rn(d3f, xs[g][3], s);
                acc += s;
            }
        }
        lp[r * 33 + lane] = acc;
    }
    __syncthreads();

    // ---- 3. per-(warp,row) lane reduction, shfl_down tree order ----
    for (int t = tid; t < 8 * RPB; t += 256) {
        const float* __restrict__ a = lane_part + t * 33;
        float s1[16];
        #pragma unroll
        for (int i = 0; i < 16; i++) s1[i] = a[i] + a[i + 16];
        float s2[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) s2[i] = s1[i] + s1[i + 8];
        float s3[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) s3[i] = s2[i] + s2[i + 4];
        const float s4a = s3[0] + s3[2];
        const float s4b = s3[1] + s3[3];
        (&part[0][0])[t] = s4a + s4b;
    }
    __syncthreads();

    // ---- 4. deterministic weighted fold: ascending slot, single owner ----
    if (tid < RPB) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++)
            a = __fmaf_rn(topk_weights[w], part[w][tid], a);
        out[r0 + tid] += a;
    }
}
