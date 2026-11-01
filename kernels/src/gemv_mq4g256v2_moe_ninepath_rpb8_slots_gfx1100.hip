// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Multi-slot gfx1100 sister of `gemv_mq4g256v2_moe_ninepath_rpb8_gfx1100`.
//
// Grid is [(down_m/8) * N, 1, 1], N <= 4. The token is folded into grid.x
// (`blockIdx.x = token * tiles + tile`). Each CTA is the single-token RPB=8
// kernel: stage that token's 8x512 activation, warp-per-krank dot, then the
// ascending-krank weighted fold into the residual.

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_mq4g256v2_moe_ninepath_rpb8_slots_gfx1100(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ topk_weights,
    const float* __restrict__ act_all,
    float* __restrict__ out_all,
    int down_m, int down_k, int n_experts, int n_tokens
) {
    constexpr int RPB = 8;
    __shared__ float act_s[8][512];
    __shared__ float part[8][RPB];
    const int tid = threadIdx.x;
    const int tiles = down_m / RPB;
    const int packed = blockIdx.x;
    const int tok = (tiles > 0) ? packed / tiles : 0;
    const int tile = packed - tok * tiles;
    // Every lane of the CTA takes the same exit, so this does not split the
    // barrier below. down_k is the sealed 512-wide shape.
    if (tok >= n_tokens || tiles <= 0 || down_k != 512) return;
    const float* __restrict__ act = act_all + (long long)tok * 8 * 512;
    const int* __restrict__ topk = topk_indices + (long long)tok * 8;
    const float* __restrict__ weights = topk_weights + (long long)tok * 8;
    float* __restrict__ out = out_all + (long long)tok * down_m;

    for (int i = tid; i < 8 * 512; i += 256)
        (&act_s[0][0])[i] = act[i];
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int expert_id = topk[warp];
    const bool expert_valid = (unsigned)expert_id < (unsigned)n_experts;
    const char* __restrict__ A =
        expert_valid ? reinterpret_cast<const char*>(expert_ptrs[expert_id]) : nullptr;
    const int r0 = tile * RPB;
    const int boff = lane * 4;
    const int hoff = (lane < 16) ? 0 : 4;
    const float* __restrict__ xa = act_s[warp];

    #define DOG_X8(gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * (x0) \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (x1) \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (x2) \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (x3) \
             + (sc * (float)(((pk) >> 16)  & 0xFu) + zp) * (x4) \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (x5) \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (x6) \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (x7); \
    } while (0)

    for (int r = 0; r < RPB; r++) {
        if (!expert_valid) {
            if (lane == 0) part[warp][r] = 0.0f;
            continue;
        }
        const int row = r0 + r;
        const char* row_ptr = A + (long long)row * 2 * 136;
        const float4 xv0a = *reinterpret_cast<const float4*>(xa + lane * 8);
        const float4 xv0b = *reinterpret_cast<const float4*>(xa + lane * 8 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(xa + lane * 8 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(xa + lane * 8 + 260);
        float acc0 = 0.0f, acc1 = 0.0f;
        DOG_X8(row_ptr + 0 * 136, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(row_ptr + 1 * 136, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        float acc = acc0 + acc1;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (lane == 0) part[warp][r] = acc;
    }
    __syncthreads();

    if (warp == 0 && lane < RPB) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++)
            a += weights[w] * part[w][lane];
        out[r0 + lane] += a;
    }
    #undef DOG_X8
}
