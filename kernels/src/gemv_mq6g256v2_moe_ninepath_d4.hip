// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt47 f16 header

// MQ6G256V2 (qt=47) sister of `gemv_mq4g256v2_moe_ninepath_d4`.
//
// Transcribed from that kernel: the LDS staging, warp-per-krank assignment,
// per-row dot, shuffle tree and deterministic weighted fold are identical line
// for line. The ONLY differences are the group stride and the payload decode:
//   - group stride 200 B (vs qt44's 136 B)
//   - dual-half f16 header at bytes 0..8 (same V2 layout as qt44; NOT the
//     V1/HFQ6 f32 scale+zero header — those identities must never collapse)
//   - 192 B legacy 6-bit packed payload at +8 (4 weights per 3 bytes), same
//     unpack as gemv_mq6g256v2 / gemv_hfq6g256_moe_down_k8_indexed_batched_expanded
//
// Affine-grid selection is LANE-CONSTANT: lane reads the u32 at
// `boff = lane*6`, covering weights [lane*8, lane*8+8) — wholly inside one
// 128-weight half, the same half in every group. So the header offset hoists
// out of the row loop:
//   lane  0..15 -> weights   0..127 -> grid 0 -> header at gp+0
//   lane 16..31 -> weights 128..255 -> grid 1 -> header at gp+4
// qt47 therefore reads ONE header u32 where a naive V1 port would read two
// f32 fields and ignore the dual-half split.
//
// Nine-path fused MoE down + weighted combine (routed k=8, decode T=1).
//
// Replaces gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded +
// moe_down_combine_k8_batched: the expanded [8 × down_m] intermediate never
// leaves LDS. One 256-thread CTA stages the rotated activation for all 8
// experts once; warp w owns expert topk_indices[w] and dots its down rows;
// after a barrier, warp 0 folds the 8 partials with topk_weights in the
// exact k=0..7 order of moe_down_combine_k8_batched and accumulates into
// the residual (single owner per row — no atomics, deterministic under
// graph replay, matching the atomic-free contract of the expanded+combine
// pair).
//
// BYTE-EXACT with the qt47 expanded+combine pair at down_k=512: per-row dot uses
// the same two live pairwise slots (group0 → acc0, group1 → acc1;
// acc = acc0 + acc1) and the same shuffle-down tree; the fold applies
// weights in ascending krank order then += into out, same as the combine.
//
// Requirements: down_k == 512 (2 groups), down_m % RPB == 0,
// blockDim == 256 (8 warps × 32).

__launch_bounds__(256, 4)
extern "C" __global__ void gemv_mq6g256v2_moe_ninepath_d4(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [8]
    const float* __restrict__ topk_weights,              // [8]
    const float* __restrict__ act,                       // [8 × 512] rotated silu out
    float* __restrict__ out,                             // [down_m] accumulated
    int down_m, int down_k, int n_experts
) {
    constexpr int RPB = 16;
    __shared__ float act_s[8][512];
    __shared__ float part[8][RPB];
    const int tid = threadIdx.x;
    for (int i = tid; i < 8 * 512; i += 256)
        (&act_s[0][0])[i] = act[i];
    __syncthreads();

    const int warp = tid >> 5;   // 0..7 = krank
    const int lane = tid & 31;
    const int expert_id = topk_indices[warp];
    const bool expert_valid = (unsigned)expert_id < (unsigned)n_experts;
    const char* __restrict__ A =
        expert_valid ? reinterpret_cast<const char*>(expert_ptrs[expert_id]) : nullptr;
    const int r0 = blockIdx.x * RPB;
    const int boff = lane * 6;
    // Lane-constant affine-grid header offset -- see the note above.
    const int hoff = (lane < 16) ? 0 : 4;
    const float* __restrict__ xa = act_s[warp];

    #define DOG_X8(gp, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * (x0) \
             + (sc * (float)q1 + zp) * (x1) \
             + (sc * (float)q2 + zp) * (x2) \
             + (sc * (float)q3 + zp) * (x3) \
             + (sc * (float)q4 + zp) * (x4) \
             + (sc * (float)q5 + zp) * (x5) \
             + (sc * (float)q6 + zp) * (x6) \
             + (sc * (float)q7 + zp) * (x7); \
    } while (0)

    for (int r = 0; r < RPB; r++) {
        if (!expert_valid) {
            if (lane == 0) part[warp][r] = 0.0f;
            continue;
        }
        const int row = r0 + r;
        const char* row_ptr = A + (long long)row * 2 * 200;

        const float4 xv0a = *reinterpret_cast<const float4*>(xa + lane * 8);
        const float4 xv0b = *reinterpret_cast<const float4*>(xa + lane * 8 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(xa + lane * 8 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(xa + lane * 8 + 260);

        float acc0 = 0.0f, acc1 = 0.0f;
        // Group 0 → pairwise slot 0 (matches baseline tail g=0 → acc0).
        DOG_X8(row_ptr + 0 * 200, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        // Group 1 → pairwise slot 1 (matches baseline tail g=1 → acc1).
        DOG_X8(row_ptr + 1 * 200, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);

        float acc = acc0 + acc1;
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (lane == 0) part[warp][r] = acc;
    }
    __syncthreads();

    // Deterministic weighted fold: ascending krank, single owner per row.
    if (warp == 0 && lane < RPB) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++)
            a += topk_weights[w] * part[w][lane];
        out[r0 + lane] += a;
    }

    #undef DOG_X8
}
