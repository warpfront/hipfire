// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Index-aware fused MoE gate_up GEMV (Phase 2b/2c) — reads expert IDs from
// a device-side topk_indices buffer and the per-expert weight base from a
// device-side expert pointer table. No CPU kernargs depend on the routing
// result, so the whole kernel replays cleanly under hipGraph capture.
//
// Expert ptrs layout: `expert_ptrs` is a raw [n_exp] array of `const char*`
// weight-base pointers (packed as `unsigned long long` slots, 8B each).
// Built once at load time from `ExpertWeights.gate_up.buf.as_ptr()`. The
// kernel reads `expert_ptrs[topk_indices[krank]]` as its weight base.
//
// Output layout matches the non-indexed variant: split into y_gate[K_TOP×MI]
// and y_up[K_TOP×MI], feeding the existing batched silu_mul_rotate.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_hfq4g256_moe_gate_up_k8_indexed(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int mi = M >> 1;
    const int row = blockIdx.x;
    if (row >= mi) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    // Read expert_id from device and index into the pointer table.
    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8(b) \
        const float x0 = x[(b) + 0]; \
        const float x1 = x[(b) + 1]; \
        const float x2 = x[(b) + 2]; \
        const float x3 = x[(b) + 3]; \
        const float x4 = x[(b) + 4]; \
        const float x5 = x[(b) + 5]; \
        const float x6 = x[(b) + 6]; \
        const float x7 = x[(b) + 7]; \
        const float xs = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7))

    #define DOG_X8(gp, a) do { \
        float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
        float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        const float dot = (float)((pk) & 0xFu) * x0 \
             + (float)(((pk) >> 4) & 0xFu) * x1 \
             + (float)(((pk) >> 8) & 0xFu) * x2 \
             + (float)(((pk) >> 12) & 0xFu) * x3 \
             + (float)(((pk) >> 16) & 0xFu) * x4 \
             + (float)(((pk) >> 20) & 0xFu) * x5 \
             + (float)(((pk) >> 24) & 0xFu) * x6 \
             + (float)(((pk) >> 28) & 0xFu) * x7; \
        (a) += sc * dot + zp * xs; \
    } while (0)

    #define DOG_PAIR_X8(gp_g, gp_u, ga, ua) do { \
        const float gsc = __builtin_bit_cast(float, *(const unsigned int*)(gp_g)); \
        const float gzp = __builtin_bit_cast(float, *(const unsigned int*)((gp_g) + 4)); \
        const unsigned int gpk = *(const unsigned int*)((gp_g) + 8 + boff); \
        const float usc = __builtin_bit_cast(float, *(const unsigned int*)(gp_u)); \
        const float uzp = __builtin_bit_cast(float, *(const unsigned int*)((gp_u) + 4)); \
        const unsigned int upk = *(const unsigned int*)((gp_u) + 8 + boff); \
        float gdot = (float)(gpk & 0xFu) * x0; \
        float udot = (float)(upk & 0xFu) * x0; \
        gdot += (float)((gpk >> 4) & 0xFu) * x1; \
        udot += (float)((upk >> 4) & 0xFu) * x1; \
        gdot += (float)((gpk >> 8) & 0xFu) * x2; \
        udot += (float)((upk >> 8) & 0xFu) * x2; \
        gdot += (float)((gpk >> 12) & 0xFu) * x3; \
        udot += (float)((upk >> 12) & 0xFu) * x3; \
        gdot += (float)((gpk >> 16) & 0xFu) * x4; \
        udot += (float)((upk >> 16) & 0xFu) * x4; \
        gdot += (float)((gpk >> 20) & 0xFu) * x5; \
        udot += (float)((upk >> 20) & 0xFu) * x5; \
        gdot += (float)((gpk >> 24) & 0xFu) * x6; \
        udot += (float)((upk >> 24) & 0xFu) * x6; \
        gdot += (float)((gpk >> 28) & 0xFu) * x7; \
        udot += (float)((upk >> 28) & 0xFu) * x7; \
        (ga) += gsc * gdot + gzp * xs; \
        (ua) += usc * udot + uzp * xs; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        {
            LOAD_X8(base0);
            DOG_PAIR_X8(gate_ptr + g * 136, up_ptr + g * 136, gacc0, uacc0);
        }
        {
            LOAD_X8(base0 + 256);
            DOG_PAIR_X8(gate_ptr + (g + 1) * 136, up_ptr + (g + 1) * 136, gacc1, uacc1);
        }
        {
            LOAD_X8(base0 + 512);
            DOG_PAIR_X8(gate_ptr + (g + 2) * 136, up_ptr + (g + 2) * 136, gacc2, uacc2);
        }
        {
            LOAD_X8(base0 + 768);
            DOG_PAIR_X8(gate_ptr + (g + 3) * 136, up_ptr + (g + 3) * 136, gacc3, uacc3);
        }
    }

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_PAIR_X8(gate_ptr + g * 136, up_ptr + g * 136, gacc0, uacc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_PAIR_X8(gate_ptr + g * 136, up_ptr + g * 136, gacc1, uacc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_PAIR_X8(gate_ptr + g * 136, up_ptr + g * 136, gacc2, uacc2);
    }
    #undef DOG_PAIR_X8
    #undef DOG_X8
    #undef LOAD_X8

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    gacc += __shfl_down(gacc, 16);
    uacc += __shfl_down(uacc, 16);
    gacc += __shfl_down(gacc, 8);
    uacc += __shfl_down(uacc, 8);
    gacc += __shfl_down(gacc, 4);
    uacc += __shfl_down(uacc, 4);
    gacc += __shfl_down(gacc, 2);
    uacc += __shfl_down(uacc, 2);
    gacc += __shfl_down(gacc, 1);
    uacc += __shfl_down(uacc, 1);

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }
}
