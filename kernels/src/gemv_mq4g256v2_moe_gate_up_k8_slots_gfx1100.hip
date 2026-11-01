// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Multi-slot gfx1100 sister of
// `gemv_mq4g256v2_moe_gate_up_k8_indexed_k2048_nolds_gfx1100`.
//
// Grid is [MI * N, 8, 1] with N <= 4. The token is folded into grid.x
// (`blockIdx.x = token * MI + row`) because the retained PM4 packet path
// replays a non-unit grid.y/grid.z unreliably. The body is the nolds kernel:
// no LDS, no barrier, same quad DOG and the same shuffle-down.

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_mq4g256v2_moe_gate_up_k8_slots_gfx1100(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_all,
    float* __restrict__ y_gate_all,
    float* __restrict__ y_up_all,
    int M, int K, int n_experts, int n_tokens
) {
    const int mi = M >> 1;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;
    const int packed = blockIdx.x;
    const int tok = (mi > 0) ? packed / mi : 0;
    const int row = packed - tok * mi;
    if (tok >= n_tokens || row >= mi || row < 0) return;

    const float* __restrict__ xsrc = x_all + (long long)tok * K;
    const int expert_id = topk_indices[(long long)tok * 8 + krank];
    float* __restrict__ y_gate = y_gate_all + (long long)tok * 8 * mi;
    float* __restrict__ y_up = y_up_all + (long long)tok * 8 * mi;
    if ((unsigned)expert_id >= (unsigned)n_experts) {
        if (tid == 0) {
            y_gate[krank * mi + row] = 0.0f;
            y_up[krank * mi + row] = 0.0f;
        }
        return;
    }
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define LOAD_X8_INTO(b, pfx) \
        const float4 pfx##v0 = *reinterpret_cast<const float4*>(xsrc + (b)); \
        const float4 pfx##v1 = *reinterpret_cast<const float4*>(xsrc + (b) + 4); \
        const float pfx##0 = pfx##v0.x, pfx##1 = pfx##v0.y, pfx##2 = pfx##v0.z, pfx##3 = pfx##v0.w; \
        const float pfx##4 = pfx##v1.x, pfx##5 = pfx##v1.y, pfx##6 = pfx##v1.z, pfx##7 = pfx##v1.w

    #define DOG_X8(gp, a, pfx) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * pfx##0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * pfx##1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * pfx##2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * pfx##3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * pfx##4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * pfx##5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * pfx##6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * pfx##7; \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;
        LOAD_X8_INTO(base0, xa);
        LOAD_X8_INTO(base0 + 256, xb);
        LOAD_X8_INTO(base0 + 512, xc);
        LOAD_X8_INTO(base0 + 768, xd);
        DOG_X8(gate_ptr + g * 136, gacc0, xa);
        DOG_X8(up_ptr + g * 136, uacc0, xa);
        DOG_X8(gate_ptr + (g + 1) * 136, gacc1, xb);
        DOG_X8(up_ptr + (g + 1) * 136, uacc1, xb);
        DOG_X8(gate_ptr + (g + 2) * 136, gacc2, xc);
        DOG_X8(up_ptr + (g + 2) * 136, uacc2, xc);
        DOG_X8(gate_ptr + (g + 3) * 136, gacc3, xd);
        DOG_X8(up_ptr + (g + 3) * 136, uacc3, xd);
    }

    #define LOAD_X8(b) \
        const float4 xv0 = *reinterpret_cast<const float4*>(xsrc + (b)); \
        const float4 xv1 = *reinterpret_cast<const float4*>(xsrc + (b) + 4); \
        const float x0 = xv0.x, x1 = xv0.y, x2 = xv0.z, x3 = xv0.w; \
        const float x4 = xv1.x, x5 = xv1.y, x6 = xv1.z, x7 = xv1.w

    #define DOG_X8_TAIL(gp, a) do { \
        unsigned int h = *(const unsigned int*)((gp) + hoff); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
        (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
    } while (0)

    if (tail >= 1) {
        const int g = quads << 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc0);
        DOG_X8_TAIL(up_ptr + g * 136, uacc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc1);
        DOG_X8_TAIL(up_ptr + g * 136, uacc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        int base = g * 256 + tid * 8;
        LOAD_X8(base);
        DOG_X8_TAIL(gate_ptr + g * 136, gacc2);
        DOG_X8_TAIL(up_ptr + g * 136, uacc2);
    }
    #undef DOG_X8_TAIL
    #undef LOAD_X8
    #undef DOG_X8
    #undef LOAD_X8_INTO

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        gacc += __shfl_down(gacc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        uacc += __shfl_down(uacc, offset);

    if (tid == 0) {
        y_gate[krank * mi + row] = gacc;
        y_up[krank * mi + row] = uacc;
    }
}
