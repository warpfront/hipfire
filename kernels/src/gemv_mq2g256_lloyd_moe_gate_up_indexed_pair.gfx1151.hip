// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Kaden Schutt <kaden@hipfire.dev>
//
// gfx1151 DeepSeek-V4 MQ2-Lloyd gate/up decode candidate. One wave computes
// matching gate and up rows for one routed expert, reusing each activation
// load across both projections. The baseline launches one wave per row across
// a combined [gate; up] matrix.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__launch_bounds__(32, 12)
extern "C" __global__ void
gemv_mq2g256_lloyd_moe_gate_up_k8_indexed_pair_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x_rot,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K
) {
    const int mi = M >> 1;
    const int row = blockIdx.x;
    if (row >= mi) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const long long row_bytes = (long long)groups_per_row * 72;
    const char* gate_ptr = A + (long long)row * row_bytes;
    const char* up_ptr = A + (long long)(row + mi) * row_bytes;
    const int boff = tid * 2;

    // [gate four groups][up four groups], four fp32 entries per codebook.
    __shared__ float cb_lds[32];

    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;
    const int quads = groups_per_row >> 2;

    #define DOT8_LDS(pk, lds, a, xa, xb) do {                           \
        (a) += cb_lds[(lds) + ( (pk)        & 3u)] * (xa).x             \
             + cb_lds[(lds) + (((pk) >>  2) & 3u)] * (xa).y             \
             + cb_lds[(lds) + (((pk) >>  4) & 3u)] * (xa).z             \
             + cb_lds[(lds) + (((pk) >>  6) & 3u)] * (xa).w             \
             + cb_lds[(lds) + (((pk) >>  8) & 3u)] * (xb).x             \
             + cb_lds[(lds) + (((pk) >> 10) & 3u)] * (xb).y             \
             + cb_lds[(lds) + (((pk) >> 12) & 3u)] * (xb).z             \
             + cb_lds[(lds) + (((pk) >> 14) & 3u)] * (xb).w;            \
    } while (0)

    #define PROCESS_PAIR(i, chain) do {                                  \
        const char* gg = gate_quad + (i) * 72;                            \
        const char* uu = up_quad + (i) * 72;                              \
        const unsigned char* gd =                                         \
            reinterpret_cast<const unsigned char*>(gg + 8) + boff;        \
        const unsigned char* ud =                                         \
            reinterpret_cast<const unsigned char*>(uu + 8) + boff;        \
        const unsigned int gpk =                                          \
            (unsigned int)gd[0] | ((unsigned int)gd[1] << 8);             \
        const unsigned int upk =                                          \
            (unsigned int)ud[0] | ((unsigned int)ud[1] << 8);             \
        const int base = (g + (i)) * 256 + tid * 8;                       \
        const float4 xa = *reinterpret_cast<const float4*>(x_rot + base); \
        const float4 xb =                                                 \
            *reinterpret_cast<const float4*>(x_rot + base + 4);           \
        DOT8_LDS(gpk, (i) * 4, gacc##chain, xa, xb);                      \
        DOT8_LDS(upk, 16 + (i) * 4, uacc##chain, xa, xb);                 \
    } while (0)

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gate_quad = gate_ptr + (long long)g * 72;
        const char* up_quad = up_ptr + (long long)g * 72;

        const int projection = tid >> 4;
        const int local = tid & 15;
        const char* quad = projection ? up_quad : gate_quad;
        const __half* cb_h = reinterpret_cast<const __half*>(
            quad + (long long)(local >> 2) * 72);
        cb_lds[tid] = __half2float(cb_h[local & 3]);
        __syncthreads();

        PROCESS_PAIR(0, 0);
        PROCESS_PAIR(1, 1);
        PROCESS_PAIR(2, 2);
        PROCESS_PAIR(3, 3);
    }

    #undef PROCESS_PAIR
    #undef DOT8_LDS

    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);
    for (int offset = 16; offset > 0; offset >>= 1) {
        gacc += __shfl_down(gacc, offset);
        uacc += __shfl_down(uacc, offset);
    }
    if (tid == 0) {
        y_gate[(long long)krank * mi + row] = gacc;
        y_up[(long long)krank * mi + row] = uacc;
    }
}
