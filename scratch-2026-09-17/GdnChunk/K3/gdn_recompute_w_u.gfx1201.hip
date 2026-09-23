// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GDN K3: recompute W=B diag(beta*exp(G)) K and U=B diag(beta) V.
// Layouts (last index contiguous):
//   k/v [N,H,128], beta [N,H], B [ceil(N/64),H,64,64],
//   G [ceil(N/64),H,64], w/u [ceil(N/64),H,64,128].
// Grid [ceil(N/64),H,1], block [256,1,1]. Invalid tail rows are zero.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) gdn_k3_h8;
typedef float __attribute__((ext_vector_type(8))) gdn_k3_f8;

__device__ __forceinline__ gdn_k3_f8 gdn_k3_mma(
    gdn_k3_h8 a, gdn_k3_h8 b, gdn_k3_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

extern "C" __launch_bounds__(256, 2)
__global__ void gdn_recompute_w_u_gfx1201(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ B,
    const float* __restrict__ G,
    _Float16* __restrict__ w,
    _Float16* __restrict__ u,
    int N,
    int H) {
    constexpr int C = 64;
    constexpr int D = 128;

    const int chunk = blockIdx.x;
    const int head = blockIdx.y;
    if (head >= H || chunk * C >= N) return;

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int half = lane & 15;
    const int half_id = lane >> 4;
    const int c0 = chunk * C;
    const int cn = min(C, N - c0);
    const unsigned token_stride = (unsigned)H * D;
    const unsigned head_off = (unsigned)head * D;
    const unsigned bh_base = ((unsigned)chunk * (unsigned)H +
                              (unsigned)head) * C * C;
    const unsigned g_base = ((unsigned)chunk * (unsigned)H +
                             (unsigned)head) * C;
    const unsigned out_base = ((unsigned)chunk * (unsigned)H +
                               (unsigned)head) * C * D;

    struct __attribute__((aligned(16))) K3Lds {
        _Float16 b[C * C];
        _Float16 rhs[C * C]; // transposed: [output-column, reduction-row]
    };
    __shared__ K3Lds lds;

#pragma clang loop unroll(disable)
    for (int x = tid; x < C * C; x += 256)
        lds.b[x] = (_Float16)B[bh_base + (unsigned)x];
    __syncthreads();

    // Four products: W/U, each streamed in two 64-column halves. B remains
    // resident while rhs is overwritten between products.
#pragma unroll
    for (int product = 0; product < 2; ++product) {
        const float* src = product == 0 ? k : v;
        _Float16* dst = product == 0 ? w : u;
#pragma unroll
        for (int col_half = 0; col_half < 2; ++col_half) {
#pragma clang loop unroll(disable)
            for (int x = tid; x < C * C; x += 256) {
                const int col = x >> 6;
                const int row = x & 63;
                float value = 0.0f;
                if (row < cn) {
                    const unsigned token = (unsigned)(c0 + row);
                    float scale = beta[token * (unsigned)H + (unsigned)head];
                    if (product == 0)
                        scale *= expf(G[g_base + (unsigned)row]);
                    value = scale * src[token * token_stride + head_off +
                                        (unsigned)(col_half * 64 + col)];
                }
                lds.rhs[x] = (_Float16)value;
            }
            __syncthreads();

#pragma unroll
            for (int round = 0; round < 2; ++round) {
                const int tile = wave + round * 8;
                const int rb = tile >> 2;
                const int cb = tile & 3;
                gdn_k3_f8 acc = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    gdn_k3_h8 a, b;
                    const int kc = kk * 16 + half_id * 8;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        a[j] = lds.b[(rb * 16 + half) * C + kc + j];
                        b[j] = lds.rhs[(cb * 16 + half) * C + kc + j];
                    }
                    acc = gdn_k3_mma(a, b, acc);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int row = rb * 16 + half_id * 8 + j;
                    const int col = col_half * 64 + cb * 16 + half;
                    const float value = row < cn ? acc[j] : 0.0f;
                    dst[out_base + (unsigned)row * D + (unsigned)col] =
                        (_Float16)value;
                }
            }
            __syncthreads();
        }
    }
}
