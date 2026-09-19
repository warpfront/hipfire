// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// GDN K2 kill-gate kernel: form the C64 causal K K^T system and invert it
// with BC16 diagonal solves plus block forward substitution.
//
// Layouts (last index contiguous):
//   k [N,H,128], gate/beta [N,H], B [ceil(N/64),H,64,64],
//   G [ceil(N/64),H,64].
// Grid [ceil(N/64),H,1], block [256,1,1]. Invalid tail rows are identity.
//
// The single 16 KiB LDS union first holds K as f16 for the gfx1201 WMMA Gram,
// then holds the complete FP32 inverse. L is staged through the output buffer
// between those phases and remains there until the final inverse store.

#include <hip/hip_runtime.h>

typedef _Float16 __attribute__((ext_vector_type(8))) gdn_k2_h8;
typedef float __attribute__((ext_vector_type(8))) gdn_k2_f8;

__device__ __forceinline__ gdn_k2_f8 gdn_k2_mma(
    gdn_k2_h8 a, gdn_k2_h8 b, gdn_k2_f8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

extern "C" __launch_bounds__(256, 2)
__global__ void gdn_kkt_bc16_solve_gfx1201(
    const float* __restrict__ k,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ B,
    float* __restrict__ G,
    int N,
    int H) {
    constexpr int C = 64;
    constexpr int D = 128;
    constexpr int BC = 16;

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
    const unsigned stride = (unsigned)H * D;
    const unsigned h_off = (unsigned)head * D;
    const unsigned bh_base = ((unsigned)chunk * (unsigned)H +
                              (unsigned)head) * C * C;
    const unsigned g_base = ((unsigned)chunk * (unsigned)H +
                             (unsigned)head) * C;

    union __attribute__((aligned(16))) K2Lds {
        _Float16 k16[C * D];
        float inv[C * C];
    };
    __shared__ K2Lds lds;

    // Prefix gates. Sixty-four threads independently reproduce the short
    // left-to-right FP32 prefix, avoiding any LDS beyond the 16 KiB union.
    if (tid < C) {
        float sum = 0.0f;
#pragma clang loop unroll(disable)
        for (int r = 0; r <= tid; ++r) {
            if (r < cn)
                sum += gate[(unsigned)(c0 + r) * (unsigned)H +
                            (unsigned)head];
        }
        G[g_base + (unsigned)tid] = sum;
    }

    // Initialize the global L staging area, including upper-block zeros.
#pragma clang loop unroll(disable)
    for (int x = tid; x < C * C; x += 256)
        B[bh_base + (unsigned)x] = 0.0f;

    // Stage the full C64 K panel to f16. Each thread owns four h8 vectors.
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int x = tid * 8 + u * 2048;
        const int row = x >> 7;
        const int col = x & 127;
        gdn_k2_h8 pack;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const bool valid = row < cn;
            pack[j] = valid
                ? (_Float16)k[(unsigned)(c0 + row) * stride + h_off +
                              (unsigned)(col + j)]
                : (_Float16)0.0f;
        }
        *reinterpret_cast<gdn_k2_h8*>(lds.k16 + x) = pack;
    }
    __syncthreads();
    __threadfence_block();
    __syncthreads();

    // Ten lower-triangular BC16 Gram tiles. Eight waves cover the first
    // eight tiles concurrently; waves 0 and 1 cover the final two tiles.
#pragma clang loop unroll(disable)
    for (int tile = wave; tile < 10; tile += 8) {
        int rb, cb;
        if (tile == 0)      { rb = 0; cb = 0; }
        else if (tile == 1) { rb = 1; cb = 0; }
        else if (tile == 2) { rb = 1; cb = 1; }
        else if (tile == 3) { rb = 2; cb = 0; }
        else if (tile == 4) { rb = 2; cb = 1; }
        else if (tile == 5) { rb = 2; cb = 2; }
        else if (tile == 6) { rb = 3; cb = 0; }
        else if (tile == 7) { rb = 3; cb = 1; }
        else if (tile == 8) { rb = 3; cb = 2; }
        else                { rb = 3; cb = 3; }

        gdn_k2_f8 acc = {0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int dc = 0; dc < 8; ++dc) {
            gdn_k2_h8 a, b;
            const int dk = dc * 16 + half_id * 8;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a[j] = lds.k16[(rb * BC + half) * D + dk + j];
                b[j] = lds.k16[(cb * BC + half) * D + dk + j];
            }
            acc = gdn_k2_mma(a, b, acc);
        }

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int row = rb * BC + half_id * 8 + j;
            const int col = cb * BC + half;
            float lv = 0.0f;
            if (row < cn && col < row) {
                const float gi = G[g_base + (unsigned)row];
                const float gj = G[g_base + (unsigned)col];
                const float bv = beta[(unsigned)(c0 + row) * (unsigned)H +
                                      (unsigned)head];
                lv = bv * expf(gi - gj) * acc[j];
            }
            B[bh_base + (unsigned)(row * C + col)] = lv;
        }
    }
    __syncthreads();
    __threadfence_block();
    __syncthreads();

    // K is dead. Load the staged strict-lower L into the same 16 KiB arena.
#pragma clang loop unroll(disable)
    for (int x = tid; x < C * C; x += 256)
        lds.inv[x] = B[bh_base + (unsigned)x];
    __syncthreads();

    // Four independent diagonal inverses P_a=(I+L_aa)^-1. One wave owns
    // each BC16 block; lanes 0..15 own inverse columns.
#pragma unroll
    for (int i = 0; i < BC; ++i) {
        if (wave < 4 && lane < BC) {
            const int block = wave;
            const int row = block * BC + i;
            const int col = block * BC + lane;
            float value = (i == lane) ? 1.0f : 0.0f;
#pragma clang loop unroll(disable)
            for (int r = 0; r < i; ++r) {
                const float l = B[bh_base + (unsigned)(row * C +
                                                       block * BC + r)];
                value -= l * lds.inv[(block * BC + r) * C + col];
            }
            lds.inv[row * C + col] = (lane <= i) ? value : 0.0f;
        }
        __syncthreads();
    }

    // Block forward substitution. The target block itself is temporary T,
    // so no exchange arena is required:
    // B_ab = -P_a * sum_{r=b}^{a-1}(L_ar * B_rb).
#pragma unroll
    for (int a = 1; a < 4; ++a) {
#pragma unroll
        for (int b = 0; b < a; ++b) {
            const int rr = tid >> 4;
            const int cc = tid & 15;
            float t = 0.0f;
#pragma clang loop unroll(disable)
            for (int rblock = b; rblock < a; ++rblock) {
#pragma unroll
                for (int kk = 0; kk < BC; ++kk) {
                    const int lr = a * BC + rr;
                    const int lc = rblock * BC + kk;
                    const float l = B[bh_base + (unsigned)(lr * C + lc)];
                    const float br = lds.inv[(rblock * BC + kk) * C +
                                             b * BC + cc];
                    t += l * br;
                }
            }
            const int target = (a * BC + rr) * C + b * BC + cc;
            lds.inv[target] = t;
            __syncthreads();

            float value = 0.0f;
#pragma unroll
            for (int kk = 0; kk < BC; ++kk)
                value -= lds.inv[(a * BC + rr) * C + a * BC + kk] *
                         lds.inv[(a * BC + kk) * C + b * BC + cc];
            __syncthreads();
            lds.inv[target] = value;
            __syncthreads();
        }
    }

    // Initialized, output-producing final store.
#pragma clang loop unroll(disable)
    for (int x = tid; x < C * C; x += 256)
        B[bh_base + (unsigned)x] = lds.inv[x];
}
