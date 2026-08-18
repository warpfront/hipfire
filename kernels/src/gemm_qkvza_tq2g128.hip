// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched 4-way fused TQ2-G128 GEMM for the Qwen3.5 DeltaNet LA preamble.
// Ternary counterpart of gemm_qkvza_hfq4g256:
//
//   Y_qkv[b]   = W_qkv   . X[b]      for b in 0..N
//   Y_z[b]     = W_z     . X[b]
//   Y_beta[b]  = W_beta  . X[b]
//   Y_alpha[b] = W_alpha . X[b]
//
// MEASURED AND NOT WIRED IN -- kept as the record of a rejected design.
//
// The hypothesis was that fusion explained the remaining prefill gap to
// qwen3.6-27b.mq4: mq4's prefill kernel does all four matrices in ONE grid
// (gemm_qkvza_hfq4g256, BATCH_TILE=8) while our path issues four launches.
// Ternary also reads 0.27 B/weight against mq4's 0.53, so fusing looked like
// it should overtake mq4 outright.
//
// It does not. At the Bonsai LA shapes (K=5120, m=[6144,4096,128,128]):
//
//   N=32   unfused 0.587 ms | fused 2.127 ms | 0.28x
//   N=128  unfused 2.110 ms | fused 8.643 ms | 0.24x
//   N=512  unfused 8.588 ms | fused 33.683 ms | 0.25x
//
// Fusion saves three launches but costs the matrix cores: the unfused path
// dispatches to gemm_tq2g128_wmma, and mq4's fused kernel contains no WMMA,
// no dp4a and no LDS. Four WMMA launches beat one scalar fused launch by 4x.
// A fused kernel is only worth revisiting if it is itself WMMA-based, which
// means solving fragment routing across four matrices of different m.
//
// So the remaining gap to mq4 is NOT launch structure. It is still unattributed
// and wants a profile before the next attempt.
//
// Each block handles (1 output row of one of the 4 projections) x (up to
// BATCH_TILE tokens). Grid = (total_m) x ceil(N / BATCH_TILE).
//
// Lane t owns code byte t of a 34-byte block, i.e. weights 4t..4t+3, so a
// wave covers one 128-weight group -- the same split the scalar GEMV uses.
// The 4 codes in a byte are decoded ONCE and reused across the token tile;
// that reuse is the point of batching.
//
// Overwrite semantics (y[b][row] = acc), matching the GEMV reference.

#ifndef HIPFIRE_QKVZA_TQ2_BATCH_TILE
#define HIPFIRE_QKVZA_TQ2_BATCH_TILE 8
#endif
#define BT HIPFIRE_QKVZA_TQ2_BATCH_TILE

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_qkvza_tq2g128(
    const char*  __restrict__ A_qkv,
    const char*  __restrict__ A_z,
    const char*  __restrict__ A_beta,
    const char*  __restrict__ A_alpha,
    const float* __restrict__ X,        // [N x K]
    float*       __restrict__ Y_qkv,    // [N x qkv_m]
    float*       __restrict__ Y_z,      // [N x z_m]
    float*       __restrict__ Y_beta,   // [N x beta_m]
    float*       __restrict__ Y_alpha,  // [N x alpha_m]
    int qkv_m, int z_m, int beta_m, int alpha_m,
    int K, int N
) {
    const int gid = blockIdx.x;
    const int total_m = qkv_m + z_m + beta_m + alpha_m;
    if (gid >= total_m) return;

    const int batch_start = blockIdx.y * BT;
    int local_bs = N - batch_start;
    if (local_bs > BT) local_bs = BT;
    if (local_bs <= 0) return;

    // Route this workgroup to its target matrix + output row + output stride.
    const char* A;
    float* Y;
    int row;
    int out_stride;
    if (gid < qkv_m) {
        A = A_qkv;   Y = Y_qkv;   row = gid;                          out_stride = qkv_m;
    } else if (gid < qkv_m + z_m) {
        A = A_z;     Y = Y_z;     row = gid - qkv_m;                  out_stride = z_m;
    } else if (gid < qkv_m + z_m + beta_m) {
        A = A_beta;  Y = Y_beta;  row = gid - (qkv_m + z_m);          out_stride = beta_m;
    } else {
        A = A_alpha; Y = Y_alpha; row = gid - (qkv_m + z_m + beta_m); out_stride = alpha_m;
    }

    const int tid = threadIdx.x;
    const int groups_per_row = K / 128;
    const char* row_ptr = A + (long long)row * groups_per_row * 34;
    const int xbase = tid * 4;

    // Four accumulator sets over g mod 4, matching the scalar kernel's
    // interleave so the pairwise combine below reproduces its FMA order.
    float acc0[BT], acc1[BT], acc2[BT], acc3[BT];
    #pragma unroll
    for (int b = 0; b < BT; b++) {
        acc0[b] = 0.0f; acc1[b] = 0.0f; acc2[b] = 0.0f; acc3[b] = 0.0f;
    }

    // Decode one code byte into 4 scaled ternary weights, then apply them to
    // every token in the tile. x is 16B-aligned at g*128 + tid*4, so the
    // float4 load is legal for every lane.
    #define TQ2_GROUP(gidx, accA)                                              \
        do {                                                                   \
            const char* gp = row_ptr + (gidx) * 34;                            \
            const float d = (float)(*(const _Float16*)gp);                     \
            const unsigned int c = *((const unsigned char*)(gp + 2 + tid));    \
            const float w0 = d * (float)((int)( c       & 0x3) - 1);           \
            const float w1 = d * (float)((int)((c >> 2) & 0x3) - 1);           \
            const float w2 = d * (float)((int)((c >> 4) & 0x3) - 1);           \
            const float w3 = d * (float)((int)((c >> 6) & 0x3) - 1);           \
            const int xoff = (gidx) * 128 + xbase;                             \
            _Pragma("unroll")                                                  \
            for (int b = 0; b < BT; b++) {                                     \
                if (b >= local_bs) break;                                      \
                const float4 xv =                                              \
                    *(const float4*)(X + (long long)(batch_start + b) * K + xoff); \
                (accA)[b] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;    \
            }                                                                  \
        } while (0)

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        TQ2_GROUP(g,     acc0);
        TQ2_GROUP(g + 1, acc1);
        TQ2_GROUP(g + 2, acc2);
        TQ2_GROUP(g + 3, acc3);
    }
    for (int t = 0; t < tail; t++) {
        const int g = (quads << 2) + t;
        if (t == 0)      { TQ2_GROUP(g, acc0); }
        else if (t == 1) { TQ2_GROUP(g, acc1); }
        else             { TQ2_GROUP(g, acc2); }
    }
    #undef TQ2_GROUP

    #pragma unroll
    for (int b = 0; b < BT; b++) {
        if (b >= local_bs) break;
        float acc = (acc0[b] + acc1[b]) + (acc2[b] + acc3[b]);
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (tid == 0) {
            Y[(long long)(batch_start + b) * out_stride + row] = acc;
        }
    }
}
