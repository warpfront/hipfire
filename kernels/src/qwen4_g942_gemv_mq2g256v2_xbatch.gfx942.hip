// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_g942_gemv_mq2g256v2_xbatch{4,8}_k{2048,2560}: the x-batched MQ2G256V2 GEMV of the MTP
// draft head's ranking (`Y[b, row] = W[row] . X[b]` for up to NB rotated rows `X`, exact
// gfx942), one weight pass for all NB rows.
//
// Every output element equals the one-row `gemv_mq2g256v2` of that row bit for bit (kernel
// oracle: `examples/qwen4_g942_mq2_xbatch_oracle.rs`). What the one-row kernel compiles to on
// gfx942 (gemv_mq2g256v2.hip, hipcc -O3; the v_pk_* sequence of its ISA), per lane `p` of a
// 32-lane row group and per 256-value group `g` (8 values `p*8 .. p*8+7`):
//   t_j  = fma(scale, q_j, zero)                     q_j the 2-bit code j of the lane's 16 bits
//   main (g < 4 * (K / 1024)):  s = fma(t_0, x_0, t_1 * x_1)  (the t_1 product rounded), then
//                               s = fma(t_j, x_j, s) for j = 2..7
//   tail (the K % 1024 groups): every product rounded, s = (((p_0 + p_1) + p_2) + ... + p_7)
//   acc[g % 4] += s                                  groups in ascending order per accumulator
//   lane value = (acc0 + acc1) + (acc2 + acc3), then the 5-step shfl_down tree over the 32 lanes
//   (offsets 16, 8, 4, 2, 1; lane 0 is the row).
// This kernel reproduces exactly those operations. The four accumulators of a row live on four
// different lanes (chain c = g % 4 of lane p sits on thread c * 32 + p), so a thread holds its
// x values for <= 3 groups in registers for all the rows it visits and the one-row kernel's
// per-lane x re-reads disappear; the four chains meet through LDS in the same
// (acc0 + acc1) + (acc2 + acc3) association, and the lane tree is the same shfl_down tree.
//
// Block = 256 threads = two row slots of 128 threads (4 chains x 32 lanes). A slot walks
// ROWS_PER_SLOT consecutive rows; all 256 threads run the same iteration count (the barrier is
// uniform), a row past M only skips its loads and its store.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define MQ2X_ROWS_PER_SLOT 16
#define MQ2X_ROWS_PER_ITER 2
#define MQ2X_MAX_CHAIN_GROUPS 3

// Tail-group sum: every product rounded, left-associated adds (no fma contraction).
static __device__ __forceinline__ float mq2x_tail_sum(const float (&t)[8], const float (&x)[8]) {
#pragma clang fp contract(off)
    float s = t[0] * x[0] + t[1] * x[1];
    s = s + t[2] * x[2];
    s = s + t[3] * x[3];
    s = s + t[4] * x[4];
    s = s + t[5] * x[5];
    s = s + t[6] * x[6];
    s = s + t[7] * x[7];
    return s;
}

// Main-group sum: first-term fma over the rounded second product, then an fma chain.
static __device__ __forceinline__ float mq2x_main_sum(const float (&t)[8], const float (&x)[8]) {
#pragma clang fp contract(off)
    float s = __fmaf_rn(t[0], x[0], t[1] * x[1]);
    s = __fmaf_rn(t[2], x[2], s);
    s = __fmaf_rn(t[3], x[3], s);
    s = __fmaf_rn(t[4], x[4], s);
    s = __fmaf_rn(t[5], x[5], s);
    s = __fmaf_rn(t[6], x[6], s);
    s = __fmaf_rn(t[7], x[7], s);
    return s;
}

template <int NB, int QUADS, int TAIL>
static __device__ __forceinline__ void mq2_xbatch_body(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B x K]
    float* __restrict__ y,         // [B x M]
    int M, int K, int B
) {
    constexpr int G = QUADS * 4 + TAIL;
    constexpr int CG = (G + 3) / 4;  // groups of the longest chain
    constexpr int RPI = MQ2X_ROWS_PER_ITER;
    static_assert(CG <= MQ2X_MAX_CHAIN_GROUPS, "K too wide for the register-resident x");
    static_assert(MQ2X_ROWS_PER_SLOT % RPI == 0, "rows per slot must be a multiple of RPI");
    // [parity][slot][row of the iteration][chain][row b][lane]
    __shared__ float part[2][2][RPI][4][NB][32];

    const int tid = (int)threadIdx.x;
    const int slot = tid >> 7;
    const int cl = tid & 127;
    const int c = cl >> 5;   // accumulator chain
    const int p = cl & 31;   // lane of the row group

    // x values of this thread: groups g = c + 4 i, 8 values p*8.. each, for the NB rows.
    float xr[CG][NB][8];
#pragma unroll
    for (int i = 0; i < CG; ++i) {
        const int g = c + 4 * i;
#pragma unroll
        for (int b = 0; b < NB; ++b) {
            if (g < G && b < B) {
                const float4* src =
                    reinterpret_cast<const float4*>(x + (long long)b * K + g * 256 + p * 8);
                const float4 lo = src[0];
                const float4 hi = src[1];
                xr[i][b][0] = lo.x; xr[i][b][1] = lo.y; xr[i][b][2] = lo.z; xr[i][b][3] = lo.w;
                xr[i][b][4] = hi.x; xr[i][b][5] = hi.y; xr[i][b][6] = hi.z; xr[i][b][7] = hi.w;
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) xr[i][b][j] = 0.0f;
            }
        }
    }

    const long long row_bytes = (long long)G * 72;
    const int slot_row0 = ((int)blockIdx.x * 2 + slot) * MQ2X_ROWS_PER_SLOT;
    const int hdr_off = (p < 16) ? 0 : 4;

    // Weight words of RPI rows of this thread: header word and 16 code bits per group.
    unsigned int pre_hs[RPI][CG];
    unsigned int pre_pk[RPI][CG];
    auto load_rows = [&](int row0) {
#pragma unroll
        for (int r = 0; r < RPI; ++r) {
            const int row = row0 + r;
#pragma unroll
            for (int i = 0; i < CG; ++i) {
                const int g = c + 4 * i;
                pre_hs[r][i] = 0u;
                pre_pk[r][i] = 0u;
                if (row < M && g < G) {
                    const char* rp = A + (long long)row * row_bytes + g * 72;
                    pre_hs[r][i] = *reinterpret_cast<const unsigned int*>(rp + hdr_off);
                    pre_pk[r][i] = *reinterpret_cast<const unsigned short*>(rp + 8 + p * 2);
                }
            }
        }
    };
    load_rows(slot_row0);

    for (int it = 0; it < MQ2X_ROWS_PER_SLOT / RPI; ++it) {
        const int row0 = slot_row0 + it * RPI;
        unsigned int cur_hs[RPI][CG];
        unsigned int cur_pk[RPI][CG];
#pragma unroll
        for (int r = 0; r < RPI; ++r)
#pragma unroll
            for (int i = 0; i < CG; ++i) { cur_hs[r][i] = pre_hs[r][i]; cur_pk[r][i] = pre_pk[r][i]; }
        if (it + 1 < MQ2X_ROWS_PER_SLOT / RPI) load_rows(row0 + RPI);

        float (*pp)[RPI][4][NB][32] = part[it & 1];
#pragma unroll
        for (int r = 0; r < RPI; ++r) {
            float acc[NB];
#pragma unroll
            for (int b = 0; b < NB; ++b) acc[b] = 0.0f;
#pragma unroll
            for (int i = 0; i < CG; ++i) {
                const int g = c + 4 * i;
                if (g < G) {
                    const float sc =
                        __half2float(__ushort_as_half((unsigned short)(cur_hs[r][i] & 0xFFFFu)));
                    const float zp =
                        __half2float(__ushort_as_half((unsigned short)(cur_hs[r][i] >> 16)));
                    float t[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        t[j] = __fmaf_rn(sc, (float)((cur_pk[r][i] >> (2 * j)) & 0x3u), zp);
                    const bool tail_group = (i >= QUADS);
#pragma unroll
                    for (int b = 0; b < NB; ++b) {
                        if (b < B) {
                            const float s = tail_group ? mq2x_tail_sum(t, xr[i][b])
                                                       : mq2x_main_sum(t, xr[i][b]);
                            acc[b] = acc[b] + s;
                        }
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < NB; ++b) pp[slot][r][c][b][p] = acc[b];
        }
        // Exchange the four chains and finish the rows: (acc0 + acc1) + (acc2 + acc3), then the
        // lane tree.
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RPI; ++r) {
#pragma unroll
            for (int k = 0; k < NB / 4; ++k) {
                const int b = (cl >> 5) + 4 * k;   // this half-wave reduces row b
                float v = (pp[slot][r][0][b][p] + pp[slot][r][1][b][p]) +
                          (pp[slot][r][2][b][p] + pp[slot][r][3][b][p]);
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    v += __shfl_down(v, offset, 32);
                const int row = row0 + r;
                if (p == 0 && row < M && b < B) y[(long long)b * M + row] = v;
            }
        }
    }
}

#define MQ2X_ENTRY(NAME, NB, QUADS, TAIL)                                                   \
    __launch_bounds__(256)                                                                  \
    extern "C" __global__ void NAME(const char* __restrict__ A, const float* __restrict__ x, \
                                    float* __restrict__ y, int M, int K, int B) {           \
        mq2_xbatch_body<NB, QUADS, TAIL>(A, x, y, M, K, B);                                 \
    }

MQ2X_ENTRY(qwen4_g942_gemv_mq2g256v2_xbatch4_k2048, 4, 2, 0)
MQ2X_ENTRY(qwen4_g942_gemv_mq2g256v2_xbatch4_k2560, 4, 2, 2)
MQ2X_ENTRY(qwen4_g942_gemv_mq2g256v2_xbatch8_k2048, 8, 2, 0)
MQ2X_ENTRY(qwen4_g942_gemv_mq2g256v2_xbatch8_k2560, 8, 2, 2)
