// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1010-only cooperative FP32 MQ4G256V2 set kernel (Y = A·X, no residual).
//
// Tile M4×N32, block 256 (eight wave32s), fixed LDS 36864:
//   x_lds[32][8][32]  — one 256-K group of X for 32 tokens (32768 B)
//   w_lds[4][8][32]   — dequantized weights for 4 rows (4096 B)
//
// Wave mapping: wave=tid>>5, lane=tid&31, row_slot=wave&3, token_half=wave>>2.
// Paired waves (r, r+4) jointly stage row r weights; wave r consumes tokens
// 0..15 and wave r+4 consumes 16..31. Four private chains preserve g&3 routing,
// eight-term product order, pairwise combine, and width-32 shuffle tree of the
// residual/SIMT oracle. Body compiles only under __gfx1010__.

#if defined(__gfx1010__)

__launch_bounds__(256, 1)
extern "C" __global__ void gemm_mq4g256v2_set_simt_m4n32_lds_gfx1010(
    const char* __restrict__ A,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int batch_size
) {
    __shared__ float x_lds[32][8][32];
    __shared__ float w_lds[4][8][32];

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row_slot = wave & 3;
    const int token_half = wave >> 2;

    const int row_base = (int)blockIdx.x * 4;
    const int token_base = (int)blockIdx.y * 32;

    // Whole-workgroup-uniform early exit only — before any barrier.
    if (row_base >= M || token_base >= batch_size) return;

    const int row = row_base + row_slot;
    const int row_valid = row < M;
    // Invalid final-M row still stages A from safe row 0 (M is nonzero here).
    const int a_row = row_valid ? row : 0;

    const int half_base = token_half * 16;
    int local_n = batch_size - (token_base + half_base);
    if (local_n > 16) local_n = 16;
    if (local_n < 0) local_n = 0;

    const int groups_per_row = K / 256;
    const char* row_ptr =
        A + (long long)a_row * (long long)groups_per_row * 136LL;

    // Four chains × 16 local tokens; private across all K groups.
    float acc0[16], acc1[16], acc2[16], acc3[16];
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        acc0[t] = 0.0f;
        acc1[t] = 0.0f;
        acc2[t] = 0.0f;
        acc3[t] = 0.0f;
    }

    for (int g = 0; g < groups_per_row; g++) {
        // ---------------------------------------------------------------
        // Phase 1a: cooperative X stage — 2048 float4 tasks, 8 per thread.
        // token = task>>6, pack = task&63, logical_lane = pack>>1,
        // j0 = (pack&1)*4. Contiguous aligned float4 → scalar LDS scatter.
        // ---------------------------------------------------------------
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int task = tid + i * 256;
            const int token = task >> 6;
            const int pack = task & 63;
            const int logical_lane = pack >> 1;
            const int j0 = (pack & 1) * 4;
            const int gtok = token_base + token;

            float x0, x1, x2, x3;
            if (gtok < batch_size) {
                const float* src =
                    X + (long long)gtok * (long long)K
                      + (long long)g * 256LL
                      + (long long)(pack * 4);
                const float4 v = *reinterpret_cast<const float4*>(src);
                x0 = v.x;
                x1 = v.y;
                x2 = v.z;
                x3 = v.w;
            } else {
                x0 = 0.0f;
                x1 = 0.0f;
                x2 = 0.0f;
                x3 = 0.0f;
            }
            x_lds[token][j0 + 0][logical_lane] = x0;
            x_lds[token][j0 + 1][logical_lane] = x1;
            x_lds[token][j0 + 2][logical_lane] = x2;
            x_lds[token][j0 + 3][logical_lane] = x3;
        }

        // ---------------------------------------------------------------
        // Phase 1b: paired-wave weight dequant stage.
        // Lanes 0..15 of every wave stage half-row logical_lane =
        // lane + 16*token_half. Lane 0 loads hA (half 0) or hB (half 1);
        // full-wave width-32 broadcast before any lane<16 store.
        // ---------------------------------------------------------------
        {
            const char* gp = row_ptr + (long long)g * 136LL;
            unsigned int hs = 0u;
            if (lane == 0) {
                const int hoff = token_half * 4; // 0 → hA, 4 → hB
                hs = *reinterpret_cast<const unsigned int*>(gp + hoff);
            }
            hs = __shfl(hs, 0, 32);

            const float sc =
                __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
            const float zp =
                __half2float(__ushort_as_half((unsigned short)(hs >> 16)));

            if (lane < 16) {
                const int logical_lane = lane + token_half * 16;
                const unsigned int pk =
                    *reinterpret_cast<const unsigned int*>(
                        gp + 8 + logical_lane * 4);
                w_lds[row_slot][0][logical_lane] =
                    sc * (float)((pk      ) & 0xFu) + zp;
                w_lds[row_slot][1][logical_lane] =
                    sc * (float)((pk >>  4) & 0xFu) + zp;
                w_lds[row_slot][2][logical_lane] =
                    sc * (float)((pk >>  8) & 0xFu) + zp;
                w_lds[row_slot][3][logical_lane] =
                    sc * (float)((pk >> 12) & 0xFu) + zp;
                w_lds[row_slot][4][logical_lane] =
                    sc * (float)((pk >> 16) & 0xFu) + zp;
                w_lds[row_slot][5][logical_lane] =
                    sc * (float)((pk >> 20) & 0xFu) + zp;
                w_lds[row_slot][6][logical_lane] =
                    sc * (float)((pk >> 24) & 0xFu) + zp;
                w_lds[row_slot][7][logical_lane] =
                    sc * (float)((pk >> 28) & 0xFu) + zp;
            }
        }

        // Barrier 1: LDS becomes immutable snapshot of group g.
        __syncthreads();

        // ---------------------------------------------------------------
        // Phase 2: consume. Load w0..w7 once; for each valid local token
        // load x0..x7 and update only chain g&3 with oracle expression.
        // Explicit switch — no dynamic accumulator-array index.
        // ---------------------------------------------------------------
        {
            const float w0 = w_lds[row_slot][0][lane];
            const float w1 = w_lds[row_slot][1][lane];
            const float w2 = w_lds[row_slot][2][lane];
            const float w3 = w_lds[row_slot][3][lane];
            const float w4 = w_lds[row_slot][4][lane];
            const float w5 = w_lds[row_slot][5][lane];
            const float w6 = w_lds[row_slot][6][lane];
            const float w7 = w_lds[row_slot][7][lane];

            switch (g & 3) {
            case 0:
                #pragma unroll
                for (int t = 0; t < 16; t++) {
                    if (t >= local_n) break;
                    const int lt = half_base + t;
                    const float x0 = x_lds[lt][0][lane];
                    const float x1 = x_lds[lt][1][lane];
                    const float x2 = x_lds[lt][2][lane];
                    const float x3 = x_lds[lt][3][lane];
                    const float x4 = x_lds[lt][4][lane];
                    const float x5 = x_lds[lt][5][lane];
                    const float x6 = x_lds[lt][6][lane];
                    const float x7 = x_lds[lt][7][lane];
                    acc0[t] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
                             + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
                }
                break;
            case 1:
                #pragma unroll
                for (int t = 0; t < 16; t++) {
                    if (t >= local_n) break;
                    const int lt = half_base + t;
                    const float x0 = x_lds[lt][0][lane];
                    const float x1 = x_lds[lt][1][lane];
                    const float x2 = x_lds[lt][2][lane];
                    const float x3 = x_lds[lt][3][lane];
                    const float x4 = x_lds[lt][4][lane];
                    const float x5 = x_lds[lt][5][lane];
                    const float x6 = x_lds[lt][6][lane];
                    const float x7 = x_lds[lt][7][lane];
                    acc1[t] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
                             + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
                }
                break;
            case 2:
                #pragma unroll
                for (int t = 0; t < 16; t++) {
                    if (t >= local_n) break;
                    const int lt = half_base + t;
                    const float x0 = x_lds[lt][0][lane];
                    const float x1 = x_lds[lt][1][lane];
                    const float x2 = x_lds[lt][2][lane];
                    const float x3 = x_lds[lt][3][lane];
                    const float x4 = x_lds[lt][4][lane];
                    const float x5 = x_lds[lt][5][lane];
                    const float x6 = x_lds[lt][6][lane];
                    const float x7 = x_lds[lt][7][lane];
                    acc2[t] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
                             + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
                }
                break;
            default:
                #pragma unroll
                for (int t = 0; t < 16; t++) {
                    if (t >= local_n) break;
                    const int lt = half_base + t;
                    const float x0 = x_lds[lt][0][lane];
                    const float x1 = x_lds[lt][1][lane];
                    const float x2 = x_lds[lt][2][lane];
                    const float x3 = x_lds[lt][3][lane];
                    const float x4 = x_lds[lt][4][lane];
                    const float x5 = x_lds[lt][5][lane];
                    const float x6 = x_lds[lt][6][lane];
                    const float x7 = x_lds[lt][7][lane];
                    acc3[t] += w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
                             + w4 * x4 + w5 * x5 + w6 * x6 + w7 * x7;
                }
                break;
            }
        }

        // Barrier 2: all consumers done before next group overwrites LDS.
        __syncthreads();
    }

    // Epilogue: pairwise combine + width-32 shuffle-down 16/8/4/2/1.
    // Lane 0 stores only when row and global token are both valid.
    #pragma unroll
    for (int t = 0; t < 16; t++) {
        if (t >= local_n) break;
        float val = (acc0[t] + acc1[t]) + (acc2[t] + acc3[t]);
        val += __shfl_down(val, 16, 32);
        val += __shfl_down(val,  8, 32);
        val += __shfl_down(val,  4, 32);
        val += __shfl_down(val,  2, 32);
        val += __shfl_down(val,  1, 32);
        if (lane == 0 && row_valid) {
            const int gtok = token_base + half_base + t;
            Y[(long long)gtok * (long long)M + (long long)row] = val;
        }
    }
}

#endif // defined(__gfx1010__)
