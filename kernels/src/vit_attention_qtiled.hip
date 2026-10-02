// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q-tiled bidirectional attention for the vision encoder (flash-style
// online softmax). Replaces the per-(head, query) `vit_attention_f32`,
// whose blocks each re-read all of K and V from DRAM — ~220 GB of traffic
// per tower layer at N=4624 (measured 1.02 s/layer on gfx1101, i.e. the
// whole 25.5 s vision forward at 68x68 grid). Here QB queries of one head
// share every K tile pulled through LDS: DRAM traffic drops by ~QB.
//
// V is NOT staged in LDS: in the P@V accumulation each (j, d) element is
// read by exactly one lane (d = lane + a*LANES), so there is no intra-block
// reuse to pay LDS for — lanes read V straight from global/L2, fully
// coalesced. Keeping only K+Q+scores in LDS holds the footprint near
// 30 KB, which lets two blocks share a CU.
//
// Computes: out[qi, head] = softmax(Q[qi] @ K^T / sqrt(d)) @ V   (no mask)
//
// Grid:  [num_heads, ceil(N / QB)]
// Block: [256]           — QB * LANES threads; each 16-lane subgroup owns
//                        — one query for the whole tile.
// LDS (QB=16, K_TILE=64, hd=64): k_tile 16 KB + q_tile 4 KB + s_tile 4 KB
// + reductions ~1 KB + out_run 4 KB ≈ 29 KB.

#define QB 16      // queries per block (one 16-lane subgroup each)
#define LANES 16   // threads per query subgroup
#define K_TILE 64  // keys streamed through LDS per iteration
#define A_MAX 8    // max head_dim/LANES (supports head_dim up to 128)

extern "C" __global__ __launch_bounds__(256, 2) void vit_attention_qtiled_f32(
    const float* __restrict__ qkv,   // [N, 3*hidden] row-major, interleaved Q/K/V
    float* __restrict__ out,         // [N, hidden] row-major
    int N, int hidden, int num_heads, int head_dim, float scale)
{
    const int head = blockIdx.x;
    const int q0 = blockIdx.y * QB;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (head >= num_heads || q0 >= N) return;

    const int qid = tid / LANES;      // which query this subgroup owns
    const int lane = tid % LANES;
    const int qi = q0 + qid;
    const int nq = min(QB, N - q0);   // queries live in this block
    const bool alive = qid < nq;

    const int stride = 3 * hidden;
    const int q_off = head * head_dim;
    const int k_off = hidden + head * head_dim;
    const int v_off = 2 * hidden + head * head_dim;
    const int A = head_dim / LANES;   // out elements owned per lane

    extern __shared__ float smem[];
    float* k_tile  = smem;                       // [K_TILE * head_dim]
    float* q_tile  = k_tile + K_TILE * head_dim; // [QB * head_dim]
    float* s_tile  = q_tile + QB * head_dim;     // [QB * K_TILE]
    float* ws_max  = s_tile + QB * K_TILE;       // [QB * LANES]
    float* ws_sum  = ws_max + QB * LANES;        // [QB * LANES]
    float* m_l     = ws_sum + QB * LANES;        // [QB * 2] running (max, sum)
    float* out_run = m_l + QB * 2;               // [QB * head_dim]

    // Load this block's Q rows once.
    for (int i = tid; i < nq * head_dim; i += nthreads) {
        int q = i / head_dim, d = i % head_dim;
        q_tile[q * head_dim + d] = qkv[(q0 + q) * stride + q_off + d];
    }
    if (alive && lane == 0) {
        m_l[qid * 2 + 0] = -INFINITY;
        m_l[qid * 2 + 1] = 0.0f;
    }
    for (int i = tid; i < nq * head_dim; i += nthreads) out_run[i] = 0.0f;
    __syncthreads();

    float s_reg[A_MAX];  // this lane's scores/p for the current tile

    for (int t0 = 0; t0 < N; t0 += K_TILE) {
        const int tlen = min(K_TILE, N - t0);

        // Cooperative K tile load.
        for (int i = tid; i < tlen * head_dim; i += nthreads) {
            int j = i / head_dim, d = i % head_dim;
            k_tile[j * head_dim + d] = qkv[(t0 + j) * stride + k_off + d];
        }
        __syncthreads();

        if (alive) {
            // Phase A: full scores for this lane's K rows (registers).
            float smax = -INFINITY;
            for (int jj = 0; jj < K_TILE / LANES; jj++) {
                int j = lane + jj * LANES;
                if (j < tlen) {
                    const float* qrow = q_tile + qid * head_dim;
                    const float* krow = k_tile + j * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) dot += qrow[d] * krow[d];
                    float s = dot * scale;
                    s_reg[jj] = s;
                    smax = fmaxf(smax, s);
                } else {
                    s_reg[jj] = -INFINITY;
                }
            }
            ws_max[qid * LANES + lane] = smax;
        }
        __syncthreads();

        if (alive) {
            // Tile max / running max, shared through LDS so every lane of
            // the subgroup recomputes the same m_new.
            float tmax = -INFINITY;
            for (int l = 0; l < LANES; l++)
                tmax = fmaxf(tmax, ws_max[qid * LANES + l]);
            const float m_old = m_l[qid * 2 + 0];
            const float m_new = fmaxf(m_old, tmax);

            // Scaled probabilities, shared through s_tile; partial sums.
            float psum = 0.0f;
            for (int jj = 0; jj < K_TILE / LANES; jj++) {
                int j = lane + jj * LANES;
                float p = (j < tlen) ? expf(s_reg[jj] - m_new) : 0.0f;
                s_reg[jj] = p;
                s_tile[qid * K_TILE + j] = p;
                psum += p;
            }
            ws_sum[qid * LANES + lane] = psum;
        }
        __syncthreads();

        if (alive) {
            float tmax = -INFINITY;
            for (int l = 0; l < LANES; l++)
                tmax = fmaxf(tmax, ws_max[qid * LANES + l]);
            const float m_old = m_l[qid * 2 + 0];
            const float m_new = fmaxf(m_old, tmax);
            const float r = expf(m_old - m_new);
            float tsum = 0.0f;
            for (int l = 0; l < LANES; l++)
                tsum += ws_sum[qid * LANES + l];

            // Rescale the running output and fold in this tile's P@V.
            // V comes straight from global: each (j, d) is touched by
            // exactly one lane of this block.
            const float* vrow_base = qkv + v_off;
            for (int a = 0; a < A; a++) {
                int d = lane + a * LANES;
                float acc = out_run[qid * head_dim + d] * r;
                const float* prow = s_tile + qid * K_TILE;
                #pragma unroll 8
                for (int j = 0; j < tlen; j++)
                    acc += prow[j] * vrow_base[(t0 + j) * stride + d];
                out_run[qid * head_dim + d] = acc;
            }
            if (lane == 0) {
                m_l[qid * 2 + 0] = m_new;
                m_l[qid * 2 + 1] = m_l[qid * 2 + 1] * r + tsum;
            }
        }
        __syncthreads();
    }

    // Normalize and write out.
    if (alive) {
        const float l = m_l[qid * 2 + 1];
        for (int a = 0; a < A; a++) {
            int d = lane + a * LANES;
            out[qi * hidden + head * head_dim + d] =
                out_run[qid * head_dim + d] / l;
        }
    }
}
