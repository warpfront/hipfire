// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q-tiled bidirectional attention for the qwen3.5 embedding encoder
// (arch 17). Flash-style online softmax, no mask, GQA.
//
// Why it exists: `attention_t5_bias_f32` launches one 256-thread block per
// (query row, head), so every one of the N query rows re-reads all of K and V
// for its head. At N=4096, H=8, hd=256 that is ~275 GB of traffic per layer
// and measured 486 ms/layer on gfx1101 — 82% of the whole encode
// (rocprofv3, 2026-10-12). Here QB queries of one head share every K tile
// pulled through LDS, cutting that traffic by ~QB.
//
// Same structure as `vit_attention_qtiled_f32`, with the three differences
// the encoder needs:
//   1. Q, K and V are separate tensors (three GEMMs), not one interleaved
//      [N, 3*hidden] buffer.
//   2. GQA: `n_heads` Q heads over `n_kv_heads` K/V heads
//      (`n_heads % n_kv_heads == 0`), so K/V index the head as
//      `h / (n_heads / n_kv_heads)`.
//   3. `head_dim` up to 256 (A_MAX = 256/LANES), where the vision tower's
//      kernel stops at 128 — `k_tile` is `K_TILE * head_dim` floats, so the
//      larger head needs a smaller K tile to stay inside the 64 KB LDS.
//      `K_TILE` is a launch argument; the host sizes it (and the LDS) from
//      `head_dim`.
//
// K AND V are both staged in LDS. K was always; V is the newer fix, and the
// reason is latency rather than bandwidth. The P@V pass reads A (16) V values
// per key per lane, all independent, so the compiler can issue them back to
// back — but each one was an L2/DRAM read of ~500 cycles against a block with
// two warps per SIMD, and that latency was the measured bottleneck: the kernel
// ran at 1.5% of its FMA issue rate while moving only 33 GB/s (rocprofv3,
// gfx1101, N=4096). Through LDS the same reads are ~30 cycles. Staging V costs
// LDS, so K_TILE drops to 16 (K_TILE/LANES = 1 score per lane, twice the
// tiles) — a trade this kernel makes deliberately, and the host sizes both.
//
// LATENCY, NOT BANDWIDTH (rocprofv3, gfx1101, N=4096): the first version of
// this kernel land measured 531 GFLOP/s — 1.8% of the card's f32 peak — while
// only moving 33 GB/s of DRAM. It was stalled, not starved. Two causes, both
// fixed here:
//   1. the score dot was a 256-long chain through ONE accumulator, so ~1100
//      cycles of FMA latency were exposed per score with only 2 warps per
//      SIMD to hide them. It now runs 4 independent accumulators over
//      `float4` loads.
//   2. the running output lived in LDS, which pushed the block over half the
//      64 KB budget and pinned occupancy at one block per CU. It now lives in
//      registers (out_reg), which frees 16 KB at head_dim 256 and lets the K
//      tile grow.
//
// Computes: out[qi, h] = softmax(Q[qi, h] @ K[:, kv(h)]^T * scale) @ V[:, kv(h)]
//
// Grid:  [n_heads, ceil(N / QB)]
// Block: [QB * LANES]  — each LANES-lane subgroup owns one query row.
// LDS:   2*K_TILE*(head_dim+4) + QB*head_dim + QB*K_TILE + 2*QB*LANES + 2*QB
// floats (the running output is register-resident, so it costs no LDS; the K
// and V strides each carry 4 floats of padding against LDS bank conflicts)

// K-tile row stride padding. With stride == head_dim (256) two different key
// rows sit exactly 8 LDS rows (1024 B) apart, so the 16 lanes of a subgroup —
// which read the SAME `d` of DIFFERENT `j` — all land in one bank: a 16-way
// conflict on every K read of the score pass. Padding the stride by 4 floats
// (16 B, keeps float4 alignment) spreads consecutive rows across banks.
#define K_PAD 4

#define QB 16      // queries per block (one LANES-lane subgroup each)
#define LANES 16   // threads per query subgroup
#define A_MAX 16   // max head_dim/LANES (supports head_dim up to 256)

extern "C" __global__ __launch_bounds__(QB* LANES, 1) void attention_encoder_qtiled_f32(
    const float* __restrict__ q,     // [N, n_heads    * head_dim]
    const float* __restrict__ k,     // [N, n_kv_heads * head_dim]
    const float* __restrict__ v,     // [N, n_kv_heads * head_dim]
    float* __restrict__ out,         // [N, n_heads    * head_dim]
    int N, int n_valid, int n_heads, int n_kv_heads, int head_dim, float scale, int k_tile_arg) {
    const int head = (int)blockIdx.x;
    const int q0 = (int)blockIdx.y * QB;
    const int nq = min(QB, N - q0);
    const int tid = (int)threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int K_TILE = k_tile_arg;
    // The score pass computes K_TILE/LANES scores per lane; a smaller tile
    // would compute none, and a non-multiple would drop tail keys. The host
    // validates via encoder_qtiled_tile before launch, but if a future
    // caller bypasses it, refuse loudly: write zeros rather than leave
    // whatever the allocation contained.
    if (K_TILE < LANES || (K_TILE % LANES) != 0) {
        const int d_q0 = n_heads * head_dim;
        for (int i = tid; i < nq * head_dim; i += nthreads) {
            int qi = i / head_dim, d = i % head_dim;
            out[(size_t)(q0 + qi) * d_q0 + head * head_dim + d] = 0.0f;
        }
        return;
    }
    // Keys in [n_valid, N) are sequence padding (zeroed rows): they must not
    // take a softmax share. Queries there are also padding — computed harm-
    // lessly and dropped by the caller.
    const int kn = min(n_valid, N);

    const int d_q = n_heads * head_dim;
    const int d_kv = n_kv_heads * head_dim;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int q_off = head * head_dim;
    const int kv_off = kv_head * head_dim;
    const int A = head_dim / LANES;  // out elements owned per lane

    const int K_STRIDE = head_dim + K_PAD;
    extern __shared__ float smem[];
    float* k_tile = smem;                              // [K_TILE * K_STRIDE]
    float* v_tile = k_tile + K_TILE * K_STRIDE;        // [K_TILE * K_STRIDE]
    float* q_tile = v_tile + K_TILE * K_STRIDE;        // [QB * head_dim]
    float* s_tile = q_tile + QB * head_dim;            // [QB * K_TILE]
    float* ws_max = s_tile + QB * K_TILE;              // [QB * LANES]
    float* ws_sum = ws_max + QB * LANES;               // [QB * LANES]
    float* m_l = ws_sum + QB * LANES;                  // [QB * 2] (running max, sum)

    const int lane = tid % LANES;
    const int qid = tid / LANES;
    const bool alive = qid < nq;

    // Load this block's Q rows once.
    for (int i = tid; i < nq * head_dim; i += nthreads) {
        int qi = i / head_dim, d = i % head_dim;
        q_tile[qi * head_dim + d] = q[(size_t)(q0 + qi) * d_q + q_off + d];
    }
    if (alive && lane == 0) {
        m_l[qid * 2 + 0] = -INFINITY;
        m_l[qid * 2 + 1] = 0.0f;
    }
    // Running output in registers: this lane owns `A` elements of its query
    // (d = lane + a*LANES), so there is no cross-lane sharing to pay LDS for.
    float out_reg[A_MAX];
    for (int a = 0; a < A; a++) out_reg[a] = 0.0f;

    float s_reg[A_MAX];
    __syncthreads();

    for (int t0 = 0; t0 < kn; t0 += K_TILE) {
        const int tlen = min(K_TILE, kn - t0);

        // Cooperative K tile load (coalesced over head_dim).
        for (int i = tid; i < tlen * head_dim; i += nthreads) {
            int j = i / head_dim, d = i % head_dim;
            k_tile[j * K_STRIDE + d] = k[(size_t)(t0 + j) * d_kv + kv_off + d];
            v_tile[j * K_STRIDE + d] = v[(size_t)(t0 + j) * d_kv + kv_off + d];
        }
        __syncthreads();

        if (alive) {
            // Phase A: this lane's scores for the tile, in registers.
            float smax = -INFINITY;
            for (int jj = 0; jj < K_TILE / LANES; jj++) {
                int j = lane + jj * LANES;
                if (j < tlen) {
                    const float* qrow = q_tile + qid * head_dim;
                    const float* krow = k_tile + j * K_STRIDE;
                    // Four independent accumulator chains over float4 loads.
                    // A single accumulator exposes the full 256-step FMA
                    // latency chain (~1100 cycles) and was the measured
                    // bottleneck at 2 warps/SIMD.
                    const float4* qv = (const float4*)qrow;
                    const float4* kv4 = (const float4*)krow;
                    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                    const int nq4 = head_dim >> 2;
                    #pragma unroll 4
                    for (int q4 = 0; q4 < nq4; q4++) {
                        const float4 qa = qv[q4];
                        const float4 ka = kv4[q4];
                        a0 = fmaf(qa.x, ka.x, a0);
                        a1 = fmaf(qa.y, ka.y, a1);
                        a2 = fmaf(qa.z, ka.z, a2);
                        a3 = fmaf(qa.w, ka.w, a3);
                    }
                    const float dot = (a0 + a1) + (a2 + a3);
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
            // Tile max -> running max, shared through LDS (every lane of the
            // subgroup recomputes the same m_new).
            float tmax = -INFINITY;
            for (int l = 0; l < LANES; l++) tmax = fmaxf(tmax, ws_max[qid * LANES + l]);
            const float m_old = m_l[qid * 2 + 0];
            const float m_new = fmaxf(m_old, tmax);

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
            for (int l = 0; l < LANES; l++) tmax = fmaxf(tmax, ws_max[qid * LANES + l]);
            const float m_old = m_l[qid * 2 + 0];
            const float m_new = fmaxf(m_old, tmax);
            const float r = expf(m_old - m_new);
            float tsum = 0.0f;
            for (int l = 0; l < LANES; l++) tsum += ws_sum[qid * LANES + l];

            // Rescale the running output and fold in this tile's P@V.
            const float* prow = s_tile + qid * K_TILE;
            // Rescale this lane's running output first ...
            #pragma unroll
            for (int a = 0; a < A; a++) out_reg[a] *= r;
            // ... then fold in P@V with `p` hoisted into a register and V read
            // from LDS: reading `p` inside the `a` loop re-read it A (16) times
            // per key for one multiply, and reading V from global put a
            // full-memory-latency load under every one of those multiplies.
            for (int j = 0; j < tlen; j++) {
                const float p = prow[j];
                const float* vrow = v_tile + j * K_STRIDE;
                #pragma unroll
                for (int a = 0; a < A; a++)
                    out_reg[a] = fmaf(p, vrow[lane + a * LANES], out_reg[a]);
            }
            if (lane == 0) {
                m_l[qid * 2 + 0] = m_new;
                m_l[qid * 2 + 1] = m_l[qid * 2 + 1] * r + tsum;
            }
        }
        __syncthreads();
    }

    // Normalize and store.
    if (alive) {
        const float l = m_l[qid * 2 + 1];
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (int a = 0; a < A; a++) {
            int d = lane + a * LANES;
            out[(size_t)(q0 + qid) * d_q + q_off + d] = out_reg[a] * inv;
        }
    }
}

