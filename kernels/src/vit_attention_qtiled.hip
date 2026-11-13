// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Q-tiled bidirectional attention for the vision encoder (flash-style
// online softmax). Replaces the per-(head, query) `vit_attention_f32`,
// whose blocks each re-read all of K and V from DRAM — ~220 GB of traffic
// per tower layer at N=4624 (measured 1.02 s/layer on gfx1101). Here QB
// queries of one head share every K/V tile pulled through LDS: DRAM
// traffic drops by ~QB.
//
// Four rules keep the block fast, each learned from the sister
// `attention_encoder_qtiled_f32` (same shape class, arch 17):
//   1. K/Q/V row strides are padded by K_PAD floats. With stride == head_dim
//      (64) the 16 lanes of a subgroup — reading the SAME `d` of DIFFERENT
//      rows — land in one bank: a 16-way conflict on every score-pass read.
//      Measured on gfx1101 at N=7564: 471.3 -> 170.5 ms per call.
//   2. V is staged in LDS alongside K. The P@V pass reads each (j, d) once
//      per lane, but straight from global that is a full-memory-latency
//      load under every FMA; the encoder kernel measured 1.5% FMA issue and
//      33 GB/s with global V. Staging V costs LDS, so K_TILE drops to 32.
//   3. The dot product uses float4 LDS loads with four independent
//      accumulator chains (a single accumulator exposes the FMA latency
//      chain), and the running output lives in registers, not LDS.
//   4. QB is a launch parameter, host-sized to the largest value whose LDS
//      carve-up fits 64 KB (64 at head_dim<=64, 32 above). One K/V tile then
//      serves up to 64 queries instead of 16. Measured N=7564/hd=64:
//      170.5 -> 126.0 ms per call (3.7x the pre-padding kernel).
//
// head_dim is any value in 1..=128: per-lane accumulators are sized by
// the rounded-up A, and every d the lane owns is guarded (`d < head_dim`)
// in the P@V fold and the store, so head_dims that are not multiples of
// 16 (the Qwen ViT's 72) keep the Q-tiled path as the non-WMMA fallback.
// The float4 score pass is only taken when head_dim % 4 == 0, where the
// padded row stride stays 16-byte aligned; other widths use a scalar dot.
//
// This kernel is the fallback exactly on the arches without wave32 WMMA
// (gfx9/CDNA are wave64), so its block size is wave-width aware: 1024
// threads on wave32, 512 on wave64. A 1024-thread block on wave64 needs
// 16 waves on one CU, which caps __launch_bounds__ at 64 VGPRs/thread;
// measured on gfx906 that spilled 6 VGPRs and a 28 B private segment (21
// `v_accvgpr` moves on gfx908). 512 threads leaves 128 VGPRs and fits the
// 71-VGPR kernel without scratch. The host mirrors MAX_THREADS in
// `vit_qtiled_launch`, so the query tile shrinks to 32 there.
//
// The LDS carve-up (below) is the kernel's own, and the host sizes the
// dynamic allocation from `vit_qtiled_launch`. The two are compared at
// launch (`lds_limit`): a constant edited on only one side used to write
// past the allocation and read another region's data — silently, with
// plausible numbers — so it traps instead.
//
// LDS (QB=64, K_TILE=32, hd=64): 51,968 B; 71 VGPR, no scratch, one
// 1024-thread block per CU (wave32).
//
// Computes: out[qi, head] = softmax(Q[qi] @ K^T / sqrt(d)) @ V   (no mask)
//
// Grid:  [num_heads, ceil(N / qpb)]
// Block: [qpb * LANES]   — each 16-lane subgroup owns one query for the
//                        — whole tile (256..1024 threads; 1024 wave32).

#define LANES 16   // threads per query subgroup
#define K_TILE 32  // keys streamed through LDS per iteration
#define A_MAX 8    // max ceil(head_dim/LANES) (supports head_dim up to 128)
#define K_PAD 4    // row-stride padding (16 B: keeps float4 LDS alignment)
#define S_PAD 1    // probability-tile stride padding across subgroups

// Biggest legal block. gfx9 / CDNA (wave64) halves it to keep 128 VGPRs.
// `__AMDGCN_WAVEFRONT_SIZE` is not defined by this toolchain (clang 24), so
// the generation macro is the discriminator: __GFX9__ is wave64, and
// __GFX10__/__GFX11__/__GFX12__ are wave32.
#if defined(__GFX9__)
#define VIT_QTLED_MAX_THREADS 512
#else
#define VIT_QTLED_MAX_THREADS 1024
#endif

static_assert(K_TILE % LANES == 0, "K_TILE must be a multiple of LANES");
static_assert(K_PAD % 4 == 0, "K_PAD keeps the padded row stride float4-aligned");
static_assert(A_MAX * LANES >= 128, "A_MAX must cover head_dim up to 128");
static_assert(VIT_QTLED_MAX_THREADS % LANES == 0, "block must split into subgroups");

// Bytes of `extern __shared__` this block's carve-up needs. Must match the
// pointer arithmetic below; the host allocates at least this much.
__device__ __forceinline__ int vit_qtiled_lds_bytes(int nqb, int hd) {
    return (2 * K_TILE * (hd + K_PAD)   // k_tile, v_tile
            + nqb * (hd + K_PAD)        // q_tile
            + nqb * (K_TILE + S_PAD)    // s_tile
            + 2 * nqb * LANES           // ws_max, ws_sum
            + 2 * nqb)                  // m_l
           * (int)sizeof(float);
}

extern "C" __global__ __launch_bounds__(VIT_QTLED_MAX_THREADS, 1) void vit_attention_qtiled_f32(
    const float* __restrict__ qkv,   // [N, 3*hidden] row-major, interleaved Q/K/V
    float* __restrict__ out,         // [N, hidden] row-major
    int N, int hidden, int num_heads, int head_dim, float scale, int qpb,
    int lds_limit)                   // dynamic LDS bytes this launch allocated
{
    const int head = blockIdx.x;
    const int nqb = qpb;              // queries per block, host-sized (16..64)
    const int q0 = blockIdx.y * nqb;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    if (head >= num_heads || q0 >= N) return;
    // Launch-contract violations are host bugs, and every one of them
    // corrupts silently if the block simply carries on: a qpb that does not
    // match blockDim leaves queries unwritten, and an undersized LDS
    // allocation makes the carve-up below overlap itself. Trap (a queue
    // failure the caller cannot mistake for a result) instead of guessing.
    if (nqb < LANES || (nqb % LANES) != 0 || nqb * LANES > VIT_QTLED_MAX_THREADS ||
        nqb * LANES != (int)blockDim.x ||
        vit_qtiled_lds_bytes(nqb, head_dim) > lds_limit) {
        __builtin_trap();
    }

    const int qid = tid / LANES;      // which query this subgroup owns
    const int lane = tid % LANES;
    const int qi = q0 + qid;
    const int nq = min(nqb, N - q0);  // queries live in this block
    const bool alive = qid < nq;

    const int stride = 3 * hidden;
    const int q_off = head * head_dim;
    const int k_off = hidden + head * head_dim;
    const int v_off = 2 * hidden + head * head_dim;
    // Out elements owned per lane; the tail above head_dim is skipped by
    // the `d < head_dim` guards below.
    const int A = (head_dim + LANES - 1) / LANES;

    const int K_STRIDE = head_dim + K_PAD;
    const int Q_STRIDE = head_dim + K_PAD;
    const int S_STRIDE = K_TILE + S_PAD;

    extern __shared__ float smem[];
    float* k_tile = smem;                             // [K_TILE * K_STRIDE]
    float* v_tile = k_tile + K_TILE * K_STRIDE;       // [K_TILE * K_STRIDE]
    float* q_tile = v_tile + K_TILE * K_STRIDE;       // [nqb * Q_STRIDE]
    float* s_tile = q_tile + nqb * Q_STRIDE;          // [nqb * S_STRIDE]
    float* ws_max = s_tile + nqb * S_STRIDE;          // [nqb * LANES]
    float* ws_sum = ws_max + nqb * LANES;             // [nqb * LANES]
    float* m_l    = ws_sum + nqb * LANES;             // [nqb * 2] running (max, sum)

    // Load this block's Q rows once.
    for (int i = tid; i < nq * head_dim; i += nthreads) {
        int q = i / head_dim, d = i % head_dim;
        q_tile[q * Q_STRIDE + d] = qkv[(q0 + q) * stride + q_off + d];
    }
    if (alive && lane == 0) {
        m_l[qid * 2 + 0] = -INFINITY;
        m_l[qid * 2 + 1] = 0.0f;
    }
    // Running output in registers: this lane owns up to `A` elements of its
    // query (d = lane + a*LANES), so there is no cross-lane sharing to pay
    // LDS for.
    float out_reg[A_MAX];
    for (int a = 0; a < A; a++) out_reg[a] = 0.0f;

    float s_reg[A_MAX];  // this lane's scores/p for the current tile
    __syncthreads();

    for (int t0 = 0; t0 < N; t0 += K_TILE) {
        const int tlen = min(K_TILE, N - t0);

        // Cooperative K+V tile load (coalesced over head_dim).
        for (int i = tid; i < tlen * head_dim; i += nthreads) {
            int j = i / head_dim, d = i % head_dim;
            k_tile[j * K_STRIDE + d] = qkv[(t0 + j) * stride + k_off + d];
            v_tile[j * K_STRIDE + d] = qkv[(t0 + j) * stride + v_off + d];
        }
        __syncthreads();

        if (alive) {
            // Phase A: full scores for this lane's K rows (registers).
            float smax = -INFINITY;
            for (int jj = 0; jj < K_TILE / LANES; jj++) {
                int j = lane + jj * LANES;
                if (j < tlen) {
                    const float* qrow = q_tile + qid * Q_STRIDE;
                    const float* krow = k_tile + j * K_STRIDE;
                    float dot;
                    if ((head_dim & 3) == 0) {
                        // Four independent accumulator chains over float4 LDS
                        // loads: a single accumulator exposes the full FMA
                        // latency chain and was the measured bottleneck at
                        // two warps/SIMD. 16-byte alignment holds because
                        // head_dim % 4 == 0 keeps Q_STRIDE a float4 multiple.
                        const float4* qv = (const float4*)qrow;
                        const float4* kv4 = (const float4*)krow;
                        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                        const int nq4 = head_dim >> 2;
                        #pragma unroll 4
                        for (int d4 = 0; d4 < nq4; d4++) {
                            const float4 qa = qv[d4];
                            const float4 ka = kv4[d4];
                            a0 = fmaf(qa.x, ka.x, a0);
                            a1 = fmaf(qa.y, ka.y, a1);
                            a2 = fmaf(qa.z, ka.z, a2);
                            a3 = fmaf(qa.w, ka.w, a3);
                        }
                        dot = (a0 + a1) + (a2 + a3);
                    } else {
                        // Widths that are not float4 multiples (e.g. 70) fall
                        // back to one scalar chain; correctness over speed,
                        // since no VisionTower contract ships such a head.
                        float acc = 0.0f;
                        for (int d = 0; d < head_dim; d++)
                            acc = fmaf(qrow[d], krow[d], acc);
                        dot = acc;
                    }
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
                s_tile[qid * S_STRIDE + j] = p;
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

            // Rescale this lane's running output, then fold in P@V with `p`
            // hoisted into a register and V read from LDS (reading `p` inside
            // the `a` loop re-read it A times per key for one multiply).
            // Exact widths (head_dim % LANES == 0, every shipped tower but
            // the Qwen ViT's 72) take a guard-free loop: the per-element
            // `d < head_dim` compare measured 122.0 -> 140.9 ms/call at
            // N=7564 when it ran on every shape.
            const float* prow = s_tile + qid * S_STRIDE;
            const bool exact = (head_dim % LANES) == 0;
            #pragma unroll
            for (int a = 0; a < A; a++) out_reg[a] *= r;
            if (exact) {
                for (int j = 0; j < tlen; j++) {
                    const float p = prow[j];
                    const float* vrow = v_tile + j * K_STRIDE;
                    #pragma unroll
                    for (int a = 0; a < A; a++)
                        out_reg[a] = fmaf(p, vrow[lane + a * LANES], out_reg[a]);
                }
            } else {
                for (int j = 0; j < tlen; j++) {
                    const float p = prow[j];
                    const float* vrow = v_tile + j * K_STRIDE;
                    #pragma unroll
                    for (int a = 0; a < A; a++) {
                        const int d = lane + a * LANES;
                        if (d < head_dim)
                            out_reg[a] = fmaf(p, vrow[d], out_reg[a]);
                    }
                }
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
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        float* orow = out + qi * hidden + head * head_dim;
        if ((head_dim % LANES) == 0) {
            for (int a = 0; a < A; a++) orow[lane + a * LANES] = out_reg[a] * inv;
        } else {
            for (int a = 0; a < A; a++) {
                const int d = lane + a * LANES;
                if (d < head_dim) orow[d] = out_reg[a] * inv;
            }
        }
    }
}
