// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched MoE top-K + renorm. Companion to moe_topk_renorm_k8 for the
// prefill path.
//
// One workgroup per token. blockIdx.x selects the token row in
// [N × n_exp] input. Same algorithm as the single-token kernel, just
// strided by bid * n_exp / bid * K_TOP.
//
// fuse_softmax == 0 (original behavior, block 256): `probs_in` holds rows that
// were already softmaxed by a separate softmax_f32 launch.
//
// fuse_softmax != 0 (prefill fusion, block = blockDim.x physical lanes):
// `probs_in` holds raw router logits and the kernel computes the softmax
// in-block, then the same top-8 + renorm. The result is byte-identical to the
// unfused softmax_f32 -> mode-0 pair because each virtual 256-thread block of
// the incumbent kernels is emulated stage for stage by the physical lanes
// (virtual thread t handled by lane t % blockDim.x via
// `t = tid; t < N; t += blockDim.x`), with a barrier after every reduction
// stage so the exact tree order, the exact strided accumulation order, and the
// exact (non-index-ascending) tie resolution of the 256-thread tree are
// preserved. No reciprocal, no shuffle/DPP reassociation, no approximation.
// The fused path is correct for any blockDim.x >= 8 (the final clamp/store
// uses lanes 0..K_TOP-1) and <= BLOCK_SIZE; the host launcher narrows the
// valid set to 32/64/128/256 and rejects everything else. The block size is
// launch geometry only (not a kernel argument).
//
// moe_router_softmax_topk_k8_wave (below, same module) is the A3B prefill
// route: one wave32 per row, 8 rows per 256-thread block, n_exp == 256. It
// reproduces the result of the fused path above bit for bit (see its header);
// this symbol keeps the one-row-per-block modes 0 and 1.

#define BLOCK_SIZE 256
#define MAX_NEXP   1024
#define K_TOP      8

extern "C" __launch_bounds__(BLOCK_SIZE, 2)
__global__ void moe_topk_renorm_k8_batched(
    const float* __restrict__ probs_in,    // [N × n_exp] probs (mode 0) / logits (mode 1)
    int*         __restrict__ topk_idx,    // [N × 8]
    float*       __restrict__ topk_w,      // [N × 8]
    int n_exp,
    int norm_topk,
    int fuse_softmax
) {
    __shared__ float smem_v[MAX_NEXP];
    __shared__ int   smem_i[MAX_NEXP];     // mode 0 only
    __shared__ float reduce_v[BLOCK_SIZE];
    __shared__ int   reduce_i[BLOCK_SIZE];
    __shared__ int   picked_idx[K_TOP];
    __shared__ float picked_w[K_TOP];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const float* __restrict__ row_probs = probs_in + (long long)bid * n_exp;
    int*   __restrict__ row_idx = topk_idx + bid * K_TOP;
    float* __restrict__ row_w   = topk_w   + bid * K_TOP;

    if (fuse_softmax) {
        // ---- Fused prefill path: block = blockDim.x lanes, emulating virtual 256 ----
        //
        // Softmax: bit-for-bit the incumbent kernels/src/softmax.hip
        // (softmax_f32 launched with block = min(256, n_exp)). `vblock` is that
        // virtual block size.
        const int vblock = n_exp < 256 ? n_exp : 256;

        // Row max. Each virtual thread folds i = t, t+vblock, ... starting
        // from -1e30f; reduced by the incumbent tree s = vblock/2 .. 1.
        for (int t = tid; t < vblock; t += blockDim.x) {
            float m = -1e30f;
            for (int i = t; i < n_exp; i += vblock) {
                m = fmaxf(m, row_probs[i]);
            }
            reduce_v[t] = m;
        }
        __syncthreads();
        for (int s = vblock >> 1; s > 0; s >>= 1) {
            for (int t = tid; t < s; t += blockDim.x) {
                reduce_v[t] = fmaxf(reduce_v[t], reduce_v[t + s]);
            }
            __syncthreads();
        }
        const float max_val = reduce_v[0];
        __syncthreads();

        // exp(x - max) per element, accumulated per virtual thread in the
        // incumbent strided increasing-index order from 0.0f, then the same
        // tree. The incumbent stores expf() back to the row and re-reads it
        // (an exact fp32 round trip); here it is kept in smem_v.
        for (int t = tid; t < vblock; t += blockDim.x) {
            float sum = 0.0f;
            for (int i = t; i < n_exp; i += vblock) {
                const float e = expf(row_probs[i] - max_val);
                smem_v[i] = e;
                sum += e;
            }
            reduce_v[t] = sum;
        }
        __syncthreads();
        for (int s = vblock >> 1; s > 0; s >>= 1) {
            for (int t = tid; t < s; t += blockDim.x) {
                reduce_v[t] += reduce_v[t + s];
            }
            __syncthreads();
        }
        const float sum_all = reduce_v[0];
        __syncthreads();

        // Direct divide (no reciprocal multiply), as softmax_f32 does.
        for (int i = tid; i < n_exp; i += blockDim.x) {
            smem_v[i] = smem_v[i] / sum_all;
        }
        __syncthreads();

        // Top-8: the incumbent mode-0 algorithm with its 256 virtual threads
        // (always 256 here, independent of n_exp). Local scan is strict `>`
        // over i = t, t+256, ...; the reduction tree s = 128..1 replaces only
        // on strictly greater (`vb > va`). Equal values therefore resolve by
        // tree position, NOT by lowest index (all-equal picks 0,128,64,192,..),
        // and that tree is reproduced exactly.
        for (int k = 0; k < K_TOP; k++) {
            for (int t = tid; t < BLOCK_SIZE; t += blockDim.x) {
                float best_v = -INFINITY;
                int   best_i = -1;
                for (int i = t; i < n_exp; i += BLOCK_SIZE) {
                    const float v = smem_v[i];
                    if (v > best_v) { best_v = v; best_i = i; }
                }
                reduce_v[t] = best_v;
                reduce_i[t] = best_i;
            }
            __syncthreads();
            for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
                for (int t = tid; t < s; t += blockDim.x) {
                    const float va = reduce_v[t];
                    const float vb = reduce_v[t + s];
                    if (vb > va) {
                        reduce_v[t] = vb;
                        reduce_i[t] = reduce_i[t + s];
                    }
                }
                __syncthreads();
            }
            const int   winner   = reduce_i[0];
            const float winner_w = reduce_v[0];
            if (tid == 0) {
                picked_idx[k] = winner;
                picked_w[k]   = winner_w;
                if (winner >= 0 && winner < n_exp) {
                    smem_v[winner] = -INFINITY;
                }
            }
            __syncthreads();
        }

        // Renorm: sequential sum of the eight picks from 0.0f, direct divide.
        if (tid == 0) {
            float s = 0.0f;
            for (int k = 0; k < K_TOP; k++) s += picked_w[k];
            if (norm_topk && s > 0.0f) {
                for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
            }
        }
        __syncthreads();

        // Same NaN/OOB clamp as the mode-0 path.
        if (tid < K_TOP) {
            int   safe_idx = picked_idx[tid];
            float safe_w   = picked_w[tid];
            if (safe_idx < 0 || safe_idx >= n_exp) {
                safe_idx = 0;
                safe_w   = 0.0f;
            }
            row_idx[tid] = safe_idx;
            row_w[tid]   = safe_w;
        }
        return;
    }

    for (int i = tid; i < MAX_NEXP; i += BLOCK_SIZE) {
        smem_v[i] = (i < n_exp) ? row_probs[i] : -INFINITY;
        smem_i[i] = i;
    }
    __syncthreads();

    for (int k = 0; k < K_TOP; k++) {
        float best_v = -INFINITY;
        int   best_i = -1;
        for (int i = tid; i < n_exp; i += BLOCK_SIZE) {
            float v = smem_v[i];
            if (v > best_v) { best_v = v; best_i = smem_i[i]; }
        }
        reduce_v[tid] = best_v;
        reduce_i[tid] = best_i;
        __syncthreads();

        for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                float va = reduce_v[tid];
                float vb = reduce_v[tid + s];
                if (vb > va) {
                    reduce_v[tid] = vb;
                    reduce_i[tid] = reduce_i[tid + s];
                }
            }
            __syncthreads();
        }
        const int   winner = reduce_i[0];
        const float winner_w = reduce_v[0];
        if (tid == 0) {
            picked_idx[k] = winner;
            picked_w[k]   = winner_w;
        }
        if (winner >= 0 && winner < n_exp) {
            if ((winner & (BLOCK_SIZE - 1)) == tid) {
                smem_v[winner] = -INFINITY;
            }
        }
        __syncthreads();
    }

    // Renorm via direct division (matches CPU reference; mul-by-reciprocal
    // adds 1 ULP/element which the MoE layer stack amplifies).
    if (tid == 0) {
        float s = 0.0f;
        for (int k = 0; k < K_TOP; k++) s += picked_w[k];
        if (norm_topk && s > 0.0f) {
            for (int k = 0; k < K_TOP; k++) picked_w[k] /= s;
        }
    }
    __syncthreads();

    // Defense-in-depth NaN/OOB clamp (issue #152 sibling).
    if (tid < K_TOP) {
        int   safe_idx = picked_idx[tid];
        float safe_w   = picked_w[tid];
        if (safe_idx < 0 || safe_idx >= n_exp) {
            safe_idx = 0;
            safe_w   = 0.0f;
        }
        row_idx[tid] = safe_idx;
        row_w[tid]   = safe_w;
    }
}

// ---------------------------------------------------------------------------
// moe_router_softmax_topk_k8_wave: fused softmax + top-8 + renorm, one wave32
// per row, n_exp == 256 (A3B), k == 8. Byte-identical to moe_topk_renorm_k8_batched
// mode 1 (and so to softmax_f32 + mode 0): every float operation of a row keeps
// the incumbent value and operand order, only the thread that computes it moves.
//
// Lane l of the wave holds experts 4l..4l+3 (x[0..3]) and 128+4l..128+4l+3
// (x[4..7]) from two float4 loads. The incumbent's 256 virtual threads own one
// expert each (virtual thread t = expert t), so in this layout:
//   * max: order independent (fmaxf over a set, seeded with -1e30f like the
//     incumbent's per-thread seed) -> in-lane fold + xor butterfly.
//   * sum: the incumbent tree reduce[t] += reduce[t+s], s = 128..1, is
//     s=128 in-lane (x[c] + x[4+c]), s=64/32/16/8/4 as shfl_down by 16/8/4/2/1
//     lanes (t+s lives in lane l + s/4, same c), s=2/1 in-lane in lane 0.
//     Each add is the same two operands in the same order (v[t] + v[t+s]).
//   * exp(x - max) and the direct divide by the tree sum are per element.
//   * top-8: the incumbent scan + tree s = 128..1 replaces only on strictly
//     greater, so the winner among equal values is the one with the smallest
//     bit-reversed (8-bit) expert index (bit 0 is the last tree level and has
//     top priority: all-equal rows pick 0,128,64,192,...). Each lane keeps
//     its 8 candidates with that priority; a xor butterfly over (value,
//     priority) pairs, a strict total order, yields the same winner on every
//     lane. NaN probabilities never win (v > -inf is false), exactly as in
//     the incumbent scan; a round with no valid candidate records (-1, -inf)
//     and the common clamp maps it to (0, 0.0f).
//   * renorm: sequential sum of the eight picks from 0.0f, direct divide, same
//     clamp as the other modes.
// No barriers: a wave only talks to itself, so waves past the last row just exit.
// Wave32 only (the launcher rejects other devices).
// ---------------------------------------------------------------------------

#define WAVE_ROWS 8  // rows (waves) per 256-thread block
#define WAVE_NEXP 256

__device__ __forceinline__ bool topk_wave_better(float va, int pa, float vb, int pb) {
    return va > vb || (va == vb && pa < pb);
}

extern "C" __launch_bounds__(WAVE_ROWS * 32)
__global__ void moe_router_softmax_topk_k8_wave(
    const float* __restrict__ logits,      // [n_rows x 256] raw router logits (left untouched)
    int*         __restrict__ topk_idx,    // [n_rows x 8]
    float*       __restrict__ topk_w,      // [n_rows x 8]
    int n_rows,
    int norm_topk
) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * WAVE_ROWS + (threadIdx.x >> 5);
    if (row >= n_rows) return;

    const float4* __restrict__ rp =
        reinterpret_cast<const float4*>(logits + (long long)row * WAVE_NEXP);
    const float4 qa = rp[lane];
    const float4 qb = rp[32 + lane];
    float x[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};

    // Row max.
    float m = -1e30f;
    #pragma unroll
    for (int j = 0; j < 8; j++) m = fmaxf(m, x[j]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off));
    const float max_val = m;

    // exp(x - max), tree sum.
    float e[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) e[j] = expf(x[j] - max_val);
    float u[4];
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = e[c] + e[4 + c];                  // s = 128
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = u[c] + __shfl_down(u[c], 16);     // s = 64
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = u[c] + __shfl_down(u[c], 8);      // s = 32
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = u[c] + __shfl_down(u[c], 4);      // s = 16
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = u[c] + __shfl_down(u[c], 2);      // s = 8
    #pragma unroll
    for (int c = 0; c < 4; c++) u[c] = u[c] + __shfl_down(u[c], 1);      // s = 4
    const float q0 = u[0] + u[2];                                        // s = 2
    const float q1 = u[1] + u[3];
    const float sum_all = __shfl(q0 + q1, 0);                            // s = 1

    // Probabilities, candidate priorities.
    float v[8];
    int   pr[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const float p = e[j] / sum_all;
        v[j] = (p > -INFINITY) ? p : -INFINITY;
        const unsigned pos = (j < 4 ? 0u : 128u) + 4u * (unsigned)lane + (unsigned)(j & 3);
        pr[j] = (int)(__brev(pos) >> 24);
    }

    int   pick_i[K_TOP];
    float pick_w[K_TOP];
    #pragma unroll
    for (int k = 0; k < K_TOP; k++) {
        float bv = v[0];
        int   bp = pr[0];
        #pragma unroll
        for (int j = 1; j < 8; j++) {
            if (topk_wave_better(v[j], pr[j], bv, bp)) { bv = v[j]; bp = pr[j]; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off);
            const int   op = __shfl_xor(bp, off);
            if (topk_wave_better(ov, op, bv, bp)) { bv = ov; bp = op; }
        }
        pick_i[k] = (bv > -INFINITY) ? (int)(__brev((unsigned)bp) >> 24) : -1;
        pick_w[k] = bv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            if (pr[j] == bp) v[j] = -INFINITY;
        }
    }

    // Renorm: sequential sum from 0.0f, direct divide.
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < K_TOP; k++) s += pick_w[k];
    if (norm_topk && s > 0.0f) {
        #pragma unroll
        for (int k = 0; k < K_TOP; k++) pick_w[k] /= s;
    }

    int   oi = 0;
    float ow = 0.0f;
    #pragma unroll
    for (int k = 0; k < K_TOP; k++) {
        if (lane == k) {
            const bool ok = pick_i[k] >= 0 && pick_i[k] < WAVE_NEXP;
            oi = ok ? pick_i[k] : 0;
            ow = ok ? pick_w[k] : 0.0f;
        }
    }
    if (lane < K_TOP) {
        topk_idx[row * K_TOP + lane] = oi;
        topk_w[row * K_TOP + lane]   = ow;
    }
}
