// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Tie-row recompute of the raw top-K selector (`topk_values_regrid_f32` marks
// a row with top_idx[b*K] == -1 when a tie sits among its leading K+1 values).
// Output is byte-identical to the shipping kernel body
// (`topk_logsumexp_batched_f32`, raw mode, kept behind `topk_values_fixup_f32`),
// which ran this row on 256 threads and then merged 256 x K candidates on ONE
// thread (~1.6 ms per window for a 260K vocabulary).
//
// Per row, one block of 256 threads, the shipping layout:
//   Pass A: thread t scans i = t, t+256, ... and keeps its K best (val, idx) in
//           registers: a candidate replaces the FIRST minimal slot iff v > min.
//           (Identical to the shipping scan; the minimum is tracked instead of
//           rescanned per element, which changes nothing because the slots only
//           change on a replacement.)
//   Merge:  the shipping thread 0 inserted the candidates in order
//           c = t * K + slot into a sorted-desc list of K with a strict `>`
//           (a tie goes behind the earlier arrival; -inf and NaN are never
//           inserted, an unfilled list entry stays (-inf, 0)). That list is
//           exactly the K best of the candidates under the total order
//           (value desc, c asc). K block-wide rounds pick the best remaining
//           candidate under that order instead: per-thread best -> wave
//           butterfly -> cross-wave stage; the winner clears its slot.
//
// No floating-point arithmetic: outputs are selections of input values.

#define MAX_K 16
#define NTH 256

__device__ __forceinline__ bool tie_better(float va, int ta, float vb, int tb) {
    return va > vb || (va == vb && ta < tb);
}

extern "C" __launch_bounds__(NTH, 4)
__global__ void topk_values_tie_exact_f32(
    const float* __restrict__ logits,  // [B x vocab]
    int* __restrict__ top_idx,         // [B x K]
    float* __restrict__ top_val,       // [B x K]
    int vocab,
    int K) {
    // Only rows the re-grid kernel marked. Every thread reads the marker before
    // the first barrier, so the final stores of thread 0 cannot race it.
    if (top_idx[(size_t)blockIdx.x * K] != -1) return;

    __shared__ float wave_v[NTH / 32];
    __shared__ int wave_t[NTH / 32];
    __shared__ float win_v;
    __shared__ int win_t;
    __shared__ float out_v[MAX_K];
    __shared__ int out_i[MAX_K];

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const float* row = logits + (size_t)b * vocab;

    float loc_val[MAX_K];
    int loc_idx[MAX_K];
#pragma unroll
    for (int j = 0; j < MAX_K; j++) {
        loc_val[j] = -INFINITY;
        loc_idx[j] = 0;
    }

    // Pass A. min_v / min_j: the first minimal slot among the K live slots.
    float min_v = loc_val[0];
    int min_j = 0;
    for (int i = tid; i < vocab; i += NTH) {
        const float v = row[i];
        if (v > min_v) {
#pragma unroll
            for (int j = 0; j < MAX_K; j++) {
                if (j == min_j) {
                    loc_val[j] = v;
                    loc_idx[j] = i;
                }
            }
            min_j = 0;
            min_v = loc_val[0];
#pragma unroll
            for (int j = 1; j < MAX_K; j++) {
                if (j < K && loc_val[j] < min_v) {
                    min_v = loc_val[j];
                    min_j = j;
                }
            }
        }
    }

    for (int r = 0; r < K; r++) {
        // This thread's best live candidate: largest value, lowest slot on ties.
        float bv = -INFINITY;
        int bj = -1;
#pragma unroll
        for (int j = 0; j < MAX_K; j++) {
            if (j < K && loc_val[j] > bv) {
                bv = loc_val[j];
                bj = j;
            }
        }
        // Best across the block under (value desc, thread asc): candidate order
        // c = t * K + slot, so a lower thread id is always the earlier arrival.
        float v = bv;
        int t = tid;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(v, o);
            const int ot = __shfl_xor(t, o);
            if (tie_better(ov, ot, v, t)) {
                v = ov;
                t = ot;
            }
        }
        if (lane == 0) {
            wave_v[wave] = v;
            wave_t[wave] = t;
        }
        __syncthreads();
        if (wave == 0) {
            float fv = (lane < NTH / 32) ? wave_v[lane] : -INFINITY;
            int ft = (lane < NTH / 32) ? wave_t[lane] : 0x7fffffff;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor(fv, o);
                const int ot = __shfl_xor(ft, o);
                if (tie_better(ov, ot, fv, ft)) {
                    fv = ov;
                    ft = ot;
                }
            }
            if (lane == 0) {
                win_v = fv;
                win_t = ft;
            }
        }
        __syncthreads();
        const float wv = win_v;
        const int wt = win_t;
        if (wv > -INFINITY) {
            if (tid == wt) {
                out_v[r] = wv;
                out_i[r] = loc_idx[bj];
#pragma unroll
                for (int j = 0; j < MAX_K; j++) {
                    if (j == bj) loc_val[j] = -INFINITY;
                }
            }
        } else if (tid == 0) {
            // No candidate left: the shipping list keeps its (-inf, 0) entry.
            out_v[r] = -INFINITY;
            out_i[r] = 0;
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int j = 0; j < K; j++) {
            top_idx[(size_t)b * K + j] = out_i[j];
            top_val[(size_t)b * K + j] = out_v[j];
        }
    }
}
