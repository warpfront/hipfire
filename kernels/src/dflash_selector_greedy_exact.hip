// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! dflash_selector_greedy_exact (Design A §4): the DFlash2 greedy candidate
//! selector on the device, byte-identical to the CPU reference
//! `hipfire_runtime::dflash::propose_candidates_from_rows` at temperature 0.
//!
//! One wave32 (block 32, grid 1). Lane j < k owns candidate j of every row.
//! Rows 0..rows-1 run sequentially; the predecessor of row 0 is `seed` and
//! of row r+1 is the id selected at row r. For each lane and row:
//!
//!   dot = 0.0f
//!   for d = 0..rank-1 (ascending):
//!       dot = dot + ((p[d] * proj[r*rank+d]) * s[d])   // three separately
//!                                                      // rounded f32 ops
//!   score = top_vals[r*k+j] + dot
//!
//! with p = pred_cb[pred][d], s = succ_cb[cand][d] (f16 codebooks widen
//! exactly; the widening mirrors `half::f16::to_f32`, subnormals and quiet
//! NaN included). No FMA, no reassociation, no parallel reduction:
//! `__fmul_rn`/`__fadd_rn` are not contractable, and the pragma is a second
//! guard. Scores are shared through LDS; lane 0 scans j = 0..k-1 with strict
//! `>` starting from score[0] (first candidate wins ties), and publishes the
//! selected id to every lane as the next predecessor.
//!
//! Outputs: scores[rows*k] (f32), block[0] = seed, block[r+1] = selected id of
//! row r, status[0]. Status is written exactly once:
//!   0                       ok, every id published
//!   code | (row+1) << 8     the first malformed input. Nothing at or after
//!                           `block[row+1]` is published; block[0..=row] may
//!                           hold earlier published ids and the caller MUST
//!                           treat a non-zero status as "no proposals".
//! codes: 1 bad shape (rows 1..=15, k 1..=16, rank >= 1, vocab >= 1, flags
//! 0/1), 2 seed >= vocab, 3 candidate id >= vocab (checked before any
//! codebook dereference), 4 predecessor >= vocab (cannot occur after a
//! validated seed/candidate; kept as a guard).
//! Codebook extents (vocab*rank elements) are the host wrapper's contract.

#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

#define DSG_MAX_ROWS 15u
#define DSG_MAX_K 16u

__device__ __forceinline__ float dsg_f16_to_f32(unsigned int h) {
    const unsigned int s = (h & 0x8000u) << 16;
    unsigned int e = (h >> 10) & 0x1Fu;
    unsigned int m = h & 0x3FFu;
    unsigned int o;
    if (e == 0u) {
        if (m == 0u) {
            o = s;
        } else {
            const unsigned int sh = (unsigned int)__builtin_clz(m) - 21u;
            m = (m << sh) & 0x3FFu;
            o = s | ((112u + 1u - sh) << 23) | (m << 13);
        }
    } else if (e == 31u) {
        o = m == 0u ? (s | 0x7F800000u) : (s | 0x7FC00000u | (m << 13));
    } else {
        o = s | ((e + 112u) << 23) | (m << 13);
    }
    return __uint_as_float(o);
}

__device__ __forceinline__ float dsg_load(const void* cb, unsigned long long idx, unsigned int is_f16) {
    if (is_f16) return dsg_f16_to_f32((unsigned int)((const unsigned short*)cb)[idx]);
    return ((const float*)cb)[idx];
}

extern "C" __launch_bounds__(32)
__global__ void dflash_selector_greedy_exact(
    const void* __restrict__ pred_cb,
    const void* __restrict__ succ_cb,
    const float* __restrict__ proj,          // [rows, rank]
    const unsigned int* __restrict__ top_ids, // [rows, k]
    const float* __restrict__ top_vals,       // [rows, k]
    const unsigned int* __restrict__ seed,    // [1]
    float* __restrict__ scores,               // [rows, k]
    unsigned int* __restrict__ block,         // [rows + 1]
    unsigned int* __restrict__ status,        // [1]
    unsigned int rows,
    unsigned int k,
    unsigned int rank,
    unsigned int vocab,
    unsigned int pred_f16,
    unsigned int succ_f16)
{
    __shared__ float s_score[DSG_MAX_K];
    __shared__ unsigned int s_pred;
    __shared__ unsigned int s_bad;
    const unsigned int lane = threadIdx.x;

    if (rows < 1u || rows > DSG_MAX_ROWS || k < 1u || k > DSG_MAX_K || rank < 1u ||
        vocab < 1u || pred_f16 > 1u || succ_f16 > 1u) {
        if (lane == 0) status[0] = 1u;
        return;
    }
    if (lane == 0) {
        const unsigned int sd = seed[0];
        if (sd >= vocab) {
            status[0] = 2u;
            s_bad = 2u;
        } else {
            block[0] = sd;
            s_pred = sd;
            s_bad = 0u;
        }
    }
    __syncthreads();
    if (s_bad) return;

    for (unsigned int r = 0; r < rows; ++r) {
        const unsigned int pred = s_pred;
        if (lane < k) {
            const unsigned int cand = top_ids[r * k + lane];
            float score = 0.0f;
            unsigned int bad = 0u;
            if (cand >= vocab) {
                bad = 3u;
            } else if (pred >= vocab) {
                bad = 4u;
            } else {
                const unsigned long long po = (unsigned long long)pred * rank;
                const unsigned long long so = (unsigned long long)cand * rank;
                const float* pr = proj + (unsigned long long)r * rank;
                float dot = 0.0f;
                for (unsigned int d = 0; d < rank; ++d) {
                    const float p = dsg_load(pred_cb, po + d, pred_f16);
                    const float s = dsg_load(succ_cb, so + d, succ_f16);
                    float t = __fmul_rn(__fmul_rn(p, pr[d]), s);
                    // HIP's __fmul_rn/__fadd_rn are plain operators, so
                    // fp-contract can still fuse `dot + t` into v_fmac.
                    // The empty asm makes t opaque, forcing a separately
                    // rounded multiply then add (the CPU scorer's order).
                    asm volatile("" : "+v"(t));
                    dot = __fadd_rn(dot, t);
                }
                score = __fadd_rn(top_vals[r * k + lane], dot);
            }
            s_score[lane] = score;
            scores[r * k + lane] = score;
            if (bad) atomicMax(&s_bad, bad);
        }
        __syncthreads();
        if (lane == 0) {
            if (s_bad) {
                status[0] = s_bad | ((r + 1u) << 8);
            } else {
                unsigned int best = 0u;
                float best_s = s_score[0];
                for (unsigned int j = 1; j < k; ++j) {
                    const float sj = s_score[j];
                    if (sj > best_s) {
                        best_s = sj;
                        best = j;
                    }
                }
                const unsigned int id = top_ids[r * k + best];
                block[r + 1u] = id;
                s_pred = id;
            }
        }
        __syncthreads();
        if (s_bad) return;
    }
    if (lane == 0) status[0] = 0u;
}
