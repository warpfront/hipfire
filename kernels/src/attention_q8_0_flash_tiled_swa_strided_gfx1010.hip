// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx1010 FP32 query-tiled Q8 attention experiment, head dimension 256.
//
// Source algorithm: attention_q8_0_flash_prefill.hip (scalar FP32 SIMT online
// softmax). Strided QKV-row Q load + packed out + sliding-window mask deltas
// on top of that path. Plain single-arena K/V (no descriptors/slots/tree).
//
// Grid:  [ceil(batch_size / BR), n_heads, 1]
// Block: NTHREADS (256)
// BR=8, BC=32, hd=256 (bph=8). One geometry only.
//
// LDS (dynamic shared, constant 26976 B — NOT fixed+dynamic split):
//   s[BR*BC] 1024 | m/l/corr[BR] 96 | q[BR*hd] 8192
//   | kt[BC*KROW] 8832 | vt[BC*KROW] 8832
// KROW=276 = 272 payload (bph*34) + 4 pad dword per staged row. Global arena
// stays 34 B Q8_0 blocks; padding is LDS-addressing only. Stage payload bytes
// only; every K/V LDS access uses KROW stride.
//
// Per-thread private qpos[BR]/tlo[BR] (no shared exchange). Uniform maxseq /
// min_t_lo recomputed identically on every lane. Empty row-tile state is
// decided from geometric [t_lo,qpos] ∩ [t0,t0+t_len) intersection — preserve
// m/l/O, corr=1, zero weights, bypass exp (no -inf/-inf NaN).
//
// Bodies compile only when __gfx1010__ is defined (headers above stay live).

#if defined(__gfx1010__)

#ifndef BR
#define BR 8
#endif
#ifndef BC
#define BC 32
#endif
#ifndef NTHREADS
#define NTHREADS 256
#endif
// LDS row stride for staged K/V: 8 blocks * 34 B = 272 payload + 4 pad.
#ifndef KROW
#define KROW 276
#endif

extern "C" __global__ __launch_bounds__(NTHREADS) void
attention_q8_0_flash_tiled_swa_strided_gfx1010(
    const float* __restrict__ qkv,             // [batch_size × q_stride] F32
    const unsigned char* __restrict__ k_cache, // plain arena Q8_0
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,                   // [batch_size × n_heads × head_dim]
    const int* __restrict__ positions,         // [batch_size]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int window,
    int q_stride,
    int batch_size
) {
    const int tile = blockIdx.x;
    const int h    = blockIdx.y;
    const int tid  = threadIdx.x;
    // Whole-WG early exits only (uniform blockIdx) — never cross-wave return
    // past a required barrier below.
    if (h >= n_heads) return;

    const int qbase = tile * BR;
    if (qbase >= batch_size) return;

    // max_seq is plain-arena capacity; keep symbol live for ABI parity with
    // the 13-arg baseline launch (host validates max_ctx_len <= max_seq).
    (void)max_seq;

    const int bph                 = head_dim / 32; // 8 at hd=256
    const int total_blocks_per_pos = n_kv_heads * bph;
    const int kv_group            = n_heads / n_kv_heads;
    const int kv_h                = h / kv_group;
    const int kv_head_block_start = kv_h * bph;
    const int out_stride          = n_heads * head_dim;

    // Dynamic LDS 26976 B: s 1024 + m/l/corr 96 + q 8192 + kt 8832 + vt 8832.
    extern __shared__ float smem[];
    float* s_lds    = smem;                          // [BR*BC]
    float* m_lds    = s_lds + BR * BC;               // [BR]
    float* l_lds    = m_lds + BR;                    // [BR]
    float* corr_lds = l_lds + BR;                    // [BR]
    float* q_lds    = corr_lds + BR;                 // [BR*head_dim]
    unsigned char* kt = (unsigned char*)(q_lds + BR * head_dim);
    unsigned char* vt = kt + (size_t)BC * KROW;

    const int rows = min(BR, batch_size - qbase);

    // ── Stage Q tile (strided QKV row; packed out later) ────────────────
    for (int i = tid; i < rows * head_dim; i += NTHREADS) {
        const int r = i / head_dim;
        const int d = i % head_dim;
        q_lds[r * head_dim + d] =
            qkv[(long long)(qbase + r) * q_stride + h * head_dim + d];
    }
    // Per-row running state (flash_prefill.hip:132).
    if (tid < BR) {
        m_lds[tid]    = -1e30f;
        l_lds[tid]    = 0.0f;
        corr_lds[tid] = 1.0f;
    }
    __syncthreads();

    // Output accumulator: lanes_per_row = 256/8 = 32; dpt = 256/32 = 8.
    const int lanes_per_row = NTHREADS / BR;
    const int acc_r = tid / lanes_per_row;
    const int acc_c = tid % lanes_per_row;
    const int dpt   = head_dim / lanes_per_row;
    const int d0    = acc_c * dpt;
    float acc[32]; // host asserts dpt <= 32
    for (int i = 0; i < dpt; i++) acc[i] = 0.0f;

    // ── Private per-thread qpos/tlo (no shared exchange, no mono assume) ─
    // Each lane independently loads every valid row's position and derives
    // SWA t_lo; uniform maxseq / min_t_lo follow from the same private copies.
    int qpos[BR];
    int tlo[BR];
    int maxseq   = 0;
    int min_t_lo = 0x7fffffff;
    for (int r = 0; r < rows; r++) {
        qpos[r] = positions[qbase + r];
        // window==0 → full causal t_lo=0; else [max(0,qpos+1-window), qpos].
        tlo[r] = (window > 0 && qpos[r] + 1 > window) ? (qpos[r] + 1 - window) : 0;
        maxseq   = max(maxseq, qpos[r] + 1);
        min_t_lo = min(min_t_lo, tlo[r]);
    }
    if (rows <= 0) {
        min_t_lo = 0;
        maxseq   = 0;
    }

    // ── K/V tile loop with online softmax ───────────────────────────────
    for (int t0 = 0; t0 < maxseq; t0 += BC) {
        // Uniform tile skip BEFORE any barrier of this iteration: fires only
        // when EVERY valid query's window starts at/after t0+BC (window=0 ⇒
        // tlo all 0 ⇒ never skips). All lanes take the same branch.
        if (t0 + BC <= min_t_lo) continue;

        const int t_len = min(BC, maxseq - t0);

        __syncthreads(); // protect kt/vt from the previous iteration
        // Stage payload bytes only into KROW-strided LDS rows.
        for (int i = tid; i < t_len * bph * 34; i += NTHREADS) {
            const int j  = i / (bph * 34);
            const int bo = i % (bph * 34);
            const int t  = t0 + j;
            // Plain-arena offsets identical to gfx1010 baseline SWA strided.
            const size_t gbase =
                (size_t)t * (size_t)total_blocks_per_pos * 34u
                + (size_t)kv_head_block_start * 34u;
            const size_t lds = (size_t)j * (size_t)KROW + (size_t)bo;
            kt[lds] = k_cache[gbase + (size_t)bo];
            vt[lds] = v_cache[gbase + (size_t)bo];
        }
        __syncthreads();

        // S = Q · K^T with per-row causal + SWA mask.
        // Mask key t for query r: tlo[r] <= t <= qpos[r].
        for (int e = tid; e < BR * BC; e += NTHREADS) {
            const int r = e / BC;
            const int j = e % BC;
            float sv = -INFINITY;
            if (r < rows && j < t_len) {
                const int t = t0 + j;
                if (t >= tlo[r] && t <= qpos[r]) {
                    float dot = 0.0f;
                    const unsigned char* kb = kt + (size_t)j * (size_t)KROW;
                    // Serial bi/c FMA order preserved (flash_prefill.hip:201-206).
                    for (int bi = 0; bi < bph; bi++) {
                        const float ks =
                            (float)*((const _Float16*)(kb + bi * 34));
                        const float* qb = q_lds + r * head_dim + bi * 32;
                        for (int c = 0; c < 32; c++)
                            dot += qb[c]
                                   * (ks * (float)((signed char)kb[bi * 34 + 2 + c]));
                    }
                    sv = dot * scale_attn;
                }
            }
            s_lds[e] = sv;
        }
        __syncthreads();

        // Online softmax: new running max, rescale factor, running sum.
        // Geometry-empty row-tile: interval [tlo,qpos] ∩ [t0,t0+t_len) empty
        // → keep m/l, corr=1, zero weights, bypass exp (no sentinel-only path).
        if (tid < rows) {
            const int row_lo = tlo[tid];
            const int row_hi = qpos[tid]; // inclusive
            // Empty iff tile ends before window or starts after causal bound.
            const bool row_empty =
                (t0 + t_len - 1 < row_lo) || (t0 > row_hi);

            if (row_empty) {
                // Identity online step: m/l unchanged, O later sees p==0.
                corr_lds[tid] = 1.0f;
                for (int j = 0; j < BC; j++) s_lds[tid * BC + j] = 0.0f;
            } else {
                float mx = m_lds[tid];
                for (int j = 0; j < BC; j++)
                    mx = fmaxf(mx, s_lds[tid * BC + j]);
                // Preserve -1e30f first-tile / still-empty running-max path.
                const float corr =
                    (m_lds[tid] <= -1e30f) ? 1.0f : __expf(m_lds[tid] - mx);
                float sum = 0.0f;
                for (int j = 0; j < BC; j++) {
                    const float sv = s_lds[tid * BC + j];
                    const float e  = (sv == -INFINITY) ? 0.0f : __expf(sv - mx);
                    s_lds[tid * BC + j] = e;
                    sum += e;
                }
                m_lds[tid]    = mx;
                l_lds[tid]    = l_lds[tid] * corr + sum;
                corr_lds[tid] = corr;
            }
        }
        __syncthreads();

        // O = O * corr + P · V (ascending j within tile; flash_prefill:232-247).
        if (acc_r < rows) {
            const float corr = corr_lds[acc_r];
            if (corr != 1.0f)
                for (int i = 0; i < dpt; i++) acc[i] *= corr;
            for (int j = 0; j < t_len; j++) {
                const float p = s_lds[acc_r * BC + j];
                if (p == 0.0f) continue;
                const unsigned char* vb = vt + (size_t)j * (size_t)KROW;
                for (int i = 0; i < dpt; i++) {
                    const int d  = d0 + i;
                    const int bi = d >> 5;
                    const int bj = d & 31;
                    const float vs =
                        (float)*((const _Float16*)(vb + bi * 34));
                    acc[i] += p * (vs * (float)((signed char)vb[bi * 34 + 2 + bj]));
                }
            }
        }
    }

    // ── Write packed output (ragged tail: only valid rows) ──────────────
    if (acc_r < rows) {
        float* o = out + (long long)(qbase + acc_r) * out_stride + h * head_dim;
        const float denom = l_lds[acc_r];
        const float inv   = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
        for (int i = 0; i < dpt; i++) o[d0 + i] = acc[i] * inv;
    }
}

#endif // defined(__gfx1010__)
