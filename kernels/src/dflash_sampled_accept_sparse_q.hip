// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Device accept / residual / bonus of a sampled singleton DFlash window
// (HIPFIRE_DFLASH_DEVICE_ACCEPT), over the resident FAST_SAMPLE target rows
// of `softmax_temp_topp_batched_f32` and the host's SPARSE selector q rows.
// No target row, no compaction and no dense q ever leaves or touches the host:
// the host uploads one payload (tokens, pre-drawn uniforms, q) and downloads
// four words.
//
// One block of 256 threads. Rows are the verify rows of the window: `n` draft
// positions plus the bonus row `n` (the softmax ran over n + 1 rows).
//
// ── Law ──────────────────────────────────────────────────────────────────
// Chen & Leviathan rejection sampling against the AR-truncated target row
//   p_i(c) = tau_i <= 0 ? probs_i[c]
//                       : (probs_i[c] >= tau_i ? probs_i[c] * (1 / max(z_i, FLT_MIN)) : 0)
// (`apply_topp_trunc`'s exact rule: a NaN tau truncates everything to 0) and
// the dense-with-last-write selector row q_i the host prepared (`SparseQRows`).
//   accept_i   = u_i * max(qsel_i, FLT_MIN) <= p_i(t_i)             (t_i = proposed token)
//   accept_len = first rejected row, bounded by `max_accept`
//   rejected   -> bonus ~ normalize(max(p_r - q_r, 0))              (residual, uniform u_fin)
//   else       -> bonus ~ p_{accept_len}                            (target row,  uniform u_fin)
// Row i's acceptance only depends on its own uniform, so the accept test of
// every row runs in parallel and `accept_len` is the minimum rejected index.
// The residual and the bonus are mutually exclusive, so ONE final uniform
// serves both: n + 1 pre-drawn uniforms per window, a fixed count, drawn by
// the host from its own RNG (deterministic per seed).
//
// ── Draw order ───────────────────────────────────────────────────────────
// The categorical walk is over ascending vocab ids with a DEFINED, thread
// count independent association: thread t owns the contiguous segment
// [t * seg, min((t + 1) * seg, vocab)), seg = ceil(vocab / 256); segment sums
// are accumulated sequentially from 0, then chained in segment order into
// base[t]; the running CDF at id c of segment t is base[t] + (sequential sum
// of the segment's positive summands up to c). The CDF is monotone and ends at
// base[256] = the total. The draw is the first id whose CDF exceeds u (u *
// total for the residual); only the claiming segment is re-walked. The result
// is deterministic (no atomics, fixed association) and the same ordered CDF
// law as the host loop, whose own sum association differs only by f32
// rounding. `kernels` fp contract is OFF so the host reference twin
// (rdna-compute `dflash_sampled_accept` module) matches bit for bit.
//
// ── Fallbacks (the host loop's, with one deliberate fix) ─────────────────
//   residual total <= 0  -> argmax of the truncated target row (first max)
//   draw falls off the end (u * total rounds to total) -> the LAST id of the
//     support (the host returns vocab - 1, an id outside the kept set).
//     code 2 in out[3]; code 1 = degenerate argmax; 0 = normal draw.
//
// ── Payload (u32 words; floats as bits) ──────────────────────────────────
//   [0, n)            proposed tokens t_i
//   [n, 2n)           accept uniforms u_i
//   [2n]              final uniform u_fin
//   [2n + 1]          max_accept (rows tested; host clamps to <= n)
//   [2n + 2, 3n + 2)  selected q (q at the proposed token, raw)
//   [3n + 2, 4n + 3)  q row offsets (n + 1 words, offset[0] = 0)
//   [4n + 3, +nnz)    q ids, ascending per row, nnz = offset[n]
//   [.., +nnz)        q values
//
// out (4 x i32): accept_len, bonus token, rejected (0/1), fallback code.
//
// Kernarg order: probs, tau, z, payload, out, n, vocab. Grid [1], block 256.

#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

#define DSA_THREADS 256
#define DSA_FLT_MIN 1.17549435e-38f

__device__ __forceinline__ float dsa_eff(float p, float tau, float inv_z) {
    if (tau <= 0.0f) return p;
    return (p >= tau) ? (p * inv_z) : 0.0f;
}

// First index in qidx[0, qn) that is >= x.
__device__ __forceinline__ int dsa_lower_bound(const unsigned int* qidx, int qn, unsigned int x) {
    int lo = 0, hi = qn;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (qidx[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

extern "C" __global__ __launch_bounds__(DSA_THREADS) void dflash_sampled_accept_sparse_q_f32(
    const float* __restrict__ probs,
    const float* __restrict__ tau_rows,
    const float* __restrict__ z_rows,
    const unsigned int* __restrict__ payload,
    int* __restrict__ out,
    int n,
    int vocab)
{
    __shared__ int   s_min[DSA_THREADS];
    __shared__ float s_seg[DSA_THREADS];
    __shared__ int   s_last[DSA_THREADS];
    __shared__ float s_bv[DSA_THREADS];
    __shared__ int   s_bi[DSA_THREADS];
    __shared__ float s_base[DSA_THREADS + 1];
    __shared__ float s_us;
    __shared__ int   s_bonus;
    __shared__ int   s_code;

    const int tid = threadIdx.x;

    const unsigned int* tokens = payload;
    const unsigned int* uacc   = payload + n;
    const float u_fin          = __uint_as_float(payload[2 * n]);
    unsigned int max_accept    = payload[2 * n + 1];
    const unsigned int* selq   = payload + 2 * n + 2;
    const unsigned int* qoff   = payload + 3 * n + 2;
    const unsigned int* qids   = payload + 4 * n + 3;
    const unsigned int* qvals  = qids + qoff[n];

    const int lim = (int)min((unsigned int)n, max_accept);

    // ── Parallel accept tests; accept_len = first rejected row ────────────
    int local_rej = 0x7fffffff;
    for (int i = tid; i < lim; i += DSA_THREADS) {
        const unsigned int t = tokens[i];
        const float tau = tau_rows[i];
        const float inv_z = __fdiv_rn(1.0f, fmaxf(z_rows[i], DSA_FLT_MIN));
        const float p = probs[(size_t)i * (size_t)vocab + t];
        const float p_t = dsa_eff(p, tau, inv_z);
        const float p_d = fmaxf(__uint_as_float(selq[i]), DSA_FLT_MIN);
        const float u = __uint_as_float(uacc[i]);
        if (!(u * p_d <= p_t)) local_rej = min(local_rej, i);
    }
    s_min[tid] = local_rej;
    __syncthreads();
    for (int s = DSA_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) s_min[tid] = min(s_min[tid], s_min[tid + s]);
        __syncthreads();
    }
    const bool rejected = s_min[0] < lim;
    const int row = rejected ? s_min[0] : lim;

    // ── Row scalars and segment geometry ───────────────────────────────────
    const float tau = tau_rows[row];
    const float inv_z = __fdiv_rn(1.0f, fmaxf(z_rows[row], DSA_FLT_MIN));
    const float* __restrict__ prow = probs + (size_t)row * (size_t)vocab;
    const int q_lo = rejected ? (int)qoff[row] : 0;
    const int qn = rejected ? (int)qoff[row + 1] - q_lo : 0;
    const unsigned int* qrow_ids = qids + q_lo;
    const unsigned int* qrow_val = qvals + q_lo;

    const int seg = (vocab + DSA_THREADS - 1) / DSA_THREADS;
    const int start = min(tid * seg, vocab);
    const int end = min(start + seg, vocab);

    // ── Pass 1: segment sums of the positive summands, last positive id ───
    // residual: d = p - q (q = 0 off its support); bonus: d = p.
    {
        int qp = rejected ? dsa_lower_bound(qrow_ids, qn, (unsigned int)start) : 0;
        float L = 0.0f;
        int last = -1;
        for (int c = start; c < end; ++c) {
            float d = dsa_eff(prow[c], tau, inv_z);
            if (rejected) {
                float q = 0.0f;
                if (qp < qn && qrow_ids[qp] == (unsigned int)c) {
                    q = __uint_as_float(qrow_val[qp]);
                    ++qp;
                }
                d = d - q;
            }
            if (d > 0.0f) {
                L = L + d;
                last = c;
            }
        }
        s_seg[tid] = L;
        s_last[tid] = last;
    }
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        s_base[0] = 0.0f;
        for (int t = 0; t < DSA_THREADS; ++t) {
            acc = acc + s_seg[t];
            s_base[t + 1] = acc;
        }
        const float total = acc;
        s_code = 0;
        if (rejected) {
            if (total <= 0.0f) {
                s_code = 1;           // degenerate: argmax of the target row
                s_us = 0.0f;
            } else {
                s_us = u_fin * total;
            }
        } else {
            s_us = u_fin;
        }
        s_bonus = -1;
    }
    __syncthreads();

    if (s_code == 1) {
        // Degenerate residual: first maximum of the truncated target row
        // (strict >, start -inf, like `argmax_u32`).
        float best_v = -INFINITY;
        int best_i = -1;
        for (int c = start; c < end; ++c) {
            float v = dsa_eff(prow[c], tau, inv_z);
            if (v > best_v) { best_v = v; best_i = c; }
        }
        s_bv[tid] = best_v;
        s_bi[tid] = best_i;
        __syncthreads();
        if (tid == 0) {
            float bv = -INFINITY;
            int bi = 0;
            for (int t = 0; t < DSA_THREADS; ++t) {
                if (s_bi[t] >= 0 && s_bv[t] > bv) { bv = s_bv[t]; bi = s_bi[t]; }
            }
            s_bonus = bi;
        }
    } else {
        // Only the segment whose CDF interval holds u walks again.
        const float us = s_us;
        if (s_base[tid] <= us && us < s_base[tid + 1]) {
            const float base = s_base[tid];
            int qp = rejected ? dsa_lower_bound(qrow_ids, qn, (unsigned int)start) : 0;
            float L = 0.0f;
            for (int c = start; c < end; ++c) {
                float d = dsa_eff(prow[c], tau, inv_z);
                if (rejected) {
                    float q = 0.0f;
                    if (qp < qn && qrow_ids[qp] == (unsigned int)c) {
                        q = __uint_as_float(qrow_val[qp]);
                        ++qp;
                    }
                    d = d - q;
                }
                if (d > 0.0f) {
                    L = L + d;
                    if (us < base + L) { s_bonus = c; break; }
                }
            }
        }
        __syncthreads();
        if (tid == 0 && s_bonus < 0) {
            // Fell off the end (u * total rounded up to total, or NaN): the
            // last support id, or vocab - 1 when the support is empty.
            int fb = vocab - 1;
            for (int t = DSA_THREADS - 1; t >= 0; --t) {
                if (s_last[t] >= 0) { fb = s_last[t]; break; }
            }
            s_bonus = fb;
            s_code = 2;
        }
    }
    __syncthreads();

    if (tid == 0) {
        out[0] = row;
        out[1] = s_bonus;
        out[2] = rejected ? 1 : 0;
        out[3] = s_code;
    }
}
