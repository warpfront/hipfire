// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Batched temperature-scaled softmax, out-of-place.
//
// For each row r (one block per row), computes
//
//     s_i   = logits[r*n + i] * inv_t
//     m     = max_i s_i
//     e_i   = expf(s_i - m)
//     probs[r*n + i] = e_i / sum_i e_i
//
// This MIRRORS the host `softmax_temp_into` in
// crates/hipfire-arch-qwen35/src/speculative.rs: temperature is applied to
// each logit FIRST (v * inv_t), the max is taken over the scaled logits, and
// exp is taken of (scaled - max). The arithmetic ORDER of the reduction is a
// tree reduction here vs. a sequential sum on the host, so the produced
// probabilities are DISTRIBUTION-parity, NOT byte-identical, to the host
// softmax (see HIPFIRE_DFLASH_FAST_SAMPLE gating in speculative.rs).
//
// Out-of-place: the input `logits` buffer is left untouched (the caller still
// needs the raw logits for argmax on the same rows).

#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_temp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int n,
    float inv_t
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // max over scaled logits
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // exp(scaled - max), accumulate sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

// Variant that additionally emits the per-row truncation threshold and kept
// mass of the AR sampler's law, WITHOUT modifying `probs` (consumers apply
// `p >= tau_cut ? p / Z : 0`). For each row r it writes:
//
//   tau_cut[r] = probability of the last kept token; consumers keep tokens
//                with p_i >= tau_cut[r].
//   Z[r]       = the kept mass = sum of p_i for p_i >= tau_cut[r].
//
// The kept set is EXACTLY `llama::sample_top_k_p`'s (the AR producer's host
// sampler; the GPU `sample_top_p` kernel shares its semantics):
//
//   1. cap = top_k, the caller's AR candidate cap (`llama::ar_candidate_cap`:
//      absent top_k → 20, 0 → 64, k → min(k, 64)); `top_k <= 0` here → 64.
//   2. The cap largest probabilities, sorted descending — selected EXACTLY by
//      an 8-bit-digit radix select on the (non-negative) float bit patterns.
//   3. min_p > 0: cap shrinks to the first rank i >= 1 whose prob is below
//      min_p * p_max.
//   4. Nucleus over the cap's OWN mass: keep the shortest descending prefix
//      whose cumulative mass reaches top_p * mass(cap), crossing token
//      included. The full-vocab mass never enters (AR samples from its pool).
//
// Exact float ties at the cap boundary or at tau keep every tied token
// (measure-zero divergence from AR's index-order tie-break).
extern "C" __global__ void softmax_temp_topp_batched_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    float* __restrict__ tau_cut,   // [rows] out
    float* __restrict__ Z,         // [rows] out
    int n,
    float inv_t,
    float top_p,
    int top_k,
    float min_p
) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    const float* row_in = logits + (size_t)row * n;
    float* row_out = probs + (size_t)row * n;

    // ── Phase 1: max over scaled logits ────────────────────────────────────
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float s = row_in[i] * inv_t;
        max_val = fmaxf(max_val, s);
    }
    sdata[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();

    // ── Phase 2: exp(scaled - max), accumulate sum ─────────────────────────
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row_in[i] * inv_t - max_val);
        row_out[i] = e;
        sum += e;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    sum = sdata[0];
    __syncthreads();

    // ── Phase 3: normalize → row_out holds the full softmax p_i, sum == 1 ───
    float inv_sum = 1.0f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
    __syncthreads();

    // ── Phase 4: exact radix select of the cap-th largest probability ──────
    // p_max == inv_sum (the max logit's exp is 1). Non-negative floats order
    // like their bit patterns, so four 8-bit digit passes (MSB first) find the
    // exact bits of the cap-th largest prob. Tokens below p_max * 2^-20 are
    // skipped (they would only pile atomics on a few low bins) unless fewer
    // than `cap` tokens clear that floor, in which case no floor is used.
    const int MAX_CAP = 64;
    __shared__ unsigned int s_hist[256];
    __shared__ unsigned int s_digit;
    __shared__ unsigned int s_need;     // tokens still to take at the current prefix
    __shared__ unsigned int s_count;    // tokens clearing the floor
    __shared__ unsigned int s_n_gt;
    __shared__ unsigned int s_n_eq;
    __shared__ float s_cand_p[MAX_CAP];
    __shared__ int   s_cand_i[MAX_CAP];
    __shared__ float s_tau;

    int cap = (top_k <= 0 || top_k > MAX_CAP) ? MAX_CAP : top_k;
    if (cap > n) cap = n;
    const float pmax = inv_sum;

    if (threadIdx.x == 0) s_count = 0u;
    __syncthreads();
    float floor_p = pmax * (1.0f / 1048576.0f);
    unsigned int local_cnt = 0u;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (row_out[i] >= floor_p) ++local_cnt;
    }
    atomicAdd(&s_count, local_cnt);
    __syncthreads();
    if (s_count < (unsigned int)cap) floor_p = 0.0f;
    const unsigned int floor_bits = __float_as_uint(floor_p);

    unsigned int prefix = 0u, mask = 0u;
    if (threadIdx.x == 0) s_need = (unsigned int)cap;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) s_hist[b] = 0u;
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            unsigned int bits = __float_as_uint(row_out[i]);
            if (bits >= floor_bits && (bits & mask) == prefix) {
                atomicAdd(&s_hist[(bits >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int need = s_need, above = 0u;
            int d = 255;
            for (; d > 0; --d) {
                if (above + s_hist[d] >= need) break;
                above += s_hist[d];
            }
            s_digit = (unsigned int)d;
            s_need = need - above;
        }
        __syncthreads();
        prefix |= s_digit << shift;
        mask |= 255u << shift;
    }
    const float vk = __uint_as_float(prefix);   // exact cap-th largest prob
    const unsigned int ties_needed = s_need;    // of the tokens equal to vk

    // ── Phase 5: gather the cap candidates, sort, apply AR's cut ────────────
    if (threadIdx.x == 0) { s_n_gt = 0u; s_n_eq = 0u; }
    __syncthreads();
    const unsigned int n_gt_total = (unsigned int)cap - ties_needed;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row_out[i];
        if (p > vk) {
            unsigned int slot = atomicAdd(&s_n_gt, 1u);
            s_cand_p[slot] = p;
            s_cand_i[slot] = i;
        } else if (p == vk) {
            unsigned int slot = atomicAdd(&s_n_eq, 1u);
            if (slot < ties_needed) {
                s_cand_p[n_gt_total + slot] = p;
                s_cand_i[n_gt_total + slot] = i;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Insertion sort, descending prob, ascending id on ties.
        for (int a = 1; a < cap; ++a) {
            float p = s_cand_p[a];
            int id = s_cand_i[a];
            int j = a;
            while (j > 0 && (s_cand_p[j - 1] < p || (s_cand_p[j - 1] == p && s_cand_i[j - 1] > id))) {
                s_cand_p[j] = s_cand_p[j - 1];
                s_cand_i[j] = s_cand_i[j - 1];
                --j;
            }
            s_cand_p[j] = p;
            s_cand_i[j] = id;
        }
        int kept_cap = cap;
        if (min_p > 0.0f) {
            float floor_min = fminf(min_p, 1.0f) * s_cand_p[0];
            for (int a = 1; a < cap; ++a) {
                if (s_cand_p[a] < floor_min) { kept_cap = a; break; }
            }
        }
        float mass = 0.0f;
        for (int a = 0; a < kept_cap; ++a) mass += s_cand_p[a];
        float threshold = fminf(fmaxf(top_p, 0.0f), 1.0f) * mass;
        float cum = 0.0f;
        int m = kept_cap;
        for (int a = 0; a < kept_cap; ++a) {
            cum += s_cand_p[a];
            if (cum >= threshold) { m = a + 1; break; }
        }
        // A zero (underflowed) last kept prob must not read as "no cut".
        s_tau = fmaxf(s_cand_p[m - 1], 1.17549435e-38f);
    }
    __syncthreads();
    const float tau = s_tau;

    // ── Phase 6: exact Z = mass(tau) ────────────────────────────────────────
    float kept = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float p = row_out[i];
        if (p >= tau) kept += p;
    }
    sdata[threadIdx.x] = kept;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    kept = sdata[0];

    if (threadIdx.x == 0) {
        tau_cut[row] = tau;
        Z[row] = kept;
    }
}
