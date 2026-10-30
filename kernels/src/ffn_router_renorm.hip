// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MoE router: per-row top-k selection and renormalised weights from raw router
// logits.  One block of 256 threads per row.
//
//   kind 0 (softmax + renorm):
//       p = softmax(logits); select top-k on p; w[r] = p[idx_r] / sum(p[idx_*])
//   kind 1 (sigmoid + bias + renorm):
//       s = 1 / (1 + expf(-logit)); select top-k on s + bias[e] (bias optional);
//       w[r] = s[idx_r] / (sum(s[idx_*]) + eps) * scale   (unbiased s)
//
// Selection order is descending selection score; a tie goes to the LOWER expert
// id.  The chosen sum runs sequentially from 0.0f in selection (rank) order.
// A NaN / -inf selection score is ranked as the lowest finite value (it stays
// selectable, so top_k <= n_experts always yields top_k distinct experts).
//
// Softmax: the exponent sum is the 256 per-thread strided partial sums folded by
// an LDS tree (stride 128, 64, ..., 1); every probability is a direct divide.
//
// Grid  (rows)      Block 256
// LDS   2 * 1024 * 4 (scores, selection scores) + 2 * 256 * 4 (reduction)
//       + 8 * 4 * 3 (chosen idx / score / weight)
//
// Kernarg order:
//   logits*, bias* (nullable), idx_out*, w_out*, n_experts, top_k, kind, eps, scale
// Limits: n_experts <= 1024, top_k <= 8, top_k <= n_experts.

#include <hip/hip_runtime.h>

#define FFN_ROUTER_THREADS 256
#define FFN_ROUTER_MAX_E 1024
#define FFN_ROUTER_MAX_K 8
#define FFN_ROUTER_LOWEST (-3.4028234663852886e38f)  /* -FLT_MAX */

extern "C" __global__ __launch_bounds__(FFN_ROUTER_THREADS) void ffn_moe_router_topk_f32(
    const float* logits,
    const float* bias,
    int* idx_out,
    float* w_out,
    int n_experts,
    int top_k,
    int kind,
    float eps,
    float scale
) {
    __shared__ float scores[FFN_ROUTER_MAX_E];
    __shared__ float sel[FFN_ROUTER_MAX_E];
    __shared__ float red_val[FFN_ROUTER_THREADS];
    __shared__ int red_idx[FFN_ROUTER_THREADS];
    __shared__ int chosen_idx[FFN_ROUTER_MAX_K];
    __shared__ float chosen_score[FFN_ROUTER_MAX_K];
    __shared__ float chosen_w[FFN_ROUTER_MAX_K];

    const int tid = threadIdx.x;
    const long long row = blockIdx.x;
    const float* x = logits + row * (long long)n_experts;

    // ---- scores ----
    if (kind == 0) {
        float m = -INFINITY;
        for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
            const float v = x[e];
            scores[e] = v;
            if (v > m) m = v;
        }
        red_val[tid] = m;
        __syncthreads();
        for (int s = FFN_ROUTER_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s && red_val[tid + s] > red_val[tid]) red_val[tid] = red_val[tid + s];
            __syncthreads();
        }
        const float row_max = red_val[0];
        __syncthreads();

        float part = 0.0f;
        for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
            const float ex = expf(scores[e] - row_max);
            scores[e] = ex;
            part += ex;
        }
        red_val[tid] = part;
        __syncthreads();
        for (int s = FFN_ROUTER_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) red_val[tid] = red_val[tid] + red_val[tid + s];
            __syncthreads();
        }
        const float total = red_val[0];
        __syncthreads();

        for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
            const float p = scores[e] / total;
            scores[e] = p;
            sel[e] = p;
        }
    } else {
        for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
            const float sg = 1.0f / (1.0f + expf(-x[e]));
            scores[e] = sg;
            sel[e] = (bias != nullptr) ? (sg + bias[e]) : sg;
        }
    }
    __syncthreads();
    for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
        const float v = sel[e];
        // NaN and -inf rank lowest but stay above the -INFINITY "taken" marker.
        sel[e] = (v >= FFN_ROUTER_LOWEST) ? v : FFN_ROUTER_LOWEST;
    }
    __syncthreads();

    // ---- top-k rounds ----
    for (int r = 0; r < top_k; ++r) {
        float best_v = -INFINITY;
        int best_i = 0x7fffffff;
        for (int e = tid; e < n_experts; e += FFN_ROUTER_THREADS) {
            const float v = sel[e];
            if (v > best_v) {
                best_v = v;
                best_i = e;
            }
        }
        red_val[tid] = best_v;
        red_idx[tid] = best_i;
        __syncthreads();
        for (int s = FFN_ROUTER_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const float ov = red_val[tid + s];
                const int oi = red_idx[tid + s];
                const float mv = red_val[tid];
                const int mi = red_idx[tid];
                if (ov > mv || (ov == mv && oi < mi)) {
                    red_val[tid] = ov;
                    red_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int win = red_idx[0];
            chosen_idx[r] = win;
            chosen_score[r] = scores[win];
            sel[win] = -INFINITY;
        }
        __syncthreads();
    }

    // ---- weights ----
    if (tid == 0) {
        float sum = 0.0f;
        for (int r = 0; r < top_k; ++r) sum += chosen_score[r];
        if (kind == 0) {
            for (int r = 0; r < top_k; ++r) chosen_w[r] = chosen_score[r] / sum;
        } else {
            const float denom = sum + eps;
            for (int r = 0; r < top_k; ++r) chosen_w[r] = (chosen_score[r] / denom) * scale;
        }
    }
    __syncthreads();
    if (tid < top_k) {
        idx_out[row * (long long)top_k + tid] = chosen_idx[tid];
        w_out[row * (long long)top_k + tid] = chosen_w[tid];
    }
}
