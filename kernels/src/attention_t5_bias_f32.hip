// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Small-sequence f32 self-attention for the FLUX **text encoders** (T5-XXL
// conditioning and CLIP-L pooling). Deliberately NOT the FLUX MMDiT attention
// kernel: those are WMMA/f16 paths tuned for n_img in the thousands, while
// this one runs at n = 256 (T5) / 77 (CLIP) where the whole score row fits in
// LDS and a plain tiled kernel is bandwidth-trivial.
//
// It is a structural mirror of the CPU reference loops in
// `hipfire_arch_diffusion::t5::encode` / `clip::encode`:
//
//     scores[q,k] = (Σ_t Q[q, h*hd+t] · K[k, h*hd+t]) * scale + bias[h,q,k]
//     probs       = softmax over the row (max-subtracted)
//     out[q, h*hd+t] = Σ_k probs[q,k] · V[k, h*hd+t]
//
// Layouts (all row-major, batch 1):
//   q, k, v : [n, heads*hd] f32, HEAD-INTERLEAVED — head `h` occupies columns
//             [h*hd, (h+1)*hd), exactly the `x[pos*d + h*hd + t]` indexing the
//             CPU reference uses, so no transpose is needed either side.
//   bias    : [heads, n, n] f32, or nullptr. T5 passes the relative-position
//             bias (data-INDEPENDENT: it is a pure function of the bucket
//             table and the sequence length, so the host builds it once per
//             frame length). CLIP passes nullptr.
//   out     : [n, heads*hd] f32.
//
// `scale` folds the dot-product scaling: T5 passes **1.0** (transformers 5
// folds the scale into the relative bias and does not scale the logits — the
// golden was captured against that), CLIP passes 1/sqrt(hd).
//
// `causal != 0` masks future keys (k > q), which is CLIP's mask. Masked keys
// are SKIPPED (the loops stop at `kmax`) rather than given a masked logit —
// the CPU reference adds `f32::NEG_INFINITY` and lets `exp(-inf - max)`
// underflow to exactly 0, and the diagonal is always unmasked, so the two
// forms are numerically identical and skipping cannot produce a NaN.
//
// key_mask (F32 [n], 1.0 = visible, 0.0 = masked), when non-null, adds a
// key-padding mask on top of the causal/bias logic above. This is the Qwen3
// GQA path (Task 15); T5 and CLIP keep passing nullptr and are numerically
// unaffected — see PADDING IS NOT MASKED above, which still describes their
// behaviour exactly.
//
// NO IEEE INFINITIES: a masked key gets the large FINITE sentinel logit
// MASKED_LOGIT = -1e30f, never -INFINITY, and `local_max` starts at that same
// sentinel rather than -INFINITY. Every value this kernel computes is
// therefore finite on every path:
//   * mixed row (some key visible): row_max is a real logit, and
//     `expf(-1e30f - row_max)` underflows to exactly 0.0f — the same
//     contribution -inf would have made, with no inf ever materialised.
//   * fully-masked row (every key in the causal window masked, or kmax == 0):
//     row_max == -1e30f, so every `expf(0.0f) == 1.0f`, `sum == kmax >= 1`
//     and `1.0f / sum` is finite. `row_masked` (row_max <= -1e29f) then
//     selects an exact 0.0f for the store, which is the reference
//     convention for a row with nothing to attend to.
// The previous form put a real -INFINITY in LDS and relied on
// `isfinite(row_max)` plus a block-uniform early `return`, which computed
// `exp(-inf - -inf) = NaN` on a fully-masked row and discarded it by
// returning early. That form is also correct — it was re-measured against
// this exact test and passes (Task 14 report, Fix round 2) — so this rewrite
// is defence in depth, not a bug fix: it keeps NaN out of the softmax
// pipeline entirely rather than producing one and branching around it, and
// it drops the early `return` so every thread runs the same instruction
// stream past every barrier. `row_masked` is a select on the final store.
//
// The sentinel is safe because a real logit can never reach -1e29f: T5/CLIP
// logits live within a few tens of units of zero, and the mask branch
// overwrites `acc` outright, so `row_max <= -1e29f` holds if and only if the
// row is fully masked.
//
// GQA: `q` is `[n, heads*hd]`, `k`/`v` are `[n, n_kv_heads*hd]` (heads %
// n_kv_heads == 0). Head `h` reads KV head `h / (heads / n_kv_heads)`. T5 and
// CLIP pass `n_kv_heads == heads`, which collapses `kv_off` to `off` and
// `d_kv` to `d` — the exact indexing this kernel used before this change.
//
// Launch: grid (n, heads), block 256 (power of two, required by the two tree
// reductions), dynamic LDS = (n + blockDim.x) floats.

// Finite stand-in for a -inf logit, and the threshold that recognises it
// after the max reduction. See NO IEEE INFINITIES above.
#define ATTN_MASKED_LOGIT (-1e30f)
#define ATTN_MASKED_ROW   (-1e29f)

extern "C" __global__ void attention_t5_bias_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bias,
    const float* __restrict__ key_mask,
    float* __restrict__ out,
    int n, int heads, int n_kv_heads, int hd, float scale, int causal) {

    extern __shared__ float smem[];
    float* probs = smem;               // [n]
    float* red   = smem + n;           // [blockDim.x]

    const int qpos = (int)blockIdx.x;
    const int h    = (int)blockIdx.y;
    if (qpos >= n || h >= heads) return;

    const int   d      = heads * hd;
    const int   d_kv   = n_kv_heads * hd;
    const int   off    = h * hd;
    const int   kv_off = (h / (heads / n_kv_heads)) * hd;
    const int   tid  = (int)threadIdx.x;
    const int   nthr = (int)blockDim.x;
    const int   kmax = causal ? (qpos + 1) : n;

    const float* qr = q + (size_t)qpos * d + off;

    // ── scores + running max ──────────────────────────────────────────
    float local_max = ATTN_MASKED_LOGIT;
    for (int kp = tid; kp < kmax; kp += nthr) {
        const float* kr = k + (size_t)kp * d_kv + kv_off;
        float acc = 0.0f;
        for (int t = 0; t < hd; ++t) {
            acc = fmaf(qr[t], kr[t], acc);
        }
        acc *= scale;
        if (bias != nullptr) {
            acc += bias[((size_t)h * n + qpos) * n + kp];
        }
        if (key_mask != nullptr && key_mask[kp] == 0.0f) {
            acc = ATTN_MASKED_LOGIT;
        }
        probs[kp] = acc;
        local_max = fmaxf(local_max, acc);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    // Always finite (worst case the sentinel itself). `row_masked` is block
    // uniform and is applied as a select on the final store, not as a branch.
    const float row_max    = red[0];
    const bool  row_masked = (row_max <= ATTN_MASKED_ROW);
    __syncthreads();

    // ── exp + sum ─────────────────────────────────────────────────────
    float local_sum = 0.0f;
    for (int kp = tid; kp < kmax; kp += nthr) {
        const float e = expf(probs[kp] - row_max);
        probs[kp] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    // The max element always contributes expf(0) == 1, so red[0] >= 1 for
    // every row — masked or not — and this division is never 1.0f / 0.0f.
    const float inv_sum = 1.0f / red[0];
    __syncthreads();

    // ── context ───────────────────────────────────────────────────────
    // One thread per head-dim lane; each walks the (LDS-resident) prob row.
    for (int t = tid; t < hd; t += nthr) {
        float acc = 0.0f;
        for (int kp = 0; kp < kmax; ++kp) {
            acc = fmaf(probs[kp], v[(size_t)kp * d_kv + kv_off + t], acc);
        }
        out[(size_t)qpos * d + off + t] = row_masked ? 0.0f : (acc * inv_sum);
    }
}
