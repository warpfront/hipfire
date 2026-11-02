// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Railgun E0 / L6d: re-gridded row selectors for the DFlash cycle (gfx1201).
//
// The shipping `topk_logsumexp_batched_f32` (raw mode, DFlash2 selector) and
// `argmax_f32_batched` (verify argmax) run 256 threads per row and, for
// top-K, merge 256 x K candidates on ONE thread. Both kernels below keep one
// block per row but widen it to 1024 threads and replace every serial step
// with wave32 butterflies plus one cross-wave stage. Neither does any
// floating-point arithmetic: outputs are selections of input values, so the
// only exactness question is which element wins a tie.
//
// argmax: the shipping kernel's winner is fully characterized by the total
// order (value desc, bitrev8(i & 255) asc, i asc) over elements with
// value > -1e30 (its initial running max), and 0 when no element qualifies:
// per-thread strict `>` keeps the first index of a stride-256 lane, and the
// shared-memory tree keeps the left slot on ties, which prefers low
// bit-reversed lane ids. This kernel maximizes that exact key, so it returns
// the shipping index for every input, ties and NaNs included.
//
// top-K (raw values): when the K+1 largest real values (value > -inf, not
// NaN) are pairwise distinct, the shipping output is exactly those K values
// sorted descending with their indices, whatever its internal lane layout.
// This kernel computes the top 17 by (value desc, index asc) and, if any of
// the leading K+1 values tie (including -inf padding), writes -1 to
// top_idx[b*K] instead; `topk_values_fixup_f32` (the shipping kernel body,
// entered only for marked rows) then recomputes that row verbatim.

#define SEL_BLOCK 1024
#define SEL_WAVES (SEL_BLOCK / 32)
#define SEL_KP 17
#define SEL_EMPTY 0x7fffffff
#define SEL_UNROLL 4

__device__ __forceinline__ bool sel_topk_better(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

// Pops the 17 best (value desc, index asc) heads of one sorted-desc 17-list
// per lane of the calling wave32, writing round r's winner to out_v/out_i[r]
// from lane 0. Consumes the lists.
__device__ __forceinline__ void sel_wave_top17(
    float (&a)[SEL_KP], int (&ai)[SEL_KP], float* out_v, int* out_i, int lane) {
#pragma unroll
    for (int r = 0; r < SEL_KP; ++r) {
        float bv = a[0];
        int bi = ai[0];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o);
            const int oi = __shfl_xor(bi, o);
            if (sel_topk_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            out_v[r] = bv;
            out_i[r] = bi;
        }
        if (bi != SEL_EMPTY && ai[0] == bi) {
#pragma unroll
            for (int j = 0; j < SEL_KP - 1; ++j) {
                a[j] = a[j + 1];
                ai[j] = ai[j + 1];
            }
            a[SEL_KP - 1] = -INFINITY;
            ai[SEL_KP - 1] = SEL_EMPTY;
        }
    }
}

extern "C" __launch_bounds__(SEL_BLOCK)
__global__ void topk_values_regrid_f32(
    const float* __restrict__ logits,  // [B x vocab]
    int* __restrict__ top_idx,         // [B x K]
    float* __restrict__ top_val,       // [B x K]
    int vocab,
    int K) {
    __shared__ float wave_v[SEL_WAVES][SEL_KP];
    __shared__ int wave_i[SEL_WAVES][SEL_KP];
    __shared__ float fin_v[SEL_KP];
    __shared__ int fin_i[SEL_KP];
    __shared__ float thr_s;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const float* row = logits + (size_t)b * vocab;

    const bool vec = (vocab & 3) == 0 && (reinterpret_cast<unsigned long long>(row) & 15) == 0;
    const float4* row4 = reinterpret_cast<const float4*>(row);
    const int n4 = vocab >> 2;

    // Pass 1: per-thread max. The 17th largest of the 1024 thread maxima is a
    // lower bound T on the row's 17th largest real value (17 distinct
    // elements are >= T), so pass 2 only has to admit values >= T. NaN never
    // wins `>`; a thread with no real value reports -inf (empty).
    float m = -INFINITY;
    if (vec) {
        for (int base = tid; base < n4; base += SEL_BLOCK * SEL_UNROLL) {
            float4 x[SEL_UNROLL];
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                const int q = base + u * SEL_BLOCK;
                x[u] = (q < n4) ? row4[q]
                                : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                if (x[u].x > m) m = x[u].x;
                if (x[u].y > m) m = x[u].y;
                if (x[u].z > m) m = x[u].z;
                if (x[u].w > m) m = x[u].w;
            }
        }
    } else {
        for (int i = tid; i < vocab; i += SEL_BLOCK) {
            const float v = row[i];
            if (v > m) m = v;
        }
    }
    float a[SEL_KP];
    int ai[SEL_KP];
#pragma unroll
    for (int j = 0; j < SEL_KP; ++j) {
        a[j] = -INFINITY;
        ai[j] = SEL_EMPTY;
    }
    if (m > -INFINITY) {
        a[0] = m;
        ai[0] = tid;
    }
    sel_wave_top17(a, ai, wave_v[wave], wave_i[wave], lane);
    __syncthreads();
    if (wave == 0) {
#pragma unroll
        for (int j = 0; j < SEL_KP; ++j) {
            a[j] = wave_v[lane][j];
            ai[j] = wave_i[lane][j];
        }
        sel_wave_top17(a, ai, fin_v, fin_i, lane);
        if (lane == 0) thr_s = fin_v[SEL_KP - 1];
    }
    __syncthreads();
    const float thr = thr_s;

    // Pass 2: per-thread sorted-desc top 17 of values >= T. `v > a[16]`
    // rejects NaN and -inf, the same values the shipping kernel can never
    // admit. Visit order is free: only the resulting multiset matters.
#pragma unroll
    for (int j = 0; j < SEL_KP; ++j) {
        a[j] = -INFINITY;
        ai[j] = SEL_EMPTY;
    }
#define SEL_TOPK_PUSH(V, I)                                                   \
    do {                                                                      \
        const float v_ = (V);                                                 \
        const int i_ = (I);                                                   \
        if (v_ >= thr && v_ > a[SEL_KP - 1]) {                                \
            _Pragma("unroll")                                                 \
            for (int j = SEL_KP - 1; j > 0; --j) {                            \
                const bool up = v_ > a[j - 1];                                \
                const bool here = v_ > a[j];                                  \
                a[j] = up ? a[j - 1] : (here ? v_ : a[j]);                    \
                ai[j] = up ? ai[j - 1] : (here ? i_ : ai[j]);                 \
            }                                                                 \
            if (v_ > a[0]) {                                                  \
                a[0] = v_;                                                    \
                ai[0] = i_;                                                   \
            }                                                                 \
        }                                                                     \
    } while (0)
    if (vec) {
        for (int base = tid; base < n4; base += SEL_BLOCK * SEL_UNROLL) {
            float4 x[SEL_UNROLL];
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                const int q = base + u * SEL_BLOCK;
                x[u] = (q < n4) ? row4[q]
                                : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
            }
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                const int i0 = (base + u * SEL_BLOCK) * 4;
                SEL_TOPK_PUSH(x[u].x, i0);
                SEL_TOPK_PUSH(x[u].y, i0 + 1);
                SEL_TOPK_PUSH(x[u].z, i0 + 2);
                SEL_TOPK_PUSH(x[u].w, i0 + 3);
            }
        }
    } else {
        for (int i = tid; i < vocab; i += SEL_BLOCK) {
            SEL_TOPK_PUSH(row[i], i);
        }
    }
#undef SEL_TOPK_PUSH
    sel_wave_top17(a, ai, wave_v[wave], wave_i[wave], lane);
    __syncthreads();
    if (wave == 0) {
#pragma unroll
        for (int j = 0; j < SEL_KP; ++j) {
            a[j] = wave_v[lane][j];
            ai[j] = wave_i[lane][j];
        }
        sel_wave_top17(a, ai, fin_v, fin_i, lane);
        if (lane == 0) {
            bool tie = false;
#pragma unroll
            for (int j = 0; j < SEL_KP - 1; ++j) {
                if (j < K && !(fin_v[j] > fin_v[j + 1])) tie = true;
            }
            if (tie) {
                top_idx[(size_t)b * K] = -1;
            } else {
                for (int j = 0; j < K; ++j) {
                    top_idx[(size_t)b * K + j] = fin_i[j];
                    top_val[(size_t)b * K + j] = fin_v[j];
                }
            }
        }
    }
}

// Shipping-argmax key: larger value first; on equal values the lower
// bit-reversed stride-256 lane, then the lower index.
__device__ __forceinline__ bool sel_argmax_better(float va, int ia, float vb, int ib) {
    if (va > vb) return true;
    if (!(va == vb)) return false;
    const unsigned ra = __brev((unsigned)(ia & 255)) >> 24;
    const unsigned rb = __brev((unsigned)(ib & 255)) >> 24;
    return ra < rb || (ra == rb && ia < ib);
}

extern "C" __launch_bounds__(SEL_BLOCK)
__global__ void argmax_f32_batched_regrid(
    const float* __restrict__ data,  // [batch, n] row-major
    int* __restrict__ result,        // [batch]
    int n) {
    __shared__ float wave_v[SEL_WAVES];
    __shared__ int wave_i[SEL_WAVES];
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const float* row = data + (size_t)b * n;

    // Empty = (-1e30, SEL_EMPTY): every qualifying element (v > -1e30) beats it.
    float bv = -1e30f;
    int bi = SEL_EMPTY;
#define SEL_ARGMAX_PUSH(V, I)                                                 \
    do {                                                                      \
        const float v_ = (V);                                                 \
        const int i_ = (I);                                                   \
        if (v_ > -1e30f && sel_argmax_better(v_, i_, bv, bi)) {               \
            bv = v_;                                                          \
            bi = i_;                                                          \
        }                                                                     \
    } while (0)
    // Visit order is free (the key is a total order); aligned rows stream
    // SEL_UNROLL float4 loads per thread per step.
    if ((n & 3) == 0 && (reinterpret_cast<unsigned long long>(row) & 15) == 0) {
        const float4* row4 = reinterpret_cast<const float4*>(row);
        const int n4 = n >> 2;
        for (int base = tid; base < n4; base += SEL_BLOCK * SEL_UNROLL) {
            float4 x[SEL_UNROLL];
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                const int q = base + u * SEL_BLOCK;
                x[u] = (q < n4) ? row4[q] : make_float4(-1e30f, -1e30f, -1e30f, -1e30f);
            }
#pragma unroll
            for (int u = 0; u < SEL_UNROLL; ++u) {
                const int i0 = (base + u * SEL_BLOCK) * 4;
                SEL_ARGMAX_PUSH(x[u].x, i0);
                SEL_ARGMAX_PUSH(x[u].y, i0 + 1);
                SEL_ARGMAX_PUSH(x[u].z, i0 + 2);
                SEL_ARGMAX_PUSH(x[u].w, i0 + 3);
            }
        }
    } else {
        for (int i = tid; i < n; i += SEL_BLOCK) {
            SEL_ARGMAX_PUSH(row[i], i);
        }
    }
#undef SEL_ARGMAX_PUSH
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor(bv, o);
        const int oi = __shfl_xor(bi, o);
        if (sel_argmax_better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        wave_v[wave] = bv;
        wave_i[wave] = bi;
    }
    __syncthreads();
    if (wave == 0) {
        bv = wave_v[lane];
        bi = wave_i[lane];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor(bv, o);
            const int oi = __shfl_xor(bi, o);
            if (sel_argmax_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) result[b] = (bi == SEL_EMPTY) ? 0 : bi;
    }
}
