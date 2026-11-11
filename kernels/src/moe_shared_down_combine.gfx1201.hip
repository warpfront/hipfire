// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — Qwen4 MoE epilogue: the routed ten-row combine and the shared-expert fold ride the
// shared-down rows GEMV (`HIPFIRE_QWEN4_MOE_EPILOGUE_FUSE`, developer, default off).
//
// Bitwise the three launches
//   moe_down_combine_top10_batched   (moe_down_combine_top10_batched.hip): expanded rows -> target
//   gemv_bf16_xf32_x4_rows_r<R>      (qwen4_gemv_bf16_xf32.hip): shared down F32 rows
//   bf16_scaled_add_batched_f32      (tensor_ops.hip): target = bf16(bf16(target) + bf16(bf16(sel) * bf16(row)))
// i.e. `target` ends exactly as the chain leaves it; the shared-down F32 rows are not stored
// (nothing reads them).  Per (token, column) the epilogue is:
//   acc = bf16(target)                                   (the combine's accumulator seed)
//   per route, ascending expert id (a tie keeps rank order; ids outside [0, n_experts) are
//   unrouted and never selected):  acc = bf16(acc + bf16(bf16(out) * bf16(w)))
//   target = bf16(bf16(acc) + bf16(bf16(selector) * bf16(shared_dot)))
// and the shared dot is gemv_bf16_xf32_rows_body's, copied textually (four chains a0..a3 per
// token, (a0 + a1) + (a2 + a3), shfl_down tree), so every token's dot is bitwise the one-row
// kernel's.
//
// Geometry: block = 10 waves (320 threads), grid [M / 8], M % 8 == 0.  Waves 0..7: wave w computes
// output column 8 * blockIdx.x + w for all R tokens, exactly the rows kernel's wave-per-output
// decomposition; its lane-0 dots go through LDS.  Waves 8 and 9 run concurrently: thread q owns
// one (token, column) = (q >> 3, 8 * blockIdx.x + (q & 7)) with q < 8R, reads its ten expert-row
// values as 32-byte column segments, orders the routes by position counting (no data-dependent
// branches or loads) and runs the BF16 accumulate chain; after the barrier it folds the GEMV dot
// in and stores `target`, so the combine's latency hides under the GEMV.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define MSC_BF16_LO(u) __uint_as_float((u) << 16)
#define MSC_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

static __device__ __forceinline__ float msc_bf16(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

// BF16 rounding of a value that is the product / sum of two BF16-exact floats (or one rounded
// value): RNE `(u + 0x7fff + bit16) & 0xffff0000`. Identical to the converting cast for every such
// value -- inf rounds to inf, and a NaN carries no bits below bit 16 so the add cannot reach the
// payload (the same argument as hc_row_fold's FAST rounding).
static __device__ __forceinline__ float msc_bf16_exact(float v) {
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}

// ZINIT (`HIPFIRE_QWEN4_FFN_CLEAR_ABSORB_ROWS`, the `_zinit` entries): the combine's accumulator
// seed is +0.0 instead of bf16(target), `target` is never read, and its dirty / poisoned contents
// are fully overwritten: every (token < R, column = 8 * blockIdx.x + c < M) is stored by exactly
// one lane below (grid M / 8, M % 8 == 0; unrouted slots are skipped inside the accumulate, never
// around the store).  Bitwise `zero_f32(target)` + the non-ZINIT entry, since bf16(+0.0) == +0.0.
template <int R, bool ZINIT>
__device__ __forceinline__ void moe_shared_down_combine_body(
    const unsigned short* __restrict__ W,      // [M, K] bf16 shared down
    const float* __restrict__ x,               // [R, K] F32 shared hidden
    float* __restrict__ target,                // [R, M] combine target: seed in (not ZINIT), result out
    const float* __restrict__ expert_outputs,  // [R * 10 * M] unweighted expanded down rows
    const int* __restrict__ topk_indices,      // [R * 10]
    const float* __restrict__ topk_weights,    // [R * 10]
    const float* __restrict__ scalar,          // [R] shared selector
    int M, int K, int n_experts
) {
    __shared__ float dots[R][8];
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave < 8) {
        const int row = (int)blockIdx.x * 8 + wave;   // M % 8 == 0
        const unsigned short* w_row = W + (size_t)row * K;

        // ---- gemv_bf16_xf32_rows_body<R>, tid -> lane ----
        float a0[R], a1[R], a2[R], a3[R];
#pragma unroll
        for (int r = 0; r < R; ++r) a0[r] = a1[r] = a2[r] = a3[r] = 0.0f;
        if ((K & 7) == 0) {
            int i = lane * 8;
            for (; i + 8 * 32 * 8 <= K; i += 32 * 8 * 8) {
                uint4 wv[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) wv[s] = *(const uint4*)(w_row + i + s * (32 * 8));
#pragma unroll
                for (int s = 0; s < 8; ++s) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float* xr = x + (size_t)r * K;
                        const float4 xa = *(const float4*)(xr + i + s * (32 * 8));
                        const float4 xb = *(const float4*)(xr + i + s * (32 * 8) + 4);
                        a0[r] += MSC_BF16_LO(wv[s].x) * xa.x;
                        a1[r] += MSC_BF16_HI(wv[s].x) * xa.y;
                        a2[r] += MSC_BF16_LO(wv[s].y) * xa.z;
                        a3[r] += MSC_BF16_HI(wv[s].y) * xa.w;
                        a0[r] += MSC_BF16_LO(wv[s].z) * xb.x;
                        a1[r] += MSC_BF16_HI(wv[s].z) * xb.y;
                        a2[r] += MSC_BF16_LO(wv[s].w) * xb.z;
                        a3[r] += MSC_BF16_HI(wv[s].w) * xb.w;
                    }
                }
            }
            for (; i + 8 <= K; i += 32 * 8) {
                const uint4 wv = *(const uint4*)(w_row + i);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float* xr = x + (size_t)r * K;
                    const float4 xa = *(const float4*)(xr + i);
                    const float4 xb = *(const float4*)(xr + i + 4);
                    a0[r] += MSC_BF16_LO(wv.x) * xa.x;
                    a1[r] += MSC_BF16_HI(wv.x) * xa.y;
                    a2[r] += MSC_BF16_LO(wv.y) * xa.z;
                    a3[r] += MSC_BF16_HI(wv.y) * xa.w;
                    a0[r] += MSC_BF16_LO(wv.z) * xb.x;
                    a1[r] += MSC_BF16_HI(wv.z) * xb.y;
                    a2[r] += MSC_BF16_LO(wv.w) * xb.z;
                    a3[r] += MSC_BF16_HI(wv.w) * xb.w;
                }
            }
        }
        for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + lane; i < K; i += 32) {
            const float w = __uint_as_float(((unsigned int)w_row[i]) << 16);
#pragma unroll
            for (int r = 0; r < R; ++r) a0[r] += w * x[(size_t)r * K + i];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float sum = (a0[r] + a1[r]) + (a2[r] + a3[r]);
            for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset);
            if (lane == 0) dots[r][wave] = sum;
        }
        __syncthreads();
        return;
    }

    // ---- waves 8 and 9: the routed combine of (token, column) = (q >> 3, 8 * blockIdx.x + (q & 7)),
    // concurrent with the GEMV waves.  Then the shared fold once the GEMV dots are in LDS.
    const int q = tid - 256;
    const int token = q >> 3;
    const int c = q & 7;
    const bool mine = token < R;
    const int col = (int)blockIdx.x * 8 + c;
    const size_t slot = (size_t)(mine ? token : 0) * 10;
    int ids[10];
    float outs[10];
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        ids[k] = topk_indices[slot + k];
        // A dead rank's row is still valid memory (never used).
        outs[k] = expert_outputs[(slot + k) * (size_t)M + col];
    }
    // The lane owns ranks c and 8 + (c & 1) for the order and the weight rounding; the other
    // lanes of its token get them by shuffle.
    const int own_b = 8 + (c & 1);
    const int id_a = topk_indices[slot + c];
    const int id_b = topk_indices[slot + own_b];
    const float w_a = topk_weights[slot + c];
    const float w_b = topk_weights[slot + own_b];
    const float seed = ZINIT ? 0.0f : target[(slot / 10) * M + col];
    const float gate_in = scalar[slot / 10];
    // Sort keys: (expert id, rank), unique; ids outside [0, n_experts) (unrouted: never
    // selected) sort after every routed one.
    int key[10];
    int count = 0;
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const bool live = (unsigned)ids[k] < (unsigned)n_experts;
        key[k] = live ? (int)(((unsigned)ids[k] << 4) | (unsigned)k) : (0x40000000 | k);
        count += live ? 1 : 0;
    }
    const int key_a = (unsigned)id_a < (unsigned)n_experts ? (int)(((unsigned)id_a << 4) | (unsigned)c) : (0x40000000 | c);
    const int key_b = (unsigned)id_b < (unsigned)n_experts ? (int)(((unsigned)id_b << 4) | (unsigned)own_b) : (0x40000000 | own_b);
    // Position in the combine's repeated-minimum order = number of smaller keys.
    int pos_a = 0, pos_b = 0;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        pos_a += (int)((unsigned)(key[j] - key_a) >> 31);
        pos_b += (int)((unsigned)(key[j] - key_b) >> 31);
    }
    const float wb_a = msc_bf16(w_a);
    const float wb_b = msc_bf16(w_b);
    // Each lane scatters its ten (column-wise) terms to their positions and reads them back
    // in position order: every slot of its (token, column) is written by this lane.
    __shared__ float sorted[R][10][8];
    const int tok = mine ? token : 0;
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        const int p = __shfl(i < 8 ? pos_a : pos_b, i & 7, 8);
        const float wb = __shfl(i < 8 ? wb_a : wb_b, i & 7, 8);
        // Both factors are BF16-exact: the product is rounded by the cheap exact formula.
        sorted[tok][p][c] = msc_bf16_exact(msc_bf16(outs[i]) * wb);
    }
    float acc = msc_bf16(seed);
#pragma unroll
    for (int k = 0; k < 10; ++k) {
        const float term = sorted[tok][k][c];
        const float next = msc_bf16_exact(acc + term);
        acc = k < count ? next : acc;
    }
    __syncthreads();
    if (!mine) return;
    // bf16_scaled_add_batched_f32: residual = the combine's stored value (BF16-exact `acc`).
    const float rhs = msc_bf16(dots[token][c]);
    const float gate = msc_bf16(gate_in);
    const float product = msc_bf16_exact(gate * rhs);
    target[(size_t)token * M + col] = msc_bf16_exact(acc + product);
}

#define MOE_SHARED_DOWN_COMBINE_ROWS_NAMED(NAME, R, ZINIT)                                    \
    __launch_bounds__(320) extern "C" __global__ void NAME(                                   \
        const unsigned short* __restrict__ W, const float* __restrict__ x,                    \
        float* __restrict__ target, const float* __restrict__ expert_outputs,                 \
        const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,         \
        const float* __restrict__ scalar, int M, int K, int n_experts) {                      \
        moe_shared_down_combine_body<R, ZINIT>(W, x, target, expert_outputs, topk_indices,    \
                                               topk_weights, scalar, M, K, n_experts);        \
    }
#define MOE_SHARED_DOWN_COMBINE_ROWS(R)                                                       \
    MOE_SHARED_DOWN_COMBINE_ROWS_NAMED(moe_shared_down_combine_rows_r##R, R, false)           \
    MOE_SHARED_DOWN_COMBINE_ROWS_NAMED(moe_shared_down_combine_rows_r##R##_zinit, R, true)
MOE_SHARED_DOWN_COMBINE_ROWS(1)
MOE_SHARED_DOWN_COMBINE_ROWS(2)
MOE_SHARED_DOWN_COMBINE_ROWS(3)
MOE_SHARED_DOWN_COMBINE_ROWS(4)
MOE_SHARED_DOWN_COMBINE_ROWS(5)
MOE_SHARED_DOWN_COMBINE_ROWS(6)
MOE_SHARED_DOWN_COMBINE_ROWS(7)
MOE_SHARED_DOWN_COMBINE_ROWS(8)
#undef MOE_SHARED_DOWN_COMBINE_ROWS
#undef MOE_SHARED_DOWN_COMBINE_ROWS_NAMED
