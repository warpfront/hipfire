// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx1201 Qwen3.5 full-attention preparation: one launch replaces
//   deinterleave_q_rmsnorm_f32_batched (Q + gate)
//   rmsnorm_f32 (K, in place)
//   rope_partial_halfsplit_batched_f32 (Q and K)
// One wave32 owns one 256-wide (token, head) row, lane l holding dims
// l + 32k (k = 0..7): no LDS and no workgroup barriers.
//
// Bit-exactness vs the three-launch sequence (standalone-verified on
// gfx1201, and by the WT2 kldseq md5):
// - rmsnorm_f32 at blockDim == n == 256 sums v*v per thread (one element)
//   and halves over LDS: node (t, s) = node(t) + node(t + s) for
//   s = 128, 64, ..., 1. Lane l holds leaves l + 32k, so levels 128/64/32
//   are the same pairs added in-lane and levels 16..1 are shfl_xor pairs;
//   fp add is commutative, so every node is the identical rounded sum.
// - normalized = (x * w) * rms and the RoPE outputs use the gfx1201 batched
//   kernel's formation: out[i] = fma(x0, cos, -(x1 * sin)),
//   out[i + 32] = fma(x0, sin, x1 * cos); angles use the same expressions.
//   Pair (i, i + 32) is (k = 0, k = 1) of lane i: RoPE stays in-lane.
//
// qwen35_fa_prep_fp8q_batched_gfx1201 additionally replaces the F32 Q rows by
// the exact E4M3 codes and row scales the native-fp8 Q-resident attention
// prologue derives from them: a = max_d |q[d]| (exact, order-free),
// s = (a == 0) ? 1 : a / 448, code[d] = e4m3(q[d] / s) (IEEE quotients).
// Code layout per (token, head): dim d = 16f + 8kg + j at byte kg*128 + f*8 + j
// (one attention lane half reads 128 contiguous bytes); q_scale [N x NQ].
//
// qwen35_fa_prep_fp8q_nogate_batched_gfx1201 is the fp8q twin that leaves the
// gate where the QKV projection wrote it (the second half of each 512-float
// head row of q_interleaved): it neither reads that half nor writes a gate
// copy, and the FP8 FA-output producer reads the gate in place
// (HIPFIRE_ROTATE_FP8_GATE_IL in mq_rotate_x_fp8.gfx12.hip). Q codes, row
// scales and K are byte-identical to qwen35_fa_prep_fp8q_batched_gfx1201.
//
// Admitted: NQ 24 / NK 4 (args), head_dim 256, n_rot 64.
// Grid: [ceil((NQ + NK) * N / 8), 1, 1]. Block: [256, 1, 1] (8 rows).

template <bool FP8Q, bool GATE = true>
static __device__ __forceinline__ void qwen35_fa_prep_row_gfx1201(
    const float* __restrict__ q_interleaved,
    float* __restrict__ q,
    unsigned char* __restrict__ q_codes,
    float* __restrict__ q_scale,
    float* __restrict__ gate,
    float* __restrict__ k,
    const float* __restrict__ q_weight,
    const float* __restrict__ k_weight,
    const int* __restrict__ positions,
    float eps,
    float freq_base,
    int pos_offset,
    int n_q_heads,
    int n_kv_heads,
    int batch_size)
{
    // Every rounding point is explicit (the reference sums RN(v*v) leaves and
    // the RoPE fmas are written out): no implicit contraction here.
#pragma clang fp contract(off)
    constexpr int HD = 256;
    constexpr int NROT = 64;
    const int lane = (int)threadIdx.x & 31;
    const int slots = n_q_heads + n_kv_heads;
    const long long r = (long long)blockIdx.x * 8 + ((int)threadIdx.x >> 5);
    if (r >= (long long)slots * batch_size)
        return;
    const int b = (int)(r / slots);
    const int head_slot = (int)(r - (long long)b * slots);
    const bool is_q = head_slot < n_q_heads;
    const int head = is_q ? head_slot : head_slot - n_q_heads;

    float v[8];
    const float* weight;
    long long row;
    if (is_q) {
        row = (long long)b * n_q_heads * HD + (long long)head * HD;
        const float* src =
            q_interleaved + (long long)b * n_q_heads * HD * 2 + (long long)head * HD * 2;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            v[kk] = src[lane + 32 * kk];
            if (GATE)
                gate[row + lane + 32 * kk] = src[HD + lane + 32 * kk];
        }
        weight = q_weight;
    } else {
        row = (long long)b * n_kv_heads * HD + (long long)head * HD;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
            v[kk] = k[row + lane + 32 * kk];
        weight = k_weight;
    }
    // rmsnorm_f32 halving tree over leaves t = lane + 32k.
    float s[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        s[kk] = v[kk] * v[kk];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)  // s = 128
        s[kk] = s[kk] + s[kk + 4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk)  // s = 64
        s[kk] = s[kk] + s[kk + 2];
    float sum = s[0] + s[1];        // s = 32
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)  // s = 16 .. 1
        sum = sum + __shfl_xor(sum, off, 32);
    const float rms = rsqrtf(sum / (float)HD + eps);
    float o[8];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        o[kk] = v[kk] * weight[lane + 32 * kk] * rms;
    {
        const int pos = positions[b] + pos_offset;
        const float freq = 1.0f / powf(freq_base, (float)(2 * lane) / (float)NROT);
        const float angle = (float)pos * freq;
        const float cos_a = cosf(angle);
        const float sin_a = sinf(angle);
        const float x0 = o[0];
        const float x1 = o[1];
        o[0] = fmaf(x0, cos_a, -(x1 * sin_a));
        o[1] = fmaf(x0, sin_a, x1 * cos_a);
    }
    if (!FP8Q || !is_q) {
        float* out = is_q ? q : k;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
            out[row + lane + 32 * kk] = o[kk];
        return;
    }
    float a = 0.0f;
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
        a = fmaxf(a, fabsf(o[kk]));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor(a, off, 32));
    const float scale = (a == 0.0f) ? 1.0f : a / 448.0f;
    const long long rowq = (long long)b * n_q_heads + head;
    unsigned char* dst = q_codes + rowq * HD + ((lane >> 3) & 1) * 128
        + (lane >> 4) * 8 + (lane & 7);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        // dim lane + 32kk: f = (lane >> 4) + 2kk, kg = (lane >> 3) & 1.
        const unsigned code =
            __builtin_amdgcn_cvt_pk_fp8_f32(o[kk] / scale, 0.0f, 0u, false) & 0xffu;
        dst[kk * 16] = (unsigned char)code;
    }
    if (lane == 0)
        q_scale[rowq] = scale;
}

extern "C" __launch_bounds__(256)
__global__ void qwen35_fa_prep_batched_gfx1201(
    const float* __restrict__ q_interleaved, // [N x NQ x 256 x 2]
    float* __restrict__ q,                   // [N x NQ x 256]
    float* __restrict__ gate,                // [N x NQ x 256]
    float* __restrict__ k,                   // [N x NK x 256], in-place in/out
    const float* __restrict__ q_weight,      // [256]
    const float* __restrict__ k_weight,      // [256]
    const int* __restrict__ positions,       // [N] physical KV slots
    float eps,
    float freq_base,
    int pos_offset,
    int n_q_heads,
    int n_kv_heads,
    int batch_size)
{
    qwen35_fa_prep_row_gfx1201<false>(
        q_interleaved, q, nullptr, nullptr, gate, k, q_weight, k_weight,
        positions, eps, freq_base, pos_offset, n_q_heads, n_kv_heads, batch_size);
}

extern "C" __launch_bounds__(256)
__global__ void qwen35_fa_prep_fp8q_batched_gfx1201(
    const float* __restrict__ q_interleaved, // [N x NQ x 256 x 2]
    unsigned char* __restrict__ q_codes,     // [N x NQ x 256] permuted E4M3
    float* __restrict__ q_scale,             // [N x NQ]
    float* __restrict__ gate,                // [N x NQ x 256]
    float* __restrict__ k,                   // [N x NK x 256], in-place in/out
    const float* __restrict__ q_weight,      // [256]
    const float* __restrict__ k_weight,      // [256]
    const int* __restrict__ positions,       // [N] physical KV slots
    float eps,
    float freq_base,
    int pos_offset,
    int n_q_heads,
    int n_kv_heads,
    int batch_size)
{
    qwen35_fa_prep_row_gfx1201<true>(
        q_interleaved, nullptr, q_codes, q_scale, gate, k, q_weight, k_weight,
        positions, eps, freq_base, pos_offset, n_q_heads, n_kv_heads, batch_size);
}

extern "C" __launch_bounds__(256)
__global__ void qwen35_fa_prep_fp8q_nogate_batched_gfx1201(
    const float* __restrict__ q_interleaved, // [N x NQ x 256 x 2], gate read in place later
    unsigned char* __restrict__ q_codes,     // [N x NQ x 256] permuted E4M3
    float* __restrict__ q_scale,             // [N x NQ]
    float* __restrict__ k,                   // [N x NK x 256], in-place in/out
    const float* __restrict__ q_weight,      // [256]
    const float* __restrict__ k_weight,      // [256]
    const int* __restrict__ positions,       // [N] physical KV slots
    float eps,
    float freq_base,
    int pos_offset,
    int n_q_heads,
    int n_kv_heads,
    int batch_size)
{
    qwen35_fa_prep_row_gfx1201<true, false>(
        q_interleaved, nullptr, q_codes, q_scale, nullptr, k, q_weight, k_weight,
        positions, eps, freq_base, pos_offset, n_q_heads, n_kv_heads, batch_size);
}
