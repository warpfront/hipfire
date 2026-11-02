// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// BF16-pooled multi-row QSA block scores (module qsa_bf16_scores), the BF16
// twins of indexed_attention_select_scores_rows{8,16}_f32 (tensor_ops.hip),
// plus the D_i=128 row-exact wide-verify tile rows32 (end of file).
// Byte-exact to the fused BF16 selector's indexed_attention_select_score4_f32
// <__hip_bfloat16>: each pooled element widens exactly (bits << 16), each
// (row, head) dot is one serial chain in ascending d order written as
// `dot += a * p` (same compiler FP policy as the fused selector, no explicit
// fma), `dot / sqrtf(index_dim)` is the incumbent's literal expression
// divide (no reciprocal), `fmaxf(., 0)` clamps, and the four heads are summed
// in head order starting from 0.0f.
//
// ABI (same as the F32 kernels, pooled pointer is BF16):
//   (const float* query, const __hip_bfloat16* pooled, float* scores,
//    int rows, int query_row_stride, int block_count, int index_dim,
//    int compress, int position_start, int score_stride)
// index_heads is fixed at 4, index_dim % 4 == 0 and <= 128, pooled rows are
// 8-byte aligned. Row r (0-based within the launch) scores blocks below
// (position_start + r + 1) / compress into scores[r * score_stride + block];
// every other score element is left untouched.
// Grid [ceil(bound / 256), ceil(rows / R)], block 256, where bound may exceed
// the live block_count (replay launches a fixed grid and patches block_count /
// position_start): workgroups and threads past the rows' visible-block ceiling
// return before touching the pooled keys.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float4 qsa_bf16_scores_load4(
    const __hip_bfloat16* row, int d4) {
    const uint2 w = reinterpret_cast<const uint2*>(row)[d4];
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xFFFF0000u),
                       __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xFFFF0000u));
}

template <int R>
static __device__ __forceinline__ void qsa_bf16_scores_rows(
    const float* query,
    const __hip_bfloat16* pooled,
    float* scores,
    int rows,
    int query_row_stride,
    int block_count,
    int index_dim,
    int compress,
    int position_start,
    int score_stride) {
    const int row0 = (int)blockIdx.y * R;
    const int nrows = min(R, rows - row0);
    // Largest visible-block count over this workgroup's rows (row r sees
    // (position_start + row0 + r + 1) / compress; the last row dominates).
    // Uniform per workgroup, so returning before __syncthreads is safe.
    const int live = nrows > 0 ? min(block_count, (position_start + row0 + nrows) / compress) : 0;
    if ((int)blockIdx.x * 256 >= live) return;

    __shared__ __align__(16) float qs[R * 4 * 128];
    const int width = 4 * index_dim;
    for (int i = (int)threadIdx.x; i < nrows * width; i += 256) {
        const int r = i / width;
        qs[r * 4 * 128 + (i - r * width) / index_dim * 128 + (i - r * width) % index_dim] =
            query[(row0 + r) * query_row_stride + (i - r * width)];
    }
    __syncthreads();
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (block >= live) return;

    const __hip_bfloat16* row = pooled + block * index_dim;
    float dot[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int h = 0; h < 4; ++h) dot[r][h] = 0.0f;
    for (int d4 = 0; d4 < index_dim / 4; ++d4) {
        const float4 p = qsa_bf16_scores_load4(row, d4);
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float4 a =
                    *reinterpret_cast<const float4*>(qs + r * 4 * 128 + h * 128 + d4 * 4);
                dot[r][h] += a.x * p.x;
                dot[r][h] += a.y * p.y;
                dot[r][h] += a.z * p.z;
                dot[r][h] += a.w * p.w;
            }
        }
    }
    // No break/continue: the loop must fully unroll so `dot` stays in
    // registers (the 16-row kernel spills it to scratch otherwise).
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int visible_blocks = (position_start + row0 + r + 1) / compress;
        if (r < nrows && block < visible_blocks) {
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < 4; ++h) score += fmaxf(dot[r][h] / sqrtf((float)index_dim), 0.0f);
            scores[(row0 + r) * score_stride + block] = score;
        }
    }
}

// Eight rows per pooled read: decode and few-row verify.
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows8_bf16(
    const float* query, const __hip_bfloat16* pooled, float* scores, int rows,
    int query_row_stride, int block_count, int index_dim, int compress, int position_start,
    int score_stride) {
    qsa_bf16_scores_rows<8>(query, pooled, scores, rows, query_row_stride, block_count,
                            index_dim, compress, position_start, score_stride);
}

// Sixteen rows per pooled read: prefill.
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows16_bf16(
    const float* query, const __hip_bfloat16* pooled, float* scores, int rows,
    int query_row_stride, int block_count, int index_dim, int compress, int position_start,
    int score_stride) {
    qsa_bf16_scores_rows<16>(query, pooled, scores, rows, query_row_stride, block_count,
                             index_dim, compress, position_start, score_stride);
}

// Thirty-two rows per pooled read: row-exact wide verify (index_heads == 4,
// index_dim == 128 ONLY; the host admits no other width for this symbol).
// Same ABI, grid x and score cells as rows8/rows16; grid y is
// ceil(rows / 32), so a workgroup scores up to 32 consecutive query rows.
//
// Schedule (frozen in the wide-verify QSA design, section 3B):
//  * Each lane widens its 128 pooled values once into the fixed-index register
//    array `key` (every access is a compile-time index after full unrolling;
//    no private memory). Lanes past the live count hold zeros and issue no
//    pooled load.
//  * Queries are staged in banks of four rows, `qs[4][4][128]` = 8192 B LDS,
//    loading only the bank's live rows (never a row >= nrows).
//  * A bank's rows are scored serially with four live dot accumulators. The
//    per-(row, head) chain, the division and the head sum are exactly the
//    rows8/rows16 body in `qsa_bf16_scores_rows<R>` above: dot starts at +0,
//    `dot += a * p` over ascending d (x, y, z, w within each float4), then
//    `score += fmaxf(dot / sqrtf((float)index_dim), 0.0f)` over heads 0..3
//    from 0.0f. Only cells with block < (position_start + row + 1) / compress
//    are stored.
//  * Barriers are uniform: a workgroup wholly past its largest live ceiling
//    returns before the first one; afterwards no lane returns early (past-live
//    lanes only skip loads, FMAs and stores), so every lane reaches the same
//    2 * ceil(nrows / 4) - 1 barriers (at most 15 for a full tile): one after
//    each bank load and one before each query-bank reuse (none after the
//    last bank).
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows32_bf16(
    const float* query, const __hip_bfloat16* pooled, float* scores, int rows,
    int query_row_stride, int block_count, int index_dim, int compress, int position_start,
    int score_stride) {
    const int row0 = (int)blockIdx.y * 32;
    const int nrows = min(32, rows - row0);
    // Largest visible-block count over this workgroup's rows (the last row
    // dominates). Uniform per workgroup: this is the only return before a
    // barrier.
    const int live = nrows > 0 ? min(block_count, (position_start + row0 + nrows) / compress) : 0;
    if ((int)blockIdx.x * 256 >= live) return;

    __shared__ __align__(16) float qs[4 * 4 * 128];
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const bool lane_live = block < live;

    float key[128];
#pragma unroll
    for (int d4 = 0; d4 < 32; ++d4) {
        float4 p = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (lane_live) p = qsa_bf16_scores_load4(pooled + block * 128, d4);
        key[4 * d4 + 0] = p.x;
        key[4 * d4 + 1] = p.y;
        key[4 * d4 + 2] = p.z;
        key[4 * d4 + 3] = p.w;
    }

    const int banks = (nrows + 3) / 4;
#pragma unroll 1
    for (int bank = 0; bank < banks; ++bank) {
        const int brow0 = bank * 4;
        const int brows = min(4, nrows - brow0);
        for (int i = (int)threadIdx.x; i < brows * 512; i += 256) {
            const int r = i >> 9;
            qs[i] = query[(row0 + brow0 + r) * query_row_stride + (i & 511)];
        }
        __syncthreads();
#pragma unroll 1
        for (int r = 0; r < brows; ++r) {
            const int visible_blocks = (position_start + row0 + brow0 + r + 1) / compress;
            if (lane_live && block < visible_blocks) {
                float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int d4 = 0; d4 < 32; ++d4) {
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float4 a =
                            *reinterpret_cast<const float4*>(qs + r * 512 + h * 128 + d4 * 4);
                        dot[h] += a.x * key[4 * d4 + 0];
                        dot[h] += a.y * key[4 * d4 + 1];
                        dot[h] += a.z * key[4 * d4 + 2];
                        dot[h] += a.w * key[4 * d4 + 3];
                    }
                }
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < 4; ++h) score += fmaxf(dot[h] / sqrtf((float)index_dim), 0.0f);
                scores[(row0 + brow0 + r) * score_stride + block] = score;
            }
        }
        if (bank + 1 < banks) __syncthreads();
    }
}
