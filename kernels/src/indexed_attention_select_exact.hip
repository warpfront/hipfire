// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact pp8192 selector. Host gates live pooled count <= 2048,
// budget_blocks <= 512, blockDim.x == 256, and excludes capture. Dynamic LDS
// is max(6144, scores_global ? 0 : 4 * max_live_count) bytes. ABI and score
// association match tensor_ops.hip; larger contexts keep the incumbent.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float qsa_exact_f32(float value) { return value; }
static __device__ __forceinline__ float qsa_exact_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}
static __device__ __forceinline__ float4 qsa_exact_load4(const float* row, int d4) {
    return reinterpret_cast<const float4*>(row)[d4];
}

static __device__ __forceinline__ float4 qsa_exact_load4(const __hip_bfloat16* row, int d4) {
    const uint2 w = reinterpret_cast<const uint2*>(row)[d4];
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xFFFF0000u),
                       __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xFFFF0000u));
}
template <typename P>
static __device__ __forceinline__ float qsa_exact_score4(
    const float* query,
    const P* pooled,
    int block,
    int index_heads,
    int index_dim) {
    const P* row = pooled + block * index_dim;
    float score = 0.0f;
    int head = 0;
    if ((index_dim & 3) == 0
        && ((reinterpret_cast<unsigned long long>(query) & 15)
            | (reinterpret_cast<unsigned long long>(row) & (4 * sizeof(P) - 1))) == 0) {
        for (; head + 4 <= index_heads; head += 4) {
            const float4* q0 = reinterpret_cast<const float4*>(query + head * index_dim);
            const float4* q1 = q0 + index_dim / 4;
            const float4* q2 = q1 + index_dim / 4;
            const float4* q3 = q2 + index_dim / 4;
            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#pragma unroll 4
            for (int d4 = 0; d4 < index_dim / 4; ++d4) {
                const float4 p = qsa_exact_load4(row, d4);
                const float4 a = q0[d4], b = q1[d4], c = q2[d4], e = q3[d4];
                dot0 += a.x * p.x; dot0 += a.y * p.y; dot0 += a.z * p.z; dot0 += a.w * p.w;
                dot1 += b.x * p.x; dot1 += b.y * p.y; dot1 += b.z * p.z; dot1 += b.w * p.w;
                dot2 += c.x * p.x; dot2 += c.y * p.y; dot2 += c.z * p.z; dot2 += c.w * p.w;
                dot3 += e.x * p.x; dot3 += e.y * p.y; dot3 += e.z * p.z; dot3 += e.w * p.w;
            }
            score += fmaxf(dot0 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot1 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot2 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot3 / sqrtf((float)index_dim), 0.0f);
        }
    }
    for (; head < index_heads; ++head) {
        float dot = 0.0f;
        for (int d = 0; d < index_dim; ++d) {
            dot += query[head * index_dim + d] * qsa_exact_f32(row[d]);
        }
        score += fmaxf(dot / sqrtf((float)index_dim), 0.0f);
    }
    return score;
}

// fmaxf removes dot NaNs. Scores are nonnegative, so their unsigned IEEE
// bit pattern gives exact score DESC ordering (normalize signed zero only in
// this key). Complementing the index gives index ASC without a second load.
using QsaExactKey = unsigned long long;
static __device__ __forceinline__ QsaExactKey qsa_exact_key(float score, int index) {
    return (static_cast<QsaExactKey>(__float_as_uint(score) & 0x7fffffffu) << 32)
        | (0xffffffffu - static_cast<unsigned int>(index));
}

// Fixed integer XOR exchanges within each logical 32-lane wave group.
// DPP handles distances 1/2/4; bit-mode swizzles preserve group boundaries
// for distances 8/16 on both wave32 and wave64.
static __device__ __forceinline__ int qsa_exact_xor32(int value, int step) {
    if (step == 1) return __builtin_amdgcn_mov_dpp(value, 0xb1, 0xf, 0xf, false);
    if (step == 2) return __builtin_amdgcn_mov_dpp(value, 0x4e, 0xf, 0xf, false);
    if (step == 4) return __builtin_amdgcn_mov_dpp8(value, 0x688fac);
    if (step == 8) return __builtin_amdgcn_ds_swizzle(value, 0x201f);
    return __builtin_amdgcn_ds_swizzle(value, 0x401f); // step == 16
}

static __device__ __forceinline__ QsaExactKey qsa_exact_xor64(QsaExactKey value, int step) {
    return (static_cast<QsaExactKey>(static_cast<unsigned int>(qsa_exact_xor32(value >> 32, step))) << 32)
        | static_cast<unsigned int>(qsa_exact_xor32(value, step));
}

template <typename P>
static __device__ __forceinline__ void qsa_select_exact(
    const float* query, const P* pooled, int* selected,
    int rows, int query_row_stride, int block_count,
    int index_heads, int index_dim, int budget_blocks,
    int compress, int position_start, int capacity,
    int* mirror, float* scores_global, int scores_stride) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    if (row != rows - 1) mirror = nullptr;
    extern __shared__ __align__(16) unsigned char scratch[];
    float* scores_lds = reinterpret_cast<float*>(scratch);
    float* scores = scores_global != nullptr
        ? scores_global + (size_t)row * (size_t)scores_stride : scores_lds;
    const int tid = (int)threadIdx.x;
    const int visible = position_start + row + 1;
    const int count = min(visible / compress, block_count);
    const float* row_query = query + row * query_row_stride;
    int* output = selected + row * capacity;
    if (count > 0) {
        for (int b = (int)threadIdx.x; b < count; b += (int)blockDim.x) {
            scores[b] = qsa_exact_score4(row_query, pooled, b, index_heads, index_dim);
        }
    }
    // Reuse score LDS after every score has been copied into register keys.
    // The merge scratch is one tile (2 KiB) plus an in-place top512 (4 KiB).
    QsaExactKey* tile_indices = reinterpret_cast<QsaExactKey*>(scratch);
    QsaExactKey* top = tile_indices + 256;
    __syncthreads();
    const int chosen = min(budget_blocks, count);
    // Each logical wave32 sorts one 256-element tile in registers. This
    // removes inter-wave sort barriers; tiles still merge left-to-right.
    const int tile_wave = tid / 32;
    const int lane = tid % 32;
    QsaExactKey keys[8];
#pragma unroll 8
    for (int j = 0; j < 8; ++j) {
        const int b = tile_wave * 256 + lane * 8 + j;
        keys[j] = b < count ? qsa_exact_key(scores[b], b) : 0;
    }
    if (tile_wave * 256 < count) {
#pragma unroll 8
        for (int size = 2; size <= 256; size <<= 1) {
#pragma unroll 8
            for (int step = size >> 1; step; step >>= 1) {
#pragma unroll 8
                for (int j = 0; j < 8; ++j) {
                    if (step < 8) {
                        if ((j & step) == 0) {
                            const int k = j ^ step;
                            const bool first = ((lane * 8 + j) & size) == 0;
                            const bool before = keys[j] > keys[k];
                            if (first != before) {
                                const QsaExactKey key = keys[j];
                                keys[j] = keys[k]; keys[k] = key;
                            }
                        }
                    } else {
                        const QsaExactKey other_key = qsa_exact_xor64(keys[j], step / 8);
                        const bool first = (((lane * 8 + j) & size) == 0)
                            == (((lane * 8 + j) & step) == 0);
                        const bool before = keys[j] > other_key;
                        if (first != before) keys[j] = other_key;
                    }
                }
            }
        }
    }
    // No score LDS read may remain when the first tile overwrites scratch.
    __syncthreads();
    int length = 0;
    for (int base = 0; base < count; base += 256) {
        const int tile_length = min(256, count - base);
        if (tile_wave == base / 256) {
#pragma unroll 8
            for (int j = 0; j < 8; ++j) {
                const int rank = lane * 8 + j;
                // XOR-swizzle the tile so eight-register wave stores do not
                // serialize on eight-way LDS bank conflicts.
                tile_indices[rank ^ (rank >> 5)] = keys[j];
            }
        }
        __syncthreads();
        if (base == 0) {
            if (tid < tile_length) top[tid] = tile_indices[tid ^ (tid >> 5)];
            length = tile_length;
        } else {
            // Merge in fixed tile order. Binary-search co-ranks are unique
            // under index-ASC ties; each top512 slot has exactly one writer.
            QsaExactKey candidates[2] = {0, 0};
            const int next_length = min(512, length + tile_length);
            const int rank = tid * 2;
            if (rank < next_length) {
                // One co-rank locates a two-element output interval. Consume
                // its next two keys serially, preserving the exact merge order.
                int lo = max(0, rank - tile_length);
                int hi = min(rank, length);
                while (lo < hi) {
                    const int mid = (lo + hi + 1) >> 1;
                    const int right = rank - mid;
                    if (right < tile_length
                        && tile_indices[right ^ (right >> 5)] > top[mid - 1])
                        hi = mid - 1;
                    else lo = mid;
                }
                int right = rank - lo;
#pragma unroll 2
                for (int part = 0; part < 2; ++part) {
                    if (rank + part < next_length) {
                        const bool take_left = right == tile_length
                            || (lo < length && top[lo] > tile_indices[right ^ (right >> 5)]);
                        candidates[part] = take_left ? top[lo++]
                            : tile_indices[right ^ (right >> 5)];
                        if (!take_left) ++right;
                    }
                }
            }
            // Both output keys must be read before any in-place overwrite.
            __syncthreads();
#pragma unroll 2
            for (int part = 0; part < 2; ++part) {
                if (rank + part < next_length) top[rank + part] = candidates[part];
            }
            length = min(512, length + tile_length);
        }
        __syncthreads();
    }
    const int tail_start = count * compress;
    const int offset = chosen * compress;
    // Selected blocks and the causal tail write a contiguous prefix. Only
    // the suffix owes -1; clearing that prefix first would duplicate stores
    // and evict pooled rows during the score phase.
    const int written = min(capacity, offset + (visible - tail_start));
    for (int i = written + tid; i < capacity; i += 256) {
        output[i] = -1;
        if (mirror) mirror[i] = -1;
    }
    for (int rank = tid; rank < chosen; rank += 256) {
        const int block = 0xffffffffu - static_cast<unsigned int>(top[rank]);
        const int first = rank * compress;
        if (compress == 4 && first + 4 <= capacity) {
            const int base_token = block * 4;
            const int4 tokens = make_int4(base_token, base_token + 1, base_token + 2, base_token + 3);
            // Capacity can misalign row bases; memcpy only promises int alignment.
            __builtin_memcpy(output + first, &tokens, sizeof(tokens));
            if (mirror) __builtin_memcpy(mirror + first, &tokens, sizeof(tokens));
        } else for (int slot = 0; slot < compress; ++slot) {
            const int index = rank * compress + slot;
            const int token = block * compress + slot;
            if (index < capacity && token < visible) {
                output[index] = token;
                if (mirror) mirror[index] = token;
            }
        }
    }
    for (int token = tail_start + tid;
         token < visible && offset + token - tail_start < capacity; token += 256) {
        output[offset + token - tail_start] = token;
        if (mirror) mirror[offset + token - tail_start] = token;
    }
}

#define QSA_SELECT_EXACT(NAME, POOLED) \
extern "C" __global__ void NAME(const float* query, const POOLED* pooled, int* selected, \
    int rows, int query_row_stride, int block_count, int index_heads, int index_dim, \
    int budget_blocks, int compress, int position_start, int capacity, \
    int* mirror, float* scores_global, int scores_stride) { \
    qsa_select_exact(query, pooled, selected, rows, query_row_stride, block_count, \
        index_heads, index_dim, budget_blocks, compress, position_start, capacity, \
        mirror, scores_global, scores_stride); \
}
QSA_SELECT_EXACT(indexed_attention_select_f32_batched_exact, float)
QSA_SELECT_EXACT(indexed_attention_select_bf16_batched_exact, __hip_bfloat16)
#undef QSA_SELECT_EXACT
