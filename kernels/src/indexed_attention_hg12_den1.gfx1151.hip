// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QSA F32 grouped attention (twelve query heads per KV
// head), one denominator chain per head.
//
// Bitwise twin of `indexed_attention_attention_f32_batched_hg12`
// (tensor_ops.hip) with the same arguments, grid [n_kv_heads, 1, rows] and
// block 256.  The incumbent has every one of its 256 channel threads carry
// the twelve softmax denominators: each thread adds the same weights in the
// same row order, so all 256 copies are equal (twelve adds per row per
// thread, half the value-phase VALU work).  Here each lane carries ONE
// denominator, lane l for head l % 12 (lanes 0-11 own the twelve heads),
// adding weights[u][h] for u = 0, 1, ... exactly as every incumbent thread
// does: one LDS read and one add per row instead of twelve adds.  The adds
// stay branch-free in every wave so the dependent chain interleaves with the
// numerator FMAs (a single-wave copy serialized that chain behind the
// barrier); wave 0's lanes 0-11 publish them through LDS before the
// epilogue.
// Everything else is the incumbent's text: the per-thread serial d-order
// score dots, the 1/sqrt(256) scale, the order-free fmaxf maxima, the
// expf(score - max) weights (0 for invalid rows), the in-order numerators
// (`numerator += w * value`) and the epilogue
// `denominator > 0 ? numerator / denominator / (1 + expf(-gate)) : 0`.
#include <hip/hip_runtime.h>

#define QSA_G 12
#define QSA_T 9
#define QSA_HD 256
extern "C" __global__ __launch_bounds__(256) void indexed_attention_attention_f32_batched_hg12_den1(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    __shared__ __align__(16) float weights[256 * QSA_G];  // one tile, [row][head]
    __shared__ float wave_max[8 * QSA_G];
    __shared__ float maxes[QSA_G];
    __shared__ float denominators[QSA_G];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const bool den_wave = __builtin_amdgcn_readfirstlane(tid >> 5) == 0;
    const int den_head = lane % QSA_G;
    const int kv_head = (int)blockIdx.x;
    const int head0 = kv_head * QSA_G;
    const int kv_stride = n_kv_heads * QSA_HD;
    const float* keys = full_keys + kv_head * QSA_HD;
    const float* values = full_values + kv_head * QSA_HD;
    const int* row_selected = selected_indices + row * capacity;
    const int q_row = row * 2 * n_heads * QSA_HD;
    const float* __restrict__ qg = q_with_gate + q_row + head0 * QSA_HD * 2;

    float scores[QSA_T][QSA_G];
    unsigned valid_rows = 0u;  // bit t: this thread's tile-t row is a live key
    float partial_max[QSA_G];
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) partial_max[h] = -3.402823466e+38f;
#pragma unroll
    for (int t = 0; t < QSA_T; ++t) {
        const int index = t * 256 + tid;
        int token = index < selected_len ? row_selected[index] : -1;
        if (!(token >= 0 && token < full_capacity && token < visible)) token = -1;
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) scores[t][h] = 0.0f;
        if (token >= 0) {
            const float4* key4 = reinterpret_cast<const float4*>(keys + token * kv_stride);
#pragma unroll 2
            for (int d4 = 0; d4 < QSA_HD / 4; ++d4) {
                const float4 k4 = key4[d4];
#pragma unroll
                for (int h = 0; h < QSA_G; ++h) {
                    const float4 q4 = reinterpret_cast<const float4*>(qg + h * QSA_HD * 2)[d4];
                    scores[t][h] += q4.x * k4.x;
                    scores[t][h] += q4.y * k4.y;
                    scores[t][h] += q4.z * k4.z;
                    scores[t][h] += q4.w * k4.w;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) {
            scores[t][h] = scores[t][h] / sqrtf((float)QSA_HD);
            if (token >= 0) partial_max[h] = fmaxf(partial_max[h], scores[t][h]);
        }
        if (token >= 0) valid_rows |= 1u << t;
    }
    // fmaxf is exact and order-independent: any reduction tree matches.
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        float m = partial_max[h];
        for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor(m, offset, 32));
        partial_max[h] = m;
    }
    if ((tid & 31) == 0) {
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) wave_max[(tid >> 5) * QSA_G + h] = partial_max[h];
    }
    __syncthreads();
    if (tid < QSA_G) {
        float m = -3.402823466e+38f;
        for (int w = 0; w < 8; ++w) m = fmaxf(m, wave_max[w * QSA_G + tid]);
        maxes[tid] = m;
    }
    __syncthreads();

    const int channel = tid;
    // Lane l: head (l % 12)'s denominator; wave 0's lanes 0-11 publish.
    float denominator = 0.0f;
    float numerator[QSA_G];
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) numerator[h] = 0.0f;
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x31004000);
    const int voff = channel * 4;
    auto value_at = [&](int index) {
        const int token = __builtin_amdgcn_readfirstlane(row_selected[index]);
        const bool ok = token >= 0 && token < full_capacity && token < visible;
        const float v = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(
            vrsrc, voff, ok ? token * kv_stride * 4 : 0, 0));
        return ok ? v : 0.0f;
    };
#pragma unroll
    for (int t = 0; t < QSA_T; ++t) {
        const int base = t * 256;
        if (base >= selected_len) break;
        // Invalid rows keep weight 0: their score is never exponentiated.
        const bool valid = (valid_rows >> t) & 1u;
#pragma unroll
        for (int h = 0; h < QSA_G; ++h)
            weights[tid * QSA_G + h] = valid ? expf(scores[t][h] - maxes[h]) : 0.0f;
        __syncthreads();
        const int count = min(256, selected_len - base);
        int u = 0;
        for (; u + 8 <= count; u += 8) {
            float value[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) value[k] = value_at(base + u + k);
#pragma unroll
            for (int k = 0; k < 8; ++k) denominator += weights[(u + k) * QSA_G + den_head];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float4* w4 = reinterpret_cast<const float4*>(weights + (u + k) * QSA_G);
#pragma unroll
                for (int q = 0; q < QSA_G / 4; ++q) {
                    const float4 w = w4[q];
                    numerator[4 * q] += w.x * value[k];
                    numerator[4 * q + 1] += w.y * value[k];
                    numerator[4 * q + 2] += w.z * value[k];
                    numerator[4 * q + 3] += w.w * value[k];
                }
            }
        }
        for (; u < count; ++u) {
            const float value = value_at(base + u);
            denominator += weights[u * QSA_G + den_head];
            const float4* w4 = reinterpret_cast<const float4*>(weights + u * QSA_G);
#pragma unroll
            for (int q = 0; q < QSA_G / 4; ++q) {
                const float4 w = w4[q];
                numerator[4 * q] += w.x * value;
                numerator[4 * q + 1] += w.y * value;
                numerator[4 * q + 2] += w.z * value;
                numerator[4 * q + 3] += w.w * value;
            }
        }
        __syncthreads();
    }
    if (den_wave && lane < QSA_G) denominators[lane] = denominator;
    __syncthreads();
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        int head = head0 + h;
        const float den = denominators[h];
        float gate = q_with_gate[q_row + head * QSA_HD * 2 + QSA_HD + channel];
        output[row * n_heads * QSA_HD + head * QSA_HD + channel] = den > 0.0f
            ? numerator[h] / den / (1.0f + expf(-gate))
            : 0.0f;
    }
}
#undef QSA_G
#undef QSA_T
#undef QSA_HD
