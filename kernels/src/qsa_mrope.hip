// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.
//
// Image-only QSA RMSNorm / pooled-key mRoPE, preserving the arithmetic of
// tensor_ops while leaving that measured text module's source unchanged.
// RopeYarn and rope_yarn_sincos are prepended by kernels::QSA_MROPE_SRC.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float qsa_index_f32(float value) {
    return value;
}

static __device__ __forceinline__ float qsa_index_f32(__hip_bfloat16 value) {
    return bf16_to_f32(value);
}

static __device__ __forceinline__ void qsa_index_store(float* slot, __hip_bfloat16 value) {
    *slot = bf16_to_f32(value);
}

static __device__ __forceinline__ void qsa_index_store(__hip_bfloat16* slot, __hip_bfloat16 value) {
    *slot = value;
}

static __device__ __forceinline__ int qsa_mrope_position(
    int position, int frequency, const int* positions, int prompt_rows,
    int delta, int section_h, int section_w) {
    if (positions == nullptr) return position;
    if (position >= prompt_rows) return position + delta;
    int axis = frequency % 3 == 1 && frequency < 3 * section_h ? 1
        : frequency % 3 == 2 && frequency < 3 * section_w ? 2 : 0;
    return positions[position * 3 + axis];
}

static __device__ __forceinline__ float qsa_norm_mrope_head(
    const float* head_values,
    const __hip_bfloat16* norm,
    int head_dim,
    int position,
    int rotary_dim,
    RopeYarn yarn,
    float* input_row,
    float* row_sum,
    const int* positions = nullptr, int prompt_rows = 0, int delta = 0,
    int section_h = 0, int section_w = 0) {
    const int channel = (int)threadIdx.x;
    if (channel < head_dim) {
        input_row[channel] = head_values[channel];
    }
    __syncthreads();
    // One wave runs the serial i-order sum (identical in every lane) instead
    // of every wave repeating it.
    if (threadIdx.x < 32) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float row_value = input_row[i];
            sum += row_value * row_value;
        }
        if (threadIdx.x == 0) *row_sum = sum;
    }
    __syncthreads();
    if (channel >= head_dim) return 0.0f;
    float inv = rsqrtf(*row_sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)qsa_mrope_position(position, base, positions,
            prompt_rows, delta, section_h, section_w)
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, base, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    return value;
}

extern "C" __global__ void indexed_attention_norm_mrope_f32_batched(
    float* values, const __hip_bfloat16* norm, const int* positions,
    int rows, int row_stride, int heads, int head_dim, int head_stride,
    int position_start, int rotary_dim, int prompt_rows, int delta,
    int section_h, int section_w, RopeYarn yarn) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    if (row >= rows || head >= heads) return;
    int start = row * row_stride + head * head_stride;
    __shared__ float input_row[256];
    __shared__ float row_sum;
    const float value = qsa_norm_mrope_head(values + start, norm, head_dim,
        position_start + row, rotary_dim, yarn, input_row, &row_sum,
        positions, prompt_rows, delta, section_h, section_w);
    if ((int)threadIdx.x < head_dim) values[start + threadIdx.x] = value;
}

template <typename Key>
static __device__ __forceinline__ void qsa_pool_mrope(
    const Key* raw_keys,
    Key* pooled,
    const __hip_bfloat16* norm,
    int block_count,
    int compress,
    int index_dim,
    int first_block,
    RopeYarn yarn,
    const int* positions = nullptr, int prompt_rows = 0, int delta = 0,
    int section_h = 0, int section_w = 0) {
    // The grid starts at `first_block` (0 for the full pool): blocks below it
    // already hold this kernel's output for their unchanged raw keys (the
    // incremental decode caller), so an incremental launch covers only the
    // blocks its rows complete.
    int block = first_block + (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if ( block >= block_count || compress <= 0) return;
    bool active = channel < index_dim;
    int base = block * compress * index_dim;
    float mean = 0.0f;
    if (active) {
        for (int row = 0; row < compress; ++row) {
            mean += qsa_index_f32(raw_keys[base + row * index_dim + channel])
                / (float)compress;
        }
    }
    __hip_bfloat16 mean_bf16 = bf16_round(mean);
    __shared__ float partial[256];
    float squared = active
        ? bf16_to_f32(bf16_round(bf16_to_f32(mean_bf16)
            * bf16_to_f32(mean_bf16)))
        : 0.0f;
    partial[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!active) return;
    float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
    float norm_value = norm == nullptr
        ? 0.0f
        : bf16_to_f32(norm[channel]);
    float value = bf16_to_f32(bf16_round(
        bf16_to_f32(mean_bf16) * inv * (1.0f + norm_value)));
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    int half = rotary_dim / 2;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other_mean = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other_mean += qsa_index_f32(raw_keys[base + row * index_dim + partner])
                / (float)compress;
        }
        __hip_bfloat16 other_bf16 = bf16_round(other_mean);
        float other_norm = norm == nullptr
            ? 0.0f
            : bf16_to_f32(norm[partner]);
        float other = bf16_to_f32(bf16_round(
            bf16_to_f32(other_bf16) * inv * (1.0f + other_norm)));
        int pair = channel < half ? channel : partner;
        float angle = (float)qsa_mrope_position(block * compress, pair,
            positions, prompt_rows, delta, section_h, section_w)
            * powf(10000000.0f, -((float)(2 * pair)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, pair, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    qsa_index_store(pooled + block * index_dim + channel, bf16_round(value));
}

extern "C" __global__ void indexed_attention_pool_mrope_f32(
    const float* raw_keys, float* pooled, const __hip_bfloat16* norm,
    const int* positions, int block_count, int compress, int index_dim,
    int first_block, int prompt_rows, int delta, int section_h, int section_w,
    RopeYarn yarn) {
    qsa_pool_mrope(raw_keys, pooled, norm, block_count, compress, index_dim,
        first_block, yarn, positions, prompt_rows, delta, section_h, section_w);
}

extern "C" __global__ void indexed_attention_pool_mrope_bf16(
    const __hip_bfloat16* raw_keys, __hip_bfloat16* pooled,
    const __hip_bfloat16* norm, const int* positions, int block_count,
    int compress, int index_dim, int first_block, int prompt_rows, int delta,
    int section_h, int section_w, RopeYarn yarn) {
    qsa_pool_mrope(raw_keys, pooled, norm, block_count, compress, index_dim,
        first_block, yarn, positions, prompt_rows, delta, section_h, section_w);
}
