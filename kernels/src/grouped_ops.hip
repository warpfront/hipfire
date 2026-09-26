// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

static __device__ __forceinline__ float grouped_sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

static __device__ __forceinline__ float grouped_silu(float value) {
    return value * grouped_sigmoid(value);
}

extern "C" __global__ void grouped_gather_convert_bf16(
    const unsigned short* __restrict__ staged,
    float* __restrict__ output,
    int tokens,
    int rows_per_token,
    int row_width) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = tokens * rows_per_token * row_width;
    if (index >= total) {
        return;
    }
    output[index] = __uint_as_float(static_cast<unsigned int>(staged[index]) << 16);
}

extern "C" __global__ void grouped_linear_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int tokens,
    int in_dim,
    int out_dim) {
    const int out = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (token >= tokens || out >= out_dim) {
        return;
    }
    const float* input_row = input + token * in_dim;
    const float* weight_row = weight + out * in_dim;
    float sum = 0.0f;
    for (int index = 0; index < in_dim; ++index) {
        sum += input_row[index] * weight_row[index];
    }
    output[token * out_dim + out] = sum;
}

// One block/thread handles one token and one residual branch.  This is a
// correctness-first fallback; the surrounding projection kernels remain
// independently batchable and can be replaced by a tiled implementation
// without changing this ABI or equation.
extern "C" __global__ void grouped_gate_f32(
    const float* __restrict__ key,
    const float* __restrict__ query,
    const float* __restrict__ value,
    const float* __restrict__ key_norm,
    const float* __restrict__ query_norm,
    float* __restrict__ gated,
    int tokens,
    int hc_count,
    int hidden_size,
    float epsilon) {
    if (threadIdx.x != 0) {
        return;
    }
    const int branch = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (branch >= hc_count || token >= tokens) {
        return;
    }
    const int channels = hc_count * hidden_size;
    const int base = token * channels + branch * hidden_size;
    float key_square = 0.0f;
    float query_square = 0.0f;
    for (int index = 0; index < hidden_size; ++index) {
        const float key_value = key[base + index];
        const float query_value = query[base + index];
        key_square += key_value * key_value;
        query_square += query_value * query_value;
    }
    const float key_inverse = rsqrtf(key_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
    const float query_inverse = rsqrtf(query_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
    float dot = 0.0f;
    for (int index = 0; index < hidden_size; ++index) {
        const int norm_index = branch * hidden_size + index;
        const float key_value = key[base + index] * key_inverse * (1.0f + key_norm[norm_index]);
        const float query_value = query[base + index] * query_inverse * (1.0f + query_norm[norm_index]);
        dot += key_value * query_value;
    }
    dot /= sqrtf(static_cast<float>(hidden_size));
    const float sign = dot >= 0.0f ? 1.0f : -1.0f;
    const float transformed = sign * sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
    const float gate = grouped_sigmoid(transformed);
    const float* value_row = value + token * hidden_size;
    float* gated_row = gated + base;
    for (int index = 0; index < hidden_size; ++index) {
        gated_row[index] = gate * value_row[index];
    }
}

// Grouped RMS normalization with the checkpoint's `(1 + weight)` multiplier.
// Grid is [groups, tokens], block is 256 threads.
extern "C" __global__ void grouped_norm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int tokens,
    int groups,
    int group_size,
    float epsilon) {
    const int group = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (group >= groups || token >= tokens) {
        return;
    }
    __shared__ float partial[256];
    const int base = token * groups * group_size + group * group_size;
    float sum = 0.0f;
    for (int index = lane; index < group_size; index += static_cast<int>(blockDim.x)) {
        const float value = input[base + index];
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float inverse = rsqrtf(partial[0] / static_cast<float>(group_size) + fmaxf(epsilon, 1.0e-12f));
    for (int index = lane; index < group_size; index += static_cast<int>(blockDim.x)) {
        output[base + index] = input[base + index] * inverse * (1.0f + weight[group * group_size + index]);
    }
}

// Each thread owns one channel and evaluates the fixture's dilated window over
// the concatenated history/input stream.  The state is the trailing history
// rows, so token chunks compose without changing the oracle's indexing.
extern "C" __global__ void grouped_depthwise_conv_silu_add_f32(
    const float* __restrict__ gated,
    const float* __restrict__ normed,
    const float* __restrict__ conv_weight,
    float* __restrict__ state,
    float* __restrict__ output,
    int tokens,
    int channels,
    int kernel_size,
    int dilation) {
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) {
        return;
    }
    const int history_rows = (kernel_size - 1) * dilation;
    for (int token = 0; token < tokens; ++token) {
        float total = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int source = token + tap * dilation;
            float value = 0.0f;
            if (source < history_rows) {
                value = state[source * channels + channel];
            } else {
                const int input_index = source - history_rows;
                if (input_index < tokens) {
                    value = normed[input_index * channels + channel];
                }
            }
            total += value * conv_weight[channel * kernel_size + tap];
        }
        output[token * channels + channel] =
            gated[token * channels + channel] + grouped_silu(total);
    }
    for (int row = 0; row < history_rows; ++row) {
        const int source = tokens + row;
        if (source < history_rows) {
            state[row * channels + channel] = state[source * channels + channel];
        } else {
            state[row * channels + channel] =
                normed[(source - history_rows) * channels + channel];
        }
    }
}
// One block per (branch, token), dynamic LDS 2 * hidden_size floats.  The
// block stages the key/query rows and the normalized products in LDS; thread 0
// runs the three serial reductions in the reference's index order and every
// thread writes the gated row.  Same expressions per element, so the result is
// bitwise the single-thread reference's.
extern "C" __global__ void grouped_gate_bf16(
    const float* __restrict__ key,
    const float* __restrict__ query,
    const float* __restrict__ value,
    const hip_bfloat16* __restrict__ key_norm,
    const hip_bfloat16* __restrict__ query_norm,
    float* __restrict__ gated,
    int tokens,
    int hc_count,
    int hidden_size,
    float epsilon) {
    extern __shared__ float gate_lds[];
    float* key_row = gate_lds;
    float* query_row = gate_lds + hidden_size;
    __shared__ float key_inverse, query_inverse, gate;
    const int branch = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (branch >= hc_count || token >= tokens) return;
    const int channels = hc_count * hidden_size;
    const int base = token * channels + branch * hidden_size;
    for (int index = tid; index < hidden_size; index += static_cast<int>(blockDim.x)) {
        key_row[index] = key[base + index];
        query_row[index] = query[base + index];
    }
    __syncthreads();
    if (tid == 0) {
        float key_square = 0.0f;
        float query_square = 0.0f;
        for (int index = 0; index < hidden_size; ++index) {
            const float key_value = key_row[index];
            const float query_value = query_row[index];
            key_square += key_value * key_value;
            query_square += query_value * query_value;
        }
        key_inverse = rsqrtf(key_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
        query_inverse = rsqrtf(query_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
    }
    __syncthreads();
    for (int index = tid; index < hidden_size; index += static_cast<int>(blockDim.x)) {
        const int norm_index = branch * hidden_size + index;
        key_row[index] = key_row[index] * key_inverse
            * (1.0f + static_cast<float>(key_norm[norm_index]));
        query_row[index] = query_row[index] * query_inverse
            * (1.0f + static_cast<float>(query_norm[norm_index]));
    }
    __syncthreads();
    if (tid == 0) {
        float dot = 0.0f;
        for (int index = 0; index < hidden_size; ++index) {
            dot += key_row[index] * query_row[index];
        }
        dot /= sqrtf(static_cast<float>(hidden_size));
        const float sign = dot >= 0.0f ? 1.0f : -1.0f;
        const float transformed = sign * sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
        gate = grouped_sigmoid(transformed);
    }
    __syncthreads();
    const float* value_row = value + token * hidden_size;
    float* gated_row = gated + base;
    for (int index = tid; index < hidden_size; index += static_cast<int>(blockDim.x)) {
        gated_row[index] = gate * value_row[index];
    }
}

extern "C" __global__ void grouped_norm_bf16(
    const float* __restrict__ input,
    const hip_bfloat16* __restrict__ weight,
    float* __restrict__ output,
    int tokens,
    int groups,
    int group_size,
    float epsilon) {
    const int group = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (group >= groups || token >= tokens) return;
    __shared__ float partial[256];
    const int base = token * groups * group_size + group * group_size;
    float sum = 0.0f;
    for (int index = lane; index < group_size; index += static_cast<int>(blockDim.x)) {
        const float value = input[base + index];
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) partial[lane] += partial[lane + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(partial[0] / static_cast<float>(group_size)
        + fmaxf(epsilon, 1.0e-12f));
    for (int index = lane; index < group_size; index += static_cast<int>(blockDim.x)) {
        output[base + index] = input[base + index] * inverse
            * (1.0f + static_cast<float>(weight[group * group_size + index]));
    }
}

extern "C" __global__ void grouped_depthwise_conv_silu_add_bf16(
    const float* __restrict__ gated,
    const float* __restrict__ normed,
    const hip_bfloat16* __restrict__ conv_weight,
    float* __restrict__ state,
    float* __restrict__ output,
    int tokens,
    int channels,
    int kernel_size,
    int dilation,
    int token_chunk) {
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    const int history_rows = (kernel_size - 1) * dilation;
    // Token chunk blockIdx.y; token_chunk >= history_rows, so only chunk 0
    // reads `state`, and it alone rewrites it after its own reads.
    const int token_begin = static_cast<int>(blockIdx.y) * token_chunk;
    const int token_end = min(tokens, token_begin + token_chunk);
    for (int token = token_begin; token < token_end; ++token) {
        float total = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int source = token + tap * dilation;
            float value = 0.0f;
            if (source < history_rows) {
                value = state[source * channels + channel];
            } else {
                const int input_index = source - history_rows;
                if (input_index < tokens) {
                    value = normed[input_index * channels + channel];
                }
            }
            total += value * static_cast<float>(conv_weight[channel * kernel_size + tap]);
        }
        output[token * channels + channel] =
            gated[token * channels + channel] + grouped_silu(total);
    }
    if (blockIdx.y != 0) return;
    for (int row = 0; row < history_rows; ++row) {
        const int source = tokens + row;
        if (source < history_rows) {
            state[row * channels + channel] = state[source * channels + channel];
        } else {
            state[row * channels + channel] =
                normed[(source - history_rows) * channels + channel];
        }
    }
}
