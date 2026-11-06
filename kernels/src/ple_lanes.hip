// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Lane-batched PLE tail for the Qwen4 lane (C8) decode step
// (`HIPFIRE_QWEN4_LANE_PLE_BATCH`, exact gfx1201, F32 HC streams, one row per
// lane).
//
// The incumbent runs, once per lane (1 row, 1 token):
//     copy_f32_buffer(query <- streams)
//     grouped_gate_bf16        (key, query, value -> gated)
//     grouped_norm_bf16        (gated -> normed)
//     grouped_depthwise_conv_silu_add_bf16  (gated, normed, conv state -> output,
//                                            state rotated in place)
//     add_f32                  (streams += output)
// = 5 launches x lanes. This kernel is the same chain in ONE launch: grid
// [hc_count, lanes], block 256, one block per (branch, lane), dynamic LDS
// 2 * hidden floats. Lane `l` owns row `l` of the contiguous key / streams /
// value / normed buffers and its own convolution state (`t.state[l]`, by value
// in the kernarg segment, no device table).
//
// Byte-exactness vs the per-lane chain (the arithmetic of every element is the
// incumbent's, expression for expression):
//   * the three serial gate reductions run on thread 0 in index order with the
//     same `+= a * b` expressions (the loads are vectorised to float4 and the
//     products are applied x, y, z, w in order: the same dependency chain, only
//     the LDS latency is hidden);
//   * `gated = gate * value` is stored to LDS and re-read after a barrier, so
//     the product is rounded exactly where the incumbent stores it to global
//     memory and `gated + silu(total)` contracts the same way;
//   * the norm is the incumbent's 256-lane strided partial sums and tree
//     (stride 128, 64, ..., 1) with the same epsilon / rsqrtf expression;
//   * the convolution is the one-token body of
//     `grouped_depthwise_conv_silu_add_bf16` (token 0, then the in-place state
//     rotation by one row) per channel; the product `normed` is pinned behind
//     an empty asm (it is a stored F32 value in the incumbent);
//   * the stream add is `streams + output`, one IEEE add of the stored value.
// `normed` is still written (the kept PLE normed rows read by verify rollback);
// the incumbent's scratch-only `query` / `gated` / `output` rows are not.

#define PLE_LANES_MAX 8

struct PleLaneTable {
    unsigned long long state[PLE_LANES_MAX];
};

static __device__ __forceinline__ float ple_lanes_sigmoid(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

static __device__ __forceinline__ float ple_lanes_silu(float value) {
    return value * ple_lanes_sigmoid(value);
}

// A product the incumbent stores to F32 before its consumer reads it back.
static __device__ __forceinline__ float ple_lanes_pin(float value) {
    float r = value;
    asm volatile("" : "+v"(r));
    return r;
}

extern "C" __global__ __launch_bounds__(256) void ple_lanes_tail_f32(
    const float* __restrict__ key,
    float* __restrict__ streams,
    const float* __restrict__ value,
    const hip_bfloat16* __restrict__ key_norm,
    const hip_bfloat16* __restrict__ query_norm,
    const hip_bfloat16* __restrict__ conv_norm,
    const hip_bfloat16* __restrict__ conv_weight,
    float* __restrict__ normed,
    const PleLaneTable t,
    int hc_count,
    int hidden_size,
    int kernel_size,
    int dilation,
    float epsilon) {
    extern __shared__ __attribute__((aligned(16))) float ple_lds[];
    float* key_row = ple_lds;
    float* query_row = ple_lds + hidden_size;
    __shared__ float key_inverse, query_inverse, gate, norm_inverse;
    __shared__ float partial[256];
    const int branch = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (branch >= hc_count) return;
    const int channels = hc_count * hidden_size;
    const int base = lane * channels + branch * hidden_size;

    for (int index = tid; index < hidden_size; index += 256) {
        key_row[index] = key[base + index];
        query_row[index] = streams[base + index];
    }
    __syncthreads();
    if (tid == 0) {
        float key_square = 0.0f;
        float query_square = 0.0f;
        int index = 0;
        if ((hidden_size & 3) == 0) {
            for (; index < hidden_size; index += 4) {
                const float4 k4 = *reinterpret_cast<const float4*>(key_row + index);
                const float4 q4 = *reinterpret_cast<const float4*>(query_row + index);
                key_square += k4.x * k4.x;
                query_square += q4.x * q4.x;
                key_square += k4.y * k4.y;
                query_square += q4.y * q4.y;
                key_square += k4.z * k4.z;
                query_square += q4.z * q4.z;
                key_square += k4.w * k4.w;
                query_square += q4.w * q4.w;
            }
        }
        for (; index < hidden_size; ++index) {
            const float key_value = key_row[index];
            const float query_value = query_row[index];
            key_square += key_value * key_value;
            query_square += query_value * query_value;
        }
        key_inverse = rsqrtf(key_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
        query_inverse = rsqrtf(query_square / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
    }
    __syncthreads();
    for (int index = tid; index < hidden_size; index += 256) {
        const int norm_index = branch * hidden_size + index;
        key_row[index] = key_row[index] * key_inverse
            * (1.0f + static_cast<float>(key_norm[norm_index]));
        query_row[index] = query_row[index] * query_inverse
            * (1.0f + static_cast<float>(query_norm[norm_index]));
    }
    __syncthreads();
    if (tid == 0) {
        float dot = 0.0f;
        int index = 0;
        if ((hidden_size & 3) == 0) {
            for (; index < hidden_size; index += 4) {
                const float4 k4 = *reinterpret_cast<const float4*>(key_row + index);
                const float4 q4 = *reinterpret_cast<const float4*>(query_row + index);
                dot += k4.x * q4.x;
                dot += k4.y * q4.y;
                dot += k4.z * q4.z;
                dot += k4.w * q4.w;
            }
        }
        for (; index < hidden_size; ++index) {
            dot += key_row[index] * query_row[index];
        }
        dot /= sqrtf(static_cast<float>(hidden_size));
        const float sign = dot >= 0.0f ? 1.0f : -1.0f;
        const float transformed = sign * sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
        gate = ple_lanes_sigmoid(transformed);
    }
    __syncthreads();
    // gated = gate * value, rounded to F32 in LDS (key_row is dead past the dot).
    const float* value_row = value + lane * hidden_size;
    for (int index = tid; index < hidden_size; index += 256) {
        key_row[index] = gate * value_row[index];
    }
    __syncthreads();
    // grouped_norm_bf16 over the gated row: 256 strided partial sums + tree.
    float sum = 0.0f;
    for (int index = tid; index < hidden_size; index += 256) {
        const float gated_value = key_row[index];
        sum += gated_value * gated_value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        norm_inverse = rsqrtf(partial[0] / static_cast<float>(hidden_size)
            + fmaxf(epsilon, 1.0e-12f));
    }
    __syncthreads();
    // One-token depthwise convolution + SiLU + gated add, state rotation, and
    // the stream add, per channel of this branch.
    float* __restrict__ state = reinterpret_cast<float*>(t.state[lane]);
    const int history_rows = (kernel_size - 1) * dilation;
    for (int index = tid; index < hidden_size; index += 256) {
        const int channel = branch * hidden_size + index;
        const float gated_value = key_row[index];
        const float normed_value = ple_lanes_pin(
            gated_value * norm_inverse
            * (1.0f + static_cast<float>(conv_norm[branch * hidden_size + index])));
        normed[lane * channels + channel] = normed_value;
        float total = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int source = tap * dilation;
            float tap_value = 0.0f;
            if (source < history_rows) {
                tap_value = state[source * channels + channel];
            } else if (source - history_rows < 1) {
                tap_value = normed_value;
            }
            total += tap_value * static_cast<float>(conv_weight[channel * kernel_size + tap]);
        }
        const float output = gated_value + ple_lanes_silu(total);
        for (int row = 0; row < history_rows; ++row) {
            const int source = 1 + row;
            if (source < history_rows) {
                state[row * channels + channel] = state[source * channels + channel];
            } else {
                state[row * channels + channel] = normed_value;
            }
        }
        streams[base + index] = streams[base + index] + output;
    }
}
