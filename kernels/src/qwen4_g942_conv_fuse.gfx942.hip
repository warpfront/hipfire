// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Exact-gfx942 convert-elimination entries of the C8 lane-verify window (module
// qwen4_g942_conv_fuse, twin hook HIPFIRE_QWEN4_G942_CONV_FUSE_IMAGE).
//
// qwen4_g942_hyper_norm_bf16x: `hyper_norm_f32`'s body verbatim (the F32 `normalized` rows are
// stored exactly as before) with ONE extra store per element: the same value as BF16 bits into
// `normalized_bf16`. The value is already BF16-rounded (`bf16_to_f32(bf16_round(..))`), so its top
// 16 bits are the bytes `convert_f32_to_bf16` would stage from the F32 row. The BF16 rows feed the
// wide MFMA down projection directly, replacing its `convert_f32_to_bf16` launch.
//
// Lines ending in `// G12` are the only additions to the tensor_ops.hip blocks; with them removed
// each BEGIN/END block is a substring of tensor_ops.hip (unit test
// `g942_conv_fuse_source_is_tensor_ops_verbatim`).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// BEGIN verbatim tensor_ops.hip
using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
// HC residual stream element: F32, or BF16 bits on the Qwen4 F16 prefill
// route.  Every reader rounds the stream to BF16 first, so both layouts
// yield the value it uses.
static __device__ __forceinline__ float hc_stream_value(float v) { return v; }
static __device__ __forceinline__ float hc_stream_value(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

// Sum of squares of the BF16-rounded stream values at j = tid, tid + 256,
// ... < n, accumulated in that order (the 256-thread HC norms' partial).  Ten
// loads are issued before they are consumed: the plain strided loop waits a
// full memory round trip per element.
template <typename T>
static __device__ __forceinline__ float hc_bf16_square_partial(
    const T* values, int tid, int n) {
    float partial = 0.0f;
    for (int base = tid; base < n; base += 10 * 256) {
        float raw[10];
        // A full batch loads unconditionally: guarded loads compile to one
        // branch + wait per element, serializing ten memory latencies.
        if (base + 9 * 256 < n) {
#pragma unroll
            for (int u = 0; u < 10; ++u) raw[u] = hc_stream_value(values[base + u * 256]);
        } else {
#pragma unroll
            for (int u = 0; u < 10; ++u) {
                const int j = base + u * 256;
                raw[u] = j < n ? hc_stream_value(values[j]) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            if (base + u * 256 < n) {
                const float value = bf16_to_f32(bf16_round(raw[u]));
                partial += value * value;
            }
        }
    }
    return partial;
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
// `normalized16`, when non-null, also receives each normalized value rounded
// to F16 (RNE, as convert_f32_to_f16) for the F16 WMMA projection, and
// `normalized`, when non-null, then holds the (already BF16-rounded) values
// as BF16 bits.
template <typename T>
static __device__ __forceinline__ void hyper_norm_rows(
    const T* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows,
    unsigned short* normalized_bf16,  // G12
    _Float16* normalized16) {
    const int branch = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (branch >= branches || row >= rows) return;
    const int tid = (int)threadIdx.x;
    const int offset = (row * branches + branch) * hidden;
    const int weight_offset = branch * hidden;
    __shared__ float rms_sum[256];
    // One 2560-wide batch (the Qwen4 decode shape): the second pass reuses
    // the first pass's loads, and the weights load alongside them.
    const bool single = hidden == 10 * 256;
    float raw[10], weight[10];
    if (single) {
        float partial = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            raw[u] = hc_stream_value(input[offset + tid + u * 256]);
            weight[u] = bf16_to_f32(norm_weight[weight_offset + tid + u * 256]);
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const float value = bf16_to_f32(bf16_round(raw[u]));
            partial += value * value;
        }
        rms_sum[tid] = partial;
    } else {
        rms_sum[tid] = hc_bf16_square_partial(input + offset, tid, hidden);
    }
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rms_sum[tid] += rms_sum[tid + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
    for (int base = tid; base < hidden; base += 10 * 256) {
        // Batched loads as in hc_bf16_square_partial.
        if (!single) {
            if (base + 9 * 256 < hidden) {
#pragma unroll
                for (int u = 0; u < 10; ++u) {
                    raw[u] = hc_stream_value(input[offset + base + u * 256]);
                    weight[u] = bf16_to_f32(norm_weight[weight_offset + base + u * 256]);
                }
            } else {
#pragma unroll
                for (int u = 0; u < 10; ++u) {
                    const int j = base + u * 256;
                    raw[u] = j < hidden ? hc_stream_value(input[offset + j]) : 0.0f;
                    weight[u] = j < hidden ? bf16_to_f32(norm_weight[weight_offset + j]) : 0.0f;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = base + u * 256;
            if (j >= hidden) break;
            const int column = offset + j;
            const float value = bf16_to_f32(bf16_round(
                bf16_to_f32(bf16_round(raw[u])) * inv * (1.0f + weight[u])));
            if (normalized16) {
                normalized16[column] = (_Float16)value;
                if (normalized)
                    ((unsigned short*)normalized)[column] =
                        (unsigned short)(__float_as_uint(value) >> 16);
            } else {
                normalized[column] = value;
                if (normalized_bf16)  // G12
                    normalized_bf16[column] = (unsigned short)(__float_as_uint(value) >> 16);  // G12
            }
        }
    }
}
// END verbatim tensor_ops.hip

// `input` is the HC streams (BF16 bits when `state_bf16`). Grid [branches, rows], block 256.
extern "C" __global__ void qwen4_g942_hyper_norm_bf16x(
    const void* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows,
    unsigned short* normalized_bf16,
    int state_bf16) {
    if (state_bf16)
        hyper_norm_rows((const unsigned short*)input, norm_weight, normalized, branches, hidden,
                        rows, normalized_bf16, (_Float16*)nullptr);
    else
        hyper_norm_rows((const float*)input, norm_weight, normalized, branches, hidden, rows,
                        normalized_bf16, (_Float16*)nullptr);
}
