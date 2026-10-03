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

// ── Fused PLE tail (`HIPFIRE_QWEN4_PLE_FUSE`, exact gfx1151, BF16 HC streams) ──
// Idea from Gufo upstream (gufo-org/gufo @1071b361, MIT).
//
// The incumbent chain is hc_state_bf16_to_f32 -> grouped_gate_bf16 ->
// grouped_norm_bf16 -> grouped_depthwise_conv_silu_add_bf16 ->
// hc_state_bf16_add_f32.  The gate row is `gate * value_row` (one scalar per
// (token, branch) times the per-token value row) and the gate scalar's three
// reductions are serial in index order, so the fused route keeps those orders
// bit for bit and drops every F32 round trip:
//   ple_gate_rows_bf16s : one thread per (token, branch) row runs the three
//                         serial reductions over LDS-staged 32-column tiles
//                         (the BF16 stream row is widened on load) -> gate[row].
//   ple_norm_inv        : the 256-lane tree reduction of grouped_norm_bf16 over
//                         `gate * value`, one block per token -> inv[row].
//   ple_conv_add_bf16s  : the convolution recomputes gated/normed from
//                         (gate, inv, value) and adds its output straight into
//                         the BF16 stream: bf16(stream + (gated + silu(conv))).
// Every product that the incumbent stores to F32 and re-reads goes through
// ple_mul_rn here so no contraction can fuse across the old kernel boundary.
#ifndef PLE_GATE_ROWS
#define PLE_GATE_ROWS 64
#endif
#ifndef PLE_GATE_LDS_PAD
#define PLE_GATE_LDS_PAD 0
#endif
#define PLE_GATE_COLS 32
#define PLE_GATE_PAD 33
#define PLE_NORM_MAX_BRANCHES 4

// The product the incumbent stores to F32 before its consumer adds to it.
// `__fmul_rn` is a plain `*` in HIP and still contracts into a following add,
// so pin the rounded product behind an empty asm.
static __device__ __forceinline__ float ple_mul_rn(float a, float b) {
    float r = a * b;
    asm volatile("" : "+v"(r));
    return r;
}

static __device__ __forceinline__ float ple_bf16_to_f32(unsigned int bits) {
    return __uint_as_float(bits << 16);
}

extern "C" __global__ __launch_bounds__(PLE_GATE_ROWS) void ple_gate_rows_bf16s(
    const float* __restrict__ key,
    const unsigned short* __restrict__ streams,
    const unsigned short* __restrict__ key_norm,
    const unsigned short* __restrict__ query_norm,
    float* __restrict__ gate_out,
    int rows,
    int hc_count,
    int hidden_size,
    float epsilon) {
    __shared__ float key_tile[PLE_GATE_ROWS * PLE_GATE_PAD];
    __shared__ float query_tile[PLE_GATE_ROWS * PLE_GATE_PAD];
#if PLE_GATE_LDS_PAD > 0
    __shared__ float occupancy_pad[PLE_GATE_LDS_PAD];
    if (hidden_size < 0) occupancy_pad[threadIdx.x] = 0.0f;
#endif
    const int tid = static_cast<int>(threadIdx.x);
    const int row0 = static_cast<int>(blockIdx.x) * PLE_GATE_ROWS;
    const int row = row0 + tid;
    const int branch = row % hc_count;
    const float eps = fmaxf(epsilon, 1.0e-12f);
    float key_square = 0.0f;
    float query_square = 0.0f;
    float key_inverse = 0.0f;
    float query_inverse = 0.0f;
    float dot = 0.0f;
    for (int pass = 0; pass < 2; ++pass) {
        for (int c0 = 0; c0 < hidden_size; c0 += PLE_GATE_COLS) {
            for (int i = tid; i < PLE_GATE_ROWS * 8; i += PLE_GATE_ROWS) {
                const int r = i >> 3;
                const int q = i & 7;
                float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                uint2 qv = make_uint2(0u, 0u);
                if (row0 + r < rows) {
                    const size_t at = static_cast<size_t>(row0 + r) * hidden_size + c0 + 4 * q;
                    kv = *reinterpret_cast<const float4*>(key + at);
                    qv = *reinterpret_cast<const uint2*>(streams + at);
                }
                float* kd = key_tile + r * PLE_GATE_PAD + 4 * q;
                float* qd = query_tile + r * PLE_GATE_PAD + 4 * q;
                kd[0] = kv.x; kd[1] = kv.y; kd[2] = kv.z; kd[3] = kv.w;
                qd[0] = ple_bf16_to_f32(qv.x & 0xffffu);
                qd[1] = ple_bf16_to_f32(qv.x >> 16);
                qd[2] = ple_bf16_to_f32(qv.y & 0xffffu);
                qd[3] = ple_bf16_to_f32(qv.y >> 16);
            }
            __syncthreads();
            const float* kt = key_tile + tid * PLE_GATE_PAD;
            const float* qt = query_tile + tid * PLE_GATE_PAD;
            if (pass == 0) {
#pragma unroll
                for (int j = 0; j < PLE_GATE_COLS; ++j) {
                    const float key_value = kt[j];
                    const float query_value = qt[j];
                    key_square += key_value * key_value;
                    query_square += query_value * query_value;
                }
            } else {
                const int norm_at = branch * hidden_size + c0;
                const uint4* kn = reinterpret_cast<const uint4*>(key_norm + norm_at);
                const uint4* qn = reinterpret_cast<const uint4*>(query_norm + norm_at);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const uint4 kw = kn[v];
                    const uint4 qw = qn[v];
                    const unsigned int kword[4] = {kw.x, kw.y, kw.z, kw.w};
                    const unsigned int qword[4] = {qw.x, qw.y, qw.z, qw.w};
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int j = v * 8 + w * 2 + h;
                            const float kweight = ple_bf16_to_f32(h ? (kword[w] >> 16) : (kword[w] & 0xffffu));
                            const float qweight = ple_bf16_to_f32(h ? (qword[w] >> 16) : (qword[w] & 0xffffu));
                            const float key_value = kt[j] * key_inverse * (1.0f + kweight);
                            const float query_value = qt[j] * query_inverse * (1.0f + qweight);
                            dot += key_value * query_value;
                        }
                    }
                }
            }
            __syncthreads();
        }
        if (pass == 0) {
            key_inverse = rsqrtf(key_square / static_cast<float>(hidden_size) + eps);
            query_inverse = rsqrtf(query_square / static_cast<float>(hidden_size) + eps);
        }
    }
    if (row < rows) {
        dot /= sqrtf(static_cast<float>(hidden_size));
        const float sign = dot >= 0.0f ? 1.0f : -1.0f;
        const float transformed = sign * sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
        gate_out[row] = grouped_sigmoid(transformed);
    }
}

extern "C" __global__ __launch_bounds__(256) void ple_norm_inv(
    const float* __restrict__ gate,
    const float* __restrict__ value,
    float* __restrict__ inverse_out,
    int tokens,
    int hc_count,
    int hidden_size,
    float epsilon) {
    __shared__ float partial[PLE_NORM_MAX_BRANCHES][256];
    const int token = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    float sum[PLE_NORM_MAX_BRANCHES];
    float scale[PLE_NORM_MAX_BRANCHES];
#pragma unroll
    for (int b = 0; b < PLE_NORM_MAX_BRANCHES; ++b) {
        sum[b] = 0.0f;
        scale[b] = b < hc_count ? gate[token * hc_count + b] : 0.0f;
    }
    for (int index = lane; index < hidden_size; index += 256) {
        const float v = value[token * hidden_size + index];
#pragma unroll
        for (int b = 0; b < PLE_NORM_MAX_BRANCHES; ++b) {
            const float g = ple_mul_rn(scale[b], v);
            sum[b] += g * g;
        }
    }
#pragma unroll
    for (int b = 0; b < PLE_NORM_MAX_BRANCHES; ++b) partial[b][lane] = sum[b];
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) {
#pragma unroll
            for (int b = 0; b < PLE_NORM_MAX_BRANCHES; ++b) partial[b][lane] += partial[b][lane + stride];
        }
        __syncthreads();
    }
    if (lane < hc_count) {
        inverse_out[token * hc_count + lane] =
            rsqrtf(partial[lane][0] / static_cast<float>(hidden_size) + fmaxf(epsilon, 1.0e-12f));
    }
}

extern "C" __global__ void ple_conv_add_bf16s(
    const float* __restrict__ gate,
    const float* __restrict__ inverse,
    const float* __restrict__ value,
    const hip_bfloat16* __restrict__ norm_weight,
    const hip_bfloat16* __restrict__ conv_weight,
    float* __restrict__ state,
    unsigned short* __restrict__ streams,
    int tokens,
    int hc_count,
    int hidden_size,
    int kernel_size,
    int dilation,
    int token_chunk) {
    const int channels = hc_count * hidden_size;
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    const int branch = channel / hidden_size;
    const int h = channel - branch * hidden_size;
    const float norm_scale = 1.0f + static_cast<float>(norm_weight[channel]);
    const int history_rows = (kernel_size - 1) * dilation;
    const int token_begin = static_cast<int>(blockIdx.y) * token_chunk;
    const int token_end = min(tokens, token_begin + token_chunk);
    for (int token = token_begin; token < token_end; ++token) {
        float total = 0.0f;
        for (int tap = 0; tap < kernel_size; ++tap) {
            const int source = token + tap * dilation;
            float v = 0.0f;
            if (source < history_rows) {
                v = state[source * channels + channel];
            } else {
                const int input_index = source - history_rows;
                if (input_index < tokens) {
                    const float g = ple_mul_rn(gate[input_index * hc_count + branch],
                                              value[input_index * hidden_size + h]);
                    v = ple_mul_rn(ple_mul_rn(g, inverse[input_index * hc_count + branch]), norm_scale);
                }
            }
            total += v * static_cast<float>(conv_weight[channel * kernel_size + tap]);
        }
        const float gated = ple_mul_rn(gate[token * hc_count + branch], value[token * hidden_size + h]);
        const float out = gated + grouped_silu(total);
        const int at = token * channels + channel;
        streams[at] = static_cast<hip_bfloat16>(ple_bf16_to_f32(streams[at]) + out).data;
    }
    if (blockIdx.y != 0) return;
    for (int row = 0; row < history_rows; ++row) {
        const int source = tokens + row;
        if (source < history_rows) {
            state[row * channels + channel] = state[source * channels + channel];
        } else {
            const int t = source - history_rows;
            const float g = ple_mul_rn(gate[t * hc_count + branch], value[t * hidden_size + h]);
            state[row * channels + channel] =
                ple_mul_rn(ple_mul_rn(g, inverse[t * hc_count + branch]), norm_scale);
        }
    }
}

// PLE rows straight to the F16 WMMA input (`HIPFIRE_QWEN4_PLE_FUSE`): the
// gather's F32 widening followed by `convert_f32_to_f16`'s `(_Float16)` cast,
// in one launch.  Eight elements per thread; `total` is a multiple of 8.
extern "C" __global__ __launch_bounds__(256) void grouped_gather_convert_bf16_f16(
    const unsigned short* __restrict__ staged,
    _Float16* __restrict__ output,
    int total) {
    const int at = (static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)) * 8;
    if (at >= total) return;
    const uint4 packed = *reinterpret_cast<const uint4*>(staged + at);
    const unsigned int word[4] = {packed.x, packed.y, packed.z, packed.w};
    _Float16 half_out[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        half_out[2 * i] = (_Float16)__uint_as_float((word[i] & 0xffffu) << 16);
        half_out[2 * i + 1] = (_Float16)__uint_as_float(word[i] & 0xffff0000u);
    }
    *reinterpret_cast<uint4*>(output + at) = *reinterpret_cast<const uint4*>(half_out);
}

// `ple_conv_add_bf16s` specialised to the model's 4-tap, dilation-3 window
// (history 9): two adjacent channels per thread, a 10-deep register window of
// the convolution input so each token's gated/normed value is computed once.
// Same per-element expressions and tap order as `ple_conv_add_bf16s`.
extern "C" __global__ __launch_bounds__(256) void ple_conv_add_bf16s_k4d3(
    const float* __restrict__ gate,
    const float* __restrict__ inverse,
    const float* __restrict__ value,
    const unsigned short* __restrict__ norm_weight,
    const unsigned short* __restrict__ conv_weight,
    float* __restrict__ state,
    unsigned short* __restrict__ streams,
    int tokens,
    int hc_count,
    int hidden_size,
    int token_chunk) {
    const int channels = hc_count * hidden_size;
    const int channel = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (channel >= channels) return;
    const int branch = channel / hidden_size;
    const int h = channel - branch * hidden_size;
    const unsigned int nw = *reinterpret_cast<const unsigned int*>(norm_weight + channel);
    const float scale0 = 1.0f + ple_bf16_to_f32(nw & 0xffffu);
    const float scale1 = 1.0f + ple_bf16_to_f32(nw >> 16);
    const uint4 cwv = *reinterpret_cast<const uint4*>(conv_weight + channel * 4);
    const unsigned int cww[4] = {cwv.x, cwv.y, cwv.z, cwv.w};
    float cw0[4], cw1[4];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        cw0[2 * w] = ple_bf16_to_f32(cww[w] & 0xffffu);
        cw0[2 * w + 1] = ple_bf16_to_f32(cww[w] >> 16);
        cw1[2 * w] = ple_bf16_to_f32(cww[2 + w] & 0xffffu);
        cw1[2 * w + 1] = ple_bf16_to_f32(cww[2 + w] >> 16);
    }
    const int token_begin = static_cast<int>(blockIdx.y) * token_chunk;
    const int token_end = min(tokens, token_begin + token_chunk);
    // gated (`gate * value`) and normed pairs of token t.
    auto gated_at = [&](int t, float& g0, float& g1, float& n0, float& n1) {
        const float scalar = gate[t * hc_count + branch];
        const float inv = inverse[t * hc_count + branch];
        const float2 v = *reinterpret_cast<const float2*>(value + t * hidden_size + h);
        g0 = ple_mul_rn(scalar, v.x);
        g1 = ple_mul_rn(scalar, v.y);
        n0 = ple_mul_rn(ple_mul_rn(g0, inv), scale0);
        n1 = ple_mul_rn(ple_mul_rn(g1, inv), scale1);
    };
    // Convolution input at source row `source` (history rows, then normed).
    auto input_at = [&](int source, float& u0, float& u1) {
        if (source < 9) {
            const float2 st = *reinterpret_cast<const float2*>(state + source * channels + channel);
            u0 = st.x;
            u1 = st.y;
        } else if (source - 9 < tokens) {
            float g0, g1;
            gated_at(source - 9, g0, g1, u0, u1);
        } else {
            u0 = 0.0f;
            u1 = 0.0f;
        }
    };
    float win0[10], win1[10];
#pragma unroll
    for (int i = 0; i < 10; ++i) input_at(token_begin + i, win0[i], win1[i]);
    float g0c = 0.0f, g1c = 0.0f, ignore0, ignore1;
    if (token_begin < tokens) gated_at(token_begin, g0c, g1c, ignore0, ignore1);
    for (int token = token_begin; token < token_end; ++token) {
        float total0 = 0.0f, total1 = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) {
            total0 += win0[3 * tap] * cw0[tap];
            total1 += win1[3 * tap] * cw1[tap];
        }
        const float out0 = g0c + grouped_silu(total0);
        const float out1 = g1c + grouped_silu(total1);
        const int at = token * channels + channel;
        const unsigned int word = *reinterpret_cast<const unsigned int*>(streams + at);
        const unsigned short r0 = static_cast<hip_bfloat16>(ple_bf16_to_f32(word & 0xffffu) + out0).data;
        const unsigned short r1 = static_cast<hip_bfloat16>(ple_bf16_to_f32(word >> 16) + out1).data;
        *reinterpret_cast<unsigned int*>(streams + at) = static_cast<unsigned int>(r0) | (static_cast<unsigned int>(r1) << 16);
#pragma unroll
        for (int i = 0; i < 9; ++i) { win0[i] = win0[i + 1]; win1[i] = win1[i + 1]; }
        if (token + 1 < tokens) {
            gated_at(token + 1, g0c, g1c, win0[9], win1[9]);
        } else {
            win0[9] = 0.0f;
            win1[9] = 0.0f;
        }
    }
    if (blockIdx.y != 0) return;
    for (int row = 0; row < 9; ++row) {
        const int source = tokens + row;
        float u0, u1;
        if (source < 9) {
            const float2 st = *reinterpret_cast<const float2*>(state + source * channels + channel);
            u0 = st.x;
            u1 = st.y;
        } else {
            float g0, g1;
            gated_at(source - 9, g0, g1, u0, u1);
        }
        *reinterpret_cast<float2*>(state + row * channels + channel) = make_float2(u0, u1);
    }
}
