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
// qwen4_g942_hyper_write_norm_bf16x: `hyper_write_norm_f32` (the HC write + next HC read norm + next-gate
// quarters + folded Clear launch) with the same extra BF16-bits store of the read's `normalized`
// rows (entry renamed; `normalized_bf16` may be null).
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

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
// bf16 -> f32 is a lossless 16-bit shift. A dwordx4 holds 8 bf16: within each
// u32 the LOW half is the even element, the HIGH half the odd one (LE).
#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)
// END verbatim qwen4_gemv_bf16_xf32.hip

// BEGIN verbatim qwen4_gemv_bf16_xf32.hip
// Decode HC write + next HC read norm (tensor_ops.hip's hyper_write_f32 and
// hyper_norm_f32 on one F32 row, bitwise): one block per branch; each thread
// writes its ten columns of the new streams, keeps them, and normalizes them
// with hyper_norm_rows' partials, 256-wide LDS tree and expression.
//
// `quarters_in` (optional) replaces `gates`: gate[b] is the k4 fold
// (q[b][0] + q[b][1]) + (q[b][2] + q[b][3]) of four quarter dots.
// `quarters_out` (optional) produces those quarters for the NEXT hyper write
// of these same streams: its norm (`next_norm`) has hyper_norm's `inv` (same
// rows), so block b normalizes its branch into LDS and waves 0..3 run
// gemv_bf16_xf32_k4's quarter b of gate rows 0..3 (gemv_bf16_xf32_row over
// that branch) — the next write then needs neither its norm nor its gate
// GEMV. `clear` (optional) zeroes `clear_elements` floats, the
// zero_f32 of a Clear step this launch absorbs. Grid [branches = 4],
// block 256, hidden == 2560.
static __device__ __forceinline__ float hf_bf16(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

__launch_bounds__(256)
extern "C" __global__ void qwen4_g942_hyper_write_norm_bf16x(
    const float* __restrict__ input,
    const float* __restrict__ mixed,
    const float* __restrict__ gates,
    const float* __restrict__ quarters_in,
    float* output,
    const unsigned short* __restrict__ norm_weight,
    float* __restrict__ normalized,
    unsigned short* __restrict__ normalized_bf16,  // G12
    const unsigned short* __restrict__ next_norm,
    const unsigned short* __restrict__ next_inject,
    float* __restrict__ quarters_out,
    float* __restrict__ clear,
    int clear_elements,
    int branches,
    int hidden
) {
    // Grid [branches, rows]: block (branch, row) owns one branch of one token
    // row; the weights are per branch.
    const int branch = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int woff = branch * hidden;
    const int offset = row * branches * hidden + woff;
    mixed += (size_t)row * hidden;
    gates += (size_t)row * branches;
    if (quarters_in) quarters_in += (size_t)row * 16;
    if (quarters_out) quarters_out += (size_t)row * 16;
    // Optional: a scratch prefix the following steps expect zeroed.
    if (clear) {
        const int blocks = (int)gridDim.y * branches;
        for (int i = (row * branches + branch) * 256 + tid; i < clear_elements; i += blocks * 256)
            clear[i] = 0.0f;
    }
    __shared__ float rms_sum[256];
    __shared__ __align__(16) float xn[2560];
    const int wave = tid >> 5;
    const int lane = tid & 31;
    // Every load issues up front, including the next write's norm weights and
    // (waves 0..3) its gate row quarter: gemv_bf16_xf32_row's eight bf16 per
    // lane at lane * 8 + s * 256, s = 0..9 for K = 2560.
    float raw[10], mix[10], weight[10], next_weight[10];
    uint4 wv[10];
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        raw[u] = input[offset + tid + u * 256];
        mix[u] = mixed[tid + u * 256];
        weight[u] = HF_BF16_LO((unsigned int)norm_weight[woff + tid + u * 256]);
    }
    if (quarters_out) {
#pragma unroll
        for (int u = 0; u < 10; ++u)
            next_weight[u] = HF_BF16_LO((unsigned int)next_norm[woff + tid + u * 256]);
        if (wave < 4) {
            const unsigned short* w_row = next_inject + (size_t)wave * 4 * hidden + woff;
#pragma unroll
            for (int s = 0; s < 10; ++s) wv[s] = *(const uint4*)(w_row + lane * 8 + s * 256);
        }
    }
    const float* q = quarters_in + branch * 4;
    const float gate_logit = hf_bf16(
        quarters_in ? (q[0] + q[1]) + (q[2] + q[3]) : gates[branch]);
    const float gate_scaled = hf_bf16(gate_logit / (float)branches);
    const float gate = hf_bf16(2.0f * hf_bf16(1.0f / (1.0f + expf(-gate_scaled))));
    float partial = 0.0f;
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        const float injection = hf_bf16(gate * hf_bf16(mix[u]));
        raw[u] = hf_bf16(hf_bf16(raw[u]) + injection);
        output[offset + tid + u * 256] = raw[u];
        const float normed_in = hf_bf16(raw[u]);
        partial += normed_in * normed_in;
    }
    rms_sum[tid] = partial;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) rms_sum[tid] += rms_sum[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        normalized[offset + tid + u * 256] = hf_bf16(hf_bf16(raw[u]) * inv * (1.0f + weight[u]));
        if (normalized_bf16)  // G12
            normalized_bf16[offset + tid + u * 256] = (unsigned short)(__float_as_uint(hf_bf16(hf_bf16(raw[u]) * inv * (1.0f + weight[u]))) >> 16);  // G12
    }
    if (!quarters_out) return;
#pragma unroll
    for (int u = 0; u < 10; ++u) {
        xn[tid + u * 256] = hf_bf16(hf_bf16(raw[u]) * inv * (1.0f + next_weight[u]));
    }
    __syncthreads();
    if (wave >= 4) return;
    // gemv_bf16_xf32_row's accumulation for K = 2560, on the preloaded row.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
    for (int s = 0; s < 10; ++s) {
        const float4 xa = *(const float4*)(xn + lane * 8 + s * 256);
        const float4 xb = *(const float4*)(xn + lane * 8 + s * 256 + 4);
        a0 += HF_BF16_LO(wv[s].x) * xa.x;
        a1 += HF_BF16_HI(wv[s].x) * xa.y;
        a2 += HF_BF16_LO(wv[s].y) * xa.z;
        a3 += HF_BF16_HI(wv[s].y) * xa.w;
        a0 += HF_BF16_LO(wv[s].z) * xb.x;
        a1 += HF_BF16_HI(wv[s].z) * xb.y;
        a2 += HF_BF16_LO(wv[s].w) * xb.z;
        a3 += HF_BF16_HI(wv[s].w) * xb.w;
    }
    float sum = (a0 + a1) + (a2 + a3);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down(sum, off);
    if (lane == 0) quarters_out[wave * 4 + branch] = sum;
}
// END verbatim qwen4_gemv_bf16_xf32.hip
