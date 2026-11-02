// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Qwen4 companions of mq_rotate_x (gemv_mq4g256.hip): the same 256-point
// signed FWHT fused into the HC read and written as F16. Kept out of the
// gemv_mq4g256 / mq_rotate_x modules so their code objects stay unchanged.
// mq_fwht256_lane comes from mq_fwht256.h (prepended to this source).

// HC read branch mix (tensor_ops.hip's hyper_read_projected_f32 for four
// branches, same expression) that also writes each mixed row's mq_rotate_x
// rotation: a block per 256-column group and row, wave 0 rotates the group
// from LDS. Grid [hidden / 256, rows], block 256; rows are [rows, 4, hidden]
// in and [rows, hidden] out.
static __device__ __forceinline__ float mq_bf16(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

extern "C" __launch_bounds__(256)
__global__ void hyper_read_projected_rotate_f32(
    const float* __restrict__ normalized,
    const float* __restrict__ projected_up,
    float* __restrict__ mixed,
    float* __restrict__ rotated,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int hidden
) {
    __shared__ float group_values[256];
    const int tid = threadIdx.x;
    const int column = blockIdx.x * 256 + tid;
    const long long row = blockIdx.y;
    normalized += row * 4 * hidden;
    projected_up += row * 4 * hidden;
    mixed += row * hidden;
    rotated += row * hidden;
    float value = 0.0f;
#pragma unroll
    for (int branch = 0; branch < 4; ++branch) {
        const int offset = branch * hidden + column;
        const float gate = mq_bf16(1.0f / (1.0f + expf(-mq_bf16(projected_up[offset]))));
        value += mq_bf16(gate * mq_bf16(normalized[offset]));
    }
    value = mq_bf16(value / 4.0f);
    mixed[column] = value;
    group_values[tid] = value;
    __syncthreads();
    if (tid >= 32) return;
    const int d0 = tid * 8;
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = group_values[d0 + e] * signs1[d0 + e];
    mq_fwht256_lane(v, tid);
    const float s = 0.0625f;
#pragma unroll
    for (int e = 0; e < 8; ++e) rotated[blockIdx.x * 256 + d0 + e] = v[e] * s * signs2[d0 + e];
}

// mq_rotate_x writing F16 (the same float value, rounded as
// convert_f32_to_f16 does): the MQ WMMA GEMMs' input without the F32
// round trip and separate conversion pass.
// F32 or BF16 input (the chunked GDN output stores its BF16-rounded values
// as BF16 bits; widening them is exact, so both entries rotate equal values).
static __device__ __forceinline__ float mq_rotate_load(const float* p) { return *p; }
static __device__ __forceinline__ float mq_rotate_load(const unsigned short* p) {
    return __uint_as_float((unsigned int)*p << 16);
}
template <typename XT>
__device__ __forceinline__ void mq_rotate_x_f16_body(
    const XT* __restrict__ x_in,
    _Float16* __restrict__ x_out,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int K
) {
    const int tid = threadIdx.x;
    const int groups_total = K / 256;
    const int linear_group = blockIdx.x;
    const int batch = linear_group / groups_total;
    const int group = linear_group - batch * groups_total;

    const long long batch_off = (long long)batch * K;
    const XT* x_in_b  = x_in  + batch_off;
          _Float16* x_out_b = x_out + batch_off;

    int d0 = tid * 8;
    int base = group * 256 + d0;

    float v0 = mq_rotate_load(x_in_b + base)     * signs1[d0];
    float v1 = mq_rotate_load(x_in_b + base + 1) * signs1[d0 + 1];
    float v2 = mq_rotate_load(x_in_b + base + 2) * signs1[d0 + 2];
    float v3 = mq_rotate_load(x_in_b + base + 3) * signs1[d0 + 3];
    float v4 = mq_rotate_load(x_in_b + base + 4) * signs1[d0 + 4];
    float v5 = mq_rotate_load(x_in_b + base + 5) * signs1[d0 + 5];
    float v6 = mq_rotate_load(x_in_b + base + 6) * signs1[d0 + 6];
    float v7 = mq_rotate_load(x_in_b + base + 7) * signs1[d0 + 7];

    // Local butterfly: strides 1, 2, 4
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    // The F32 product is pinned before the conversion: fast-math may otherwise
    // fold the scale into an F16 multiply and round differently from
    // mq_rotate_x followed by convert_f32_to_f16.
    // One 16-byte store of the eight halves.
    _Float16 h_[8];
    #define MQ_ROT_F16(o, v, d) do { \
        float r_ = (v) * s * signs2[d]; \
        asm volatile("" : "+v"(r_)); \
        h_[o] = (_Float16)r_; \
    } while (0)
    MQ_ROT_F16(0, v0, d0);
    MQ_ROT_F16(1, v1, d0 + 1);
    MQ_ROT_F16(2, v2, d0 + 2);
    MQ_ROT_F16(3, v3, d0 + 3);
    MQ_ROT_F16(4, v4, d0 + 4);
    MQ_ROT_F16(5, v5, d0 + 5);
    MQ_ROT_F16(6, v6, d0 + 6);
    MQ_ROT_F16(7, v7, d0 + 7);
    #undef MQ_ROT_F16
    *(uint4*)(x_out_b + base) = __builtin_bit_cast(uint4, h_);
}
extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x_f16(
    const float* __restrict__ x_in, _Float16* __restrict__ x_out,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int K) {
    mq_rotate_x_f16_body(x_in, x_out, signs1, signs2, K);
}
extern "C" __launch_bounds__(32, 16)
__global__ void mq_rotate_x_bf16_f16(
    const unsigned short* __restrict__ x_in, _Float16* __restrict__ x_out,
    const float* __restrict__ signs1, const float* __restrict__ signs2, int K) {
    mq_rotate_x_f16_body(x_in, x_out, signs1, signs2, K);
}
