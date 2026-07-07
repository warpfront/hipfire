// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fast variant of gated_delta_net_q8 for the default MQ4/HFQ4 path.
// Supports EF residual (sigma-delta) but always does single-end requant
// OUTSIDE the per-token loop. No per-token requant mode.
//
// Why: the full kernel moved the requant inside the loop to support
// per-token requant (PARO path). This increased register pressure because
// the compiler must carry the requant live-set (q0..q3, scale, rng, EF
// temporaries) across the entire loop body. On gfx906 wave64 (256 VGPRs/file)
// this causes spill-to-scratch: 108→172 bytes per workitem (+60%), costing
// +38% per dispatch. This variant avoids that by keeping the loop lean.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HD 128
#define TILE_ROWS 4

__device__ __forceinline__ void gdn_q8_fast_step_row(
    int tid,
    int row,
    const float* __restrict__ v_t,
    float* __restrict__ out_base,
    float alpha,
    float beta_v,
    float k0,
    float k1,
    float k2,
    float k3,
    float q0,
    float q1,
    float q2,
    float q3,
    float& s0,
    float& s1,
    float& s2,
    float& s3
) {
    float kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
    for (int o = 16; o > 0; o >>= 1)
        kv += __shfl_down(kv, o);
    kv = __shfl(kv, 0);

    float delta = (v_t[row] - alpha * kv) * beta_v;
    s0 = alpha * s0 + k0 * delta;
    s1 = alpha * s1 + k1 * delta;
    s2 = alpha * s2 + k2 * delta;
    s3 = alpha * s3 + k3 * delta;

    float out_v = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    for (int o = 16; o > 0; o >>= 1)
        out_v += __shfl_down(out_v, o);
    if (tid == 0)
        out_base[row] = out_v;
}

__device__ __forceinline__ void gdn_q8_fast_requant_row(
    int tid,
    int h,
    int rr,
    unsigned int frame,
    bool use_ef,
    signed char* __restrict__ dst,
    float* __restrict__ s_scales,
    __half* __restrict__ s_ef_residual,
    float s0,
    float s1,
    float s2,
    float s3
) {
    __half* efr = use_ef ? (s_ef_residual + h * HD * HD + rr * HD) : nullptr;
    if (use_ef) {
        s0 += __half2float(efr[tid]);
        s1 += __half2float(efr[tid + 32]);
        s2 += __half2float(efr[tid + 64]);
        s3 += __half2float(efr[tid + 96]);
    }

    float my_max = fmaxf(fmaxf(fabsf(s0), fabsf(s1)), fmaxf(fabsf(s2), fabsf(s3)));
    for (int o = 16; o > 0; o >>= 1)
        my_max = fmaxf(my_max, __shfl_xor(my_max, o));

    float inv_s = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;
    if (use_ef) {
        float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
        float qf0 = fminf(fmaxf(rintf(s0 * inv_s), -128.0f), 127.0f);
        float qf1 = fminf(fmaxf(rintf(s1 * inv_s), -128.0f), 127.0f);
        float qf2 = fminf(fmaxf(rintf(s2 * inv_s), -128.0f), 127.0f);
        float qf3 = fminf(fmaxf(rintf(s3 * inv_s), -128.0f), 127.0f);
        efr[tid]      = __float2half(s0 - qf0 * scale);
        efr[tid + 32] = __float2half(s1 - qf1 * scale);
        efr[tid + 64] = __float2half(s2 - qf2 * scale);
        efr[tid + 96] = __float2half(s3 - qf3 * scale);
        dst[tid]      = (signed char)qf0;
        dst[tid + 32] = (signed char)qf1;
        dst[tid + 64] = (signed char)qf2;
        dst[tid + 96] = (signed char)qf3;
        if (tid == 0) s_scales[h * HD + rr] = scale;
    } else {
        unsigned int rng = (unsigned int)(h * 1664525u + rr * 1013904223u
                         + tid * 214013u + frame * 2531011u);
        #define STOCH_ROUND(val) ({ \
            rng = rng * 1664525u + 1013904223u; \
            float noise = (float)(rng >> 8) * (1.0f / 16777216.0f); \
            (signed char)fminf(fmaxf(floorf((val) * inv_s + noise), -128.0f), 127.0f); \
        })
        dst[tid]      = STOCH_ROUND(s0);
        dst[tid + 32] = STOCH_ROUND(s1);
        dst[tid + 64] = STOCH_ROUND(s2);
        dst[tid + 96] = STOCH_ROUND(s3);
        #undef STOCH_ROUND
        if (tid == 0) {
            float scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
            s_scales[h * HD + rr] = scale;
        }
    }
}

extern "C" __launch_bounds__(32, 8)
__global__ void gated_delta_net_q8_fast(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    signed char* __restrict__ s_q8,
    float* __restrict__ s_scales,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim,
    unsigned int frame,
    __half* __restrict__ s_ef_residual
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    signed char* sq_base = s_q8 + h * HD * HD + row_start * HD;
    const float scale0 = s_scales[h * HD + row_start + 0];
    const float scale1 = s_scales[h * HD + row_start + 1];
    const float scale2 = s_scales[h * HD + row_start + 2];
    const float scale3 = s_scales[h * HD + row_start + 3];

    float s00 = scale0 * (float)sq_base[tid];
    float s01 = scale0 * (float)sq_base[tid + 32];
    float s02 = scale0 * (float)sq_base[tid + 64];
    float s03 = scale0 * (float)sq_base[tid + 96];
    float s10 = scale1 * (float)sq_base[HD + tid];
    float s11 = scale1 * (float)sq_base[HD + tid + 32];
    float s12 = scale1 * (float)sq_base[HD + tid + 64];
    float s13 = scale1 * (float)sq_base[HD + tid + 96];
    float s20 = scale2 * (float)sq_base[2 * HD + tid];
    float s21 = scale2 * (float)sq_base[2 * HD + tid + 32];
    float s22 = scale2 * (float)sq_base[2 * HD + tid + 64];
    float s23 = scale2 * (float)sq_base[2 * HD + tid + 96];
    float s30 = scale3 * (float)sq_base[3 * HD + tid];
    float s31 = scale3 * (float)sq_base[3 * HD + tid + 32];
    float s32 = scale3 * (float)sq_base[3 * HD + tid + 64];
    float s33 = scale3 * (float)sq_base[3 * HD + tid + 96];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        float alpha = expf(gate[t * n_heads + h]);
        float beta_v = beta[t * n_heads + h];

        float k0 = k_t[tid], k1 = k_t[tid+32], k2 = k_t[tid+64], k3 = k_t[tid+96];
        float q0 = q_t[tid], q1 = q_t[tid+32], q2 = q_t[tid+64], q3 = q_t[tid+96];

        float* out_base = output + t * stride + h * HD;
        gdn_q8_fast_step_row(tid, row_start + 0, v_t, out_base, alpha, beta_v,
                             k0, k1, k2, k3, q0, q1, q2, q3, s00, s01, s02, s03);
        gdn_q8_fast_step_row(tid, row_start + 1, v_t, out_base, alpha, beta_v,
                             k0, k1, k2, k3, q0, q1, q2, q3, s10, s11, s12, s13);
        gdn_q8_fast_step_row(tid, row_start + 2, v_t, out_base, alpha, beta_v,
                             k0, k1, k2, k3, q0, q1, q2, q3, s20, s21, s22, s23);
        gdn_q8_fast_step_row(tid, row_start + 3, v_t, out_base, alpha, beta_v,
                             k0, k1, k2, k3, q0, q1, q2, q3, s30, s31, s32, s33);
    }

    // Requant FP32 → Q8 — OUTSIDE the token loop.
    // Only the final state is quantized (intermediate tokens stay FP32 in LDS).
    const bool use_ef = (s_ef_residual != nullptr);
    gdn_q8_fast_requant_row(tid, h, row_start + 0, frame, use_ef,
                            sq_base, s_scales, s_ef_residual, s00, s01, s02, s03);
    gdn_q8_fast_requant_row(tid, h, row_start + 1, frame, use_ef,
                            sq_base + HD, s_scales, s_ef_residual, s10, s11, s12, s13);
    gdn_q8_fast_requant_row(tid, h, row_start + 2, frame, use_ef,
                            sq_base + 2 * HD, s_scales, s_ef_residual, s20, s21, s22, s23);
    gdn_q8_fast_requant_row(tid, h, row_start + 3, frame, use_ef,
                            sq_base + 3 * HD, s_scales, s_ef_residual, s30, s31, s32, s33);
}
