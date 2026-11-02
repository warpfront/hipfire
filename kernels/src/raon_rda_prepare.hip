// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Checkpoint-specific RDA gate/normalization prep (the only Raon-named kernel;
// everything downstream is the generic vector-decay gated delta kernel).
// Inputs are post-conv, BF16-rounded FP32 planes. Per (row, head):
//   q  = bf16( bf16( qconv / max(||qconv||, 1e-6) ) * q_scale )
//   k  = bf16( kconv / max(||kconv||, 1e-6) )
//   u[k] = exp(A_log[h]) * (f[k] + dt_bias[h,k]);  log_alpha[k] = lower_bound * sigmoid(u[k])
//   beta = bf16(sigmoid(b));  ratio = exp(logsigmoid(b+delta) - logsigmoid(b))
//   vprime = bf16( vconv * ratio )
// With round_bf16 = 0 (FP32 model dtype) every bf16() above is the identity.
// One workgroup per (head, row); blockDim.x == head_dim (<= 256).
// Credit: gate/ratio structure cross-checked against the contributor FP32 port
// by u/Queasy_Asparagus69 (HF datayoda).

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float rrp_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

static __device__ __forceinline__ float rrp_rn(float x, int round) { return round ? rrp_bf16_rn(x) : x; }

static __device__ __forceinline__ float rrp_logsigmoid(float x) {
    return fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
}

extern "C" __global__ void raon_rda_prepare_f32(
    float* __restrict__ q_out,        // [N, H*K]
    float* __restrict__ k_out,        // [N, H*K]
    float* __restrict__ vprime_out,   // [N, H*K]
    float* __restrict__ log_alpha_out,// [N, H*K]
    float* __restrict__ beta_out,     // [N, H]
    const float* __restrict__ qconv,
    const float* __restrict__ kconv,
    const float* __restrict__ vconv,
    const float* __restrict__ f,
    const float* __restrict__ b,
    const float* __restrict__ delta,
    const float* __restrict__ a_log,  // [H]
    const float* __restrict__ dt_bias,// [H*K]
    int head_dim,
    int n_heads,
    int conv_row_stride,              // qconv/kconv/vconv row stride
    int f_row_stride,
    int bd_row_stride,                // b / delta row stride
    int out_row_stride,               // q/k/vprime/log_alpha row stride
    int beta_row_stride,
    float q_scale,
    float norm_eps,
    float lower_bound,
    int round_bf16                    // 1 = model dtype BF16 boundaries, 0 = FP32 model dtype
) {
    __shared__ float rq[256];
    __shared__ float rk[256];
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int d = threadIdx.x;
    const long long off = (long long)h * head_dim + d;
    const float qv = qconv[(long long)row * conv_row_stride + off];
    const float kv = kconv[(long long)row * conv_row_stride + off];
    rq[d] = qv * qv;
    rk[d] = kv * kv;
    __syncthreads();
    for (int s = head_dim >> 1; s > 0; s >>= 1) {
        if (d < s) { rq[d] += rq[d + s]; rk[d] += rk[d + s]; }
        __syncthreads();
    }
    const float qn = fmaxf(sqrtf(rq[0]), norm_eps);
    const float kn = fmaxf(sqrtf(rk[0]), norm_eps);
    const long long ooff = (long long)row * out_row_stride + off;
    q_out[ooff] = rrp_rn(rrp_rn(qv / qn, round_bf16) * q_scale, round_bf16);
    k_out[ooff] = rrp_rn(kv / kn, round_bf16);

    const float u = expf(a_log[h]) * (f[(long long)row * f_row_stride + off] + dt_bias[off]);
    log_alpha_out[ooff] = lower_bound * (1.0f / (1.0f + expf(-u)));

    const float bv = b[(long long)row * bd_row_stride + h];
    const float dv = delta[(long long)row * bd_row_stride + h];
    const float ratio = expf(rrp_logsigmoid(bv + dv) - rrp_logsigmoid(bv));
    vprime_out[ooff] = rrp_rn(vconv[(long long)row * conv_row_stride + off] * ratio, round_bf16);
    if (d == 0) beta_out[(long long)row * beta_row_stride + h] = rrp_rn(1.0f / (1.0f + expf(-bv)), round_bf16);
}
