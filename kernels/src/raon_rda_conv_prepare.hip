// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fused producer of the Raon RDA recurrence inputs (gfx1201, head_dim 128, conv kernel 4):
//   3x causal_conv_depthwise_silu_rows_f32 (q, k, v; K=4, no bias, SiLU, BF16 round, in-place raw history)
//   + raon_rda_prepare_f32 (q/k L2 norm, log_alpha, beta, write-ratio v scaling)
// in ONE launch with no materialized conv planes. Every arithmetic expression, rounding point and
// reduction order is the incumbent's, so the outputs are bit-identical to the two launches:
//   conv   y = bf16( silu( ((((0 + w0*a0) + w1*a1) + w2*a2) + w3*x) ) ),  explicit __fmul_rn/__fadd_rn
//   prepare q = bf16( bf16( qconv / max(||qconv||, eps) ) * q_scale ),  k = bf16( kconv / max(||kconv||, eps) )
//           u[k] = exp(A_log[h]) * (f[k] + dt_bias[h,k]);  log_alpha[k] = lower_bound * sigmoid(u[k])
//           beta = bf16(sigmoid(b));  ratio = exp(logsigmoid(b+delta) - logsigmoid(b));  vprime = bf16(vconv * ratio)
// The head L2 sum of squares is the incumbent's fixed LDS tree (s = 64, 32, ..., 1 over the 128 channels)
// evaluated by a wave-shuffle tree: a wave owns one (head, row range), lane l holds channels 4l..4l+3, so
// tree stage s = 64/32/16/8/4 are lane offsets 16/8/4/2/1 and the last two stages are in-lane.
// A wave walks `rows_per_wave` consecutive rows with the conv window in registers (the raw rows are read
// once each; the incumbent wrote and re-read three conv planes). `hist` is read (rows < 3 of the padded
// stream) and updated in place by the wave that owns rows [0, rows_per_wave), which is the only wave that
// reads it, so in-place is safe (requires rows_per_wave >= 3).
//
// Launch: block 256 (8 waves, one (head, row tile) each), grid ceil(n_heads * ceil(n_rows / rows_per_wave) / 8).

#include <hip/hip_runtime.h>

static __device__ __forceinline__ float rcp_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

static __device__ __forceinline__ float rcp_rn(float x, int round) { return round ? rcp_bf16_rn(x) : x; }

static __device__ __forceinline__ float rcp_logsigmoid(float x) {
    return fminf(x, 0.0f) - log1pf(expf(-fabsf(x)));
}

// depthwise conv tap chain + SiLU + optional BF16 round (the incumbent's expression order)
static __device__ __forceinline__ float rcp_conv(float w0, float w1, float w2, float w3,
                                                  float a0, float a1, float a2, float x, int round) {
    float acc = 0.0f;
    acc = __fadd_rn(acc, __fmul_rn(w0, a0));
    acc = __fadd_rn(acc, __fmul_rn(w1, a1));
    acc = __fadd_rn(acc, __fmul_rn(w2, a2));
    acc = __fadd_rn(acc, __fmul_rn(w3, x));
    const float y = acc / (1.0f + expf(-acc));
    return rcp_rn(y, round);
}

// Sum of the 128 per-channel values (lane l holds channels 4l..4l+3 as s0..s3) in the incumbent's tree
// order, returned in every lane.
static __device__ __forceinline__ float rcp_tree128(float s0, float s1, float s2, float s3) {
#pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        s0 = __fadd_rn(s0, __shfl_down(s0, off, 32));
        s1 = __fadd_rn(s1, __shfl_down(s1, off, 32));
        s2 = __fadd_rn(s2, __shfl_down(s2, off, 32));
        s3 = __fadd_rn(s3, __shfl_down(s3, off, 32));
    }
    const float r = __fadd_rn(__fadd_rn(s0, s2), __fadd_rn(s1, s3));
    return __shfl(r, 0, 32);
}

static __device__ __forceinline__ float4 rcp_ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
static __device__ __forceinline__ void rcp_st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

extern "C" __global__ __launch_bounds__(256) void raon_rda_conv_prepare_w128_f32(
    float* __restrict__ q_out,         // [N, H*128]
    float* __restrict__ k_out,
    float* __restrict__ vprime_out,
    float* __restrict__ log_alpha_out,
    float* __restrict__ beta_out,      // [N, H]
    const float* __restrict__ q_raw,   // pre-conv projections [N, H*128]
    const float* __restrict__ k_raw,
    const float* __restrict__ v_raw,
    const float* __restrict__ w_q,     // conv weights [H*128, 4]
    const float* __restrict__ w_k,
    const float* __restrict__ w_v,
    float* hist_q,                     // raw conv history [3, H*128] oldest -> newest, read + updated in place
    float* hist_k,
    float* hist_v,
    const float* __restrict__ f,
    const float* __restrict__ b,
    const float* __restrict__ delta,
    const float* __restrict__ a_log,   // [H]
    const float* __restrict__ dt_bias, // [H*128]
    int head_dim,                      // 128
    int n_heads,
    int n_rows,
    int rows_per_wave,                 // >= 3
    int raw_row_stride,
    int f_row_stride,
    int bd_row_stride,
    int out_row_stride,
    int beta_row_stride,
    int hist_tap_stride,
    float q_scale,
    float norm_eps,
    float lower_bound,
    int round_bf16
) {
    const int lane = threadIdx.x & 31;
    const int wid = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int h = wid % n_heads;
    const int r_begin = (wid / n_heads) * rows_per_wave;
    if (r_begin >= n_rows) return;
    int r_end = r_begin + rows_per_wave;
    if (r_end > n_rows) r_end = n_rows;

    const long long c0 = (long long)h * head_dim + lane * 4;   // first of this lane's 4 channels

    // conv taps per channel (weights [channel, tap], tap stride 1, channel stride 4)
    float4 wq[4], wk[4], wv[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        wq[i] = rcp_ld4(w_q + (c0 + i) * 4);
        wk[i] = rcp_ld4(w_k + (c0 + i) * 4);
        wv[i] = rcp_ld4(w_v + (c0 + i) * 4);
    }

    // conv windows: padded rows r_begin .. r_begin+2 (history for padded index < 3, else raw row p - 3)
    float4 aq[3], ak[3], av[3];
#pragma unroll
    for (int j = 0; j < 3; j++) {
        const int p = r_begin + j;
        if (p < 3) {
            aq[j] = rcp_ld4(hist_q + (long long)p * hist_tap_stride + c0);
            ak[j] = rcp_ld4(hist_k + (long long)p * hist_tap_stride + c0);
            av[j] = rcp_ld4(hist_v + (long long)p * hist_tap_stride + c0);
        } else {
            aq[j] = rcp_ld4(q_raw + (long long)(p - 3) * raw_row_stride + c0);
            ak[j] = rcp_ld4(k_raw + (long long)(p - 3) * raw_row_stride + c0);
            av[j] = rcp_ld4(v_raw + (long long)(p - 3) * raw_row_stride + c0);
        }
    }

    // History update (after this wave's own reads above, same lanes/channels): padded rows n_rows .. n_rows+2.
    if (r_begin == 0) {
        for (int j = 0; j < 3; j++) {
            const int p = n_rows + j;
            float4 vq, vk, vv;
            if (p < 3) {
                vq = rcp_ld4(hist_q + (long long)p * hist_tap_stride + c0);
                vk = rcp_ld4(hist_k + (long long)p * hist_tap_stride + c0);
                vv = rcp_ld4(hist_v + (long long)p * hist_tap_stride + c0);
            } else {
                vq = rcp_ld4(q_raw + (long long)(p - 3) * raw_row_stride + c0);
                vk = rcp_ld4(k_raw + (long long)(p - 3) * raw_row_stride + c0);
                vv = rcp_ld4(v_raw + (long long)(p - 3) * raw_row_stride + c0);
            }
            rcp_st4(hist_q + (long long)j * hist_tap_stride + c0, vq);
            rcp_st4(hist_k + (long long)j * hist_tap_stride + c0, vk);
            rcp_st4(hist_v + (long long)j * hist_tap_stride + c0, vv);
        }
    }

    const float ea = expf(a_log[h]);
    const float4 dt = rcp_ld4(dt_bias + c0);

    float4 xq = rcp_ld4(q_raw + (long long)r_begin * raw_row_stride + c0);
    float4 xk = rcp_ld4(k_raw + (long long)r_begin * raw_row_stride + c0);
    float4 xv = rcp_ld4(v_raw + (long long)r_begin * raw_row_stride + c0);
    float4 xf = rcp_ld4(f + (long long)r_begin * f_row_stride + c0);

    for (int r = r_begin; r < r_end; r++) {
        // one-row lookahead of the four streamed planes
        float4 nq = xq, nk = xk, nv = xv, nf = xf;
        if (r + 1 < r_end) {
            nq = rcp_ld4(q_raw + (long long)(r + 1) * raw_row_stride + c0);
            nk = rcp_ld4(k_raw + (long long)(r + 1) * raw_row_stride + c0);
            nv = rcp_ld4(v_raw + (long long)(r + 1) * raw_row_stride + c0);
            nf = rcp_ld4(f + (long long)(r + 1) * f_row_stride + c0);
        }
        const float bv = b[(long long)r * bd_row_stride + h];
        const float dv = delta[(long long)r * bd_row_stride + h];

        const float xqa[4] = {xq.x, xq.y, xq.z, xq.w};
        const float xka[4] = {xk.x, xk.y, xk.z, xk.w};
        const float xva[4] = {xv.x, xv.y, xv.z, xv.w};
        const float xfa[4] = {xf.x, xf.y, xf.z, xf.w};
        float cq[4], ck[4], cv[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 q4 = wq[i], k4 = wk[i], v4 = wv[i];
            const float aq0 = ((const float*)&aq[0])[i], aq1 = ((const float*)&aq[1])[i], aq2 = ((const float*)&aq[2])[i];
            const float ak0 = ((const float*)&ak[0])[i], ak1 = ((const float*)&ak[1])[i], ak2 = ((const float*)&ak[2])[i];
            const float av0 = ((const float*)&av[0])[i], av1 = ((const float*)&av[1])[i], av2 = ((const float*)&av[2])[i];
            cq[i] = rcp_conv(q4.x, q4.y, q4.z, q4.w, aq0, aq1, aq2, xqa[i], round_bf16);
            ck[i] = rcp_conv(k4.x, k4.y, k4.z, k4.w, ak0, ak1, ak2, xka[i], round_bf16);
            cv[i] = rcp_conv(v4.x, v4.y, v4.z, v4.w, av0, av1, av2, xva[i], round_bf16);
        }
        // slide the windows
        aq[0] = aq[1]; aq[1] = aq[2]; aq[2] = xq;
        ak[0] = ak[1]; ak[1] = ak[2]; ak[2] = xk;
        av[0] = av[1]; av[1] = av[2]; av[2] = xv;

        const float rq = rcp_tree128(__fmul_rn(cq[0], cq[0]), __fmul_rn(cq[1], cq[1]),
                                     __fmul_rn(cq[2], cq[2]), __fmul_rn(cq[3], cq[3]));
        const float rk = rcp_tree128(__fmul_rn(ck[0], ck[0]), __fmul_rn(ck[1], ck[1]),
                                     __fmul_rn(ck[2], ck[2]), __fmul_rn(ck[3], ck[3]));
        const float qn = fmaxf(sqrtf(rq), norm_eps);
        const float kn = fmaxf(sqrtf(rk), norm_eps);
        const float ratio = expf(rcp_logsigmoid(bv + dv) - rcp_logsigmoid(bv));

        const float dta[4] = {dt.x, dt.y, dt.z, dt.w};
        float oq[4], ok[4], ov[4], ol[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            oq[i] = rcp_rn(rcp_rn(cq[i] / qn, round_bf16) * q_scale, round_bf16);
            ok[i] = rcp_rn(ck[i] / kn, round_bf16);
            const float u = ea * (xfa[i] + dta[i]);
            ol[i] = lower_bound * (1.0f / (1.0f + expf(-u)));
            ov[i] = rcp_rn(cv[i] * ratio, round_bf16);
        }
        const long long ooff = (long long)r * out_row_stride + c0;
        rcp_st4(q_out + ooff, make_float4(oq[0], oq[1], oq[2], oq[3]));
        rcp_st4(k_out + ooff, make_float4(ok[0], ok[1], ok[2], ok[3]));
        rcp_st4(vprime_out + ooff, make_float4(ov[0], ov[1], ov[2], ov[3]));
        rcp_st4(log_alpha_out + ooff, make_float4(ol[0], ol[1], ol[2], ol[3]));
        if (lane == 0) beta_out[(long long)r * beta_row_stride + h] = rcp_rn(1.0f / (1.0f + expf(-bv)), round_bf16);

        xq = nq; xk = nk; xv = nv; xf = nf;
    }
}
