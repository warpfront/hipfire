// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Generic gated delta rule with per-head or per-key-channel log decay.
// One launch covers T>=1 tokens; the FP32 state tile stays in registers for the
// whole sequence (no per-token launch, no state tape, no atomics).
//
// Per token t and head h, state S[v,k] (value-major, key contiguous):
//   D[k]   = exp(log_decay[t,h,k])            (PerKeyChannel) or exp(log_decay[t,h]) (PerHead)
//   S[v,k] = S[v,k] * D[k]
//   pred[v]= sum_k S[v,k] * key[k]
//   err[v] = value[v] - pred[v]
//   S[v,k] = S[v,k] + beta * (err[v] * key[k])
//   y[v]   = sum_k S[v,k] * query[k]           (read AFTER write)
// The decay is applied to each key column BEFORE the reduction, so a vector
// decay is never factored out of the dot product.
//
// Geometry: head_dim (K) and value dim (V) are both 128 in this build. One
// wave32 workgroup owns (head, V-tile of GDV_TV rows); each lane owns four key
// columns (lane + 32*i) of every row in the tile. Reductions are xor
// butterflies, so every lane holds the bit-identical sum.
//
// Slot/token-count device indirection (graph-capture safe): when `slot_idx` /
// `n_dev` are non-null they override the baked `slot` / `n_tokens` (the baked
// n_tokens is then the launch upper bound).

#include <hip/hip_runtime.h>

#define GDV_D 128
#define GDV_TV 8
#define GDV_LANES 32
#define GDV_KPL (GDV_D / GDV_LANES)  // key columns per lane = 4

static __device__ __forceinline__ float gdv_bf16_rn(float x) {
    unsigned u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return x;  // NaN passes through
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

static __device__ __forceinline__ float gdv_wave_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o, GDV_LANES);
    return x;
}

extern "C" __launch_bounds__(GDV_LANES, 8)
__global__ void gated_delta_net_vector_serial_f32(
    const float* __restrict__ query,       // [T, Hqk, K] row stride q_row_stride
    const float* __restrict__ key,         // [T, Hqk, K] row stride k_row_stride
    const float* __restrict__ value,       // [T, H,   V] row stride v_row_stride
    const float* __restrict__ log_decay,   // PerKey: [T,H,K]; PerHead: [T,H]
    const float* __restrict__ beta,        // [T, H]       row stride beta_row_stride
    float* __restrict__ state,             // slot base; [H,V,K] via head/row strides
    float* __restrict__ output,            // [T, H, V]    row stride out_row_stride
    const int* __restrict__ slot_idx,      // nullable device slot index
    const int* __restrict__ n_dev,         // nullable device token count
    int n_tokens,
    int n_heads,
    int qk_group,                          // value heads per q/k head (1 = no repeat)
    int decay_per_key,                     // 1 = PerKeyChannel, 0 = PerHead
    int round_out_bf16,
    int slot,
    int q_row_stride,
    int k_row_stride,
    int v_row_stride,
    int decay_row_stride,
    int beta_row_stride,
    int out_row_stride,
    int state_head_stride,
    int state_row_stride,
    unsigned long long state_slot_stride
) {
    const int h = blockIdx.x;
    const int row0 = blockIdx.y * GDV_TV;
    if (h >= n_heads) return;
    const int lane = threadIdx.x;
    const int hq = h / qk_group;

    int n = n_tokens;
    if (n_dev != nullptr) {
        int nd = n_dev[0];
        n = nd < n ? nd : n;
    }
    const int slot_sel = (slot_idx != nullptr) ? slot_idx[0] : slot;
    float* S_base = state + (unsigned long long)slot_sel * state_slot_stride
                  + (long long)h * state_head_stride + (long long)row0 * state_row_stride;

    float S[GDV_TV][GDV_KPL];
#pragma unroll
    for (int r = 0; r < GDV_TV; r++) {
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++)
            S[r][i] = S_base[(long long)r * state_row_stride + lane + GDV_LANES * i];
    }

    // Operands for the current token; the next token's are fetched while the
    // current one computes.
    float kc[GDV_KPL], qc[GDV_KPL], dc[GDV_KPL], vc[GDV_TV];
    float bc = 0.0f;
    if (n > 0) {
        const float* kp = key + hq * GDV_D;
        const float* qp = query + hq * GDV_D;
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++) {
            kc[i] = kp[lane + GDV_LANES * i];
            qc[i] = qp[lane + GDV_LANES * i];
        }
        if (decay_per_key) {
            const float* dp = log_decay + (long long)h * GDV_D;
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) dc[i] = expf(dp[lane + GDV_LANES * i]);
        } else {
            float d = expf(log_decay[h]);
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) dc[i] = d;
        }
        const float* vp = value + h * GDV_D + row0;
#pragma unroll
        for (int r = 0; r < GDV_TV; r++) vc[r] = vp[r];
        bc = beta[h];
    }

    for (int t = 0; t < n; t++) {
        float kn[GDV_KPL], qn[GDV_KPL], dn[GDV_KPL], vn[GDV_TV];
        float bn = 0.0f;
        const bool has_next = (t + 1) < n;
        if (has_next) {
            const int t1 = t + 1;
            const float* kp = key + (long long)t1 * k_row_stride + hq * GDV_D;
            const float* qp = query + (long long)t1 * q_row_stride + hq * GDV_D;
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) {
                kn[i] = kp[lane + GDV_LANES * i];
                qn[i] = qp[lane + GDV_LANES * i];
            }
            if (decay_per_key) {
                const float* dp = log_decay + (long long)t1 * decay_row_stride + (long long)h * GDV_D;
#pragma unroll
                for (int i = 0; i < GDV_KPL; i++) dn[i] = expf(dp[lane + GDV_LANES * i]);
            } else {
                float d = expf(log_decay[(long long)t1 * decay_row_stride + h]);
#pragma unroll
                for (int i = 0; i < GDV_KPL; i++) dn[i] = d;
            }
            const float* vp = value + (long long)t1 * v_row_stride + h * GDV_D + row0;
#pragma unroll
            for (int r = 0; r < GDV_TV; r++) vn[r] = vp[r];
            bn = beta[(long long)t1 * beta_row_stride + h];
        }

        float* out_row = output + (long long)t * out_row_stride + h * GDV_D + row0;
#pragma unroll
        for (int r = 0; r < GDV_TV; r++) {
            float s0 = S[r][0] * dc[0];
            float s1 = S[r][1] * dc[1];
            float s2 = S[r][2] * dc[2];
            float s3 = S[r][3] * dc[3];
            float pred = gdv_wave_sum(s0 * kc[0] + s1 * kc[1] + s2 * kc[2] + s3 * kc[3]);
            float err = vc[r] - pred;
            s0 = s0 + bc * (err * kc[0]);
            s1 = s1 + bc * (err * kc[1]);
            s2 = s2 + bc * (err * kc[2]);
            s3 = s3 + bc * (err * kc[3]);
            S[r][0] = s0; S[r][1] = s1; S[r][2] = s2; S[r][3] = s3;
            float y = gdv_wave_sum(s0 * qc[0] + s1 * qc[1] + s2 * qc[2] + s3 * qc[3]);
            if (lane == 0) out_row[r] = round_out_bf16 ? gdv_bf16_rn(y) : y;
        }

        if (has_next) {
#pragma unroll
            for (int i = 0; i < GDV_KPL; i++) { kc[i] = kn[i]; qc[i] = qn[i]; dc[i] = dn[i]; }
#pragma unroll
            for (int r = 0; r < GDV_TV; r++) vc[r] = vn[r];
            bc = bn;
        }
    }

#pragma unroll
    for (int r = 0; r < GDV_TV; r++) {
#pragma unroll
        for (int i = 0; i < GDV_KPL; i++)
            S_base[(long long)r * state_row_stride + lane + GDV_LANES * i] = S[r][i];
    }
}
