// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Batched prefill GDN preamble fusion for gfx1201.
//
// Fuses three sequential launches per LA layer-chunk into ONE:
//   fused_sigmoid_alpha_gate_f32_batched          (sigmoid(beta) + alpha gate)
//   conv1d_silu_split_f32_n                       (4-tap causal conv + SiLU + split)
//   fused_qk_l2_norm_scale_interleave_f32_batched (L2 norm + scale + interleave)
//
// Grid: [QK ░ V ░ PREP, 1, 1], block [256, 1, 1], where with R = 32 rows per
// group and n_groups = ceil(n_tokens / R):
//   QK   = n_key_heads * n_groups blocks. Block (g, h) owns key head h's Q/K
//          channels over rows [g*R, g*R+R'). Threads 0-127 take Q, 128-255 K.
//   V    = v_blocks * n_groups blocks (v_blocks = ceil(v_dim / 256)).
//   PREP = n_groups blocks doing the elementwise sigmoid + alpha gate rows.
//
// Token-axis parallelisation is exact: output row t of channel c depends only
// on input rows t..t-3, all present in qkv_in (priors for t < 3 come from the
// conv ring). The per-element accumulation order is unchanged, so every output
// bit matches the serial kernel. Only group 0 touches the ring: it reads the
// old ring for rows t < 3 and writes the new ring from the input tail, in the
// same thread and in program order — no cross-block race. Later groups never
// touch the ring.
//
// Bit-exactness contract (same recipe as dflash_gdn_pre.gfx1100.hip, whose
// byte-identity is certified by test_dflash_gdn_pre_gfx1100.rs):
// - conv: explicit fmaf tree `acc = w3*x; acc += w2*s0; acc += w1*s1;
//   acc += w0*s2` with in-place weight indexing (the `+`-chain source form
//   does not reliably recompile to that tree in a new block shape).
// - Q/K norm: verbatim strided accumulation + __shfl_xor wave32 tree + the TWO
//   ordered multiplies (`qv *= inv; qv *= scale` — folding breaks identity).
// - gate: verbatim sigmoid / softplus guards / `sp * (-exp(a_log))`.
// - LDS float staging of conv outputs is exact; q_raw/k_raw stores are kept.
//
// Host-side admission (checked by the launcher, anything else stays on the
// old 3-launch path): exact gfx1201, sequential batch semantics, no tree
// parents, no DFlash tape, head_dim == 128, ratio >= 1 with exact division,
// n_tokens >= 1 (ragged tails run with a short last group).

#define GDN_PREB_HD 128
#define GDN_PREB_BLOCK 256
#define GDN_PREB_R 32

// Same fmaf tree as GDN_PRE_CONV_STEP (dflash_gdn_pre.gfx1100.hip:66-80),
// decoupled from the ring: priors arrive as values, no state writeback here.
#define GDN_PREB_CONV_Y(X, S0, S1, S2, C, Y)                                   \
    do {                                                                        \
        float y = fmaf((W)[(C) * 4 + 3], (X), (W)[(C) * 4 + 2] * (S0));          \
        y = fmaf((W)[(C) * 4 + 1], (S1), y);                                    \
        y = fmaf((W)[(C) * 4], (S2), y);                                        \
        (Y) = y / (1.0f + expf(-y));                                            \
    } while (0)

extern "C" __launch_bounds__(GDN_PREB_BLOCK)
__global__ void gdn_pre_batched_gfx1201(
    float* __restrict__ beta,         // [N x n_v_heads] in/out, sigmoid
    float* __restrict__ alpha,        // [N x n_v_heads] in/out, alpha gate
    const float* __restrict__ dt_bias,// [n_v_heads]
    const float* __restrict__ a_log,  // [n_v_heads]
    const float* __restrict__ qkv_in, // [N x qkv_dim] raw projection
    const float* __restrict__ conv_w, // [n_channels x 4]
    float* __restrict__ conv_state,   // [n_channels x 3] single lane
    float* __restrict__ q_raw,        // [N x k_dim] conv Q (kept)
    float* __restrict__ k_raw,        // [N x k_dim] conv K (kept)
    float* __restrict__ v_out,        // [N x v_dim]
    float* __restrict__ q_dst,        // [N x n_v_heads*HD] normed+scaled+repeated
    float* __restrict__ k_dst,        // [N x n_v_heads*HD] normed+repeated
    int n_v_heads,
    int n_key_heads,
    int ratio,
    int k_dim,
    int v_dim,
    int qkv_dim,
    int n_tokens,
    float q_scale,
    float eps) {
    __shared__ float q_s[GDN_PREB_R][GDN_PREB_HD];
    __shared__ float k_s[GDN_PREB_R][GDN_PREB_HD];
    __shared__ float q_inv[GDN_PREB_R];
    __shared__ float k_inv[GDN_PREB_R];

    const float* W = conv_w;
    float* S = conv_state;
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int n_groups = (n_tokens + GDN_PREB_R - 1) / GDN_PREB_R;
    const int qk_blocks = n_key_heads * n_groups;
    const int v_blocks = (v_dim + GDN_PREB_BLOCK - 1) / GDN_PREB_BLOCK;

    if (bx < qk_blocks) {
        // Q/K head block: owns Q channels [h*HD,(h+1)*HD) and K channels
        // [k_dim+h*HD,k_dim+(h+1)*HD) over this group's rows. Barrier-free
        // conv staging (channel-exclusive), ONE barrier, verbatim 32-lane
        // norm trees, then a 256-wide interleave scatter.
        const int g = bx / n_key_heads;
        const int h = bx % n_key_heads;
        const int t0 = g * GDN_PREB_R;
        int nrows = n_tokens - t0;
        if (nrows > GDN_PREB_R) nrows = GDN_PREB_R;
        const int is_k = (tid >= GDN_PREB_HD) ? 1 : 0;
        const int d = tid & (GDN_PREB_HD - 1);
        const int c = is_k ? (k_dim + h * GDN_PREB_HD + d) : (h * GDN_PREB_HD + d);
        for (int r = 0; r < nrows; ++r) {
            const int t = t0 + r;
            const long long in_row = (long long)t * qkv_dim;
            float x = qkv_in[in_row + c];
            float s0 = (t >= 1) ? qkv_in[(long long)(t - 1) * qkv_dim + c] : S[c * 3];
            float s1 = (t >= 2) ? qkv_in[(long long)(t - 2) * qkv_dim + c]
                                : ((t == 1) ? S[c * 3] : S[c * 3 + 1]);
            float s2 = (t >= 3) ? qkv_in[(long long)(t - 3) * qkv_dim + c]
                                : ((t == 2) ? S[c * 3] : ((t == 1) ? S[c * 3 + 1] : S[c * 3 + 2]));
            float result = 0.0f;
            GDN_PREB_CONV_Y(x, s0, s1, s2, c, result);
            const long long qk_row = (long long)t * k_dim;
            if (is_k) {
                k_raw[qk_row + h * GDN_PREB_HD + d] = result;
                k_s[r][d] = result;
            } else {
                q_raw[qk_row + h * GDN_PREB_HD + d] = result;
                q_s[r][d] = result;
            }
        }
        if (g == 0) {
            // Sole ring writer (same thread that read it above, in order;
            // later groups never touch the ring, so no cross-block race).
            // Final ring is (x_{N-1}, x_{N-2}, x_{N-3}) with pre-chunk ring
            // contents standing in for negative indices.
            float o0 = S[c * 3];
            float o1 = S[c * 3 + 1];
            S[c * 3] = qkv_in[(long long)(n_tokens - 1) * qkv_dim + c];
            S[c * 3 + 1] = (n_tokens >= 2)
                ? qkv_in[(long long)(n_tokens - 2) * qkv_dim + c]
                : o0;
            S[c * 3 + 2] = (n_tokens >= 3)
                ? qkv_in[(long long)(n_tokens - 3) * qkv_dim + c]
                : ((n_tokens == 2) ? o0 : o1);
        }
        __syncthreads();
        // Verbatim interleave-kernel reduction, one row at a time.
        if (tid < 32) {
            for (int r = 0; r < nrows; ++r) {
                float q_sq = 0.0f;
                for (int dd = tid; dd < GDN_PREB_HD; dd += 32)
                    q_sq += q_s[r][dd] * q_s[r][dd];
                for (int o = 16; o > 0; o >>= 1)
                    q_sq += __shfl_xor(q_sq, o);
                q_inv[r] = rsqrtf(q_sq + eps);
                float k_sq = 0.0f;
                for (int dd = tid; dd < GDN_PREB_HD; dd += 32)
                    k_sq += k_s[r][dd] * k_s[r][dd];
                for (int o = 16; o > 0; o >>= 1)
                    k_sq += __shfl_xor(k_sq, o);
                k_inv[r] = rsqrtf(k_sq + eps);
            }
        }
        __syncthreads();
        for (int r = 0; r < nrows; ++r) {
            const float qi = q_inv[r];
            const float ki = k_inv[r];
            const long long dst_row = (long long)(t0 + r) * n_v_heads * GDN_PREB_HD;
            for (int i = tid; i < GDN_PREB_HD * ratio; i += GDN_PREB_BLOCK) {
                const int dd = i / ratio;
                const int vh = h * ratio + (i % ratio);
                float qv = q_s[r][dd] * qi;
                qv *= q_scale;
                float kv = k_s[r][dd] * ki;
                q_dst[dst_row + vh * GDN_PREB_HD + dd] = qv;
                k_dst[dst_row + vh * GDN_PREB_HD + dd] = kv;
            }
        }
        return;
    }

    const int v_total = v_blocks * n_groups;
    if (bx < qk_blocks + v_total) {
        // V block: owns V channels [vi0, vi0+256) over this group's rows.
        // No norm follows, so outputs stream straight to global; no barrier.
        const int vbx = bx - qk_blocks;
        const int g = vbx / v_blocks;
        const int vb = vbx % v_blocks;
        const int t0 = g * GDN_PREB_R;
        int nrows = n_tokens - t0;
        if (nrows > GDN_PREB_R) nrows = GDN_PREB_R;
        const int vi = vb * GDN_PREB_BLOCK + tid;
        if (vi >= v_dim) return;
        const int c = 2 * k_dim + vi;
        for (int r = 0; r < nrows; ++r) {
            const int t = t0 + r;
            const long long in_row = (long long)t * qkv_dim;
            float x = qkv_in[in_row + c];
            float s0 = (t >= 1) ? qkv_in[(long long)(t - 1) * qkv_dim + c] : S[c * 3];
            float s1 = (t >= 2) ? qkv_in[(long long)(t - 2) * qkv_dim + c]
                                : ((t == 1) ? S[c * 3] : S[c * 3 + 1]);
            float s2 = (t >= 3) ? qkv_in[(long long)(t - 3) * qkv_dim + c]
                                : ((t == 2) ? S[c * 3] : ((t == 1) ? S[c * 3 + 1] : S[c * 3 + 2]));
            float result = 0.0f;
            GDN_PREB_CONV_Y(x, s0, s1, s2, c, result);
            v_out[(long long)t * v_dim + vi] = result;
        }
        if (g == 0) {
            float o0 = S[c * 3];
            float o1 = S[c * 3 + 1];
            S[c * 3] = qkv_in[(long long)(n_tokens - 1) * qkv_dim + c];
            S[c * 3 + 1] = (n_tokens >= 2)
                ? qkv_in[(long long)(n_tokens - 2) * qkv_dim + c]
                : o0;
            S[c * 3 + 2] = (n_tokens >= 3)
                ? qkv_in[(long long)(n_tokens - 3) * qkv_dim + c]
                : ((n_tokens == 2) ? o0 : o1);
        }
        return;
    }

    // Prep block: elementwise sigmoid(beta) + alpha gate over this group's
    // rows. Verbatim fused_sigmoid_alpha_gate.hip math, flattened so all 256
    // threads share the rows.
    {
        const int g = bx - qk_blocks - v_total;
        const int t0 = g * GDN_PREB_R;
        int nrows = n_tokens - t0;
        if (nrows > GDN_PREB_R) nrows = GDN_PREB_R;
        const int total = n_v_heads * nrows;
        for (int i = tid; i < total; i += GDN_PREB_BLOCK) {
            const int r = i / n_v_heads;
            const int j = i % n_v_heads;
            const long long ab = (long long)(t0 + r) * n_v_heads + j;
            float b = beta[ab];
            b = 1.0f / (1.0f + expf(-b));
            beta[ab] = b;
            float a = alpha[ab];
            float biased = a + dt_bias[j];
            float sp = (biased > 20.0f)
                ? biased
                : ((biased < -20.0f) ? expf(biased) : logf(1.0f + expf(biased)));
            a = sp * (-expf(a_log[j]));
            alpha[ab] = a;
        }
    }
}
