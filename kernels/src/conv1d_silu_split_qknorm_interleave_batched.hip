// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Prefill fusion of `conv1d_silu_split_f32` (n-tokens variant) with the
// DeltaNet Q/K normalization + repeat-interleave tail.
//
// Replaces, per LinearAttention layer on the Sequential (single-lane) prefill
// path, either
//   conv1d_silu_split_f32_n(q_raw, k_raw, v)               (1 launch)
//   fused_qk_l2_norm_scale_interleave_f32_batched           (1 launch, GQA)
// or the non-GQA triple
//   conv1d_silu_split_f32_n + fused_qk_l2_norm_scale_f32_batched
//   + repeat_interleave_qk_f32_batched / memcpy pair       (3-4 launches)
// with ONE launch. The conv's Q/K outputs never touch global memory: each
// (key head, row) block stages its 2*HD conv results in LDS, reduces, and
// stores normed+scaled+replicated values straight to the GDN inputs.
//
// Bit-exactness contract (eval md5 gate):
// - Conv: verbatim 4-tap causal depthwise + SiLU + ring-buffer update from
//   conv1d_silu_split.hip, INCLUDING in-place weight[] indexing (hoisting the
//   loads changes FMA fusion and nudges numerics ~1 ULP). The row loop is
//   sequential inside each block, so the state trajectory matches the old
//   token loop exactly; channels are partitioned across blocks with no
//   overlap, and state is per-channel, so the partition is invisible.
// - Norm: verbatim per-(head,row) strided accumulation + __shfl_xor wave32
//   tree + rsqrt + the TWO ordered Q multiplies (`qv *= inv; qv *= scale` —
//   folding them breaks byte-identity per fused_qk_l2_norm_scale.hip:43),
//   then the same ratio replication stores as
//   fused_qk_l2_norm_scale_interleave_f32_batched.hip.
// - V channels: verbatim conv path, same as the old kernel.
//
// Constraints (enforced host-side; anything else stays on the old path):
// - head_dim == 128 (HD). The LDS stage is sized for it.
// - Single-lane sequential batch: state is [(2*k_dim+v_dim) x 3], input and
//   outputs are dense [N x stride] row-major with lane 0, exactly like the
//   old `*_f32_n` batched kernels. Tree/independent/masked variants keep
//   their own kernels.
// - q_raw/k_raw are NOT populated (nothing downstream reads them; the old
//   interleave kernel already left them unmodified on the GQA route and no
//   consumer reads them after the norm — see prefill.rs). q_dst/k_dst/v_out
//   are bit-identical to the unfused sequence.
//
// Grid: [n_key_heads + v_blocks, 1, 1]. Block: [256]. Shared: 2*HD floats.

#define CQN_IB_HD 128
#define CQN_IB_BLOCK 256

extern "C" __launch_bounds__(CQN_IB_BLOCK)
__global__ void conv1d_silu_split_qknorm_interleave_batched(
    float* __restrict__ q_dst,   // [N x n_value_heads*HD], normed+scaled+replicated
    float* __restrict__ k_dst,   // [N x n_value_heads*HD], normed+replicated
    float* __restrict__ v_out,   // [N x v_dim]
    const float* __restrict__ input,  // [N x (2*k_dim+v_dim)]
    const float* __restrict__ weight, // [n_channels x 4]
    float* __restrict__ state,   // [n_channels x 3] single lane
    int k_dim,
    int v_dim,
    int n_tokens,
    int n_key_heads,
    int ratio,
    int head_dim,  // must be CQN_IB_HD (host-checked); kept for ABI clarity
    float q_scale,
    float eps
) {
    __shared__ float q_s[CQN_IB_HD];
    __shared__ float k_s[CQN_IB_HD];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int n_channels = 2 * k_dim + v_dim;
    const int n_value_heads = n_key_heads * ratio;
    (void)head_dim;

    if (bx < n_key_heads) {
        // Q/K head block: threads 0..127 own Q channels [h*HD,(h+1)*HD),
        // threads 128..255 own K channels [k_dim+h*HD,k_dim+(h+1)*HD).
        const int h = bx;
        const bool is_k = tid >= CQN_IB_HD;
        const int d = tid & (CQN_IB_HD - 1);
        const int c = is_k ? (k_dim + h * CQN_IB_HD + d) : (h * CQN_IB_HD + d);

        for (int t = 0; t < n_tokens; t++) {
            // Verbatim conv1d_silu_split_f32 math and state-update order.
            float x = input[(long long)t * n_channels + c];
            float s0 = state[c * 3];
            float s1 = state[c * 3 + 1];
            float s2 = state[c * 3 + 2];
            float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                    + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
            float result = y / (1.0f + expf(-y));
            state[c * 3 + 2] = s1;
            state[c * 3 + 1] = s0;
            state[c * 3] = x;
            if (is_k) k_s[d] = result;
            else      q_s[d] = result;
            __syncthreads();

            // Verbatim fused_qk_l2_norm_scale_interleave reduction, one
            // wave32 per (head, row). Only lanes 0..31 participate; every
            // thread still reaches both barriers.
            if (tid < 32) {
                float q_sq = 0.0f;
                float k_sq = 0.0f;
                for (int i = tid; i < CQN_IB_HD; i += 32) {
                    const float qv = q_s[i];
                    const float kv = k_s[i];
                    q_sq += qv * qv;
                    k_sq += kv * kv;
                }
                for (int o = 16; o > 0; o >>= 1) {
                    q_sq += __shfl_xor(q_sq, o);
                    k_sq += __shfl_xor(k_sq, o);
                }
                const float q_inv = rsqrtf(q_sq + eps);
                const float k_inv = rsqrtf(k_sq + eps);
                // Replicate-and-write across `ratio` value-head slots.
                // Keep the legacy two ordered Q multiplies bit-for-bit.
                const long long dst_row = (long long)t * n_value_heads * CQN_IB_HD;
                for (int r = 0; r < ratio; r++) {
                    const int vh = h * ratio + r;
                    float* q_o = q_dst + dst_row + (long long)vh * CQN_IB_HD;
                    float* k_o = k_dst + dst_row + (long long)vh * CQN_IB_HD;
                    for (int i = tid; i < CQN_IB_HD; i += 32) {
                        float qv = q_s[i] * q_inv;
                        qv *= q_scale;
                        q_o[i] = qv;
                        k_o[i] = k_s[i] * k_inv;
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    // V blocks: verbatim V-channel path from conv1d_silu_split_f32.
    const int v_block = bx - n_key_heads;
    const int vi = v_block * CQN_IB_BLOCK + tid;
    if (vi >= v_dim) return;
    const int c = 2 * k_dim + vi;
    for (int t = 0; t < n_tokens; t++) {
        float x = input[(long long)t * n_channels + c];
        float s0 = state[c * 3];
        float s1 = state[c * 3 + 1];
        float s2 = state[c * 3 + 2];
        float y = weight[c * 4 + 3] * x + weight[c * 4 + 2] * s0
                + weight[c * 4 + 1] * s1 + weight[c * 4] * s2;
        float result = y / (1.0f + expf(-y));
        state[c * 3 + 2] = s1;
        state[c * 3 + 1] = s0;
        state[c * 3] = x;
        v_out[(long long)t * v_dim + vi] = result;
    }
}
