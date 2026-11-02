// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Railgun E0 / L6c: multi-layer GDN replay preamble (gfx1201).
//
// `GdnTape::replay_gdn` issues, per LinearAttention layer, conv1d + QK norm
// + repeat-interleave (3 launches on gfx1201) and then the GDN recurrence —
// 192 launches for 48 layers. Layers are independent (per-layer tape, conv
// state and recurrent state), so this kernel runs the preamble of EVERY
// layer in one launch: grid = [n_key_heads + v_blocks, n_layers, 1],
// block = [256, 1, 1]; blockIdx.y selects the layer's row in a device
// pointer table. `gated_delta_net_q8_fast_ml` then runs the recurrence of
// every layer in a second launch (2 launches per replay instead of 192).
//
// The per-block body is `dflash_gdn_pre_replay_gfx1100` verbatim (same
// explicit-fmaf conv tree, same 32-lane norm reduction, same two ordered Q
// multiplies, same interleave mapping) with ONE change: the normed q_raw /
// k_raw stores are dropped. Those scratch rows are dead after replay (the
// recurrence reads the repeated q/k), and each layer here owns its own
// q/k/v rows, so nothing reads them. Byte parity against the gfx1201
// 3-launch sequence is gated by `test_dflash_replay_ml`.
//
// Constraints (host-enforced): head_dim == 128, 1 <= n_steps <= 16,
// n_v_heads % n_key_heads == 0, single-lane dense rows.

#define GDN_PRE_HD 128
#define GDN_PRE_BLOCK 256
#define GDN_PRE_MAXN 16

// Railgun D8 (`DFLASH_GDN_REPLAY_STATE_SRC`): the first step reads the conv
// ring from the snapshot (`conv_state_src`); every step writes `conv_state`,
// and later steps read back what this thread wrote there. Each channel is
// owned by one thread for all steps, so the result equals an in-place
// replay on a restored ring. Without the define `GDN_PRE_CONV_RD(t)` is `S`.
#if defined(DFLASH_GDN_REPLAY_STATE_SRC)
#define GDN_PRE_CONV_RD(T) ((T) == 0 ? S_src : S)
#else
#define GDN_PRE_CONV_RD(T) S
#endif

#define GDN_PRE_CONV_STEP(X, C, Y)                                            \
    do {                                                                      \
        float x = (X);                                                        \
        float s0 = (GDN_PRE_CONV_RD(t))[(C) * 3];                             \
        float s1 = (GDN_PRE_CONV_RD(t))[(C) * 3 + 1];                         \
        float s2 = (GDN_PRE_CONV_RD(t))[(C) * 3 + 2];                         \
        float y = fmaf((W)[(C) * 4 + 3], x, (W)[(C) * 4 + 2] * s0);            \
        y = fmaf((W)[(C) * 4 + 1], s1, y);                                    \
        y = fmaf((W)[(C) * 4], s2, y);                                        \
        float r = y / (1.0f + expf(-y));                                      \
        (S)[(C) * 3 + 2] = s1;                                                \
        (S)[(C) * 3 + 1] = s0;                                                \
        (S)[(C) * 3] = x;                                                     \
        (Y) = r;                                                              \
    } while (0)

// One row per LinearAttention layer. Layout mirrors
// `DflashReplayPreLayer` in crates/rdna-compute/src/dflash_gdn_replay.rs.
struct DflashReplayPreLayer {
    const float* qkv_tape;  // [max_n x qkv_dim] taped raw qkv
    const float* conv_w;    // [n_channels x 4]
    float* conv_state;      // [n_channels x 3] single lane
    float* v_out;           // [n_steps x v_dim]
    float* q_dst;           // [n_steps x n_v_heads*HD] normed+scaled+repeated
    float* k_dst;           // [n_steps x n_v_heads*HD] normed+repeated
#if defined(DFLASH_GDN_REPLAY_STATE_SRC)
    const float* conv_state_src;  // [n_channels x 3] pre-step ring (snapshot)
#endif
};

#ifndef DFLASH_GDN_REPLAY_PRE_KERNEL
#define DFLASH_GDN_REPLAY_PRE_KERNEL dflash_gdn_replay_pre_ml
#endif

extern "C" __launch_bounds__(GDN_PRE_BLOCK)
__global__ void DFLASH_GDN_REPLAY_PRE_KERNEL(
    const DflashReplayPreLayer* __restrict__ layers,
    int n_v_heads,
    int n_key_heads,
    int ratio,
    int k_dim,
    int v_dim,
    int qkv_dim,
    int n_steps,
    float q_scale,
    float eps) {
    __shared__ float q_s[GDN_PRE_MAXN][GDN_PRE_HD];
    __shared__ float k_s[GDN_PRE_MAXN][GDN_PRE_HD];
    __shared__ float q_inv[GDN_PRE_MAXN];
    __shared__ float k_inv[GDN_PRE_MAXN];

    const DflashReplayPreLayer layer = layers[blockIdx.y];
    const float* __restrict__ qkv_tape = layer.qkv_tape;
    float* __restrict__ v_out = layer.v_out;
    float* __restrict__ q_dst = layer.q_dst;
    float* __restrict__ k_dst = layer.k_dst;
    const float* W = layer.conv_w;
    float* S = layer.conv_state;
#if defined(DFLASH_GDN_REPLAY_STATE_SRC)
    const float* S_src = layer.conv_state_src;
#endif
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int v_blocks = (v_dim + GDN_PRE_BLOCK - 1) / GDN_PRE_BLOCK;
    if (bx < n_key_heads) {
        const int h = bx;
        for (int t = 0; t < n_steps; ++t) {
            const long long in_row = (long long)t * qkv_dim;
            float result = 0.0f;
            if (tid < GDN_PRE_HD) {
                const int c = h * GDN_PRE_HD + tid;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                q_s[t][tid] = result;
            } else {
                const int d = tid - GDN_PRE_HD;
                const int c = k_dim + h * GDN_PRE_HD + d;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                k_s[t][d] = result;
            }
        }
        __syncthreads();
        if (tid < 32) {
            for (int t = 0; t < n_steps; ++t) {
                float q_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    q_sq += q_s[t][d] * q_s[t][d];
                for (int o = 16; o > 0; o >>= 1)
                    q_sq += __shfl_xor(q_sq, o);
                q_inv[t] = rsqrtf(q_sq + eps);
                float k_sq = 0.0f;
                for (int d = tid; d < GDN_PRE_HD; d += 32)
                    k_sq += k_s[t][d] * k_s[t][d];
                for (int o = 16; o > 0; o >>= 1)
                    k_sq += __shfl_xor(k_sq, o);
                k_inv[t] = rsqrtf(k_sq + eps);
            }
        }
        __syncthreads();
        for (int t = 0; t < n_steps; ++t) {
            const float qi = q_inv[t];
            const float ki = k_inv[t];
            const long long dst_row = (long long)t * n_v_heads * GDN_PRE_HD;
            for (int i = tid; i < GDN_PRE_HD * ratio; i += GDN_PRE_BLOCK) {
                const int d = i / ratio;
                const int r = i % ratio;
                // Old in-place order: q *= inv, then q *= scale.
                float qv = q_s[t][d] * qi;
                qv *= q_scale;
                float kv = k_s[t][d] * ki;
                const int vh = h * ratio + r;
                q_dst[dst_row + vh * GDN_PRE_HD + d] = qv;
                k_dst[dst_row + vh * GDN_PRE_HD + d] = kv;
            }
        }
        return;
    }

    if (bx < n_key_heads + v_blocks) {
        for (int t = 0; t < n_steps; ++t) {
            const long long in_row = (long long)t * qkv_dim;
            for (int stripe = bx - n_key_heads; ; stripe += v_blocks) {
                const int vi = stripe * GDN_PRE_BLOCK + tid;
                if (vi >= v_dim)
                    break;
                const int c = 2 * k_dim + vi;
                float result = 0.0f;
                GDN_PRE_CONV_STEP(qkv_tape[in_row + c], c, result);
                v_out[(long long)t * v_dim + vi] = result;
            }
        }
    }
}
