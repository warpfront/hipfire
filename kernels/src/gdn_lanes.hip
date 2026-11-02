// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// Multi-lane one-row GDN decode core (fn-lanes): the Core phase of several AR
// lanes in one launch per kernel instead of one per lane. Lane l runs exactly
// the per-lane kernels' bodies on its own rows and state:
//   gated_delta_conv_params_bf16_f32_lanes  = gated_delta_conv_params_bf16_f32
//     (grid [conv_blocks + 1, lanes], block 256; lane = blockIdx.y)
//   gated_delta_step_gate_norm128_q8_lanes  = gated_delta_step_gate_norm128_q8
//     without the optional rotation (grid [value_heads, 1, lanes], block 128;
//     lane = blockIdx.z)
// Appended to tensor_ops.hip (same helpers, same compile flags). Lanes share
// the weights and never share a state, so their blocks are independent.
#define GDN_LANES_MAX 8

struct GdnConvLanes {
    const float* input[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    float* history[GDN_LANES_MAX];
    const float* a[GDN_LANES_MAX];
    const float* b[GDN_LANES_MAX];
    float* gate[GDN_LANES_MAX];
    float* beta[GDN_LANES_MAX];
    int cursor[GDN_LANES_MAX];
};

extern "C" __global__ void gated_delta_conv_params_bf16_f32_lanes(
    const __hip_bfloat16* kernel,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    int channels,
    int history_rows,
    int kernel_size,
    int heads,
    int conv_blocks,
    GdnConvLanes lanes) {
    const int l = (int)blockIdx.y;
    if ((int)blockIdx.x < conv_blocks) {
        int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (channel >= channels) return;
        gated_delta_conv_channel(lanes.input[l], kernel, lanes.history[l], lanes.output[l],
                                 lanes.history[l], channels, history_rows, kernel_size,
                                 lanes.cursor[l], channel);
    } else {
        int head = (int)threadIdx.x;
        if (head >= heads) return;
        gated_delta_params_head(lanes.a[l], lanes.b[l], a_log, dt_bias, lanes.gate[l],
                                lanes.beta[l], head);
    }
}

struct GdnStepLanes {
    // The lane's convolution output row: q (qk) | k (qk) | v.
    const float* qkv[GDN_LANES_MAX];
    const float* gate[GDN_LANES_MAX];
    const float* beta[GDN_LANES_MAX];
    signed char* state[GDN_LANES_MAX];
    const float* z[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    unsigned int frame[GDN_LANES_MAX];
};

extern "C" __global__ void gated_delta_step_gate_norm128_q8_lanes(
    const __hip_bfloat16* norm,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    int qk,
    GdnStepLanes lanes) {
    const int l = (int)blockIdx.z;
    const float* qkv = lanes.qkv[l];
    gated_delta_step_gate_norm128_body<true>(qkv, qkv + qk, qkv + 2 * qk, lanes.gate[l],
        lanes.beta[l], lanes.state[l], lanes.z[l], norm, lanes.output[l], key_heads, value_heads,
        key_dim, value_dim, query_scale, nullptr, nullptr, nullptr, nullptr, lanes.frame[l]);
}
#undef GDN_LANES_MAX
