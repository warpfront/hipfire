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

// ---------------------------------------------------------------------------
// HIPFIRE_QWEN4_GDN_LANES_FUSE (default off): the Core of up to 8 lanes in ONE
// launch. gated_delta_conv_params_bf16_f32_lanes is the prologue of the step
// block, and the output projection's mq_rotate_x is the epilogue (the head-pair
// arrival-counter rotation of gated_delta_step_gate_norm128_body, one counter
// set per lane). Grid [value_heads, 1, lanes], block 128 (exact 128 x 128 heads,
// K = 4 convolution, ring of 3).
//
// Block (lane, value head vh, key head kh = vh / (value_heads / key_heads)):
//   * thread 0 runs gated_delta_params_head for vh (gate / beta stored as the
//     conv_params kernel stores them);
//   * thread t convolves the three channels q[kh * 128 + t], k[kh * 128 + t],
//     v[vh * 128 + t] with gated_delta_conv_channel's arithmetic and stores them
//     to the lane's conv output row (the q / k channels are shared by the
//     `repeat` value heads of kh: each block stores the identical bytes);
//   * the ring update of gated_delta_conv_channel (every slot rewritten
//     BF16-rounded, slot `cursor` = the BF16-rounded input) is deferred past
//     every read of the channel: the v channels belong to this block alone,
//     the q / k channels are written by the LAST of kh's `repeat` blocks to
//     arrive on key_counters[lane * key_heads + kh] (reset by that block).
// then the unchanged step + gated-norm body runs on the stored conv output.
// ---------------------------------------------------------------------------
struct GdnFuseLanes {
    const float* input[GDN_LANES_MAX];
    float* conv_out[GDN_LANES_MAX];
    float* history[GDN_LANES_MAX];
    const float* a[GDN_LANES_MAX];
    const float* b[GDN_LANES_MAX];
    float* gate[GDN_LANES_MAX];
    float* beta[GDN_LANES_MAX];
    signed char* state[GDN_LANES_MAX];
    const float* z[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    float* rotated[GDN_LANES_MAX];
    int cursor[GDN_LANES_MAX];
    unsigned int frame[GDN_LANES_MAX];
};

// gated_delta_conv_channel (history_rows == 3, kernel_size == 4) up to, not
// including, the ring update. `h[tap]` is the raw ring entry of tap `tap`
// (ring slot (cursor + tap) % 3); `current` the BF16-rounded input.
static __device__ __forceinline__ float gdn_conv_k4_channel(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    int channels,
    int cursor,
    int channel,
    float (&h)[3],
    float& current) {
    current = bf16_to_f32(bf16_round(input[channel]));
    float value = bf16_to_f32(kernel[channel * 4 + 3]) * current;
    float w[3];
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) {
        w[tap] = bf16_to_f32(kernel[channel * 4 + tap]);
        h[tap] = history[((cursor + tap) % 3) * channels + channel];
    }
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) value += w[tap] * bf16_to_f32(bf16_round(h[tap]));
    return bf16_to_f32(bf16_round(value * (1.0f / (1.0f + expf(-value)))));
}

// The ring update of gated_delta_conv_channel for one channel from the raw
// entries `h` the channel read: every slot rewritten BF16-rounded, then slot
// `cursor` takes the BF16-rounded input.
static __device__ __forceinline__ void gdn_conv_k4_roll(
    float* history,
    int channels,
    int cursor,
    int channel,
    const float (&h)[3],
    float current) {
#pragma unroll
    for (int tap = 1; tap < 3; ++tap) {
        history[((cursor + tap) % 3) * channels + channel] = bf16_to_f32(bf16_round(h[tap]));
    }
    history[cursor * channels + channel] = current;
}

extern "C" __global__ void gated_delta_step_gate_norm128_q8_lanes_fuse(
    const __hip_bfloat16* conv_kernel,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    const __hip_bfloat16* norm,
    const float* signs1,
    const float* signs2,
    unsigned int* pair_counters,
    unsigned int* key_counters,
    int channels,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    int qk,
    GdnFuseLanes lanes) {
    __shared__ int last_arrival;
    const int l = (int)blockIdx.z;
    const int vh = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int repeat = value_heads / key_heads;
    const int kh = vh / repeat;
    const float* input = lanes.input[l];
    float* conv_out = lanes.conv_out[l];
    float* history = lanes.history[l];
    const int cursor = lanes.cursor[l];
#if defined(__gfx1201__)
    {
        // Hint only: bring this head's 16 KiB of Q8 codes (one 4 KiB quarter per wave) and its
        // 512 B of scales toward the cache while the convolution below runs; the step body then
        // finds them there instead of paying the cold HBM latency after its norm.
        const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
        const char* codes = reinterpret_cast<const char*>(lanes.state[l]) + (size_t)vh * 16384;
        __builtin_amdgcn_s_prefetch_data((const void*)(codes + wave * 4096), 31);
        if (wave == 0) {
            __builtin_amdgcn_s_prefetch_data(
                (const void*)(reinterpret_cast<const char*>(lanes.state[l])
                              + (size_t)value_heads * 16384 + (size_t)vh * 512), 3);
        }
    }
#endif
    if (tid == 0) {
        gated_delta_params_head(lanes.a[l], lanes.b[l], a_log, dt_bias, lanes.gate[l],
                                lanes.beta[l], vh);
    }
    const int cq = kh * 128 + tid;
    const int ck = qk + kh * 128 + tid;
    const int cv = 2 * qk + vh * 128 + tid;
    float hq[3], hk[3], hv[3];
    float curq, curk, curv;
    conv_out[cq] = gdn_conv_k4_channel(input, conv_kernel, history, channels, cursor, cq, hq, curq);
    conv_out[ck] = gdn_conv_k4_channel(input, conv_kernel, history, channels, cursor, ck, hk, curk);
    conv_out[cv] = gdn_conv_k4_channel(input, conv_kernel, history, channels, cursor, cv, hv, curv);
    // The v channels are this block's alone.
    gdn_conv_k4_roll(history, channels, cursor, cv, hv, curv);
    // Every read of the shared q / k ring channels is complete, and the conv output row and
    // this head's gate / beta are visible to the block, past this barrier.
    __syncthreads();
    bool last = true;
    if (repeat > 1) {
        if (tid == 0) {
            last_arrival = __hip_atomic_fetch_add(&key_counters[l * key_heads + kh], 1u,
                                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT)
                == (unsigned int)(repeat - 1);
        }
        __syncthreads();
        last = last_arrival != 0;
    }
    if (last) {
        gdn_conv_k4_roll(history, channels, cursor, cq, hq, curq);
        gdn_conv_k4_roll(history, channels, cursor, ck, hk, curk);
        if (repeat > 1 && tid == 0) key_counters[l * key_heads + kh] = 0u;
    }
    gated_delta_step_gate_norm128_body<true>(conv_out, conv_out + qk, conv_out + 2 * qk,
        lanes.gate[l], lanes.beta[l], lanes.state[l], lanes.z[l], norm, lanes.output[l],
        key_heads, value_heads, key_dim, value_dim, query_scale, lanes.rotated[l], signs1, signs2,
        pair_counters + l * (value_heads / 2), lanes.frame[l]);
}

// ---------------------------------------------------------------------------
// Multi-lane verify core (phase 3): lanes with 2..=8 rows each under the armed
// capture ring, the per-lane chain
//   copy_d2d projection -> capture.inputs
//   gated_delta_conv_params_batched  (gated_delta_conv_bf16_f32_batched_k4)
//   gated_delta_step_batched         (..._persistent256_capture_q8)
// as one launch per kernel for every lane. The gate that follows is row
// independent: gated_delta_gate_bf16_f32_rows_lanes runs the same head body per
// lane (the lanes' row scratch need not be adjacent: AR lanes of the same step
// sit between verify lanes in the packed rows).
// ---------------------------------------------------------------------------

struct GdnConvRowsLanes {
    // The lane's projection rows [rows, channels] (read) and the capture inputs
    // [rows, channels] they are copied into (the fused copy_d2d; may be null).
    const float* input[GDN_LANES_MAX];
    float* inputs[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    float* history[GDN_LANES_MAX];
    const float* a[GDN_LANES_MAX];
    const float* b[GDN_LANES_MAX];
    float* gate[GDN_LANES_MAX];
    float* beta[GDN_LANES_MAX];
    int rows[GDN_LANES_MAX];
    int cursor[GDN_LANES_MAX];
};

// gated_delta_conv_bf16_f32_batched_k4 (F32 in/out, no qk-norm, with the gate
// parameters) on each lane's own rows. Grid [conv_blocks * ceil(max_rows / 16) +
// ceil(max_rows * heads / 256), lanes], block 256, lane = blockIdx.y. Tile i <
// conv_total is (channel block i % conv_blocks, 16-row block i / conv_blocks); the
// rest are parameter blocks. The tile body is the singleton kernel's, row for
// row (the same expression sequence and ring tail); additionally every input row
// value the tile reads for its own rows is stored to `inputs` (bit copy).
extern "C" __global__ void gated_delta_conv_params_bf16_f32_rows_lanes(
    const __hip_bfloat16* kernel,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    int channels,
    int heads,
    int max_rows,
    GdnConvRowsLanes lanes) {
    const int l = (int)blockIdx.y;
    const int rows = lanes.rows[l];
    const unsigned conv_blocks = (unsigned)(channels + 255) / 256u;
    const unsigned conv_total = conv_blocks * (((unsigned)max_rows + 15u) / 16u);
    if (blockIdx.x >= conv_total) {
        const int flat = (int)(blockIdx.x - conv_total) * 256 + (int)threadIdx.x;
        if (flat < rows * heads) {
            gated_delta_params_flat(lanes.a[l], lanes.b[l], a_log, dt_bias, lanes.gate[l],
                                    lanes.beta[l], flat, heads);
        }
        return;
    }
    const unsigned by = blockIdx.x / conv_blocks;
    const unsigned bx = blockIdx.x - by * conv_blocks;
    const int channel = (int)(bx * blockDim.x + threadIdx.x);
    const int r0 = (int)by * 16;
    if (channel >= channels || r0 >= rows) return;
    const int start_cursor = lanes.cursor[l];
    const float* input = lanes.input[l];
    float* inputs = lanes.inputs[l];
    float* output = lanes.output[l];
    float* next_history = lanes.history[l];
    const float* history = lanes.history[l];
    auto x = [&](int j) {
        return j >= 0 ? bf16_round(input[j * channels + channel])
                      : bf16_round(history[((start_cursor + j + 3) % 3) * channels + channel]);
    };
    const float k0 = bf16_to_f32(kernel[channel * 4 + 0]);
    const float k1 = bf16_to_f32(kernel[channel * 4 + 1]);
    const float k2 = bf16_to_f32(kernel[channel * 4 + 2]);
    const float k3 = bf16_to_f32(kernel[channel * 4 + 3]);
    __hip_bfloat16 h0 = x(r0 - 3), h1 = x(r0 - 2), h2 = x(r0 - 1);
    __hip_bfloat16 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    const int r1 = min(r0 + 16, rows);
    for (int r = r0; r < r1; ++r) {
        const float raw = input[r * channels + channel];
        if (inputs != nullptr) inputs[r * channels + channel] = raw;
        const __hip_bfloat16 current_bf16 = bf16_round(raw);
        float value = k3 * bf16_to_f32(current_bf16);
        value += k0 * bf16_to_f32(h0);
        value += k1 * bf16_to_f32(h1);
        value += k2 * bf16_to_f32(h2);
        const __hip_bfloat16 out = bf16_round(value * (1.0f / (1.0f + expf(-value))));
        output[r * channels + channel] = bf16_to_f32(out);
        h0 = h1;
        h1 = h2;
        h2 = current_bf16;
    }
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s)
            next_history[s * channels + channel] = bf16_to_f32(tail[s]);
    }
}

struct GdnStepRowsLanes {
    // The lane's convolution output rows [rows, qkv] (q | k | v per row), gate and
    // beta rows, the live Q8 slot (read only), the recurrent output rows, and the
    // base of its capture ring (the final state lands in slot rows - 1).
    const float* projection[GDN_LANES_MAX];
    const float* gate[GDN_LANES_MAX];
    const float* beta[GDN_LANES_MAX];
    signed char* state[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    signed char* ring[GDN_LANES_MAX];
    int rows[GDN_LANES_MAX];
    unsigned int frame[GDN_LANES_MAX];
};

// gated_delta_step_halves_state128_persistent256_capture_q8 per lane (grid
// [value_heads, 1, lanes], block 256, lane = blockIdx.z): the same device body
// and template instantiation, a lane's rows run in order inside one block per
// value head. `frame` is the lane's last row's position.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_capture_q8_lanes(
    int qkv_width,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    GdnStepRowsLanes lanes) {
    const int l = (int)blockIdx.z;
    gated_delta_step_persistent256_body<true, true>(lanes.projection[l], lanes.gate[l],
        lanes.beta[l], lanes.state[l], lanes.output[l], lanes.ring[l], lanes.rows[l], qkv_width,
        key_heads, value_heads, key_dim, value_dim, query_scale, lanes.frame[l]);
}

struct GdnGateRowsLanes {
    // The lane's recurrent output rows, z rows and gdn output rows
    // [rows, value_heads * 128]; lanes need not be adjacent in memory.
    float* recurrent[GDN_LANES_MAX];
    const float* z[GDN_LANES_MAX];
    float* output[GDN_LANES_MAX];
    int rows[GDN_LANES_MAX];
};

// gated_delta_gate_bf16_f32_batched on each lane's own rows (the gate is
// row independent: the same head body on the same flat head index). Grid
// [max_rows * value_heads, lanes], block 32, lane = blockIdx.y.
extern "C" __global__ __launch_bounds__(32) void gated_delta_gate_bf16_f32_rows_lanes(
    const __hip_bfloat16* norm,
    int value_heads,
    GdnGateRowsLanes lanes) {
    const int l = (int)blockIdx.y;
    const int flat_head = (int)blockIdx.x;
    if (flat_head >= lanes.rows[l] * value_heads) return;
    __shared__ float rounded[128];
    gated_delta_gate_head128(lanes.recurrent[l], lanes.z[l], norm, lanes.output[l],
                             flat_head * 128, (int)threadIdx.x, rounded);
}
#undef GDN_LANES_MAX
