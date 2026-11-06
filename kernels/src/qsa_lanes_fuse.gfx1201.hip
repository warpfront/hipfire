// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// Fused multi-lane QSA decode core (HIPFIRE_QWEN4_QSA_LANES_FUSE, default off), exact
// gfx1201, fp8 K/V, BF16 index keys, ONE row per lane (AR decode). Appended after
// qsa_lanes.gfx1201.hip (struct QsaLanes) onto the tensor_ops.hip translation unit as its
// own module (`qsa_lanes_fuse`), so no unfused kernel's code object changes.
//
// The unfused lane chain is prologue -> pool -> select -> attention -> mq_rotate_x
// (five launches). A launch boundary is only needed where a later kernel reads what ALL
// workgroups of the earlier one wrote; with one row per lane the chain has two such
// boundaries, not four:
//
//   K1 indexed_attention_core_fused_fp8_lanes = prologue + pool + select
//     grid [1 + heads + kv_heads + 1, 1, lanes], block 256, dynamic LDS = the select scores.
//       x == 0                      the lane's index path, in ONE workgroup:
//                                   index-query norm+RoPE (in place, head by head: the
//                                   prologue's index heads), the pool block the row
//                                   completes (qsa_pool_rope's arithmetic; the newest raw
//                                   key is rebuilt from the index row's key part instead of
//                                   read back from the prologue's raw-key store, so no
//                                   workgroup waits on another), then the incumbent select
//                                   (qsa_select_batched, blockIdx.x == row 0) with the
//                                   persistent mirror.
//       x in [1, 1 + heads)         query head `x - 1` norm+RoPE (the prologue's q heads).
//       x in [1 + heads, ..+ kv)    K/V head: norm+RoPE, fp8 cache append (prologue kv heads).
//       x == 1 + heads + kv_heads   the index key's BF16 round trip + raw-key copy.
//     The index row's key part is read by the pool (x == 0) while the index-key block
//     rounds it in place; the rounding is idempotent, so either read yields the same value.
//     The index row's query part is only written/read by x == 0.
//   K2 indexed_attention_attention_rot_fp8_lanes = attention + mq_rotate_x
//     grid [heads, 1, lanes], block 256, dynamic LDS = the widest selected list: the
//     incumbent qsa_attention_row_parallel, then (one attention head is one 256-value FWHT
//     group) wave 0 rotates the head's output row exactly as mq_rotate_x does into the
//     out-projection's rotation scratch row. A null `x_rot[lane]` skips the rotation.
//
// Every output / state byte equals the unfused chain's (the oracle compares them).
#if defined(__GFX12__)
#define QSA_LANES_MAX 8

struct QsaLanesRot {
    float* x_rot[QSA_LANES_MAX];  // lane's rotated row (q_width floats); null = no rotation
};

extern "C" __global__ __launch_bounds__(256) void indexed_attention_core_fused_fp8_lanes(
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    const __hip_bfloat16* pool_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    int query_row_stride,
    int budget_blocks,
    int compress,
    int capacity,
    RopeYarn yarn,
    RopeYarn pool_yarn,
    QsaLanes lanes) {
    __shared__ float input_row[256];
    __shared__ float row_sum;
    __shared__ float kv_amax[8];
    __shared__ float partial[256];
    __shared__ float input4[4][256];
    __shared__ float row_sum4[4];
    const int l = (int)blockIdx.z;
    const int c = (int)threadIdx.x;
    const int position = lanes.pos[l];
    const int index_q_width = index_heads * index_dim;
    const int kv_width = kv_heads * head_dim;
    float* index_row = lanes.index_row[l];
    int role = (int)blockIdx.x;

    if (role == 0) {
        // ---- index queries: the prologue's index-head blocks (qsa_norm_rope_head's
        // operations), four heads at a time in one workgroup: head `j` of a group is
        // normalised by wave `j` (the same serial i-order sum), every thread finishes its
        // channel for all the group's heads. The RoPE angle depends on (position, channel)
        // only, so its sin/cos is evaluated once per thread, not once per head. ----
        const int rotary_dim = min(index_dim, 64);
        const int half = rotary_dim / 2;
        float sine = 0.0f, cosine = 0.0f;
        if (c < rotary_dim) {
            const int pair = c < half ? c : c - half;
            const float angle = (float)position
                * powf(10000000.0f, -((float)(2 * pair)) / (float)rotary_dim);
            rope_yarn_sincos(angle, pair, yarn, &sine, &cosine);
        }
        for (int h0 = 0; h0 < index_heads; h0 += 4) {
            const int group = min(4, index_heads - h0);
            if (c < index_dim) {
                for (int j = 0; j < group; ++j) input4[j][c] = index_row[(h0 + j) * index_dim + c];
            }
            __syncthreads();
            if (c < 32 * group) {
                const int j = c >> 5;
                float sum = 0.0f;
                for (int i = 0; i < index_dim; ++i) {
                    float row_value = input4[j][i];
                    sum += row_value * row_value;
                }
                if ((c & 31) == 0) row_sum4[j] = sum;
            }
            __syncthreads();
            if (c < index_dim) {
                const float norm_c = 1.0f + bf16_to_f32(index_q_norm[c]);
                const int partner = c < half ? c + half : c - half;
                const float norm_p = 1.0f + bf16_to_f32(index_q_norm[partner < index_dim ? partner : c]);
                for (int j = 0; j < group; ++j) {
                    const float inv = rsqrtf(row_sum4[j] / (float)index_dim + 1.0e-6f);
                    float value = input4[j][c] * inv * norm_c;
                    if (c < rotary_dim) {
                        float other = input4[j][partner] * inv * norm_p;
                        value = c < half ? value * cosine - other * sine
                            : other * sine + value * cosine;
                    }
                    index_row[(h0 + j) * index_dim + c] = value;
                }
            }
            // input4 / row_sum4 are reused by the next group; the rows are read below.
            __syncthreads();
        }
        // ---- pool: the blocks this row completes (qsa_pool_rope, same operations). ----
        const __hip_bfloat16* raw_keys = lanes.raw_idx[l];
        __hip_bfloat16* pooled = lanes.pooled[l];
        const float* index_key = index_row + index_q_width;
        for (int block = lanes.first_block[l]; block < lanes.blocks[l]; ++block) {
            // The raw key of `token`: the prologue's raw-key store for the row in flight
            // (BF16 round trip of the index key, then bf16_round), the arena otherwise.
            auto raw = [&](int token, int channel) -> float {
                if (token == position) {
                    float v = index_key[channel];
                    if (isfinite(v)) {
                        v = __uint_as_float((__float_as_uint(v) + 0x7FFFu
                                             + ((__float_as_uint(v) >> 16) & 1u))
                                            & 0xFFFF0000u);
                    }
                    return qsa_index_f32(bf16_round(v));
                }
                return qsa_index_f32(raw_keys[token * index_dim + channel]);
            };
            const bool active = c < index_dim;
            float mean = 0.0f;
            if (active) {
                for (int row = 0; row < compress; ++row) {
                    mean += raw(block * compress + row, c) / (float)compress;
                }
            }
            __hip_bfloat16 mean_bf16 = bf16_round(mean);
            float squared = active
                ? bf16_to_f32(bf16_round(bf16_to_f32(mean_bf16) * bf16_to_f32(mean_bf16)))
                : 0.0f;
            partial[c] = squared;
            __syncthreads();
            for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
                if (c < stride) partial[c] += partial[c + stride];
                __syncthreads();
            }
            if (active) {
                float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
                float norm_value = pool_norm == nullptr ? 0.0f : bf16_to_f32(pool_norm[c]);
                float value = bf16_to_f32(bf16_round(
                    bf16_to_f32(mean_bf16) * inv * (1.0f + norm_value)));
                int rotary_dim = index_dim < 64 ? index_dim : 64;
                int half = rotary_dim / 2;
                if (c < rotary_dim) {
                    int partner = c < half ? c + half : c - half;
                    float other_mean = 0.0f;
                    for (int row = 0; row < compress; ++row) {
                        other_mean += raw(block * compress + row, partner) / (float)compress;
                    }
                    __hip_bfloat16 other_bf16 = bf16_round(other_mean);
                    float other_norm = pool_norm == nullptr ? 0.0f : bf16_to_f32(pool_norm[partner]);
                    float other = bf16_to_f32(bf16_round(
                        bf16_to_f32(other_bf16) * inv * (1.0f + other_norm)));
                    int pair = c < half ? c : partner;
                    float angle = (float)(block * compress)
                        * powf(10000000.0f, -((float)(2 * pair)) / (float)rotary_dim);
                    float sine, cosine;
                    rope_yarn_sincos(angle, pair, pool_yarn, &sine, &cosine);
                    value = c < half ? value * cosine - other * sine
                        : other * sine + value * cosine;
                }
                qsa_index_store(pooled + block * index_dim + c, bf16_round(value));
            }
            // `partial` is reused by the next block; the pooled keys are read below.
            __syncthreads();
        }
        // ---- select: the incumbent body, row 0 of 1 (blockIdx.x == 0). ----
        qsa_select_batched<__hip_bfloat16>(
            index_row, pooled, lanes.selected[l], 1, query_row_stride, lanes.blocks[l],
            index_heads, index_dim, budget_blocks, compress, position, capacity,
            lanes.sel_persist[l], nullptr, 0);
        return;
    }
    role -= 1;
    if (role < heads) {
        // ---- q head (qgate row 0, `[q | gate]` per head). ----
        float* row = lanes.qgate[l] + role * 2 * head_dim;
        const float value = qsa_norm_rope_head(row, q_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        if (c < head_dim) row[c] = value;
        return;
    }
    role -= heads;
    if (role < kv_heads) {
        // ---- K/V head: norm+RoPE of the key, fp8 append of K and V at `position`. ----
        float* row = lanes.keys[l] + role * head_dim;
        const float value = qsa_norm_rope_head(row, k_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        const float v = c < head_dim ? lanes.values[l][role * head_dim + c] : 0.0f;
        if (c < head_dim) row[c] = value;
        const size_t row_bytes = (size_t)qsa_kv_row_bytes<QSA_KV_FP8>(kv_heads, head_dim);
        qsa_kv_store<QSA_KV_FP8>(lanes.full_keys[l] + (size_t)position * row_bytes, value,
                                 kv_heads, head_dim, role, c, kv_amax);
        qsa_kv_store<QSA_KV_FP8>(lanes.full_values[l] + (size_t)position * row_bytes, v,
                                 kv_heads, head_dim, role, c, kv_amax);
        return;
    }
    // ---- index key: BF16 round trip in place + raw-key copy. ----
    __hip_bfloat16* raw_index_keys = lanes.raw_idx[l];
    for (int i = c; i < index_kv_width; i += 256) {
        float v = index_row[index_q_width + i];
        if (isfinite(v)) {
            v = __uint_as_float((__float_as_uint(v) + 0x7FFFu + ((__float_as_uint(v) >> 16) & 1u))
                                & 0xFFFF0000u);
            index_row[index_q_width + i] = v;
        }
        raw_index_keys[(size_t)position * index_kv_width + i] = bf16_round(v);
    }
}

extern "C" __global__ __launch_bounds__(256) void indexed_attention_attention_rot_fp8_lanes(
    int max_rows,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    const float* signs1,
    const float* signs2,
    QsaLanes lanes,
    QsaLanesRot rot) {
    const int l = (int)blockIdx.z / max_rows;
    const int row = (int)blockIdx.z - l * max_rows;
    if (row >= lanes.rows[l]) return;
    const int visible = lanes.pos[l] + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    qsa_attention_row_parallel<QSA_KV_FP8>(
        lanes.qgate[l] + row * 2 * n_heads * head_dim + (int)blockIdx.x * head_dim * 2,
        lanes.full_keys[l], lanes.full_values[l], lanes.selected[l] + row * capacity,
        lanes.output[l] + row * n_heads * head_dim, selected_len, visible, n_heads,
        n_kv_heads, head_dim, full_capacity);
    float* x_rot = rot.x_rot[l];
    if (x_rot == nullptr) return;
    // One head is one 256-value MQ group (head_dim == 256): the head's 256 outputs were
    // stored by this workgroup; wave 0 takes eight contiguous values per lane exactly as
    // mq_rotate_x (x * signs1, FWHT, v * 0.0625f * signs2).
    __syncthreads();
    const int tid = (int)threadIdx.x;
    if (tid >= 32) return;
    const int d0 = tid * 8;
    const float* out = lanes.output[l] + row * n_heads * head_dim + (int)blockIdx.x * head_dim;
    float v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = out[d0 + i] * signs1[d0 + i];
    mq_fwht256_lane(v, tid);
    float* dst = x_rot + row * n_heads * head_dim + (int)blockIdx.x * head_dim;
    const float s = 0.0625f;
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[d0 + i] = v[i] * s * signs2[d0 + i];
}
#undef QSA_LANES_MAX
#endif
