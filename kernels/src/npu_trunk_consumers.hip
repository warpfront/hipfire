// SPDX-License-Identifier: Apache-2.0
// BF16 arena twins; concatenated after tensor_ops.hip, in a separate module.



extern "C" __global__ void gated_delta_params_batched_arena(
    const __hip_bfloat16* a, const __hip_bfloat16* b, const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias, float* gate, float* beta, int rows, int heads,
    int a_pitch, int a_col0, int b_pitch, int b_col0) {
    const int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat < rows * heads)
        gated_delta_params_flat_arena(a, b, a_log, dt_bias, gate, beta, flat, heads,
                                      a_pitch, a_col0, b_pitch, b_col0);
}

static __device__ __forceinline__ float qsa_norm_rope_head_arena(
    const __hip_bfloat16* head_values,
    const __hip_bfloat16* norm,
    int head_dim,
    int position,
    int rotary_dim,
    RopeYarn yarn,
    float* input_row,
    float* row_sum) {
    const int channel = (int)threadIdx.x;
    if (channel < head_dim) {
        input_row[channel] = bf16_to_f32(head_values[channel]);
    }
    __syncthreads();
    // One wave runs the serial i-order sum (identical in every lane) instead
    // of every wave repeating it.
    if (threadIdx.x < 32) {
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            float row_value = input_row[i];
            sum += row_value * row_value;
        }
        if (threadIdx.x == 0) *row_sum = sum;
    }
    __syncthreads();
    if (channel >= head_dim) return 0.0f;
    float inv = rsqrtf(*row_sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)position
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, base, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    return value;
}

// Each head copies its whole head_stride, including the untouched output gate
// or index-key tail. The last head owns the remaining row_stride tail.
extern "C" __global__ void indexed_attention_norm_rope_batch_arena(
    const __hip_bfloat16* src, float* values, const __hip_bfloat16* norm,
    int rows, int row_stride, int heads, int head_dim, int head_stride,
    int position_start, int rotary_dim, int src_pitch, int src_col0, RopeYarn yarn) {
    const int row = (int)blockIdx.z, head = (int)blockIdx.x;
    if (row >= rows || head >= heads || blockIdx.y != 0) return;
    const int start = head * head_stride;
    const __hip_bfloat16* in = src + (size_t)row * src_pitch + src_col0 + start;
    float* out = values + (size_t)row * row_stride + start;
    const int span = head + 1 == heads ? row_stride - start : head_stride;
    for (int c = threadIdx.x + head_dim; c < span; c += blockDim.x)
        out[c] = bf16_to_f32(in[c]);
    __shared__ float input_row[256];
    __shared__ float row_sum;
    const float value = qsa_norm_rope_head_arena(in, norm, head_dim, position_start + row,
                                                rotary_dim, yarn, input_row, &row_sum);
    if ((int)threadIdx.x < head_dim) out[threadIdx.x] = value;
}

static __device__ __forceinline__ float4 gated_delta_gate_head128_arena(
    float* recurrent,
    const __hip_bfloat16* z,
    const __hip_bfloat16* norm,
    float* output,
    int start,
    int lane,
    float* rounded, int z_start) {
    const float4 raw = *reinterpret_cast<const float4*>(recurrent + start + lane * 4);
    const float4 value = make_float4(
        bf16_to_f32(bf16_round(raw.x)), bf16_to_f32(bf16_round(raw.y)),
        bf16_to_f32(bf16_round(raw.z)), bf16_to_f32(bf16_round(raw.w)));
    *reinterpret_cast<float4*>(recurrent + start + lane * 4) = value;
    *reinterpret_cast<float4*>(rounded + lane * 4) = value;
    __syncthreads();

    // `rounded` already holds BF16 values (rounding them again is exact).
    float sum = 0.0f;
    for (int i = 0; i < 128; i += 4) {
        const float4 v = *reinterpret_cast<const float4*>(rounded + i);
        sum += v.x * v.x;
        sum += v.y * v.y;
        sum += v.z * v.z;
        sum += v.w * v.w;
    }
    const float inv = rsqrtf(sum / 128.0f + 1.0e-6f);
    const float4 zv = make_float4(bf16_to_f32(z[z_start + lane * 4]),
        bf16_to_f32(z[z_start + lane * 4 + 1]), bf16_to_f32(z[z_start + lane * 4 + 2]),
        bf16_to_f32(z[z_start + lane * 4 + 3]));
    const float in[4] = {value.x, value.y, value.z, value.w};
    const float zs[4] = {zv.x, zv.y, zv.z, zv.w};
    float out[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int channel = lane * 4 + e;
        const float normalized = bf16_to_f32(bf16_round(
            in[e] * inv * bf16_to_f32(norm[channel])));
        const float gate = bf16_to_f32(bf16_round(zs[e]));
        out[e] = bf16_to_f32(bf16_round(normalized / (1.0f + expf(-gate))));
    }
    const float4 result = make_float4(out[0], out[1], out[2], out[3]);
    *reinterpret_cast<float4*>(output + start + lane * 4) = result;
    return result;
}

extern "C" __global__ __launch_bounds__(32) void gated_delta_gate_batched_arena(
    float* recurrent, const __hip_bfloat16* z, const __hip_bfloat16* norm,
    float* output, int rows, int value_heads, int value_dim, int z_pitch, int z_col0) {
    const int flat_head = (int)blockIdx.x;
    if (flat_head >= rows * value_heads || value_dim != 128) return;
    __shared__ float rounded[128];
    gated_delta_gate_head128_arena(recurrent, z, norm, output, flat_head * 128,
        (int)threadIdx.x, rounded,
        (flat_head / value_heads) * z_pitch + z_col0 + (flat_head % value_heads) * 128);
}

static __device__ __forceinline__ void gated_delta_gate_channel_arena(
    const float* head_values,
    const __hip_bfloat16* z_head,
    const __hip_bfloat16* norm,
    float* out_head,
    int channel,
    int value_dim) {
    float sum = 0.0f;
    for (int i = 0; i < value_dim; ++i) {
        float value = bf16_to_f32(bf16_round(head_values[i]));
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)value_dim + 1.0e-6f);
    // The recurrent output is read through its source BF16 boundary here too,
    // so callers need no separate round trip (a pre-rounded input is unchanged).
    float normalized = bf16_to_f32(bf16_round(
        bf16_to_f32(bf16_round(head_values[channel])) * inv
        * bf16_to_f32(norm[channel])));
    float gate = bf16_to_f32(z_head[channel]);
    out_head[channel] = bf16_to_f32(bf16_round(
        normalized / (1.0f + expf(-gate))));
}

extern "C" __global__ void gated_delta_gate_arena(
    const float* recurrent, const __hip_bfloat16* z, const __hip_bfloat16* norm,
    float* output, int value_heads, int value_dim, int z_pitch, int z_col0) {
    const int head = (int)blockIdx.x;
    const int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= value_heads || channel >= value_dim) return;
    const int start = head * value_dim;
    gated_delta_gate_channel_arena(recurrent + start, z + z_col0 + start, norm,
                                    output + start, channel, value_dim);
}

template <int F>
static __device__ __forceinline__ void qsa_cache_append_batched_arena(
    const void* key,
    const __hip_bfloat16* value,
    unsigned char* full_keys,
    unsigned char* full_values,
    int rows,
    int position_start,
    int kv_heads,
    int head_dim, int key_pitch, int key_col0, int value_pitch, int value_col0, int key_bf16) {
    __shared__ float kv_amax[8];
    const int head = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    const int c = (int)threadIdx.x;
    if (head >= kv_heads || row >= rows) return;
    const int kc = row * key_pitch + key_col0 + head * head_dim + c;
    const float k = c < head_dim ? (key_bf16 ? bf16_to_f32(((const __hip_bfloat16*)key)[kc]) : ((const float*)key)[kc]) : 0.0f;
    const float v = c < head_dim ? bf16_to_f32(value[(size_t)row * value_pitch + value_col0 + head * head_dim + c]) : 0.0f;
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(kv_heads, head_dim);
    const size_t offset = (size_t)(position_start + row) * row_bytes;
    qsa_kv_store<F>(full_keys + offset, k, kv_heads, head_dim, head, c, kv_amax);
    qsa_kv_store<F>(full_values + offset, v, kv_heads, head_dim, head, c, kv_amax);
}

#if defined(QSA_HAS_FP8)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_fp8_batch_arena(
    const void* key, const __hip_bfloat16* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim,
    int key_pitch, int key_col0, int value_pitch, int value_col0, int key_bf16) {
    qsa_cache_append_batched_arena<QSA_KV_FP8>(key, value, full_keys, full_values, rows,
        position_start, kv_heads, head_dim, key_pitch, key_col0, value_pitch, value_col0, key_bf16);
}
#endif

#if defined(QSA_HAS_Q8)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_q8_batch_arena(
    const void* key, const __hip_bfloat16* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim,
    int key_pitch, int key_col0, int value_pitch, int value_col0, int key_bf16) {
    qsa_cache_append_batched_arena<QSA_KV_Q8>(key, value, full_keys, full_values, rows,
        position_start, kv_heads, head_dim, key_pitch, key_col0, value_pitch, value_col0, key_bf16);
}
#endif

// Non-chunk prefill's fused gate + output-projection rotation consumer.
extern "C" __global__ __launch_bounds__(64) void gated_delta_gate_batched_rotate_arena(
    float* recurrent, const __hip_bfloat16* z, const __hip_bfloat16* norm,
    float* output, float* rotated, const float* signs1, const float* signs2,
    int z_pitch, int z_col0, int value_heads) {
    const int wave = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int flat_head = (int)blockIdx.x * 2 + wave;
    __shared__ float rounded[2][128];
    __shared__ float pair[256];
    const float4 result = gated_delta_gate_head128_arena(
        recurrent, z, norm, output, flat_head * 128, lane, rounded[wave],
        (flat_head / value_heads) * z_pitch + z_col0 + (flat_head % value_heads) * 128);
    *reinterpret_cast<float4*>(pair + wave * 128 + lane * 4) = result;
    __syncthreads();
    if (wave != 0) return;
    const int d0 = lane * 8;
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) values[e] = pair[d0 + e] * signs1[d0 + e];
    mq_fwht256_lane(values, lane);
#pragma unroll
    for (int e = 0; e < 8; ++e)
        rotated[(int)blockIdx.x * 256 + d0 + e] = values[e] * 0.0625f * signs2[d0 + e];
}
