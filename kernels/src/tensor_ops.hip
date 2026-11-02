// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q/K input values are source BF16 activations.  The recurrent state and
// projections remain F32, but source l2norm intentionally performs its
// multiply, reduction, rsqrt, and output multiply in BF16 before promotion.
// Keep these helpers explicit so the boundary is not widened by F32 operators.
// The host side owns bounded arenas and passes explicit active lengths; no kernel
// performs host I/O or allocates memory.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_add(
    __hip_bfloat16 lhs,
    __hip_bfloat16 rhs) {
    return bf16_round(bf16_to_f32(lhs) + bf16_to_f32(rhs));
}

// Q8 GDN recurrent state: Qwen3.5's DeltaNet Q8 state format
// (gated_delta_net_q8_fast.hip). Per value head, 128 rows of 128 signed
// bytes (row = value channel, byte = key channel), then, after every head's
// codes, one F32 scale per row (`value_heads * 128`). A row dequantizes as
// scale * code. A token requantizes each row once, after its output, by the
// row's absmax with Qwen3.5's stochastic rounding (floor(x * 127 / absmax
// + u), u uniform in [0, 1) from an LCG seeded by head, row, key channel and
// the token's frame), so repeated decay does not bias the stored state. The
// unarmed persistent batch requantizes only after its last row.
static __device__ __forceinline__ signed char* gdn_q8_row(void* slot, int head, int row) {
    return reinterpret_cast<signed char*>(slot) + ((size_t)head * 128 + row) * 128;
}

static __device__ __forceinline__ float* gdn_q8_scales(void* slot, int value_heads) {
    return reinterpret_cast<float*>(
        reinterpret_cast<signed char*>(slot) + (size_t)value_heads * 128 * 128);
}

static __device__ __forceinline__ size_t gdn_q8_slot_bytes(int value_heads) {
    return (size_t)value_heads * 128 * (128 + sizeof(float));
}

static __device__ __forceinline__ signed char gdn_q8_round(
    float value, float inv, int head, int row, int column, unsigned int frame) {
    unsigned int rng = (unsigned int)head * 1664525u + (unsigned int)row * 1013904223u
        + (unsigned int)column * 214013u + frame * 2531011u;
    rng = rng * 1664525u + 1013904223u;
    const float noise = (float)(rng >> 8) * (1.0f / 16777216.0f);
    return (signed char)fminf(fmaxf(floorf(value * inv + noise), -128.0f), 127.0f);
}

static __device__ __forceinline__ float gdn_q8_inv(float peak) {
    return peak > 0.0f ? 127.0f / peak : 0.0f;
}

static __device__ __forceinline__ float gdn_q8_scale(float peak) {
    return peak > 0.0f ? peak / 127.0f : 1.0f;
}

// Code `b` (0..15, a constant after unrolling) of a 16-byte code word.
static __device__ __forceinline__ float gdn_q8_code(int4 word, int b) {
    const int lane = b < 4 ? word.x : b < 8 ? word.y : b < 12 ? word.z : word.w;
    return (float)(signed char)(lane >> ((b & 3) * 8));
}

// Q8-EF (`q8_ef`) helpers. The recurrent slot stays the plain Q8 layout above;
// a separate live F16 residual plane `ef` (one f16 per code, same element
// index `(head * 128 + row) * 128 + column`) carries the requantization error.
// This module is compiled with the default FP contraction, so every EF
// boundary expression uses explicit non-contracted intrinsics and a
// nearest-even rint: no RNG, no frame, and the residual is added exactly once
// (to the incoming state), never again at the store.
static __device__ __forceinline__ float gdn_q8ef_f16(unsigned int bits) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)bits);
}

// f32 -> f16 bits, round-to-nearest-even (the default rounding mode); overflow
// yields an infinity and NaN stays NaN, so a non-finite row stays visible.
static __device__ __forceinline__ unsigned short gdn_q8ef_f16_bits(float value) {
    return __builtin_bit_cast(unsigned short, (_Float16)value);
}

// Incoming state element: f32(code) * scale + f32(f16 residual), the product
// and the sum each rounded once.
static __device__ __forceinline__ float gdn_q8ef_recon(float code, float scale, float residual) {
    return __fadd_rn(__fmul_rn(code, scale), residual);
}

// Code of an updated state element: round-half-even(value * inv), clamped to
// the signed byte range, as a float holding an integer.
static __device__ __forceinline__ float gdn_q8ef_code(float value, float inv) {
    return fminf(fmaxf(rintf(__fmul_rn(value, inv)), -128.0f), 127.0f);
}

// F16 residual bits against the stored F32 scale and the stored integer code.
static __device__ __forceinline__ unsigned short gdn_q8ef_residual_bits(
    float value, float code, float scale) {
    return gdn_q8ef_f16_bits(__fsub_rn(value, __fmul_rn(code, scale)));
}



// q/k/v are source BF16 activations promoted to the explicit F32 recurrent
// state.  Each value head repeats one q/k head.  The source l2norm call runs
// before that promotion, so its intermediate products, reduction, rsqrt, and
// output are BF16.  Query scale is a separate F32 multiply after promotion.
extern "C" __global__ void gated_delta_step_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale) {
    int value_head = (int)blockIdx.x;
    int value_index = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (value_head >= value_heads || value_index >= value_dim) return;
    int repeat = value_heads / key_heads;
    int key_head = value_head / repeat;
    const int key_base = key_head * key_dim;
    float q_sum_f32 = 0.0f;
    float k_sum_f32 = 0.0f;
    const __hip_bfloat16 eps = bf16_round(1.0e-6f);
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = bf16_round(q[key_base + kd]);
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const float q_float = bf16_to_f32(q_value);
        const float k_float = bf16_to_f32(k_value);
        q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
        k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
    }
    const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
    const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
    const __hip_bfloat16 q_inv = bf16_round(
        rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
    const __hip_bfloat16 k_inv = bf16_round(
        rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
    float decay = expf(gate[value_head]);
    float beta_value = beta[value_head];
    int state_base = value_head * key_dim * value_dim;
    for (int kd = 0; kd < key_dim; ++kd) {
        state[state_base + kd * value_dim + value_index] *= decay;
    }
    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = bf16_round(
            bf16_to_f32(k_value) * bf16_to_f32(k_inv));
        float kv = bf16_to_f32(k_normalized);
        kv_mem += state[state_base + kd * value_dim + value_index] * kv;
    }
    const __hip_bfloat16 v_value = bf16_round(
        v[value_head * value_dim + value_index]);
    float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = bf16_round(
            bf16_to_f32(k_value) * bf16_to_f32(k_inv));
        float kv = bf16_to_f32(k_normalized);
        state[state_base + kd * value_dim + value_index] += kv * delta;
    }
    float out = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = bf16_round(q[key_base + kd]);
        const __hip_bfloat16 q_normalized = bf16_round(
            bf16_to_f32(q_value) * bf16_to_f32(q_inv));
        float qv = bf16_to_f32(q_normalized);
        qv *= query_scale;
        out += state[state_base + kd * value_dim + value_index] * qv;
    }
    output[value_head * value_dim + value_index] = out;
}
// gfx1151-only exact-128 path.  Q/K normalization is shared by every value
// channel in the block, but the recurrence remains one independent row per
// value channel and keeps the original decay -> kv_mem -> delta -> update ->
// output order.
// One decode step of the 128x128 recurrence for block `value_head`; returns
// false for threads outside the exact shape, else writes the thread's
// channel output to `*out`. Shared by the step kernel and the fused
// step + gated-norm kernel below. `kQ8`: `state` is a Q8 GDN state slot
// (see gdn_q8_row) and `frame` seeds its requantization; else F32
// [head][key][value] and `frame` is unused.
template <bool kQ8, bool kEf = false>
static __device__ __forceinline__ bool gated_delta_step_norm128_body(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    void* state,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    unsigned int frame,
    float* out_value,
    unsigned short* ef = nullptr) {
    const int value_head = (int)blockIdx.x;
    const int value_index = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    const bool exact_shape = key_dim == 128 && value_dim == 128;
    const bool active = exact_shape
        && value_head < value_heads
        && value_index < 128;
    __shared__ __hip_bfloat16 q_norm[128];
    __shared__ __hip_bfloat16 k_norm[128];
    __shared__ __hip_bfloat16 q_inv_shared;
    __shared__ __hip_bfloat16 k_inv_shared;
    __shared__ __align__(16) float q_square[128];
    __shared__ __align__(16) float k_square[128];

    // Load the BF16-rounded q/k vectors cooperatively, with each element's
    // BF16-rounded square for the serial sums below.  Threads beyond the
    // exact 128-wide vector remain barrier participants so the old 256-wide
    // launch remains safe for direct callers.
    if (exact_shape && value_head < value_heads && threadIdx.x < 128) {
        const int repeat = value_heads / key_heads;
        const int key_head = value_head / repeat;
        const int key_base = key_head * 128;
        const __hip_bfloat16 q_value = bf16_round(q[key_base + threadIdx.x]);
        const __hip_bfloat16 k_value = bf16_round(k[key_base + threadIdx.x]);
        q_norm[threadIdx.x] = q_value;
        k_norm[threadIdx.x] = k_value;
        const float q_float = bf16_to_f32(q_value);
        const float k_float = bf16_to_f32(k_value);
        q_square[threadIdx.x] = bf16_to_f32(bf16_round(q_float * q_float));
        k_square[threadIdx.x] = bf16_to_f32(bf16_round(k_float * k_float));
    }
    __syncthreads();

    if (threadIdx.x == 0 && value_head < value_heads && exact_shape) {
        float q_sum_f32 = 0.0f;
        float k_sum_f32 = 0.0f;
        const __hip_bfloat16 eps = bf16_round(1.0e-6f);
        // The serial kd-order sums, four LDS squares per read.
#pragma unroll 4
        for (int kd = 0; kd < 128; kd += 4) {
            const float4 qs = *reinterpret_cast<const float4*>(q_square + kd);
            const float4 ks = *reinterpret_cast<const float4*>(k_square + kd);
            q_sum_f32 += qs.x;
            k_sum_f32 += ks.x;
            q_sum_f32 += qs.y;
            k_sum_f32 += ks.y;
            q_sum_f32 += qs.z;
            k_sum_f32 += ks.z;
            q_sum_f32 += qs.w;
            k_sum_f32 += ks.w;
        }
        const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
        const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
        q_inv_shared = bf16_round(
            rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
        k_inv_shared = bf16_round(
            rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        const __hip_bfloat16 q_inv = q_inv_shared;
        const __hip_bfloat16 k_inv = k_inv_shared;
        q_norm[threadIdx.x] = bf16_round(
            bf16_to_f32(q_norm[threadIdx.x]) * bf16_to_f32(q_inv));
        k_norm[threadIdx.x] = bf16_round(
            bf16_to_f32(k_norm[threadIdx.x]) * bf16_to_f32(k_inv));
    }
    __syncthreads();
    if (!active) return false;

    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const float decay = expf(gate[value_head]);
    const float beta_value = beta[value_head];
    const __hip_bfloat16 v_value = bf16_round(
        v[value_head * 128 + value_index]);
    if constexpr (kEf) {
        // Q8-EF: the same three passes as the Q8 branch below, but every pass
        // reconstructs the incoming element as code * scale + f16 residual
        // (explicit roundings), the new row is quantized nearest-even, and the
        // residual against the stored scale goes to the live `ef` plane. The
        // row's code/residual words are read and rewritten only by this thread.
        static_assert(kQ8, "q8_ef needs the Q8 slot layout");
        int4* row = reinterpret_cast<int4*>(gdn_q8_row(state, value_head, value_index));
        float* scale_slot = gdn_q8_scales(state, value_heads) + value_head * 128 + value_index;
        int4* ef_row =
            reinterpret_cast<int4*>(ef + ((size_t)value_head * 128 + value_index) * 128);
        const float scale = *scale_slot;
        float kv_mem = 0.0f;
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
            int4 efw[2] = {ef_row[2 * word], ef_row[2 * word + 1]};
            const unsigned short* efh = reinterpret_cast<const unsigned short*>(efw);
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float s_in =
                    gdn_q8ef_recon(gdn_q8_code(codes, b), scale, gdn_q8ef_f16(efh[b]));
                const float decayed = s_in * decay;
                kv_mem += decayed * kv;
            }
        }
        const float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
        float out = 0.0f;
        float peak = 0.0f;
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
            int4 efw[2] = {ef_row[2 * word], ef_row[2 * word + 1]};
            const unsigned short* efh = reinterpret_cast<const unsigned short*>(efw);
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float s_in =
                    gdn_q8ef_recon(gdn_q8_code(codes, b), scale, gdn_q8ef_f16(efh[b]));
                const float decayed = fmaf(kv, delta, s_in * decay);
                peak = fmaxf(peak, fabsf(decayed));
                float qv = bf16_to_f32(q_norm[kd]);
                qv *= query_scale;
                out += decayed * qv;
            }
        }
        const float inv = gdn_q8_inv(peak);
        const float new_scale = gdn_q8_scale(peak);
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
            int4 efw[2] = {ef_row[2 * word], ef_row[2 * word + 1]};
            const unsigned short* efh = reinterpret_cast<const unsigned short*>(efw);
            int packed[4] = {0, 0, 0, 0};
            int4 efo[2];
            unsigned short* efo_h = reinterpret_cast<unsigned short*>(efo);
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float s_in =
                    gdn_q8ef_recon(gdn_q8_code(codes, b), scale, gdn_q8ef_f16(efh[b]));
                const float decayed = fmaf(kv, delta, s_in * decay);
                const float code = gdn_q8ef_code(decayed, inv);
                packed[b >> 2] |= ((int)(signed char)code & 0xff) << ((b & 3) * 8);
                efo_h[b] = gdn_q8ef_residual_bits(decayed, code, new_scale);
            }
            row[word] = make_int4(packed[0], packed[1], packed[2], packed[3]);
            ef_row[2 * word] = efo[0];
            ef_row[2 * word + 1] = efo[1];
        }
        *scale_slot = new_scale;
        *out_value = out;
        return true;
    }
    if constexpr (kQ8) {
        // The thread's value row: 128 key-channel codes, read 16 at a time
        // (a 16-byte word per step, from cache after the first pass) and
        // dequantized on use; the second and third passes load the next word
        // while they work on the current one. The update is the F32 route's
        // expression on the dequantized state; the row is requantized from the
        // updated values (recomputed, not stored) after the output.
        int4* row = reinterpret_cast<int4*>(gdn_q8_row(state, value_head, value_index));
        float* scale_slot = gdn_q8_scales(state, value_heads) + value_head * 128 + value_index;
        const float scale = *scale_slot;
        float kv_mem = 0.0f;
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float decayed = (scale * gdn_q8_code(codes, b)) * decay;
                kv_mem += decayed * kv;
            }
        }
        const float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
        float out = 0.0f;
        float peak = 0.0f;
        int4 next = row[0];
        for (int word = 0; word < 8; ++word) {
            const int4 codes = next;
            if (word < 7) next = row[word + 1];
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float decayed = fmaf(kv, delta, (scale * gdn_q8_code(codes, b)) * decay);
                peak = fmaxf(peak, fabsf(decayed));
                float qv = bf16_to_f32(q_norm[kd]);
                qv *= query_scale;
                out += decayed * qv;
            }
        }
        const float inv = gdn_q8_inv(peak);
        next = row[0];
        for (int word = 0; word < 8; ++word) {
            const int4 codes = next;
            if (word < 7) next = row[word + 1];
            int packed[4] = {0, 0, 0, 0};
#pragma unroll
            for (int b = 0; b < 16; ++b) {
                const int kd = word * 16 + b;
                const float kv = bf16_to_f32(k_norm[kd]);
                const float decayed = fmaf(kv, delta, (scale * gdn_q8_code(codes, b)) * decay);
                const int code = gdn_q8_round(decayed, inv, value_head, value_index, kd, frame);
                packed[b >> 2] |= (code & 0xff) << ((b & 3) * 8);
            }
            row[word] = make_int4(packed[0], packed[1], packed[2], packed[3]);
        }
        *scale_slot = gdn_q8_scale(peak);
        *out_value = out;
        return true;
    }
    const int state_base = value_head * 128 * 128;
    float* column = reinterpret_cast<float*>(state) + state_base + value_index;
    // Two passes over the state column (read; read + write) instead of four
    // reads and two writes: the decayed value is recomputed, not stored, and
    // the update and output share a pass. Same per-element operations.
    float kv_mem = 0.0f;
    for (int kd = 0; kd < 128; ++kd) {
        const float kv = bf16_to_f32(k_norm[kd]);
        const float decayed = column[kd * 128] * decay;
        kv_mem += decayed * kv;
    }
    const float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
    float out = 0.0f;
    for (int kd = 0; kd < 128; ++kd) {
        const float kv = bf16_to_f32(k_norm[kd]);
        // Explicit fma: the decayed value (stored, then reloaded, in the
        // four-pass form) must not contract into fma(state, decay, kv * delta).
        const float decayed = fmaf(kv, delta, column[kd * 128] * decay);
        column[kd * 128] = decayed;
        float qv = bf16_to_f32(q_norm[kd]);
        qv *= query_scale;
        out += decayed * qv;
    }
    *out_value = out;
    return true;
}

extern "C" __global__ void gated_delta_step_shared_norm128_gfx1151(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    float* state, float* output, int key_heads, int value_heads, int key_dim, int value_dim,
    float query_scale) {
    float out;
    if (gated_delta_step_norm128_body<false>(q, k, v, gate, beta, state, key_heads, value_heads,
                                             key_dim, value_dim, query_scale, 0u, &out))
        output[blockIdx.x * 128 + (blockIdx.y * blockDim.x + threadIdx.x)] = out;
}

// gated_delta_step_shared_norm128_gfx1151 on a Q8 GDN state slot.
extern "C" __global__ void gated_delta_step_norm128_q8(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    signed char* state, float* output, int key_heads, int value_heads, int key_dim,
    int value_dim, float query_scale, unsigned int frame) {
    float out;
    if (gated_delta_step_norm128_body<true>(q, k, v, gate, beta, state, key_heads, value_heads,
                                            key_dim, value_dim, query_scale, frame, &out))
        output[blockIdx.x * 128 + (blockIdx.y * blockDim.x + threadIdx.x)] = out;
}

// gated_delta_step_norm128_q8 with a live f16 error-feedback residual plane `ef`
// (kernargs: the Q8 twin's, `ef` right after `state`, no frame).
extern "C" __global__ void gated_delta_step_norm128_q8ef(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    signed char* state, unsigned short* ef, float* output, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale) {
    float out;
    if (gated_delta_step_norm128_body<true, true>(q, k, v, gate, beta, state, key_heads,
                                                  value_heads, key_dim, value_dim, query_scale,
                                                  0u, &out, ef))
        output[blockIdx.x * 128 + (blockIdx.y * blockDim.x + threadIdx.x)] = out;
}


// Workgroup barrier ordering LDS only: every handoff in the persistent GDN
// step is through LDS, and a full __syncthreads would also wait at each
// barrier for the row's global stores (outputs, captured states) to land.
// Each wave waits for its own LDS ops before it signals: s_wait_dscnt 0 on
// gfx12 (no clang builtin, so the intrinsic is bound by name), lgkmcnt(0) with
// vmcnt/expcnt left open (s_waitcnt 0xFC07) on gfx11. The release fence alone
// adds a soft wait that SIInsertWaitcnts may erase as redundant without seeing
// loop-carried stores (llvm/llvm-project#220357), e.g. q_norm[buf ^ 1] stored
// at a row's tail and read after the next row's first barrier; in WGP mode a
// read past the barrier could then overtake another wave's queued store (the
// gfx1151 VerifyAttn race). Other arches (no persistent route) keep the fence.
#if defined(__GFX12__)
extern "C" __device__ void gdn_lds_wait_dscnt(unsigned short) __asm("llvm.amdgcn.s.wait.dscnt");
#endif
__device__ __forceinline__ void gdn_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
#if defined(__GFX12__)
    gdn_lds_wait_dscnt(0);
#elif defined(__GFX11__)
    __builtin_amdgcn_s_waitcnt(0xFC07);
#endif
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The absmax of a Q8 state row the persistent GDN body holds as two 64-key
// halves (threads t and t + 128): each half's peak is exchanged through LDS,
// then both take the max of the two, so both halves see the same row peak.
// fmaxf is exact, so the result equals a serial absmax over all 128 keys.
static __device__ __forceinline__ float gdn_q8_halves_peak(
    const float (&state_half)[64], int half, int value_index) {
    __shared__ float peak_half[2][128];
    float peak = 0.0f;
    #pragma unroll
    for (int kd = 0; kd < 64; ++kd) peak = fmaxf(peak, fabsf(state_half[kd]));
    peak_half[half][value_index] = peak;
    gdn_lds_barrier();
    return fmaxf(peak_half[0][value_index], peak_half[1][value_index]);
}

// Wave-size-independent persistent row-batched path for exact 128x128 geometry.
// A block owns one value head and 256 threads.  tid%128 is the output channel
// and tid/128 selects the half; each thread retains 64 state values in
// registers.  Shared handoffs serialize the kd=0..127 kv and output chains:
// half 1 continues half 0's accumulator rather than combining partial sums.
//
// Nothing but the state recurrence is serial across rows, so the per-row Q/K
// norm factors, decay and beta of up to 256 rows are computed up front (one
// row per thread, each with the exact serial sum the per-row path uses) and
// the next row's raw q/k/v are prefetched while the current row runs.  The
// normalized q/k vectors are double-buffered so a row needs no trailing
// barrier.  Every value is produced by the same expression sequence as
// gated_delta_step_f32, so the result is bitwise unchanged.
// `kCapture`: leave the state after the last row in `row_states` slot
// `rows - 1` instead of updating `state` (speculative verify: the
// pre-verify state stays intact, and a rollback re-runs the kept rows from
// it); a separate instantiation so the plain route's register allocation is
// unchanged.
// `kQ8`: `state` and the `row_states` slots are Q8 GDN state slots (see
// gdn_q8_row) and `frame` (the last row's position) seeds the final
// requantization; the recurrence runs on the dequantized F32 values. With
// `kCapture` too (verify and its rollback), every earlier row also
// requantizes at its own position, as single-row decode does, so the window
// is token-exact with AR; the unarmed batch keeps one requantization.
// `kTokenQ8` (default `kCapture`): every row requantizes at its own position
// as described above. The `canon_q8` entry sets it without `kCapture`: the
// final state is stored to `state` (not a ring slot), so the result is the
// token-canonical recurrence, independent of how rows split into launches.
template <bool kCapture, bool kQ8 = false, bool kTokenQ8 = kCapture, bool kEf = false>
__device__ __forceinline__ void gated_delta_step_persistent256_body(
    const float* projection,
    const float* gate,
    const float* beta,
    void* state,
    float* output,
    void* row_states,
    int rows,
    int qkv_width,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    unsigned int frame = 0u,
    unsigned short* ef = nullptr) {
    const int value_head = (int)blockIdx.x;
    const int thread = (int)threadIdx.x;
    const int value_index = thread & 127;
    const int half = thread >> 7;
    const bool exact_shape = key_dim == 128 && value_dim == 128;
    if (!exact_shape || value_head >= value_heads) return;

    const int qk_width = key_heads * 128;
    const int value_width = value_heads * 128;
    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const int kd_base = half * 64;
    const int state_base = value_head * 128 * 128;
    const int q_offset = key_head * 128;
    const int k_offset = qk_width + key_head * 128;
    const int v_offset = 2 * qk_width + value_head * 128;

    __shared__ __hip_bfloat16 q_norm[2][128];
    __shared__ __hip_bfloat16 k_norm[2][128];
    __shared__ __hip_bfloat16 q_inv_rows[256];
    __shared__ __hip_bfloat16 k_inv_rows[256];
    __shared__ float decay_rows[256];
    __shared__ float beta_rows[256];
    __shared__ float handoff_kv[128];
    __shared__ float handoff_delta[128];
    __shared__ float handoff_out[128];

    float state_half[64];
    if constexpr (kQ8) {
        const int4* codes4 =
            reinterpret_cast<const int4*>(gdn_q8_row(state, value_head, value_index) + kd_base);
        const float scale =
            gdn_q8_scales(state, value_heads)[value_head * 128 + value_index];
        int4 packed[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) packed[i] = codes4[i];
        const signed char* codes = reinterpret_cast<const signed char*>(packed);
        if constexpr (kEf) {
            // Q8-EF: code * scale + the live f16 residual, once.
            const int4* ef4 = reinterpret_cast<const int4*>(
                ef + ((size_t)value_head * 128 + value_index) * 128 + kd_base);
            int4 efw[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) efw[i] = ef4[i];
            const unsigned short* efh = reinterpret_cast<const unsigned short*>(efw);
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                state_half[kd] = gdn_q8ef_recon((float)codes[kd], scale, gdn_q8ef_f16(efh[kd]));
            }
        } else {
        #pragma unroll
        for (int kd = 0; kd < 64; ++kd) state_half[kd] = scale * (float)codes[kd];
        }
    } else {
        #pragma unroll
        for (int kd = 0; kd < 64; ++kd) {
            state_half[kd] = reinterpret_cast<const float*>(
                state)[state_base + (kd_base + kd) * 128 + value_index];
        }
    }

    // Half 0 carries q/k channel value_index, half 1 carries v.
    float raw_a = 0.0f;
    float raw_b = 0.0f;
    if (half == 0) {
        raw_a = projection[q_offset + value_index];
        raw_b = projection[k_offset + value_index];
    } else {
        raw_a = projection[v_offset + value_index];
    }

    for (int block = 0; block < rows; block += 256) {
        const int block_rows = min(256, rows - block);
        gdn_lds_barrier();
        if (thread < block_rows) {
            const int row = block + thread;
            const int row_base = row * qkv_width;
            float q_sum_f32 = 0.0f;
            float k_sum_f32 = 0.0f;
            const __hip_bfloat16 eps = bf16_round(1.0e-6f);
            // float4 loads, the next pair issued before the current one is
            // consumed (a scalar load per kd waited a full memory round trip
            // each); the kd-order sums are unchanged.  More loads in flight
            // cost registers the live state needs.
            const float4* q_row =
                reinterpret_cast<const float4*>(projection + row_base + q_offset);
            const float4* k_row =
                reinterpret_cast<const float4*>(projection + row_base + k_offset);
            float4 qn = q_row[0], kn = k_row[0];
            for (int c = 0; c < 32; ++c) {
                const float4 qc = qn, kc = kn;
                if (c + 1 < 32) { qn = q_row[c + 1]; kn = k_row[c + 1]; }
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float q_float = bf16_to_f32(bf16_round((&qc.x)[u]));
                    const float k_float = bf16_to_f32(bf16_round((&kc.x)[u]));
                    q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
                    k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
                }
            }
            const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
            const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
            q_inv_rows[thread] = bf16_round(
                rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
            k_inv_rows[thread] = bf16_round(
                rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
            decay_rows[thread] = expf(gate[row * value_heads + value_head]);
            beta_rows[thread] = beta[row * value_heads + value_head];
        }
        gdn_lds_barrier();

        for (int r = 0; r < block_rows; ++r) {
            const int row = block + r;
            const int buf = r & 1;
            const float cur_a = raw_a;
            const float cur_b = raw_b;
            if (row + 1 < rows) {
                const int next_base = (row + 1) * qkv_width;
                if (half == 0) {
                    raw_a = projection[next_base + q_offset + value_index];
                    raw_b = projection[next_base + k_offset + value_index];
                } else {
                    raw_a = projection[next_base + v_offset + value_index];
                }
            }
            // The first row of a parameter block stages its own normalized
            // q/k; every later row's were staged by half 0 at the end of the
            // previous row, behind that row's last barrier.
            if (r == 0) {
                if (half == 0) {
                    q_norm[buf][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(cur_a)) * bf16_to_f32(q_inv_rows[r]));
                    k_norm[buf][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(cur_b)) * bf16_to_f32(k_inv_rows[r]));
                }
                gdn_lds_barrier();
            }

            const float decay = decay_rows[r];
            const float beta_value = beta_rows[r];

            // gfx942 F32 (!kQ8): pin the recurrence to the sequential kernel's
            // exact arithmetic. The sequential path rounds state*decay through
            // its global state store, so the decay product is rounded here with
            // __fmul_rn (register-resident state would otherwise let the
            // compiler contract it into the following FMAs). The kv_mem and
            // out accumulators stay serial __fmaf_rn chains in kd order
            // (half 0 then half 1 via the LDS handoff); delta, the state
            // update and the qv scaling use the same explicit roundings.
            // Q8 and every other architecture keep the original expressions.
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
#if defined(__gfx942__)
                if constexpr (!kQ8) state_half[kd] = __fmul_rn(state_half[kd], decay); else
#endif
                state_half[kd] *= decay;
            }

            float kv_mem = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
#if defined(__gfx942__)
                if constexpr (!kQ8) kv_mem = __fmaf_rn(state_half[kd], kv, kv_mem); else
#endif
                kv_mem += state_half[kd] * kv;
            }
            if (half == 0) {
                handoff_kv[value_index] = kv_mem;
            }
            gdn_lds_barrier();

            if (half == 1) {
                kv_mem = handoff_kv[value_index];
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) {
                    const float kv = bf16_to_f32(k_norm[buf][64 + kd]);
#if defined(__gfx942__)
                    if constexpr (!kQ8) kv_mem = __fmaf_rn(state_half[kd], kv, kv_mem); else
#endif
                    kv_mem += state_half[kd] * kv;
                }
                const __hip_bfloat16 v_value = bf16_round(cur_a);
#if defined(__gfx942__)
                const float delta = kQ8
                    ? (bf16_to_f32(v_value) - kv_mem) * beta_value
                    : __fmul_rn(__fsub_rn(bf16_to_f32(v_value), kv_mem), beta_value);
#else
                const float delta =
                    (bf16_to_f32(v_value) - kv_mem) * beta_value;
#endif
                handoff_delta[value_index] = delta;
            }
            gdn_lds_barrier();

            const float delta = handoff_delta[value_index];
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
#if defined(__gfx942__)
                if constexpr (!kQ8) state_half[kd] = __fmaf_rn(kv, delta, state_half[kd]); else
#endif
                state_half[kd] += kv * delta;
            }

            float out = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                float qv = bf16_to_f32(q_norm[buf][kd_base + kd]);
#if defined(__gfx942__)
                if constexpr (!kQ8) {
                    qv = __fmul_rn(qv, query_scale);
                    out = __fmaf_rn(state_half[kd], qv, out);
                } else {
#endif
                qv *= query_scale;
                out += state_half[kd] * qv;
#if defined(__gfx942__)
                }
#endif
            }
            if (half == 0) {
                handoff_out[value_index] = out;
                if (r + 1 < block_rows) {
                    q_norm[buf ^ 1][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(raw_a)) * bf16_to_f32(q_inv_rows[r + 1]));
                    k_norm[buf ^ 1][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(raw_b)) * bf16_to_f32(k_inv_rows[r + 1]));
                }
            }
            gdn_lds_barrier();

            if (half == 1) {
                out = handoff_out[value_index];
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) {
                    float qv = bf16_to_f32(q_norm[buf][64 + kd]);
#if defined(__gfx942__)
                    if constexpr (!kQ8) {
                        qv = __fmul_rn(qv, query_scale);
                        out = __fmaf_rn(state_half[kd], qv, out);
                    } else {
#endif
                    qv *= query_scale;
                    out += state_half[kd] * qv;
#if defined(__gfx942__)
                    }
#endif
                }
                output[row * value_width + value_head * 128 + value_index] = out;
            }
            // Verify capture on a Q8 state: gated_delta_step_norm128_body<true>'s
            // per-token boundary. The row's output above used its updated F32
            // state; the next row runs on that state requantized at the row's
            // own frame and dequantized (scale * code, the next decode's read).
            // The last row's requantization is the final store below.
            if constexpr (kTokenQ8 && kQ8) {
                if (row + 1 < rows) {
                    const float peak = gdn_q8_halves_peak(state_half, half, value_index);
                    const float inv = gdn_q8_inv(peak);
                    const float scale = gdn_q8_scale(peak);
                    const unsigned int row_frame = frame - (unsigned int)(rows - 1 - row);
                    if constexpr (kEf) {
                        // Q8-EF: the next row's S_in is rebuilt in registers from
                        // the new code, the new scale and the f16-rounded residual.
                        #pragma unroll
                        for (int kd = 0; kd < 64; ++kd) {
                            const float code = gdn_q8ef_code(state_half[kd], inv);
                            const float residual = gdn_q8ef_f16(
                                gdn_q8ef_residual_bits(state_half[kd], code, scale));
                            state_half[kd] = gdn_q8ef_recon(code, scale, residual);
                        }
                    } else {
                    #pragma unroll
                    for (int kd = 0; kd < 64; ++kd) {
                        state_half[kd] = scale * (float)gdn_q8_round(
                            state_half[kd], inv, value_head, value_index, kd_base + kd,
                            row_frame);
                    }
                    }
                }
            }
        }
    }

    if constexpr (kQ8) {
        const float peak = gdn_q8_halves_peak(state_half, half, value_index);
        const float inv = gdn_q8_inv(peak);
        void* const final_slot = kCapture
            ? reinterpret_cast<signed char*>(row_states)
                  + (size_t)(rows - 1) * gdn_q8_slot_bytes(value_heads)
            : state;
        int4 packed[4];
        signed char* codes = reinterpret_cast<signed char*>(packed);
        int4 efp[kEf ? 8 : 1];
        unsigned short* efh = reinterpret_cast<unsigned short*>(efp);
        if constexpr (kEf) {
            // Q8-EF: fresh codes, and the residual against the stored scale; the
            // old residual already entered the state and is not added again.
            const float new_scale = gdn_q8_scale(peak);
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float code = gdn_q8ef_code(state_half[kd], inv);
                codes[kd] = (signed char)code;
                efh[kd] = gdn_q8ef_residual_bits(state_half[kd], code, new_scale);
            }
        } else {
        #pragma unroll
        for (int kd = 0; kd < 64; ++kd) {
            codes[kd] = gdn_q8_round(state_half[kd], inv, value_head, value_index,
                                     kd_base + kd, frame);
        }
        }
        int4* dst =
            reinterpret_cast<int4*>(gdn_q8_row(final_slot, value_head, value_index) + kd_base);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = packed[i];
        if constexpr (kEf) {
            // The final residual goes to the live plane, also when capturing: a
            // ring slot holds Q8 codes and scales only.
            int4* ef_dst = reinterpret_cast<int4*>(
                ef + ((size_t)value_head * 128 + value_index) * 128 + kd_base);
            #pragma unroll
            for (int i = 0; i < 8; ++i) ef_dst[i] = efp[i];
        }
        if (half == 0) {
            gdn_q8_scales(final_slot, value_heads)[value_head * 128 + value_index] =
                gdn_q8_scale(peak);
        }
        return;
    }
    float* const final_state = kCapture
        ? reinterpret_cast<float*>(row_states) + (size_t)(rows - 1) * value_heads * 128 * 128
        : reinterpret_cast<float*>(state);
    #pragma unroll
    for (int kd = 0; kd < 64; ++kd) {
        final_state[state_base + (kd_base + kd) * 128 + value_index] =
            state_half[kd];
    }
}

__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_f32(
    const float* projection, const float* gate, const float* beta, float* state,
    float* output, int rows, int qkv_width, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<false>(projection, gate, beta, state, output, nullptr,
        rows, qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale);
}

__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_capture_f32(
    const float* projection, const float* gate, const float* beta, float* state,
    float* output, float* row_states, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<true>(projection, gate, beta, state, output, row_states,
        rows, qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale);
}

// The two persistent kernels above on Q8 GDN state slots (`row_states`: a
// ring of slots); `frame`, the last row's position, seeds the final
// requantization, and the capture's row `r` of `rows` requantizes at
// `frame - (rows - 1) + r` (single-row decode's per-token boundary).
__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_q8(
    const float* projection, const float* gate, const float* beta, signed char* state,
    float* output, int rows, int qkv_width, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale, unsigned int frame) {
    gated_delta_step_persistent256_body<false, true>(projection, gate, beta, state, output,
        nullptr, rows, qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale,
        frame);
}

__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_capture_q8(
    const float* projection, const float* gate, const float* beta, signed char* state,
    float* output, signed char* row_states, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale, unsigned int frame) {
    gated_delta_step_persistent256_body<true, true>(projection, gate, beta, state, output,
        row_states, rows, qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale,
        frame);
}

// Token-canonical Q8 prefill recurrence (`HIPFIRE_QWEN4_GDN_PREFILL=canon`):
// the plain `_q8` signature and final store (`state`, no row ring), with the
// capture kernel's per-row requantization at `frame - (rows - 1 - row)`.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_canon_q8(
    const float* projection, const float* gate, const float* beta, signed char* state,
    float* output, int rows, int qkv_width, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale, unsigned int frame) {
    gated_delta_step_persistent256_body<false, true, true>(projection, gate, beta, state,
        output, nullptr, rows, qkv_width, key_heads, value_heads, key_dim, value_dim,
        query_scale, frame);
}

// The three persistent kernels above with a live f16 error-feedback residual
// plane `ef` (Q8 twin's kernargs, `ef` right after `state`, no frame). Capture
// reads the live slot + `ef`, writes the final Q8 slot to ring slot rows-1 and
// the final residual to `ef`; the live Q8 slot is never written.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_q8ef(
    const float* projection, const float* gate, const float* beta, signed char* state,
    unsigned short* ef, float* output, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<false, true, false, true>(projection, gate, beta,
        state, output, nullptr, rows, qkv_width, key_heads, value_heads, key_dim, value_dim,
        query_scale, 0u, ef);
}

__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_capture_q8ef(
    const float* projection, const float* gate, const float* beta, signed char* state,
    unsigned short* ef, float* output, signed char* row_states, int rows, int qkv_width,
    int key_heads, int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<true, true, true, true>(projection, gate, beta,
        state, output, row_states, rows, qkv_width, key_heads, value_heads, key_dim,
        value_dim, query_scale, 0u, ef);
}

__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_canon_q8ef(
    const float* projection, const float* gate, const float* beta, signed char* state,
    unsigned short* ef, float* output, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<false, true, true, true>(projection, gate, beta,
        state, output, nullptr, rows, qkv_width, key_heads, value_heads, key_dim, value_dim,
        query_scale, 0u, ef);
}

#if defined(__GFX12__) || defined(__gfx942__)
// Per-row norm factors, decay and beta of the persistent GDN recurrence: the
// exact expression sequence of gated_delta_step_persistent256_body's
// parameter pass (one row, the serial kd-order sums the per-row path uses).
static __device__ __forceinline__ void gdn_pipe128_row_params(
    const float* projection, const float* gate, const float* beta, int row, int qkv_width,
    int q_offset, int k_offset, int value_heads, int value_head,
    __hip_bfloat16& q_inv, __hip_bfloat16& k_inv, float& decay, float& beta_value) {
    const int row_base = row * qkv_width;
    float q_sum_f32 = 0.0f;
    float k_sum_f32 = 0.0f;
    const __hip_bfloat16 eps = bf16_round(1.0e-6f);
    const float4* q_row = reinterpret_cast<const float4*>(projection + row_base + q_offset);
    const float4* k_row = reinterpret_cast<const float4*>(projection + row_base + k_offset);
    float4 qn = q_row[0], kn = k_row[0];
    for (int c = 0; c < 32; ++c) {
        const float4 qc = qn, kc = kn;
        if (c + 1 < 32) { qn = q_row[c + 1]; kn = k_row[c + 1]; }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const float q_float = bf16_to_f32(bf16_round((&qc.x)[u]));
            const float k_float = bf16_to_f32(bf16_round((&kc.x)[u]));
            q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
            k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
        }
    }
    const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
    const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
    q_inv = bf16_round(rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
    k_inv = bf16_round(rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
    decay = expf(gate[row * value_heads + value_head]);
    beta_value = beta[row * value_heads + value_head];
}

// A per-lane zero the compiler cannot see through.  The q/k slots are read at
// a wave-uniform address, so the compiler would treat the loaded LDS data as
// uniform and move it through v_readfirstlane into SALU math (a VALU -> SALU
// -> VALU round trip per kd step).  Adding this to the slot address makes the
// loads divergent, so they stay ds_load_b128 into VGPRs feeding v_fmac.
static __device__ __forceinline__ int gdn_pipe128_lane_zero() {
    int zero;
    __asm__ volatile("v_mov_b32 %0, 0" : "=v"(zero));
    return zero;
}

// The kd loops of the pipe128 body.  A slot holds 128 floats read as float4,
// consumed in groups of kGdnPipeGroup float4 with the next group's loads
// issued before the current group's FMAs.  Left alone, the compiler hoists
// the whole row's loads and the decay multiplies of the fused loop above the
// out chain (the longest dependence chain), which keeps the old and the new
// state live together and spills.  The empty volatile asm statements carry
// the chain values through each element, so loads and arithmetic stay in
// source order and the window of live loads stays one group; they emit no
// instruction and change no value.  Per-element expressions and chain order
// are exactly the scalar kd = 0..127 loops of the original.
constexpr int kGdnPipeGroup = 4;
constexpr int kGdnPipeGroups = 32 / kGdnPipeGroup;

// out += s[kd] * slot[kd] over kd = 0..127.
static __device__ __forceinline__ void gdn_pipe128_dot(
    const float (&s)[128], const float* slot, int lane_zero, float& out) {
    const float4* p = reinterpret_cast<const float4*>(slot + lane_zero);
    float4 cur[kGdnPipeGroup];
    #pragma unroll
    for (int e = 0; e < kGdnPipeGroup; ++e) cur[e] = p[e];
    #pragma unroll
    for (int g = 0; g < kGdnPipeGroups; ++g) {
        float4 next[kGdnPipeGroup];
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) next[e] = p[(g + 1) * kGdnPipeGroup + e];
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int e = 0; e < kGdnPipeGroup; ++e) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
#if defined(__gfx942__)
                out = __fmaf_rn(s[4 * (g * kGdnPipeGroup + e) + u], (&cur[e].x)[u], out);
#else
                out += s[4 * (g * kGdnPipeGroup + e) + u] * (&cur[e].x)[u];
#endif
            }
            __asm__ volatile("" : "+v"(out));
        }
        __builtin_amdgcn_sched_barrier(0);
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) cur[e] = next[e];
        }
    }
}

// s[kd] += slot[kd] * delta over kd = 0..127.
static __device__ __forceinline__ void gdn_pipe128_update(
    float (&s)[128], const float* slot, int lane_zero, float delta) {
    const float4* p = reinterpret_cast<const float4*>(slot + lane_zero);
    float4 cur[kGdnPipeGroup];
    #pragma unroll
    for (int e = 0; e < kGdnPipeGroup; ++e) cur[e] = p[e];
    #pragma unroll
    for (int g = 0; g < kGdnPipeGroups; ++g) {
        float4 next[kGdnPipeGroup];
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) next[e] = p[(g + 1) * kGdnPipeGroup + e];
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int e = 0; e < kGdnPipeGroup; ++e) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
#if defined(__gfx942__)
                s[4 * (g * kGdnPipeGroup + e) + u] = __fmaf_rn(
                    (&cur[e].x)[u], delta, s[4 * (g * kGdnPipeGroup + e) + u]);
#else
                s[4 * (g * kGdnPipeGroup + e) + u] += (&cur[e].x)[u] * delta;
#endif
                __asm__ volatile("" : "+v"(s[4 * (g * kGdnPipeGroup + e) + u]));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) cur[e] = next[e];
        }
    }
}

// One pass over kd = 0..127: out += s[kd] * q[kd]; s[kd] *= decay;
// kv += s[kd] * k[kd]  (the out chain of row i, decay and the kv chain of row i + 1).
static __device__ __forceinline__ void gdn_pipe128_out_decay_kv(
    float (&s)[128], const float* qslot, const float* kslot, int lane_zero, float decay,
    float& out, float& kv_mem) {
    const float4* qp = reinterpret_cast<const float4*>(qslot + lane_zero);
    const float4* kp = reinterpret_cast<const float4*>(kslot + lane_zero);
    float4 qcur[kGdnPipeGroup], kcur[kGdnPipeGroup];
    #pragma unroll
    for (int e = 0; e < kGdnPipeGroup; ++e) {
        qcur[e] = qp[e];
        kcur[e] = kp[e];
    }
    #pragma unroll
    for (int g = 0; g < kGdnPipeGroups; ++g) {
        float4 qnext[kGdnPipeGroup], knext[kGdnPipeGroup];
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) {
                qnext[e] = qp[(g + 1) * kGdnPipeGroup + e];
                knext[e] = kp[(g + 1) * kGdnPipeGroup + e];
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for (int e = 0; e < kGdnPipeGroup; ++e) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int kd = 4 * (g * kGdnPipeGroup + e) + u;
#if defined(__gfx942__)
                out = __fmaf_rn(s[kd], (&qcur[e].x)[u], out);
                s[kd] = __fmul_rn(s[kd], decay);
                kv_mem = __fmaf_rn(s[kd], (&kcur[e].x)[u], kv_mem);
#else
                out += s[kd] * (&qcur[e].x)[u];
                s[kd] *= decay;
                kv_mem += s[kd] * (&kcur[e].x)[u];
#endif
                // Orders out(kd) -> decay(kd) -> kv(kd) -> kd + 1 (see above).
                __asm__ volatile("" : "+v"(s[kd]), "+v"(out), "+v"(kv_mem));
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        if (g + 1 < kGdnPipeGroups) {
            #pragma unroll
            for (int e = 0; e < kGdnPipeGroup; ++e) {
                qcur[e] = qnext[e];
                kcur[e] = knext[e];
            }
        }
    }
}

// gfx12 / gfx942 (F32 only) prefill GDN recurrence, exact 128x128 geometry: one value head per
// 128-thread block, thread c owns value column c and keeps all 128 state
// values in registers, so a row needs no cross-thread handoff of kv, delta or
// out.  The per-row chains are the same serial kd = 0..127 chains as
// gated_delta_step_persistent256_body (that kernel splits them over two
// 64-key halves and passes the accumulator between the halves; here one
// thread runs the whole chain).  The recurrence is software-pipelined: the
// loop body for row i computes row i's output chain together with row i+1's
// decay and kv chain (they are independent serial chains, interleaved per kd
// for ILP), then row i+1's delta and state update.  Every per-element
// expression and its operand order is the original's, so state and outputs
// are bitwise identical.  Nothing here depends on the wave size (no shuffles,
// ballots or lane arithmetic; the LDS ring is synchronized by the block
// barrier), so the 128-thread block is four waves on gfx12 (wave32) and two
// on gfx942 (wave64) alike.  Under __gfx942__ every F32 expression is the
// pinned __fmul_rn / __fmaf_rn / __fsub_rn form of persistent256's gfx942
// branch (same operand order, serial kd chains), because the register state
// would otherwise let the compiler contract the decay product.
template <bool kQ8, bool kEf = false>
__device__ __forceinline__ void gated_delta_step_pipe128_body(
    const float* projection,
    const float* gate,
    const float* beta,
    void* state,
    float* output,
    int rows,
    int qkv_width,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    unsigned int frame = 0u,
    unsigned short* ef = nullptr) {
    const int value_head = (int)blockIdx.x;
    const int c = (int)threadIdx.x;
    const int lane_zero = gdn_pipe128_lane_zero();
    const bool exact_shape = key_dim == 128 && value_dim == 128;
    if (!exact_shape || value_head >= value_heads) return;

    const int qk_width = key_heads * 128;
    const int value_width = value_heads * 128;
    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const int state_base = value_head * 128 * 128;
    const int q_offset = key_head * 128;
    const int k_offset = qk_width + key_head * 128;
    const int v_offset = 2 * qk_width + value_head * 128;

    // Per-row parameters of 256-row blocks, double-buffered by block parity.
    __shared__ __hip_bfloat16 q_inv_rows[2][256];
    __shared__ __hip_bfloat16 k_inv_rows[2][256];
    __shared__ float decay_rows[2][256];
    __shared__ float beta_rows[2][256];
    // Normalized, widened q (already times query_scale) and k of the rows in
    // flight: a 3-slot ring (see the loop).  Float slots: the kd loops read
    // them as float4, with no per-element bf16 widening or scaling.
    __shared__ __attribute__((aligned(16))) float qs[3][128];
    __shared__ __attribute__((aligned(16))) float kf[3][128];

    float s[128];
    if constexpr (kQ8) {
        const int4* codes4 =
            reinterpret_cast<const int4*>(gdn_q8_row(state, value_head, c));
        const float scale = gdn_q8_scales(state, value_heads)[value_head * 128 + c];
        int4 packed[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) packed[i] = codes4[i];
        const signed char* codes = reinterpret_cast<const signed char*>(packed);
        if constexpr (kEf) {
            // Q8-EF: code * scale + the live f16 residual, once.
            const int4* ef4 =
                reinterpret_cast<const int4*>(ef + ((size_t)value_head * 128 + c) * 128);
            #pragma unroll
            for (int g = 0; g < 16; ++g) {
                const int4 efw = ef4[g];
                const unsigned short* efh = reinterpret_cast<const unsigned short*>(&efw);
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    s[g * 8 + e] = gdn_q8ef_recon(
                        (float)codes[g * 8 + e], scale, gdn_q8ef_f16(efh[e]));
                }
            }
        } else {
        #pragma unroll
        for (int kd = 0; kd < 128; ++kd) s[kd] = scale * (float)codes[kd];
        }
    } else {
        #pragma unroll
        for (int kd = 0; kd < 128; ++kd) {
            s[kd] = reinterpret_cast<const float*>(state)[state_base + kd * 128 + c];
        }
    }

    // Raw q/k/v of this thread's channel c for the row about to be staged.
    float raw_q = projection[q_offset + c];
    float raw_k = projection[k_offset + c];
    float raw_v = projection[v_offset + c];

    // Parameters of block 0: thread c takes rows c and c + 128.
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = c + h * 128;
        if (row < rows) {
            __hip_bfloat16 q_inv, k_inv;
            float decay, beta_value;
            gdn_pipe128_row_params(projection, gate, beta, row, qkv_width, q_offset, k_offset,
                                   value_heads, value_head, q_inv, k_inv, decay, beta_value);
            q_inv_rows[0][row] = q_inv;
            k_inv_rows[0][row] = k_inv;
            decay_rows[0][row] = decay;
            beta_rows[0][row] = beta_value;
        }
    }
    gdn_lds_barrier();

    // Row 0: stage, then decay + kv + delta + update on the loaded state.
#if defined(__gfx942__)
    qs[0][c] = __fmul_rn(bf16_to_f32(bf16_round(
        bf16_to_f32(bf16_round(raw_q)) * bf16_to_f32(q_inv_rows[0][0]))), query_scale);
#else
    qs[0][c] = bf16_to_f32(bf16_round(
        bf16_to_f32(bf16_round(raw_q)) * bf16_to_f32(q_inv_rows[0][0]))) * query_scale;
#endif
    kf[0][c] = bf16_to_f32(bf16_round(
        bf16_to_f32(bf16_round(raw_k)) * bf16_to_f32(k_inv_rows[0][0])));
    const float v0 = raw_v;
    if (rows > 1) {
        const int next_base = qkv_width;
        raw_q = projection[next_base + q_offset + c];
        raw_k = projection[next_base + k_offset + c];
        raw_v = projection[next_base + v_offset + c];
    }
    gdn_lds_barrier();
    {
        const float decay = decay_rows[0][0];
        const float beta_value = beta_rows[0][0];
        #pragma unroll
        for (int kd = 0; kd < 128; ++kd) {
#if defined(__gfx942__)
            s[kd] = __fmul_rn(s[kd], decay);
#else
            s[kd] *= decay;
#endif
        }
        float kv_mem = 0.0f;
        gdn_pipe128_dot(s, &kf[0][0], lane_zero, kv_mem);
#if defined(__gfx942__)
        const float delta =
            __fmul_rn(__fsub_rn(bf16_to_f32(bf16_round(v0)), kv_mem), beta_value);
#else
        const float delta = (bf16_to_f32(bf16_round(v0)) - kv_mem) * beta_value;
#endif
        gdn_pipe128_update(s, &kf[0][0], lane_zero, delta);
    }

    // Slot ring: row i's q is in qs[cur], row i+1's q/k are staged into
    // qs/kf[nxt] at the top of iteration i.  Iteration i reads qs[cur] and
    // kf[nxt] after its one barrier and writes qs/kf[nxt] before it.  A wave
    // that has crossed iteration i's barrier knows every wave has finished
    // iteration i-1's pre-barrier work, and its own iteration-i writes land
    // in slot (i+1)%3.  The slowest wave can still be reading iteration
    // i-1's slots (qs[(i-1)%3], kf[i%3]) or iteration i's, which differ
    // from (i+1)%3; with only two slots (i+1)%2 == (i-1)%2 and a fast wave
    // would overwrite q still being read.  The slots iteration i reads were
    // written before its barrier (qs[cur] in iteration i-1, before that
    // iteration's barrier; kf[nxt] in this iteration).
    int cur = 0;
    int nxt = 1;
    for (int i = 0; i + 1 < rows; ++i) {
        const int row = i + 1;
        const int pbuf = (row >> 8) & 1;
        const int pidx = row & 255;
#if defined(__gfx942__)
        qs[nxt][c] = __fmul_rn(bf16_to_f32(bf16_round(
            bf16_to_f32(bf16_round(raw_q)) * bf16_to_f32(q_inv_rows[pbuf][pidx]))),
            query_scale);
#else
        qs[nxt][c] = bf16_to_f32(bf16_round(
            bf16_to_f32(bf16_round(raw_q)) * bf16_to_f32(q_inv_rows[pbuf][pidx])))
            * query_scale;
#endif
        kf[nxt][c] = bf16_to_f32(bf16_round(
            bf16_to_f32(bf16_round(raw_k)) * bf16_to_f32(k_inv_rows[pbuf][pidx])));
        const float cur_v = raw_v;
        if (row + 1 < rows) {
            const int next_base = (row + 1) * qkv_width;
            raw_q = projection[next_base + q_offset + c];
            raw_k = projection[next_base + k_offset + c];
            raw_v = projection[next_base + v_offset + c];
        }
        // Parameters of the next 256-row block, into the buffer of the block
        // before the current one (whose last row's parameters were read two
        // iterations ago, before the previous barrier).  Their first reader
        // is 255 iterations away, past many per-row barriers.
        if ((i & 255) == 0 && i + 256 < rows) {
            const int nbuf = ((i >> 8) + 1) & 1;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int prow = i + 256 + c + h * 128;
                if (prow < rows) {
                    __hip_bfloat16 q_inv, k_inv;
                    float decay, beta_value;
                    gdn_pipe128_row_params(projection, gate, beta, prow, qkv_width, q_offset,
                                           k_offset, value_heads, value_head, q_inv, k_inv,
                                           decay, beta_value);
                    q_inv_rows[nbuf][prow & 255] = q_inv;
                    k_inv_rows[nbuf][prow & 255] = k_inv;
                    decay_rows[nbuf][prow & 255] = decay;
                    beta_rows[nbuf][prow & 255] = beta_value;
                }
            }
        }
        gdn_lds_barrier();

        const float decay = decay_rows[pbuf][pidx];
        const float beta_value = beta_rows[pbuf][pidx];
        float out = 0.0f;
        float kv_mem = 0.0f;
        gdn_pipe128_out_decay_kv(s, &qs[cur][0], &kf[nxt][0], lane_zero, decay, out, kv_mem);
        output[i * value_width + value_head * 128 + c] = out;
#if defined(__gfx942__)
        const float delta =
            __fmul_rn(__fsub_rn(bf16_to_f32(bf16_round(cur_v)), kv_mem), beta_value);
#else
        const float delta = (bf16_to_f32(bf16_round(cur_v)) - kv_mem) * beta_value;
#endif
        gdn_pipe128_update(s, &kf[nxt][0], lane_zero, delta);
        cur = nxt;
        nxt = nxt == 2 ? 0 : nxt + 1;
    }

    // Last row's output chain; its q was staged before the last barrier.
    {
        float out = 0.0f;
        gdn_pipe128_dot(s, &qs[cur][0], lane_zero, out);
        output[(rows - 1) * value_width + value_head * 128 + c] = out;
    }

    if constexpr (kQ8) {
        float peak = 0.0f;
        #pragma unroll
        for (int kd = 0; kd < 128; ++kd) peak = fmaxf(peak, fabsf(s[kd]));
        const float inv = gdn_q8_inv(peak);
        int4 packed[8];
        signed char* codes = reinterpret_cast<signed char*>(packed);
        int4 efp[kEf ? 16 : 1];
        unsigned short* efh = reinterpret_cast<unsigned short*>(efp);
        if constexpr (kEf) {
            // Q8-EF: one boundary at launch end; the old residual already
            // entered `s` and is not added again.
            const float new_scale = gdn_q8_scale(peak);
            #pragma unroll
            for (int kd = 0; kd < 128; ++kd) {
                const float code = gdn_q8ef_code(s[kd], inv);
                codes[kd] = (signed char)code;
                efh[kd] = gdn_q8ef_residual_bits(s[kd], code, new_scale);
            }
        } else {
        #pragma unroll
        for (int kd = 0; kd < 128; ++kd) {
            codes[kd] = gdn_q8_round(s[kd], inv, value_head, c, kd, frame);
        }
        }
        int4* dst = reinterpret_cast<int4*>(gdn_q8_row(state, value_head, c));
        #pragma unroll
        for (int i = 0; i < 8; ++i) dst[i] = packed[i];
        if constexpr (kEf) {
            int4* ef_dst =
                reinterpret_cast<int4*>(ef + ((size_t)value_head * 128 + c) * 128);
            #pragma unroll
            for (int i = 0; i < 16; ++i) ef_dst[i] = efp[i];
        }
        gdn_q8_scales(state, value_heads)[value_head * 128 + c] = gdn_q8_scale(peak);
        return;
    }
    float* const final_state = reinterpret_cast<float*>(state);
    #pragma unroll
    for (int kd = 0; kd < 128; ++kd) {
        final_state[state_base + kd * 128 + c] = s[kd];
    }
}

__launch_bounds__(128)
extern "C" __global__ void gated_delta_step_pipe128_f32(
    const float* projection, const float* gate, const float* beta, float* state,
    float* output, int rows, int qkv_width, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale) {
    gated_delta_step_pipe128_body<false>(projection, gate, beta, state, output, rows,
        qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale);
}

#if defined(__GFX12__)
__launch_bounds__(128)
extern "C" __global__ void gated_delta_step_pipe128_q8(
    const float* projection, const float* gate, const float* beta, signed char* state,
    float* output, int rows, int qkv_width, int key_heads, int value_heads,
    int key_dim, int value_dim, float query_scale, unsigned int frame) {
    gated_delta_step_pipe128_body<true>(projection, gate, beta, state, output, rows,
        qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale, frame);
}

// gated_delta_step_pipe128_q8 with a live f16 error-feedback residual plane `ef`
// (Q8 twin's kernargs, `ef` right after `state`, no frame).
__launch_bounds__(128)
extern "C" __global__ void gated_delta_step_pipe128_q8ef(
    const float* projection, const float* gate, const float* beta, signed char* state,
    unsigned short* ef, float* output, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_pipe128_body<true, true>(projection, gate, beta, state, output, rows,
        qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale, 0u, ef);
}
#endif
#endif

// Few-row verify rollback: re-run the `keep` kept rows of every GDN layer in
// one launch (blockIdx.y = layer; the layers' recurrences are independent).
// `table[2 * layer]` is the layer's captured recurrence input (the verify's
// `rows`-row convolution output [rows, qkv], then gate and beta
// [rows, value_heads] each), `table[2 * layer + 1]` its state ring: the rows
// re-run from ring slot `from` and the last kept row's state lands in slot
// `to + keep - 1`, bitwise the verify's own prefix. `discard` takes the
// recurrence output ([keep, value_heads * 128], shared by all layers).
__launch_bounds__(256)
extern "C" __global__ void gated_delta_rollback_layers_f32(
    const unsigned long long* table, float* discard, int rows, int keep, int from, int to,
    int qkv_width, int key_heads, int value_heads, float query_scale) {
    const float* recurrence = reinterpret_cast<const float*>(table[2 * blockIdx.y]);
    float* ring = reinterpret_cast<float*>(table[2 * blockIdx.y + 1]);
    const size_t state = (size_t)value_heads * 128 * 128;
    const size_t gate = (size_t)rows * qkv_width;
    gated_delta_step_persistent256_body<true>(recurrence, recurrence + gate,
        recurrence + gate + (size_t)rows * value_heads, ring + from * state, discard,
        ring + to * state, keep, qkv_width, key_heads, value_heads, 128, 128, query_scale);
}

// gated_delta_rollback_layers_f32 over rings of Q8 GDN state slots; `frame`
// is the last kept row's position, so each kept row requantizes at its own
// position and slot `to + keep - 1` is single-row decode's state.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_rollback_layers_q8(
    const unsigned long long* table, float* discard, int rows, int keep, int from, int to,
    int qkv_width, int key_heads, int value_heads, float query_scale, unsigned int frame) {
    const float* recurrence = reinterpret_cast<const float*>(table[2 * blockIdx.y]);
    signed char* ring = reinterpret_cast<signed char*>(table[2 * blockIdx.y + 1]);
    const size_t slot = gdn_q8_slot_bytes(value_heads);
    const size_t gate = (size_t)rows * qkv_width;
    gated_delta_step_persistent256_body<true, true>(recurrence, recurrence + gate,
        recurrence + gate + (size_t)rows * value_heads, ring + from * slot, discard,
        ring + to * slot, keep, qkv_width, key_heads, value_heads, 128, 128, query_scale,
        frame);
}

// A Q8 GDN state slot as F32 [head][key][value] (the F32 recurrence's
// order), for kernels that run on F32 state. Grid value_heads * 128 (one
// block per value row), block 128 (key channel).
extern "C" __global__ __launch_bounds__(128) void gdn_state_q8_to_f32(
    signed char* slot, float* state, int value_heads) {
    const int flat = (int)blockIdx.x;
    const int head = flat >> 7;
    const int row = flat & 127;
    const int column = (int)threadIdx.x;
    if (head >= value_heads) return;
    const float scale = gdn_q8_scales(slot, value_heads)[flat];
    state[((size_t)head * 128 + column) * 128 + row] =
        scale * (float)gdn_q8_row(slot, head, row)[column];
}

// F32 [head][key][value] state requantized into a Q8 GDN state slot, each
// value row by its absmax (the kernels' own requantization). Grid
// value_heads * 128, block 128.
extern "C" __global__ __launch_bounds__(128) void gdn_state_f32_to_q8(
    const float* state, signed char* slot, int value_heads, unsigned int frame) {
    const int flat = (int)blockIdx.x;
    const int head = flat >> 7;
    const int row = flat & 127;
    const int column = (int)threadIdx.x;
    if (head >= value_heads) return;
    __shared__ float peaks[128];
    const float value = state[((size_t)head * 128 + column) * 128 + row];
    peaks[column] = fabsf(value);
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (column < stride) peaks[column] = fmaxf(peaks[column], peaks[column + stride]);
        __syncthreads();
    }
    const float peak = peaks[0];
    gdn_q8_row(slot, head, row)[column] =
        gdn_q8_round(value, gdn_q8_inv(peak), head, row, column, frame);
    if (column == 0) gdn_q8_scales(slot, value_heads)[flat] = gdn_q8_scale(peak);
}

// gdn_state_q8_to_f32 with the live f16 residual added once:
// f32(code) * scale + f32(ef) (grid value_heads * 128, block 128).
extern "C" __global__ __launch_bounds__(128) void gdn_state_q8ef_to_f32(
    signed char* slot, const unsigned short* ef, float* state, int value_heads) {
    const int flat = (int)blockIdx.x;
    const int head = flat >> 7;
    const int row = flat & 127;
    const int column = (int)threadIdx.x;
    if (head >= value_heads) return;
    const float scale = gdn_q8_scales(slot, value_heads)[flat];
    state[((size_t)head * 128 + column) * 128 + row] = gdn_q8ef_recon(
        (float)gdn_q8_row(slot, head, row)[column], scale,
        gdn_q8ef_f16(ef[(size_t)flat * 128 + column]));
}

// F32 [head][key][value] state into a Q8 slot plus its live f16 residual
// plane: fresh nearest-even codes, scales and residuals; any old residual is
// ignored (it entered the F32 state through gdn_state_q8ef_to_f32). Grid
// value_heads * 128, block 128; the row peak reduction is gdn_state_f32_to_q8's.
extern "C" __global__ __launch_bounds__(128) void gdn_state_f32_to_q8ef(
    const float* state, signed char* slot, unsigned short* ef, int value_heads) {
    const int flat = (int)blockIdx.x;
    const int head = flat >> 7;
    const int row = flat & 127;
    const int column = (int)threadIdx.x;
    if (head >= value_heads) return;
    __shared__ float peaks[128];
    const float value = state[((size_t)head * 128 + column) * 128 + row];
    peaks[column] = fabsf(value);
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (column < stride) peaks[column] = fmaxf(peaks[column], peaks[column + stride]);
        __syncthreads();
    }
    const float peak = peaks[0];
    const float scale = gdn_q8_scale(peak);
    const float code = gdn_q8ef_code(value, gdn_q8_inv(peak));
    gdn_q8_row(slot, head, row)[column] = (signed char)code;
    ef[(size_t)flat * 128 + column] = gdn_q8ef_residual_bits(value, code, scale);
    if (column == 0) gdn_q8_scales(slot, value_heads)[flat] = scale;
}

// Zero `count` f16 residual elements (live-EF reset). Grid ceil(count / 256),
// block 256.
extern "C" __global__ __launch_bounds__(256) void gdn_q8ef_residual_zero(
    unsigned short* ef, unsigned int count) {
    const unsigned int i = blockIdx.x * 256u + threadIdx.x;
    if (i < count) ef[i] = 0;
}



// Every row's normalized q and k for the chunked GDN prefill arm
// (gated_delta_chunk_wmma.gfx1151.hip), at the reference's BF16 rounding
// points (the square sum as lane partials + a wave tree), stored as BF16.
// Grid [rows], block 256; wave w takes items w, w + 8, ... of the row's
// 2 * key_heads (q heads, then k heads).  `projection` is packed BF16 bits
// (the convolution's BF16 output).
extern "C" __global__ __launch_bounds__(256) void gated_delta_qk_norm_bf16_batched(
    const unsigned short* projection, __hip_bfloat16* qn, __hip_bfloat16* kn, int rows,
    int qkv_width, int key_heads) {
    const int row = blockIdx.x, wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned short* rowp = projection + (size_t)row * qkv_width;
    for (int item = wave; item < 2 * key_heads; item += 8) {
        const int is_k = item >= key_heads, kh = is_k ? item - key_heads : item;
        const uint2 w = *(const uint2*)(rowp + item * 128 + lane * 4);
        const float4 raw = {__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xFFFF0000u),
                            __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xFFFF0000u)};
        float v[4] = {bf16_to_f32(bf16_round(raw.x)), bf16_to_f32(bf16_round(raw.y)),
                      bf16_to_f32(bf16_round(raw.z)), bf16_to_f32(bf16_round(raw.w))};
        float part = 0.0f;
#pragma unroll
        for (int u = 0; u < 4; ++u) part += bf16_to_f32(bf16_round(v[u] * v[u]));
        part += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(part), 0xB1, 0xF, 0xF, false));
        part += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(part), 0x4E, 0xF, 0xF, false));
        part += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(part), 0x101F));
        part += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(part), 0x201F));
        part += __shfl_xor(part, 16, 32);
        const __hip_bfloat16 eps = bf16_round(1.0e-6f);
        const float inv = bf16_to_f32(bf16_round(rsqrtf(bf16_to_f32(bf16_round(part)) + bf16_to_f32(eps))));
        __hip_bfloat16* dst = (is_k ? kn : qn) + ((size_t)row * key_heads + kh) * 128 + lane * 4;
#pragma unroll
        for (int u = 0; u < 4; ++u) dst[u] = bf16_round(v[u] * inv);
    }
}


// BF16-stored HC streams (Qwen4 F16 prefill route) for the PLE block, which
// reads them as F32 (the values are BF16-exact there) and adds its output:
// the F32 path stores streams + output unrounded and every reader rounds it
// to BF16 first, so storing the rounded sum is the same value.
extern "C" __global__ void hc_state_bf16_to_f32(
    const unsigned short* state, float* out, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) out[i] = __uint_as_float((unsigned int)state[i] << 16);
}
extern "C" __global__ void hc_state_bf16_add_f32(
    unsigned short* state, const float* addend, int n) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        state[i] = bf16_round(__uint_as_float((unsigned int)state[i] << 16) + addend[i]).data;
}

// Source HF executes the recurrent/chunk result through a hidden BF16 cast
// before RMSNormGated.  Keep the compact production buffer F32 for the
// existing gate kernel while forcing the exact bounded BF16 storage boundary
// through a caller-owned device scratch tensor.
extern "C" __global__ void bf16_roundtrip_f32(
    const float* input,
    __hip_bfloat16* scratch,
    float* output,
    int elements) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    __hip_bfloat16 rounded = static_cast<__hip_bfloat16>(input[i]);
    scratch[i] = rounded;
    output[i] = bf16_to_f32(scratch[i]);
}
// Source shared-expert folding is BF16 at both multiply and add:
// routed output, shared down output, and sigmoid gate are each stored as
// BF16, then the product and residual sum are rounded independently.
extern "C" __global__ void bf16_scaled_add_f32(
    float* residual,
    const float* value,
    const float* scalar,
    int elements) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    const __hip_bfloat16 lhs = bf16_round(residual[i]);
    const __hip_bfloat16 rhs = bf16_round(value[i]);
    const __hip_bfloat16 gate = bf16_round(scalar[0]);
    const __hip_bfloat16 product = bf16_round(
        bf16_to_f32(gate) * bf16_to_f32(rhs));
    residual[i] = bf16_to_f32(bf16_round(
        bf16_to_f32(lhs) + bf16_to_f32(product)));
}
// Row-batched sibling of bf16_scaled_add_f32.  The scalar gate is one
// BF16 value per row; every product and residual sum keeps the same source
// rounding boundaries as the decode operation.
extern "C" __global__ void bf16_scaled_add_batched_f32(
    float* residual,
    const float* value,
    const float* scalar,
    int rows,
    int elements) {
    const int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * elements;
    if (flat >= total) return;
    const int row = flat / elements;
    const __hip_bfloat16 lhs = bf16_round(residual[flat]);
    const __hip_bfloat16 rhs = bf16_round(value[flat]);
    const __hip_bfloat16 gate = bf16_round(scalar[row]);
    const __hip_bfloat16 product = bf16_round(
        bf16_to_f32(gate) * bf16_to_f32(rhs));
    residual[flat] = bf16_to_f32(bf16_round(
        bf16_to_f32(lhs) + bf16_to_f32(product)));
}

// Learned HC read/mix. `norm_weight` is resident BF16; down/up projections
// have already run through the native quantized GEMV path.
extern "C" __global__ void hyper_read_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const float* low,
    const float* up,
    float* normalized,
    float* mixed,
    int branches,
    int hidden,
    int rank) {
    const int tid = (int)threadIdx.x;
    __shared__ float rms_sum[256];
    for (int branch = 0; branch < branches; ++branch) {
        const int offset = branch * hidden;
        float partial = 0.0f;
        for (int j = tid; j < hidden; j += (int)blockDim.x) {
            const float value =
                bf16_to_f32(bf16_round(input[offset + j]));
            partial += value * value;
        }
        rms_sum[tid] = partial;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                rms_sum[tid] += rms_sum[tid + stride];
            }
            __syncthreads();
        }
        const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
        for (int j = tid; j < hidden; j += (int)blockDim.x) {
            const int column = offset + j;
            normalized[column] = bf16_to_f32(bf16_round(
                bf16_to_f32(bf16_round(input[column]))
                * inv * (1.0f + bf16_to_f32(norm_weight[column]))));
        }
        __syncthreads();
    }
    for (int j = tid; j < hidden; j += (int)blockDim.x) {
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            const int offset = branch * hidden + j;
            float gate = 0.0f;
            for (int r = 0; r < rank; ++r) {
                gate += up[offset * rank + r] * low[r];
            }
            gate = bf16_to_f32(bf16_round(
                1.0f / (1.0f + expf(-gate))));
            value += bf16_to_f32(bf16_round(
                gate * normalized[offset]));
        }
        mixed[j] = bf16_to_f32(bf16_round(value / (float)branches));
    }
}
// Learned HC read/mix when the up projection has already been materialized.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column rather than the
// raw `[branches*hidden, rank]` weight matrix consumed above.
// Learned HC read/mix after the grouped RMSNorm has materialized `normalized`.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column.
extern "C" __global__ void hyper_read_projected_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const float* projected_up,
    float* normalized,
    float* mixed,
    int branches,
    int hidden,
    int rows) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    if (column < hidden) {
        const int row_offset = row * wide;
        const int mixed_offset = row * hidden;
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            const int offset = row_offset + branch * hidden + column;
            const float gate_input = bf16_to_f32(bf16_round(projected_up[offset]));
            const float gate = bf16_to_f32(bf16_round(
                1.0f / (1.0f + expf(-gate_input))));
            const float normed = bf16_to_f32(bf16_round(normalized[offset]));
            value += bf16_to_f32(bf16_round(gate * normed));
        }
        mixed[mixed_offset + column] =
            bf16_to_f32(bf16_round(value / (float)branches));
    }
    (void)input;
    (void)norm_weight;
}


// HC residual stream element: F32, or BF16 bits on the Qwen4 F16 prefill
// route.  Every reader rounds the stream to BF16 first, so both layouts
// yield the value it uses.
static __device__ __forceinline__ float hc_stream_value(float v) { return v; }
static __device__ __forceinline__ float hc_stream_value(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

// Sum of squares of the BF16-rounded stream values at j = tid, tid + 256,
// ... < n, accumulated in that order (the 256-thread HC norms' partial).  Ten
// loads are issued before they are consumed: the plain strided loop waits a
// full memory round trip per element.
template <typename T>
static __device__ __forceinline__ float hc_bf16_square_partial(
    const T* values, int tid, int n) {
    float partial = 0.0f;
    for (int base = tid; base < n; base += 10 * 256) {
        float raw[10];
        // A full batch loads unconditionally: guarded loads compile to one
        // branch + wait per element, serializing ten memory latencies.
        if (base + 9 * 256 < n) {
#pragma unroll
            for (int u = 0; u < 10; ++u) raw[u] = hc_stream_value(values[base + u * 256]);
        } else {
#pragma unroll
            for (int u = 0; u < 10; ++u) {
                const int j = base + u * 256;
                raw[u] = j < n ? hc_stream_value(values[j]) : 0.0f;
            }
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            if (base + u * 256 < n) {
                const float value = bf16_to_f32(bf16_round(raw[u]));
                partial += value * value;
            }
        }
    }
    return partial;
}

extern "C" __global__ void hyper_write_f32(
    const float* input,
    const float* normalized,
    const float* mixed,
    const float* gates,
    float* output,

    int branches,
    int hidden,
    int rows) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    const int row_offset = row * wide;
    const int branch = column / hidden;
    const float gate_logit =
        bf16_to_f32(bf16_round(gates[row * branches + branch]));
    const float gate_scaled = bf16_to_f32(
        bf16_round(gate_logit / (float)branches));
    const float gate = bf16_to_f32(bf16_round(
        2.0f * bf16_to_f32(bf16_round(
            1.0f / (1.0f + expf(-gate_scaled))))));
    const float injection = bf16_to_f32(bf16_round(
        gate * bf16_to_f32(
            bf16_round(mixed[row * hidden + column % hidden]))));
    const float value =
        bf16_to_f32(bf16_round(input[row_offset + column])) + injection;
    output[row_offset + column] = bf16_to_f32(bf16_round(value));
}

// hyper_write_f32 on BF16-stored HC streams (the Qwen4 F16 prefill route):
// two adjacent columns per thread (one 32-bit pair of BF16 values), the same
// expression per column, the result stored as BF16 bits.  `hidden` is even.
// Grid [ceil(branches * hidden / 512), rows], block 256.
extern "C" __global__ void hyper_write_bf16x2(
    const unsigned int* input,
    const float* mixed,
    const float* gates,
    unsigned int* output,
    int branches,
    int hidden,
    int rows) {
    const int pair = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int column = 2 * pair;
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    const int branch = column / hidden;
    const float gate_logit =
        bf16_to_f32(bf16_round(gates[row * branches + branch]));
    const float gate_scaled = bf16_to_f32(
        bf16_round(gate_logit / (float)branches));
    const float gate = bf16_to_f32(bf16_round(
        2.0f * bf16_to_f32(bf16_round(
            1.0f / (1.0f + expf(-gate_scaled))))));
    const float2 m = *(const float2*)(mixed + row * hidden + column % hidden);
    const unsigned int w = input[(row * wide + column) >> 1];
    const float lo = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(m.x))))
        + __uint_as_float(w << 16);
    const float hi = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(m.y))))
        + __uint_as_float(w & 0xFFFF0000u);
    output[(row * wide + column) >> 1] =
        (unsigned int)bf16_round(lo).data | ((unsigned int)bf16_round(hi).data << 16);
}

// hyper_write_bf16x8_gfx1201: production eight-columns-per-thread widening of
// hyper_write_bf16x2 (four BF16 pairs = one uint4 stream transaction, mixed as
// two float4 loads).  EXACT same seven-argument ABI as hyper_write_bf16x2; the
// per-column arithmetic and operand order are the incumbent's verbatim
// (injection + stream), NOT the row-fold reversed add or BF16-staged mixed.
// The gate is evaluated once for the eight columns.  Production geometry:
// branches == 4, hidden == 2560 (hidden % 8 == 0, so no 8-column block crosses
// a branch).  Grid [5, rows, 1] (wide = 10240 = 1280 units of 8), block
// [256, 1, 1], LDS 0.  The host launcher fails closed unless the stream/mixed
// pointers are 16-byte aligned for the uint4/float4 transactions; unaligned
// subviews stay on the x2 production path.
#if defined(__GFX12__)
extern "C" __global__ void hyper_write_bf16x8_gfx1201(
    const unsigned int* input,
    const float* mixed,
    const float* gates,
    unsigned int* output,
    int branches,
    int hidden,
    int rows) {
    const int unit = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int column = 8 * unit;
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    const int branch = column / hidden;
    const float gate_logit =
        bf16_to_f32(bf16_round(gates[row * branches + branch]));
    const float gate_scaled = bf16_to_f32(
        bf16_round(gate_logit / (float)branches));
    const float gate = bf16_to_f32(bf16_round(
        2.0f * bf16_to_f32(bf16_round(
            1.0f / (1.0f + expf(-gate_scaled))))));
    const float4 m0 = *(const float4*)(mixed + row * hidden + column % hidden);
    const float4 m1 = *(const float4*)(mixed + row * hidden + column % hidden + 4);
    const uint4 stream = *(const uint4*)(input + ((row * wide + column) >> 1));
    const unsigned int wv[4] = {stream.x, stream.y, stream.z, stream.w};
    const float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    unsigned int out[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float lo = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(mv[2 * p]))))
            + __uint_as_float(wv[p] << 16);
        const float hi = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(mv[2 * p + 1]))))
            + __uint_as_float(wv[p] & 0xFFFF0000u);
        out[p] = (unsigned int)bf16_round(lo).data | ((unsigned int)bf16_round(hi).data << 16);
    }
    *(uint4*)(output + ((row * wide + column) >> 1)) =
        make_uint4(out[0], out[1], out[2], out[3]);
}
#endif

// `normalized16`, when non-null, also receives each normalized value rounded
// to F16 (RNE, as convert_f32_to_f16) for the F16 WMMA projection, and
// `normalized`, when non-null, then holds the (already BF16-rounded) values
// as BF16 bits.
template <typename T>
static __device__ __forceinline__ void hyper_norm_rows(
    const T* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows,
    _Float16* normalized16) {
    const int branch = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (branch >= branches || row >= rows) return;
    const int tid = (int)threadIdx.x;
    const int offset = (row * branches + branch) * hidden;
    const int weight_offset = branch * hidden;
    __shared__ float rms_sum[256];
    // One 2560-wide batch (the Qwen4 decode shape): the second pass reuses
    // the first pass's loads, and the weights load alongside them.
    const bool single = hidden == 10 * 256;
    float raw[10], weight[10];
    if (single) {
        float partial = 0.0f;
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            raw[u] = hc_stream_value(input[offset + tid + u * 256]);
            weight[u] = bf16_to_f32(norm_weight[weight_offset + tid + u * 256]);
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const float value = bf16_to_f32(bf16_round(raw[u]));
            partial += value * value;
        }
        rms_sum[tid] = partial;
    } else {
        rms_sum[tid] = hc_bf16_square_partial(input + offset, tid, hidden);
    }
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rms_sum[tid] += rms_sum[tid + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
    for (int base = tid; base < hidden; base += 10 * 256) {
        // Batched loads as in hc_bf16_square_partial.
        if (!single) {
            if (base + 9 * 256 < hidden) {
#pragma unroll
                for (int u = 0; u < 10; ++u) {
                    raw[u] = hc_stream_value(input[offset + base + u * 256]);
                    weight[u] = bf16_to_f32(norm_weight[weight_offset + base + u * 256]);
                }
            } else {
#pragma unroll
                for (int u = 0; u < 10; ++u) {
                    const int j = base + u * 256;
                    raw[u] = j < hidden ? hc_stream_value(input[offset + j]) : 0.0f;
                    weight[u] = j < hidden ? bf16_to_f32(norm_weight[weight_offset + j]) : 0.0f;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 10; ++u) {
            const int j = base + u * 256;
            if (j >= hidden) break;
            const int column = offset + j;
            const float value = bf16_to_f32(bf16_round(
                bf16_to_f32(bf16_round(raw[u])) * inv * (1.0f + weight[u])));
            if (normalized16) {
                normalized16[column] = (_Float16)value;
                if (normalized)
                    ((unsigned short*)normalized)[column] =
                        (unsigned short)(__float_as_uint(value) >> 16);
            } else {
                normalized[column] = value;
            }
        }
    }
}

// `input` is the HC streams (BF16 bits when `state_bf16`).
extern "C" __global__ void hyper_norm_f32(
    const void* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows,
    _Float16* normalized16,
    int state_bf16) {
    if (state_bf16)
        hyper_norm_rows((const unsigned short*)input, norm_weight, normalized, branches, hidden,
                        rows, normalized16);
    else
        hyper_norm_rows((const float*)input, norm_weight, normalized, branches, hidden, rows,
                        normalized16);
}
// Lane 0 of `v += __shfl_down(v, off, 32)` for off = 16..1, with the
// partners moved by permlanex16 / row DPP on gfx11/gfx12 (same additions and
// order as gemm_bf16_xf32_multirow's reduction).
static __device__ __forceinline__ float hc_lane0_tree_sum(float v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x108, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x104, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x102, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x101, 0xf, 0xf, true));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
#endif
    return v;
}

// hyper_norm_f32 followed by the BF16 [4, wide] gate projection of that
// normalized row (gemm_bf16_xf32_multirow / gemv_bf16_xf32), for one token
// per block, without materializing `normalized` in global memory.  Each
// thread holds its row values in registers; every branch uses
// hyper_norm_f32's exact partial sums (j = tid + 256u in u order) and
// 256-wide LDS tree (the four trees run side by side); the normalized row
// lands in LDS and wave w < 4 replays the BF16 kernels' lane traversal
// (lane*8, +256), four chains, (a0+a1)+(a2+a3) fold and shuffle tree for gate
// row w, so both results are bitwise unchanged.
// Grid [rows], block 256, dynamic LDS (2 * hidden + 4 * 256) * 4 bytes.
// Requires four branches, hidden % 256 == 0 and hidden <= 2560.
// `input` is the HC streams (BF16 bits when `state_bf16`).
extern "C" __global__ __launch_bounds__(256) void hyper_norm_gate_f32(
    const void* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gates,
    int hidden,
    int state_bf16) {
    constexpr int branches = 4;
    constexpr int U = 10;
    extern __shared__ float hc_lds[];
    const int wide = branches * hidden;
    const int per = hidden / 256;
    // The normalized row is BF16-rounded: LDS keeps its BF16 bits.
    unsigned short* xsb = (unsigned short*)hc_lds;
    float* rms_sum = hc_lds + wide / 2;
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const size_t in = (size_t)row * wide;
    // The whole row is loaded up front (ten values per branch per thread).
    float raw[branches][U];
    if (state_bf16) {
        // The BF16 row arrives with 16-byte loads into the LDS the normalized
        // row later takes, then each thread reads its j = tid + 256u values.
        const uint4* s = (const uint4*)((const unsigned short*)input + in);
        for (int v = tid; v < wide / 8; v += 256) ((uint4*)xsb)[v] = s[v];
        __syncthreads();
#pragma unroll
        for (int b = 0; b < branches; ++b)
#pragma unroll
            for (int u = 0; u < U; ++u)
                raw[b][u] = u < per
                    ? __uint_as_float((unsigned int)xsb[b * hidden + tid + 256 * u] << 16)
                    : 0.0f;
    } else {
        const float* s = (const float*)input + in;
#pragma unroll
        for (int b = 0; b < branches; ++b)
#pragma unroll
            for (int u = 0; u < U; ++u)
                raw[b][u] = u < per ? s[b * hidden + tid + 256 * u] : 0.0f;
    }
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        float partial = 0.0f;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            raw[b][u] = bf16_to_f32(bf16_round(raw[b][u]));
            if (u < per) partial += raw[b][u] * raw[b][u];
        }
        rms_sum[b * 256 + tid] = partial;
    }
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int b = 0; b < branches; ++b)
                rms_sum[b * 256 + tid] += rms_sum[b * 256 + tid + stride];
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        const float inv = rsqrtf(rms_sum[b * 256] / (float)hidden + 1.0e-6f);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = b * hidden + tid + 256 * u;
            if (u < per)
                xsb[j] = (unsigned short)(__float_as_uint(bf16_to_f32(bf16_round(
                    raw[b][u] * inv * (1.0f + bf16_to_f32(norm_weight[j]))))) >> 16);
        }
    }
    __syncthreads();
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave >= branches) return;
    const unsigned short* w_row = gate_weight + (size_t)wave * wide;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Weights are fetched ten steps at a time before they are consumed: one
    // load per step waits a full memory round trip each.
    constexpr int BATCH = 10;
    int i = lane * 8;
    for (; i + 8 + (BATCH - 1) * 256 <= wide; i += BATCH * 256) {
        uint4 wb[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) wb[u] = *(const uint4*)(w_row + i + u * 256);
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const uint4 wv = wb[u];
            const uint4 xv = *(const uint4*)(xsb + i + u * 256);
            const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
            const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
            a0 += __uint_as_float(wv.x << 16) * xa.x;
            a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
            a2 += __uint_as_float(wv.y << 16) * xa.z;
            a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
            a0 += __uint_as_float(wv.z << 16) * xb.x;
            a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
            a2 += __uint_as_float(wv.w << 16) * xb.z;
            a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
        }
    }
    for (; i + 8 <= wide; i += 32 * 8) {
        const uint4 wv = *(const uint4*)(w_row + i);
        const uint4 xv = *(const uint4*)(xsb + i);
        const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
        const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
        a0 += __uint_as_float(wv.x << 16) * xa.x;
        a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
        a2 += __uint_as_float(wv.y << 16) * xa.z;
        a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
        a0 += __uint_as_float(wv.z << 16) * xb.x;
        a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
        a2 += __uint_as_float(wv.w << 16) * xb.z;
        a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
    }
    const float sum = hc_lane0_tree_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) gates[(size_t)row * branches + wave] = sum;
}

// H4 read side: hyper_norm_gate_f32's exact norm + BF16 gate projection of the
// streams, plus the normalized row for the HC down projection, in ONE launch,
// so the paired HC write consumes `gate_logits` instead of re-reading the
// streams (the streams cannot change between a read and its paired write).
// Same arithmetic as hyper_norm_gate_f32 (gate projection reads the BF16 LDS
// row, never an F16-rounded value).  The gate projection is run by waves 0..3
// with hyper_norm_gate_f32's text; waves 4..7 store the BF16 LDS row.  The two
// entries differ only in WHICH row copy waves 4..7 store, because adding the
// second store loop to one kernel changes the gate projection's instruction
// schedule and with it the sign of a NaN gate when the row mixes NaN signs
// (the fuzz in fn-wmo/h4 gates this; keep the entries separate):
//   hyper_norm_gate_outputs      : normalized16 (RNE F16, the F16 WMMA down input)
//   hyper_norm_gate_outputs_f32  : normalized32 (BF16-rounded value as F32)
// Both: inv4[row*4+branch] = the branch inverse RMS before normalization
// (optional), gate_logits[row*4 + branch] always.  The other row pointer is
// ignored.  Any optional output may be null.
// Grid [rows], block 256, dynamic LDS (hidden * 8 + 4 * 256 * 4) bytes.
// Requires four branches, hidden % 256 == 0 and hidden <= 2560.
template <bool F32OUT>
static __device__ __forceinline__ void hyper_norm_gate_outputs_body(
    const void* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gate_logits,
    _Float16* normalized16,
    float* normalized32,
    float* inv4,
    int hidden,
    int state_bf16) {
    constexpr int branches = 4;
    constexpr int U = 10;
    extern __shared__ float hc_lds[];
    const int wide = branches * hidden;
    const int per = hidden / 256;
    unsigned short* xsb = (unsigned short*)hc_lds;
    float* rms_sum = hc_lds + wide / 2;
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const size_t in = (size_t)row * wide;
    float raw[branches][U];
    if (state_bf16) {
        const uint4* s = (const uint4*)((const unsigned short*)input + in);
        for (int v = tid; v < wide / 8; v += 256) ((uint4*)xsb)[v] = s[v];
        __syncthreads();
#pragma unroll
        for (int b = 0; b < branches; ++b)
#pragma unroll
            for (int u = 0; u < U; ++u)
                raw[b][u] = u < per
                    ? __uint_as_float((unsigned int)xsb[b * hidden + tid + 256 * u] << 16)
                    : 0.0f;
    } else {
        const float* s = (const float*)input + in;
#pragma unroll
        for (int b = 0; b < branches; ++b)
#pragma unroll
            for (int u = 0; u < U; ++u)
                raw[b][u] = u < per ? s[b * hidden + tid + 256 * u] : 0.0f;
    }
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        float partial = 0.0f;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            raw[b][u] = bf16_to_f32(bf16_round(raw[b][u]));
            if (u < per) partial += raw[b][u] * raw[b][u];
        }
        rms_sum[b * 256 + tid] = partial;
    }
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int b = 0; b < branches; ++b)
                rms_sum[b * 256 + tid] += rms_sum[b * 256 + tid + stride];
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        const float inv = rsqrtf(rms_sum[b * 256] / (float)hidden + 1.0e-6f);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = b * hidden + tid + 256 * u;
            if (u < per)
                xsb[j] = (unsigned short)(__float_as_uint(bf16_to_f32(bf16_round(
                    raw[b][u] * inv * (1.0f + bf16_to_f32(norm_weight[j]))))) >> 16);
        }
    }
    __syncthreads();
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave >= branches) {
        // Waves 4..7 (idle in the gate projection) store the BF16 LDS row; the
        // projection below, run by waves 0..3, is hyper_norm_gate_f32's text.
        const int t2 = tid - branches * 32;
        if (inv4 && t2 < branches)
            inv4[(size_t)row * branches + t2] =
                rsqrtf(rms_sum[t2 * 256] / (float)hidden + 1.0e-6f);
        if constexpr (!F32OUT) if (normalized16) {
            for (int v = t2; v < wide / 8; v += 4 * 32) {
                const uint4 xv = ((const uint4*)xsb)[v];
                typedef _Float16 half8v __attribute__((ext_vector_type(8)));
                half8v h;
                h[0] = (_Float16)__uint_as_float(xv.x << 16);
                h[1] = (_Float16)__uint_as_float(xv.x & 0xFFFF0000u);
                h[2] = (_Float16)__uint_as_float(xv.y << 16);
                h[3] = (_Float16)__uint_as_float(xv.y & 0xFFFF0000u);
                h[4] = (_Float16)__uint_as_float(xv.z << 16);
                h[5] = (_Float16)__uint_as_float(xv.z & 0xFFFF0000u);
                h[6] = (_Float16)__uint_as_float(xv.w << 16);
                h[7] = (_Float16)__uint_as_float(xv.w & 0xFFFF0000u);
                ((half8v*)(normalized16 + in))[v] = h;
            }
        }
        if constexpr (F32OUT) if (normalized32) {
            for (int v = t2; v < wide / 8; v += 4 * 32) {
                const uint4 xv = ((const uint4*)xsb)[v];
                float4* o = (float4*)(normalized32 + in) + 2 * v;
                o[0] = make_float4(__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u),
                                   __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u));
                o[1] = make_float4(__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u),
                                   __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u));
            }
        }
        return;
    }
    const unsigned short* w_row = gate_weight + (size_t)wave * wide;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    constexpr int BATCH = 10;
    int i = lane * 8;
    for (; i + 8 + (BATCH - 1) * 256 <= wide; i += BATCH * 256) {
        uint4 wb[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) wb[u] = *(const uint4*)(w_row + i + u * 256);
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const uint4 wv = wb[u];
            const uint4 xv = *(const uint4*)(xsb + i + u * 256);
            const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
            const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
            a0 += __uint_as_float(wv.x << 16) * xa.x;
            a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
            a2 += __uint_as_float(wv.y << 16) * xa.z;
            a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
            a0 += __uint_as_float(wv.z << 16) * xb.x;
            a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
            a2 += __uint_as_float(wv.w << 16) * xb.z;
            a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
        }
    }
    for (; i + 8 <= wide; i += 32 * 8) {
        const uint4 wv = *(const uint4*)(w_row + i);
        const uint4 xv = *(const uint4*)(xsb + i);
        const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
        const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
        a0 += __uint_as_float(wv.x << 16) * xa.x;
        a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
        a2 += __uint_as_float(wv.y << 16) * xa.z;
        a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
        a0 += __uint_as_float(wv.z << 16) * xb.x;
        a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
        a2 += __uint_as_float(wv.w << 16) * xb.z;
        a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
    }
    const float sum = hc_lane0_tree_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) gate_logits[(size_t)row * branches + wave] = sum;
}
extern "C" __global__ __launch_bounds__(256) void hyper_norm_gate_outputs(
    const void* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gate_logits,
    _Float16* normalized16,
    float* normalized32,
    float* inv4,
    int hidden,
    int state_bf16) {
    hyper_norm_gate_outputs_body<false>(input, norm_weight, gate_weight, gate_logits,
                                        normalized16, normalized32, inv4, hidden, state_bf16);
}
extern "C" __global__ __launch_bounds__(256) void hyper_norm_gate_outputs_f32(
    const void* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gate_logits,
    _Float16* normalized16,
    float* normalized32,
    float* inv4,
    int hidden,
    int state_bf16) {
    hyper_norm_gate_outputs_body<true>(input, norm_weight, gate_weight, gate_logits,
                                       normalized16, normalized32, inv4, hidden, state_bf16);
}
// HC read tail on gfx1151: the BF16 [wide, low_rank] up projection
// (gemm_bf16_xf32_multirow's per-thread route, 256 < low_rank <= 512) fused
// with hyper_read_projected_f32, so the projected gate logits never reach
// global memory.  A block owns hidden columns j0..j0+7 of all four branches:
// lane l computes wide row (l >> 3) * hidden + j0 + (l & 7) exactly as the
// per-thread GEMM does (32 replayed lane partials, shuffle tree), then the
// four branch lanes of a column combine in branch order exactly as
// hyper_read_projected_f32 does.  Eight waves take 4 tokens each, 4 times
// (each widened weight serves four tokens).
// Grid [hidden / 8, ceil(rows / 128)], block 256, dynamic LDS
// 32 * (low_rank / 2 + 1) * 4 bytes.  branches == 4, hidden % 8 == 0.
// `normalized_bf16` != 0: `normalized` holds BF16 bits (hyper_norm_f32 with
// an F16 output), which widen to the value the F32 path rounds to.
extern "C" __global__ __launch_bounds__(256, 1) void hyper_read_up_fused_f32(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N,
    int normalized_bf16) {
    constexpr int TPW = 4;
    constexpr int TGROUPS = 4;
    const int branches = 4;
    const int wide = branches * hidden;
    extern __shared__ unsigned int ws[];
    const int stride = K / 2 + 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int j0 = blockIdx.x * 8;
    const int vpr = K / 8;
    for (int v = tid; v < 32 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 3) * hidden + j0 + (r & 7);
        const uint4 u = *(const uint4*)(W + (size_t)row * K + c * 8);
        unsigned int* d = ws + r * stride + c * 4;
        d[0] = u.x; d[1] = u.y; d[2] = u.z; d[3] = u.w;
    }
    __syncthreads();
    const unsigned int* wr = ws + lane * stride;
    const int my_row = (lane >> 3) * hidden + j0 + (lane & 7);
    for (int tg = 0; tg < TGROUPS; ++tg) {
        const int tbase = ((blockIdx.y * TGROUPS + tg) * 8 + wave) * TPW;
        if (tbase >= N) break;
        int tok[TPW];
#pragma unroll
        for (int t = 0; t < TPW; ++t) tok[t] = tbase + t < N ? tbase + t : tbase;
        float h[TPW][16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float sp[2][TPW];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int vl = l + 16 * p;
                float a[TPW][4];
#pragma unroll
                for (int t = 0; t < TPW; ++t) a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int i = vl * 8 + it * 256;
                    if (i + 8 > K) break;
                    const unsigned int u0 = wr[i / 2], u1 = wr[i / 2 + 1], u2 = wr[i / 2 + 2], u3 = wr[i / 2 + 3];
                    const float w0 = __uint_as_float(u0 << 16), w1 = __uint_as_float(u0 & 0xFFFF0000u);
                    const float w2 = __uint_as_float(u1 << 16), w3 = __uint_as_float(u1 & 0xFFFF0000u);
                    const float w4 = __uint_as_float(u2 << 16), w5 = __uint_as_float(u2 & 0xFFFF0000u);
                    const float w6 = __uint_as_float(u3 << 16), w7 = __uint_as_float(u3 & 0xFFFF0000u);
#pragma unroll
                    for (int t = 0; t < TPW; ++t) {
                        const float* xr = X + (size_t)tok[t] * K + i;
                        a[t][0] += w0 * xr[0];
                        a[t][1] += w1 * xr[1];
                        a[t][2] += w2 * xr[2];
                        a[t][3] += w3 * xr[3];
                        a[t][0] += w4 * xr[4];
                        a[t][1] += w5 * xr[5];
                        a[t][2] += w6 * xr[6];
                        a[t][3] += w7 * xr[7];
                    }
                }
#pragma unroll
                for (int t = 0; t < TPW; ++t) sp[p][t] = (a[t][0] + a[t][1]) + (a[t][2] + a[t][3]);
            }
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                h[t][l] = sp[0][t] + sp[1][t];
                asm volatile("" : "+v"(h[t][l]) :: "memory");
            }
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
#pragma unroll
                for (int l = 0; l < off; ++l) h[t][l] = h[t][l] + h[t][l + off];
            // hyper_read_projected_f32 for (token, column) from the branch
            // lanes; the sum runs in branch order from +0.0.
            const size_t token = (size_t)tok[t];
            const float gate_input = bf16_to_f32(bf16_round(h[t][0]));
            const float gate = bf16_to_f32(bf16_round(1.0f / (1.0f + expf(-gate_input))));
            const float normed = normalized_bf16
                ? __uint_as_float((unsigned int)((const unsigned short*)normalized)[token * wide + my_row] << 16)
                : bf16_to_f32(bf16_round(normalized[token * wide + my_row]));
            const float term = bf16_to_f32(bf16_round(gate * normed));
            float value = 0.0f;
#pragma unroll
            for (int branch = 0; branch < 4; ++branch)
                value += __shfl(term, (lane & 7) + 8 * branch, 32);
            if (lane < 8 && tbase + t < N)
                mixed[token * hidden + j0 + lane] = bf16_to_f32(bf16_round(value / (float)branches));
        }
    }
}
static __device__ __forceinline__ void gated_delta_conv_channel(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int channels,
    int history_rows,
    int kernel_size,
    int cursor,
    int channel) {
    const float current = bf16_to_f32(bf16_round(input[channel]));
    float value = bf16_to_f32(kernel[channel * kernel_size + history_rows]) * current;
    // causal_conv1d uses the raw HF weight order: oldest history first,
    // newest history last, and the current token at the final tap.
    if (history_rows == 3) {
        // K = 4 (Qwen4): the three taps' loads issue together.
        float w[3], h[3];
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) {
            w[tap] = bf16_to_f32(kernel[channel * kernel_size + tap]);
            h[tap] = history[((cursor + tap) % 3) * channels + channel];
        }
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) value += w[tap] * bf16_to_f32(bf16_round(h[tap]));
    } else for (int tap = 0; tap < history_rows; ++tap) {
        int row = (cursor + tap) % history_rows;
        value += bf16_to_f32(kernel[channel * kernel_size + tap])
            * bf16_to_f32(bf16_round(history[row * channels + channel]));
    }
    output[channel] = bf16_to_f32(bf16_round(
        value * (1.0f / (1.0f + expf(-value)))));
    for (int row = 0; row < history_rows; ++row) {
        next_history[row * channels + channel] =
            bf16_to_f32(bf16_round(history[row * channels + channel]));
    }
    if (history_rows > 0) {
        next_history[cursor * channels + channel] = current;
    }
}

extern "C" __global__ void gated_delta_conv_bf16_f32(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int channels,
    int history_rows,
    int kernel_size,
    int cursor) {
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    gated_delta_conv_channel(input, kernel, history, output, next_history, channels,
                             history_rows, kernel_size, cursor, channel);
}
// K=4 causal convolution over row-major [rows, channels]: the per-row
// kernel's ring, unrolled.  Row r reads x(r-3), x(r-2), x(r-1) (oldest
// first), where x(j) is the BF16-rounded input row j, or for j < 0 the history
// slot (start_cursor + j) mod 3 that the ring would still hold.  A thread owns
// one channel and 16 consecutive rows, so every row is independent.  Only the
// row-0 chunk reads the history, and it alone writes next_history after its
// reads: history may alias next_history.
// `output_bf16`: store the (BF16-rounded) outputs as packed BF16 bits.
// Optional gate parameters (`param_elements` > 0): the trailing blocks past
// the convolution tiles run gated_delta_params_bf16_f32_batched's body.
// Grid [ceil(channels / 256) * ceil(rows / 16) + param blocks], block 256: block
// i < conv_blocks * row_grid is conv tile (bx = i % conv_blocks, by = i / conv_blocks),
// the rest are parameter blocks i - conv_blocks * row_grid.
static __device__ __forceinline__ void gated_delta_params_flat(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int flat,
    int heads) {
    const int head = flat % heads;
    const float al = bf16_to_f32(a_log[head]);
    const float dt = bf16_to_f32(dt_bias[head]);
    const float value = bf16_to_f32(bf16_round(a[flat])) + dt;
    const float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[flat] = -expf(al) * softplus;
    beta[flat] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[flat]))));
}

#ifdef NPU_TRUNK_ARENA
static __device__ __forceinline__ void gated_delta_params_flat_arena(
    const __hip_bfloat16* a, const __hip_bfloat16* b,
    const __hip_bfloat16* a_log, const __hip_bfloat16* dt_bias,
    float* gate, float* beta, int flat, int heads,
    int a_pitch, int a_col0, int b_pitch, int b_col0) {
    const int head = flat % heads;
    const float al = bf16_to_f32(a_log[head]);
    const float dt = bf16_to_f32(dt_bias[head]);
    const float av = a_pitch ? bf16_to_f32(a[(size_t)(flat / heads) * a_pitch + a_col0 + head])
        : bf16_to_f32(bf16_round(((const float*)a)[flat]));
    const float bv = b_pitch ? bf16_to_f32(b[(size_t)(flat / heads) * b_pitch + b_col0 + head])
        : bf16_to_f32(bf16_round(((const float*)b)[flat]));
    const float value = av + dt;
    const float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[flat] = -expf(al) * softplus;
    beta[flat] = 1.0f / (1.0f + expf(-bv));
}
#endif

extern "C" __global__ void gated_delta_conv_bf16_f32_batched_k4(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int rows,
    int channels,
    int history_rows,
    int kernel_size,
    int start_cursor,
    int output_bf16,
    int input_bf16,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int param_elements,
    int heads
#ifdef NPU_TRUNK_ARENA
    , int input_pitch, int input_col0, int param_a_pitch, int param_a_col0,
    int param_b_pitch, int param_b_col0
#endif
    ) {
    // Compact 1-D grid: blocks [0, conv_blocks * row_grid) are the convolution
    // tiles (bx = channel block, by = 16-row block), the trailing blocks are
    // the parameter blocks.
    const unsigned conv_blocks = (unsigned)(channels + 255) / 256u;
    const unsigned conv_total = conv_blocks * (((unsigned)rows + 15u) / 16u);
    if (blockIdx.x >= conv_total) {
        const int flat = (int)(blockIdx.x - conv_total) * 256 + (int)threadIdx.x;
        if (flat < param_elements)
#ifdef NPU_TRUNK_ARENA
            gated_delta_params_flat_arena((const __hip_bfloat16*)param_a,
                (const __hip_bfloat16*)param_b, a_log, dt_bias, gate, beta, flat, heads,
                param_a_pitch, param_a_col0, param_b_pitch, param_b_col0);
#else
            gated_delta_params_flat(param_a, param_b, a_log, dt_bias, gate, beta, flat, heads);
#endif
        return;
    }
    const unsigned by = blockIdx.x / conv_blocks;
    const unsigned bx = blockIdx.x - by * conv_blocks;
    const int channel = (int)(bx * blockDim.x + threadIdx.x);
    const int r0 = (int)by * 16;
    if (channel >= channels || r0 >= rows
        || history_rows != 3 || kernel_size != 4) {
        return;
    }
    // `input_bf16`: the projection is stored as BF16 bits (its rounded values).
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
#ifdef NPU_TRUNK_ARENA
            b.data = ((const unsigned short*)input)[input_pitch ? (size_t)j * input_pitch + input_col0 + channel : j * channels + channel];
#else
            b.data = ((const unsigned short*)input)[j * channels + channel];
#endif
            return b;
        }
        return bf16_round(input[j * channels + channel]);
    };
    auto x = [&](int j) {
        return j >= 0 ? in(j)
                      : bf16_round(history[((start_cursor + j + 3) % 3) * channels + channel]);
    };
    const float k0 = bf16_to_f32(kernel[channel * 4 + 0]);
    const float k1 = bf16_to_f32(kernel[channel * 4 + 1]);
    const float k2 = bf16_to_f32(kernel[channel * 4 + 2]);
    const float k3 = bf16_to_f32(kernel[channel * 4 + 3]);
    __hip_bfloat16 h0 = x(r0 - 3), h1 = x(r0 - 2), h2 = x(r0 - 1);
    // Ring slot s after the last row: the newest row j in [rows-3, rows-1]
    // with (start_cursor + j) % 3 == s (the history slot itself when j < 0).
    __hip_bfloat16 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    const int r1 = min(r0 + 16, rows);
    for (int r = r0; r < r1; ++r) {
        const __hip_bfloat16 current_bf16 = in(r);
        float value = k3 * bf16_to_f32(current_bf16);
        value += k0 * bf16_to_f32(h0);
        value += k1 * bf16_to_f32(h1);
        value += k2 * bf16_to_f32(h2);
        const __hip_bfloat16 out = bf16_round(value * (1.0f / (1.0f + expf(-value))));
        if (output_bf16)
            ((unsigned short*)output)[r * channels + channel] = out.data;
        else
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

// ---------------------------------------------------------------------------
// Convolution + Q/K norm in one launch (Flash-Next H5; opt-in, chunked-GDN
// wave32 route only: gfx1151 / gfx1201 with a gated_delta_qk_norm_bf16_batched
// consumer).  `gated_delta_conv_qknorm_bf16_f32_batched_k4` is
// gated_delta_conv_bf16_f32_batched_k4's body plus the normalized Q/K that
// gated_delta_qk_norm_bf16_batched stores, byte for byte:
//   * the four-tap multiply/add order, the BF16 SiLU rounding and the ring
//     history are the plain kernel's expressions (one helper, same order);
//   * the first 16 blocks of the channel grid (channels [0, 4096) = the 16 q
//     heads then the 16 k heads, 128 channels each) are remapped: a wave owns
//     one head (32 lanes x 4 contiguous channels) and 4 rows of the 16-row
//     tile, wave w -> head bx * 2 + (w & 1), rows r0 + 4 * (w >> 1)
//     .. +3.  Each output row is independent (only the 3-row history
//     window is replayed from the input at the wave's first row), so the
//     conv values are the plain kernel's;
//   * the BF16 conv result is normalized in registers by the qk-norm kernel's
//     exact expressions (lane partial of 4 BF16-rounded squares in element
//     order, then the same DPP/swizzle/xor tree) and stored as BF16 to
//     qn / kn [rows, 16, 128];
//   * the V channels use the same wave tile (4 contiguous channels per lane,
//     no norm) when channels % 256 == 0, else the plain scalar body; the
//     parameter blocks run gated_delta_params_flat as in the plain kernel.
// Extra ABI vs the plain kernel: `qn`, `kn` (8-byte aligned, [rows,16,128]
// BF16).  Layout contract (the wrapper must check): channels >= 4096, q|k|v
// channel order, key_heads == 16, key_dim == 128; otherwise do not use this
// symbol (below 4096 channels the kernel degrades to the plain body and
// leaves qn / kn untouched).  `output_bf16` is a flag word: bit 0 = store
// the V output as packed BF16 (the plain kernel's meaning); bit 1 = also
// store the *unnormalized* Q/K convolution output into `output` (debug /
// capture); without bit 1 the Q/K columns of `output` are NOT written.
// Grid [ceil(channels / 256) * ceil(rows / 16) + param blocks] (1-D, the plain
// kernel's compact grid), block 256.  wave32 (the DPP rows/xor-16
// shuffle are the plain qk-norm kernel's).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ __hip_bfloat16 gated_delta_conv_silu_tap(
    float k0, float k1, float k2, float k3,
    __hip_bfloat16 current_bf16, __hip_bfloat16 h0, __hip_bfloat16 h1, __hip_bfloat16 h2) {
    float value = k3 * bf16_to_f32(current_bf16);
    value += k0 * bf16_to_f32(h0);
    value += k1 * bf16_to_f32(h1);
    value += k2 * bf16_to_f32(h2);
    return bf16_round(value * (1.0f / (1.0f + expf(-value))));
}

// gated_delta_qk_norm_bf16_batched's inverse-RMS for one lane's 4 BF16 values
// of a 128-channel head (a full wave32 must be converged here).
static __device__ __forceinline__ float gated_delta_qk_norm_inv_wave32(const float v[4]) {
    float part = 0.0f;
#pragma unroll
    for (int u = 0; u < 4; ++u) part += bf16_to_f32(bf16_round(v[u] * v[u]));
    part += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(part), 0xB1, 0xF, 0xF, false));
    part += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(part), 0x4E, 0xF, 0xF, false));
    part += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(part), 0x101F));
    part += __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(part), 0x201F));
    part += __shfl_xor(part, 16, 32);
    const __hip_bfloat16 eps = bf16_round(1.0e-6f);
    return bf16_to_f32(bf16_round(rsqrtf(bf16_to_f32(bf16_round(part)) + bf16_to_f32(eps))));
}

extern "C" __global__ __launch_bounds__(256) void gated_delta_conv_qknorm_bf16_f32_batched_k4(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int rows,
    int channels,
    int history_rows,
    int kernel_size,
    int start_cursor,
    int output_bf16,
    int input_bf16,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int param_elements,
    int heads,
    __hip_bfloat16* qn,
    __hip_bfloat16* kn
#ifdef NPU_TRUNK_ARENA
    , int input_pitch, int input_col0, int param_a_pitch, int param_a_col0,
    int param_b_pitch, int param_b_col0
#endif
    ) {
    const unsigned conv_blocks = (unsigned)(channels + 255) / 256u;
    const unsigned conv_total = conv_blocks * (((unsigned)rows + 15u) / 16u);
    if (blockIdx.x >= conv_total) {
        const int flat = (int)(blockIdx.x - conv_total) * 256 + (int)threadIdx.x;
        if (flat < param_elements)
#ifdef NPU_TRUNK_ARENA
            gated_delta_params_flat_arena((const __hip_bfloat16*)param_a,
                (const __hip_bfloat16*)param_b, a_log, dt_bias, gate, beta, flat, heads,
                param_a_pitch, param_a_col0, param_b_pitch, param_b_col0);
#else
            gated_delta_params_flat(param_a, param_b, a_log, dt_bias, gate, beta, flat, heads);
#endif
        return;
    }
    const unsigned by = blockIdx.x / conv_blocks;
    const unsigned bx = blockIdx.x - by * conv_blocks;
    const int r0 = (int)by * 16;
    if (r0 >= rows || history_rows != 3 || kernel_size != 4) return;
    const int store_bf16 = output_bf16 & 1;

    // 128-channel slices (4 contiguous channels per lane) x 4-row quarters.  The
    // q / k heads (blocks 0..15) are also normalized; the V channels use the
    // same tile when channels is a multiple of 256, else the scalar body below.
    const bool norm = bx < 16u && channels >= 4096;
    if (norm || (channels & 255) == 0) {
        const int wave = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31);
        const int head = (int)bx * 2 + (wave & 1);
        const int rb = r0 + (wave >> 1) * 4;
        if (rb >= rows) return;  // wave-uniform; no block barrier follows
        const int re = min(rb + 4, min(r0 + 16, rows));
        const int c0 = head * 128 + lane * 4;
        const bool is_k = head >= 16;
        const int kh = head & 15;
        const unsigned short* in_u16 = (const unsigned short*)input;
        const unsigned short* kern_u16 = (const unsigned short*)kernel;
        auto in4 = [&](int j, __hip_bfloat16 (&o)[4]) {
            if (input_bf16) {
#ifdef NPU_TRUNK_ARENA
                const uint2 w = *(const uint2*)(in_u16 + (input_pitch ? (size_t)j * input_pitch + input_col0 + c0 : (size_t)j * channels + c0));
#else
                const uint2 w = *(const uint2*)(in_u16 + (size_t)j * channels + c0);
#endif
                o[0].data = (unsigned short)(w.x & 0xFFFFu);
                o[1].data = (unsigned short)(w.x >> 16);
                o[2].data = (unsigned short)(w.y & 0xFFFFu);
                o[3].data = (unsigned short)(w.y >> 16);
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) o[u] = bf16_round(input[(size_t)j * channels + c0 + u]);
            }
        };
        auto x4 = [&](int j, __hip_bfloat16 (&o)[4]) {
            if (j >= 0) {
                in4(j, o);
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    o[u] = bf16_round(history[((start_cursor + j + 3) % 3) * channels + c0 + u]);
            }
        };
        float k0[4], k1[4], k2[4], k3[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint2 w = *(const uint2*)(kern_u16 + (size_t)(c0 + u) * 4);
            k0[u] = __uint_as_float(w.x << 16);
            k1[u] = __uint_as_float(w.x & 0xFFFF0000u);
            k2[u] = __uint_as_float(w.y << 16);
            k3[u] = __uint_as_float(w.y & 0xFFFF0000u);
        }
        __hip_bfloat16 h0[4], h1[4], h2[4];
        x4(rb - 3, h0);
        x4(rb - 2, h1);
        x4(rb - 1, h2);
        // Ring slot s after the last row (see the plain kernel).  Only the
        // row-0 quarter-0 wave reads the history (rb - 3 < 0) or writes
        // next_history, each lane its own channels, reads first.
        const bool writes_tail = (r0 == 0) && (wave >> 1) == 0;
        __hip_bfloat16 tail[3][4];
        if (writes_tail) {
            for (int s = 0; s < 3; ++s) {
                const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
                x4(j, tail[s]);
            }
        }
        for (int r = rb; r < re; ++r) {
            __hip_bfloat16 cur[4];
            in4(r, cur);
            float v[4];
            unsigned short bits[4];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const __hip_bfloat16 out =
                    gated_delta_conv_silu_tap(k0[u], k1[u], k2[u], k3[u], cur[u], h0[u], h1[u], h2[u]);
                v[u] = bf16_to_f32(out);
                bits[u] = out.data;
            }
            if (!norm || (output_bf16 & 2)) {
                if (store_bf16) {
                    uint2 packed;
                    packed.x = (unsigned)bits[0] | ((unsigned)bits[1] << 16);
                    packed.y = (unsigned)bits[2] | ((unsigned)bits[3] << 16);
                    *(uint2*)((unsigned short*)output + (size_t)r * channels + c0) = packed;
                } else {
#pragma unroll
                    for (int u = 0; u < 4; ++u) output[(size_t)r * channels + c0 + u] = v[u];
                }
            }
            if (norm) {
                const float inv = gated_delta_qk_norm_inv_wave32(v);
                uint2 packed;
                packed.x = (unsigned)bf16_round(v[0] * inv).data | ((unsigned)bf16_round(v[1] * inv).data << 16);
                packed.y = (unsigned)bf16_round(v[2] * inv).data | ((unsigned)bf16_round(v[3] * inv).data << 16);
                *(uint2*)((unsigned short*)(is_k ? kn : qn) + ((size_t)r * 16 + kh) * 128 + lane * 4) = packed;
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                h0[u] = h1[u];
                h1[u] = h2[u];
                h2[u] = cur[u];
            }
        }
        if (writes_tail) {
            for (int s = 0; s < 3; ++s)
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    next_history[s * channels + c0 + u] = bf16_to_f32(tail[s][u]);
        }
        return;
    }

    // V channels: the plain kernel's body.
    const int channel = (int)(bx * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
#ifdef NPU_TRUNK_ARENA
            b.data = ((const unsigned short*)input)[input_pitch ? (size_t)j * input_pitch + input_col0 + channel : j * channels + channel];
#else
            b.data = ((const unsigned short*)input)[j * channels + channel];
#endif
            return b;
        }
        return bf16_round(input[j * channels + channel]);
    };
    auto x = [&](int j) {
        return j >= 0 ? in(j)
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
        const __hip_bfloat16 current_bf16 = in(r);
        const __hip_bfloat16 out = gated_delta_conv_silu_tap(k0, k1, k2, k3, current_bf16, h0, h1, h2);
        if (store_bf16)
            ((unsigned short*)output)[r * channels + channel] = out.data;
        else
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

// Row-batched parameter expansion; source BF16 boundaries match decode.
extern "C" __global__ void gated_delta_params_bf16_f32_batched(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int rows,
    int heads) {
    const int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (flat >= rows * heads) return;
    gated_delta_params_flat(a, b, a_log, dt_bias, gate, beta, flat, heads);
}

static __device__ __forceinline__ void gated_delta_params_head(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int head) {
    float al = bf16_to_f32(a_log[head]);
    float dt = bf16_to_f32(dt_bias[head]);
    float value = bf16_to_f32(bf16_round(a[head])) + dt;
    float softplus = value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(al) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[head]))));
}

extern "C" __global__ void gated_delta_params_bf16_f32(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int heads) {
    int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    gated_delta_params_head(a, b, a_log, dt_bias, gate, beta, head);
}

// One decode row's convolution and gate parameters in one launch: blocks
// [0, conv_blocks) run gated_delta_conv_bf16_f32, block conv_blocks runs
// gated_delta_params_bf16_f32 (heads <= blockDim.x). Same per-element bodies,
// so both outputs are bitwise the two launches.
extern "C" __global__ void gated_delta_conv_params_bf16_f32(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int channels,
    int history_rows,
    int kernel_size,
    int cursor,
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int heads,
    int conv_blocks) {
    if ((int)blockIdx.x < conv_blocks) {
        int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (channel >= channels) return;
        gated_delta_conv_channel(input, kernel, history, output, next_history, channels,
                                 history_rows, kernel_size, cursor, channel);
    } else {
        int head = (int)threadIdx.x;
        if (head >= heads) return;
        gated_delta_params_head(a, b, a_log, dt_bias, gate, beta, head);
    }
}

extern "C" __global__ void gated_delta_params_f32(
    const float* a,
    const float* b,
    const float* a_log,
    const float* dt_bias,
    float* gate,
    float* beta,
    int heads) {
    int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    float value = a[head] + dt_bias[head];
    float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(a_log[head]) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-b[head]));
}

// Gated RMSNorm of one channel: `head_values` is the head's recurrent
// output (value_dim values), `z_head` / `out_head` the head's slices.
static __device__ __forceinline__ void gated_delta_gate_channel(
    const float* head_values,
    const float* z_head,
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
    float gate = bf16_to_f32(bf16_round(z_head[channel]));
    out_head[channel] = bf16_to_f32(bf16_round(
        normalized / (1.0f + expf(-gate))));
}

extern "C" __global__ void gated_delta_gate_bf16_f32(
    const float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int value_heads,
    int value_dim) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= value_heads || channel >= value_dim) return;
    int start = head * value_dim;
    gated_delta_gate_channel(recurrent + start, z + start, norm, output + start, channel,
                             value_dim);
}

// The decode step and its gated RMSNorm in one launch: the head's recurrent
// output stays in LDS for the norm (same values the separate launches pass
// through memory, so bitwise equal). The recurrent output is not stored.
template <bool kQ8, bool kEf = false>
static __device__ __forceinline__ void gated_delta_step_gate_norm128_body(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    void* state,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale,
    float* rotated,
    const float* signs1,
    const float* signs2,
    unsigned int* pair_counters,
    unsigned int frame,
    unsigned short* ef = nullptr) {
    __shared__ float head_values[128];
    __shared__ int rotate_pair;
    float out;
    if (!gated_delta_step_norm128_body<kQ8, kEf>(q, k, v, gate, beta, state, key_heads,
                                                 value_heads, key_dim, value_dim, query_scale,
                                                 frame, &out, ef))
        return;
    const int channel = (int)threadIdx.x;
    head_values[channel] = out;
    __syncthreads();
    const int start = (int)blockIdx.x * 128;
    gated_delta_gate_channel(head_values, z + start, norm, output + start, channel, 128);
    if (!rotated) return;
    // Optional: `rotated` receives mq_rotate_x(output) (256-wide groups = head
    // pairs). The second head of a pair to finish rotates the pair's group
    // (a zeroed arrival counter per pair, reset by that block).
    __threadfence();
    __syncthreads();
    const int pair = (int)blockIdx.x >> 1;
    if (threadIdx.x == 0) rotate_pair = atomicAdd(&pair_counters[pair], 1u) == 1u;
    __syncthreads();
    if (!rotate_pair || threadIdx.x >= 32) return;
    __threadfence();
    const int d0 = (int)threadIdx.x * 8;
    const int base = pair * 256;
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        values[e] = __hip_atomic_load(&output[base + d0 + e], __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT) * signs1[d0 + e];
    }
    mq_fwht256_lane(values, (int)threadIdx.x);
#pragma unroll
    for (int e = 0; e < 8; ++e) rotated[base + d0 + e] = values[e] * 0.0625f * signs2[d0 + e];
    if (threadIdx.x == 0) pair_counters[pair] = 0u;
}

extern "C" __global__ void gated_delta_step_gate_norm128_gfx1151(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    float* state, const float* z, const __hip_bfloat16* norm, float* output, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale, float* rotated,
    const float* signs1, const float* signs2, unsigned int* pair_counters) {
    gated_delta_step_gate_norm128_body<false>(q, k, v, gate, beta, state, z, norm, output,
        key_heads, value_heads, key_dim, value_dim, query_scale, rotated, signs1, signs2,
        pair_counters, 0u);
}

// gated_delta_step_gate_norm128_gfx1151 on a Q8 GDN state slot.
extern "C" __global__ void gated_delta_step_gate_norm128_q8(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    signed char* state, const float* z, const __hip_bfloat16* norm, float* output,
    int key_heads, int value_heads, int key_dim, int value_dim, float query_scale,
    float* rotated, const float* signs1, const float* signs2, unsigned int* pair_counters,
    unsigned int frame) {
    gated_delta_step_gate_norm128_body<true>(q, k, v, gate, beta, state, z, norm, output,
        key_heads, value_heads, key_dim, value_dim, query_scale, rotated, signs1, signs2,
        pair_counters, frame);
}

// gated_delta_step_gate_norm128_q8 with a live f16 error-feedback residual plane
// `ef` (Q8 twin's kernargs, `ef` right after `state`, no frame).
extern "C" __global__ void gated_delta_step_gate_norm128_q8ef(
    const float* q, const float* k, const float* v, const float* gate, const float* beta,
    signed char* state, unsigned short* ef, const float* z, const __hip_bfloat16* norm,
    float* output, int key_heads, int value_heads, int key_dim, int value_dim,
    float query_scale, float* rotated, const float* signs1, const float* signs2,
    unsigned int* pair_counters) {
    gated_delta_step_gate_norm128_body<true, true>(q, k, v, gate, beta, state, z, norm, output,
        key_heads, value_heads, key_dim, value_dim, query_scale, rotated, signs1, signs2,
        pair_counters, 0u, ef);
}
// Row-batched gated RMSNorm. The recurrent output is rounded in place before
// the norm reduction, matching the scalar BF16 boundary.  One wave owns one
// head (four channels per lane): the norm sum is the scalar kernel's serial
// i = 0..127 chain, run once per wave over the LDS-staged rounded head rather
// than once per channel thread.  Returns the lane's four outputs.
static __device__ __forceinline__ float4 gated_delta_gate_head128(
    float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int start,
    int lane,
    float* rounded) {
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
    const float4 zv = *reinterpret_cast<const float4*>(z + start + lane * 4);
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

// Grid [rows * value_heads], block 32.
extern "C" __global__ __launch_bounds__(32) void gated_delta_gate_bf16_f32_batched(
    float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int rows,
    int value_heads,
    int value_dim) {
    const int flat_head = (int)blockIdx.x;
    if (flat_head >= rows * value_heads || value_dim != 128) return;
    __shared__ float rounded[128];
    gated_delta_gate_head128(recurrent, z, norm, output, flat_head * 128,
                             (int)threadIdx.x, rounded);
}

// gated_delta_gate_bf16_f32_batched over head pairs (even value_heads), also
// writing mq_rotate_x(output) to `rotated`: the pair's 256 outputs form one
// rotation group, which wave 0 rotates from LDS.  Grid [rows * value_heads /
// 2], block 64.
extern "C" __global__ __launch_bounds__(64) void gated_delta_gate_rotate_bf16_f32_batched(
    float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    float* rotated,
    const float* signs1,
    const float* signs2) {
    const int wave = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    __shared__ float rounded[2][128];
    __shared__ float pair[256];
    const float4 result = gated_delta_gate_head128(
        recurrent, z, norm, output, ((int)blockIdx.x * 2 + wave) * 128, lane, rounded[wave]);
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

extern "C" __global__ void indexed_attention_norm_rope_f32(
    float* values,
    const __hip_bfloat16* norm,
    int heads,
    int head_dim,
    int head_stride,
    int position,
    int rotary_dim,
    RopeYarn yarn) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    int start = head * head_stride;
    __shared__ float input_row[256];
    if (channel < head_dim) {
        input_row[channel] = values[start + channel];
    }
    __syncthreads();
    if (channel >= head_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float row_value = input_row[i];
        sum += row_value * row_value;
    }
    float inv = rsqrtf(sum / (float)head_dim + 1.0e-6f);
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
    values[start + channel] = value;
}
 
// QSA state formats (`QsaKvFormat` in tensor_ops.rs). K and V are separate
// token-major caches, one row per position:
//   F32  kv_heads * head_dim f32.
//   FP8  kv_heads * head_dim E4M3 codes, then kv_heads f16 scales: the qwen35
//        native fp8 row, quantized as kv_cache_write_fp8_e4m3 does (gfx12).
//   Q8   kv_heads * head_dim / 32 Q8_0 blocks (f16 scale, 32 int8 codes;
//        34 bytes), head-major: the dense q8 KV row, quantized as
//        kv_cache_write_q8_0 does (gfx11 and gfx12).
// An fp8 key score is the head's scale times the in-order sum of q * code; a
// q8 key score sums, block by block, the block's scale times its in-order
// sum of q * code; a value is scale * code. Every attention kernel goes
// through the helpers below, so for each format the per-head, grouped and
// serial kernels stay bitwise equal. The indexer's raw and pooled keys are
// BF16 values in every format. RDNA quantized formats store them as BF16;
// gfx942 FP8 FNUZ keeps the F32 index arenas and the reference pool/select.
#define QSA_KV_F32 0
#define QSA_KV_FP8 1
#define QSA_KV_Q8 2
#define QSA_KV_FP8_FNUZ 3
#if defined(__GFX12__) || defined(__gfx942__)
#define QSA_HAS_FP8 1
#endif
#if defined(__GFX11__) || defined(__GFX12__)
#define QSA_HAS_Q8 1
#endif

typedef float qsa_f32x2 __attribute__((ext_vector_type(2)));

template <int F>
static __device__ __forceinline__ int qsa_kv_row_bytes(int kv_heads, int head_dim) {
    if constexpr (F == QSA_KV_F32) {
        return kv_heads * head_dim * 4;
    } else if constexpr (F == QSA_KV_FP8 || F == QSA_KV_FP8_FNUZ) {
        return kv_heads * (head_dim + 2);
    } else {
        return kv_heads * (head_dim / 32) * 34;
    }
}

// Signed int8 code `byte` (0..3) of a little-endian word, as a float.
static __device__ __forceinline__ float qsa_q8_code(unsigned int word, int byte) {
    return (float)((int)(word << (24 - 8 * byte)) >> 24);
}

static __device__ __forceinline__ float qsa_half_bits(unsigned int bits) {
    return (float)__builtin_bit_cast(_Float16, (unsigned short)bits);
}

#if defined(QSA_HAS_FP8)
static __device__ __forceinline__ float qsa_fp8_code(unsigned int code) {
    const qsa_f32x2 pair = __builtin_amdgcn_cvt_pk_f32_fp8(code, false);
    return pair.x;
}
#endif

// Value channel `channel` of KV head `kv_head` in the quantized token row `row`.
template <int F>
static __device__ __forceinline__ float qsa_kv_value(
    const unsigned char* row,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel) {
    static_assert(F != QSA_KV_F32, "F32 values keep each kernel's own loads");
    if constexpr (F == QSA_KV_Q8) {
        const unsigned char* block = row + (kv_head * (head_dim / 32) + channel / 32) * 34;
        return (float)(signed char)block[2 + (channel & 31)]
            * qsa_half_bits(*reinterpret_cast<const unsigned short*>(block));
    } else {
#if defined(QSA_HAS_FP8)
    const float scale = qsa_half_bits(*reinterpret_cast<const unsigned short*>(
        row + kv_heads * head_dim + 2 * kv_head));
#if defined(__gfx942__)
    return qsa_fp8_code(row[kv_head * head_dim + channel]) * (2.0f * scale);
#else
    return qsa_fp8_code(row[kv_head * head_dim + channel]) * scale;
#endif
#else
    return 0.0f;
#endif
    }
}

// Unscaled QK dot of `heads` query rows (`q + h * q_stride`, head_dim floats
// each) with KV head `kv_head` of the quantized token row `row`, added to
// `score[h]`; every head sees the same key loads and the association above.
template <int F, int HEADS>
static __device__ __forceinline__ void qsa_kv_key_dots(
    const float* __restrict__ q,
    int q_stride,
    const unsigned char* row,
    int kv_heads,
    int head_dim,
    int kv_head,
    float (&score)[HEADS]) {
#if defined(__gfx942__)
#pragma clang fp contract(off)
    // Keep one-head and grouped-head instantiations on the same rounded
    // multiply/add chain; the compiler otherwise contracts only the latter.
#endif
    static_assert(F != QSA_KV_F32, "F32 keys keep each kernel's own loop");
    if constexpr (F == QSA_KV_Q8) {
        // A pair of blocks is 68 bytes, 4-byte aligned (the host admits q8 for
        // head_dim 256 only, so every head and token row is):
        // word 0 = scale0 | codes0[0..1], words 1..7 = codes0[2..29],
        // word 8 = codes0[30..31] | scale1, words 9..16 = codes1[0..31].
        const unsigned int* pairs =
            reinterpret_cast<const unsigned int*>(row + kv_head * (head_dim / 32) * 34);
        float total[HEADS];
#pragma unroll
        for (int h = 0; h < HEADS; ++h) total[h] = 0.0f;
        for (int pair = 0; pair < head_dim / 64; ++pair) {
            unsigned int w[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) w[i] = pairs[pair * 17 + i];
            const float scale0 = qsa_half_bits(w[0] & 0xFFFFu);
            const float scale1 = qsa_half_bits(w[8] >> 16);
            float code[64];
            code[0] = qsa_q8_code(w[0], 2);
            code[1] = qsa_q8_code(w[0], 3);
#pragma unroll
            for (int i = 1; i < 8; ++i) {
#pragma unroll
                for (int b = 0; b < 4; ++b) code[4 * i - 2 + b] = qsa_q8_code(w[i], b);
            }
            code[30] = qsa_q8_code(w[8], 0);
            code[31] = qsa_q8_code(w[8], 1);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int b = 0; b < 4; ++b) code[32 + 4 * i + b] = qsa_q8_code(w[9 + i], b);
            }
#pragma unroll
            for (int h = 0; h < HEADS; ++h) {
                const float* qh = q + h * q_stride + pair * 64;
                float dot0 = 0.0f;
                float dot1 = 0.0f;
#pragma unroll
                for (int d = 0; d < 32; ++d) dot0 += qh[d] * code[d];
#pragma unroll
                for (int d = 0; d < 32; ++d) dot1 += qh[32 + d] * code[32 + d];
                total[h] += dot0 * scale0;
                total[h] += dot1 * scale1;
            }
        }
#pragma unroll
        for (int h = 0; h < HEADS; ++h) score[h] += total[h];
    } else {
#if defined(QSA_HAS_FP8)
    // Codes are 4-byte aligned: the host admits fp8 for head_dim 256 and an
    // even KV-head count only.
    const unsigned int* codes = reinterpret_cast<const unsigned int*>(row + kv_head * head_dim);
    const float scale = qsa_half_bits(*reinterpret_cast<const unsigned short*>(
        row + kv_heads * head_dim + 2 * kv_head));
    float dot[HEADS];
#pragma unroll
    for (int h = 0; h < HEADS; ++h) dot[h] = 0.0f;
#pragma unroll 4
    for (int d4 = 0; d4 < head_dim / 4; ++d4) {
        const unsigned int word = codes[d4];
        const qsa_f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(word, false);
        const qsa_f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8(word, true);
#pragma unroll
        for (int h = 0; h < HEADS; ++h) {
            const float* qh = q + h * q_stride + 4 * d4;
            dot[h] += qh[0] * lo.x;
            dot[h] += qh[1] * lo.y;
            dot[h] += qh[2] * hi.x;
            dot[h] += qh[3] * hi.y;
        }
    }
#pragma unroll
#if defined(__gfx942__)
    for (int h = 0; h < HEADS; ++h) score[h] += dot[h] * (2.0f * scale);
#else
    for (int h = 0; h < HEADS; ++h) score[h] += dot[h] * scale;
#endif
#endif
    }
}

// Store one channel of a K or V row in a quantized format. Every thread of
// the block calls it (block 256 over one KV head, `channel` = threadIdx.x;
// lanes at or past head_dim pass any value and store nothing). fp8: the
// head's absmax is reduced through width-32 shuffles and `amax_lds` (one float
// per 32-lane group, including both halves of a gfx942 wave64), which
// synchronizes. q8: each wave32 is one 32-channel Q8_0 block (no barrier).
template <int F>
static __device__ __forceinline__ void qsa_kv_store(
    unsigned char* row,
    float value,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel,
    float* amax_lds) {
    static_assert(F != QSA_KV_F32, "F32 rows are stored directly");
    if constexpr (F == QSA_KV_Q8) {
        const bool active = channel < head_dim;
        float amax = active ? fabsf(value) : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
        }
        const float scale = amax / 127.0f;
        const float inv_scale = amax > 0.0f ? 127.0f / amax : 0.0f;
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        if (!active) return;
        unsigned char* block = row + (kv_head * (head_dim / 32) + channel / 32) * 34;
        if ((channel & 31) == 0) *reinterpret_cast<_Float16*>(block) = (_Float16)scale;
        block[2 + (channel & 31)] = (unsigned char)(signed char)q;
    } else {
#if defined(QSA_HAS_FP8)
    const bool active = channel < head_dim;
    float amax = active ? fabsf(value) : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }
    if ((threadIdx.x & 31) == 0) amax_lds[threadIdx.x >> 5] = amax;
    __syncthreads();
    amax = amax_lds[0];
    for (int wave = 1; wave < ((int)blockDim.x + 31) / 32; ++wave) {
        amax = fmaxf(amax, amax_lds[wave]);
    }
    __syncthreads();
    _Float16 s16 = amax == 0.0f ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f) {
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    }
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = bits >= 0x7bff ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }
    if (!active) return;
#if defined(__gfx942__)
    // FNUZ uses 0x80 for NaN, so every quantized negative zero (including
    // nonzero negative underflow) becomes 0x00. All other finite OCP codes
    // are identical after this exact half-range fold.
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32((value / sf) * 0.5f, 0.0f, 0u, false);
#else
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value / sf, 0.0f, 0u, false);
#endif
    if (channel == 0) {
        *reinterpret_cast<_Float16*>(row + kv_heads * head_dim + 2 * kv_head) = s16;
    }
#endif
    }
}

static __device__ __forceinline__ float qsa_index_f32(float value) {
    return value;
}

static __device__ __forceinline__ float qsa_index_f32(__hip_bfloat16 value) {
    return bf16_to_f32(value);
}

// Store a BF16-exact index-key value in an F32 or BF16 arena.
static __device__ __forceinline__ void qsa_index_store(float* slot, __hip_bfloat16 value) {
    *slot = bf16_to_f32(value);
}

static __device__ __forceinline__ void qsa_index_store(__hip_bfloat16* slot, __hip_bfloat16 value) {
    *slot = value;
}

// One head's RMSNorm + HalfSplit RoPE at `position` (block 256, one channel
// per thread; returns the channel's value, `head_dim` <= 256). Every block
// thread must call it (it synchronizes).
static __device__ __forceinline__ float qsa_norm_rope_head(
    const float* head_values,
    const __hip_bfloat16* norm,
    int head_dim,
    int position,
    int rotary_dim,
    RopeYarn yarn,
    float* input_row,
    float* row_sum) {
    const int channel = (int)threadIdx.x;
    if (channel < head_dim) {
        input_row[channel] = head_values[channel];
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

extern "C" __global__ void indexed_attention_norm_rope_f32_batched(
    float* values,
    const __hip_bfloat16* norm,
    int rows,
    int row_stride,
    int heads,
    int head_dim,
    int head_stride,
    int position_start,
    int rotary_dim,
    RopeYarn yarn) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    if (row >= rows || head >= heads || blockIdx.y != 0) return;
    int start = row * row_stride + head * head_stride;
    __shared__ float input_row[256];
    __shared__ float row_sum;
    const float value = qsa_norm_rope_head(values + start, norm, head_dim, position_start + row,
                                           rotary_dim, yarn, input_row, &row_sum);
    if ((int)threadIdx.x < head_dim) values[start + threadIdx.x] = value;
}

// RoPE angle table of `rows` consecutive positions for the fused QSA q/k
// norm+RoPE GEMM epilogue: row r holds the cosines of bases 0..rotary_dim/2
// then their sines (yarn mscale applied by rope_yarn_sincos) at
// `table[r * rotary_dim ...]`, from the exact expression of
// qsa_norm_rope_head. One thread per (row, base).
extern "C" __global__ void indexed_attention_rope_table_f32(
    float* table,
    int rows,
    int position_start,
    int rotary_dim,
    RopeYarn yarn) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int half = rotary_dim / 2;
    int row = idx / half;
    int base = idx - row * half;
    if (row >= rows) return;
    int position = position_start + row;
    float angle = (float)position
        * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
    float sine, cosine;
    rope_yarn_sincos(angle, base, yarn, &sine, &cosine);
    table[row * rotary_dim + base] = cosine;
    table[row * rotary_dim + half + base] = sine;
}

// QSA prologue of `rows` consecutive rows in one launch (block 256, one block
// per head and row): the index-query, query and key RMSNorm + RoPE (the three
// indexed_attention_norm_rope_f32_batched launches), the key/value cache
// append at `position + row`, and the index key's BF16 round trip and raw-key
// copy (bf16_round_trip_f32_strided + copy_rows_strided_f32). Same
// per-element operations. Grid [index_heads + heads + kv_heads + 1, rows];
// row r's inputs sit at r times each buffer's row width. `F` is the K/V cache
// format; the raw index keys are F32 for F32 caches and BF16 otherwise, and
// the fp8 rows are written exactly as the prefill append writes them.
template <int F, typename RawKey>
static __device__ __forceinline__ void qsa_decode_prologue(
    float* index_row,
    float* qgate,
    float* keys,
    const float* values,
    unsigned char* full_keys,
    unsigned char* full_values,
    RawKey* raw_index_keys,
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    int position_start,
    RopeYarn yarn) {
    __shared__ float input_row[256];
    __shared__ float row_sum;
    __shared__ float kv_amax[8];
    const int c = (int)threadIdx.x;
    const int r = (int)blockIdx.y;
    const int position = position_start + r;
    const int index_q_width = index_heads * index_dim;
    const int kv_width = kv_heads * head_dim;
    index_row += r * (index_q_width + index_kv_width);
    qgate += r * 2 * heads * head_dim;
    keys += r * kv_width;
    values += r * kv_width;
    int block = (int)blockIdx.x;
    if (block < index_heads) {
        float* row = index_row + block * index_dim;
        const float value = qsa_norm_rope_head(row, index_q_norm, index_dim, position,
                                               min(index_dim, 64), yarn, input_row, &row_sum);
        if (c < index_dim) row[c] = value;
        return;
    }
    block -= index_heads;
    if (block < heads) {
        float* row = qgate + block * 2 * head_dim;
        const float value = qsa_norm_rope_head(row, q_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        if (c < head_dim) row[c] = value;
        return;
    }
    block -= heads;
    if (block < kv_heads) {
        float* row = keys + block * head_dim;
        const float value = qsa_norm_rope_head(row, k_norm, head_dim, position,
                                               min(head_dim, 64), yarn, input_row, &row_sum);
        if constexpr (F == QSA_KV_F32) {
            if (c < head_dim) {
                const int channel = block * head_dim + c;
                row[c] = value;
                reinterpret_cast<float*>(full_keys)[position * kv_width + channel] = value;
                reinterpret_cast<float*>(full_values)[position * kv_width + channel] =
                    values[channel];
            }
        } else {
            const float v = c < head_dim ? values[block * head_dim + c] : 0.0f;
            if (c < head_dim) row[c] = value;
            const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(kv_heads, head_dim);
            qsa_kv_store<F>(full_keys + (size_t)position * row_bytes, value, kv_heads, head_dim,
                            block, c, kv_amax);
            qsa_kv_store<F>(full_values + (size_t)position * row_bytes, v, kv_heads, head_dim,
                            block, c, kv_amax);
        }
        return;
    }
    for (int i = c; i < index_kv_width; i += 256) {
        float v = index_row[index_q_width + i];
        if (isfinite(v)) {
            v = __uint_as_float((__float_as_uint(v) + 0x7FFFu + ((__float_as_uint(v) >> 16) & 1u))
                                & 0xFFFF0000u);
            index_row[index_q_width + i] = v;
        }
        if constexpr (F == QSA_KV_F32 || F == QSA_KV_FP8_FNUZ) {
            raw_index_keys[position * index_kv_width + i] = v;
        } else {
            raw_index_keys[(size_t)position * index_kv_width + i] = bf16_round(v);
        }
    }
}

#define QSA_DECODE_PROLOGUE(NAME, FORMAT, CACHE, RAW)                                      \
    extern "C" __global__ __launch_bounds__(256) void NAME(                               \
        float* index_row, float* qgate, float* keys, const float* values, CACHE* full_keys, \
        CACHE* full_values, RAW* raw_index_keys, const __hip_bfloat16* index_q_norm,       \
        const __hip_bfloat16* q_norm, const __hip_bfloat16* k_norm, int index_heads,       \
        int index_dim, int index_kv_width, int heads, int kv_heads, int head_dim,          \
        int position_start, RopeYarn yarn) {                                               \
        qsa_decode_prologue<FORMAT, RAW>(                                                  \
            index_row, qgate, keys, values, reinterpret_cast<unsigned char*>(full_keys),   \
            reinterpret_cast<unsigned char*>(full_values), raw_index_keys, index_q_norm,   \
            q_norm, k_norm, index_heads, index_dim, index_kv_width, heads, kv_heads,       \
            head_dim, position_start, yarn);                                               \
    }
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_f32, QSA_KV_F32, float, float)
#if defined(__GFX12__)
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_fp8, QSA_KV_FP8, unsigned char,
                    __hip_bfloat16)
#endif
#if defined(__gfx942__)
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_fp8_fnuz, QSA_KV_FP8_FNUZ, unsigned char,
                    float)
#endif
#if defined(QSA_HAS_Q8)
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_q8, QSA_KV_Q8, unsigned char,
                    __hip_bfloat16)
#endif
#undef QSA_DECODE_PROLOGUE
 

extern "C" __global__ void indexed_attention_cache_append_f32(
    const float* key,
    const float* value,
    float* full_keys,
    float* full_values,
    int position,
    int kv_width) {
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= kv_width) return;
    full_keys[position * kv_width + channel] = key[channel];
    full_values[position * kv_width + channel] = value[channel];
}
 
extern "C" __global__ void indexed_attention_cache_append_f32_batched(
    const float* key,
    const float* value,
    float* full_keys,
    float* full_values,
    int rows,
    int position_start,
    int kv_width) {
    int row = (int)blockIdx.y;
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || channel >= kv_width) return;
    int position = position_start + row;
    full_keys[position * kv_width + channel] = key[row * kv_width + channel];
    full_values[position * kv_width + channel] = value[row * kv_width + channel];
}

// Quantized twin of indexed_attention_cache_append_f32_batched: grid
// [kv_heads, rows], block 256, one block per KV head of one row. The rows are
// byte-identical to the decode prologue's for the same K/V.
template <int F>
static __device__ __forceinline__ void qsa_cache_append_batched(
    const float* key,
    const float* value,
    unsigned char* full_keys,
    unsigned char* full_values,
    int rows,
    int position_start,
    int kv_heads,
    int head_dim) {
    __shared__ float kv_amax[8];
    const int head = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    const int c = (int)threadIdx.x;
    if (head >= kv_heads || row >= rows) return;
    const int source = row * kv_heads * head_dim + head * head_dim + c;
    const float k = c < head_dim ? key[source] : 0.0f;
    const float v = c < head_dim ? value[source] : 0.0f;
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(kv_heads, head_dim);
    const size_t offset = (size_t)(position_start + row) * row_bytes;
    qsa_kv_store<F>(full_keys + offset, k, kv_heads, head_dim, head, c, kv_amax);
    qsa_kv_store<F>(full_values + offset, v, kv_heads, head_dim, head, c, kv_amax);
}

#if defined(__GFX12__)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_fp8_batched(
    const float* key, const float* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim) {
    qsa_cache_append_batched<QSA_KV_FP8>(key, value, full_keys, full_values, rows,
                                         position_start, kv_heads, head_dim);
}
#endif
#if defined(__gfx942__)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_fp8_fnuz_batched(
    const float* key, const float* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim) {
    qsa_cache_append_batched<QSA_KV_FP8_FNUZ>(key, value, full_keys, full_values, rows,
                                              position_start, kv_heads, head_dim);
}
#endif
#if defined(QSA_HAS_Q8)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_q8_batched(
    const float* key, const float* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim) {
    qsa_cache_append_batched<QSA_KV_Q8>(key, value, full_keys, full_values, rows,
                                        position_start, kv_heads, head_dim);
}
#endif

// Prefill index-key tail for a BF16 raw-key arena: each row's index key is
// rounded to BF16 in place (bf16_round_trip_f32_strided's arithmetic) and
// appended at `position_start + row`, as the decode prologue does per row.
// Grid [ceil(index_kv_width / 256), rows], block 256.
extern "C" __global__ void indexed_attention_index_key_append_bf16_batched(
    float* index_rows,
    __hip_bfloat16* raw_index_keys,
    int rows,
    int index_q_width,
    int index_width,
    int index_kv_width,
    int position_start) {
    const int row = (int)blockIdx.y;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || i >= index_kv_width) return;
    float* key = index_rows + row * index_width + index_q_width;
    float v = key[i];
    if (isfinite(v)) {
        v = __uint_as_float((__float_as_uint(v) + 0x7FFFu + ((__float_as_uint(v) >> 16) & 1u))
                            & 0xFFFF0000u);
        key[i] = v;
    }
    raw_index_keys[(size_t)(position_start + row) * index_kv_width + i] = bf16_round(v);
}

// Complete-block QSA pooling: mean over `compress` raw index keys, grouped
// RMS, then the same HalfSplit RoPE used by the index key stream.  The output
// remains in the resident pooled-key arena; no host selection metadata is
// involved.
// Complete-block QSA pooling: mean over raw BF16 index keys, followed by the
// learned source key RMSNorm and HalfSplit RoPE at the block start. A null
// `norm` is used only by synthetic parity buffers. `Key` is the arena storage
// (F32 or BF16, see the QSA format note): both hold the same values, so the
// pooled keys are the same values too.
template <typename Key>
static __device__ __forceinline__ void qsa_pool_rope(
    const Key* raw_keys,
    Key* pooled,
    const __hip_bfloat16* norm,
    int block_count,
    int compress,
    int index_dim,
    int first_block,
    RopeYarn yarn) {
    // The grid starts at `first_block` (0 for the full pool): blocks below it
    // already hold this kernel's output for their unchanged raw keys (the
    // incremental decode caller), so an incremental launch covers only the
    // blocks its rows complete.
    int block = first_block + (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if ( block >= block_count || compress <= 0) return;
    bool active = channel < index_dim;
    int base = block * compress * index_dim;
    float mean = 0.0f;
    if (active) {
        for (int row = 0; row < compress; ++row) {
            mean += qsa_index_f32(raw_keys[base + row * index_dim + channel])
                / (float)compress;
        }
    }
    __hip_bfloat16 mean_bf16 = bf16_round(mean);
    __shared__ float partial[256];
    float squared = active
        ? bf16_to_f32(bf16_round(bf16_to_f32(mean_bf16)
            * bf16_to_f32(mean_bf16)))
        : 0.0f;
    partial[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!active) return;
    float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
    float norm_value = norm == nullptr
        ? 0.0f
        : bf16_to_f32(norm[channel]);
    float value = bf16_to_f32(bf16_round(
        bf16_to_f32(mean_bf16) * inv * (1.0f + norm_value)));
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    int half = rotary_dim / 2;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other_mean = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other_mean += qsa_index_f32(raw_keys[base + row * index_dim + partner])
                / (float)compress;
        }
        __hip_bfloat16 other_bf16 = bf16_round(other_mean);
        float other_norm = norm == nullptr
            ? 0.0f
            : bf16_to_f32(norm[partner]);
        float other = bf16_to_f32(bf16_round(
            bf16_to_f32(other_bf16) * inv * (1.0f + other_norm)));
        int pair = channel < half ? channel : partner;
        float angle = (float)(block * compress)
            * powf(10000000.0f, -((float)(2 * pair)) / (float)rotary_dim);
        float sine, cosine;
        rope_yarn_sincos(angle, pair, yarn, &sine, &cosine);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    qsa_index_store(pooled + block * index_dim + channel, bf16_round(value));
}

extern "C" __global__ void indexed_attention_pool_rope_f32(
    const float* raw_keys, float* pooled, const __hip_bfloat16* norm, int block_count,
    int compress, int index_dim, int first_block, RopeYarn yarn) {
    qsa_pool_rope(raw_keys, pooled, norm, block_count, compress, index_dim, first_block, yarn);
}

extern "C" __global__ void indexed_attention_pool_rope_bf16(
    const __hip_bfloat16* raw_keys, __hip_bfloat16* pooled, const __hip_bfloat16* norm,
    int block_count, int compress, int index_dim, int first_block, RopeYarn yarn) {
    qsa_pool_rope(raw_keys, pooled, norm, block_count, compress, index_dim, first_block, yarn);
}

static __device__ __forceinline__ float4 qsa_index_load4(const float* row, int d4) {
    return reinterpret_cast<const float4*>(row)[d4];
}

static __device__ __forceinline__ float4 qsa_index_load4(const __hip_bfloat16* row, int d4) {
    const uint2 w = reinterpret_cast<const uint2*>(row)[d4];
    return make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xFFFF0000u),
                       __uint_as_float(w.y << 16), __uint_as_float(w.y & 0xFFFF0000u));
}

// Compute one block score with the original head/dimension traversal. Keeping
// this as one thread's serial reduction preserves the F32 association and the
// `fmaxf` NaN/tie behavior while allowing independent blocks to run in parallel.
// `P` is the pooled-key storage (F32 or BF16; the same values either way).
template <typename P>
static __device__ __forceinline__ float indexed_attention_select_score_f32(
    const float* query,
    const P* pooled,
    int block,
    int index_heads,
    int index_dim) {
    float score = 0.0f;
    for (int head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        for (int d = 0; d < index_dim; ++d) {
            dot += query[head * index_dim + d]
                * qsa_index_f32(pooled[block * index_dim + d]);
        }
        score += fmaxf(dot / sqrtf((float)index_dim), 0.0f);
    }
    return score;
}

// indexed_attention_select_score_f32 with four heads' dot chains side by
// side, so each pooled value is loaded once for all four (float4 loads when
// both rows are 16-byte aligned and index_dim % 4 == 0).  Every chain keeps
// its d order and heads are added in order: the score is bitwise unchanged.
template <typename P>
static __device__ __forceinline__ float indexed_attention_select_score4_f32(
    const float* query,
    const P* pooled,
    int block,
    int index_heads,
    int index_dim) {
    const P* row = pooled + block * index_dim;
    float score = 0.0f;
    int head = 0;
    if ((index_dim & 3) == 0
        && ((reinterpret_cast<unsigned long long>(query) & 15)
            | (reinterpret_cast<unsigned long long>(row) & (4 * sizeof(P) - 1))) == 0) {
        for (; head + 4 <= index_heads; head += 4) {
            const float4* q0 = reinterpret_cast<const float4*>(query + head * index_dim);
            const float4* q1 = q0 + index_dim / 4;
            const float4* q2 = q1 + index_dim / 4;
            const float4* q3 = q2 + index_dim / 4;
            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#pragma unroll 4
            for (int d4 = 0; d4 < index_dim / 4; ++d4) {
                const float4 p = qsa_index_load4(row, d4);
                const float4 a = q0[d4], b = q1[d4], c = q2[d4], e = q3[d4];
                dot0 += a.x * p.x; dot0 += a.y * p.y; dot0 += a.z * p.z; dot0 += a.w * p.w;
                dot1 += b.x * p.x; dot1 += b.y * p.y; dot1 += b.z * p.z; dot1 += b.w * p.w;
                dot2 += c.x * p.x; dot2 += c.y * p.y; dot2 += c.z * p.z; dot2 += c.w * p.w;
                dot3 += e.x * p.x; dot3 += e.y * p.y; dot3 += e.z * p.z; dot3 += e.w * p.w;
            }
            score += fmaxf(dot0 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot1 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot2 / sqrtf((float)index_dim), 0.0f);
            score += fmaxf(dot3 / sqrtf((float)index_dim), 0.0f);
        }
    }
    for (; head < index_heads; ++head) {
        float dot = 0.0f;
        for (int d = 0; d < index_dim; ++d) {
            dot += query[head * index_dim + d] * qsa_index_f32(row[d]);
        }
        score += fmaxf(dot / sqrtf((float)index_dim), 0.0f);
    }
    return score;
}

// Serial fallback off gfx1151 or when the score array exceeds its LDS limit.
// Keep stable ascending-block tie selection and the causal tail unchanged.
// Batched selection keeps one independent workgroup per query row. The row's
// visible prefix derives from position_start, so complete blocks that belong
// only to later rows can never enter an earlier row's candidate set.
template <typename P>
static __device__ __forceinline__ void qsa_select_batched_serial(
    const float* query,
    const P* pooled,
    int* selected,
    int rows,
    int query_row_stride,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity) {
    int row = (int)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    int visible = position_start + row + 1;
    int row_block_count = visible / compress;
    if (row_block_count > block_count) row_block_count = block_count;
    int* row_selected = selected + row * capacity;
    const float* row_query = query + row * query_row_stride;
    for (int i = 0; i < capacity; ++i) row_selected[i] = -1;
    int chosen = budget_blocks < row_block_count ? budget_blocks : row_block_count;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < row_block_count; ++block) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (row_selected[prior * compress] == block * compress) used = true;
            }
            if (used) continue;
            float score = indexed_attention_select_score_f32(
                row_query, pooled, block, index_heads, index_dim);
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            for (int slot_row = 0;
                 slot_row < compress && slot * compress + slot_row < capacity;
                 ++slot_row) {
                int token = best_block * compress + slot_row;
                if (token < visible) row_selected[slot * compress + slot_row] = token;
            }
        }
    }
    int tail_start = row_block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start; token < visible && offset < capacity; ++token) {
        row_selected[offset++] = token;
    }
}

#define QSA_SELECT_SERIAL(NAME, POOLED)                                                    \
    extern "C" __global__ void NAME(const float* query, const POOLED* pooled, int* selected, \
                                    int rows, int query_row_stride, int block_count,         \
                                    int index_heads, int index_dim, int budget_blocks,       \
                                    int compress, int position_start, int capacity) {        \
        qsa_select_batched_serial(query, pooled, selected, rows, query_row_stride,           \
                                  block_count, index_heads, index_dim, budget_blocks,        \
                                  compress, position_start, capacity);                       \
    }
QSA_SELECT_SERIAL(indexed_attention_select_f32_batched_serial, float)
QSA_SELECT_SERIAL(indexed_attention_select_bf16_batched_serial, __hip_bfloat16)
#undef QSA_SELECT_SERIAL

// Chosen-block list of the past-budget radix path in
// indexed_attention_select_{f32,bf16}_batched (static LDS with its 256 radix
// bins; QSA_SELECT_BATCHED_STATIC_LDS_BYTES in tensor_ops.rs bounds both).
#define QSA_SELECT_LIST_CAPACITY 512
template <typename P>
static __device__ __forceinline__ void qsa_select_batched(
    const float* query,
    const P* pooled,
    int* selected,
    int rows,
    int query_row_stride,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity,
    int* mirror,
    float* scores_global,
    int scores_stride) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    // The persistent selection is the final row's.
    if (row != rows - 1) mirror = nullptr;
    // Scores live in dynamic LDS when the arena's block count fits it, else in
    // a global row of `scores_stride` floats (a multiple of 4, so the float4
    // reads below stay 16-byte aligned); every access is the same.
    extern __shared__ __align__(16) float scores_lds[];
    float* scores = scores_global != nullptr
        ? scores_global + (size_t)row * (size_t)scores_stride
        : scores_lds;
    int visible = position_start + row + 1;
    int row_block_count = visible / compress;
    if (row_block_count > block_count) row_block_count = block_count;
    const float* row_query = query + row * query_row_stride;
    int* row_selected = selected + row * capacity;
    // Every slot the caller can read owes a -1: the ones above `chosen`, the
    // tail a short row never reaches, and the whole row when it has no
    // complete block. This is the fill the old single-threaded path did.
    for (int i = (int)threadIdx.x; i < capacity; i += (int)blockDim.x) {
        row_selected[i] = -1;
        if (mirror) mirror[i] = -1;
    }
    if (row_block_count > 0) {
        for (int block = (int)threadIdx.x;
             block < row_block_count;
             block += (int)blockDim.x) {
            scores[block] = indexed_attention_select_score4_f32(
                row_query, pooled, block, index_heads, index_dim);
        }
    }
    __syncthreads();
    int chosen = budget_blocks < row_block_count ? budget_blocks : row_block_count;
    const int tid = (int)threadIdx.x;
    const int threads = (int)blockDim.x;
    // rank(block) counts the blocks that sort before it under
    // (score desc, index asc) — the exact order the serial selection sort
    // produced — so writing block `b` at slot `rank(b)` emits the same bytes.
    // Live scores are >= 0 (fmaxf against 0, summed per head), so no block can
    // hold the old sentinel and the order is strict: the ranks are distinct.
    // Non-negative, non-NaN F32 scores order exactly as their bit patterns.
    if (chosen < row_block_count && chosen <= QSA_SELECT_LIST_CAPACITY) {
        // Past the budget: find the chosen-th largest score by an 8-bit
        // radix select, list the chosen blocks (every score above it, then
        // the lowest-index blocks equal to it), and rank only the list. A
        // block that sorts before a listed block is itself listed, so its
        // rank among the list is its rank among all blocks.
        __shared__ unsigned int bins[256];
        __shared__ int list[QSA_SELECT_LIST_CAPACITY];
        __shared__ unsigned int radix_prefix;
        __shared__ int radix_need;
        __shared__ int list_len;
        unsigned int prefix = 0u, mask = 0u;
        int need = chosen;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += threads) bins[i] = 0u;
            __syncthreads();
            for (int block = tid; block < row_block_count; block += threads) {
                const unsigned int key = __float_as_uint(scores[block]);
                if ((key & mask) == prefix) atomicAdd(&bins[(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (tid == 0) {
                int above = 0;
                int digit = 255;
                for (; digit > 0; --digit) {
                    if (above + (int)bins[digit] >= need) break;
                    above += (int)bins[digit];
                }
                radix_need = need - above;
                radix_prefix = prefix | ((unsigned int)digit << shift);
            }
            __syncthreads();
            need = radix_need;
            prefix = radix_prefix;
            mask |= 255u << shift;
        }
        // `prefix` is the threshold key; the `need` lowest-index blocks equal
        // to it complete the list. Ordinals by index: contiguous ranges and an
        // exclusive scan of their equal counts (reusing `bins`).
        const int per = (row_block_count + threads - 1) / threads;
        const int first = min(tid * per, row_block_count);
        const int last = min(first + per, row_block_count);
        int equal = 0;
        for (int block = first; block < last; ++block) {
            equal += __float_as_uint(scores[block]) == prefix;
        }
        __syncthreads();
        bins[tid] = (unsigned int)equal;
        if (tid == 0) list_len = 0;
        __syncthreads();
        if (tid == 0) {
            unsigned int sum = 0u;
            for (int i = 0; i < threads; ++i) {
                const unsigned int count = bins[i];
                bins[i] = sum;
                sum += count;
            }
        }
        __syncthreads();
        int ordinal = (int)bins[tid];
        for (int block = first; block < last; ++block) {
            const unsigned int key = __float_as_uint(scores[block]);
            bool take = key > prefix;
            if (key == prefix) take = ordinal++ < need;
            if (take) list[atomicAdd(&list_len, 1)] = block;
        }
        __syncthreads();
        for (int i = tid; i < chosen; i += threads) {
            const int block = list[i];
            const float score = scores[block];
            int rank = 0;
            for (int j = 0; j < chosen; ++j) {
                const int other = list[j];
                const float value = scores[other];
                rank += (value > score || (value == score && other < block));
            }
            for (int slot = 0; slot < compress; ++slot) {
                int index = rank * compress + slot;
                int token = block * compress + slot;
                if (index < capacity && token < visible) {
                    row_selected[index] = token;
                    if (mirror) mirror[index] = token;
                }
            }
        }
    } else {
        for (int block = tid; block < row_block_count; block += threads) {
            float score = scores[block];
            int rank = 0;
            // Four LDS scores per read (`scores` is 16-byte aligned),
            // several reads in flight; the count is order-independent.
            int other = 0;
#pragma unroll 8
            for (; other + 4 <= row_block_count; other += 4) {
                const float4 v = *reinterpret_cast<const float4*>(scores + other);
                rank += (v.x > score || (v.x == score && other < block));
                rank += (v.y > score || (v.y == score && other + 1 < block));
                rank += (v.z > score || (v.z == score && other + 2 < block));
                rank += (v.w > score || (v.w == score && other + 3 < block));
            }
            for (; other < row_block_count; ++other) {
                float value = scores[other];
                if (value > score || (value == score && other < block)) ++rank;
            }
            if (rank < chosen) {
                for (int slot = 0; slot < compress; ++slot) {
                    int index = rank * compress + slot;
                    int token = block * compress + slot;
                    if (index < capacity && token < visible) {
                        row_selected[index] = token;
                        if (mirror) mirror[index] = token;
                    }
                }
            }
        }
    }
    int tail_start = row_block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start + (int)threadIdx.x;
         token < visible && offset + (token - tail_start) < capacity;
         token += (int)blockDim.x) {
        row_selected[offset + (token - tail_start)] = token;
        if (mirror) mirror[offset + (token - tail_start)] = token;
    }
}

#define QSA_SELECT_PARALLEL(NAME, POOLED)                                                      \
    extern "C" __global__ void NAME(const float* query, const POOLED* pooled, int* selected,    \
                                    int rows, int query_row_stride, int block_count,            \
                                    int index_heads, int index_dim, int budget_blocks,          \
                                    int compress, int position_start, int capacity,             \
                                    int* mirror, float* scores_global, int scores_stride) {     \
        qsa_select_batched(query, pooled, selected, rows, query_row_stride, block_count,        \
                           index_heads, index_dim, budget_blocks, compress, position_start,     \
                           capacity, mirror, scores_global, scores_stride);                     \
    }
QSA_SELECT_PARALLEL(indexed_attention_select_f32_batched, float)
QSA_SELECT_PARALLEL(indexed_attention_select_bf16_batched, __hip_bfloat16)
#undef QSA_SELECT_PARALLEL

// Block scores of up to R (8 or 16) consecutive query rows (index_heads == 4,
// index_dim % 4 == 0, index_dim <= 128): each pooled key row is read once for
// the R rows instead of once per row.  Every (row, block) score is
// indexed_attention_select_score4_f32's exact arithmetic (per-head serial
// d-order dots, heads summed in order).  Row r scores blocks below its own
// visible count into scores[r * score_stride + block].
// Grid [ceil(block_count / 256), ceil(rows / R)], block 256.
template <int R>
__device__ __forceinline__ void indexed_attention_select_scores_rows(
    const float* query,
    const float* pooled,
    float* scores,
    int rows,
    int query_row_stride,
    int block_count,
    int index_dim,
    int compress,
    int position_start,
    int score_stride) {
    __shared__ __align__(16) float qs[R * 4 * 128];
    const int row0 = (int)blockIdx.y * R;
    const int nrows = min(R, rows - row0);
    const int width = 4 * index_dim;
    for (int i = (int)threadIdx.x; i < nrows * width; i += 256) {
        const int r = i / width;
        qs[r * 4 * 128 + (i - r * width) / index_dim * 128 + (i - r * width) % index_dim] =
            query[(row0 + r) * query_row_stride + (i - r * width)];
    }
    __syncthreads();
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (block >= block_count) return;
    const float4* p4 = reinterpret_cast<const float4*>(pooled + block * index_dim);
    float dot[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int h = 0; h < 4; ++h) dot[r][h] = 0.0f;
    for (int d4 = 0; d4 < index_dim / 4; ++d4) {
        const float4 p = p4[d4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const float4 a = *reinterpret_cast<const float4*>(qs + r * 4 * 128 + h * 128 + d4 * 4);
                dot[r][h] += a.x * p.x;
                dot[r][h] += a.y * p.y;
                dot[r][h] += a.z * p.z;
                dot[r][h] += a.w * p.w;
            }
        }
    }
    // No break/continue: the loop must fully unroll so `dot` stays in
    // registers (with them, the 16-row kernel spills it to scratch).
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int visible_blocks = (position_start + row0 + r + 1) / compress;
        if (r < nrows && block < visible_blocks) {
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < 4; ++h) score += fmaxf(dot[r][h] / sqrtf((float)index_dim), 0.0f);
            scores[(row0 + r) * score_stride + block] = score;
        }
    }
}

// Eight rows per pooled read: decode and few-row verify.
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows8_f32(
    const float* query, const float* pooled, float* scores, int rows, int query_row_stride,
    int block_count, int index_dim, int compress, int position_start, int score_stride) {
    indexed_attention_select_scores_rows<8>(query, pooled, scores, rows, query_row_stride,
                                            block_count, index_dim, compress, position_start,
                                            score_stride);
}

// Sixteen rows per pooled read: prefill (half the pooled-key reads per row).
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows16_f32(
    const float* query, const float* pooled, float* scores, int rows, int query_row_stride,
    int block_count, int index_dim, int compress, int position_start, int score_stride) {
    indexed_attention_select_scores_rows<16>(query, pooled, scores, rows, query_row_stride,
                                             block_count, index_dim, compress, position_start,
                                             score_stride);
}

// Thirty-two rows per pooled read: row-exact wide verify (index_heads == 4,
// index_dim == 128 ONLY; the host admits no other width for this symbol).
// Same ABI, grid x and score cells as rows8/rows16_f32; grid y is
// ceil(rows / 32). Schedule (frozen in the wide-verify QSA design, 3B):
//  * each lane loads its 128 pooled F32 values once into the fixed-index
//    register array `key` (full unrolling, no private memory); lanes past the
//    live count hold zeros and issue no pooled load;
//  * queries are staged in banks of four rows, `qs[4][4][128]` = 8192 B LDS,
//    loading only the bank's live rows (never a row >= nrows);
//  * a bank's rows are scored serially with four live dot accumulators, each
//    (row, head) chain being exactly the rows<R> body above: dot from +0,
//    `dot += a * p` over ascending d (x, y, z, w per float4), then
//    `score += fmaxf(dot / sqrtf((float)index_dim), 0.0f)` over heads 0..3
//    from 0.0f; only cells with block < (position_start + row + 1) / compress
//    are stored;
//  * barriers are uniform: a workgroup wholly past its largest live ceiling
//    returns before the first one; afterwards no lane returns early, so every
//    lane reaches the same 2 * ceil(nrows / 4) - 1 barriers (at most 15): one
//    after each bank load, one before each query-bank reuse.
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_scores_rows32_f32(
    const float* query, const float* pooled, float* scores, int rows, int query_row_stride,
    int block_count, int index_dim, int compress, int position_start, int score_stride) {
    const int row0 = (int)blockIdx.y * 32;
    const int nrows = min(32, rows - row0);
    // Largest visible-block count over this workgroup's rows (the last row
    // dominates). Uniform per workgroup: the only return before a barrier.
    const int live = nrows > 0 ? min(block_count, (position_start + row0 + nrows) / compress) : 0;
    if ((int)blockIdx.x * 256 >= live) return;

    __shared__ __align__(16) float qs[4 * 4 * 128];
    const int block = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const bool lane_live = block < live;

    float key[128];
#pragma unroll
    for (int d4 = 0; d4 < 32; ++d4) {
        float4 p = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (lane_live) p = reinterpret_cast<const float4*>(pooled + block * 128)[d4];
        key[4 * d4 + 0] = p.x;
        key[4 * d4 + 1] = p.y;
        key[4 * d4 + 2] = p.z;
        key[4 * d4 + 3] = p.w;
    }

    const int banks = (nrows + 3) / 4;
#pragma unroll 1
    for (int bank = 0; bank < banks; ++bank) {
        const int brow0 = bank * 4;
        const int brows = min(4, nrows - brow0);
        for (int i = (int)threadIdx.x; i < brows * 512; i += 256) {
            const int r = i >> 9;
            qs[i] = query[(row0 + brow0 + r) * query_row_stride + (i & 511)];
        }
        __syncthreads();
#pragma unroll 1
        for (int r = 0; r < brows; ++r) {
            const int visible_blocks = (position_start + row0 + brow0 + r + 1) / compress;
            if (lane_live && block < visible_blocks) {
                float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
                for (int d4 = 0; d4 < 32; ++d4) {
#pragma unroll
                    for (int h = 0; h < 4; ++h) {
                        const float4 a =
                            *reinterpret_cast<const float4*>(qs + r * 512 + h * 128 + d4 * 4);
                        dot[h] += a.x * key[4 * d4 + 0];
                        dot[h] += a.y * key[4 * d4 + 1];
                        dot[h] += a.z * key[4 * d4 + 2];
                        dot[h] += a.w * key[4 * d4 + 3];
                    }
                }
                float score = 0.0f;
#pragma unroll
                for (int h = 0; h < 4; ++h) score += fmaxf(dot[h] / sqrtf((float)index_dim), 0.0f);
                scores[(row0 + brow0 + r) * score_stride + block] = score;
            }
        }
        if (bank + 1 < banks) __syncthreads();
    }
}
// indexed_attention_select_f32_batched's selection over block scores already
// in memory (indexed_attention_select_scores_rows8_f32, row stride
// `score_stride`), emitting the same bytes: the top `chosen` blocks by
// (score desc, index asc), then the causal tail. Scores are non-negative,
// non-NaN, never -0 floats, so their bits order like the values: a four-pass
// 8-bit radix select over the bits finds T, the chosen-th largest key, with
// LDS histograms (no score row in LDS, so many rows share a CU). Keys above
// T are ranked among themselves; the first keys equal to T, in index order,
// take the remaining ranks. budget_blocks <= 512.
// Grid [rows], block 256.
extern "C" __global__ __launch_bounds__(256) void indexed_attention_select_from_scores(
    const float* scores,
    int score_stride,
    int* selected,
    int rows,
    int block_count,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity,
    int* mirror) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    if (row != rows - 1) mirror = nullptr;
    __shared__ unsigned hist[8][256];  // one copy per wave: fewer atomic collisions
    __shared__ int wave_totals[8];
    __shared__ int wave_equals[8];
    __shared__ unsigned radix_prefix;
    __shared__ int radix_need;
    __shared__ int2 entries[512];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int visible = position_start + row + 1;
    const int row_block_count = min(visible / compress, block_count);
    const int chosen = min(budget_blocks, row_block_count);
    const unsigned* keys = reinterpret_cast<const unsigned*>(scores + row * score_stride);
    int* row_selected = selected + row * capacity;
    for (int i = tid; i < capacity; i += 256) {
        row_selected[i] = -1;
        if (mirror) mirror[i] = -1;
    }
    if (chosen > 0) {
        unsigned prefix = 0u;
        int need = chosen;
        for (int shift = 24; shift >= 0; shift -= 8) {
            const unsigned high = shift == 24 ? 0u : ~0u << (shift + 8);
#pragma unroll
            for (int w = 0; w < 8; ++w) hist[w][tid] = 0u;
            __syncthreads();
            for (int block = tid; block < row_block_count; block += 256) {
                const unsigned key = keys[block];
                if ((key & high) == prefix) atomicAdd(&hist[wave][(key >> shift) & 255u], 1u);
            }
            __syncthreads();
            // Thread t owns digit 255 - t: an inclusive scan over t gives the
            // count at or above each digit; the digit where it first reaches
            // `need` holds the chosen-th largest key.
            int count = 0;
#pragma unroll
            for (int w = 0; w < 8; ++w) count += (int)hist[w][255 - tid];
            int at_or_above = count;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const int up = __shfl_up(at_or_above, offset, 32);
                if (lane >= offset) at_or_above += up;
            }
            if (lane == 31) wave_totals[wave] = at_or_above;
            __syncthreads();
            for (int w = 0; w < wave; ++w) at_or_above += wave_totals[w];
            if (at_or_above >= need && at_or_above - count < need) {
                radix_prefix = prefix | ((unsigned)(255 - tid) << shift);
                radix_need = need - (at_or_above - count);
            }
            __syncthreads();
            prefix = radix_prefix;
            need = radix_need;
        }
        // prefix = T; `need` keys equal to T are taken, every key above T is.
        const int above = chosen - need;
        const int per = (row_block_count + 255) / 256;
        const int first = min(tid * per, row_block_count);
        const int last = min(first + per, row_block_count);
        // Above and equal counts scan separately: ties can number every block
        // (65,536 at 262K context), which overflows a 16-bit packed field.
        int above_count = 0;
        int equal_count = 0;
        for (int block = first; block < last; ++block) {
            const unsigned key = keys[block];
            above_count += key > prefix;
            equal_count += key == prefix;
        }
        int above_inclusive = above_count;
        int equal_inclusive = equal_count;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int above_up = __shfl_up(above_inclusive, offset, 32);
            const int equal_up = __shfl_up(equal_inclusive, offset, 32);
            if (lane >= offset) {
                above_inclusive += above_up;
                equal_inclusive += equal_up;
            }
        }
        if (lane == 31) {
            wave_totals[wave] = above_inclusive;
            wave_equals[wave] = equal_inclusive;
        }
        __syncthreads();
        int above_slot = above_inclusive - above_count;
        int equal_seen = equal_inclusive - equal_count;
        for (int w = 0; w < wave; ++w) {
            above_slot += wave_totals[w];
            equal_seen += wave_equals[w];
        }
        for (int block = first; block < last; ++block) {
            const unsigned key = keys[block];
            if (key > prefix) {
                entries[above_slot++] = make_int2((int)key, block);
            } else if (key == prefix) {
                if (equal_seen < need) entries[above + equal_seen] = make_int2((int)key, block);
                ++equal_seen;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = tid + k * 256;
            if (i >= chosen) break;
            const int2 entry = entries[i];
            int rank = i;
            if (i < above) {
                rank = 0;
                const unsigned key = (unsigned)entry.x;
                for (int j = 0; j < above; ++j) {
                    const int2 other = entries[j];
                    rank += (unsigned)other.x > key
                        || ((unsigned)other.x == key && other.y < entry.y);
                }
            }
            for (int slot = 0; slot < compress; ++slot) {
                const int index = rank * compress + slot;
                const int token = entry.y * compress + slot;
                if (index < capacity && token < visible) {
                    row_selected[index] = token;
                    if (mirror) mirror[index] = token;
                }
            }
        }
    }
    __syncthreads();  // the -1 fill above precedes the tail writes
    const int tail_start = row_block_count * compress;
    const int offset = chosen * compress;
    for (int token = tail_start + tid;
         token < visible && offset + (token - tail_start) < capacity;
         token += 256) {
        row_selected[offset + (token - tail_start)] = token;
        if (mirror) mirror[offset + (token - tail_start)] = token;
    }
}

extern "C" __global__ void indexed_attention_select_f32_serial(
    const float* query,
    const float* pooled,
    int* selected,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int visible,
    int capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < capacity; ++i) selected[i] = -1;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < block_count; ++block) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (selected[prior * compress] == block * compress) used = true;
            }
            if (used) continue;
            float score = indexed_attention_select_score_f32(
                query, pooled, block, index_heads, index_dim);
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            for (int row = 0; row < compress && slot * compress + row < capacity; ++row) {
                int token = best_block * compress + row;
                if (token < visible) selected[slot * compress + row] = token;
            }
        }
    }
    int tail_start = block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start; token < visible && offset < capacity; ++token) {
        selected[offset++] = token;
    }
}

// Select complete four-token blocks by the learned index score and append the
// causal tail. One workgroup owns the current query row; independent block
// scores are staged in dynamic LDS and the stable selection reuses a sentinel
// in that score array for the old first-slot membership predicate.
extern "C" __global__ void indexed_attention_select_f32(
    const float* query,
    const float* pooled,
    int* selected,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int visible,
    int capacity) {
    if (blockIdx.x != 0) return;
    extern __shared__ float scores[];
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    if (chosen > 0) {
        for (int block = (int)threadIdx.x;
             block < block_count;
             block += (int)blockDim.x) {
            scores[block] = indexed_attention_select_score_f32(
                query, pooled, block, index_heads, index_dim);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < capacity; ++i) selected[i] = -1;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < block_count; ++block) {
            float score = scores[block];
            if (score == -3.402823466e+38f) continue;
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            if (slot * compress < capacity && best_block * compress < visible) {
                scores[best_block] = -3.402823466e+38f;
            }
            for (int row = 0; row < compress && slot * compress + row < capacity; ++row) {
                int token = best_block * compress + row;
                if (token < visible) selected[slot * compress + row] = token;
            }
        }
    }
    int tail_start = block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start; token < visible && offset < capacity; ++token) {
        selected[offset++] = token;
    }
}
// Many independent small device copies in one launch (copy_regions in
// tensor_ops.rs): the region table rides in the kernarg segment by value;
// grid.y selects the region, grid.x strides over its 4-byte words.
#define COPY_REGIONS_MAX 64
struct CopyRegions {
    unsigned long long dst[COPY_REGIONS_MAX];
    unsigned long long src[COPY_REGIONS_MAX];
    unsigned int words[COPY_REGIONS_MAX];
};
extern "C" __global__ void copy_regions_u32(CopyRegions regions, int count) {
    const int region = (int)blockIdx.y;
    if (region >= count) return;
    unsigned int* dst = reinterpret_cast<unsigned int*>(regions.dst[region]);
    const unsigned int* src = reinterpret_cast<const unsigned int*>(regions.src[region]);
    const unsigned int words = regions.words[region];
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}
#undef COPY_REGIONS_MAX

// Reuse the previous MTP QSA sparse row after step zero. The selected buffer
// stores i32 indices in a byte-addressed Raw allocation. One workgroup owns
// the row: it snapshots the prior entries to LDS, keeps the valid ones (>= 0,
// <= position) in order via a block scan over contiguous per-thread ranges,
// appends `position` unless present, and fills the rest with -1 — the serial
// in-place compaction's exact result, with no host round-trip.
// Block 256; dynamic LDS min(selected_len, capacity) ints.
extern "C" __global__ void indexed_attention_reuse_selection(
    int* selected,
    int* selected_len_out,
    int selected_len,
    int position,
    int capacity) {
    if (blockIdx.x != 0) return;
    extern __shared__ int prior[];
    __shared__ int wave_totals[32];
    __shared__ int has_current;
    const int tid = (int)threadIdx.x;
    const int threads = (int)blockDim.x;
    const int n = min(selected_len, capacity);
    if (tid == 0) has_current = 0;
    for (int i = tid; i < n; i += threads) prior[i] = selected[i];
    __syncthreads();
    const int per = (n + threads - 1) / threads;
    const int first = min(tid * per, n);
    const int last = min(first + per, n);
    int keep = 0;
    for (int i = first; i < last; ++i) {
        const int token = prior[i];
        if (token < 0 || token > position) continue;
        ++keep;
        if (token == position) has_current = 1;
    }
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int waves = (threads + 31) >> 5;
    int inclusive = keep;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int up = __shfl_up(inclusive, offset, 32);
        if (lane >= offset) inclusive += up;
    }
    if (lane == 31 || tid == threads - 1) wave_totals[wave] = inclusive;
    __syncthreads();
    if (tid == 0)
        for (int w = 1; w < waves; ++w) wave_totals[w] += wave_totals[w - 1];
    __syncthreads();
    int write = inclusive - keep + (wave > 0 ? wave_totals[wave - 1] : 0);
    for (int i = first; i < last; ++i) {
        const int token = prior[i];
        if (token < 0 || token > position) continue;
        selected[write++] = token;
    }
    int count = wave_totals[waves - 1];
    if (!has_current && count < capacity) {
        if (tid == 0) selected[count] = position;
        ++count;
    }
    for (int i = count + tid; i < capacity; i += threads) selected[i] = -1;
    if (tid == 0) *selected_len_out = count;
}


// Serial fallback off gfx1151 or when score/token rows exceed its LDS limit.
// Keep the body identical to the original per-channel implementation. `F` is
// the K/V format (see the QSA format note).
template <int F>
static __device__ __forceinline__ void qsa_attention_batched_serial(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (row >= rows || head >= n_heads || channel >= head_dim) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    int kv_head = head / (n_heads / n_kv_heads);
    int q_row_stride = 2 * n_heads * head_dim;
    int q_head_offset = row * q_row_stride + head * head_dim * 2;
    const int* row_selected = selected_indices + row * capacity;
    const float* keys = reinterpret_cast<const float*>(full_keys);
    const float* values = reinterpret_cast<const float*>(full_values);
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, head_dim);
    auto key_score = [&](int token) {
        float score = 0.0f;
        if constexpr (F == QSA_KV_F32) {
            for (int d = 0; d < head_dim; ++d) {
                score += q_with_gate[q_head_offset + d]
                    * keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
            }
        } else {
            float dots[1] = {0.0f};
            qsa_kv_key_dots<F, 1>(q_with_gate + q_head_offset, 0,
                                  full_keys + (size_t)token * row_bytes, n_kv_heads, head_dim,
                                  kv_head, dots);
            score = dots[0];
        }
        return score;
    };
    auto value_of = [&](int token) {
        if constexpr (F == QSA_KV_F32) {
            return values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
        } else {
            return qsa_kv_value<F>(full_values + (size_t)token * row_bytes, n_kv_heads, head_dim,
                                   kv_head, channel);
        }
    };
    float max_score = -3.402823466e+38f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_selected[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        float score = key_score(token);
        max_score = fmaxf(max_score, score / sqrtf((float)head_dim));
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_selected[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        float score = key_score(token);
        float weight = expf(score / sqrtf((float)head_dim) - max_score);
        denominator += weight;
        numerator += weight * value_of(token);
    }
    float gate = q_with_gate[q_head_offset + head_dim + channel];
    int output_offset = row * n_heads * head_dim + head * head_dim + channel;
    output[output_offset] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}

// gfx942 pins for the parallel attention kernels below (qsa_attention_row_parallel,
// _hg4, _hg12). gfx942 hipcc -O3 packs independent multiplies into
// v_pk_mul_f32 (SLP) and then adds them unfused, and fuses others into
// v_pk_fma_f32, so a bare `acc += a * b` does not compile to the same
// rounding as the serial kernel's. Observed in the gfx942 ISA of
// indexed_attention_attention_f32_batched_serial: the key-dot chain is a
// fused v_fmac chain in d order, while `numerator += weight * value` is a
// v_mul_f32 followed by a packed add (product rounded, then added).
// QSA_DOT_ACC / QSA_SUM_ACC pin exactly those two roundings in the parallel
// kernels. __fadd_rn/__fmul_rn are plain operators in this HIP toolchain
// (__clang_hip_math.h), so the unfused sum needs contract(off) to keep the
// compiler from fusing it back into an fma. Every other architecture keeps
// the bare expressions.
#if defined(__gfx942__)
static __device__ __forceinline__ float qsa_sum_acc_unfused(float acc, float w, float v) {
#pragma clang fp contract(off)
    return acc + w * v;
}
#define QSA_DOT_ACC(acc, a, b) ((acc) = __fmaf_rn((a), (b), (acc)))
#define QSA_SUM_ACC(acc, w, v) ((acc) = qsa_sum_acc_unfused((acc), (w), (v)))
#else
#define QSA_DOT_ACC(acc, a, b) ((acc) += (a) * (b))
#define QSA_SUM_ACC(acc, w, v) ((acc) += (w) * (v))
#endif

// One query row of the parallel attention below against `selected_len`
// entries of `row_selected`; rows at or past `visible` count as invalid.
template <int F>
static __device__ __forceinline__ void qsa_attention_row_parallel(
    const float* __restrict__ q,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* row_selected,
    float* output_row,
    int selected_len,
    int visible,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int full_capacity) {
    extern __shared__ float row_scores[];
    int* row_tokens = reinterpret_cast<int*>(row_scores + selected_len);
    __shared__ float wave_max[8];
    const int tid = (int)threadIdx.x;
    const int head = (int)blockIdx.x;
    const int kv_head = head / (n_heads / n_kv_heads);
    const int kv_stride = n_kv_heads * head_dim;
    // F32 rows are kept as cache element offsets (token * kv_stride), the
    // quantized formats' as token indices.
    const int row_unit = F == QSA_KV_F32 ? kv_stride : 1;
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, head_dim);
    const float* keys = reinterpret_cast<const float*>(full_keys) + kv_head * head_dim;
    float partial_max = -3.402823466e+38f;
    for (int selected_row = tid; selected_row < selected_len; selected_row += (int)blockDim.x) {
        int token = row_selected[selected_row];
        bool ok = token >= 0 && token < full_capacity && token < visible;
        row_tokens[selected_row] = ok ? token * row_unit : -1;
        float score = 0.0f;
        if (ok) {
            if constexpr (F == QSA_KV_F32) {
                const float* key = keys + token * kv_stride;
                if (head_dim == 256) {
                    // Thirty-two key float4 loads in flight per batch; the FMAs
                    // still run in serial d order.
                    const float4* key4 = reinterpret_cast<const float4*>(key);
                    const float4* q4 = reinterpret_cast<const float4*>(q);
                    for (int c = 0; c < 64; c += 32) {
                        float4 k4[32];
#pragma unroll
                        for (int u = 0; u < 32; ++u) k4[u] = key4[c + u];
#pragma unroll
                        for (int u = 0; u < 32; ++u) {
                            const float4 qv = q4[c + u];
                            QSA_DOT_ACC(score, qv.x, k4[u].x);
                            QSA_DOT_ACC(score, qv.y, k4[u].y);
                            QSA_DOT_ACC(score, qv.z, k4[u].z);
                            QSA_DOT_ACC(score, qv.w, k4[u].w);
                        }
                    }
                } else if ((head_dim & 3) == 0) {
                    for (int d = 0; d < head_dim; d += 4) {
                        const float4 k4 = *reinterpret_cast<const float4*>(key + d);
                        QSA_DOT_ACC(score, q[d], k4.x);
                        QSA_DOT_ACC(score, q[d + 1], k4.y);
                        QSA_DOT_ACC(score, q[d + 2], k4.z);
                        QSA_DOT_ACC(score, q[d + 3], k4.w);
                    }
                } else {
                    for (int d = 0; d < head_dim; ++d) QSA_DOT_ACC(score, q[d], key[d]);
                }
            } else {
                float dots[1] = {0.0f};
                qsa_kv_key_dots<F, 1>(q, 0, full_keys + (size_t)token * row_bytes, n_kv_heads,
                                      head_dim, kv_head, dots);
                score = dots[0];
            }
            score /= sqrtf((float)head_dim);
            partial_max = fmaxf(partial_max, score);
        }
        row_scores[selected_row] = score;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_max = fmaxf(partial_max, __shfl_xor(partial_max, offset, 32));
    }
    if ((tid & 31) == 0) wave_max[tid >> 5] = partial_max;
    __syncthreads();
    float max_score = wave_max[0];
    for (int w = 1; w < ((int)blockDim.x + 31) / 32; ++w) max_score = fmaxf(max_score, wave_max[w]);
    for (int selected_row = tid; selected_row < selected_len; selected_row += (int)blockDim.x) {
        row_scores[selected_row] = row_tokens[selected_row] >= 0
            ? expf(row_scores[selected_row] - max_score)
            : 0.0f;
    }
    __syncthreads();
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    const float* values = reinterpret_cast<const float*>(full_values) + kv_head * head_dim + channel;
    // `entry` is a row_tokens entry (>= 0).
    auto load = [&](int entry) -> float {
        if constexpr (F == QSA_KV_F32) {
            return values[(unsigned)entry];
        } else {
            return qsa_kv_value<F>(full_values + (size_t)entry * row_bytes, n_kv_heads, head_dim,
                                   kv_head, channel);
        }
    };
    float denominator = 0.0f;
    float numerator = 0.0f;
    // Value loads run one batch ahead of the in-order accumulation.
    constexpr int UNROLL = 24;
    int i = 0;
    float value[UNROLL];
    if (selected_len >= UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const int offset = row_tokens[u];
            value[u] = load(offset < 0 ? 0 : offset);
        }
    }
    for (; i + UNROLL <= selected_len; i += UNROLL) {
        float next[UNROLL];
        if (i + 2 * UNROLL <= selected_len) {
#pragma unroll
            for (int u = 0; u < UNROLL; ++u) {
                const int offset = row_tokens[i + UNROLL + u];
                next[u] = load(offset < 0 ? 0 : offset);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            const float weight = row_scores[i + u];
            denominator += weight;
            QSA_SUM_ACC(numerator, weight, (row_tokens[i + u] < 0 ? 0.0f : value[u]));
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) value[u] = next[u];
    }
    for (; i < selected_len; ++i) {
        const int token = row_tokens[i];
        const float weight = row_scores[i];
        const float value = token < 0 ? 0.0f : load(token);
        denominator += weight;
        QSA_SUM_ACC(numerator, weight, value);
    }
    float gate = q[head_dim + channel];
    output_row[head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}

// Bitwise-equal to indexed_attention_attention_f32_batched_serial: every score
// is the same serial d-order dot, fmaxf is exact and order-independent, each
// row's weight is the same expf, and every channel sums its weights and
// products in selected-row order.  Weights are exponentiated once per row
// (not per channel), and the value loads of UNROLL rows are issued before
// their in-order accumulation (rows are kept as cache offsets, token *
// kv_stride).  Invalid rows keep weight 0 and select value
// 0, adding +0.0 to sums that are never -0.0. Each quantized format keeps the
// same equality with its serial twin.
template <int F>
static __device__ __forceinline__ void qsa_attention_batched(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    qsa_attention_row_parallel<F>(
        q_with_gate + row * 2 * n_heads * head_dim + (int)blockIdx.x * head_dim * 2,
        full_keys, full_values, selected_indices + row * capacity,
        output + row * n_heads * head_dim, selected_len, visible,
        n_heads, n_kv_heads, head_dim, full_capacity);
}

// Bitwise-equal to indexed_attention_attention_f32_batched_serial for
// head_dim == 256: every score is the same serial d-order dot and every
// softmax sum runs in selected-row order per channel. One workgroup owns four
// query heads of one KV group, so each key/value row is fetched once for all
// four, and expf runs once per (head, token) instead of once per channel.
// Each quantized format keeps the same equality with its serial twin.
#define QSA_HG 4
#define QSA_HD 256
template <int F>
static __device__ __forceinline__ void qsa_attention_batched_hg4(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    // Every region is a multiple of 16 bytes so float4 LDS accesses stay aligned.
    extern __shared__ float4 lds4[];
    float* lds = reinterpret_cast<float*>(lds4);
    float* weights = lds;                              // [selected_len][QSA_HG]
    float* kt = weights + QSA_HG * selected_len;       // [QSA_HG][256] partial maxes
    float* maxes = kt + QSA_HG * 256;                  // [QSA_HG]
    int* tokens = reinterpret_cast<int*>(maxes + QSA_HG);  // [selected_len]
    int tid = (int)threadIdx.x;
    int head0 = (int)blockIdx.x * QSA_HG;
    int kv_head = head0 / (n_heads / n_kv_heads);
    int kv_stride = n_kv_heads * QSA_HD;
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, QSA_HD);
    const float* keys = reinterpret_cast<const float*>(full_keys) + kv_head * QSA_HD;
    const float* values = reinterpret_cast<const float*>(full_values) + kv_head * QSA_HD;
    const int* row_selected = selected_indices + row * capacity;
    int q_row = row * 2 * n_heads * QSA_HD;
    // The query rows are block-uniform: the score loop reads them as scalar
    // loads (an SGPR FMA operand) instead of LDS broadcasts.
    const float* __restrict__ qg = q_with_gate + q_row + head0 * QSA_HD * 2;
    for (int i = tid; i < selected_len; i += 256) {
        int token = row_selected[i];
        tokens[i] = (token >= 0 && token < full_capacity && token < visible) ? token : -1;
    }
    __syncthreads();

    // Scores: each thread owns one selected row and all four heads, reading
    // its key straight from global memory (four independent serial d-order
    // chains; q comes from scalar loads).  No key tile, no barriers.
    float partial_max[QSA_HG];
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) partial_max[h] = -3.402823466e+38f;
    for (int base = 0; base < selected_len; base += 256) {
        const int index = base + tid;
        const int token = index < selected_len ? tokens[index] : -1;
        float score[QSA_HG];
#pragma unroll
        for (int h = 0; h < QSA_HG; ++h) score[h] = 0.0f;
        if (token >= 0) {
            if constexpr (F == QSA_KV_F32) {
                const float4* key4 = reinterpret_cast<const float4*>(keys + token * kv_stride);
#pragma unroll 4
                for (int d4 = 0; d4 < QSA_HD / 4; ++d4) {
                    const float4 k4 = key4[d4];
#pragma unroll
                    for (int h = 0; h < QSA_HG; ++h) {
                        const float4 q4 = reinterpret_cast<const float4*>(qg + h * QSA_HD * 2)[d4];
                        QSA_DOT_ACC(score[h], q4.x, k4.x);
                        QSA_DOT_ACC(score[h], q4.y, k4.y);
                        QSA_DOT_ACC(score[h], q4.z, k4.z);
                        QSA_DOT_ACC(score[h], q4.w, k4.w);
                    }
                }
            } else {
                qsa_kv_key_dots<F, QSA_HG>(qg, QSA_HD * 2, full_keys + (size_t)token * row_bytes,
                                           n_kv_heads, QSA_HD, kv_head, score);
            }
        }
        if (index < selected_len) {
#pragma unroll
            for (int h = 0; h < QSA_HG; ++h) {
                const float scaled = score[h] / sqrtf((float)QSA_HD);
                weights[index * QSA_HG + h] = scaled;
                if (token >= 0) partial_max[h] = fmaxf(partial_max[h], scaled);
            }
        }
    }
    // fmaxf is exact and order-independent, so the tree matches the serial scan.
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) kt[h * 256 + tid] = partial_max[h];
    __syncthreads();
    if (tid < QSA_HG) {
        float head_max = -3.402823466e+38f;
        for (int i = 0; i < 256; ++i) head_max = fmaxf(head_max, kt[tid * 256 + i]);
        maxes[tid] = head_max;
    }
    __syncthreads();
    for (int i = tid; i < QSA_HG * selected_len; i += 256) {
        if (tokens[i / QSA_HG] >= 0) weights[i] = expf(weights[i] - maxes[i % QSA_HG]);
    }
    __syncthreads();

    int channel = tid;
    float denominator[QSA_HG];
    float numerator[QSA_HG];
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) {
        denominator[h] = 0.0f;
        numerator[h] = 0.0f;
    }
    // Invalid (-1) rows hold weight 0 (their score was never exponentiated)
    // and read value 0, so they add +0.0 to sums that are never -0.0: the
    // accumulation is branch-free and still bitwise the skipping version.
    // Sixteen value loads are issued before their in-order accumulation.
    // Token indices are block-uniform: read them as scalars and fetch each
    // value row through a scalar buffer offset (invalid rows read row 0 and
    // select 0.0, exactly the value the skipped row contributed).
#if defined(__gfx942__)
    // gfx9 raw buffer descriptor dword3 (CK_TILE_BUFFER_RESOURCE_3RD_DWORD for
    // __gfx9__): DATA_FORMAT = 4 (BUF_DATA_FORMAT_32, bits 18:15), DST_SEL_* = 0,
    // NUM_FORMAT = 0. 0x31004000 is the RDNA3/RDNA4 layout, not valid on CDNA.
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x00020000);
#else
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x31004000);
#endif
    const int voff = channel * 4;
    auto value_at = [&](int index) {
        const int token = __builtin_amdgcn_readfirstlane(row_selected[index]);
        const bool ok = token >= 0 && token < full_capacity && token < visible;
        if constexpr (F == QSA_KV_F32) {
            const float v = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(
                vrsrc, voff, ok ? token * kv_stride * 4 : 0, 0));
            return ok ? v : 0.0f;
        } else {
            const float v = qsa_kv_value<F>(full_values + (size_t)(ok ? token : 0) * row_bytes,
                                            n_kv_heads, QSA_HD, kv_head, channel);
            return ok ? v : 0.0f;
        }
    };
    int i = 0;
    for (; i + 16 <= selected_len; i += 16) {
        float value[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) value[u] = value_at(i + u);
#pragma unroll
        for (int u = 0; u < 16; ++u) {
            const float4 w4 = *reinterpret_cast<const float4*>(weights + (i + u) * QSA_HG);
            denominator[0] += w4.x;
            QSA_SUM_ACC(numerator[0], w4.x, value[u]);
            denominator[1] += w4.y;
            QSA_SUM_ACC(numerator[1], w4.y, value[u]);
            denominator[2] += w4.z;
            QSA_SUM_ACC(numerator[2], w4.z, value[u]);
            denominator[3] += w4.w;
            QSA_SUM_ACC(numerator[3], w4.w, value[u]);
        }
    }
    for (; i < selected_len; ++i) {
        const float value = value_at(i);
        const float4 w4 = *reinterpret_cast<const float4*>(weights + i * QSA_HG);
        denominator[0] += w4.x;
        QSA_SUM_ACC(numerator[0], w4.x, value);
        denominator[1] += w4.y;
        QSA_SUM_ACC(numerator[1], w4.y, value);
        denominator[2] += w4.z;
        QSA_SUM_ACC(numerator[2], w4.z, value);
        denominator[3] += w4.w;
        QSA_SUM_ACC(numerator[3], w4.w, value);
    }
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) {
        int head = head0 + h;
        float gate = q_with_gate[q_row + head * QSA_HD * 2 + QSA_HD + channel];
        output[row * n_heads * QSA_HD + head * QSA_HD + channel] = denominator[h] > 0.0f
            ? numerator[h] / denominator[h] / (1.0f + expf(-gate))
            : 0.0f;
    }
}

#if defined(QSA_HAS_Q8)
// Q8 twin of qsa_attention_batched_hg4 with the same arithmetic in the same
// order (bitwise-equal output), restructured for less LDS and fewer exposed
// latencies (card-proven bit-identical, 0.467 mid / 0.438 late of the shipped
// kernel's time: R2Qsa-05-hg4-pipe-kpf):
//  - no tokens[] array in LDS: the validated token is recomputed from the
//    selected list where it is needed (one unsigned compare against
//    min(visible, full_capacity));
//  - LDS = [8 waves x 4 partial maxes, later the 4 denominators][selected x 4
//    weights]: the per-thread max array and its serial 256-step reduction are
//    replaced by wave-shuffle maxes (fmaxf is exact and order-independent);
//  - score loop: each thread owns one selected row (per 256-row tile) and the
//    next pair's 17 key words, the next tile's pair 0 and the token indices
//    two tiles ahead are requested before the current pair is decoded, with
//    sched_barrier(0) fences so the loads are neither sunk nor the q loads
//    hoisted into SGPR spills;
//  - value loads use a wave-uniform buffer descriptor on the wave's Q8_0
//    block of row 0 plus an SGPR row offset, in two 16-token ping-pong
//    batches (the loop body is one branch-free block so every value is
//    converted next to its load);
//  - every channel's denominator is the same in-order sum of the head's
//    weights, so wave 0 computes it once (one lane per head) and publishes
//    it through LDS instead of all eight waves adding it redundantly.
// Requires dynamic LDS of selected_len * 16 + 4 * 8 * 4 bytes.
// Bit-identity: score = the incumbent's qsa_kv_key_dots<Q8> chain per head
// (same code[] values, dot0/dot1 ascending d from +0, `total += dot0*scale0;
// total += dot1*scale1` per 64-wide pair in ascending pair order), then
// `/ sqrtf(256)`; weights are expf(scaled - max[h]); numerator/denominator
// keep their selected-row-ascending chains from +0.
template <bool B>
struct qsa_bool {
    static constexpr bool value = B;
};

// One 64-wide pair (17 words, 68 bytes) of a Q8 key row against HG query heads: the incumbent's code[] decode and
// dot chains. `total[h]` carries across pairs exactly like qsa_kv_key_dots<Q8, HG>.
template <int HG>
static __device__ __forceinline__ void qsa_lite_key_pair(
    const unsigned int* w, const float* __restrict__ q, int q_stride, int pair, float (&total)[HG]) {
    const float scale0 = qsa_half_bits(w[0] & 0xFFFFu);
    const float scale1 = qsa_half_bits(w[8] >> 16);
    float code[64];
    code[0] = qsa_q8_code(w[0], 2);
    code[1] = qsa_q8_code(w[0], 3);
#pragma unroll
    for (int i = 1; i < 8; ++i) {
#pragma unroll
        for (int b = 0; b < 4; ++b) code[4 * i - 2 + b] = qsa_q8_code(w[i], b);
    }
    code[30] = qsa_q8_code(w[8], 0);
    code[31] = qsa_q8_code(w[8], 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int b = 0; b < 4; ++b) code[32 + 4 * i + b] = qsa_q8_code(w[9 + i], b);
    }
    const float* qbase = q;
#pragma unroll
    for (int h = 0; h < HG; ++h) {
        const float* qh = qbase + h * q_stride + pair * 64;
        float dot0 = 0.0f;
        float dot1 = 0.0f;
#pragma unroll
        for (int d = 0; d < 32; ++d) dot0 += qh[d] * code[d];
#pragma unroll
        for (int d = 0; d < 32; ++d) dot1 += qh[32 + d] * code[32 + d];
        total[h] += dot0 * scale0;
        total[h] += dot1 * scale1;
        __builtin_amdgcn_sched_barrier(0);
    }
}

#define QSA_TOK_OK(t) ((unsigned)(t) < (unsigned)valid_limit)
template <int F>
static __device__ __forceinline__ void qsa_attention_batched_hg4_lite(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    static_assert(F == QSA_KV_Q8, "the lite hg4 kernel is the Q8 K/V twin");
    constexpr int HG = QSA_HG;
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    // token >= 0 && token < full_capacity && token < visible, as one unsigned compare.
    const int valid_limit = max(min(visible, full_capacity), 0);
    // LDS: [kt: 8 waves x HG partial maxes, later the HG denominators][weights: selected_len x HG].
    // Both regions are multiples of 16 bytes. No token array and no per-thread max array.
    extern __shared__ float4 lds4[];
    float* lds = reinterpret_cast<float*>(lds4);
    float* kt = lds;
    float* weights = lds + 8 * HG;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave_id = __builtin_amdgcn_readfirstlane(tid >> 5);  // wave-uniform
    const int head0 = (int)blockIdx.x * HG;
    const int kv_head = head0 / (n_heads / n_kv_heads);
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, QSA_HD);
    const int* row_selected = selected_indices + row * capacity;
    const int q_row = row * 2 * n_heads * QSA_HD;
    // The query rows are block-uniform: the score loop reads them as scalar loads (SGPR FMA operands).
    const float* __restrict__ qg = q_with_gate + q_row + head0 * QSA_HD * 2;
    // Validated token of selected row `index` (or -1), straight from the selected list.
    auto token_at = [&](int index) -> int {
        if (index >= selected_len) return -1;
        const int token = row_selected[index];
        return QSA_TOK_OK(token) ? token : -1;
    };

    // ---- scores: each thread owns one selected row (per 256-row tile) and all HG heads ----
    float partial_max[HG];
#pragma unroll
    for (int h = 0; h < HG; ++h) partial_max[h] = -3.402823466e+38f;
    // Token indices run two tiles ahead and the next tile's first key pair is requested before this tile is
    // decoded, so no tile waits on a dependent global load chain (index -> row -> key words).
    int tok0 = token_at(tid);
    int raw1 = row_selected[min(tid + 256, selected_len - 1)];  // next tile's raw index, validated one tile later
    unsigned int wt[17];  // pair 0 of this tile's key row
    {
        const unsigned int* p0 = reinterpret_cast<const unsigned int*>(
            full_keys + (size_t)(tok0 >= 0 ? tok0 : 0) * row_bytes + kv_head * (QSA_HD / 32) * 34);
#pragma unroll
        for (int i = 0; i < 17; ++i) wt[i] = p0[i];
    }
    for (int base = 0; base < selected_len; base += 256) {
        const int index = base + tid;
        const int token = tok0;
        tok0 = (index + 256 < selected_len && QSA_TOK_OK(raw1)) ? raw1 : -1;  // next tile's token
        raw1 = row_selected[min(index + 512, selected_len - 1)];            // the tile after: stays in flight
        unsigned int wx[17];  // pair 0 of the next tile's key row (row 0 stands in for a dead lane)
        {
            const unsigned int* px = reinterpret_cast<const unsigned int*>(
                full_keys + (size_t)(tok0 >= 0 ? tok0 : 0) * row_bytes + kv_head * (QSA_HD / 32) * 34);
#pragma unroll
            for (int i = 0; i < 17; ++i) wx[i] = px[i];
        }
        __builtin_amdgcn_sched_barrier(0);
        float score[HG];
#pragma unroll
        for (int h = 0; h < HG; ++h) score[h] = 0.0f;
        if (token >= 0) {
            const unsigned int* pairs = reinterpret_cast<const unsigned int*>(
                full_keys + (size_t)token * row_bytes + kv_head * (QSA_HD / 32) * 34);
            float total[HG];
#pragma unroll
            for (int h = 0; h < HG; ++h) total[h] = 0.0f;
            unsigned int wc[17];
#pragma unroll
            for (int i = 0; i < 17; ++i) wc[i] = wt[i];
#pragma unroll 1
            for (int pair = 0; pair < QSA_HD / 64; ++pair) {
                unsigned int wn[17];
                const int next_pair = pair + 1 < QSA_HD / 64 ? pair + 1 : QSA_HD / 64 - 1;
#pragma unroll
                for (int i = 0; i < 17; ++i) wn[i] = pairs[next_pair * 17 + i];
                __builtin_amdgcn_sched_barrier(0);
                qsa_lite_key_pair<HG>(wc, qg, QSA_HD * 2, pair, total);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int i = 0; i < 17; ++i) wc[i] = wn[i];
            }
#pragma unroll
            for (int h = 0; h < HG; ++h) score[h] += total[h];
        }
#pragma unroll
        for (int i = 0; i < 17; ++i) wt[i] = wx[i];
        if (index < selected_len) {
#pragma unroll
            for (int h = 0; h < HG; ++h) {
                const float scaled = score[h] / sqrtf((float)QSA_HD);
                weights[index * HG + h] = scaled;
                if (token >= 0) partial_max[h] = fmaxf(partial_max[h], scaled);
            }
        }
    }
    // fmaxf is exact and order-independent: reduce inside each wave with shuffles, then across the 8 waves.
#pragma unroll
    for (int h = 0; h < HG; ++h) {
        float m = partial_max[h];
        for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor(m, offset, 32));
        if (lane == 0) kt[wave_id * HG + h] = m;
    }
    __syncthreads();
    float maxes[HG];
#pragma unroll
    for (int h = 0; h < HG; ++h) {
        float head_max = -3.402823466e+38f;
#pragma unroll
        for (int w = 0; w < 8; ++w) head_max = fmaxf(head_max, kt[w * HG + h]);
        maxes[h] = head_max;
    }
    // exp phase: same thread/row mapping as the score loop (no barrier needed between them); the next row's raw
    // index is requested before this row's exps are computed.
    int exp_raw = row_selected[min(tid, selected_len - 1)];
    for (int index = tid; index < selected_len; index += 256) {
        const int exp_next = row_selected[min(index + 256, selected_len - 1)];
        const bool ok = QSA_TOK_OK(exp_raw);
        exp_raw = exp_next;
        if (ok) {
            float* wp = weights + index * HG;
            float4 w4 = *reinterpret_cast<const float4*>(wp);
            w4.x = expf(w4.x - maxes[0]);
            w4.y = expf(w4.y - maxes[1]);
            w4.z = expf(w4.z - maxes[2]);
            w4.w = expf(w4.w - maxes[3]);
            *reinterpret_cast<float4*>(wp) = w4;
        }
    }
    __syncthreads();

    // ---- PV: lane = channel; token indices are block-uniform scalar loads, value rows go through one
    // wave-uniform buffer descriptor + SGPR row offset (invalid rows read row 0 and select 0.0) ----
    const int channel = tid;
    float numerator[HG];
#pragma unroll
    for (int h = 0; h < HG; ++h) numerator[h] = 0.0f;
    const __amdgpu_buffer_rsrc_t qrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)(full_values + (size_t)(kv_head * (QSA_HD / 32) + wave_id) * 34), 0,
        0x7FFFFFFF, 0x31004000);
    auto value_at = [&](int index) {
        const int token = __builtin_amdgcn_readfirstlane(row_selected[index]);
        const bool ok = QSA_TOK_OK(token);
        const unsigned soff = ok ? (unsigned)token * (unsigned)row_bytes : 0u;
        const unsigned char code = __builtin_amdgcn_raw_buffer_load_b8(qrsrc, lane + 2, soff, 0);
        const unsigned short scale = __builtin_amdgcn_raw_buffer_load_b16(qrsrc, 0, soff, 0);
        const float v = (float)(signed char)code * qsa_half_bits(scale);
        return ok ? v : 0.0f;
    };
    // Wave 0, lane l runs head l's denominator chain (every channel's denominator is the same in-order sum).
    float den_lane = 0.0f;
    const int den_h = lane < HG ? lane : 0;
    auto accumulate = [&](auto with_den, int index, float value) {
        const float* wp = weights + index * HG;
        float wv[HG];
        const float4 w4 = *reinterpret_cast<const float4*>(wp);
        wv[0] = w4.x; wv[1] = w4.y; wv[2] = w4.z; wv[3] = w4.w;
        if constexpr (decltype(with_den)::value) den_lane += wp[den_h];
#pragma unroll
        for (int h = 0; h < HG; ++h) QSA_SUM_ACC(numerator[h], wv[h], value);
    };
    auto consume = [&](auto with_den, const float (&value)[16], int i0) {
#pragma unroll
        for (int u = 0; u < 16; ++u) accumulate(with_den, i0 + u, value[u]);
    };
    auto pv = [&](auto with_den) {
        int i = 0;
        // Two value batches in flight: the next batch's loads (and token-index scalar loads) are issued before
        // the current batch is consumed. The loop body is one branch-free block (indices past the end are clamped
        // to the last full batch: valid reads whose values are never consumed), so every value is converted next to its
        // load and no raw byte is carried across blocks.
        if (selected_len >= 16) {
            const int last_batch = selected_len - 16;
            float va[16];
            float vb[16];
#pragma unroll
            for (int u = 0; u < 16; ++u) va[u] = value_at(u);
            for (; i + 32 <= selected_len; i += 32) {
#pragma unroll
                for (int u = 0; u < 16; ++u) vb[u] = value_at(i + 16 + u);
                consume(with_den, va, i);
                const int next_batch = min(i + 32, last_batch);  // contiguous, so the index loads stay one scalar load
#pragma unroll
                for (int u = 0; u < 16; ++u) va[u] = value_at(next_batch + u);
                consume(with_den, vb, i + 16);
            }
            if (i + 16 <= selected_len) {
                consume(with_den, va, i);
                i += 16;
            }
        }
        for (; i < selected_len; ++i) {
            const float value = value_at(i);
            accumulate(with_den, i, value);
        }
    };
    if (wave_id == 0) pv(qsa_bool<true>{}); else pv(qsa_bool<false>{});
    // Every thread read the partial maxes in kt before the barrier that precedes PV: the region is free again.
    if (tid < HG) kt[tid] = den_lane;
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HG; ++h) {
        const float denominator = kt[h];
        const int head = head0 + h;
        const float gate = q_with_gate[q_row + head * QSA_HD * 2 + QSA_HD + channel];
        output[row * n_heads * QSA_HD + head * QSA_HD + channel] = denominator > 0.0f
            ? numerator[h] / denominator / (1.0f + expf(-gate))
            : 0.0f;
    }
}
#undef QSA_HD
#undef QSA_TOK_OK
#endif
#undef QSA_HG
#undef QSA_HD

// Bitwise-equal to indexed_attention_attention_f32_batched_hg4 (and so to the
// serial kernel) for head_dim 256 and twelve query heads per KV head: one
// workgroup owns the whole KV group, so every selected key and value row is
// read once instead of once per four heads. Each thread keeps the scores of
// its (up to QSA_T) selected rows for all twelve heads in registers — the
// same serial d-order dots — so no score array lives in LDS; the weights of
// one 256-row tile at a time go through LDS for the in-order channel sums.
#define QSA_G 12
#define QSA_T 9
#define QSA_HD 256
extern "C" __global__ __launch_bounds__(256) void indexed_attention_attention_f32_batched_hg12(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    __shared__ __align__(16) float weights[256 * QSA_G];  // one tile, [row][head]
    __shared__ float wave_max[8 * QSA_G];
    __shared__ float maxes[QSA_G];
    const int tid = (int)threadIdx.x;
    const int kv_head = (int)blockIdx.x;
    const int head0 = kv_head * QSA_G;
    const int kv_stride = n_kv_heads * QSA_HD;
    const float* keys = full_keys + kv_head * QSA_HD;
    const float* values = full_values + kv_head * QSA_HD;
    const int* row_selected = selected_indices + row * capacity;
    const int q_row = row * 2 * n_heads * QSA_HD;
    const float* __restrict__ qg = q_with_gate + q_row + head0 * QSA_HD * 2;

    float scores[QSA_T][QSA_G];
    unsigned valid_rows = 0u;  // bit t: this thread's tile-t row is a live key
    float partial_max[QSA_G];
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) partial_max[h] = -3.402823466e+38f;
#pragma unroll
    for (int t = 0; t < QSA_T; ++t) {
        const int index = t * 256 + tid;
        int token = index < selected_len ? row_selected[index] : -1;
        if (!(token >= 0 && token < full_capacity && token < visible)) token = -1;
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) scores[t][h] = 0.0f;
        if (token >= 0) {
            const float4* key4 = reinterpret_cast<const float4*>(keys + token * kv_stride);
#pragma unroll 2
            for (int d4 = 0; d4 < QSA_HD / 4; ++d4) {
                const float4 k4 = key4[d4];
#pragma unroll
                for (int h = 0; h < QSA_G; ++h) {
                    const float4 q4 = reinterpret_cast<const float4*>(qg + h * QSA_HD * 2)[d4];
                    QSA_DOT_ACC(scores[t][h], q4.x, k4.x);
                    QSA_DOT_ACC(scores[t][h], q4.y, k4.y);
                    QSA_DOT_ACC(scores[t][h], q4.z, k4.z);
                    QSA_DOT_ACC(scores[t][h], q4.w, k4.w);
                }
            }
        }
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) {
            scores[t][h] = scores[t][h] / sqrtf((float)QSA_HD);
            if (token >= 0) partial_max[h] = fmaxf(partial_max[h], scores[t][h]);
        }
        if (token >= 0) valid_rows |= 1u << t;
    }
    // fmaxf is exact and order-independent: any reduction tree matches.
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        float m = partial_max[h];
        for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor(m, offset, 32));
        partial_max[h] = m;
    }
    if ((tid & 31) == 0) {
#pragma unroll
        for (int h = 0; h < QSA_G; ++h) wave_max[(tid >> 5) * QSA_G + h] = partial_max[h];
    }
    __syncthreads();
    if (tid < QSA_G) {
        float m = -3.402823466e+38f;
        for (int w = 0; w < 8; ++w) m = fmaxf(m, wave_max[w * QSA_G + tid]);
        maxes[tid] = m;
    }
    __syncthreads();

    const int channel = tid;
    float denominator[QSA_G];
    float numerator[QSA_G];
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        denominator[h] = 0.0f;
        numerator[h] = 0.0f;
    }
#if defined(__gfx942__)
    // gfx9 raw buffer descriptor dword3 (CK_TILE_BUFFER_RESOURCE_3RD_DWORD for
    // __gfx9__): DATA_FORMAT = 4 (BUF_DATA_FORMAT_32, bits 18:15), DST_SEL_* = 0,
    // NUM_FORMAT = 0. 0x31004000 is the RDNA3/RDNA4 layout, not valid on CDNA.
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x00020000);
#else
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x31004000);
#endif
    const int voff = channel * 4;
    auto value_at = [&](int index) {
        const int token = __builtin_amdgcn_readfirstlane(row_selected[index]);
        const bool ok = token >= 0 && token < full_capacity && token < visible;
        const float v = __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(
            vrsrc, voff, ok ? token * kv_stride * 4 : 0, 0));
        return ok ? v : 0.0f;
    };
#pragma unroll
    for (int t = 0; t < QSA_T; ++t) {
        const int base = t * 256;
        if (base >= selected_len) break;
        // Invalid rows keep weight 0: their score is never exponentiated.
        const bool valid = (valid_rows >> t) & 1u;
#pragma unroll
        for (int h = 0; h < QSA_G; ++h)
            weights[tid * QSA_G + h] = valid ? expf(scores[t][h] - maxes[h]) : 0.0f;
        __syncthreads();
        const int count = min(256, selected_len - base);
        int u = 0;
        for (; u + 8 <= count; u += 8) {
            float value[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) value[k] = value_at(base + u + k);
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float4* w4 = reinterpret_cast<const float4*>(weights + (u + k) * QSA_G);
#pragma unroll
                for (int q = 0; q < QSA_G / 4; ++q) {
                    const float4 w = w4[q];
                    denominator[4 * q] += w.x;
                    QSA_SUM_ACC(numerator[4 * q], w.x, value[k]);
                    denominator[4 * q + 1] += w.y;
                    QSA_SUM_ACC(numerator[4 * q + 1], w.y, value[k]);
                    denominator[4 * q + 2] += w.z;
                    QSA_SUM_ACC(numerator[4 * q + 2], w.z, value[k]);
                    denominator[4 * q + 3] += w.w;
                    QSA_SUM_ACC(numerator[4 * q + 3], w.w, value[k]);
                }
            }
        }
        for (; u < count; ++u) {
            const float value = value_at(base + u);
            const float4* w4 = reinterpret_cast<const float4*>(weights + u * QSA_G);
#pragma unroll
            for (int q = 0; q < QSA_G / 4; ++q) {
                const float4 w = w4[q];
                denominator[4 * q] += w.x;
                QSA_SUM_ACC(numerator[4 * q], w.x, value);
                denominator[4 * q + 1] += w.y;
                QSA_SUM_ACC(numerator[4 * q + 1], w.y, value);
                denominator[4 * q + 2] += w.z;
                QSA_SUM_ACC(numerator[4 * q + 2], w.z, value);
                denominator[4 * q + 3] += w.w;
                QSA_SUM_ACC(numerator[4 * q + 3], w.w, value);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = 0; h < QSA_G; ++h) {
        int head = head0 + h;
        float gate = q_with_gate[q_row + head * QSA_HD * 2 + QSA_HD + channel];
        output[row * n_heads * QSA_HD + head * QSA_HD + channel] = denominator[h] > 0.0f
            ? numerator[h] / denominator[h] / (1.0f + expf(-gate))
            : 0.0f;
    }
}
#undef QSA_G
#undef QSA_DOT_ACC
#undef QSA_SUM_ACC
#undef QSA_T
#undef QSA_HD

template <int F>
static __device__ __forceinline__ void qsa_attention_serial(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int selected_len,
    int full_capacity) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    int kv_head = head / (n_heads / n_kv_heads);
    const float* keys = reinterpret_cast<const float*>(full_keys);
    const float* values = reinterpret_cast<const float*>(full_values);
    const size_t row_bytes = (size_t)qsa_kv_row_bytes<F>(n_kv_heads, head_dim);
    auto key_score = [&](int token) {
        float score = 0.0f;
        if constexpr (F == QSA_KV_F32) {
            for (int d = 0; d < head_dim; ++d) {
                score += q_with_gate[head * head_dim * 2 + d]
                    * keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
            }
        } else {
            float dots[1] = {0.0f};
            qsa_kv_key_dots<F, 1>(q_with_gate + head * head_dim * 2, 0,
                                  full_keys + (size_t)token * row_bytes, n_kv_heads, head_dim,
                                  kv_head, dots);
            score = dots[0];
        }
        return score;
    };
    auto value_of = [&](int token) {
        if constexpr (F == QSA_KV_F32) {
            return values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
        } else {
            return qsa_kv_value<F>(full_values + (size_t)token * row_bytes, n_kv_heads, head_dim,
                                   kv_head, channel);
        }
    };
    float max_score = -3.402823466e+38f;
    for (int row = 0; row < selected_len; ++row) {
        int token = selected_indices[row];
        if (token < 0 || token >= full_capacity) continue;
        float score = key_score(token);
        max_score = fmaxf(max_score, score / sqrtf((float)head_dim));
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int row = 0; row < selected_len; ++row) {
        int token = selected_indices[row];
        if (token < 0 || token >= full_capacity) continue;
        float score = key_score(token);
        float weight = expf(score / sqrtf((float)head_dim) - max_score);
        denominator += weight;
        numerator += weight * value_of(token);
    }
    float gate = q_with_gate[head * head_dim * 2 + head_dim + channel];
    output[head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}

// Single query row with an explicit selection length (the MTP head's reused
// selections do not follow the batched length formula); same arithmetic as
// the batched kernel.
template <int F>
static __device__ __forceinline__ void qsa_attention_single(
    const float* q_with_gate,
    const unsigned char* full_keys,
    const unsigned char* full_values,
    const int* selected_indices,
    float* output,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int selected_len,
    int full_capacity) {
    qsa_attention_row_parallel<F>(
        q_with_gate + (int)blockIdx.x * head_dim * 2, full_keys, full_values,
        selected_indices, output, selected_len, full_capacity,
        n_heads, n_kv_heads, head_dim, full_capacity);
}

// One exported kernel per (body, format): F32 keeps its historical names and
// float cache pointers, the quantized formats take byte rows.
#define QSA_ATTENTION_BATCHED(NAME, BODY, BOUNDS, FORMAT, CACHE)                            \
    extern "C" __global__ BOUNDS void NAME(                                                 \
        const float* q_with_gate, const CACHE* full_keys, const CACHE* full_values,        \
        const int* selected_indices, float* output, int rows, int position_start,          \
        int n_heads, int n_kv_heads, int head_dim, int budget_blocks, int compress,        \
        int capacity, int full_capacity) {                                                  \
        BODY<FORMAT>(q_with_gate, reinterpret_cast<const unsigned char*>(full_keys),        \
                     reinterpret_cast<const unsigned char*>(full_values), selected_indices, \
                     output, rows, position_start, n_heads, n_kv_heads, head_dim,           \
                     budget_blocks, compress, capacity, full_capacity);                     \
    }
#define QSA_ATTENTION_SINGLE(NAME, BODY, BOUNDS, FORMAT, CACHE)                             \
    extern "C" __global__ BOUNDS void NAME(                                                 \
        const float* q_with_gate, const CACHE* full_keys, const CACHE* full_values,        \
        const int* selected_indices, float* output, int n_heads, int n_kv_heads,           \
        int head_dim, int selected_len, int full_capacity) {                                \
        BODY<FORMAT>(q_with_gate, reinterpret_cast<const unsigned char*>(full_keys),        \
                     reinterpret_cast<const unsigned char*>(full_values), selected_indices, \
                     output, n_heads, n_kv_heads, head_dim, selected_len, full_capacity);   \
    }
#define QSA_NO_BOUNDS
#define QSA_ATTENTION_KERNELS(TAG, FORMAT, CACHE)                                           \
    QSA_ATTENTION_BATCHED(indexed_attention_attention_##TAG##_batched_serial,               \
                          qsa_attention_batched_serial, QSA_NO_BOUNDS, FORMAT, CACHE)       \
    QSA_ATTENTION_BATCHED(indexed_attention_attention_##TAG##_batched,                      \
                          qsa_attention_batched, __launch_bounds__(256), FORMAT, CACHE)     \
    QSA_ATTENTION_BATCHED(indexed_attention_attention_##TAG##_batched_hg4,                  \
                          qsa_attention_batched_hg4, __launch_bounds__(256), FORMAT, CACHE) \
    QSA_ATTENTION_SINGLE(indexed_attention_attention_##TAG##_serial, qsa_attention_serial,  \
                         QSA_NO_BOUNDS, FORMAT, CACHE)                                      \
    QSA_ATTENTION_SINGLE(indexed_attention_attention_##TAG, qsa_attention_single,           \
                         __launch_bounds__(256), FORMAT, CACHE)
QSA_ATTENTION_KERNELS(f32, QSA_KV_F32, float)
#if defined(__GFX12__)
QSA_ATTENTION_KERNELS(fp8, QSA_KV_FP8, unsigned char)
#endif
#if defined(__gfx942__)
QSA_ATTENTION_KERNELS(fp8_fnuz, QSA_KV_FP8_FNUZ, unsigned char)
#endif
#if defined(QSA_HAS_Q8)
QSA_ATTENTION_KERNELS(q8, QSA_KV_Q8, unsigned char)
// Same arithmetic as indexed_attention_attention_q8_batched_hg4, less LDS
// (selected_len * 16 + 128 bytes) and pipelined key/value gathers: routed on
// exact gfx1100 prefill only.
QSA_ATTENTION_BATCHED(indexed_attention_attention_q8_batched_hg4_lite,
                      qsa_attention_batched_hg4_lite, __launch_bounds__(256), QSA_KV_Q8,
                      unsigned char)
#endif
#undef QSA_ATTENTION_KERNELS
#undef QSA_NO_BOUNDS
#undef QSA_ATTENTION_SINGLE
#undef QSA_ATTENTION_BATCHED
extern "C" __global__ void scale_f32(
    float* values,
    int elements,
    float scale) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) values[index] *= scale;
}
// HC-specific pointwise fusion.  This preserves the source order of the
// three F32->BF16->F32 boundaries around scale and SiLU while replacing the
// per-row roundtrip, flat scale, flat SiLU, and per-row roundtrip launches
// with one launch over the contiguous low-rank buffer.  With `bf16_out` the
// (BF16-exact) results go there as packed BF16 bits and `values` is left as is.
extern "C" __global__ void hc_activation_fused_f32(
    float* values,
    int elements,
    float scale,
    unsigned short* bf16_out) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= elements) return;

    float value = bf16_to_f32(bf16_round(values[index]));
    value *= scale;
    value = bf16_to_f32(bf16_round(value));
    value = value / (1.0f + expf(-value));
    if (bf16_out)
        bf16_out[index] = bf16_round(value).data;
    else
        values[index] = bf16_to_f32(bf16_round(value));
}

// Deterministic device-side greedy top-1 over one or more contiguous logits
// rows.  Ties retain the lowest vocabulary id, matching the reference route.
// Row argmax with the host sampler's semantics (llama::argmax): the first
// index of the largest finite value; non-finite values never win, and a row
// with no finite value yields 0. Ties resolve to the lowest index, so the
// reduction shape does not matter. Block <= 1024 threads.
extern "C" __global__ void argmax_f32(
    const float* logits,
    int* indices,
    int rows,
    int vocab) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    __shared__ float values[1024];
    __shared__ int ids[1024];
    float best = -INFINITY;
    int best_id = 0;
    int base = row * vocab;
#pragma unroll 8
    for (int id = (int)threadIdx.x; id < vocab; id += (int)blockDim.x) {
        float value = logits[base + id];
        if (isfinite(value) && value > best) {
            best = value;
            best_id = id;
        }
    }
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            float other = values[threadIdx.x + stride];
            int other_id = ids[threadIdx.x + stride];
            if (other > values[threadIdx.x]
                || (other == values[threadIdx.x] && other_id < ids[threadIdx.x])) {
                values[threadIdx.x] = other;
                ids[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) indices[row] = ids[0];
}
