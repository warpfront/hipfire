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
// scale * code. Every token requantizes each row, by the row's absmax with
// Qwen3.5's stochastic rounding (floor(x * 127 / absmax + u), u uniform in
// [0, 1) from an LCG seeded by head, row, key channel and the token's
// position `frame`), so repeated decay does not bias the stored state. The
// few-row persistent route does the same per row, which keeps an MTP verify
// bitwise the single-row decode route; only the chunked prefill scan
// requantizes once, after its last token.
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
template <bool kQ8>
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
    float* out_value) {
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
    if constexpr (kQ8) {
        // The thread's value row: 128 key-channel codes, read 16 at a time
        // (a 16-byte word per step, from cache after the first pass) and
        // dequantized on use. The update is the F32 route's expression on
        // the dequantized state; the row is requantized from the updated
        // values (recomputed, not stored) after the output.
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
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
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
        for (int word = 0; word < 8; ++word) {
            const int4 codes = row[word];
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
// gdn_q8_row) and `frame` is the last row's position; the recurrence runs on
// the dequantized F32 values, requantized after every row with that row's
// position as a decode step would.
template <bool kCapture, bool kQ8 = false>
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
    unsigned int frame = 0u) {
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
    // kQ8: the codes and scale of the state after the latest row.
    int4 packed[4];
    signed char* const codes = reinterpret_cast<signed char*>(packed);
    float row_scale = 0.0f;
    __shared__ float peak_half[2][128];
    if constexpr (kQ8) {
        const int4* codes4 =
            reinterpret_cast<const int4*>(gdn_q8_row(state, value_head, value_index) + kd_base);
        row_scale = gdn_q8_scales(state, value_heads)[value_head * 128 + value_index];
        #pragma unroll
        for (int i = 0; i < 4; ++i) packed[i] = codes4[i];
        #pragma unroll
        for (int kd = 0; kd < 64; ++kd) state_half[kd] = row_scale * (float)codes[kd];
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

            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                state_half[kd] *= decay;
            }

            float kv_mem = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
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
                    kv_mem += state_half[kd] * kv;
                }
                const __hip_bfloat16 v_value = bf16_round(cur_a);
                const float delta =
                    (bf16_to_f32(v_value) - kv_mem) * beta_value;
                handoff_delta[value_index] = delta;
            }
            gdn_lds_barrier();

            const float delta = handoff_delta[value_index];
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
                state_half[kd] += kv * delta;
            }

            float out = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                float qv = bf16_to_f32(q_norm[buf][kd_base + kd]);
                qv *= query_scale;
                out += state_half[kd] * qv;
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
                    qv *= query_scale;
                    out += state_half[kd] * qv;
                }
                output[row * value_width + value_head * 128 + value_index] = out;
            }
            if constexpr (kQ8) {
                // Requantize as a decode step at this row's position does:
                // the row's absmax spans both halves (exchange the half
                // peaks), and the next row decays the dequantized codes.
                float peak = 0.0f;
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) peak = fmaxf(peak, fabsf(state_half[kd]));
                peak_half[half][value_index] = peak;
                gdn_lds_barrier();
                peak = fmaxf(peak_half[0][value_index], peak_half[1][value_index]);
                const float inv = gdn_q8_inv(peak);
                row_scale = gdn_q8_scale(peak);
                const unsigned int row_frame = frame - (unsigned int)(rows - 1 - row);
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) {
                    codes[kd] = gdn_q8_round(state_half[kd], inv, value_head, value_index,
                                             kd_base + kd, row_frame);
                    state_half[kd] = row_scale * (float)codes[kd];
                }
            }
        }
    }

    if constexpr (kQ8) {
        void* const final_slot = kCapture
            ? reinterpret_cast<signed char*>(row_states)
                  + (size_t)(rows - 1) * gdn_q8_slot_bytes(value_heads)
            : state;
        int4* dst =
            reinterpret_cast<int4*>(gdn_q8_row(final_slot, value_head, value_index) + kd_base);
        #pragma unroll
        for (int i = 0; i < 4; ++i) dst[i] = packed[i];
        if (half == 0) {
            gdn_q8_scales(final_slot, value_heads)[value_head * 128 + value_index] = row_scale;
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
// ring of slots); `frame` seeds the final requantization.
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

// gated_delta_rollback_layers_f32 over rings of Q8 GDN state slots.
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
// eight adjacent columns per thread (16-byte loads and store, one gate per
// thread), the same expression per column, the result stored as BF16 bits.
// `hidden` % 8 == 0.  Grid [ceil(branches * hidden / 2048), rows], block 256.
extern "C" __global__ void hyper_write_bf16x8(
    const uint4* input,
    const float* mixed,
    const float* gates,
    uint4* output,
    int branches,
    int hidden,
    int rows) {
    const int column = 8 * (int)(blockIdx.x * blockDim.x + threadIdx.x);
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
    const float* m = mixed + (size_t)row * hidden + column % hidden;
    const float4 m0 = *(const float4*)m, m1 = *(const float4*)(m + 4);
    const float mv[8] = {m0.x, m0.y, m0.z, m0.w, m1.x, m1.y, m1.z, m1.w};
    const size_t at = ((size_t)row * wide + column) >> 3;
    const uint4 w = input[at];
    const unsigned int wv[4] = {w.x, w.y, w.z, w.w};
    unsigned int ov[4];
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const float lo = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(mv[2 * p]))))
            + __uint_as_float(wv[p] << 16);
        const float hi = bf16_to_f32(bf16_round(gate * bf16_to_f32(bf16_round(mv[2 * p + 1]))))
            + __uint_as_float(wv[p] & 0xFFFF0000u);
        ov[p] = (unsigned int)bf16_round(lo).data | ((unsigned int)bf16_round(hi).data << 16);
    }
    output[at] = make_uint4(ov[0], ov[1], ov[2], ov[3]);
}

// `normalized16`, when non-null, also receives each normalized value rounded
// to F16 (RNE, as convert_f32_to_f16) for the F16 WMMA projection, at row
// pitch `ld16` elements, and `normalized`, when non-null, then holds the
// (already BF16-rounded) values as BF16 bits.
template <typename T>
static __device__ __forceinline__ void hyper_norm_rows(
    const T* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows,
    _Float16* normalized16,
    int ld16) {
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
                normalized16[(size_t)row * ld16 + branch * hidden + j] = (_Float16)value;
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
    int state_bf16,
    int ld16) {
    if (state_bf16)
        hyper_norm_rows((const unsigned short*)input, norm_weight, normalized, branches, hidden,
                        rows, normalized16, ld16);
    else
        hyper_norm_rows((const float*)input, norm_weight, normalized, branches, hidden, rows,
                        normalized16, ld16);
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
// With `normalized16` non-null the block also writes the row normalized with
// `read_norm_weight` as F16 at row pitch `ld16`: hyper_norm_f32's F16 output
// for a hyper read of the same streams (same inverse RMS, bitwise).
// Grid [rows], block 256, dynamic LDS (2 * hidden + 4 * 256) * 4 bytes.
// Requires four branches, hidden % 256 == 0 and hidden <= 2560.
// `input` is the HC streams (BF16 bits when `state_bf16`).
extern "C" __global__ __launch_bounds__(256) void hyper_norm_gate_f32(
    const void* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gates,
    int hidden,
    int state_bf16,
    const __hip_bfloat16* read_norm_weight,
    _Float16* normalized16,
    int ld16) {
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
    if (state_bf16 && normalized16) {
        // The BF16 row is still in LDS: each thread normalizes eight
        // consecutive values, so the weights load and the F16 read output
        // store as 16-byte runs (the strided per-value form spent most of
        // the kernel on 2-byte stores).  Same expressions, same values.
        for (int v = tid; v < wide / 8; v += 256) {
            const int j0 = v * 8;
            const int b = j0 / hidden;
            const float inv = rsqrtf(rms_sum[b * 256] / (float)hidden + 1.0e-6f);
            const uint4 rv = ((const uint4*)xsb)[v];
            const uint4 wv = *(const uint4*)(norm_weight + j0);
            const uint4 qv = *(const uint4*)(read_norm_weight + j0);
            const unsigned int rw[4] = {rv.x, rv.y, rv.z, rv.w};
            const unsigned int ww[4] = {wv.x, wv.y, wv.z, wv.w};
            const unsigned int qw[4] = {qv.x, qv.y, qv.z, qv.w};
            unsigned int ow[4];
            _Float16 oq[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int sh = (e & 1) ? 0 : 16;
                const float x = __uint_as_float((rw[e >> 1] << sh) & 0xFFFF0000u);
                const float w = __uint_as_float((ww[e >> 1] << sh) & 0xFFFF0000u);
                const float q = __uint_as_float((qw[e >> 1] << sh) & 0xFFFF0000u);
                const unsigned int nb =
                    __float_as_uint(bf16_to_f32(bf16_round(x * inv * (1.0f + w)))) >> 16;
                ow[e >> 1] = (e & 1) ? (ow[e >> 1] | (nb << 16)) : nb;
                oq[e] = (_Float16)bf16_to_f32(bf16_round(x * inv * (1.0f + q)));
            }
            ((uint4*)xsb)[v] = make_uint4(ow[0], ow[1], ow[2], ow[3]);
            *(uint4*)(normalized16 + (size_t)row * ld16 + j0) = __builtin_bit_cast(uint4, oq);
        }
    } else {
#pragma unroll
        for (int b = 0; b < branches; ++b) {
            const float inv = rsqrtf(rms_sum[b * 256] / (float)hidden + 1.0e-6f);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int j = b * hidden + tid + 256 * u;
                if (u < per) {
                    xsb[j] = (unsigned short)(__float_as_uint(bf16_to_f32(bf16_round(
                        raw[b][u] * inv * (1.0f + bf16_to_f32(norm_weight[j]))))) >> 16);
                    if (normalized16)
                        normalized16[(size_t)row * ld16 + j] = (_Float16)bf16_to_f32(bf16_round(
                            raw[b][u] * inv * (1.0f + bf16_to_f32(read_norm_weight[j]))));
                }
            }
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
// Optional gate parameters (`param_elements` > 0): blocks past the channel
// grid run gated_delta_params_bf16_f32_batched's body (block row 0 only).
// Grid [ceil(channels / 256) + param blocks, ceil(rows / 16)], block 256.
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
    int heads) {
    const int conv_blocks = (channels + 255) / 256;
    if ((int)blockIdx.x >= conv_blocks) {
        const int flat = ((int)blockIdx.x - conv_blocks) * 256 + (int)threadIdx.x;
        if (blockIdx.y == 0 && flat < param_elements)
            gated_delta_params_flat(param_a, param_b, a_log, dt_bias, gate, beta, flat, heads);
        return;
    }
    const int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int r0 = (int)blockIdx.y * 16;
    if (channel >= channels || r0 >= rows
        || history_rows != 3 || kernel_size != 4) {
        return;
    }
    // `input_bf16`: the projection is stored as BF16 bits (its rounded values).
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
            b.data = ((const unsigned short*)input)[j * channels + channel];
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

// gated_delta_conv_bf16_f32_batched_k4 with BF16 input and output, four
// channels per thread (8-byte row accesses instead of 2-byte ones); every
// channel's arithmetic is the scalar kernel's.  The gate parameters take the
// blocks past the channel grid on every row of the grid, not just row 0.
// Grid [channels / 1024 + param blocks per row, ceil(rows / 8)], block 256.
// channels % 4 == 0.
extern "C" __global__ void gated_delta_conv_bf16x4_batched_k4(
    const unsigned short* input,
    const __hip_bfloat16* kernel,
    const float* history,
    unsigned short* output,
    float* next_history,
    int rows,
    int channels,
    int start_cursor,
    const float* param_a,
    const float* param_b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int param_elements,
    int heads) {
    const int conv_blocks = (channels / 4 + 255) / 256;
    if ((int)blockIdx.x >= conv_blocks) {
        const int per_row = (int)gridDim.x - conv_blocks;
        const int flat = ((int)blockIdx.y * per_row + (int)blockIdx.x - conv_blocks) * 256 + (int)threadIdx.x;
        if (flat < param_elements)
            gated_delta_params_flat(param_a, param_b, a_log, dt_bias, gate, beta, flat, heads);
        return;
    }
    const int channel = ((int)blockIdx.x * 256 + (int)threadIdx.x) * 4;
    const int r0 = (int)blockIdx.y * 8;
    if (channel >= channels || r0 >= rows) return;
    struct bf4 { __hip_bfloat16 v[4]; };
    auto in = [&](int j) {
        const uint2 u = *(const uint2*)(input + (size_t)j * channels + channel);
        bf4 o;
        o.v[0].data = (unsigned short)(u.x & 0xFFFFu);
        o.v[1].data = (unsigned short)(u.x >> 16);
        o.v[2].data = (unsigned short)(u.y & 0xFFFFu);
        o.v[3].data = (unsigned short)(u.y >> 16);
        return o;
    };
    auto x = [&](int j) {
        if (j >= 0) return in(j);
        const float4 h = *(const float4*)(history + ((start_cursor + j + 3) % 3) * channels + channel);
        bf4 o;
        o.v[0] = bf16_round(h.x);
        o.v[1] = bf16_round(h.y);
        o.v[2] = bf16_round(h.z);
        o.v[3] = bf16_round(h.w);
        return o;
    };
    float k[4][4];
    for (int c = 0; c < 4; ++c)
        for (int t = 0; t < 4; ++t) k[c][t] = bf16_to_f32(kernel[(channel + c) * 4 + t]);
    bf4 h0 = x(r0 - 3), h1 = x(r0 - 2), h2 = x(r0 - 1);
    bf4 tail[3];
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s) {
            const int j = rows - 3 + ((s - (start_cursor + rows) % 3) + 6) % 3;
            tail[s] = x(j);
        }
    }
    const int r1 = min(r0 + 8, rows);
#pragma unroll
    for (int r = r0; r < r0 + 8; ++r) {
        if (r >= r1) break;
        const bf4 current = in(r);
        unsigned short o[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float value = k[c][3] * bf16_to_f32(current.v[c]);
            value += k[c][0] * bf16_to_f32(h0.v[c]);
            value += k[c][1] * bf16_to_f32(h1.v[c]);
            value += k[c][2] * bf16_to_f32(h2.v[c]);
            o[c] = bf16_round(value * (1.0f / (1.0f + expf(-value)))).data;
        }
        *(uint2*)(output + (size_t)r * channels + channel) =
            make_uint2(o[0] | ((unsigned)o[1] << 16), o[2] | ((unsigned)o[3] << 16));
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    if (r0 == 0) {
        for (int s = 0; s < 3; ++s)
            *(float4*)(next_history + s * channels + channel) = make_float4(
                bf16_to_f32(tail[s].v[0]), bf16_to_f32(tail[s].v[1]),
                bf16_to_f32(tail[s].v[2]), bf16_to_f32(tail[s].v[3]));
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
//     tile, wave w -> head blockIdx.x * 2 + (w & 1), rows r0 + 4 * (w >> 1)
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
// Grid [ceil(channels / 256) + param blocks, ceil(rows / 16)], block 256
// (identical to the plain kernel's launch).  wave32 (the DPP rows/xor-16
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
    __hip_bfloat16* kn) {
    const int conv_blocks = (channels + 255) / 256;
    if ((int)blockIdx.x >= conv_blocks) {
        const int flat = ((int)blockIdx.x - conv_blocks) * 256 + (int)threadIdx.x;
        if (blockIdx.y == 0 && flat < param_elements)
            gated_delta_params_flat(param_a, param_b, a_log, dt_bias, gate, beta, flat, heads);
        return;
    }
    const int r0 = (int)blockIdx.y * 16;
    if (r0 >= rows || history_rows != 3 || kernel_size != 4) return;
    const int store_bf16 = output_bf16 & 1;

    // 128-channel slices (4 contiguous channels per lane) x 4-row quarters.  The
    // q / k heads (blocks 0..15) are also normalized; the V channels use the
    // same tile when channels is a multiple of 256, else the scalar body below.
    const bool norm = (int)blockIdx.x < 16 && channels >= 4096;
    if (norm || (channels & 255) == 0) {
        const int wave = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31);
        const int head = (int)blockIdx.x * 2 + (wave & 1);
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
                const uint2 w = *(const uint2*)(in_u16 + (size_t)j * channels + c0);
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
    const int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    auto in = [&](int j) {
        if (input_bf16) {
            __hip_bfloat16 b;
            b.data = ((const unsigned short*)input)[j * channels + channel];
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
template <bool kQ8>
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
    unsigned int frame) {
    __shared__ float head_values[128];
    __shared__ int rotate_pair;
    float out;
    if (!gated_delta_step_norm128_body<kQ8>(q, k, v, gate, beta, state, key_heads, value_heads,
                                            key_dim, value_dim, query_scale, frame, &out))
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
    int rotary_dim) {
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
        float sine = sinf(angle);
        float cosine = cosf(angle);
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
// An fp8 key score is the head's scale times the in-order sum of q * code; a
// value is scale * code. Every attention kernel goes through the helpers
// below, so for each format the per-head, grouped and serial kernels stay
// bitwise equal. The indexer's raw and pooled keys are BF16 values in every
// format; the fp8 format stores them as BF16, which holds the same values as
// the F32 arenas.
#define QSA_KV_F32 0
#define QSA_KV_FP8 1
#if defined(__GFX12__)
#define QSA_HAS_FP8 1
#endif

typedef float qsa_f32x2 __attribute__((ext_vector_type(2)));

template <int F>
static __device__ __forceinline__ int qsa_kv_row_bytes(int kv_heads, int head_dim) {
    if constexpr (F == QSA_KV_F32) {
        return kv_heads * head_dim * 4;
    } else {
        return kv_heads * (head_dim + 2);
    }
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

// Value channel `channel` of KV head `kv_head` in the fp8 token row `row`.
template <int F>
static __device__ __forceinline__ float qsa_kv_value(
    const unsigned char* row,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel) {
    static_assert(F == QSA_KV_FP8, "F32 values keep each kernel's own loads");
#if defined(QSA_HAS_FP8)
    const float scale = qsa_half_bits(*reinterpret_cast<const unsigned short*>(
        row + kv_heads * head_dim + 2 * kv_head));
    return qsa_fp8_code(row[kv_head * head_dim + channel]) * scale;
#else
    return 0.0f;
#endif
}

// Unscaled QK dot of `heads` query rows (`q + h * q_stride`, head_dim floats
// each) with KV head `kv_head` of the fp8 token row `row`, added to
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
    static_assert(F == QSA_KV_FP8, "F32 keys keep each kernel's own loop");
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
    for (int h = 0; h < HEADS; ++h) score[h] += dot[h] * scale;
#endif
}

// Store one channel of a K or V row in the fp8 format. Every thread of the
// block calls it (block 256 over one KV head, `channel` = threadIdx.x; lanes
// at or past head_dim pass any value and store nothing). The head's absmax is
// reduced through wave shuffles and `amax_lds` (one float per wave), which
// synchronizes.
template <int F>
static __device__ __forceinline__ void qsa_kv_store(
    unsigned char* row,
    float value,
    int kv_heads,
    int head_dim,
    int kv_head,
    int channel,
    float* amax_lds) {
    static_assert(F == QSA_KV_FP8, "F32 rows are stored directly");
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
    row[kv_head * head_dim + channel] =
        (unsigned char)__builtin_amdgcn_cvt_pk_fp8_f32(value / sf, 0.0f, 0u, false);
    if (channel == 0) {
        *reinterpret_cast<_Float16*>(row + kv_heads * head_dim + 2 * kv_head) = s16;
    }
#endif
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
        float sine = sinf(angle);
        float cosine = cosf(angle);
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
    int rotary_dim) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    if (row >= rows || head >= heads || blockIdx.y != 0) return;
    int start = row * row_stride + head * head_stride;
    __shared__ float input_row[256];
    __shared__ float row_sum;
    const float value = qsa_norm_rope_head(values + start, norm, head_dim, position_start + row,
                                           rotary_dim, input_row, &row_sum);
    if ((int)threadIdx.x < head_dim) values[start + threadIdx.x] = value;
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
    int position_start) {
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
                                               min(index_dim, 64), input_row, &row_sum);
        if (c < index_dim) row[c] = value;
        return;
    }
    block -= index_heads;
    if (block < heads) {
        float* row = qgate + block * 2 * head_dim;
        const float value = qsa_norm_rope_head(row, q_norm, head_dim, position,
                                               min(head_dim, 64), input_row, &row_sum);
        if (c < head_dim) row[c] = value;
        return;
    }
    block -= heads;
    if (block < kv_heads) {
        float* row = keys + block * head_dim;
        const float value = qsa_norm_rope_head(row, k_norm, head_dim, position,
                                               min(head_dim, 64), input_row, &row_sum);
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
        if constexpr (F == QSA_KV_F32) {
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
        int position_start) {                                                              \
        qsa_decode_prologue<FORMAT, RAW>(                                                  \
            index_row, qgate, keys, values, reinterpret_cast<unsigned char*>(full_keys),   \
            reinterpret_cast<unsigned char*>(full_values), raw_index_keys, index_q_norm,   \
            q_norm, k_norm, index_heads, index_dim, index_kv_width, heads, kv_heads,       \
            head_dim, position_start);                                                     \
    }
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_f32, QSA_KV_F32, float, float)
#if defined(QSA_HAS_FP8)
QSA_DECODE_PROLOGUE(indexed_attention_decode_prologue_fp8, QSA_KV_FP8, unsigned char,
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

// Fp8 twin of indexed_attention_cache_append_f32_batched: grid
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

#if defined(QSA_HAS_FP8)
extern "C" __global__ __launch_bounds__(256) void indexed_attention_cache_append_fp8_batched(
    const float* key, const float* value, unsigned char* full_keys, unsigned char* full_values,
    int rows, int position_start, int kv_heads, int head_dim) {
    qsa_cache_append_batched<QSA_KV_FP8>(key, value, full_keys, full_values, rows,
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
    int first_block) {
    int block = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    // Blocks below `first_block` already hold this kernel's output for their
    // unchanged raw keys (the incremental decode caller).
    if (block < first_block || block >= block_count || compress <= 0) return;
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
        float angle = (float)(block * compress)
            * powf(10000000.0f, -((float)(2 * (channel < half ? channel : partner)))
                / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    qsa_index_store(pooled + block * index_dim + channel, bf16_round(value));
}

extern "C" __global__ void indexed_attention_pool_rope_f32(
    const float* raw_keys, float* pooled, const __hip_bfloat16* norm, int block_count,
    int compress, int index_dim, int first_block) {
    qsa_pool_rope(raw_keys, pooled, norm, block_count, compress, index_dim, first_block);
}

extern "C" __global__ void indexed_attention_pool_rope_bf16(
    const __hip_bfloat16* raw_keys, __hip_bfloat16* pooled, const __hip_bfloat16* norm,
    int block_count, int compress, int index_dim, int first_block) {
    qsa_pool_rope(raw_keys, pooled, norm, block_count, compress, index_dim, first_block);
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
        int packed = 0;
        for (int block = first; block < last; ++block) {
            const unsigned key = keys[block];
            packed += key > prefix ? 1 : (key == prefix ? 1 << 16 : 0);
        }
        int inclusive = packed;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int up = __shfl_up(inclusive, offset, 32);
            if (lane >= offset) inclusive += up;
        }
        if (lane == 31) wave_totals[wave] = inclusive;
        __syncthreads();
        int before = 0;
        for (int w = 0; w < wave; ++w) before += wave_totals[w];
        const int exclusive = before + inclusive - packed;
        int above_slot = exclusive & 0xffff;
        int equal_seen = exclusive >> 16;
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
                            score += qv.x * k4[u].x;
                            score += qv.y * k4[u].y;
                            score += qv.z * k4[u].z;
                            score += qv.w * k4[u].w;
                        }
                    }
                } else if ((head_dim & 3) == 0) {
                    for (int d = 0; d < head_dim; d += 4) {
                        const float4 k4 = *reinterpret_cast<const float4*>(key + d);
                        score += q[d] * k4.x;
                        score += q[d + 1] * k4.y;
                        score += q[d + 2] * k4.z;
                        score += q[d + 3] * k4.w;
                    }
                } else {
                    for (int d = 0; d < head_dim; ++d) score += q[d] * key[d];
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
            numerator += weight * (row_tokens[i + u] < 0 ? 0.0f : value[u]);
        }
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) value[u] = next[u];
    }
    for (; i < selected_len; ++i) {
        const int token = row_tokens[i];
        const float weight = row_scores[i];
        const float value = token < 0 ? 0.0f : load(token);
        denominator += weight;
        numerator += weight * value;
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
                        score[h] += q4.x * k4.x;
                        score[h] += q4.y * k4.y;
                        score[h] += q4.z * k4.z;
                        score[h] += q4.w * k4.w;
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
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x31004000);
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
            numerator[0] += w4.x * value[u];
            denominator[1] += w4.y;
            numerator[1] += w4.y * value[u];
            denominator[2] += w4.z;
            numerator[2] += w4.z * value[u];
            denominator[3] += w4.w;
            numerator[3] += w4.w * value[u];
        }
    }
    for (; i < selected_len; ++i) {
        const float value = value_at(i);
        const float4 w4 = *reinterpret_cast<const float4*>(weights + i * QSA_HG);
        denominator[0] += w4.x;
        numerator[0] += w4.x * value;
        denominator[1] += w4.y;
        numerator[1] += w4.y * value;
        denominator[2] += w4.z;
        numerator[2] += w4.z * value;
        denominator[3] += w4.w;
        numerator[3] += w4.w * value;
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
                    scores[t][h] += q4.x * k4.x;
                    scores[t][h] += q4.y * k4.y;
                    scores[t][h] += q4.z * k4.z;
                    scores[t][h] += q4.w * k4.w;
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
    const __amdgpu_buffer_rsrc_t vrsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)values, 0, 0x7FFFFFFF, 0x31004000);
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
                    numerator[4 * q] += w.x * value[k];
                    denominator[4 * q + 1] += w.y;
                    numerator[4 * q + 1] += w.y * value[k];
                    denominator[4 * q + 2] += w.z;
                    numerator[4 * q + 2] += w.z * value[k];
                    denominator[4 * q + 3] += w.w;
                    numerator[4 * q + 3] += w.w * value[k];
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
                numerator[4 * q] += w.x * value;
                denominator[4 * q + 1] += w.y;
                numerator[4 * q + 1] += w.y * value;
                denominator[4 * q + 2] += w.z;
                numerator[4 * q + 2] += w.z * value;
                denominator[4 * q + 3] += w.w;
                numerator[4 * q + 3] += w.w * value;
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
// float cache pointers, fp8 takes byte rows.
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
#if defined(QSA_HAS_FP8)
QSA_ATTENTION_KERNELS(fp8, QSA_KV_FP8, unsigned char)
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
