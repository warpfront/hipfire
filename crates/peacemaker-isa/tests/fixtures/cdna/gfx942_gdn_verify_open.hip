// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_g942_gdn_verify: the exact-gfx942 GDN verify-window entries of native MTP, in a module
// of their own (gfx942 only, its own cache key and its own frozen hipcc anchor / twin seat).
//
//   qwen4_g942_gdn_verify_capture_f32      = gated_delta_step_halves_state128_persistent256_capture_f32
//   qwen4_g942_gdn_verify_rollback_f32     = gated_delta_rollback_layers_f32
//   qwen4_g942_gdn_verify_conv_params_k4   = gated_delta_conv_bf16_f32_batched_k4
//   qwen4_g942_gdn_verify_gate_batched     = gated_delta_gate_bf16_f32_batched
//
// Every block between the `BEGIN` / `END verbatim tensor_ops.hip` markers is a byte-for-byte copy of
// tensor_ops.hip up to the four entry names above (the entries are renamed so they cannot collide
// with the tensor_ops symbols in the per-context function table); a unit test in rdna-compute
// (`g942_gdn_verify_source_is_tensor_ops_verbatim`) maps the names back and asserts each block
// occurs in TENSOR_OPS_SRC, so an edit of the production bodies that is not mirrored here fails
// the build. The gfx12-only staged parameter pass of the persistent body is not defined here
// (`GDN_PERSISTENT_STAGED` stays undefined on gfx942, exactly as in the tensor_ops compile).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// BEGIN verbatim tensor_ops.hip
using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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
#if defined(GDN_PERSISTENT_STAGED)
    __shared__ __align__(16) unsigned short square_rows[GDN_PERSISTENT_STAGE_ROWS][2][128];
#endif

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
#if defined(GDN_PERSISTENT_STAGED)
        // Few-row blocks: the whole block stages every row's BF16-rounded q/k
        // element squares (independent per element) into LDS, so the row
        // threads below only add them in kd order. The thread-per-row pass
        // ran each row's 128-element load-and-round chain on one wave while
        // the other seven waited, most of a few-row launch's fixed cost.
        // Same values, same serial sum order.
        const bool staged = block_rows <= GDN_PERSISTENT_STAGE_ROWS;
        if (staged) {
            const int kd = thread & 127;
            const int column = (thread < 128 ? q_offset : k_offset) + kd;
            float x[GDN_PERSISTENT_STAGE_ROWS];
            #pragma unroll
            for (int r = 0; r < GDN_PERSISTENT_STAGE_ROWS; ++r) {
                x[r] = projection[(block + min(r, block_rows - 1)) * qkv_width + column];
            }
            #pragma unroll
            for (int r = 0; r < GDN_PERSISTENT_STAGE_ROWS; ++r) {
                if (r < block_rows) {
                    const float rounded = gdn_bf16_value(gdn_bf16_bits(x[r]));
                    square_rows[r][thread >> 7][kd] = gdn_bf16_bits(rounded * rounded);
                }
            }
            gdn_lds_barrier();
        }
#endif
        if (thread < block_rows) {
            const int row = block + thread;
            const int row_base = row * qkv_width;
            float q_sum_f32 = 0.0f;
            float k_sum_f32 = 0.0f;
            const __hip_bfloat16 eps = bf16_round(1.0e-6f);
#if defined(GDN_PERSISTENT_STAGED)
            if (staged) {
                const uint4* q_squares = reinterpret_cast<const uint4*>(square_rows[thread][0]);
                const uint4* k_squares = reinterpret_cast<const uint4*>(square_rows[thread][1]);
                #pragma unroll
                for (int c = 0; c < 16; ++c) {
                    const uint4 qw = q_squares[c];
                    const uint4 kw = k_squares[c];
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        const unsigned int qp = (&qw.x)[w];
                        const unsigned int kp = (&kw.x)[w];
                        q_sum_f32 += gdn_bf16_value(qp & 0xffffu);
                        k_sum_f32 += gdn_bf16_value(kp & 0xffffu);
                        q_sum_f32 += gdn_bf16_value(qp >> 16);
                        k_sum_f32 += gdn_bf16_value(kp >> 16);
                    }
                }
            } else {
#endif
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
#if defined(GDN_PERSISTENT_STAGED)
            }
#endif
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
#if defined(GDN_PERSISTENT_STAGED)
                        // The code's exact F32 value: no float -> int8 -> float trip.
                        state_half[kd] = scale * gdn_q8_round_value(
                            state_half[kd], inv, value_head, value_index, kd_base + kd,
                            row_frame);
#else
                        state_half[kd] = scale * (float)gdn_q8_round(
                            state_half[kd], inv, value_head, value_index, kd_base + kd,
                            row_frame);
#endif
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
__launch_bounds__(256)
extern "C" __global__ void qwen4_g942_gdn_verify_capture_f32(
    const float* projection, const float* gate, const float* beta, float* state,
    float* output, float* row_states, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    gated_delta_step_persistent256_body<true>(projection, gate, beta, state, output, row_states,
        rows, qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale);
}
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
__launch_bounds__(256)
extern "C" __global__ void qwen4_g942_gdn_verify_rollback_f32(
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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

extern "C" __global__ void qwen4_g942_gdn_verify_conv_params_k4(
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
// END verbatim tensor_ops.hip

// BEGIN verbatim tensor_ops.hip
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
extern "C" __global__ __launch_bounds__(32) void qwen4_g942_gdn_verify_gate_batched(
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
// END verbatim tensor_ops.hip
