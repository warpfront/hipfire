// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_gdn_lanes_step_capture_f32_gfx942: the Flash-Next GDN VERIFY-window recurrence of many
// lanes in ONE launch (gfx942, F32 state), with the state capture of native MTP.
//
// It is `qwen4_gdn_lanes_step_f32_gfx942` (kernels/src/qwen4_gdn_lanes_f32.gfx942.hip: the
// `gated_delta_step_pipe128_f32` body, lane on `blockIdx.y`, grid (value_heads, lanes), block
// 128) with one change: the lane's state is READ from `state` and the final state (after the
// lane's `rows` rows) is STORED to `final_state`, the capture ring's slot `rows - 1`, so the
// pre-window state stays intact for a partial-accept rollback
// (`gated_delta_rollback_layers_f32`) exactly as the per-lane capture kernel
// `gated_delta_step_halves_state128_persistent256_capture_f32` leaves it. Each lane's
// projection / gate / beta rows, state, final-state slot and output rows come from a device
// lane table (one `Qwen4GdnLaneCaptureDesc` per lane); a lane's blocks write only that lane's
// final-state slot and output rows.
//
// The recurrence is the AR singleton's (pipe128) arithmetic, bit for bit: the final store is the
// only edit of the body. tensor_ops.hip is JIT-compiled by every arch and is not edited for this
// route, so the helpers between the `BEGIN verbatim` / `END verbatim` markers are copies of it
// and the body between the `BEGIN derived` / `END derived` markers is its pipe128 body with
// the two documented edits (a `final_out` parameter and the final store); unit tests in
// rdna-compute (`gdn_lanes_capture_f32_gfx942_source_*`) assert both against TENSOR_OPS_SRC.

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
// END verbatim tensor_ops.hip

// BEGIN derived tensor_ops.hip pipe128 body (final_out)
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
    unsigned short* ef = nullptr,
    float* final_out = nullptr) {
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
    float* const final_state = final_out;
    #pragma unroll
    for (int kd = 0; kd < 128; ++kd) {
        final_state[state_base + kd * 128 + c] = s[kd];
    }
}
// END derived tensor_ops.hip pipe128 body (final_out)

// One lane of a launch: 6 x 8 bytes (`Qwen4GdnLaneCaptureDesc` in
// crates/rdna-compute/src/cdna/gfx942.rs, `#[repr(C)]`). The pointers are typed global (address
// space 1) for the reason documented in qwen4_gdn_lanes_f32.gfx942.hip.
#define QWEN4_GDN_GLOBAL __attribute__((address_space(1)))
struct Qwen4GdnLaneCaptureDesc {
    const float QWEN4_GDN_GLOBAL* projection;  // [rows, qkv_width] convolution output rows
    const float QWEN4_GDN_GLOBAL* gate;        // [rows, value_heads]
    const float QWEN4_GDN_GLOBAL* beta;        // [rows, value_heads]
    const float QWEN4_GDN_GLOBAL* state;       // F32 [value_heads, 128, 128], read only
    float QWEN4_GDN_GLOBAL* final_state;       // F32 [value_heads, 128, 128]: the state after `rows` rows
    float QWEN4_GDN_GLOBAL* output;            // [rows, value_heads * 128]
};

// Grid (value_heads, lanes), block 128. A lane's `rows` rows run serially in its blocks exactly as
// in the singleton kernel; the final state goes to the lane's `final_state`.
__launch_bounds__(128)
extern "C" __global__ void qwen4_gdn_lanes_step_capture_f32_gfx942(
    const Qwen4GdnLaneCaptureDesc* lanes, int rows, int qkv_width, int key_heads,
    int value_heads, int key_dim, int value_dim, float query_scale) {
    const Qwen4GdnLaneCaptureDesc lane = lanes[blockIdx.y];
    gated_delta_step_pipe128_body<false>((const float*)lane.projection, (const float*)lane.gate,
        (const float*)lane.beta, (void*)(float*)lane.state, (float*)lane.output, rows,
        qkv_width, key_heads, value_heads, key_dim, value_dim, query_scale, 0u, nullptr,
        (float*)lane.final_state);
}
#endif
