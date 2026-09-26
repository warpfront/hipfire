// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — Qwen4 fixed top-10 grouped gate/up unscatter fused with SwiGLU.
//
// One pass replaces moe_gate_up_unscatter_top10 -> bf16_round_trip_f32 (gate,
// up) -> silu_mul_f32 -> bf16_round_trip_f32 (activation): for each live
// grouped slot it reads the gate/up logits, applies the same BF16 round trips
// (when `bf16_rt` is set), the same SiLU·up expression, and writes the
// activation to the slot's flat (token, k_rank) row.  Every live flat row is
// produced exactly once (each token owns exactly top-10 routed slots), so the
// output buffer is fully written, bit for bit as the unfused sequence.
//
// Grid: (m_total, ceil(mi / 256)). Block: [256].

#include <hip/hip_runtime.h>

// kernels/src/bf16_round_trip.hip: non-finite values pass through unchanged.
static __device__ __forceinline__ float moe_bf16_round_trip(float v) {
    if (!isfinite(v)) return v;
    union { float f; unsigned int u; } box;
    box.f = v;
    const unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    return box.f;
}

static __device__ __forceinline__ float moe_widen(float v) { return v; }
static __device__ __forceinline__ float moe_widen(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

template <typename T>
__device__ __forceinline__ void moe_unscatter_silu(
    const T* __restrict__ Y_grouped,           // [m_total × 2*mi]
    const int*   __restrict__ sorted_slot_index,
    float*       __restrict__ activation,      // [N × 10 × mi] (written)
    int mi, int m_total, int bf16_rt
) {
    const int slot = blockIdx.x;
    if (slot >= m_total) return;
    const int flat = sorted_slot_index[slot];
    if (flat < 0) return;
    const int m_col = blockIdx.y * blockDim.x + threadIdx.x;
    if (m_col >= mi) return;

    const long long y_base = (long long)slot * 2 * mi;
    float v = moe_widen(Y_grouped[y_base + m_col]);
    float u = moe_widen(Y_grouped[y_base + mi + m_col]);
    if (bf16_rt) {
        v = moe_bf16_round_trip(v);
        u = moe_bf16_round_trip(u);
    }
    // kernels/src/silu_mul.hip expression.
    float out = (v / (1.0f + expf(-v))) * u;
    if (bf16_rt) out = moe_bf16_round_trip(out);
    activation[(long long)flat * mi + m_col] = out;
}

extern "C" __global__ void moe_gate_up_unscatter_silu_top10(
    const float* __restrict__ Y_grouped, const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation, int mi, int m_total, int bf16_rt) {
    moe_unscatter_silu(Y_grouped, sorted_slot_index, activation, mi, m_total, bf16_rt);
}

// Grouped gate/up stored as BF16 bits (already the round trip's values).
extern "C" __global__ void moe_gate_up_unscatter_silu_top10_bf16in(
    const unsigned short* __restrict__ Y_grouped, const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation, int mi, int m_total, int bf16_rt) {
    moe_unscatter_silu(Y_grouped, sorted_slot_index, activation, mi, m_total, bf16_rt);
}

// Unscatter of the SwiGLU activation the gate/up GEMM's SILU epilogue wrote
// as BF16 ([m_total x mi], already the unfused round trip's values) fused
// with the 128-wide FWHT of kernels/src/mq_rotate_x_128_v2.hip, writing the
// MoE down GEMM's F16 input straight to `x16` ([N*10 x mi], flat row order).
// Per value: the rotation's butterfly and pinned F32 product before the F16
// cast, so the bytes equal unscatter -> mq_rotate_x_128_v2 ->
// convert_f32_to_f16.  mi % 128 == 0.  Grid (m_total, mi / 128), block 32.
extern "C" __global__ __launch_bounds__(32, 16) void moe_unscatter_rotate128_f16(
    const unsigned short* __restrict__ act_grouped,
    const int*   __restrict__ sorted_slot_index,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    _Float16*    __restrict__ x16,
    int mi, int m_total
) {
    const int slot = blockIdx.x;
    if (slot >= m_total) return;
    const int flat = sorted_slot_index[slot];
    if (flat < 0) return;
    const int tid = threadIdx.x;
    const int d0 = tid * 4;
    const int base = blockIdx.y * 128 + d0;
    // Four activations arrive as one 8-byte load.
    const uint2 aw = *(const uint2*)(act_grouped + (long long)slot * mi + base);
    const float a[4] = {moe_widen((unsigned short)aw.x), moe_widen((unsigned short)(aw.x >> 16)),
                        moe_widen((unsigned short)aw.y), moe_widen((unsigned short)(aw.y >> 16))};
    float v0 = a[0] * signs1[d0];
    float v1 = a[1] * signs1[d0 + 1];
    float v2 = a[2] * signs1[d0 + 2];
    float v3 = a[3] * signs1[d0 + 3];
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
#define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define HBFLY4(pat, str) \
    HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); \
    HBFLY(v3, pat, str)
    HBFLY4(0x041F, 1);
    HBFLY4(0x081F, 2);
    HBFLY4(0x101F, 4);
    HBFLY4(0x201F, 8);
    HBFLY4(0x401F, 16);
#undef HBFLY4
#undef HBFLY
    const float scale = 0.0883883476f;
    // The four halves leave as one 8-byte store.
    _Float16 h[4];
#define ROT128_F16(i, v) do { \
        float r_ = (v) * scale * signs2[d0 + (i)]; \
        asm volatile("" : "+v"(r_)); \
        h[i] = (_Float16)r_; \
    } while (0)
    ROT128_F16(0, v0);
    ROT128_F16(1, v1);
    ROT128_F16(2, v2);
    ROT128_F16(3, v3);
#undef ROT128_F16
    *(uint2*)(x16 + (long long)flat * mi + base) = __builtin_bit_cast(uint2, h);
}
