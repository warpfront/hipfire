// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Down-projection A8 producer of the Qwen4 EP owner-routed symmetric MoE prefill chain
// (`qwen4_moe_ep_a8.gfx12.hip`): the int8 twin of `qwen4_moe_rotate128_i4.hip` with the same grouped BF16
// SwiGLU row input `[P, K]`, the same transform (lossless BF16 widen -> sign1 -> radix-2 FWHT128 -> *0.0883883476f
// -> sign2; lane l holds elements 4l..4l+3) and the same compact flat-slot epoch-major output
// `Xq[group * L + sorted[p]]`, but through `quantize_block_i8_128_wave` (`block_i8_128_quant.hip`,
// the shared A8 quantizer contract: d = RN(amax / 127), q = clamp(rint(RN(x / d)), -127, 127)).
// Each valid flat slot appears once (no atomics); padding rows (`sorted[p] < 0`) emit nothing.
// The host concatenates `block_i8_128_quant.hip` (HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE) in front.
// Grid (P * K/128, 1, 1), block 32: x folds (p, group) as x = p * (K/128) + group.

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "qwen4_moe_rotate128_i8.hip is gfx1201-only"
#endif

extern "C" __global__ __launch_bounds__(32) void qwen4_moe_rotate128_i8(
    const uint16_t* __restrict__ h, const int* __restrict__ sorted,
    block_i8_128* __restrict__ Xq, const float* __restrict__ signs1,
    const float* __restrict__ signs2, int K, int P, int L) {
    const unsigned groups = (unsigned)K >> 7;
    const unsigned bx = blockIdx.x;
    const int p = (int)(bx / groups);
    const int group = (int)(bx - (unsigned)p * groups);
    const int slot = sorted[p];
    if (slot < 0) return;
    const int tid = threadIdx.x;
    const int d0 = tid * 4;
    const uint2 raw = *(const uint2*)(h + (size_t)p * K + group * 128 + d0);
    float v0 = __uint_as_float(raw.x << 16) * signs1[d0];
    float v1 = __uint_as_float(raw.x & 0xFFFF0000u) * signs1[d0 + 1];
    float v2 = __uint_as_float(raw.y << 16) * signs1[d0 + 2];
    float v3 = __uint_as_float(raw.y & 0xFFFF0000u) * signs1[d0 + 3];

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
    float xv[4];
    xv[0] = v0 * scale * signs2[d0];
    xv[1] = v1 * scale * signs2[d0 + 1];
    xv[2] = v2 * scale * signs2[d0 + 2];
    xv[3] = v3 * scale * signs2[d0 + 3];
    quantize_block_i8_128_wave(xv, Xq + (size_t)group * L + slot, tid);
}
