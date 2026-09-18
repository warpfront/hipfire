// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_write_q8_0(
    unsigned char* __restrict__ dst,     // quantized cache
    const float* __restrict__ src,       // [kv_dim] FP32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    // Grid: [n_kv_heads × blocks_per_head, 1, 1]. Block: [32, 1, 1].
    // Each threadblock quantizes one Q8_0 block (32 elements).
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    const int pos = pos_buf[0];

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;

    float val = src[elem_offset];

    // Warp reduction for max absolute value
    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    float scale = amax / 127.0f;
    float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    // Output: pos * total_blocks * 34 + gid * 34
    unsigned char* out = dst + (size_t)pos * total_blocks * 34 + gid * 34;

    // Thread 0 writes the f16 scale
    if (tid == 0) {
        *((_Float16*)(out)) = (_Float16)scale;
    }
    // All threads write their int8 value
    out[2 + tid] = (unsigned char)(signed char)q;
}

// Sliding-window ring variant. The logical position remains absolute in
// pos_buf, while storage wraps within cache_capacity physical rows.
extern "C" __global__ void kv_cache_write_q8_0_ring(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim,
    int cache_capacity
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];
    if (cache_capacity <= 0) return;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;
    const int elem_offset = head_idx * head_dim + block_idx * 32 + tid;
    const float val = src[elem_offset];

    float amax = fabsf(val);
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    const float scale = amax / 127.0f;
    const float inv_scale = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * inv_scale);
    q = max(-127, min(127, q));

    const int slot = pos % cache_capacity;
    unsigned char* out = dst + (size_t)slot * total_blocks * 34 + gid * 34;
    if (tid == 0) *((_Float16*)(out)) = (_Float16)scale;
    out[2 + tid] = (unsigned char)(signed char)q;
}

#ifdef HIPFIRE_KV_FP8_E4M3
// Native E4M3FN KV write (stage-b Q0 format, §2.1). Distinct symbol; the q8
// kernels above are byte-unchanged.
//
// Layout per token row, per side: [codes: n_kv_heads × head_dim bytes,
// head-major][scales: n_kv_heads little-endian f16]. stride =
// n_kv_heads * (head_dim + 2) (1032 at Hkv=4/D=256).
//
// Grid: [n_kv_heads, 1, 1]. Block: [32, 1, 1]. One wave per head, 8 floats
// per lane, full-head absmax → one f16 scale, RNE E4M3 codes, contiguous
// packed code stores. Admitted geometry is D256 (enforced host-side).
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3 requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif
extern "C" __global__ void kv_cache_write_fp8_e4m3(
    unsigned char* __restrict__ dst,     // fp8 cache, token-local rows
    const float* __restrict__ src,       // [n_kv_heads * head_dim] f32 KV vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    if (head >= n_kv_heads) return;
    const int pos = pos_buf[0];

    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = src[(size_t)head * head_dim + tid * 8 + i];

    // Full-head absmax (uniform across the wave afterwards).
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    // Scale: smallest f16 s >= amax/448, floor 2^-24, zero row → 1.0, with
    // the upward-half next-up bump when 448*f32(s) < amax. The f32→f16
    // conversion is RNE; the ==0.0f floor check also catches any FTZ zero.
    // In-domain inputs never reach f16 infinity (oracle rejects the rest).
    _Float16 s16 = (amax == 0.0f) ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f)
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = (bits >= 0x7bff) ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }

    // Quantize against the STORED scale (divide, not reciprocal multiply).
    unsigned w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] / sf, v[1] / sf, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] / sf, v[3] / sf, w0, true);
    unsigned w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] / sf, v[5] / sf, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] / sf, v[7] / sf, w1, true);

    const size_t stride = (size_t)n_kv_heads * (head_dim + 2);
    unsigned char* codes = dst + (size_t)pos * stride + (size_t)head * head_dim + tid * 8;
    __builtin_memcpy(codes + 0, &w0, 4);
    __builtin_memcpy(codes + 4, &w1, 4);
    if (tid == 0)
        *((_Float16*)(dst + (size_t)pos * stride + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
#ifdef HIPFIRE_KV_FP8_E4M3
// V-side grouped-fragment write (Vfill layout). Values are bit-identical to
// kv_cache_write_fp8_e4m3 above (same amax/scale/cvt rule); only the code
// placement differs. Codes for (head, dim) land dim-major within the token's
// 8-key group: byte for dim tid*8+i at group_base + head*D*8 + (tid*8+i)*8
// + (pos&7), where group_base = (pos>>3)*8*stride. The 8 codes of one
// (head, group, dim) fragment are contiguous (one b64) for the FA2 route-N V
// fill. Scales stay home (same token-local address as the K writer). K keeps
// the token-major writer above; ONLY V call sites route here.
//
// Grid/block/ABI identical to kv_cache_write_fp8_e4m3.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3_v requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif
extern "C" __global__ void kv_cache_write_fp8_e4m3_v(
    unsigned char* __restrict__ dst,     // fp8 V cache, grouped-fragment codes
    const float* __restrict__ src,       // [n_kv_heads * head_dim] f32 V vector
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;  // 0..31
    if (head >= n_kv_heads) return;
    const int pos = pos_buf[0];

    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = src[(size_t)head * head_dim + tid * 8 + i];

    // Full-head absmax (uniform across the wave afterwards).
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

    // Scale rule identical to the K writer (stored-scale divide below).
    _Float16 s16 = (amax == 0.0f) ? (_Float16)1.0f : (_Float16)(amax / 448.0f);
    if (amax > 0.0f && s16 == (_Float16)0.0f)
        s16 = __builtin_bit_cast(_Float16, (unsigned short)0x0001);
    float sf = (float)s16;
    if (amax > 0.0f && 448.0f * sf < amax) {
        unsigned short bits = __builtin_bit_cast(unsigned short, s16);
        bits = (bits >= 0x7bff) ? (unsigned short)0x7bff : (unsigned short)(bits + 1u);
        s16 = __builtin_bit_cast(_Float16, bits);
        sf = (float)s16;
    }

    // Quantize against the STORED scale (divide, not reciprocal multiply).
    unsigned w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] / sf, v[1] / sf, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] / sf, v[3] / sf, w0, true);
    unsigned w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] / sf, v[5] / sf, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] / sf, v[7] / sf, w1, true);

    // Grouped-fragment scatter: lane tid owns dims tid*8..tid*8+7; dim d of
    // this head lives at group_base + head*D*8 + d*8 + j (stride 8 between
    // dims, byte-exact single-lane stores so neighbours are undisturbed).
    const size_t stride = (size_t)n_kv_heads * (head_dim + 2);
    const unsigned char* w0b = (const unsigned char*)&w0;
    const unsigned char* w1b = (const unsigned char*)&w1;
    unsigned char* base = dst + (size_t)(pos >> 3) * stride * 8u
        + (size_t)head * (size_t)head_dim * 8u + (size_t)tid * 64u + (size_t)(pos & 7);
    base[0] = w0b[0]; base[8] = w0b[1]; base[16] = w0b[2]; base[24] = w0b[3];
    base[32] = w1b[0]; base[40] = w1b[1]; base[48] = w1b[2]; base[56] = w1b[3];
    if (tid == 0)
        *((_Float16*)(dst + (size_t)pos * stride + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
#endif
#endif
