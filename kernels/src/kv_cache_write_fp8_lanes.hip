// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3_lanes requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif

// Lane-table sibling of `kv_cache_write_fp8_e4m3_batched`
// (kv_cache_write_q8_0_batched.hip, HIPFIRE_KV_FP8_E4M3 section) for the
// continuous-batching DFlash verify trunk: ONE launch writes the K (or V)
// rows of every lane (request) of the chunk into that lane's own cache.
//
// Lane i writes `n[i]` rows of its row-major `src[i]` ([row, n_kv_heads *
// head_dim] f32) into `dst[i]` (the lane's fp8 cache, `[pos][n_kv_heads *
// (head_dim + 2)]` bytes) at the absolute positions `positions[i][row]`.
// grid.z selects the lane; the lane table rides in the kernarg segment BY
// VALUE (no device table, no H2D upload).
//
// Byte-exactness vs one `kv_cache_write_fp8_e4m3_batched` launch per lane
// (grid [n_kv_heads, n_i, 1]): the per-(head, row) body is the original's
// VERBATIM (same loads, same amax / f16 scale rule, same cvt_pk_fp8 chain,
// same store addressing); only the origin of `dst` / `src` / `positions` /
// `batch_size` (table instead of kernel arguments) changes. A wave owns one
// (head, row) of one lane and writes only that lane's cache row, so running
// lanes concurrently instead of one launch after another changes no byte; the
// grid's y extent is the max row count over lanes and shorter lanes return on
// `bid >= batch_size` exactly as the original's guard does.
//
// Grid: [n_kv_heads, max_l n_l, lanes]. Block: [32, 1, 1].

#define KVW_LANES_MAX 16

struct KvWriteLaneTable {
    unsigned long long dst[KVW_LANES_MAX];
    unsigned long long src[KVW_LANES_MAX];
    unsigned long long positions[KVW_LANES_MAX];
    int n[KVW_LANES_MAX];
};

extern "C" __global__ void kv_cache_write_fp8_e4m3_lanes(
    const KvWriteLaneTable t,
    int n_kv_heads,
    int head_dim
) {
    const int lane = blockIdx.z;
    unsigned char* __restrict__ dst = (unsigned char*)t.dst[lane];
    const float* __restrict__ src = (const float*)t.src[lane];
    const int* __restrict__ positions = (const int*)t.positions[lane];
    const int batch_size = t.n[lane];

    const int head = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    if (head >= n_kv_heads) return;
    const int tid = threadIdx.x;

    const int kv_dim = n_kv_heads * head_dim;
    float v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = src[(size_t)bid * kv_dim + (size_t)head * head_dim + tid * 8 + i];

    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset));

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

    unsigned w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[0] / sf, v[1] / sf, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v[2] / sf, v[3] / sf, w0, true);
    unsigned w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[4] / sf, v[5] / sf, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v[6] / sf, v[7] / sf, w1, true);

    const int pos = positions[bid];
    const size_t stride = (size_t)n_kv_heads * (head_dim + 2);
    unsigned char* codes = dst + (size_t)pos * stride + (size_t)head * head_dim + tid * 8;
    __builtin_memcpy(codes + 0, &w0, 4);
    __builtin_memcpy(codes + 4, &w1, 4);
    if (tid == 0)
        *((_Float16*)(dst + (size_t)pos * stride + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
