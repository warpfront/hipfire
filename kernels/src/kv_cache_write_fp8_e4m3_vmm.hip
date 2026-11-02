// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Per-request VMM native fp8-E4M3 KV write (gfx1201). Arithmetic is the
// exact body of `kv_cache_write_fp8_e4m3_batched` (kv_cache_write_q8_0_batched.hip);
// only the destination differs: row `bid` writes into the absolute VMM VA of
// the request selected by `row_slot[bid]` (k_base or v_base by `use_v_base`),
// bounds-checked by kv_offset_for_k/v. Token row layout is unchanged:
// [Hkv x D fp8 codes][Hkv f16 scales].
//
// Grid: [n_kv_heads, batch_size, 1]. Block: [32, 1, 1].

#include <hip/hip_runtime.h>
#include "kv_slot_desc_vmm.h"

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "kv_cache_write_fp8_e4m3_batched_vmm requires --offload-arch=gfx1201 (OCP E4M3FN cvt)"
#endif
extern "C" __global__ void kv_cache_write_fp8_e4m3_batched_vmm(
    const float* __restrict__ src,        // [batch_size × n_kv_heads × head_dim]
    const int* __restrict__ positions,    // [batch_size]
    int n_kv_heads,
    int head_dim,
    int batch_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots], never null
    const int* __restrict__ row_slot,           // [batch_size], never null
    int use_v_base
) {
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
    const KvSlotDesc desc = slot_descs[row_slot[bid]];
    const int stride = n_kv_heads * (head_dim + 2);
    unsigned char* row = (unsigned char*)(use_v_base
        ? kv_offset_for_v(desc, pos, stride)
        : kv_offset_for_k(desc, pos, stride));
    unsigned char* codes = row + (size_t)head * head_dim + tid * 8;
    __builtin_memcpy(codes + 0, &w0, 4);
    __builtin_memcpy(codes + 4, &w1, 4);
    if (tid == 0)
        *((_Float16*)(row + (size_t)n_kv_heads * head_dim + (size_t)head * 2)) = s16;
}
