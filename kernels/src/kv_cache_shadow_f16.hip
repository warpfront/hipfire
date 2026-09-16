// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// f16 KV shadow writers (Halo-only, exact gfx1151).
//
// Each kernel re-reads the JUST-WRITTEN admitted Q8_0 / fwht3 records and
// emits the f16 value the gfx11 FA2 fill would compute for the same bytes —
// the dequant expressions below are copied verbatim from
// kernels/src/attention_q8_0_fa2_gqa.gfx11.hip (same cast, same operand
// order), so the shadow bits are identical by construction and the shadow-fed
// FA2 fill is bitwise-identical to the Q8/fwht3 fill with no dequant.
//
// Shadow layout (both sides): row-major f16 `[pos, n_kv_heads, head_dim]`
// (`shadow + pos*n_kv_heads*head_dim + h*head_dim + d`). Legacy single-slot
// addressing only (base 0, direct `pos`); multi-slot arenas keep the
// unchanged Q8 path (no shadow). Launched immediately after the matching
// KV write in the same dispatch arm via `launch_maybe_blob`, so graph
// capture/replay covers the shadow write together with the KV write.
//
// Grids mirror the matching writers:
//   kv_cache_shadow_q8_0            — [n_kv_heads * head_dim/32],       block 32
//   kv_cache_shadow_q8_0_batched    — [n_kv_heads * head_dim/32, batch], block 32
//   kv_cache_shadow_fwht3_k         — [n_kv_heads],                     block 32
//   kv_cache_shadow_fwht3_k_batched — [n_kv_heads, batch],              block 32
//
// TURBO_C3_256 comes from turbo_common.h (prepended by the launcher — the
// runtime compile has no -I to kernels/src, same pattern as the fwht3
// writers).
#include <hip/hip_runtime.h>
#include "turbo_common.h"

typedef union {
    uint32_t u;
    _Float16 h[2];
} shadow_dw_t;

// One Q8_0 block (f16 scale + 32 int8 codes) -> 32 f16 shadow values.
// Verbatim FA2 Q8 fill arithmetic: s = f16 scale bits reloaded,
// sf = (float)s, c = (float)(int8_t)code, out = (_Float16)(sf * c).
extern "C" __global__ void kv_cache_shadow_q8_0(
    const unsigned char* __restrict__ cache, // Q8_0 rows just written
    _Float16* __restrict__ shadow,           // f16 rows [pos, heads, dim]
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int gid = blockIdx.x;
    const int tid = threadIdx.x; // 0..31
    const int pos = pos_buf[0];

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;

    const unsigned char* blk = cache + (size_t)pos * total_blocks * 34 + gid * 34;
    uint16_t su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    _Float16 s = ((shadow_dw_t){.u = (uint32_t)su}).h[0];
    const float sf = (float)s;
    const float c = (float)(int8_t)blk[2 + tid];

    shadow_dw_t d;
    d.h[0] = (_Float16)(sf * c);
    shadow[(size_t)pos * n_kv_heads * head_dim + head_idx * head_dim + block_idx * 32 + tid] = d.h[0];
}

// Batched sibling: one Q8_0 block per (position, block). `positions[bid]`
// selects the token row on both the cache and shadow sides.
extern "C" __global__ void kv_cache_shadow_q8_0_batched(
    const unsigned char* __restrict__ cache,
    _Float16* __restrict__ shadow,
    const int* __restrict__ positions,
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int gid = blockIdx.x;
    const int bid = blockIdx.y;
    if (bid >= batch_size) return;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks = n_kv_heads * blocks_per_head;
    if (gid >= total_blocks) return;

    const int pos = positions[bid];
    const int head_idx = gid / blocks_per_head;
    const int block_idx = gid % blocks_per_head;

    const unsigned char* blk = cache + (size_t)pos * total_blocks * 34 + gid * 34;
    uint16_t su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    _Float16 s = ((shadow_dw_t){.u = (uint32_t)su}).h[0];
    const float sf = (float)s;
    const float c = (float)(int8_t)blk[2 + tid];

    shadow_dw_t d;
    d.h[0] = (_Float16)(sf * c);
    shadow[(size_t)pos * n_kv_heads * head_dim + head_idx * head_dim + block_idx * 32 + tid] = d.h[0];
}

// One fwht3 K record ([f32 cnorm][32 lanes x 3 B packed 3-bit codes],
// 100 B/head at hd=256) -> 256 f16 shadow values, post-rotation.
// Verbatim FA2 fwht3 fill arithmetic: value = (_Float16)(cnorm *
// TURBO_C3_256[code]), code (packed >> 3*i) & 7 lane-mapped identically
// (lane l owns dims l*8..l*8+7 via 3 bytes at +4+l*3; packed dword split
// code0 = (packed >> 6*i) & 7, code1 = (packed >> 6*i+3) & 7).

extern "C" __global__ void kv_cache_shadow_fwht3_k(
    const unsigned char* __restrict__ k_cache, // fwht3 records just written
    _Float16* __restrict__ k_shadow,           // f16 rows [pos, heads, dim]
    const int* __restrict__ pos_buf,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x; // 0..31, lane l owns dims l*8..+8
    const int pos = pos_buf[0];

    const int bytes_per_head = 4 + (head_dim * 3) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const unsigned char* rec = k_cache + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    float cnorm = 0.0f;
    __builtin_memcpy(&cnorm, rec, 4);
    uint32_t packed = 0;
    __builtin_memcpy(&packed, rec + 4 + tid * 3, 3);

    _Float16* out = k_shadow + (size_t)pos * n_kv_heads * head_dim + h * head_dim + tid * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int code0 = (packed >> (6 * i)) & 7;
        const int code1 = (packed >> (6 * i + 3)) & 7;
        shadow_dw_t dw;
        dw.h[0] = (_Float16)(cnorm * TURBO_C3_256[code0]);
        dw.h[1] = (_Float16)(cnorm * TURBO_C3_256[code1]);
        out[2 * i] = dw.h[0];
        out[2 * i + 1] = dw.h[1];
    }
}

// Batched fwht3-K sibling: grid [n_kv_heads, batch_size].
extern "C" __global__ void kv_cache_shadow_fwht3_k_batched(
    const unsigned char* __restrict__ k_cache,
    _Float16* __restrict__ k_shadow,
    const int* __restrict__ positions,
    int n_kv_heads,
    int head_dim,
    int batch_size
) {
    const int h = blockIdx.x;
    const int bid = blockIdx.y;
    if (h >= n_kv_heads || bid >= batch_size) return;
    const int tid = threadIdx.x;
    const int pos = positions[bid];

    const int bytes_per_head = 4 + (head_dim * 3) / 8;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const unsigned char* rec = k_cache + (size_t)pos * bytes_per_pos + h * bytes_per_head;

    float cnorm = 0.0f;
    __builtin_memcpy(&cnorm, rec, 4);
    uint32_t packed = 0;
    __builtin_memcpy(&packed, rec + 4 + tid * 3, 3);

    _Float16* out = k_shadow + (size_t)pos * n_kv_heads * head_dim + h * head_dim + tid * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int code0 = (packed >> (6 * i)) & 7;
        const int code1 = (packed >> (6 * i + 3)) & 7;
        shadow_dw_t dw;
        dw.h[0] = (_Float16)(cnorm * TURBO_C3_256[code0]);
        dw.h[1] = (_Float16)(cnorm * TURBO_C3_256[code1]);
        out[2 * i] = dw.h[0];
        out[2 * i + 1] = dw.h[1];
    }
}
