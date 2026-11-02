// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Lane-table siblings of the VerifyAttn Stage-0 gfx1201 pair
// (`attention_verify_gqa_fp8_gfx1201` + `attention_verify_reduce_gfx1201`)
// for the continuous-batching DFlash verify trunk: ONE launch of each runs
// every lane (request) of the chunk instead of one launch pair per lane.
//
// NOT a standalone translation unit. The host builds the module source as
// concat!(attention_verify_gqa.gfx1201.hip, this file), so the `va_verify_gqa`
// template, `VA_*` macros and `VA_STRIDE` below are the original's, untouched
// (no edit to the existing file, no new copy of its arithmetic).
//
// Each lane's arguments ride in the kernarg segment BY VALUE (lane tables
// below, no device table, no H2D upload, capture-safe like conv1d lanes);
// grid.z selects the lane.
//
// Byte-exactness vs one singleton launch pair per lane with that lane's
// arguments:
//   * verify (`attention_verify_gqa_fp8_lanes_gfx1201`): the body is
//     `va_verify_gqa<VA_KV_FP8>` itself (a `__device__ __forceinline__`
//     template) called with the lane's q / caches / partials / positions /
//     n_rows / max_tiles / batch_offset; the arithmetic is the singleton
//     entry's, expression for expression. The only differences are where the
//     arguments come from (table vs kernel arguments), none of which enters
//     the arithmetic.
//       - grid.x is the MAX over lanes of ceil(n_rows/8) * n_kv_heads: the
//         lane's own decode `kv_h = blockIdx.x / n_rg` yields kv_h >=
//         n_kv_heads for the surplus blocks, which `va_verify_gqa` already
//         turns into an immediate return before touching memory;
//       - grid.y may be any split count: the tile loop strides by gridDim.y
//         and every tile writes its own partial slot
//         (`pair_base + tile`), so the partials bytes (and hence the reduce
//         result) do not depend on the split count (the singleton's own
//         `verify_gqa_splits`: "the split count never changes output bytes");
//       - lanes own DISJOINT partials regions (host-provided base pointers),
//         each indexed with the lane's own `max_tiles` stride as in the
//         singleton, so concurrent lanes never alias.
//   * reduce (`attention_verify_reduce_lanes_gfx1201`): the body is
//     `attention_verify_reduce_gfx1201` VERBATIM (same expressions, same
//     order, same `__launch_bounds__`); the lane's partials / out / positions
//     / max_tiles / batch_offset come from the table, and blocks of rows past
//     the lane's own row count return before reading anything.

#define VA_LANES_MAX 16

struct VaLaneTable {
    unsigned long long q[VA_LANES_MAX];
    unsigned long long k_cache[VA_LANES_MAX];
    unsigned long long v_cache[VA_LANES_MAX];
    unsigned long long partials[VA_LANES_MAX];
    unsigned long long positions[VA_LANES_MAX];
    int n_rows[VA_LANES_MAX];
    int max_tiles[VA_LANES_MAX];
    int batch_offset[VA_LANES_MAX];
};

// Grid: [max_l ceil(n_rows_l/8) * n_kv_heads, S, lanes], block 256.
extern "C" __launch_bounds__(VA_THREADS, 1)
__global__ void attention_verify_gqa_fp8_lanes_gfx1201(
    const VaLaneTable t, int n_kv_heads, float scale_attn) {
    const int lane = (int)blockIdx.z;
    va_verify_gqa<VA_KV_FP8>(
        (const float*)t.q[lane], (const unsigned char*)t.k_cache[lane],
        (const unsigned char*)t.v_cache[lane], (float*)t.partials[lane],
        (const int*)t.positions[lane], t.n_rows[lane], n_kv_heads, t.max_tiles[lane],
        t.batch_offset[lane], scale_attn);
}

struct VaReduceLaneTable {
    unsigned long long partials[VA_LANES_MAX];
    unsigned long long out[VA_LANES_MAX];
    unsigned long long positions[VA_LANES_MAX];
    int n_rows[VA_LANES_MAX];
    int max_tiles[VA_LANES_MAX];
    int batch_offset[VA_LANES_MAX];
};

// Grid: [n_heads, max_l n_rows_l, lanes], block 256.
extern "C" __launch_bounds__(VA_HD, 1)
__global__ void attention_verify_reduce_lanes_gfx1201(
    const VaReduceLaneTable t, int n_heads) {
    const int lane = (int)blockIdx.z;
    const int h = blockIdx.x;
    const int local_bid = blockIdx.y;
    if (local_bid >= t.n_rows[lane]) return;
    const float* partials = (const float*)t.partials[lane];
    float* out = (float*)t.out[lane];
    const int* positions = (const int*)t.positions[lane];
    const int max_tiles = t.max_tiles[lane];
    const int batch_offset = t.batch_offset[lane];

    const int d = threadIdx.x;
    const int seq_len = positions[batch_offset + local_bid] + 1;
    const int n_tiles = (seq_len + VA_TILE - 1) / VA_TILE;
    const float* my_partials = partials + ((long long)local_bid * n_heads + h) * max_tiles * VA_STRIDE;

    float global_max = -1e30f;
    for (int tile = 0; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        if (p[1] > 0.0f)
            global_max = fmaxf(global_max, p[0]);
    }

    // Same serial per-dim order as the reference; the loads of 8 tiles are
    // issued ahead of their (in-order) accumulation.
    float global_sum = 0.0f;
    float g = 0.0f;
    int tile = 0;
    for (; tile + 8 <= n_tiles; tile += 8) {
        float pm[8], ps[8], pv[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float* p = my_partials + (tile + u) * VA_STRIDE;
            pm[u] = p[0];
            ps[u] = p[1];
            pv[u] = p[2 + d];
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            if (ps[u] <= 0.0f) continue;
            const float corr = expf(pm[u] - global_max);
            g += pv[u] * corr;
            global_sum += ps[u] * corr;
        }
    }
    for (; tile < n_tiles; tile++) {
        const float* p = my_partials + tile * VA_STRIDE;
        const float tile_max = p[0];
        const float tile_sum = p[1];
        if (tile_sum <= 0.0f) continue;
        const float corr = expf(tile_max - global_max);
        g += p[2 + d] * corr;
        global_sum += tile_sum * corr;
    }

    const float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    out[(size_t)local_bid * n_heads * VA_HD + (size_t)h * VA_HD + d] = g * inv_sum;
}
