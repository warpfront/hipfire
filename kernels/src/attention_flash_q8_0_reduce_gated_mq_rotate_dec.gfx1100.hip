// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Decode flash reduce + sigmoid gate + AWQ MQ rotation, gfx1100
// (`attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100`).
// Byte-identical to `attention_flash_q8_0_reduce_gated_mq_rotate_awq_gfx1100`
// on the same partials: the same 2-pass combine —
//   global max over tiles with sum > 0, corr = expf(tile_max - global_max),
//   increasing-tile global_sum = fma(sum, corr, global_sum),
//   per-dim g = fma(out_d, corr, g) in increasing tile order,
//   value = g * (global_sum > 0 ? 1 / global_sum : 0),
//   gated = value * (1 / (1 + expf(-gate))),
// then the reference's own (x / awq) * signs1 -> FWHT-256 -> * 1/16 * signs2.
//
// The reference walks tile max/sum with one thread through dependent loads
// and each dim thread through 8-deep batches of dependent tile loads. Here
// one 1024-thread workgroup per head issues its loads at entry: the tile
// max/sum of every tile (one per thread per 1024 tiles) and, per 256-tile
// round, 64 tiles of one dim per thread (thread group k = tid / 256 owns
// tiles 64k .. 64k+63 of the round, dim tid % 256). The per-dim chains then
// run group after group through an LDS carry, the global-sum chain alongside
// on an idle thread through its own LDS carry, both in the reference's
// increasing-tile order. The global max is a workgroup reduction: the fmax
// of a fixed set is order-free except for the sign of a zero maximum, and
// its only consumer, tile_max - global_max inside expf, gives identical bits
// for either sign.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
#error "attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100 requires --offload-arch=gfx1100"
#endif

#define RX_THREADS 1024
#define RX_WAVES (RX_THREADS / 32)
#define RX_HD 256
#define RX_GROUPS (RX_THREADS / RX_HD)   // 4
#define RX_RB 64                          // tiles per group per round
#define RX_ROUND (RX_GROUPS * RX_RB)      // 256
#define RX_STRIDE (2 + RX_HD)

// v[i] = partial value of tile t0 + i, dim d (0 past n_tiles).
__device__ __forceinline__ void rx_fetch(const float* __restrict__ ph, int d, int t0, int n_tiles,
                                         float (&v)[RX_RB]) {
#pragma unroll
    for (int i = 0; i < RX_RB; ++i) {
        const int t = t0 + i;
        v[i] = t < n_tiles ? ph[(size_t)t * RX_STRIDE + 2 + d] : 0.0f;
    }
}

#define RX_BATCH 16

// acc = fma(x[i], c[i], acc) for i < n (n <= RX_RB), in order; c (LDS) is
// read RX_BATCH entries ahead of the dependent FMAs.
__device__ __forceinline__ float rx_chain(const float (&x)[RX_RB], const float* __restrict__ c, int n,
                                          float acc) {
#pragma unroll
    for (int b = 0; b < RX_RB; b += RX_BATCH) {
        if (b >= n) break;
        float cr[RX_BATCH];
#pragma unroll
        for (int i = 0; i < RX_BATCH; ++i) cr[i] = c[min(b + i, n - 1)];
#pragma unroll
        for (int i = 0; i < RX_BATCH; ++i)
            if (b + i < n) acc = __builtin_fmaf(x[b + i], cr[i], acc);
    }
    return acc;
}

// Same chain with both operands in LDS (global sum: sums x corrections).
__device__ __forceinline__ float rx_chain_lds(const float* __restrict__ s, const float* __restrict__ c, int n,
                                              float acc) {
    for (int b = 0; b < n; b += RX_BATCH) {
        float sr[RX_BATCH], cr[RX_BATCH];
#pragma unroll
        for (int i = 0; i < RX_BATCH; ++i) {
            sr[i] = s[min(b + i, n - 1)];
            cr[i] = c[min(b + i, n - 1)];
        }
#pragma unroll
        for (int i = 0; i < RX_BATCH; ++i)
            if (b + i < n) acc = __builtin_fmaf(sr[i], cr[i], acc);
    }
    return acc;
}

// Same 11-argument ABI as the AWQ reference. Grid: [n_heads]. Block: [1024].
// Dynamic LDS: 2 * max_tiles floats.
extern "C" __launch_bounds__(RX_THREADS, 1)
__global__ void attention_flash_q8_0_reduce_gated_mq_rotate_awq_dec_gfx1100(
    const float* __restrict__ partials,   // [n_heads, max_tiles, 2 + 256]
    float* __restrict__ out,              // [n_heads, 256]
    const float* __restrict__ gate,
    const float* __restrict__ awq_scale,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,
    int tile_size,
    int max_tiles
) {
    const int h = blockIdx.x;
    if (h >= n_heads || head_dim != RX_HD) return;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int grp = tid / RX_HD;
    const int dim = tid % RX_HD;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);
    const float* ph = partials + (size_t)h * max_tiles * RX_STRIDE;

    extern __shared__ float rx_lds[];
    float* corr = rx_lds;              // [max_tiles]
    float* sums = rx_lds + max_tiles;  // [max_tiles]
    __shared__ float carry[RX_HD];
    __shared__ float rotate_values[RX_HD];
    __shared__ float wave_max[RX_WAVES];
    __shared__ float gs_carry;

    // 1. Loads at entry: this thread's first-round values, its tiles'
    //    max/sum, its gate element, and (wave 0) the rotation operands.
    float v[RX_RB];
    rx_fetch(ph, dim, grp * RX_RB, n_tiles, v);
    const float gate_v = grp == 0 ? gate[h * RX_HD + dim] : 0.0f;
    const int d0 = lane * 8;
    float aw[8], s1[8], s2[8];
    if (tid < 32) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            aw[i] = awq_scale[h * RX_HD + d0 + i];
            s1[i] = signs1[d0 + i];
            s2[i] = signs2[d0 + i];
        }
    }
    float local_max = -1e30f;
    for (int t = tid; t < n_tiles; t += RX_THREADS) {
        const float m = ph[(size_t)t * RX_STRIDE];
        const float s = ph[(size_t)t * RX_STRIDE + 1];
        corr[t] = m;  // tile max, until the corrections replace it
        sums[t] = s;
        if (s > 0.0f) local_max = fmaxf(local_max, m);
    }

    // 2. Global max (workgroup reduction), then the corrections.
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    if (lane == 0) wave_max[wave] = local_max;
    if (tid < RX_HD) carry[tid] = 0.0f;
    if (tid == 0) gs_carry = 0.0f;
    __syncthreads();
    float global_max = wave_max[0];
#pragma unroll
    for (int w = 1; w < RX_WAVES; ++w) global_max = fmaxf(global_max, wave_max[w]);
    for (int t = tid; t < n_tiles; t += RX_THREADS)
        corr[t] = sums[t] > 0.0f ? expf(corr[t] - global_max) : 0.0f;
    __syncthreads();

    // 3. Rounds of 256 tiles: group k extends the per-dim chains over tiles
    //    base + 64k .. +63 while the last thread of the next group extends
    //    the global-sum chain over the same tiles.
    for (int base = 0; base < n_tiles; base += RX_ROUND) {
        if (base > 0) rx_fetch(ph, dim, base + grp * RX_RB, n_tiles, v);
#pragma unroll
        for (int k = 0; k < RX_GROUPS; ++k) {
            const int t0 = base + k * RX_RB;
            const int nb = min(RX_RB, n_tiles - t0);
            if (nb <= 0) break;
            const float* c = corr + t0;
            if (grp == k) carry[dim] = rx_chain(v, c, nb, carry[dim]);
            if (tid == ((k + 1) % RX_GROUPS) * RX_HD + RX_HD - 1)
                gs_carry = rx_chain_lds(sums + t0, c, nb, gs_carry);
            __syncthreads();
        }
    }

    // 4. Normalize and gate (reference expressions), then the rotation.
    if (tid < RX_HD) {
        const float global_sum = gs_carry;
        const float inv_sum = global_sum > 0.0f ? 1.0f / global_sum : 0.0f;
        const float value = carry[tid] * inv_sum;
        const float gate_scale = 1.0f / (1.0f + expf(-gate_v));
        rotate_values[tid] = value * gate_scale;
    }
    __syncthreads();

    // Match mq_rotate_x exactly. One attention head is one 256-value MQ group,
    // and the first wave owns the same eight contiguous values per lane.
    if (tid >= 32) return;
    float v0 = (rotate_values[d0] / aw[0]) * s1[0];
    float v1 = (rotate_values[d0 + 1] / aw[1]) * s1[1];
    float v2 = (rotate_values[d0 + 2] / aw[2]) * s1[2];
    float v3 = (rotate_values[d0 + 3] / aw[3]) * s1[3];
    float v4 = (rotate_values[d0 + 4] / aw[4]) * s1[4];
    float v5 = (rotate_values[d0 + 5] / aw[5]) * s1[5];
    float v6 = (rotate_values[d0 + 6] / aw[6]) * s1[6];
    float v7 = (rotate_values[d0 + 7] / aw[7]) * s1[7];

    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;
    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float scale = 0.0625f;
    float* dst = out + h * RX_HD + d0;
    dst[0] = v0 * scale * s2[0];
    dst[1] = v1 * scale * s2[1];
    dst[2] = v2 * scale * s2[2];
    dst[3] = v3 * scale * s2[3];
    dst[4] = v4 * scale * s2[4];
    dst[5] = v5 * scale * s2[5];
    dst[6] = v6 * scale * s2[6];
    dst[7] = v7 * scale * s2[7];
}
