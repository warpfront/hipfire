// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Head-dim-split flash-decode reduce (gfx1151). Byte-identical to
// `attention_flash_q8_0_reduce`: same 2-pass combine —
// global max over tiles with sum > 0, corr = expf(tile_max - global_max),
// increasing-tile global_sum = fma(sum, corr, global_sum), per-dim
// g = fma(out_d, corr, g) in increasing tile order, out = g * (1 / global_sum).
//
// The reference walks the tile max/sum with one thread through dependent
// scalar loads (one memory round trip per tile) on 24 workgroups. Here each
// workgroup owns one head and a 32-dim chunk (grid [n_heads, head_dim / 32],
// 20 KB LDS so all of them are resident at once):
//   1. every thread issues all of its loads at entry: the chunk's per-tile
//      values of up to 512 tiles (register-resident) and its tiles' max/sum;
//   2. the global max is a workgroup reduction (the fmax of a fixed set is
//      order-free except for the sign of a zero maximum, and its only
//      consumer, tile_max - global_max inside expf, gives identical bits for
//      either sign); corrections go to LDS;
//   3. per 64-tile block the values are staged through LDS and lane 0 of
//      wave 0 extends the serial global sum while wave 1 extends the 32
//      serial per-dim chains, both in the reference's increasing-tile order.
#include <hip/hip_runtime.h>

#define RD_THREADS 256
#define RD_WAVES (RD_THREADS / 32)
#define RD_DCHUNK 32
#define RD_BT 64                                   // tiles per LDS block
#define RD_NB 8                                    // blocks per super-block
#define RD_SB (RD_BT * RD_NB)                      // 512 tiles register-resident
#define RD_REGS (RD_SB * RD_DCHUNK / RD_THREADS)   // 64
#define RD_BATCH 16

// Serial acc = fma(a[k*as], b[k], acc) over k < n, loads batched.
__device__ __forceinline__ float rd_serial_fma(const float* __restrict__ a, int as,
                                               const float* __restrict__ b, int n, float acc) {
    int k = 0;
    for (; k + RD_BATCH <= n; k += RD_BATCH) {
        float x[RD_BATCH], y[RD_BATCH];
#pragma unroll
        for (int i = 0; i < RD_BATCH; ++i) { x[i] = a[(k + i) * as]; y[i] = b[k + i]; }
#pragma unroll
        for (int i = 0; i < RD_BATCH; ++i) acc = __builtin_fmaf(x[i], y[i], acc);
    }
    for (; k < n; ++k) acc = __builtin_fmaf(a[k * as], b[k], acc);
    return acc;
}

// d[RD_BT/RD_WAVES * b + j] = value of tile sb + RD_BT*b + wave + RD_WAVES*j, dim d0 + lane.
__device__ __forceinline__ void rd_fetch(const float* __restrict__ ph, int stride, int d0, int sb,
                                         int n_tiles, int wave, int lane, float (&d)[RD_REGS]) {
#pragma unroll
    for (int i = 0; i < RD_REGS; ++i) {
        const int tile = sb + wave + RD_WAVES * i;
        d[i] = tile < n_tiles ? ph[(size_t)tile * stride + 2 + d0 + lane] : 0.0f;
    }
}

// Same 7-argument ABI as attention_flash_q8_0_reduce. Grid: [n_heads,
// head_dim / 32]. Block: [256]. No dynamic LDS.
extern "C" __launch_bounds__(RD_THREADS, 1)
__global__ void attention_flash_reduce_dsplit_gfx1151(
    const float* __restrict__ partials,  // [n_heads, max_tiles, 2 + head_dim]
    float* __restrict__ out,             // [n_heads, head_dim]
    int n_heads,
    int head_dim,
    const int* __restrict__ pos_buf,     // pos_buf[0] = last position
    int tile_size,
    int max_tiles
) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int d0 = blockIdx.y * RD_DCHUNK;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int stride = 2 + head_dim;
    const int seq_len = pos_buf[0] + 1;
    const int n_tiles = min((seq_len + tile_size - 1) / tile_size, max_tiles);
    const float* ph = partials + (size_t)h * max_tiles * stride;

    __shared__ float corr[RD_SB];
    __shared__ float sums[RD_SB];
    __shared__ float blk[2][RD_BT * RD_DCHUNK];
    __shared__ float wave_max[RD_WAVES];
    __shared__ float global_sum_shared;

    // 1. All loads at entry: first super-block values, this thread's tile max/sum.
    float d[RD_REGS];
    rd_fetch(ph, stride, d0, 0, n_tiles, wave, lane, d);
    float m[RD_SB / RD_THREADS], s[RD_SB / RD_THREADS];
#pragma unroll
    for (int i = 0; i < RD_SB / RD_THREADS; ++i) {
        const int tile = tid + RD_THREADS * i;
        m[i] = tile < n_tiles ? ph[(size_t)tile * stride] : 0.0f;
        s[i] = tile < n_tiles ? ph[(size_t)tile * stride + 1] : 0.0f;
    }

    // 2. Global max (workgroup reduction).
    float local_max = -1e30f;
#pragma unroll
    for (int i = 0; i < RD_SB / RD_THREADS; ++i)
        if (s[i] > 0.0f) local_max = fmaxf(local_max, m[i]);
    for (int tile = RD_SB + tid; tile < n_tiles; tile += RD_THREADS) {
        if (ph[(size_t)tile * stride + 1] > 0.0f) local_max = fmaxf(local_max, ph[(size_t)tile * stride]);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    if (lane == 0) wave_max[wave] = local_max;
    __syncthreads();
    float global_max = wave_max[0];
#pragma unroll
    for (int w = 1; w < RD_WAVES; ++w) global_max = fmaxf(global_max, wave_max[w]);

    // 3. Per super-block: corrections, then 64-tile blocks of serial sum / chains.
    float global_sum = 0.0f;
    float g = 0.0f;
    for (int sb = 0; sb < n_tiles; sb += RD_SB) {
        if (sb > 0) {
            rd_fetch(ph, stride, d0, sb, n_tiles, wave, lane, d);
#pragma unroll
            for (int i = 0; i < RD_SB / RD_THREADS; ++i) {
                const int tile = sb + tid + RD_THREADS * i;
                m[i] = tile < n_tiles ? ph[(size_t)tile * stride] : 0.0f;
                s[i] = tile < n_tiles ? ph[(size_t)tile * stride + 1] : 0.0f;
            }
            __syncthreads();  // previous super-block's corr/sums/blk readers are done
        }
#pragma unroll
        for (int i = 0; i < RD_SB / RD_THREADS; ++i) {
            const int k = tid + RD_THREADS * i;
            corr[k] = s[i] > 0.0f ? expf(m[i] - global_max) : 0.0f;
            sums[k] = s[i];
        }
        const int n_sb = min(RD_SB, n_tiles - sb);
#pragma unroll
        for (int b = 0; b < RD_NB; ++b) {
            if (RD_BT * b < n_sb) {
                float* buf = blk[b & 1];
#pragma unroll
                for (int j = 0; j < RD_BT / RD_WAVES; ++j)
                    buf[(wave + RD_WAVES * j) * RD_DCHUNK + lane] = d[(RD_BT / RD_WAVES) * b + j];
                __syncthreads();
                const int nb = min(RD_BT, n_sb - RD_BT * b);
                if (tid == 0)
                    global_sum = rd_serial_fma(sums + RD_BT * b, 1, corr + RD_BT * b, nb, global_sum);
                if (wave == 1)
                    g = rd_serial_fma(buf + lane, RD_DCHUNK, corr + RD_BT * b, nb, g);
            }
        }
    }
    if (tid == 0) global_sum_shared = global_sum;
    __syncthreads();
    if (wave == 1) {
        const float inv_sum = global_sum_shared > 0.0f ? 1.0f / global_sum_shared : 0.0f;
        out[(size_t)h * head_dim + d0 + lane] = g * inv_sum;
    }
}
