// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// High-occupancy Q8_0 GEMV for K > 1536 (`gemv_q8_0.hip`'s arm), BIT-IDENTICAL to it.
//
// Per output the arithmetic is the narrow kernel's, instruction for instruction:
//   * one lane per element of a 32-wide Q8_0 block (lane l owns element l of every
//     block of the row), ONE accumulator chain per lane over the row's blocks in
//     ascending order: sum = fma(dq, x[bi*32 + l], sum), dq = f16 scale * (float)int8
//     (an exact product, so any evaluation form gives the same bits);
//   * the wave `__shfl_down` tree 16, 8, 4, 2, 1; lane 0 stores.
// What changes is only how the operands reach the lanes:
//   * 8 waves per workgroup, one weight row per wave (the old launch was 1 wave per
//     block), so a CU holds many more rows in flight;
//   * the row's bytes are fetched with 32-bit global loads (the old kernel issued a
//     byte load, a scale load and an x load per lane per 32 elements) into a per-wave
//     LDS region, `C` blocks at a time, and read back as bytes / a broadcast f16;
//   * the activation rows of the chunk are staged ONCE per workgroup (float4 loads)
//     and shared by its 8 waves; with R > 1 rows every weight byte is read once and
//     applied to the R activation rows (R independent chains per lane, each the
//     single-row chain above), so row r of an R-row launch is bit-identical to the
//     R == 1 launch on x row r (and to the old kernel);
//   * the next chunk's global loads are issued before the current chunk is consumed
//     (software pipeline over two LDS buffers, one barrier per chunk).
//
// Launch contract (checked by the launcher; otherwise the old kernel runs):
//   K % 64 == 0 (an even number of 34-byte blocks keeps every row start and every
//   chunk 4-byte aligned), A 4-byte aligned, x 16-byte aligned, wave32 (gfx11/gfx12),
//   grid ceil(M / 8), block 256. y is [R, M] row-major, x is [R, K].
//
// `gemv_q8_0_fast_r1` is the decode kernel; `_r2` .. `_r16` are the row-exact
// verify kernels (one per row count so a small R does not carry R = 16's registers).

#define Q8F_WAVES 8
#define Q8F_THREADS (Q8F_WAVES * 32)

// Blocks per chunk: the x tile of a chunk is R * C * 32 floats, kept at <= 2048 floats
// (8 KB) per buffer, C even so every chunk is a whole number of dwords (34 * C % 4 == 0),
// at most 16.
template <int R>
struct Q8FChunk {
    static constexpr int raw = 2048 / (R * 32);
    static constexpr int capped = raw > 16 ? 16 : raw;
    static constexpr int value = (capped & ~1) < 2 ? 2 : (capped & ~1);
};

template <int R>
static __device__ __forceinline__ void gemv_q8_0_fast_body(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    constexpr int C = Q8FChunk<R>::value;
    constexpr int WD = C / 2 * 17;                 // dwords of one wave's chunk (34 * C / 4)
    constexpr int XF4 = C * 8;                     // float4 per row of the x tile
    constexpr int XT = (R * XF4 + Q8F_THREADS - 1) / Q8F_THREADS;
    constexpr int WT = (WD + 31) / 32;

    __shared__ __attribute__((aligned(16))) float sx[2][R * C * 32];
    __shared__ __attribute__((aligned(16))) unsigned int sw[2][Q8F_WAVES][WD];

    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int row = blockIdx.x * Q8F_WAVES + wave;
    const bool live = row < M;

    const int nb = K >> 5;                          // blocks per row (even)
    const int nchunks = (nb + C - 1) / C;
    const unsigned int* rowdw =
        reinterpret_cast<const unsigned int*>(A_q8 + (size_t)row * (size_t)nb * 34);

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = 0.0f;

    float4 xr[XT];
    unsigned int wr[WT];

    // Issue the global loads of chunk `c` into registers.
    auto load_chunk = [&](int c) {
        const int cnt = (nb - c * C < C) ? (nb - c * C) : C;
        const size_t xbase = (size_t)c * (C * 32);
#pragma unroll
        for (int t = 0; t < XT; t++) {
            const int idx = tid + Q8F_THREADS * t;
            const int r = idx / XF4;
            const int j4 = idx - r * XF4;
            xr[t] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < R && j4 < cnt * 8)
                xr[t] = *reinterpret_cast<const float4*>(x + (size_t)r * K + xbase + j4 * 4);
        }
        const int ndw = cnt / 2 * 17;
        const size_t wbase = (size_t)c * (C / 2 * 17);
#pragma unroll
        for (int t = 0; t < WT; t++) {
            const int i = lane + 32 * t;
            wr[t] = 0u;
            if (live && i < ndw) wr[t] = rowdw[wbase + i];
        }
    };
    // Publish the registers into LDS buffer `buf`.
    auto store_chunk = [&](int buf, int c) {
        const int cnt = (nb - c * C < C) ? (nb - c * C) : C;
#pragma unroll
        for (int t = 0; t < XT; t++) {
            const int idx = tid + Q8F_THREADS * t;
            const int r = idx / XF4;
            const int j4 = idx - r * XF4;
            if (r < R && j4 < cnt * 8)
                *reinterpret_cast<float4*>(&sx[buf][r * C * 32 + j4 * 4]) = xr[t];
        }
        const int ndw = cnt / 2 * 17;
#pragma unroll
        for (int t = 0; t < WT; t++) {
            const int i = lane + 32 * t;
            if (i < ndw) sw[buf][wave][i] = wr[t];
        }
    };

    load_chunk(0);
    for (int c = 0; c < nchunks; c++) {
        const int buf = c & 1;
        store_chunk(buf, c);
        if (c + 1 < nchunks) load_chunk(c + 1);     // overlaps the barrier and the compute below
        __syncthreads();

        if (live) {
            const int cnt = (nb - c * C < C) ? (nb - c * C) : C;
            const unsigned char* wb = reinterpret_cast<const unsigned char*>(&sw[buf][wave][0]);
#pragma unroll 4
            for (int b = 0; b < cnt; b++) {
                const unsigned char* blk = wb + b * 34;
                const float d = (float)*reinterpret_cast<const _Float16*>(blk);
                const float dq = d * (float)(signed char)blk[2 + lane];
#pragma unroll
                for (int r = 0; r < R; r++)
                    acc[r] = __builtin_fmaf(dq, sx[buf][r * C * 32 + b * 32 + lane], acc[r]);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < R; r++) {
        float s = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        if (live && lane == 0) y[(size_t)r * M + row] = s;
    }
}

#define GEMV_Q8_0_FAST(R)                                                     \
    __launch_bounds__(Q8F_THREADS)                                            \
    extern "C" __global__ void gemv_q8_0_fast_r##R(                           \
        const unsigned char* __restrict__ A_q8,                               \
        const float* __restrict__ x,                                          \
        float* __restrict__ y,                                                \
        int M, int K                                                          \
    ) {                                                                       \
        gemv_q8_0_fast_body<R>(A_q8, x, y, M, K);                             \
    }

GEMV_Q8_0_FAST(1)
