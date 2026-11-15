// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx1151 F16-weight WMMA GEMM for small batches (the DSpark drafter's
// 151936 x 4096 F16 lm_head at 1..7 rows). Same contract as
// `gemm_mw16_residual_wmma_f16x`: Y[N, M] = X[N, K] @ W[M, K]^T, W and X FP16,
// Y FP32 written as 0 + acc (that launcher pre-zeroes Y and adds).
//
// Per output the arithmetic is gemm_mw16_residual_wmma's exactly, so outputs
// are byte-identical to it: one wave owns 16 weight rows x 16 batch rows and
// runs the whole K chain acc = wmma_f32_16x16x16_f16(W_t, X_t, acc) for
// t = 0..K/16-1 in order from acc = 0; invalid batch rows read row 0.
// What differs is memory access only. The base kernel runs one single-wave
// block per 16 rows and keeps 64 B per lane in flight, ~70 GB/s on the 1.24 GB
// head. Here a block runs 4 waves (64 weight rows), stages each 128-K slab of
// the X panel in LDS once (double-buffered), and each lane loads its row's next
// 256 B (16 b128 loads) while the current slab's 8 WMMAs run: ~150 GB/s.
//
// Grid: [ceil(M / 64), ceil(N / 16), 1]. Block: [128, 1, 1].
// Static LDS: 2 x 16 x 136 halfs = 8,704 B. Requires K % 128 == 0 and a 16 B
// aligned W. Waves whose 16-row tile starts past M still run the loop (clamped
// rows) so the block barriers stay uniform; their stores are masked.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef unsigned int __attribute__((ext_vector_type(4))) u4_t;

#define F16X_WAVES 4
#define F16X_THREADS (32 * F16X_WAVES)
#define F16X_GROUP 128
// X slab row pitch 136 halfs (272 B): 16 lanes' b128 reads hit distinct banks.
#define F16X_PITCH 136
#define F16X_SLAB (16 * F16X_PITCH)
// 16 batch rows x 128 halfs = 256 vectors of 16 B.
#define F16X_PER ((16 * F16X_GROUP / 8 + F16X_THREADS - 1) / F16X_THREADS)

__launch_bounds__(F16X_THREADS)
extern "C" __global__ void gemm_f16_wmma_gfx1151_xlds(
    const _Float16* __restrict__ W, // [M, K] FP16
    const _Float16* __restrict__ X, // [N, K] FP16
    float* __restrict__ Y,          // [N, M] FP32
    int M, int K, int N) {
    __shared__ __align__(16) _Float16 xs_lds[2 * F16X_SLAB];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = (blockIdx.x * F16X_WAVES + (tid >> 5)) * 16;
    const int batch_start = blockIdx.y * 16;
    const int my_row = row_start + lane16;
    const int safe_row = (my_row < M) ? my_row : (M - 1);
    const int G = K / F16X_GROUP;
    const _Float16* row_base = W + (long long)safe_row * K;

    u4_t xr[F16X_PER];
    auto x_fetch = [&](int g) {
#pragma unroll
        for (int i = 0; i < F16X_PER; ++i) {
            const int v = tid + i * F16X_THREADS;
            const int b = v >> 4, c = v & 15;
            const int sb = (batch_start + b < N) ? (batch_start + b) : 0;
            xr[i] = *(const u4_t*)(X + (long long)sb * K + g * F16X_GROUP + c * 8);
        }
    };
    auto x_store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < F16X_PER; ++i) {
            const int v = tid + i * F16X_THREADS;
            const int b = v >> 4, c = v & 15;
            *(u4_t*)(xs_lds + buf * F16X_SLAB + b * F16X_PITCH + c * 8) = xr[i];
        }
    };

    float8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // The next group's 256 B row slice is in flight while this group's 8 WMMAs run.
    half16_t w[8];
    auto w_fetch = [&](int g, half16_t* dst) {
        const half16_t* gp = (const half16_t*)(row_base + (long long)g * F16X_GROUP);
#pragma unroll
        for (int t = 0; t < 8; ++t) dst[t] = gp[t];
    };
    w_fetch(0, w);
    x_fetch(0);
    x_store(0);
    __syncthreads();
    for (int g = 0; g < G; ++g) {
        half16_t wn[8];
        // X first: vmcnt retires in order, so x_store below waits only for xr.
        if (g + 1 < G) {
            x_fetch(g + 1);
            w_fetch(g + 1, wn);
        }
        const _Float16* xs = xs_lds + (g & 1) * F16X_SLAB + lane16 * F16X_PITCH;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                w[t], *(const half16_t*)(xs + t * 16), acc);
        }
        // Slab (g + 1) & 1 was last read in group g - 1, before that group's barrier.
        if (g + 1 < G) x_store((g + 1) & 1);
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 8; ++t) w[t] = wn[t];
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (lane>>4)][lane & 15].
    const int out_col = batch_start + lane16;
    if (row_start < M && out_col < N) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            // 0 + acc: the base launcher's pre-zeroed Y += acc (-0 becomes +0).
            if (out_row < M) Y[(long long)out_col * M + out_row] = 0.0f + acc[j];
        }
    }
}
