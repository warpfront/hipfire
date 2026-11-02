// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// MagnumWave MW16 WMMA GEMM — pre-dequanted FP16 weights + K2 pipelining.
//
// W is FP16 [M, K] row-major (dequanted at model load time).
// X is FP16 [N, K] row-major (converted per-prefill).
// Y is FP32 [N, M] (residual add).
//
// No dequant → ~30 VGPRs (vs 47 with 4-bit dequant).
// K2 unroll: 2 WMMAs per iteration, vmcnt(2) pipelining.
// The A-load is now a simple FP16 vector load (no nibble extraction),
// so both A and B loads can pipeline with WMMA compute.
//
// Grid: [ceil(M/16), ceil(N/16)]. Block: [32]. K must be multiple of 32.

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mw16_residual_wmma(
    const _Float16* __restrict__ W,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;

    if (row_start >= M || batch_start >= N) return;

    const int safe_row = (row_start + (tid & 15) < M) ? (row_start + (tid & 15)) : (M - 1);
    const int safe_batch = (batch_start + (tid & 15) < N) ? (batch_start + (tid & 15)) : 0;

    const _Float16* wp = W + (long long)safe_row * K;
    const _Float16* xp = X + (long long)safe_batch * K;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    // K2: process 32 K-elements per iteration (2 WMMAs)
    for (int k = 0; k < K; k += 32) {
        // Load both A and B tiles for 2 consecutive K-tiles up front
        half16_t a0 = *(const half16_t*)(wp + k);
        half16_t b0 = *(const half16_t*)(xp + k);
        half16_t a1 = *(const half16_t*)(wp + k + 16);
        half16_t b1 = *(const half16_t*)(xp + k + 16);

        // First WMMA — a1/b1 loads may still be in flight
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a0, b0, acc);
        // Second WMMA — fully pipelined
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, acc);
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (m-dim), B operand = batch X cols (n-dim).
    // Matches `gemm_hfq4g256_residual_wmma` post-correctness-fix convention.
    // (Pre-fix mapping `out_row = row_start + (tid & 15)` + col_base = (tid>>4)*8
    // shipped silently wrong in this kernel from 1c9cc99 2026-04-12 until the
    // DFlash F16-draft path exposed it.)
    const int out_col = batch_start + (tid & 15);
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y[(long long)out_col * M + out_row] += acc[j];
            }
        }
    }
}

// gfx1151 twin of gemm_mw16_residual_wmma for the 16-row DFlash draft GEMMs.
// Per output tile the arithmetic is the base kernel's exactly (one wave owns
// 16 weight rows x 16 batch rows and runs acc = wmma(A_t, B_t, acc) over
// t = 0..K/16-1 in order from 0, then Y += acc), so Y is bitwise equal.
// What differs is memory access only: a block runs 4 waves (64 weight rows)
// and stages each 128-K slab of the 16 x K X panel in LDS once, double-
// buffered, instead of every wave re-reading the whole panel from L2; each
// lane loads its row's 256 B of the slab with sixteen b128 loads up front.
// W rows are `ldw` halfs apart (ldw >= K, ldw % 8 == 0): with a power-of-two
// row pitch (K = 4096 -> 8 KiB) every wave's rows hit the same DRAM channels at
// the same k and gfx1151 streams ~50 GB/s instead of ~210, so the loader pads.
// Grid: [ceil(M / 64), ceil(N / 16)]. Block: [128]. K % 128 == 0.
// Waves past M run the loop on clamped rows (uniform barriers), stores masked.
#define MWX_WAVES 4
#define MWX_KS 128
#define MWX_PITCH (MWX_KS + 8)
typedef unsigned int __attribute__((ext_vector_type(4))) mwx_u4_t;
typedef unsigned int __attribute__((ext_vector_type(8))) mwx_u8_t;

__launch_bounds__(32 * MWX_WAVES, 1)
extern "C" __global__ void gemm_mw16_residual_wmma_gfx1151_xlds(
    const _Float16* __restrict__ W,
    const _Float16* __restrict__ X,
    float* __restrict__ Y,
    int M, int K, int N, int ldw
) {
    __shared__ _Float16 xs_lds[2 * 16 * MWX_PITCH];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int lane16 = lane & 15;
    const int row_start = blockIdx.x * 16 * MWX_WAVES + (tid >> 5) * 16;
    const int batch_start = blockIdx.y * 16;
    if (batch_start >= N) return;

    const int r = row_start + lane16;
    const _Float16* wp = W + (long long)(r < M ? r : M - 1) * ldw;

    // X slab: 16 rows x 128 halfs = 256 vectors of 8 halfs, 2 per thread.
    // Invalid batch rows read batch row 0, as the base kernel does.
    mwx_u4_t xr[2];
    auto x_fetch = [&](int s) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * 32 * MWX_WAVES;
            const int b = v >> 4, c = v & 15;
            const int sb = (batch_start + b < N) ? (batch_start + b) : 0;
            xr[i] = *(const mwx_u4_t*)(X + (long long)sb * K + s * MWX_KS + c * 8);
        }
    };
    auto x_store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * 32 * MWX_WAVES;
            const int b = v >> 4, c = v & 15;
            *(mwx_u4_t*)(xs_lds + buf * 16 * MWX_PITCH + b * MWX_PITCH + c * 8) = xr[i];
        }
    };

    const int S = K / MWX_KS;
    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    x_fetch(0);
    x_store(0);
    __syncthreads();
    for (int s = 0; s < S; ++s) {
        const mwx_u4_t* wv = (const mwx_u4_t*)(wp + s * MWX_KS);
        mwx_u4_t w[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) w[i] = wv[i];
        if (s + 1 < S) x_fetch(s + 1);
        const _Float16* xs = xs_lds + (s & 1) * 16 * MWX_PITCH + lane16 * MWX_PITCH;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            mwx_u8_t a8;
            a8[0] = w[2 * t].x; a8[1] = w[2 * t].y; a8[2] = w[2 * t].z; a8[3] = w[2 * t].w;
            a8[4] = w[2 * t + 1].x; a8[5] = w[2 * t + 1].y;
            a8[6] = w[2 * t + 1].z; a8[7] = w[2 * t + 1].w;
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                __builtin_bit_cast(half16_t, a8), *(const half16_t*)(xs + t * 16), acc);
        }
        if (s + 1 < S) x_store((s + 1) & 1);
        __syncthreads();
    }

    const int out_col = batch_start + lane16;
    if (out_col < N) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (lane >> 4);
            if (out_row < M) Y[(long long)out_col * M + out_row] += acc[j];
        }
    }
}
