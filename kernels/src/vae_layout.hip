// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Layout helpers for the FLUX VAE GEMM route.
//
// - vae_transpose_f32_naive: `[m][n]` -> `[n][m]`, one thread per element.
//   The default: on the VAE's skinny shapes it measured faster than the LDS
//   tile below. The WMMA GEMM writes conv outputs position-major
//   `[n_pix][c_out]`; the decoder walks channel-major `[c_out][n_pix]`, so
//   each GEMM conv transposes once.
// - vae_transpose_f32: 32x32 LDS-tiled variant (padded +1 against bank
//   conflicts), kept for A/B behind HIPFIRE_VAE_TRANSPOSE=tiled.
// - vae_transpose_cast_f16: f32 `[m][n]` -> f16 `[n][m]`, the fused
//   transpose+cast that prepares a channel-major activation as the
//   position-major f16 GEMM operand (shared by the q/k/v projections of the
//   mid attention).
// - vae_cast_scale_f16: f32 -> f16 elementwise cast with a scale folded in,
//   used to fold the attention `1/sqrt(c)` into q during the GEMM cast so
//   the scores need no separate scaling pass.

#include <hip/hip_runtime.h>

#define VAE_TILE 32

extern "C" __global__ void vae_transpose_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m,
    int n
) {
    __shared__ float tile[VAE_TILE][VAE_TILE + 1];
    int x0 = blockIdx.x * VAE_TILE; // src col
    int y0 = blockIdx.y * VAE_TILE; // src row
    for (int j = threadIdx.y; j < VAE_TILE && y0 + j < m; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && x0 + i < n; i += blockDim.x) {
            tile[j][i] = src[(long)(y0 + j) * n + (x0 + i)];
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < VAE_TILE && x0 + j < n; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && y0 + i < m; i += blockDim.x) {
            dst[(long)(x0 + j) * m + (y0 + i)] = tile[i][j];
        }
    }
}

// One-thread-per-element scatter variant, kept for A/B (`HIPFIRE_VAE_TRANSPOSE=naive`):
// on the VAE's skinny shapes (n = 128..512 channels, m up to 1M pixels) it
// measured within reach of the tiled kernel without the LDS round trip.
extern "C" __global__ void vae_transpose_f32_naive(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m,
    int n
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)m * n;
    if (t >= total) return;
    int i = (int)(t / n);
    int j = (int)(t % n);
    dst[(long)j * m + i] = src[t];
}

extern "C" __global__ void vae_transpose_cast_f16(
    const float* __restrict__ src,
    _Float16* __restrict__ dst,
    int m,
    int n
) {
    __shared__ float tile[VAE_TILE][VAE_TILE + 1];
    int x0 = blockIdx.x * VAE_TILE; // src col
    int y0 = blockIdx.y * VAE_TILE; // src row
    for (int j = threadIdx.y; j < VAE_TILE && y0 + j < m; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && x0 + i < n; i += blockDim.x) {
            tile[j][i] = src[(long)(y0 + j) * n + (x0 + i)];
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < VAE_TILE && x0 + j < n; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && y0 + i < m; i += blockDim.x) {
            dst[(long)(x0 + j) * m + (y0 + i)] = (_Float16)tile[i][j];
        }
    }
}

extern "C" __global__ void vae_cast_scale_f16(
    const float* __restrict__ src,
    _Float16* __restrict__ dst,
    float scale,
    long n
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (t >= n) return;
    dst[t] = (_Float16)(src[t] * scale);
}

// ─── banded (strided-destination) transposes ────────────────────────────
//
// The GEMM conv route stages the image in horizontal bands so the f16 column
// matrix stays bounded (see `vae_gpu.rs::conv3x3`). A band's GEMM output is
// position-major `[m][n]` over `m` band pixels, and it has to land inside the
// full channel-major `[n][dst_stride]` image at column `dst_off`:
//
//     dst[j * dst_stride + dst_off + i] = src[i * n + j]
//
// `dst_stride = h*w` and `dst_off = y0*w` recover the whole-image transpose,
// which is what the unbanded callers pass.
extern "C" __global__ void vae_transpose_f32_naive_banded(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m,
    int n,
    long dst_stride,
    long dst_off
) {
    long t = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)m * n;
    if (t >= total) return;
    int i = (int)(t / n);
    int j = (int)(t % n);
    dst[(long)j * dst_stride + dst_off + i] = src[t];
}

extern "C" __global__ void vae_transpose_f32_banded(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int m,
    int n,
    long dst_stride,
    long dst_off
) {
    __shared__ float tile[VAE_TILE][VAE_TILE + 1];
    int x0 = blockIdx.x * VAE_TILE; // src col
    int y0 = blockIdx.y * VAE_TILE; // src row
    for (int j = threadIdx.y; j < VAE_TILE && y0 + j < m; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && x0 + i < n; i += blockDim.x) {
            tile[j][i] = src[(long)(y0 + j) * n + (x0 + i)];
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < VAE_TILE && x0 + j < n; j += blockDim.y) {
        for (int i = threadIdx.x; i < VAE_TILE && y0 + i < m; i += blockDim.x) {
            dst[(long)(x0 + j) * dst_stride + dst_off + (y0 + i)] = tile[i][j];
        }
    }
}
