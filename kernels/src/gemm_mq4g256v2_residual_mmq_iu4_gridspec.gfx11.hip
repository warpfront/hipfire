// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Appended to the shipped gfx11 IU4 source by the grid-specialization
// experiment. The generic body needs the original N stride for every Xq
// K-block, so pointer rebasing cannot target the final column tile. This
// wrapper preserves that stride and offsets only the logical column tile.

#if HIPFIRE_RDNA3
extern "C" __global__ __launch_bounds__(IU4_MMQ_NWARPS * IU4_WARP_SIZE, 2)
void gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec(
    const char* __restrict__ A,
    const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y,
    int M,
    int K,
    int N,
    int add
) {
    const int tail_col_tile = N / IU4_MMQ_X + blockIdx.y;
    gemm_iu4_body<false>(A, Xq, Y, M, K, N, add, blockIdx.x, tail_col_tile);
}
#else
extern "C" __global__ void gemm_mq4g256v2_residual_mmq_iu4_tail_gridspec(
    const char*, const block_i4_128*, float*, int, int, int, int) {}
#endif
