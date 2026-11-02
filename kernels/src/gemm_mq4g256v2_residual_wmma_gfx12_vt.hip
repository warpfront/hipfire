// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 MQ4G256V2 residual GEMM (Y += W @ X; also the batched lm_head),
// verify-tile twin of gemm_mq4g256v2_residual_wmma_gfx12 (16 < N <= 128).
// Prefixed at build time by gemm_mq4g256v2_wmma_gfx12_vt_core.hip (kernels.rs
// concat!), which documents the design and the byte-exactness argument. Same
// ABI as the one-tile kernel.
//
// Symbols: gemm_mq4g256v2_residual_wmma_gfx12_vt{2,3,4}w{4,8} — BT = ceil(N/16), 4 or 8 waves per block,
// 128-K slabs (16 < N < 64); gemm_mq4g256v2_residual_wmma_gfx12_vt{4,8}w{4,8}_k32 — wide exact verify,
// 32-K slabs, BT4 for N = 64, BT8 for 65 <= N <= 128 (static LDS 10,240 / 20,480 B).
// Grid: [ceil(M/(16*W)), ceil(N/(16*BT))]. Block: [32*W].
// Accumulate semantics (Y += acc, once per output after the whole K chain): the caller pre-initializes Y.

struct ResidualVtOp {
    const char* A;
    float* Y;
    int M;
    __device__ __forceinline__ const char* row(int sr, long long row_stride) const {
        return A + (long long)sr * row_stride;
    }
    __device__ __forceinline__ void store(int out_row, int out_col, float v) const {
        Y[(long long)out_col * M + out_row] += v;
    }
};

#define GEN_RESIDUAL_VT(BT, W, SLAB, SFX) \
extern "C" __launch_bounds__(W * 32) __global__ void gemm_mq4g256v2_residual_wmma_gfx12_vt##BT##w##W##SFX( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int batch_size) { \
    const ResidualVtOp op{A, Y, M}; \
    mq4v2_vt_gemm<BT, W, SLAB>(op, X, M, K, batch_size); \
}

GEN_RESIDUAL_VT(2, 4, 128, )
GEN_RESIDUAL_VT(2, 8, 128, )
GEN_RESIDUAL_VT(3, 4, 128, )
GEN_RESIDUAL_VT(3, 8, 128, )
GEN_RESIDUAL_VT(4, 4, 128, )
GEN_RESIDUAL_VT(4, 8, 128, )
GEN_RESIDUAL_VT(4, 4, 32, _k32)
GEN_RESIDUAL_VT(4, 8, 32, _k32)
GEN_RESIDUAL_VT(8, 4, 32, _k32)
GEN_RESIDUAL_VT(8, 8, 32, _k32)
