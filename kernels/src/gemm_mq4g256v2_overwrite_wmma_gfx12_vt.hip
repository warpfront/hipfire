// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 MQ4G256V2 OVERWRITE GEMM (Y = W @ X, no pre-zero fill), verify-tile
// twin of gemm_mq4g256v2_overwrite_wmma_gfx12 (16 < N <= 63) and of
// gemm_mq4g256v2_residual_wmma_gfx12_vt. Prefixed at build time by
// gemm_mq4g256v2_wmma_gfx12_vt_core.hip (kernels.rs concat!), which documents
// the design. Same ABI and per-output arithmetic as the residual vt kernels.
//
// Byte-exactness: the legacy path memsets Y to +0.0f and the residual epilogue
// does one `Y += acc` per output, i.e. (+0.0f) + acc. Each output is stored
// exactly once per launch, so `Y = acc + 0.0f` is bit-identical (including
// acc == -0.0f -> +0.0f; a plain `= acc` would differ). No fast-math in the
// hipcc flags, so the add is not folded.
//
// Symbols: gemm_mq4g256v2_overwrite_wmma_gfx12_vt{2,3,4}w{4,8} — BT = ceil(N/16), 4 or 8 waves per block,
// 128-K slabs (16 < N < 64); gemm_mq4g256v2_overwrite_wmma_gfx12_vt{4,8}w{4,8}_k32 — 32-K slabs
// (same set as the residual file; only the vt2..4 w4/w8 ones are launched by the draft route).
// Grid: [ceil(M/(16*W)), ceil(N/(16*BT))]. Block: [32*W].

struct OverwriteVtOp {
    const char* A;
    float* Y;
    int M;
    __device__ __forceinline__ const char* row(int sr, long long row_stride) const {
        return A + (long long)sr * row_stride;
    }
    __device__ __forceinline__ void store(int out_row, int out_col, float v) const {
        Y[(long long)out_col * M + out_row] = v + 0.0f;
    }
};

#define GEN_OVERWRITE_VT(BT, W, SLAB, SFX) \
extern "C" __launch_bounds__(W * 32) __global__ void gemm_mq4g256v2_overwrite_wmma_gfx12_vt##BT##w##W##SFX( \
    const char* __restrict__ A, const _Float16* __restrict__ X, float* __restrict__ Y, \
    int M, int K, int batch_size) { \
    const OverwriteVtOp op{A, Y, M}; \
    mq4v2_vt_gemm<BT, W, SLAB>(op, X, M, K, batch_size); \
}

GEN_OVERWRITE_VT(2, 4, 128, )
GEN_OVERWRITE_VT(2, 8, 128, )
GEN_OVERWRITE_VT(3, 4, 128, )
GEN_OVERWRITE_VT(3, 8, 128, )
GEN_OVERWRITE_VT(4, 4, 128, )
GEN_OVERWRITE_VT(4, 8, 128, )
GEN_OVERWRITE_VT(4, 4, 32, _k32)
GEN_OVERWRITE_VT(4, 8, 32, _k32)
GEN_OVERWRITE_VT(8, 4, 32, _k32)
GEN_OVERWRITE_VT(8, 8, 32, _k32)
