// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 MQ4G256V2 4-way fused qkv+z+beta+alpha GEMM (Qwen3.5 DeltaNet
// preamble), verify-tile twin of gemm_qkvza_mq4g256v2_wmma_gfx12
// (16 < N <= 128). Prefixed at build time by gemm_mq4g256v2_wmma_gfx12_vt_core.hip
// (kernels.rs concat!), which documents the design and the byte-exactness
// argument. Same ABI as the one-tile kernel.
//
// Symbols: gemm_qkvza_mq4g256v2_wmma_gfx12_vt{2,3,4}w{4,8} — BT = ceil(N/16), 4 or 8 waves per block,
// 128-K slabs (16 < N < 64); gemm_qkvza_mq4g256v2_wmma_gfx12_vt{4,8}w{4,8}_k32 — wide exact verify,
// 32-K slabs, BT4 for N = 64, BT8 for 65 <= N <= 128 (static LDS 10,240 / 20,480 B).
// Grid: [ceil(total_m/(16*W)), ceil(N/(16*BT))]. Block: [32*W].
// Overwrite semantics (y = acc), rows routed across the four projections.

struct QkvzaVtOp {
    const char* A_qkv; const char* A_z; const char* A_beta; const char* A_alpha;
    float* Y_qkv; float* Y_z; float* Y_beta; float* Y_alpha;
    int qkv_m, z_m, beta_m, alpha_m;
    __device__ __forceinline__ const char* row(int sr, long long row_stride) const {
        if (sr < qkv_m) return A_qkv + (long long)sr * row_stride;
        if (sr < qkv_m + z_m) return A_z + (long long)(sr - qkv_m) * row_stride;
        if (sr < qkv_m + z_m + beta_m) return A_beta + (long long)(sr - (qkv_m + z_m)) * row_stride;
        return A_alpha + (long long)(sr - (qkv_m + z_m + beta_m)) * row_stride;
    }
    __device__ __forceinline__ void store(int out_row, int out_col, float v) const {
        if (out_row < qkv_m) {
            Y_qkv[(long long)out_col * qkv_m + out_row] = v;
        } else if (out_row < qkv_m + z_m) {
            Y_z[(long long)out_col * z_m + (out_row - qkv_m)] = v;
        } else if (out_row < qkv_m + z_m + beta_m) {
            Y_beta[(long long)out_col * beta_m + (out_row - (qkv_m + z_m))] = v;
        } else {
            Y_alpha[(long long)out_col * alpha_m + (out_row - (qkv_m + z_m + beta_m))] = v;
        }
    }
};

#define GEN_QKVZA_VT(BT, W, SLAB, SFX) \
extern "C" __launch_bounds__(W * 32) __global__ void gemm_qkvza_mq4g256v2_wmma_gfx12_vt##BT##w##W##SFX( \
    const char* __restrict__ A_qkv, const char* __restrict__ A_z, \
    const char* __restrict__ A_beta, const char* __restrict__ A_alpha, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_qkv, float* __restrict__ Y_z, \
    float* __restrict__ Y_beta, float* __restrict__ Y_alpha, \
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N) { \
    const QkvzaVtOp op{A_qkv, A_z, A_beta, A_alpha, Y_qkv, Y_z, Y_beta, Y_alpha, \
                       qkv_m, z_m, beta_m, alpha_m}; \
    mq4v2_vt_gemm<BT, W, SLAB>(op, X, qkv_m + z_m + beta_m + alpha_m, K, N); \
}

GEN_QKVZA_VT(2, 4, 128, )
GEN_QKVZA_VT(2, 8, 128, )
GEN_QKVZA_VT(3, 4, 128, )
GEN_QKVZA_VT(3, 8, 128, )
GEN_QKVZA_VT(4, 4, 128, )
GEN_QKVZA_VT(4, 8, 128, )
GEN_QKVZA_VT(4, 4, 32, _k32)
GEN_QKVZA_VT(4, 8, 32, _k32)
GEN_QKVZA_VT(8, 4, 32, _k32)
GEN_QKVZA_VT(8, 8, 32, _k32)
