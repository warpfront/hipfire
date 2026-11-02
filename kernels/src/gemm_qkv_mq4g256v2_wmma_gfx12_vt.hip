// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 MQ4G256V2 3-way fused q+k+v GEMM (Qwen3.5 FullAttention preamble),
// verify-tile twin of gemm_qkv_mq4g256v2_wmma_gfx12 (16 < N <= 128). Prefixed at
// build time by gemm_mq4g256v2_wmma_gfx12_vt_core.hip (kernels.rs concat!),
// which documents the design and the byte-exactness argument. Same ABI as the
// one-tile kernel.
//
// Symbols: gemm_qkv_mq4g256v2_wmma_gfx12_vt{2,3,4}w{4,8} — BT = ceil(N/16), 4 or 8 waves per block,
// 128-K slabs (16 < N < 64); gemm_qkv_mq4g256v2_wmma_gfx12_vt{4,8}w{4,8}_k32 — wide exact verify,
// 32-K slabs, BT4 for N = 64, BT8 for 65 <= N <= 128 (static LDS 10,240 / 20,480 B).
// Grid: [ceil((q_m+k_m+v_m)/(16*W)), ceil(N/(16*BT))]. Block: [32*W].
// Overwrite semantics (y = acc), rows routed across Y_q / Y_k / Y_v.

struct QkvVtOp {
    const char* A_q; const char* A_k; const char* A_v;
    float* Y_q; float* Y_k; float* Y_v;
    int q_m, k_m, v_m;
    __device__ __forceinline__ const char* row(int sr, long long row_stride) const {
        if (sr < q_m) return A_q + (long long)sr * row_stride;
        if (sr < q_m + k_m) return A_k + (long long)(sr - q_m) * row_stride;
        return A_v + (long long)(sr - q_m - k_m) * row_stride;
    }
    __device__ __forceinline__ void store(int out_row, int out_col, float v) const {
        if (out_row < q_m) {
            Y_q[(long long)out_col * q_m + out_row] = v;
        } else if (out_row < q_m + k_m) {
            Y_k[(long long)out_col * k_m + (out_row - q_m)] = v;
        } else {
            Y_v[(long long)out_col * v_m + (out_row - q_m - k_m)] = v;
        }
    }
};

#define GEN_QKV_VT(BT, W, SLAB, SFX) \
extern "C" __launch_bounds__(W * 32) __global__ void gemm_qkv_mq4g256v2_wmma_gfx12_vt##BT##w##W##SFX( \
    const char* __restrict__ A_q, const char* __restrict__ A_k, const char* __restrict__ A_v, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_q, float* __restrict__ Y_k, float* __restrict__ Y_v, \
    int q_m, int k_m, int v_m, int K, int N) { \
    const QkvVtOp op{A_q, A_k, A_v, Y_q, Y_k, Y_v, q_m, k_m, v_m}; \
    mq4v2_vt_gemm<BT, W, SLAB>(op, X, q_m + k_m + v_m, K, N); \
}

GEN_QKV_VT(2, 4, 128, )
GEN_QKV_VT(2, 8, 128, )
GEN_QKV_VT(3, 4, 128, )
GEN_QKV_VT(3, 8, 128, )
GEN_QKV_VT(4, 4, 128, )
GEN_QKV_VT(4, 8, 128, )
GEN_QKV_VT(4, 4, 32, _k32)
GEN_QKV_VT(4, 8, 32, _k32)
GEN_QKV_VT(8, 4, 32, _k32)
GEN_QKV_VT(8, 8, 32, _k32)
