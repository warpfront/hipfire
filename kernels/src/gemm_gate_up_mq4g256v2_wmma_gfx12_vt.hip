// SPDX-License-Identifier: MIT OR Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx12 MQ4G256V2 fused gate+up GEMM, verify-tile twin of
// gemm_gate_up_mq4g256v2_wmma_gfx12 (16 < N <= 128). Prefixed at build time by
// gemm_mq4g256v2_wmma_gfx12_vt_core.hip (kernels.rs concat!), which documents
// the design and the byte-exactness argument. Same ABI as the one-tile kernel.
//
// Symbols: gemm_gate_up_mq4g256v2_wmma_gfx12_vt{2,3,4}w{4,8} — BT = ceil(N/16), 4 or 8 waves per block,
// 128-K slabs (16 < N < 64); gemm_gate_up_mq4g256v2_wmma_gfx12_vt{4,8}w{4,8}_k32 — wide exact verify,
// 32-K slabs, BT4 for N = 64, BT8 for 65 <= N <= 128 (static LDS 10,240 / 20,480 B).
// Grid: [ceil((gate_m+up_m)/(16*W)), ceil(N/(16*BT))]. Block: [32*W].
// Overwrite semantics (y = acc), rows routed across Y_gate / Y_up.

struct GateUpVtOp {
    const char* A_gate; const char* A_up;
    float* Y_gate; float* Y_up;
    int gate_m, up_m;
    __device__ __forceinline__ const char* row(int sr, long long row_stride) const {
        return (sr < gate_m) ? (A_gate + (long long)sr * row_stride)
                             : (A_up + (long long)(sr - gate_m) * row_stride);
    }
    __device__ __forceinline__ void store(int out_row, int out_col, float v) const {
        if (out_row < gate_m) {
            Y_gate[(long long)out_col * gate_m + out_row] = v;
        } else {
            Y_up[(long long)out_col * up_m + (out_row - gate_m)] = v;
        }
    }
};

#define GEN_GATE_UP_VT(BT, W, SLAB, SFX) \
extern "C" __launch_bounds__(W * 32) __global__ void gemm_gate_up_mq4g256v2_wmma_gfx12_vt##BT##w##W##SFX( \
    const char* __restrict__ A_gate, const char* __restrict__ A_up, \
    const _Float16* __restrict__ X, \
    float* __restrict__ Y_gate, float* __restrict__ Y_up, \
    int gate_m, int up_m, int K, int N) { \
    const GateUpVtOp op{A_gate, A_up, Y_gate, Y_up, gate_m, up_m}; \
    mq4v2_vt_gemm<BT, W, SLAB>(op, X, gate_m + up_m, K, N); \
}

GEN_GATE_UP_VT(2, 4, 128, )
GEN_GATE_UP_VT(2, 8, 128, )
GEN_GATE_UP_VT(3, 4, 128, )
GEN_GATE_UP_VT(3, 8, 128, )
GEN_GATE_UP_VT(4, 4, 128, )
GEN_GATE_UP_VT(4, 8, 128, )
GEN_GATE_UP_VT(4, 4, 32, _k32)
GEN_GATE_UP_VT(4, 8, 32, _k32)
GEN_GATE_UP_VT(8, 4, 32, _k32)
GEN_GATE_UP_VT(8, 8, 32, _k32)
