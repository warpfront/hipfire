// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) Q8_0-weighted 4-way fused GEMV for the PARO MoE
// gate-side preamble. Mirror of fused_4way_f32_gemv_gfx12 but reads
// Q8_0 quantized weights (4× less BW) instead of F32.
//
// Q8_0 block format (32 elements per block):
//   2 bytes: F16 scale (d)
//  32 bytes: int8 quantized values
//  Total 34 bytes per 32 K-elements = 1.0625 bytes/element vs F32's 4
//
// Per call: 4 weights (router + seg + gate + up). One launch handles all
// from a single shared F32 X input. Wave32 (32 threads), one WG per
// output row.
//
// rocprof shows fused_4way_f32_gemv at 11.6 percent of decode GPU time
// (~537 ms / 500 tokens = 1.07 ms/token). Q8 version expected to save
// most of the BW component → 0.3-0.5 ms/token savings on z-lab A3B-PARO.

#if defined(__gfx1200__) || defined(__gfx1201__)

__launch_bounds__(32, 16)
extern "C" __global__ void fused_4way_q8_0_gemv_gfx12(
    const char*  __restrict__ A_router,    // [router_m * (K/32) * 34] Q8_0
    const char*  __restrict__ A_seg,
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const float* __restrict__ x,           // [K] F32 input
    float*       __restrict__ y_router,    // [router_m] F32 output
    float*       __restrict__ y_seg,
    float*       __restrict__ y_gate,
    float*       __restrict__ y_up,
    int router_m, int seg_m, int gate_m, int up_m,
    int K
) {
    const int gid = blockIdx.x;
    const int total = router_m + seg_m + gate_m + up_m;
    if (gid >= total) return;

    // Route this WG to its target matrix + output row.
    const char* A;
    float* y;
    int row;
    if (gid < router_m) {
        A = A_router; y = y_router; row = gid;
    } else if (gid < router_m + seg_m) {
        A = A_seg;    y = y_seg;    row = gid - router_m;
    } else if (gid < router_m + seg_m + gate_m) {
        A = A_gate;   y = y_gate;   row = gid - router_m - seg_m;
    } else {
        A = A_up;     y = y_up;     row = gid - router_m - seg_m - gate_m;
    }

    const int tid = threadIdx.x;
    const int num_blocks = K / 32;  // 64 for K=2048
    const char* row_base = A + (long long)row * num_blocks * 34;

    // Each wave iterates blocks; each thread within a wave handles 1 of the
    // 32 elements in each block.
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        const char* block_base = row_base + b * 34;
        // F16 scale, broadcast — all threads read the same 2 bytes via L1.
        const _Float16 d = *reinterpret_cast<const _Float16*>(block_base);
        const float scale = (float)d;
        // i8 value at this thread's offset within the block.
        const int8_t v = *reinterpret_cast<const int8_t*>(block_base + 2 + tid);
        const float w = scale * (float)v;
        const int k = b * 32 + tid;
        sum += w * x[k];
    }

    // Wave32 shfl_down reduction.
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) {
        y[row] = sum;
    }
}

#else

extern "C" __global__ void fused_4way_q8_0_gemv_gfx12(
    const char*, const char*, const char*, const char*,
    const float*,
    float*, float*, float*, float*,
    int, int, int, int, int
) {}

#endif
