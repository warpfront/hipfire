// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) HFQ4G256-weighted 4-way fused GEMV. Mirror of fused_4way_q8
// but reads HFQ4G256 quantized weights for 2× less BW than Q8 (4× less than F32).
//
// HFQ4G256 block format (256 elements per block):
//    4 bytes: F32 scale
//    4 bytes: F32 min
//  128 bytes: 4-bit packed weights (low nibble = elem[2i], high nibble = elem[2i+1])
//  Total 136 bytes per 256 K-elements = 0.53 bytes/element.
//
// Per call: 4 weights from a shared F32 X. Wave32 (32 threads), one WG
// per output row.

#if defined(__gfx1200__) || defined(__gfx1201__)

__launch_bounds__(32, 16)
extern "C" __global__ void fused_4way_hfq4g256_gemv_gfx12(
    const char*  __restrict__ A_router,
    const char*  __restrict__ A_seg,
    const char*  __restrict__ A_gate,
    const char*  __restrict__ A_up,
    const float* __restrict__ x,
    float*       __restrict__ y_router,
    float*       __restrict__ y_seg,
    float*       __restrict__ y_gate,
    float*       __restrict__ y_up,
    int router_m, int seg_m, int gate_m, int up_m,
    int K
) {
    const int gid = blockIdx.x;
    const int total = router_m + seg_m + gate_m + up_m;
    if (gid >= total) return;

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
    const int num_blocks = K / 256;  // 8 for K=2048
    const char* row_base = A + (long long)row * num_blocks * 136;

    // 32 threads × 8 nibbles per block (256 elem / 32 threads × 2 elem-per-nibble-byte = needs careful)
    // Layout: nibble[i] for elem i. 2 elements per byte. 128 bytes per block.
    // Each thread covers 8 elements (4 bytes) per block: tid * 4 .. tid * 4 + 3 bytes.

    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        const char* block_base = row_base + b * 136;
        const float scale = __builtin_bit_cast(float, *(const unsigned int*)(block_base));
        const float min_val = __builtin_bit_cast(float, *(const unsigned int*)(block_base + 4));

        // Read 4 packed nibble bytes = 8 weights for this thread.
        const unsigned int packed = *(const unsigned int*)(block_base + 8 + tid * 4);
        const int k_base = b * 256 + tid * 8;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned int nib = (packed >> (i * 4)) & 0xFu;
            const float w = scale * (float)nib + min_val;
            sum += w * x[k_base + i];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (tid == 0) {
        y[row] = sum;
    }
}

#else

extern "C" __global__ void fused_4way_hfq4g256_gemv_gfx12(
    const char*, const char*, const char*, const char*,
    const float*,
    float*, float*, float*, float*,
    int, int, int, int, int
) {}

#endif
