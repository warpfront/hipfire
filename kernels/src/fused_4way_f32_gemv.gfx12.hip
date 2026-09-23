// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// gfx12 (RDNA4) 4-way fused F32 GEMV for the PARO MoE gate-side preamble.
// One kernel launch performs all four F32 dense projections from a single
// shared x input:
//
//   y_router = W_router · x        (M=num_experts, e.g. 256)
//   y_seg    = W_seg    · x        (M=1, shared_expert_gate)
//   y_gate   = W_gate   · x        (M=smi=512)
//   y_up     = W_up     · x        (M=smi=512)
//
// Grid = (router_m + seg_m + gate_m + up_m) workgroups × 32 threads.
// Each WG picks its target matrix/row via blockIdx.x range check, then
// runs a wave32 reduction inner loop.
//
// Mirrors the MQ4 `fused_qkvza_hfq4g256.hip` pattern. The MQ4 sister
// claims ~36 µs/layer saved on gfx1100 (4 launches → 1 = 3 saved per
// layer). For z-lab A3B-PARO decode (40 layers × ~3 launches saved
// × ~12 µs gfx12 launch overhead = ~1.4 ms/token wall savings ≈ +9 %).
//
// === Why this kernel ===
//
// Per-layer F32 GEMV decomposition on z-lab A3B-PARO:
//   1. router            M=256, K=2048
//   2. shared_expert_gate M=1,  K=2048
//   3. shared_expert.gate M=512, K=2048
//   4. shared_expert.up   M=512, K=2048
//
// All four read the SAME x (post-rmsnorm). Currently 4 separate
// gemv_f32 launches at ~24.5 µs each. Fusion saves 3 launches × 40 layers
// = 120 launches/token of host+dispatch overhead. Bit-exact with 4
// sequential gemv_f32 calls.

#if defined(__gfx1200__) || defined(__gfx1201__)

__launch_bounds__(32, 16)
extern "C" __global__ void fused_4way_f32_gemv_gfx12(
    const float* __restrict__ A_router,
    const float* __restrict__ A_seg,
    const float* __restrict__ A_gate,
    const float* __restrict__ A_up,
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

    // Route this workgroup to its target matrix + output row.
    const float* A;
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
    const float* a_row = A + (long long)row * K;

    // Stride-32 accumulation across K. Wave32 hardware → all 32 lanes
    // active. No LDS — single-wave shfl_down reduction at end.
    float sum = 0.0f;
    for (int k = tid; k < K; k += 32) {
        sum += a_row[k] * x[k];
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

extern "C" __global__ void fused_4way_f32_gemv_gfx12(
    const float*, const float*, const float*, const float*,
    const float*,
    float*, float*, float*, float*,
    int, int, int, int, int
) {}

#endif
