// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Lane-batched twins of the A3B (`qwen3_5_moe`, `.mq4rs`) decode MoE block,
// exact gfx1201, for the VMM continuous-batching exact route
// (`forward_slots/vmm/exact.rs`): N decode lanes (1..=8) share one launch of
// each kernel, and lane r's bytes are the singleton kernel's bytes on lane r's
// rows.
//
// The host (crates/rdna-compute/src/moe_decode_lanes.rs `moe_lane_module_source`)
// assembles each module as
//
//   [#define <entry macro> <kernel>_row_body]      (macro-named singletons)
//   <singleton source; its `extern "C" ... __global__` entry header rewritten
//    to `__device__ __forceinline__ void <kernel>_row_body(`; nothing else
//    changed>
//   #define HIPFIRE_MOE_LANES_<X> 1
//   <this file>
//
// so every lane runs the singleton kernel's own body (same arithmetic, same
// reduction order, same FMA contraction, same compiler flags: the router module
// keeps the singleton's whole-TU `-mwavefrontsize64`, the others add none) with
// the singleton's grid x/y and block, plus the lane index in a spare grid
// dimension and the lane's row of every per-lane operand. Weights, expert
// pointer tables and sign tables are shared by all lanes. The wrappers pass the
// singleton exactly the arguments a per-lane singleton launch passes (per-lane
// pointers advanced by one lane's footprint).
//
// Layout contract (row-major f32/i32 `[lanes][len]`, the `PrefillBatchScratch`
// fields):
//   router rows    logits [lanes][n_exp]      -> topk_idx/topk_w [lanes][8]
//   shared down    x (silu rotated) [lanes][K=512], c_buf [lanes][1],
//                  y (residual stream) [lanes][M=2048], weights shared
//   gate/up        x_rot [lanes][K=2048], topk_idx [lanes][8],
//                  y_gate/y_up [lanes][8][M/2=512], expert table shared
//   nine-path      topk_idx/topk_w [lanes][8], gate/up [lanes][8][512],
//                  out (residual stream) [lanes][down_m=2048], table shared
//
// Lane axes: router grid.x = lane (the singleton grid is 1), shared down
// grid.z, gate/up grid.z, nine-path grid.y.

#ifdef HIPFIRE_MOE_LANES_ROUTER
// Singleton: `moe_router_softmax_topk_k8_wave64`, grid 1 x block 64.
extern "C" __launch_bounds__(64, 4)
__global__ void moe_router_softmax_topk_k8_wave64_rows(
    const float* __restrict__ logits,   // [lanes, n_exp]
    int* __restrict__ topk_idx,         // [lanes, 8]
    float* __restrict__ topk_w,         // [lanes, 8]
    int n_exp,
    int norm_topk
) {
    const size_t r = blockIdx.x;
    moe_router_softmax_topk_k8_wave64_row_body(
        logits + r * (size_t)n_exp,
        topk_idx + r * 8,
        topk_w + r * 8,
        n_exp, norm_topk);
}
#endif

#ifdef HIPFIRE_MOE_LANES_SHARED_DOWN
// Singleton: `gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2`, grid M/2 x block 32.
extern "C" __launch_bounds__(32, 16)
__global__ void gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_rows(
    const char* __restrict__ A,         // shared [M] rows of 272 B
    const float* __restrict__ x,        // [lanes, K]
    float* __restrict__ y,              // [lanes, M]
    const float* __restrict__ c_buf,    // [lanes]
    int M, int K
) {
    const size_t r = blockIdx.z;
    gemv_mq4g256v2_residual_sigmoid_scaled_k512_r2_row_body(
        A,
        x + r * (size_t)K,
        y + r * (size_t)M,
        c_buf + r,
        M, K);
}
#endif

#ifdef HIPFIRE_MOE_LANES_GATE_UP
// Singleton: `gemv_mq4g256v2_moe_gate_up_k8_indexed`, grid (M/2, 8) x block 32.
extern "C" __launch_bounds__(32, 16)
__global__ void gemv_mq4g256v2_moe_gate_up_k8_indexed_rows(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp], shared
    const int* __restrict__ topk_indices,               // [lanes, 8]
    const float* __restrict__ x,                        // [lanes, K]
    float* __restrict__ y_gate,                         // [lanes, 8, M/2]
    float* __restrict__ y_up,                           // [lanes, 8, M/2]
    int M, int K, int n_experts
) {
    const size_t r = blockIdx.z;
    const size_t mi = (size_t)(M >> 1);
    gemv_mq4g256v2_moe_gate_up_k8_indexed_row_body(
        expert_ptrs,
        topk_indices + r * 8,
        x + r * (size_t)K,
        y_gate + r * 8 * mi,
        y_up + r * 8 * mi,
        M, K, n_experts);
}
#endif

#ifdef HIPFIRE_MOE_LANES_NINEPATH
// Singleton: `gemv_mq4g128v2_moe_ninepath_d4`, grid down_m/16 x block 256.
extern "C" __launch_bounds__(256, 4)
__global__ void gemv_mq4g128v2_moe_ninepath_d4_rows(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp], shared
    const int* __restrict__ topk_indices,               // [lanes, 8]
    const float* __restrict__ topk_weights,             // [lanes, 8]
    const float* __restrict__ gate,                     // [lanes, 8, down_k]
    const float* __restrict__ up,                       // [lanes, 8, down_k]
    const float* __restrict__ signs1,                   // [128], shared
    const float* __restrict__ signs2,                   // [128], shared
    float* __restrict__ out,                            // [lanes, down_m]
    int down_m, int down_k, int n_experts
) {
    const size_t r = blockIdx.y;
    gemv_mq4g128v2_moe_ninepath_d4_row_body(
        expert_ptrs,
        topk_indices + r * 8,
        topk_weights + r * 8,
        gate + r * 8 * (size_t)down_k,
        up + r * 8 * (size_t)down_k,
        signs1, signs2,
        out + r * (size_t)down_m,
        down_m, down_k, n_experts);
}
#endif
