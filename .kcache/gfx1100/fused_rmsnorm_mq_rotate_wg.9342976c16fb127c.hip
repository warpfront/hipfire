// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Workgroup-per-FWHT-group variant of fused_rmsnorm_mq_rotate (BIT-IDENTICAL).
//
// Baseline launches grid=[1]: one 256-thread block owns all K/256 FWHT groups
// (groups_per_warp groups per warp), so occupancy is structurally capped at
// 1 block. Here grid = K/256 workgroups of 256 threads; workgroup b owns
// exactly FWHT group b.
//
// Bit-exactness argument (verified against fused_rmsnorm_mq_rotate.hip):
//  - Phase 1a: EVERY workgroup recomputes the full sum-of-squares with the
//    baseline's exact thread striding (`for i = tid; i < K; i += 256`), so
//    each tid's local_sum is identical to the baseline's.
//  - Phase 1b: the exact same block-reduction tree (same shuffle offsets,
//    same LDS order, same `rsqrtf(warp_sum / (float)K + eps)` expression),
//    so every workgroup derives a bit-identical `rms` with no inter-WG sync.
//  - Phase 2: the FWHT stage uses only lane-local values (d0 = lane_id * 8),
//    lane-indexed signs1/2 loads (group-independent), the group-indexed
//    x/weight/x_rot base, and ds_swizzle shuffles *within* the executing
//    wave. warp_id appears only in group assignment and reduce[warp_id]
//    indexing — never in the arithmetic — and no LDS is shared across warps
//    in this stage. Executing group b on warp 0 of workgroup b is therefore
//    identical to executing it on the baseline's assigned warp.
//
// Kernarg signature is identical to the baseline. No batch offset: blockIdx.x
// is the FWHT group index (the baseline's grid=[1] batch_off is always 0 in
// the default decode route).

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_mq_rotate_wg(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
    int K,
    float eps
) {
    extern __shared__ float smem[];
    float* reduce = smem;              // [256]

    const int tid = threadIdx.x;
    const int block_size = blockDim.x; // 256
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;      // 0..7
    const int num_warps = block_size >> 5;

    const float* x_b     = x;
          float* x_rot_b = x_rot;

    // Phase 1a: accumulate sum-of-squares (same stride/order as baseline).
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        local_sum += v * v;
    }

    const int d0 = lane_id * 8;

    // signs1/2 are indexed only by lane (not group) — same loads as baseline.
    float4 s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    float4 s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    float4 s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    float4 s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

    // Phase 1b: block reduction -> RMS (unchanged tree).
    float warp_sum = local_sum;
    warp_sum += __shfl_down(warp_sum, 16);
    warp_sum += __shfl_down(warp_sum, 8);
    warp_sum += __shfl_down(warp_sum, 4);
    warp_sum += __shfl_down(warp_sum, 2);
    warp_sum += __shfl_down(warp_sum, 1);
    if (lane_id == 0) reduce[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        warp_sum = (tid < num_warps) ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)K + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

    // Phase 2: FWHT. This workgroup owns exactly one group; warp 0 executes
    // it with the identical sequence of operations. Other warps idle here
    // (no __syncthreads below, so early exit is safe).
    const int groups_total = K / 256;
    const int group = (int)blockIdx.x;
    if (group >= groups_total) return;
    if (warp_id != 0) return;

    const int base = group * 256 + d0;

    float4 x0 = *reinterpret_cast<const float4*>(x_b + base);
    float4 x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
    float4 w0 = *reinterpret_cast<const float4*>(weight + base);
    float4 w1 = *reinterpret_cast<const float4*>(weight + base + 4);

    // Exact multiply order: ((x * weight) * rms) * sign.
    float v0 = x0.x * w0.x * rms * s10.x;
    float v1 = x0.y * w0.y * rms * s10.y;
    float v2 = x0.z * w0.z * rms * s10.z;
    float v3 = x0.w * w0.w * rms * s10.w;
    float v4 = x1.x * w1.x * rms * s11.x;
    float v5 = x1.y * w1.y * rms * s11.y;
    float v6 = x1.z * w1.z * rms * s11.z;
    float v7 = x1.w * w1.w * rms * s11.w;

    // Local butterfly: strides 1, 2, 4.
    float t;
    t = v0; v0 = v0 + v1; v1 = t - v1;
    t = v2; v2 = v2 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v5; v5 = t - v5;
    t = v6; v6 = v6 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v2; v2 = t - v2;
    t = v1; v1 = v1 + v3; v3 = t - v3;
    t = v4; v4 = v4 + v6; v6 = t - v6;
    t = v5; v5 = v5 + v7; v7 = t - v7;

    t = v0; v0 = v0 + v4; v4 = t - v4;
    t = v1; v1 = v1 + v5; v5 = t - v5;
    t = v2; v2 = v2 + v6; v6 = t - v6;
    t = v3; v3 = v3 + v7; v7 = t - v7;

    #define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
    } while (0)
    #define HBFLY8(pat, str) \
        HBFLY(v0,pat,str); HBFLY(v1,pat,str); HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
        HBFLY(v4,pat,str); HBFLY(v5,pat,str); HBFLY(v6,pat,str); HBFLY(v7,pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY

    const float s = 0.0625f;
    float4 out0;
    float4 out1;
    out0.x = v0 * s * s20.x;
    out0.y = v1 * s * s20.y;
    out0.z = v2 * s * s20.z;
    out0.w = v3 * s * s20.w;
    out1.x = v4 * s * s21.x;
    out1.y = v5 * s * s21.y;
    out1.z = v6 * s * s21.z;
    out1.w = v7 * s * s21.w;
    *reinterpret_cast<float4*>(x_rot_b + base) = out0;
    *reinterpret_cast<float4*>(x_rot_b + base + 4) = out1;
}
