// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if HIPFIRE_RMSNORM_EMIT_Q81
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_q8_1
struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");
#else
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate
#endif

// Fused RMSNorm + FWHT rotation for MagnumQuant inputs.
//
// VARIANT R29c1 rms_p1_pref_xw:
// BOD latency/occ-starved (occ=0.2%, mem_busy=18.3%, L2=73.8%, vgpr=48):
// decode launches grid=[1] so occupancy is STRUCTURALLY capped (1 block /
// 64 CUs). R11 fwht_softpipe1 (next-group softpipe) was INCONCLUSIVE and
// only added VGPR (48→72) without raising waves. Softpipe of NEXT group
// is also a no-op for the A3B hot shape K=2048 (groups_total=8 =
// num_warps → groups_per_warp=1).
//
// Hypothesis: the real latency to hide is Phase-1b's two barriers + warp
// reduce + rsqrt — currently the first (and only) group's x/weight loads
// wait until AFTER rms is published. Prefetch this warp's first group's
// x+weight (+ lane-local signs1/2) immediately after Phase-1a sum-of-
// squares, so those global loads overlap the reduction tree. Math is
// identical: still ((x*w)*rms)*s1 then FWHT then *s2/16. Only load
// issue time moves earlier.

__launch_bounds__(256, 1)
extern "C" __global__ void HIPFIRE_RMSNORM_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ x_rot,
#if HIPFIRE_RMSNORM_EMIT_Q81
    block_q8_1_mmq* __restrict__ x_q8,
#endif
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

    const long long batch_off = (long long)blockIdx.x * K;
    const float* x_b     = x     + batch_off;
          float* x_rot_b = x_rot + batch_off;

    // Phase 1a: accumulate sum-of-squares (same stride/order as baseline).
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        local_sum += v * v;
    }

    // Prefetch first FWHT group's x/weight (+ lane-local signs) so the
    // loads ride under Phase-1b barriers. groups layout is wave-uniform
    // (depends only on K and warp_id), so this is safe before rms exists.
    const int groups_total = K / 256;
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;
    const int d0 = lane_id * 8;

    float4 pref_x0, pref_x1, pref_w0, pref_w1;
    float4 s10, s11, s20, s21;
    const int pref_group = warp_group_start;
    const bool have_pref = (groups_per_warp > 0 && pref_group < groups_total);
    if (have_pref) {
        const int base = pref_group * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x_b + base);
        pref_x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
        pref_w0 = *reinterpret_cast<const float4*>(weight + base);
        pref_w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    }
    // signs1/2 are indexed only by lane (not group) — load once for the warp.
    s10 = *reinterpret_cast<const float4*>(signs1 + d0);
    s11 = *reinterpret_cast<const float4*>(signs1 + d0 + 4);
    s20 = *reinterpret_cast<const float4*>(signs2 + d0);
    s21 = *reinterpret_cast<const float4*>(signs2 + d0 + 4);

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

    // Phase 2: FWHT. First group uses prefetched x/w; later groups (if any)
    // load as before. signs already in registers.
    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int base = group * 256 + d0;

        float4 x0, x1, w0, w1;
        if (gi == 0 && have_pref) {
            x0 = pref_x0; x1 = pref_x1;
            w0 = pref_w0; w1 = pref_w1;
        } else {
            x0 = *reinterpret_cast<const float4*>(x_b + base);
            x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
            w0 = *reinterpret_cast<const float4*>(weight + base);
            w1 = *reinterpret_cast<const float4*>(weight + base + 4);
        }

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

#if HIPFIRE_RMSNORM_EMIT_Q81
        // Emit the two Q8_1 blocks covered by this 256-element FWHT group
        // while all eight rotated values are still live. Four adjacent lanes
        // own one 32-value Q8_1 sub-block (8 values/lane). The reduction below
        // reproduces the standalone quantizer's virtual 8-thread tree:
        // first values[0..3] and values[4..7] across lanes 0/2 then 0/1,
        // followed by the final half combine.
        float amax_a = fmaxf(fmaxf(fabsf(out0.x), fabsf(out0.y)),
                             fmaxf(fabsf(out0.z), fabsf(out0.w)));
        float amax_b = fmaxf(fmaxf(fabsf(out1.x), fabsf(out1.y)),
                             fmaxf(fabsf(out1.z), fabsf(out1.w)));
        float sum_a = out0.x + out0.y + out0.z + out0.w;
        float sum_b = out1.x + out1.y + out1.z + out1.w;
        amax_a = fmaxf(amax_a, __shfl_xor(amax_a, 2, 32));
        amax_b = fmaxf(amax_b, __shfl_xor(amax_b, 2, 32));
        sum_a += __shfl_xor(sum_a, 2, 32);
        sum_b += __shfl_xor(sum_b, 2, 32);
        amax_a = fmaxf(amax_a, __shfl_xor(amax_a, 1, 32));
        amax_b = fmaxf(amax_b, __shfl_xor(amax_b, 1, 32));
        sum_a += __shfl_xor(sum_a, 1, 32);
        sum_b += __shfl_xor(sum_b, 1, 32);
        const float amax = fmaxf(amax_a, amax_b);
        const float sum = sum_a + sum_b;
        const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;

        #define Q81(v) ((char)lrintf(fmaxf(-127.0f, fminf(127.0f, (v) * id))))
        const char4 q0 = make_char4(Q81(out0.x), Q81(out0.y), Q81(out0.z), Q81(out0.w));
        const char4 q1 = make_char4(Q81(out1.x), Q81(out1.y), Q81(out1.z), Q81(out1.w));
        #undef Q81

        const int block_in_group = lane_id >> 4;
        const int qblock = group * 2 + block_in_group;
        const int qbyte = (lane_id & 15) * 8;
        block_q8_1_mmq* qb = x_q8 + qblock;
        *reinterpret_cast<char4*>(qb->qs + qbyte) = q0;
        *reinterpret_cast<char4*>(qb->qs + qbyte + 4) = q1;
        if ((lane_id & 3) == 0) {
            qb->ds4[(lane_id >> 2) & 3] = make_half2((_Float16)d, (_Float16)sum);
        }
#endif
    }
}

#undef HIPFIRE_RMSNORM_KERNEL
