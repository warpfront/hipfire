// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#ifndef HIPFIRE_IU4_GROUP_K
#define HIPFIRE_IU4_GROUP_K 128
#endif
#if HIPFIRE_IU4_GROUP_K != 128 && HIPFIRE_IU4_GROUP_K != 256 && HIPFIRE_IU4_GROUP_K != 512
#error "HIPFIRE_IU4_GROUP_K must be 128, 256, or 512"
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

// -DHIPFIRE_RMSNORM_AWQ=1 builds the AWQ twin `fused_rmsnorm_mq_rotate_awq`:
// one extra `awq_scale[K]` argument, divided in AFTER the rms multiply and
// BEFORE the sign/FWHT (`(((x*w)*rms)/s)*sign`), completing `(W·s)·(x/s)`.
// Same phases, prefetch, and reduction tree as the uniform symbol; the scale
// rides the same float4 prefetch as x/weight. Uniform builds are textually
// unchanged.
#ifdef HIPFIRE_RMSNORM_KERNEL
/* caller (host JIT concat) already named the symbol */
#elif defined(HIPFIRE_RMSNORM_AWQ) && defined(HIPFIRE_FP8_STREAM)
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_mq4v2_fp8_gfx12
#elif defined(HIPFIRE_FP8_STREAM)
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_mq4v2_fp8_gfx12
#elif defined(HIPFIRE_RMSNORM_AWQ) && defined(HIPFIRE_IU4_SIDECAR)
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq_i4
#elif defined(HIPFIRE_IU4_SIDECAR)
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_i4
#elif defined(HIPFIRE_RMSNORM_AWQ)
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate_awq
#else
#define HIPFIRE_RMSNORM_KERNEL fused_rmsnorm_mq_rotate
#endif

__launch_bounds__(256, 1)
extern "C" __global__ void HIPFIRE_RMSNORM_KERNEL(
    const float* __restrict__ x,
    const float* __restrict__ weight,
#ifdef HIPFIRE_RMSNORM_AWQ
    const float* __restrict__ awq_scale,
#endif
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
#ifdef HIPFIRE_IU4_SIDECAR
    float* __restrict__ x_rot,          // nullable when emit_f32=false
    block_i4_128* __restrict__ x_i4,    // [K/128, N] block_i4_128
    int K,
    float eps,
    int N
#elif defined(HIPFIRE_FP8_STREAM)
    float* __restrict__ x_rot,          // [N, K] f32, always written (v1)
    unsigned char* __restrict__ x_fp8,  // [N, K] E4M3, standalone-pack layout
    float* __restrict__ fp8_half_sums,  // [N, K/256, 2] f32 decoded-code sums
    float* __restrict__ fp8_row_scales, // [N] f32 power-of-two-to-224 scales
    int K,
    float eps,
    int N
#else
    float* __restrict__ x_rot,
    int K,
    float eps
#endif
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
#ifdef HIPFIRE_IU4_SIDECAR
          float* x_rot_b = (x_rot != nullptr) ? (x_rot + batch_off) : nullptr;
#else
          float* x_rot_b = x_rot + batch_off;
#endif

    // Phase 1a: accumulate sum-of-squares (same stride/order as baseline).
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += block_size) {
        float v = x_b[i];
        local_sum += v * v;
    }

    // Prefetch this wave's first group. Coarse IU4 variants schedule complete
    // 256/512-K windows per wave; all other variants keep the incumbent
    // contiguous groups-per-wave schedule.
    const int groups_total = K / 256;
    const int d0 = lane_id * 8;
#if defined(HIPFIRE_IU4_SIDECAR) && HIPFIRE_IU4_GROUP_K > 128
    constexpr int groups_per_window = HIPFIRE_IU4_GROUP_K / 256;
    const int windows_total = K / HIPFIRE_IU4_GROUP_K;
    const int pref_group = warp_id * groups_per_window;
    const bool have_pref = warp_id < windows_total;
#else
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
    const int warp_group_start = warp_id * groups_per_warp;
    const int pref_group = warp_group_start;
    const bool have_pref = (groups_per_warp > 0 && pref_group < groups_total);
#endif

    float4 pref_x0, pref_x1, pref_w0, pref_w1;
#ifdef HIPFIRE_RMSNORM_AWQ
    float4 pref_a0, pref_a1;
#endif
    float4 s10, s11, s20, s21;
    if (have_pref) {
        const int base = pref_group * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x_b + base);
        pref_x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
        pref_w0 = *reinterpret_cast<const float4*>(weight + base);
        pref_w1 = *reinterpret_cast<const float4*>(weight + base + 4);
#ifdef HIPFIRE_RMSNORM_AWQ
        pref_a0 = *reinterpret_cast<const float4*>(awq_scale + base);
        pref_a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);
#endif
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
#ifdef HIPFIRE_FP8_STREAM
    float row_amax = 0.0f;
#endif

#if defined(HIPFIRE_IU4_SIDECAR) && HIPFIRE_IU4_GROUP_K > 128
    // One wave owns each complete coarse K window. Its final FWHT values stay
    // in registers until one wave reduction supplies the shared scale; no
    // activation-plane write/re-read or additional launch is involved.
    for (int window = warp_id; window < windows_total; window += num_warps) {
        float values[groups_per_window][8];
        #pragma unroll
        for (int wg = 0; wg < groups_per_window; ++wg) {
            const int group = window * groups_per_window + wg;
            const int base = group * 256 + d0;
            float4 x0, x1, w0, w1;
#ifdef HIPFIRE_RMSNORM_AWQ
            float4 a0, a1;
#endif
            if (window == warp_id && wg == 0 && have_pref) {
                x0 = pref_x0; x1 = pref_x1;
                w0 = pref_w0; w1 = pref_w1;
#ifdef HIPFIRE_RMSNORM_AWQ
                a0 = pref_a0; a1 = pref_a1;
#endif
            } else {
                x0 = *reinterpret_cast<const float4*>(x_b + base);
                x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
                w0 = *reinterpret_cast<const float4*>(weight + base);
                w1 = *reinterpret_cast<const float4*>(weight + base + 4);
#ifdef HIPFIRE_RMSNORM_AWQ
                a0 = *reinterpret_cast<const float4*>(awq_scale + base);
                a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);
#endif
            }

#ifdef HIPFIRE_RMSNORM_AWQ
            float v0 = (x0.x * w0.x * rms / a0.x) * s10.x;
            float v1 = (x0.y * w0.y * rms / a0.y) * s10.y;
            float v2 = (x0.z * w0.z * rms / a0.z) * s10.z;
            float v3 = (x0.w * w0.w * rms / a0.w) * s10.w;
            float v4 = (x1.x * w1.x * rms / a1.x) * s11.x;
            float v5 = (x1.y * w1.y * rms / a1.y) * s11.y;
            float v6 = (x1.z * w1.z * rms / a1.z) * s11.z;
            float v7 = (x1.w * w1.w * rms / a1.w) * s11.w;
#else
            float v0 = x0.x * w0.x * rms * s10.x;
            float v1 = x0.y * w0.y * rms * s10.y;
            float v2 = x0.z * w0.z * rms * s10.z;
            float v3 = x0.w * w0.w * rms * s10.w;
            float v4 = x1.x * w1.x * rms * s11.x;
            float v5 = x1.y * w1.y * rms * s11.y;
            float v6 = x1.z * w1.z * rms * s11.z;
            float v7 = x1.w * w1.w * rms * s11.w;
#endif

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

            #define HBFLY_COARSE(v, pat, str) do { \
                float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
                if (lane_id & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
            } while (0)
            #define HBFLY8_COARSE(pat, str) \
                HBFLY_COARSE(v0,pat,str); HBFLY_COARSE(v1,pat,str); \
                HBFLY_COARSE(v2,pat,str); HBFLY_COARSE(v3,pat,str); \
                HBFLY_COARSE(v4,pat,str); HBFLY_COARSE(v5,pat,str); \
                HBFLY_COARSE(v6,pat,str); HBFLY_COARSE(v7,pat,str)
            HBFLY8_COARSE(0x041F, 1);
            HBFLY8_COARSE(0x081F, 2);
            HBFLY8_COARSE(0x101F, 4);
            HBFLY8_COARSE(0x201F, 8);
            HBFLY8_COARSE(0x401F, 16);
            #undef HBFLY8_COARSE
            #undef HBFLY_COARSE

            const float scale = 0.0625f;
            values[wg][0] = v0 * scale * s20.x;
            values[wg][1] = v1 * scale * s20.y;
            values[wg][2] = v2 * scale * s20.z;
            values[wg][3] = v3 * scale * s20.w;
            values[wg][4] = v4 * scale * s21.x;
            values[wg][5] = v5 * scale * s21.y;
            values[wg][6] = v6 * scale * s21.z;
            values[wg][7] = v7 * scale * s21.w;

            if (x_rot != nullptr) {
                float4 out0, out1;
                out0.x = values[wg][0]; out0.y = values[wg][1];
                out0.z = values[wg][2]; out0.w = values[wg][3];
                out1.x = values[wg][4]; out1.y = values[wg][5];
                out1.z = values[wg][6]; out1.w = values[wg][7];
                *reinterpret_cast<float4*>(x_rot_b + base) = out0;
                *reinterpret_cast<float4*>(x_rot_b + base + 4) = out1;
            }
        }

        emit_iu4_sidecar_window_from_producer8<groups_per_window>(
            values, x_i4, window * groups_per_window,
            (int)blockIdx.x, N, lane_id);
    }
#else
    // Phase 2: FWHT. First group uses prefetched x/w; later groups (if any)
    // load as before. signs already in registers.
    for (int gi = 0; gi < groups_per_warp; gi++) {
        const int group = warp_group_start + gi;
        if (group >= groups_total) break;

        const int base = group * 256 + d0;

        float4 x0, x1, w0, w1;
#ifdef HIPFIRE_RMSNORM_AWQ
        float4 a0, a1;
#endif
        if (gi == 0 && have_pref) {
            x0 = pref_x0; x1 = pref_x1;
            w0 = pref_w0; w1 = pref_w1;
#ifdef HIPFIRE_RMSNORM_AWQ
            a0 = pref_a0; a1 = pref_a1;
#endif
        } else {
            x0 = *reinterpret_cast<const float4*>(x_b + base);
            x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
            w0 = *reinterpret_cast<const float4*>(weight + base);
            w1 = *reinterpret_cast<const float4*>(weight + base + 4);
#ifdef HIPFIRE_RMSNORM_AWQ
            a0 = *reinterpret_cast<const float4*>(awq_scale + base);
            a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);
#endif
        }

#ifdef HIPFIRE_RMSNORM_AWQ
        // Exact order of the retired LDS-staged AWQ kernel:
        // (((x * weight) * rms) / awq_scale) * sign.
        float v0 = (x0.x * w0.x * rms / a0.x) * s10.x;
        float v1 = (x0.y * w0.y * rms / a0.y) * s10.y;
        float v2 = (x0.z * w0.z * rms / a0.z) * s10.z;
        float v3 = (x0.w * w0.w * rms / a0.w) * s10.w;
        float v4 = (x1.x * w1.x * rms / a1.x) * s11.x;
        float v5 = (x1.y * w1.y * rms / a1.y) * s11.y;
        float v6 = (x1.z * w1.z * rms / a1.z) * s11.z;
        float v7 = (x1.w * w1.w * rms / a1.w) * s11.w;
#else
        // Exact multiply order: ((x * weight) * rms) * sign.
        float v0 = x0.x * w0.x * rms * s10.x;
        float v1 = x0.y * w0.y * rms * s10.y;
        float v2 = x0.z * w0.z * rms * s10.z;
        float v3 = x0.w * w0.w * rms * s10.w;
        float v4 = x1.x * w1.x * rms * s11.x;
        float v5 = x1.y * w1.y * rms * s11.y;
        float v6 = x1.z * w1.z * rms * s11.z;
        float v7 = x1.w * w1.w * rms * s11.w;
#endif

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
        float f0 = v0 * s * s20.x;
        float f1 = v1 * s * s20.y;
        float f2 = v2 * s * s20.z;
        float f3 = v3 * s * s20.w;
        float f4 = v4 * s * s21.x;
        float f5 = v5 * s * s21.y;
        float f6 = v6 * s * s21.z;
        float f7 = v7 * s * s21.w;
#ifdef HIPFIRE_FP8_STREAM
        row_amax = fmaxf(row_amax, fabsf(f0));
        row_amax = fmaxf(row_amax, fabsf(f1));
        row_amax = fmaxf(row_amax, fabsf(f2));
        row_amax = fmaxf(row_amax, fabsf(f3));
        row_amax = fmaxf(row_amax, fabsf(f4));
        row_amax = fmaxf(row_amax, fabsf(f5));
        row_amax = fmaxf(row_amax, fabsf(f6));
        row_amax = fmaxf(row_amax, fabsf(f7));
#endif
#ifdef HIPFIRE_IU4_SIDECAR
        if (x_rot != nullptr) {
            float4 out0, out1;
            out0.x = f0; out0.y = f1; out0.z = f2; out0.w = f3;
            out1.x = f4; out1.y = f5; out1.z = f6; out1.w = f7;
            *reinterpret_cast<float4*>(x_rot_b + base) = out0;
            *reinterpret_cast<float4*>(x_rot_b + base + 4) = out1;
        }
        // token = blockIdx.x; N is the batch extent for [K/128, N] layout.
        emit_iu4_sidecar_from_producer8(
            f0, f1, f2, f3, f4, f5, f6, f7,
            x_i4, group, (int)blockIdx.x, N, lane_id);
#else
        float4 out0;
        float4 out1;
        out0.x = f0;
        out0.y = f1;
        out0.z = f2;
        out0.w = f3;
        out1.x = f4;
        out1.y = f5;
        out1.z = f6;
        out1.w = f7;
        *reinterpret_cast<float4*>(x_rot_b + base) = out0;
        *reinterpret_cast<float4*>(x_rot_b + base + 4) = out1;
#endif
    }
#endif
    #ifdef HIPFIRE_FP8_STREAM
    // FP8 stream: whole-row scale + pack from the just-written F32 row
    // (L1-resident re-read). Byte-identical to pack_f32_to_fp8_mq4v2_gfx12
    // with scale_mode=1 on finite rows; reuses the reduction smem.
    {
        const int token = (int)blockIdx.x;
        const long long frow = (long long)token * K;
        const int gpr = K >> 8;
        fp8stream_pack_row_from_f32(
            x_rot_b,
            x_fp8 + frow,
            fp8_half_sums + (long long)token * gpr * 2,
            fp8_row_scales + token,
            row_amax,
            reduce,
            K,
            tid,
            lane_id,
            warp_id);
    }
    #endif
}
