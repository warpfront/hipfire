// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

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

// -DHIPFIRE_RMSNORM_FOLD=1 (IU4 sidecar builds only) adds a `delta` input
// after `x`: the preceding residual GEMM wrote its sum there (SET) instead of
// adding it into `x`, and this kernel performs that add, RN(x + delta), in
// both phases, writes the sum back to `x`, and normalizes it. Bit-identical
// to the GEMM's ADD epilogue followed by the plain kernel; turns the GEMM's
// residual read-modify-write into streams this kernel already pays for.
//
// -DHIPFIRE_RMSNORM_AWQ=1 builds the AWQ twin `fused_rmsnorm_mq_rotate_awq`:
// one extra `awq_scale[K]` argument, divided in AFTER the rms multiply and
// BEFORE the sign/FWHT (`(((x*w)*rms)/s)*sign`), completing `(W·s)·(x/s)`.
// Same phases, prefetch, and reduction tree as the uniform symbol; the scale
// rides the same float4 prefetch as x/weight. Uniform builds are textually
// unchanged.
//
// -DHIPFIRE_RMSNORM_P1A_BATCHED=1 (gfx1201 `_v2` IU4 symbols, gfx1100/gfx1151
// `_b8` IU4 symbols, gfx1151 `_fold_b8`) keeps up to 20 Phase-1a row loads in
// flight per wave instead of one: the incumbent loop waits on each x[i] before
// issuing x[i + 256], so a wave held a single 128-B request and the
// sum-of-squares pass was DRAM-latency bound. Same elements, same strictly
// sequential fma(v, v, acc) chain from +0 as the incumbent loop's
// contraction, same reduction tree: bit-identical rms. Under
// HIPFIRE_RMSNORM_FOLD the batch loads x and delta pairs and forms each
// element as the incumbent's RN(x + delta) after the drain (MAX8 + DRAIN
// only).
//
// -DHIPFIRE_RMSNORM_AWQ_RCP=1 (gfx1201 A4 `_v2_slab_fdiv`, AWQ only) adds an
// `awq_rcp` input after `awq_scale`: the planes [R 0..K-1][Rlo K..2K-1]
// that the loader computes once per AWQ scale a (rdna-compute
// `awq_rcp_planes`): R = RN(1/a), Rlo = RN(fma(-a, R, 1) * R), both NaN
// unless 2^-20 <= |a| <= 2^20. The divide y / a becomes
//   q0 = fma(y, R, y * Rlo), q = fma(fma(-a, q0, y), R, q0).
// y * (R + Rlo) is within 2^-47 of y / a relative, so q0 is a faithful
// rounding of y / a, and with R = RN(1/a) one Markstein correction returns
// RN(y / a). The wave keeps q only when every lane's eight |q| lie in
// [2^-80, 2^80] (no NaN); otherwise the whole wave recomputes the IEEE
// divides. q equals the divide for every f32 y and every f16 AWQ scale
// (exhaustive gfx1201 check, `a4-5k3/rms-row`), so the codes are
// bit-identical.
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
#if defined(HIPFIRE_RMSNORM_FOLD) && !defined(HIPFIRE_IU4_SIDECAR)
#error "HIPFIRE_RMSNORM_FOLD is an IU4-sidecar producer variant"
#endif
#if defined(HIPFIRE_RMSNORM_P1A_BATCHED) && defined(HIPFIRE_RMSNORM_FOLD) \
    && !(defined(HIPFIRE_RMSNORM_P1A_DRAIN) && defined(HIPFIRE_RMSNORM_P1A_MAX8))
#error "HIPFIRE_RMSNORM_FOLD batches only the MAX8 + DRAIN Phase-1a"
#endif
#if defined(HIPFIRE_RMSNORM_AWQ_RCP) && !defined(HIPFIRE_RMSNORM_AWQ)
#error "HIPFIRE_RMSNORM_AWQ_RCP replaces the AWQ divide"
#endif
#if defined(HIPFIRE_RMSNORM_GROUP_GRID) && (!defined(HIPFIRE_RMSNORM_AWQ) \
    || defined(HIPFIRE_IU4_SIDECAR) || defined(HIPFIRE_FP8_STREAM) || defined(HIPFIRE_RMSNORM_AWQ_RCP))
#error "HIPFIRE_RMSNORM_GROUP_GRID is the f32 AWQ decode variant"
#endif
#ifdef HIPFIRE_RMSNORM_AWQ_RCP
// RN(y / a) from the loader's planes (see the header): exact where the
// caller's wave check admits q.
__device__ __forceinline__ float rmsnorm_awq_rcp_div(float y, float a, float r, float rl) {
    const float q0 = __fmaf_rn(y, r, __fmul_rn(y, rl));
    return __fmaf_rn(__fmaf_rn(-a, q0, y), r, q0);
}
#endif
#ifdef HIPFIRE_FP8_PROD_SHORT
#define HIPFIRE_RMSNORM_FP8_GROUPS 3
#else
#define HIPFIRE_RMSNORM_FP8_GROUPS 9
#endif


__launch_bounds__(256, 1)
extern "C" __global__ void HIPFIRE_RMSNORM_KERNEL(
#ifdef HIPFIRE_RMSNORM_FOLD
    float* __restrict__ x,              // residual, updated in place: x += delta
    const float* __restrict__ delta,    // deferred residual-GEMM sum [N, K]
#else
    const float* __restrict__ x,
#endif
    const float* __restrict__ weight,
#ifdef HIPFIRE_RMSNORM_AWQ
    const float* __restrict__ awq_scale,
#ifdef HIPFIRE_RMSNORM_AWQ_RCP
    const float* __restrict__ awq_rcp,  // [R K][Rlo K], loader planes of awq_scale
#endif
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
    int N,
    int row_scale_shift
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

#ifdef HIPFIRE_RMSNORM_GROUP_GRID
    // Group grid (gfx1201 decode, `HIPFIRE_G12_DEC_NORM`): grid = (K/256, N).
    // Every workgroup recomputes the row's rms with the unchanged 256-thread
    // Phase 1a/1b (bit-identical), then wave 0 rotates only group blockIdx.x.
    // The single-workgroup launch ran up to three groups per wave serially.
    const long long batch_off = (long long)blockIdx.y * K;
#else
    const long long batch_off = (long long)blockIdx.x * K;
#endif
#ifdef HIPFIRE_RMSNORM_FOLD
    float* x_b           = x     + batch_off;
    const float* d_b     = delta + batch_off;
#else
    const float* x_b     = x     + batch_off;
#endif
#ifdef HIPFIRE_IU4_SIDECAR
          float* x_rot_b = (x_rot != nullptr) ? (x_rot + batch_off) : nullptr;
#elif defined(HIPFIRE_FP8_PROD_INREG)
    if (K > HIPFIRE_RMSNORM_FP8_GROUPS * 2048 || (K & 255)) return;
    float held[HIPFIRE_RMSNORM_FP8_GROUPS][8];
#else
    float* x_rot_b = x_rot + batch_off;
#endif

    // Phase 1a: accumulate sum-of-squares (same stride/order as baseline).
    float local_sum = 0.0f;
#ifdef HIPFIRE_RMSNORM_P1A_BATCHED
    // Batched phase-1a: same elements, same strictly sequential fma(v, v, acc)
    // chain from +0 as the incumbent loop's contraction (the explicit fma also
    // keeps the incumbent's per-element rounding regardless of how the backend
    // would otherwise pair two unrolled products), same reduction tree:
    // bit-identical rms.
    // Batch width: 20-wide batches corrupt intermittently under sustained
    // back-to-back pressure at N >= 2048 (single-row wrong-scale, ~5%/launch;
    // N <= 1024 clean; serial SEQ clean over 6000 launches). Batches of <= 8
    // with the same fence are stable over thousands of back-to-back launches
    // (serial + single-CU) and bit-identical to SEQ, so the fp8 SHORT
    // producers cap the cascade at 8 (`HIPFIRE_RMSNORM_P1A_MAX8`). The IU4
    // `_v2` producers keep 20/8/4/1 (their shapes/pressures are unchanged);
    // they need their own poisoned stress before any width change.
    // With HIPFIRE_RMSNORM_P1A_DRAIN, a full load wait separates each batch's
    // issue from its use, so no fma reads a load that has not retired.
#ifdef HIPFIRE_RMSNORM_P1A_DRAIN
#if defined(__GFX12__)
#define HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER() asm volatile("s_wait_loadcnt 0x0" ::: "memory")
#else
#define HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER() asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
#endif
#else
#define HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER() ((void)0)
#endif
    int i = tid;
#ifdef HIPFIRE_RMSNORM_FOLD
#define HIPFIRE_RMSNORM_P1A_STEP(B) \
    for (; i + ((B) - 1) * block_size < K; i += (B) * block_size) { \
        float vb[B], db[B]; \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) { \
            vb[u] = x_b[i + u * block_size]; \
            db[u] = d_b[i + u * block_size]; \
        } \
        HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER(); \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) { \
            const float v = __fadd_rn(vb[u], db[u]); \
            local_sum = __fmaf_rn(v, v, local_sum); \
        } \
    }
#else
#define HIPFIRE_RMSNORM_P1A_STEP(B) \
    for (; i + ((B) - 1) * block_size < K; i += (B) * block_size) { \
        float vb[B]; \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) vb[u] = x_b[i + u * block_size]; \
        HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER(); \
        _Pragma("unroll") for (int u = 0; u < (B); ++u) \
            local_sum = __fmaf_rn(vb[u], vb[u], local_sum); \
    }
#endif
#ifdef HIPFIRE_RMSNORM_P1A_MAX8
    HIPFIRE_RMSNORM_P1A_STEP(8)
#else
    HIPFIRE_RMSNORM_P1A_STEP(20)
    HIPFIRE_RMSNORM_P1A_STEP(8)
#endif
    HIPFIRE_RMSNORM_P1A_STEP(4)
#undef HIPFIRE_RMSNORM_P1A_STEP
#undef HIPFIRE_RMSNORM_P1A_ISSUE_BARRIER
#ifdef HIPFIRE_RMSNORM_FOLD
    for (; i < K; i += block_size) {
        const float v = __fadd_rn(x_b[i], d_b[i]);
        local_sum = __fmaf_rn(v, v, local_sum);
    }
#else
    for (; i < K; i += block_size) local_sum = __fmaf_rn(x_b[i], x_b[i], local_sum);
#endif
#else
    for (int i = tid; i < K; i += block_size) {
#ifdef HIPFIRE_RMSNORM_FOLD
        float v = __fadd_rn(x_b[i], d_b[i]);
#else
        float v = x_b[i];
#endif
        local_sum += v * v;
    }
#endif

    // Prefetch first FWHT group's x/weight (+ lane-local signs) so the
    // loads ride under Phase-1b barriers. groups layout is wave-uniform
    // (depends only on K and warp_id), so this is safe before rms exists.
    const int groups_total = K / 256;
#ifdef HIPFIRE_RMSNORM_GROUP_GRID
    const int groups_per_warp = (warp_id == 0) ? 1 : 0;
    const int warp_group_start = (int)blockIdx.x;
#else
    const int groups_per_warp = (groups_total + num_warps - 1) / num_warps;
#ifdef HIPFIRE_RMSNORM_FP8_STRIDED
    // Spread consecutive 256-value groups across the eight waves; each wave
    // still owns its own groups through rotation and exact FP8 half sums.
    const int warp_group_start = warp_id;
#else
    const int warp_group_start = warp_id * groups_per_warp;
#endif
#endif
    const int d0 = lane_id * 8;

    float4 pref_x0, pref_x1, pref_w0, pref_w1;
#ifdef HIPFIRE_RMSNORM_AWQ
    float4 pref_a0, pref_a1;
#endif
    float4 s10, s11, s20, s21;
    const int pref_group = warp_group_start;
    const bool have_pref = (groups_per_warp > 0 && pref_group < groups_total);
    if (have_pref) {
        const int base = pref_group * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x_b + base);
        pref_x1 = *reinterpret_cast<const float4*>(x_b + base + 4);
#ifdef HIPFIRE_RMSNORM_FOLD
        {
            const float4 q0 = *reinterpret_cast<const float4*>(d_b + base);
            const float4 q1 = *reinterpret_cast<const float4*>(d_b + base + 4);
            pref_x0.x = __fadd_rn(pref_x0.x, q0.x); pref_x0.y = __fadd_rn(pref_x0.y, q0.y);
            pref_x0.z = __fadd_rn(pref_x0.z, q0.z); pref_x0.w = __fadd_rn(pref_x0.w, q0.w);
            pref_x1.x = __fadd_rn(pref_x1.x, q1.x); pref_x1.y = __fadd_rn(pref_x1.y, q1.y);
            pref_x1.z = __fadd_rn(pref_x1.z, q1.z); pref_x1.w = __fadd_rn(pref_x1.w, q1.w);
        }
#endif
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

    // Phase 2: FWHT. First group uses prefetched x/w; later groups (if any)
    // load as before. signs already in registers.
#ifdef HIPFIRE_FP8_PROD_INREG
    #pragma unroll
    for (int gi = 0; gi < HIPFIRE_RMSNORM_FP8_GROUPS; gi++) {
#else
    for (int gi = 0; gi < groups_per_warp; gi++) {
#endif
#ifdef HIPFIRE_RMSNORM_FP8_STRIDED
        const int group = warp_group_start + gi * num_warps;
#else
        const int group = warp_group_start + gi;
#endif
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
#ifdef HIPFIRE_RMSNORM_FOLD
            {
                const float4 q0 = *reinterpret_cast<const float4*>(d_b + base);
                const float4 q1 = *reinterpret_cast<const float4*>(d_b + base + 4);
                x0.x = __fadd_rn(x0.x, q0.x); x0.y = __fadd_rn(x0.y, q0.y);
                x0.z = __fadd_rn(x0.z, q0.z); x0.w = __fadd_rn(x0.w, q0.w);
                x1.x = __fadd_rn(x1.x, q1.x); x1.y = __fadd_rn(x1.y, q1.y);
                x1.z = __fadd_rn(x1.z, q1.z); x1.w = __fadd_rn(x1.w, q1.w);
            }
#endif
            w0 = *reinterpret_cast<const float4*>(weight + base);
            w1 = *reinterpret_cast<const float4*>(weight + base + 4);
#ifdef HIPFIRE_RMSNORM_AWQ
            a0 = *reinterpret_cast<const float4*>(awq_scale + base);
            a1 = *reinterpret_cast<const float4*>(awq_scale + base + 4);
#endif
        }
#ifdef HIPFIRE_RMSNORM_FOLD
        // Write the updated residual back: every element of the row is owned
        // by exactly one lane here, and all Phase-1a reads of x retired at the
        // reduction barrier.
        *reinterpret_cast<float4*>(x_b + base) = x0;
        *reinterpret_cast<float4*>(x_b + base + 4) = x1;
#endif

#if defined(HIPFIRE_RMSNORM_AWQ) && defined(HIPFIRE_RMSNORM_AWQ_RCP)
        // (((x * weight) * rms) / awq_scale) * sign with the divide from the
        // loader's reciprocal planes; any lane outside the admitted range
        // sends the wave to the IEEE divides.
        const float4 r0 = *reinterpret_cast<const float4*>(awq_rcp + base);
        const float4 r1 = *reinterpret_cast<const float4*>(awq_rcp + base + 4);
        const float4 l0 = *reinterpret_cast<const float4*>(awq_rcp + K + base);
        const float4 l1 = *reinterpret_cast<const float4*>(awq_rcp + K + base + 4);
        float qd0 = rmsnorm_awq_rcp_div(x0.x * w0.x * rms, a0.x, r0.x, l0.x);
        float qd1 = rmsnorm_awq_rcp_div(x0.y * w0.y * rms, a0.y, r0.y, l0.y);
        float qd2 = rmsnorm_awq_rcp_div(x0.z * w0.z * rms, a0.z, r0.z, l0.z);
        float qd3 = rmsnorm_awq_rcp_div(x0.w * w0.w * rms, a0.w, r0.w, l0.w);
        float qd4 = rmsnorm_awq_rcp_div(x1.x * w1.x * rms, a1.x, r1.x, l1.x);
        float qd5 = rmsnorm_awq_rcp_div(x1.y * w1.y * rms, a1.y, r1.y, l1.y);
        float qd6 = rmsnorm_awq_rcp_div(x1.z * w1.z * rms, a1.z, r1.z, l1.z);
        float qd7 = rmsnorm_awq_rcp_div(x1.w * w1.w * rms, a1.w, r1.w, l1.w);
        {
            // minimum propagates NaN, so mn >= 2^-80 also rejects NaN.
            float mn = __builtin_elementwise_minimum(
                __builtin_elementwise_minimum(fabsf(qd0), fabsf(qd1)),
                __builtin_elementwise_minimum(fabsf(qd2), fabsf(qd3)));
            mn = __builtin_elementwise_minimum(mn, __builtin_elementwise_minimum(
                __builtin_elementwise_minimum(fabsf(qd4), fabsf(qd5)),
                __builtin_elementwise_minimum(fabsf(qd6), fabsf(qd7))));
            const float mx = fmaxf(fmaxf(fmaxf(fabsf(qd0), fabsf(qd1)), fmaxf(fabsf(qd2), fabsf(qd3))),
                                   fmaxf(fmaxf(fabsf(qd4), fabsf(qd5)), fmaxf(fabsf(qd6), fabsf(qd7))));
            const bool admitted = mn >= 0x1p-80f && mx <= 0x1p80f;
            if (__builtin_amdgcn_ballot_w32(!admitted) != 0u) {
                qd0 = x0.x * w0.x * rms / a0.x;
                qd1 = x0.y * w0.y * rms / a0.y;
                qd2 = x0.z * w0.z * rms / a0.z;
                qd3 = x0.w * w0.w * rms / a0.w;
                qd4 = x1.x * w1.x * rms / a1.x;
                qd5 = x1.y * w1.y * rms / a1.y;
                qd6 = x1.z * w1.z * rms / a1.z;
                qd7 = x1.w * w1.w * rms / a1.w;
            }
        }
        float v0 = qd0 * s10.x;
        float v1 = qd1 * s10.y;
        float v2 = qd2 * s10.z;
        float v3 = qd3 * s10.w;
        float v4 = qd4 * s11.x;
        float v5 = qd5 * s11.y;
        float v6 = qd6 * s11.z;
        float v7 = qd7 * s11.w;
#elif defined(HIPFIRE_RMSNORM_AWQ)
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
#ifdef HIPFIRE_FP8_PROD_INREG
        held[gi][0] = f0; held[gi][1] = f1; held[gi][2] = f2; held[gi][3] = f3;
        held[gi][4] = f4; held[gi][5] = f5; held[gi][6] = f6; held[gi][7] = f7;
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
#endif
    }
    #ifdef HIPFIRE_FP8_STREAM
    // FP8 stream: whole-row scale + pack from the just-written F32 row
    // (L1-resident re-read). Byte-identical to pack_f32_to_fp8_mq4v2_gfx12
    // with scale_mode=1 on finite rows.
    //
    // The pack gets reduce[8..15], never reduce[0..7]. No barrier separates
    // Phase 2 from the pack's first `smem8[wave] = amax` store, so wave 0 can
    // reach that store while another wave has not yet executed its
    // `rms = reduce[0]` load (Phase 1b's last barrier releases all waves,
    // but nothing orders their LDS accesses after it). Aliasing reduce[0]
    // then hands that wave wave 0's amax as its rms: a wrong-scale row that
    // appears only when the waves of a workgroup drift apart, e.g. next to
    // another process's ALU-bound waves. reduce[8..15] is untouched by
    // Phase 1b and inside the launch's (K + 256) * 4 B of dynamic LDS.
    {
        float* pack8 = reduce + 8;
        const int token = (int)blockIdx.x;
        const long long frow = (long long)token * K;
        const int gpr = K >> 8;
#ifdef HIPFIRE_FP8_PROD_INREG
        fp8stream_pack_row_inreg(
            held, x_fp8 + frow, fp8_half_sums + (long long)token * (K >> 7),
            fp8_row_scales + token, row_amax, pack8, K, lane_id, warp_id,
#ifdef HIPFIRE_RMSNORM_FP8_STRIDED
            warp_group_start, num_warps, row_scale_shift);
#else
            warp_group_start, 1, row_scale_shift);
#endif
#else
        fp8stream_pack_row_from_f32(
            x_rot_b,
            x_fp8 + frow,
            fp8_half_sums + (long long)token * gpr * 2,
            fp8_row_scales + token,
            row_amax, pack8, K, tid, lane_id, warp_id, row_scale_shift);
#endif
    }
    #endif
}
