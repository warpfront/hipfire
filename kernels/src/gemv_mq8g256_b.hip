// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// MQ8G256 decode GEMV (R = 1) and verify rows-exact (R = 2..16) on weight LAYOUT B.
//
// Layout B (`gemm_mq8g256::mq8g256_repack_b`): per tensor `[M, K/256]` f16 scale plane (row-major), zero
// padding to the next 256 B boundary (`code_off`), then the int8 code plane `[M, K]` row-major. The base is
// 16-byte aligned, so every code load below is one aligned 16-byte `global_load_dwordx4`, and the per-row
// scale plane (2 * K/256 bytes at `row * K/128`) is read with uniform dword loads (scalar cache).
//
// X operand = the planes of the batched producer `mq8_rotate_quantize_x_batched` (shared by prefill,
// decode and verify): int8 codes `[N][K]` (row r at `r * K`) and f32 scales `[K/256][ldxs]`
// (`xs[g * ldxs + r]`, `ldxs` = the producer's N). A decode row is `N = ldxs = 1`.
//
// ASSOCIATION (the CPU reference in `mq8_decode_lab.rs::cpu_row_b` spells it out; every kernel here and
// every R computes exactly this, so row r of an R-row launch is bit-identical to the R = 1 kernel on X row
// r):
//   * one wave32 per weight row; lane l = 16*h + s (h = l >> 4 half, s = l & 15 slice);
//   * "pair" p covers groups 2p (half 0) and 2p + 1 (half 1); lane (h, s) owns bytes [16 s, 16 s + 16) of
//     group g = 2p + h; a lane whose g >= K/256 (odd group count, last pair) contributes nothing;
//   * per (lane, pair): dot = four chained `sdot4` (exact int32, |dot| <= 16 * 128 * 128 = 2^18) and
//     `acc = fmaf(RN(RN(f16 ws) * xs), (float)dot, acc)`; pairs ascending, ONE chain per lane;
//   * reduction: inside each 16-lane row the butterfly `a += a[l ^ 1]`, `a += a[l ^ 2]`, `a += a[l ^ 4]`,
//     `a += a[l ^ 8]` (every lane of the row holds the row total T_h), then lane 0 stores T_0 + T_1.
// The association does not depend on R, on the unroll U (a scheduling choice only) or on the block shape.

// RDNA3/RDNA4 have no dot1-insts; `sudot4` with both operands signed is the same exact int32 signed dot.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
#define MQ8B_SDOT4(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)
#else
#define MQ8B_SDOT4(a, b, c) __builtin_amdgcn_sdot4((a), (b), (c), false)
#endif

static __device__ __forceinline__ int mq8b_dot16(const uint4 w, const uint4 x) {
    int d = MQ8B_SDOT4((int)w.x, (int)x.x, 0);
    d = MQ8B_SDOT4((int)w.y, (int)x.y, d);
    d = MQ8B_SDOT4((int)w.z, (int)x.z, d);
    d = MQ8B_SDOT4((int)w.w, (int)x.w, d);
    return d;
}

// f16 scale at byte offset `off` of the tensor (uniform; one aligned dword load, global address space kept
// so the load stays a scalar-cache `s_load`).
static __device__ __forceinline__ float mq8b_scale(const unsigned char* __restrict__ W, size_t off) {
    const unsigned v = *(const unsigned*)(W + (off & ~(size_t)3));
    const unsigned short h = (unsigned short)((off & 2u) ? (v >> 16) : (v & 0xFFFFu));
    return (float)__builtin_bit_cast(_Float16, h);
}

// Row totals: butterfly inside each 16-lane row, then T_0 + T_1 at lane 0.
static __device__ __forceinline__ float mq8b_reduce(float a) {
#if defined(MQ8B_NO_DPP)
    a += __shfl_xor(a, 1, 32);
    a += __shfl_xor(a, 2, 32);
    a += __shfl_xor(a, 4, 32);
    a += __shfl_xor(a, 8, 32);
    a += __shfl_xor(a, 16, 32);
#else
    a += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(a), 0x161, 0xF, 0xF, false));
    a += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(a), 0x162, 0xF, 0xF, false));
    a += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(a), 0x164, 0xF, 0xF, false));
    a += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(a), 0x168, 0xF, 0xF, false));
    a += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(a), 0x76543210u, 0xfedcba98u, false, false));
#endif
    return a;
}

// R: X rows per launch; U: group pairs whose loads are issued before their math (a scheduling choice);
// WPB: waves (weight-row groups) per block; P: weight rows per wave (they share the X loads). None of them
// changes a bit.
template <int R, int U, int WPB, int P>
static __device__ __forceinline__ void mq8b_body(
    const unsigned char* __restrict__ W,
    const signed char* __restrict__ xq,
    const float* __restrict__ xs,
    float* __restrict__ y,
    int M, int K, int ldxs
) {
    const int row0 = __builtin_amdgcn_readfirstlane(blockIdx.x * WPB + (threadIdx.x >> 5)) * P;
    if (row0 >= M) return;
    const int lane = threadIdx.x & 31;
    const int h = lane >> 4;
    const int sub = lane & 15;
    const int gpr = K >> 8;
    const int npairs = (gpr + 1) >> 1;
    const size_t code_off = ((size_t)M * (size_t)gpr * 2 + 255) & ~(size_t)255;
    size_t srow[P];
    const unsigned char* crow[P];
#pragma unroll
    for (int q = 0; q < P; q++) {
        const int rc = min(row0 + q, M - 1);   // rows past M re-read the last row; their result is not stored
        srow[q] = (size_t)rc * (size_t)gpr * 2;
        crow[q] = W + code_off + (size_t)rc * (size_t)K + (size_t)sub * 16;
    }
    const unsigned char* xcol = (const unsigned char*)xq + (size_t)sub * 16;

    float acc[P][R];
#pragma unroll
    for (int q = 0; q < P; q++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[q][r] = 0.0f;

    for (int p0 = 0; p0 < npairs; p0 += U) {
        uint4 w[P][U];
        float sl[P][U], sh[P][U];
        int g[U], gl[U], gh[U];
        bool ok[U];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int p = p0 + u;
            gl[u] = min(2 * p, gpr - 1);
            gh[u] = min(2 * p + 1, gpr - 1);
            g[u] = h ? gh[u] : gl[u];
            ok[u] = (2 * p + h) < gpr;
#pragma unroll
            for (int q = 0; q < P; q++) {
                w[q][u] = *(const uint4*)(crow[q] + (size_t)g[u] * 256);
                sl[q][u] = mq8b_scale(W, srow[q] + 2 * (size_t)gl[u]);
                sh[q][u] = mq8b_scale(W, srow[q] + 2 * (size_t)gh[u]);
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
#pragma unroll
            for (int u = 0; u < U; u++) {
                const uint4 x = *(const uint4*)(xcol + (size_t)r * K + (size_t)g[u] * 256);
                const float xl = xs[(size_t)gl[u] * ldxs + r];
                const float xh = xs[(size_t)gh[u] * ldxs + r];
#pragma unroll
                for (int q = 0; q < P; q++) {
                    const int d = mq8b_dot16(w[q][u], x);
                    const float scl = sl[q][u] * xl;
                    const float sch = sh[q][u] * xh;
                    const float sc = h ? sch : scl;
                    const float nacc = __builtin_fmaf(sc, (float)d, acc[q][r]);
                    acc[q][r] = ok[u] ? nacc : acc[q][r];
                }
            }
        }
    }

#pragma unroll
    for (int q = 0; q < P; q++) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float a = mq8b_reduce(acc[q][r]);
            if (lane == 0 && row0 + q < M) y[(size_t)r * M + row0 + q] = a;
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Fused-producer variant (decode and small-R verify): the same GEMV, but X arrives as F32 `[R][K]` and every
// block rotates + quantizes it itself (the batched producer's math, operation for operation, one wave per
// (row, group)), exchanging the codes and scales through LDS. The weight loads are issued BEFORE the
// rotation so the rotation hides under the weight-load latency; no separate producer launch (the ~3 us
// per-launch floor of a captured graph). The codes and scales equal the producer's bit for bit, so the
// result equals the unfused kernels' (guarded in the lab). K <= MQ8B_FX_KMAX.
// ---------------------------------------------------------------------------------------------------
#define MQ8B_FX_KMAX 5120

// Value of `v` in lane (lane ^ MASK) of the same 32-lane wave, MASK in {1, 2, 4, 8, 16}: DPP row_xmask inside a
// 16-lane row, `v_permlanex16` (identity selects) for 16. No LDS round trip, so the rotation below is plain VALU.
template <int MASK>
static __device__ __forceinline__ float mq8b_xor(float v) {
#if defined(MQ8B_NO_DPP)
    return __shfl_xor(v, MASK, 32);
#else
    if (MASK == 16)
        return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + MASK, 0xF, 0xF, false));
#endif
}

// The rotation inputs of one (row, group) for this lane: 8 X floats and the two sign tables' 8 entries each.
struct Mq8bRotIn {
    float4 a, b, s1a, s1b, s2a, s2b;
};

// Keeps the loaded inputs live in registers at this point of the program (the compiler would otherwise sink
// the loads to their first use, behind the weight loads, and the in-order load counter would then make the
// rotation wait for the whole weight stream).
#define MQ8B_PIN4(f) asm volatile("" : "+v"((f).x), "+v"((f).y), "+v"((f).z), "+v"((f).w))
static __device__ __forceinline__ void mq8b_rot_pin(Mq8bRotIn& in) {
    MQ8B_PIN4(in.a); MQ8B_PIN4(in.b); MQ8B_PIN4(in.s1a); MQ8B_PIN4(in.s1b); MQ8B_PIN4(in.s2a); MQ8B_PIN4(in.s2b);
}

static __device__ __forceinline__ Mq8bRotIn mq8b_rot_load(
    const float* __restrict__ xg,        // 256 floats of one (row, group)
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int lane
) {
    const int d0 = lane * 8;
    Mq8bRotIn in;
    in.a = *(const float4*)(xg + d0);
    in.b = *(const float4*)(xg + d0 + 4);
    in.s1a = *(const float4*)(signs1 + d0);
    in.s1b = *(const float4*)(signs1 + d0 + 4);
    in.s2a = *(const float4*)(signs2 + d0);
    in.s2b = *(const float4*)(signs2 + d0 + 4);
    return in;
}

// `mq8_rotate_quantize_x_batched` for one (row, group), operation for operation:
//   v = x * signs1; FWHT-256 (local butterflies over strides 1, 2, 4 of the lane's 8 elements, then lane
//   butterflies over strides 1, 2, 4, 8, 16: v_lo' = v + p, v_hi' = p - v, written as p + (+-v), the same
//   IEEE result); v *= 0.0625 * signs2; amax = max |v|; scale = amax > 0 ? amax / 127 : 1;
//   q = clamp(rintf(v * (127 / amax)), -128, 127).
static __device__ __forceinline__ void mq8b_rot_compute(
    const Mq8bRotIn& in, int lane, unsigned& w0, unsigned& w1, float& group_scale
) {
    float v0 = in.a.x * in.s1a.x, v1 = in.a.y * in.s1a.y, v2 = in.a.z * in.s1a.z, v3 = in.a.w * in.s1a.w;
    float v4 = in.b.x * in.s1b.x, v5 = in.b.y * in.s1b.y, v6 = in.b.z * in.s1b.z, v7 = in.b.w * in.s1b.w;

    float t;
    t = v0; v0 += v1; v1 = t - v1;
    t = v2; v2 += v3; v3 = t - v3;
    t = v4; v4 += v5; v5 = t - v5;
    t = v6; v6 += v7; v7 = t - v7;
    t = v0; v0 += v2; v2 = t - v2;
    t = v1; v1 += v3; v3 = t - v3;
    t = v4; v4 += v6; v6 = t - v6;
    t = v5; v5 += v7; v7 = t - v7;
    t = v0; v0 += v4; v4 = t - v4;
    t = v1; v1 += v5; v5 = t - v5;
    t = v2; v2 += v6; v6 = t - v6;
    t = v3; v3 += v7; v7 = t - v7;

    #define MQ8B_HBFLY(v, str, neg) do { \
        const float _p = mq8b_xor<str>(v); \
        (v) = _p + __uint_as_float(__float_as_uint(v) ^ (neg)); \
    } while (0)
    #define MQ8B_HBFLY8(str) do { \
        const unsigned _neg = (lane & (str)) ? 0x80000000u : 0u; \
        MQ8B_HBFLY(v0, str, _neg); MQ8B_HBFLY(v1, str, _neg); MQ8B_HBFLY(v2, str, _neg); MQ8B_HBFLY(v3, str, _neg); \
        MQ8B_HBFLY(v4, str, _neg); MQ8B_HBFLY(v5, str, _neg); MQ8B_HBFLY(v6, str, _neg); MQ8B_HBFLY(v7, str, _neg); \
    } while (0)
    MQ8B_HBFLY8(1); MQ8B_HBFLY8(2); MQ8B_HBFLY8(4); MQ8B_HBFLY8(8); MQ8B_HBFLY8(16);
    #undef MQ8B_HBFLY8
    #undef MQ8B_HBFLY

    const float s = 0.0625f;
    v0 *= s * in.s2a.x; v1 *= s * in.s2a.y; v2 *= s * in.s2a.z; v3 *= s * in.s2a.w;
    v4 *= s * in.s2b.x; v5 *= s * in.s2b.y; v6 *= s * in.s2b.z; v7 *= s * in.s2b.w;

    float my_max = fmaxf(fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3))),
                         fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7))));
    my_max = fmaxf(my_max, mq8b_xor<1>(my_max));
    my_max = fmaxf(my_max, mq8b_xor<2>(my_max));
    my_max = fmaxf(my_max, mq8b_xor<4>(my_max));
    my_max = fmaxf(my_max, mq8b_xor<8>(my_max));
    my_max = fmaxf(my_max, mq8b_xor<16>(my_max));

    group_scale = (my_max > 0.0f) ? (my_max / 127.0f) : 1.0f;
    const float inv_scale = (my_max > 0.0f) ? (127.0f / my_max) : 0.0f;

    const float xv[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    w0 = 0u; w1 = 0u;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int q0 = (int)fminf(fmaxf(rintf(xv[e] * inv_scale), -128.0f), 127.0f);
        const int q1 = (int)fminf(fmaxf(rintf(xv[e + 4] * inv_scale), -128.0f), 127.0f);
        w0 |= ((unsigned)q0 & 255u) << (8 * e);
        w1 |= ((unsigned)q1 & 255u) << (8 * e);
    }
}

// U >= ceil(K/512): every weight load of the row is issued before the rotation math (one chunk); the X /
// sign loads of this wave's first rotation item are issued before even those, so (loads complete in
// order) the rotation does not wait for the weight stream.
template <int R, int U, int WPB, int P>
static __device__ __forceinline__ void mq8b_fx_body(
    const unsigned char* __restrict__ W,
    const float* __restrict__ x,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ y,
    int M, int K
) {
    __shared__ __attribute__((aligned(16))) unsigned char lq[R * MQ8B_FX_KMAX];
    __shared__ float ls[R * (MQ8B_FX_KMAX / 256)];
    const int wid = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int row0 = (blockIdx.x * WPB + wid) * P;
    const bool live = row0 < M;
    const int lane = threadIdx.x & 31;
    const int h = lane >> 4;
    const int sub = lane & 15;
    const int gpr = K >> 8;
    const int npairs = (gpr + 1) >> 1;
    const int items = R * gpr;
    const size_t code_off = ((size_t)M * (size_t)gpr * 2 + 255) & ~(size_t)255;

    // 1. rotation inputs of this wave's first item (clamped: waves past the item count rotate nothing)
    const int item0 = min(wid, items - 1);
    const int r0 = item0 / gpr;
    Mq8bRotIn rin0 = mq8b_rot_load(x + (size_t)r0 * K + (size_t)(item0 - r0 * gpr) * 256, signs1, signs2, lane);
    __builtin_amdgcn_sched_barrier(0);

    // 2. weights
    uint4 w[P][U];
    float sl[P][U], sh[P][U];
    int g[U], gl[U], gh[U];
    bool ok[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
        gl[u] = min(2 * u, gpr - 1);
        gh[u] = min(2 * u + 1, gpr - 1);
        g[u] = h ? gh[u] : gl[u];
        ok[u] = (2 * u + h) < gpr && u < npairs;
    }
#pragma unroll
    for (int q = 0; q < P; q++) {
        const int rc = min(row0 + q, M - 1);
        const size_t srow = (size_t)rc * (size_t)gpr * 2;
        const unsigned char* crow = W + code_off + (size_t)rc * (size_t)K + (size_t)sub * 16;
#pragma unroll
        for (int u = 0; u < U; u++) {
            w[q][u] = *(const uint4*)(crow + (size_t)g[u] * 256);
            sl[q][u] = mq8b_scale(W, srow + 2 * (size_t)gl[u]);
            sh[q][u] = mq8b_scale(W, srow + 2 * (size_t)gh[u]);
        }
    }
    __builtin_amdgcn_sched_barrier(0);
    mq8b_rot_pin(rin0);
    __builtin_amdgcn_sched_barrier(0);

    // 3. rotate + quantize X: item = (row r, group g), one wave each, round-robin over the block's waves
    if (wid < items) {
        unsigned c0, c1;
        float sc;
        mq8b_rot_compute(rin0, lane, c0, c1, sc);
        *(uint2*)(lq + (size_t)r0 * K + (size_t)(item0 - r0 * gpr) * 256 + lane * 8) = make_uint2(c0, c1);
        if (lane == 0) ls[r0 * gpr + (item0 - r0 * gpr)] = sc;
    }
    for (int item = wid + WPB; item < items; item += WPB) {
        const int r = item / gpr;
        const int gg = item - r * gpr;
        const Mq8bRotIn rin = mq8b_rot_load(x + (size_t)r * K + (size_t)gg * 256, signs1, signs2, lane);
        unsigned c0, c1;
        float sc;
        mq8b_rot_compute(rin, lane, c0, c1, sc);
        *(uint2*)(lq + (size_t)r * K + (size_t)gg * 256 + lane * 8) = make_uint2(c0, c1);
        if (lane == 0) ls[r * gpr + gg] = sc;
    }
    // Barrier with an LDS-only fence: __syncthreads() would also wait for the in-flight weight loads.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    if (!live) return;

    float acc[P][R];
#pragma unroll
    for (int q = 0; q < P; q++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[q][r] = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int u = 0; u < U; u++) {
            const uint4 xv = *(const uint4*)(lq + (size_t)r * K + (size_t)g[u] * 256 + sub * 16);
            const float xl = ls[r * gpr + gl[u]];
            const float xh = ls[r * gpr + gh[u]];
#pragma unroll
            for (int q = 0; q < P; q++) {
                const int d = mq8b_dot16(w[q][u], xv);
                const float scl = sl[q][u] * xl;
                const float sch = sh[q][u] * xh;
                const float sc = h ? sch : scl;
                const float nacc = __builtin_fmaf(sc, (float)d, acc[q][r]);
                acc[q][r] = ok[u] ? nacc : acc[q][r];
            }
        }
    }
#pragma unroll
    for (int q = 0; q < P; q++) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float a = mq8b_reduce(acc[q][r]);
            if (lane == 0 && row0 + q < M) y[(size_t)r * M + row0 + q] = a;
        }
    }
}

#define GEMV_MQ8G256_BX(R, U, WPB, P)                                         \
    extern "C" __global__ __launch_bounds__(32 * WPB)                         \
    void gemv_mq8g256_bx_r##R##_u##U##_w##WPB##_p##P(                         \
        const unsigned char* __restrict__ W,                                  \
        const float* __restrict__ x,                                          \
        const float* __restrict__ signs1,                                     \
        const float* __restrict__ signs2,                                     \
        float* __restrict__ y,                                                \
        int M, int K                                                          \
    ) {                                                                       \
        mq8b_fx_body<R, U, WPB, P>(W, x, signs1, signs2, y, M, K);            \
    }

#define GEMV_MQ8G256_B(R, U, WPB, P)                                          \
    extern "C" __global__ __launch_bounds__(32 * WPB)                         \
    void gemv_mq8g256_b_r##R##_u##U##_w##WPB##_p##P(                          \
        const unsigned char* __restrict__ W,                                  \
        const signed char* __restrict__ xq,                                   \
        const float* __restrict__ xs,                                         \
        float* __restrict__ y,                                                \
        int M, int K, int ldxs                                                \
    ) {                                                                       \
        mq8b_body<R, U, WPB, P>(W, xq, xs, y, M, K, ldxs);                    \
    }

// Production set: one pair per loop step (U = 1), 4 waves per block, one weight row per wave, for R = 1..16.
// (Measured on gfx1201 / gfx1100: deeper pair unrolls and 2 rows per wave are slower at every Raon shape.)
#define GEMV_MQ8G256_B_U(R) GEMV_MQ8G256_B(R, 1, 4, 1)
GEMV_MQ8G256_B_U(1)
GEMV_MQ8G256_B_U(2)
GEMV_MQ8G256_B_U(3)
GEMV_MQ8G256_B_U(4)
GEMV_MQ8G256_B_U(5)
GEMV_MQ8G256_B_U(6)
GEMV_MQ8G256_B_U(7)
GEMV_MQ8G256_B_U(8)
GEMV_MQ8G256_B_U(9)
GEMV_MQ8G256_B_U(10)
GEMV_MQ8G256_B_U(11)
GEMV_MQ8G256_B_U(12)
GEMV_MQ8G256_B_U(13)
GEMV_MQ8G256_B_U(14)
GEMV_MQ8G256_B_U(15)
GEMV_MQ8G256_B_U(16)

// Fused-producer set (X given as F32; decode n = 1 and verify n = 2): one rotation group per wave of a block of
// W waves, U = ceil(K/512) pairs, one weight row per wave. K <= 512: W 10; K in (1536, 2560]: W 24;
// K in (2560, 5120]: W 20 (measured best on gfx1201 and gfx1100 at the Raon shapes).
#define GEMV_MQ8G256_BX_R(R) GEMV_MQ8G256_BX(R, 1, 10, 1) GEMV_MQ8G256_BX(R, 5, 24, 1) GEMV_MQ8G256_BX(R, 10, 20, 1)
GEMV_MQ8G256_BX_R(1)
GEMV_MQ8G256_BX_R(2)
