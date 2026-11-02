// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

// A3B (hidden 2048) decode: RMSNorm + FWHT rotate folded into the MQ4G256V2
// projection GEMV that consumes its output (exact gfx1201, K == 2048, rows == 1).
//
// The incumbent chain is two launches: `fused_rmsnorm_mq_rotate` on ONE
// workgroup, then `fused_qkvza_mq4g256v2` / `fused_qkv_mq4g256v2` with one
// 32-lane wave per output row. This kernel is ONE launch of G workgroups x 256
// threads. Every workgroup recomputes the row's rms and FWHT with the
// incumbent's exact 256-thread code (Phase 1a strided sum of squares, the
// shfl_down tree, rsqrt, per-wave group FWHT, same operand order), keeps the
// rotated activation in LDS, then each of its eight waves walks output rows
// `wave + 8 * G * i` with the consumer's exact per-row arithmetic: four
// accumulators over the 4x-unrolled group loop, `(acc0+acc1)+(acc2+acc3)`, the
// 16/8/4/2/1 shfl_down tree. Weights of a wave's first row are issued before
// the norm so their latency overlaps it; each later row's weights are issued
// before the previous row's FMAs.
//
// Four weight sources route rows in order (source 0 first). The qkvza consumer
// uses all four; the qkv consumer passes m3 == 0. Workgroup 0 also stores the
// rotated vector to `x_rot` (later launches of the layer read it).
//
// Output bytes equal the incumbent chain: x_rot is the plain
// `fused_rmsnorm_mq_rotate` output, and every y row is the consumer's.

#define NF_K 2048
#define NF_GROUPS 8
#define NF_WAVES 8

struct NfRow {
    unsigned int hs[NF_GROUPS];  // this lane's scale/zero word (lane < 16: half 0, else half 1)
    unsigned int pk[NF_GROUPS];  // this lane's packed nibbles (8 elements)
};

__device__ __forceinline__ void nf_load_row(NfRow& r, const char* row_ptr, int lane) {
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif
    const int hoff = (lane < 16) ? 0 : 4;
    const int boff = lane * 4;
    #pragma unroll
    for (int g = 0; g < NF_GROUPS; ++g) {
        const char* gp = row_ptr + g * 136;
        r.hs[g] = HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, gp + hoff, g * 136 + hoff);
        r.pk[g] = HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, gp + 8 + boff, g * 136 + 8 + boff);
    }
}

#define NF_DOG(pk, sc, zp, xa, xb, a) \
    (a) += (sc * (float)((pk) & 0xFu)         + zp) * (xa.x) \
         + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * (xa.y) \
         + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * (xa.z) \
         + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * (xa.w) \
         + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * (xb.x) \
         + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * (xb.y) \
         + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * (xb.z) \
         + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * (xb.w)

__device__ __forceinline__ float nf_row_dot(const NfRow& r, const float* x_lds, int lane) {
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        float4 xa[4], xb[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int base = (q * 4 + i) * 256 + lane * 8;
            xa[i] = *reinterpret_cast<const float4*>(x_lds + base);
            xb[i] = *reinterpret_cast<const float4*>(x_lds + base + 4);
        }
        float sc[4], zp[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int h = r.hs[q * 4 + i];
            sc[i] = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
            zp[i] = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        }
        const unsigned int p0 = r.pk[q * 4 + 0], p1 = r.pk[q * 4 + 1];
        const unsigned int p2 = r.pk[q * 4 + 2], p3 = r.pk[q * 4 + 3];
        NF_DOG(p0, sc[0], zp[0], xa[0], xb[0], acc0);
        NF_DOG(p1, sc[1], zp[1], xa[1], xb[1], acc1);
        NF_DOG(p2, sc[2], zp[2], xa[2], xb[2], acc2);
        NF_DOG(p3, sc[3], zp[3], xa[3], xb[3], acc3);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    return acc;
}
#undef NF_DOG

__launch_bounds__(256, 1)
extern "C" __global__ void fused_rmsnorm_proj4_mq4g256v2_fold_gfx1201(
    const char*  __restrict__ A0,
    const char*  __restrict__ A1,
    const char*  __restrict__ A2,
    const char*  __restrict__ A3,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float*       __restrict__ y0,
    float*       __restrict__ y1,
    float*       __restrict__ y2,
    float*       __restrict__ y3,
    float*       __restrict__ x_rot,
    int m0, int m1, int m2, int m3,
    int nw,                      // total row waves = gridDim.x * 8
    float eps
) {
    __shared__ float x_lds[NF_K];
    __shared__ float reduce[256];

    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    // Wave-uniform by construction; the compiler cannot prove it, and a divergent
    // row pointer would turn every weight buffer load into a waterfall loop.
    const int warp_id = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int K = NF_K;

    const int total_m = m0 + m1 + m2 + m3;
    int gr = blockIdx.x * NF_WAVES + warp_id;

    // Row routing: matrix, output row. Wave-uniform.
#define NF_ROUTE(gid, A, y, row) \
    do { \
        if ((gid) < m0) { A = A0; y = y0; row = (gid); } \
        else if ((gid) < m0 + m1) { A = A1; y = y1; row = (gid) - m0; } \
        else if ((gid) < m0 + m1 + m2) { A = A2; y = y2; row = (gid) - m0 - m1; } \
        else { A = A3; y = y3; row = (gid) - m0 - m1 - m2; } \
    } while (0)

    // Weights do not depend on the norm: issue the first row's loads now.
    NfRow cur;
    float* cur_y = y0;
    int cur_row = 0;
    if (gr < total_m) {
        const char* A; float* y; int row;
        NF_ROUTE(gr, A, y, row);
        cur_y = y; cur_row = row;
        nf_load_row(cur, A + (long long)row * NF_GROUPS * 136, lane_id);
    }

    // ---- incumbent fused_rmsnorm_mq_rotate, K = 2048, one group per wave ----
    // Phase 1a: sum of squares, same stride/order as the incumbent.
    float local_sum = 0.0f;
    for (int i = tid; i < K; i += 256) {
        float v = x[i];
        local_sum += v * v;
    }

    const int d0 = lane_id * 8;
    float4 pref_x0, pref_x1, pref_w0, pref_w1;
    float4 s10, s11, s20, s21;
    {
        const int base = warp_id * 256 + d0;
        pref_x0 = *reinterpret_cast<const float4*>(x + base);
        pref_x1 = *reinterpret_cast<const float4*>(x + base + 4);
        pref_w0 = *reinterpret_cast<const float4*>(weight + base);
        pref_w1 = *reinterpret_cast<const float4*>(weight + base + 4);
    }
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
        warp_sum = (tid < NF_WAVES) ? reduce[tid] : 0.0f;
        warp_sum += __shfl_down(warp_sum, 4);
        warp_sum += __shfl_down(warp_sum, 2);
        warp_sum += __shfl_down(warp_sum, 1);
        if (tid == 0) reduce[0] = rsqrtf(warp_sum / (float)K + eps);
    }
    __syncthreads();
    const float rms = reduce[0];

    {
        const int base = warp_id * 256 + d0;
        const float4 x0 = pref_x0, x1 = pref_x1, w0 = pref_w0, w1 = pref_w1;
        // Exact multiply order: ((x * weight) * rms) * sign.
        float v0 = x0.x * w0.x * rms * s10.x;
        float v1 = x0.y * w0.y * rms * s10.y;
        float v2 = x0.z * w0.z * rms * s10.z;
        float v3 = x0.w * w0.w * rms * s10.w;
        float v4 = x1.x * w1.x * rms * s11.x;
        float v5 = x1.y * w1.y * rms * s11.y;
        float v6 = x1.z * w1.z * rms * s11.z;
        float v7 = x1.w * w1.w * rms * s11.w;

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
        float4 out0, out1;
        out0.x = v0 * s * s20.x;
        out0.y = v1 * s * s20.y;
        out0.z = v2 * s * s20.z;
        out0.w = v3 * s * s20.w;
        out1.x = v4 * s * s21.x;
        out1.y = v5 * s * s21.y;
        out1.z = v6 * s * s21.z;
        out1.w = v7 * s * s21.w;
        *reinterpret_cast<float4*>(x_lds + base) = out0;
        *reinterpret_cast<float4*>(x_lds + base + 4) = out1;
        if (blockIdx.x == 0) {
            *reinterpret_cast<float4*>(x_rot + base) = out0;
            *reinterpret_cast<float4*>(x_rot + base + 4) = out1;
        }
    }
    __syncthreads();

    // ---- rows: the next row's weights are in flight while this row accumulates ----
    while (gr < total_m) {
        const int gn = gr + nw;
        NfRow nxt;
        float* nxt_y = y0;
        int nxt_row = 0;
        if (gn < total_m) {
            const char* A; float* y; int row;
            NF_ROUTE(gn, A, y, row);
            nxt_y = y; nxt_row = row;
            nf_load_row(nxt, A + (long long)row * NF_GROUPS * 136, lane_id);
        }
        const float acc = nf_row_dot(cur, x_lds, lane_id);
        if (lane_id == 0) cur_y[cur_row] = acc;
        cur = nxt;
        cur_y = nxt_y;
        cur_row = nxt_row;
        gr = gn;
    }
#undef NF_ROUTE
}
