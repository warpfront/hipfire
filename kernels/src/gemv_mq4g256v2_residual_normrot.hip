// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

// Folded residual GEMV + RMSNorm/FWHT-rotate: the GEMV body below is verbatim
// the generic (plain-load) path of gemv_mq4g256v2_residual (dual-row,
// 32-thread WG, grid M with early return for row0>=M, epilogue y[row]+=acc
// float2). After the y store, the LAST-arriving WG (via *ctr) performs the
// fused_rmsnorm_mq_rotate math over y[0..M) (norm length K == M here) with
// exactly the baseline arithmetic order (see notes in the epilogue), then
// resets *ctr to 0 so tape replay starts clean.
__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_residual_normrot(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K,
    const float* __restrict__ nw,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    float* __restrict__ x_rot,
    float eps,
    unsigned* __restrict__ ctr
) {
    __shared__ float nrm_part[256];
    __shared__ float nrm_warp[8];

    const int row0 = blockIdx.x << 1;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const bool have_row1 = row1 < M;
    const int tid = threadIdx.x;

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Exact Qwen3.6 A3B decode contract. Making both quads static lets LLVM
    // schedule the two independent row chains across the full K extent and
    // removes the dead generic tail machinery on gfx1151.
    const int groups_per_row = 16;
#else
    const int groups_per_row = K / 256;
#endif
    const long long row_stride = (long long)groups_per_row * 136;
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const int boff = tid * 4;
    const unsigned int row_weight_offset0 = (unsigned int)((long long)row0 * row_stride);
    const unsigned int row_weight_offset1 =
        (unsigned int)((long long)(have_row1 ? row1 : row0) * row_stride);
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    // One matrix-wide SRD keeps the dual-row kernel at 94 VGPR / zero scratch.
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float bcc0 = 0.0f, bcc1 = 0.0f, bcc2 = 0.0f, bcc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    #define DOG_X8(sc, zp, pk, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        (a) += ((sc) * (float)((pk) & 0xFu)         + (zp)) * (x0) \
             + ((sc) * (float)(((pk) >> 4) & 0xFu)  + (zp)) * (x1) \
             + ((sc) * (float)(((pk) >> 8) & 0xFu)  + (zp)) * (x2) \
             + ((sc) * (float)(((pk) >> 12) & 0xFu) + (zp)) * (x3) \
             + ((sc) * (float)(((pk) >> 16) & 0xFu) + (zp)) * (x4) \
             + ((sc) * (float)(((pk) >> 20) & 0xFu) + (zp)) * (x5) \
             + ((sc) * (float)(((pk) >> 24) & 0xFu) + (zp)) * (x6) \
             + ((sc) * (float)(((pk) >> 28) & 0xFu) + (zp)) * (x7); \
    } while (0)

    #if HIPFIRE_GFX1151_RESIDUAL_HYBRID_BUFFER
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) (*(const unsigned int*)(gp))
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + 4))
    #else
    #define HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, gp, weight_offset)
    #define HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset) HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 4, (weight_offset) + 4)
    #endif

    #define LOAD_W(gp, weight_offset, sc, zp, pk) do { \
        const unsigned int hA = HIPFIRE_RESIDUAL_LOAD_SC(weight_rsrc, gp, weight_offset); \
        const unsigned int hB = HIPFIRE_RESIDUAL_LOAD_ZP(weight_rsrc, gp, weight_offset); \
        const unsigned int hs = (tid < 16) ? hA : hB; \
        (sc) = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu))); \
        (zp) = __half2float(__ushort_as_half((unsigned short)(hs >> 16))); \
        (pk) = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, (gp) + 8 + boff, (weight_offset) + 8 + boff); \
    } while (0)

#if defined(HIPFIRE_GFX1151_RESIDUAL_K4096)
    // Keep the fixed four-quad bound without expanding the body. Full unroll
    // doubles the instruction footprint; factor-two unroll spills private=16
    // and is not admissible for direct gfx1151 PM4 replay.
    #pragma unroll 1
#endif
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base0 = g * 256 + tid * 8;

        // Hoist full 32-wide x for this quad (4 groups × 8) via float4 pairs.
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        const float4 xv2a = *reinterpret_cast<const float4*>(x + base0 + 512);
        const float4 xv2b = *reinterpret_cast<const float4*>(x + base0 + 516);
        const float4 xv3a = *reinterpret_cast<const float4*>(x + base0 + 768);
        const float4 xv3b = *reinterpret_cast<const float4*>(x + base0 + 772);

        // Weight MEMORY phase: sequential along row0 then row1 (8 headers).
        float sc0a, zp0a, sc1a, zp1a, sc2a, zp2a, sc3a, zp3a;
        float sc0b, zp0b, sc1b, zp1b, sc2b, zp2b, sc3b, zp3b;
        unsigned int pk0a, pk1a, pk2a, pk3a, pk0b, pk1b, pk2b, pk3b;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0a, zp0a, pk0a);
        LOAD_W(row_ptr0 + (g + 1) * 136, row_weight_offset0 + (g + 1) * 136, sc1a, zp1a, pk1a);
        LOAD_W(row_ptr0 + (g + 2) * 136, row_weight_offset0 + (g + 2) * 136, sc2a, zp2a, pk2a);
        LOAD_W(row_ptr0 + (g + 3) * 136, row_weight_offset0 + (g + 3) * 136, sc3a, zp3a, pk3a);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc0b, zp0b, pk0b);
        LOAD_W(row_ptr1 + (g + 1) * 136, row_weight_offset1 + (g + 1) * 136, sc1b, zp1b, pk1b);
        LOAD_W(row_ptr1 + (g + 2) * 136, row_weight_offset1 + (g + 2) * 136, sc2b, zp2b, pk2b);
        LOAD_W(row_ptr1 + (g + 3) * 136, row_weight_offset1 + (g + 3) * 136, sc3b, zp3b, pk3b);

        // Pure FMA phase: same dual-row DOG order and acc slots as baseline.
        DOG_X8(sc0a, zp0a, pk0a, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(sc0b, zp0b, pk0b, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);

        DOG_X8(sc1a, zp1a, pk1a, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(sc1b, zp1b, pk1b, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);

        DOG_X8(sc2a, zp2a, pk2a, acc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);
        DOG_X8(sc2b, zp2b, pk2b, bcc2,
               xv2a.x, xv2a.y, xv2a.z, xv2a.w, xv2b.x, xv2b.y, xv2b.z, xv2b.w);

        DOG_X8(sc3a, zp3a, pk3a, acc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
        DOG_X8(sc3b, zp3b, pk3b, bcc3,
               xv3a.x, xv3a.y, xv3a.z, xv3a.w, xv3b.x, xv3b.y, xv3b.z, xv3b.w);
    }

    if (tail >= 1) {
        const int g = quads << 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc0,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc1,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const int base = g * 256 + tid * 8;
        const float4 xa = *reinterpret_cast<const float4*>(x + base);
        const float4 xb = *reinterpret_cast<const float4*>(x + base + 4);
        float sc0, zp0, sc1, zp1;
        unsigned int pk0, pk1;
        LOAD_W(row_ptr0 + g * 136, row_weight_offset0 + g * 136, sc0, zp0, pk0);
        LOAD_W(row_ptr1 + g * 136, row_weight_offset1 + g * 136, sc1, zp1, pk1);
        DOG_X8(sc0, zp0, pk0, acc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
        DOG_X8(sc1, zp1, pk1, bcc2,
               xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
    }
    #undef LOAD_W
    #undef HIPFIRE_RESIDUAL_LOAD_SC
    #undef HIPFIRE_RESIDUAL_LOAD_ZP
    #undef DOG_X8

    float acc = (acc0 + acc1) + (acc2 + acc3);
    float bcc = (bcc0 + bcc1) + (bcc2 + bcc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);

    if (tid == 0) {
        if (have_row1) {
            float2 yy = *reinterpret_cast<float2*>(y + row0);
            yy.x += acc;
            yy.y += bcc;
            *reinterpret_cast<float2*>(y + row0) = yy;
        } else {
            y[row0] += acc;
        }
    }

    // ——— folded norm+rotate epilogue: last-arriving WG only ———
    // WGs that early-returned above never reach here, so the arrival count
    // is (M+1)/2 (== grid when launched with grid (M+1)/2).
    __threadfence();
    unsigned nrm_prev = 0u;
    if (tid == 0) nrm_prev = atomicAdd(ctr, 1u);
    nrm_prev = __shfl(nrm_prev, 0);
    const int nrm_active = (M + 1) >> 1;
    if (nrm_prev == (unsigned)(nrm_active - 1)) {
        // Other WGs' y stores must be visible; plain loads after the fence.
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        const int NK = M; // norm length (== M here)

        // Phase 1a: emulate the 256 virtual threads with 32 lanes × 8 slots
        // — lane l handles virtual threads t = l + 32*j (j=0..7), same
        // increasing-i order as the baseline's `for (i=tid; i<K; i+=256)`.
        float nrm_ls[8];
        for (int j = 0; j < 8; j++) {
            float s = 0.0f;
            for (int i = tid + j * 32; i < NK; i += 256) {
                float v = y[i];
                s += v * v;
            }
            nrm_ls[j] = s;
            nrm_part[tid + j * 32] = s;
        }

        // Phase 1b, stage 1: baseline reduces each warp's 32 partials with
        // __shfl_down 16/8/4/2/1 then lane 0 publishes reduce[warp_id].
        // Virtual warp w's partials are nrm_part[w*32 + lane]; lane l of this
        // wave holds nrm_ls[w] == nrm_part[w*32+l], so the real shuffle
        // sequence on this wave reproduces each warp sum bit-exactly.
        for (int w = 0; w < 8; w++) {
            float ws = nrm_part[w * 32 + tid];
            ws += __shfl_down(ws, 16);
            ws += __shfl_down(ws, 8);
            ws += __shfl_down(ws, 4);
            ws += __shfl_down(ws, 2);
            ws += __shfl_down(ws, 1);
            if (tid == 0) nrm_warp[w] = ws;
        }
        __syncthreads();

        // Phase 1b, stage 2: baseline (tid<32) loads reduce[tid] (0 above
        // num_warps=8), then __shfl_down 4/2/1 and lane 0 publishes
        // rsqrtf(sum/K + eps). Same values + same shuffle sequence here.
        float nrm_ws = (tid < 8) ? nrm_warp[tid] : 0.0f;
        nrm_ws += __shfl_down(nrm_ws, 4);
        nrm_ws += __shfl_down(nrm_ws, 2);
        nrm_ws += __shfl_down(nrm_ws, 1);
        if (tid == 0) nrm_warp[0] = rsqrtf(nrm_ws / (float)NK + eps);
        __syncthreads();
        const float rms = nrm_warp[0];

        // Phase 2: baseline assigns warp w groups [w*gpw, w*gpw+gpw) with
        // lane mapping d0 = lane*8. Emulate each virtual warp in turn with
        // this wave's lane l == virtual lane l, so intra-wave swizzles see
        // the identical lane layout. Groups are independent; sequential
        // order across warps changes nothing elementwise. The baseline's
        // prefetch only moved load issue time; values are identical, so
        // plain loads here are bit-exact.
        const int nrm_groups_total = NK / 256;
        const int nrm_gpw = (nrm_groups_total + 8 - 1) / 8;
        const int nrm_d0 = tid * 8;
        const float4 nrm_s10 = *reinterpret_cast<const float4*>(s1 + nrm_d0);
        const float4 nrm_s11 = *reinterpret_cast<const float4*>(s1 + nrm_d0 + 4);
        const float4 nrm_s20 = *reinterpret_cast<const float4*>(s2 + nrm_d0);
        const float4 nrm_s21 = *reinterpret_cast<const float4*>(s2 + nrm_d0 + 4);
        for (int w = 0; w < 8; w++) {
            for (int gi = 0; gi < nrm_gpw; gi++) {
                const int group = w * nrm_gpw + gi;
                if (group >= nrm_groups_total) break;
                const int base = group * 256 + nrm_d0;

                const float4 x0 = *reinterpret_cast<const float4*>(y + base);
                const float4 x1 = *reinterpret_cast<const float4*>(y + base + 4);
                const float4 w0 = *reinterpret_cast<const float4*>(nw + base);
                const float4 w1 = *reinterpret_cast<const float4*>(nw + base + 4);

                // Exact multiply order: ((x * weight) * rms) * sign.
                float v0 = x0.x * w0.x * rms * nrm_s10.x;
                float v1 = x0.y * w0.y * rms * nrm_s10.y;
                float v2 = x0.z * w0.z * rms * nrm_s10.z;
                float v3 = x0.w * w0.w * rms * nrm_s10.w;
                float v4 = x1.x * w1.x * rms * nrm_s11.x;
                float v5 = x1.y * w1.y * rms * nrm_s11.y;
                float v6 = x1.z * w1.z * rms * nrm_s11.z;
                float v7 = x1.w * w1.w * rms * nrm_s11.w;

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

                #define NRM_HBFLY(v, pat, str) do { \
                    float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
                    if (tid & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
                } while (0)
                #define NRM_HBFLY8(pat, str) \
                    NRM_HBFLY(v0,pat,str); NRM_HBFLY(v1,pat,str); NRM_HBFLY(v2,pat,str); NRM_HBFLY(v3,pat,str); \
                    NRM_HBFLY(v4,pat,str); NRM_HBFLY(v5,pat,str); NRM_HBFLY(v6,pat,str); NRM_HBFLY(v7,pat,str)

                NRM_HBFLY8(0x041F, 1);
                NRM_HBFLY8(0x081F, 2);
                NRM_HBFLY8(0x101F, 4);
                NRM_HBFLY8(0x201F, 8);
                NRM_HBFLY8(0x401F, 16);
                #undef NRM_HBFLY8
                #undef NRM_HBFLY

                const float sc = 0.0625f;
                float4 out0;
                float4 out1;
                out0.x = v0 * sc * nrm_s20.x;
                out0.y = v1 * sc * nrm_s20.y;
                out0.z = v2 * sc * nrm_s20.z;
                out0.w = v3 * sc * nrm_s20.w;
                out1.x = v4 * sc * nrm_s21.x;
                out1.y = v5 * sc * nrm_s21.y;
                out1.z = v6 * sc * nrm_s21.z;
                out1.w = v7 * sc * nrm_s21.w;
                *reinterpret_cast<float4*>(x_rot + base) = out0;
                *reinterpret_cast<float4*>(x_rot + base + 4) = out1;
            }
        }

        // Only the last WG runs this, so the reset is safe for tape replay.
        if (tid == 0) *ctr = 0;
    }
}
