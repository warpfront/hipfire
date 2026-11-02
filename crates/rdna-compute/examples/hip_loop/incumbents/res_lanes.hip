#define HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE 1
// Architecture-neutral HFQ4 weight-buffer addressing, with per-ISA cache
// policy selection.
//
// HIPFIRE_WEIGHT_CPOL_AUX is the raw-buffer cache-policy immediate. GFX12 uses
// the TH/SCOPE encoding:
//   16 = TH_RT    + SCOPE_DEV (buffer-addressing control)
//   18 = TH_HT    + SCOPE_DEV
//   20 = TH_NT_RT + SCOPE_DEV
//   22 = TH_NT_HT + SCOPE_DEV
// Earlier RDNA generations use cache-policy zero for ordinary temporal reuse.
//
// This preamble is concatenated ahead of selected HFQ4 kernel sources by
// kernels.rs. gfx1201 defaults to buffer RT after matched model-level gates;
// gfx1100 now defaults to temporal buffer addressing after its own exact
// model-level A/B. Other targets remain byte-identical unless Radiowave
// explicitly selects HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN. Flat gfx1100 GEMV
// stays global unless HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN is set.
// HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS keeps the historical gfx12 control.
#if defined(HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE) && \
    (defined(__gfx1201__) || defined(__gfx1100__) || \
     defined(HIPFIRE_WEIGHT_BUFFER_LOADS_OPT_IN)) && \
    (!defined(HIPFIRE_WEIGHT_CACHE_FLAT_GEMV) || \
     defined(HIPFIRE_WEIGHT_BUFFER_LOADS_FLAT_GEMV_OPT_IN)) && \
    !defined(HIPFIRE_GFX11_WEIGHT_GLOBAL_LOADS) && \
    !defined(HIPFIRE_GFX12_WEIGHT_GLOBAL_LOADS)
#ifndef HIPFIRE_WEIGHT_CPOL_AUX
#if defined(__gfx1201__)
#define HIPFIRE_WEIGHT_CPOL_AUX 16
#else
#define HIPFIRE_WEIGHT_CPOL_AUX 0
#endif
#endif
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 1
#define HIPFIRE_WEIGHT_RSRC(base) \
    __builtin_amdgcn_make_buffer_rsrc( \
        const_cast<char*>(base), 0, 0xffffffffu, 0x31004000u)
#define HIPFIRE_WEIGHT_LOAD_U32(rsrc, byte_offset) \
    __builtin_amdgcn_raw_buffer_load_b32( \
        (rsrc), (unsigned int)(byte_offset), 0, HIPFIRE_WEIGHT_CPOL_AUX)
#define HIPFIRE_WEIGHT_LOAD_SPLIT_U32(rsrc, vector_offset, scalar_offset) \
    __builtin_amdgcn_raw_buffer_load_b32( \
        (rsrc), (unsigned int)(vector_offset), (unsigned int)(scalar_offset), \
        HIPFIRE_WEIGHT_CPOL_AUX)
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    HIPFIRE_WEIGHT_LOAD_U32((rsrc), (byte_offset))
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    HIPFIRE_WEIGHT_LOAD_SPLIT_U32((rsrc), (vector_offset), (scalar_offset))
#else
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    (*(const unsigned int*)(ptr))
#endif
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(rsrc, ptr, vector_offset, scalar_offset) \
    (*(const unsigned int*)(ptr))
#endif

#ifndef HIPFIRE_RESIDUAL_LANES_KERNEL
#define HIPFIRE_RESIDUAL_LANES_KERNEL gemv_mq4g256v2_residual_lanes_gfx1201
#endif

// Lane-batched MQ4G256V2 GEMV with fused residual add for exact gfx1201:
//
//   y[b][row] += A[row] . x[b]        b < B <= 8 lanes, K = 4096 (the A3B `wo`)
//
// Per lane b the bytes are those of the singleton `gemv_mq4g256v2_residual`
// (gemv_mq4g256v2_residual.hip) run on (x[b], y[b]): the same per-lane
// accumulation order (four group-interleaved accumulators, accumulator j
// holding groups j, j+4, j+8, j+12 in order, each group an 8-term product sum
// added whole), the same `(acc0 + acc1) + (acc2 + acc3)` combine, the same
// wave32 shfl_down tree and the same `y + acc` residual add. The weight row is
// streamed ONCE into registers (the singleton's buffer-load path and cache
// policy, a group ahead) and decoded once per group; only the per-column FMA
// chains repeat, on the lane's x elements read straight from global memory
// (L0/L2 hits: every wave of a workgroup reads the same activations).
//
// The four accumulators are independent until the combine, so the kernel walks
// one accumulator ("chain") at a time and folds the finished chain into the
// combine as it completes: only one accumulator set (R rows x 8 lanes) plus the
// partial sums are live.
//
// ABI: A, x, y, M, B. x is [B][4096] f32 row-major, y is [B][M] f32
// row-major, 1 <= B <= 8, M arbitrary. Grid = ceil(M / (W * R)) workgroups of
// W waves (block W * 32), no LDS and no barriers; wave w of workgroup g owns the
// R consecutive rows (g * W + w) * R ... + R - 1 (R even: row parity selects
// the contraction form below).

#ifndef HIPFIRE_RES_LANES_W
#define HIPFIRE_RES_LANES_W 4
#endif
#ifndef HIPFIRE_RES_LANES_R
#define HIPFIRE_RES_LANES_R 2
#endif
#define RL_W HIPFIRE_RES_LANES_W
#define RL_R HIPFIRE_RES_LANES_R
#define RL_NT (RL_W * 32)
#define RL_B 8
#define RL_K 4096
#define RL_NQ 4                       // groups per chain = RL_K / 1024
static_assert((RL_R & 1) == 0, "rows per wave must be even: row parity selects the singleton's contraction form");

// One 8-term group dot, the singleton's DOG_X8 sum `w0*x0 + w1*x1 + ... +
// w7*x7` added whole into the accumulator. The singleton leaves the contraction
// of the first two products to the compiler, which fuses them differently for
// the two rows its dual-row workgroup owns (measured per group of the quad and
// per row parity on the hardware, forms below): the even row (`acc*`) computes
// F0 = fma(w1, x1, w0*x0), the odd row (`bcc*`) F1 = fma(w0, x0, w1*x1), for
// all four groups; both then fuse w2*x2 .. w7*x7 in index order. The explicit
// fused chains pin those forms, independent of this kernel's own instruction
// order, so rows keep the singleton's parity (this kernel's rows per wave is
// even and its first row even).
#define DOG_F0(a, wv, xlo, xhi) do { \
    float t_ = (wv[0]) * (xlo.x); \
    t_ = __builtin_fmaf((wv[1]), (xlo.y), t_); \
    t_ = __builtin_fmaf((wv[2]), (xlo.z), t_); \
    t_ = __builtin_fmaf((wv[3]), (xlo.w), t_); \
    t_ = __builtin_fmaf((wv[4]), (xhi.x), t_); \
    t_ = __builtin_fmaf((wv[5]), (xhi.y), t_); \
    t_ = __builtin_fmaf((wv[6]), (xhi.z), t_); \
    t_ = __builtin_fmaf((wv[7]), (xhi.w), t_); \
    (a) += t_; \
} while (0)
#define DOG_F1(a, wv, xlo, xhi) do { \
    float t_ = (wv[1]) * (xlo.y); \
    t_ = __builtin_fmaf((wv[0]), (xlo.x), t_); \
    t_ = __builtin_fmaf((wv[2]), (xlo.z), t_); \
    t_ = __builtin_fmaf((wv[3]), (xlo.w), t_); \
    t_ = __builtin_fmaf((wv[4]), (xhi.x), t_); \
    t_ = __builtin_fmaf((wv[5]), (xhi.y), t_); \
    t_ = __builtin_fmaf((wv[6]), (xhi.z), t_); \
    t_ = __builtin_fmaf((wv[7]), (xhi.w), t_); \
    (a) += t_; \
} while (0)

__launch_bounds__(RL_NT, 1)
extern "C" __global__ void HIPFIRE_RESIDUAL_LANES_KERNEL(
    const char* __restrict__ A,
    const float* __restrict__ x,   // [B][4096]
    float* __restrict__ y,         // [B][M]
    int M, int B
) {
    const int tid = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const long long row_stride = (long long)(RL_K / 256) * 136;

    const int row_base = ((int)blockIdx.x * RL_W + wave) * RL_R;
    unsigned int woff[RL_R];
    #pragma unroll
    for (int r = 0; r < RL_R; r++) {
        const int row = (row_base + r < M) ? (row_base + r) : (M - 1);
        woff[r] = (unsigned int)((long long)row * row_stride);
    }
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
#else
    const int weight_rsrc = 0;
#endif
    // Per-lane byte offsets inside a group record: header dword (first half of
    // the group for lanes < 16, second half otherwise) and this lane's packed
    // dword.
    const unsigned int hoff = (tid < 16) ? 0u : 4u;
    const unsigned int poff = 8u + (unsigned int)tid * 4u;
    const float* xl = x + tid * 8;

    // raw weights of group g of every owned row, into slot `buf`
    unsigned int hs[2][RL_R], pk[2][RL_R];
#define RL_LOAD_W(buf, g) do { \
        _Pragma("unroll") \
        for (int r = 0; r < RL_R; r++) { \
            const unsigned int g_off = woff[r] + (unsigned int)((g) * 136); \
            hs[buf][r] = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(weight_rsrc, A + (long long)g_off + hoff, hoff, g_off); \
            pk[buf][r] = HIPFIRE_WEIGHT_LOAD_SPLIT_PTR_U32(weight_rsrc, A + (long long)g_off + poff, poff, g_off); \
        } \
    } while (0)

    float part[RL_R][RL_B];   // (acc0 + acc1), then (acc0 + acc1) + (acc2 + acc3)
    float part2[RL_R][RL_B];  // acc2
    #pragma unroll
    for (int r = 0; r < RL_R; r++)
        #pragma unroll
        for (int b = 0; b < RL_B; b++) { part[r][b] = 0.0f; part2[r][b] = 0.0f; }

    RL_LOAD_W(0, 0);

    #pragma unroll 1
    for (int j = 0; j < 4; j++) {
        float acc[RL_R][RL_B];
        #pragma unroll
        for (int r = 0; r < RL_R; r++)
            #pragma unroll
            for (int b = 0; b < RL_B; b++) acc[r][b] = 0.0f;

        #pragma unroll
        for (int q = 0; q < RL_NQ; q++) {
            const int buf = q & 1;
            const int g = j + 4 * q;
            // next step's raw weights: group j + 4 (q + 1) of this chain, or
            // group j + 1 of the next
            if (q + 1 < RL_NQ) { RL_LOAD_W(buf ^ 1, g + 4); }
            else if (j + 1 < 4) { RL_LOAD_W(buf ^ 1, j + 1); }
            float w[RL_R][8];
            #pragma unroll
            for (int r = 0; r < RL_R; r++) {
                const unsigned int h = hs[buf][r];
                const float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
                const float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
                const unsigned int p = pk[buf][r];
                const unsigned int lo = p & 0x0F0F0F0Fu;
                const unsigned int hi = (p >> 4) & 0x0F0F0F0Fu;
                w[r][0] = sc * (float)(lo & 0xFFu) + zp;
                w[r][1] = sc * (float)(hi & 0xFFu) + zp;
                w[r][2] = sc * (float)((lo >> 8) & 0xFFu) + zp;
                w[r][3] = sc * (float)((hi >> 8) & 0xFFu) + zp;
                w[r][4] = sc * (float)((lo >> 16) & 0xFFu) + zp;
                w[r][5] = sc * (float)((hi >> 16) & 0xFFu) + zp;
                w[r][6] = sc * (float)((lo >> 24) & 0xFFu) + zp;
                w[r][7] = sc * (float)((hi >> 24) & 0xFFu) + zp;
            }
            #pragma unroll
            for (int b = 0; b < RL_B; b++) {
                if (b >= B) break;
                const float* xp = xl + (long long)b * RL_K + g * 256;
                const float4 xa = *reinterpret_cast<const float4*>(xp);
                const float4 xb = *reinterpret_cast<const float4*>(xp + 4);
                #pragma unroll
                for (int r = 0; r < RL_R; r++) {
                    if (r & 1) DOG_F1(acc[r][b], w[r], xa, xb);
                    else DOG_F0(acc[r][b], w[r], xa, xb);
                }
            }
        }

        // fold the finished chain into (acc0 + acc1) + (acc2 + acc3)
        #pragma unroll
        for (int r = 0; r < RL_R; r++) {
            #pragma unroll
            for (int b = 0; b < RL_B; b++) {
                if (j == 0) part[r][b] = acc[r][b];
                else if (j == 1) part[r][b] = part[r][b] + acc[r][b];
                else if (j == 2) part2[r][b] = acc[r][b];
                else part[r][b] = part[r][b] + (part2[r][b] + acc[r][b]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < RL_R; r++) {
        #pragma unroll
        for (int b = 0; b < RL_B; b++) {
            if (b >= B) break;
            float s = part[r][b];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down(s, offset);
            if (tid == 0 && row_base + r < M) y[(long long)b * M + row_base + r] += s;
        }
    }
}
