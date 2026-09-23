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
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
using hipfire_qkv_i32x4 = int __attribute__((ext_vector_type(4)));
using hipfire_qkv_f32x4 = float __attribute__((ext_vector_type(4)));
#endif

#ifndef HIPFIRE_QKV_KERNEL_NAME
#define HIPFIRE_QKV_KERNEL_NAME fused_qkv_mq4g256v2
#endif

#if defined(HIPFIRE_QKV_WITH_BIAS)
#define HIPFIRE_QKV_BIAS_ARGS \
    , const float* __restrict__ bias_q \
    , const float* __restrict__ bias_k \
    , const float* __restrict__ bias_v
#define HIPFIRE_QKV_BIAS_DECL const float* bias;
#define HIPFIRE_QKV_BIAS_Q bias = bias_q;
#define HIPFIRE_QKV_BIAS_K bias = bias_k;
#define HIPFIRE_QKV_BIAS_V bias = bias_v;
#else
#define HIPFIRE_QKV_BIAS_ARGS
#define HIPFIRE_QKV_BIAS_DECL
#define HIPFIRE_QKV_BIAS_Q
#define HIPFIRE_QKV_BIAS_K
#define HIPFIRE_QKV_BIAS_V
#endif

// 3-way fused HFQ4-G256 GEMV for the Qwen3.5 FullAttention layer preamble.
// One kernel launch performs all three projections from a single shared
// x_rot input:
//
//   y_q = W_q · x_rot   (M=q_m, n_heads × head_dim × 2 — Q+gate interleaved)
//   y_k = W_k · x_rot   (M=k_m, n_kv_heads × head_dim)
//   y_v = W_v · x_rot   (M=v_m, n_kv_heads × head_dim)
//
// Same 4x-unroll inner loop and tail handling as gemv_hfq4g256.hip
// (5302926 accumulator-interleave fix). Grid = q_m + k_m + v_m.
//
// Arch coverage: works on every RDNA generation (gfx1010 / gfx1013 /
// gfx1030 / gfx1100+). See fused_qkvza_hfq4g256.hip for the full
// rationale on the unified 4-accumulator inner loop and the per-arch
// register-budget assumptions baked into launch_bounds(32, 16).
//
// Bit-exact with three sequential gemv_hfq4g256 calls.

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_QKV_KERNEL_NAME(
    const char*  __restrict__ A_q,
    const char*  __restrict__ A_k,
    const char*  __restrict__ A_v,
    const float* __restrict__ x,
    float*       __restrict__ y_q,
    float*       __restrict__ y_k,
    float*       __restrict__ y_v,
    int q_m, int k_m, int v_m,
    int K HIPFIRE_QKV_BIAS_ARGS
#ifdef HIPFIRE_MQ4G256V2_LUT
    // Lloyd (qt=52): three per-tensor centered f16 codebooks (q, k, v),
    // 8 dwords each, staged to LDS once; the block indexes its source.
    , const unsigned int L0, const unsigned int L1, const unsigned int L2, const unsigned int L3,
    const unsigned int L4, const unsigned int L5, const unsigned int L6, const unsigned int L7,
    const unsigned int L8, const unsigned int L9, const unsigned int L10, const unsigned int L11,
    const unsigned int L12, const unsigned int L13, const unsigned int L14, const unsigned int L15,
    const unsigned int L16, const unsigned int L17, const unsigned int L18, const unsigned int L19,
    const unsigned int L20, const unsigned int L21, const unsigned int L22, const unsigned int L23
#endif
) {
    const int gid = blockIdx.x;
    const int total_m = q_m + k_m + v_m;
    if (gid >= total_m) return;

    // Route to one of the three projections.
    const char* A;
    float* y;
    int row;
    HIPFIRE_QKV_BIAS_DECL
    int lut_base = 0;
    if (gid < q_m) {
        A = A_q; y = y_q; row = gid; HIPFIRE_QKV_BIAS_Q
    } else if (gid < q_m + k_m) {
        A = A_k; y = y_k; row = gid - q_m; HIPFIRE_QKV_BIAS_K              lut_base = 16;
    } else {
        A = A_v; y = y_v; row = gid - q_m - k_m; HIPFIRE_QKV_BIAS_V        lut_base = 32;
    }

    const int tid = threadIdx.x;
#ifdef HIPFIRE_MQ4G256V2_LUT
    __shared__ unsigned short LUT_SMEM[48];
    if (tid < 24) {
        unsigned int d;
        switch (tid) {
            case 0: d = L0; break;   case 1: d = L1; break;   case 2: d = L2; break;   case 3: d = L3; break;
            case 4: d = L4; break;   case 5: d = L5; break;   case 6: d = L6; break;   case 7: d = L7; break;
            case 8: d = L8; break;   case 9: d = L9; break;   case 10: d = L10; break; case 11: d = L11; break;
            case 12: d = L12; break; case 13: d = L13; break; case 14: d = L14; break; case 15: d = L15; break;
            case 16: d = L16; break; case 17: d = L17; break; case 18: d = L18; break; case 19: d = L19; break;
            case 20: d = L20; break; case 21: d = L21; break; case 22: d = L22; break; default: d = L23; break;
        }
        ((unsigned int*)LUT_SMEM)[tid] = d;
    }
    __syncthreads();
    #define MQ4V2_DEQ(q) (__half2float(__ushort_as_half(LUT_SMEM[lut_base + (q)])))
#else
    (void)lut_base;
    #define MQ4V2_DEQ(q) ((float)(q))
#endif
#if defined(HIPFIRE_RDNA3_QKV_K2048)
    const int groups_per_row = 8;
#else
    const int groups_per_row = K / 256;
#endif
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc = HIPFIRE_WEIGHT_RSRC(row_ptr);
#endif
#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
    const auto x_rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(x), 0, 0xffffffffu, 0x31004000u);
#endif

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail  = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        const unsigned int hA_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0, (g + 0) * 136);
        const unsigned int hB_sc0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 4, (g + 0) * 136 + 4);
        const unsigned int hs_sc0 = (tid < 16) ? hA_sc0 : hB_sc0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs_sc0 >> 16)));
        const unsigned int hA_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1, (g + 1) * 136);
        const unsigned int hB_sc1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 4, (g + 1) * 136 + 4);
        const unsigned int hs_sc1 = (tid < 16) ? hA_sc1 : hB_sc1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs_sc1 >> 16)));
        const unsigned int hA_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2, (g + 2) * 136);
        const unsigned int hB_sc2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 4, (g + 2) * 136 + 4);
        const unsigned int hs_sc2 = (tid < 16) ? hA_sc2 : hB_sc2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs_sc2 >> 16)));
        const unsigned int hA_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3, (g + 3) * 136);
        const unsigned int hB_sc3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 4, (g + 3) * 136 + 4);
        const unsigned int hs_sc3 = (tid < 16) ? hA_sc3 : hB_sc3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs_sc3 >> 16)));
        unsigned int pk0 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp0 + 8 + boff, (g + 0) * 136 + 8 + boff);
        unsigned int pk1 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp1 + 8 + boff, (g + 1) * 136 + 8 + boff);
        unsigned int pk2 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp2 + 8 + boff, (g + 2) * 136 + 8 + boff);
        unsigned int pk3 = HIPFIRE_WEIGHT_LOAD_PTR_U32(row_rsrc, gp3 + 8 + boff, (g + 3) * 136 + 8 + boff);

        int base = g * 256 + tid * 8;

#if defined(HIPFIRE_RDNA3_QKV_X_BUFFER)
        const hipfire_qkv_f32x4 xv0a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 0) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv0b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 4) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv1a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 256) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv1b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 260) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv2a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 512) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv2b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 516) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv3a = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 768) * 4, 0, 0));
        const hipfire_qkv_f32x4 xv3b = __builtin_bit_cast(
            hipfire_qkv_f32x4, __builtin_amdgcn_raw_buffer_load_b128(
                x_rsrc, (base + 772) * 4, 0, 0));
        #define DOG_V(pk, sc, zp, a, x0, x1, x2, x3, x4, x5, x6, x7) \
            (a) += (sc * MQ4V2_DEQ((pk) & 0xFu)         + zp) * (x0) \
                 + (sc * MQ4V2_DEQ(((pk) >> 4) & 0xFu)  + zp) * (x1) \
                 + (sc * MQ4V2_DEQ(((pk) >> 8) & 0xFu)  + zp) * (x2) \
                 + (sc * MQ4V2_DEQ(((pk) >> 12) & 0xFu) + zp) * (x3) \
                 + (sc * MQ4V2_DEQ(((pk) >> 16) & 0xFu) + zp) * (x4) \
                 + (sc * MQ4V2_DEQ(((pk) >> 20) & 0xFu) + zp) * (x5) \
                 + (sc * MQ4V2_DEQ(((pk) >> 24) & 0xFu) + zp) * (x6) \
                 + (sc * MQ4V2_DEQ(((pk) >> 28) & 0xFu) + zp) * (x7)
        DOG_V(pk0, sc0, zp0, acc0, xv0a.x,xv0a.y,xv0a.z,xv0a.w,xv0b.x,xv0b.y,xv0b.z,xv0b.w);
        DOG_V(pk1, sc1, zp1, acc1, xv1a.x,xv1a.y,xv1a.z,xv1a.w,xv1b.x,xv1b.y,xv1b.z,xv1b.w);
        DOG_V(pk2, sc2, zp2, acc2, xv2a.x,xv2a.y,xv2a.z,xv2a.w,xv2b.x,xv2b.y,xv2b.z,xv2b.w);
        DOG_V(pk3, sc3, zp3, acc3, xv3a.x,xv3a.y,xv3a.z,xv3a.w,xv3b.x,xv3b.y,xv3b.z,xv3b.w);
        #undef DOG_V
#else
        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * MQ4V2_DEQ((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * MQ4V2_DEQ(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * MQ4V2_DEQ(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * MQ4V2_DEQ(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * MQ4V2_DEQ(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * MQ4V2_DEQ(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * MQ4V2_DEQ(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * MQ4V2_DEQ(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base,       acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
#endif
    }

    // Tail handling: preserve acc[gi] interleaving. See 5302926 bug report.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * MQ4V2_DEQ((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * MQ4V2_DEQ(((pk) >>  4) & 0xFu) + zp) * x[(b) + 1] \
             + (sc * MQ4V2_DEQ(((pk) >>  8) & 0xFu) + zp) * x[(b) + 2] \
             + (sc * MQ4V2_DEQ(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * MQ4V2_DEQ(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * MQ4V2_DEQ(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * MQ4V2_DEQ(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * MQ4V2_DEQ(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        const unsigned int hA_sc = *(const unsigned int*)(gp);
        const unsigned int hB_sc = *(const unsigned int*)(gp + 4);
        const unsigned int hs_sc = (tid < 16) ? hA_sc : hB_sc;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs_sc & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs_sc >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

#if defined(HIPFIRE_QKV_WITH_BIAS)
    if (tid == 0) y[row] = acc + (bias ? bias[row] : 0.0f);
#else
    if (tid == 0) y[row] = acc;
#endif
}
