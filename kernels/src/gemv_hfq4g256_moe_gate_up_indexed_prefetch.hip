// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// S_PREFETCH_DATA sweep variants of the index-aware fused MoE gate_up GEMV.
// Byte-identical math to gemv_hfq4g256_moe_gate_up_indexed.hip — the ONLY
// change is a software scalar prefetch (`__builtin_amdgcn_s_prefetch_data`)
// issued at the top of each quad iteration for a FUTURE quad's gate/up weight
// block. gfx12 (RDNA4) only: the builtin lowers to `s_prefetch_data` which is
// a gfx12-insts instruction.
//
// Encoding (verified via clang -S, gfx1201):
//   s_prefetch_data s[base:base+1], <byte-offset>, null, <LEN>
//   - base is a SCALAR (wave-uniform) address (SGPR pair). gate_ptr/up_ptr are
//     wave-uniform (depend on blockIdx/expert, not tid), so this is legal.
//   - LEN is an immediate cache-line count (0..31 valid). One RDNA4 line = 128B.
//
// A quad = 4 groups = 4*136 = 544 B of gate weights (contiguous) + 544 B up.
// 544 B ~= 5 cache lines. The divergent VECTOR nibble loads
// (`*(gp + 8 + tid*4)`, tid=0..31 → 128 B block) read the SAME lines this
// scalar prefetch warms — IFF s_prefetch_data fills a tier shared by the
// vector path. That is exactly the hypothesis under test.
//
// Variants (select via HIPFIRE_GFX1201_MOE_PREFETCH ∈ {1,2,3,4} at dispatch):
//   1 (_pf1): prefetch NEXT quad (g+4), gate+up, LEN=5 (whole quad).
//   2 (_pf2): prefetch 2 quads ahead (g+8), gate+up, LEN=5 (deeper overlap).
//   3 (_pf3): prefetch NEXT quad (g+4), gate+up, LEN=2 (light, leading lines).
//   4 (_pf0): prefetch NEXT quad (g+4), gate+up, LEN=0 (probes what the
//             existing gfx1201-file idiom's `,0` actually does).

#define MOE_GATE_UP_PREFETCH_BODY(FN, PF_DIST_GROUPS, PF_LEN)                 \
extern "C" __global__ void FN(                                                \
    const unsigned long long* __restrict__ expert_ptrs,                       \
    const int* __restrict__ topk_indices,                                     \
    const float* __restrict__ x,                                              \
    float* __restrict__ y_gate,                                               \
    float* __restrict__ y_up,                                                 \
    int M, int K                                                              \
) {                                                                           \
    const int mi = M >> 1;                                                     \
    const int row = blockIdx.x;                                                \
    if (row >= mi) return;                                                     \
    const int krank = blockIdx.y;                                              \
    const int tid = threadIdx.x;                                              \
    const int expert_id = topk_indices[krank];                                \
    const char* __restrict__ A =                                              \
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);                \
    const int groups_per_row = K / 256;                                       \
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;         \
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;    \
    const int boff = tid * 4;                                                  \
    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;             \
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;             \
    const int quads = groups_per_row >> 2;                                    \
    const int tail = groups_per_row & 3;                                      \
    for (int q = 0; q < quads; q++) {                                         \
        const int g = q << 2;                                                  \
        const int base0 = g * 256 + tid * 8;                                  \
        const int gpf = g + (PF_DIST_GROUPS);                                 \
        if (gpf < groups_per_row) {                                           \
            __builtin_amdgcn_s_prefetch_data(                                 \
                (const void*)(gate_ptr + (long long)gpf * 136), (PF_LEN));    \
            __builtin_amdgcn_s_prefetch_data(                                 \
                (const void*)(up_ptr + (long long)gpf * 136), (PF_LEN));      \
        }                                                                     \
        { LOAD_X8(base0);       DOG_X8(gate_ptr + g * 136, gacc0);            \
                                DOG_X8(up_ptr + g * 136, uacc0); }            \
        { LOAD_X8(base0 + 256); DOG_X8(gate_ptr + (g + 1) * 136, gacc1);      \
                                DOG_X8(up_ptr + (g + 1) * 136, uacc1); }      \
        { LOAD_X8(base0 + 512); DOG_X8(gate_ptr + (g + 2) * 136, gacc2);      \
                                DOG_X8(up_ptr + (g + 2) * 136, uacc2); }      \
        { LOAD_X8(base0 + 768); DOG_X8(gate_ptr + (g + 3) * 136, gacc3);      \
                                DOG_X8(up_ptr + (g + 3) * 136, uacc3); }      \
    }                                                                         \
    if (tail >= 1) { const int g = quads << 2; int base = g * 256 + tid * 8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc0);                     \
        DOG_X8(up_ptr + g * 136, uacc0); }                                    \
    if (tail >= 2) { const int g = (quads << 2) + 1; int base = g*256+tid*8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc1);                     \
        DOG_X8(up_ptr + g * 136, uacc1); }                                    \
    if (tail >= 3) { const int g = (quads << 2) + 2; int base = g*256+tid*8;  \
        LOAD_X8(base); DOG_X8(gate_ptr + g * 136, gacc2);                     \
        DOG_X8(up_ptr + g * 136, uacc2); }                                    \
    float gacc = (gacc0 + gacc1) + (gacc2 + gacc3);                           \
    float uacc = (uacc0 + uacc1) + (uacc2 + uacc3);                           \
    for (int offset = 16; offset > 0; offset >>= 1) gacc += __shfl_down(gacc, offset); \
    for (int offset = 16; offset > 0; offset >>= 1) uacc += __shfl_down(uacc, offset); \
    if (tid == 0) { y_gate[krank * mi + row] = gacc; y_up[krank * mi + row] = uacc; } \
}

#define LOAD_X8(b) \
    const float x0 = x[(b) + 0]; \
    const float x1 = x[(b) + 1]; \
    const float x2 = x[(b) + 2]; \
    const float x3 = x[(b) + 3]; \
    const float x4 = x[(b) + 4]; \
    const float x5 = x[(b) + 5]; \
    const float x6 = x[(b) + 6]; \
    const float x7 = x[(b) + 7]

#define DOG_X8(gp, a) do { \
    float sc = __builtin_bit_cast(float, *(const unsigned int*)(gp)); \
    float zp = __builtin_bit_cast(float, *(const unsigned int*)((gp) + 4)); \
    unsigned int pk = *(const unsigned int*)((gp) + 8 + boff); \
    (a) += (sc * (float)((pk) & 0xFu)         + zp) * x0 \
         + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x1 \
         + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x2 \
         + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x3 \
         + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x4 \
         + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x5 \
         + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x6 \
         + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x7; \
} while (0)

// LEN=5 → whole next quad (544 B ≈ 5×128 B lines). gate + up each.
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf1, 4, 5)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf2, 8, 5)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf3, 4, 2)
MOE_GATE_UP_PREFETCH_BODY(gemv_hfq4g256_moe_gate_up_k8_indexed_pf0, 4, 0)

#undef DOG_X8
#undef LOAD_X8
#undef MOE_GATE_UP_PREFETCH_BODY
