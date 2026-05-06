#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// HFQ4-G256 multi-row GEMV with FP16 packed inner loop (gfx1010 experiment).
//
// Same shape as gemv_hfq4g256_multirow.hip (R rows per warp, x-hoist
// amortized across rows) but the inner FMA chain is __hfma2 (v_pk_fma_f16,
// 2 FP16 FMAs/cycle) instead of scalar v_fma_f32. Cross-group accumulation
// stays FP32 for precision (matches gemm_hfq4g256_residual_fp16.hip).
//
// X is pre-converted to FP16 by the caller via the existing convert_f32_to_f16
// kernel (one-shot, amortized across all GEMV rows in the layer). This avoids
// the inline cvt cost that would cancel the FMA savings.
//
// Expected wins on gfx1010 (5700 XT, RDNA1):
//   1. VALU FMA count halved (32 ops → 16 v_pk_fma_f16 per K=64 inner stride)
//   2. VGPR pressure for x storage cut roughly 2× (4 __half2 vs 8 floats per
//      quad-half), potentially raising R=2 occupancy past 18 waves/SIMD
//   3. X bandwidth halved (FP16 reads — but x is small for batch=1 GEMV, so
//      this is a minor effect)
//
// Honest caveat: the FP32 baseline measures at ~88% peak DRAM BW on 5700 XT
// 9B decode (per project_gfx1010_5700xt_validated_2026_05_06.md), so VALU
// reductions may run in the BW shadow without moving wall-clock tok/s. Bench
// will say.
//
// Opt-in via HIPFIRE_GEMV_FP16=1 (gfx1010-only dispatch path).

template<int R>
__device__ __forceinline__ void gemv_hfq4g256_multirow_fp16_body(
    const char* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row_base = blockIdx.x * R;
    if (row_base >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const int boff = tid * 4;

    const char* row_ptrs[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        int row = row_base + r;
        int safe_row = (row < M) ? row : row_base;
        row_ptrs[r] = A + (long long)safe_row * row_stride;
    }

    // FP32 accumulators — cross-group precision matches FP32 baseline.
    float acc[R][4];
    #pragma unroll
    for (int r = 0; r < R; r++) {
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
        acc[r][2] = 0.0f;
        acc[r][3] = 0.0f;
    }

    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    // Dequant 8 nibbles → 4 × __half2 (8 FP16 weights), via two __hfma2.
    #define DEQUANT_H2(pk, sc_h, zp_h, h01, h23, h45, h67)            \
        do {                                                            \
            __half w0 = __float2half((float)((pk      ) & 0xFu));       \
            __half w1 = __float2half((float)((pk >>  4) & 0xFu));       \
            __half w2 = __float2half((float)((pk >>  8) & 0xFu));       \
            __half w3 = __float2half((float)((pk >> 12) & 0xFu));       \
            __half w4 = __float2half((float)((pk >> 16) & 0xFu));       \
            __half w5 = __float2half((float)((pk >> 20) & 0xFu));       \
            __half w6 = __float2half((float)((pk >> 24) & 0xFu));       \
            __half w7 = __float2half((float)((pk >> 28) & 0xFu));       \
            __half2 sc2 = __half2half2(sc_h);                           \
            __half2 zp2 = __half2half2(zp_h);                           \
            h01 = __hfma2(__halves2half2(w0, w1), sc2, zp2);            \
            h23 = __hfma2(__halves2half2(w2, w3), sc2, zp2);            \
            h45 = __hfma2(__halves2half2(w4, w5), sc2, zp2);            \
            h67 = __hfma2(__halves2half2(w6, w7), sc2, zp2);            \
        } while(0)

    // Dot: __hmul2 + 3× __hfma2 packed FP16, then extract to FP32 and add.
    #define DOT_H2_ACC(wh01, wh23, wh45, wh67, xh01, xh23, xh45, xh67, acc) \
        do {                                                                  \
            __half2 s = __hmul2(wh01, xh01);                                  \
            s = __hfma2(wh23, xh23, s);                                       \
            s = __hfma2(wh45, xh45, s);                                       \
            s = __hfma2(wh67, xh67, s);                                       \
            (acc) += __low2float(s) + __high2float(s);                        \
        } while(0)

    // Main loop: 4 groups per iteration. X load amortized across R rows.
    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const int base = g * 256 + tid * 8;

        // Hoist x as 4 × __half2 per K-position (vs 8 floats in FP32 path).
        // VGPR savings = 4 per K-position × 4 positions = 16 VGPRs vs FP32.
        const __half2* xp0 = (const __half2*)(x + base);
        const __half2* xp1 = (const __half2*)(x + base + 256);
        const __half2* xp2 = (const __half2*)(x + base + 512);
        const __half2* xp3 = (const __half2*)(x + base + 768);
        const __half2 xh001 = xp0[0], xh023 = xp0[1], xh045 = xp0[2], xh067 = xp0[3];
        const __half2 xh101 = xp1[0], xh123 = xp1[1], xh145 = xp1[2], xh167 = xp1[3];
        const __half2 xh201 = xp2[0], xh223 = xp2[1], xh245 = xp2[2], xh267 = xp2[3];
        const __half2 xh301 = xp3[0], xh323 = xp3[1], xh345 = xp3[2], xh367 = xp3[3];

        #pragma unroll
        for (int r = 0; r < R; r++) {
            const char* gp0 = row_ptrs[r] + g * 136;
            const char* gp1 = gp0 + 136;
            const char* gp2 = gp1 + 136;
            const char* gp3 = gp2 + 136;

            __half sc0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0)));
            __half zp0_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4)));
            __half sc1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1)));
            __half zp1_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4)));
            __half sc2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2)));
            __half zp2_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4)));
            __half sc3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3)));
            __half zp3_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4)));

            unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
            unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
            unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
            unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

            __half2 wh001, wh023, wh045, wh067;
            __half2 wh101, wh123, wh145, wh167;
            __half2 wh201, wh223, wh245, wh267;
            __half2 wh301, wh323, wh345, wh367;
            DEQUANT_H2(pk0, sc0_h, zp0_h, wh001, wh023, wh045, wh067);
            DEQUANT_H2(pk1, sc1_h, zp1_h, wh101, wh123, wh145, wh167);
            DEQUANT_H2(pk2, sc2_h, zp2_h, wh201, wh223, wh245, wh267);
            DEQUANT_H2(pk3, sc3_h, zp3_h, wh301, wh323, wh345, wh367);

            DOT_H2_ACC(wh001, wh023, wh045, wh067, xh001, xh023, xh045, xh067, acc[r][0]);
            DOT_H2_ACC(wh101, wh123, wh145, wh167, xh101, xh123, xh145, xh167, acc[r][1]);
            DOT_H2_ACC(wh201, wh223, wh245, wh267, xh201, xh223, xh245, xh267, acc[r][2]);
            DOT_H2_ACC(wh301, wh323, wh345, wh367, xh301, xh323, xh345, xh367, acc[r][3]);
        }
    }

    // Tail: per-row, accumulate into acc[r][gi] to preserve 4-way interleaving
    // (same constraint as FP32 multirow — see commit 5302926).
    #define TAIL_BODY(gi)                                                        \
        do {                                                                       \
            const int g = (quads << 2) + (gi);                                     \
            const int base = g * 256 + tid * 8;                                    \
            const __half2* xp = (const __half2*)(x + base);                        \
            const __half2 xh01 = xp[0], xh23 = xp[1], xh45 = xp[2], xh67 = xp[3];  \
            _Pragma("unroll") for (int r = 0; r < R; r++) {                        \
                const char* gp = row_ptrs[r] + g * 136;                            \
                __half sc_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp))); \
                __half zp_h = __float2half(__builtin_bit_cast(float, *(const unsigned int*)(gp + 4))); \
                unsigned int pk = *(const unsigned int*)(gp + 8 + boff);           \
                __half2 wh01, wh23, wh45, wh67;                                    \
                DEQUANT_H2(pk, sc_h, zp_h, wh01, wh23, wh45, wh67);                \
                DOT_H2_ACC(wh01, wh23, wh45, wh67, xh01, xh23, xh45, xh67, acc[r][gi]); \
            }                                                                       \
        } while(0)

    if (tail >= 1) TAIL_BODY(0);
    if (tail >= 2) TAIL_BODY(1);
    if (tail >= 3) TAIL_BODY(2);

    #undef TAIL_BODY
    #undef DEQUANT_H2
    #undef DOT_H2_ACC

    // Per-row warp reduction and write-back (FP32 throughout, matches FP32 path).
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down(s, offset);
        int row = row_base + r;
        if (tid == 0 && row < M) y[row] = s;
    }
}

// ── extern "C" entry points per R ──
// Tighter launch_bounds because the inner-loop VGPR footprint shrinks vs FP32
// (4 __half2 x-pairs replace 8 floats per K-position). Conservative starting
// point: same occupancy ceiling as FP32 — bench will determine if we can
// raise it.

__launch_bounds__(32, 18)
extern "C" __global__ void gemv_hfq4g256_multirow_fp16_r2(
    const char* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_multirow_fp16_body<2>(A, x, y, M, K);
}

__launch_bounds__(32, 18)
extern "C" __global__ void gemv_hfq4g256_multirow_fp16_r4(
    const char* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_multirow_fp16_body<4>(A, x, y, M, K);
}

__launch_bounds__(32, 6)
extern "C" __global__ void gemv_hfq4g256_multirow_fp16_r8(
    const char* __restrict__ A,
    const _Float16* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    gemv_hfq4g256_multirow_fp16_body<8>(A, x, y, M, K);
}
