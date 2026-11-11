#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

// W8 twin of hyper_read_up_wmma_bf16_gfx1201_v2 (hyper_read_up_wmma.gfx1201.hip), the persistent
// W-stationary gfx1201 HC up+mix read tail that is the beta default for low_rank 320:
//   h[br][token][j] = dot(W[br * hidden + j, :], low[token, :]),  W [4 * hidden, 320],
//   mixed[token, j] = bf16r(sum_br bf16r(sigmoid(bf16r(h)) * normalized) / 4)   (same epilogue).
// The weight comes from the PLANAR W8 format instead of BF16:
//   q : i8  [4 * hidden][320] row-major,   d : f16 [4 * hidden][10] (one scale per 32-K block),
//   weight value = (float)q * (float)d.
//
// What changed against the BF16 v2 kernel:
//   * Staging: the block's 64 W rows (4 branches x 16 columns) are staged into LDS as exact BF16
//     in the same [row][K + 8] layout (an int8 is exactly representable in BF16: the F32 bits of
//     (float)q >> 16), and the block's 64 x 10 scales as F16 into a [block][64 rows] table
//     (1280 B).  Dynamic LDS: 64 * (320 + 8) * 2 + 16384 + 1280 = 59648 bytes.
//   * Operand roles are swapped back to the baseline's orientation: X (tokens) is the WMMA A
//     operand and W the B operand, so the accumulator element is C[token t0 + 8 * (l >> 4) + i]
//     [column j0 + (l & 15)].  Every accumulator element of a lane then shares ONE output
//     column per branch, i.e. ONE scale per lane per branch per 32-K block (4 scalars),
//     instead of 8 per branch (the original's v2 orientation has the 8 j of a lane
//     in the accumulator elements).  Fragment layouts of A and B are the same
//     (row l & 15, K elements [8 * (l >> 4), +8) of each 16-substep), so the operand loads
//     are unchanged.
//   * The scale is NEVER folded into an operand.  Per 32-K block (two 16-substeps) a block
//     accumulator starts at zero, takes the block's two WMMAs, and is folded into the running
//     F32 accumulator with the block's scale:
//         acc[br][i] = fmaf(d[br][block], blk[br][i], acc[br][i])
//     The fold of block b is issued after the first substep's WMMAs of block b + 1 (the block
//     accumulators ping-pong), so it never waits on the WMMA it just issued.
//   * Epilogue: same expressions, same LUT gate (built at block start with the baseline
//     expression), but the lane holds one column of eight tokens, so `normalized` is read as 32
//     scalar F16 loads (issued mid-loop) and each output is a 4-byte store (16 lanes = 64
//     contiguous bytes of one token row).  A lane with a NaN logit or a non-finite normalized
//     value recomputes its 8 outputs with the baseline expression (NaN payloads not pinned).
//   * Not byte-identical to the BF16 kernel (different weights).  `X` is packed BF16 `low`,
//     `normalized` the F16 copy (hyper_norm_f16), exactly as the BF16 route.
//
// Grid [hidden / 16, G], G <= ceil(N / 16) (the host uses hc_up_v2_groups); block 512 threads;
// dynamic LDS 59648 bytes.  hidden % 16 == 0, K == 320, 16-byte aligned q / X.

#define HRW_K 320
#define HRW_KSTEPS (HRW_K / 16)
#define HRW_NB (HRW_K / 32)
#define HRW_XB 4            // X substeps prefetched per group (even: whole 32-K blocks)
#define HRW_WAVES 16
#define HRW_LUT_LO 0x3b80u
#define HRW_LUT_HI 0x42b2u
#define HRW_LUT_SPAN 2048u
#define HRW_LDW (HRW_K + 8)
#define HRW_SC_OFF (64 * HRW_LDW * 2 + 2 * HRW_LUT_SPAN * 4)   // byte offset of the scale table
#define HRW_LDS_BYTES (HRW_SC_OFF + HRW_NB * 64 * 2)

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "hyper_read_up_wmma_w8.gfx1201.hip: gfx1201 only"
#endif

static_assert(HRW_KSTEPS % HRW_XB == 0 && HRW_XB % 2 == 0, "X groups hold whole scale blocks");
static_assert(HRW_LDS_BYTES <= 65536, "LDS");

typedef short __attribute__((ext_vector_type(8))) hrw_bf16x8_t;
typedef float __attribute__((ext_vector_type(8))) hrw_f8_t;

// tensor_ops.hip's bf16_to_f32(bf16_round(v)).
static __device__ __forceinline__ float hrw_bf16r(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}

static __device__ __forceinline__ float hrw_bf16r_fin(float v) {
    unsigned u = __float_as_uint(v);
    u = u + 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// Two signed ints (|q| <= 128) -> two exact BF16 in one word: the BF16 of an integer of at
// most 8 significant bits is the top half of its F32 bits.
static __device__ __forceinline__ unsigned hrw_bf16x2(int lo, int hi) {
    const unsigned a = __float_as_uint((float)lo);
    const unsigned b = __float_as_uint((float)hi);
    return __builtin_amdgcn_perm(b, a, 0x07060302u);   // { a.b2, a.b3, b.b2, b.b3 }
}

// 8 signed bytes (two words) -> 8 exact BF16 (one uint4).
static __device__ __forceinline__ uint4 hrw_q8x8_to_bf16(unsigned x0, unsigned x1) {
    uint4 o;
    o.x = hrw_bf16x2((int)(signed char)(x0), (int)(signed char)(x0 >> 8));
    o.y = hrw_bf16x2((int)(signed char)(x0 >> 16), (int)(signed char)(x0 >> 24));
    o.z = hrw_bf16x2((int)(signed char)(x1), (int)(signed char)(x1 >> 8));
    o.w = hrw_bf16x2((int)(signed char)(x1 >> 16), (int)(signed char)(x1 >> 24));
    return o;
}

#define HRW_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c)

extern "C" __global__ __launch_bounds__(HRW_WAVES * 32) void hyper_read_up_wmma_w8_gfx1201_v2(
    const signed char* __restrict__ Wq,
    const _Float16* __restrict__ Wd,
    const unsigned short* __restrict__ X,
    const _Float16* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    (void)K;   // host guarantees K == HRW_K
    extern __shared__ unsigned short hrw_w[];
    unsigned* const tab = (unsigned*)((char*)hrw_w + 64 * HRW_LDW * 2);
    unsigned short* const sc = (unsigned short*)((char*)hrw_w + HRW_SC_OFF);   // [block][64 rows]
    const int j0 = blockIdx.x * 16;
    const int tid = threadIdx.x;
    // W: 64 rows x 20 chunks of 16 int8 -> 2 x uint4 of BF16 each.
    for (int v = tid; v < 64 * (HRW_K / 16); v += HRW_WAVES * 32) {
        const int r = v / (HRW_K / 16), c = v % (HRW_K / 16);
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        const uint4 raw = *(const uint4*)(Wq + (size_t)row * HRW_K + c * 16);
        *(uint4*)(hrw_w + r * HRW_LDW + c * 16) = hrw_q8x8_to_bf16(raw.x, raw.y);
        *(uint4*)(hrw_w + r * HRW_LDW + c * 16 + 8) = hrw_q8x8_to_bf16(raw.z, raw.w);
    }
    // Scales: 64 rows x 10 F16 = 5 words per row -> sc[block][row].
    if (tid < 64 * (HRW_NB / 2)) {
        const int r = tid / (HRW_NB / 2), w = tid % (HRW_NB / 2);
        const int row = (r >> 4) * hidden + j0 + (r & 15);
        const unsigned word = ((const unsigned*)Wd)[(size_t)row * (HRW_NB / 2) + w];
        sc[(2 * w) * 64 + r] = (unsigned short)(word & 0xffffu);
        sc[(2 * w + 1) * 64 + r] = (unsigned short)(word >> 16);
    }
    for (int idx = tid; idx < 2 * (int)HRW_LUT_SPAN; idx += HRW_WAVES * 32) {
        const unsigned sg = (unsigned)idx / HRW_LUT_SPAN;
        const unsigned e = min((unsigned)idx % HRW_LUT_SPAN, HRW_LUT_HI - HRW_LUT_LO);
        const float x = __uint_as_float(((sg << 15) | (HRW_LUT_LO + e)) << 16);
        tab[idx] = __float_as_uint(hrw_bf16r(1.0f / (1.0f + expf(-x))));
    }
    __syncthreads();
    const int wave = tid >> 5, lane = tid & 31, l16 = lane & 15, kh = lane >> 4;
    const int wide = 4 * hidden;
    const int ntiles = (N + 15) >> 4;
    const int tpg = (ntiles + (int)gridDim.y - 1) / (int)gridDim.y;
    const int tbeg = (int)blockIdx.y * tpg;
    const int tend = min(ntiles, tbeg + tpg);
    const int j = j0 + l16;
    const unsigned short* const wb = hrw_w + l16 * HRW_LDW + 8 * kh;
    const _Float16* const scl = (const _Float16*)sc + l16;
    const hrw_f8_t zero8 = hrw_f8_t{0, 0, 0, 0, 0, 0, 0, 0};
    for (int tile = tbeg + wave; tile < tend; tile += HRW_WAVES) {
        const int tok0 = tile * 16;
        const int tokc = min(tok0 + l16, N - 1);     // the X row this lane feeds the A operand
        const unsigned short* const xr = X + (size_t)tokc * HRW_K + 8 * kh;
        hrw_f8_t acc[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acc[br] = zero8;
        uint4 xcur[HRW_XB];
#pragma unroll
        for (int s = 0; s < HRW_XB; s++) xcur[s] = *(const uint4*)(xr + 16 * s);
        hrw_bf16x8_t acur[4];
#pragma unroll
        for (int br = 0; br < 4; br++) acur[br] = *(const hrw_bf16x8_t*)(wb + br * 16 * HRW_LDW);
        hrw_f8_t blkp[4], blkc[4];
        float dp[4], dc[4];
        _Float16 nv[4][8];
#pragma unroll
        for (int g = 0; g < HRW_KSTEPS / HRW_XB; g++) {
            __builtin_amdgcn_sched_barrier(0);
            uint4 xnext[HRW_XB];
            if (g + 1 < HRW_KSTEPS / HRW_XB) {
#pragma unroll
                for (int s = 0; s < HRW_XB; s++)
                    xnext[s] = *(const uint4*)(xr + 16 * ((g + 1) * HRW_XB + s));
            }
            if (g == 3) {
                // The epilogue's normalized values: column j of the lane's eight tokens, in
                // flight over the last two groups' WMMA chains.
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const int token = min(tok0 + 8 * kh + i, N - 1);
#pragma unroll
                    for (int br = 0; br < 4; br++)
                        nv[br][i] = normalized[(size_t)token * wide + br * hidden + j];
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int s = 0; s < HRW_XB; s++) {
                const int ks = g * HRW_XB + s;
                const int kb = ks >> 1;
                hrw_bf16x8_t anext[4];
                if (ks + 1 < HRW_KSTEPS) {
#pragma unroll
                    for (int br = 0; br < 4; br++)
                        anext[br] = *(const hrw_bf16x8_t*)(wb + br * 16 * HRW_LDW + 16 * (ks + 1));
                }
                if ((ks & 1) == 0) {
#pragma unroll
                    for (int br = 0; br < 4; br++) dc[br] = (float)scl[kb * 64 + br * 16];
                }
                __builtin_amdgcn_sched_barrier(0);
                const hrw_bf16x8_t a = __builtin_bit_cast(hrw_bf16x8_t, xcur[s]);
                if ((ks & 1) == 0) {
#pragma unroll
                    for (int br = 0; br < 4; br++) blkc[br] = HRW_WMMA(a, acur[br], zero8);
                    if (kb > 0) {
                        // fold the previous block (its second WMMAs were issued a full
                        // substep ago)
#pragma unroll
                        for (int br = 0; br < 4; br++)
#pragma unroll
                            for (int i = 0; i < 8; i++)
                                acc[br][i] = __builtin_fmaf(dp[br], blkp[br][i], acc[br][i]);
                    }
                } else {
#pragma unroll
                    for (int br = 0; br < 4; br++) blkc[br] = HRW_WMMA(a, acur[br], blkc[br]);
#pragma unroll
                    for (int br = 0; br < 4; br++) { blkp[br] = blkc[br]; dp[br] = dc[br]; }
                }
                if (ks + 1 < HRW_KSTEPS) {
#pragma unroll
                    for (int br = 0; br < 4; br++) acur[br] = anext[br];
                }
            }
            if (g + 1 < HRW_KSTEPS / HRW_XB) {
#pragma unroll
                for (int s = 0; s < HRW_XB; s++) xcur[s] = xnext[s];
            }
        }
        // the last block
#pragma unroll
        for (int br = 0; br < 4; br++)
#pragma unroll
            for (int i = 0; i < 8; i++)
                acc[br][i] = __builtin_fmaf(dp[br], blkp[br][i], acc[br][i]);

        // Epilogue: hyper_read_projected_f32's mix, branch order from +0.0.  The gate of a finite
        // (or infinite) logit is its LUT entry; a NaN logit or a non-finite normalized value sends
        // the lane to the baseline expression.
        float out[8];
        bool bad = false;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float value = 0.0f;
#pragma unroll
            for (int br = 0; br < 4; br++) {
                const float a = acc[br][i];
                bad |= (a != a);
                const unsigned u = __float_as_uint(a);
                const unsigned r = u + 0x7fffu + ((u >> 16) & 1u);
                const unsigned mag = min(max((r >> 16) & 0x7fffu, HRW_LUT_LO), HRW_LUT_HI);
                const float gate = __uint_as_float(tab[(r >> 31) * HRW_LUT_SPAN + (mag - HRW_LUT_LO)]);
                const float normed = (float)nv[br][i];
                bad |= !(fabsf(normed) <= 65504.0f);
                value += hrw_bf16r_fin(gate * normed);
            }
            out[i] = hrw_bf16r_fin(value * 0.25f);
        }
        if (bad) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                float value = 0.0f;
#pragma unroll
                for (int br = 0; br < 4; br++) {
                    const float gate_input = hrw_bf16r(acc[br][i]);
                    const float gate = hrw_bf16r(1.0f / (1.0f + expf(-gate_input)));
                    const float normed = (float)nv[br][i];
                    value += hrw_bf16r(gate * normed);
                }
                out[i] = hrw_bf16r(value / 4.0f);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int token = tok0 + 8 * kh + i;
            if (token < N) mixed[(size_t)token * hidden + j] = out[i];
        }
    }
}
