#define HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE 1
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Daniil Markevich
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared int4 activation block + wave32 MSE-clip quant recipe for the
// iu4-direct MMQ path. Concatenated by the host JIT into:
//   - gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip (standalone quantize_int4_mmq_ds128)
//   - fused RMSNorm/FWHT and SwiGLU/FWHT IU4 producer sidecars (C2)
//
// Do not edit a second copy of the amax / 8-candidate MSE / rintf+clamp /
// integer-sum path. Every consumer of this fragment must remain bit-identical
// on the 72-byte block_i4_128 layout.

#ifndef HIPFIRE_BLOCK_I4_128_QUANT_HIP
#define HIPFIRE_BLOCK_I4_128_QUANT_HIP

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

// int4 activation block: one 128-K group for one token (column).
// qs: 64 B, even K low nibble / odd K high nibble, two's complement.
// d: per-128 scale (MSE-clip grid argmin). s: exact int sum of the 128 x_q.
struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};
static_assert(sizeof(block_i4_128) == 72, "bad block size");

// Wave32-synchronous quant of one 128-element group.
// Lane l owns elements 4l..4l+3 in xv[4]. Reductions use offsets 16..1
// (wave64-safe width-32 shuffles). Writes qs for every lane; lane 0 of the
// group publishes d and s. Arithmetic matches the original standalone body
// (amax → 8 MSE candidates with strict-`<` → rintf/clamp → exact int sum).
__device__ __forceinline__ void quantize_block_i4_128_wave(
    const float xv[4],
    block_i4_128* __restrict__ out,
    int lane
) {
    float amax = fmaxf(fmaxf(fabsf(xv[0]), fabsf(xv[1])), fmaxf(fabsf(xv[2]), fabsf(xv[3])));
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    }

    float best_d = 1.0f;
    if (amax != 0.0f) {
        float best_mse = 1.0e30f;
        for (int j = 0; j < 8; ++j) {
            const float dj = (amax / 7.0f) * 0.5f * (1.0f + (float)j / 7.0f);
            float mse = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                float q = rintf(xv[e] / dj);
                q = fminf(fmaxf(q, -8.0f), 7.0f);
                const float err = __fmaf_rn(-q, dj, xv[e]);
                mse = __fmaf_rn(err, err, mse);
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                mse += __shfl_xor(mse, offset, 32);
            }
            if (mse < best_mse) {
                best_mse = mse;
                best_d = dj;
            }
        }
    }

    int q4[4];
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        float q = (amax == 0.0f) ? 0.0f : rintf(xv[e] / best_d);
        q = fminf(fmaxf(q, -8.0f), 7.0f);
        q4[e] = (int)q;
    }
    int s = q4[0] + q4[1] + q4[2] + q4[3];
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_xor(s, offset, 32);
    }

    unsigned char* qb = out->qs;
    qb[lane * 2 + 0] = (unsigned char)((q4[0] & 15) | ((q4[1] & 15) << 4));
    qb[lane * 2 + 1] = (unsigned char)((q4[2] & 15) | ((q4[3] & 15) << 4));
    if (lane == 0) {
        out->d = best_d;
        out->s = s;
    }
}

// Remap producer-final 8 floats/lane into the standalone quant ownership
// (lane l owns 4l..4l+3 of each 128-half) and emit two block_i4_128 rows.
// Producer lane p holds values 8p..8p+7. For half h∈{0,1}:
//   source lane = 16*h + (l >> 1); even l → first float4, odd l → second.
// Sidecar index: (2*group + h)*N + token.
__device__ __forceinline__ void emit_iu4_sidecar_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    int group,
    int token,
    int N,
    int lane
) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int src = 16 * h + (lane >> 1);
        float s0 = __shfl(f0, src, 32);
        float s1 = __shfl(f1, src, 32);
        float s2 = __shfl(f2, src, 32);
        float s3 = __shfl(f3, src, 32);
        float s4 = __shfl(f4, src, 32);
        float s5 = __shfl(f5, src, 32);
        float s6 = __shfl(f6, src, 32);
        float s7 = __shfl(f7, src, 32);
        float xv[4];
        if ((lane & 1) == 0) {
            xv[0] = s0; xv[1] = s1; xv[2] = s2; xv[3] = s3;
        } else {
            xv[0] = s4; xv[1] = s5; xv[2] = s6; xv[3] = s7;
        }
        block_i4_128* out = Y + (long long)(2 * group + h) * N + token;
        quantize_block_i4_128_wave(xv, out, lane);
    }
}

// Standalone quantizer: same (X, Y, K, N) grid contract as
// quantize_q8_1_mmq_ds4_x128 (block (256,1,1), each thread 4 floats,
// grid [(K+1023)/1024, N]). One wave32 covers exactly one 128-K group.
#ifndef HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE
extern "C" __global__ void quantize_int4_mmq_ds128(
    const float* __restrict__ X,
    block_i4_128* __restrict__ Y,
    int K,
    int N
) {
    const int token = blockIdx.y;
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = t * 4;
    if (token >= N || i0 >= K) return;
    const int lane = threadIdx.x & 31;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};

    const int kblock = i0 / 128;
    block_i4_128* out = Y + (long long)kblock * N + token;
    quantize_block_i4_128_wave(xv, out, lane);
}
#endif // HIPFIRE_BLOCK_I4_128_QUANT_NO_STANDALONE

#endif // HIPFIRE_BLOCK_I4_128_QUANT_HIP

// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2 up projection with the exact SwiGLU/AWQ/FWHT/int4 producer
// epilogue for gfx1201. The gate projection remains a separate launch.
//
// Frozen ABI:
//   (Wup, Xin, Gate, Awq, S1, S2, Down, I, K, N)
// Wup is the 136-byte MQ4V2 group format, Xin is block_i4_128 [K/128,N],
// Gate is F32 [N,I], and Down is block_i4_128 [I/128,N].
//
// Geometry: one 256-feature x 64-token tile per 256-thread workgroup.
// Wave w owns features 32w..32w+31 for all 64 tokens during the GEMM.
// Its 64 FP32 accumulators preserve the incumbent iu4 K32 grouping and
// IU4_FOLD_RN sequence. After the K loop, the dead 25,600-byte compute arena
// is reused as an eight-token x 256-feature F32 epoch. Each epoch changes
// ownership to one producer wave per token, then executes the incumbent
// fused_silu_mul_mq_rotate_awq_i4_gfx12 arithmetic and shared int4 recipe.
//
// LDS (25,600 bytes):
//   A0 64x32 B @ 0; W0 256x32 B; W1 256x32 B;
//   DS0/DS1 64x8 B; SZ0/SZ1 256x8 B; A1 64x32 B.
// The A/W sizes denote packed-nibble bytes for one 64-K slab. The producer
// overlay occupies only the first 8x256x4 = 8,192 bytes.
//
// Grid [ceil(I/256), ceil(N/64)], block [256,1,1], dynamic LDS 25,600.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1201__)

// block_i4_128 plus emit_iu4_sidecar_from_producer8 are supplied by the host
// JIT prelude (kernels/src/block_i4_128_quant.hip).

// Workgroup tile geometry (tokens x rows x staged-K, waves). Overridable
// for sweeps; host + oracle must match.
#define IU4_BM 64
#define IU4_BN 256
#ifndef IU4_BK
#define IU4_BK 64
#endif
#ifndef IU4_WAVES
#define IU4_WAVES 8
#endif
#define IU4_NTHREADS (IU4_WAVES * 32)
// Fragment-order slab: each 64-K slab (IU4_BK nibbles/row) holds 8 row/token
// groups of 16 x 2 K32 steps = 16 fragments of 256 contiguous bytes
// (32 lanes x 8 B, lane L at frag*256 + L*8). Same 4096 data bytes as the
// old [128][32] rows, no pad: contiguous-lane b64 reads sit at the b64
// width floor (see bank enumeration above).
#define IU4_FRAG_BYTES 256
#define IU4_A_FRAGS ((IU4_BM / 16) * (IU4_BK / 32))
#define IU4_W_FRAGS ((IU4_BN / 16) * (IU4_BK / 32))
#define IU4_A_BYTES (IU4_A_FRAGS * IU4_FRAG_BYTES)
#define IU4_W_BYTES (IU4_W_FRAGS * IU4_FRAG_BYTES)
#define IU4_A_OFF 0
#define IU4_W0_OFF (IU4_A_OFF + IU4_A_BYTES)
#define IU4_W1_OFF (IU4_W0_OFF + IU4_W_BYTES)
#define IU4_DS_OFF (IU4_W1_OFF + IU4_W_BYTES)
#define IU4_DS_BYTES (IU4_BM * 8)
#define IU4_DS1_OFF (IU4_DS_OFF + IU4_DS_BYTES)
#define IU4_SZ0_OFF (IU4_DS1_OFF + IU4_DS_BYTES)
#define IU4_SZ_BYTES (IU4_BN * 8)
#define IU4_SZ1_OFF (IU4_SZ0_OFF + IU4_SZ_BYTES)
#define IU4_A1_OFF (IU4_SZ1_OFF + IU4_SZ_BYTES)
#define IU4_LDS_BYTES (IU4_A1_OFF + IU4_A_BYTES)
_Static_assert(IU4_LDS_BYTES == 25600, "epilogue tile must use 25,600 B LDS");

typedef int8_t  __attribute__((ext_vector_type(8)))  int8x8_t;
typedef int32_t __attribute__((ext_vector_type(2)))  int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  float8_t;
// 8-byte staging word: the width ceiling for A/W traffic (see the staging-map
// note at the K-loop top). Ext-vector so global/LDS accesses codegen to
// single b64 instructions with elementwise branch-free selects.
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;
// IU4_FOLD_RN — W4.0/LF.0 bit-exact fold pin, shared with the gfx11 iu4
// kernel. Every output element folds one int32 WMMA result C through EXACTLY
// this DAG:
//   t1   = RN(sc * d)
//   p    = RN(t1 * float(C))
//   t2   = RN(zp * d)
//   term = RN(t2 * float(s) + p)
//   sum  = RN(sum + term)
#define IU4_FOLD_RN(sum_lvalue, sc, zp, d, acc_i32, s_i32) \
    do { \
        const float iu4_fold_d_ = (d); \
        const float iu4_fold_t1_ = __fmul_rn((sc), iu4_fold_d_); \
        const float iu4_fold_p_ = __fmul_rn(iu4_fold_t1_, (float)(acc_i32)); \
        const float iu4_fold_t2_ = __fmul_rn((zp), iu4_fold_d_); \
        const float iu4_fold_term_ = __fmaf_rn(iu4_fold_t2_, (float)(s_i32), iu4_fold_p_); \
        (sum_lvalue) = __fadd_rn((sum_lvalue), iu4_fold_term_); \
    } while (0)
#if defined(__gfx1201__)
// IU4_FOLD_RN_PAIR — gfx1201 VOPD fold pairing (exact-arithmetic form 1).
// Two independent j-components through the pinned IU4_FOLD_RN DAG with
// stages interleaved pairwise (t1_0,t1_1 then p_0,p_1 ...) so the backend
// can form V_DUAL_MUL_F32 / V_DUAL_FMAC_F32 / V_DUAL_ADD_F32 packets.
// Bit-exact by construction: the same RN multiply/FMA/add nodes in the
// same per-component order, no contraction, no precision change; i32->f32
// conversions stay scalar (no dual conversion form exists on gfx1201).
// By-value sums + float2 return (ext_vector elements cannot bind to
// float&); forceinline keeps acc traffic direct. The gfx1200 path keeps
// the scalar IU4_FOLD_RN loop below untouched.
static __device__ __forceinline__ float2 iu4_fold_rn_pair(
    float sum0, float sum1,
    float sc0, float sc1, float zp0, float zp1,
    float d, int c0, int c1, float sf) {
    const float t1_0 = __fmul_rn(sc0, d);
    const float t1_1 = __fmul_rn(sc1, d);
    const float p_0 = __fmul_rn(t1_0, (float)c0);
    const float p_1 = __fmul_rn(t1_1, (float)c1);
    const float t2_0 = __fmul_rn(zp0, d);
    const float t2_1 = __fmul_rn(zp1, d);
    const float term_0 = __fmaf_rn(t2_0, sf, p_0);
    const float term_1 = __fmaf_rn(t2_1, sf, p_1);
    float2 out;
    out.x = __fadd_rn(sum0, term_0);
    out.y = __fadd_rn(sum1, term_1);
    return out;
}
#endif

// One K32-sub-block compute bundle: 2 W-frags + 4 A-frags +
// 8 WMMAs (2 row-groups x 4 column blocks), then a scheduling fence.
// SB is a template constant (folded into LDS immediates); FIRST picks
// the shared zero C vector for each chain's first step (avoids 64 separate
// Compute reads slab sb of A from the single A plane and
// of W from Wn (n = sb); the caller re-points w_lds per phase. Both W planes
// share the fragment-order layout, so the hoisted frA/frW offsets apply to
// either. Each fragment read is one ds_read_b64 at frag*256 + lane*8.
// K-step VALU: ds_reads + WMMAs only.
template <int SB, bool FIRST>
static __device__ __forceinline__ void iu4_bundle(
    const char* __restrict__ w_lds,
    const char* __restrict__ a_lds,
    const unsigned (&frW)[2][2],
    const unsigned (&frA)[4][2],
    int32x8_t cacc[4][2],
    const int32x8_t& zc
) {
    const int32x2_t af0 = *(const int32x2_t*)(w_lds + frW[0][SB]);
    const int32x2_t af1 = *(const int32x2_t*)(w_lds + frW[1][SB]);
    const int32x2_t bf0 = *(const int32x2_t*)(a_lds + frA[0][SB]);
    const int32x2_t bf1 = *(const int32x2_t*)(a_lds + frA[1][SB]);
    const int32x2_t bf2 = *(const int32x2_t*)(a_lds + frA[2][SB]);
    const int32x2_t bf3 = *(const int32x2_t*)(a_lds + frA[3][SB]);
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const int32x2_t a = (rg == 0) ? af0 : af1;
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int32x2_t b = (nb == 0) ? bf0 : (nb == 1) ? bf1 : (nb == 2) ? bf2 : bf3;
            if constexpr (FIRST) {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a, true, b, zc, false
                );
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    false, a, true, b, cacc[nb][rg], false
                );
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

static __device__ __forceinline__ void gemm_up_silu_mq4g256v2_iu4_gfx1201_body(
    const char*           __restrict__ Wup,
    const block_i4_128*   __restrict__ Xin,
    const float*          __restrict__ Gate,
    const float*          __restrict__ Awq,
    const float*          __restrict__ S1,
    const float*          __restrict__ S2,
    block_i4_128*         __restrict__ Down,
    int I, int K, int N
) {
    const char* __restrict__ A = Wup;
    const block_i4_128* __restrict__ Xq = Xin;
    const int M = I;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rs = blockIdx.x * IU4_BN;
    const int bs = blockIdx.y * IU4_BM;

    if (rs >= M || bs >= N) return;

    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int wave_row0 = wave * 32;
    const int wave_tok0 = 0;

    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];
    char* __restrict__ a_lds = LDS + IU4_A_OFF;
    char* __restrict__ w_lds = LDS + IU4_W0_OFF;

    unsigned frA[4][2];
    unsigned frW[2][2];
    const unsigned lane8 = (unsigned)lane * 8u;
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frA[nb][sb] = ((unsigned)nb * 2u + (unsigned)sb)
                * (unsigned)IU4_FRAG_BYTES + lane8;
        }
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const unsigned g8w = (unsigned)(wave * 2 + rg);
        #pragma unroll
        for (int sb = 0; sb < 2; sb++) {
            frW[rg][sb] = (g8w * 2u + (unsigned)sb)
                * (unsigned)IU4_FRAG_BYTES + lane8;
        }
    }

    // acc[nb][rg][j] owns token nb*16+m_lane and feature
    // wave*32+rg*16+8*k_grp+j within this workgroup tile.
    float8_t acc[4][2];

    // One 64-token activation slab: 256 lanes x one b64.
    const int tok_r = bs + (tid >> 2);
    const int tok_c = (tok_r < N) ? tok_r : (N - 1);
    const unsigned st_aok = (tok_r < N) ? 1u : 0u;
    const unsigned st_avoff = (unsigned)tok_c * (unsigned)sizeof(block_i4_128)
        + (unsigned)(tid & 3) * 8u;
    const unsigned st_aldsoff =
        (((unsigned)(tid >> 2) >> 4) * 2u + ((unsigned)(tid & 3) >> 1))
            * (unsigned)IU4_FRAG_BYTES
        + (((unsigned)(tid >> 2) & 15u) + ((unsigned)(tid & 3) & 1u) * 16u) * 8u;

    // One 256-feature weight slab: 256 lanes x four b64 rounds.
    unsigned st_wvoff[4];
    unsigned st_wldsoff[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int grow_r = rs + r * 64 + (tid >> 2);
        const int src_c = (grow_r < M) ? grow_r : (M - 1);
        st_wvoff[r] = (unsigned)src_c * (unsigned)groups_per_row * 136u
            + (unsigned)(tid & 3) * 8u;
        const unsigned R = (unsigned)(r * 64 + (tid >> 2));
        const unsigned q = (unsigned)(tid & 3);
        st_wldsoff[r] = ((R >> 4) * 2u + (q >> 1))
                * (unsigned)IU4_FRAG_BYTES
            + ((R & 15u) + (q & 1u) * 16u) * 8u;
    }

    const u32x2_t z64 = {0u, 0u};
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (float8_t){
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
            };
        }
    }
    const int32x8_t zc = (int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
    int32x8_t cacc[4][2];
    u32x2_t A_pf;
    u32x2_t W_pf[4];

    // Prologue: metadata and the first packed K64 slab.
    for (int i = tid; i < IU4_BM * 2; i += IU4_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N)
            ? ((const unsigned*)(Xq + tok))[w]
            : 0u;
        ((unsigned*)(LDS + IU4_DS_OFF + tok_idx * 8))[w] = dw;
    }
    for (int i = tid; i < IU4_BN * 2; i += IU4_NTHREADS) {
        const int row_idx = i >> 1;
        const int w = i & 1;
        const int grow = rs + row_idx;
        const int src_row = (grow < M) ? grow : (M - 1);
        const unsigned hw =
            *(const unsigned*)(A + (long long)src_row * groups_per_row * 136);
        const float f = (w == 0)
            ? __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)))
            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
        ((float*)(LDS + IU4_SZ0_OFF + row_idx * 8))[w] =
            (grow < M) ? f : 0.0f;
    }
    {
        const char* __restrict__ xq0 = (const char*)Xq + 8;
        const char* __restrict__ w0 = A + 8;
        const u32x2_t av = *(const u32x2_t*)(xq0 + st_avoff);
        *(u32x2_t*)(LDS + IU4_A_OFF + st_aldsoff) =
            st_aok ? av : z64;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            *(u32x2_t*)(LDS + IU4_W0_OFF + st_wldsoff[r]) =
                *(const u32x2_t*)(w0 + st_wvoff[r]);
        }
    }
    __syncthreads();

    for (int g = 0; g < groups_per_row; g++) {
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; h++) {
            const int kb = 2 * g + h;
            const bool last = h == 1 && g + 1 == groups_per_row;
            char* __restrict__ ds_lds =
                LDS + ((h == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
            char* __restrict__ sz_lds =
                LDS + ((h == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);

            // Prefetch current slab one while slot zero computes.
            const char* __restrict__ xq1 =
                (const char*)(Xq + (long long)kb * N) + 8 + 32;
            const char* __restrict__ w1 =
                A + (long long)g * 136 + 8 + (long long)h * 64 + 32;
            A_pf = *(const u32x2_t*)(xq1 + st_avoff);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                W_pf[r] = *(const u32x2_t*)(w1 + st_wvoff[r]);
            }

            a_lds = LDS + IU4_A_OFF;
            w_lds = LDS + IU4_W0_OFF;
            iu4_bundle<0, true>(w_lds, a_lds, frW, frA, cacc, zc);
            iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);

            asm volatile("" : "+v"(A_pf[0]), "+v"(A_pf[1]));
            *(u32x2_t*)(LDS + IU4_A1_OFF + st_aldsoff) =
                st_aok ? A_pf : z64;
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                *(u32x2_t*)(LDS + IU4_W1_OFF + st_wldsoff[r]) = W_pf[r];
            }
            __syncthreads();

            // Issue the next K128 block before computing/folding slot one.
            if (!last) {
                const int kb_nx = kb + 1;
                const char* __restrict__ xq_nx =
                    (const char*)(Xq + (long long)kb_nx * N) + 8;
                A_pf = *(const u32x2_t*)(xq_nx + st_avoff);
                const int g_nx = g + h;
                const int h_nx = h ^ 1;
                const char* __restrict__ w_nx =
                    A + (long long)g_nx * 136 + 8 + (long long)h_nx * 64;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    W_pf[r] = *(const u32x2_t*)(w_nx + st_wvoff[r]);
                }
            }

            a_lds = LDS + IU4_A1_OFF;
            w_lds = LDS + IU4_W1_OFF;
            iu4_bundle<0, false>(w_lds, a_lds, frW, frA, cacc, zc);
            iu4_bundle<1, false>(w_lds, a_lds, frW, frA, cacc, zc);

            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                float sc_row[8];
                float zp_row[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const unsigned jbase =
                        (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + j) * 8u;
                    sc_row[j] = *(const float*)(sz_lds + jbase);
                    zp_row[j] = *(const float*)(sz_lds + jbase + 4u);
                }
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    int* acc_i = (int*)&cacc[nb][rg];
                    const unsigned tbase =
                        (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                    const float d_x = *(const float*)(ds_lds + tbase);
                    const int s_x = *(const int*)(ds_lds + tbase + 4u);
                    const float sf_x = (float)s_x;
                    #pragma unroll
                    for (int j = 0; j < 8; j += 2) {
                        const float2 pair = iu4_fold_rn_pair(
                            acc[nb][rg][j], acc[nb][rg][j + 1],
                            sc_row[j], sc_row[j + 1],
                            zp_row[j], zp_row[j + 1],
                            d_x, acc_i[j], acc_i[j + 1], sf_x);
                        acc[nb][rg][j] = pair.x;
                        acc[nb][rg][j + 1] = pair.y;
                    }
                }
            }

            if (!last) {
                *(u32x2_t*)(LDS + IU4_A_OFF + st_aldsoff) =
                    st_aok ? A_pf : z64;
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    *(u32x2_t*)(LDS + IU4_W0_OFF + st_wldsoff[r]) =
                        W_pf[r];
                }
                const int kb_nx = kb + 1;
                const int g_nx = g + h;
                const int h_nx = h ^ 1;
                char* __restrict__ ds_n =
                    LDS + ((h_nx == 0) ? IU4_DS_OFF : IU4_DS1_OFF);
                if (tid < IU4_BM * 2) {
                    const int tok_idx = tid >> 1;
                    const int tok = bs + tok_idx;
                    const int tok_c2 = (tok < N) ? tok : (N - 1);
                    const unsigned dw = ((const unsigned*)(
                        Xq + (long long)kb_nx * N + tok_c2))[tid & 1];
                    ((unsigned*)(ds_n + tok_idx * 8))[tid & 1] =
                        (tok < N) ? dw : 0u;
                }
                char* __restrict__ sz_n =
                    LDS + ((h_nx == 0) ? IU4_SZ0_OFF : IU4_SZ1_OFF);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const int i = tid + r * IU4_NTHREADS;
                    const int row_idx = i >> 1;
                    const int w = i & 1;
                    const int grow = rs + row_idx;
                    const int src_row = (grow < M) ? grow : (M - 1);
                    const unsigned hw = *(const unsigned*)(
                        A + (long long)src_row * groups_per_row * 136
                            + (long long)g_nx * 136 + h_nx * 4);
                    const float f = (w == 0)
                        ? __half2float(__ushort_as_half(
                            (unsigned short)(hw & 0xFFFFu)))
                        : __half2float(__ushort_as_half(
                            (unsigned short)(hw >> 16)));
                    ((float*)(sz_n + row_idx * 8))[w] =
                        (grow < M) ? f : 0.0f;
                }
            }
            __syncthreads();
        }
    }
    // Token-epoch ownership transpose. In epoch e, the accumulator lanes
    // scatter tokens 8e..8e+7. Wave w then owns token 8e+w and lane p owns
    // the same eight consecutive features as the incumbent producer.
    float* overlay = (float*)LDS;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int nb = e >> 1;
        const int token_half8 = e & 1;
        if ((m_lane >> 3) == token_half8) {
            const int token8 = m_lane & 7;
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int feature =
                        wave_row0 + rg * 16 + 8 * k_grp + j;
                    overlay[token8 * 256 + feature] = acc[nb][rg][j];
                }
            }
        }
        __syncthreads();

        const int token = bs + 8 * e + wave;
        if (token < N) {
            const int d0 = lane * 8;
            const int global_base = rs + d0;
            const float* gate_b = Gate + (long long)token * I;

            float u0 = overlay[wave * 256 + d0    ];
            float u1 = overlay[wave * 256 + d0 + 1];
            float u2 = overlay[wave * 256 + d0 + 2];
            float u3 = overlay[wave * 256 + d0 + 3];
            float u4 = overlay[wave * 256 + d0 + 4];
            float u5 = overlay[wave * 256 + d0 + 5];
            float u6 = overlay[wave * 256 + d0 + 6];
            float u7 = overlay[wave * 256 + d0 + 7];

            float g0 = gate_b[global_base    ];
            float g1 = gate_b[global_base + 1];
            float g2 = gate_b[global_base + 2];
            float g3 = gate_b[global_base + 3];
            float g4 = gate_b[global_base + 4];
            float g5 = gate_b[global_base + 5];
            float g6 = gate_b[global_base + 6];
            float g7 = gate_b[global_base + 7];

            #define SILU_UP(g, u, i) (((((g) / (1.0f + expf(-(g)))) * (u)) / Awq[(i)]) * S1[(i) - rs])
            float v0 = SILU_UP(g0, u0, global_base    );
            float v1 = SILU_UP(g1, u1, global_base + 1);
            float v2 = SILU_UP(g2, u2, global_base + 2);
            float v3 = SILU_UP(g3, u3, global_base + 3);
            float v4 = SILU_UP(g4, u4, global_base + 4);
            float v5 = SILU_UP(g5, u5, global_base + 5);
            float v6 = SILU_UP(g6, u6, global_base + 6);
            float v7 = SILU_UP(g7, u7, global_base + 7);
            #undef SILU_UP

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
                float _p = __int_as_float(__builtin_amdgcn_ds_swizzle( \
                    __float_as_int(v), (pat))); \
                if (lane & (str)) { (v) = _p - (v); } \
                else { (v) = (v) + _p; } \
            } while (0)
            #define HBFLY8(pat, str) \
                HBFLY(v0,pat,str); HBFLY(v1,pat,str); \
                HBFLY(v2,pat,str); HBFLY(v3,pat,str); \
                HBFLY(v4,pat,str); HBFLY(v5,pat,str); \
                HBFLY(v6,pat,str); HBFLY(v7,pat,str)
            HBFLY8(0x041F, 1);
            HBFLY8(0x081F, 2);
            HBFLY8(0x101F, 4);
            HBFLY8(0x201F, 8);
            HBFLY8(0x401F, 16);
            #undef HBFLY8
            #undef HBFLY
            float f0 = (v0 * 0.0625f) * S2[d0    ];
            float f1 = (v1 * 0.0625f) * S2[d0 + 1];
            float f2 = (v2 * 0.0625f) * S2[d0 + 2];
            float f3 = (v3 * 0.0625f) * S2[d0 + 3];
            float f4 = (v4 * 0.0625f) * S2[d0 + 4];
            float f5 = (v5 * 0.0625f) * S2[d0 + 5];
            float f6 = (v6 * 0.0625f) * S2[d0 + 6];
            float f7 = (v7 * 0.0625f) * S2[d0 + 7];
            emit_iu4_sidecar_from_producer8(
                f0, f1, f2, f3, f4, f5, f6, f7,
                Down, (int)blockIdx.x, token, N, lane);
        }
        __syncthreads();
    }
}

extern "C" __global__ __launch_bounds__(IU4_NTHREADS, 2)
__attribute__((amdgpu_num_vgpr(208)))
void gemm_up_silu_mq4g256v2_iu4_gfx1201(
    const char*           __restrict__ Wup,
    const block_i4_128*   __restrict__ Xin,
    const float*          __restrict__ Gate,
    const float*          __restrict__ Awq,
    const float*          __restrict__ S1,
    const float*          __restrict__ S2,
    block_i4_128*         __restrict__ Down,
    int I, int K, int N
) {
    gemm_up_silu_mq4g256v2_iu4_gfx1201_body(
        Wup, Xin, Gate, Awq, S1, S2, Down, I, K, N);
}

#else

extern "C" __global__ void gemm_up_silu_mq4g256v2_iu4_gfx1201(
    const char*, const void*, const float*, const float*, const float*,
    const float*, void*, int, int, int
) {}

#endif