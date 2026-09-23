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
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Research-only 8-wave64, 128-row x 256-token retile of the iu4 MMQ.
// W is staged once for all eight waves; activation fragments are read directly
// from block_i4_128 so the 20480-byte launch allocation still admits 3 blocks/CU.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

#define W64B_BN 128
#define W64B_BM 256
#define W64B_THREADS 512
#define W64B_FRAG_BYTES 256
#define W64B_W_BYTES 4096
#define W64B_W0_OFF 0
#define W64B_W1_OFF 4096
#define W64B_DS0_OFF 8192
#define W64B_DS_BYTES (W64B_BM * 8)
#define W64B_DS1_OFF (W64B_DS0_OFF + W64B_DS_BYTES)
#define W64B_SZ0_OFF (W64B_DS1_OFF + W64B_DS_BYTES)
#define W64B_SZ_BYTES (W64B_BN * 8)
#define W64B_SZ1_OFF (W64B_SZ0_OFF + W64B_SZ_BYTES)
#define W64B_COMPUTE_LDS (W64B_SZ1_OFF + W64B_SZ_BYTES)
#define W64B_SLOT_FLOATS (16 * 20)
_Static_assert(W64B_COMPUTE_LDS <= 20480, "wave64b compute LDS exceeds launch contract");
_Static_assert(8 * W64B_SLOT_FLOATS * 4 <= 20480, "wave64b store slots exceed launch contract");

typedef int32_t __attribute__((ext_vector_type(4))) i32x4;
typedef float __attribute__((ext_vector_type(4))) f32x4;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2;

static __device__ __forceinline__ float2 w64b_fold_pair(
    float sum0, float sum1, float sc0, float sc1, float zp0, float zp1,
    float d, int c0, int c1, float sf) {
    const float t1_0 = __fmul_rn(sc0, d);
    const float t1_1 = __fmul_rn(sc1, d);
    const float p_0 = __fmul_rn(t1_0, (float)c0);
    const float p_1 = __fmul_rn(t1_1, (float)c1);
    const float t2_0 = __fmul_rn(zp0, d);
    const float t2_1 = __fmul_rn(zp1, d);
    const float term_0 = __fmaf_rn(t2_0, sf, p_0);
    const float term_1 = __fmaf_rn(t2_1, sf, p_1);
    return (float2){__fadd_rn(sum0, term_0), __fadd_rn(sum1, term_1)};
}

template <int SB, bool FIRST>
static __device__ __forceinline__ void w64b_bundle(
    const char* __restrict__ w_lds,
    const block_i4_128* __restrict__ xq_block,
    int bs, int N, int wave_tok0, int lane, int m_lane, int k_grp,
    const unsigned (&frW)[4][2], i32x4 cacc[4][4], const i32x4& zc,
    int slab) {
    int af[4];
    #pragma unroll
    for (int rg = 0; rg < 4; ++rg) af[rg] = *(const int*)(w_lds + frW[rg][SB]);
    int bf[4];
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb) {
        const int tok = bs + wave_tok0 + nb * 16 + m_lane;
        const int tc = tok < N ? tok : N - 1;
        const char* qs = (const char*)(xq_block + tc) + 8;
        const int v = *(const int*)(qs + slab * 32 + SB * 16 + k_grp * 4);
        bf[nb] = tok < N ? v : 0;
    }
    #pragma unroll
    for (int rg = 0; rg < 4; ++rg) {
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            if constexpr (FIRST) {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(
                    false, af[rg], true, bf[nb], zc, false);
            } else {
                cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w64_gfx12(
                    false, af[rg], true, bf[nb], cacc[nb][rg], false);
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

template <bool ADD>
static __device__ __forceinline__ void w64b_body(
    const char* __restrict__ A, const block_i4_128* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N) {
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int m_lane = lane & 15;
    const int k_grp = lane >> 4;
    const int c_row = ((k_grp & 1) << 3) | ((k_grp & 2) << 1);
    const int row_half = wave >> 2;
    const int tok_quarter = wave & 3;
    const int wave_row0 = row_half * 64;
    const int wave_tok0 = tok_quarter * 64;
    const int rs = blockIdx.x * W64B_BN;
    const int bs = blockIdx.y * W64B_BM;
    if (rs >= M || bs >= N) return;
    const int groups_per_row = K / 256;
    extern __shared__ char LDS[];
    char* w_lds = LDS + W64B_W0_OFF;

    unsigned frW[4][2];
    const unsigned lane4 = (unsigned)lane * 4u;
    #pragma unroll
    for (int rg = 0; rg < 4; ++rg) {
        const unsigned g8 = (unsigned)(wave_row0 / 16 + rg);
        #pragma unroll
        for (int sb = 0; sb < 2; ++sb)
            frW[rg][sb] = (g8 * 2u + (unsigned)sb) * W64B_FRAG_BYTES + lane4;
    }

    const int stage_row = tid >> 2;
    const int stage_q = tid & 3;
    const int grow = rs + stage_row;
    const int src_row = grow < M ? grow : M - 1;
    const unsigned w_voff = (unsigned)src_row * (unsigned)groups_per_row * 136u
        + (unsigned)stage_q * 8u;
    const unsigned frag_base = ((unsigned)(stage_row >> 4) * 2u
        + (unsigned)(stage_q >> 1)) * W64B_FRAG_BYTES;
    const unsigned kg0 = 2u * (unsigned)(stage_q & 1);
    const unsigned w_loff0 = frag_base + ((unsigned)(stage_row & 15) + kg0 * 16u) * 4u;
    const unsigned w_loff1 = frag_base + ((unsigned)(stage_row & 15) + (kg0 + 1u) * 16u) * 4u;

    // Plane zero metadata and first weight slab.
    for (int i = tid; i < W64B_BM * 2; i += W64B_THREADS) {
        const int tok_idx = i >> 1;
        const int word = i & 1;
        const int tok = bs + tok_idx;
        const unsigned v = tok < N ? ((const unsigned*)(Xq + tok))[word] : 0u;
        ((unsigned*)(LDS + W64B_DS0_OFF + tok_idx * 8))[word] = v;
    }
    for (int i = tid; i < W64B_BN * 2; i += W64B_THREADS) {
        const int row = i >> 1;
        const int word = i & 1;
        const int gr = rs + row;
        const int sr = gr < M ? gr : M - 1;
        const unsigned hw = *(const unsigned*)(A + (long long)sr * groups_per_row * 136);
        const float f = word == 0
            ? __half2float(__ushort_as_half((unsigned short)hw))
            : __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
        ((float*)(LDS + W64B_SZ0_OFF + row * 8))[word] = gr < M ? f : 0.0f;
    }
    {
        const u32x2 wv = *(const u32x2*)(A + 8 + w_voff);
        *(unsigned*)(w_lds + w_loff0) = wv[0];
        *(unsigned*)(w_lds + w_loff1) = wv[1];
    }
    __syncthreads();

    f32x4 acc[4][4];
    #pragma unroll
    for (int nb = 0; nb < 4; ++nb)
        #pragma unroll
        for (int rg = 0; rg < 4; ++rg) acc[nb][rg] = (f32x4){0, 0, 0, 0};
    const i32x4 zc = (i32x4){0, 0, 0, 0};
    i32x4 cacc[4][4];
    u32x2 W_pf;
    unsigned ds_nx = 0;
    unsigned sz_nx = 0;

    for (int g = 0; g < groups_per_row; ++g) {
        #pragma clang loop unroll_count(1)
        for (int h = 0; h < 2; ++h) {
            const int kb = 2 * g + h;
            const bool last = h == 1 && g + 1 == groups_per_row;
            char* ds_lds = LDS + (h == 0 ? W64B_DS0_OFF : W64B_DS1_OFF);
            char* sz_lds = LDS + (h == 0 ? W64B_SZ0_OFF : W64B_SZ1_OFF);
            const block_i4_128* xq_block = Xq + (long long)kb * N;

            W_pf = *(const u32x2*)(A + (long long)g * 136 + 8 + (long long)h * 64 + 32 + w_voff);
            w64b_bundle<0, true>(w_lds, xq_block, bs, N, wave_tok0, lane, m_lane, k_grp,
                                  frW, cacc, zc, 0);
            w64b_bundle<1, false>(w_lds, xq_block, bs, N, wave_tok0, lane, m_lane, k_grp,
                                   frW, cacc, zc, 0);
            w_lds = LDS + W64B_W1_OFF;
            *(unsigned*)(w_lds + w_loff0) = W_pf[0];
            *(unsigned*)(w_lds + w_loff1) = W_pf[1];
            __syncthreads();

            if (!last) {
                const int kb_nx = kb + 1;
                const int g_nx = g + h;
                const int h_nx = h ^ 1;
                W_pf = *(const u32x2*)(A + (long long)g_nx * 136 + 8
                    + (long long)h_nx * 64 + w_voff);
                const int tok = bs + (tid >> 1);
                const int tc = tok < N ? tok : N - 1;
                ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tc))[tid & 1];
                const int row = rs + (tid >> 1);
                const int sr = row < M ? row : M - 1;
                sz_nx = *(const unsigned*)(A + (long long)sr * groups_per_row * 136
                    + g_nx * 136 + h_nx * 4);
            }

            w64b_bundle<0, false>(w_lds, xq_block, bs, N, wave_tok0, lane, m_lane, k_grp,
                                   frW, cacc, zc, 1);
            w64b_bundle<1, false>(w_lds, xq_block, bs, N, wave_tok0, lane, m_lane, k_grp,
                                   frW, cacc, zc, 1);
            #pragma unroll
            for (int rg = 0; rg < 4; ++rg) {
                float sc[4], zp[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const unsigned off = (unsigned)(wave_row0 + rg * 16 + c_row + j) * 8u;
                    sc[j] = *(const float*)(sz_lds + off);
                    zp[j] = *(const float*)(sz_lds + off + 4);
                }
                #pragma unroll
                for (int nb = 0; nb < 4; ++nb) {
                    const unsigned off = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                    const float d = *(const float*)(ds_lds + off);
                    const int s = *(const int*)(ds_lds + off + 4);
                    const float sf = (float)s;
                    int* ci = (int*)&cacc[nb][rg];
                    #pragma unroll
                    for (int j = 0; j < 4; j += 2) {
                        const float2 p = w64b_fold_pair(acc[nb][rg][j], acc[nb][rg][j + 1],
                            sc[j], sc[j + 1], zp[j], zp[j + 1], d, ci[j], ci[j + 1], sf);
                        acc[nb][rg][j] = p.x;
                        acc[nb][rg][j + 1] = p.y;
                    }
                }
            }

            if (!last) {
                w_lds = LDS + W64B_W0_OFF;
                *(unsigned*)(w_lds + w_loff0) = W_pf[0];
                *(unsigned*)(w_lds + w_loff1) = W_pf[1];
                const int hp = h ^ 1;
                char* dsn = LDS + (hp == 0 ? W64B_DS0_OFF : W64B_DS1_OFF);
                const int tok = bs + (tid >> 1);
                ((unsigned*)(dsn + (tid >> 1) * 8))[tid & 1] = tok < N ? ds_nx : 0u;
                if (tid < W64B_BN * 2) {
                    char* szn = LDS + (hp == 0 ? W64B_SZ0_OFF : W64B_SZ1_OFF);
                    const int row = rs + (tid >> 1);
                    const int word = tid & 1;
                    const float f = word == 0
                        ? __half2float(__ushort_as_half((unsigned short)sz_nx))
                        : __half2float(__ushort_as_half((unsigned short)(sz_nx >> 16)));
                    ((float*)(szn + (tid >> 1) * 8))[word] = row < M ? f : 0.0f;
                }
            }
            __syncthreads();
        }
    }

    float* slots = (float*)LDS;
    const int store_i = tid & 15;
    const int store_x = (tid >> 4) & 15;
    const int owner_group = tid >> 8;
    const bool full = rs + W64B_BN <= M && bs + W64B_BM <= N;
    #pragma unroll
    for (int rg = 0; rg < 4; ++rg) {
        #pragma unroll
        for (int nb = 0; nb < 4; ++nb) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                slots[wave * W64B_SLOT_FLOATS + (c_row + j) * 20 + m_lane] = acc[nb][rg][j];
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int owner = owner_group * 4 + q;
                const int row_rel = (owner >> 2) * 64 + rg * 16 + store_i;
                const int col_rel = (owner & 3) * 64 + nb * 16 + store_x;
                const float v = slots[owner * W64B_SLOT_FLOATS + store_i * 20 + store_x];
                const int row = rs + row_rel;
                const int col = bs + col_rel;
                if (full || (row < M && col < N)) {
                    float* yp = Y + (long long)col * M + row;
                    if constexpr (ADD) *yp += v;
                    else *yp = v;
                }
            }
            __syncthreads();
        }
    }
}

#ifndef IU4_W64B_ADD_KERNEL
#define IU4_W64B_ADD_KERNEL gemm_mq4g256v2_residual_mmq_iu4_w64b_full_add
#endif
#ifndef IU4_W64B_SET_KERNEL
#define IU4_W64B_SET_KERNEL gemm_mq4g256v2_residual_mmq_iu4_w64b_full_set
#endif
extern "C" __global__ __launch_bounds__(W64B_THREADS, 1)
void IU4_W64B_ADD_KERNEL(const char* A, const block_i4_128* Xq, float* Y,
                          int M, int K, int N, int) { w64b_body<true>(A, Xq, Y, M, K, N); }
extern "C" __global__ __launch_bounds__(W64B_THREADS, 1)
void IU4_W64B_SET_KERNEL(const char* A, const block_i4_128* Xq, float* Y,
                          int M, int K, int N, int) { w64b_body<false>(A, Xq, Y, M, K, N); }

#endif
