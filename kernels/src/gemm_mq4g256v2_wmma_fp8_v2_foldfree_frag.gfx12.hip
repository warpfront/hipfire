// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Fold-free MQ4V2 FP8 WMMA body for gfx1201.  The activation producer emits
// WMMA fragments directly: for logical `(row,k)`,
//   lane=(row&15)+16*((k>>3)&1)
// A wave therefore issues its four token fragments x four K16 fragments from
// four uniform scalar bases plus one lane offset as a single sixteen-load
// clause before draining 64 WMMAs. Compact W expands once per K64 tile through
// LDS; A never enters LDS. The 9,728-byte W+LUT footprint permits four
// blocks/CU by LDS capacity, though VGPR allocation limits the measured result
// to three.
// Per-128 d=m*2^e is folded into E4M3 W bytes relative to the loader-provided
// per-row reference 2^e_row; the epilogue restores it with the activation scale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int   __attribute__((ext_vector_type(2))) ff_v2i_t;
typedef int   __attribute__((ext_vector_type(4))) ff_v4i_t;
typedef float __attribute__((ext_vector_type(8))) ff_v8f_t;

#define FF_BM 256
#define FF_BN 128
#define FF_BK 64
#define FF_THREADS 256
#define FF_FRAG 288
#define FF_K16 4
#define FF_W_BYTES ((FF_BN / 16) * FF_K16 * FF_FRAG)
#define FF_LUT_BYTES 512
#define FF_LDS_BYTES (FF_W_BYTES + FF_LUT_BYTES)

#ifndef HIPFIRE_FP8_WPRESHUFFLE
#define HIPFIRE_FP8_WPRESHUFFLE 0
#endif

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "fold-free MQ4V2 FP8 kernel requires gfx1201"
#endif

// [mantissa (1,1.5)][e_row-e clamped to 0..15][code 0..15].  Each entry is
// E4M3_RNE(E4M3_RNE((code-8)*mantissa) * 2^-delta), including subnormals.
// The first RNE is intentional: it is the quantizer's fold-honest contract.
__device__ __constant__ unsigned int ff_fold_lut[2][16][4] = {
  {
    {0xCACCCED0u,0xB8C0C4C8u,0x44403800u,0x4E4C4A48u},
    {0xC2C4C6C8u,0xB0B8BCC0u,0x3C383000u,0x46444240u},
    {0xBABCBEC0u,0xA8B0B4B8u,0x34302800u,0x3E3C3A38u},
    {0xB2B4B6B8u,0xA0A8ACB0u,0x2C282000u,0x36343230u},
    {0xAAACAEB0u,0x98A0A4A8u,0x24201800u,0x2E2C2A28u},
    {0xA2A4A6A8u,0x90989CA0u,0x1C181000u,0x26242220u},
    {0x9A9C9EA0u,0x88909498u,0x14100800u,0x1E1C1A18u},
    {0x92949698u,0x84888C90u,0x0C080400u,0x16141210u},
    {0x8A8C8E90u,0x82848688u,0x06040200u,0x0E0C0A08u},
    {0x85868788u,0x81828384u,0x03020100u,0x07060504u},
    {0x82838484u,0x00818282u,0x02010000u,0x04030202u},
    {0x81828282u,0x00008181u,0x01000000u,0x02020101u},
    {0x81818181u,0x00000000u,0x00000000u,0x01010100u},
    {0u,0u,0u,0u},{0u,0u,0u,0u},{0u,0u,0u,0u}
  },
  {
    {0xCFD1D2D4u,0xBCC4C9CCu,0x49443C00u,0x52514F4Cu},
    {0xC7C9CACCu,0xB4BCC1C4u,0x413C3400u,0x4A494744u},
    {0xBFC1C2C4u,0xACB4B9BCu,0x39342C00u,0x42413F3Cu},
    {0xB7B9BABCu,0xA4ACB1B4u,0x312C2400u,0x3A393734u},
    {0xAFB1B2B4u,0x9CA4A9ACu,0x29241C00u,0x32312F2Cu},
    {0xA7A9AAACu,0x949CA1A4u,0x211C1400u,0x2A292724u},
    {0x9FA1A2A4u,0x8C94999Cu,0x19140C00u,0x22211F1Cu},
    {0x97999A9Cu,0x868C9194u,0x110C0600u,0x1A191714u},
    {0x8F919294u,0x8386898Cu,0x09060300u,0x12110F0Cu},
    {0x88898A8Cu,0x82838486u,0x04030200u,0x0A090806u},
    {0x84848586u,0x81828283u,0x02020100u,0x05040403u},
    {0x82828283u,0x00818182u,0x01010000u,0x02020202u},
    {0x81818182u,0x00008181u,0x01000000u,0x01010101u},
    {0x00818181u,0x00000000u,0x00000000u,0x01010000u},
    {0u,0u,0u,0u},{0u,0u,0u,0u}
  }
};

__device__ __forceinline__ void ff_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ unsigned int ff_lut4(unsigned int sel,
    unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3) {
    const unsigned int idx = sel & 0x07070707u;
    const unsigned int lo = __builtin_amdgcn_perm(c1, c0, idx);
    const unsigned int hi = __builtin_amdgcn_perm(c3, c2, idx);
    const unsigned int pick = 0x03020100u | ((sel & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(hi, lo, pick);
}

__device__ __forceinline__ ff_v2i_t ff_expand(unsigned int pk, const unsigned int* lut) {
    const unsigned int ev = pk & 0x0F0F0F0Fu;
    const unsigned int od = (pk >> 4) & 0x0F0F0F0Fu;
    const unsigned int g0 = __builtin_amdgcn_perm(od, ev, 0x05010400u);
    const unsigned int g1 = __builtin_amdgcn_perm(od, ev, 0x07030602u);
    ff_v2i_t out;
    out[0] = (int)ff_lut4(g0, lut[0], lut[1], lut[2], lut[3]);
    out[1] = (int)ff_lut4(g1, lut[0], lut[1], lut[2], lut[3]);
    return out;
}

extern "C" __launch_bounds__(FF_THREADS, 2) __global__
#if HIPFIRE_FP8_QKVZA
void HIPFIRE_FP8_FOLDFREE_KERNEL(
    const char* A_qkv, const char* A_z, const char* A_beta, const char* A_alpha,
    const float* E_qkv, const float* E_z, const float* E_beta, const float* E_alpha,
    const unsigned char* __restrict__ X_fp8, const float* X_half_sums, const float* X_row_scales,
    float* Y_qkv, float* Y_z, float* Y_beta, float* Y_alpha,
    int qkv_m, int z_m, int beta_m, int alpha_m, int K, int N) {
    const int tm = qkv_m + z_m + beta_m + alpha_m;
#elif HIPFIRE_FP8_RESIDUAL
void HIPFIRE_FP8_FOLDFREE_KERNEL(
    const char* A_w, const float* E_w,
    const unsigned char* __restrict__ X_fp8, const float* X_half_sums, const float* X_row_scales,
    float* Y_res, int res_m, int K, int N) {
    const int tm = res_m;
#elif HIPFIRE_FP8_QKV
void HIPFIRE_FP8_FOLDFREE_KERNEL(
    const char* A_q, const char* A_k, const char* A_v,
    const float* E_q, const float* E_k, const float* E_v,
    const unsigned char* __restrict__ X_fp8, const float* X_half_sums, const float* X_row_scales,
    float* Y_q, float* Y_k, float* Y_v,
    int q_m, int k_m, int v_m, int K, int N) {
    const int tm = q_m + k_m + v_m;
#else
void HIPFIRE_FP8_FOLDFREE_KERNEL(
    const char* A_gate, const char* A_up, const float* E_gate, const float* E_up,
    const unsigned char* __restrict__ X_fp8, const float* X_half_sums, const float* X_row_scales,
    float* Y_gate, float* Y_up, int gate_m, int up_m, int K, int N) {
    const int tm = gate_m + up_m;
#endif
    (void)X_half_sums;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int tok_part = wave >> 1;
    const int wt_part = wave & 1;
    const int wave_tok = tok_part * 64;
    const int wave_wt = wt_part * 64;
    const int bs = (int)blockIdx.y * FF_BM;
    const int rs = (int)blockIdx.x * FF_BN;
    const int gpr = K / 256;

    extern __shared__ char lds[];
    char* lds_W = lds;
    unsigned int* lds_lut = (unsigned int*)(lds + FF_W_BYTES);
    if (tid < 128) lds_lut[tid] = ((const unsigned int*)ff_fold_lut)[tid];
    ff_lds_barrier();

#define FF_PICK(GR, WBASE, EBASE, LR) do { \
    const int _gr = (GR); \
    /* family-specific pointer routing; every partition is row-contiguous */ \
    FF_PICK_IMPL(_gr, WBASE, EBASE, LR); \
} while (0)

#if HIPFIRE_FP8_QKVZA
#define FF_PICK_IMPL(GR,WBASE,EBASE,LR) do { \
    (WBASE)=A_qkv; (EBASE)=E_qkv; (LR)=(GR); \
    if ((GR)>=qkv_m+z_m+beta_m) {(WBASE)=A_alpha;(EBASE)=E_alpha;(LR)=(GR)-(qkv_m+z_m+beta_m);} \
    else if ((GR)>=qkv_m+z_m) {(WBASE)=A_beta;(EBASE)=E_beta;(LR)=(GR)-(qkv_m+z_m);} \
    else if ((GR)>=qkv_m) {(WBASE)=A_z;(EBASE)=E_z;(LR)=(GR)-qkv_m;} \
} while(0)
#elif HIPFIRE_FP8_RESIDUAL
#define FF_PICK_IMPL(GR,WBASE,EBASE,LR) do {(WBASE)=A_w;(EBASE)=E_w;(LR)=(GR);} while(0)
#elif HIPFIRE_FP8_QKV
#define FF_PICK_IMPL(GR,WBASE,EBASE,LR) do { \
    (WBASE)=A_q; (EBASE)=E_q; (LR)=(GR); \
    if ((GR)>=q_m+k_m) {(WBASE)=A_v;(EBASE)=E_v;(LR)=(GR)-(q_m+k_m);} \
    else if ((GR)>=q_m) {(WBASE)=A_k;(EBASE)=E_k;(LR)=(GR)-q_m;} \
} while(0)
#else
#define FF_PICK_IMPL(GR,WBASE,EBASE,LR) do { \
    (WBASE)=A_gate; (EBASE)=E_gate; (LR)=(GR); \
    if ((GR)>=gate_m) {(WBASE)=A_up;(EBASE)=E_up;(LR)=(GR)-gate_m;} \
} while(0)
#endif

    ff_v8f_t acc[4][4];
    const ff_v8f_t zero = ff_v8f_t{0,0,0,0,0,0,0,0};
#pragma unroll
    for (int nb=0; nb<4; ++nb)
#pragma unroll
        for (int wf=0; wf<4; ++wf) acc[nb][wf] = zero;

    const int ksteps = K >> 4;
    const unsigned int aoff = (unsigned int)lane * 8u;
    const int mt0 = (bs + wave_tok) >> 4;
    const unsigned char* abase[4];
#pragma unroll
    for (int nb=0; nb<4; ++nb) {
        const int tile_off = (mt0 + nb) * ksteps * 256;
        abase[nb] = X_fp8
            + (unsigned int)__builtin_amdgcn_readfirstlane(tile_off);
    }

#pragma nounroll
    for (int k0=0; k0<K; k0+=FF_BK) {
        // All sixteen A fragments issue together from uniform bases plus the
        // lane's unsigned 32-bit byte offset.  Keep this loop fully unrolled:
        // the required gfx1201 form is one `s_clause 0xf` of SADDR b64 loads.
        ff_v2i_t af[4][FF_K16];
#pragma unroll
        for (int nb=0; nb<4; ++nb)
#pragma unroll
            for (int ks=0; ks<FF_K16; ++ks)
                af[nb][ks] = *(const ff_v2i_t*)(
                    abase[nb] + (aoff
                    + (unsigned int)(ks * 256 + k0 * 16)));
        // Keep the global clause intact and ahead of the independent W expand.
        // Otherwise the scheduler can strand one token fragment above the
        // expansion and reintroduce loadcnt waits in the WMMA drain.
        asm volatile("" ::: "memory");
#if HIPFIRE_FP8_WPRESHUFFLE
#if HIPFIRE_FP8_QKV
#error "weight preshuffle is not implemented for qkv"
#endif
        // One lane owns four adjacent output rows at one K8 half-fragment.
        // The load-time layout makes those four packed dwords contiguous, so
        // the entire compact-W contribution is one b128.  The high bytes of
        // the four positive f16 scales form a parallel b32 plane; the other
        // three header bytes are retained in unused planes to keep the
        // transform byte-for-byte bijective.
        const unsigned int wslot = (unsigned int)tid * 4u;
        const int wlane = (int)(wslot & 31u);
        const int wrest = (int)(wslot >> 5);
        const int wkst = wrest & 3;
        const int wntl = wrest >> 2;
        const int wrow = wntl * 16 + (wlane & 15);
        const int wgr = rs + wrow;
        const int wgrc = wgr < tm ? wgr : tm - 4;
        const char* wtile;
        const float* etile;
        int wlr;
        int wtile_rows;
#if HIPFIRE_FP8_QKVZA
        wtile = A_qkv; etile = E_qkv; wlr = wgrc; wtile_rows = qkv_m;
        if (wgrc >= qkv_m + z_m + beta_m) {
            wtile = A_alpha; etile = E_alpha;
            wlr = wgrc - (qkv_m + z_m + beta_m); wtile_rows = alpha_m;
        } else if (wgrc >= qkv_m + z_m) {
            wtile = A_beta; etile = E_beta;
            wlr = wgrc - (qkv_m + z_m); wtile_rows = beta_m;
        } else if (wgrc >= qkv_m) {
            wtile = A_z; etile = E_z;
            wlr = wgrc - qkv_m; wtile_rows = z_m;
        }
#elif HIPFIRE_FP8_RESIDUAL
        wtile = A_w; etile = E_w; wlr = wgrc; wtile_rows = res_m;
#else
        const bool wtile_up = wgrc >= gate_m;
        wtile = wtile_up ? A_up : A_gate;
        etile = wtile_up ? E_up : E_gate;
        wlr = wtile_up ? wgrc - gate_m : wgrc;
        wtile_rows = wtile_up ? up_m : gate_m;
#endif
        const int wlane_idx = (wlane & 16) | (wlr & 15);
        const size_t wpayload_bytes =
            (size_t)wtile_rows * (size_t)K / 2u;
        const size_t wmeta_half_stride =
            (size_t)(wtile_rows >> 4) * 64u;
        const char* wpacked_base = wtile
            + (((size_t)(wlr >> 4) * (size_t)ksteps
                + (size_t)wkst) * 32u + (unsigned)wlane_idx) * 4u;
        const unsigned char* wscale_base =
            (const unsigned char*)wtile + wpayload_bytes
            + ((size_t)(wlr >> 4) * 4u) * 16u + (unsigned)(wlr & 15);
        const float4 wref = *(const float4*)(etile + wlr);
        ff_v4i_t pk4 =
            *(const ff_v4i_t*)(wpacked_base + (size_t)k0 * 8u);
        if (wgr >= tm) pk4 = ff_v4i_t{0, 0, 0, 0};
        const unsigned int sc4 = *(const unsigned int*)(
            wscale_base + (size_t)(k0 >> 7) * wmeta_half_stride);
#pragma unroll
        for (int q=0; q<4; ++q) {
            const unsigned int dbits = ((sc4 >> (q * 8)) & 0xffu) << 8;
            const unsigned int erow_bits = __float_as_uint(wref[q]);
            const int erow = (int)((erow_bits >> 23) & 0xffu) - 127;
            const int e = (int)((dbits >> 10) & 0x1fu) - 15;
            int delta = erow - e;
            delta = delta < 0 ? 0 : (delta > 15 ? 15 : delta);
            const int mant = (dbits & 0x03ffu) != 0;
            const ff_v2i_t w8 =
                ff_expand((unsigned int)pk4[q],
                          lds_lut + (mant * 16 + delta) * 4);
            const int row = wrow + q;
            const int frag = ((row >> 4) * FF_K16 + wkst) * FF_FRAG;
            *(ff_v2i_t*)(lds_W + frag
                         + ((row & 15) + (wlane >> 4) * 16) * 8) = w8;
        }
#else
        // Expand one 4 KiB compact W tile into fragment order for all waves.
#pragma unroll
        for (int it=0; it<4; ++it) {
            const int slot = it * FF_THREADS + tid;
            const int wr = slot >> 3;
            const int pkidx = slot & 7;
            const int gr = rs + wr;
            const int grc = gr < tm ? gr : tm - 1;
            const char* wb; const float* eb; int lr;
            FF_PICK(grc, wb, eb, lr);
            const int group = k0 >> 8;
            const int half = (k0 >> 7) & 1;
            const int slab = (k0 >> 6) & 1;
            const char* gh = wb + (size_t)lr * (size_t)gpr * 136u
                              + (unsigned)group * 136u;
            const unsigned short dbits = *(const unsigned short*)(gh + half * 4);
            unsigned int pk = *(const unsigned int*)(gh + 8 + half * 64
                                                    + slab * 32 + pkidx * 4);
            if (gr >= tm) pk = 0;
            const unsigned int erow_bits = __float_as_uint(eb[lr]);
            const int erow = (int)((erow_bits >> 23) & 0xffu) - 127;
            const int e = (int)((dbits >> 10) & 0x1fu) - 15;
            int delta = erow - e;
            delta = delta < 0 ? 0 : (delta > 15 ? 15 : delta);
            const int mant = (dbits & 0x03ffu) != 0;
            const ff_v2i_t w8 =
                ff_expand(pk, lds_lut + (mant * 16 + delta) * 4);
            const int ks = pkidx >> 1;
            const int hk = pkidx & 1;
            const int frag = ((wr >> 4) * FF_K16 + ks) * FF_FRAG;
            *(ff_v2i_t*)(lds_W + frag + ((wr & 15) + hk * 16) * 8) = w8;
        }
#endif
        ff_lds_barrier();

#pragma unroll
        for (int ks=0; ks<FF_K16; ++ks) {
            ff_v2i_t wv[4];
#pragma unroll
            for (int wf=0; wf<4; ++wf) {
                const int frag = (((wave_wt >> 4) + wf) * FF_K16 + ks)
                               * FF_FRAG;
                wv[wf] = *(const ff_v2i_t*)(lds_W + frag + lane * 8);
            }
#pragma unroll
            for (int nb=0; nb<4; ++nb)
#pragma unroll
                for (int wf=0; wf<4; ++wf)
                    acc[nb][wf] =
                        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            wv[wf], af[nb][ks], acc[nb][wf]);
            __builtin_amdgcn_sched_barrier(0);
        }
        ff_lds_barrier();
    }

    // Direct stores avoid a 32 KiB transpose plane and preserve exact arithmetic.
#pragma unroll
    for (int nb=0; nb<4; ++nb) {
        const int t = bs + wave_tok + nb * 16 + ml;
        if (t < N) {
            const float as = X_row_scales[t];
#pragma unroll
            for (int wf=0; wf<4; ++wf) {
#pragma unroll
                for (int j=0; j<8; ++j) {
                    const int gr = rs + wave_wt + wf * 16 + 8 * kg + j;
                    if (gr < tm) {
                        const char* wb; const float* eb; int lr;
                        FF_PICK(gr, wb, eb, lr);
                        (void)wb;
                        const float yv = (as * eb[lr]) * acc[nb][wf][j];
#if HIPFIRE_FP8_QKVZA
                        if (gr < qkv_m) Y_qkv[(size_t)t*qkv_m+gr] = yv;
                        else if (gr < qkv_m+z_m) Y_z[(size_t)t*z_m+gr-qkv_m] = yv;
                        else if (gr < qkv_m+z_m+beta_m) Y_beta[(size_t)t*beta_m+gr-qkv_m-z_m] = yv;
                        else Y_alpha[(size_t)t*alpha_m+gr-qkv_m-z_m-beta_m] = yv;
#elif HIPFIRE_FP8_RESIDUAL
                        Y_res[(size_t)t*res_m+gr] += yv;
#elif HIPFIRE_FP8_QKV
                        if (gr < q_m) Y_q[(size_t)t*q_m+gr] = yv;
                        else if (gr < q_m+k_m) Y_k[(size_t)t*k_m+gr-q_m] = yv;
                        else Y_v[(size_t)t*v_m+gr-q_m-k_m] = yv;
#else
                        if (gr < gate_m) Y_gate[(size_t)t*gate_m+gr] = yv;
                        else Y_up[(size_t)t*up_m+gr-gate_m] = yv;
#endif
                    }
                }
            }
        }
    }
#undef FF_PICK_IMPL
#undef FF_PICK
}
