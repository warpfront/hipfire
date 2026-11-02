// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Native-FP8 prefill GEMM for MQ6G256V2 weights on exact gfx1201 (RDNA4).
//
//   Y[t*M + r] = a_t * sum_k w_r(k) * x8_t(k)        (overwrite)
//
// w_r(k) = s*q + z (exact affine, f32 header), q = the 6-bit code; x8_t(k) is
// the decoded E4M3 activation byte (activation / a_t) produced by the MQ4v2
// FP8 producers (Mq4v2Fp8Prepared: x_fp8, half_sums, row_scales).
//
// No weight conversion pass and no resident FP8 weight copy: the 6-bit codes
// are decoded in-register at W staging straight into E4M3 WMMA operands.
// Centred q' = q - 32 in [-32, 31] and z' = z + 32 s, so
//   sum w x = s * sum q' x + z' * sum x            (per 128-K half)
// where sum x is the half-sum the producer already emits.
//
//   SPLIT (default, exact): q' = 8*hi + lo, hi = (q>>3)-4 in [-4,3],
//     lo = q&7 in [0,7]. E4M3(8*hi) in {-32,-24,-16,-8,0,8,16,24} and
//     E4M3(lo) in {0..7} are exact, so the weight side is bit-exact: two FP8
//     WMMAs per (tile, K16) into the same accumulator, each plane one 8-entry
//     v_perm table.
//   ROUND (lossy): one plane E4M3_RNE(q'), one WMMA per (tile, K16).
//
// Geometry: the MQ4 v2 FRAG_ORDER residual skeleton
// (gemm_gate_up_mq4g256v2_wmma_fp8.gfx12.hip, HIPFIRE_FP8_V2_TILE): 128 tokens
// x 128 weight rows x BK64, 8 wave32 (256 threads), fixed 64-token x 32-row
// wave tile (accumulators T[4][2] / P[4][2] v8f), fragment-order LDS with the
// 288 B frag stride, raw-prefetch R1 pipeline, half-local P folded into T at
// each 128-K half. Static LDS only.
//
// Grid [ceil(M/128), ceil(N/128), 1], block [256, 1, 1].
// Host preconditions: M>0, N>0, K>0, K%256==0, exact gfx1201.
//
// MQ6G256V2 group (200 B): [0,4) u32 {f16 s0 | f16 z0 << 16} (codes 0..127),
// [4,8) u32 {s1 | z1 << 16} (codes 128..255), [8,200) little-endian 6-bit
// bitstream (code c = bits [6c, 6c+6)); half h = payload bytes [96h, 96h+96).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef int   __attribute__((ext_vector_type(2))) v2i_t;
typedef float __attribute__((ext_vector_type(8))) v8f_t;

// 1 = exact hi/lo two-plane arm (default), 0 = ROUND single-plane lossy arm.
#ifndef HIPFIRE_MQ6_FP8_SPLIT
#define HIPFIRE_MQ6_FP8_SPLIT 1
#endif
// 0 = F32 output, 1 = BF16 bits (RNE, the incumbent's formula).
#ifndef HIPFIRE_MQ6_FP8_OUT_BF16
#define HIPFIRE_MQ6_FP8_OUT_BF16 0
#endif
#ifndef HIPFIRE_MQ6_FP8_KERNEL
#define HIPFIRE_MQ6_FP8_KERNEL gemm_mq6g256v2_wmma_fp8_split_gfx1201
#endif
// 1 = double-buffered continuous-pipeline variant (see the DB block in the
// kernel), 0 = the single-buffer per-half-prologue kernel.
#ifndef HIPFIRE_MQ6_FP8_DB
#define HIPFIRE_MQ6_FP8_DB 0
#endif

// gfx1201 device only. The !__HIP_DEVICE_COMPILE__ arm lets the host pass
// (which also preprocesses this TU) through; any other device compile hits
// the #error instead of silently building the wrong idiom.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1201__)
#error "gemm_mq6g256v2_wmma_fp8 requires gfx1201"
#endif

// E4M3 (OCP, sign|exp4 bias 7|mant3) v_perm tables, 8 bytes each as two
// dwords (little-endian byte order: byte i of CX0 = entry i, of CX1 = entry
// i+4). The host unit test pins these literals.
//   E4M3(lo)        lo = 0..7 : 00 38 40 44 48 4A 4C 4E
//   E4M3(8*(h-4))   h  = 0..7 : E0 DC D8 D0 00 50 58 5C   (-32,-24,-16,-8,0,8,16,24)
#define HIPFIRE_MQ6_FP8_CL0 0x44403800u
#define HIPFIRE_MQ6_FP8_CL1 0x4E4C4A48u
#define HIPFIRE_MQ6_FP8_CH0 0xD0D8DCE0u
#define HIPFIRE_MQ6_FP8_CH1 0x5C585000u

// Tile geometry (fixed).
#define MQ6_BM 128          // tokens per block
#define MQ6_BN 128          // weight rows per block
#define MQ6_NT 256          // threads per block
#define MQ6_K16 4           // K16 steps per K64 slab
#define MQ6_FRAG 288        // bank-shifted fragment stride (bytes)
#define MQ6_A_OFF 0
#define MQ6_A_BYTES ((MQ6_BM / 16) * MQ6_K16 * MQ6_FRAG)       // 9216
#define MQ6_W_BYTES ((MQ6_BN / 16) * MQ6_K16 * MQ6_FRAG)       // 9216
#if HIPFIRE_MQ6_FP8_DB
// Double-buffered layout: A[2], then per buffer {W_hi, W_lo (SPLIT)} (the lo
// plane stays W_BYTES above the hi plane of its buffer, so the K-loop reads
// are unchanged), then S[2] and SZ[2] (one per half parity).
#define MQ6_MIN_BLOCKS 1
#define MQ6_WH_OFF (2 * MQ6_A_BYTES)
#if HIPFIRE_MQ6_FP8_SPLIT
#define MQ6_WL_OFF (MQ6_WH_OFF + MQ6_W_BYTES)
#define MQ6_WBUF_STRIDE (2 * MQ6_W_BYTES)
#else
#define MQ6_WBUF_STRIDE MQ6_W_BYTES
#endif
#define MQ6_S_OFF (MQ6_WH_OFF + 2 * MQ6_WBUF_STRIDE)
#define MQ6_S_BYTES (MQ6_BM * 4)                               // 512 per half-buffer
#define MQ6_SZ_OFF (MQ6_S_OFF + 2 * MQ6_S_BYTES)
#define MQ6_SZ_BYTES (MQ6_BN * 8)                              // 1024 per half-buffer
#define MQ6_LDS_BYTES (MQ6_SZ_OFF + 2 * MQ6_SZ_BYTES)          // ROUND 39936, SPLIT 58368
#else
#define MQ6_MIN_BLOCKS 2
#define MQ6_WH_OFF MQ6_A_BYTES
#if HIPFIRE_MQ6_FP8_SPLIT
#define MQ6_WL_OFF (MQ6_WH_OFF + MQ6_W_BYTES)
#define MQ6_S_OFF (MQ6_WL_OFF + MQ6_W_BYTES)
#else
#define MQ6_S_OFF (MQ6_WH_OFF + MQ6_W_BYTES)
#endif
#define MQ6_S_BYTES (MQ6_BM * 4)                               // 512
#define MQ6_SZ_OFF (MQ6_S_OFF + MQ6_S_BYTES)
#define MQ6_SZ_BYTES (MQ6_BN * 8)                              // 1024
#define MQ6_LDS_BYTES (MQ6_SZ_OFF + MQ6_SZ_BYTES)              // SPLIT 29184, ROUND 19968
#endif
#define MQ6_A_ROUNDS (MQ6_BM * 4 / MQ6_NT)                     // 2
#define MQ6_W_ROUNDS (MQ6_BN * 4 / MQ6_NT)                     // 2
#define MQ6_RPR (MQ6_NT / 4)                                   // rows per round: 64
#define MQ6_GROUP_BYTES 200u

#if MQ6_LDS_BYTES < (MQ6_NT / 32) * 2048
#error "LDS must hold the 512-float/wave store transpose tile"
#endif

// Spread the four 6-bit codes of a 24-bit field into the four bytes of a dword.
__device__ static inline unsigned int mq6_sp4(unsigned int t) {
    return (t & 0x3Fu) | ((t << 2) & 0x3F00u) | ((t << 4) & 0x3F0000u)
        | ((t << 6) & 0x3F000000u);
}

// 16 codes in w0 | w1<<32 | w2<<64 (code i = bits [6i, 6i+6)) -> four dwords
// b[j], byte i of b[j] = code 4j+i (0..63).
__device__ static inline void mq6_codes16(unsigned int w0, unsigned int w1,
    unsigned int w2, unsigned int (&b)[4]) {
    b[0] = mq6_sp4(w0) & 0x3F3F3F3Fu;
    b[1] = mq6_sp4((w0 >> 24) | (w1 << 8)) & 0x3F3F3F3Fu;
    b[2] = mq6_sp4((w1 >> 16) | (w2 << 16)) & 0x3F3F3F3Fu;
    b[3] = mq6_sp4(w2 >> 8) & 0x3F3F3F3Fu;
}

// Expand 16 codes into the two 8-byte fragment slots (pp=0: codes 0..7 = the
// K16 fragment's kg=0 half, pp=1: codes 8..15 = kg=1). hp = hi plane (the
// only plane under ROUND), lp = lo plane (SPLIT only).
__device__ static inline void mq6_expand16(unsigned int w0, unsigned int w1,
    unsigned int w2, v2i_t (&hp)[2], v2i_t (&lp)[2]) {
    unsigned int b[4];
    mq6_codes16(w0, w1, w2, b);
#if HIPFIRE_MQ6_FP8_SPLIT
    // v_perm_b32(S0, S1, sel): result byte i = {S1 bytes 0-3, S0 bytes 4-7}[sel_i];
    // sel bytes are 0..7 here (masked), so the tables never see the special
    // selector values.
    unsigned int lo_e[4];
    unsigned int hi_e[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        lo_e[j] = __builtin_amdgcn_perm(HIPFIRE_MQ6_FP8_CL1, HIPFIRE_MQ6_FP8_CL0,
            b[j] & 0x07070707u);
        hi_e[j] = __builtin_amdgcn_perm(HIPFIRE_MQ6_FP8_CH1, HIPFIRE_MQ6_FP8_CH0,
            (b[j] >> 3) & 0x07070707u);
    }
    hp[0] = v2i_t{(int)hi_e[0], (int)hi_e[1]};
    hp[1] = v2i_t{(int)hi_e[2], (int)hi_e[3]};
    lp[0] = v2i_t{(int)lo_e[0], (int)lo_e[1]};
    lp[1] = v2i_t{(int)lo_e[2], (int)lo_e[3]};
#else
    unsigned int e[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float f0 = (float)(b[j] & 0xFFu) - 32.0f;
        const float f1 = (float)((b[j] >> 8) & 0xFFu) - 32.0f;
        const float f2 = (float)((b[j] >> 16) & 0xFFu) - 32.0f;
        const float f3 = (float)(b[j] >> 24) - 32.0f;
        unsigned int p = 0u;
        p = __builtin_amdgcn_cvt_pk_fp8_f32(f0, f1, p, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32(f2, f3, p, true);
        e[j] = p;
    }
    hp[0] = v2i_t{(int)e[0], (int)e[1]};
    hp[1] = v2i_t{(int)e[2], (int)e[3]};
    lp[0] = v2i_t{0, 0};
    lp[1] = v2i_t{0, 0};
#endif
}

// One K16 step of one K64 slab for the wave's 64-token x 32-row tile. KS and
// FIRST are compile-time so every LDS read is an immediate off the loop
// invariant a_fptr / w_fptr. FIRST (slab 0, ks 0) starts the half's P chain
// from zero.
template <int KS, bool FIRST>
__device__ __forceinline__ void mq6_compute_ks(v8f_t (&P)[4][2],
    const char* a_fptr, const char* w_fptr) {
    const v8f_t zc = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const v2i_t wh0 = *(const v2i_t*)(w_fptr + (0 * MQ6_K16 + KS) * MQ6_FRAG);
    const v2i_t wh1 = *(const v2i_t*)(w_fptr + (1 * MQ6_K16 + KS) * MQ6_FRAG);
#if HIPFIRE_MQ6_FP8_SPLIT
    const v2i_t wl0 = *(const v2i_t*)(w_fptr + MQ6_W_BYTES + (0 * MQ6_K16 + KS) * MQ6_FRAG);
    const v2i_t wl1 = *(const v2i_t*)(w_fptr + MQ6_W_BYTES + (1 * MQ6_K16 + KS) * MQ6_FRAG);
#endif
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        const v2i_t a = *(const v2i_t*)(a_fptr + (nb * MQ6_K16 + KS) * MQ6_FRAG);
        P[nb][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            wh0, a, FIRST ? zc : P[nb][0]);
        P[nb][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            wh1, a, FIRST ? zc : P[nb][1]);
#if HIPFIRE_MQ6_FP8_SPLIT
        P[nb][0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wl0, a, P[nb][0]);
        P[nb][1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(wl1, a, P[nb][1]);
#endif
    }
    __builtin_amdgcn_sched_barrier(0);
}


extern "C" __launch_bounds__(MQ6_NT, MQ6_MIN_BLOCKS) __global__
void HIPFIRE_MQ6_FP8_KERNEL(
    const char* __restrict__ A,
    const unsigned char* __restrict__ X_fp8,
    const float* __restrict__ X_half_sums,
    const float* __restrict__ X_row_scales,
    void* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int rs = (int)blockIdx.x * MQ6_BN;
    const int bs = (int)blockIdx.y * MQ6_BM;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const int n_wt_parts = MQ6_BN / 32;                 // 4
    const int tok_part = wave / n_wt_parts;
    const int wt_part = wave - tok_part * n_wt_parts;
    const int wave_tok = tok_part * 64;
    const int wave_wt = wt_part * 32;
    const int gpr = K / 256;

    __shared__ __attribute__((aligned(16))) char lds[MQ6_LDS_BYTES];
    char* lds_A = lds + MQ6_A_OFF;
    char* lds_Wh = lds + MQ6_WH_OFF;
#if HIPFIRE_MQ6_FP8_SPLIT
    char* lds_Wl = lds + MQ6_WL_OFF;
#endif
    char* lds_S = lds + MQ6_S_OFF;
    char* lds_SZ = lds + MQ6_SZ_OFF;

    // Fragment-order K-loop read pointers: loop-invariant (wave-uniform frag
    // base + lane*8); every K-loop LDS read is *(v2i_t*)(ptr + IMM).
    const unsigned lane8 = (unsigned)lane * 8u;
    const char* a_fptr = lds_A + (unsigned)(wave_tok >> 4)
        * (unsigned)MQ6_K16 * (unsigned)MQ6_FRAG + lane8;
    const char* w_fptr = lds_Wh + (unsigned)(wave_wt >> 4)
        * (unsigned)MQ6_K16 * (unsigned)MQ6_FRAG + lane8;

    // ---- staging map (loop-invariant) ----
    const unsigned st_quad = (unsigned)(tid & 3);   // K16 step of the slab
    const unsigned st_grp = (unsigned)(tid >> 2);
    const v2i_t st_z64 = {0, 0};

    // A (activations): one b128 pair (2 b64 chunks) per round, frag order.
    unsigned st_agoff[MQ6_A_ROUNDS];
    unsigned st_ab0[MQ6_A_ROUNDS];
    unsigned st_aok[MQ6_A_ROUNDS];
    #pragma unroll
    for (int r = 0; r < MQ6_A_ROUNDS; r++) {
        const int a_row = r * MQ6_RPR + (int)st_grp;
        const int a_tok = bs + a_row;
        const int a_tok_c = (a_tok < N) ? a_tok : (N - 1);
        st_aok[r] = (a_tok < N) ? 1u : 0u;
        st_agoff[r] = (unsigned)a_tok_c * (unsigned)K + st_quad * 16u;
        st_ab0[r] = ((unsigned)(a_row >> 4) * (unsigned)MQ6_K16 + st_quad)
            * (unsigned)MQ6_FRAG + (unsigned)(a_row & 15) * 8u;
    }

    // W (weights): per round, the lane's 12 payload bytes (3 dwords = one K16
    // step, 16 codes) of row w_row; the 16 codes publish as two b64 frag slots
    // (pp=0 codes 0..7, pp=1 codes 8..15) per plane.
    const char* st_wbase[MQ6_W_ROUNDS];
    unsigned st_wok[MQ6_W_ROUNDS];
    unsigned st_wslot[MQ6_W_ROUNDS][2];
    const unsigned st_wq = st_quad * 12u;
    #pragma unroll
    for (int r = 0; r < MQ6_W_ROUNDS; r++) {
        const int w_row = r * MQ6_RPR + (int)st_grp;
        const int w_grow = rs + w_row;
        const int w_grow_c = (w_grow < M) ? w_grow : (M - 1);
        st_wbase[r] = A + (size_t)w_grow_c * (size_t)gpr * (size_t)MQ6_GROUP_BYTES;
        st_wok[r] = (w_grow < M) ? 1u : 0u;
        #pragma unroll
        for (int pp = 0; pp < 2; pp++) {
            st_wslot[r][pp] = ((unsigned)(w_row >> 4) * (unsigned)MQ6_K16 + st_quad)
                * (unsigned)MQ6_FRAG + (unsigned)((w_row & 15) + pp * 16) * 8u;
        }
    }

    // S entry: one f32 per lane (two lanes share a token row, dup benign).
    const int st_srow = tid >> 1;
    const int st_stok = bs + st_srow;
    const int st_stok_c = (st_stok < N) ? st_stok : (N - 1);
    const unsigned st_sok = (st_stok < N) ? 1u : 0u;
    const unsigned st_soff = (unsigned)st_stok_c * (unsigned)gpr * 2u;
    const unsigned st_slds = (unsigned)st_srow * 4u;

    // SZ entry: threads tid < 128 own weight row rs+tid ({s, z + 32 s}).
    const int sz_grow = rs + (tid & (MQ6_BN - 1));
    const int sz_grow_c = (sz_grow < M) ? sz_grow : (M - 1);
    const char* st_szbase = A + (size_t)sz_grow_c * (size_t)gpr * (size_t)MQ6_GROUP_BYTES;
    const unsigned st_szok = (sz_grow < M) ? 1u : 0u;
    const unsigned st_szlds = (unsigned)(tid & (MQ6_BN - 1)) * 8u;

    const v8f_t zc = v8f_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    v8f_t T[4][2];
    v8f_t P[4][2];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        T[nb][0] = zc;
        T[nb][1] = zc;
    }

#if HIPFIRE_MQ6_FP8_DB
    // ---- double-buffered continuous pipeline (HIPFIRE_MQ6_FP8_DB=1) ----
    // The K loop is flattened to K64 slabs s = 4g + j (j = 0..3; slab parity
    // = j&1, half = j>>1). A, W_hi, W_lo (SPLIT) are double-buffered by
    // slab parity (s&1); S and SZ by half parity ((s>>1)&1 == j>>1). Slab s+1's
    // raw global loads (and, when s+1 starts a half, that half's S float and SZ
    // header word) are issued before slab s's compute and published into the
    // other buffer after it, so there is no per-half prologue bubble and ONE
    // barrier per slab.
    //
    // Correctness (why one barrier per slab suffices): the publish at the end
    // of slab s writes buffer (s+1)&1, which was last READ by slab s-1's
    // compute (and, for S/SZ, by a fold at least one half earlier); every wave
    // passed the barrier that ends slab s-1 before any wave reaches slab s's
    // publish, so no wave can still be reading it. The data published at slab
    // s is first read at slab s+1 (compute) or at the fold of its half, both
    // after the barrier that ends slab s. The fold of slab s reads half-buffer
    // j>>1 while the publish of the same slab writes the other half-buffer.
    {
        // Prologue: slab 0 (g = 0, h = 0, slab 0) + half 0's S/SZ -> buffer 0.
        {
            const float s_ld = X_half_sums[st_soff];
            *(float*)(lds_S + st_slds) = st_sok ? s_ld : 0.0f;
        }
        if (tid < MQ6_BN) {
            const unsigned int hw = *(const unsigned int*)st_szbase;
            const float s_f = __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)));
            const float z_f = __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
            float* szp = (float*)(lds_SZ + st_szlds);
            szp[0] = st_szok ? s_f : 0.0f;
            szp[1] = st_szok ? (z_f + 32.0f * s_f) : 0.0f;
        }
        #pragma unroll
        for (int r = 0; r < MQ6_A_ROUNDS; r++) {
            const unsigned char* asrc = X_fp8 + st_agoff[r];
            const v2i_t v0l = *(const v2i_t*)asrc;
            const v2i_t v1l = *(const v2i_t*)(asrc + 8u);
            char* adst = lds_A + st_ab0[r];
            *(v2i_t*)adst = st_aok[r] ? v0l : st_z64;
            *(v2i_t*)(adst + 128u) = st_aok[r] ? v1l : st_z64;
        }
        #pragma unroll
        for (int r = 0; r < MQ6_W_ROUNDS; r++) {
            const unsigned int w0 = *(const unsigned int*)(st_wbase[r] + 8u + st_wq);
            const unsigned int w1 = *(const unsigned int*)(st_wbase[r] + 8u + st_wq + 4u);
            const unsigned int w2 = *(const unsigned int*)(st_wbase[r] + 8u + st_wq + 8u);
            v2i_t hp[2];
            v2i_t lp[2];
            mq6_expand16(w0, w1, w2, hp, lp);
            #pragma unroll
            for (int pp = 0; pp < 2; pp++) {
                *(v2i_t*)(lds_Wh + st_wslot[r][pp]) = st_wok[r] ? hp[pp] : st_z64;
#if HIPFIRE_MQ6_FP8_SPLIT
                *(v2i_t*)(lds_Wl + st_wslot[r][pp]) = st_wok[r] ? lp[pp] : st_z64;
#endif
            }
        }
    }
    __syncthreads();
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        auto slab = [&](const int J, const int g) __attribute__((always_inline)) {
            const int SL = J & 1;                // slab within the half
            const int HC = J >> 1;               // current half == S/SZ half-buffer
            const int NJ = (J + 1) & 3;          // next slab's j
            const int NBUF = (J + 1) & 1;        // next slab's A/W buffer
            const int NH = NJ >> 1;              // next slab's half (== its half-buffer)
            const int NSL = NJ & 1;
            const bool WRAP = (J == 3);          // next slab is slab 0 of group g+1
            const bool NEW_HALF = (J & 1) == 1;  // next slab starts a half
            const bool has_next = !WRAP || (g + 1 < gpr);
            const unsigned int gn = (unsigned int)g + (WRAP ? 1u : 0u);

            v2i_t A_pf[MQ6_A_ROUNDS * 2];
            unsigned int W_pf[MQ6_W_ROUNDS * 3];
            float s_pf = 0.0f;
            unsigned int hw_pf = 0u;
            // (a) raw global loads of slab s+1 (R1: stay raw until publish).
            if (has_next) {
                const unsigned char* xbase = X_fp8 + gn * 256u + (unsigned int)NH * 128u
                    + (unsigned int)NSL * 64u;
                #pragma unroll
                for (int r = 0; r < MQ6_A_ROUNDS; r++) {
                    const unsigned char* asrc = xbase + st_agoff[r];
                    A_pf[r * 2] = *(const v2i_t*)asrc;
                    A_pf[r * 2 + 1] = *(const v2i_t*)(asrc + 8u);
                }
                const unsigned int wpb = gn * MQ6_GROUP_BYTES + 8u + (unsigned int)NH * 96u
                    + (unsigned int)NSL * 48u + st_wq;
                #pragma unroll
                for (int r = 0; r < MQ6_W_ROUNDS; r++) {
                    W_pf[r * 3] = *(const unsigned int*)(st_wbase[r] + wpb);
                    W_pf[r * 3 + 1] = *(const unsigned int*)(st_wbase[r] + wpb + 4u);
                    W_pf[r * 3 + 2] = *(const unsigned int*)(st_wbase[r] + wpb + 8u);
                }
                if (NEW_HALF) {
                    s_pf = X_half_sums[st_soff + gn * 2u + (unsigned int)NH];
                    if (tid < MQ6_BN) {
                        hw_pf = *(const unsigned int*)(st_szbase + gn * MQ6_GROUP_BYTES
                            + (unsigned int)NH * 4u);
                    }
                }
            }

            // (b) compute slab s from buffer s&1; FIRST starts the half's P.
            {
                const char* a_cur = a_fptr + (unsigned int)(J & 1) * (unsigned int)MQ6_A_BYTES;
                const char* w_cur = w_fptr + (unsigned int)(J & 1) * (unsigned int)MQ6_WBUF_STRIDE;
                if (SL == 0) {
                    mq6_compute_ks<0, true>(P, a_cur, w_cur);
                } else {
                    mq6_compute_ks<0, false>(P, a_cur, w_cur);
                }
                mq6_compute_ks<1, false>(P, a_cur, w_cur);
                mq6_compute_ks<2, false>(P, a_cur, w_cur);
                mq6_compute_ks<3, false>(P, a_cur, w_cur);
            }

            // (c) fold the finished half: T += s*P + z'*S (half-buffer HC).
            if (SL == 1) {
                float Sh[4];
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    Sh[nb] = *(const float*)(lds_S + (unsigned int)HC * (unsigned int)MQ6_S_BYTES
                        + (unsigned int)(wave_tok + nb * 16 + ml) * 4u);
                }
                #pragma unroll
                for (int wf = 0; wf < 2; wf++) {
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int grow = wave_wt + wf * 16 + 8 * kg + j;
                        const float* szp = (const float*)(lds_SZ
                            + (unsigned int)HC * (unsigned int)MQ6_SZ_BYTES
                            + (unsigned int)grow * 8u);
                        const float s_h = szp[0];
                        const float z_h = szp[1];
                        #pragma unroll
                        for (int nb = 0; nb < 4; nb++) {
                            T[nb][wf][j] = fmaf(s_h, P[nb][wf][j], T[nb][wf][j]);
                            T[nb][wf][j] = fmaf(z_h, Sh[nb], T[nb][wf][j]);
                        }
                    }
                }
            }

            // (d) R1 pin, then publish slab s+1 into buffer (s+1)&1 (and, on a
            // half start, S/SZ into half-buffer NH).
            if (has_next) {
                #pragma unroll
                for (int r = 0; r < MQ6_A_ROUNDS * 2; r++) {
                    asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                }
                #pragma unroll
                for (int r = 0; r < MQ6_W_ROUNDS * 3; r++) {
                    asm volatile("" : "+v"(W_pf[r]));
                }
                if (NEW_HALF) {
                    asm volatile("" : "+v"(s_pf));
                    asm volatile("" : "+v"(hw_pf));
                }
                #pragma unroll
                for (int r = 0; r < MQ6_A_ROUNDS; r++) {
                    char* adst = lds_A + (unsigned int)NBUF * (unsigned int)MQ6_A_BYTES + st_ab0[r];
                    *(v2i_t*)adst = st_aok[r] ? A_pf[r * 2] : st_z64;
                    *(v2i_t*)(adst + 128u) = st_aok[r] ? A_pf[r * 2 + 1] : st_z64;
                }
                #pragma unroll
                for (int r = 0; r < MQ6_W_ROUNDS; r++) {
                    v2i_t hp[2];
                    v2i_t lp[2];
                    mq6_expand16(W_pf[r * 3], W_pf[r * 3 + 1], W_pf[r * 3 + 2], hp, lp);
                    #pragma unroll
                    for (int pp = 0; pp < 2; pp++) {
                        *(v2i_t*)(lds_Wh + (unsigned int)NBUF * (unsigned int)MQ6_WBUF_STRIDE
                            + st_wslot[r][pp]) = st_wok[r] ? hp[pp] : st_z64;
#if HIPFIRE_MQ6_FP8_SPLIT
                        *(v2i_t*)(lds_Wl + (unsigned int)NBUF * (unsigned int)MQ6_WBUF_STRIDE
                            + st_wslot[r][pp]) = st_wok[r] ? lp[pp] : st_z64;
#endif
                    }
                }
                if (NEW_HALF) {
                    *(float*)(lds_S + (unsigned int)NH * (unsigned int)MQ6_S_BYTES + st_slds)
                        = st_sok ? s_pf : 0.0f;
                    if (tid < MQ6_BN) {
                        const float s_f = __half2float(__ushort_as_half((unsigned short)(hw_pf & 0xFFFFu)));
                        const float z_f = __half2float(__ushort_as_half((unsigned short)(hw_pf >> 16)));
                        float* szp = (float*)(lds_SZ + (unsigned int)NH * (unsigned int)MQ6_SZ_BYTES
                            + st_szlds);
                        szp[0] = st_szok ? s_f : 0.0f;
                        szp[1] = st_szok ? (z_f + 32.0f * s_f) : 0.0f;
                    }
                }
            }
            // (e) the one barrier of the slab.
            __syncthreads();
        };
        // Unrolling the four slabs of a group (compile-time parity) needs >256
        // VGPRs (spills); the parity-runtime loop keeps ~190.
        #pragma nounroll
        for (int j = 0; j < 4; j++) slab(j, g);
    }
#else
    #pragma nounroll
    for (int g = 0; g < gpr; g++) {
        #pragma nounroll
        for (int h = 0; h < 2; h++) {
            // Raw prefetch (R1): slab s+1's global loads are issued into
            // registers BEFORE slab s's compute and stay RAW until after it
            // (behind an opaque register pin); decode + LDS publish of s+1
            // follow compute (single buffer).
            v2i_t A_pf[MQ6_A_ROUNDS * 2];
            unsigned int W_pf[MQ6_W_ROUNDS * 3];
            // Prologue: stage S/SZ/A/W for slab 0.
            {
                {
                    const unsigned int s_idx = st_soff + (unsigned int)g * 2u + (unsigned int)h;
                    const float s_ld = X_half_sums[s_idx];
                    *(float*)(lds_S + st_slds) = st_sok ? s_ld : 0.0f;
                }
                if (tid < MQ6_BN) {
                    const unsigned int hw = *(const unsigned int*)(st_szbase
                        + (unsigned int)g * MQ6_GROUP_BYTES + (unsigned int)h * 4u);
                    const float s_f = __half2float(__ushort_as_half((unsigned short)(hw & 0xFFFFu)));
                    const float z_f = __half2float(__ushort_as_half((unsigned short)(hw >> 16)));
                    float* szp = (float*)(lds_SZ + st_szlds);
                    szp[0] = st_szok ? s_f : 0.0f;
                    szp[1] = st_szok ? (z_f + 32.0f * s_f) : 0.0f;
                }
                {
                    const unsigned char* xbase = X_fp8 + (unsigned int)g * 256u + (unsigned int)h * 128u;
                    #pragma unroll
                    for (int r = 0; r < MQ6_A_ROUNDS; r++) {
                        const unsigned char* asrc = xbase + st_agoff[r];
                        const v2i_t v0l = *(const v2i_t*)asrc;
                        const v2i_t v1l = *(const v2i_t*)(asrc + 8u);
                        const v2i_t v0 = st_aok[r] ? v0l : st_z64;
                        const v2i_t v1 = st_aok[r] ? v1l : st_z64;
                        char* adst = lds_A + st_ab0[r];
                        *(v2i_t*)adst = v0;
                        *(v2i_t*)(adst + 128u) = v1;
                    }
                }
                {
                    const unsigned int wpb = (unsigned int)g * MQ6_GROUP_BYTES + 8u
                        + (unsigned int)h * 96u + st_wq;
                    #pragma unroll
                    for (int r = 0; r < MQ6_W_ROUNDS; r++) {
                        const unsigned int w0 = *(const unsigned int*)(st_wbase[r] + wpb);
                        const unsigned int w1 = *(const unsigned int*)(st_wbase[r] + wpb + 4u);
                        const unsigned int w2 = *(const unsigned int*)(st_wbase[r] + wpb + 8u);
                        v2i_t hp[2];
                        v2i_t lp[2];
                        mq6_expand16(w0, w1, w2, hp, lp);
                        // OOB rows publish all-zero bytes in every plane
                        // (SPLIT's hi table maps code 0 to -32).
                        #pragma unroll
                        for (int pp = 0; pp < 2; pp++) {
                            *(v2i_t*)(lds_Wh + st_wslot[r][pp]) = st_wok[r] ? hp[pp] : st_z64;
#if HIPFIRE_MQ6_FP8_SPLIT
                            *(v2i_t*)(lds_Wl + st_wslot[r][pp]) = st_wok[r] ? lp[pp] : st_z64;
#endif
                        }
                    }
                }
            }
            __syncthreads();
            #pragma nounroll
            for (int slab = 0; slab < 2; slab++) {
                // Prefetch slab s+1 globals before compute s.
                if (slab + 1 < 2) {
                    {
                        const unsigned char* xbase = X_fp8 + (unsigned int)g * 256u + (unsigned int)h * 128u
                            + (unsigned int)(slab + 1) * 64u;
                        #pragma unroll
                        for (int r = 0; r < MQ6_A_ROUNDS; r++) {
                            const unsigned char* asrc = xbase + st_agoff[r];
                            A_pf[r * 2] = *(const v2i_t*)asrc;
                            A_pf[r * 2 + 1] = *(const v2i_t*)(asrc + 8u);
                        }
                    }
                    {
                        const unsigned int wpb = (unsigned int)g * MQ6_GROUP_BYTES + 8u
                            + (unsigned int)h * 96u + (unsigned int)(slab + 1) * 48u + st_wq;
                        #pragma unroll
                        for (int r = 0; r < MQ6_W_ROUNDS; r++) {
                            W_pf[r * 3] = *(const unsigned int*)(st_wbase[r] + wpb);
                            W_pf[r * 3 + 1] = *(const unsigned int*)(st_wbase[r] + wpb + 4u);
                            W_pf[r * 3 + 2] = *(const unsigned int*)(st_wbase[r] + wpb + 8u);
                        }
                    }
                }

                // Inner loop: ds_read_b64 A/W -> v_wmma fp8 16x16x16.
                if (slab == 0) {
                    mq6_compute_ks<0, true>(P, a_fptr, w_fptr);
                    mq6_compute_ks<1, false>(P, a_fptr, w_fptr);
                    mq6_compute_ks<2, false>(P, a_fptr, w_fptr);
                    mq6_compute_ks<3, false>(P, a_fptr, w_fptr);
                } else {
                    mq6_compute_ks<0, false>(P, a_fptr, w_fptr);
                    mq6_compute_ks<1, false>(P, a_fptr, w_fptr);
                    mq6_compute_ks<2, false>(P, a_fptr, w_fptr);
                    mq6_compute_ks<3, false>(P, a_fptr, w_fptr);
                }
                // R1 pin: opaque register-only identity on the raw future
                // slab so the OOB zero-selects / decode consume it only here,
                // after the current slab's compute.
                if (slab + 1 < 2) {
                    #pragma unroll
                    for (int r = 0; r < MQ6_A_ROUNDS * 2; r++) {
                        asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                    }
                    #pragma unroll
                    for (int r = 0; r < MQ6_W_ROUNDS * 3; r++) {
                        asm volatile("" : "+v"(W_pf[r]));
                    }
                }
                // Publish prefetched slab s+1 (buffer free once compute s is
                // done). The barrier is REQUIRED: fast waves would otherwise
                // overwrite slab s while stragglers still read it.
                if (slab + 1 < 2) {
                    __syncthreads();
                    #pragma unroll
                    for (int r = 0; r < MQ6_A_ROUNDS; r++) {
                        char* adst = lds_A + st_ab0[r];
                        *(v2i_t*)adst = st_aok[r] ? A_pf[r * 2] : st_z64;
                        *(v2i_t*)(adst + 128u) = st_aok[r] ? A_pf[r * 2 + 1] : st_z64;
                    }
                    #pragma unroll
                    for (int r = 0; r < MQ6_W_ROUNDS; r++) {
                        v2i_t hp[2];
                        v2i_t lp[2];
                        mq6_expand16(W_pf[r * 3], W_pf[r * 3 + 1], W_pf[r * 3 + 2], hp, lp);
                        #pragma unroll
                        for (int pp = 0; pp < 2; pp++) {
                            *(v2i_t*)(lds_Wh + st_wslot[r][pp]) = st_wok[r] ? hp[pp] : st_z64;
#if HIPFIRE_MQ6_FP8_SPLIT
                            *(v2i_t*)(lds_Wl + st_wslot[r][pp]) = st_wok[r] ? lp[pp] : st_z64;
#endif
                        }
                    }
                }

                if (slab == 1) {
                    // Fold at the 128-K half boundary: T += s*P + z'*S, with
                    // z' = z + 32 s from SZ (P holds sum (q-32) x).
                    float Sh[4];
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        Sh[nb] = *(const float*)(lds_S + (unsigned int)(wave_tok + nb * 16 + ml) * 4u);
                    }
                    #pragma unroll
                    for (int wf = 0; wf < 2; wf++) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            const int grow = wave_wt + wf * 16 + 8 * kg + j;
                            const float* szp = (const float*)(lds_SZ + (unsigned int)grow * 8u);
                            const float s_h = szp[0];
                            const float z_h = szp[1];
                            #pragma unroll
                            for (int nb = 0; nb < 4; nb++) {
                                T[nb][wf][j] = fmaf(s_h, P[nb][wf][j], T[nb][wf][j]);
                                T[nb][wf][j] = fmaf(z_h, Sh[nb], T[nb][wf][j]);
                            }
                        }
                    }
                }

                __syncthreads();
            }
        }
    }
#endif  // HIPFIRE_MQ6_FP8_DB

    // Coalesced store: lane (ml,kg) holds T[nb][wf][j] for token
    // wave_tok+nb*16+ml and rows wave_wt+wf*16+8*kg+j. Transpose each nb's
    // 16x32 wave tile through LDS (row skew (col+ml)&31 avoids the 16-way
    // bank conflict) so every global store is one contiguous segment with
    // lane = output row. A/W/S/SZ are dead (all folds complete): the tile
    // reuses the LDS base (8 waves x 16 x 32 floats = 16 KiB <= total).
    float* lds_tile = (float*)lds;
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int wf = 0; wf < 2; wf++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const int col = (wf * 16 + 8 * kg + j + ml) & 31;
                lds_tile[(unsigned int)(wave * 512 + ml * 32 + col)] = T[nb][wf][j];
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r16 = 0; r16 < 16; r16++) {
            const int t = bs + wave_tok + nb * 16 + r16;
            if (t < N) {
                const float at = X_row_scales[t];
                const int r = rs + wave_wt + lane;
                const float v = lds_tile[(unsigned int)(wave * 512 + r16 * 32 + ((lane + r16) & 31))];
                const float y = at * v;
                if (r < M) {
#if HIPFIRE_MQ6_FP8_OUT_BF16
                    // Incumbent bf16 formula: v = 0.0f + y, RNE on the bit
                    // pattern for finite values, truncation otherwise.
                    const float yv = 0.0f + y;
                    const unsigned int u = __float_as_uint(yv);
                    const unsigned short hb = isfinite(yv)
                        ? (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16)
                        : (unsigned short)(u >> 16);
                    ((unsigned short*)Y)[(size_t)t * (size_t)M + (size_t)r] = hb;
#else
                    ((float*)Y)[(size_t)t * (size_t)M + (size_t)r] = y;
#endif
                }
            }
        }
        __syncthreads();
    }
}
