// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Qwen4 shared-expert INT8 x INT8 prefill GEMM for gfx12 (RDNA4 — gfx1200/gfx1201):
// both operands are A8 `block_i8_128` planes ([K/128][rows] weights, [K/128][tokens]
// activations, 136 B per 128-K block: float d; int s; signed char qs[128], qs in the
// nibble-pair order of block_i8_128_quant.hip), multiplied on
// `v_wmma_i32_16x16x16_iu8` signed x signed.
//
// A fork of `gemm_mq6g256v2_mmq_i8.gfx12.hip` (same 128x128 tile, 8 waves, 32-row x
// 64-token wave tile, K64 slabs in a two-slot LDS ring with two barriers per K128
// block, banded CTA raster, next-block fetch epoch). What changes:
//   - Weights are already unpacked int8 in the activation's byte order, so the W plane
//     is staged exactly like the A plane (8-byte chunks, the same swizzled fragment
//     order) from the weight blocks; there is no 6-bit decode.
//   - Signed x signed (|C| <= 127*127*128 = 2,064,512 < 2^24): the per-128-block int32
//     accumulator converts to float EXACTLY, no zero-point seed. The fold in K128-block
//     order is acc = fma(RN(d_w * d_x), (float)C_kb, acc), acc starting at +0.0f.
//   - K is any multiple of 128 (the block loop is over K128 blocks, plane parity kb & 1).
//   - Epilogues: DOWN (F32 [tokens][M] rows, overwrite) and GATE_UP (the weight rows
//     are interleaved in groups of 8: rows 8q..8q+3 = gate channels 4q..4q+3, rows
//     8q+4..8q+7 = up channels 4q..4q+3, so one lane holds both operands of the
//     SwiGLU: h = bf16(silu(bf16(g)) * bf16(u)) with the `shared_expert_activation_bf16_f32`
//     expressions, stored as BF16 bits [tokens][M/2]).
//
// Entries (ABI shared by both): (const block_i8_128* Wq, const block_i8_128* Xq,
// void* Y, int M, int K, int N), K % 128 == 0.
//   qwen4_shared_i8_down_set_f32       Y = [N][M] F32 (M rows)
//   qwen4_shared_i8_gate_up_silu_bf16  Y = [N][M/2] BF16 bits, M % 128 == 0, Y 8-byte aligned
// Grid [ceil(M/128), ceil(N/128)], block [256,1,1], dynamic LDS =
// QWEN4_SHARED_I8_LDS_BYTES (36864 B). Compile gate: gfx1200 / gfx1201 only.
//
// LDS map (36864 B; the epilogue store tiles overlap the dead compute planes):
//   slot s (s = 0, 1) @ s*16384: A_s = 32 K16 fragments x 256 B (8192 B), W_s likewise.
//   DS0/DS1 = [128][8] token (d f32, s i32) per kb parity  @ 32768 (2 x 1024)
//   SZ0/SZ1 = [128][8] row (d f32, s i32) per kb parity    @ 34816 (2 x 1024)
// Fragment order (A and W): fragment f = g8*4 + sb*2 + e (g8 = 16-token / 16-row
// group, sb = K32 sub-block, e = K16 half) holds the 8-byte operand lanes of one K16
// step; lane L's bytes sit at f*256 + (L ^ sw(e, L>>4))*8 with sw = (e << 3) ^ ((L >> 4) << 2).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

// block_i8_128 comes from the host JIT prelude (block_i8_128_quant.hip).

#define S8_BM 128
#define S8_BN 128
#define S8_BK 64
#define S8_WAVES 8
#define S8_NTHREADS (S8_WAVES * 32)
#ifndef S8_RASTER_BAND
#define S8_RASTER_BAND 8
#endif

#define S8_A_BYTES (S8_BM * S8_BK)
#define S8_W_BYTES (S8_BN * S8_BK)
#define S8_SLOT_BYTES (S8_A_BYTES + S8_W_BYTES)
#define S8_SLOT0_OFF 0
#define S8_SLOT1_OFF S8_SLOT_BYTES
#define S8_DS0_OFF (2 * S8_SLOT_BYTES)
#define S8_DS_BYTES (S8_BM * 8)
#define S8_DS1_OFF (S8_DS0_OFF + S8_DS_BYTES)
#define S8_SZ0_OFF (S8_DS1_OFF + S8_DS_BYTES)
#define S8_SZ_BYTES (S8_BN * 8)
#define S8_SZ1_OFF (S8_SZ0_OFF + S8_SZ_BYTES)
#define S8_LDS_BYTES (S8_SZ1_OFF + S8_SZ_BYTES)
#define QWEN4_SHARED_I8_LDS_BYTES 36864
_Static_assert(S8_LDS_BYTES == QWEN4_SHARED_I8_LDS_BYTES, "planes must match the host LDS contract");
_Static_assert(32 * 132 * 4 <= QWEN4_SHARED_I8_LDS_BYTES, "wide store tile exceeds LDS");

typedef int32_t __attribute__((ext_vector_type(2)))  s8_int32x2_t;
typedef int32_t __attribute__((ext_vector_type(8)))  s8_int32x8_t;
typedef float   __attribute__((ext_vector_type(8)))  s8_float8_t;
typedef unsigned __attribute__((ext_vector_type(2))) s8_u32x2_t;

static __device__ __forceinline__ s8_int32x8_t s8_zero() {
    return (s8_int32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
}

// One K32 sub-block (two K16 steps): per step 2 W and 4 A b64 fragment reads, 8 WMMAs.
// SB is a template constant (LDS immediates); FIRST starts each chain at zero.
template <int SB, bool FIRST>
static __device__ __forceinline__ void s8_bundle(
    const char* __restrict__ slot,
    const unsigned (&a_lane)[2],
    const unsigned (&w_lane)[2],
    s8_int32x8_t cacc[4][2]
) {
    #pragma unroll
    for (int e = 0; e < 2; e++) {
        const char* a_base = slot + a_lane[e] + SB * 512 + e * 256;
        const char* w_base = slot + S8_A_BYTES + w_lane[e] + SB * 512 + e * 256;
        const s8_int32x2_t b0 = *(const s8_int32x2_t*)(a_base + 0 * 1024);
        const s8_int32x2_t b1 = *(const s8_int32x2_t*)(a_base + 1 * 1024);
        const s8_int32x2_t b2 = *(const s8_int32x2_t*)(a_base + 2 * 1024);
        const s8_int32x2_t b3 = *(const s8_int32x2_t*)(a_base + 3 * 1024);
        const s8_int32x2_t w0 = *(const s8_int32x2_t*)(w_base);
        const s8_int32x2_t w1 = *(const s8_int32x2_t*)(w_base + 1024);
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const s8_int32x2_t b = (nb == 0) ? b0 : (nb == 1) ? b1 : (nb == 2) ? b2 : b3;
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const s8_int32x2_t a = (rg == 0) ? w0 : w1;
                if (FIRST && e == 0) {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a, true, b, s8_zero(), false);
                } else {
                    cacc[nb][rg] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a, true, b, cacc[nb][rg], false);
                }
            }
        }
    }
    __builtin_amdgcn_sched_barrier(0);
}

// bf16_round_trip_f32's rounding (RNE; non-finite values unchanged), `silu_mul_bf16_rt`.
static __device__ __forceinline__ float s8_bf16_rt(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

template <bool GATE_UP>
static __device__ __forceinline__ void qwen4_shared_i8_gemm_body(
    const block_i8_128*   __restrict__ Wq,  // [K/128 x M]
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    void*                 __restrict__ Yv,  // DOWN [N x M] F32; GATE_UP [N x M/2] BF16 bits
    int M, int K, int N
) {
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs;
    int bs;
    {
        // Banded raster (K1 IU4_G12_RASTER): bijective for any grid.
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = S8_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * S8_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, S8_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * S8_BN;
        bs = (in_band / band_rows) * S8_BM;
    }
    if (rs >= M || bs >= N) return;

    const int m_lane = lane & 15;
    const int k_grp  = lane >> 4;
    const int wt_pair = wave >> 1;
    const int tok_half = wave & 1;
    const int wave_row0 = wt_pair * 32;
    const int wave_tok0 = tok_half * 64;

    const int n_kb = K / 128;
    extern __shared__ char LDS[];

    // Swizzled lane byte offsets (K1): parity-0 / parity-1 fragments.
    const unsigned lane_lo = (unsigned)lane & 15u;
    const unsigned lane_half = (unsigned)lane & 16u;
    const unsigned lane_even = ((lane_lo ^ (lane_half >> 2)) | lane_half) * 8u;
    const unsigned lane_odd = ((lane_lo ^ (8u ^ (lane_half >> 2))) | lane_half) * 8u;
    const unsigned a_lane[2] = {
        (unsigned)tok_half * 4096u + lane_even,
        (unsigned)tok_half * 4096u + lane_odd,
    };
    const unsigned w_lane[2] = {
        (unsigned)wt_pair * 2048u + lane_even,
        (unsigned)wt_pair * 2048u + lane_odd,
    };

    s8_float8_t acc[4][2];
    // Staging maps (all loop-invariant), shared by the A and the W plane: round r
    // stages row/token T = r*64 + (tid>>2), 8-byte chunk c = tid&3 (lane half c>>1,
    // K16 half e = c&1) of one K32 half: global T*136 + c*8 (+ 8 + slab*64 + sb*32 in
    // the scalar offset), LDS frag (T>>4)*4 + sb*2 + e, lane (T&15) + 16*(c>>1), swizzled.
    unsigned st_avoff[2];
    unsigned st_wvoff[2];
    unsigned st_lds[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        const unsigned T = (unsigned)(r * 64 + (tid >> 2));
        const unsigned c = (unsigned)(tid & 3);
        st_avoff[r] = (unsigned)(bs + (int)T) * (unsigned)sizeof(block_i8_128) + c * 8u;
        const int grow_r = rs + (int)T;
        const int src_c = (grow_r < M) ? grow_r : (M - 1);
        st_wvoff[r] = (unsigned)src_c * (unsigned)sizeof(block_i8_128) + c * 8u;
        const unsigned e = c & 1u;
        const unsigned L = (T & 15u) + (c >> 1) * 16u;
        const unsigned sw = (e << 3) ^ ((L >> 4) << 2);
        st_lds[r] = (T >> 4) * 1024u + e * 256u + (L ^ sw) * 8u;
    }
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            acc[nb][rg] = (s8_float8_t){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
    s8_int32x8_t cacc[4][2];
    // The A descriptor spans exactly one K128 block of N tokens (tokens >= N read zero);
    // the W descriptor one K128 block of M rows (rows >= M are clamped to M-1).
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i8_128);
    const unsigned w_bytes = (unsigned)M * (unsigned)sizeof(block_i8_128);
    s8_u32x2_t A_pf[2];
    s8_u32x2_t W_pf[2];
    unsigned ds_nx = 0u;
    unsigned sz_nx = 0u;

    // --- Prologue: DS[0] + SZ[0] + slab 0 of block 0 into slot 0. ---
    for (int i = tid; i < S8_BM * 2; i += S8_NTHREADS) {
        const int tok_idx = i >> 1;
        const int w = i & 1;
        const int tok = bs + tok_idx;
        const unsigned dw = (tok < N) ? ((const unsigned*)(Xq + tok))[w] : 0u;
        ((unsigned*)(LDS + S8_DS0_OFF + tok_idx * 8))[w] = dw;
    }
    for (int i = tid; i < S8_BN * 2; i += S8_NTHREADS) {
        const int row_idx = i >> 1;
        const int w = i & 1;
        const int grow = rs + row_idx;
        const int src_row = (grow < M) ? grow : (M - 1);
        const unsigned hw = ((const unsigned*)(Wq + src_row))[w];
        ((unsigned*)(LDS + S8_SZ0_OFF + row_idx * 8))[w] = (grow < M) ? hw : 0u;
    }
    {
        const auto a0_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)Xq, 0, a_bytes, 0x31004000u);
        const auto w0_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)Wq, 0, w_bytes, 0x31004000u);
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            #pragma unroll
            for (int sb = 0; sb < 2; sb++) {
                const s8_u32x2_t av = __builtin_amdgcn_raw_buffer_load_b64(
                    a0_rsrc, st_avoff[r], 8u + (unsigned)sb * 32u, 0);
                *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + st_lds[r] + sb * 512) = av;
                const s8_u32x2_t wv = __builtin_amdgcn_raw_buffer_load_b64(
                    w0_rsrc, st_wvoff[r], 8u + (unsigned)sb * 32u, 0);
                *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + S8_A_BYTES + st_lds[r] + sb * 512) = wv;
            }
        }
    }
    __syncthreads();

    #pragma clang loop unroll_count(1)
    for (int kb = 0; kb < n_kb; kb++) {
        const int h = kb & 1;
        const bool last = (kb + 1 == n_kb);
        const char* __restrict__ ds_lds = LDS + ((h == 0) ? S8_DS0_OFF : S8_DS1_OFF);
        const char* __restrict__ sz_lds = LDS + ((h == 0) ? S8_SZ0_OFF : S8_SZ1_OFF);
        const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Wq + (long long)kb * M), 0, w_bytes, 0x31004000u);
        // --- Slot 0 compute; slab 1 of this block -> slot 1. ---
        {
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 72u, 0);
                W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(w_rsrc, st_wvoff[r], 72u, 0);
            }
            s8_bundle<0, true>(LDS + S8_SLOT0_OFF, a_lane, w_lane, cacc);
            // Slab 1 K32 half 0 -> slot 1 (free since the last rendezvous), then
            // fetch half 1 into the same private registers.
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                asm volatile("" : "+v"(W_pf[r][0]), "+v"(W_pf[r][1]));
                *(s8_u32x2_t*)(LDS + S8_SLOT1_OFF + st_lds[r]) = A_pf[r];
                *(s8_u32x2_t*)(LDS + S8_SLOT1_OFF + S8_A_BYTES + st_lds[r]) = W_pf[r];
            }
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_avoff[r], 104u, 0);
                W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(w_rsrc, st_wvoff[r], 104u, 0);
            }
            s8_bundle<1, false>(LDS + S8_SLOT0_OFF, a_lane, w_lane, cacc);
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                asm volatile("" : "+v"(W_pf[r][0]), "+v"(W_pf[r][1]));
                *(s8_u32x2_t*)(LDS + S8_SLOT1_OFF + st_lds[r] + 512) = A_pf[r];
                *(s8_u32x2_t*)(LDS + S8_SLOT1_OFF + S8_A_BYTES + st_lds[r] + 512) = W_pf[r];
            }
            // Slot-0 retirement plus slot-1 readiness.
            __syncthreads();
            if (!last) {
                // Next-block fetch epoch: raw loads into private registers, first use
                // at the slot-0 publishes below.
                const int kb_nx = kb + 1;
                const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    (void*)(Xq + (long long)kb_nx * N), 0, a_bytes, 0x31004000u);
                const auto w_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    (void*)(Wq + (long long)kb_nx * M), 0, w_bytes, 0x31004000u);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 8u, 0);
                    W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(w_next_rsrc, st_wvoff[r], 8u, 0);
                }
                const int tok_ds = bs + (tid >> 1);
                const int tok_dsc = (tok_ds < N) ? tok_ds : (N - 1);
                ds_nx = ((const unsigned*)(Xq + (long long)kb_nx * N + tok_dsc))[tid & 1];
                const int row_sz = rs + (tid >> 1);
                const int src_sz = (row_sz < M) ? row_sz : (M - 1);
                sz_nx = ((const unsigned*)(Wq + (long long)kb_nx * M + src_sz))[tid & 1];
            }
        }
        // --- Slot 1 compute, fold, next slab 0 -> slot 0. ---
        {
            s8_bundle<0, false>(LDS + S8_SLOT1_OFF, a_lane, w_lane, cacc);
            // Next slab 0 K32 half 0 -> slot 0 (retired at the rendezvous above; a
            // no-op rewrite on the last block), then half 1. The last block's
            // descriptors have zero records: no access.
            #pragma unroll
            for (int r = 0; r < 2; r++) {
                asm volatile("" : "+v"(A_pf[r][0]), "+v"(A_pf[r][1]));
                asm volatile("" : "+v"(W_pf[r][0]), "+v"(W_pf[r][1]));
                *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + st_lds[r]) = A_pf[r];
                *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + S8_A_BYTES + st_lds[r]) = W_pf[r];
            }
            {
                const auto a_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    (void*)(Xq + (long long)(kb + 1) * N), 0, last ? 0u : a_bytes, 0x31004000u);
                const auto w_next_rsrc = __builtin_amdgcn_make_buffer_rsrc(
                    (void*)(Wq + (long long)(kb + 1) * M), 0, last ? 0u : w_bytes, 0x31004000u);
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    A_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(a_next_rsrc, st_avoff[r], 40u, 0);
                    W_pf[r] = __builtin_amdgcn_raw_buffer_load_b64(w_next_rsrc, st_wvoff[r], 40u, 0);
                }
            }
            s8_bundle<1, false>(LDS + S8_SLOT1_OFF, a_lane, w_lane, cacc);
            // Fold (per 128-K block, reads only cacc and the current DS[h]/SZ[h] planes):
            //   acc = fma(RN(d_w * d_x), (float)C, acc), C exact.
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                float dw_row[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const unsigned jbase = (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + j) * 8u;
                    dw_row[j] = *(const float*)(sz_lds + jbase);
                }
                #pragma unroll
                for (int nb = 0; nb < 4; nb++) {
                    int* acc_i = (int*)&cacc[nb][rg];
                    const unsigned tbase = (unsigned)(wave_tok0 + nb * 16 + m_lane) * 8u;
                    const float d_x = *(const float*)(ds_lds + tbase);
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const float fc = (float)acc_i[j];
                        acc[nb][rg][j] = __builtin_fmaf(__fmul_rn(dw_row[j], d_x), fc, acc[nb][rg][j]);
                    }
                }
            }
            if (!last) {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + st_lds[r] + 512) = A_pf[r];
                    *(s8_u32x2_t*)(LDS + S8_SLOT0_OFF + S8_A_BYTES + st_lds[r] + 512) = W_pf[r];
                }
                const int h_np = h ^ 1;
                char* __restrict__ ds_n = LDS + ((h_np == 0) ? S8_DS0_OFF : S8_DS1_OFF);
                {
                    const int tok_dsp = bs + (tid >> 1);
                    ((unsigned*)(ds_n + (tid >> 1) * 8))[tid & 1] = (tok_dsp < N) ? ds_nx : 0u;
                }
                char* __restrict__ sz_n = LDS + ((h_np == 0) ? S8_SZ0_OFF : S8_SZ1_OFF);
                {
                    const int row_szp = rs + (tid >> 1);
                    ((unsigned*)(sz_n + (tid >> 1) * 8))[tid & 1] = (row_szp < M) ? sz_nx : 0u;
                }
            }
            // Slot-1 retirement plus next slot-0 readiness; on the last block this
            // guards store-tile reuse of LDS.
            __syncthreads();
        }
    } // for kb

    if constexpr (GATE_UP) {
        // Rows 8q..8q+7 of the lane's 8 accumulators (row 8*k_grp + j of the fragment) are
        // gate 4q..4q+3 then up 4q..4q+3: the SwiGLU is lane-local. M % 128 == 0.
        const int hid = M >> 1;
        unsigned short* const Y = (unsigned short*)Yv;
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            const int tok = bs + wave_tok0 + nb * 16 + m_lane;
            #pragma unroll
            for (int rg = 0; rg < 2; rg++) {
                const int row0 = rs + wave_row0 + rg * 16 + 8 * k_grp;
                if (tok < N && row0 < M) {
                    unsigned short hb[4];
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float v = s8_bf16_rt(acc[nb][rg][j]);
                        const float u = s8_bf16_rt(acc[nb][rg][4 + j]);
                        const float p = s8_bf16_rt((v / (1.0f + expf(-v))) * u);
                        hb[j] = (unsigned short)(__float_as_uint(p) >> 16);
                    }
                    uint2 pk;
                    pk.x = (unsigned)hb[0] | ((unsigned)hb[1] << 16);
                    pk.y = (unsigned)hb[2] | ((unsigned)hb[3] << 16);
                    *(uint2*)(Y + (long long)tok * hid + (row0 >> 1)) = pk;
                }
            }
        }
        return;
    } else {
        // Epilogue (K1: wide tile stores for full tiles, LDS transpose + checked stores
        // otherwise). `slots` reuses the dead compute planes.
        float* slots = (float*)LDS;
        const int store_i = tid & 15;
        const int store_x = (tid >> 4) & 15;
        const bool full_tile = rs + S8_BM <= M && bs + S8_BM <= N;
        const long long tile_elem0 = (long long)bs * M + rs;
        if (full_tile && (M & 3) == 0 && ((reinterpret_cast<uintptr_t>(Yv) & 15u) == 0)) {
            // Wide path: each lane stores 4 consecutive rows (b128 F32).
            const long long wide_lane_el = (long long)wave * M + lane * 4;
            const unsigned dump_tok = (unsigned)(tok_half * 16 + m_lane);
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    #pragma unroll
                    for (int c = 0; c < 2; c++) {
                        float4 v4;
                        v4.x = acc[nb][rg][4 * c + 0];
                        v4.y = acc[nb][rg][4 * c + 1];
                        v4.z = acc[nb][rg][4 * c + 2];
                        v4.w = acc[nb][rg][4 * c + 3];
                        *(float4*)(slots + dump_tok * 132u
                            + (unsigned)(wave_row0 + rg * 16 + 8 * k_grp + 4 * c)) = v4;
                    }
                }
                __syncthreads();
                #pragma unroll
                for (int it = 0; it < 4; it++) {
                    const float4 v4 = *(const float4*)(slots + (unsigned)(it * 8 + wave) * 132u
                        + (unsigned)lane * 4u);
                    const long long el = tile_elem0 + wide_lane_el
                        + (long long)((it >> 1) * 64 + nb * 16 + (it & 1) * 8) * M;
                    *(float4*)((float*)Yv + el) = v4;
                }
                __syncthreads();
            }
            return;
        }
        // Guarded path (M / N tails, unaligned Y).
        #pragma unroll
        for (int rg = 0; rg < 2; rg++) {
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    slots[(unsigned)(wave * 320 + (8 * k_grp + j) * 20 + m_lane)] = acc[nb][rg][j];
                }
                __syncthreads();
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    const int p = k >> 1;
                    const int hh = k & 1;
                    const int row_rel = p * 32 + rg * 16 + store_i;
                    const int col_rel = hh * 64 + nb * 16 + store_x;
                    const int owner = (p << 1) | hh;
                    const float v = slots[(unsigned)(owner * 320 + store_i * 20 + store_x)];
                    const int row = rs + row_rel;
                    const int col = bs + col_rel;
                    if (row < M && col < N) {
                        ((float*)Yv)[(long long)col * M + row] = v;
                    }
                }
                __syncthreads();
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(S8_NTHREADS, 2)
void qwen4_shared_i8_down_set_f32(
    const block_i8_128*   __restrict__ Wq,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    qwen4_shared_i8_gemm_body<false>(Wq, Xq, Y, M, K, N);
}

extern "C" __global__ __launch_bounds__(S8_NTHREADS, 2)
void qwen4_shared_i8_gate_up_silu_bf16(
    const block_i8_128*   __restrict__ Wq,
    const block_i8_128*   __restrict__ Xq,
    void*                 __restrict__ Y,
    int M, int K, int N
) {
    qwen4_shared_i8_gemm_body<true>(Wq, Xq, Y, M, K, N);
}

#else

extern "C" __global__ void qwen4_shared_i8_down_set_f32(
    const void*, const void*, void*, int, int, int
) {}

extern "C" __global__ void qwen4_shared_i8_gate_up_silu_bf16(
    const void*, const void*, void*, int, int, int
) {}

#endif
