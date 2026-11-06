// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// qwen4 Flash-Next routed-expert DOWN projection (qt53 MQ4G128V2 / F1 planar MQ4G128V2P, M = 2560,
// K = 640, top-10), integer lane: symmetric IU4 expert codes x block_i8_128 (A8) activations on
// gfx1201 (KERNEL-PLAN §1.1 integer lane, §1.3 D-indexed / D-grouped ABIs, §5 E1).
//
// The experts are symmetric (every group header has z == -8 s exactly, verified at load by
// qwen4_moe_sym_check): w = s * (q - 8), q the raw 0..15 nibble. The kernels never read z.
//
// Arithmetic (one definition shared with the CPU oracle crates/rdna-compute/tests/qwen4_moe_down_a8_hw.rs):
//   for every 128-K block g = 0..4 in increasing order, for every (row, pick):
//     C  = sum_{k in block} (q_w[k] - 8) * qx[k]        exact int32 (|C| <= 8 * 127 * 128 = 130048)
//        = sum q_w * qx - 8 * s_x                       (s_x = block_i8_128.s, the exact sum of qx)
//     sd = RN(f32(s_h) * d_x)                           s_h the f16 group scale, d_x the block's f32 d
//     acc = fmaf(sd, (float)C, acc)                     acc starts at +0.0f
//   Y = acc (F32). No FP atomics, no reassociated f32 sums; the integer C is order-free, the f32 fold
//   is one sequential chain per (row, pick), the SAME chain in every entry (dot4 and WMMA, singleton
//   and batched), so a pick's bytes do not depend on the entry / row count / tile position.
//
// Activation sidecar: `block_i8_128` (136 B {f32 d; i32 s; i8 qs[128]}, block_i8_128_quant.hip,
// from the host JIT prelude), ROW-MAJOR [x_src_rows][5]: row r (a flat pick, the same row index the
// incumbent F32 X tensor has) owns 5 consecutive blocks (K = 640 -> 680 B), qs in the "nibble-pair order":
// within every 8-element K group, qs[8g + 0..3] = q[8g + 0, 2, 4, 6] and qs[8g + 4..7] =
// q[8g + 1, 3, 5, 7] -- exactly what one packed weight dword gives with lo = pk & 0x0F0F0F0F and
// hi = (pk >> 4) & 0x0F0F0F0F, so a weight dword pairs with its own activation dwords with no shuffle.
//
// Expert tensor layouts (both read as the shipped model / loader define them, no repack):
//   shipped : [M][5 groups][ f16 s, f16 z, 64 nibble bytes ]   row = 340 B, group = 68 B, payload at +4
//   planar  : [codes plane M * 320 B][scales plane M * 5 LE f16]   row codes at row*320 + g*64,
//             scale of (row, g) at M*320 + row*10 + 2g          (hipfire_runtime::mq4_planar)
// Both: nibble byte b of a 128-wide group = elements 2b (low nibble) and 2b + 1 (high nibble).
//
// Entries (explicit typed names; none shares an entry with a flag-dependent pointer meaning):
//   qwen4_moe_down_a8_quant_gfx1201            producer: F32 [rows][K] -> block_i8_128 [rows][K/128]
//     (const f32* X, block_i8_128* Xq, i32 K, i32 rows)                        24 B
//     grid [rows], block (K/128)*32 (one wave per 128-K block), no LDS; quantizer = the shared
//     reference (quantize_block_i8_128_wave), so bytes == quantize_int8_mmq_ds128 on the same X.
//   qwen4_moe_down_a8_indexed_gfx1201 / _indexed_planar_gfx1201   D-indexed, rows 1..8 (any N)
//     (const u64* expert_ptrs, const i32* topk, const block_i8_128* xq, f32* expanded,
//      i32 M, i32 K, i32 N, i32 n_exp, i32 k_top)                              52 B (+0 pad to 56)
//     grid [M / (A8_DN_WPB * 8 * A8_DN_RG), k_top, N], block A8_DN_WPB * 32 (defaults 4/1: [M/32, k_top, N] x 128),
//     no LDS. expanded[(token * k_top + rank) * M + row] (the incumbent expanded layout); expert < 0 or
//     >= n_exp writes +0 for every row, K != 640 / token >= N / rank >= k_top return without writing.
//   qwen4_moe_down_a8_grouped_gfx1201 / _grouped_planar_gfx1201   D-grouped, rows 9..48
//     (const u64* ptrs, const i32* tile_ids, const i32* sorted_slots, const block_i8_128* xq, f32* Y,
//      i32 M, i32 K, i32 x_row_div, i32 grouped_rows, i32 x_src_rows, i32 n_exp)   64 B
//     grid [ceil(M / (A8_GR_NW * A8_GR_RG * 16)), ceil(grouped_rows / 16)], block A8_GR_NW * 32,
//     static LDS 10944 B. Y[(grouped_slot) * M + row] for EVERY grouped_slot < grouped_rows (the beta
//     default dead-tile/pad contract: sentinel tiles, pad slots (sorted == -1) and out-of-range x rows
//     store +0.0f; no dead-store skipping).
//
// Test-only knobs (never defined in production): -DA8_ABL=<bits> timing ablations (see A8_ABL below),
// -DA8_DEBUG_C (+ -DA8_SUF=_dbgc to coexist with the production entries) also stores the per-block integer C after Y:
//   indexed: i32 [5][N * k_top][M] at (i32*)out + N * k_top * M; grouped: i32 [5][grouped_rows][M] at
//   (i32*)Y + grouped_rows * M (0 for dead slots / sentinel tiles).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// block_i8_128 + quantize_block_i8_128_wave come from the host JIT prelude (block_i8_128_quant.hip,
// concatenated ahead with HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE defined by the caller).

#if defined(__gfx1201__)

typedef int32_t __attribute__((ext_vector_type(2))) a8_i32x2;
typedef int32_t __attribute__((ext_vector_type(8))) a8_i32x8;
typedef unsigned __attribute__((ext_vector_type(2))) a8_u32x2;
typedef unsigned __attribute__((ext_vector_type(4), aligned(4))) a8_u32x4;  // dword-aligned b128

// Entry-name suffix hook: empty in production; the test-only -DA8_DEBUG_C build is compiled with
// -DA8_SUF=_dbgc so its entries (qwen4_moe_down_a8_indexed_dbgc_gfx1201 ...) coexist with the production ones.
#ifndef A8_SUF
#define A8_SUF
#endif
#define A8_ENTRY2(a, b, c) a##b##c
#define A8_ENTRY(a, b, c) A8_ENTRY2(a, b, c)

#ifndef A8_DN_WPB
#define A8_DN_WPB 4
#endif
#ifndef A8_DN_RG
#define A8_DN_RG 1
#endif
#ifndef A8_GR_NW
#define A8_GR_NW 4
#endif
#ifndef A8_GR_RG
#define A8_GR_RG 1
#endif

// Timing ablations (test only; 0 in production; results are numerically meaningless): 1 fully coalesced
// weight addressing (grouped), 2 no WMMA / fold compute (grouped; loaded words xor-folded).
#ifndef A8_ABL
#define A8_ABL 0
#endif

#define A8_G 5                      // 128-K blocks per K = 640 row
#define A8_BLK 136                  // sizeof(block_i8_128)
#define A8_ROWQ (A8_G * A8_BLK)     // 680 B of sidecar per row

static __device__ __forceinline__ float a8_f16(unsigned short h) {
    return __half2float(__ushort_as_half(h));
}

#ifdef A8_DEBUG_C
#define A8_DBG_C(ptr, idx, v) (((int*)(ptr))[(idx)] = (v))
#else
#define A8_DBG_C(ptr, idx, v) ((void)0)
#endif

// ===================================================================================================
// Producer: one wave per (row, 128-K block). Same quantizer as quantize_int8_mmq_ds128.
// ===================================================================================================
extern "C" __global__ __launch_bounds__(1024) void qwen4_moe_down_a8_quant_gfx1201(
    const float* __restrict__ X, block_i8_128* __restrict__ Xq, int K, int rows) {
    const int nb = K >> 7;
    const int row = (int)blockIdx.x;
    const int wave = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    if (row >= rows || wave >= nb || (K & 127) != 0) return;
    const float4 xi = *(const float4*)(X + (long long)row * K + wave * 128 + lane * 4);
    const float xv[4] = {xi.x, xi.y, xi.z, xi.w};
    quantize_block_i8_128_wave(xv, Xq + (long long)row * nb + wave, lane);
}

// ===================================================================================================
// D-indexed decode (rows 1..8): signed-activation x unsigned-nibble dot4.
//   Wave = 8 * RGN adjacent output rows of ONE pick. Lane = (row-in-8 = lane >> 2, quarter q = lane & 3):
//   per 128-K group the lane owns 16 weight bytes (elements 32 q .. 32 q + 31 = 4 weight dwords) and the
//   matching 32 qs bytes (8 dwords); 8 sudot4 give the lane's partial sum of q_w * qx, the quad adds its
//   four partials (DPP), C = that - 8 s_x.
// ===================================================================================================
static __device__ __forceinline__ int a8_dpp_x1(int v) { return __builtin_amdgcn_update_dpp(0, v, 0xB1, 0xF, 0xF, false); }
static __device__ __forceinline__ int a8_dpp_x2(int v) { return __builtin_amdgcn_update_dpp(0, v, 0x4E, 0xF, 0xF, false); }

template <bool PLANAR, int WPB, int RGN>
static __device__ __forceinline__ void a8_dn_indexed_body(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk,
    const block_i8_128* __restrict__ xq, float* __restrict__ out, int M, int K, int N, int n_exp, int k_top) {
    const int lane = (int)(threadIdx.x & 31);
    const int wave = __builtin_amdgcn_readfirstlane((int)(threadIdx.x >> 5));
    const int row0 = (blockIdx.x * WPB + wave) * (8 * RGN);
    const int rank = (int)blockIdx.y;
    const int token = (int)blockIdx.z;
    if (row0 >= M || k_top <= 0 || rank >= k_top || token >= N || K != A8_G * 128) return;
    const int flat = token * k_top + rank;
    const int expert = topk[flat];
    const int q = lane & 3;
    const int rl = lane >> 2;

    float res[RGN];
#pragma unroll
    for (int r = 0; r < RGN; ++r) res[r] = 0.0f;

    if (expert >= 0 && expert < n_exp) {
        const unsigned char* wb = (const unsigned char*)expert_ptrs[expert];
        const block_i8_128* xb = xq + (long long)flat * A8_G;

        a8_u32x4 xv[A8_G][2];
        float dv[A8_G];
        int sv[A8_G];
#pragma unroll
        for (int g = 0; g < A8_G; ++g) {
            const unsigned char* qp = (const unsigned char*)(xb + g) + 8 + 32 * q;
            xv[g][0] = *(const a8_u32x4*)qp;
            xv[g][1] = *(const a8_u32x4*)(qp + 16);
            dv[g] = xb[g].d;
            sv[g] = xb[g].s;
        }
        a8_u32x4 wv[RGN][A8_G];
        unsigned short sc[RGN][A8_G];
#pragma unroll
        for (int r = 0; r < RGN; ++r) {
            const int row = row0 + 8 * r + rl;
            const int rowc = row < M ? row : M - 1;
#pragma unroll
            for (int g = 0; g < A8_G; ++g) {
                if constexpr (PLANAR) {
                    wv[r][g] = *(const a8_u32x4*)(wb + (long long)rowc * 320 + g * 64 + 16 * q);
                    sc[r][g] = *(const unsigned short*)(wb + (long long)M * 320 + (long long)rowc * 10 + 2 * g);
                } else {
                    wv[r][g] = *(const a8_u32x4*)(wb + (long long)rowc * 340 + g * 68 + 4 + 16 * q);
                    sc[r][g] = *(const unsigned short*)(wb + (long long)rowc * 340 + g * 68);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RGN; ++r) {
            float acc = 0.0f;
#pragma unroll
            for (int g = 0; g < A8_G; ++g) {
                int c = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const unsigned w = wv[r][g][i];
                    const unsigned lo = w & 0x0F0F0F0Fu;
                    const unsigned hi = (w >> 4) & 0x0F0F0F0Fu;
                    const unsigned xa = xv[g][i >> 1][(i & 1) * 2 + 0];
                    const unsigned xc = xv[g][i >> 1][(i & 1) * 2 + 1];
                    c = __builtin_amdgcn_sudot4(false, (int)lo, true, (int)xa, c, false);
                    c = __builtin_amdgcn_sudot4(false, (int)hi, true, (int)xc, c, false);
                }
                c += a8_dpp_x1(c);
                c += a8_dpp_x2(c);
                const int C = c - 8 * sv[g];
                const float sd = __fmul_rn(a8_f16(sc[r][g]), dv[g]);
                acc = __builtin_fmaf(sd, (float)C, acc);
#ifdef A8_DEBUG_C
                if (q == 0) {
                    const int row = row0 + 8 * r + rl;
                    if (row < M) A8_DBG_C(out + (long long)N * k_top * M, ((long long)g * N * k_top + flat) * M + row, C);
                }
#endif
            }
            res[r] = acc;
        }
    }
    if (q == 0) {
#pragma unroll
        for (int r = 0; r < RGN; ++r) {
            const int row = row0 + 8 * r + rl;
            if (row < M) out[(long long)flat * M + row] = res[r];
        }
    }
#ifdef A8_DEBUG_C
    if (!(expert >= 0 && expert < n_exp) && q == 0) {
        for (int r = 0; r < RGN; ++r) {
            const int row = row0 + 8 * r + rl;
            if (row < M)
                for (int g = 0; g < A8_G; ++g) A8_DBG_C(out + (long long)N * k_top * M, ((long long)g * N * k_top + flat) * M + row, 0);
        }
    }
#endif
}

extern "C" __global__ __launch_bounds__(A8_DN_WPB * 32) void A8_ENTRY(qwen4_moe_down_a8_indexed, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk,
    const block_i8_128* __restrict__ xq, float* __restrict__ out, int M, int K, int N, int n_exp, int k_top) {
    a8_dn_indexed_body<false, A8_DN_WPB, A8_DN_RG>(expert_ptrs, topk, xq, out, M, K, N, n_exp, k_top);
}
extern "C" __global__ __launch_bounds__(A8_DN_WPB * 32) void A8_ENTRY(qwen4_moe_down_a8_indexed_planar, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ topk,
    const block_i8_128* __restrict__ xq, float* __restrict__ out, int M, int K, int N, int n_exp, int k_top) {
    a8_dn_indexed_body<true, A8_DN_WPB, A8_DN_RG>(expert_ptrs, topk, xq, out, M, K, N, n_exp, k_top);
}

// ===================================================================================================
// D-grouped verify (rows 9..48): signed iu8 WMMA 16 x 16 x 16, A = activations (WMMA row = grouped
// slot of the 16-slot tile), B = UNSIGNED nibble codes (WMMA column = weight row). D[8 kg + j][lo]:
// the lane owns weight row `lo` of a 16-row group and slots 8 kg + j, j = 0..7 (kg = lane >> 4).
// Within a 128-K block, lane half kg owns elements 64 kg .. 64 kg + 63 (32 weight bytes = 8 dwords);
// WMMA step j covers octet 8 kg + j (weight dword j of the lane, qs bytes 64 kg + 8 j .. + 7).
// The WMMA chain of a block starts from -8 s_x(slot) so D is exactly C. The tile's 16 sidecar rows
// are staged once in LDS (all 5 blocks); dead slots (sorted == -1, sentinel tile, x row out of range)
// stage as zero blocks and their outputs are forced to +0.0f.
// ===================================================================================================
template <bool PLANAR, int NW, int RG, bool NODZ = false>
static __device__ __forceinline__ void a8_dn_grouped_body(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots, const block_i8_128* __restrict__ xq, float* __restrict__ Y,
    int M, int K, int x_row_div, int grouped_rows, int x_src_rows, int n_exp) {
    constexpr int NT = NW * 32;
    constexpr int ROWS = NW * RG * 16;
    constexpr int WQ = A8_ROWQ / 8;  // 85 b64 words per sidecar row
    __shared__ __attribute__((aligned(16))) unsigned char xs[16 * A8_ROWQ];
    __shared__ unsigned live_lds[16];

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int lo = lane & 15;
    const int kg = lane >> 4;
    const int tile = (int)blockIdx.y;
    const int slot_start = tile * 16;
    if ((int)blockIdx.x * ROWS >= M || slot_start >= grouped_rows || K != A8_G * 128) return;
    const int rbase = (int)blockIdx.x * ROWS + wave * (RG * 16);
    const int expert = tile_ids[tile];
    const bool eok = expert >= 0 && expert < n_exp && x_row_div > 0;
#ifndef A8_DEBUG_C
    if (!eok) {
        if (NODZ) return;  // no dead store (the `_nodz` entries): the tile is never read
        // Dead (sentinel / invalid) tile: the beta default stores +0.0f for every (slot, row); nothing to read.
        const int r0 = (int)blockIdx.x * ROWS;
        const int rend = (r0 + ROWS < M) ? (r0 + ROWS) : M;
        if ((M & 3) == 0) {
            const int c4 = (rend - r0) >> 2;
            for (int i = tid; i < 16 * c4; i += NT) {
                const int s = i / c4;
                const int cc = i - s * c4;
                const int gs = slot_start + s;
                if (gs < grouped_rows) *(float4*)(Y + (long long)gs * M + r0 + 4 * cc) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        } else {
            for (int i = tid; i < 16 * (rend - r0); i += NT) {
                const int s = i / (rend - r0);
                const int gs = slot_start + s;
                if (gs < grouped_rows) Y[(long long)gs * M + r0 + (i - s * (rend - r0))] = 0.0f;
            }
        }
        return;
    }
#endif

    // 1) stage the 16 slots' sidecar rows (global -> regs here; regs -> LDS after the weight loads issue)
    constexpr int SPW = (16 + NW - 1) / NW;         // slots staged per wave
    constexpr int WIT = (WQ + 31) / 32;             // b64 words per lane per slot
    a8_u32x2 xr[SPW][WIT];
    int xrow[SPW];
#pragma unroll
    for (int i = 0; i < SPW; ++i) {
        const int s = wave + i * NW;
        int xrw = -1;
        if (s < 16 && eok) {
            const int gs = slot_start + s;
            if (gs < grouped_rows) {
                const int flat = sorted_slots[gs];
                if (flat >= 0) {
                    const int r = (x_row_div == 1) ? flat : flat / x_row_div;
                    if (r >= 0 && r < x_src_rows) xrw = r;
                }
            }
        }
        xrow[i] = xrw;
#pragma unroll
        for (int t = 0; t < WIT; ++t) {
            const int w = lane + 32 * t;
            a8_u32x2 v = (a8_u32x2){0u, 0u};
            if (xrw >= 0 && w < WQ) v = *(const a8_u32x2*)((const unsigned char*)xq + (long long)xrw * A8_ROWQ + 8 * w);
            xr[i][t] = v;
        }
    }

    // 2) weights + headers of this lane's (row, kg) for all 5 groups, issued before the LDS publish
    a8_u32x4 wv[RG][A8_G][2];
    unsigned short sc[RG][A8_G];
    const unsigned char* wb = eok ? (const unsigned char*)expert_ptrs[expert] : nullptr;
    if (eok) {
#pragma unroll
        for (int rg = 0; rg < RG; ++rg) {
            const int row = rbase + rg * 16 + lo;
            const int rowc = row < M ? row : M - 1;
#pragma unroll
            for (int g = 0; g < A8_G; ++g) {
                if constexpr (PLANAR) {
#if A8_ABL & 1
                    const unsigned char* p = wb + (long long)(rbase + rg * 16) * 320 + g * 1024 + 32 * lane;
#else
                    const unsigned char* p = wb + (long long)rowc * 320 + g * 64 + 32 * kg;
#endif
                    wv[rg][g][0] = *(const a8_u32x4*)p;
                    wv[rg][g][1] = *(const a8_u32x4*)(p + 16);
                    sc[rg][g] = *(const unsigned short*)(wb + (long long)M * 320 + (long long)rowc * 10 + 2 * g);
                } else {
                    const unsigned char* p = wb + (long long)rowc * 340 + g * 68 + 4 + 32 * kg;
                    wv[rg][g][0] = *(const a8_u32x4*)p;
                    wv[rg][g][1] = *(const a8_u32x4*)(p + 16);
                    sc[rg][g] = *(const unsigned short*)(wb + (long long)rowc * 340 + g * 68);
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < SPW; ++i) {
        const int s = wave + i * NW;
        if (s < 16) {
#pragma unroll
            for (int t = 0; t < WIT; ++t) {
                const int w = lane + 32 * t;
                if (w < WQ) *(a8_u32x2*)(xs + s * A8_ROWQ + 8 * w) = xr[i][t];
            }
            if (lane == 0) live_lds[s] = xrow[i] >= 0 ? 1u : 0u;
        }
    }
    __syncthreads();

    float acc[RG][8];
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[rg][j] = 0.0f;

    if (eok) {
#pragma unroll
        for (int g = 0; g < A8_G; ++g) {
            a8_i32x2 a[8];
            float dvj[8];
            a8_i32x8 cinit;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                a[j] = *(const a8_i32x2*)(xs + lo * A8_ROWQ + g * A8_BLK + 8 + 64 * kg + 8 * j);
                dvj[j] = *(const float*)(xs + (8 * kg + j) * A8_ROWQ + g * A8_BLK);
                cinit[j] = -8 * *(const int*)(xs + (8 * kg + j) * A8_ROWQ + g * A8_BLK + 4);
            }
#pragma unroll
            for (int rg = 0; rg < RG; ++rg) {
#if A8_ABL & 2
                {
                    unsigned xr_ = 0;
#pragma unroll
                    for (int j = 0; j < 8; ++j) xr_ ^= wv[rg][g][j >> 2][j & 3] ^ (unsigned)a[j][0] ^ (unsigned)a[j][1];
                    acc[rg][0] += (float)(int)xr_;
                    continue;
                }
#endif
                a8_i32x8 c = cinit;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned w = wv[rg][g][j >> 2][j & 3];
                    const a8_i32x2 b = (a8_i32x2){(int)(w & 0x0F0F0F0Fu), (int)((w >> 4) & 0x0F0F0F0Fu)};
                    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a[j], false, b, c, false);
                }
                const float sf = a8_f16(sc[rg][g]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float sd = __fmul_rn(sf, dvj[j]);
                    acc[rg][j] = __builtin_fmaf(sd, (float)c[j], acc[rg][j]);
                }
#ifdef A8_DEBUG_C
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int gs = slot_start + 8 * kg + j;
                    const int row = rbase + rg * 16 + lo;
                    if (gs < grouped_rows && row < M)
                        A8_DBG_C(Y + (long long)grouped_rows * M, ((long long)g * grouped_rows + gs) * M + row, live_lds[8 * kg + j] ? c[j] : 0);
                }
#endif
            }
        }
    }

#pragma unroll
    for (int rg = 0; rg < RG; ++rg) {
        const int row = rbase + rg * 16 + lo;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int gs = slot_start + 8 * kg + j;
            if (gs < grouped_rows && row < M) {
                const bool live = live_lds[8 * kg + j] != 0u;
                if (!NODZ || live) Y[(long long)gs * M + row] = live ? acc[rg][j] : 0.0f;
            }
        }
    }
#ifdef A8_DEBUG_C
    if (!eok) {
        // sentinel / invalid tile: C = 0 for every block (the compute loop above never ran)
#pragma unroll
        for (int rg = 0; rg < RG; ++rg) {
            const int row = rbase + rg * 16 + lo;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int gs = slot_start + 8 * kg + j;
                if (gs < grouped_rows && row < M)
#pragma unroll
                    for (int g = 0; g < A8_G; ++g)
                        A8_DBG_C(Y + (long long)grouped_rows * M, ((long long)g * grouped_rows + gs) * M + row, 0);
            }
        }
    }
#endif
}

extern "C" __global__ __launch_bounds__(A8_GR_NW * 32) void A8_ENTRY(qwen4_moe_down_a8_grouped, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots, const block_i8_128* __restrict__ xq, float* __restrict__ Y,
    int M, int K, int x_row_div, int grouped_rows, int x_src_rows, int n_exp) {
    a8_dn_grouped_body<false, A8_GR_NW, A8_GR_RG>(
        expert_ptrs, tile_ids, sorted_slots, xq, Y, M, K, x_row_div, grouped_rows, x_src_rows, n_exp);
}
extern "C" __global__ __launch_bounds__(A8_GR_NW * 32) void A8_ENTRY(qwen4_moe_down_a8_grouped_planar, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots, const block_i8_128* __restrict__ xq, float* __restrict__ Y,
    int M, int K, int x_row_div, int grouped_rows, int x_src_rows, int n_exp) {
    a8_dn_grouped_body<true, A8_GR_NW, A8_GR_RG>(
        expert_ptrs, tile_ids, sorted_slots, xq, Y, M, K, x_row_div, grouped_rows, x_src_rows, n_exp);
}
// `_nodz` twins (HIPFIRE_QWEN4_MOE_DOWN_A8_NODZ): the same arithmetic, but NO store for a dead row: a sentinel /
// invalid tile returns before any store and a dead slot (sorted == -1, x row out of range) of a live tile is
// not written. Live rows are byte-identical to the entries above; dead rows keep whatever the buffer held.
// Valid only when every consumer reads live rows through inverse_perm / the combine order (all of them do:
// moe_down_combine_grouped_top10*, hc_row_fold_*; the tile exchange copies tiles verbatim, never read dead).
extern "C" __global__ __launch_bounds__(A8_GR_NW * 32) void A8_ENTRY(qwen4_moe_down_a8_grouped_nodz, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots, const block_i8_128* __restrict__ xq, float* __restrict__ Y,
    int M, int K, int x_row_div, int grouped_rows, int x_src_rows, int n_exp) {
    a8_dn_grouped_body<false, A8_GR_NW, A8_GR_RG, true>(
        expert_ptrs, tile_ids, sorted_slots, xq, Y, M, K, x_row_div, grouped_rows, x_src_rows, n_exp);
}
extern "C" __global__ __launch_bounds__(A8_GR_NW * 32) void A8_ENTRY(qwen4_moe_down_a8_grouped_planar_nodz, A8_SUF, _gfx1201)(
    const unsigned long long* __restrict__ expert_ptrs, const int* __restrict__ tile_ids,
    const int* __restrict__ sorted_slots, const block_i8_128* __restrict__ xq, float* __restrict__ Y,
    int M, int K, int x_row_div, int grouped_rows, int x_src_rows, int n_exp) {
    a8_dn_grouped_body<true, A8_GR_NW, A8_GR_RG, true>(
        expert_ptrs, tile_ids, sorted_slots, xq, Y, M, K, x_row_div, grouped_rows, x_src_rows, n_exp);
}

#endif  // __gfx1201__
