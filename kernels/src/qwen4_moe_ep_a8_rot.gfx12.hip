// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 Flash-Next EP2 wide-verify (9..=64 source rows) gate/up A8 GEMM with the down producer folded into its
// epilogue (`HIPFIRE_QWEN4_MOE_GLUE_FOLD` sub-fold `c`, X8c): ONE launch in place of
// `qwen4_moe_ep_a8_gate_up_planar_nt{1,2}` followed by `qwen4_moe_rotate128_i8`.
//
// The host concatenates `block_i8_128_quant.hip` (HIPFIRE_BLOCK_I8_128_QUANT_NO_STANDALONE) and
// `qwen4_moe_ep_a8.gfx12.hip` in front, so every helper below (`ep8_wrow_of`, `ep8_load_a`, `ep8_load_x`,
// `ep8_epoch`, `ep8_bf16_rt`, `ep8_bf16_bits`) is the unchanged legacy one, and `quantize_block_i8_128_wave` is
// the shared quantizer.
//
// A legacy gate/up CTA owns 64 of the 128 columns of one rotate group, so this CTA is 8 waves (block 256), each
// wave exactly the legacy wave: 16 gate rows plus the matching 16 up rows, the same per-output K128 epoch
// chain (`ep8_epoch`), 128 gate columns per CTA (`blockIdx.x` = rotate group, grid x = M/2 / 128). The SwiGLU
// rows (the BF16 values the legacy epilogue stores to `Y`) of the CTA's NT tiles x 128 columns go to LDS
// instead; after a barrier every wave runs the `qwen4_moe_rotate128_i8` per-row body on rows
// q = wave, wave + 8, ... of the live slots (sign1 -> radix-2 FWHT128 with the same ds_swizzle butterflies ->
// *0.0883883476f -> sign2 -> `quantize_block_i8_128_wave`) and stores the record to
// `Xdown[group * L + slot]`, L = total flat slots (`block_i8_128` `[K_down/128, L]`, the layout
// `qwen4_moe_ep_a8_down_planar_nt{1,2}` reads with x_row_div 1). Rows of dead slots (sorted < 0) emit nothing,
// exactly as the standalone rotate. Sentinel / non-leader tiles return before any load or store (the standalone
// rotate would have written stale-derived records of masked tiles' slots; the down GEMM never reads them).
// The BF16 `Y` rows are not written: nothing on this route reads them.
//
// Kernarg ABI (7 pointers, 7 i32):
//   (expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Xdown, signs1, signs2, M, K, x_row_div, m_total,
//    x_src_rows, n_experts, L)
//   Xq: gate sidecar `[K/128, x_src_rows]` (X row of slot s = s / x_row_div); Xdown: `[M/2/128, L]`;
//   signs1 / signs2: the 128-entry F32 sign tables of `qwen4_moe_rotate128_i8`.
// Grid `(M/2 / 128, m_total / 16)`, block 256. Planar experts only (the served route).

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_moe_ep_a8_rot.gfx12.hip is gfx1201-only"
#endif

// Row stride (bf16 elements) of the staged SwiGLU rows: 136 x 2 B = 272 B, a 16 B multiple (uint4 stores) that
// spreads the 16 lane rows of one wave's store over distinct banks.
#define EP8R_LD 136

template <int NT, bool FULL>
__device__ __forceinline__ void ep8r_body(const ep8_wrow (&wrow)[2], const ep8_block* const (&xrow)[NT],
                                          const bool (&live)[NT], unsigned short* __restrict__ stage, int ntv,
                                          int wave, int K, size_t x_src_rows, int lr, int hi) {
    constexpr int NF = 2;
    float sum[NF][NT][8];
#pragma unroll
    for (int f = 0; f < NF; ++f)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[f][i][j] = 0.0f;

    const int epochs = K >> 7;
    ep8_aop a0[NF], a1[NF];
    ep8_xop x0[NT], x1[NT];
#define EP8R_LOAD(h, A, X)                                                                              \
    do {                                                                                                \
        _Pragma("unroll") for (int i = 0; i < NT; ++i) {                                                \
            if (FULL || i < ntv) ep8_load_x(X[i], xrow[i], live[i], (h), x_src_rows, hi);               \
        }                                                                                               \
        _Pragma("unroll") for (int f = 0; f < NF; ++f) ep8_load_a<false, true>(A[f], wrow[f], (h), hi); \
    } while (0)
    EP8R_LOAD(0, a0, x0);
    for (int h = 0; h < epochs; h += 2) {
        const bool two = (h + 1) < epochs;
        if (two) EP8R_LOAD(h + 1, a1, x1);
        ep8_epoch<NF, NT, FULL>(sum, a0, x0, ntv, hi);
        if (two) {
            if (h + 2 < epochs) EP8R_LOAD(h + 2, a0, x0);
            ep8_epoch<NF, NT, FULL>(sum, a1, x1, ntv, hi);
        }
    }
#undef EP8R_LOAD

    // The legacy SwiGLU epilogue, stored to LDS (column wave * 16 + 8 hi + j of the CTA's 128-column group).
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        if (!FULL && i >= ntv) break;
        unsigned short o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float v = ep8_bf16_rt(sum[0][i][j]);
            const float w = ep8_bf16_rt(sum[NF - 1][i][j]);
            const float r = live[i] ? ep8_bf16_rt((v / (1.0f + expf(-v))) * w) : 0.0f;
            o[j] = ep8_bf16_bits(r);
        }
        uint4 pk;
        pk.x = (unsigned)o[0] | ((unsigned)o[1] << 16);
        pk.y = (unsigned)o[2] | ((unsigned)o[3] << 16);
        pk.z = (unsigned)o[4] | ((unsigned)o[5] << 16);
        pk.w = (unsigned)o[6] | ((unsigned)o[7] << 16);
        *(uint4*)(stage + (i * 16 + lr) * EP8R_LD + wave * 16 + 8 * hi) = pk;
    }
}

template <int NT>
__device__ __forceinline__ void ep8r_moe(const unsigned long long* __restrict__ expert_weight_ptrs,
                                         const int* __restrict__ expert_tile_ids,
                                         const int* __restrict__ sorted_slot_index,
                                         const ep8_block* __restrict__ Xq, ep8_block* __restrict__ Xdown,
                                         const float* __restrict__ signs1, const float* __restrict__ signs2, int M,
                                         int K, int x_row_div, int m_total, int x_src_rows, int n_experts, int L) {
    __shared__ __attribute__((aligned(16))) unsigned short stage[NT * 16 * EP8R_LD];

    // Tile selection: the legacy ep8_moe prologue, uniform over the CTA (all returns precede the barrier).
    const int tile = blockIdx.y;
    if (tile * 16 >= m_total) return;
    const int e = expert_tile_ids[tile];
    if (e < 0 || e >= n_experts) return;
    if ((tile % NT) != 0 && expert_tile_ids[tile - 1] == e) return;  // not a leader
    const int tiles_total = m_total >> 4;
    int ntv = 1;
    while (ntv < NT && ((tile + ntv) % NT) != 0 && tile + ntv < tiles_total && expert_tile_ids[tile + ntv] == e) ++ntv;

    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, lr = lane & 15, hi = lane >> 4;
    const int group = blockIdx.x;
    const int rbase = group * 128 + wave * 16;
    const unsigned char* W = (const unsigned char*)expert_weight_ptrs[e];
    ep8_wrow wrow[2];
    wrow[0] = ep8_wrow_of<false, true>(W, rbase + lr, M, K);
    wrow[1] = ep8_wrow_of<false, true>(W, rbase + lr + (M >> 1), M, K);

    const ep8_block* xrow[NT];
    bool live[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        int slot = -1;
        if (i < ntv) slot = sorted_slot_index[(tile + i) * 16 + lr];
        live[i] = slot >= 0;
        xrow[i] = Xq + (live[i] ? slot / x_row_div : 0);
    }
    if (ntv == NT) {
        ep8r_body<NT, true>(wrow, xrow, live, stage, ntv, wave, K, (size_t)x_src_rows, lr, hi);
    } else {
        ep8r_body<NT, false>(wrow, xrow, live, stage, ntv, wave, K, (size_t)x_src_rows, lr, hi);
    }
    __syncthreads();

    // The `qwen4_moe_rotate128_i8` per-row body, lane = its `tid`, over the staged rows.
    const int d0 = lane * 4;
    const float s1_0 = signs1[d0], s1_1 = signs1[d0 + 1], s1_2 = signs1[d0 + 2], s1_3 = signs1[d0 + 3];
    const float s2_0 = signs2[d0], s2_1 = signs2[d0 + 1], s2_2 = signs2[d0 + 2], s2_3 = signs2[d0 + 3];
    const int tid = lane;
    for (int q = wave; q < NT * 16; q += 8) {
        const int i = q >> 4;
        if (i >= ntv) break;
        const int slot = sorted_slot_index[(tile + i) * 16 + (q & 15)];
        if (slot < 0) continue;
        const uint2 raw = *(const uint2*)(stage + q * EP8R_LD + d0);
        float v0 = __uint_as_float(raw.x << 16) * s1_0;
        float v1 = __uint_as_float(raw.x & 0xFFFF0000u) * s1_1;
        float v2 = __uint_as_float(raw.y << 16) * s1_2;
        float v3 = __uint_as_float(raw.y & 0xFFFF0000u) * s1_3;

        float t;
        t = v0; v0 = v0 + v1; v1 = t - v1;
        t = v2; v2 = v2 + v3; v3 = t - v3;
        t = v0; v0 = v0 + v2; v2 = t - v2;
        t = v1; v1 = v1 + v3; v3 = t - v3;

#define HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (tid & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define HBFLY4(pat, str) \
    HBFLY(v0, pat, str); HBFLY(v1, pat, str); HBFLY(v2, pat, str); \
    HBFLY(v3, pat, str)
        HBFLY4(0x041F, 1);
        HBFLY4(0x081F, 2);
        HBFLY4(0x101F, 4);
        HBFLY4(0x201F, 8);
        HBFLY4(0x401F, 16);
#undef HBFLY4
#undef HBFLY

        const float scale = 0.0883883476f;
        float xv[4];
        xv[0] = v0 * scale * s2_0;
        xv[1] = v1 * scale * s2_1;
        xv[2] = v2 * scale * s2_2;
        xv[3] = v3 * scale * s2_3;
        quantize_block_i8_128_wave(xv, (block_i8_128*)(Xdown + (size_t)group * (size_t)L + slot), tid);
    }
}

#define EP8R_ENTRY(NAME, NT)                                                                                       \
    extern "C" __global__ __launch_bounds__(256) void NAME(                                                       \
        const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,      \
        const int* __restrict__ sorted_slot_index, const ep8_block* __restrict__ Xq,                              \
        ep8_block* __restrict__ Xdown, const float* __restrict__ signs1, const float* __restrict__ signs2,       \
        int M, int K, int x_row_div, int m_total, int x_src_rows, int n_experts, int L) {                         \
        ep8r_moe<NT>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Xdown, signs1, signs2, M, K,    \
                     x_row_div, m_total, x_src_rows, n_experts, L);                                               \
    }

EP8R_ENTRY(qwen4_moe_ep_a8_gate_up_rot_planar_nt1, 1)
EP8R_ENTRY(qwen4_moe_ep_a8_gate_up_rot_planar_nt2, 2)
