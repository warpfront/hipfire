// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11 wave32 WMMA (ArchCaps::has_wmma_w32); tuned on gfx1151.
#if defined(__GFX11__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_p_t;
typedef unsigned int __attribute__((ext_vector_type(8))) uint8_p_t;
// Weights 2j, 2j+1 of nibble words (lo, hi) as sc*q + zp in F16.
__device__ __forceinline__ half16_t dq_nibbles(unsigned int lo, unsigned int hi,
                                              _Float16 sc, _Float16 zp) {
    // Byte p of a word holds weights 2p (low nibble) and 2p + 1 (high
    // nibble).  One byte permute copies byte p into bytes 0 and 2, one and-or
    // keeps the low nibble at bits 0-3 of half 0 (0x6400 | q is exactly
    // 1024 + q) and the high nibble at bits 4-7 of half 1 (0x5400 | q << 4 is
    // exactly 64 + q), so subtracting {1024, 64} yields both codes exactly and
    // one fused multiply-add per pair is the per-element sc * q + zp.
    const half2_p_t sc2 = {sc, sc};
    const half2_p_t zp2 = {zp, zp};
    const half2_p_t bias = {(_Float16)1024.0f, (_Float16)64.0f};
    uint8_p_t out;
#pragma unroll
    for (int p = 0; p < 8; p++) {
        const unsigned int w = p < 4 ? lo : hi;
        const unsigned int b = (unsigned)(p & 3);
        const unsigned int bits =
            (__builtin_amdgcn_perm(w, w, b | (12u << 8) | (b << 16) | (12u << 24)) & 0x00F0000Fu)
            | 0x54006400u;
        const half2_p_t q2 = __builtin_bit_cast(half2_p_t, bits) - bias;
        out[p] = __builtin_bit_cast(unsigned int, __builtin_elementwise_fma(sc2, q2, zp2));
    }
    return __builtin_bit_cast(half16_t, out);
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat

// Grouped WMMA GEMM for MoE down over MQ4G128V2 (qt=53) routed experts:
// gemm_mq4g256v2_moe_grouped_wmma_k2 (PR #775 gate/up arm) with the 68 B
// single-header group ([0..2) f16 scale, [2..4) f16 zero, [4..68) 64 B
// nibbles: byte i holds weight 2i low / 2i+1 high).  Eight K-tiles per group.
// F16 dequant and F16 inputs: not bit-exact with the F32 SIMT kernels.
// K % 128 == 0.  Same expert-tile grouping (up to four tiles per
// wave) and unrolled K-tile loop as the gate/up kernel.  Grid [ceil(M/16), m_total/16], block 32.
// F32 -> BF16 bits, round to nearest even; non-finite values truncate (the
// BF16 round trip leaves them unchanged).  Widening the bits back gives the
// value moe_bf16_round_trip / hip_bfloat16 rounding yields for finite input.
__device__ __forceinline__ unsigned short moe_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

template <int NT, bool OUT_BF16>
__device__ __forceinline__ void moe_down_wmma_tiles(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int tid, int row_start, int slot_start,
    unsigned int* __restrict__ a_lds, int nlive) {
    const int lane = tid & 15;
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const _Float16* x_base[NT] = {};
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int slot_idx = slot_start + 16 * t + lane;
        if (t < nlive && slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0)
                x_base[t] = X_src + (long long)((x_row_div > 1) ? (flat / x_row_div) : flat) * K;
        }
    }
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 68;
    float8_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    // The 16 rows' groups are staged in LDS one 68-byte group at a time,
    // double-buffered: group g+1's 16 x 68 bytes (272 dwords, nine per lane)
    // load while group g computes, so each wave overlaps its own weight
    // stream (staging the whole 16-row tile first left every wave waiting
    // on it before any compute).
    const int rows_live = min(16, M - row_start);
    const char* Ablk = A + (long long)row_start * row_bytes;
    unsigned int pf[9];
    auto load_group = [&](int g) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int d = min(tid + 32 * i, 16 * 17 - 1);
            const int r = min(d / 17, rows_live - 1), w = d - (d / 17) * 17;
            pf[i] = *(const unsigned int*)(Ablk + r * row_bytes + g * 68 + w * 4);
        }
    };
    auto store_group = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int d = min(tid + 32 * i, 16 * 17 - 1);
            a_lds[buf * 16 * 17 + d] = pf[i];
        }
    };
    load_group(0);
    store_group(0);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    const int my_row = min(lane, rows_live - 1);
    for (int g = 0; g < groups_per_row; g++) {
        if (g + 1 < groups_per_row) load_group(g + 1);
        const unsigned int* gw = a_lds + (g & 1) * 16 * 17 + my_row * 17;
        const unsigned int hdr = gw[0];
        const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
        const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
#pragma unroll (NT == 4 ? 2 : 4)  // full unroll spills at NT == 4
        for (int kt = 0; kt < 8; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            const half16_t a_a = dq_nibbles(gw[1 + k_off_a / 8], gw[2 + k_off_a / 8], sc_h, zp_h);
            const half16_t a_b = dq_nibbles(gw[1 + k_off_b / 8], gw[2 + k_off_b / 8], sc_h, zp_h);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                half16_t b_a = zero_b, b_b = zero_b;
                if (x_base[t] != nullptr) {
                    b_a = *(const half16_t*)(x_base[t] + g * 128 + k_off_a);
                    b_b = *(const half16_t*)(x_base[t] + g * 128 + k_off_b);
                }
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
            }
        }
        if (g + 1 < groups_per_row) {
            store_group((g + 1) & 1);
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        }
    }
    // Lane l < 16 holds rows 2j of column l, lane l + 16 rows 2j + 1.  One
    // cross-half swap gives the low lane rows 0..7 and the high lane rows
    // 8..15 of its column, stored as one 16-byte BF16 run (or two 16-byte
    // F32 runs) instead of eight scattered 2-byte stores.
    const int hi = tid >> 4;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float give = hi ? acc[t][i] : acc[t][4 + i];
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
            const float own = hi ? acc[t][4 + i] : acc[t][i];
            r[2 * i] = hi ? got : own;
            r[2 * i + 1] = hi ? own : got;
        }
        const int out_col = slot_start + 16 * t + lane;
        if (t >= nlive) continue;
        const int row0 = row_start + 8 * hi;
        if (out_col >= m_total || row0 >= M) continue;
        const long long o = (long long)out_col * M + row0;
        if (row0 + 8 > M) {
            for (int i = 0; row0 + i < M; i++) {
                if (OUT_BF16)
                    ((unsigned short*)Y_grouped)[o + i] = moe_bf16_bits(r[i]);
                else
                    ((float*)Y_grouped)[o + i] = r[i];
            }
        } else {
            if (OUT_BF16) {
                uint4 w;
                w.x = moe_bf16_bits(r[0]) | ((unsigned)moe_bf16_bits(r[1]) << 16);
                w.y = moe_bf16_bits(r[2]) | ((unsigned)moe_bf16_bits(r[3]) << 16);
                w.z = moe_bf16_bits(r[4]) | ((unsigned)moe_bf16_bits(r[5]) << 16);
                w.w = moe_bf16_bits(r[6]) | ((unsigned)moe_bf16_bits(r[7]) << 16);
                *(uint4*)((unsigned short*)Y_grouped + o) = w;
            } else {
                *(float4*)((float*)Y_grouped + o) = make_float4(r[0], r[1], r[2], r[3]);
                *(float4*)((float*)Y_grouped + o + 4) = make_float4(r[4], r[5], r[6], r[7]);
            }
        }
    }
}

template <bool OUT_BF16>
__device__ __forceinline__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    int local = 0;
    while (tile_y - local - 1 >= 0 && expert_tile_ids[tile_y - local - 1] == expert_id) ++local;
    if (local % 4) return;
    int run = 1;
    while (run < 4 && (tile_y + run) * 16 < m_total && expert_tile_ids[tile_y + run] == expert_id) ++run;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    // Declared here, not in the NT-templated body: one allocation, not three.
    __shared__ __align__(16) unsigned int a_lds[2 * 16 * 17];
    // A three-tile run takes the four-tile body with its last tile masked.
    if (run >= 3)
        moe_down_wmma_tiles<4, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, run);
    else if (run == 2)
        moe_down_wmma_tiles<2, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, 2);
    else
        moe_down_wmma_tiles<1, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start, a_lds, 1);
}

#define MOE_WMMA_ARGS \
    const unsigned long long* __restrict__ expert_weight_ptrs, \
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index, \
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped, \
    int M, int K, int x_row_div, int m_total
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151(MOE_WMMA_ARGS) {
    gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry<false>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                         Y_grouped, M, K, x_row_div, m_total);
}
// Y_grouped as BF16 (RNE) for consumers that round each value to BF16 first.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out(MOE_WMMA_ARGS) {
    gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry<true>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                        Y_grouped, M, K, x_row_div, m_total);
}
#undef MOE_WMMA_ARGS

// Eight row-tile waves (128 rows) of one expert run share its X through LDS,
// 64 K (half a group) at a time: the per-wave kernel above re-reads every
// tile's X from L2 in each of M/16 waves.  Each wave keeps its own
// double-buffered weight staging.  The next half group's X (and at a group
// boundary the next group's weights) are loaded before the current one
// computes from LDS, so the compute waits on no global load (vmcnt retires in
// order).  Every tile's WMMA sequence, operands and epilogue are
// moe_down_wmma_tiles', so outputs are bitwise identical.
// Grid [M / 128, m_total / 16], block 256.  M % 128 == 0, K % 128 == 0.
template <int NT, bool OUT_BF16>
__device__ __forceinline__ void moe_down_wmma_tiles_x8(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int m_total, int tid, int slot_start,
    unsigned int* __restrict__ a_lds, _Float16* __restrict__ xs, int nlive) {
    constexpr int WAVES = 8;
    constexpr int XS = 72;                   // halves per staged slot row (64 + 8)
    constexpr int NTH = 32 * WAVES;
    constexpr int PIECES = NT * 16 * 8;      // 16 B pieces of a half group's X
    constexpr int PER = (PIECES + NTH - 1) / NTH;
    const int wt = tid & 31;
    const int lane = wt & 15;
    const int row_start = (blockIdx.x * WAVES + (tid >> 5)) * 16;
    const _Float16* src[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int p = tid + NTH * i;
        const int s = p >> 3;
        const int slot_idx = slot_start + s;
        src[i] = nullptr;
        if (p < PIECES && (s >> 4) < nlive && slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0) src[i] = X_src + (long long)flat * K + (p & 7) * 8;
        }
    }
    uint4 xp[PER];
    auto load_x = [&](int hg) {
#pragma unroll
        for (int i = 0; i < PER; i++)
            xp[i] = src[i] ? *(const uint4*)(src[i] + hg * 64) : make_uint4(0, 0, 0, 0);
    };
    auto store_x = [&]() {
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int p = tid + NTH * i;
            if (p < PIECES) *(uint4*)(xs + (p >> 3) * XS + (p & 7) * 8) = xp[i];
        }
    };
    const int groups_per_row = K / 128;
    const int row_bytes = groups_per_row * 68;
    float8_t acc[NT];
    for (int t = 0; t < NT; t++) acc[t] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    const int rows_live = min(16, M - row_start);
    const char* Ablk = A + (long long)row_start * row_bytes;
    unsigned int pf[9];
    auto load_group = [&](int g) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int d = min(wt + 32 * i, 16 * 17 - 1);
            const int r = min(d / 17, rows_live - 1), w = d - (d / 17) * 17;
            pf[i] = *(const unsigned int*)(Ablk + r * row_bytes + g * 68 + w * 4);
        }
    };
    auto store_group = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            const int d = min(wt + 32 * i, 16 * 17 - 1);
            a_lds[buf * 16 * 17 + d] = pf[i];
        }
    };
    // LDS-only barriers: __syncthreads would also wait for the in-flight
    // next-stage global loads (WGP-mode workgroup fence).
#define MOE_X8_BARRIER() asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory")
    load_group(0);
    load_x(0);
    store_group(0);
    store_x();
    MOE_X8_BARRIER();
    const int my_row = min(lane, rows_live - 1);
    const int halves = 2 * groups_per_row;
    for (int hg = 0; hg < halves; hg++) {
        const int g = hg >> 1, hh = hg & 1;
        if (hg + 1 < halves) {
            if (hh) load_group(g + 1);
            load_x(hg + 1);
        }
        const unsigned int* gw = a_lds + (g & 1) * 16 * 17 + my_row * 17;
        const unsigned int hdr = gw[0];
        const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
        const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
#pragma unroll
        for (int kq = 0; kq < 4; kq += 2) {
            const int kt = hh * 4 + kq;
            const half16_t a_a = dq_nibbles(gw[1 + kt * 2], gw[2 + kt * 2], sc_h, zp_h);
            const half16_t a_b = dq_nibbles(gw[1 + (kt + 1) * 2], gw[2 + (kt + 1) * 2], sc_h, zp_h);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const _Float16* xr = xs + (t * 16 + lane) * XS;
                const half16_t b_a = *(const half16_t*)(xr + kq * 16);
                const half16_t b_b = *(const half16_t*)(xr + (kq + 1) * 16);
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
            }
        }
        // WMMA has no memory chain: keep the WMMAs ahead of the barrier.
#pragma unroll
        for (int t = 0; t < NT; t++) asm volatile("" : "+v"(acc[t]));
        if (hg + 1 < halves) {
            MOE_X8_BARRIER();  // every wave is done reading this half group's X
            if (hh) store_group((g + 1) & 1);
            store_x();
            MOE_X8_BARRIER();
        }
    }
#undef MOE_X8_BARRIER
    const int hi = wt >> 4;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        float r[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float give = hi ? acc[t][i] : acc[t][4 + i];
            const float got = __int_as_float(__builtin_amdgcn_permlanex16(
                __float_as_int(give), __float_as_int(give), 0x76543210u, 0xfedcba98u, false, false));
            const float own = hi ? acc[t][4 + i] : acc[t][i];
            r[2 * i] = hi ? got : own;
            r[2 * i + 1] = hi ? own : got;
        }
        const int out_col = slot_start + 16 * t + lane;
        if (t >= nlive) continue;
        const int row0 = row_start + 8 * hi;
        if (out_col >= m_total || row0 >= M) continue;
        const long long o = (long long)out_col * M + row0;
        if (row0 + 8 > M) {
            for (int i = 0; row0 + i < M; i++) {
                if (OUT_BF16)
                    ((unsigned short*)Y_grouped)[o + i] = moe_bf16_bits(r[i]);
                else
                    ((float*)Y_grouped)[o + i] = r[i];
            }
        } else if (OUT_BF16) {
            uint4 w;
            w.x = moe_bf16_bits(r[0]) | ((unsigned)moe_bf16_bits(r[1]) << 16);
            w.y = moe_bf16_bits(r[2]) | ((unsigned)moe_bf16_bits(r[3]) << 16);
            w.z = moe_bf16_bits(r[4]) | ((unsigned)moe_bf16_bits(r[5]) << 16);
            w.w = moe_bf16_bits(r[6]) | ((unsigned)moe_bf16_bits(r[7]) << 16);
            *(uint4*)((unsigned short*)Y_grouped + o) = w;
        } else {
            *(float4*)((float*)Y_grouped + o) = make_float4(r[0], r[1], r[2], r[3]);
            *(float4*)((float*)Y_grouped + o + 4) = make_float4(r[4], r[5], r[6], r[7]);
        }
    }
}

// Y_grouped as BF16 (RNE), eight row tiles per workgroup (see above).
__launch_bounds__(256, 1)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151_x8_bf16out(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total) {
    (void)x_row_div;
    const int tid = threadIdx.x;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    int local = 0;
    while (tile_y - local - 1 >= 0 && expert_tile_ids[tile_y - local - 1] == expert_id) ++local;
    if (local % 4) return;
    int run = 1;
    while (run < 4 && (tile_y + run) * 16 < m_total && expert_tile_ids[tile_y + run] == expert_id) ++run;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    __shared__ __align__(16) unsigned int a_lds[8][2 * 16 * 17];
    __shared__ __align__(16) _Float16 xs[4 * 16 * 72];
    unsigned int* my_lds = a_lds[tid >> 5];
    if (run == 4)
        moe_down_wmma_tiles_x8<4, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, m_total, tid, slot_start, my_lds, xs, 4);
    else if (run == 3)
        moe_down_wmma_tiles_x8<3, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, m_total, tid, slot_start, my_lds, xs, 3);
    else if (run == 2)
        moe_down_wmma_tiles_x8<2, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, m_total, tid, slot_start, my_lds, xs, 2);
    else
        moe_down_wmma_tiles_x8<1, true>(A, sorted_slot_index, X_src, Y_grouped, M, K, m_total, tid, slot_start, my_lds, xs, 1);
}

#endif
