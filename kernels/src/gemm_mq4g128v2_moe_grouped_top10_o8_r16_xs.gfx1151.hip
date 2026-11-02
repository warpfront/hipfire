// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT53 grouped top-10 down O8×R16, X shared through LDS.
//
// Bitwise twin of `gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151` for the
// exact Qwen4 shape M = 2560 (a multiple of 8 * W), K = 640 (five full
// QT53 groups), x_row_div = 1.  Every output element is computed by one wave
// with the incumbent's per-element arithmetic: the same QT53 decode
// (fma(scale, nibble, zero) per row, f16 scale/zero from the group header),
// the same lane -> K map (lane t owns values t*4 .. t*4 + 3 of each 128-value
// group), the same per-group expression evaluated as the incumbent's code
// object evaluates it, t = w1*x1; t = fma(w0, x0, t); t = fma(w2, x2, t);
// t = fma(w3, x3, t); acc += t (spelled out with explicit fmas so the
// surrounding schedule cannot flip the backend's contraction), groups in
// increasing order into one accumulator, and the same hf_lane0_tree_sum2
// reduction and stores.  Semantic corners are the incumbent's: slots without
// an X row (or with an invalid expert) accumulate nothing, skip the tree and
// store +0.0; slots past grouped_rows are never stored.
//
// Only scheduling changes: the incumbent's wave owns eight rows of a 16-slot
// tile and loads that tile's X itself, so every X byte is fetched once per
// eight output rows (about fifteen times the expert-weight traffic).  Here a
// block of W waves owns 8 * W adjacent rows of the same tile; each group's X
// chunk (16 slots x 128 floats) is staged into LDS once per block,
// double-buffered with one LDS-only barrier per group.  W = 4 is the routed
// instantiation (Halo oracle, 5d6ae67e8: W = 4 beat 8 and 16 at every batch).
// Grid [M / (8 * W), ceil(grouped_rows / 16), 1], block 32 * W.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

static __device__ __forceinline__ void hf_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}

// LDS-only workgroup barrier (attention_verify_wmma.gfx1151 idiom): each
// wave waits for its own LDS traffic (lgkmcnt(0), vmcnt left open) and the
// global prefetches stay in flight.  X and the weights are read-only and Y is
// written after the walk.
#define HF_LDS_BARRIER()                                                  \
    do {                                                                  \
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");  \
        __builtin_amdgcn_s_waitcnt(0xFC07);                               \
        __builtin_amdgcn_s_barrier();                                     \
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");  \
    } while (0)

#define HF_ROWS 8
#define HF_GROUPS 5   // K = 640
template <int W>
__device__ __forceinline__ void grouped_down_o8_r16_xs_body(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int x_src_rows,
    int n_exp,
    float4* __restrict__ xs
) {
    constexpr int S = 16 * 32 / (32 * W);  // float4 staged per thread per group
    static_assert(S >= 1 && S * W == 16, "X chunk split");
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int tid = threadIdx.x & 31;
    const int row0 = (blockIdx.x * W + wave) * HF_ROWS;
    const int tile = blockIdx.y;
    const int slot_start = tile * 16;
    // Block-uniform exit only: every wave takes part in the barriers.
    if (slot_start >= grouped_rows) return;

    int flat_idx[16];
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_idx[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }
    const int expert = expert_tile_ids[tile];
    asm volatile("" :: "s"(flat_idx[0]), "s"(flat_idx[1]), "s"(flat_idx[2]),
                 "s"(flat_idx[3]), "s"(flat_idx[4]), "s"(flat_idx[5]),
                 "s"(flat_idx[6]), "s"(flat_idx[7]), "s"(flat_idx[8]),
                 "s"(flat_idx[9]), "s"(flat_idx[10]), "s"(flat_idx[11]),
                 "s"(flat_idx[12]), "s"(flat_idx[13]), "s"(flat_idx[14]),
                 "s"(flat_idx[15]), "s"(expert));
    float acc[HF_ROWS][16];
    bool x_ok[16];
    int x_soff[16];
    bool row_valid[HF_ROWS];
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
#pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) acc[r][slot] = 0.0f;
        x_ok[slot] = false;
        x_soff[slot] = 0;
    }
#pragma unroll
    for (int r = 0; r < HF_ROWS; ++r) row_valid[r] = (row0 + r) < M;

    if (expert >= 0 && expert < n_exp && K > 0 && x_row_div > 0) {
#pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            const int flat = flat_idx[slot];
            if (flat < 0) continue;
            const int x_row = (x_row_div == 1) ? flat : flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows) {
                x_ok[slot] = true;
                x_soff[slot] = x_row * K * 4;
            }
        }

        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int row_stride = HF_GROUPS * 68;
        int row_soff[HF_ROWS];
#pragma unroll
        for (int r = 0; r < HF_ROWS; ++r)
            row_soff[r] = (row_valid[r] ? (row0 + r) : (M - 1)) * row_stride;
        const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)A, 0, M * row_stride, 0x31004000);
        const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)X_src, 0, x_src_rows * K * 4, 0x31004000);
        // Staging: float4 v = j * 32W + threadIdx.x of the 16 x 32 chunk
        // belongs to slot j * W + wave (wave-uniform), float4 tid within it.
        int stage_soff[S];
#pragma unroll
        for (int j = 0; j < S; ++j) {
            int s = x_soff[j * W];
#pragma unroll
            for (int q = 1; q < W; ++q) s = (wave == q) ? x_soff[j * W + q] : s;
            stage_soff[j] = s;
        }
        const int stage_voff = tid * 16;

        unsigned int header[HF_ROWS];
        unsigned short payload[HF_ROWS];
        float4 st[S];
        auto load = [&](int group) {
#pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                const int goff = row_soff[r] + group * 68;
                header[r] = *(const __attribute__((address_space(4))) unsigned int*)(
                    (const __attribute__((address_space(4))) char*)expert_ptrs[expert] + goff);
                payload[r] = __builtin_amdgcn_raw_buffer_load_b16(wrsrc, 4 + tid * 2, goff, 0);
            }
#pragma unroll
            for (int j = 0; j < S; ++j)
                st[j] = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128(
                    xrsrc, stage_voff, stage_soff[j] + group * 512, 0));
        };

        load(0);
#pragma unroll
        for (int group = 0; group < HF_GROUPS; ++group) {
            float4* __restrict__ xg = xs + (group & 1) * (16 * 32);
#pragma unroll
            for (int j = 0; j < S; ++j) xg[j * 32 * W + threadIdx.x] = st[j];
            float scale[HF_ROWS], zero[HF_ROWS];
            float n0[HF_ROWS], n1[HF_ROWS], n2[HF_ROWS], n3[HF_ROWS];
#pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                scale[r] = __half2float(__ushort_as_half((unsigned short)(header[r] & 0xFFFFU)));
                zero[r] = __half2float(__ushort_as_half((unsigned short)(header[r] >> 16)));
                unsigned int lo, hi;
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(lo) : "v"((unsigned int)payload[r]));
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(hi) : "v"((unsigned int)payload[r] >> 4));
                n0[r] = (float)(lo & 0xFFU);
                n1[r] = (float)(hi & 0xFFU);
                n2[r] = (float)(lo >> 8);
                n3[r] = (float)(hi >> 8);
            }
            // Buffer group & 1 was last read in group - 2; every wave passed
            // the previous group's barrier after that.
            HF_LDS_BARRIER();
            if (group + 1 < HF_GROUPS) load(group + 1);
            float w0[HF_ROWS], w1[HF_ROWS], w2[HF_ROWS], w3[HF_ROWS];
#pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                w0[r] = __builtin_fmaf(scale[r], n0[r], zero[r]);
                w1[r] = __builtin_fmaf(scale[r], n1[r], zero[r]);
                w2[r] = __builtin_fmaf(scale[r], n2[r], zero[r]);
                w3[r] = __builtin_fmaf(scale[r], n3[r], zero[r]);
            }
#pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                if (!x_ok[slot]) continue;
                const float4 xv = xg[slot * 32 + tid];
#pragma unroll
                for (int r = 0; r < HF_ROWS; ++r) {
                    float t = w1[r] * xv.y;
                    t = __builtin_fmaf(w0[r], xv.x, t);
                    t = __builtin_fmaf(w2[r], xv.z, t);
                    t = __builtin_fmaf(w3[r], xv.w, t);
                    acc[r][slot] += t;
                }
            }
        }
    }

#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        if (!x_ok[slot]) continue;
#pragma unroll
        for (int r = 0; r < HF_ROWS; r += 2) hf_lane0_tree_sum2(acc[r][slot], acc[r + 1][slot]);
    }
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
#pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] = acc[r][slot];
            }
        }
    }
}

#define HF_DOWN_XS_KERNEL(NAME, W) \
__launch_bounds__(32 * (W), 1) \
extern "C" __global__ void NAME( \
    const unsigned long long* __restrict__ expert_ptrs, \
    const int* __restrict__ expert_tile_ids, \
    const int* __restrict__ sorted_slot_index, \
    const float* __restrict__ X_src, \
    float* __restrict__ Y_grouped, \
    int M, \
    int K, \
    int x_row_div, \
    int grouped_rows, \
    int x_src_rows, \
    int n_exp \
) { \
    __shared__ float4 xs[2 * 16 * 32]; \
    grouped_down_o8_r16_xs_body<(W)>(expert_ptrs, expert_tile_ids, sorted_slot_index, X_src, \
        Y_grouped, M, K, x_row_div, grouped_rows, x_src_rows, n_exp, xs); \
}

HF_DOWN_XS_KERNEL(gemm_mq4g128v2_moe_grouped_top10_o8_r16_xs4_gfx1151, 4)
#undef HF_GROUPS
#undef HF_ROWS
