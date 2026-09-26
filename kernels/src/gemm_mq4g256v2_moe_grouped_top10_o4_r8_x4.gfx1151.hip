// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT44 grouped top-10 gate/up O4×R8 with X in LDS.
//
// This exact-shape gfx1151 path keeps the static production kernel's F32 input,
// MQ4G256V2 decoding, four static accumulator chains per output row, and full
// wave32 reduction.  Four waves per block each own four adjacent output rows
// of one eight-slot subtile; the subtile's X is staged through LDS once per
// block, and each decoded weight vector serves eight slots.
//
// Layouts are the live grouped gate/up contract:
//   expert_weight_ptrs: [E] device pointers to [M, K] MQ4G256V2 rows
//   expert_tile_ids:   [ceil(grouped_rows / 16)] expert ID per 16-slot tile
//   sorted_slot_index: [grouped_rows] flat token/top-k slot, -1 for padding
//   X_src:             [N, K] F32 rotated token rows
//   Y_grouped:         [grouped_rows, M] F32 grouped output
//
// Arithmetic invariants (per output row and slot):
//   * Groups are processed in the complete-quad/tail order of the O2×R8 and
//     O1×R16 kernels: group g accumulates into static chain g & 3 with the
//     8-term expression w0*x0 + ... + w7*x7.  The final association is
//     (acc0 + acc1) + (acc2 + acc3), followed by lane 0 of the full-wave
//     shuffle tree (offsets 16, 8, 4, 2, 1).
//   * A subtile with no live slot accumulates nothing, so eliding its walk
//     leaves every accumulator at zero and the epilogue writes exactly the
//     zeros the full walk wrote.  Dead slots skip accumulation and store +0.0.
//   * Rows past M read row M - 1 and are never stored.  A negative expert ID
//     leaves the output sentinel untouched.  A partial final subtile writes
//     only grouped_rows-valid slots.
// The max-ILP machine scheduler interleaves the 32 accumulator chains better
// (~4% on the routed 1131-token shape); scheduling cannot change results.
// HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Two lane-0 trees in one pass: lanes 0-15 carry a, lanes 16-31 carry b.
// After the cross-row step lane i holds a[i] + a[i+16] and lane 16+i holds
// b[16+i] + b[i] (the same sum, IEEE add commutes); the in-row DPP steps then
// reduce both rows with the reference offsets, so a ends in lane 0 and b in
// lane 16 bit-identical to lane 0 of the `v += __shfl_down(v, off, 32)` tree
// (off = 16..1) applied to each.  a is returned in
// lane 0; b is broadcast (readlane 16).
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
// x & 0x0F0F0F0F, opaque to the optimizer so each nibble converts with one
// v_cvt_f32_ubyteN instead of being re-canonicalized into v_bfe + cvt.
static __device__ __forceinline__ unsigned int hf_nibble_bytes(unsigned int x) {
    unsigned int r;
    asm("v_and_b32 %0, 0x0f0f0f0f, %1" : "=v"(r) : "v"(x));
    return r;
}
// ---------------------------------------------------------------------------
// O4 x R8 with X shared through LDS.  Four waves per block own adjacent
// four-row blocks (16 rows) of one eight-slot subtile.  Each group's X chunk
// (8 slots x 256 floats) is staged once per block into LDS instead of every
// wave pulling it through the vector memory path, which frees the X registers
// for eight slots per wave: each decoded weight vector now serves eight slots
// (half the weight re-reads of the four-slot subtiles).  Per output element
// the decode, the 8-term group expression, the static four-chain split, the
// (c0 + c1) + (c2 + c3) fold and the lane-0 tree are the production ones.
// Grid: [ceil(M / 16), ceil(grouped_rows / 8), 1], block 128.
// expert_tile_ids is indexed per 16-slot tile: subtile s reads tile s / 2.
// ---------------------------------------------------------------------------
#define HF_R8 8
__launch_bounds__(128, 1)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows
) {
    // readfirstlane: the compiler cannot see that threadIdx.x >> 5 is
    // wave-uniform, and a VGPR row offset turns every buffer load into a
    // waterfall loop.
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int tid = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + wave) * 4;
    const int slot_start = blockIdx.y * HF_R8;
    // Block-uniform exits only: every wave takes part in the barriers.
    if (slot_start >= grouped_rows) return;

    int flat_raw[HF_R8];
#pragma unroll
    for (int slot = 0; slot < HF_R8; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_raw[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }
    const int expert_id = expert_tile_ids[blockIdx.y >> 1];
    asm volatile("" :: "s"(flat_raw[0]), "s"(flat_raw[1]), "s"(flat_raw[2]),
                 "s"(flat_raw[3]), "s"(flat_raw[4]), "s"(flat_raw[5]),
                 "s"(flat_raw[6]), "s"(flat_raw[7]));
    if (expert_id < 0) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // Rows past M read row M - 1 and are never stored.
    bool row_valid[4];
    int row_soff[4];
    const int groups_per_row = K / 256;
    const int row_stride = groups_per_row * 136;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        row_valid[r] = row0 + r < M;
        row_soff[r] = (row_valid[r] ? row0 + r : M - 1) * row_stride;
    }

    bool live[HF_R8];
    int x_soff[HF_R8];
    bool any_live = false;
#pragma unroll
    for (int slot = 0; slot < HF_R8; ++slot) {
        const int grouped_slot = slot_start + slot;
        const int flat = (grouped_slot < grouped_rows) ? flat_raw[slot] : -1;
        live[slot] = flat >= 0;
        const int x_row = !live[slot] ? 0
            : (x_row_div > 1) ? (flat / x_row_div) : flat;
        x_soff[slot] = x_row * K * 4;
        any_live |= live[slot];
    }
    const int group_walk = any_live ? groups_per_row : 0;

    const __amdgpu_buffer_rsrc_t wrsrc =
        __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const __amdgpu_buffer_rsrc_t xrsrc =
        __builtin_amdgcn_make_buffer_rsrc((void*)X_src, 0, 0x7FFFFFFF, 0x31004000);
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;

    // Staging: float4 v = j * 128 + threadIdx.x of the 8 x 64 chunk belongs
    // to slot 2j + (wave >> 1) (wave-uniform), float4 (wave & 1) * 32 + tid
    // within it.
    __shared__ float4 xs[HF_R8 * 64];
    const int stage_part = (wave & 1) * 32 + tid;
    int stage_soff[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int s0 = x_soff[2 * j], s1 = x_soff[2 * j + 1];
        stage_soff[j] = (wave >> 1) ? s1 : s0;
    }

    float acc[4][4][HF_R8];  // [chain][row][slot]
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int slot = 0; slot < HF_R8; ++slot) acc[c][r][slot] = 0.0f;

    unsigned int nh[4], npk[4];
    float4 st0, st1, st2, st3;
#define R8_LOAD(G) do { \
        _Pragma("unroll") \
        for (int r = 0; r < 4; ++r) { \
            nh[r] = __builtin_amdgcn_raw_buffer_load_b32( \
                wrsrc, hoff, row_soff[r] + (G) * 136, 0); \
            npk[r] = __builtin_amdgcn_raw_buffer_load_b32( \
                wrsrc, 8 + boff, row_soff[r] + (G) * 136, 0); \
        } \
        st0 = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128( \
            xrsrc, stage_part * 16, stage_soff[0] + (G) * 1024, 0)); \
        st1 = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128( \
            xrsrc, stage_part * 16, stage_soff[1] + (G) * 1024, 0)); \
        st2 = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128( \
            xrsrc, stage_part * 16, stage_soff[2] + (G) * 1024, 0)); \
        st3 = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128( \
            xrsrc, stage_part * 16, stage_soff[3] + (G) * 1024, 0)); \
    } while (0)
#define R8_DECODE(R) \
    float w##R[8]; \
    { \
        const float sc = __half2float(__ushort_as_half( \
            (unsigned short)(ch[R] & 0xFFFFu))); \
        const float zp = __half2float(__ushort_as_half( \
            (unsigned short)(ch[R] >> 16))); \
        const unsigned int lo = hf_nibble_bytes(cpk[R]); \
        const unsigned int hi = hf_nibble_bytes(cpk[R] >> 4); \
        w##R[0] = sc * (float)(lo & 0xFFu) + zp; \
        w##R[1] = sc * (float)(hi & 0xFFu) + zp; \
        w##R[2] = sc * (float)((lo >> 8) & 0xFFu) + zp; \
        w##R[3] = sc * (float)((hi >> 8) & 0xFFu) + zp; \
        w##R[4] = sc * (float)((lo >> 16) & 0xFFu) + zp; \
        w##R[5] = sc * (float)((hi >> 16) & 0xFFu) + zp; \
        w##R[6] = sc * (float)(lo >> 24) + zp; \
        w##R[7] = sc * (float)(hi >> 24) + zp; \
    }
#define R8_ROW(R, C) \
    acc[C][R][slot] += w##R[0] * x0 + w##R[1] * x1 + w##R[2] * x2 + w##R[3] * x3 \
        + w##R[4] * x4 + w##R[5] * x5 + w##R[6] * x6 + w##R[7] * x7
#define R8_GROUP(G, C) do { \
        unsigned int ch[4], cpk[4]; \
        _Pragma("unroll") \
        for (int r = 0; r < 4; ++r) { ch[r] = nh[r]; cpk[r] = npk[r]; } \
        xs[0 * 128 + threadIdx.x] = st0; \
        xs[1 * 128 + threadIdx.x] = st1; \
        xs[2 * 128 + threadIdx.x] = st2; \
        xs[3 * 128 + threadIdx.x] = st3; \
        __syncthreads(); \
        if ((G) + 1 < group_walk) R8_LOAD((G) + 1); \
        R8_DECODE(0) R8_DECODE(1) R8_DECODE(2) R8_DECODE(3) \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_R8; ++slot) { \
            if (!live[slot]) continue; \
            const float4 xa = xs[slot * 64 + tid * 2]; \
            const float4 xb = xs[slot * 64 + tid * 2 + 1]; \
            const float x0 = xa.x, x1 = xa.y, x2 = xa.z, x3 = xa.w; \
            const float x4 = xb.x, x5 = xb.y, x6 = xb.z, x7 = xb.w; \
            R8_ROW(0, C); R8_ROW(1, C); R8_ROW(2, C); R8_ROW(3, C); \
        } \
        /* ALU results carry no memory chain; pin them before the barrier. */ \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_R8; ++slot) \
            asm volatile("" : "+v"(acc[C][0][slot]), "+v"(acc[C][1][slot]), \
                         "+v"(acc[C][2][slot]), "+v"(acc[C][3][slot])); \
        __syncthreads(); \
    } while (0)

    if (group_walk > 0) R8_LOAD(0);
    int g = 0;
    for (; g + 3 < group_walk; g += 4) {
        R8_GROUP(g, 0);
        R8_GROUP(g + 1, 1);
        R8_GROUP(g + 2, 2);
        R8_GROUP(g + 3, 3);
    }
    for (; g < group_walk; ++g) {
        switch (g & 3) {
        case 0: R8_GROUP(g, 0); break;
        case 1: R8_GROUP(g, 1); break;
        case 2: R8_GROUP(g, 2); break;
        default: R8_GROUP(g, 3); break;
        }
    }
#undef R8_GROUP
#undef R8_ROW
#undef R8_DECODE
#undef R8_LOAD

    float sum[4][HF_R8];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int slot = 0; slot < HF_R8; ++slot)
            sum[r][slot] = (acc[0][r][slot] + acc[1][r][slot])
                + (acc[2][r][slot] + acc[3][r][slot]);
#pragma unroll
    for (int slot = 0; slot < HF_R8; ++slot) {
        hf_lane0_tree_sum2(sum[0][slot], sum[1][slot]);
        hf_lane0_tree_sum2(sum[2][slot], sum[3][slot]);
    }
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < HF_R8; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
#pragma unroll
            for (int r = 0; r < 4; ++r)
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] = sum[r][slot];
        }
    }
}
#undef HF_R8
