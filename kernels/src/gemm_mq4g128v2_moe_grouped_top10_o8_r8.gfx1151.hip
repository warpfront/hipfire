// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT53 grouped top-10 down O8×R8 (z-split twin of O8×R16).
//
// This is the O8×R16 kernel (gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151)
// with the slot tile split across two waves. The R16 wave owns eight adjacent
// output rows and sixteen grouped route slots (8 x 16 = 128 accumulators per
// lane). Here one wave owns the same eight rows but only eight slots
// (8 x 8 = 64 accumulators per lane), so register pressure and live X state
// are halved and more waves fit per SIMD; the two waves of a tile are
// scheduled as grid z = 0 and z = 1.
//
// Per-output arithmetic is the R16 text, verbatim: the same QT53 decode, the
// same left-to-right `acc += (scale*n0 + zero)*x0 + ... + (scale*n3 + zero)*x3`
// association (no fast-math, no explicit fused intrinsics), the same float4
// vector path for full groups (K % 4 == 0) and guarded scalar tail group with
// padded values multiplied by zero, the same batch-of-four X load issue, and
// the same hf_lane0_tree_sum2 wave32 reduction (offsets 16, 8, 4, 2, 1). Each
// output element sees exactly the same operations in the same order as in R16,
// so results are bitwise identical to the R16 kernel; only the slot->wave
// assignment changes.
//
// Launch contract (same arguments and output layout as R16):
//   grid = [ceil(M / 8), ceil(grouped_rows / 16), 2], block = [32, 1, 1]
//   expert_tile_ids[blockIdx.y] is the expert of the whole 16-slot tile
//   (-1 for a sentinel tile). blockIdx.z selects the half of that tile:
//   slots tile*16 + z*8 .. tile*16 + z*8 + 7.
//   Slots beyond grouped_rows are padding and are never written; a wave whose
//   first slot is already past grouped_rows returns immediately.
//   Y_grouped[grouped_slot * M + row] is unweighted grouped output.
//   Rows at or beyond M are never written and their accumulator chains stay
//   zero. Their weight loads are clamped to the last valid row of the same
//   expert, so no read leaves the expert matrix.
//
// Semantic corners, all identical to R16:
//   * Every in-range grouped slot is written. A slot with a negative sorted
//     index, an out-of-range X row, a sentinel/out-of-range expert, K <= 0 or
//     x_row_div <= 0 is written as exactly +0.0f.
//   * If none of the wave's eight slots has an X row (a uniform, scalar
//     decision), all weight and X work is skipped and the zeros are written
//     directly; the output is never left unwritten.
//   * Slots without an X row contribute nothing and skip the reduction (their
//     accumulator is still exactly +0.0f).
//
// QT53 layout: each logical K=128 group occupies 68 bytes: fp16 scale,
// fp16 zero, and 64 packed nibbles. The final group is logically padded to
// 128 values, but values at k >= K are multiplied by zero exactly as in the
// production sibling. The caller owns the unchanged BF16 round-trip boundary.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11+ wave32 SIMT ISA (ArchCaps::has_gfx11_plus_simt); tuned on gfx1151.
#if defined(__GFX11__) || defined(__GFX12__)

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

// Eight adjacent output rows per wave: each X vector feeds eight rows.
#define HF_ROWS 8
// Eight grouped route slots per wave (half of the 16-slot expert tile).
#define HF_SLOTS 8
__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_top10_o8_r8_gfx1151(
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
    int n_exp
) {
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * HF_ROWS;
    const int tile = blockIdx.y;
    const int slot_start = tile * 16 + (int)blockIdx.z * HF_SLOTS;
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32) return;

    // Issue all eight slot-index loads with the expert ID, before any is
    // tested (clamped in range): a load behind each `continue` serializes
    // eight dependent scalar round trips into the prologue, and the asm
    // keeps the compiler from sinking them back below the branches.
    int flat_idx[HF_SLOTS];
    #pragma unroll
    for (int slot = 0; slot < HF_SLOTS; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_idx[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }
    const int expert = expert_tile_ids[tile];
    asm volatile("" :: "s"(flat_idx[0]), "s"(flat_idx[1]), "s"(flat_idx[2]),
                 "s"(flat_idx[3]), "s"(flat_idx[4]), "s"(flat_idx[5]),
                 "s"(flat_idx[6]), "s"(flat_idx[7]), "s"(expert));
    float acc[HF_ROWS][HF_SLOTS];
    const float* x_rows[HF_SLOTS];
    bool row_valid[HF_ROWS];
    #pragma unroll
    for (int slot = 0; slot < HF_SLOTS; ++slot) {
        #pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) acc[r][slot] = 0.0f;
        x_rows[slot] = nullptr;
    }
    #pragma unroll
    for (int r = 0; r < HF_ROWS; ++r) row_valid[r] = (row0 + r) < M;

    if (expert >= 0 && expert < n_exp && K > 0 && x_row_div > 0) {
        #pragma unroll
        for (int slot = 0; slot < HF_SLOTS; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            const int flat = flat_idx[slot];
            if (flat < 0) continue;
            // The Qwen4 grouped-down caller passes x_row_div == 1: every route
            // slot owns its own X row.  Skip the runtime integer division for
            // that case; the general divisor keeps its own path.
            const int x_row = (x_row_div == 1) ? flat : flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows)
                x_rows[slot] = X_src + (long long)x_row * K;
        }

        // Uniform: all inputs are scalar. When no slot of this half-tile has
        // an X row, every accumulator stays +0.0f, so all weight/X work is
        // skipped and control falls through to the zero stores below.
        bool any_x = false;
        #pragma unroll
        for (int slot = 0; slot < HF_SLOTS; ++slot) any_x |= (x_rows[slot] != nullptr);

        if (any_x) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr[HF_ROWS];
        #pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) {
            const int row = row_valid[r] ? (row0 + r) : (M - 1);
            row_ptr[r] = A + (long long)row * row_stride;
        }
        // A full group needs all four lane K values in range and a 16-byte
        // aligned row stride. Anything else keeps the production sibling's
        // guarded scalar path.
        const int vector_groups = (K % 4 == 0) ? (K / 128) : 0;

        // Pinned production contract: dequantize first, then accumulate the
        // four K values in order, every product fused into its accumulation:
        //   (((acc + w0*x0) + w1*x1) + w2*x2) + w3*x3
        // Clang's fast-math reassociation is free to permute an unpinned
        // chain, which changes the last bits of every output.
        auto accumulate = [&](int slot, float x0, float x1, float x2, float x3,
                              const float (&scale)[HF_ROWS], const float (&zero)[HF_ROWS],
                              const unsigned char (&b0)[HF_ROWS],
                              const unsigned char (&b1)[HF_ROWS]) {
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                acc[r][slot] += (scale[r] * (float)(b0[r] & 0xF) + zero[r]) * x0
                    + (scale[r] * (float)(b0[r] >> 4) + zero[r]) * x1
                    + (scale[r] * (float)(b1[r] & 0xF) + zero[r]) * x2
                    + (scale[r] * (float)(b1[r] >> 4) + zero[r]) * x3;
            }
        };

        // Buffer loads: uniform row/slot/group offsets ride in soffset, the
        // lane offset in voffset, so the loop carries no 64-bit VGPR address
        // arithmetic. Slots with no X row read row 0 (in bounds) and skip the
        // accumulation, so their loads need no branch.
        const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)A, 0, (int)((long long)M * row_stride), 0x31004000);
        const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)X_src, 0, (int)((long long)x_src_rows * K * 4), 0x31004000);
        int row_soff[HF_ROWS];
        #pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) row_soff[r] = (int)(row_ptr[r] - A);
        int x_soff[HF_SLOTS];
        #pragma unroll
        for (int slot = 0; slot < HF_SLOTS; ++slot)
            x_soff[slot] = x_rows[slot] ? (int)((x_rows[slot] - X_src) * 4) : 0;

        for (int group = 0; group < vector_groups; ++group) {
            // Issue every row's raw group bytes before decoding any of them:
            // eight independent loads in flight instead of four dependent
            // load/decode pairs. The header is uniform: a scalar load.
            unsigned int header[HF_ROWS];
            unsigned short payload[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                if (row_valid[r]) {
                    const int goff = row_soff[r] + group * 68;
                    header[r] = *(const __attribute__((address_space(4))) unsigned int*)(
                        (const __attribute__((address_space(4))) char*)expert_ptrs[expert] + goff);
                    payload[r] = __builtin_amdgcn_raw_buffer_load_b16(
                        wrsrc, 4 + tid * 2, goff, 0);
                } else {
                    header[r] = 0U;
                    payload[r] = static_cast<unsigned short>(0);
                }
            }
            // The nibbles as floats: the 0x0F0F mask is opaque so each one
            // converts with a single v_cvt_f32_ubyteN (values unchanged).
            float scale[HF_ROWS];
            float zero[HF_ROWS];
            float n0[HF_ROWS], n1[HF_ROWS], n2[HF_ROWS], n3[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                scale[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] & 0xFFFFU)));
                zero[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] >> 16)));
                unsigned int lo, hi;
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(lo) : "v"((unsigned int)payload[r]));
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(hi) : "v"((unsigned int)payload[r] >> 4));
                n0[r] = (float)(lo & 0xFFU);
                n1[r] = (float)(hi & 0xFFU);
                n2[r] = (float)(lo >> 8);
                n3[r] = (float)(hi >> 8);
            }
            const int xvoff = tid * 16;
            // Four slots per batch: the four X loads are independent, so
            // issuing them together gives the scheduler four in-flight loads
            // before the first FMA needs one. Slots with no X row keep the
            // exact previous behaviour of contributing nothing.
            #pragma unroll
            for (int batch = 0; batch < HF_SLOTS; batch += 4) {
                float4 values[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    values[i] = __builtin_bit_cast(float4,
                        __builtin_amdgcn_raw_buffer_load_b128(
                            xrsrc, xvoff, x_soff[batch + i] + group * 512, 0));
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (x_rows[batch + i] == nullptr) continue;
                    const float4 xv = values[i];
                    #pragma unroll
                    for (int r = 0; r < HF_ROWS; ++r) {
                        acc[r][batch + i] += (scale[r] * n0[r] + zero[r]) * xv.x
                            + (scale[r] * n1[r] + zero[r]) * xv.y
                            + (scale[r] * n2[r] + zero[r]) * xv.z
                            + (scale[r] * n3[r] + zero[r]) * xv.w;
                    }
                }
            }
        }

        // The single partial group a K that is not a multiple of 128 can
        // produce keeps the production sibling's guarded scalar loads.
        for (int group = vector_groups; group < groups; ++group) {
            const int base = group * 128 + tid * 4;
            const int i0 = base;
            const int i1 = base + 1;
            const int i2 = base + 2;
            const int i3 = base + 3;
            // Issue every row's raw group bytes before decoding any of them:
            // eight independent loads in flight instead of four dependent
            // load/decode pairs.
            unsigned int header[HF_ROWS];
            unsigned short payload[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                if (row_valid[r]) {
                    const char* gp = row_ptr[r] + (long long)group * 68;
                    header[r] = *(const unsigned int*)(gp + 0);
                    payload[r] = *(const unsigned short*)(gp + 4 + tid * 2);
                } else {
                    header[r] = 0U;
                    payload[r] = static_cast<unsigned short>(0);
                }
            }
            float scale[HF_ROWS];
            float zero[HF_ROWS];
            unsigned char b0[HF_ROWS];
            unsigned char b1[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                scale[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] & 0xFFFFU)));
                zero[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] >> 16)));
                b0[r] = (unsigned char)(payload[r] & 0xFFU);
                b1[r] = (unsigned char)(payload[r] >> 8);
            }
            #pragma unroll
            for (int slot = 0; slot < HF_SLOTS; ++slot) {
                const float* x = x_rows[slot];
                if (x == nullptr) continue;
                const float x0 = i0 < K ? x[i0] : 0.0f;
                const float x1 = i1 < K ? x[i1] : 0.0f;
                const float x2 = i2 < K ? x[i2] : 0.0f;
                const float x3 = i3 < K ? x[i3] : 0.0f;
                accumulate(slot, x0, x1, x2, x3, scale, zero, b0, b1);
            }
        }
        }
    }

    // Match the production sibling's wave32 association independently for all
    // eight output rows. A slot with no X row never accumulated anything, so
    // its accumulator is still exactly +0.0 and needs no shuffle rounds: the
    // reduction is skipped for those slots, a uniform (whole-wave) decision.
    // Every reduced accumulator keeps the production sibling's offsets 16, 8,
    // 4, 2, 1 and association. Rows are reduced in pairs (rows past M are
    // reduced too but never stored).
    #pragma unroll
    for (int slot = 0; slot < HF_SLOTS; ++slot) {
        if (x_rows[slot] == nullptr) continue;
        #pragma unroll
        for (int r = 0; r < HF_ROWS; r += 2) hf_lane0_tree_sum2(acc[r][slot], acc[r + 1][slot]);
    }

    if (tid == 0) {
        #pragma unroll
        for (int slot = 0; slot < HF_SLOTS; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] =
                        acc[r][slot];
            }
        }
    }
}

#undef HF_SLOTS
#undef HF_ROWS
#endif
