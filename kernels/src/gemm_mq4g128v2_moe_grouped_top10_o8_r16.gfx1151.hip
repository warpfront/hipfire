// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT53 grouped top-10 down O8×R16.
//
// The previous production sibling assigned one wave to two adjacent output
// rows and sixteen grouped route slots. This kernel keeps that exact R16
// mapping, the per-element decode arithmetic, the padded-K handling, and the
// wave32 shuffle association, but one wave computes eight adjacent output rows.
// The X value for each lane/slot/group is loaded once and consumed by all eight
// row accumulator chains, which quarters the X bytes each output element pays
// for; each row still decodes its own QT53 weights and owns its own
// independent F32 accumulator chain.
//
// Measured on the exact routed shape (E=512, M=2560, K=640, grouped_rows=10592,
// N=291) this is 1.83x the previous sibling's time in the focused driver.
// Three structural changes carry that, all arithmetic-invisible:
//
//   * The accumulation statement is the production sibling's text, verbatim,
//     per row: acc += (scale*n0 + zero)*x0 + (scale*n1 + zero)*x1 + ... with
//     the same left-to-right association. Production compiles these kernels
//     without fast-math (hipcc --genco -O3, no -ffast-math), so the source
//     association is the whole contract: no reassociation is permitted and a
//     multi-row body cannot disagree with the two-row sibling. Dequantizing
//     with explicit fused intrinsics or re-associating the sum changes the
//     last bits and is therefore wrong here.
//   * Full QT53 groups read each lane's four K values with one aligned
//     float4 load instead of four guarded scalar loads. A lane's slice
//     (group * 128 + 4 * tid) is 16-byte aligned whenever the row stride
//     K * 4 is, so the vector path is admitted only for K % 4 == 0. The
//     arithmetically visible values are identical. The one partial group a
//     K that is not a multiple of 128 can produce is peeled into a tail loop
//     that keeps the production sibling's guarded scalar loads and
//     padded-value handling exactly.
//   * Loads are batched, because the sibling's codegen issued one X load per
//     slot with an immediate full wait. Four slots' X loads are now issued
//     together before their FMA blocks, and each row's raw weight bytes are
//     loaded for all rows before any row is decoded.
//
// Launch contract (same arguments and output layout as the production sibling):
//   grid = [ceil(M / 8), ceil(grouped_rows / 16), 1], block = [32, 1, 1]
//   expert_tile_ids[tile] is -1 for a sentinel tile; slots beyond
//   grouped_rows are padding and are never written.
//   Y_grouped[grouped_slot * M + row] is unweighted grouped output.
//   Rows at or beyond M are never written and their accumulator chains stay
//   zero. Their weight loads are clamped to the last valid row of the same
//   expert, so no read leaves the expert matrix.
//
// QT53 layout: each logical K=128 group occupies 68 bytes: fp16 scale,
// fp16 zero, and 64 packed nibbles. The final group is logically padded to
// 128 values, but values at k >= K are multiplied by zero exactly as in the
// production sibling. The caller owns the unchanged BF16 round-trip boundary.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// gfx11 wave32 ISA; tuned on gfx1151 (ArchCaps::qwen4_tuned_routes admits it).
#if defined(__GFX11__)

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
__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_top10_o8_r16_gfx1151(
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
    const int slot_start = tile * 16;
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32) return;

    // Issue all sixteen slot-index loads with the expert ID, before any is
    // tested (clamped in range): a load behind each `continue` serializes
    // sixteen dependent scalar round trips into the prologue, and the asm
    // keeps the compiler from sinking them back below the branches.
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
    const float* x_rows[16];
    bool row_valid[HF_ROWS];
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        #pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) acc[r][slot] = 0.0f;
        x_rows[slot] = nullptr;
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
            // The Qwen4 grouped-down caller passes x_row_div == 1: every route
            // slot owns its own X row.  Skip the runtime integer division for
            // that case; the general divisor keeps its own path.
            const int x_row = (x_row_div == 1) ? flat : flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows)
                x_rows[slot] = X_src + (long long)x_row * K;
        }

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
        int x_soff[16];
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot)
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
            for (int batch = 0; batch < 16; batch += 4) {
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
            for (int slot = 0; slot < 16; ++slot) {
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

    // Match the production sibling's wave32 association independently for all
    // four output rows. A slot with no X row never accumulated anything, so
    // its accumulator is still exactly +0.0 and needs no shuffle rounds: the
    // reduction is skipped for those slots and for rows past M, both of which
    // are uniform (whole-wave) decisions. Every reduced accumulator keeps the
    // production sibling's offsets 16, 8, 4, 2, 1 and association.
    // Rows are reduced in pairs (rows past M are reduced too but never
    // stored).
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        if (x_rows[slot] == nullptr) continue;
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
                    Y_grouped[(long long)grouped_slot * M + row0 + r] =
                        acc[r][slot];
            }
        }
    }
}

#undef HF_ROWS
#endif
