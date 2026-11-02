// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx11+/gfx12 QT53 grouped top-10 down O8×R16, drop-zero planar twin
// (DESIGN F1, MQ4G128V2P, qt=138) of gemm_mq4g128v2_moe_grouped_top10_o8_r16.gfx1151.hip.
//
// Same launch contract, R16 slot mapping, F32 per-element decode/accumulation text and
// wave32 reduction association as the shipped kernel; only the expert tensor addressing and
// the zero change. Expert tensor [M, K], K % 128 == 0, planar (hipfire_runtime::mq4_planar):
//   [codes plane: M*K/2 bytes][scales plane: M*(K/128) LE f16]
//   row r codes  : base + r*(K/2)   (64 nibble bytes per 128-column group)
//   row r scales : base + M*(K/2) + r*(K/128)*2   (one f16 per group)
// zp = -8.0f * sc (the shipped f16 zero widened, exactly). Whole groups only (K = 640 for
// Flash-Next): the shipped kernel's partial-group tail does not exist here and any K that is
// not a multiple of 128 writes nothing.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__GFX11__) || defined(__GFX12__)

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

#define HF_ROWS 8
__launch_bounds__(32, 4)
extern "C" __global__ void gemm_mq4g128v2p_moe_grouped_top10_o8_r16_gfx1201(
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
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32 || (K & 127) != 0) return;

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
            const int x_row = (x_row_div == 1) ? flat : flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows)
                x_rows[slot] = X_src + (long long)x_row * K;
        }

        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = K / 128;
        const long long codes_row_bytes = K / 2;
        const long long scale_plane = (long long)M * codes_row_bytes;
        int row_csoff[HF_ROWS];
        int row_ssoff[HF_ROWS];
        #pragma unroll
        for (int r = 0; r < HF_ROWS; ++r) {
            const int row = row_valid[r] ? (row0 + r) : (M - 1);
            row_csoff[r] = (int)((long long)row * codes_row_bytes);
            row_ssoff[r] = (int)(scale_plane + (long long)row * groups * 2);
        }

        const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)A, 0, (int)(scale_plane + (long long)M * groups * 2), 0x31004000);
        const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)X_src, 0, (int)((long long)x_src_rows * K * 4), 0x31004000);
        int x_soff[16];
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot)
            x_soff[slot] = x_rows[slot] ? (int)((x_rows[slot] - X_src) * 4) : 0;

        for (int group = 0; group < groups; ++group) {
            unsigned short scale_bits[HF_ROWS];
            unsigned short payload[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                if (row_valid[r]) {
                    scale_bits[r] = *(const unsigned short*)(A + row_ssoff[r] + group * 2);
                    payload[r] = __builtin_amdgcn_raw_buffer_load_b16(
                        wrsrc, tid * 2, row_csoff[r] + group * 64, 0);
                } else {
                    scale_bits[r] = static_cast<unsigned short>(0);
                    payload[r] = static_cast<unsigned short>(0);
                }
            }
            float scale[HF_ROWS];
            float zero[HF_ROWS];
            float n0[HF_ROWS], n1[HF_ROWS], n2[HF_ROWS], n3[HF_ROWS];
            #pragma unroll
            for (int r = 0; r < HF_ROWS; ++r) {
                scale[r] = __half2float(__ushort_as_half(scale_bits[r]));
                zero[r] = -8.0f * scale[r];
                unsigned int lo, hi;
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(lo) : "v"((unsigned int)payload[r]));
                asm("v_and_b32 %0, 0x0f0f, %1" : "=v"(hi) : "v"((unsigned int)payload[r] >> 4));
                n0[r] = (float)(lo & 0xFFU);
                n1[r] = (float)(hi & 0xFFU);
                n2[r] = (float)(lo >> 8);
                n3[r] = (float)(hi >> 8);
            }
            const int xvoff = tid * 16;
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
    }

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
