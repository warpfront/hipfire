// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8 (qt=42, 128-wide rotation) grouped-WMMA-GEMM for MoE
// prefill, gfx1151 (Strix Halo).
//
// Same SGLang-style scatter pipeline and wave32 WMMA 16x16x16 F16 layout as
// gemm_mfp4g32_e8_moe_grouped_wmma.gfx1151.hip, with one structural change: the
// K loop is FLAT over 16-value WMMA tiles instead of nested group -> tile.
//
// Why: the shipped kernel nests `for g in 0..K/256 { for kt in 0..16 }`, so it
// covers the reduction axis only when K % 256 == 0. Nothing in the math needs
// that 256: the AoS row's 17-byte blocks are contiguous, so the absolute block
// index of tile t is just `t >> 1` and the codeword pair within it is
// `(t & 1) * 2`. Flattening the loop therefore covers any K % 32 == 0 with the
// same arithmetic, and for K % 256 == 0 it visits the identical (g, kt)
// sequence in the identical order, so the accumulation order — and the result —
// is unchanged.
//
// E8 weight layout (bit-identical decode to gemv_mfp4g32_e8.hip):
//   per row: 16-byte header (f16 row_scale at byte 0) + (K/32) x 17-byte blocks.
//   block  : 1 byte E4M3 block_scale + 4 x 32-bit E8 codewords (8 weights each).
//   value  : row_scale x cvt_e4m3(block_scale) x 0.88 x e8_decode(codeword)[i].
//
// Row/tile arithmetic is untouched by the flattening: `safe_row`, the lane ->
// slot gather, `x_row_div`, and the masked output store depend on M and m_total,
// not on K's group count; the only K-derived values are the row stride
// (16 + (K/32)*17) and the tile count (K/16).
//
// X is the FWHT-rotated activation, auto-converted f32->f16 by the wrapper via
// ensure_fp16_x (cached) and read here as f16 for the WMMA B operand.
//
// Grid: [ceil(M/16), m_total/16]. Block: [32]. LDS: 0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__device__ __forceinline__ float cvt_e4m3_scale_to_f32(unsigned char b) {
    const int exp = (b >> 3) & 0xF;
    const unsigned mant = (unsigned)(b & 0x7);
    if (exp == 0) return 0.015625f * ((float)mant) * 0.125f;
    if (exp == 0xF && mant == 7u) return 448.0f;
    const float pow2 = __builtin_bit_cast(float, ((unsigned int)(exp + 120)) << 23);
    return pow2 * (1.0f + ((float)mant) * 0.125f);
}

__device__ __forceinline__ void e8_decode_index(unsigned int idx, float p[8]) {
    const int coset = (int)((idx >> 31) & 1u);
    unsigned int e[8];
    unsigned int sl = 0u;
    for (int i = 0; i < 7; i++) { e[i] = (idx >> (unsigned)(4 * i)) & 0xFu; sl += e[i]; }
    const unsigned int e7_high = (idx >> 28) & 0x7u;
    const unsigned int p7 = e7_high << 1;
    const unsigned int lsb = (sl + p7) & 1u;
    e[7] = p7 | lsb;
    for (int i = 0; i < 8; i++) { float c = (float)((int)e[i] - 7); p[i] = coset ? c + 0.5f : c; }
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N x K], x_row_div = K_TOP -> x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots x K], x_row_div = 1 -> x_row = flat
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mfp4g32_e8g128_moe_grouped_wmma(
    const unsigned long long* __restrict__ expert_weight_ptrs, // [E]
    const int*                __restrict__ expert_tile_ids,    // [m_total / 16]
    const int*                __restrict__ sorted_slot_index,  // [m_total]
    const _Float16*           __restrict__ X_src,              // [n_rows x K] f16
    float*                    __restrict__ Y_grouped,          // [m_total x M]
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;

    if (row_start >= M || slot_start >= m_total) return;

    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;

    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    // X gather: each lane (tid & 15) gets its own routed slot; -1 -> zero row.
    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) {
            x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
        }
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;

    // A-row selection: pad OOB rows to the last valid one; masked store discards.
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int row_block_bytes = (K / 32) * 17;
    const char* row_ptr = A + (long long)safe_row * (16 + (long long)row_block_bytes);
    const float row_scale_f = (float)__builtin_bit_cast(_Float16, *(const unsigned short*)row_ptr);
    const char* blocks_ptr = row_ptr + 16;

    // Flat tile loop: tile t covers K values [16t, 16t+16), which is the second
    // or first codeword pair of block (t >> 1). Covers any K % 32 == 0 exactly;
    // for K % 256 == 0 it is the shipped (group, tile) order, flattened.
    const int tiles_per_row = K / 16;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for (int t = 0; t < tiles_per_row; t++) {
        const char* block_ptr = blocks_ptr + (long long)(t >> 1) * 17;
        const float ssc = row_scale_f
            * cvt_e4m3_scale_to_f32(*(const unsigned char*)block_ptr) * 0.88f;
        const int cw_base = (t & 1) * 2;
        const unsigned int cw0 = *(const unsigned int*)(block_ptr + 1 + cw_base * 4);
        const unsigned int cw1 = *(const unsigned int*)(block_ptr + 1 + (cw_base + 1) * 4);

        float p0[8], p1[8];
        e8_decode_index(cw0, p0);
        e8_decode_index(cw1, p1);

        half16_t a_a;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            a_a[i]     = (_Float16)(p0[i] * ssc);
            a_a[8 + i] = (_Float16)(p1[i] * ssc);
        }

        half16_t b = (x_base != nullptr) ? *(const half16_t*)(x_base + (long long)t * 16) : zero_b;

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b, acc);
    }

    // RDNA3 wave32 WMMA output: acc[j] = C[2*j + (tid>>4)][tid & 15].
    // A operand = weight rows (M-dim), B operand = slot lane (m_total-dim).
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) {
                Y_grouped[(long long)out_col * M + out_row] = acc[j];
            }
        }
    }
}
