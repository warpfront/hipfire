// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// G128-native m4 (4 rows per WG) MoE gate_up GEMV with LDS-cached X for
// HFQ4G128 (ParoQuant) on gfx12 (RDNA4 — R9700/gfx1201, gfx1200).
//
// === Bottleneck analysis ===
//
// The baseline `gemv_paro_q4g128_moe_gate_up_k8_indexed` dispatches ONE WG
// per (output_row, krank). For A3B-PARO at K_TOP=8 and gate_up M ≈ 1536:
//
//   - WGs per layer per token (decode): M × K_TOP = 1536 × 8 = 12 288
//   - X reads per layer per token: 12 288 × K floats × 4 B
//     For K=2048: 12 288 × 8 KB = ~96 MB of X re-reads per layer per token
//   - A reads per layer per token: M × K_TOP × groups × 72 B
//     = 1536 × 8 × 16 × 72 = ~14 MB
//
// So X re-reads dominate by ~6.9×. The token's hidden state (the X operand
// for gate_up) is identical across all M × K_TOP WG dispatches; the
// baseline re-fetches it from global memory once per WG.
//
// === m4 lever ===
//
// Each WG now owns 4 output rows × 1 krank. X is loaded ONCE into LDS at
// WG entry, then re-read from LDS for each of the 4 rows. X-load BW per
// layer drops 4×:
//
//   - WGs per layer per token: (M/4) × K_TOP = 384 × 8 = 3 072
//   - X reads per layer: 3 072 × K × 4 B = 24 MB (4× lower)
//   - A reads per layer: unchanged (~14 MB)
//
// Each WG processes the same 4 output rows for ONE krank. The expert
// pointer table `expert_ptrs[topk_indices[krank]]` is shared across the
// 4 rows. The 4 rows each have their own scale/zero/qweight per group.
//
// === Per-block contract ===
//
//   - blockIdx.x = row_base / 4 (4 rows per WG)
//   - blockIdx.y = krank (which top-K expert this WG processes)
//   - 32 threads per WG, each owns byte_off = tid * 2 of every group's
//     qweight payload (each thread → 4 nibbles per group per row).
//   - LDS: K floats (4 × K bytes, sized at launch via dynamic shmem).
//
// Output split: y_gate[krank × mi + row] for row < mi, else y_up[krank × mi + (row - mi)].

__launch_bounds__(32, 8)
extern "C" __global__ void gemv_paro_q4g128_moe_gate_up_m4_indexed_gfx12(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp] device pointers
    const int* __restrict__ topk_indices,                // [k_top]
    const float* __restrict__ x,                         // [K]
    float* __restrict__ y_gate,                          // [K_TOP × MI]
    float* __restrict__ y_up,                            // [K_TOP × MI]
    int M, int K
) {
    const int row_base = blockIdx.x * 4;  // 4 rows per WG
    if (row_base >= M) return;
    const int krank = blockIdx.y;
    const int tid = threadIdx.x;

    const int expert_id = topk_indices[krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);

    const int groups_per_row = K / 128;

    // Cooperative LDS load of X (K floats). All 32 threads stride-load.
    // Wait at the barrier before any reads from LDS in the compute body.
    extern __shared__ float x_lds[];
    for (int k = tid; k < K; k += 32) {
        x_lds[k] = x[k];
    }
    __syncthreads();

    // 4 row accumulators. Each thread covers all 4 rows for its 4
    // qweight nibbles per group.
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int byte_off = tid * 2;

    // Row pointers (only need the part of A specific to this WG's 4 rows).
    const char* row_ptr_0 = A + (long long)(row_base + 0) * groups_per_row * 72;
    const char* row_ptr_1 = A + (long long)(row_base + 1) * groups_per_row * 72;
    const char* row_ptr_2 = A + (long long)(row_base + 2) * groups_per_row * 72;
    const char* row_ptr_3 = A + (long long)(row_base + 3) * groups_per_row * 72;

    // OOB guards: if row >= M, ptr stays valid but acc is discarded at write.
    const bool live_1 = (row_base + 1) < M;
    const bool live_2 = (row_base + 2) < M;
    const bool live_3 = (row_base + 3) < M;

    for (int g = 0; g < groups_per_row; g++) {
        // X tile shared across all 4 rows — read ONCE from LDS.
        const int base = g * 128 + tid * 4;
        const float x0 = x_lds[base];
        const float x1 = x_lds[base + 1];
        const float x2 = x_lds[base + 2];
        const float x3 = x_lds[base + 3];

        // Per-row scale/zero + 2 qweight bytes.
        const char* gp0 = row_ptr_0 + g * 72;
        const float sc0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0));
        const float zp0 = __builtin_bit_cast(float, *(const unsigned int*)(gp0 + 4));
        unsigned char b00 = *((const unsigned char*)(gp0 + 8 + byte_off));
        unsigned char b01 = *((const unsigned char*)(gp0 + 8 + byte_off + 1));
        acc0 += (sc0 * (float)(b00 & 0xFu)        + zp0) * x0
              + (sc0 * (float)((b00 >> 4) & 0xFu) + zp0) * x1
              + (sc0 * (float)(b01 & 0xFu)        + zp0) * x2
              + (sc0 * (float)((b01 >> 4) & 0xFu) + zp0) * x3;

        if (live_1) {
            const char* gp1 = row_ptr_1 + g * 72;
            const float sc1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1));
            const float zp1 = __builtin_bit_cast(float, *(const unsigned int*)(gp1 + 4));
            unsigned char b10 = *((const unsigned char*)(gp1 + 8 + byte_off));
            unsigned char b11 = *((const unsigned char*)(gp1 + 8 + byte_off + 1));
            acc1 += (sc1 * (float)(b10 & 0xFu)        + zp1) * x0
                  + (sc1 * (float)((b10 >> 4) & 0xFu) + zp1) * x1
                  + (sc1 * (float)(b11 & 0xFu)        + zp1) * x2
                  + (sc1 * (float)((b11 >> 4) & 0xFu) + zp1) * x3;
        }

        if (live_2) {
            const char* gp2 = row_ptr_2 + g * 72;
            const float sc2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2));
            const float zp2 = __builtin_bit_cast(float, *(const unsigned int*)(gp2 + 4));
            unsigned char b20 = *((const unsigned char*)(gp2 + 8 + byte_off));
            unsigned char b21 = *((const unsigned char*)(gp2 + 8 + byte_off + 1));
            acc2 += (sc2 * (float)(b20 & 0xFu)        + zp2) * x0
                  + (sc2 * (float)((b20 >> 4) & 0xFu) + zp2) * x1
                  + (sc2 * (float)(b21 & 0xFu)        + zp2) * x2
                  + (sc2 * (float)((b21 >> 4) & 0xFu) + zp2) * x3;
        }

        if (live_3) {
            const char* gp3 = row_ptr_3 + g * 72;
            const float sc3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3));
            const float zp3 = __builtin_bit_cast(float, *(const unsigned int*)(gp3 + 4));
            unsigned char b30 = *((const unsigned char*)(gp3 + 8 + byte_off));
            unsigned char b31 = *((const unsigned char*)(gp3 + 8 + byte_off + 1));
            acc3 += (sc3 * (float)(b30 & 0xFu)        + zp3) * x0
                  + (sc3 * (float)((b30 >> 4) & 0xFu) + zp3) * x1
                  + (sc3 * (float)(b31 & 0xFu)        + zp3) * x2
                  + (sc3 * (float)((b31 >> 4) & 0xFu) + zp3) * x3;
        }
    }

    // Per-row wavefront reduction, then write to y_gate or y_up.
    auto write_row = [&](float a, int row) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            a += __shfl_down(a, offset);
        }
        if (tid == 0) {
            const int mi = M >> 1;
            if (row < mi) {
                y_gate[krank * mi + row] = a;
            } else {
                y_up[krank * mi + (row - mi)] = a;
            }
        }
    };

    write_row(acc0, row_base + 0);
    if (live_1) write_row(acc1, row_base + 1);
    if (live_2) write_row(acc2, row_base + 2);
    if (live_3) write_row(acc3, row_base + 3);
}
