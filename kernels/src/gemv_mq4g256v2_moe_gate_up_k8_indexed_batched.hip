// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt44 f16 header

// MQ4G256V2 (qt=44) sister of `gemv_hfq4g256_moe_gate_up_k8_indexed_batched`
// (kernels/src/gemv_hfq4g256_moe_gate_up_indexed_batched.hip).
//
// Transcribed from that kernel. The staging is absent there (global x_row),
// tiling, single-row-per-block gate/up split, wavefront reduction, N-token
// batching (grid.z = N, K_TOP ABI), and output layout are identical line for
// line. The ONLY difference is the group header decode.
//
// Group layout, 136 B stride for BOTH formats:
//
//   qt13 MQ4G256      [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 min   (all 256 weights)
//                     [8..136) 128 nibble bytes
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes   (identical packing)
//
// Affine-grid selection is LANE-CONSTANT. Thread `tid` reads the u32 at
// `boff = tid*4` within the payload, which covers weights [tid*8, tid*8+8) —
// wholly inside one 128-weight half for every tid, and the same half in every
// group. So the header offset is computed ONCE outside the loops:
//   tid  0..15 → weights   0..127 → grid 0 → header at gp+0
//   tid 16..31 → weights 128..255 → grid 1 → header at gp+4
// This makes the qt44 header read strictly cheaper than qt13's: one 4-byte
// load instead of two.
//
// Dequant math is f32, matching `gemv_mq4g256v2_moe_gate_up_k8_indexed.hip`
// (the single-token qt44 MoE GEMV): the f16 scale/zero are widened to float
// and the accumulation is identical to qt13's. Note this differs from the
// qt44 grouped GEMM, which accumulates the dequant in fp16 — the two paths
// have always had different precision conventions and are not expected to be
// bit-identical to each other.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.
//
// Shapes:
//   expert_ptrs : [n_exp] device pointer table, shared across tokens.
//   topk_indices: [N × K_TOP]
//   x           : [N × K]                           (FWHT-rotated post-RMSNorm)
//   y_gate      : [N × K_TOP × MI]                  (MI = M / 2)
//   y_up        : [N × K_TOP × MI]
//
// Grid: (M, K_TOP, N). Each block reduces one output row for one expert
// for one token, mirroring the single-token layout. Block = 32 threads.
//
// V1/V2 incompatibility: qt13 and qt44 share the 136 B stride and nibble
// packing but NOT the header. Feeding qt13 weights through this kernel (or
// qt44 through the V1 batched sister) silently mis-decodes scale/zero and
// produces fluent corruption. Dispatch must select by exact wire identity.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    // Per-token slices into routing + input.
    const int expert_id = topk_indices[bid * K_TOP + krank];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;
    const char* row_ptr = A + (long long)row * groups_per_row * 136;
    const int boff = tid * 4;
    // Lane-constant affine-grid header offset — see the note above.
    const int hoff = (tid < 16) ? 0 : 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * 136;
        const char* gp1 = gp0 + 136;
        const char* gp2 = gp1 + 136;
        const char* gp3 = gp2 + 136;

        unsigned int h0 = *(const unsigned int*)(gp0 + hoff);
        float sc0 = __half2float(__ushort_as_half((unsigned short)(h0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(h0 >> 16)));
        unsigned int h1 = *(const unsigned int*)(gp1 + hoff);
        float sc1 = __half2float(__ushort_as_half((unsigned short)(h1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(h1 >> 16)));
        unsigned int h2 = *(const unsigned int*)(gp2 + hoff);
        float sc2 = __half2float(__ushort_as_half((unsigned short)(h2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(h2 >> 16)));
        unsigned int h3 = *(const unsigned int*)(gp3 + hoff);
        float sc3 = __half2float(__ushort_as_half((unsigned short)(h3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(h3 >> 16)));

        unsigned int pk0 = *(const unsigned int*)(gp0 + 8 + boff);
        unsigned int pk1 = *(const unsigned int*)(gp1 + 8 + boff);
        unsigned int pk2 = *(const unsigned int*)(gp2 + 8 + boff);
        unsigned int pk3 = *(const unsigned int*)(gp3 + 8 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x_row[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x_row[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x_row[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x_row[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x_row[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x_row[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x_row[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x_row[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const char* gp = row_ptr + g * 136;
        unsigned int h = *(const unsigned int*)(gp + hoff);
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const char* gp = row_ptr + g * 136;
        unsigned int h = *(const unsigned int*)(gp + hoff);
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const char* gp = row_ptr + g * 136;
        unsigned int h = *(const unsigned int*)(gp + hoff);
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        unsigned int pk = *(const unsigned int*)(gp + 8 + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
