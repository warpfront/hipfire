// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt47 f16 header

// MQ6G256V2 (qt=47) sister of `gemv_hfq6g256_moe_gate_up_k8_indexed_batched`
// (kernels/src/gemv_hfq6g256_moe_gate_up_k8_indexed_batched.hip).
//
// Batched N-token indexed MoE gate_up GEMV. Identical kernarg signature,
// expert indexing (expert_ptrs[topk_indices[bid*K_TOP + krank]]), grid
// mapping (M, K_TOP, N), group-size-256 loop, gate/up output split
// (M = 2*MI: first half → y_gate, second half → y_up), batch handling,
// and wave32 reduction. The ONLY difference is the per-group header decode.
//
// V1 and V2 share the 200 B/group stride and 6-bit payload packing, but the
// headers are wire-incompatible — wrong dispatch yields fluent corruption:
//
//   HFQ6/MQ6G256 V1   [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 zero  (all 256 weights)
//                     [8..200) 192 B of 6-bit packed quants
//
//   qt47 MQ6G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..200) 192 B of 6-bit packed quants (identical packing)
//
// Affine-grid selection is LANE-CONSTANT. Thread `tid` reads 6 payload bytes
// covering weights [tid*8, tid*8+8) — wholly inside one 128-weight half for
// every tid, and the same half in every group. So the header offset is
// computed ONCE outside the loop:
//   tid  0..15 → weights   0..127 → pair 0 → header at gp+0
//   tid 16..31 → weights 128..255 → pair 1 → header at gp+4
// Header values widen to F32; accumulation/output remain F32, matching
// `gemv_mq6g256v2.hip` (the dense qt47 GEMV).
//
// 6-bit dequant matches the V1 batched sister and dense MQ6V2: 4 weights per
// 3 bytes → 8 weights = 6 bytes/thread; 256*6/8 = 192, + 8 B header = 200 B.
//
// Note: MQ6V2 weights require FWHT-G256 rotated activations. The kernel is
// rotation-agnostic; the caller passes the already-rotated F32 tensor via `x`.

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq6g256v2_moe_gate_up_k8_indexed_batched(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int* __restrict__ topk_indices,                // [N × K_TOP]
    const float* __restrict__ x,                         // [N × K]
    float* __restrict__ y_gate,                          // [N × K_TOP × MI]
    float* __restrict__ y_up,                            // [N × K_TOP × MI]
    int M, int K, int K_TOP, int n_experts
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    // Per-token slices into routing + input.
    const int expert_id = topk_indices[bid * K_TOP + krank];
    if ((unsigned)expert_id >= (unsigned)n_experts) {
        if (tid == 0) {
            const int mi_inv = M >> 1;
            const long long base_inv = (long long)bid * K_TOP * mi_inv + (long long)krank * mi_inv;
            if (row < mi_inv) y_gate[base_inv + row] = 0.0f;
            else              y_up[base_inv + (row - mi_inv)] = 0.0f;
        }
        return;
    }
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 200;
    const char* row_ptr = A + (long long)row * row_bytes;
    // Lane-constant dual-half header offset — see the note above.
    const int hoff = (tid < 16) ? 0 : 4;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 200;
        unsigned int h = *(const unsigned int*)(gptr + hoff);
        float scale = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu)));
        float zero  = __half2float(__ushort_as_half((unsigned short)(h >> 16)));
        const unsigned char* data = (const unsigned char*)(gptr + 8);

        // 256 weights / 32 threads = 8 weights per thread
        // 4 weights per 3 bytes → 8 weights = 6 bytes per thread
        int base_idx = g * 256 + tid * 8;
        int byte_off = tid * 6;

        unsigned char b0 = data[byte_off];
        unsigned char b1 = data[byte_off + 1];
        unsigned char b2 = data[byte_off + 2];
        unsigned char b3 = data[byte_off + 3];
        unsigned char b4 = data[byte_off + 4];
        unsigned char b5 = data[byte_off + 5];

        // 4 weights from first 3 bytes
        int q0 = b0 & 63;
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2);
        int q2 = (b1 >> 4) | ((b2 & 3) << 4);
        int q3 = b2 >> 2;
        // 4 weights from next 3 bytes
        int q4 = b3 & 63;
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2);
        int q6 = (b4 >> 4) | ((b5 & 3) << 4);
        int q7 = b5 >> 2;

        acc += (scale * (float)q0 + zero) * x_row[base_idx]
             + (scale * (float)q1 + zero) * x_row[base_idx + 1]
             + (scale * (float)q2 + zero) * x_row[base_idx + 2]
             + (scale * (float)q3 + zero) * x_row[base_idx + 3]
             + (scale * (float)q4 + zero) * x_row[base_idx + 4]
             + (scale * (float)q5 + zero) * x_row[base_idx + 5]
             + (scale * (float)q6 + zero) * x_row[base_idx + 6]
             + (scale * (float)q7 + zero) * x_row[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) {
        // Output layout matches gemv_hfq6g256_moe_gate_up_k8_indexed_batched:
        // M = 2 * MI, first half is gate, second half is up.
        const int mi = M >> 1;
        const long long base_out = (long long)bid * K_TOP * mi + (long long)krank * mi;
        if (row < mi) {
            y_gate[base_out + row] = acc;
        } else {
            y_up[base_out + (row - mi)] = acc;
        }
    }
}
