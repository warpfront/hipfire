// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>  // __ushort_as_half / __half2float for the qt47 f16 header

#ifndef HIPFIRE_MOE_DOWN_LAST_COMBINE
#define HIPFIRE_MOE_DOWN_LAST_COMBINE 0
#endif

#ifndef HIPFIRE_MOE_DOWN_KERNEL
#define HIPFIRE_MOE_DOWN_KERNEL gemv_mq6g256v2_moe_down_k8_indexed_batched_expanded
#endif

// MQ6G256V2 (qt=47) sister of
// `gemv_mq4g256v2_moe_down_k8_indexed_batched_expanded` (layout/counter tail)
// and `gemv_hfq6g256_moe_down_k8_indexed_batched_expanded` (6-bit payload).
//
// Transcribed from the MQ4V2 expanded-down kernel; the tiling, x hoisting,
// pairwise fold, wavefront reduction, expanded output layout, and optional
// LAST_COMBINE counter tail are identical. The layout axis that changes is
// the group wire:
//
//   qt44 MQ4G256V2  136 B/G256: dual fp16 s/z header + 128 nibble bytes
//   qt47 MQ6G256V2  200 B/G256: dual fp16 s/z header + 192 B 6-bit payload
//
// Header identity MUST NOT collapse with HFQ6/MQ6 V1 (qt=6 / legacy f32
// scale+zero over all 256). Wrong dispatch yields fluent corruption:
//
//   qt6  HFQ6G256     [0..4)  f32 scale (all 256)
//                     [4..8)  f32 zero  (all 256)
//                     [8..200) 192 B 6-bit packed quants
//
//   qt47 MQ6G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..200) 192 B 6-bit packed quants (identical packing)
//
// Affine-grid selection is LANE-CONSTANT (same as MQ4V2): tid 0..15 → header
// at gp+0; tid 16..31 → header at gp+4. Payload is 8 weights / 6 bytes per
// lane (boff = tid*6), decoded exactly as dense `gemv_mq6g256v2` / HFQ6 MoE.
//
// Atomic-free variant: each (token, krank) GEMV writes its own row in
// `expert_outputs[N × K_TOP × M]`. Optional LAST_COMBINE folds K_TOP into
// x_residual via the scratch counter tail immediately after the expanded
// payload — identical deterministic layout to the MQ4V2 sister.
//
// Shapes:
//   expert_ptrs    : [n_exp]
//   topk_indices   : [N × K_TOP]
//   rot_batch      : [N × K_TOP × K]   (already FWHT-G256 rotated F32)
//   expert_outputs : [N × K_TOP × M]   non-atomic, one writer per cell
// Note: `topk_weights` is NOT consumed unless LAST_COMBINE is enabled.
//
// Grid: (ceil(M/4), K_TOP, N). Block = 32 threads. K%256==0 → groups=K/256
// (K=512 → 2 groups, matches the MQ4V2 expanded-down specialization this clones).

__launch_bounds__(32, 16)
extern "C" __global__ void HIPFIRE_MOE_DOWN_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp]
    const int*   __restrict__ topk_indices,              // [N × K_TOP]
    const float* __restrict__ rot_batch,                 // [N × K_TOP × K]
    float*       __restrict__ expert_outputs,            // [N × K_TOP × M]
#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    const float* __restrict__ topk_weights,               // [N × K_TOP]
    float*       __restrict__ x_residual,                 // [N × M], in-place
#endif
    int M, int K, int K_TOP, int n_experts
) {
    const int row0 = blockIdx.x * 4;
    if (row0 >= M) return;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const bool have_row1 = row1 < M;
    const bool have_row2 = row2 < M;
    const bool have_row3 = row3 < M;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int tid = threadIdx.x;

    const int routing_base = bid * K_TOP;
    const int expert_id = topk_indices[routing_base + krank];
    const bool expert_valid = (unsigned)expert_id < (unsigned)n_experts;
    if (!expert_valid) {
        if (tid == 0) {
            const long long out_inv = ((long long)bid * K_TOP + krank) * (long long)M;
            expert_outputs[out_inv + row0] = 0.0f;
            if (have_row1) expert_outputs[out_inv + row1] = 0.0f;
            if (have_row2) expert_outputs[out_inv + row2] = 0.0f;
            if (have_row3) expert_outputs[out_inv + row3] = 0.0f;
        }
    } else {
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x =
        rot_batch + ((long long)bid * K_TOP + krank) * K;

    // K%256==0 → groups_per_row = K/256; row stride is groups*200 (qt47), not groups*136 (qt44).
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 200;
    // Clamp OOB rows to row0 for weight reads (safe in-bounds); store still masked.
    const char* row_ptr0 = A + (long long)row0 * row_stride;
    const char* row_ptr1 = A + (long long)(have_row1 ? row1 : row0) * row_stride;
    const char* row_ptr2 = A + (long long)(have_row2 ? row2 : row0) * row_stride;
    const char* row_ptr3 = A + (long long)(have_row3 ? row3 : row0) * row_stride;
    // 8 weights × 6 bits = 48 bits = 6 bytes per lane.
    const int boff = tid * 6;
    // Lane-constant affine-grid header offset: tid 0..15 -> weights 0..127
    // (grid 0, header at gp+0); tid 16..31 -> weights 128..255 (grid 1, gp+4).
    const int hoff = (tid < 16) ? 0 : 4;
#if HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS
    const auto row_rsrc0 = HIPFIRE_WEIGHT_RSRC(row_ptr0);
    const auto row_rsrc1 = HIPFIRE_WEIGHT_RSRC(row_ptr1);
    const auto row_rsrc2 = HIPFIRE_WEIGHT_RSRC(row_ptr2);
    const auto row_rsrc3 = HIPFIRE_WEIGHT_RSRC(row_ptr3);
#endif

    // gfx12: initiate the four first-demand group-0 cacheline fetches while
    // the existing x float4 hoist provides independent VMEM work.  Group 1
    // is adjacent and already covered by the group-0 compute window.
#if defined(__gfx1200__) || defined(__gfx1201__)
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr0, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr1, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr2, 0);
    __builtin_amdgcn_s_prefetch_data((const void*)row_ptr3, 0);
#endif

    // Arbitrary K%256==0: groups_per_row = K/256 (K=512 → 2, K=256 → 1, etc.)
    // Preserve K=512 unrolled fast path bitwise-identical to prior specialization;
    // otherwise use general group loop. 4-row tiling/output/counters unchanged.
    float acc = 0.0f;
    float bcc = 0.0f;
    float ccc = 0.0f;
    float dcc = 0.0f;

    #if HIPFIRE_GFX1151_MOE_DOWN_HYBRID_BUFFER
    // gfx1151 hybrid cache lowering: scale/zero-point are wave-uniform within
    // each half, so preserve their scalar/global path. 6-bit packed payload
    // stays on the lean direct byte path (same as dense gemv_mq6g256v2).
    #define HIPFIRE_MOE_DOWN_LOAD_HDR(rsrc, gp, weight_offset) (*(const unsigned int*)((gp) + hoff))
    #else
    #define HIPFIRE_MOE_DOWN_LOAD_HDR(rsrc, gp, weight_offset) \
        HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, (gp) + hoff, (weight_offset) + hoff)
    #endif

    #define DOG_X8(gp, rsrc, weight_offset, a, x0, x1, x2, x3, x4, x5, x6, x7) do { \
        unsigned int h = HIPFIRE_MOE_DOWN_LOAD_HDR(rsrc, gp, weight_offset); \
        float sc = __half2float(__ushort_as_half((unsigned short)(h & 0xFFFFu))); \
        float zp = __half2float(__ushort_as_half((unsigned short)(h >> 16))); \
        const unsigned char* d = (const unsigned char*)((gp) + 8 + boff); \
        unsigned char b0 = d[0]; unsigned char b1 = d[1]; unsigned char b2 = d[2]; \
        unsigned char b3 = d[3]; unsigned char b4 = d[4]; unsigned char b5 = d[5]; \
        int q0 = b0 & 63; \
        int q1 = (b0 >> 6) | ((b1 & 0xF) << 2); \
        int q2 = (b1 >> 4) | ((b2 & 3) << 4); \
        int q3 = b2 >> 2; \
        int q4 = b3 & 63; \
        int q5 = (b3 >> 6) | ((b4 & 0xF) << 2); \
        int q6 = (b4 >> 4) | ((b5 & 3) << 4); \
        int q7 = b5 >> 2; \
        (a) += (sc * (float)q0 + zp) * (x0) \
             + (sc * (float)q1 + zp) * (x1) \
             + (sc * (float)q2 + zp) * (x2) \
             + (sc * (float)q3 + zp) * (x3) \
             + (sc * (float)q4 + zp) * (x4) \
             + (sc * (float)q5 + zp) * (x5) \
             + (sc * (float)q6 + zp) * (x6) \
             + (sc * (float)q7 + zp) * (x7); \
    } while (0)

    if (groups_per_row == 2) {
        // K=512 fast path — bitwise identical to prior 2-group specialization.
        float acc0 = 0.0f, acc1 = 0.0f;
        float bcc0 = 0.0f, bcc1 = 0.0f;
        float ccc0 = 0.0f, ccc1 = 0.0f;
        float dcc0 = 0.0f, dcc1 = 0.0f;
        const int base0 = tid * 8;
        const float4 xv0a = *reinterpret_cast<const float4*>(x + base0);
        const float4 xv0b = *reinterpret_cast<const float4*>(x + base0 + 4);
        const float4 xv1a = *reinterpret_cast<const float4*>(x + base0 + 256);
        const float4 xv1b = *reinterpret_cast<const float4*>(x + base0 + 260);
        DOG_X8(row_ptr0 + 0 * 200, row_rsrc0, 0 * 200, acc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(row_ptr1 + 0 * 200, row_rsrc1, 0 * 200, bcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(row_ptr2 + 0 * 200, row_rsrc2, 0 * 200, ccc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(row_ptr3 + 0 * 200, row_rsrc3, 0 * 200, dcc0,
               xv0a.x, xv0a.y, xv0a.z, xv0a.w, xv0b.x, xv0b.y, xv0b.z, xv0b.w);
        DOG_X8(row_ptr0 + 1 * 200, row_rsrc0, 1 * 200, acc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(row_ptr1 + 1 * 200, row_rsrc1, 1 * 200, bcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(row_ptr2 + 1 * 200, row_rsrc2, 1 * 200, ccc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        DOG_X8(row_ptr3 + 1 * 200, row_rsrc3, 1 * 200, dcc1,
               xv1a.x, xv1a.y, xv1a.z, xv1a.w, xv1b.x, xv1b.y, xv1b.z, xv1b.w);
        acc = acc0 + acc1;
        bcc = bcc0 + bcc1;
        ccc = ccc0 + ccc1;
        dcc = dcc0 + dcc1;
    } else {
        for (int g = 0; g < groups_per_row; ++g) {
            const int g_off = g * 200;
            const char* gp0 = row_ptr0 + g_off;
            const char* gp1 = row_ptr1 + g_off;
            const char* gp2 = row_ptr2 + g_off;
            const char* gp3 = row_ptr3 + g_off;
            const int x_base = g * 256 + tid * 8;
            const float4 xv_a = *reinterpret_cast<const float4*>(x + x_base);
            const float4 xv_b = *reinterpret_cast<const float4*>(x + x_base + 4);
            DOG_X8(gp0, row_rsrc0, g_off, acc,
                   xv_a.x, xv_a.y, xv_a.z, xv_a.w, xv_b.x, xv_b.y, xv_b.z, xv_b.w);
            DOG_X8(gp1, row_rsrc1, g_off, bcc,
                   xv_a.x, xv_a.y, xv_a.z, xv_a.w, xv_b.x, xv_b.y, xv_b.z, xv_b.w);
            DOG_X8(gp2, row_rsrc2, g_off, ccc,
                   xv_a.x, xv_a.y, xv_a.z, xv_a.w, xv_b.x, xv_b.y, xv_b.z, xv_b.w);
            DOG_X8(gp3, row_rsrc3, g_off, dcc,
                   xv_a.x, xv_a.y, xv_a.z, xv_a.w, xv_b.x, xv_b.y, xv_b.z, xv_b.w);
        }
    }
    #undef DOG_X8
    #undef HIPFIRE_MOE_DOWN_LOAD_HDR
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        bcc += __shfl_down(bcc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        ccc += __shfl_down(ccc, offset);
    for (int offset = 16; offset > 0; offset >>= 1)
        dcc += __shfl_down(dcc, offset);

    if (tid == 0) {
        // Direct store to per-(token, krank) row — no atomic contention.
        // Layout: expert_outputs[bid][krank][row].
        const long long out_base =
            ((long long)bid * K_TOP + krank) * (long long)M;
        expert_outputs[out_base + row0] = acc;
        if (have_row1) expert_outputs[out_base + row1] = bcc;
        if (have_row2) expert_outputs[out_base + row2] = ccc;
        if (have_row3) expert_outputs[out_base + row3] = dcc;
    }

    }  // expert_valid

#if HIPFIRE_MOE_DOWN_LAST_COMBINE
    // Preserve one independently schedulable expert wave per workgroup. Each
    // row tile publishes its four expanded outputs, then increments a reusable
    // counter in the scratch tail. Only the last expert wave performs the
    // fixed k=0..7 weighted fold, replacing the separate combine dispatch.
    // The scratch tail is [N × ceil(M/4)] u32 immediately after the expanded
    // [N × K_TOP × M] payload and is zeroed once at scratch allocation.
    int prior = -1;
    const int row_tiles = (M + 3) / 4;
    int* counters = reinterpret_cast<int*>(
        expert_outputs + (long long)gridDim.z * K_TOP * M);
    const int counter_index = bid * row_tiles + blockIdx.x;
    if (tid == 0) {
        // Publish this block's output stores before advertising arrival.
        __threadfence();
        prior = atomicAdd(counters + counter_index, 1);
    }
    prior = __shfl(prior, 0);
    if (prior == K_TOP - 1) {
        if (tid < 4) {
            const int row = row0 + tid;
            if (row < M) {
                const long long token_base = (long long)bid * K_TOP * M;
                const int weights_base = bid * K_TOP;
                float combined = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    if (k < K_TOP) {
                        combined += topk_weights[weights_base + k]
                                  * expert_outputs[token_base + (long long)k * M + row];
                    }
                }
                x_residual[(long long)bid * M + row] += combined;
            }
        }
        __syncwarp();
        if (tid == 0)
            atomicExch(counters + counter_index, 0);
    }
#endif
}
