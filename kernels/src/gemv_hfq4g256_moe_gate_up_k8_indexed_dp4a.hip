// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef HIPFIRE_MOE_GU_DP4A_KERNEL
#define HIPFIRE_MOE_GU_DP4A_KERNEL gemv_hfq4g256_moe_gate_up_k8_indexed_dp4a
#endif
#ifndef HIPFIRE_MOE_GU_DP4A_ROWS_PER_BLOCK
#define HIPFIRE_MOE_GU_DP4A_ROWS_PER_BLOCK 1
#endif

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Device-routed MoE gate+up using one Q8_1 activation shared by all K_TOP
// experts. Each wave owns one output row and computes gate and up together.
// The mixed unsigned-weight/signed-activation dot maps directly to
// v_dot4_i32_iu8 on gfx1201; v_perm_b32 expands eight packed nibbles into
// two unsigned-byte vectors without the scalar BFE chain.
extern "C" __global__ void HIPFIRE_MOE_GU_DP4A_KERNEL(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ topk_indices,
    const block_q8_1_mmq* __restrict__ Xq,
    float* __restrict__ y_gate,
    float* __restrict__ y_up,
    int M,
    int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int mi = M >> 1;
    const int row = blockIdx.x * HIPFIRE_MOE_GU_DP4A_ROWS_PER_BLOCK + warp_id;
    if (row >= mi) return;

    const int krank = blockIdx.y;
    const int expert_id = topk_indices[krank];
    const char* A = reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const int groups_per_row = K / 256;
    const char* gate_ptr = A + (long long)row * groups_per_row * 136;
    const char* up_ptr = A + (long long)(row + mi) * groups_per_row * 136;

    const int boff = lane * 4;
    const int sub_in_block = (lane >> 2) & 3;
    const int block_within_group = lane >> 4;
    const int qs_byte_off = 8 * (lane & 15);
    float gacc = 0.0f;
    float uacc = 0.0f;

#ifdef HIPFIRE_MOE_GU_DP4A_UNROLL2
#pragma unroll 2
#endif
    for (int g = 0; g < groups_per_row; ++g) {
        const char* gg = gate_ptr + (long long)g * 136;
        const char* ug = up_ptr + (long long)g * 136;
        const float gsc = __builtin_bit_cast(float, *(const unsigned int*)gg);
        const float gzp = __builtin_bit_cast(float, *(const unsigned int*)(gg + 4));
        const float usc = __builtin_bit_cast(float, *(const unsigned int*)ug);
        const float uzp = __builtin_bit_cast(float, *(const unsigned int*)(ug + 4));
        const unsigned int gpk = *(const unsigned int*)(gg + 8 + boff);
        const unsigned int upk = *(const unsigned int*)(ug + 8 + boff);

        const unsigned int geven = gpk & 0x0F0F0F0Fu;
        const unsigned int godd = (gpk >> 4) & 0x0F0F0F0Fu;
        const int ga = (int)__builtin_amdgcn_perm(godd, geven, 0x05010400u);
        const int gb = (int)__builtin_amdgcn_perm(godd, geven, 0x07030602u);
        const unsigned int ueven = upk & 0x0F0F0F0Fu;
        const unsigned int uodd = (upk >> 4) & 0x0F0F0F0Fu;
        const int ua = (int)__builtin_amdgcn_perm(uodd, ueven, 0x05010400u);
        const int ub = (int)__builtin_amdgcn_perm(uodd, ueven, 0x07030602u);

        const block_q8_1_mmq* xb = Xq + 2 * g + block_within_group;
        const int* xqs = reinterpret_cast<const int*>(xb->qs + qs_byte_off);
        const int xqa = xqs[0];
        const int xqb = xqs[1];
        int gsum = __builtin_amdgcn_sudot4(false, ga, true, xqa, 0, false);
        gsum = __builtin_amdgcn_sudot4(false, gb, true, xqb, gsum, false);
        int usum = __builtin_amdgcn_sudot4(false, ua, true, xqa, 0, false);
        usum = __builtin_amdgcn_sudot4(false, ub, true, xqb, usum, false);

        const float2 ds = __half22float2(xb->ds4[sub_in_block]);
        gacc += gsc * ds.x * (float)gsum + gzp * ds.y * 0.25f;
        uacc += usc * ds.x * (float)usum + uzp * ds.y * 0.25f;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        gacc += __shfl_down(gacc, offset);
        uacc += __shfl_down(uacc, offset);
    }
    if (lane == 0) {
        y_gate[(long long)krank * mi + row] = gacc;
        y_up[(long long)krank * mi + row] = uacc;
    }
}
