// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
// HIPFIRE_COMPILER_FLAGS: -mwavefrontsize64

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#if defined(__gfx1200__) || defined(__gfx1201__)

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

static __device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (int)(((unsigned int)a & 0xffu)
        | (((unsigned int)b & 0xffu) << 8)
        | (((unsigned int)c & 0xffu) << 16)
        | (((unsigned int)d & 0xffu) << 24));
}

// Mixed MQ4P gate+up/SwiGLU with a Q8_1 activation sidecar.  Affine MQ6
// and MQ4 experts use signed DP4A; MQ3-Lloyd keeps the established F32
// lookup/FMA path so this experiment changes only the tiers with an exact
// affine zero-point correction.
//
// One wave64 owns one output row.  Lanes 0..31 compute gate and lanes 32..63
// compute up.  Each half-lane owns eight adjacent K values, matching both the
// packed expert layout and two int32 words in the Q8_1 sidecar.
extern "C" __launch_bounds__(64, 8)
__global__ void gemv_mixed_moe_gate_up_swiglu_k8_indexed_batched_wave64_dp4a_gfx12(
    const unsigned long long* __restrict__ expert_ptrs,
    const unsigned char* __restrict__ dtype_tags,
    const int* __restrict__ topk_indices,
    const float* __restrict__ x,
    const block_q8_1_mmq* __restrict__ xq,
    float* __restrict__ y_hidden,
    int MI, int K, int K_TOP
) {
    const int row = blockIdx.x;
    if (row >= MI) return;
    const int krank = blockIdx.y;
    const int bid = blockIdx.z;
    const int lane = threadIdx.x;
    const int half_lane = lane & 31;
    const int matrix_row = row + ((lane >> 5) * MI);

    const int expert_id = topk_indices[bid * K_TOP + krank];
    const unsigned char tag = dtype_tags[expert_id];
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_ptrs[expert_id]);
    const float* __restrict__ x_row = x + (long long)bid * K;
    const block_q8_1_mmq* __restrict__ xq_row = xq + (long long)bid * (K / 128);
    const int groups = K / 256;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    __shared__ float cb[16];

    if (tag == 0 || tag == 2) {
        const int block_in_group = half_lane >> 4;
        const int sub_in_block = (half_lane >> 2) & 3;
        const int q8_byte = (half_lane & 15) * 8;

        for (int g = 0; g < groups; ++g) {
            const block_q8_1_mmq* xb = xq_row + g * 2 + block_in_group;
            const int* xqi = reinterpret_cast<const int*>(xb->qs + q8_byte);
            const int xqa = xqi[0];
            const int xqb = xqi[1];
            const float2 ds = __half22float2(xb->ds4[sub_in_block]);

            int wa, wb;
            float sc, zp_eff;
            if (tag == 0) {
                const char* gp = A + ((long long)matrix_row * groups + g) * 200;
                sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                zp_eff = zp + 32.0f * sc;
                const unsigned char* d = (const unsigned char*)(gp + 8) + half_lane * 6;
                const int q0 = d[0] & 63;
                const int q1 = (d[0] >> 6) | ((d[1] & 15) << 2);
                const int q2 = (d[1] >> 4) | ((d[2] & 3) << 4);
                const int q3 = d[2] >> 2;
                const int q4 = d[3] & 63;
                const int q5 = (d[3] >> 6) | ((d[4] & 15) << 2);
                const int q6 = (d[4] >> 4) | ((d[5] & 3) << 4);
                const int q7 = d[5] >> 2;
                wa = pack_i8x4(q0 - 32, q1 - 32, q2 - 32, q3 - 32);
                wb = pack_i8x4(q4 - 32, q5 - 32, q6 - 32, q7 - 32);
            } else {
                const char* gp = A + ((long long)matrix_row * groups + g) * 136;
                sc = __builtin_bit_cast(float, *(const unsigned int*)gp);
                const float zp = __builtin_bit_cast(float, *(const unsigned int*)(gp + 4));
                zp_eff = zp + 8.0f * sc;
                const unsigned int p = *(const unsigned int*)(gp + 8 + half_lane * 4);
                wa = pack_i8x4(
                    (int)(p & 15u) - 8, (int)((p >> 4) & 15u) - 8,
                    (int)((p >> 8) & 15u) - 8, (int)((p >> 12) & 15u) - 8);
                wb = pack_i8x4(
                    (int)((p >> 16) & 15u) - 8, (int)((p >> 20) & 15u) - 8,
                    (int)((p >> 24) & 15u) - 8, (int)((p >> 28) & 15u) - 8);
            }

            int sumi = __builtin_amdgcn_sudot4(true, wa, true, xqa, 0, false);
            sumi = __builtin_amdgcn_sudot4(true, wb, true, xqb, sumi, false);
            const float v = sc * ds.x * (float)sumi + zp_eff * ds.y * 0.25f;
            if ((g & 3) == 0) a0 += v;
            else if ((g & 3) == 1) a1 += v;
            else if ((g & 3) == 2) a2 += v;
            else a3 += v;
        }
    } else if (tag == 3) {
        const int row_bytes = groups * 112;
        const char* gp = A + (long long)matrix_row * row_bytes;
        const int boff = half_lane * 3;
        for (int g = 0; g < groups; ++g, gp += 112) {
            if (half_lane < 8) {
                const unsigned short hc = *(const unsigned short*)(gp + half_lane * 2);
                cb[(lane >> 5) * 8 + half_lane] =
                    __half2float(__builtin_bit_cast(__half, hc));
            }
            __syncthreads();
            const unsigned char* d = (const unsigned char*)(gp + 16) + boff;
            const unsigned int p = (unsigned int)d[0]
                | ((unsigned int)d[1] << 8) | ((unsigned int)d[2] << 16);
            const int xb = g * 256 + half_lane * 8;
            const float* cp = cb + (lane >> 5) * 8;
            const float v = cp[p & 7u] * x_row[xb]
                + cp[(p >> 3) & 7u] * x_row[xb + 1]
                + cp[(p >> 6) & 7u] * x_row[xb + 2]
                + cp[(p >> 9) & 7u] * x_row[xb + 3]
                + cp[(p >> 12) & 7u] * x_row[xb + 4]
                + cp[(p >> 15) & 7u] * x_row[xb + 5]
                + cp[(p >> 18) & 7u] * x_row[xb + 6]
                + cp[(p >> 21) & 7u] * x_row[xb + 7];
            __syncthreads();
            if ((g & 3) == 0) a0 += v;
            else if ((g & 3) == 1) a1 += v;
            else if ((g & 3) == 2) a2 += v;
            else a3 += v;
        }
    }

    float acc = (a0 + a1) + (a2 + a3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset, 32);
    const float up = __shfl(acc, 32, 64);
    if (lane == 0) {
        const long long out = ((long long)bid * K_TOP + krank) * MI + row;
        y_hidden[out] = (acc / (1.0f + expf(-acc))) * up;
    }
}

#endif
