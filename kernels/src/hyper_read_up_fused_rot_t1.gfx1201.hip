// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1201 HC read tail with the next mixer's FWHT rotation, K == 320, <= 8 rows.
//
// `hyper_read_up_fused_f32_sx_t1_k320_rot` is `hyper_read_up_fused_f32_sx_t1_k320`
// (hyper_read_up_fused_twin_t1.gfx1201.hip) plus the `mq_rotate_x` of the `mixed` rows it writes,
// in the same launch (opt-in `HIPFIRE_QWEN4_HC_UP_ROT`).  The mixed values are computed by the
// same text as the t1 / k320 entries, so `mixed` is bitwise theirs.  The rotation follows the
// fused GDN step's pair-arrival pattern (tensor_ops.hip `gated_delta_step_gate_norm128_*`): a
// 256-column group of one token is written by 32 blocks (8 columns each); the wave of the
// 32nd block to arrive (a zeroed counter per (token, group), reset by that wave) reloads the 256
// stored values with agent-scope atomic loads and runs `mq_fwht256_lane` exactly as
// `hyper_read_projected_rotate_f32` / `mq_rotate_x` do: v = mixed * signs1, FWHT, then
// `* 0.0625f * signs2`.  Counters: `rows * (hidden / 256)` u32, zero on entry and on exit.
// Requires hidden % 256 == 0 (grid.x = hidden / 8 = 32 blocks per group) and K == 320.
// mq_fwht256.h is prepended by the host.
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

// `hyper_read_up_fused_f32_sx_t1` specialised to K == 320 (Flash-Next `hc_lowrank`), opt-in
// `HIPFIRE_QWEN4_HC_UP_T1_K320`.  Same arguments, block, dynamic LDS and grid as the t1 entry, and
// the same per-(token, row) arithmetic text (the sx twin's chains, folds, halving tree and gate /
// branch-order mix), so the output is bitwise the t1 entry's.  Two scheduling changes only:
//  * with K a compile-time constant the weight staging is five straight-line 16-byte loads per
//    thread, all issued before the first LDS store (the runtime-K loop of the t1 entry waits for
//    each load before issuing the next: five dependent memory round trips per block);
//  * the wave's `normalized` element is loaded with the weights, ahead of the gate, instead of
//    after the 16-step accumulation.
// `K` is ignored; the launcher only selects this entry for low_rank == 320.
extern "C" __global__ __launch_bounds__(256, 1) void hyper_read_up_fused_f32_sx_t1_k320_rot(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K_unused,
    int N,
    int normalized_bf16,
    float* __restrict__ rotated,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    unsigned int* __restrict__ counters) {
    constexpr int K = 320;
    const int branches = 4;
    const int wide = branches * hidden;
    extern __shared__ unsigned int ws[];
    constexpr int stride = K / 2 + 1;
    constexpr int vpr = K / 8;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int j0 = blockIdx.x * 8;
    uint4 uu[5];
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        const int v = tid + q * 256;
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 3) * hidden + j0 + (r & 7);
        uu[q] = *(const uint4*)(W + (size_t)row * K + c * 8);
    }
    const int my_row = (lane >> 3) * hidden + j0 + (lane & 7);
    const int tbase = blockIdx.y * 8 + wave;
    const int tok = tbase;
    float nv = 0.0f;
    if (tbase < N) {
        nv = normalized_bf16
            ? __uint_as_float((unsigned int)((const unsigned short*)normalized)[(size_t)tok * wide + my_row] << 16)
            : bf16_to_f32(bf16_round(normalized[(size_t)tok * wide + my_row]));
    }
#pragma unroll
    for (int q = 0; q < 5; ++q) {
        const int v = tid + q * 256;
        const int r = v / vpr, c = v - r * vpr;
        unsigned int* d = ws + r * stride + c * 4;
        d[0] = uu[q].x; d[1] = uu[q].y; d[2] = uu[q].z; d[3] = uu[q].w;
    }
    __syncthreads();
    if (tbase >= N) return;
    const unsigned int* wr = ws + lane * stride;
    float h[16];
#pragma unroll
    for (int l = 0; l < 16; ++l) {
        float sp[2];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int vl = l + 16 * p;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int i = vl * 8 + it * 256;
                if (i + 8 > K) break;
                const unsigned int u0 = wr[i / 2], u1 = wr[i / 2 + 1], u2 = wr[i / 2 + 2], u3 = wr[i / 2 + 3];
                const float w0 = __uint_as_float(u0 << 16), w1 = __uint_as_float(u0 & 0xFFFF0000u);
                const float w2 = __uint_as_float(u1 << 16), w3 = __uint_as_float(u1 & 0xFFFF0000u);
                const float w4 = __uint_as_float(u2 << 16), w5 = __uint_as_float(u2 & 0xFFFF0000u);
                const float w6 = __uint_as_float(u3 << 16), w7 = __uint_as_float(u3 & 0xFFFF0000u);
                const float* xr = X + (size_t)tok * K + i;
                a0 += w0 * xr[0];
                a1 += w1 * xr[1];
                a2 += w2 * xr[2];
                a3 += w3 * xr[3];
                a0 += w4 * xr[4];
                a1 += w5 * xr[5];
                a2 += w6 * xr[6];
                a3 += w7 * xr[7];
            }
            sp[p] = (a0 + a1) + (a2 + a3);
        }
        h[l] = sp[0] + sp[1];
        asm volatile("" : "+v"(h[l]));
    }
#pragma unroll
    for (int off = 8; off > 0; off >>= 1)
#pragma unroll
        for (int l = 0; l < off; ++l) h[l] = h[l] + h[l + off];
    const float gate_input = bf16_to_f32(bf16_round(h[0]));
    const float gate = bf16_to_f32(bf16_round(1.0f / (1.0f + expf(-gate_input))));
    const float term = bf16_to_f32(bf16_round(gate * nv));
    float value = 0.0f;
#pragma unroll
    for (int branch = 0; branch < 4; ++branch)
        value += __shfl(term, (lane & 7) + 8 * branch, 32);
    const float out = bf16_to_f32(bf16_round(value / (float)branches));
    if (lane < 8) mixed[(size_t)tok * hidden + j0 + lane] = out;
    // Rotation of this token's 256-column group by the last of its 32 writer waves.
    __threadfence();
    const int groups = hidden >> 8;
    const int group = (int)blockIdx.x >> 5;
    unsigned int* counter = counters + (size_t)tok * groups + group;
    unsigned int arrived = 0u;
    if (lane == 0) arrived = atomicAdd(counter, 1u);
    arrived = (unsigned int)__shfl((int)arrived, 0, 32);
    if (arrived != 31u) return;
    __threadfence();
    const int d0 = lane * 8;
    const size_t base = (size_t)tok * hidden + (size_t)group * 256;
    float values[8];
#pragma unroll
    for (int e = 0; e < 8; ++e)
        values[e] = __hip_atomic_load(&mixed[base + d0 + e], __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_AGENT) * signs1[d0 + e];
    mq_fwht256_lane(values, lane);
#pragma unroll
    for (int e = 0; e < 8; ++e) rotated[base + d0 + e] = values[e] * 0.0625f * signs2[d0 + e];
    if (lane == 0) *counter = 0u;
}
