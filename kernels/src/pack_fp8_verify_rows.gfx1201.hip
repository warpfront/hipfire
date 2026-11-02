// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// Lean per-row E4M3 activation pack for the PeaceMaker FP8 verify GEMM family
// (`mq4_fp8_verify_*_pm_gfx1201_*`, see rdna-compute `gemm_fp8_verify.rs`).
//
// Bit-identical to `pack_f32_to_fp8_mq4v2_gfx12` with scale_mode 1
// (kernels/src/pack_f16_to_fp8_mq4v2.gfx12.hip) in both outputs:
//   - a_n  = mq4v2_row_scale_from_absmax(absmax(row)), times
//            2^row_scale_shift when row_scale_shift != 0;
//   - byte = E4M3 RNE of v * (1.0f / a_n) via __builtin_amdgcn_cvt_pk_fp8_f32.
// The helper functions below are verbatim copies of that file's and MUST be
// kept in sync with it. Differences are layout only: no half sums (the
// symmetric verify kernels need none), one global read per element (the row
// is held in registers between the absmax and the pack, instead of being
// re-read), and every thread resolves a_n from the eight published wave
// maxima itself (one barrier, no republish). The absmax is an fmaxf
// reduction (order-independent), so the ownership of elements by threads
// changes no output bit.
//
// Row independence: block n reads only row n of `x` and writes only row n of
// `x8` and `scales[n]`; nothing depends on N or on any other row, so a row's
// bytes and scale are the same in every batch composition.
//
// Geometry:
//   grid  [N, 1, 1]
//   block [256, 1, 1]  — eight wave32s per row
//   LDS   32 B         — eight wave maxima
// Preconditions (host-enforced): N > 0, K % 256 == 0, 0 < K <= 32768
// (128 values per thread in registers), `x` 16-byte aligned.
// Element ownership: thread t owns float4 vectors t, t+256, ... of the row.

#define PMV_MAX_K 32768
#define PMV_VEC_PER_THREAD (PMV_MAX_K / 4 / 256)  // 32

__device__ __forceinline__ int mq4v2_fast_log2_ceil(float x) {
    unsigned b = __builtin_bit_cast(unsigned, x);
    int e = (int)((b >> 23) & 0xFFu);
    unsigned m = b & 0x7FFFFFu;
    return e - 127 + (m != 0u ? 1 : 0);
}

__device__ __forceinline__ float mq4v2_fast_pow2(int n) {
    return __builtin_bit_cast(float, (unsigned)(n + 127) << 23);
}

__device__ __forceinline__ float mq4v2_wave_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

// a_n = 2^{ceil(log2(amax/224))} clamped to exp [-31, 9]; zero → 1.
__device__ __forceinline__ float mq4v2_row_scale_from_absmax(float amax) {
    if (!(amax > 0.0f))
        return 1.0f;
    const float t = amax * (1.0f / 224.0f);
    int exp = mq4v2_fast_log2_ceil(t);
    if (exp < -31) exp = -31;
    if (exp > 9) exp = 9;
    return mq4v2_fast_pow2(exp);
}

extern "C" __launch_bounds__(256, 1) __global__
void pack_f32_to_fp8_verify_rows_gfx1201(
    const float* x,              // [N,K] F32, 16-byte aligned
    unsigned char* x8,           // [N,K] E4M3 bytes
    float* scales,               // [N]   a_n
    int K,
    int N,
    int row_scale_shift)
{
    __shared__ float smem[8];

    const int n = (int)blockIdx.x;
    if (n >= N || K <= 0 || K > PMV_MAX_K) return;

    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5; // 0..7

    const float4* row_in = reinterpret_cast<const float4*>(x + (long long)n * K);
    unsigned int* row_out = reinterpret_cast<unsigned int*>(x8 + (long long)n * K);
    const int vecs = K >> 2;   // float4 vectors per row

    // 1) One read of the whole row into registers + per-thread absmax.
    float4 v[PMV_VEC_PER_THREAD];
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < PMV_VEC_PER_THREAD; ++i) {
        const int idx = tid + i * 256;
        if (idx < vecs) {
            const float4 t = row_in[idx];
            v[i] = t;
            local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)),
                                               fmaxf(fabsf(t.z), fabsf(t.w))));
        }
    }

    // 2) Per-wave reduce, publish, every thread resolves a_n identically.
    local_max = mq4v2_wave_max(local_max);
    if (lane == 0)
        smem[wave] = local_max;
    __syncthreads();
    float amax = smem[0];
    #pragma unroll
    for (int w = 1; w < 8; ++w)
        amax = fmaxf(amax, smem[w]);
    float a_n = mq4v2_row_scale_from_absmax(amax);
    if (row_scale_shift != 0) a_n *= mq4v2_fast_pow2(row_scale_shift);
    if (tid == 0)
        scales[n] = a_n;

    const float inv_a = 1.0f / a_n;

    // 3) Pack from registers: four floats -> one E4M3 dword per vector.
    #pragma unroll
    for (int i = 0; i < PMV_VEC_PER_THREAD; ++i) {
        const int idx = tid + i * 256;
        if (idx < vecs) {
            const float v0 = v[i].x * inv_a;
            const float v1 = v[i].y * inv_a;
            const float v2 = v[i].z * inv_a;
            const float v3 = v[i].w * inv_a;
            unsigned int pack = 0u;
            pack = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, pack, false);
            pack = __builtin_amdgcn_cvt_pk_fp8_f32(v2, v3, pack, true);
            row_out[idx] = pack;
        }
    }
}
