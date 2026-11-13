// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 symmetric-grid header checker of the qt47 (MQ6G256V2) trunk A8 route
// (`HIPFIRE_QWEN4_TRUNK_A8`). A qt47 group (256 weights) is 200 B:
//   u32 {f16 s0 | f16 z0 << 16}, u32 {f16 s1 | f16 z1 << 16}, 192 B of 6-bit codes
// (weights 0..127 use half 0, 128..255 half 1). A symmetric artifact has
// z_h == -32 * s_h exactly (f32 compare of the f16 values), finite s and z.
// One thread per half-group of the M x (K/256) groups (2 per group); one status word per
// 256 half-groups (a block), nonzero = some half-group in the chunk violates. No atomics.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_mq6_sym_check.gfx1201.hip is gfx1201-only"
#endif

extern "C" __global__ void qwen4_mq6_sym_check_gfx1201(
    const unsigned char* __restrict__ W, uint32_t* __restrict__ bad, int M, int K) {
    const long long total = (long long)M * (K >> 7);
    const long long base = (long long)blockIdx.x * 256;
    const long long i = base + threadIdx.x;
    bool viol = false;
    if (i < total) {
        const long long group = i >> 1;
        const int h = (int)(i & 1);
        const unsigned int hd =
            *(const unsigned int*)(W + (size_t)group * 200 + 4 * (size_t)h);
        const float s = (float)__builtin_bit_cast(_Float16, (unsigned short)(hd & 0xFFFFu));
        const float z = (float)__builtin_bit_cast(_Float16, (unsigned short)(hd >> 16));
        viol = !isfinite(s) || !isfinite(z) || z != -32.0f * s;
    }
    const int any = __syncthreads_or(viol ? 1 : 0);
    if (threadIdx.x == 0 && base < total) bad[blockIdx.x] = any ? 1u : 0u;
}
