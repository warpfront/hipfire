// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201 symmetric-grid header checker of the Qwen4 symmetric IU4 MoE route
// (`HIPFIRE_QWEN4_MOE_SYM_IU4=1`); the GEMMs are the builder module
// `qwen4_moe_iu4_sym_pm_gfx1201`. Same contract as
// `qwen4_moe_sym_check_gfx1151`: one status word per 256 checked K128
// headers, nonzero = some header in that chunk is non-finite or violates
// zp == -8*sc. QT44 (group_bytes 136) header h of a row sits at
// (h/2)*136 + 4*(h%2), QT53 (68) at h*68. No atomics.

#include <hip/hip_runtime.h>
#include <stdint.h>

#if !defined(__gfx1201__) && defined(__HIP_DEVICE_COMPILE__)
#error "qwen4_moe_sym_check.gfx1201.hip is gfx1201-only"
#endif

extern "C" __global__ void qwen4_moe_sym_check_gfx1201(
    const uint64_t* __restrict__ ptrs, uint32_t* __restrict__ bad,
    int M, int K, int E, int group_bytes) {
    const long long per_expert = (long long)M * (K >> 7);
    const long long total = per_expert * E;
    const long long base = (long long)blockIdx.x * 256;
    const long long i = base + threadIdx.x;
    bool viol = false;
    if (i < total) {
        const int e = (int)(i / per_expert);
        const long long r = i - (long long)e * per_expert;
        const int row = (int)(r / (K >> 7));
        const int h = (int)(r - (long long)row * (K >> 7));
        const unsigned char* W = (const unsigned char*)ptrs[e];
        size_t off;
        if (group_bytes == 136) off = (size_t)row * (K / 256) * 136 + (h / 2) * 136 + 4 * (h % 2);
        else off = (size_t)row * (K / 128) * 68 + (size_t)h * 68;
        const unsigned int hd = *(const unsigned int*)(W + off);
        const float sc = (float)__builtin_bit_cast(_Float16, (unsigned short)(hd & 0xFFFFu));
        const float zp = (float)__builtin_bit_cast(_Float16, (unsigned short)(hd >> 16));
        viol = !isfinite(sc) || !isfinite(zp) || zp != -8.0f * sc;
    }
    const int any = __syncthreads_or(viol ? 1 : 0);
    if (threadIdx.x == 0 && base < total) bad[blockIdx.x] = any ? 1u : 0u;
}
