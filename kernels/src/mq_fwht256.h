// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Shared by qwen4_gemv_mq4g256.hip and tensor_ops.hip (prepended by kernels.rs /
// tensor_ops.rs): the in-wave 256-point FWHT of mq_rotate_x.

#include <hip/hip_runtime.h>

// The 256-point signed FWHT of one group held as eight values per lane of a
// wave (lane `tid` owns elements tid*8 .. tid*8+7, `signs1` already
// applied): local butterflies at strides 1, 2, 4, then strides 8..128 across
// lanes via ds_swizzle. The caller applies 0.0625 * signs2.
static __device__ __forceinline__ void mq_fwht256_lane(float (&v)[8], int tid) {
    float t;
    t = v[0]; v[0] = v[0] + v[1]; v[1] = t - v[1];
    t = v[2]; v[2] = v[2] + v[3]; v[3] = t - v[3];
    t = v[4]; v[4] = v[4] + v[5]; v[5] = t - v[5];
    t = v[6]; v[6] = v[6] + v[7]; v[7] = t - v[7];

    t = v[0]; v[0] = v[0] + v[2]; v[2] = t - v[2];
    t = v[1]; v[1] = v[1] + v[3]; v[3] = t - v[3];
    t = v[4]; v[4] = v[4] + v[6]; v[6] = t - v[6];
    t = v[5]; v[5] = v[5] + v[7]; v[7] = t - v[7];

    t = v[0]; v[0] = v[0] + v[4]; v[4] = t - v[4];
    t = v[1]; v[1] = v[1] + v[5]; v[5] = t - v[5];
    t = v[2]; v[2] = v[2] + v[6]; v[6] = t - v[6];
    t = v[3]; v[3] = v[3] + v[7]; v[7] = t - v[7];

    // Wave butterfly: strides 1-16 in thread space via ds_swizzle
    #define HBFLY(x, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(x), (pat))); \
        if (tid & (str)) { (x) = _p - (x); } else { (x) = (x) + _p; } \
    } while(0)

    #define HBFLY8(pat, str) \
        HBFLY(v[0],pat,str); HBFLY(v[1],pat,str); HBFLY(v[2],pat,str); HBFLY(v[3],pat,str); \
        HBFLY(v[4],pat,str); HBFLY(v[5],pat,str); HBFLY(v[6],pat,str); HBFLY(v[7],pat,str)

    HBFLY8(0x041F, 1);
    HBFLY8(0x081F, 2);
    HBFLY8(0x101F, 4);
    HBFLY8(0x201F, 8);
    HBFLY8(0x401F, 16);
    #undef HBFLY8
    #undef HBFLY
}

