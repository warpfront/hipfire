// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>

struct block_i4_128 {
    float d;
    int s;
    unsigned char qs[64];
};

// Producer-final 8 floats/lane cover one K256 tile. Publish its maximum to
// the row-major scale plane; the second stage assembles one scale per row.
__device__ __forceinline__ void emit_iu4_sidecar_from_producer8(
    float f0, float f1, float f2, float f3,
    float f4, float f5, float f6, float f7,
    block_i4_128* __restrict__ Y,
    float* __restrict__ tile_amax,
    int tiles,
    int group,
    int token,
    int N,
    int lane
) {
    float amax = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f4), fabsf(f5)), fmaxf(fabsf(f6), fabsf(f7))));
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset, 32));
    if (lane == 0)
        tile_amax[(long long)token * tiles + group] = amax;
    (void)Y;
    (void)N;
}
