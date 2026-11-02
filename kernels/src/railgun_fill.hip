// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! railgun typed fill (DFlash window design A §2): sibling of
//! `railgun_copy.hip`, a separate module so the certified copy module's
//! bytes are unchanged. Rust side of the ABI: `railgun::copy::fill_launch`.

#include <hip/hip_runtime.h>

typedef __attribute__((address_space(1))) unsigned char railgun_gu8;

// `railgun_fill_u32(dst, value, words)`: the typed fill. Writes `value` to
// each of the `words` 32-bit words at `dst` (4-byte aligned by the caller),
// reads nothing. `words ≤ 2^30`, so every byte offset `i << 2` and every
// index sum `i + stride` (stride ≤ 2^31) stays in 32 bits. Every access is
// `dst` plus a 32-bit byte offset, so A1 attributes the only write to `dst`.
// Grid (gx, 1, 1), block (256, 1, 1); result is independent of the grid.
typedef __attribute__((address_space(1))) unsigned int railgun_gu32;
extern "C" __launch_bounds__(256)
__global__ void railgun_fill_u32(
    unsigned char* __restrict__ dst,
    unsigned int value,
    unsigned int words)
{
    const unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = gridDim.x * blockDim.x;
    railgun_gu8* d = (railgun_gu8*)dst;
    for (unsigned int i = g; i < words; i += stride) *(railgun_gu32*)(d + (i << 2)) = value;
}
