// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Codec tile mask of the sealed expert-slab K2 prefill on gfx1151: the hipcc
// twin of the gfx1201 certified builder kernel `ep_tile_mask_pm_gfx1201`
// (`Gpu::ep_tile_mask`), same six arguments and the same bytes out. Writes a COPY
// of the grouped-GEMM tile ids in which every tile whose expert's table entry is
// not `rank` (here: the codec) is the sentinel -1, which the grouped IU4 GEMMs
// return on before any weight or activation load. `tile_ids` is never written.
//
//   live(t) = 0 <= tile_ids[t] < n_experts                     (unsigned compare)
//   take(t) = live(t) && owner_table[tile_ids[t]] == rank
//   tile_ids_owned[t] = take(t) ? tile_ids[t] : -1             (t < tiles)
//
// A negative id (the grouping's padding tail) and an id outside the table map to
// -1. Grid `[ceil(tiles / 256), 1, 1]`, block 256, one thread per tile, no LDS.

#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1151__)
#error "qwen4_slab_tile_mask.hip is gfx1151-only (gfx1201 uses ep_tile_mask_pm_gfx1201)"
#endif

extern "C" __global__ __launch_bounds__(256) void qwen4_slab_tile_mask(
    const int* __restrict__ tile_ids, int* __restrict__ tile_ids_owned,
    const int* __restrict__ owner_table, int tiles, int n_experts, int rank) {
    const int t = blockIdx.x * 256 + threadIdx.x;
    if (t >= tiles) return;
    const int id = tile_ids[t];
    const bool live = (unsigned)id < (unsigned)n_experts;
    tile_ids_owned[t] = (live && owner_table[id] == rank) ? id : -1;
}
