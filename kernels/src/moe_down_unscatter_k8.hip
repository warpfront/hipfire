#include <hip/hip_runtime.h>

// Path 2 EP contribution materialization: restore grouped down rows to the
// canonical flat (token, k_rank) slot order before cross-rank gathering.
//
// `moe_scatter_permute_k8` writes inverse_perm[flat_slot] = grouped_slot.
// Unlike the local grouped combine, this pass deliberately does not apply
// top-k weights or touch the residual. The root's ordinary canonical combine
// owns that arithmetic after all rank-local contributions have been gathered.
//
// Shapes:
//   Y_down_grouped : [m_total × dim] f32
//   inverse_perm   : [total_slots] i32 (flat slot -> grouped row)
//   down_expanded  : [capacity × dim] f32 (writes only total_slots rows)
//
// Grid: (total_slots, ceil(dim/256)). Block: [256]. LDS: 0.

extern "C" __global__ void moe_down_unscatter_k8(
    const float* __restrict__ Y_down_grouped,
    const int* __restrict__ inverse_perm,
    float* __restrict__ down_expanded,
    int dim,
    int total_slots
) {
    const int slot = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (slot >= total_slots || col >= dim) return;

    const int grouped_slot = inverse_perm[slot];
    down_expanded[(long long)slot * dim + col] =
        Y_down_grouped[(long long)grouped_slot * dim + col];
}
