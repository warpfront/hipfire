// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Row-batched per-request VMM entry points for the singleton flash-decode
// tile/reduce kernels. The host assembles each module as
//
//   kv_slot_desc_vmm.h + "#define VMM_ROWS_<X> 1" + <singleton body with its
//   __global__ entry rewritten to `__device__ __forceinline__ <name>_vmm_row_body`>
//   + this file
//
// so every row runs the singleton kernel's exact body (same arithmetic,
// reduction order and tile partitioning) with:
//   * blockIdx.z = query row r,
//   * K/V bases = the absolute VMM VAs of descs[row_slot[r]],
//   * pos_buf = positions + r (causal end positions[r] + 1),
//   * q / out / partials advanced by one row's singleton footprint
//     (partials row stride = n_heads * max_tiles * (2 + head_dim), with
//     max_tiles = ceil(max_seq / tile_size) from the max_seq kernarg).
// Before touching KV, the row's causal end is checked against the
// descriptor: positions[r] < mapped_positions <= logical_bound, else trap.

#ifdef VMM_ROWS_FP8_GQA_TILE
extern "C" __launch_bounds__(GQA_THREADS, 1)
__global__ void attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm(
    const float* __restrict__ q,                // [rows, n_heads, 256]
    float* __restrict__ partials,               // [rows, n_heads, max_tiles, 258]
    const int* __restrict__ positions,          // [rows]
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    const KvSlotDesc* __restrict__ slot_descs,  // [n_slots]
    const int* __restrict__ row_slot            // [rows]
) {
    const int r = blockIdx.z;
    const KvSlotDesc d = slot_descs[row_slot[r]];
    kv_vmm_check(d, positions[r]);
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    attention_flash_fp8_e4m3_tile_gqa_gfx1201_vmm_row_body(
        q + (size_t)r * n_heads * head_dim,
        (const unsigned char*)d.k_base,
        (const unsigned char*)d.v_base,
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        positions + r,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, tile_size, 0, 0);
}
#endif

#ifdef VMM_ROWS_Q8_TILE
extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile_vmm(
    const float* __restrict__ q,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    const KvSlotDesc* __restrict__ slot_descs,
    const int* __restrict__ row_slot
) {
    const int r = blockIdx.z;
    const KvSlotDesc d = slot_descs[row_slot[r]];
    kv_vmm_check(d, positions[r]);
    const int max_tiles = (max_seq + tile_size - 1) / tile_size;
    attention_flash_q8_0_tile_vmm_row_body(
        q + (size_t)r * n_heads * head_dim,
        (const unsigned char*)d.k_base,
        (const unsigned char*)d.v_base,
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        positions + r,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, tile_size, 0, 0);
}
#endif

#ifdef VMM_ROWS_DSPLIT_REDUCE
extern "C" __launch_bounds__(RD_THREADS, 1)
__global__ void attention_flash_reduce_dsplit_gfx1201_vmm(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    const int* __restrict__ positions,
    int tile_size,
    int max_tiles
) {
    const int r = blockIdx.z;
    attention_flash_reduce_dsplit_gfx1201_vmm_row_body(
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        out + (size_t)r * n_heads * head_dim,
        n_heads, head_dim, positions + r, tile_size, max_tiles);
}
#endif

#ifdef VMM_ROWS_Q8_REDUCE
extern "C" __launch_bounds__(256, 4)
__global__ void attention_flash_q8_0_reduce_vmm(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    const int* __restrict__ positions,
    int tile_size,
    int max_tiles
) {
    const int r = blockIdx.z;
    attention_flash_q8_0_reduce_vmm_row_body(
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        out + (size_t)r * n_heads * head_dim,
        n_heads, head_dim, positions + r, tile_size, max_tiles);
}
#endif
