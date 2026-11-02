// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//
// Row-batched per-request VMM entry points for the A3B (H16/KV2, head_dim
// 256, tile 128) singleton decode-attention kernels. Assembled exactly like
// `attention_flash_decode_vmm_rows.hip` (`vmm_rows_source_from`): the host
// emits
//
//   kv_slot_desc_vmm.h + "#define VMM_ROWS_<X> 1" + <singleton body with its
//   __global__ entry rewritten to `__device__ __forceinline__ <name>_vmm_row_body`>
//   + this file
//
// so every row runs the singleton kernel's own body (same arithmetic,
// reduction order and tile partitioning). Kept apart from the group-6 file so
// the shipped 27B `_vmm` modules' sources are untouched.
//
// Row addressing (shared with the group-6 wrappers):
//   * blockIdx.z = query row r,
//   * pos_buf = positions + r (causal end positions[r] + 1),
//   * q / gate / out advance by one row's singleton footprint,
//   * partials advance by n_heads * max_tiles * (2 + head_dim) f32 per row
//     (max_tiles = ceil(max_seq / tile_size) from the tile's max_seq kernarg;
//     the reduce takes max_tiles directly).

#ifdef VMM_ROWS_FP8_GQA8_TILE
// Group-8 fp8 flash tile (`attention_flash_fp8_e4m3_tile_gqa8_gfx1201`): K/V
// bases are the absolute VMM VAs of descs[row_slot[r]]; before touching KV the
// row's causal end is checked against the descriptor (positions[r] <
// mapped_positions <= logical_bound, else trap).
extern "C" __launch_bounds__(GQA_THREADS, 1)
__global__ void attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm(
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
    attention_flash_fp8_e4m3_tile_gqa8_gfx1201_vmm_row_body(
        q + (size_t)r * n_heads * head_dim,
        (const unsigned char*)d.k_base,
        (const unsigned char*)d.v_base,
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        positions + r,
        n_heads, n_kv_heads, head_dim, max_seq, scale_attn, tile_size, 0, 0);
}
#endif

#ifdef VMM_ROWS_ATTN_REDUCE_GATED_MQ
// Fused reduce + output gate + MQ FWHT rotate (the singleton A3B epilogue
// `attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201`, non-AWQ). Row r reads
// its own partials / gate rows and writes its rotated `wo` input row.
extern "C" __launch_bounds__(256, 4)
__global__ void attention_flash_q8_0_reduce_gated_mq_rotate_gfx1201_vmm(
    const float* __restrict__ partials,
    float* __restrict__ out,                    // [rows, n_heads * head_dim]
    const float* __restrict__ gate,             // [rows, n_heads * head_dim]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int head_dim,
    const int* __restrict__ positions,          // [rows]
    int tile_size,
    int max_tiles
) {
    const int r = blockIdx.z;
    HIPFIRE_ATTENTION_REDUCE_GATED_MQ_KERNEL_vmm_row_body(
        partials + (size_t)r * n_heads * max_tiles * (2 + head_dim),
        out + (size_t)r * n_heads * head_dim,
        gate + (size_t)r * n_heads * head_dim,
        signs1, signs2, n_heads, head_dim, positions + r, tile_size, max_tiles);
}
#endif
