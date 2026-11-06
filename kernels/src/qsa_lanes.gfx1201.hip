// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

// Multi-lane QSA decode core (fn-lanes), exact gfx1201, fp8 K/V, BF16 index keys:
// the prologue / pool / select / attention launches of up to QSA_LANES_MAX AR lanes
// (or verify windows of 1..=8 rows each) of one QSA layer as ONE launch per kernel,
// instead of four per lane. Lane `l` runs EXACTLY the incumbent device templates of
// tensor_ops.hip on its own state pointers, its own position and its own rows, so
// every output and state byte equals the per-lane launches':
//   indexed_attention_decode_prologue_fp8_lanes = indexed_attention_decode_prologue_fp8
//     (qsa_decode_prologue<QSA_KV_FP8, bf16>; grid [31, max_rows, lanes], lane = blockIdx.z)
//   indexed_attention_pool_rope_bf16_lanes      = indexed_attention_pool_rope_bf16
//     (qsa_pool_rope<bf16>; grid [max_l ceil(rows/4), dim_blocks, lanes], lane = blockIdx.z)
//   indexed_attention_select_bf16_lanes         = indexed_attention_select_bf16_batched
//     (qsa_select_batched<bf16>, fused LDS-resident parallel route, persistent mirror on the
//      lane's last row; grid [max_rows, 1, lanes], lane = blockIdx.z)
//   indexed_attention_attention_fp8_lanes       = indexed_attention_attention_fp8_batched
//     (qsa_attention_row_parallel<QSA_KV_FP8>, PerHead policy; grid
//      [heads, 1, lanes * max_rows], lane = blockIdx.z / max_rows, row = blockIdx.z % max_rows:
//      the template owns blockIdx.y as the channel chunk)
// Position, RoPE angle and `visible` derive from the lane's `pos` and its lane-local row,
// never from a global scratch row. A block outside its lane's rows returns before any
// __syncthreads. Appended to tensor_ops.hip (same helpers, same compile flags).
#if defined(__GFX12__)
#define QSA_LANES_MAX 8

struct QsaLanes {
    // Persistent per-lane state.
    unsigned char* full_keys[QSA_LANES_MAX];
    unsigned char* full_values[QSA_LANES_MAX];
    __hip_bfloat16* raw_idx[QSA_LANES_MAX];
    __hip_bfloat16* pooled[QSA_LANES_MAX];
    int* sel_persist[QSA_LANES_MAX];
    // Per-lane scratch: the lane's FIRST row; rows are contiguous inside a lane.
    float* index_row[QSA_LANES_MAX];
    float* qgate[QSA_LANES_MAX];
    float* keys[QSA_LANES_MAX];
    const float* values[QSA_LANES_MAX];
    int* selected[QSA_LANES_MAX];
    float* output[QSA_LANES_MAX];
    int pos[QSA_LANES_MAX];          // state.position of the lane (position_start)
    int rows[QSA_LANES_MAX];         // 1..=8
    int blocks[QSA_LANES_MAX];       // (pos + rows) / compress
    int first_block[QSA_LANES_MAX];  // pos / compress
};

extern "C" __global__ __launch_bounds__(256) void indexed_attention_decode_prologue_fp8_lanes(
    const __hip_bfloat16* index_q_norm,
    const __hip_bfloat16* q_norm,
    const __hip_bfloat16* k_norm,
    int index_heads,
    int index_dim,
    int index_kv_width,
    int heads,
    int kv_heads,
    int head_dim,
    RopeYarn yarn,
    QsaLanes lanes) {
    const int l = (int)blockIdx.z;
    if ((int)blockIdx.y >= lanes.rows[l]) return;
    qsa_decode_prologue<QSA_KV_FP8, __hip_bfloat16>(
        lanes.index_row[l], lanes.qgate[l], lanes.keys[l], lanes.values[l],
        lanes.full_keys[l], lanes.full_values[l], lanes.raw_idx[l], index_q_norm, q_norm,
        k_norm, index_heads, index_dim, index_kv_width, heads, kv_heads, head_dim,
        lanes.pos[l], yarn);
}

extern "C" __global__ void indexed_attention_pool_rope_bf16_lanes(
    const __hip_bfloat16* norm,
    int compress,
    int index_dim,
    RopeYarn yarn,
    QsaLanes lanes) {
    const int l = (int)blockIdx.z;
    // Grid x covers the widest lane; the template returns, before its first read or
    // barrier, for every block at or past the lane's block count (surplus blocks and
    // lanes with no completing block).
    qsa_pool_rope<__hip_bfloat16>(lanes.raw_idx[l], lanes.pooled[l], norm, lanes.blocks[l],
                                  compress, index_dim, lanes.first_block[l], yarn);
}

extern "C" __global__ void indexed_attention_select_bf16_lanes(
    int query_row_stride,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int capacity,
    QsaLanes lanes) {
    const int l = (int)blockIdx.z;
    const int rows = lanes.rows[l];
    if ((int)blockIdx.x >= rows) return;
    // The template mirrors only the lane's last row (and fills it with -1 when the
    // lane has no complete block), never touches global scores (scores_global null).
    qsa_select_batched<__hip_bfloat16>(
        lanes.index_row[l], lanes.pooled[l], lanes.selected[l], rows, query_row_stride,
        lanes.blocks[l], index_heads, index_dim, budget_blocks, compress, lanes.pos[l],
        capacity, lanes.sel_persist[l], nullptr, 0);
}

extern "C" __global__ __launch_bounds__(256) void indexed_attention_attention_fp8_lanes(
    int max_rows,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity,
    QsaLanes lanes) {
    const int l = (int)blockIdx.z / max_rows;
    const int row = (int)blockIdx.z - l * max_rows;
    if (row >= lanes.rows[l]) return;
    // qsa_attention_batched's per-row derivation with the lane's own position.
    const int visible = lanes.pos[l] + row + 1;
    const int block_count = visible / compress;
    const int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    qsa_attention_row_parallel<QSA_KV_FP8>(
        lanes.qgate[l] + row * 2 * n_heads * head_dim + (int)blockIdx.x * head_dim * 2,
        lanes.full_keys[l], lanes.full_values[l], lanes.selected[l] + row * capacity,
        lanes.output[l] + row * n_heads * head_dim, selected_len, visible, n_heads,
        n_kv_heads, head_dim, full_capacity);
}
#undef QSA_LANES_MAX
#endif
