// Flash attention tile kernel — sequential dot product + 2-pass softmax.
//
// Grid: [n_heads, n_tiles, 1]. Block: [32, 1, 1] (one WAVE32).
// LDS: tile_size floats for scores + head_dim floats for Q = (128+128)*4 = 1024 B.
//
// QK uses one WAVE32 per head/tile. Each lane handles four dims per 128-dim
// half, then the wave reduces partial sums. V accumulation mirrors that layout:
// each lane writes four output dims per half. The partials stride must use the
// graph-capture max_tiles value, not the current launch tile count, because the
// reduce kernel always indexes the max_tiles-shaped scratch allocation.
//
// Works on RDNA1 (gfx1010), RDNA2 (gfx1030), RDNA3 (gfx1100).
#include <hip/hip_runtime.h>

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile(
    const float* __restrict__ q,         // [n_heads, head_dim]
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,        // [n_heads, n_tiles, 2 + head_dim]
    const int* __restrict__ pos_buf,     // pos_buf[0] = last position
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int max_tiles
) {
    const int seq_len = pos_buf[0] + 1;
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int tile_id = blockIdx.y;
    const int tile_start = tile_id * tile_size;
    const int tile_end = min(tile_start + tile_size, seq_len);
    if (tile_start >= seq_len) return;
    const int tile_len = tile_end - tile_start;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;

    const int blocks_per_head = head_dim / 32;
    const int total_blocks_per_pos = n_kv_heads * blocks_per_head;
    const int kv_head_block_start = kv_h * blocks_per_head;
    const int row_stride = total_blocks_per_pos * 34;
    const int n_halves = head_dim / 128;

    // LDS layout: [tile_size scores][head_dim Q values]
    extern __shared__ float sdata[];
    float* scores = sdata;
    float* q_lds  = sdata + tile_size;

    // Load Q into LDS. Each half covers 128 dims as 32 threads × 4 values.
    const float* q_head = q + h * head_dim;
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        q_lds[d_base + 0] = q_head[d_base + 0];
        q_lds[d_base + 1] = q_head[d_base + 1];
        q_lds[d_base + 2] = q_head[d_base + 2];
        q_lds[d_base + 3] = q_head[d_base + 3];
    }
    __syncthreads();

    // ═══ Phase A: compute tile_len dot products (wave-cooperative) ═══
    // Each thread handles 4 elements per 128-dim half, then the wave reduces
    // all partial sums via __shfl_xor.
    const int k_off  = (tid % 8) * 4;

    for (int t_local = 0; t_local < tile_len; t_local++) {
        int t = tile_start + t_local;
        float partial = 0.0f;
        for (int half = 0; half < n_halves; half++) {
            const int k_bi = half * 4 + tid / 8;
            const int k_blk_off = (kv_head_block_start + k_bi) * 34;
            const float* qb_thread = q_lds + k_bi * 32 + k_off;
            const unsigned char* kb = k_cache + (size_t)t * row_stride + k_blk_off;
            float ks = (float)*((const _Float16*)kb);
            partial += qb_thread[0] * (ks * (float)((signed char)kb[2 + k_off + 0]))
                     + qb_thread[1] * (ks * (float)((signed char)kb[2 + k_off + 1]))
                     + qb_thread[2] * (ks * (float)((signed char)kb[2 + k_off + 2]))
                     + qb_thread[3] * (ks * (float)((signed char)kb[2 + k_off + 3]));
        }
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor(partial, off);
        if (tid == 0)
            scores[t_local] = partial * scale_attn;
    }
    __syncthreads();

    // ═══ Phase B: find tile max ═══
    float local_max = -1e30f;
    for (int i = tid; i < tile_len; i += 32)
        local_max = fmaxf(local_max, scores[i]);
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, off));
    float tile_max = local_max;

    // ═══ Phase C: exp + sum ═══
    float local_sum = 0.0f;
    for (int i = tid; i < tile_len; i += 32) {
        float e = expf(scores[i] - tile_max);
        scores[i] = e;
        local_sum += e;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor(local_sum, off);
    float tile_sum = local_sum;
    __syncthreads();

    // ═══ Phase D: V-weighted accumulation ═══
    float out_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (int t_local = 0; t_local < tile_len; t_local++) {
        float w = scores[t_local];
        int t = tile_start + t_local;
        const unsigned char* vrow = v_cache + (size_t)t * row_stride;
        for (int half = 0; half < n_halves; half++) {
            const int d_base = half * 128 + tid * 4;
            const int bi = d_base / 32;
            const int bj_base = d_base % 32;
            const int v_blk_off = (kv_head_block_start + bi) * 34;
            const unsigned char* vb = vrow + v_blk_off;
            float vs = (float)*((const _Float16*)vb);
            out_vec[half * 4 + 0] += w * (vs * (float)((signed char)vb[2 + bj_base + 0]));
            out_vec[half * 4 + 1] += w * (vs * (float)((signed char)vb[2 + bj_base + 1]));
            out_vec[half * 4 + 2] += w * (vs * (float)((signed char)vb[2 + bj_base + 2]));
            out_vec[half * 4 + 3] += w * (vs * (float)((signed char)vb[2 + bj_base + 3]));
        }
    }

    // ═══ Write tile partials ═══
    // `partials` is allocated and reduced with max_tiles stride. This matters
    // for graph capture, where the launch grid is max_tiles even when the
    // current seq_len covers fewer tiles.
    float* p = partials + (h * max_tiles + tile_id) * (2 + head_dim);
    if (tid == 0) {
        p[0] = tile_max;
        p[1] = tile_sum;
    }
    for (int half = 0; half < n_halves; half++) {
        const int d_base = half * 128 + tid * 4;
        p[2 + d_base + 0] = out_vec[half * 4 + 0];
        p[2 + d_base + 1] = out_vec[half * 4 + 1];
        p[2 + d_base + 2] = out_vec[half * 4 + 2];
        p[2 + d_base + 3] = out_vec[half * 4 + 3];
    }
}
