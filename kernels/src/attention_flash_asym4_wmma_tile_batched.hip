// Phase 1.1 WMMA flash attention — asym4 batched, hd=128, fp32 P, scalar P@V.
//
// Drop-in for attention_flash_asym4_tile_batched semantics: same args, same
// partials layout (consumed by attention_flash_asym_reduce_batched), same
// tile_size=128 contract. Different grid shape: [n_heads, ceil(N/BLOCK_M),
// max_tiles] instead of [n_heads, max_tiles, N]. One workgroup processes
// BLOCK_M=16 query rows at once for one (head, partition). blockIdx.z is
// the partition id; each row online-merges its balanced contiguous KV-tile
// range into one (m,l,N) partial. P=max_tiles, T=ceil(seq/TILE_SIZE):
// base=T/P, extra=T%P, count=base+(p<extra), first=p*base+min(p,extra).
// P=T is one KV tile per active partition; p>=T writes m=-1e30, l=0.
//
// Per Phase 1.0 precision spike result (ΔNLL/tok 0.0187 at 6× the gate),
// the P @ V multiply stays scalar fp32 — we DO NOT narrow P to fp16. WMMA
// is only used for the QK^T half. This trades ~half the spike's 5.91× ALU
// win for correctness.
//
// Per Phase 1.0 spike kernel-metadata, the full-WMMA spike sat at 239 VGPRs
// (38% of max occupancy on gfx1151). Phase 1.1 should land lower: dropping
// the second WMMA call removes p_reg / v_reg (16 VGPRs) and the 8-fragment
// acc_o is replaced with a 64-VGPR per-lane scalar accumulator (same total).
//
// LDS layout (static, per workgroup, hd=128):
//   scores[BLOCK_M][TILE_SIZE]  16 × 128 × 4 B = 8 KB (fp32 QK pre-softmax,
//                                                       then P post-softmax)
//   lds_q_pos[BLOCK_M]          64 B
//   lds_q_rotated[BLOCK_M][HD]  16 × 128 × 4 B = 8 KB (Givens-rotated Q in fp32,
//                                                       loaded once per workgroup)
//   ─────
//   ~16.1 KB per workgroup. Well under 64 KB limit.
//
// Tree-bias support: kept symmetric with attention_flash_asym4_tile_batched.
// When tree_bias != nullptr, seq_len is block_start + block_cols, and the
// in-block KV positions (t >= block_start) get the per-row bias added.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "turbo_common.h"
#include "givens_common.h"

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define BLOCK_M    16
#define WMMA_N     16
#define WMMA_K     16
#define TILE_SIZE  128

// No __launch_bounds__ — let the compiler choose VGPR count freely.
// With (32, 2) the compiler aggressively kept VGPRs low (76) and spilled
// out_vec[128] to private memory (528 B/lane scratch), making the kernel
// SLOWER than the scalar reference. Removing the hint lets it allocate up
// to ~256 VGPRs/wave, fit out_vec in registers, and run scratch-free.
extern "C"
__global__ void attention_flash_asym4_wmma_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ cos_theta,
    const float* __restrict__ sin_theta,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,                              // assumed = 128 for this kernel
    int max_seq,
    float scale_attn,
    int tile_size,                             // assumed = TILE_SIZE = 128
    int max_tiles,
    int batch_offset,
    int block_start,
    int block_cols
) {
    const int h = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int partition_id = blockIdx.z;
    if (h >= n_heads) return;

    const int tid = threadIdx.x;
    const int lane_row = tid & 15;             // 0..15: which row this lane owns in the M-tile
    const int half = tid >> 4;                 // 0 (lanes 0..15) or 1 (lanes 16..31)

    const int m_start = m_tile * BLOCK_M;
    const int my_bid_local = m_start + lane_row;
    const bool tree_mode = (tree_bias != nullptr);
    // hipGraph-safe block_start: under captured tree-verify replay the
    // `block_start` scalar kernarg is baked at capture time and goes stale as
    // the committed prefix grows. positions[] is re-uploaded (uncaptured) every
    // cycle and the linearized tree root sits at positions[0] == start_pos ==
    // block_start, so derive it from the device buffer instead of the scalar.
    // Eager (non-capture) value is identical, so this is a no-op off-graph.
    const int eff_block_start = tree_mode ? positions[0] : block_start;

    // ── Per-row OOB guard for this lane's row ───────────────────────────
    // For threads with my_bid_local >= chunk_batch_size, lane_row position
    // is OOB. We mask seq_len = 0 → tile_start always exceeds seq_len → skip.
    // The grid caller must dispatch ceil(N/BLOCK_M) workgroups; the trailing
    // OOB rows have positions undefined, so we don't dereference positions[]
    // for OOB lanes. Instead, batch_size is implied via the caller passing a
    // chunk size — we use the same trick the scalar kernel uses: bid >= chunk
    // is OK as long as we mask before writing to partials.
    //
    // For simplicity and matching the scalar kernel contract, we don't pass
    // an explicit chunk size; we rely on the caller to pass a positions[] of
    // appropriate length. OOB rows will simply read garbage positions[] and
    // get masked when their seq_len_for_this_lane is checked. This matches
    // the scalar kernel's "trust the caller" pattern.

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    // Runtime head_dim. Supports {128, 256}. LDS + out_vec are sized for the
    // max (256). For hd=128 the upper halves of those buffers are unused.
    const int HD = head_dim;
    const int MAX_HD = 256;                    // sizing-only constant

    const int k_bytes_per_head = 4 + HD / 2;   // 68 (hd=128) / 132 (hd=256)
    const int k_bytes_per_pos  = n_kv_heads * k_bytes_per_head;
    const int k_head_off       = kv_h * k_bytes_per_head;

    const int v_blocks_per_head = HD / 32;     // 4 (hd=128) / 8 (hd=256)
    const int v_total_blocks    = n_kv_heads * v_blocks_per_head;
    const int v_kv_head_blk     = kv_h * v_blocks_per_head;
    const int v_row_stride      = v_total_blocks * 34;

    // ── LDS storage ──────────────────────────────────────────────────────
    __shared__ float scores[BLOCK_M][TILE_SIZE];           // 8 KB
    __shared__ int   lds_q_pos[BLOCK_M];                   // 64 B
    __shared__ int   lds_seq_len[BLOCK_M];                 // 64 B
    __shared__ float lds_q_rot[BLOCK_M][MAX_HD];           // 16 KB (8 KB used for hd=128)
    __shared__ int   lds_first_tile[BLOCK_M];
    __shared__ int   lds_last_tile[BLOCK_M];
    __shared__ float lds_part_max[BLOCK_M];
    __shared__ float lds_part_sum[BLOCK_M];
    __shared__ float lds_prev_corr[BLOCK_M];
    __shared__ float lds_tile_corr[BLOCK_M];
    __shared__ float lds_row_max[BLOCK_M];
    __shared__ float lds_row_sum[BLOCK_M];
    __shared__ int   lds_row_tile_len[BLOCK_M];

    // ── Preload per-row positions + seq_len (16 lanes × 1 row each) ─────
    if (tid < BLOCK_M) {
        int bid = batch_offset + m_start + tid;
        // Note: callers must ensure positions[] is sized to cover the largest
        // workgroup's row range. For the trailing tile this may read garbage
        // for OOB rows — that's fine as long as we mask before writing.
        int p = positions[bid];
        lds_q_pos[tid]   = p;
        int seq = tree_mode ? (eff_block_start + block_cols) : (p + 1);
        lds_seq_len[tid] = seq;
        // Balanced contiguous partitions (Q8 authority): differ by at most
        // one tile and keep all partitions active whenever T >= P.
        const int n_context_tiles = (seq + TILE_SIZE - 1) / TILE_SIZE;
        const int base_tiles = n_context_tiles / max_tiles;
        const int extra_tiles = n_context_tiles % max_tiles;
        const int partition_tiles = base_tiles + (partition_id < extra_tiles ? 1 : 0);
        const int first_tile = partition_id * base_tiles + min(partition_id, extra_tiles);
        lds_first_tile[tid] = first_tile;
        lds_last_tile[tid] = first_tile + partition_tiles;
        lds_part_max[tid] = -1e30f;
        lds_part_sum[tid] = 0.0f;
    }
    __syncthreads();

    int wg_first = 0x7fffffff;
    int wg_last = 0;
    #pragma unroll
    for (int j = 0; j < BLOCK_M; j++) {
        if (lds_last_tile[j] > lds_first_tile[j]) {
            wg_first = min(wg_first, lds_first_tile[j]);
            wg_last = max(wg_last, lds_last_tile[j]);
        }
    }

    const int stride = 2 + HD;
    const int my_row = lane_row;
    const int my_local_bid = m_start + my_row;
    float* my_partials_row = partials
        + ((long long)my_local_bid * n_heads + h) * max_tiles * stride
        + partition_id * stride;

    // Uniform inactive partition: every row has p>=T. Write neutrals and
    // return together — no barrier lives past this point.
    if (wg_last <= wg_first) {
        if (tid < BLOCK_M) {
            float* pr = partials
                + ((long long)(m_start + tid) * n_heads + h) * max_tiles * stride
                + partition_id * stride;
            pr[0] = -1e30f;
            pr[1] = 0.0f;
        }
        return;
    }

    // ────────────────────────────────────────────────────────────────────
    // Load Q + apply Givens, staging into lds_q_rot[16][HD].
    // Gated to half=0 (lanes 0..15) — each lane handles row lane_row.
    // Lanes 16..31 idle here; their work pickups in the QK WMMA phase.
    // Saves ~50% of the Q-load + Givens cost vs running on all 32 lanes.
    // ────────────────────────────────────────────────────────────────────
    if (half == 0) {
        const float* q_head = q + (size_t)my_bid_local * q_dim + h * head_dim;
        for (int k0 = 0; k0 < HD; k0 += WMMA_K) {
            float buf[WMMA_K];
            #pragma unroll
            for (int j = 0; j < WMMA_K; j++) buf[j] = q_head[k0 + j];

            // Apply Givens to 4 (a,b,c,d) quadruples in this 16-wide chunk.
            // Each quadruple covers 2 pair-blocks. The scalar kernel uses
            // block_tid = half * 32 + tid_within_half; for our WMMA path
            // block_tid = half * 32 + (k0 % 128)/4 + q*2.
            // For hd=128, half is always 0 → block_tid = k0/4 + q*2.
            #pragma unroll
            for (int q = 0; q < WMMA_K / 4; q++) {
                int qs = q * 4;
                int dim = k0 + qs;
                int half_idx = dim / 128;
                int block_tid = half_idx * 32 + (dim % 128) / 4;
                givens_forward(buf[qs], buf[qs+1], buf[qs+2], buf[qs+3],
                               cos_theta, sin_theta, block_tid);
            }
            #pragma unroll
            for (int j = 0; j < WMMA_K; j++) {
                lds_q_rot[lane_row][k0 + j] = buf[j];
            }
        }
    }
    __syncthreads();

    const int n_kv_sub = TILE_SIZE / WMMA_N;   // 8 sub-blocks per tile
    constexpr int OUT_CHUNK = 32;
    const int v_dim_per_lane = HD / 2;
    const int v_chunks_per_lane = v_dim_per_lane / OUT_CHUNK;  // 2 (hd=128) / 4 (hd=256)
    const int v_dim_start = half * v_dim_per_lane;

    for (int tile = wg_first; tile < wg_last; tile++) {
        const int tile_start = tile * TILE_SIZE;

        // ────────────────────────────────────────────────────────────────
        // Phase A: QK^T via WMMA.
        //
        // For each 16-KV-position sub-block within the tile, accumulate
        // 8 K-chunks (head_dim=128 / 16) of fp16 WMMA. Result is C-layout
        // 16x16 fragment in acc_qk: lane has 8 cells at (row=2j+half, col=lane_row).
        //
        // After WMMA: scale + causal mask + tree-bias, then scatter to scores
        // LDS in (row, col) order.
        // ────────────────────────────────────────────────────────────────
        for (int sub = 0; sub < n_kv_sub; sub++) {
            const int kv_sub_start = tile_start + sub * WMMA_N;
            if (kv_sub_start >= max_seq) {
                // Past the cache — zero this 16-col range and continue.
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int row = 2 * j + half;
                    scores[row][sub * WMMA_N + lane_row] = -1e30f;
                }
                continue;
            }

            float8_t acc_qk = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};

            // Inner loop: 8 K-chunks of 16 head_dim values each.
            #pragma unroll
            for (int k0 = 0; k0 < HD; k0 += WMMA_K) {
                // A-fragment: lane lane_row's row, k0..k0+15 head_dim values
                // (already Givens-rotated, sitting in LDS).
                half16_t a_reg;
                #pragma unroll
                for (int j = 0; j < WMMA_K; j++) {
                    a_reg[j] = (_Float16) lds_q_rot[lane_row][k0 + j];
                }

                // B-fragment: lane lane_row's KV position, dequantized asym4
                // values at k0..k0+15.
                int my_kv = kv_sub_start + lane_row;
                half16_t b_reg;
                if (my_kv < max_seq) {
                    const unsigned char* kb = k_cache + (size_t)my_kv * k_bytes_per_pos + k_head_off;
                    float cnorm = *((const float*)kb);
                    // 16 nibbles at this k0 offset = 4 ushorts.
                    // Nibble offset in the K-byte stream: 4 (cnorm) + k0/2.
                    // For hd=128 there's no half multiplier (single half).
                    #pragma unroll
                    for (int w = 0; w < 4; w++) {
                        unsigned short pk = *((const unsigned short*)(kb + 4 + k0 / 2 + w * 2));
                        b_reg[w * 4 + 0] = (_Float16)(cnorm * TURBO_C4[(pk      ) & 0xF]);
                        b_reg[w * 4 + 1] = (_Float16)(cnorm * TURBO_C4[(pk >>  4) & 0xF]);
                        b_reg[w * 4 + 2] = (_Float16)(cnorm * TURBO_C4[(pk >>  8) & 0xF]);
                        b_reg[w * 4 + 3] = (_Float16)(cnorm * TURBO_C4[(pk >> 12) & 0xF]);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < WMMA_K; j++) b_reg[j] = (_Float16)0.0f;
                }

                acc_qk = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc_qk);
            }

            // Scale + per-cell causal mask + optional tree-bias.
            const int my_col_pos = kv_sub_start + lane_row;
            const int col_in_tile = sub * WMMA_N + lane_row;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                int cell_row = 2 * j + half;
                int q_pos_j  = lds_q_pos[cell_row];
                int sl_j     = lds_seq_len[cell_row];

                float val = acc_qk[j] * scale_attn;
                bool ok_kv   = (my_col_pos < max_seq) && (my_col_pos < sl_j);
                bool ok_mask = tree_mode ? true : (my_col_pos <= q_pos_j);
                float masked = (ok_kv && ok_mask) ? val : -1e30f;

                // Tree-bias addition (in-block only).
                if (tree_mode && ok_kv && my_col_pos >= eff_block_start) {
                    int bias_col = my_col_pos - eff_block_start;
                    int gbid = batch_offset + m_start + cell_row;
                    masked += tree_bias[gbid * block_cols + bias_col];
                }

                // Scatter to scores LDS: scores[cell_row][col_in_tile].
                // Only ONE half (half=0 or half=1) writes each row's cell — they
                // don't collide because cell_row depends on half.
                scores[cell_row][col_in_tile] = masked;
            }
        }
        __syncthreads();

        // ────────────────────────────────────────────────────────────────
        // Phase B + C: per-row softmax (max + exp + sum).
        //
        // Gated to half=0 (lanes 0..15): each lane owns one unique row, so there
        // is no LDS write race. Half=1 lanes idle here and pick back up in
        // Phase D where they own the upper V-dim half. Phase B/C state
        // (my_max, my_sum, my_tile_len) is staged into LDS for half=1 lanes to
        // consume in Phase D's per-row work.
        // ────────────────────────────────────────────────────────────────
        if (half == 0) {
            int my_seq = lds_seq_len[my_row];
            const bool row_owns =
                (tile >= lds_first_tile[my_row] && tile < lds_last_tile[my_row]);
            int my_tile_len = 0;
            if (row_owns) {
                my_tile_len = my_seq - tile_start;
                if (my_tile_len < 0) my_tile_len = 0;
                if (my_tile_len > TILE_SIZE) my_tile_len = TILE_SIZE;
            }

            float my_max = -1e30f;
            for (int i = 0; i < my_tile_len; i++) {
                my_max = fmaxf(my_max, scores[my_row][i]);
            }

            float my_sum = 0.0f;
            for (int i = 0; i < my_tile_len; i++) {
                float e = expf(scores[my_row][i] - my_max);
                scores[my_row][i] = e;
                my_sum += e;
            }

            // Zero unused columns so Phase D can safely iterate.
            for (int i = my_tile_len; i < TILE_SIZE; i++) {
                scores[my_row][i] = 0.0f;
            }

            lds_row_max[my_row]      = my_max;
            lds_row_sum[my_row]      = my_sum;
            lds_row_tile_len[my_row] = my_tile_len;

            if (row_owns) {
                const float merged_max = fmaxf(lds_part_max[my_row], my_max);
                const float previous_corr =
                    (lds_part_sum[my_row] > 0.0f)
                        ? expf(lds_part_max[my_row] - merged_max) : 0.0f;
                const float tile_corr = expf(my_max - merged_max);
                lds_prev_corr[my_row] = previous_corr;
                lds_tile_corr[my_row] = tile_corr;
                lds_part_sum[my_row] =
                    lds_part_sum[my_row] * previous_corr + my_sum * tile_corr;
                lds_part_max[my_row] = merged_max;
            } else {
                lds_prev_corr[my_row] = 1.0f;
                lds_tile_corr[my_row] = 0.0f;
            }
        }
        __syncthreads();

        // All lanes read per-row state from LDS for Phase D.
        int my_tile_len = lds_row_tile_len[my_row];
        const float previous_corr = lds_prev_corr[my_row];
        const float tile_corr = lds_tile_corr[my_row];
        const bool row_owns =
            (tile >= lds_first_tile[my_row] && tile < lds_last_tile[my_row]);

        // ────────────────────────────────────────────────────────────────
        // Phase D: scalar P @ V, fp32 accumulate.
        //
        // Process V-dims in chunks of OUT_CHUNK fp32 per lane to keep the
        // accumulator in registers (avoid the private-memory spill seen with
        // a single 128-element accumulator on hd=256). Chunks of 32 → out_chunk
        // fits in 32 VGPRs/lane (≈ same as scalar's 4-dim accumulator scaled by
        // 16-row tile width). With BLOCK_M=16 and 2 lanes/row, each chunk covers
        // 32 V-dims per lane = 64 V-dims per row; 2 chunks for hd=128, 4 for hd=256.
        // Online-merge N through the partials slot rather than a persistent
        // HD/2 register vector: out_vec[128] spilled 528 B/lane under
        // launch_bounds (see file header).
        // ────────────────────────────────────────────────────────────────
        for (int c = 0; c < v_chunks_per_lane; c++) {
            float out_chunk[OUT_CHUNK];
            #pragma unroll
            for (int d = 0; d < OUT_CHUNK; d++) out_chunk[d] = 0.0f;

            // OUT_CHUNK == 32 → covers exactly one Q8_0 block.
            int d_base = v_dim_start + c * OUT_CHUNK;
            int v_bi   = d_base / 32;
            int v_blk_off = (v_kv_head_blk + v_bi) * 34;

            if (my_tile_len > 0) {
                for (int k = 0; k < my_tile_len; k++) {
                    float w = scores[my_row][k] * tile_corr;
                    int t = tile_start + k;
                    const unsigned char* vbh = v_cache + (size_t)t * v_row_stride + v_blk_off;
                    float vs = (float)*((const _Float16*)vbh);
                    #pragma unroll
                    for (int d = 0; d < 32; d++) {
                        out_chunk[d] += w * (vs * (float)((signed char)vbh[2 + d]));
                    }
                }
            }

            if (row_owns) {
                if (previous_corr == 0.0f) {
                    #pragma unroll
                    for (int d = 0; d < OUT_CHUNK; d++) {
                        my_partials_row[2 + d_base + d] = out_chunk[d];
                    }
                } else {
                    #pragma unroll
                    for (int d = 0; d < OUT_CHUNK; d++) {
                        my_partials_row[2 + d_base + d] =
                            my_partials_row[2 + d_base + d] * previous_corr + out_chunk[d];
                    }
                }
            }
        }
        // Next tile overwrites scores; every lane must finish V first.
        __syncthreads();
    }

    if (half == 0) {
        my_partials_row[0] = lds_part_max[my_row];
        my_partials_row[1] = lds_part_sum[my_row];
    }
}
