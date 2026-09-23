// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Batched flash attention for Q8_0 KV (both K and V at Q8_0, 34-byte
// blocks of 32 elements: fp16 scale + 32 signed bytes). Stage 1 emits at
// most 64 balanced contiguous split-K partitions per query/head; each
// partition online-merges several `tile_size` KV tiles before writing one
// partial. Partials layout:
// [sub_batch × n_heads × partition_count × (2+head_dim)].
//
// LDS is tile_size*4 bytes regardless of seq_len, so this has NO context
// cap. Bounded partitions also keep the global partial buffer independent
// of context beyond the host-selected partition cap. This replaces the
// per-position attention_flash_q8_0 fallback that
// attention_q8_0_kv_batched_masked degenerated to above 15000 tokens
// (that kernel stages scores[max_ctx_len] in LDS → overflows ~64 KB at
// 16k+). See fix/q8-batched-masked-no-lds-cap.
//
// cos_theta/sin_theta are accepted for ABI compatibility with the shared
// launch_asym_flash_batched dispatcher but ignored — Q8 K is unrotated.
//
// Grid: [n_heads, partition_count, sub_batch_size]. Block: [32, 1, 1].
#include <hip/hip_runtime.h>
#include <assert.h>
#include "kv_slot_desc.h"

static __device__ __forceinline__ unsigned int q8_tile_load_u32(
    const unsigned char* p
) {
    unsigned int packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

static __device__ __forceinline__ unsigned long long q8_tile_load_u64(
    const unsigned char* p
) {
    unsigned long long packed;
    __builtin_memcpy(&packed, p, sizeof(packed));
    return packed;
}

extern "C" __launch_bounds__(32, 16)
__global__ void attention_flash_q8_0_tile_batched(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    const float* __restrict__ /*cos_theta*/,
    const float* __restrict__ /*sin_theta*/,
    const float* __restrict__ tree_bias,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    float scale_attn,
    int tile_size,
    int partition_count,
    int batch_offset,
    int block_start,
    int block_cols,
    int /*v_mode_bits*/,
    int window,
    const KvSlotDesc* __restrict__ slot_descs,
    const int* __restrict__ row_slot
) {
    const int h = blockIdx.x;
    const int partition_id = blockIdx.y;
    const int local_bid = blockIdx.z;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;

    const int global_bid = batch_offset + local_bid;
    const bool tree_mode = (tree_bias != nullptr);
    // positions[] is replay-updated while the captured block_start scalar is
    // stale, so derive the tree root from the device buffer in tree mode.
    const int eff_block_start = tree_mode ? positions[0] : block_start;
    const int seq_len = tree_mode ? (eff_block_start + block_cols) : (positions[global_bid] + 1);

    // row_slot is indexed by global row so sub-batching cannot select another
    // sequence's descriptor. The descriptor supplies addresses only; seq_len
    // remains the per-row causal bound for multi-row verification.
    const int slot = (row_slot != nullptr) ? row_slot[global_bid] : 0;
    const KvSlotDesc desc = (slot_descs != nullptr)
        ? slot_descs[slot]
        : kv_slot_legacy(seq_len, max_seq);
    const int per_pos_bytes = n_kv_heads * (head_dim / 32) * 34;
    const int win_lo = (window > 0) ? (seq_len - window) : 0;
    const int n_context_tiles = (seq_len + tile_size - 1) / tile_size;

    // Balanced contiguous partitions differ by at most one tile and keep all
    // partitions active whenever n_context_tiles >= partition_count.
    const int base_tiles = n_context_tiles / partition_count;
    const int extra_tiles = n_context_tiles % partition_count;
    const int partition_tiles = base_tiles + (partition_id < extra_tiles ? 1 : 0);
    const int first_tile = partition_id * base_tiles + min(partition_id, extra_tiles);
    const int last_tile = first_tile + partition_tiles;
    const int stride = 2 + head_dim;
    float* p = partials
        + ((long long)local_bid * n_heads + h) * partition_count * stride
        + partition_id * stride;
    if (partition_tiles == 0) {
        if (tid == 0) {
            p[0] = -1e30f;
            p[1] = 0.0f;
        }
        return;
    }

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int q_dim = n_heads * head_dim;
    const int blocks_per_head = head_dim / 32;
    const int kv_head_blk = kv_h * blocks_per_head;

    extern __shared__ float sdata[];
    float* scores = sdata;

    const float* q_head = q + local_bid * q_dim + h * head_dim;
    const int dpt = head_dim / 32;
    const int d0 = tid * dpt;
    const bool lane_one_q8_block = head_dim % 32 == 0 && dpt <= 32
                                && (d0 % 32) + dpt <= 32;
    float mq[16];
    for (int i = 0; i < dpt; i++) mq[i] = q_head[d0 + i];

    // Online-softmax state for this partition. out_vec stays unnormalised;
    // stage 2 merges partition maxima/sums and performs the final division.
    float partition_max = -1e30f;
    float partition_sum = 0.0f;
    float out_vec[16] = { 0.0f };

    for (int tile = first_tile; tile < last_tile; tile++) {
        const int tile_start = tile * tile_size;
        const int tile_end = min(tile_start + tile_size, seq_len);
        const int tile_len = tile_end - tile_start;

        // An entirely masked sliding-window tile contributes nothing.
        if (window > 0 && tile_end <= win_lo) continue;

        // Phase A: Q·K.
        for (int t_local = 0; t_local < tile_len; t_local++) {
            const int t = tile_start + t_local;
            if (t < win_lo) {
                if (tid == 0) scores[t_local] = -INFINITY;
                continue;
            }
            float partial = 0.0f;
            if (lane_one_q8_block && dpt == 8) {
                const int bi = d0 / 32;
                const int bj = d0 % 32;
                const unsigned char* kb = k_cache + kv_offset_for_k(desc, t, per_pos_bytes)
                                        + (kv_head_blk + bi) * 34;
                const float ks = (float)*((const _Float16*)kb);
                const unsigned long long packed = q8_tile_load_u64(kb + 2 + bj);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const signed char quant = (signed char)(packed >> (8 * i));
                    partial += mq[i] * (ks * (float)quant);
                }
            } else if (lane_one_q8_block && dpt == 4) {
                const int bi = d0 / 32;
                const int bj = d0 % 32;
                const unsigned char* kb = k_cache + kv_offset_for_k(desc, t, per_pos_bytes)
                                        + (kv_head_blk + bi) * 34;
                const float ks = (float)*((const _Float16*)kb);
                const unsigned int packed = q8_tile_load_u32(kb + 2 + bj);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const signed char quant = (signed char)(packed >> (8 * i));
                    partial += mq[i] * (ks * (float)quant);
                }
            } else {
                for (int i = 0; i < dpt; i++) {
                    const int d = d0 + i;
                    const int bi = d / 32;
                    const int bj = d % 32;
                    const unsigned char* kb = k_cache + kv_offset_for_k(desc, t, per_pos_bytes)
                                            + (kv_head_blk + bi) * 34;
                    const float ks = (float)*((const _Float16*)kb);
                    partial += mq[i] * (ks * (float)((signed char)kb[2 + bj]));
                }
            }
            for (int off = 16; off > 0; off >>= 1)
                partial += __shfl_xor(partial, off);
            if (tid == 0) {
                float s = partial * scale_attn;
                if (tree_mode && t >= eff_block_start)
                    s += tree_bias[global_bid * block_cols + (t - eff_block_start)];
                scores[t_local] = s;
            }
        }
        __syncthreads();

        // Phase B: tile maximum.
        float local_max = -1e30f;
        for (int i = tid; i < tile_len; i += 32)
            local_max = fmaxf(local_max, scores[i]);
        for (int off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor(local_max, off));
        const float tile_max = local_max;

        // Phase C: exponentials and tile denominator.
        float local_sum = 0.0f;
        for (int i = tid; i < tile_len; i += 32) {
            const float e = expf(scores[i] - tile_max);
            scores[i] = e;
            local_sum += e;
        }
        for (int off = 16; off > 0; off >>= 1)
            local_sum += __shfl_xor(local_sum, off);
        const float tile_sum = local_sum;
        __syncthreads();

        // Merge before accumulating V. Folding tile_corr into each V weight
        // avoids a second head_dim-sized temporary and its VGPR/scratch cost.
        const float merged_max = fmaxf(partition_max, tile_max);
        const float previous_corr =
            (partition_sum > 0.0f) ? expf(partition_max - merged_max) : 0.0f;
        const float tile_corr = expf(tile_max - merged_max);
        for (int i = 0; i < dpt; i++) out_vec[i] *= previous_corr;

        // Phase D: V-weighted accumulation (Q8_0).
        if (lane_one_q8_block && dpt == 8) {
            const int bi = d0 / 32;
            const int bj = d0 % 32;
            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float w = scores[t_local] * tile_corr;
                const int t = tile_start + t_local;
                const unsigned char* vbh = v_cache + kv_offset_for_v(desc, t, per_pos_bytes)
                                         + (kv_head_blk + bi) * 34;
                const float vs = (float)*((const _Float16*)vbh);
                const unsigned long long packed = q8_tile_load_u64(vbh + 2 + bj);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    const signed char quant = (signed char)(packed >> (8 * i));
                    out_vec[i] += w * (vs * (float)quant);
                }
            }
        } else if (lane_one_q8_block && dpt == 4) {
            const int bi = d0 / 32;
            const int bj = d0 % 32;
            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float w = scores[t_local] * tile_corr;
                const int t = tile_start + t_local;
                const unsigned char* vbh = v_cache + kv_offset_for_v(desc, t, per_pos_bytes)
                                         + (kv_head_blk + bi) * 34;
                const float vs = (float)*((const _Float16*)vbh);
                const unsigned int packed = q8_tile_load_u32(vbh + 2 + bj);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const signed char quant = (signed char)(packed >> (8 * i));
                    out_vec[i] += w * (vs * (float)quant);
                }
            }
        } else {
            for (int t_local = 0; t_local < tile_len; t_local++) {
                const float w = scores[t_local] * tile_corr;
                const int t = tile_start + t_local;
                const unsigned char* vb = v_cache + kv_offset_for_v(desc, t, per_pos_bytes);
                for (int i = 0; i < dpt; i++) {
                    const int d = d0 + i;
                    const int bi = d / 32;
                    const int bj = d % 32;
                    const unsigned char* vbh = vb + (kv_head_blk + bi) * 34;
                    const float vs = (float)*((const _Float16*)vbh);
                    out_vec[i] += w * (vs * (float)((signed char)vbh[2 + bj]));
                }
            }
        }

        partition_sum = partition_sum * previous_corr + tile_sum * tile_corr;
        partition_max = merged_max;
        // Lane 0 overwrites shared scores in the next iteration only after
        // every lane has consumed this tile's weights.
        __syncthreads();
    }

    if (tid == 0) {
        p[0] = partition_max;
        p[1] = partition_sum;
    }
    for (int i = 0; i < dpt; i++)
        p[2 + d0 + i] = out_vec[i];
}
