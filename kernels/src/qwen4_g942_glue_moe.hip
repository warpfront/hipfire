// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 stock-FP8 MoE glue folds for gfx942 (`HIPFIRE_QWEN4_G942_FOLD_MOE`): the seven small
// launches the sealed qt=55 grouped route runs around the routed expert GEMMs become three, and
// the launch count per layer drops from seven to two plus the combine that already existed.
//
//   qwen4_g942_moe_group_scatter_desc
//       = qwen4_moe_group_ranks + qwen4_moe_group_prefix + qwen4_moe_group_scatter
//         (qwen4_moe_scatter_stable_top10.hip) + gemm_fp8b128_moe_tile_desc_pad16
//         (gemm_fp8b128_moe_grouped_bf16mfma.gfx942.hip)
//   qwen4_g942_moe_unscatter_act
//       = fp8b128_moe_unscatter_silu_gfx942 + fp8b128_moe_shared_activation_gfx942
//   qwen4_g942_moe_combine_shared_add
//       = moe_down_combine_grouped_top10 + bf16_scaled_add_batched_f32 (the shared expert's
//         sigmoid-gated residual add that follows the combine on the qt=55 route)
//
// Every entry is bitwise the unfused sequence it replaces: the arithmetic below is copied
// textually from the sources named above (same casts, same rounding order, same precise expf),
// and every output word still has exactly one writer. Intermediates the unfused chain keeps in
// global scratch (ranks, per-chunk partial counts, chunk prefix, tile ids read back by the tile
// descriptor builder) live in LDS / registers here and are not outputs.
//
// Wave-size note: gfx942 is wave64, the single-block kernel is whole-block (1024 threads, no
// shuffles or wave masks; every cross-thread exchange goes through LDS behind __syncthreads).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

// ---------------------------------------------------------------------------------------------
// Fold 1: stable pad16 grouping + tile descriptors in one block.
//
// One workgroup of 1024 threads. Capacity: L = tokens * 10 slots <= G942_FOLD_MAXL (four
// 256-slot chunks, the chunking of qwen4_moe_group_ranks) and E <= 1024 experts; the host
// launcher admits nothing else.
//
//   topk       i32 [L]      routed expert per flat slot (outside [0, E) = unrouted marker)
//   counts     i32 [E]      pad16 count per expert                  (== qwen4_moe_group_prefix)
//   offsets    i32 [E+1]    pad16 exclusive offsets                 (== qwen4_moe_group_prefix)
//   sorted     i32 [Pmax]   flat slot per grouped row, -1 padding   (== qwen4_moe_group_scatter)
//   tiles      i32 [Pmax/16] expert per 16-row tile, -1 tail        (== qwen4_moe_group_scatter)
//   inverse    i32 [L]      grouped row per flat slot, -1 unrouted  (== qwen4_moe_group_scatter)
//   tile_desc  int4 [Pmax/16] {tile_id, 16 * t, 16, 0}   (== gemm_fp8b128_moe_tile_desc_pad16)
// ---------------------------------------------------------------------------------------------
#define G942_FOLD_THREADS 1024
#define G942_FOLD_MAXL 1024
#define G942_FOLD_MAXE 1024
#define G942_FOLD_MAXC 4

extern "C" __global__ __launch_bounds__(G942_FOLD_THREADS) void qwen4_g942_moe_group_scatter_desc(
    const int* __restrict__ topk, int* __restrict__ counts, int* __restrict__ offsets,
    int* __restrict__ sorted, int* __restrict__ tiles, int* __restrict__ inverse,
    int4* __restrict__ tile_desc, int L, int E, int Pmax) {
    __shared__ int s_e[G942_FOLD_MAXL];                    // routed expert, -1 = unrouted
    __shared__ int s_rank[G942_FOLD_MAXL];                 // rank among same-expert slots of the chunk
    __shared__ int s_part[G942_FOLD_MAXE * G942_FOLD_MAXC];       // [e][c] chunk count (qwen4 `partial`)
    __shared__ int s_cp[G942_FOLD_MAXE * (G942_FOLD_MAXC + 1)];   // [e][c] chunk prefix, [e][C] total
    __shared__ int s_pad[G942_FOLD_MAXE];                  // inclusive scan of the padded counts
    const int t = threadIdx.x;
    const int C = (L + 255) / 256;

    // qwen4_moe_group_ranks: the block clears its partial columns, then publishes per chunk.
    for (int k = t; k < E * C; k += G942_FOLD_THREADS) s_part[k] = 0;
    const int raw = t < L ? topk[t] : -1;
    s_e[t] = (raw >= 0 && raw < E) ? raw : -1;
    __syncthreads();
    if (t < L) {
        const int my = s_e[t];
        if (my < 0) {
            s_rank[t] = -1;
        } else {
            const int c = t >> 8;
            const int lo = c * 256;
            const int hi = min(lo + 256, L);
            int r = 0;
            for (int j = lo; j < t; ++j) r += s_e[j] == my;
            s_rank[t] = r;
            if (r == 0) {
                int cnt = 1;
                for (int j = t + 1; j < hi; ++j) cnt += s_e[j] == my;
                s_part[my * C + c] = cnt;
            }
        }
    }
    __syncthreads();

    // qwen4_moe_group_prefix: per-expert chunk prefix, padded count, inclusive Hillis-Steele scan
    // over the block (entries >= E are zero, which leaves entries < E exactly the E-wide scan).
    int total = 0;
    if (t < E) {
        for (int c = 0; c < C; ++c) {
            s_cp[t * (C + 1) + c] = total;
            total += s_part[t * C + c];
        }
        s_cp[t * (C + 1) + C] = total;
        counts[t] = (total + 15) / 16 * 16;
    }
    s_pad[t] = t < E ? (total + 15) / 16 * 16 : 0;
    __syncthreads();
    for (int off = 1; off < G942_FOLD_THREADS; off <<= 1) {
        const int v = t >= off ? s_pad[t - off] : 0;
        __syncthreads();
        s_pad[t] += v;
        __syncthreads();
    }
    if (t < E) offsets[t + 1] = s_pad[t];
    if (t == 0) offsets[0] = 0;

    // qwen4_moe_group_scatter + gemm_fp8b128_moe_tile_desc_pad16: offsets[m] is 0 for m == 0 and
    // s_pad[m - 1] otherwise; offsets[E] is the padded total.
    const int used = s_pad[E - 1];
    const int n = max(L, Pmax);
    for (int i = t; i < n; i += G942_FOLD_THREADS) {
        if (i < L) {
            const int e = s_e[i];
            if (e >= 0) {
                const int base = e == 0 ? 0 : s_pad[e - 1];
                const int pos = base + s_cp[e * (C + 1) + (i >> 8)] + s_rank[i];
                sorted[pos] = i;
                inverse[i] = pos;
            } else {
                inverse[i] = -1;  // unrouted slot: no grouped row
            }
        }
        if (i < Pmax) {
            int e = -1;
            if (i < used) {
                int lo = 0, hi = E;  // offsets[lo] <= i < offsets[hi]
                while (hi - lo > 1) {
                    const int mid = (lo + hi) >> 1;
                    const int off_mid = s_pad[mid - 1];  // mid >= 1 here
                    if (off_mid <= i) lo = mid; else hi = mid;
                }
                e = lo;
                const int local = i - (e == 0 ? 0 : s_pad[e - 1]);
                if (local >= s_cp[e * (C + 1) + C]) sorted[i] = -1;
            } else {
                sorted[i] = -1;
            }
            if ((i & 15) == 0) {
                tiles[i >> 4] = e;
                tile_desc[i >> 4] = int4{e, i, 16, 0};  // {tile_ids[t], 16 * t, 16, 0}
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fold 2: the routed SwiGLU unscatter and the shared expert's activation in one launch.
//
// Blocks [0, n_sorted) of grid.x are fp8b128_moe_unscatter_silu_gfx942 (row = blockIdx.x, column
// block = blockIdx.y, grid.y = ceil(mi / 256)); the blocks after them cover the shared
// activation, ceil(sh_rows * sh_mi / 256) blocks laid out row-major over (grid.y) columns:
// fp8b128_moe_shared_activation_gfx942's thread i is thread (blockIdx.x - n_sorted) * grid.y +
// blockIdx.y of this launch. The two parts touch disjoint buffers (routed activation vs shared
// activation + selector) and read nothing the other writes.
// ---------------------------------------------------------------------------------------------

// bf16 round-to-nearest-even of an F32 (non-finite values pass through).
static __device__ __forceinline__ float g942_bf16_rne(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

extern "C" __global__ __launch_bounds__(256) void qwen4_g942_moe_unscatter_act(
    const float* __restrict__ Y_grouped,
    const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation,
    int mi,
    int n_sorted,
    int slots,
    const float* __restrict__ sh_gate,
    const float* __restrict__ sh_up,
    float* __restrict__ sh_act,
    float* __restrict__ sh_selector,
    int sh_rows,
    int sh_mi,
    int act_blocks) {
    if ((int)blockIdx.x < n_sorted) {
        // fp8b128_moe_unscatter_silu_gfx942
        const int row = blockIdx.x;
        const int flat = sorted_slot_index[row];
        if (flat < 0 || flat >= slots) return;
        const int col = blockIdx.y * blockDim.x + threadIdx.x;
        if (col >= mi) return;
        const long long y_base = (long long)row * 2 * mi;
        const float g = g942_bf16_rne(Y_grouped[y_base + col]);
        const float u = g942_bf16_rne(Y_grouped[y_base + mi + col]);
        const float s = g942_bf16_rne(g / (1.0f + expf(-g)));
        activation[(long long)flat * mi + col] = g942_bf16_rne(s * u);
        return;
    }
    // fp8b128_moe_shared_activation_gfx942
    const long long block = (long long)(blockIdx.x - n_sorted) * gridDim.y + blockIdx.y;
    if (block >= act_blocks) return;
    const long long i = block * blockDim.x + threadIdx.x;
    if (i < sh_rows) {
        const float l = g942_bf16_rne(sh_selector[i]);
        sh_selector[i] = g942_bf16_rne(1.0f / (1.0f + expf(-l)));
    }
    if (i < (long long)sh_rows * sh_mi) {
        const float g = g942_bf16_rne(sh_gate[i]);
        const float u = g942_bf16_rne(sh_up[i]);
        const float s = g942_bf16_rne(g / (1.0f + expf(-g)));
        sh_act[i] = g942_bf16_rne(s * u);
    }
}

// ---------------------------------------------------------------------------------------------
// Fold 3: moe_down_combine_grouped_top10 with the shared expert's scaled add as its epilogue.
//
// The combine body is moe_down_combine_grouped_top10.hip's (F32 grouped rows). Its result is a
// BF16 value held in F32; bf16_scaled_add_batched_f32 then reads it back through bf16_round
// (exact for a BF16 value) as `lhs`, so the epilogue below runs on the in-register accumulator:
//   lhs     = accumulated
//   rhs     = bf16(shared_out[token * hidden + column])
//   gate    = bf16(scalar[token])
//   product = bf16(gate * rhs)
//   residual[token * hidden + column] = bf16(lhs + product)
// `shared_out` is the shared down's F32 rows ([tokens, hidden]), `scalar` the selector the
// shared activation left in place.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ __hip_bfloat16 g942_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float g942_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

__launch_bounds__(256, 4)
extern "C" __global__ void qwen4_g942_moe_combine_shared_add(
    const float* __restrict__ grouped_down,    // [grouped_rows * H], unweighted
    const int* __restrict__ inverse_perm,       // [N * 10], flat -> grouped row
    const int* __restrict__ topk_indices,       // [N * 10], expert IDs
    const float* __restrict__ topk_weights,     // [N * 10]
    float* __restrict__ residual,               // [N * H], in-place
    int hidden,
    int grouped_rows,
    int tokens,
    int n_experts,
    const float* __restrict__ shared_out,       // [N * H] shared down rows
    const float* __restrict__ scalar) {         // [N] shared selector
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    const size_t route_base = (size_t)token * 10;
    __hip_bfloat16 accumulated = g942_bf16_round(
        residual[(size_t)token * hidden + column]);
    // Ranks whose id is outside [0, n_experts) (the router's -1 marker) are
    // unrouted: never selected, contributing neither weight nor a grouped row.
    int used_ranks = 0;
    for (int rank = 0; rank < 10; ++rank) {
        const int expert = topk_indices[route_base + rank];
        if (expert < 0 || expert >= n_experts) used_ranks |= 1 << rank;
    }
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[route_base + rank];
            if (expert < selected_expert) {
                selected_expert = expert;
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;

        const __hip_bfloat16 weight =
            g942_bf16_round(topk_weights[route_base + selected_rank]);
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float raw_output =
            (grouped_row >= 0 && grouped_row < grouped_rows)
                ? grouped_down[(size_t)grouped_row * hidden + column]
                : 0.0f;
        const __hip_bfloat16 output = g942_bf16_round(raw_output);
        const __hip_bfloat16 weighted = g942_bf16_round(
            g942_bf16_to_f32(output) * g942_bf16_to_f32(weight));
        accumulated = g942_bf16_round(
            g942_bf16_to_f32(accumulated) + g942_bf16_to_f32(weighted));
    }
    // bf16_scaled_add_batched_f32 on the combined value.
    const size_t flat = (size_t)token * hidden + column;
    const __hip_bfloat16 rhs = g942_bf16_round(shared_out[flat]);
    const __hip_bfloat16 gate = g942_bf16_round(scalar[token]);
    const __hip_bfloat16 product = g942_bf16_round(
        g942_bf16_to_f32(gate) * g942_bf16_to_f32(rhs));
    residual[flat] = g942_bf16_to_f32(g942_bf16_round(
        g942_bf16_to_f32(accumulated) + g942_bf16_to_f32(product)));
}
