// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Exact-gfx942 wide-lane (>= 8 rows) glue of the stock-FP8 routed MoE: the router, the
// stable pad16 grouping + tile descriptors, the routed unscatter + shared activation and the
// combine + shared add. Each entry is BITWISE the output of the kernel it replaces on the same
// inputs (`moe_router_softmax_top10_f32`, `qwen4_g942_moe_group_scatter_desc`,
// `qwen4_g942_moe_unscatter_act`, `qwen4_g942_moe_combine_shared_add`): the arithmetic is
// copied from them (same casts, same rounding order, same expf, same left-to-right F32
// denominator); only the latency structure differs.
//
//   qwen4_g942_moe_wide_router
//       The incumbent's two serial 256-thread selection stages, three barriers and 512-term
//       LDS sum run ~15 us for 32 rows. Here one wave64 owns one token (four per workgroup, no
//       workgroup barrier): the 512 BF16-rounded logits live in 8 registers per lane, the stable
//       (value desc, index asc) top ten is ten rounds of ONE wave max over the packed key
//       `ord16(value) << 9 | (511 - index)`, and the softmax sums the same 512 exponentials left
//       to right through LDS exactly as the incumbent does. Needs the BF16 recipe
//       (`round_logits != 0`): the packed key is exact only for BF16-valued logits.
//   qwen4_g942_moe_wide_group_scatter_desc
//       Same outputs as the single-workgroup fold (counts, offsets, sorted, tiles, inverse,
//       tile descriptors), without its O(chunk) serial rank loops: a per-expert slot bitmask
//       (LDS atomicOr, order independent) gives every count and every stable rank by popcount.
//   qwen4_g942_moe_wide_unscatter_act
//       The fold's unscatter, iterated over the flat routed slots through `inverse` instead of
//       over every padded grouped row (every grouped row of a live tile has exactly one slot and
//       padding rows were skipped by the fold): ~15x fewer workgroups at 32 rows.
//   qwen4_g942_moe_wide_combine_shared_add
//       The fold's combine with the ten (expert-ordered) rank gathers issued as two independent
//       load batches instead of ten dependent load chains; the BF16 product/sum chain and its
//       ascending-expert order are untouched.
//
// gfx942 is wave64; the scatter is one 1024-thread workgroup with explicit barriers.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <math.h>

using __hip_bfloat16 = hip_bfloat16;

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "qwen4_g942_moe_wide.gfx942.hip is gfx942-only"
#endif

// bf16 round-to-nearest-even of an F32 (non-finite values pass through).
static __device__ __forceinline__ float mw_bf16_rne(float v) {
    if (!isfinite(v)) return v;
    const unsigned int u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// ---------------------------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------------------------

// Router fault word: identical to moe_router_softmax_top10_f32.hip (host-mapped, plain
// system-scope stores; the smaller (row, layer) code wins).
static __device__ __forceinline__ void mw_record_fault(unsigned long long* fault, int layer, int row) {
    if (fault == nullptr) return;
    const unsigned long long code =
        ((unsigned long long)(unsigned int)(row + 1) << 24) | (unsigned long long)(unsigned int)(layer + 1);
    const unsigned long long current =
        __hip_atomic_load(fault, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (current == 0ull || code < current)
        __hip_atomic_store(fault, code, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

static __device__ __forceinline__ void mw_fail_row(
    int* topk_idx, float* topk_w, int token, unsigned long long* fault, int layer, int row) {
    const size_t output_base = (size_t)token * 10;
    for (int rank = 0; rank < 10; ++rank) {
        topk_idx[output_base + rank] = -1;
        topk_w[output_base + rank] = 0.0f;
    }
    mw_record_fault(fault, layer, row);
}

// Wave64 unsigned max, the result uniform (SGPR) across the wave: quad / half-row / row DPP
// steps, then the four row leaders through readlane.
static __device__ __forceinline__ unsigned mw_wave_max_u32(unsigned v) {
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0xB1, 0xf, 0xf, false));   // quad_perm [1,0,3,2]
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x4E, 0xf, 0xf, false));   // quad_perm [2,3,0,1]
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x141, 0xf, 0xf, false));  // row_half_mirror
    v = max(v, (unsigned)__builtin_amdgcn_update_dpp(0, (int)v, 0x140, 0xf, 0xf, false));  // row_mirror
    const unsigned r0 = (unsigned)__builtin_amdgcn_readlane((int)v, 0);
    const unsigned r1 = (unsigned)__builtin_amdgcn_readlane((int)v, 16);
    const unsigned r2 = (unsigned)__builtin_amdgcn_readlane((int)v, 32);
    const unsigned r3 = (unsigned)__builtin_amdgcn_readlane((int)v, 48);
    return max(max(r0, r1), max(r2, r3));
}

// Monotone 16-bit order key of a finite BF16-valued float (+0 and -0 compare equal in the
// incumbent's float max, so -0 is canonicalized to +0 first).
static __device__ __forceinline__ unsigned mw_key16(float v) {
    unsigned u = __float_as_uint(v);
    if ((u << 1) == 0u) u = 0u;
    const unsigned h = u >> 16;
    return (h & 0x8000u) ? ((~h) & 0xFFFFu) : (h | 0x8000u);
}

// Inverse of mw_key16 (the canonical float of the key).
static __device__ __forceinline__ float mw_unkey16(unsigned key) {
    const unsigned h = (key & 0x8000u) ? (key & 0x7FFFu) : ((~key) & 0xFFFFu);
    return __uint_as_float(h << 16);
}

#define MW_ROUTER_WAVES 4

__launch_bounds__(256)
extern "C" __global__ void qwen4_g942_moe_wide_router(
    const float* __restrict__ logits, // [tokens, 512]
    int* __restrict__ topk_idx,       // [tokens, 10]
    float* __restrict__ topk_w,       // [tokens, 10]
    int n_exp,
    int top_k,
    int normalize_topk_prob,
    int round_logits,
    int tokens,
    unsigned long long* __restrict__ fault,
    int layer,
    int row_base) {
    __shared__ float s_exps[MW_ROUTER_WAVES][512];

    // Uniform across the workgroup; no workgroup barrier is used below. The BF16 recipe
    // (`round_logits`) is a precondition of the packed key; the host never launches without it.
    if (n_exp != 512 || top_k != 10 || round_logits == 0) return;
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int token = blockIdx.x * MW_ROUTER_WAVES + wave;
    if (token >= tokens) return;

    const float* row = logits + (size_t)token * 512;
    float x[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = mw_bf16_rne(row[lane + 64 * j]);

    // Any NaN/Inf logit (after the BF16 round trip) fails the whole row closed, exactly as the
    // incumbent: ids -1, weights 0, (row, layer) in the fault word. Wave-uniform vote.
    int nonfinite = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) nonfinite |= !isfinite(x[j]);
    if (__any(nonfinite)) {
        if (lane == 0) mw_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
        return;
    }

    unsigned key[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) key[j] = (mw_key16(x[j]) << 9) | (unsigned)(511 - (lane + 64 * j));

    float bv[10];
    int bi[10];
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        unsigned m = key[0];
#pragma unroll
        for (int j = 1; j < 8; ++j) m = max(m, key[j]);
        const unsigned best = mw_wave_max_u32(m);
        bv[rank] = mw_unkey16(best >> 9);
        bi[rank] = 511 - (int)(best & 511u);
#pragma unroll
        for (int j = 0; j < 8; ++j) key[j] = key[j] == best ? 0u : key[j];
    }

    const size_t output_base = (size_t)token * 10;
    if (lane == 0) {
#pragma unroll
        for (int rank = 0; rank < 10; ++rank) topk_idx[output_base + rank] = bi[rank];
    }

    // 512 exponentials in parallel, then the incumbent's left-to-right F32 sum.
    float* s_exp = s_exps[wave];
    const float max_logit = bv[0];
#pragma unroll
    for (int j = 0; j < 8; ++j) s_exp[lane + 64 * j] = expf(x[j] - max_logit);
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    __builtin_amdgcn_wave_barrier();
    if (lane == 0) {
        const float4* e4 = reinterpret_cast<const float4*>(s_exp);
        float denom = 0.0f;
#pragma unroll 8
        for (int q = 0; q < 128; ++q) {
            const float4 e = e4[q];
            denom += e.x;
            denom += e.y;
            denom += e.z;
            denom += e.w;
        }
        if (!(denom > 0.0f) || !isfinite(denom)) {
            mw_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
            return;
        }

        float probabilities[10];
        const float inv_denom = 1.0f / denom;
#pragma unroll
        for (int rank = 0; rank < 10; ++rank)
            probabilities[rank] = expf(bv[rank] - max_logit) * inv_denom;

        if (normalize_topk_prob != 0) {
            float selected_sum = 0.0f;
#pragma unroll
            for (int rank = 0; rank < 10; ++rank) selected_sum += probabilities[rank];
            if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) {
                mw_fail_row(topk_idx, topk_w, token, fault, layer, row_base + token);
                return;
            }
            const float inv_selected = 1.0f / selected_sum;
#pragma unroll
            for (int rank = 0; rank < 10; ++rank) probabilities[rank] *= inv_selected;
        }

#pragma unroll
        for (int rank = 0; rank < 10; ++rank)
            topk_w[output_base + rank] = mw_bf16_rne(probabilities[rank]);
    }
}

// ---------------------------------------------------------------------------------------------
// Stable pad16 grouping + tile descriptors, one workgroup.
//
//   topk       i32 [L]       routed expert per flat slot (outside [0, E) = unrouted marker)
//   counts     i32 [E]       pad16 count per expert
//   offsets    i32 [E+1]     pad16 exclusive offsets
//   sorted     i32 [Pmax]    flat slot per grouped row, -1 padding
//   tiles      i32 [Pmax/16] expert per 16-row tile, -1 tail
//   inverse    i32 [L]       grouped row per flat slot, -1 unrouted
//   tile_desc  int4 [Pmax/16]  {tile_id, 16 * t, 16, 0}
//
// Capacity: L <= MW_MAXL slots, E <= MW_MAXE experts (the host launcher admits nothing else).
// ---------------------------------------------------------------------------------------------
#define MW_THREADS 1024
#define MW_MAXL 640
#define MW_MAXE 512
#define MW_MAXW (MW_MAXL / 32)

extern "C" __global__ __launch_bounds__(MW_THREADS) void qwen4_g942_moe_wide_group_scatter_desc(
    const int* __restrict__ topk, int* __restrict__ counts, int* __restrict__ offsets,
    int* __restrict__ sorted, int* __restrict__ tiles, int* __restrict__ inverse,
    int4* __restrict__ tile_desc, int L, int E, int Pmax) {
    __shared__ unsigned s_mask[MW_MAXE * MW_MAXW];  // [e][w]: bit (t & 31) of word t >> 5 = slot t routes to e
    __shared__ int s_e[MW_MAXL];                    // routed expert per slot, -1 = unrouted
    __shared__ int s_rank[MW_MAXL];                 // stable rank among the slots of the same expert
    __shared__ int s_cnt[MW_MAXE];                  // raw count per expert
    __shared__ int s_pad[MW_MAXE];                  // inclusive scan of the padded counts
    __shared__ int s_wave[16];                      // wave totals of the scan
    const int t = threadIdx.x;
    const int W = (L + 31) >> 5;

    for (int k = t; k < E * W; k += MW_THREADS) s_mask[k] = 0u;
    if (t < L) {
        const int raw = topk[t];
        s_e[t] = (raw >= 0 && raw < E) ? raw : -1;
    }
    __syncthreads();
    if (t < L) {
        const int e = s_e[t];
        if (e >= 0) atomicOr(&s_mask[e * W + (t >> 5)], 1u << (t & 31));
    }
    __syncthreads();

    // Counts per expert (popcount of its mask) and the stable rank of every slot.
    int padded = 0;
    if (t < E) {
        int cnt = 0;
        for (int w = 0; w < W; ++w) cnt += __popc(s_mask[t * W + w]);
        padded = (cnt + 15) / 16 * 16;
        s_cnt[t] = cnt;
        counts[t] = padded;
    }
    if (t < L) {
        const int e = s_e[t];
        int r = -1;
        if (e >= 0) {
            r = 0;
            const int tw = t >> 5;
            for (int w = 0; w < tw; ++w) r += __popc(s_mask[e * W + w]);
            r += __popc(s_mask[e * W + tw] & ((1u << (t & 31)) - 1u));
        }
        s_rank[t] = r;
    }

    // Inclusive scan of the padded counts over the experts (entries >= E are zero).
    {
        const int lane = t & 63;
        const int wave = t >> 6;
        int v = padded;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const int n = __shfl_up(v, d, 64);
            if (lane >= d) v += n;
        }
        if (lane == 63) s_wave[wave] = v;
        __syncthreads();
        int add = 0;
        for (int w = 0; w < wave; ++w) add += s_wave[w];
        v += add;
        if (t < E) s_pad[t] = v;
    }
    __syncthreads();

    const int used = s_pad[E - 1];
    if (t < E) offsets[t + 1] = s_pad[t];
    if (t == 0) offsets[0] = 0;

    // Scatter: slot -> grouped row.
    if (t < L) {
        const int e = s_e[t];
        if (e >= 0) {
            const int base = e == 0 ? 0 : s_pad[e - 1];
            const int pos = base + s_rank[t];
            sorted[pos] = t;
            inverse[t] = pos;
        } else {
            inverse[t] = -1;  // unrouted slot: no grouped row
        }
    }
    // Per expert: -1 padding rows of its last tile and its tile descriptors.
    if (t < E) {
        const int cnt = s_cnt[t];
        const int base = t == 0 ? 0 : s_pad[t - 1];
        const int pad_end = s_pad[t];
        for (int p = base + cnt; p < pad_end; ++p) sorted[p] = -1;
        for (int p = base; p < pad_end; p += 16) {
            tiles[p >> 4] = t;
            tile_desc[p >> 4] = int4{t, p, 16, 0};
        }
    }
    // Tail beyond the used rows.
    for (int i = used + t; i < Pmax; i += MW_THREADS) {
        sorted[i] = -1;
        if ((i & 15) == 0) {
            tiles[i >> 4] = -1;
            tile_desc[i >> 4] = int4{-1, i, 16, 0};
        }
    }
    // A tile that starts below `used` but whose rows run past it cannot exist (padded counts are
    // multiples of 16 and `used` is their sum), so the first tail tile starts at `used` itself.
}

// ---------------------------------------------------------------------------------------------
// Routed SwiGLU unscatter (over flat slots) + shared expert activation.
//
// Blocks [0, slots) of grid.x are the routed unscatter (slot = blockIdx.x, column block =
// blockIdx.y, grid.y = ceil(mi / 256)); the blocks after them are the shared activation of
// qwen4_g942_moe_unscatter_act, laid out identically.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256) void qwen4_g942_moe_wide_unscatter_act(
    const float* __restrict__ Y_grouped,
    const int* __restrict__ inverse,
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
    if ((int)blockIdx.x < slots) {
        const int flat = blockIdx.x;
        const int row = inverse[flat];
        if (row < 0 || row >= n_sorted) return;
        const int col = blockIdx.y * blockDim.x + threadIdx.x;
        if (col >= mi) return;
        const long long y_base = (long long)row * 2 * mi;
        const float g = mw_bf16_rne(Y_grouped[y_base + col]);
        const float u = mw_bf16_rne(Y_grouped[y_base + mi + col]);
        const float s = mw_bf16_rne(g / (1.0f + expf(-g)));
        activation[(long long)flat * mi + col] = mw_bf16_rne(s * u);
        return;
    }
    const long long block = (long long)(blockIdx.x - slots) * gridDim.y + blockIdx.y;
    if (block >= act_blocks) return;
    const long long i = block * blockDim.x + threadIdx.x;
    if (i < sh_rows) {
        const float l = mw_bf16_rne(sh_selector[i]);
        sh_selector[i] = mw_bf16_rne(1.0f / (1.0f + expf(-l)));
    }
    if (i < (long long)sh_rows * sh_mi) {
        const float g = mw_bf16_rne(sh_gate[i]);
        const float u = mw_bf16_rne(sh_up[i]);
        const float s = mw_bf16_rne(g / (1.0f + expf(-g)));
        sh_act[i] = mw_bf16_rne(s * u);
    }
}

// ---------------------------------------------------------------------------------------------
// Combine + shared scaled add (batched rank gathers).
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ __hip_bfloat16 mw_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float mw_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

__launch_bounds__(256, 4)
extern "C" __global__ void qwen4_g942_moe_wide_combine_shared_add(
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
    const size_t flat = (size_t)token * hidden + column;

    // Round 1, all independent and branch free: the residual, the shared row and selector, and
    // per rank the expert id, the weight and the grouped row.
    const float residual_in = residual[flat];
    const float shared_value = shared_out[flat];
    const float selector = scalar[token];
    int experts[10];
    float weights[10];
    int rows[10];
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        experts[rank] = topk_indices[route_base + rank];
        weights[rank] = topk_weights[route_base + rank];
        rows[rank] = inverse_perm[route_base + rank];
    }
    // Round 2: the grouped down value of every rank (an unrouted or out-of-range row reads row 0
    // and is replaced by +0.0f, the incumbent's `raw_output`).
    float outputs[10];
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        const bool live_row = rows[rank] >= 0 && rows[rank] < grouped_rows;
        const float loaded = grouped_down[(size_t)(live_row ? rows[rank] : 0) * hidden + column];
        outputs[rank] = live_row ? loaded : 0.0f;
    }

    // Per-rank weighted product (independent of the visit order).
    float weighted[10];
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        const __hip_bfloat16 weight = mw_bf16_round(weights[rank]);
        const __hip_bfloat16 output = mw_bf16_round(outputs[rank]);
        weighted[rank] = mw_bf16_to_f32(mw_bf16_round(
            mw_bf16_to_f32(output) * mw_bf16_to_f32(weight)));
    }

    // Visit position of each routed rank: ascending expert id, ties by lowest rank (the
    // incumbent's strict-less selection); unrouted ranks (-1 / out of range) are never visited.
    bool routed[10];
    int n_sel = 0;
#pragma unroll
    for (int rank = 0; rank < 10; ++rank) {
        routed[rank] = experts[rank] >= 0 && experts[rank] < n_experts;
        n_sel += routed[rank] ? 1 : 0;
    }
    int pos[10];
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        int p = 0;
#pragma unroll
        for (int j = 0; j < 10; ++j) {
            if (j == i) continue;
            const bool before = experts[j] < experts[i] || (experts[j] == experts[i] && j < i);
            p += (routed[j] && before) ? 1 : 0;
        }
        pos[i] = routed[i] ? p : 16;
    }

    // Accumulate in visit order: step p adds the weighted value of the rank at position p.
    __hip_bfloat16 accumulated = mw_bf16_round(residual_in);
#pragma unroll
    for (int p = 0; p < 10; ++p) {
        float pick = 0.0f;
#pragma unroll
        for (int i = 0; i < 10; ++i) pick = pos[i] == p ? weighted[i] : pick;
        const __hip_bfloat16 next = mw_bf16_round(mw_bf16_to_f32(accumulated) + pick);
        accumulated = p < n_sel ? next : accumulated;
    }
    // bf16_scaled_add_batched_f32 on the combined value.
    const __hip_bfloat16 rhs = mw_bf16_round(shared_value);
    const __hip_bfloat16 gate = mw_bf16_round(selector);
    const __hip_bfloat16 product = mw_bf16_round(
        mw_bf16_to_f32(gate) * mw_bf16_to_f32(rhs));
    residual[flat] = mw_bf16_to_f32(mw_bf16_round(
        mw_bf16_to_f32(accumulated) + mw_bf16_to_f32(product)));
}
