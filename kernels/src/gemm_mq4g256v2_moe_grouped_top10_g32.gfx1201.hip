// SPDX-License-Identifier: Apache-2.0
// gfx1201 QT44 (MQ4G256V2) grouped top-10 gate/up, wide-pass (>= 16 rows) kernel family "g32".
// M = 1280, K = 2560 only (compile-time).  Same tensors / ABI as gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4:
//   (expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, grouped_rows, n_experts)
//
// Differences from o4_r8_x4 (all schedule/structure; the arithmetic is the production arithmetic, bit-exact):
//  * grid = [M / 32, ceil(grouped_rows / 16)], block = 256 (8 waves).  One block = one 16-slot tile x 32 output rows;
//    each wave owns 4 adjacent rows.  Live slots are processed in passes of R (R=8: one pass for <= 8 live positions).
//    Dead slots are never staged or computed (the production kernel stages 8 slots of X per subtile even for 1 live slot).
//  * X is staged through LDS ONCE per block-group for the live slots only, double buffered, with ONE barrier per group
//    (split s_barrier_signal / s_barrier_wait, no L0 invalidate: the production __syncthreads emits 2 barriers +
//    global_inv per group) and X register prefetch 2 groups ahead.
//  * weights stream through a 3-group register ring (production: distance 1, s_wait_loadcnt 0 every group).
//  * groups are visited in chain-major order 0,4,8 | 1,5,9 | 2,6 | 3,7: every static chain (g & 3) still accumulates its
//    groups in ascending order, the fold is (c0 + c1) + (c2 + c3), but only 3 accumulator sets are live at a time
//    (acc 96 VGPR instead of 128) -> 166 VGPR (R=8) / 113 VGPR (R=4) vs 253.
//  * the production `acc += w0*x0 + ... + w7*x7` lowers to t = fma(w0, x0, rn(w1*x1)); t = fma(w2, x2, t) ... fma(w7, x7, t);
//    acc += t (verified in the r8_x4 ISA); written here with explicit fmaf so the contraction cannot differ.
//  * dead slots below grouped_rows get +0.0 stores exactly like the reference (PAIR=0: every dead slot; PAIR=1: the few-row
//    o4_r2_x4 behaviour, a dead 2-slot pair is never written).  A negative / out-of-range expert id leaves the sentinel.
//  * NODZ=1 (entry points *_nodz): no +0.0 stores for dead slots at all (the consumer reads live rows only via inverse_perm).
//  * live slots may sit at any of the 16 tile positions (per-position live mask); a tile with > R live positions simply
//    makes a further pass over the weights.
// HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// x & 0x0F0F0F0F, opaque to the optimizer so each nibble converts with one v_cvt_f32_ubyteN (as production).
static __device__ __forceinline__ unsigned int gu32_nibble_bytes(unsigned int x) {
    unsigned int r;
    asm("v_and_b32 %0, 0x0f0f0f0f, %1" : "=v"(r) : "v"(x));
    return r;
}
// Two lane-0 trees in one pass (identical to production hf_lane0_tree_sum2): lanes 0-15 carry a, 16-31 carry b.
static __device__ __forceinline__ void gu32_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}
static __device__ __forceinline__ void gu32_decode(unsigned int ch, unsigned int cpk, float (&w)[8]) {
    const float sc = __half2float(__ushort_as_half((unsigned short)(ch & 0xFFFFu)));
    const float zp = __half2float(__ushort_as_half((unsigned short)(ch >> 16)));
    const unsigned int lo = gu32_nibble_bytes(cpk);
    const unsigned int hi = gu32_nibble_bytes(cpk >> 4);
    w[0] = sc * (float)(lo & 0xFFu) + zp;
    w[1] = sc * (float)(hi & 0xFFu) + zp;
    w[2] = sc * (float)((lo >> 8) & 0xFFu) + zp;
    w[3] = sc * (float)((hi >> 8) & 0xFFu) + zp;
    w[4] = sc * (float)((lo >> 16) & 0xFFu) + zp;
    w[5] = sc * (float)((hi >> 16) & 0xFFu) + zp;
    w[6] = sc * (float)(lo >> 24) + zp;
    w[7] = sc * (float)(hi >> 24) + zp;
}
static __device__ __forceinline__ void gu32_row(float& acc, const float (&w)[8], float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
    float t = __builtin_fmaf(w[0], x0, w[1] * x1);
    t = __builtin_fmaf(w[2], x2, t);
    t = __builtin_fmaf(w[3], x3, t);
    t = __builtin_fmaf(w[4], x4, t);
    t = __builtin_fmaf(w[5], x5, t);
    t = __builtin_fmaf(w[6], x6, t);
    t = __builtin_fmaf(w[7], x7, t);
    acc += t;
}
// LDS visibility + workgroup barrier without the __syncthreads fences (they add a global_inv per barrier).
static __device__ __forceinline__ void gu32_barrier() {
    asm volatile("s_wait_dscnt 0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    asm volatile("" ::: "memory");
}
// chain-major group order and chain of the i-th visited group
#define GU32_G(i) ((i) == 0 ? 0 : (i) == 1 ? 4 : (i) == 2 ? 8 : (i) == 3 ? 1 : (i) == 4 ? 5 : (i) == 5 ? 9 : (i) == 6 ? 2 : (i) == 7 ? 6 : (i) == 8 ? 3 : 7)
#define GU32_C(i) ((i) < 3 ? 0 : (i) < 6 ? 1 : (i) < 8 ? 2 : 3)

// R: live positions per pass (4 or 8), D: weight ring depth (groups), DX: X prefetch depth (groups), PAIR: dead-pair rule.
template <int R, int D, int DX, int PAIR, int NODZ = 0>
__device__ __forceinline__ void gu32_body(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int x_row_div, int grouped_rows, int n_experts, float4* xs)
{
    constexpr int NW = 8, M = 1280, K = 2560, RPB = NW * 4, LX = 2 * R / NW, NPASS = 16 / R;
    static_assert(LX * (NW / 2) == R, "stage mapping");
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int tid = threadIdx.x & 31;
    const int tile = blockIdx.y;
    const int chunk = blockIdx.x;
    const int slot_start = tile * 16;
    if (slot_start >= grouped_rows) return;  // block-uniform exits only (every wave takes part in the barriers)
    int fl[16];
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const int gs = slot_start + s;
        fl[s] = sorted_slot_index[gs < grouped_rows ? gs : grouped_rows - 1];
    }
    const int expert_id = expert_tile_ids[tile];
#pragma unroll
    for (int s = 0; s < 16; ++s) asm volatile("" :: "s"(fl[s]));
    if (expert_id < 0 || expert_id >= n_experts) return;
    unsigned int livemask = 0;
#pragma unroll
    for (int s = 0; s < 16; ++s) livemask |= (unsigned int)((slot_start + s < grouped_rows) && fl[s] >= 0) << s;
    {   // dead-slot +0.0 stores (128 B per position per block, one float4 per thread)
        constexpr int F4 = RPB / 4;
        const int pos = (int)threadIdx.x / F4;
        if (!NODZ && pos < 16 && slot_start + pos < grouped_rows && !((livemask >> pos) & 1u) && (!PAIR || ((livemask >> (pos & ~1)) & 3u)))
            *reinterpret_cast<float4*>(Y_grouped + (size_t)(slot_start + pos) * M + chunk * RPB + ((int)threadIdx.x % F4) * 4) =
                make_float4(0.f, 0.f, 0.f, 0.f);
    }
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    const __amdgpu_buffer_rsrc_t wrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const __amdgpu_buffer_rsrc_t xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)X_src, 0, 0x7FFFFFFF, 0x31004000);
    const int row0 = chunk * RPB + wave * 4;
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;
    const int part = (wave & 1) * 32 + tid;                 // float4 index inside a slot's 256-float group chunk
    const int lds_st = (part >> 1) + ((part & 1) << 5);     // transposed: a lane's two float4 sit 32 apart (conflict-free b128)
    const int xrow_b = row0 * 1360;

    for (int pass = 0; pass < NPASS; ++pass) {
        const unsigned int mask = (livemask >> (R * pass)) & ((1u << R) - 1u);
        if (mask == 0) continue;
        int stage_soff[LX];
        bool stage_live[LX];
#pragma unroll
        for (int j = 0; j < LX; ++j) {
            const int pos = j * (NW / 2) + (wave >> 1);
            stage_live[j] = (mask >> pos) & 1u;
            const int sp = slot_start + R * pass + pos;
            const int f2 = sorted_slot_index[sp < grouped_rows ? sp : grouped_rows - 1];
            const int x_row = !stage_live[j] ? 0 : (x_row_div > 1) ? (f2 / x_row_div) : f2;
            stage_soff[j] = x_row * K * 4;
            asm volatile("" :: "s"(stage_soff[j]));
        }
        float4 xl[DX][LX];
        unsigned int nh[D][4], npk[D][4];
#define GU32_ISSUE_X(GG, SL) do { \
        _Pragma("unroll") \
        for (int jj = 0; jj < LX; ++jj) \
            if (stage_live[jj]) xl[SL][jj] = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128( \
                xrsrc, part * 16, stage_soff[jj] + (GG) * 1024, 0)); \
    } while (0)
#define GU32_ISSUE_W(GG, SL) do { \
        _Pragma("unroll") \
        for (int rr = 0; rr < 4; ++rr) { \
            nh[SL][rr] = __builtin_amdgcn_raw_buffer_load_b32(wrsrc, hoff, xrow_b + rr * 1360 + (GG) * 136, 0); \
            npk[SL][rr] = __builtin_amdgcn_raw_buffer_load_b32(wrsrc, 8 + boff, xrow_b + rr * 1360 + (GG) * 136, 0); \
        } \
    } while (0)
#pragma unroll
        for (int j = 0; j < D; ++j) {
            if (j < DX) GU32_ISSUE_X(GU32_G(j), j);
            GU32_ISSUE_W(GU32_G(j), j);
        }
        float F[4][R], G[4][R], H[4][R];  // c0 then (c0+c1); c1 then c2; c3
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int s = 0; s < R; ++s) { F[r][s] = 0.f; G[r][s] = 0.f; H[r][s] = 0.f; }
#pragma unroll
        for (int i = 0; i < 10; ++i) {
#pragma unroll
            for (int j = 0; j < LX; ++j) {
                const int pos = j * (NW / 2) + (wave >> 1);
                if (stage_live[j]) xs[(i & 1) * (R * 64) + pos * 64 + lds_st] = xl[i % DX][j];
            }
            if (i + DX < 10) GU32_ISSUE_X(GU32_G(i + DX), i % DX);
            gu32_barrier();
            float w0[8], w1[8], w2[8], w3[8];
            {
                const unsigned int c0 = nh[i % D][0], c1 = nh[i % D][1], c2 = nh[i % D][2], c3 = nh[i % D][3];
                const unsigned int p0 = npk[i % D][0], p1 = npk[i % D][1], p2 = npk[i % D][2], p3 = npk[i % D][3];
                gu32_decode(c0, p0, w0); gu32_decode(c1, p1, w1); gu32_decode(c2, p2, w2); gu32_decode(c3, p3, w3);
            }
            if (i + D < 10) GU32_ISSUE_W(GU32_G(i + D), i % D);
#pragma unroll
            for (int s = 0; s < R; ++s) {
                if (!((mask >> s) & 1u)) continue;
                const float4 xa = xs[(i & 1) * (R * 64) + s * 64 + tid];
                const float4 xb = xs[(i & 1) * (R * 64) + s * 64 + 32 + tid];
                const int C = GU32_C(i);
                if (C == 0) {
                    gu32_row(F[0][s], w0, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(F[1][s], w1, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(F[2][s], w2, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(F[3][s], w3, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                } else if (C == 1 || C == 2) {
                    gu32_row(G[0][s], w0, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(G[1][s], w1, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(G[2][s], w2, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(G[3][s], w3, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                } else {
                    gu32_row(H[0][s], w0, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(H[1][s], w1, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(H[2][s], w2, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                    gu32_row(H[3][s], w3, xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w);
                }
            }
            if (i == 5) {  // chains 0 and 1 complete: F = c0 + c1; G restarts for chain 2
#pragma unroll
                for (int r = 0; r < 4; ++r)
#pragma unroll
                    for (int s = 0; s < R; ++s) { F[r][s] = F[r][s] + G[r][s]; G[r][s] = 0.f; asm volatile("" : "+v"(F[r][s]), "+v"(G[r][s])); }
            }
        }
#undef GU32_ISSUE_X
#undef GU32_ISSUE_W
        float sum[4][R];
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int s = 0; s < R; ++s) sum[r][s] = F[r][s] + (G[r][s] + H[r][s]);  // (c0 + c1) + (c2 + c3)
#pragma unroll
        for (int s = 0; s < R; ++s) {
            if (!((mask >> s) & 1u)) continue;
            gu32_lane0_tree_sum2(sum[0][s], sum[1][s]);
            gu32_lane0_tree_sum2(sum[2][s], sum[3][s]);
            if (tid == 0)
                *reinterpret_cast<float4*>(Y_grouped + (size_t)(slot_start + R * pass + s) * M + row0) =
                    make_float4(sum[0][s], sum[1][s], sum[2][s], sum[3][s]);
        }
    }
}

// rows > 16 (o4_r8_x4 dead-slot semantics).  Grid [40, ceil(grouped_rows/16)], block 256, LDS 16 KB, 166 VGPR.
extern "C" __global__ __launch_bounds__(256) void gemm_mq4g256v2_moe_grouped_top10_g32_r8(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index, const float* __restrict__ X_src, float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int grouped_rows, int n_experts) {
    __shared__ float4 xs[2 * 8 * 64];
    gu32_body<8, 3, 2, 0>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, x_row_div, grouped_rows, n_experts, xs);
}
// rows <= 16 with o4_r8_x4 dead-slot semantics.  Same grid/block, LDS 8 KB, 113 VGPR.
extern "C" __global__ __launch_bounds__(256) void gemm_mq4g256v2_moe_grouped_top10_g32_r4(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index, const float* __restrict__ X_src, float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int grouped_rows, int n_experts) {
    __shared__ float4 xs[2 * 4 * 64];
    gu32_body<4, 3, 2, 0>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, x_row_div, grouped_rows, n_experts, xs);
}
// few-row (<= 8 rows) drop-in for o4_r2_x4 (dead 2-slot pairs left untouched).  Same grid/block, LDS 8 KB.
extern "C" __global__ __launch_bounds__(256) void gemm_mq4g256v2_moe_grouped_top10_g32_r4_pair(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index, const float* __restrict__ X_src, float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int grouped_rows, int n_experts) {
    __shared__ float4 xs[2 * 4 * 64];
    gu32_body<4, 3, 2, 1>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, x_row_div, grouped_rows, n_experts, xs);
}

// *_nodz: identical, but dead slots are NOT written (consumer reads live rows only; oracle compares live rows).
extern "C" __global__ __launch_bounds__(256) void gemm_mq4g256v2_moe_grouped_top10_g32_r8_nodz(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index, const float* __restrict__ X_src, float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int grouped_rows, int n_experts) {
    __shared__ float4 xs[2 * 8 * 64];
    gu32_body<8, 3, 2, 0, 1>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, x_row_div, grouped_rows, n_experts, xs);
}
extern "C" __global__ __launch_bounds__(256) void gemm_mq4g256v2_moe_grouped_top10_g32_r4_nodz(
    const unsigned long long* __restrict__ expert_weight_ptrs, const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index, const float* __restrict__ X_src, float* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int grouped_rows, int n_experts) {
    __shared__ float4 xs[2 * 4 * 64];
    gu32_body<4, 3, 2, 0, 1>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src, Y_grouped, x_row_div, grouped_rows, n_experts, xs);
}
