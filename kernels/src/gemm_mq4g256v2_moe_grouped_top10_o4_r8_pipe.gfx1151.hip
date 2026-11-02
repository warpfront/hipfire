// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT44 grouped top-10 gate/up O4×R8, unrolled twin.
//
// Bitwise twin of `gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4`
// (`gemm_mq4g256v2_moe_grouped_top10_o4_r8_x4.gfx1151.hip`) for the exact
// Qwen4 shape M = 1280, K = 2560 (ten MQ4G256V2 groups per row).  Same nine
// arguments, grid [ceil(M / 16), ceil(grouped_rows / 8), 1] and block 128:
// four waves per block each own four adjacent output rows of one eight-slot
// subtile, the subtile's X is staged through LDS once per group, and per
// output element the arithmetic is the incumbent's:
//   * the same MQ4G256V2 decode (fma(sc, (float)nibble, zp), f16 scale/zero
//     halves selected by lane < 16) and lane -> K map (lane t owns values
//     t*8 .. t*8 + 7 of every 256-value group);
//   * group g accumulates into static chain g & 3, groups in increasing
//     order, with the incumbent's 8-term expression as its code object
//     evaluates it (see HF_ROW);
//   * the fold (acc0 + acc1) + (acc2 + acc3) and the lane-0 tree
//     (hf_lane0_tree_sum2, offsets 16, 8, 4, 2, 1);
//   * dead slots skip accumulation and store +0.0; a subtile without a live
//     slot stores zeros; rows past M read row M - 1 and are never stored; a
//     negative expert id leaves the output untouched; slots past grouped_rows
//     are never stored.
// Only scheduling changes: K = 2560 is a launch contract, so the ten-group
// walk is fully unrolled (the incumbent runs a four-group loop plus a
// four-way tail switch at 253 VGPRs; unrolled, the chain index and buffer
// offsets are static and the kernel needs 202).  Halo oracle (0b0fd7848):
// 1.045x at 145 tokens, 1.068x at 273, 1.092x at 401; deeper prefetch,
// double-buffered LDS and eight-wave blocks were all slower than this.
// HIPFIRE_COMPILER_FLAGS: -mllvm -amdgpu-sched-strategy=max-ilp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Two lane-0 trees in one pass (verbatim from the incumbent): lanes 0-15
// carry a, lanes 16-31 carry b.  After the cross-row step lane i holds
// a[i] + a[i+16] and lane 16+i holds b[16+i] + b[i] (the same sum, IEEE add
// commutes); the in-row DPP steps then reduce both rows with the reference
// offsets, so a ends in lane 0 and b in lane 16 bit-identical to lane 0 of the
// `v += __shfl_down(v, off, 32)` tree (off = 16..1) applied to each.  a is
// returned in lane 0; b is broadcast (readlane 16).
static __device__ __forceinline__ void hf_lane0_tree_sum2(float& a, float& b) {
    const bool lo = (threadIdx.x & 31) < 16;
    float v = lo ? a : b;
    const float o = lo ? b : a;
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(o), __float_as_int(o), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    a = v;
    b = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16));
}
// x & 0x0F0F0F0F, opaque to the optimizer so each nibble converts with one
// v_cvt_f32_ubyteN instead of being re-canonicalized into v_bfe + cvt.
static __device__ __forceinline__ unsigned int hf_nibble_bytes(unsigned int x) {
    unsigned int r;
    asm("v_and_b32 %0, 0x0f0f0f0f, %1" : "=v"(r) : "v"(x));
    return r;
}

namespace {

constexpr int kSlots = 8;    // slots per subtile
constexpr int kGroups = 10;  // K / 256 for K = 2560
constexpr int kGroupBytes = 136;

struct Walk {
    __amdgpu_buffer_rsrc_t wrsrc, xrsrc;
    int row_soff[4];
    int stage_soff[4];
    int hoff, boff, stage_voff;
    bool live[kSlots];
    float acc[4][4][kSlots];  // [chain][row][slot]
    unsigned int nh[4], npk[4];
    float4 st[4];

    __device__ __forceinline__ void load(int g) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            nh[r] = __builtin_amdgcn_raw_buffer_load_b32(
                wrsrc, hoff, row_soff[r] + g * kGroupBytes, 0);
            npk[r] = __builtin_amdgcn_raw_buffer_load_b32(
                wrsrc, 8 + boff, row_soff[r] + g * kGroupBytes, 0);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            st[j] = __builtin_bit_cast(float4, __builtin_amdgcn_raw_buffer_load_b128(
                xrsrc, stage_voff, stage_soff[j] + g * 1024, 0));
    }

    template <int G>
    __device__ __forceinline__ void group(float4* __restrict__ xs) {
        constexpr int C = G & 3;
        unsigned int ch[4], cpk[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) { ch[r] = nh[r]; cpk[r] = npk[r]; }
#pragma unroll
        for (int j = 0; j < 4; ++j) xs[j * 128 + threadIdx.x] = st[j];
        __syncthreads();
        if constexpr (G + 1 < kGroups) load(G + 1);

#define HF_DECODE(R) \
        float w##R[8]; \
        { \
            const float sc = __half2float(__ushort_as_half( \
                (unsigned short)(ch[R] & 0xFFFFu))); \
            const float zp = __half2float(__ushort_as_half( \
                (unsigned short)(ch[R] >> 16))); \
            const unsigned int lo = hf_nibble_bytes(cpk[R]); \
            const unsigned int hi = hf_nibble_bytes(cpk[R] >> 4); \
            w##R[0] = __builtin_fmaf(sc, (float)(lo & 0xFFu), zp); \
            w##R[1] = __builtin_fmaf(sc, (float)(hi & 0xFFu), zp); \
            w##R[2] = __builtin_fmaf(sc, (float)((lo >> 8) & 0xFFu), zp); \
            w##R[3] = __builtin_fmaf(sc, (float)((hi >> 8) & 0xFFu), zp); \
            w##R[4] = __builtin_fmaf(sc, (float)((lo >> 16) & 0xFFu), zp); \
            w##R[5] = __builtin_fmaf(sc, (float)((hi >> 16) & 0xFFu), zp); \
            w##R[6] = __builtin_fmaf(sc, (float)(lo >> 24), zp); \
            w##R[7] = __builtin_fmaf(sc, (float)(hi >> 24), zp); \
        }
// The incumbent writes `acc += w0*x0 + w1*x1 + ... + w7*x7` and leaves the
// fusion to the backend's FP contraction (HIP default fast-honor-pragmas).
// Its gfx1151 code object evaluates every instance (all 256 in the module) as
// t = w1*x1; t = fma(w0, x0, t); t = fma(wk, xk, t) for k = 2..7; acc += t
// (DAGCombiner folds the left product of the first add when both products
// have one use).  Spelled out here with explicit fmas so a different
// surrounding schedule cannot flip that choice: the same source text under
// another schedule contracted some instances as fma(w1, x1, w0*x0) instead.
#define HF_ROW(R) do { \
            float t = w##R[1] * x1; \
            t = __builtin_fmaf(w##R[0], x0, t); \
            t = __builtin_fmaf(w##R[2], x2, t); \
            t = __builtin_fmaf(w##R[3], x3, t); \
            t = __builtin_fmaf(w##R[4], x4, t); \
            t = __builtin_fmaf(w##R[5], x5, t); \
            t = __builtin_fmaf(w##R[6], x6, t); \
            t = __builtin_fmaf(w##R[7], x7, t); \
            acc[C][R][slot] += t; \
        } while (0)
        HF_DECODE(0) HF_DECODE(1) HF_DECODE(2) HF_DECODE(3)
        const int tid = threadIdx.x & 31;
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot) {
            if (!live[slot]) continue;
            const float4 xa = xs[slot * 64 + tid * 2];
            const float4 xb = xs[slot * 64 + tid * 2 + 1];
            const float x0 = xa.x, x1 = xa.y, x2 = xa.z, x3 = xa.w;
            const float x4 = xb.x, x5 = xb.y, x6 = xb.z, x7 = xb.w;
            HF_ROW(0); HF_ROW(1); HF_ROW(2); HF_ROW(3);
        }
#undef HF_ROW
#undef HF_DECODE
        // ALU results carry no memory chain; pin them before the barrier.
#pragma unroll
        for (int slot = 0; slot < kSlots; ++slot)
            asm volatile("" : "+v"(acc[C][0][slot]), "+v"(acc[C][1][slot]),
                         "+v"(acc[C][2][slot]), "+v"(acc[C][3][slot]));
        __syncthreads();
    }
};

}  // namespace

__launch_bounds__(128, 1)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_top10_o4_r8_pipe_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int n_experts
) {
    (void)K;
    __shared__ float4 xs[kSlots * 64];
    constexpr int R = kSlots;
    // readfirstlane: the compiler cannot see that threadIdx.x >> 5 is
    // wave-uniform, and a VGPR row offset turns every buffer load into a
    // waterfall loop.
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int tid = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + wave) * 4;
    const int slot_start = blockIdx.y * R;
    // Block-uniform exits only: every wave takes part in the barriers.
    if (slot_start >= grouped_rows) return;

    int flat_raw[R];
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_raw[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }
    const int expert_id = expert_tile_ids[slot_start >> 4];
#pragma unroll
    for (int slot = 0; slot < R; ++slot) asm volatile("" :: "s"(flat_raw[slot]));
    if (expert_id < 0 || expert_id >= n_experts) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    Walk wk;
    // Rows past M read row M - 1 and are never stored.
    bool row_valid[4];
    constexpr int row_stride = kGroups * kGroupBytes;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        row_valid[r] = row0 + r < M;
        wk.row_soff[r] = (row_valid[r] ? row0 + r : M - 1) * row_stride;
    }

    int x_soff[R];
    bool any_live = false;
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        const int grouped_slot = slot_start + slot;
        const int flat = (grouped_slot < grouped_rows) ? flat_raw[slot] : -1;
        wk.live[slot] = flat >= 0;
        const int x_row = !wk.live[slot] ? 0
            : (x_row_div > 1) ? (flat / x_row_div) : flat;
        x_soff[slot] = x_row * (kGroups * 256) * 4;
        any_live |= wk.live[slot];
    }

#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int slot = 0; slot < R; ++slot) wk.acc[c][r][slot] = 0.0f;

    // A subtile with no live slot accumulates nothing; the epilogue stores
    // the zeros the full walk would have stored.
    if (any_live) {
        wk.wrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
        wk.xrsrc = __builtin_amdgcn_make_buffer_rsrc((void*)X_src, 0, 0x7FFFFFFF, 0x31004000);
        wk.boff = tid * 4;
        wk.hoff = (tid < 16) ? 0 : 4;
        // Staging: float4 v = j * 128 + threadIdx.x of the 8 x 64 chunk
        // belongs to slot 2j + (wave >> 1) (wave-uniform), float4
        // (wave & 1) * 32 + tid within it.
        wk.stage_voff = ((wave & 1) * 32 + tid) * 16;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            wk.stage_soff[j] = (wave >> 1) ? x_soff[2 * j + 1] : x_soff[2 * j];
        wk.load(0);
        wk.group<0>(xs);
        wk.group<1>(xs);
        wk.group<2>(xs);
        wk.group<3>(xs);
        wk.group<4>(xs);
        wk.group<5>(xs);
        wk.group<6>(xs);
        wk.group<7>(xs);
        wk.group<8>(xs);
        wk.group<9>(xs);
    }

    float sum[4][R];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int slot = 0; slot < R; ++slot)
            sum[r][slot] = (wk.acc[0][r][slot] + wk.acc[1][r][slot])
                + (wk.acc[2][r][slot] + wk.acc[3][r][slot]);
#pragma unroll
    for (int slot = 0; slot < R; ++slot) {
        hf_lane0_tree_sum2(sum[0][slot], sum[1][slot]);
        hf_lane0_tree_sum2(sum[2][slot], sum[3][slot]);
    }
    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < R; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
#pragma unroll
            for (int r = 0; r < 4; ++r)
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] = sum[r][slot];
        }
    }
}
