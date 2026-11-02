// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Expert-deduplicating routed MoE kernels of the A3B (`qwen3_5_moe`, `.mq4rs`)
// lane-batched decode block (exact gfx1201, VMM continuous-batching exact
// route; see crates/rdna-compute/src/moe_decode_lanes.rs).
//
// With N lanes x 8 slots = up to 64 (lane, slot) "pairs" per layer-step, the
// lane-batched routed kernels read an expert's weights once per pair that
// selected it. Measured on the C8 steady set only ~46 of 64 picks are distinct
// experts, so ~28% of the routed weight bytes are re-reads of an expert another
// lane reads in the same layer-step. These kernels read each distinct expert
// ONCE: one wave per (row tile, lane, slot) pair reads the pair->expert table
// (tiny, L2-resident) and
//
//   * exits at once if an earlier pair (pair = lane * 8 + slot, ascending)
//     selected the same expert (follower), else
//   * is the LEADER of that expert: it gathers every pair that selected it
//     (the members: one per lane for router-produced ids, up to 64 in
//     general), loads the expert's weights for its row tile ONCE and computes
//     the member columns from them (MD_CHUNK columns per weight read), writing
//     each column to that pair's own output slice.
//
// No host sync, no atomics, no global ordering assumption: the leader of an
// expert is a function of the id table alone.
//
// Per-column arithmetic is the singleton kernel's, operation for operation:
//
// gate/up (singleton `gemv_mq4g256v2_moe_gate_up_k8_indexed`, grid (512, 8)
// x 32 threads, K = 2048): thread `l` owns weights l*8..l*8+7 of every
// 256-group, header dword `l < 16 ? 0 : 4`; w_i = fma(scale, (float)nibble_i,
// zero); group partial E = w0*x0 + w1*x1 + ... + w7*x7 with the singleton's
// compiler contraction, which differs between the gate stream and the up
// stream (read off the singleton's gfx1201 ISA and pinned with explicit fma /
// __fmul_rn so no column loop can change it): GATE multiplies the FIRST term
// then fma's terms 1..7 in order; UP multiplies the SECOND term, fma's the
// first onto it, then fma's terms 2..7 in order. Accumulator j (= group % 4)
// sums the groups j and j+4 onto +0.0f; lane partial = (a0 + a1) + (a2 + a3);
// cross-lane sum = shuffle-down 16, 8, 4, 2, 1.
//
// routed down (singleton `gemv_mq4g128v2_moe_ninepath_d4`): the per-slot part
// is separable from the combine. Per (row, slot) the singleton forms the
// activation silu(gate)*up -> FWHT128 (qt53) in registers, dots the expert's
// 68-byte-group row (acc = 0; acc += s per group, s = d1*x1 rounded, then
// fma d0*x0, d2*x2, d3*x3), and folds the 32 lane partials in the
// shuffle-down tree order (16, 8, 4, 2, 1); the 8 slot parts are then combined
// per row by `a = 0; a = fma(topk_w[k], part[k], a)` for k = 0..7 and added
// into the residual. Stage A (`..._down_dedup_rows`) computes the parts of every
// (lane, slot) pair into the expanded `[lane][slot][2048]` scratch (one weight
// read per distinct expert); stage B (`..._down_combine_rows`) is the
// singleton's combine, per lane.
//
// Geometry is fixed (A3B): hidden 2048, routed intermediate 512, gate/up
// K = 2048 (8 groups of 136 B), down K = 512 (4 groups of 68 B).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef MD_MINB
#define MD_MINB 8           // __launch_bounds__ min blocks per multiprocessor (register budget)
#endif
#ifndef MD_GU_ROWS
#define MD_GU_ROWS 4        // gate/up row pairs per wave (512 % MD_GU_ROWS == 0)
#endif
#ifndef MD_CHUNK
#define MD_CHUNK 2          // columns computed per weight read; more members loop (weights re-read from cache)
#endif

// ---------------------------------------------------------------------------
// Pair table
// ---------------------------------------------------------------------------

// Bit p of the result is set iff pair p (< npairs) selected expert `e`.
static __device__ __forceinline__ unsigned long long md_member_mask(
    const int* __restrict__ ids, int npairs, int e
) {
    const int t = threadIdx.x;
    const bool in0 = (t < npairs) && (ids[t] == e);
    const bool in1 = (t + 32 < npairs) && (ids[t + 32] == e);
    const unsigned long long m0 = __builtin_amdgcn_ballot_w32(in0);
    const unsigned long long m1 = __builtin_amdgcn_ballot_w32(in1);
    return m0 | (m1 << 32);
}

static __device__ __forceinline__ int md_ctz64(unsigned long long m) {
    return __builtin_ctzll(m);
}

// ---------------------------------------------------------------------------
// Gate/up
// ---------------------------------------------------------------------------

template <int C>
static __device__ __forceinline__ void md_gate_up_cols(
    const char* __restrict__ A,             // expert gate/up weight [1024][8 x 136 B]
    const float* __restrict__ x,            // [lanes][2048]
    const int* pairs,                       // C pair indices (uniform)
    float* __restrict__ y_gate,             // [pair][512]
    float* __restrict__ y_up,               // [pair][512]
    int row0
) {
    const int tid = threadIdx.x;
    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;

    const float* xc[C];
    #pragma unroll
    for (int c = 0; c < C; c++) xc[c] = x + (size_t)(pairs[c] >> 3) * 2048 + tid * 8;

    const char* gate_ptr[MD_GU_ROWS];
    const char* up_ptr[MD_GU_ROWS];
    #pragma unroll
    for (int rr = 0; rr < MD_GU_ROWS; rr++) {
        gate_ptr[rr] = A + (long long)(row0 + rr) * 8 * 136;
        up_ptr[rr] = A + (long long)(row0 + rr + 512) * 8 * 136;
    }

    float ga[C][MD_GU_ROWS][4], ua[C][MD_GU_ROWS][4];
    #pragma unroll
    for (int c = 0; c < C; c++)
        #pragma unroll
        for (int rr = 0; rr < MD_GU_ROWS; rr++)
            #pragma unroll
            for (int j = 0; j < 4; j++) { ga[c][rr][j] = 0.0f; ua[c][rr][j] = 0.0f; }

    #pragma unroll 1
    for (int q = 0; q < 2; q++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int g = q * 4 + j;
            float gw[MD_GU_ROWS][8], uw[MD_GU_ROWS][8];
            #pragma unroll
            for (int rr = 0; rr < MD_GU_ROWS; rr++) {
                const char* gp = gate_ptr[rr] + g * 136;
                const char* upp = up_ptr[rr] + g * 136;
                const unsigned int gh = *(const unsigned int*)(gp + hoff);
                const unsigned int gk = *(const unsigned int*)(gp + 8 + boff);
                const unsigned int uh = *(const unsigned int*)(upp + hoff);
                const unsigned int uk = *(const unsigned int*)(upp + 8 + boff);
                const float gsc = __half2float(__ushort_as_half((unsigned short)(gh & 0xFFFFu)));
                const float gzp = __half2float(__ushort_as_half((unsigned short)(gh >> 16)));
                const float usc = __half2float(__ushort_as_half((unsigned short)(uh & 0xFFFFu)));
                const float uzp = __half2float(__ushort_as_half((unsigned short)(uh >> 16)));
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    gw[rr][i] = __fmaf_rn(gsc, (float)((gk >> (4 * i)) & 0xFu), gzp);
                    uw[rr][i] = __fmaf_rn(usc, (float)((uk >> (4 * i)) & 0xFu), uzp);
                }
            }
            #pragma unroll
            for (int c = 0; c < C; c++) {
                const float4 v0 = *reinterpret_cast<const float4*>(xc[c] + g * 256);
                const float4 v1 = *reinterpret_cast<const float4*>(xc[c] + g * 256 + 4);
                const float xv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
                #pragma unroll
                for (int rr = 0; rr < MD_GU_ROWS; rr++) {
                    // gate stream: first term multiplied, rest fma'd in order.
                    float e = __fmul_rn(gw[rr][0], xv[0]);
                    e = __fmaf_rn(gw[rr][1], xv[1], e);
                    #pragma unroll
                    for (int i = 2; i < 8; i++) e = __fmaf_rn(gw[rr][i], xv[i], e);
                    ga[c][rr][j] += e;
                    // up stream: second term multiplied, first fma'd onto it, rest in order.
                    float f = __fmul_rn(uw[rr][1], xv[1]);
                    f = __fmaf_rn(uw[rr][0], xv[0], f);
                    #pragma unroll
                    for (int i = 2; i < 8; i++) f = __fmaf_rn(uw[rr][i], xv[i], f);
                    ua[c][rr][j] += f;
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < C; c++) {
        #pragma unroll
        for (int rr = 0; rr < MD_GU_ROWS; rr++) {
            float gacc = (ga[c][rr][0] + ga[c][rr][1]) + (ga[c][rr][2] + ga[c][rr][3]);
            float uacc = (ua[c][rr][0] + ua[c][rr][1]) + (ua[c][rr][2] + ua[c][rr][3]);
            for (int offset = 16; offset > 0; offset >>= 1)
                gacc += __shfl_down(gacc, offset);
            for (int offset = 16; offset > 0; offset >>= 1)
                uacc += __shfl_down(uacc, offset);
            if (tid == 0) {
                y_gate[(size_t)pairs[c] * 512 + row0 + rr] = gacc;
                y_up[(size_t)pairs[c] * 512 + row0 + rr] = uacc;
            }
        }
    }
}

// Grid (512 / MD_GU_ROWS row tiles, 8 slots, lanes) x 32 threads.
extern "C" __launch_bounds__(32, MD_MINB)
__global__ void gemv_mq4g256v2_moe_gate_up_k8_dedup_rows(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp], shared
    const int* __restrict__ topk_indices,               // [lanes, 8]
    const float* __restrict__ x,                        // [lanes, 2048]
    float* __restrict__ y_gate,                         // [lanes, 8, 512]
    float* __restrict__ y_up,                           // [lanes, 8, 512]
    int lanes, int n_experts
) {
    const int row0 = blockIdx.x * MD_GU_ROWS;
    const int pair = blockIdx.z * 8 + blockIdx.y;
    const int npairs = lanes * 8;
    const int e = __builtin_amdgcn_readfirstlane(topk_indices[pair]);
    unsigned long long members = md_member_mask(topk_indices, npairs, e);
    if (md_ctz64(members) != pair) return;           // follower

    if ((unsigned)e >= (unsigned)n_experts) {
        if (threadIdx.x == 0) {
            while (members) {
                const int p = md_ctz64(members);
                members &= members - 1;
                for (int rr = 0; rr < MD_GU_ROWS; rr++) {
                    y_gate[(size_t)p * 512 + row0 + rr] = 0.0f;
                    y_up[(size_t)p * 512 + row0 + rr] = 0.0f;
                }
            }
        }
        return;
    }
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_ptrs[e]);

    while (members) {
        int pc[MD_CHUNK];
        int c = 0;
        while (c < MD_CHUNK && members) {
            pc[c++] = md_ctz64(members);
            members &= members - 1;
        }
        switch (c) {
            case 1: md_gate_up_cols<1>(A, x, pc, y_gate, y_up, row0); break;
            case 2: md_gate_up_cols<2>(A, x, pc, y_gate, y_up, row0); break;
#if MD_CHUNK >= 3
            case 3: md_gate_up_cols<3>(A, x, pc, y_gate, y_up, row0); break;
#endif
#if MD_CHUNK >= 4
            default: md_gate_up_cols<4>(A, x, pc, y_gate, y_up, row0); break;
#else
            default: break;
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Routed down: stage A (per-pair slot parts, one weight read per expert)
// ---------------------------------------------------------------------------

static __device__ __forceinline__ const char* md_uniform_ptr(const char* p) {
    const unsigned long long v = (unsigned long long)p;
    const unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)v);
    const unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(v >> 32));
    return (const char*)(((unsigned long long)hi << 32) | lo);
}

// silu(gate)*up -> FWHT128 (qt53), lane `lane` owns elements 4*lane..4*lane+3
// of each of the four 128-wide groups; the singleton ninepath's own code.
static __device__ __forceinline__ void md_down_activation(
    const float* __restrict__ gate_row,     // [512]
    const float* __restrict__ up_row,       // [512]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float xs[4][4]
) {
    const int lane = threadIdx.x & 31;
    const int d0 = lane * 4;
    const float4 s1v = *reinterpret_cast<const float4*>(signs1 + d0);
    const float4 s2v = *reinterpret_cast<const float4*>(signs2 + d0);
    const float sg1[4] = {s1v.x, s1v.y, s1v.z, s1v.w};
    const float sg2[4] = {s2v.x, s2v.y, s2v.z, s2v.w};
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        const int base = g * 128 + d0;
        const float4 gv = *reinterpret_cast<const float4*>(gate_row + base);
        const float4 uv = *reinterpret_cast<const float4*>(up_row + base);
        const float gg[4] = {gv.x, gv.y, gv.z, gv.w};
        const float uu[4] = {uv.x, uv.y, uv.z, uv.w};
        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const float v = gg[i];
            a[i] = __fmul_rn(v / (1.0f + expf(-v)), uu[i]);
        }
        const float p1 = __fmul_rn(a[1], sg1[1]);
        float v0 = __fmaf_rn(a[0], sg1[0], p1);
        float v1 = __fmaf_rn(a[0], sg1[0], -p1);
        const float p3 = __fmul_rn(a[3], sg1[3]);
        float v2 = __fmaf_rn(a[2], sg1[2], p3);
        float v3 = __fmaf_rn(a[2], sg1[2], -p3);
        float t;
        t = v0; v0 = v0 + v2; v2 = t - v2;
        t = v1; v1 = v1 + v3; v3 = t - v3;
#define MD_HBFLY(v, pat, str) do { \
        float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
        if (lane & (str)) (v) = _p - (v); else (v) = (v) + _p; \
    } while (0)
#define MD_HBFLY4(pat, str) \
    MD_HBFLY(v0, pat, str); MD_HBFLY(v1, pat, str); MD_HBFLY(v2, pat, str); MD_HBFLY(v3, pat, str)
        MD_HBFLY4(0x041F, 1);
        MD_HBFLY4(0x081F, 2);
        MD_HBFLY4(0x101F, 4);
        MD_HBFLY4(0x201F, 8);
        MD_HBFLY4(0x401F, 16);
#undef MD_HBFLY4
#undef MD_HBFLY
        const float scale = 0.0883883476f;
        xs[g][0] = __fmul_rn(__fmul_rn(v0, scale), sg2[0]);
        xs[g][1] = __fmul_rn(__fmul_rn(v1, scale), sg2[1]);
        xs[g][2] = __fmul_rn(__fmul_rn(v2, scale), sg2[2]);
        xs[g][3] = __fmul_rn(__fmul_rn(v3, scale), sg2[3]);
    }
}

// 8 values per lane (one per row of an 8-row group) -> the shuffle-down sum
// tree (16, 8, 4, 2, 1) of every row at once. Value k of the group ends in
// every lane whose bits (4, 3, 2) spell k as b4 | b3 << 1 | b2 << 2 (all four
// lanes of a (b4 b3 b2) cell hold it); the per-row additions pair the same
// operands as the singleton's tree, only commuted.
static __device__ __forceinline__ float md_row8_tree(const float u[8]) {
    const int lane = threadIdx.x & 31;
    const bool h16 = (lane & 16) != 0, h8 = (lane & 8) != 0, h4 = (lane & 4) != 0;
    float w[4];
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        const float keep = h16 ? u[2 * m + 1] : u[2 * m];
        const float send = h16 ? u[2 * m] : u[2 * m + 1];
        w[m] = keep + __shfl_xor(send, 16);
    }
    float z[2];
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        const float keep = h8 ? w[2 * m + 1] : w[2 * m];
        const float send = h8 ? w[2 * m] : w[2 * m + 1];
        z[m] = keep + __shfl_xor(send, 8);
    }
    float y;
    {
        const float keep = h4 ? z[1] : z[0];
        const float send = h4 ? z[0] : z[1];
        y = keep + __shfl_xor(send, 4);
    }
    y = y + __shfl_xor(y, 2);
    y = y + __shfl_xor(y, 1);
    return y;
}

#ifndef MD_DOWN_ROWS
#define MD_DOWN_ROWS 16   // output rows per workgroup (groups of 8)
#endif

template <int C>
static __device__ __forceinline__ void md_down_cols(
    const char* __restrict__ A,             // expert down weight (uniform)
    const int* pairs,                       // C pair indices (uniform)
    const float* __restrict__ gate,         // [pair][512]
    const float* __restrict__ up,           // [pair][512]
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    float* __restrict__ expanded,           // [pair][2048]
    int r0
) {
    const int lane = threadIdx.x & 31;
    float xs[C][4][4];
    #pragma unroll
    for (int c = 0; c < C; c++)
        md_down_activation(gate + (size_t)pairs[c] * 512, up + (size_t)pairs[c] * 512,
                           signs1, signs2, xs[c]);

    #pragma unroll 1
    for (int rb = 0; rb < MD_DOWN_ROWS / 8; rb++) {
        float u[C][8];
        #pragma unroll
        for (int rr = 0; rr < 8; rr++) {
            const char* row_ptr = A + (long long)(r0 + rb * 8 + rr) * (4 * 68);
            float acc[C];
            #pragma unroll
            for (int c = 0; c < C; c++) acc[c] = 0.0f;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const char* gp = row_ptr + g * 68;
                const unsigned int hdr = *reinterpret_cast<const unsigned int*>(gp);
                const float scale = __half2float(__ushort_as_half((unsigned short)(hdr & 0xFFFFu)));
                const float zero = __half2float(__ushort_as_half((unsigned short)(hdr >> 16)));
                const unsigned int pk = *reinterpret_cast<const unsigned short*>(gp + 4 + lane * 2);
                const float d0f = __fmaf_rn(scale, (float)(pk & 0xFu), zero);
                const float d1f = __fmaf_rn(scale, (float)((pk >> 4) & 0xFu), zero);
                const float d2f = __fmaf_rn(scale, (float)((pk >> 8) & 0xFu), zero);
                const float d3f = __fmaf_rn(scale, (float)((pk >> 12) & 0xFu), zero);
                #pragma unroll
                for (int c = 0; c < C; c++) {
                    float s = __fmul_rn(d1f, xs[c][g][1]);
                    s = __fmaf_rn(d0f, xs[c][g][0], s);
                    s = __fmaf_rn(d2f, xs[c][g][2], s);
                    s = __fmaf_rn(d3f, xs[c][g][3], s);
                    acc[c] += s;
                }
            }
            #pragma unroll
            for (int c = 0; c < C; c++) u[c][rr] = acc[c];
        }
        #pragma unroll
        for (int c = 0; c < C; c++) {
            const float y = md_row8_tree(u[c]);
            if ((lane & 3) == 0) {
                const int k = ((lane >> 4) & 1) | (((lane >> 3) & 1) << 1) | (((lane >> 2) & 1) << 2);
                expanded[(size_t)pairs[c] * 2048 + r0 + rb * 8 + k] = y;
            }
        }
    }
}

// Grid (2048 / 32 row tiles, 8 slots, lanes) x 32 threads.
extern "C" __launch_bounds__(32, MD_MINB)
__global__ void gemv_mq4g128v2_moe_down_dedup_rows(
    const unsigned long long* __restrict__ expert_ptrs, // [n_exp], shared
    const int* __restrict__ topk_indices,               // [lanes, 8]
    const float* __restrict__ gate,                     // [lanes, 8, 512]
    const float* __restrict__ up,                       // [lanes, 8, 512]
    const float* __restrict__ signs1,                   // [128], shared
    const float* __restrict__ signs2,                   // [128], shared
    float* __restrict__ expanded,                       // [lanes, 8, 2048]
    int lanes, int n_experts
) {
    const int r0 = blockIdx.x * MD_DOWN_ROWS;
    const int pair = blockIdx.z * 8 + blockIdx.y;
    const int npairs = lanes * 8;
    const int e = __builtin_amdgcn_readfirstlane(topk_indices[pair]);
    unsigned long long members = md_member_mask(topk_indices, npairs, e);
    if (md_ctz64(members) != pair) return;           // follower

    const int lane = threadIdx.x & 31;
    if ((unsigned)e >= (unsigned)n_experts) {
        // The singleton parks acc = 0 for every row of an invalid expert.
        while (members) {
            const int p = md_ctz64(members);
            members &= members - 1;
            for (int i = lane; i < MD_DOWN_ROWS; i += 32)
                expanded[(size_t)p * 2048 + r0 + i] = 0.0f;
        }
        return;
    }
    const char* __restrict__ A = md_uniform_ptr(reinterpret_cast<const char*>(expert_ptrs[e]));

    while (members) {
        int pc[MD_CHUNK];
        int c = 0;
        while (c < MD_CHUNK && members) {
            pc[c++] = md_ctz64(members);
            members &= members - 1;
        }
        switch (c) {
            case 1: md_down_cols<1>(A, pc, gate, up, signs1, signs2, expanded, r0); break;
            case 2: md_down_cols<2>(A, pc, gate, up, signs1, signs2, expanded, r0); break;
#if MD_CHUNK >= 3
            case 3: md_down_cols<3>(A, pc, gate, up, signs1, signs2, expanded, r0); break;
#endif
#if MD_CHUNK >= 4
            default: md_down_cols<4>(A, pc, gate, up, signs1, signs2, expanded, r0); break;
#else
            default: break;
#endif
        }
    }
}

// ---------------------------------------------------------------------------
// Routed down: stage B (the singleton's ascending-slot weighted fold)
// ---------------------------------------------------------------------------

// Grid (2048 / 256, lanes) x 256 threads: one row per thread, single owner.
extern "C" __launch_bounds__(256, 4)
__global__ void gemv_mq4g128v2_moe_down_combine_rows(
    const float* __restrict__ topk_weights,             // [lanes, 8]
    const float* __restrict__ expanded,                 // [lanes, 8, 2048]
    float* __restrict__ out                             // [lanes, 2048] residual stream
) {
    const size_t lane = blockIdx.y;
    const int row = blockIdx.x * 256 + threadIdx.x;
    const float* __restrict__ w = topk_weights + lane * 8;
    const float* __restrict__ part = expanded + lane * 8 * 2048 + row;
    float a = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; k++)
        a = __fmaf_rn(w[k], part[(size_t)k * 2048], a);
    out[lane * 2048 + row] += a;
}
