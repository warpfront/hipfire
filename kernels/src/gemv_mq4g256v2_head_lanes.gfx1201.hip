// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Lane-batched twin of `gemv_mq4g256v2_multirow_r2` for the lm_head of the
// A3B (qwen3_5_moe, hidden 2048) exact VMM batch route, gfx1201 only.
//
// `gemv_mq4g256v2_head_lanes_n<N>` (N = 1..8) computes N independent GEMVs
// `y[slot_c] = W . x[c]` over ONE shared weight matrix `A [M, 2048]`
// (MQ4-G256-v2, 136 B per 256-group, 8 groups = 1088 B per row). Each weight
// row is read ONCE for all N columns. Per column the arithmetic is the
// singleton R2 kernel's, operation for operation:
//
//   * lane l of the wave owns elements l*8..l*8+7 of every 256-group, with the
//     header dword (scale/zero halves) selected by `l < 16 ? dword0 : dword1`;
//   * w_i = scale * (float)nibble_i + zero  (one FMA, shared by all columns);
//   * group partial E = w0*x0 + w1*x1 + ... + w7*x7 in source order, with the
//     singleton's contraction, which differs per row of its row pair (even row:
//     mul(w0,x0) then fma(w1,x1); odd row: mul(w1,x1) then fma(w0,x0); then one
//     fma per remaining term; pinned on hardware by tests/head_lanes.rs and
//     spelled with explicit fmaf here so the column loop cannot change it);
//   * accumulator a_j (j = g % 4) += E(g) for g = j, j+4 (K = 2048: 8 groups),
//     starting from +0.0f;
//   * lane partial s = (a0 + a1) + (a2 + a3);
//   * cross-lane sum: the 16, 8, 4, 2, 1 shuffle-down tree. It is evaluated
//     for 32 values at once as a transposed butterfly (one xor-shuffle serves
//     two values per step): every value's additions pair the same operands in
//     the same tree, only commutatively swapped, so the sum is bit-identical
//     to the shuffle-down chain of lane 0.
//
// The N activation columns are staged once per workgroup in LDS (exact f32,
// no rounding) and shared by the workgroup's HL_WAVES waves; each wave streams
// HL_R rows per tile. A tile's R*N <= 32 results land in the lanes of one
// register after the butterfly; lane l stores value k = rev5(l) =
// (r, c) = (k / N, k % N) to y + slot[c] * M + row. `slot[c]` is the output
// row of column c (slots need not be consecutive).
//
// Source prefix (kernels.rs): `HIPFIRE_GFX12_WEIGHT_CACHE_ELIGIBLE` define +
// `gfx12_weight_cache_policy.inc`, the singleton module's prefix.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifndef HIPFIRE_WEIGHT_LOAD_PTR_U32
#define HIPFIRE_GFX12_WEIGHT_BUFFER_LOADS 0
#define HIPFIRE_WEIGHT_RSRC(base) 0
#define HIPFIRE_WEIGHT_LOAD_PTR_U32(rsrc, ptr, byte_offset) \
    (*(const unsigned int*)(ptr))
#endif

#ifndef HL_R
#define HL_R 4        // rows per wave tile
#endif
#ifndef HL_WAVES
#define HL_WAVES 16   // waves per workgroup
#endif
#ifndef HL_TILES
#define HL_TILES 2    // tiles per wave per workgroup
#endif
#ifndef HL_CC
#define HL_CC 4       // columns per step-major block
#endif
#define HL_G 8        // 256-groups per row (K = 2048)
#define HL_ROW_BYTES (HL_G * 136)

#define HL_BYTE0(v) ((float)((v) & 0xFFu))
#define HL_BYTE1(v) ((float)(((v) >> 8) & 0xFFu))
#define HL_BYTE2(v) ((float)(((v) >> 16) & 0xFFu))
#define HL_BYTE3(v) ((float)((v) >> 24))

struct HeadSlots { int v[8]; };

__device__ __forceinline__ int head_slot_of(const HeadSlots& s, int c) {
    return c == 0 ? s.v[0] : c == 1 ? s.v[1] : c == 2 ? s.v[2] : c == 3 ? s.v[3]
         : c == 4 ? s.v[4] : c == 5 ? s.v[5] : c == 6 ? s.v[6] : s.v[7];
}

template <int N>
__device__ __forceinline__ void head_lanes_body(
    const char* __restrict__ A,
    const float4* __restrict__ x4,   // [N][2048] f32 as float4
    float* __restrict__ y,
    HeadSlots slots,
    int M
) {
    __shared__ float4 xs[N * HL_G * 64];
    const int tid = threadIdx.x;
    // Stage: global float4 index f over [c][g][q], q = lane*2 + half; LDS slot
    // [c][g][half][lane] so each lane's two ds_read_b128 are lane-contiguous.
    #pragma unroll
    for (int f = tid; f < N * HL_G * 64; f += HL_WAVES * 32) {
        const int cg = f >> 6;
        const int q = f & 63;
        xs[(cg << 6) + ((q & 1) << 5) + (q >> 1)] = x4[f];
    }
    __syncthreads();

    const int wave = tid >> 5;
    const int lane = tid & 31;
    const auto weight_rsrc = HIPFIRE_WEIGHT_RSRC(A);
    const unsigned int hoff = (unsigned int)((lane >> 4) << 2);
    const unsigned int poff = 8u + (unsigned int)lane * 4u;

    for (int t = 0; t < HL_TILES; ++t) {
        const int row0 = ((blockIdx.x * HL_TILES + t) * HL_WAVES + wave) * HL_R;
        if (row0 >= M) break;
        // All of the tile's weights are requested up front: one packed dword and
        // one header dword per (row, group) per lane.
        unsigned int pkw[HL_R][HL_G];
        unsigned int hsw[HL_R][HL_G];
        #pragma unroll
        for (int r = 0; r < HL_R; ++r) {
            const int rr = row0 + r < M ? row0 + r : M - 1;
            const unsigned int wo = (unsigned int)((long long)rr * HL_ROW_BYTES);
            #pragma unroll
            for (int g = 0; g < HL_G; ++g) {
                const unsigned int gb = wo + (unsigned int)(g * 136);
                hsw[r][g] = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, A + gb + hoff, gb + hoff);
                pkw[r][g] = HIPFIRE_WEIGHT_LOAD_PTR_U32(weight_rsrc, A + gb + poff, gb + poff);
            }
        }
        asm volatile("" ::: "memory");

        float t01[HL_R][N];
        float s[HL_R][N];
        float keep[HL_R][N];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a[HL_R][N];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int g = j + 4 * h;
                float w[HL_R][8];
                #pragma unroll
                for (int r = 0; r < HL_R; ++r) {
                    const unsigned int hs = hsw[r][g];
                    const unsigned int pk = pkw[r][g];
                    const float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
                    const float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
                    // nibble i as an exact float: two byte-spread masks, then the
                    // byte converts.
                    const unsigned int lo = pk & 0x0F0F0F0Fu;
                    const unsigned int hi = (pk >> 4) & 0x0F0F0F0Fu;
                    w[r][0] = sc * HL_BYTE0(lo) + zp;
                    w[r][1] = sc * HL_BYTE0(hi) + zp;
                    w[r][2] = sc * HL_BYTE1(lo) + zp;
                    w[r][3] = sc * HL_BYTE1(hi) + zp;
                    w[r][4] = sc * HL_BYTE2(lo) + zp;
                    w[r][5] = sc * HL_BYTE2(hi) + zp;
                    w[r][6] = sc * HL_BYTE3(lo) + zp;
                    w[r][7] = sc * HL_BYTE3(hi) + zp;
                }
                // Columns two at a time: 2*HL_R independent chains per step, issued
                // step-major with consecutive chains differing in BOTH row and column
                // (so a dual-issue pair shares no source register).
                #pragma unroll
                for (int c0 = 0; c0 < N; c0 += HL_CC) {
                    constexpr int CCMAX = HL_CC;
                    const int ncc = (N - c0) >= HL_CC ? HL_CC : (N - c0);  // compile-time per unrolled c0
                    float x[CCMAX][8];
                    #pragma unroll
                    for (int cc = 0; cc < CCMAX; ++cc) {
                        if (cc < ncc) {
                            const float4 xa = xs[(((c0 + cc) * HL_G + g) << 6) + lane];
                            const float4 xb = xs[(((c0 + cc) * HL_G + g) << 6) + 32 + lane];
                            x[cc][0] = xa.x; x[cc][1] = xa.y; x[cc][2] = xa.z; x[cc][3] = xa.w;
                            x[cc][4] = xb.x; x[cc][5] = xb.y; x[cc][6] = xb.z; x[cc][7] = xb.w;
                        }
                    }
                    float e[HL_R][CCMAX];
                    // chain order k -> (r, cc): rows cycle fastest, columns flip each row pair
                    #pragma unroll
                    for (int k = 0; k < HL_R * CCMAX; ++k) {
                        const int r = k % HL_R;
                        const int cc = ((k / HL_R) + r) % CCMAX;
                        if (cc < ncc) {
                            // The singleton R2 kernel's contraction of w0*x0 + w1*x1 + ...
                            // + w7*x7 depends on the row's slot in the pair (hipcc, read
                            // off its ISA and pinned on hardware): the even row (`a`
                            // accumulators) multiplies the FIRST term then fma's the second
                            // onto it; the odd row (`b`) multiplies the SECOND term and
                            // fma's the first onto it. Terms 2..7 are one fma each, in
                            // order. HL_R is even, so r's parity is the row's parity.
                            if ((r & 1) == 0) {
                                e[r][cc] = w[r][0] * x[cc][0];
                                e[r][cc] = __builtin_fmaf(w[r][1], x[cc][1], e[r][cc]);
                            } else {
                                e[r][cc] = w[r][1] * x[cc][1];
                                e[r][cc] = __builtin_fmaf(w[r][0], x[cc][0], e[r][cc]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int i = 2; i < 8; ++i) {
                        #pragma unroll
                        for (int k = 0; k < HL_R * CCMAX; ++k) {
                            const int r = k % HL_R;
                            const int cc = ((k / HL_R) + r) % CCMAX;
                            if (cc < ncc)
                                e[r][cc] = __builtin_fmaf(w[r][i], x[cc][i], e[r][cc]);
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < HL_R; ++r)
                        #pragma unroll
                        for (int cc = 0; cc < CCMAX; ++cc)
                            if (cc < ncc) {
                                // a_j starts at +0.0f: the first group is 0 + E.
                                if (h == 0) a[r][c0 + cc] = 0.0f + e[r][cc];
                                else a[r][c0 + cc] += e[r][cc];
                            }
                }
            }
            #pragma unroll
            for (int r = 0; r < HL_R; ++r)
                #pragma unroll
                for (int c = 0; c < N; ++c) {
                    if (j == 0) keep[r][c] = a[r][c];
                    else if (j == 1) t01[r][c] = keep[r][c] + a[r][c];
                    else if (j == 2) keep[r][c] = a[r][c];
                    else s[r][c] = t01[r][c] + (keep[r][c] + a[r][c]);
                }
        }

        // Transposed butterfly over 32 (padded) values.
        constexpr int V = HL_R * N;
        constexpr int VP = ((V + 31) / 32) * 32;
        float v[VP];
        #pragma unroll
        for (int k = 0; k < VP; ++k) v[k] = (k < V) ? s[k / N][k % N] : 0.0f;
        // VP / 32 independent groups of 32 values.
        #pragma unroll
        for (int grp = 0; grp < VP / 32; ++grp) {
            float* u = v + grp * 32;
            #pragma unroll
            for (int stage = 0; stage < 5; ++stage) {
                const int off = 16 >> stage;
                const int cnt = 16 >> stage;
                const bool hi = (lane & off) != 0;
                #pragma unroll
                for (int m = 0; m < cnt; ++m) {
                    const float keepv = hi ? u[2 * m + 1] : u[2 * m];
                    const float send = hi ? u[2 * m] : u[2 * m + 1];
                    u[m] = keepv + __shfl_xor(send, off);
                }
            }
        }
        // Lane l of group grp holds value k = grp*32 + rev5(l).
        const int rev = ((lane & 1) << 4) | ((lane & 2) << 2) | (lane & 4) | ((lane & 8) >> 2) | ((lane & 16) >> 4);
        #pragma unroll
        for (int grp = 0; grp < VP / 32; ++grp) {
            const int k = grp * 32 + rev;
            if (k < V) {
                const int r = k / N;
                const int c = k % N;
                const int row = row0 + r;
                if (row < M)
                    y[(long long)head_slot_of(slots, c) * (long long)M + row] = v[grp * 32];
            }
        }
    }
}

#define HL_ENTRY(NN) \
    extern "C" __global__ __launch_bounds__(HL_WAVES * 32) \
    void gemv_mq4g256v2_head_lanes_n##NN( \
        const char* __restrict__ A, const float* __restrict__ x, float* __restrict__ y, \
        int M, int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) { \
        HeadSlots sl; sl.v[0] = s0; sl.v[1] = s1; sl.v[2] = s2; sl.v[3] = s3; \
        sl.v[4] = s4; sl.v[5] = s5; sl.v[6] = s6; sl.v[7] = s7; \
        head_lanes_body<NN>(A, reinterpret_cast<const float4*>(x), y, sl, M); \
    }
HL_ENTRY(1)
HL_ENTRY(2)
HL_ENTRY(3)
HL_ENTRY(4)
HL_ENTRY(5)
HL_ENTRY(6)
HL_ENTRY(7)
HL_ENTRY(8)
