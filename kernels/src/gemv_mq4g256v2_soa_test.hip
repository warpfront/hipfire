// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

// Lab-only lossless global-plane MQ4G256V2 layout experiment.
// Production kernels and dispatch are unchanged.
//
// GLOBAL SoA layout (parent contract; not row-local):
//   gpr = K/256
//   N   = (size_t)M * gpr
//   n   = (size_t)row * gpr + g
//   payload = A + n * 128
//   header  = A + N * 128 + n * 8
//   total   = 136 * N bytes
// AoS group remains A + n * 136 (header @+0, payload @+8).
//
// FP tree (DOG / acc[g%4] / combine / shuffle) is byte-identical to
// kernels/src/gemv_mq4g256v2.hip:46-216; only weight addressing changes.
// RO symbols are counterfactual read probes (uint32 bits through y).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ─── plain GEMV over GLOBAL SoA weights ───────────────────────────────────

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_soa(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const size_t N = (size_t)M * (size_t)groups_per_row;
    const size_t row_base_n = (size_t)row * (size_t)groups_per_row;
    const char* payload_plane = A;
    const char* header_plane = A + N * 128;
    const int boff = tid * 4;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = groups_per_row >> 2;
    const int tail = groups_per_row & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const size_t n0 = row_base_n + (size_t)g;
        const size_t n1 = n0 + 1;
        const size_t n2 = n0 + 2;
        const size_t n3 = n0 + 3;

        const char* hp0 = header_plane + n0 * 8;
        const char* hp1 = header_plane + n1 * 8;
        const char* hp2 = header_plane + n2 * 8;
        const char* hp3 = header_plane + n3 * 8;
        const char* pp0 = payload_plane + n0 * 128;
        const char* pp1 = payload_plane + n1 * 128;
        const char* pp2 = payload_plane + n2 * 128;
        const char* pp3 = payload_plane + n3 * 128;

        const unsigned int hA0 = *(const unsigned int*)(hp0);
        const unsigned int hB0 = *(const unsigned int*)(hp0 + 4);
        const unsigned int hs0 = (tid < 16) ? hA0 : hB0;
        float sc0 = __half2float(__ushort_as_half((unsigned short)(hs0 & 0xFFFFu)));
        float zp0 = __half2float(__ushort_as_half((unsigned short)(hs0 >> 16)));
        const unsigned int hA1 = *(const unsigned int*)(hp1);
        const unsigned int hB1 = *(const unsigned int*)(hp1 + 4);
        const unsigned int hs1 = (tid < 16) ? hA1 : hB1;
        float sc1 = __half2float(__ushort_as_half((unsigned short)(hs1 & 0xFFFFu)));
        float zp1 = __half2float(__ushort_as_half((unsigned short)(hs1 >> 16)));
        const unsigned int hA2 = *(const unsigned int*)(hp2);
        const unsigned int hB2 = *(const unsigned int*)(hp2 + 4);
        const unsigned int hs2 = (tid < 16) ? hA2 : hB2;
        float sc2 = __half2float(__ushort_as_half((unsigned short)(hs2 & 0xFFFFu)));
        float zp2 = __half2float(__ushort_as_half((unsigned short)(hs2 >> 16)));
        const unsigned int hA3 = *(const unsigned int*)(hp3);
        const unsigned int hB3 = *(const unsigned int*)(hp3 + 4);
        const unsigned int hs3 = (tid < 16) ? hA3 : hB3;
        float sc3 = __half2float(__ushort_as_half((unsigned short)(hs3 & 0xFFFFu)));
        float zp3 = __half2float(__ushort_as_half((unsigned short)(hs3 >> 16)));

        unsigned int pk0 = *(const unsigned int*)(pp0 + boff);
        unsigned int pk1 = *(const unsigned int*)(pp1 + boff);
        unsigned int pk2 = *(const unsigned int*)(pp2 + boff);
        unsigned int pk3 = *(const unsigned int*)(pp3 + boff);

        int base = g * 256 + tid * 8;

        #define DOG(pk, sc, zp, b, a) \
            (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
                 + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
                 + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
                 + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
                 + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
                 + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
                 + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
                 + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

        DOG(pk0, sc0, zp0, base, acc0);
        DOG(pk1, sc1, zp1, base + 256, acc1);
        DOG(pk2, sc2, zp2, base + 512, acc2);
        DOG(pk3, sc3, zp3, base + 768, acc3);
        #undef DOG
    }

    // Tail handling: groups_per_row may not be a multiple of 4. Each tail
    // group must accumulate into its OWN accumulator (acc[g % 4]) to
    // preserve the 4-way interleaving that the combine step assumes.
    // See gemv_mq4g256v2.gfx1100.hip for the full bug writeup.
    #define TAIL_DOG(pk, sc, zp, b, a) \
        (a) += (sc * (float)((pk) & 0xFu)        + zp) * x[(b)]     \
             + (sc * (float)(((pk) >> 4) & 0xFu)  + zp) * x[(b) + 1] \
             + (sc * (float)(((pk) >> 8) & 0xFu)  + zp) * x[(b) + 2] \
             + (sc * (float)(((pk) >> 12) & 0xFu) + zp) * x[(b) + 3] \
             + (sc * (float)(((pk) >> 16) & 0xFu) + zp) * x[(b) + 4] \
             + (sc * (float)(((pk) >> 20) & 0xFu) + zp) * x[(b) + 5] \
             + (sc * (float)(((pk) >> 24) & 0xFu) + zp) * x[(b) + 6] \
             + (sc * (float)(((pk) >> 28) & 0xFu) + zp) * x[(b) + 7]

    if (tail >= 1) {
        const int g = quads << 2;
        const size_t n = row_base_n + (size_t)g;
        const char* hp = header_plane + n * 8;
        const char* pp = payload_plane + n * 128;
        const unsigned int hA = *(const unsigned int*)(hp);
        const unsigned int hB = *(const unsigned int*)(hp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(pp + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        const size_t n = row_base_n + (size_t)g;
        const char* hp = header_plane + n * 8;
        const char* pp = payload_plane + n * 128;
        const unsigned int hA = *(const unsigned int*)(hp);
        const unsigned int hB = *(const unsigned int*)(hp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(pp + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        const size_t n = row_base_n + (size_t)g;
        const char* hp = header_plane + n * 8;
        const char* pp = payload_plane + n * 128;
        const unsigned int hA = *(const unsigned int*)(hp);
        const unsigned int hB = *(const unsigned int*)(hp + 4);
        const unsigned int hs = (tid < 16) ? hA : hB;
        float sc = __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
        float zp = __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
        unsigned int pk = *(const unsigned int*)(pp + boff);
        int base = g * 256 + tid * 8;
        TAIL_DOG(pk, sc, zp, base, acc2);
    }
    #undef TAIL_DOG

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (tid == 0) y[row] = acc;
}

// ─── RO counterfactual read probes ────────────────────────────────────────
// Per-lane: acc += pk (wrapping u32) over all groups in the row.
// After groups: XOR-shuffle reduce lane accs with offsets 16,8,4,2,1.
// Lane 0 then adds header_acc = sum_g (hA + rotl32(hB, 16)) so uniform
// headers cannot cancel across the 32 lanes. Stores bits through y; x unused.

static __device__ __forceinline__ unsigned int mq4g256v2_rotl32(unsigned int x, unsigned int n) {
    return (x << n) | (x >> (32u - n));
}

template <bool SoA>
static __device__ __forceinline__ void mq4g256v2_ro_body(
    const char* __restrict__ A,
    float* __restrict__ y,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;

    const int groups_per_row = K / 256;
    const size_t N = (size_t)M * (size_t)groups_per_row;
    const size_t row_base_n = (size_t)row * (size_t)groups_per_row;
    const int boff = tid * 4;

    unsigned int acc = 0u;
    unsigned int header_acc = 0u;

    for (int g = 0; g < groups_per_row; g++) {
        const size_t n = row_base_n + (size_t)g;
        const char* hp;
        const char* pp;
        if (SoA) {
            hp = A + N * 128 + n * 8;
            pp = A + n * 128;
        } else {
            const char* gp = A + n * 136;
            hp = gp;
            pp = gp + 8;
        }
        const unsigned int hA = *(const unsigned int*)(hp);
        const unsigned int hB = *(const unsigned int*)(hp + 4);
        const unsigned int pk = *(const unsigned int*)(pp + boff);
        acc += pk;
        // Every lane loads headers (scalarizable); accumulate on all lanes
        // then only lane0 folds header_acc after the XOR reduce — keeps the
        // sum identical whether the compiler scalarizes or not.
        header_acc += hA + mq4g256v2_rotl32(hB, 16u);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc ^= __shfl_xor(acc, offset);

    if (tid == 0) {
        acc += header_acc;
        y[row] = __uint_as_float(acc);
    }
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_ro(
    const char* __restrict__ A,
    const float* __restrict__ /*x*/,
    float* __restrict__ y,
    int M, int K
) {
    mq4g256v2_ro_body</*SoA=*/false>(A, y, M, K);
}

__launch_bounds__(32, 16)
extern "C" __global__ void gemv_mq4g256v2_soa_ro(
    const char* __restrict__ A,
    const float* __restrict__ /*x*/,
    float* __restrict__ y,
    int M, int K
) {
    mq4g256v2_ro_body</*SoA=*/true>(A, y, M, K);
}
