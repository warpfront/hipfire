// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.
//
// GEMM v2 "V2B" symmetric MQ4V2 x int4-activation prefill GEMM for gfx11
// (wave32 iu4 WMMA); dispatched on gfx1151. Full tiles only: M%256 == N%256
// == 0, K%256 == 0.
//
// Tile: M256 x N256 per CTA, 16 waves (block 32x16), wave tile 64 rows x 64
// tokens (4 row fragments x 4 token fragments), 128 FP32 sums per lane.
// K is walked in K128 epochs; two 32 KiB LDS slots (65,536 B) double buffer
// the operands so one CTA barrier per epoch retires slot e%2 while epoch e+1
// is staged into slot (e+1)%2. One CTA per WGP.
//
// Operands are read from the unchanged production layouts:
//   A  : MQ4G256V2 rows, (M, K/256) groups of 136 B = [sc0 zp0 sc1 zp1] FP16
//        + 128 B nibbles (K half h&1 at +8+64*(h&1), even K in the low nibble).
//   Xq : block_i4_128[h*N + t] (72 B: f32 d, i32 s, 64 B signed nibbles).
// The staging threads apply the fragment-major permutation at the LDS store:
// K16 slice s of row/token r lands at ((r/16)*8 + s)*128 + (r%16)*8, so each
// WMMA operand fragment is one contiguous 128-byte LDS run. A is rebiased
// once while staging (XOR 0x88888888, the symmetric zp == -8*sc contract),
// so both operands feed WMMA as signed nibbles. Scales bypass LDS: lane L of
// each wave loads the 8-byte group headers of two of its wave's 64 rows (one
// for row fragments 0-1, one for 2-3) once per group, keeps sc0|sc1 packed in
// one word each, and the fold gathers the eight rows a lane owns with DPP
// row_share; the X scales d are read from Xq by the consuming lanes.
//
// Numerics (bit-identical to the symfold IU4 kernels' compiled fold DAG):
// per output and ascending K128 half h,
//   C_h = exact int32 WMMA sum over the eight K16 steps
//   sum = fma(RN(float(sc_h) * d_h), float(C_h), sum)   (sum starts +0)
// SET stores sum; ADD stores RN(oldY + sum). Integer chains start at
// 1.5*2^23 (0x4B400000) so float(C_h) is one exact v_sub (|C_h| <= 8192).
//
// A-operand lane permutation: A lane i supplies row 8*(i&1) + (i>>1), so
// result VGPR j of lane (hi, lr) is output (row 8*hi + j, token lr): each lane
// owns 8 contiguous rows (two b128 Y accesses per fragment). Y is token-major
// [N][M].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <type_traits>

#if defined(__gfx1100__) || defined(__gfx1151__)

namespace iu4_v2b {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

constexpr int TM = 256;
constexpr int TN = 256;
constexpr int NWAVES = 16;
constexpr int NT = 32 * NWAVES;
constexpr int NC = 4;                       // 16-token fragments per wave
constexpr int WAVES_N = TN / 64;
constexpr int WAVE_ROWS = TM / (NWAVES / WAVES_N);
constexpr int NA = WAVE_ROWS / 16;          // 16-row fragments per wave
constexpr int NWS = WAVE_ROWS / 32;         // W scale words per lane
constexpr int A_BYTES = TM * 64;
constexpr int X_BYTES = TN * 64;
constexpr int SLOT_BYTES = A_BYTES + X_BYTES;
constexpr int LDS_BYTES = 2 * SLOT_BYTES;
constexpr int GROUP_BYTES = 136;            // MQ4G256V2 row group
constexpr int XBLK_BYTES = 72;              // block_i4_128
constexpr int SA = TM * 8 / NT;             // A b64 staging words per thread
constexpr int SX = TN * 8 / NT;             // X b64 staging words per thread
static_assert(SA * NT == TM * 8 && SX * NT == TN * 8, "staging split");
static_assert(TM == 16 * NWAVES && TN == 16 * NWAVES && SA == 4 && SX == 4,
              "staging wave w owns rows/tokens 16w..16w+15");
static_assert(LDS_BYTES == 65536, "V2B LDS: one CTA per WGP");
static_assert(WAVE_ROWS == 64 && NA == 4 && NWS == 2, "two W scales per lane");

#define IU4_V2B_FENCE() __builtin_amdgcn_sched_barrier(0)

template <typename T>
static __device__ __forceinline__ const T* at(const void* base, uint32_t off) {
    return (const T*)((const char*)base + off);
}

// DPP row_share:k (gfx10+): every lane reads lane k of its own row of 16.
// k is a compile-time constant after unrolling; the switch folds away.
static __device__ __forceinline__ int row_share(int v, int k) {
#define IU4_V2B_SHARE(k) \
    case k: return __builtin_amdgcn_update_dpp(0, v, 0x150 + k, 0xf, 0xf, false);
    switch (k) {
        IU4_V2B_SHARE(0) IU4_V2B_SHARE(1) IU4_V2B_SHARE(2) IU4_V2B_SHARE(3)
        IU4_V2B_SHARE(4) IU4_V2B_SHARE(5) IU4_V2B_SHARE(6) IU4_V2B_SHARE(7)
        IU4_V2B_SHARE(8) IU4_V2B_SHARE(9) IU4_V2B_SHARE(10) IU4_V2B_SHARE(11)
        IU4_V2B_SHARE(12) IU4_V2B_SHARE(13) IU4_V2B_SHARE(14) IU4_V2B_SHARE(15)
        default: __builtin_unreachable();
    }
#undef IU4_V2B_SHARE
}

struct Stage {
    v2i a[SA], x[SX];
};

// Staging lane (hi, rr) of wave w owns row/token r = 16w + rr and K16 slices
// s = 2i + hi, i = 0..3: its row's 16-byte slice pair i per instruction.
// Every store instruction then fills one contiguous 256-byte LDS block
// (fragment block w, slices 2i and 2i+1), so the b64 stores are bank-conflict
// free; each wave reads its 16 rows' 64-byte K128 half over four loads.
static __device__ __forceinline__ void stage_load(
    Stage& st, const char* eA, const char* eX, uint32_t a_off, uint32_t x_off) {
#pragma unroll
    for (int i = 0; i < SA; ++i) st.a[i] = *at<v2i>(eA, a_off + 16u * i);
#pragma unroll
    for (int i = 0; i < SX; ++i) st.x[i] = *at<v2i>(eX, x_off + 16u * i);
}

static __device__ __forceinline__ void stage_store(const Stage& st, char* slot,
                                                   uint32_t lds_off) {
#pragma unroll
    for (int i = 0; i < SA; ++i) {
        const v2i v = st.a[i] ^ (v2i){(int)0x88888888u, (int)0x88888888u};
        *(v2i*)(slot + lds_off + i * 256) = v;
    }
#pragma unroll
    for (int i = 0; i < SX; ++i) *(v2i*)(slot + A_BYTES + lds_off + i * 256) = st.x[i];
}

template <bool IS_ADD>
static __device__ __forceinline__ void body(
    const char* __restrict__ A, const char* __restrict__ Xq,
    float* __restrict__ Y, int M, int K, int N) {
    extern __shared__ __attribute__((aligned(16))) char lds[];
    const int lane = threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.y);
    const int wr = wave / WAVES_N;
    const int wc = wave % WAVES_N;
    const int lr = lane & 15;
    const int hi = lane >> 4;
    const int prow = 8 * (lr & 1) + (lr >> 1);
    const int row0 = blockIdx.y * TM;
    const int col0 = blockIdx.x * TN;
    const int E = K >> 7;
    const uint32_t row_bytes = (uint32_t)(K >> 8) * GROUP_BYTES;
    const size_t x_epoch = (size_t)N * XBLK_BYTES;
    const char* gA = A + (size_t)row0 * row_bytes;
    const char* gX = Xq + (size_t)col0 * XBLK_BYTES;
    // Staging ownership (tile-relative 32-bit offsets).
    const int sr = 16 * wave + lr;
    const uint32_t a_off = (uint32_t)sr * row_bytes + 8u + 8u * (uint32_t)hi;
    const uint32_t x_off = (uint32_t)sr * XBLK_BYTES + 8u + 8u * (uint32_t)hi;
    const uint32_t lds_off = (uint32_t)(wave * 1024 + hi * 128 + lr * 8);
    // Consumer offsets.
    const uint32_t lxs = (uint32_t)(wc * 64 + lr) * XBLK_BYTES;
    const uint32_t aoff = (uint32_t)(wr * NA * 1024 + prow * 8);
    const uint32_t xoff = (uint32_t)(A_BYTES + wc * 4 * 1024 + lr * 8);
    // Scale word u of lane (hi, lr) is the scale of row 8*hi + (lr&7) of
    // fragment 2u + (lr>>3), so DPP row_share:(8(a&1)+j) on word a>>1 hands
    // every lane of half hi row 16a+8hi+j.
    uint32_t lws[NWS];
#pragma unroll
    for (int u = 0; u < NWS; ++u)
        lws[u] = (uint32_t)(wr * WAVE_ROWS + 32 * u + 16 * (lr >> 3) + 8 * hi + (lr & 7)) *
                 row_bytes;
    // Byte offset of K128 half e's payload inside a row (a_off adds the +8
    // header skip).
    auto a_epoch = [](int e) -> size_t {
        return (size_t)(e >> 1) * GROUP_BYTES + (size_t)(e & 1) * 64;
    };
    // FP16 sc0 (low half) and sc1 (high half) of the current group per scale
    // word; loaded at even epochs (one b64 header load serves two epochs).
    uint32_t wpk[NWS];
#pragma unroll
    for (int u = 0; u < NWS; ++u) wpk[u] = 0;

    float sum[NA][NC][8];
#pragma unroll
    for (int a = 0; a < NA; ++a)
#pragma unroll
        for (int c = 0; c < NC; ++c)
#pragma unroll
            for (int j = 0; j < 8; ++j) sum[a][c][j] = 0;

    Stage st;
    stage_load(st, gA + a_epoch(0), gX, a_off, x_off);
    stage_store(st, lds, lds_off);
    __syncthreads();

    auto epoch = [&](auto nxt_tag, int e) {
        constexpr bool nxt = decltype(nxt_tag)::value;
        char* slot = lds + (e & 1) * SLOT_BYTES;
        // Current-epoch metadata first (in-order vmcnt returns it before the
        // next-epoch staging packet).
        const bool even = !(e & 1);
        if (even) {
#pragma unroll
            for (int u = 0; u < NWS; ++u) {
                const v2i h = *at<v2i>(gA + (size_t)(e >> 1) * GROUP_BYTES, lws[u]);
                wpk[u] = ((uint32_t)h.x & 0xffffu) | ((uint32_t)h.y << 16);
            }
        }
        unsigned short wsb[NWS];
#pragma unroll
        for (int u = 0; u < NWS; ++u)
            wsb[u] = (unsigned short)(even ? (wpk[u] & 0xffffu) : (wpk[u] >> 16));
        float dc[NC];
        const char* xs_e = gX + (size_t)e * x_epoch;
#pragma unroll
        for (int c = 0; c < NC; ++c)
            dc[c] = *at<float>(xs_e, lxs + (uint32_t)(16 * c * XBLK_BYTES));
        IU4_V2B_FENCE();  // metadata requests precede the staging packet
        if constexpr (nxt) {
            const int en = e + 1;
            stage_load(st, gA + a_epoch(en), gX + (size_t)en * x_epoch, a_off, x_off);
        }
        IU4_V2B_FENCE();

        // Flattened (a, s) step sequence; step i+1 operands are requested
        // before step i's WMMAs.
        v2i av = *(const v2i*)(slot + aoff);
        v2i xv[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) xv[c] = *(const v2i*)(slot + xoff + c * 1024);
        v8i C[NC];
        int wsf = 0;
        constexpr int kc0 = 0x4B400000;
        const v8i kC0 = (v8i){kc0, kc0, kc0, kc0, kc0, kc0, kc0, kc0};
#pragma unroll
        for (int a = 0; a < NA; ++a) {
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int i = a * 8 + s + 1;
                v2i av_n;
                v2i xv_n[NC];
                if (i < NA * 8) {
                    const int an = i >> 3, sn = i & 7;
                    av_n = *(const v2i*)(slot + aoff + an * 1024 + sn * 128);
#pragma unroll
                    for (int c = 0; c < NC; ++c)
                        xv_n[c] = *(const v2i*)(slot + xoff + c * 1024 + sn * 128);
                }
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    C[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(
                        true, av, true, xv[c], s == 0 ? kC0 : C[c], false);
                if (i < NA * 8) {
                    av = av_n;
#pragma unroll
                    for (int c = 0; c < NC; ++c) xv[c] = xv_n[c];
                }
                if (s != 7) IU4_V2B_FENCE();
            }
            // Fold pass a (row fragment a, all token fragments c). Row
            // 16a+8hi+j's FP16 scale, widened once so t1 = sc*d is a plain
            // v_mul_f32, is shared from lane 8(a&1)+j of this lane's row of 16.
            float scf[8];
            if ((a & 1) == 0)
                wsf = __float_as_int(__half2float(__ushort_as_half(wsb[a >> 1])));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                scf[j] = __int_as_float(row_share(wsf, 8 * (a & 1) + j));
                asm volatile("" : "+v"(scf[j]));
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float t1 = __fmul_rn(scf[j], dc[c]);
                    const float cf = __fsub_rn(__int_as_float(C[c][j]), 12582912.0f);
                    sum[a][c][j] = __builtin_fmaf(t1, cf, sum[a][c][j]);
                }
                // Lifetime fence: this fragment's folded sums exist here, so
                // its int32 chain dies before the next pass.
                asm volatile("" : "+v"(sum[a][c][0]), "+v"(sum[a][c][1]),
                                  "+v"(sum[a][c][2]), "+v"(sum[a][c][3]),
                                  "+v"(sum[a][c][4]), "+v"(sum[a][c][5]),
                                  "+v"(sum[a][c][6]), "+v"(sum[a][c][7]));
            }
            IU4_V2B_FENCE();
        }
        if constexpr (nxt) stage_store(st, lds + ((e + 1) & 1) * SLOT_BYTES, lds_off);
        __syncthreads();
    };
    for (int e = 0; e + 1 < E; ++e) epoch(std::true_type{}, e);
    epoch(std::false_type{}, E - 1);

    // Epilogue: lane owns rows 8*hi..8*hi+7 of each fragment for one token.
    float* ytile = Y + (size_t)col0 * M + row0;
    uint32_t yoff[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c)
        yoff[c] = ((uint32_t)(wc * 64 + 16 * c + lr) * (uint32_t)M +
                   (uint32_t)(wr * WAVE_ROWS + 8 * hi)) * 4u;
    // ADD issues the residual loads of EG row fragments before any of their
    // stores (Y loads cannot be hoisted past Y stores by the compiler).
    constexpr int EG = 2;
#pragma unroll
    for (int a0 = 0; a0 < NA; a0 += EG) {
        v4f o[EG][NC][2];
        if constexpr (IS_ADD) {
#pragma unroll
            for (int g = 0; g < EG; ++g)
#pragma unroll
                for (int c = 0; c < NC; ++c)
#pragma unroll
                    for (int q = 0; q < 2; ++q)
                        o[g][c][q] = *at<v4f>(
                            ytile, yoff[c] + (uint32_t)(16 * (a0 + g) + 4 * q) * 4u);
        }
#pragma unroll
        for (int g = 0; g < EG; ++g) {
            const int a = a0 + g;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    v4f v = (v4f){sum[a][c][4 * q + 0], sum[a][c][4 * q + 1],
                                  sum[a][c][4 * q + 2], sum[a][c][4 * q + 3]};
                    if constexpr (IS_ADD) {
#pragma unroll
                        for (int k = 0; k < 4; ++k) v[k] = __fadd_rn(o[g][c][q][k], v[k]);
                    }
                    *(v4f*)((char*)ytile + yoff[c] + (uint32_t)(16 * a + 4 * q) * 4u) = v;
                }
            }
        }
    }
}

}  // namespace iu4_v2b

#define IU4_V2B_ENTRY(name, is_add)                                           \
    extern "C" __global__ __launch_bounds__(iu4_v2b::NT, 1) void name(        \
        const char* __restrict__ A, const char* __restrict__ Xq,              \
        float* __restrict__ Y, int M, int K, int N) {                         \
        iu4_v2b::body<is_add>(A, Xq, Y, M, K, N);                             \
    }
IU4_V2B_ENTRY(gemm_mq4g256v2_residual_iu4_v2b_set_gfx11, false)
IU4_V2B_ENTRY(gemm_mq4g256v2_residual_iu4_v2b_add_gfx11, true)
#undef IU4_V2B_ENTRY
#undef IU4_V2B_FENCE

#endif
