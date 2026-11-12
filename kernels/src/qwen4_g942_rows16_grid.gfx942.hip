// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// qwen4_g942_rows16_grid: the exact-gfx942 row-grid entries of the few-row BF16 MFMA GEMM, as
// their own module (gfx942 only).
//
// `gemm_bf16_mfma_gfx942_rows16` runs one workgroup per 16 weight rows and up to 16 batch rows;
// a projection of more rows is a host-launched chain of 16-row slices (or the wide entry). For
// a narrow, long-K weight (the HC read's down projection, M = 320, K = 10240: 20 workgroups) a
// slice launch fills 20 of the chip's 304 compute units and a 64-row wide launch stages the whole
// 64-row activation tile per 16 weight rows. These entries run ONE launch over
// `grid = [sum_j ceil(M_j / 16), ceil(batch / 16)]`: blockIdx.y selects a 16-row slice of X and of
// every D, and the verbatim rows16 body below runs on that slice with the base pointers and
// `batch = min(16, batch - 16 * blockIdx.y)` the host-side chain's slice launch passes. Every
// output element is therefore the rows16 entry's: one accumulator, one
// `v_mfma_f32_16x16x16bf16_1k` per 16-wide K step ascending from k = 0, the same operand mapping,
// seeded 0.0f, no split-K, the F32 -> BF16 X rule and the BF16 D rule unchanged. The blocks
// between the BEGIN / END markers are verbatim copies of `gemm_bf16_mfma.gfx942.hip`; a unit test
// in rdna-compute (`g942_rows16_grid_source_is_rows16_verbatim`) asserts each block occurs byte
// for byte in `GEMM_BF16_MFMA_GFX942_SRC`, so an edit of the production body that is not mirrored
// here fails the build.
//
// Requirements (checked by the launcher): K % 8 == 0, every A and X base 16-byte aligned (the X
// row pitch K * {2,4} bytes is then 16-byte aligned for every slice), 1 <= batch. Block
// [576,1,1]; dynamic LDS = 2 * (16 + min(16, batch)) * R16_LD * 2 bytes (a short last slice uses
// less). `_x2` takes an F32 X of 9..=16 rows per slice, as the rows16 entry does.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// BEGIN verbatim gemm_bf16_mfma.gfx942.hip
typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

// MFMA F32_16x16x16_BF16 intrinsic — operands are bit-cast __bf16x4 → short x4.
// LLVM's amdgcn intrinsic models the BF16 lane bits as i16; bit_cast is the
// no-op punning the compiler folds away.
__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}
// END verbatim gemm_bf16_mfma.gfx942.hip

// BEGIN verbatim gemm_bf16_mfma.gfx942.hip
#define R16_KC      256                 // K elements per LDS chunk (16 MFMA steps)
#define R16_LD      (R16_KC + 4)        // LDS row pitch (bf16): 520 B, conflict-free b64 reads
#define R16_LOADERS 512                 // waves 1..8; loader t loads A row t>>5, k8 (t&31)*8
#define R16_THREADS (64 + R16_LOADERS)  // wave 0 = the MFMA chain
#define R16_DEPTH   8                   // chunks in flight per loader thread

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// `convert_f32_to_bf16`'s rule (kernels/src/convert_f32_to_bf16.hip) on the
// F32 bit pattern, branch-free: NaN -> sign | 0x7fc0, else RNE on the low 16.
__device__ static inline unsigned int r16_f32_bits_to_bf16_bits(unsigned int bits) {
    const unsigned int nan_bits = ((bits >> 16) & 0x8000u) | 0x7fc0u;
    const unsigned int lsb = (bits >> 16) & 1u;
    const unsigned int lower = bits & 0xffffu;
    const unsigned int round_bit = (lower >> 15) & 1u;
    const unsigned int sticky = (lower & 0x7fffu) != 0u ? 1u : 0u;
    const unsigned int up = round_bit & (sticky | lsb);
    const unsigned int rne = ((bits >> 16) + up) & 0xffffu;
    return (bits & 0x7fffffffu) > 0x7f800000u ? nan_bits : rne;
}

// LDS-only workgroup barrier: orders the LDS tile writes against the chain's
// reads without waiting on the in-flight global prefetch.
__device__ static inline void r16_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int XI>
__device__ static inline void gemm_bf16_mfma_rows16_body(
    const void* __restrict__ X,
    const bf16_t* A0, const bf16_t* A1, const bf16_t* A2, const bf16_t* A3,
    void* D0, void* D1, void* D2, void* D3,
    int M0, int M1, int M2, int M3,
    int K, int batch, int x_f32, int d_bf16_mask
) {
    extern __shared__ unsigned short r16_lds[];   // [2][16 + batch][R16_LD] bf16 bits

    // This workgroup's member and output tile (uniform).
    const int t1 = (M0 + 15) >> 4;
    const int t2 = t1 + ((M1 + 15) >> 4);
    const int t3 = t2 + ((M2 + 15) >> 4);
    const int tile = blockIdx.x;
    const int member = tile < t1 ? 0 : tile < t2 ? 1 : tile < t3 ? 2 : 3;
    const bf16_t* A = member == 0 ? A0 : member == 1 ? A1 : member == 2 ? A2 : A3;
    void* D         = member == 0 ? D0 : member == 1 ? D1 : member == 2 ? D2 : D3;
    const int M     = member == 0 ? M0 : member == 1 ? M1 : member == 2 ? M2 : M3;
    const int first = member == 0 ? 0 : member == 1 ? t1 : member == 2 ? t2 : t3;
    const int m0    = (tile - first) * 16;
    const int d_bf16 = (d_bf16_mask >> member) & 1;

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int buf_elems   = (16 + batch) * R16_LD;
    const int total_steps = ((K + 127) / 128) * 8;     // the tiled kernel's K walk
    const int nchunks     = (total_steps + 15) / 16;

    if (wave == 0) {
        // ── The chain. ──
        __builtin_amdgcn_s_setprio(3);
        const int lane   = threadIdx.x;
        const int n_lane = lane & 15;
        const int strip  = lane >> 4;
        // Columns n >= batch read X row 0: their outputs are discarded and an
        // MFMA column never feeds another column.
        const int b_row = 16 + (n_lane < batch ? n_lane : 0);
        const int a_off = n_lane * R16_LD + strip * 4;
        const int b_off = b_row * R16_LD + strip * 4;
        vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};
        u32x2 a0[8], b0[8], a1[8], b1[8];

        // Half chunk H (0 / 1) of chunk C: MFMA steps 8H..8H+7.
#define R16_READ(AV, BV, C, H)                                                        \
        {                                                                             \
            const unsigned short* base = r16_lds + ((C) & 1) * buf_elems + (H) * 128; \
            const u32x2* pa = (const u32x2*)(base + a_off);                           \
            const u32x2* pb = (const u32x2*)(base + b_off);                           \
            _Pragma("unroll")                                                         \
            for (int j = 0; j < 8; j++) { AV[j] = pa[j * 4]; BV[j] = pb[j * 4]; }    \
        }
#define R16_MFMA(AV, BV)                                                              \
        {                                                                             \
            _Pragma("unroll")                                                         \
            for (int j = 0; j < 8; j++)                                               \
                c_acc = mfma_bf16(__builtin_bit_cast(bf16x4, AV[j]),                  \
                                  __builtin_bit_cast(bf16x4, BV[j]), c_acc);          \
        }

        r16_lds_barrier();
        R16_READ(a0, b0, 0, 0);
        for (int c = 0; c < nchunks; c++) {
            // A chunk is 16 steps, or 8 for the last chunk of a K that is an
            // odd multiple of 128.
            const bool full = total_steps - c * 16 > 8;
            if (full) R16_READ(a1, b1, c, 1);
            R16_MFMA(a0, b0);
            if (c + 1 < nchunks) {
                r16_lds_barrier();
                R16_READ(a0, b0, c + 1, 0);
            }
            if (full) R16_MFMA(a1, b1);
        }
#undef R16_READ
#undef R16_MFMA

        // Output: lane (strip, n_lane) holds D[n = n_lane, m = m0 + strip*4 + r].
        if (n_lane < batch) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int gm = m0 + strip * 4 + r;
                if (gm < M) {
                    const long long o = (long long)n_lane * M + gm;
                    if (d_bf16) {
                        // Copy the lane out first: a bit_cast straight off an
                        // ext_vector element reads element 0 on this toolchain.
                        const float acc = c_acc[r];
                        unsigned int bits;
                        __builtin_memcpy(&bits, &acc, 4);
                        ((unsigned short*)D)[o] = (unsigned short)r16_f32_bits_to_bf16_bits(bits);
                    } else {
                        ((float*)D)[o] = c_acc[r];
                    }
                }
            }
        }
        return;
    }

    // ── Loaders (waves 1..8). ──
    const int tid = threadIdx.x - 64;

    // A: one 16-B load per loader per chunk.
    const int  a_row  = tid >> 5;
    const int  a_k8   = (tid & 31) * 8;
    const int  a_m    = m0 + a_row;
    const bool a_live = a_m < M;
    const u32x4* a_src = (const u32x4*)(A + (long long)(a_live ? a_m : 0) * K);

    // X: 16-B items, 4 F32 or 8 BF16 elements each.
    const int x_elems     = x_f32 ? 4 : 8;
    const int x_esz       = x_f32 ? 4 : 2;
    const int x_ipr_shift = x_f32 ? 6 : 5;             // log2(R16_KC / x_elems)
    const int x_ipr       = 1 << x_ipr_shift;          // items per row per chunk
    const int x_items     = batch * x_ipr;

    u32x4 ra[R16_DEPTH];
    u32x4 rx[R16_DEPTH][XI];
    const u32x4 zero4 = {0u, 0u, 0u, 0u};

    // Every load is issued unconditionally (a dead chunk or masked item reads
    // one valid line and is never stored), so the compiler's vmcnt tracking
    // sees a fixed ring and waits only for the slot it consumes.
#define R16_LOAD(SLOT, CHUNK)                                                         \
    {                                                                                 \
        const int  kb   = (CHUNK) * R16_KC;                                           \
        const int  ka   = kb + a_k8;                                                  \
        const bool live = (CHUNK) < nchunks;                                          \
        const bool a_ok = live && a_live && ka < K;                                   \
        ra[SLOT] = a_src[a_ok ? (ka >> 3) : 0];                                       \
        _Pragma("unroll")                                                             \
        for (int xi = 0; xi < XI; xi++) {                                             \
            const int  item = tid + xi * R16_LOADERS;                                 \
            const int  n    = item >> x_ipr_shift;                                    \
            const int  kx   = kb + (item & (x_ipr - 1)) * x_elems;                    \
            const bool x_ok = live && item < x_items && kx < K;                       \
            const long long off = x_ok ? ((long long)n * K + kx) * x_esz : 0;         \
            rx[SLOT][xi] = *(const u32x4*)((const char*)X + off);                     \
        }                                                                             \
    }

    // Zero-fill past K, as the tiled kernel does.
#define R16_STORE(SLOT, CHUNK)                                                        \
    {                                                                                 \
        const int kb = (CHUNK) * R16_KC;                                              \
        unsigned short* base = r16_lds + ((CHUNK) & 1) * buf_elems;                   \
        u32x2* pa = (u32x2*)(base + a_row * R16_LD + a_k8);                           \
        const u32x4 va = (a_live && kb + a_k8 < K) ? ra[SLOT] : zero4;                \
        pa[0] = (u32x2){va.x, va.y};                                                  \
        pa[1] = (u32x2){va.z, va.w};                                                  \
        _Pragma("unroll")                                                             \
        for (int xi = 0; xi < XI; xi++) {                                             \
            const int item = tid + xi * R16_LOADERS;                                  \
            if (item < x_items) {                                                     \
                const int n   = item >> x_ipr_shift;                                  \
                const int col = (item & (x_ipr - 1)) * x_elems;                       \
                u32x2* px = (u32x2*)(base + (16 + n) * R16_LD + col);                 \
                const u32x4 v = (kb + col < K) ? rx[SLOT][xi] : zero4;                \
                if (x_f32) {                                                          \
                    const unsigned int h0 = r16_f32_bits_to_bf16_bits(v.x);           \
                    const unsigned int h1 = r16_f32_bits_to_bf16_bits(v.y);           \
                    const unsigned int h2 = r16_f32_bits_to_bf16_bits(v.z);           \
                    const unsigned int h3 = r16_f32_bits_to_bf16_bits(v.w);           \
                    px[0] = (u32x2){h0 | (h1 << 16), h2 | (h3 << 16)};                \
                } else {                                                              \
                    px[0] = (u32x2){v.x, v.y};                                        \
                    px[1] = (u32x2){v.z, v.w};                                        \
                }                                                                     \
            }                                                                         \
        }                                                                             \
    }

    #pragma unroll
    for (int u = 0; u < R16_DEPTH; u++) R16_LOAD(u, u);

    int c0 = 0;
    for (; c0 + R16_DEPTH <= nchunks; c0 += R16_DEPTH) {
        #pragma unroll
        for (int u = 0; u < R16_DEPTH; u++) {
            const int c = c0 + u;
            R16_STORE(u, c);
            R16_LOAD(u, c + R16_DEPTH);
            r16_lds_barrier();
        }
    }
    // Tail: the last nchunks % R16_DEPTH chunks are already in flight.
    #pragma unroll
    for (int u = 0; u < R16_DEPTH; u++) {
        if (c0 + u < nchunks) {
            R16_STORE(u, c0 + u);
            r16_lds_barrier();
        }
    }
#undef R16_LOAD
#undef R16_STORE
}
// END verbatim gemm_bf16_mfma.gfx942.hip

#define R16_GRID_ENTRY(NAME, XI)                                                         \
extern "C" __global__ __launch_bounds__(R16_THREADS, 2)                                  \
void NAME(                                                                               \
    const void* __restrict__ X,                                                          \
    const bf16_t* __restrict__ A0, const bf16_t* __restrict__ A1,                        \
    const bf16_t* __restrict__ A2, const bf16_t* __restrict__ A3,                        \
    void* __restrict__ D0, void* __restrict__ D1,                                        \
    void* __restrict__ D2, void* __restrict__ D3,                                        \
    int M0, int M1, int M2, int M3,                                                      \
    int K, int batch, int x_f32, int d_bf16_mask                                         \
) {                                                                                      \
    const int r0  = blockIdx.y * 16;                                                     \
    const int sub = (batch - r0) < 16 ? (batch - r0) : 16;                               \
    const char* Xs = (const char*)X + (long long)r0 * K * (x_f32 ? 4 : 2);               \
    void* E0 = (char*)D0 + (long long)r0 * M0 * (((d_bf16_mask >> 0) & 1) ? 2 : 4);      \
    void* E1 = (char*)D1 + (long long)r0 * M1 * (((d_bf16_mask >> 1) & 1) ? 2 : 4);      \
    void* E2 = (char*)D2 + (long long)r0 * M2 * (((d_bf16_mask >> 2) & 1) ? 2 : 4);      \
    void* E3 = (char*)D3 + (long long)r0 * M3 * (((d_bf16_mask >> 3) & 1) ? 2 : 4);      \
    gemm_bf16_mfma_rows16_body<XI>(Xs, A0, A1, A2, A3, E0, E1, E2, E3,                   \
                                   M0, M1, M2, M3, K, sub, x_f32, d_bf16_mask);          \
}

R16_GRID_ENTRY(qwen4_g942_rows16_grid, 1)
R16_GRID_ENTRY(qwen4_g942_rows16_grid_x2, 2)
#undef R16_GRID_ENTRY
