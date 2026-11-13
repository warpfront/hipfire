// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Qwen4 gfx942 C1 fold, elementwise class (kill switch
// HIPFIRE_QWEN4_G942_FOLD_ELEM).
//
// qwen4_g942_gemm_bf16_rows16_hc_act / _x2: the HC read's down projection and
// its activation as ONE launch. The decode route today issues, per HC read,
//
//   gemm_bf16_mfma_gfx942_rows16   low = down(normalized)          (F32 out)
//   bf16_roundtrip_f32             low = bf16(low)
//   scale_f32                      low *= scale
//   bf16_roundtrip_f32             low = bf16(low)
//   silu_f32                       low = low / (1 + expf(-low))
//   bf16_roundtrip_f32             low = bf16(low)
//
// (execute_hyper_read_inner, the non-gfx11 arm; 6 launches). The activation is
// elementwise over the GEMM's F32 outputs, so applying it to each accumulator
// element before its single store produces the same bytes. Every operation is
// copied from the unfused kernels:
//   * bf16(.)  = static_cast<hip_bfloat16> / static_cast<float>, the
//     tensor_ops.hip `bf16_roundtrip_f32` / `hc_activation_fused_f32` helpers
//     (hip_bfloat16.h round-to-nearest-even, same NaN/Inf handling);
//   * `value *= scale` and `value / (1.0f + expf(-value))` are the
//     `scale_f32` / `silu_f32` expressions verbatim.
//
// The GEMM body is a single-member copy of gemm_bf16_mfma_rows16_body in
// gemm_bf16_mfma.gfx942.hip (that file stays untouched: its hipcc object is
// the Route-A twin's anchor). The MFMA chain, the loader schedule, the LDS
// geometry and the `x_f32` staging rule are identical, so the accumulator of
// each output element equals the unfused kernel's. Only the output store
// changes: F32 destination, activation applied.
//
//   X: [batch, K] F32 (x_f32 != 0, rounded to BF16 in-kernel) or BF16
//   A: [M, K] BF16 weight          D: [batch, M] F32
//
// Requirements (checked by the launcher, which otherwise keeps the unfused
// route): K % 8 == 0, A and X bases 16-byte aligned, 1 <= batch <= 16, D
// disjoint from X. `_rows16_hc_act` takes up to 512 16-byte X items per chunk
// (BF16 X, or F32 X with batch <= 8); `_x2` up to 1024 (F32 X, batch 9..=16).
//
// Geometry: block = [576,1,1], grid = [ceil(M/16),1,1], dynamic LDS =
// 2 * (16 + batch) * R16_LD * 2 bytes.

typedef __bf16 bf16_t;
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef short  s16x4  __attribute__((ext_vector_type(4)));
typedef float  vfloat4 __attribute__((ext_vector_type(4)));

__device__ static inline vfloat4 mfma_bf16(bf16x4 a, bf16x4 b, vfloat4 c) {
    s16x4 ai = __builtin_bit_cast(s16x4, a);
    s16x4 bi = __builtin_bit_cast(s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

#define R16_KC      256                 // K elements per LDS chunk (16 MFMA steps)
#define R16_LD      (R16_KC + 4)        // LDS row pitch (bf16): 520 B, conflict-free b64 reads
#define R16_LOADERS 512                 // waves 1..8; loader t loads A row t>>5, k8 (t&31)*8
#define R16_THREADS (64 + R16_LOADERS)  // wave 0 = the MFMA chain
#define R16_DEPTH   8                   // chunks in flight per loader thread

typedef unsigned int u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// `convert_f32_to_bf16`'s rule on the F32 bit pattern (the X staging).
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

// LDS-only workgroup barrier.
__device__ static inline void r16_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The HC read activation on one F32 value: bf16 -> * scale -> bf16 -> SiLU ->
// bf16, in the unfused chain's expressions.
__device__ static inline float g942_hc_act(float value, float scale) {
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value *= scale;
    value = static_cast<float>(static_cast<hip_bfloat16>(value));
    value = value / (1.0f + expf(-value));
    return static_cast<float>(static_cast<hip_bfloat16>(value));
}

template <int XI>
__device__ static inline void g942_rows16_hc_act_body(
    const void* __restrict__ X,
    const bf16_t* A,
    float* D,
    int M, int K, int batch, int x_f32, float scale
) {
    extern __shared__ unsigned short r16_lds[];   // [2][16 + batch][R16_LD] bf16 bits

    const int m0 = blockIdx.x * 16;

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
                    // Copy the lane out first: a bit_cast straight off an
                    // ext_vector element reads element 0 on this toolchain.
                    const float acc = c_acc[r];
                    D[(long long)n_lane * M + gm] = g942_hc_act(acc, scale);
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

#define R16_HC_ACT_ENTRY(NAME, XI)                                                    \
extern "C" __global__ __launch_bounds__(R16_THREADS, 2)                               \
void NAME(                                                                            \
    const void* __restrict__ X,                                                       \
    const bf16_t* __restrict__ A,                                                     \
    float* __restrict__ D,                                                            \
    int M, int K, int batch, int x_f32, float scale                                   \
) {                                                                                   \
    g942_rows16_hc_act_body<XI>(X, A, D, M, K, batch, x_f32, scale);                  \
}

R16_HC_ACT_ENTRY(qwen4_g942_gemm_bf16_rows16_hc_act, 1)
R16_HC_ACT_ENTRY(qwen4_g942_gemm_bf16_rows16_hc_act_x2, 2)
#undef R16_HC_ACT_ENTRY
