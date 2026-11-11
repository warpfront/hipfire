#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// Wide BF16 × BF16 → FP32 MFMA GEMM entry for gfx942 (MI300X CDNA3): batch
// 1..=64 rows in ONE pass over each weight (P1b).
//
// `gemm_bf16_mfma_gfx942` (32×32 tiles) re-reads a weight for every 32-row
// tile and has a ~120 µs floor on small shapes; chaining
// `gemm_bf16_mfma_gfx942_rows16` once per 16 rows re-reads it ⌈rows/16⌉ times.
// This entry reads each weight tile from HBM ONCE per up-to-64 batch rows.
// The module is a sibling of `gemm_bf16_mfma.gfx942.hip` (its own JIT module,
// its own helper names) and takes the same kernarg ABI as
// `gemm_bf16_mfma_gfx942_rows16`.
//
// Exactness contract (every output element D[n, m] is bit-for-bit the tile
// entry's):
//   * the identical MFMA chain: accumulator seeded 0.0f, one
//     `v_mfma_f32_16x16x16bf16_1k` per 16-wide K step ascending from k = 0
//     (the tile entry runs ceil(K/128)*8 steps; this entry runs the steps
//     that start below K, rounded up to its chunk, and a step past K sees
//     zero-filled A and X, adding +0 to an accumulator that cannot be -0),
//     the same (k -> lane, slot) operand mapping
//       lane l: a[r] = A[m = l%16, k = step*16 + (l/16)*4 + r],
//               b[r] = X[n = l%16, k = step*16 + (l/16)*4 + r];
//     an MFMA output element depends only on its own A row, X column and
//     accumulator, so the weight tile, the other batch columns and the
//     unused columns do not change its bits;
//   * `x_f32 != 0`: X is F32 and is rounded to BF16 in-kernel with
//     `convert_f32_to_bf16`'s rule (RNE, NaN -> sign | 0x7fc0), once per
//     element per workgroup, as it is staged to LDS;
//   * bit j of `d_bf16_mask`: member j's D is BF16, rounded from the F32
//     accumulator by the same rule; otherwise F32.
//
// Geometry: one workgroup per MT = 16*MS output rows of one member (up to
// four members share X and K; the grid is the concatenation of their MT-row
// tiles); block = [512,1,1]. Entries: `_wide64` MS = 1, `_wide64_m32` MS = 2,
// `_wide64_m64` MS = 4. A taller tile reads the activation fewer times per
// weight byte (its F32 -> BF16 rounding too) but launches fewer workgroups.
//   * waves 0..3 are the chains: wave w owns batch columns 16w..16w+15 (waves
//     past ceil(batch/16) exit at once) and runs MS independent
//     one-accumulator chains, one per 16-row group of the tile. Both the
//     weight fragments and the X fragments come from LDS;
//   * waves 4..7 stream the weight tile and the 64 X rows of the launch:
//     16-byte global loads, WD_LOADS loads in flight per thread, into a
//     double-buffered LDS chunk of MT weight rows + 64 X rows by KC K
//     elements, with one LDS-only barrier per chunk (no vmcnt drain). The
//     chunk is KC = 128 K elements (8 MFMA steps) for MS = 1, 2 and 64 for
//     MS = 4: with the 4-element pad a 128-row, KC = 128 double buffer is
//     67,584 B, over the 65,536 B LDS of a gfx942 workgroup; at KC = 64 it
//     is 34,816 B. Dynamic LDS = 2 * (MT + 64) * (KC + 4) * 2 bytes, which
//     the launcher asserts <= 65,536.
//
// Requirements (checked by the launcher): K % 8 == 0, every A and X base
// 16-byte aligned, 1 <= batch <= 64.

typedef __bf16 wd_bf16;
typedef __bf16 wd_bf16x4 __attribute__((ext_vector_type(4)));
typedef short  wd_s16x4  __attribute__((ext_vector_type(4)));
typedef float  wd_f32x4  __attribute__((ext_vector_type(4)));
typedef unsigned int wd_u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned int wd_u32x4 __attribute__((ext_vector_type(4)));

#define WD_XROWS    64                  // X rows staged in LDS per chunk (the launch's whole batch)
#define WD_LOADERS  256                 // waves 4..7
#define WD_THREADS  (256 + WD_LOADERS)  // waves 0..3 = chains
#ifndef WD_LOADS
#define WD_LOADS    16                  // global loads in flight per loader thread (ring depth = WD_LOADS / loads per chunk)
#endif

__device__ static inline wd_f32x4 wd_mfma(wd_bf16x4 a, wd_bf16x4 b, wd_f32x4 c) {
    wd_s16x4 ai = __builtin_bit_cast(wd_s16x4, a);
    wd_s16x4 bi = __builtin_bit_cast(wd_s16x4, b);
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(ai, bi, c, 0, 0, 0);
}

// `convert_f32_to_bf16`'s rule (kernels/src/convert_f32_to_bf16.hip) on the
// F32 bit pattern, branch-free, result in the HIGH half of the word: RNE
// (`u + 0x7fff + lsb`, whose carry is the rule's round-up) and, for any NaN,
// sign | 0x7fc0. Equal to the rule for all 2^32 patterns.
__device__ static inline unsigned int wd_f32_bits_to_bf16_hi(unsigned int bits) {
    const unsigned int rne = bits + 0x7fffu + ((bits >> 16) & 1u);
    const unsigned int nan_bits = (bits & 0x80000000u) | 0x7fc00000u;
    return (bits & 0x7fffffffu) > 0x7f800000u ? nan_bits : rne;
}

__device__ static inline unsigned int wd_f32_bits_to_bf16_bits(unsigned int bits) {
    return wd_f32_bits_to_bf16_hi(bits) >> 16;
}

// Two rounded elements packed low = first, high = second.
__device__ static inline unsigned int wd_pack_bf16(unsigned int lo_bits, unsigned int hi_bits) {
    return __builtin_amdgcn_perm(wd_f32_bits_to_bf16_hi(hi_bits), wd_f32_bits_to_bf16_hi(lo_bits),
                                 0x07060302u);
}

// LDS-only workgroup barrier: orders the LDS tile writes against the chains'
// reads without waiting on the in-flight global prefetch.
__device__ static inline void wd_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <bool XF32, int MS, int KC>
__device__ static inline void gemm_bf16_mfma_wide_body(
    const void* __restrict__ X,
    const wd_bf16* A0, const wd_bf16* A1, const wd_bf16* A2, const wd_bf16* A3,
    void* D0, void* D1, void* D2, void* D3,
    int M0, int M1, int M2, int M3,
    int K, int batch, int d_bf16_mask
) {
    constexpr int MT    = 16 * MS;              // weight rows per workgroup
    constexpr int STEPS = KC / 16;              // MFMA steps per chunk
    constexpr int LD    = KC + 4;               // LDS row pitch (bf16): conflict-free b64 reads
    constexpr int BUF   = (MT + WD_XROWS) * LD; // bf16 elements of one chunk buffer
    constexpr int WIPR  = KC / 8;               // 16-B weight items per row per chunk
    constexpr int WQ    = MT * WIPR / WD_LOADERS;       // weight items per loader thread per chunk
    constexpr int XEL   = XF32 ? 4 : 8;         // X elements per 16-B item
    constexpr int ESZ   = XF32 ? 4 : 2;
    constexpr int XIPR  = KC / XEL;             // 16-B X items per row per chunk
    constexpr int XQ    = WD_XROWS * XIPR / WD_LOADERS; // X items per loader thread per chunk
    constexpr int DEPTH = (WD_LOADS / (WQ + XQ)) > 0 ? (WD_LOADS / (WQ + XQ)) : 1;
    static_assert(WQ * WD_LOADERS == MT * WIPR && XQ * WD_LOADERS == WD_XROWS * XIPR, "tile covers loader items");
    static_assert(2 * BUF * 2 <= 65536, "double-buffered chunk fits the gfx942 workgroup LDS");
    extern __shared__ unsigned short wd_lds[];  // [2][MT + 64][LD] bf16 bits

    // This workgroup's member and MT-row output tile (uniform).
    const int t1 = (M0 + MT - 1) / MT;
    const int t2 = t1 + (M1 + MT - 1) / MT;
    const int t3 = t2 + (M2 + MT - 1) / MT;
    const int tile = blockIdx.x;
    const int member = tile < t1 ? 0 : tile < t2 ? 1 : tile < t3 ? 2 : 3;
    const wd_bf16* A = member == 0 ? A0 : member == 1 ? A1 : member == 2 ? A2 : A3;
    void* D          = member == 0 ? D0 : member == 1 ? D1 : member == 2 ? D2 : D3;
    const int M      = member == 0 ? M0 : member == 1 ? M1 : member == 2 ? M2 : M3;
    const int first  = member == 0 ? 0 : member == 1 ? t1 : member == 2 ? t2 : t3;
    const int m0     = (tile - first) * MT;
    const int d_bf16 = (d_bf16_mask >> member) & 1;

    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int nchunks = (K + KC - 1) / KC;

    if (wave < 4) {
        // ── Chains: batch columns 16*wave .. 16*wave+15 over this tile's MS 16-row groups. ──
        // Columns n >= batch read staged X rows that are never real data
        // (row 0 is staged for them): their outputs are discarded and an MFMA
        // column never feeds another column.
        if (wave * 16 >= batch) return;
        __builtin_amdgcn_s_setprio(3);
        const int lane   = threadIdx.x & 63;
        const int n_lane = lane & 15;
        const int strip  = lane >> 4;
        const int n      = wave * 16 + n_lane;
        const int a_off  = n_lane * LD + strip * 4;
        const int b_off  = (MT + wave * 16 + n_lane) * LD + strip * 4;
        wd_f32x4 c_acc[MS];
        #pragma unroll
        for (int i = 0; i < MS; i++) c_acc[i] = (wd_f32x4){0.0f, 0.0f, 0.0f, 0.0f};

        for (int c = 0; c < nchunks; c++) {
            wd_lds_barrier();                                // chunk c is in LDS
            const unsigned short* base = wd_lds + (c & 1) * BUF;
            wd_u32x2 af[MS][STEPS];
            wd_u32x2 bf[STEPS];
            #pragma unroll
            for (int j = 0; j < STEPS; j++) bf[j] = *(const wd_u32x2*)(base + b_off + j * 16);
            #pragma unroll
            for (int i = 0; i < MS; i++) {
                #pragma unroll
                for (int j = 0; j < STEPS; j++)
                    af[i][j] = *(const wd_u32x2*)(base + a_off + i * 16 * LD + j * 16);
            }
            #pragma unroll
            for (int j = 0; j < STEPS; j++) {
                #pragma unroll
                for (int i = 0; i < MS; i++)
                    c_acc[i] = wd_mfma(__builtin_bit_cast(wd_bf16x4, af[i][j]),
                                       __builtin_bit_cast(wd_bf16x4, bf[j]), c_acc[i]);
            }
        }

        // Output: lane (strip, n_lane) holds D[n, m = m0 + 16*i + strip*4 + r].
        if (n < batch) {
            #pragma unroll
            for (int i = 0; i < MS; i++) {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    const int gm = m0 + i * 16 + strip * 4 + r;
                    if (gm < M) {
                        const long long o = (long long)n * M + gm;
                        if (d_bf16) {
                            // Copy the lane out first: a bit_cast straight off an
                            // ext_vector element reads element 0 on this toolchain.
                            const float acc = c_acc[i][r];
                            unsigned int bits;
                            __builtin_memcpy(&bits, &acc, 4);
                            ((unsigned short*)D)[o] = (unsigned short)wd_f32_bits_to_bf16_bits(bits);
                        } else {
                            ((float*)D)[o] = c_acc[i][r];
                        }
                    }
                }
            }
        }
        return;
    }

    // ── Loaders (waves 4..7): MT weight rows and 64 X rows x KC per chunk. ──
    const int tid = threadIdx.x - 256;
    // Weight items: item = tid + 256*g -> row item / WIPR, 8 elements at (item % WIPR) * 8.
    const int a_k8 = (tid % WIPR) * 8;
    const int a_r0 = tid / WIPR;
    constexpr int A_RSTEP = WD_LOADERS / WIPR;      // weight rows between a thread's items
    bool a_live[WQ];
    const wd_u32x4* a_src[WQ];
    #pragma unroll
    for (int g = 0; g < WQ; g++) {
        a_live[g] = m0 + a_r0 + A_RSTEP * g < M;
        a_src[g] = (const wd_u32x4*)(A + (long long)(a_live[g] ? m0 + a_r0 + A_RSTEP * g : 0) * K);
    }
    // X items: item = tid + 256*q -> row item / XIPR, XEL elements at (item % XIPR) * XEL.
    // A row at or past `batch` stages row 0 (never read as real data).
    const char* x_src[XQ];
    int x_row[XQ], x_col[XQ];
    #pragma unroll
    for (int q = 0; q < XQ; q++) {
        const int item = tid + WD_LOADERS * q;
        const int row  = item / XIPR;
        x_row[q] = row;
        x_col[q] = (item % XIPR) * XEL;
        x_src[q] = (const char*)X + ((long long)(row < batch ? row : 0) * K) * ESZ;
    }
    wd_u32x4 ra[DEPTH][WQ];
    wd_u32x4 rx[DEPTH][XQ];
    const wd_u32x4 zero4 = {0u, 0u, 0u, 0u};

    // Every load is issued unconditionally (a dead chunk or masked row reads
    // one valid line and is never stored), so the compiler's vmcnt tracking
    // sees a fixed ring and waits only for the slot it consumes.
#define WD_LOAD(SLOT, CHUNK)                                                          \
    {                                                                                 \
        const int  kb   = (CHUNK) * KC;                                               \
        const int  ka   = kb + a_k8;                                                  \
        const bool live = (CHUNK) < nchunks && ka < K;                                \
        _Pragma("unroll")                                                             \
        for (int g = 0; g < WQ; g++)                                                  \
            ra[SLOT][g] = *(a_src[g] + ((live && a_live[g]) ? (ka >> 3) : 0));        \
        _Pragma("unroll")                                                             \
        for (int q = 0; q < XQ; q++) {                                                \
            const int kx = kb + x_col[q];                                             \
            const bool xl = (CHUNK) < nchunks && kx < K;                              \
            rx[SLOT][q] = *(const wd_u32x4*)(x_src[q] + (xl ? (long long)kx * ESZ : 0)); \
        }                                                                             \
    }

    // Zero-fill past K, as the tiled kernel does.
#define WD_STORE(SLOT, CHUNK)                                                         \
    {                                                                                 \
        const int kb = (CHUNK) * KC;                                                  \
        unsigned short* base = wd_lds + ((CHUNK) & 1) * BUF;                          \
        const bool in_k = kb + a_k8 < K;                                              \
        _Pragma("unroll")                                                             \
        for (int g = 0; g < WQ; g++) {                                                \
            const wd_u32x4 v = (in_k && a_live[g]) ? ra[SLOT][g] : zero4;             \
            wd_u32x2* p = (wd_u32x2*)(base + (a_r0 + A_RSTEP * g) * LD + a_k8);       \
            p[0] = (wd_u32x2){v.x, v.y};                                              \
            p[1] = (wd_u32x2){v.z, v.w};                                              \
        }                                                                             \
        _Pragma("unroll")                                                             \
        for (int q = 0; q < XQ; q++) {                                                \
            const bool xk = kb + x_col[q] < K;                                        \
            const wd_u32x4 v = xk ? rx[SLOT][q] : zero4;                              \
            wd_u32x2* p = (wd_u32x2*)(base + (MT + x_row[q]) * LD + x_col[q]);        \
            if (XF32) {                                                               \
                p[0] = (wd_u32x2){wd_pack_bf16(v.x, v.y), wd_pack_bf16(v.z, v.w)};    \
            } else {                                                                  \
                p[0] = (wd_u32x2){v.x, v.y};                                          \
                p[1] = (wd_u32x2){v.z, v.w};                                          \
            }                                                                         \
        }                                                                             \
    }

    #pragma unroll
    for (int u = 0; u < DEPTH; u++) WD_LOAD(u, u);

    int c0 = 0;
    for (; c0 + DEPTH <= nchunks; c0 += DEPTH) {
        #pragma unroll
        for (int u = 0; u < DEPTH; u++) {
            const int c = c0 + u;
            WD_STORE(u, c);
            WD_LOAD(u, c + DEPTH);
            wd_lds_barrier();
        }
    }
    // Tail: the last nchunks % DEPTH chunks are already in flight.
    #pragma unroll
    for (int u = 0; u < DEPTH; u++) {
        if (c0 + u < nchunks) {
            WD_STORE(u, c0 + u);
            wd_lds_barrier();
        }
    }
#undef WD_LOAD
#undef WD_STORE
}

#define WD_ENTRY(NAME, MS, KC, MINB)                                                  \
extern "C" __global__ __launch_bounds__(WD_THREADS, MINB)                             \
void NAME(                                                                            \
    const void* __restrict__ X,                                                       \
    const wd_bf16* __restrict__ A0, const wd_bf16* __restrict__ A1,                   \
    const wd_bf16* __restrict__ A2, const wd_bf16* __restrict__ A3,                   \
    void* __restrict__ D0, void* __restrict__ D1,                                     \
    void* __restrict__ D2, void* __restrict__ D3,                                     \
    int M0, int M1, int M2, int M3,                                                   \
    int K, int batch, int x_f32, int d_bf16_mask                                      \
) {                                                                                   \
    if (x_f32) {                                                                      \
        gemm_bf16_mfma_wide_body<true, MS, KC>(X, A0, A1, A2, A3, D0, D1, D2, D3,     \
                                               M0, M1, M2, M3, K, batch, d_bf16_mask); \
    } else {                                                                          \
        gemm_bf16_mfma_wide_body<false, MS, KC>(X, A0, A1, A2, A3, D0, D1, D2, D3,    \
                                                M0, M1, M2, M3, K, batch, d_bf16_mask); \
    }                                                                                 \
}

WD_ENTRY(gemm_bf16_mfma_gfx942_wide64, 1, 128, 2)
WD_ENTRY(gemm_bf16_mfma_gfx942_wide64_m32, 2, 128, 1)
WD_ENTRY(gemm_bf16_mfma_gfx942_wide64_m64, 4, 64, 1)
#undef WD_ENTRY
