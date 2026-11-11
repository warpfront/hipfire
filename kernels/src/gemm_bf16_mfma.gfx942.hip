#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// BF16 × BF16 → FP32 MFMA-direct GEMM for gfx942 (MI300X CDNA3).
//
// Foundation kernel for the Tier 1 hipfire-native BF16 calibration path
// (replaces Tier 2 llama.cpp `llama-imatrix` + PyTorch `collect_hessian.py`
// subprocess wrappers). Both operands are BF16 in global memory — no
// dequantization, no FP16 shadow buffer — feeding straight into
// `__builtin_amdgcn_mfma_f32_16x16x16bf16`.
//
// Layout copies the v3 sweet-spot from `gemm_hfq4g256_residual_mfma.gfx942.hip`
// (commit 496edfbc, 97.6% of rocBLAS Tensile on HFQ4 prefill):
//
//   A:  [M, K] BF16, row-major  (M = output features, K = input dim)
//   B:  [batch_size, K] BF16, row-major (calibration activations / token tiles)
//   D:  [batch_size, M] FP32, row-major
//
// Geometry:
//   block = [256, 1, 1]                4 wave64/WG, 32×32 output tile per WG
//   grid  = [(M+31)/32, (batch+31)/32] one WG per (output-row-tile, batch-tile)
//   LDS   = 32 × K_CHUNK × 2 B         B-tile cooperative load (4 KB at K_CHUNK=128)
//
// Intrinsic: `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`.
// The `_1k` suffix is the gfx90a/gfx942 inherited name for the 16-K-wide
// BF16 MFMA (4 BF16 lanes × 16 m-rows × 16 n-cols per call). The operand
// vector type is `short __attribute__((ext_vector_type(4)))` — clang
// represents BF16 lane bits as i16 here, so we bit-cast a `__bf16` x4
// vector into the i16 x4 expected by the intrinsic. No data movement;
// pure type punning.
//
// MFMA F32_16x16x16_BF16 lane layout — identical to the F32_16x16x16_F16
// variant verified via `docs/investigations/2026-05-19-mfma-hfq4/mfma_test.cpp`
// (max_err 8.9e-08 on F16; the BF16 intrinsic shares the same lane mapping,
// only the operand dtype differs):
//
//   A operand:  lane l holds a[r] = A[m = l%16,        k = (l/16)*4 + r]
//   B operand:  lane l holds b[r] = B[k = (l/16)*4 + r, n = l%16]
//   D output:   lane l holds c[r] = D[m = (l/16)*4 + r, n = l%16]   ← strip-major
//
// The output strip-major layout (lane l holds D[m=(l/16)*4+r, n=l%16])
// differs from the A-operand m-major-by-lane input layout. Easy bug to
// introduce when swapping intrinsics. The v3 HFQ4 kernel has the same
// trap and resolves it the same way.
//
// Wave-tile arrangement within a WG (4 wave64 → 2×2 wave grid over the
// 32×32 output tile):
//
//   wave_id 0 → output sub-tile (m_off=0,  n_off=0)
//   wave_id 1 → output sub-tile (m_off=0,  n_off=16)
//   wave_id 2 → output sub-tile (m_off=16, n_off=0)
//   wave_id 3 → output sub-tile (m_off=16, n_off=16)
//
// All 4 waves cooperatively load one B-chunk [32 cols × K_CHUNK k] into LDS
// per outer-k step, then each wave's 16×16 MFMA reads B from LDS. v3's
// HFQ4 measurement showed this LDS B-tile is the lever that closes the
// gap to rocBLAS — same expectation applies here (B is now actual BF16
// activations in HBM, the cooperative load amortizes HBM bandwidth across
// 4 MFMA iterations sharing the chunk).

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

#define K_CHUNK 128
#define N_TILE  32

extern "C" __global__ __launch_bounds__(256)
void gemm_bf16_mfma_gfx942(
    const bf16_t* __restrict__ A,    // [M, K] BF16 row-major
    const bf16_t* __restrict__ B,    // [batch_size, K] BF16 row-major
    float*        __restrict__ D,    // [batch_size, M] FP32 row-major (overwritten)
    int M, int K, int batch_size
) {
    __shared__ bf16_t lds_b[N_TILE * K_CHUNK];   // 32 × 128 = 4096 bf16 = 8 KB

    const int tid       = threadIdx.x;
    const int wave_id   = tid >> 6;             // 0..3
    const int lane_in_w = tid & 63;
    const int n_lane    = lane_in_w & 15;       // 0..15  (A.m, B.n, D.n within tile)
    const int strip     = lane_in_w >> 4;       // 0..3   (k-strip / D row-strip)

    const int wave_m_off = (wave_id & 2) ? 16 : 0;
    const int wave_n_off = (wave_id & 1) ? 16 : 0;
    const int m_offset = blockIdx.x * 32 + wave_m_off;          // global output-row start
    const int b_offset_global = blockIdx.y * 32;                // global batch start (per-WG)
    const int b_offset = b_offset_global + wave_n_off;          // per-wave batch start

    const int my_m = m_offset + n_lane;                         // this lane's output row

    vfloat4 c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // K outer loop in chunks of K_CHUNK (= 8 MFMA-K=16 iterations per chunk).
    for (int kc = 0; kc < K; kc += K_CHUNK) {
        // ─── Cooperative LDS load of B chunk [32 cols × K_CHUNK k] ───
        // 256 threads load 32 × 128 = 4096 bf16 = 16 per thread.
        __syncthreads();
        #pragma unroll
        for (int li = 0; li < 16; li++) {
            int idx = tid + li * 256;
            int b_col = idx / K_CHUNK;             // 0..31
            int b_k   = idx % K_CHUNK;             // 0..K_CHUNK-1
            int gb = b_offset_global + b_col;
            int gk = kc + b_k;
            bf16_t v = (gb < batch_size && gk < K)
                ? B[(long long)gb * K + gk]
                : (bf16_t)0;
            lds_b[b_col * K_CHUNK + b_k] = v;
        }
        __syncthreads();

        // ─── MFMA loop over chunk in 16-element K steps ───
        #pragma unroll
        for (int kk = 0; kk < K_CHUNK; kk += 16) {
            int gk_start = kc + kk;

            // A operand: each lane reads 4 contiguous K-elements
            //   k = gk_start + strip*4 + r for r=0..3
            // Out-of-bounds rows / k's get zero (clean boundary handling).
            bf16x4 a_reg;
            int a_k_base = gk_start + strip * 4;
            if (my_m < M) {
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int gk = a_k_base + r;
                    a_reg[r] = (gk < K)
                        ? A[(long long)my_m * K + gk]
                        : (bf16_t)0;
                }
            } else {
                a_reg[0] = (bf16_t)0; a_reg[1] = (bf16_t)0;
                a_reg[2] = (bf16_t)0; a_reg[3] = (bf16_t)0;
            }

            // B operand: read 4 contiguous K-elements from LDS
            //   col_in_tile = wave_n_off + n_lane  (0..31, this wave's n half)
            //   k_in_chunk_base = kk + strip*4
            int b_col_in_tile   = wave_n_off + n_lane;
            int k_in_chunk_base = kk + strip * 4;
            bf16x4 b_reg;
            b_reg[0] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 0];
            b_reg[1] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 1];
            b_reg[2] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 2];
            b_reg[3] = lds_b[b_col_in_tile * K_CHUNK + k_in_chunk_base + 3];

            c_acc = mfma_bf16(a_reg, b_reg, c_acc);
        }
    }

    // ─── Output write: strip-major over m ─────────────────────────────
    // Lane (strip, n_lane) holds c_acc[r] = D[m = m_offset + strip*4 + r,
    //                                          n = b_offset + n_lane]
    // D is [batch_size, M] row-major: D[b, m] = D[b * M + m]
    //
    // Plain GEMM: overwrite (no residual add). The Tier 1 calibration
    // path uses this kernel as a forward primitive; the Σx² / outer-
    // product reductions sit on top and accumulate themselves.
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int gm = m_offset + strip * 4 + r;
        const int gb = b_offset + n_lane;
        if (gm < M && gb < batch_size) {
            D[(long long)gb * M + gm] = c_acc[r];
        }
    }
}

// ─── Few-row entry (1..=16 batch rows): the decode / short-tail route ──────
//
// `gemm_bf16_mfma_gfx942` at batch <= 16 wastes the 32×32 tile (two of its
// four waves compute only padding columns), loads A with 2-byte scalar loads
// and drains every global load at each chunk barrier, so at M=1 it streams
// weights at well under 0.1 TB/s. These entries produce the SAME BYTES per
// output element and are the launcher's route for batch <= 16:
//
//   * Each output element is the identical MFMA chain: one
//     `v_mfma_f32_16x16x16bf16_1k` per 16-wide K step, ascending from k = 0,
//     accumulator seeded with 0.0, the same (k -> lane, slot) operand mapping
//     (lane l: a[r] = A[m = l%16, k = step*16 + (l/16)*4 + r],
//              b[r] = B[k = step*16 + (l/16)*4 + r, n = l%16]),
//     and the same step count: ceil(K/128)*8, the tiled kernel's K_CHUNK
//     walk including its zero-filled tail steps past K. An MFMA output
//     element depends only on its own A row, its own B column and its own
//     accumulator, so the 16-row output tile and whatever the unused batch
//     columns hold do not change its bits.
//   * Only the data movement differs. One workgroup per 16 output rows:
//     wave 0 runs the chain and nothing else; waves 1..8 load A (and the
//     activation) with 16-byte global loads, R16_DEPTH chunks in flight per
//     thread, into a double-buffered LDS tile. One LDS-only barrier per
//     256-K chunk (no vmcnt drain); wave 0 reads the next half chunk's
//     operands from LDS while its MFMAs run on the current one.
//   * One launch serves up to four weights that share X and K (a projection
//     group): the grid is the concatenation of their 16-row tiles and each
//     workgroup picks its member from blockIdx.x. Members are independent;
//     a member's bytes do not depend on its siblings.
//   * `x_f32 != 0`: X is F32 and is rounded to BF16 in-kernel with exactly
//     `convert_f32_to_bf16`'s rule (RNE, NaN -> sign | 0x7fc0), i.e. the
//     bytes the route's separate staging launch would have produced.
//   * bit j of `d_bf16_mask`: member j's D is BF16, rounded from the F32
//     accumulator by the same rule (the route's F32 landing +
//     `convert_f32_to_bf16`); otherwise F32.
//
// Requirements (checked by the launcher, which otherwise keeps the tiled
// kernel): K % 8 == 0, every A and X base 16-byte aligned, 1 <= batch <= 16.
// `_rows16` takes up to 512 16-byte X items per chunk (BF16 X, or F32 X with
// batch <= 8); `_rows16_x2` takes up to 1024 (F32 X with batch 9..=16).
//
// Geometry: block = [576,1,1], grid = [sum_j ceil(M_j/16),1,1], dynamic LDS =
// 2 * (16 + batch) * R16_LD * 2 bytes.

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

#define R16_ENTRY(NAME, XI)                                                           \
extern "C" __global__ __launch_bounds__(R16_THREADS, 2)                               \
void NAME(                                                                            \
    const void* __restrict__ X,                                                       \
    const bf16_t* __restrict__ A0, const bf16_t* __restrict__ A1,                     \
    const bf16_t* __restrict__ A2, const bf16_t* __restrict__ A3,                     \
    void* __restrict__ D0, void* __restrict__ D1,                                     \
    void* __restrict__ D2, void* __restrict__ D3,                                     \
    int M0, int M1, int M2, int M3,                                                   \
    int K, int batch, int x_f32, int d_bf16_mask                                      \
) {                                                                                   \
    gemm_bf16_mfma_rows16_body<XI>(X, A0, A1, A2, A3, D0, D1, D2, D3,                 \
                                   M0, M1, M2, M3, K, batch, x_f32, d_bf16_mask);     \
}

R16_ENTRY(gemm_bf16_mfma_gfx942_rows16, 1)
R16_ENTRY(gemm_bf16_mfma_gfx942_rows16_x2, 2)
#undef R16_ENTRY
