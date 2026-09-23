// Scratch screening twin of kernels/src/attention_q8_0_fa2_gqa.gfx11.hip
// (Q8 K mode, Q16 tile, KT32 only). V0 (all macros 0) must reproduce the
// production body bit-for-bit and ISA-for-ISA in its hot loops.
//
// Exact candidates (bit-identical output required):
//   V_QRES=N    keep Q dim chunks dc<N resident across KV tiles (replicated
//               half16, 8 VGPR per chunk); dc>=N reload per tile as today.
//   V_QRESX=1   keep ALL of Q resident non-replicated (half 0 lanes hold dims
//               dc*16+0..7, half 1 lanes dims +8..15; 4 VGPR per chunk) and
//               rebuild the replicated half16 per chunk with permlanex16.
//   V_QPIPE=1   software-pipeline the QK chunk loop (next chunk's Q/K loads
//               issued before this chunk's WMMAs).
//   V_VPIPE=1   software-pipeline the PV chunk loop.
//   V_VPF=1     prefetch the next tile's raw V bytes into registers (F6 for V).
//   V_CODES     bit0: K int8 codes (+f16 scales) in LDS, dequantized at use;
//               bit1: V codes transposed in LDS, dequantized at use.
// Ablations (inexact, timing attribution only):
//   V_ABL_NOQ=1    Q fragment synthesized from registers (no Q loads).
//   V_ABL_NOFILL=1 K/V fill only for tile 0.
#include <hip/hip_runtime.h>

#ifndef V_QRES
#define V_QRES 0
#endif
#ifndef V_QRESX
#define V_QRESX 0
#endif
#ifndef V_QPIPE
#define V_QPIPE 0
#endif
#ifndef V_VPIPE
#define V_VPIPE 0
#endif
#ifndef V_VPF
#define V_VPF 0
#endif
#ifndef V_CODES
#define V_CODES 0
#endif
#ifndef V_ABL_NOQ
#define V_ABL_NOQ 0
#endif
#ifndef V_ABL_NOFILL
#define V_ABL_NOFILL 0
#endif
#ifndef V_QWRAP
#define V_QWRAP 0
#endif
#ifndef V_QK_UNROLL
#define V_QK_UNROLL 2
#endif
#ifndef V_NOHELP
#define V_NOHELP 0
#endif
#define V_WG_THREADS (V_NOHELP ? 192 : 256)
#define V_PRAGMA_(x) _Pragma(#x)
#define V_UNROLL_N(n) V_PRAGMA_(unroll n)
#define V_CODES_V ((V_CODES & 2) != 0)
#define V_CODES_K ((V_CODES & 1) != 0)

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;
typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;
typedef uint32_t __attribute__((ext_vector_type(4))) uint32x4_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
    half2_t h2;
} fa2_dw_t;

__device__ __forceinline__ int fa2_swiz(int c)
{
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

// ---- LDS layout ----
// f16 planes (production): K plane KT*128 dwords, V plane KT*128 dwords.
// Codes: K codes KT*256 B (+ KT*8 f16 scales), V codes 256*KT B transposed
// (+ 8*KT f16 scales, block-major).
constexpr int KT = 32;
constexpr int K_PLANE_B = V_CODES_K ? (KT * 256 + KT * 16) : KT * 512;
constexpr int V_PLANE_B = V_CODES_V ? (256 * KT + 8 * KT * 2) : KT * 512;
constexpr int LDS_B = K_PLANE_B + V_PLANE_B;
constexpr int LDS_DWORDS = LDS_B / 4;

// Exact int8 -> f16 dequant of 4 biased codes (byte = code ^ 0x80) against
// two f16 scale pairs: returns RNE(scale*code) per half, identical to
// (_Float16)((float)scale * (float)code) (product exact in f32 and in the
// f16 FMA-free multiply's infinite-precision intermediate).
__device__ __forceinline__ void fa2_deq4(uint32_t t, half2_t s01, half2_t s23,
                                         uint32_t& lo, uint32_t& hi)
{
    fa2_dw_t a, b;
    // bytes: [t.b0, 0x64, t.b1, 0x64] -> halves 1024+b0, 1024+b1
    a.u = __builtin_amdgcn_perm(0x64646464u, t, 0x04010400u);
    b.u = __builtin_amdgcn_perm(0x64646464u, t, 0x04030402u);
    const half2_t bias = {(_Float16)1152.0f, (_Float16)1152.0f};
    a.h2 = (a.h2 - bias) * s01;
    b.h2 = (b.h2 - bias) * s23;
    lo = a.u;
    hi = b.u;
}

__device__ __forceinline__ void fa2_gqa_body(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base)
{
    constexpr int WG_THREADS = V_WG_THREADS;
    constexpr int K_SLOTS = KT * 8;                  // (key, 32-dim block)
    constexpr int V_SLOTS = KT * 16;                 // (key pair, block, quarter)
    constexpr int KT_KITERS = (K_SLOTS + WG_THREADS - 1) / WG_THREADS;
    constexpr int KT_VITERS = (V_SLOTS + WG_THREADS - 1) / WG_THREADS;
    constexpr int V_ROW = KT / 2;
    constexpr int V_MASK = KT / 2 - 1;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const bool compute = wave < 6;

    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + K_PLANE_B / 4;
    unsigned char* Kc = (unsigned char*)LDS;               // codes K
    _Float16* Ksc = (_Float16*)((unsigned char*)LDS + KT * 256);
    unsigned char* Vc = (unsigned char*)LDS + K_PLANE_B;   // codes V_T
    _Float16* Vsc = (_Float16*)(Vc + 256 * KT);            // [b][k]

    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    if (compute) {
        h_ml = 6 * kv_h + wave;
        qr_ml = q_base + ml;
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    auto loadQ = [&](int dc) -> half16_t {
        half16_t qf;
        const half8_t* qd =
            (const half8_t*)(q16 + qq_base + (unsigned)(dc * 16));
        half8_t lo = {0, 0, 0, 0, 0, 0, 0, 0};
        half8_t hi = {0, 0, 0, 0, 0, 0, 0, 0};
        if (qok_ml) {
            lo = qd[0];
            hi = qd[1];
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qf[e] = lo[e];
            qf[8 + e] = hi[e];
        }
        return qf;
    };

#if V_QRES > 0
    half16_t qres[V_QRES];
    if (compute) {
#pragma unroll
        for (int dc = 0; dc < V_QRES; ++dc)
            qres[dc] = loadQ(dc);
    }
#endif
#if V_QRESX > 0
    // Non-replicated resident Q for chunks dc < V_QRESX: this lane's own
    // half8 (half 0: dims dc*16+0..7, half 1: dims dc*16+8..15).
    uint32x4_t qx[V_QRESX];
    if (compute) {
#pragma unroll
        for (int dc = 0; dc < V_QRESX; ++dc) {
            uint32x4_t v = {0u, 0u, 0u, 0u};
            if (qok_ml)
                v = *(const uint32x4_t*)(q16 + qq_base
                                          + (unsigned)(dc * 16 + half * 8));
            qx[dc] = v;
        }
    }
    auto qx_rebuild = [&](const uint32x4_t own4) -> half16_t {
        half16_t qf;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t own = own4[i];
            const uint32_t y = __builtin_amdgcn_permlanex16(
                own, own, 0x76543210u, 0xfedcba98u, false, false);
            fa2_dw_t a, b;
            a.u = half ? y : own;
            b.u = half ? own : y;
            qf[2 * i] = a.h[0];
            qf[2 * i + 1] = a.h[1];
            qf[8 + 2 * i] = b.h[0];
            qf[8 + 2 * i + 1] = b.h[1];
        }
        return qf;
    };
#endif
#if V_ABL_NOQ
    half16_t qsyn;
#pragma unroll
    for (int e = 0; e < 16; ++e)
        qsyn[e] = (_Float16)(float)((qr_ml + e) & 7);
#endif
    // Dynamic-index Q source for the rolled chunk loop.
    auto getQ = [&](int dc) -> half16_t {
#if V_ABL_NOQ
        (void)dc;
        return qsyn;
#else
        return loadQ(dc);
#endif
    };

    auto loadK = [&](int key, int dc) -> half16_t {
        half16_t kf;
#if V_CODES_K
        const uint32x4_t t = *(const uint32x4_t*)(
            Kc + key * 256 + ((dc ^ (key & 15)) & 15) * 16);
        const _Float16 s = Ksc[key * 8 + (dc >> 1)];
        const half2_t s2 = {s, s};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t lo, hi;
            fa2_deq4(t[i], s2, s2, lo, hi);
            fa2_dw_t a, b;
            a.u = lo;
            b.u = hi;
            kf[4 * i] = a.h[0];
            kf[4 * i + 1] = a.h[1];
            kf[4 * i + 2] = b.h[0];
            kf[4 * i + 3] = b.h[1];
        }
#else
        const int rot = fa2_swiz(key & 15);
        const int base = key * 128;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int off = base + ((dc * 8 + i + rot) & 127);
            const int32x2_t pair = *(const int32x2_t*)(Kdw + off);
            fa2_dw_t d0, d1;
            d0.u = (uint32_t)pair[0];
            d1.u = (uint32_t)pair[1];
            kf[2 * i] = d0.h[0];
            kf[2 * i + 1] = d0.h[1];
            kf[2 * i + 2] = d1.h[0];
            kf[2 * i + 3] = d1.h[1];
        }
#endif
        return kf;
    };

    auto loadV = [&](int sub, int dc) -> half16_t {
        half16_t vf;
        const int d = dc * 16 + ml;
#if V_CODES_V
        const int slot = sub ^ ((d >> 3) & 1);
        const uint32x4_t t = *(const uint32x4_t*)(Vc + d * KT + slot * 16);
        // Scales for keys sub*16..+15 of block dc>>1 (wave-uniform address).
        const uint32x4_t s0 = *(const uint32x4_t*)(Vsc + (dc >> 1) * KT + sub * 16);
        const uint32x4_t s1 = *(const uint32x4_t*)(Vsc + (dc >> 1) * KT + sub * 16 + 8);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            fa2_dw_t sa, sb;
            sa.u = (i < 2) ? s0[2 * i] : s1[2 * i - 4];
            sb.u = (i < 2) ? s0[2 * i + 1] : s1[2 * i - 3];
            uint32_t lo, hi;
            fa2_deq4(t[i], sa.h2, sb.h2, lo, hi);
            fa2_dw_t a, b;
            a.u = lo;
            b.u = hi;
            vf[4 * i] = a.h[0];
            vf[4 * i + 1] = a.h[1];
            vf[4 * i + 2] = b.h[0];
            vf[4 * i + 3] = b.h[1];
        }
#else
        const int rot = fa2_swiz(d & 15);
        const int vbase = d * V_ROW;
        const int ps = sub * 8;
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int off = vbase + ((ps + i + rot) & V_MASK);
            const int32x2_t pair = *(const int32x2_t*)(Vdw + off);
            fa2_dw_t d0, d1;
            d0.u = (uint32_t)pair[0];
            d1.u = (uint32_t)pair[1];
            vf[2 * i] = d0.h[0];
            vf[2 * i + 1] = d0.h[1];
            vf[2 * i + 2] = d1.h[0];
            vf[2 * i + 3] = d1.h[1];
        }
#endif
        return vf;
    };

    // ---- Per-WG causal bounds ----
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (tid < 16) {
        const int qr = q_base + tid;
        if (qr < batch_size) {
            const int p = positions[qr];
            wmx = p;
            wmn = p;
        }
    }
    wmx = max(wmx, __shfl_xor(wmx, 16, 32));
    wmx = max(wmx, __shfl_xor(wmx, 8, 32));
    wmx = max(wmx, __shfl_xor(wmx, 4, 32));
    wmx = max(wmx, __shfl_xor(wmx, 2, 32));
    wmx = max(wmx, __shfl_xor(wmx, 1, 32));
    wmn = min(wmn, __shfl_xor(wmn, 16, 32));
    wmn = min(wmn, __shfl_xor(wmn, 8, 32));
    wmn = min(wmn, __shfl_xor(wmn, 4, 32));
    wmn = min(wmn, __shfl_xor(wmn, 2, 32));
    wmn = min(wmn, __shfl_xor(wmn, 1, 32));
    if (tid == 0) {
        LDS[LDS_DWORDS - 2] = (uint32_t)wmx;
        LDS[LDS_DWORDS - 1] = (uint32_t)wmn;
    }
    __syncthreads();
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 2]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 1]);
    const int seq_len = gmax + 1;

    // K raw prefetch (production F6).
    uint16_t pf_ks[KT_KITERS];
    uint32_t pf_kc[KT_KITERS * 8];
#if V_VPF
    // V raw prefetch: per fill iteration t: scales (s0 | s1<<16), 2+2 dwords.
    uint32_t pf_vs[KT_VITERS];
    uint32_t pf_vc[KT_VITERS * 4];
#endif
#if V_QPIPE
    // Q lookahead register (with V_QWRAP it carries chunk DC0 across tiles).
    half16_t qa;
#if V_QWRAP
    if (compute)
        qa = getQ(V_QRES + V_QRESX);
#endif
#endif

    for (int tile = 0; ; ++tile) {
        const int ktile = tile * KT;
        if (ktile >= seq_len)
            break;
        const bool have_pf = (tile > 0);
#if V_ABL_NOFILL
        if (tile == 0) {
#endif

        // ---- K fill ----
#if V_CODES_K
        // Codes: slot (k, b) copies 32 biased code bytes into two swizzled
        // 16-B chunks (2b, 2b+1) of key row k, plus the f16 scale.
        for (int t = 0; t < KT_KITERS; ++t) {
            const int idx = tid + t * WG_THREADS;
            if (K_SLOTS % WG_THREADS != 0 && idx >= K_SLOTS)
                break;
            const int k = idx >> 3;
            const int b = idx & 7;
            uint16_t su;
            uint32_t w8[8];
            if (have_pf) {
                su = pf_ks[t];
#pragma unroll
                for (int w = 0; w < 8; ++w)
                    w8[w] = pf_kc[t * 8 + w];
            } else {
                const int gk = ktile + k;
                const unsigned char* blk = (gk < seq_len)
                    ? k_cache + (unsigned long long)gk * (unsigned long long)row_stride
                        + (unsigned long long)(kv_blk + b) * 34ULL
                    : nullptr;
                su = 0;
                if (blk != nullptr)
                    su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
#pragma unroll
                for (int w = 0; w < 8; ++w) {
                    uint32_t u = 0;
                    if (blk != nullptr)
                        __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                    w8[w] = u;
                }
            }
            const int j0 = 2 * b;
            const int j1 = 2 * b + 1;
            uint32x4_t c0 = {w8[0] ^ 0x80808080u, w8[1] ^ 0x80808080u,
                             w8[2] ^ 0x80808080u, w8[3] ^ 0x80808080u};
            uint32x4_t c1 = {w8[4] ^ 0x80808080u, w8[5] ^ 0x80808080u,
                             w8[6] ^ 0x80808080u, w8[7] ^ 0x80808080u};
            *(uint32x4_t*)(Kc + k * 256 + ((j0 ^ (k & 15)) & 15) * 16) = c0;
            *(uint32x4_t*)(Kc + k * 256 + ((j1 ^ (k & 15)) & 15) * 16) = c1;
            ((uint16_t*)Ksc)[k * 8 + b] = su;
        }
#else
        if (have_pf) {
#pragma unroll 1
            for (int t = 0; t < KT_KITERS; ++t) {
                const int idx = tid + t * WG_THREADS;
                if (K_SLOTS % WG_THREADS != 0 && idx >= K_SLOTS)
                    break;
                const int k = idx >> 3;
                const int b = idx & 7;
                const uint16_t su = pf_ks[t];
                _Float16 s = ((fa2_dw_t){.u = (uint32_t)su}).h[0];
                const float sf = (float)s;
                const int rot = fa2_swiz(k & 15);
                const int base = k * 128;
                const int pb = b * 16;
#pragma unroll 1
                for (int w = 0; w < 8; ++w) {
                    const uint32_t u = pf_kc[t * 8 + w];
                    const float c0 = (float)(int8_t)u;
                    const float c1 = (float)(int8_t)(u >> 8);
                    const float c2 = (float)(int8_t)(u >> 16);
                    const float c3 = (float)(int8_t)(u >> 24);
                    fa2_dw_t d0, d1;
                    d0.h[0] = (_Float16)(sf * c0);
                    d0.h[1] = (_Float16)(sf * c1);
                    d1.h[0] = (_Float16)(sf * c2);
                    d1.h[1] = (_Float16)(sf * c3);
                    Kdw[base + ((pb + w * 2 + rot) & 127)] = d0.u;
                    Kdw[base + ((pb + w * 2 + 1 + rot) & 127)] = d1.u;
                }
            }
        } else
        for (int t = 0; t < KT_KITERS; ++t) {
            const int idx = tid + t * WG_THREADS;
            if (K_SLOTS % WG_THREADS != 0 && idx >= K_SLOTS)
                break;
            const int k = idx >> 3;
            const int b = idx & 7;
            const int gk = ktile + k;
            const unsigned char* blk = (gk < seq_len)
                ? k_cache + (unsigned long long)gk * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t su = 0;
            if (blk != nullptr)
                su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            _Float16 s = ((fa2_dw_t){.u = (uint32_t)su}).h[0];
            const float sf = (float)s;
            const int rot = fa2_swiz(k & 15);
            const int base = k * 128;
            const int pb = b * 16;
#pragma unroll 1
            for (int w = 0; w < 8; ++w) {
                uint32_t u = 0;
                if (blk != nullptr)
                    __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                const float c0 = (float)(int8_t)u;
                const float c1 = (float)(int8_t)(u >> 8);
                const float c2 = (float)(int8_t)(u >> 16);
                const float c3 = (float)(int8_t)(u >> 24);
                fa2_dw_t d0, d1;
                d0.h[0] = (_Float16)(sf * c0);
                d0.h[1] = (_Float16)(sf * c1);
                d1.h[0] = (_Float16)(sf * c2);
                d1.h[1] = (_Float16)(sf * c3);
                Kdw[base + ((pb + w * 2 + rot) & 127)] = d0.u;
                Kdw[base + ((pb + w * 2 + 1 + rot) & 127)] = d1.u;
            }
        }
#endif

        // ---- V fill ----
#if V_CODES_V
        // Transposed codes: thread owns (key quad kq, dim quad dq): loads 4
        // keys' dword for dims 4dq..4dq+3, transposes 4x4 bytes, writes one
        // dword (4 keys) per dim. KT/4 * 64 = 512 slots, 2 per thread.
        for (int t = 0; t < (512 + WG_THREADS - 1) / WG_THREADS; ++t) {
            const int slot = tid + t * WG_THREADS;
            if (512 % WG_THREADS != 0 && slot >= 512)
                break;
            const int kq = slot >> 6;        // 0..7  (keys 4kq..4kq+3)
            const int dq = slot & 63;        // dims 4dq..4dq+3
            const int b = dq >> 3;           // 32-dim block
            const int w = dq & 7;            // dword within block codes
            uint32_t u[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int gk = ktile + 4 * kq + r;
                u[r] = 0;
                if (gk < seq_len) {
                    const unsigned char* blk = v_cache
                        + (unsigned long long)gk * (unsigned long long)row_stride
                        + (unsigned long long)(kv_blk + b) * 34ULL;
                    __builtin_memcpy(&u[r], blk + 2 + w * 4, 4);
                }
                u[r] ^= 0x80808080u;
            }
            // 4x4 byte transpose: out[c] = bytes c of u[0..3].
            const uint32_t x01l = __builtin_amdgcn_perm(u[1], u[0], 0x05010400u); // u0b0 u1b0 u0b1 u1b1
            const uint32_t x01h = __builtin_amdgcn_perm(u[1], u[0], 0x07030602u); // u0b2 u1b2 u0b3 u1b3
            const uint32_t x23l = __builtin_amdgcn_perm(u[3], u[2], 0x05010400u);
            const uint32_t x23h = __builtin_amdgcn_perm(u[3], u[2], 0x07030602u);
            uint32_t o[4];
            o[0] = __builtin_amdgcn_perm(x23l, x01l, 0x05040100u);
            o[1] = __builtin_amdgcn_perm(x23l, x01l, 0x07060302u);
            o[2] = __builtin_amdgcn_perm(x23h, x01h, 0x05040100u);
            o[3] = __builtin_amdgcn_perm(x23h, x01h, 0x07060302u);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int d = 4 * dq + c;
                const int kk = 4 * kq;             // key within tile
                const int sub = kk >> 4;
                const int slotv = sub ^ ((d >> 3) & 1);
                *(uint32_t*)(Vc + d * KT + slotv * 16 + (kk & 15)) = o[c];
            }
        }
        // Scales: 32 keys x 8 blocks.
        for (int sidx = tid; sidx < KT * 8; sidx += WG_THREADS) {
            const int k = sidx >> 3;
            const int b = sidx & 7;
            const int gk = ktile + k;
            uint16_t su = 0;
            if (gk < seq_len) {
                const unsigned char* blk = v_cache
                    + (unsigned long long)gk * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL;
                su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            }
            ((uint16_t*)Vsc)[b * KT + k] = su;
        }
#else
        for (int t = 0; t < KT_VITERS; ++t) {
            const int slot = tid + t * WG_THREADS;
            if (V_SLOTS % WG_THREADS != 0 && slot >= V_SLOTS)
                break;
            const int pair = slot >> 2;
            const int w_base = (slot & 3) * 2;
            const int m = pair >> 3;
            const int b = pair & 7;
#if V_VPF
            uint32_t s01, cw[4];
            if (have_pf) {
                s01 = pf_vs[t];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    cw[i] = pf_vc[t * 4 + i];
            } else {
                const int k0 = ktile + 2 * m;
                const bool ok0 = k0 < seq_len;
                const bool ok1 = k0 + 1 < seq_len;
                const unsigned char* blk0 = v_cache
                    + (unsigned long long)(ok0 ? k0 : 0) * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL;
                const unsigned char* blk1 = v_cache
                    + (unsigned long long)(ok1 ? k0 + 1 : 0) * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL;
                uint32_t a0 = (uint32_t)blk0[0] | ((uint32_t)blk0[1] << 8);
                uint32_t a1 = (uint32_t)blk1[0] | ((uint32_t)blk1[1] << 8);
                s01 = (ok0 ? a0 : 0u) | ((ok1 ? a1 : 0u) << 16);
                uint32_t x0, x1, x2, x3;
                __builtin_memcpy(&x0, blk0 + 2 + w_base * 4, 4);
                __builtin_memcpy(&x1, blk0 + 2 + (w_base + 1) * 4, 4);
                __builtin_memcpy(&x2, blk1 + 2 + w_base * 4, 4);
                __builtin_memcpy(&x3, blk1 + 2 + (w_base + 1) * 4, 4);
                cw[0] = ok0 ? x0 : 0u;
                cw[1] = ok0 ? x1 : 0u;
                cw[2] = ok1 ? x2 : 0u;
                cw[3] = ok1 ? x3 : 0u;
            }
            _Float16 s0 = ((fa2_dw_t){.u = s01 & 0xffffu}).h[0];
            _Float16 s1 = ((fa2_dw_t){.u = s01 >> 16}).h[0];
            const float sf0 = (float)s0;
            const float sf1 = (float)s1;
#pragma unroll
            for (int ww = 0; ww < 2; ++ww) {
                const int w = w_base + ww;
                const uint32_t u0 = cw[ww];
                const uint32_t u1 = cw[2 + ww];
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int d = b * 32 + w * 4 + c;
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    fa2_dw_t dw;
                    dw.h[0] = (_Float16)(sf0 * c0);
                    dw.h[1] = (_Float16)(sf1 * c1);
                    Vdw[d * V_ROW + ((m + fa2_swiz(d & 15)) & V_MASK)] = dw.u;
                }
            }
#else
            const int k0 = ktile + 2 * m;
            const unsigned char* blk0 = (k0 < seq_len)
                ? v_cache + (unsigned long long)k0 * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            const unsigned char* blk1 = (k0 + 1 < seq_len)
                ? v_cache + (unsigned long long)(k0 + 1) * (unsigned long long)row_stride
                    + (unsigned long long)(kv_blk + b) * 34ULL
                : nullptr;
            uint16_t s0u = 0;
            uint16_t s1u = 0;
            if (blk0 != nullptr)
                s0u = (uint16_t)blk0[0] | ((uint16_t)blk0[1] << 8);
            if (blk1 != nullptr)
                s1u = (uint16_t)blk1[0] | ((uint16_t)blk1[1] << 8);
            _Float16 s0 = ((fa2_dw_t){.u = (uint32_t)s0u}).h[0];
            _Float16 s1 = ((fa2_dw_t){.u = (uint32_t)s1u}).h[0];
            const float sf0 = (float)s0;
            const float sf1 = (float)s1;
#pragma unroll 1
            for (int w = w_base; w < w_base + 2; ++w) {
                uint32_t u0 = 0, u1 = 0;
                if (blk0 != nullptr)
                    __builtin_memcpy(&u0, blk0 + 2 + w * 4, 4);
                if (blk1 != nullptr)
                    __builtin_memcpy(&u1, blk1 + 2 + w * 4, 4);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int d = b * 32 + w * 4 + c;
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    fa2_dw_t dw;
                    dw.h[0] = (_Float16)(sf0 * c0);
                    dw.h[1] = (_Float16)(sf1 * c1);
                    Vdw[d * V_ROW + ((m + fa2_swiz(d & 15)) & V_MASK)] = dw.u;
                }
            }
#endif
        }
#endif
#if V_ABL_NOFILL
        }
#endif
        __syncthreads();

        // ---- Prefetch next tile's raw K (and V) bytes ----
        {
            const int ktile_next = ktile + KT;
            if (ktile_next < seq_len) {
#pragma unroll 1
                for (int t = 0; t < KT_KITERS; ++t) {
                    const int idx = tid + t * WG_THREADS;
                    if (K_SLOTS % WG_THREADS != 0 && idx >= K_SLOTS)
                        break;
                    const int k = idx >> 3;
                    const int b = idx & 7;
                    const int gk = ktile_next + k;
                    const unsigned char* blk = (gk < seq_len)
                        ? k_cache + (unsigned long long)gk * (unsigned long long)row_stride
                            + (unsigned long long)(kv_blk + b) * 34ULL
                        : nullptr;
                    uint16_t su = 0;
                    if (blk != nullptr)
                        su = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
                    pf_ks[t] = su;
#pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        uint32_t u = 0;
                        if (blk != nullptr)
                            __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                        pf_kc[t * 8 + w] = u;
                    }
                }
#if V_VPF
#pragma unroll
                for (int t = 0; t < KT_VITERS; ++t) {
                    const int slot = tid + t * WG_THREADS;
                    if (V_SLOTS % WG_THREADS != 0 && slot >= V_SLOTS)
                        break;
                    const int pair = slot >> 2;
                    const int w_base = (slot & 3) * 2;
                    const int m = pair >> 3;
                    const int b = pair & 7;
                    const int k0 = ktile_next + 2 * m;
                    const bool ok0 = k0 < seq_len;
                    const bool ok1 = k0 + 1 < seq_len;
                    const unsigned char* blk0 = v_cache
                        + (unsigned long long)(ok0 ? k0 : 0) * (unsigned long long)row_stride
                        + (unsigned long long)(kv_blk + b) * 34ULL;
                    const unsigned char* blk1 = v_cache
                        + (unsigned long long)(ok1 ? k0 + 1 : 0) * (unsigned long long)row_stride
                        + (unsigned long long)(kv_blk + b) * 34ULL;
                    uint32_t a0 = (uint32_t)blk0[0] | ((uint32_t)blk0[1] << 8);
                    uint32_t a1 = (uint32_t)blk1[0] | ((uint32_t)blk1[1] << 8);
                    pf_vs[t] = (ok0 ? a0 : 0u) | ((ok1 ? a1 : 0u) << 16);
                    uint32_t x0, x1, x2, x3;
                    __builtin_memcpy(&x0, blk0 + 2 + w_base * 4, 4);
                    __builtin_memcpy(&x1, blk0 + 2 + (w_base + 1) * 4, 4);
                    __builtin_memcpy(&x2, blk1 + 2 + w_base * 4, 4);
                    __builtin_memcpy(&x3, blk1 + 2 + (w_base + 1) * 4, 4);
                    pf_vc[t * 4 + 0] = ok0 ? x0 : 0u;
                    pf_vc[t * 4 + 1] = ok0 ? x1 : 0u;
                    pf_vc[t * 4 + 2] = ok1 ? x2 : 0u;
                    pf_vc[t * 4 + 3] = ok1 ? x3 : 0u;
                }
#endif
            }
        }

        // ---- QK / softmax / PV ----
        if (compute) {
            float8_t sacc0 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc1 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const bool do0 = ktile <= gmax;
            const bool do1 = ktile + 16 <= gmax;
            if (do0 || do1) {
                auto qk_step = [&](const half16_t qf, int dc) {
                    if (do0) {
                        const half16_t kf = loadK(ml, dc);
                        sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf, sacc0);
                    }
                    if (do1) {
                        const half16_t kf = loadK(16 + ml, dc);
                        sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf, qf, sacc1);
                    }
                };
#if V_QRES > 0
#pragma unroll
                for (int dc = 0; dc < V_QRES; ++dc) {
                    qk_step(qres[dc], dc);
                    asm volatile("" ::: "memory");
                }
#endif
#if V_QRESX > 0
#pragma unroll
                for (int dc = 0; dc < V_QRESX; ++dc) {
                    qk_step(qx_rebuild(qx[dc]), dc);
                    asm volatile("" ::: "memory");
                }
#endif
                constexpr int DC0 = V_QRES + V_QRESX;
#if V_QPIPE
#if !V_QWRAP
                qa = getQ(DC0);
#endif
#if V_QPIPE >= 2
                half16_t qb = getQ(DC0 + 1 < 16 ? DC0 + 1 : 15);
#endif
                V_UNROLL_N(V_QK_UNROLL)
                for (int dc = DC0; dc < 16; ++dc) {
                    const half16_t qf = qa;
#if V_QPIPE >= 2
                    qa = qb;
                    if (dc + 2 < 16)
                        qb = getQ(dc + 2);
#else
                    if (dc + 1 < 16)
                        qa = getQ(dc + 1);
#endif
#if V_QWRAP
                    // Next tile's first chunk (Q is tile-invariant).
                    if (dc == 15)
                        qa = getQ(DC0);
#endif
                    qk_step(qf, dc);
                    asm volatile("" ::: "memory");
                }
#else
                V_UNROLL_N(V_QK_UNROLL)
                for (int dc = DC0; dc < 16; ++dc) {
                    qk_step(getQ(dc), dc);
                    asm volatile("" ::: "memory");
                }
#endif

#pragma unroll 1
                for (int sub = 0; sub < 2; ++sub) {
                    const bool dos = (sub == 0) ? do0 : do1;
                    if (!dos)
                        continue;
                    float8_t sacc = (sub == 0) ? sacc0 : sacc1;
                    const int kt_sub = ktile + sub * 16;
                    const bool full = (kt_sub + 15) <= gmin;
                    const int pos_ml =
                        (!full && qok_ml) ? positions[qr_ml] : 0;
#if V_VPIPE
                    half16_t vn = loadV(sub, 0);
#endif
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int key_g = kt_sub + 2 * j + half;
                        const bool ok = qok_ml && (full || key_g <= pos_ml);
                        const float x = ok ? sacc[j] * scale_attn : -INFINITY;
                        sacc[j] = x;
                    }
                    float mx = -INFINITY;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        mx = fmaxf(mx, sacc[j]);
                    mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
                    const float m_new = fmaxf(m_old, mx);
                    const float alpha =
                        (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                    m_old = m_new;
                    float earr[8];
                    float sum = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float e = (m_new == -INFINITY)
                            ? 0.0f
                            : __expf(sacc[j] - m_new);
                        earr[j] = e;
                        sum += e;
                    }
                    sum += __shfl_xor(sum, 16, 32);
                    l_val = l_val * alpha + sum;
                    half16_t prow;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float peer = __shfl_xor(earr[j], 16, 32);
                        const float e0 = (half == 0) ? earr[j] : peer;
                        const float e1 = (half == 0) ? peer : earr[j];
                        prow[2 * j] = (_Float16)e0;
                        prow[2 * j + 1] = (_Float16)e1;
                    }
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        Ofr[dc] *= alpha;
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc) {
#if V_VPIPE
                        const half16_t vf = vn;
                        if (dc < 15)
                            vn = loadV(sub, dc + 1);
                        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            vf, prow, Ofr[dc]);
#else
                        const half16_t vf = loadV(sub, dc);
                        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            vf, prow, Ofr[dc]);
                        asm volatile("" ::: "memory");
#endif
                    }
                }
            }
        }
        __syncthreads();
    }

    if (compute) {
        if (qok_ml) {
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            const unsigned bo =
                (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
            for (int dc = 0; dc < 16; ++dc) {
                const unsigned bd = bo + (unsigned)(dc * 16 + half);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    out[bd + (unsigned)(j * 2)] = Ofr[dc][j] * inv;
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(V_WG_THREADS, 1) void attention_q8_0_fa2_gqa_gfx11(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 16;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body(q16, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}

// Harness metadata: dynamic LDS bytes this variant needs.
extern "C" __global__ void fa2_var_lds_bytes(int* o)
{
    if (o) {
        o[0] = LDS_B;
        o[1] = V_WG_THREADS;
    }
}
