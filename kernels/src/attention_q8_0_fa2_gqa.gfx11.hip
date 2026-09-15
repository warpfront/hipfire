// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx11-only GQA-fused FA2 prefill attention (research opt-in).
//
// One workgroup owns one KV head and eight token positions: 48 live rows
// (8 positions x 6 query heads, H=24/Hkv=4/D=256) packed into three 16-row
// WMMA tiles, plus one loader/helper wave. K and V are each dequantized
// exactly once per KT-key tile into two swizzled f16 LDS planes (KT*512 B
// each), then reused by all six query heads. Scores, online-softmax state
// (m, l), and f32 O remain in VGPRs; dynamic LDS is exactly KT*1024 B
// (65,536 for KT64, 32,768 for KT32).
//
// Grid:  [ceil(batch/8), 4, 1] direct (x=batch tiles, y=KV head).
// Block: 128 threads (four wave32s; waves 0..2 compute, wave 3 helper).
// Direct only: no PARTIAL/split-KV records, no merge kernel.
//
// Row mapping (compute wave w=0..2, packed index t=0..15):
//   head_local = 2*w + (t >> 3);  q_local = t & 7;
//   h = 6*kv_h + head_local;      q_row = 8*blockIdx.x + q_local.
//
// gfx11 wave32 WMMA mapping (measured on gfx1100 with a real probe —
// lanes r and r+16 supplied identical half16 rows, exact-valued encodings
// reproduced the C map twice):
//   The builtin consumes a FULL half16 A row and a FULL half16 B row from
//   lane r=lane&15; lanes r and r+16 MUST supply identical rows.
//   Physical accumulator: acc[j] = C[2*j + (lane>>4)][lane&15] with
//   C[i][n] = sum_k A[i][k]*B[n][k].
//   QK computes S^T = K*Q^T: A = K fragment (row ml = key sub*16+ml,
//     supplied identically by lanes ml and ml+16), B = Q fragment (row ml
//     = packed query ml, identical on both halves — there is no gfx12
//     kg*8 operand offset). sacc[j] in lane (ml=lane&15, half=lane>>4) is
//     S[query=ml][key=sub*16+2*j+half]: half 0 owns the eight even keys,
//     half 1 the eight odd keys. A lane PAIR owns one complete query row;
//     the row max and the probability sum are therefore xor-16 pair
//     reductions so both members hold identical m/l/alpha.
//   P reconstruction: each half holds eight f32 probabilities for its own
//     key parity; each lane exchanges every element with lane^16 and
//     interleaves prow[2*j]=half-0 value, prow[2*j+1]=half-1 value, so
//     both halves hold an IDENTICAL full half16 P row in key order 0..15.
//     No transpose buffer exists.
//   PV computes O^T = V^T*P^T: A = V^T fragment (row ml = dim dc*16+ml of
//     the dc tile, full 16-key row supplied identically by both halves),
//     B = the replicated prow. Ofr[dc][j] = O[query=ml][dim=dc*16+2*j+half]:
//     even dims in half 0, odd dims in half 1. Rescale/normalization use
//     the pair-local (identical) alpha/l scalars; the epilogue stores the
//     exact interleaved dimension out[query,head,dc*16+2*j+half], so every
//     output element has exactly one writer and no row shuffle is needed.
//
// LDS swizzle (f16 pairs as dwords):
//   default S(c) = 8*(c>>2)+(c&3);
//   HIPFIRE_FA2_GFX11_SWIZZLE_EVEN: S(c) = 2*bit_reverse4(c) so every
//   fragment start is 8-byte aligned and the 16 unique ml rows cover all
//   even/odd bank pairs once (broadcast on lanes ml and ml+16).
// K store/read: dword pair p for key k at K[k][(p + S(k&15)) & 127].
// V store/read: key-pair p for dim d at V_T[d][(p + S(d&15)) & (KT/2-1)].
// Even-swizzle path loads four aligned b64s per half16; default path keeps
// eight scalar dword reads. Unpack order into half16 is identical either way.
//
// Q f16 rounding matches the incumbent (Q/K/V/P operands f16,
// accumulators/state/O f32). No slots, no windows, no FP8, no other arch.

#include <hip/hip_runtime.h>

// K-cache mode, selected at JIT time (kernels.rs concat define):
//   0 = standard Q8 K (34 B blocks, f16 scale + 32 int8 codes).
//   3 = fwht3 K (100 B/head: f32 cnorm + 96 B of 3-bit codes, K stored
//       FWHT-rotated; V stays Q8_0). KMODE=3 additionally rotates this WG's
//       Q rows in place (signed FWHT-256, fwht_shfl_forward_256 from
//       turbo_common.h — prepended by the launcher, NOT #included here: the
//       runtime compile has no -I to kernels/src) before the main body, and
//       dequantizes fwht3 K into the K plane. Everything downstream of the K
//       plane (layout, swizzle, QK/PV, softmax, epilogue) is untouched.
#ifndef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 0
#endif

// Key-tile width, selected at JIT time (the host launcher pins the
// production variant with an explicit define; the file default matches it).
// Only 32 and 64 exist: the cooperative fill counts, the V row
// stride/mask, the mailbox address, and the subtile count below all
// assume KT/16 subtiles of 16 keys and KT*8/KT*4 fill slots. KT32
// (32,768 B dynamic LDS, two resident workgroups per CU) is the measured
// production choice on both gfx1100 and gfx1151; KT64 survives only for
// controlled comparison.
#ifndef HIPFIRE_FA2_KT
#define HIPFIRE_FA2_KT 32
#endif
#if HIPFIRE_FA2_KT != 64 && HIPFIRE_FA2_KT != 32
#error "HIPFIRE_FA2_KT must be 32 or 64"
#endif

// B1/B2 gate defines (JIT-prepended). Default off retains the incumbent path.
#ifndef HIPFIRE_FA2_GFX11_SWIZZLE_EVEN
#define HIPFIRE_FA2_GFX11_SWIZZLE_EVEN 0
#endif
#ifndef HIPFIRE_FA2_GFX11_NO_CLOBBER
#define HIPFIRE_FA2_GFX11_NO_CLOBBER 0
#endif
// Entry-symbol rename so variant modules cannot alias the functions cache.
#ifndef HIPFIRE_FA2_GFX11_KERNEL
#define HIPFIRE_FA2_GFX11_KERNEL attention_q8_0_fa2_gqa_gfx11
#endif
#ifndef HIPFIRE_FA2_GFX11_FWHT3K_KERNEL
#define HIPFIRE_FA2_GFX11_FWHT3K_KERNEL attention_q8_0_fa2_gqa_fwht3k_gfx11
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef int32_t __attribute__((ext_vector_type(2))) int32x2_t;


typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;

__device__ __forceinline__ int fa2_swiz(int c)
{
#if HIPFIRE_FA2_GFX11_SWIZZLE_EVEN
    // 2*bit_reverse4(c): lane starts 0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30.
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
#else
    return 8 * (c >> 2) + (c & 3);
#endif
}

// Shared workgroup body (direct only: writes normalized O to `out`).
// `kv_h` selects the KV head (blockIdx.y); `q_base` the 8-position tile
// (blockIdx.x). Causal bounds come from positions[] (mailbox-reduced to
// gmin/gmax); the tile loop clamps to the derived seq_len.
__device__ __forceinline__ void fa2_gqa_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base)
{
    constexpr int KT = HIPFIRE_FA2_KT;
    constexpr int KT_SUBS = KT / 16;
    constexpr int KT_KITERS = KT / 16;
    constexpr int KT_VITERS = KT / 32;
    constexpr int V_ROW = KT / 2;
    constexpr int V_MASK = KT / 2 - 1;
    constexpr int LDS_DWORDS = KT * 256;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const bool compute = wave < 3;

    const int q_stride = 24 * 256;
    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + KT * 128;

    // ---- Transition 1: online state init (compute waves) ----
    // Q is NOT kept persistent (the tile loop cannot spare the registers).
    // Instead each QK chunk reloads its B-fragment as four float4 loads
    // from L1-resident Q (the per-wave Q tile is read once per tile into
    // L1 and re-hit per subtile/chunk; L1 bandwidth makes the re-read
    // free). Staging is one float4 at a time so only one live half16
    // K/Q fragment pair exists per WMMA.
    typedef float __attribute__((ext_vector_type(4))) float4_t;
    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    if (compute) {
        h_ml = 6 * kv_h + (wave << 1) + (ml >> 3);
        qr_ml = q_base + (ml & 7);
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // ---- Per-WG causal bounds (positions[] is authoritative) ----
    // Lanes 0..7 read the tile's eight positions; wave 0 reduces; lane 0
    // publishes via two transient LDS mailbox dwords (overwritten by the
    // first V fill, so the KT*1024 B plane budget is unaffected).
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (tid < 8) {
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
    // Uniformize through lane 0: every lane loaded the same mailbox
    // dwords, but that uniformity is invisible through LDS.
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 2]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 1]);
    const int seq_len = gmax + 1;

    // ---- Transitions 2-4: KT tiles ----
    for (int tile = 0; ; ++tile) {
        const int ktile = tile * KT;
        if (ktile >= seq_len)
            break;

        // ---- Transition 2: cooperative KT fill (all four waves) ----
#if HIPFIRE_FA2_KMODE == 3
        // fwht3 K: per (position, kv head) record = [f32 cnorm][96 B codes]
        // (k_bytes_per_head = 100, k_bytes_per_pos = 4*100 = 400, head offset
        // kv_h*100 — the same contract as attention_flash_fwht3_tile_batched
        // and kv_cache_write_asym_k_fwht3*). KT*8 slots (k*8+b, like the Q8
        // path); slot (k, b) owns lanes b*4..b*4+3 (lane l owns dims l*8..+8
        // via 3 bytes at +4+l*3, code (packed >> 3*i) & 7, value cnorm *
        // TURBO_C3_256[code]) and dequantizes 32 dims = 16 dwords straight
        // into the swizzled dword plane (pair p for key k at
        // K[k][(p + S(k&15)) & 127] — identical layout downstream, so block
        // b covers pairs b*16..b*16+15 exactly as in the Q8 path). V fill
        // below is unchanged (Q8_0).
        for (int t = 0; t < KT_KITERS; ++t) {
            const int s = tid + t * 128;
            const int k = s >> 3;
            const int b = s & 7;
            const int gk = ktile + k;
            const unsigned char* rec = (gk < seq_len)
                ? k_cache + (unsigned long long)gk * 400ULL
                    + (unsigned long long)(kv_h * 100)
                : nullptr;
            float cnorm = 0.0f;
            if (rec != nullptr)
                __builtin_memcpy(&cnorm, rec, 4);
            const int rot = fa2_swiz(k & 15);
            const int base = k * 128;
#pragma unroll 1
            for (int m = 0; m < 4; ++m) {
                uint32_t packed = 0;
                if (rec != nullptr)
                    __builtin_memcpy(&packed, rec + 4 + (b * 4 + m) * 3, 3);
                const int p0 = (b * 4 + m) * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int code0 = (packed >> (6 * i)) & 7;
                    const int code1 = (packed >> (6 * i + 3)) & 7;
                    fa2_dw_t dw;
                    dw.h[0] = (_Float16)(cnorm * TURBO_C3_256[code0]);
                    dw.h[1] = (_Float16)(cnorm * TURBO_C3_256[code1]);
                    Kdw[base + ((p0 + i + rot) & 127)] = dw.u;
                }
            }
        }
#else
        // K: KT*8 blocks (k*8+b), one block per slot. Codes are loaded on
        // demand per pair (no staging array) and dequantized straight
        // into the swizzled dword plane.
        for (int t = 0; t < KT_KITERS; ++t) {
            const int idx = tid + t * 128;
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
            // 8 dwords cover the 32 codes (blk+2 may be 2-mod-4;
            // memcpy emits a correct unaligned load either way).
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
        // V: KT*2 key-pairs (m*8+b); each slot reads the even and odd block
        // and packs (V[2m][d], V[2m+1][d]) per output dim. Codes are loaded
        // on demand per dim (no staging array).
        for (int t = 0; t < KT_VITERS; ++t) {
            const int pair = tid + t * 128;
            const int m = pair >> 3;
            const int b = pair & 7;
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
            for (int w = 0; w < 8; ++w) {
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
        }
        __syncthreads();

        // ---- Transition 3: KT/16 QK/softmax/PV subtiles of 16 keys ----
        if (compute) {
#pragma unroll 1
            for (int sub = 0; sub < KT_SUBS; ++sub) {
                const int kt_sub = ktile + sub * 16;
                if (kt_sub > gmax)
                    continue;
                const bool full = (kt_sub + 15) <= gmin;
                // This lane pair's softmax query is ml; its position is
                // loaded once per subtile (masked path only).
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                // QK as S^T = K*Q^T: A = full K row for key sub*16+ml
                // (lanes ml and ml+16 intentionally repeat the same
                // addresses), B = full Q row for query ml reloaded per
                // chunk. sacc[j] = S[query=ml][key=sub*16+2*j+half].
                float8_t sacc =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                {
                    const int key = sub * 16 + ml;
                    const int rot = fa2_swiz(key & 15);
                    const int base = key * 128;
                    for (int dc = 0; dc < 16; ++dc) {
                        half16_t kf;
#if HIPFIRE_FA2_GFX11_SWIZZLE_EVEN
                        // Four aligned b64 LDS loads at i={0,2,4,6}. Even
                        // swizzle makes each start 8-byte aligned; consecutive
                        // dwords i,i+1 never wrap (start never 127). Unpack
                        // low then high dword into the same half16 order as
                        // the eight scalar loads below — f16 element order
                        // and WMMA A operands are unchanged.
#pragma unroll
                        for (int i = 0; i < 8; i += 2) {
                            const int off = base + ((dc * 8 + i + rot) & 127);
                            const int32x2_t pair =
                                *(const int32x2_t*)(Kdw + off);
                            fa2_dw_t d0, d1;
                            d0.u = (uint32_t)pair[0];
                            d1.u = (uint32_t)pair[1];
                            kf[2 * i] = d0.h[0];
                            kf[2 * i + 1] = d0.h[1];
                            kf[2 * i + 2] = d1.h[0];
                            kf[2 * i + 3] = d1.h[1];
                        }
#else
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            fa2_dw_t dw;
                            dw.u = Kdw[base + ((dc * 8 + i + rot) & 127)];
                            kf[2 * i] = dw.h[0];
                            kf[2 * i + 1] = dw.h[1];
                        }
#endif
                        half16_t qf;
                        {
                            const float* qd = q + qq_base + (unsigned)(dc * 16);
#pragma unroll
                            for (int g = 0; g < 4; ++g) {
                                float4_t v = {0.f, 0.f, 0.f, 0.f};
                                if (qok_ml)
                                    v = *(const float4_t*)(qd + g * 4);
                                qf[4 * g + 0] = (_Float16)v[0];
                                qf[4 * g + 1] = (_Float16)v[1];
                                qf[4 * g + 2] = (_Float16)v[2];
                                qf[4 * g + 3] = (_Float16)v[3];
                            }
                        }
                        sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            kf, qf, sacc);
#if !HIPFIRE_FA2_GFX11_NO_CLOBBER
                        // Bound K/Q-load hoisting to this chunk.
                        asm volatile("" ::: "memory");
#endif
                    }
                }

                // Mask + row max. The mx chain serializes the unrolled
                // rows; the other key parity sits in lane ml^16 and is
                // combined with one shuffle, so both pair members share
                // the same m_new and alpha.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int key_g = kt_sub + 2 * j + half;
                    const bool ok = qok_ml && (full || key_g <= pos_ml);
                    const float x = ok ? sacc[j] * scale_attn : -INFINITY;
                    // NOTE: max-accumulation happens in the second pass
                    // below; this pass only masks into sacc.
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

                // Probabilities, then full-P reconstruction: each half
                // exchanges its eight parity probabilities with lane^16
                // and interleaves them, so both halves hold the identical
                // half16 P row in key order 0..15. The sum chain likewise
                // serializes rows before the single pair shuffle.
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

                // O rescale with the pair-local alpha: Ofr rows are the
                // pair's own softmax query (O[query=ml][dim]).
#pragma unroll
                for (int dc = 0; dc < 16; ++dc)
                    Ofr[dc] *= alpha;

                // PV as O^T = V^T*P^T: A = V^T full row for dim dc*16+ml
                // over all 16 subtile keys (partner half repeats the same
                // full row), B = prow. Ofr[dc][j] =
                // O[query=ml][dim=dc*16+2*j+half].
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const int d = dc * 16 + ml;
                    const int rot = fa2_swiz(d & 15);
                    const int vbase = d * V_ROW;
                    const int ps = sub * 8;
                    half16_t vf;
#if HIPFIRE_FA2_GFX11_SWIZZLE_EVEN
                    // Four aligned b64 loads at i={0,2,4,6}. Within either
                    // V-row parity the even residues cover all even b64
                    // starts once; unpack order matches the scalar path.
#pragma unroll
                    for (int i = 0; i < 8; i += 2) {
                        const int off = vbase + ((ps + i + rot) & V_MASK);
                        const int32x2_t pair =
                            *(const int32x2_t*)(Vdw + off);
                        fa2_dw_t d0, d1;
                        d0.u = (uint32_t)pair[0];
                        d1.u = (uint32_t)pair[1];
                        vf[2 * i] = d0.h[0];
                        vf[2 * i + 1] = d0.h[1];
                        vf[2 * i + 2] = d1.h[0];
                        vf[2 * i + 3] = d1.h[1];
                    }
#else
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        fa2_dw_t dw;
                        dw.u = Vdw[vbase + ((ps + i + rot) & V_MASK)];
                        vf[2 * i] = dw.h[0];
                        vf[2 * i + 1] = dw.h[1];
                    }
#endif
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        vf, prow, Ofr[dc]);
#if !HIPFIRE_FA2_GFX11_NO_CLOBBER
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
#endif
                }
            }
        }
        // Tile turnover: no wave may overwrite LDS before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Transition 4: completion (direct epilogue) ----
    // Ofr[dc][j] at lane (ml,half) = O[query=ml][dim=dc*16+2*j+half]:
    // the pair covers every output dimension exactly once (even/odd
    // split), and the pair-local l_val normalizes. Each valid
    // out[query,head,dim] has exactly one writer.
    if (compute) {
        if (qok_ml) {
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            // 32-bit row base (whole O tensor < 10 MiB).
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

extern "C" __global__ __launch_bounds__(128, 1) void HIPFIRE_FA2_GFX11_KERNEL(
    const float* __restrict__ q,
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
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body(q, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}

#if HIPFIRE_FA2_KMODE == 3
// ---- KMODE=3 prologue: in-place signed-FWHT-256 of this WG's Q rows ----
// K is stored FWHT-rotated, so Q must be rotated before QK (orthogonal
// transform: dot products preserved; fwht_shfl_forward_256 is normalized
// 1/sqrt(256) = 1/16 and scale_attn applies downstream exactly as in the
// Q8 path — no scale change). One wave32 per row: lane `lane` owns dims
// lane*8..+8, register-only + ds_swizzle, no LDS, one __syncthreads at the
// end. This WG owns 8 positions x 6 heads = 48 rows; 4 waves cover 4 rows
// per step x 12 steps. Rows/heads are disjoint across (blockIdx.x,
// blockIdx.y), so the in-place update is race-free; tail rows
// (q_base + r >= batch_size) are never read downstream and are skipped.
// NOT replay-idempotent (a second launch would rotate twice): the launcher
// rejects capture/replay and dispatch gates tree-verify off, so the
// single-use prefill Q buffer is consumed exactly once.
__device__ __forceinline__ void fa2_gqa_rotate_q_256(
    float* __restrict__ q,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int kv_h,
    int q_base,
    int batch_size)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    for (int step = 0; step < 12; ++step) {
        const int row = step * 4 + wave;
        const int qr = q_base + row / 6;
        // Uniform across the wave (depends only on wave/step): every lane
        // of the wave takes the same branch, so the ds_swizzle group stays
        // intact.
        if (qr < batch_size) {
            float* rp = q + (unsigned)qr * 6144u
                + (unsigned)(6 * kv_h + (row % 6)) * 256u;
            float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
            float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
            float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
            float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
            fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7,
                                  signs1, signs2, lane);
            rp[lane * 8 + 0] = v0; rp[lane * 8 + 1] = v1;
            rp[lane * 8 + 2] = v2; rp[lane * 8 + 3] = v3;
            rp[lane * 8 + 4] = v4; rp[lane * 8 + 5] = v5;
            rp[lane * 8 + 6] = v6; rp[lane * 8 + 7] = v7;
        }
    }
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(128, 1) void HIPFIRE_FA2_GFX11_FWHT3K_KERNEL(
    float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
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
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_rotate_q_256(q, signs1, signs2, kv_h, q_base, batch_size);
    fa2_gqa_body(q, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}
#endif
