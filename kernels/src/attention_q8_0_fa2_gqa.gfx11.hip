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
// LDS swizzle (f16 pairs as dwords): S(c) = 2*bit_reverse4(c) so every
// fragment start is 8-byte aligned and the 16 unique ml rows cover all
// even/odd bank pairs once (broadcast on lanes ml and ml+16). Admitted B1
// on Halo gfx1151: pp8192 FA2 -4..-5% vs the old 8*(c>>2)+(c&3) path.
// K store/read: dword pair p for key k at K[k][(p + S(k&15)) & 127].
// V store/read: key-pair p for dim d at V_T[d][(p + S(d&15)) & (KT/2-1)].
// Fragment loads use four aligned b64s per half16 (unpack order unchanged).
//
// Q f16 rounding matches the incumbent (Q/K/V/P operands f16,
// accumulators/state/O f32). No slots, no windows, no FP8, no other arch.

#include <hip/hip_runtime.h>

// K-cache mode, selected at JIT time (kernels.rs concat define):
//   0 = standard Q8 K (34 B blocks, f16 scale + 32 int8 codes).
//   3 = fwht3 K (100 B/head: f32 cnorm + 96 B of 3-bit codes, K stored
//       FWHT-rotated; V stays Q8_0). KMODE=3 additionally dequantizes fwht3
//       K into the K plane; the Q-side signed-FWHT-256 rotation lives in the
//       pre-convert kernel below (fwht_shfl_forward_256 from turbo_common.h
//       — prepended by the launcher, NOT #included here: the runtime compile
//       has no -I to kernels/src), applied to the f32 row BEFORE the cast.
//       Everything downstream of the K plane (layout, swizzle, QK/PV,
//       softmax, epilogue) is untouched.
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
    // 2*bit_reverse4(c): lane starts 0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30.
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

// Shared workgroup body (direct only: writes normalized O to `out`).
// `kv_h` selects the KV head (blockIdx.y); `q_base` the 8-position tile
// (blockIdx.x). Q arrives f16 pre-converted row-major [batch, 24, 256]
// (see attention_fa2_q_preconvert_gfx11); the body never touches f32 Q.
// Causal bounds come from positions[] (mailbox-reduced to gmin/gmax);
// the tile loop clamps to the derived seq_len.
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
    // Q is not kept across tiles (register pressure). Q arrives f16
    // pre-converted (see attention_fa2_q_preconvert_gfx11); within a KT32
    // tile, each dim-chunk Q fragment is loaded once (one aligned 16-byte
    // half8 load per 8 dims, no cvt) and reused for both 16-key subtles'
    // QK WMMAs (qshare2); softmax/PV still run sub0 then sub1 so online
    // order and PV accumulation stay unchanged.
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

#if HIPFIRE_FA2_KT == 32
    // ---- F6: software prefetch of the next tile's raw K bytes ----
    // Per-lane raw K share at KT32/block128 (all four waves fill):
    //   Q8 K: KT*8 slots / 128 lanes = 2 slots/lane x 34 B (f16 scale +
    //         32 i8 codes) = 68 B = 1 dword (both f16 scales) + 16 dwords.
    //   fwht3 K: 2 slots/lane x (f32 cnorm + 4x3 B packed codes) = 32 B =
    //         2 cnorm + 8 packed dwords.
    // (Full K+V prefetch was measured first: 256 VGPR + spill on both
    // arches/both modes — over budget. K-only keeps the V fill on global
    // loads exactly as before and overlaps the K half.)
    // Staged raw (undequantized) right after the post-fill barrier so the
    // K global loads overlap this tile's QK/softmax/PV; the next fill slot
    // dequantizes K from here with no fresh K global loads. The small
    // staging arrays scalarize to VGPRs (measured: no scratch); the Q8
    // f16 scales share one dword (slot t in half t).
#if HIPFIRE_FA2_KMODE == 3
    float pf_kcn[KT_KITERS];
    uint32_t pf_kp[KT_KITERS * 4];
#else
    uint32_t pf_ks;
    uint32_t pf_kc[KT_KITERS * 8];
#endif
#endif
    // ---- Transitions 2-4: KT tiles ----
    for (int tile = 0; ; ++tile) {
        const int ktile = tile * KT;
        if (ktile >= seq_len)
            break;
#if HIPFIRE_FA2_KT == 32
        // F6: tiles after the first consume the raw bytes prefetched into
        // registers during the previous tile's compute phase (see below);
        // tile 0 fills from global exactly as before. Uniform (tile), so no
        // divergence. tile > 0 implies the prefetch ran: the previous
        // iteration saw ktile_prev + KT < seq_len, else this iteration
        // would have broken out above.
        const bool have_pf = (tile > 0);
#else
        constexpr bool have_pf = false;
#endif

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
        if (have_pf) {
#if HIPFIRE_FA2_KT == 32
            // F6: decode the prefetched raw K bytes (no global loads here).
            // Dequant/store lines are identical to the global path below;
            // the prefetch staged zeros for OOB slots, so tails zero-fill
            // bit-identically.
#pragma unroll 1
            for (int t = 0; t < KT_KITERS; ++t) {
                const int s = tid + t * 128;
                const int k = s >> 3;
                const int b = s & 7;
                const float cnorm = pf_kcn[t];
                const int rot = fa2_swiz(k & 15);
                const int base = k * 128;
#pragma unroll 1
                for (int m = 0; m < 4; ++m) {
                    const uint32_t packed = pf_kp[t * 4 + m];
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
#endif
        } else
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
        if (have_pf) {
#if HIPFIRE_FA2_KT == 32
            // F6: decode the prefetched raw K bytes (no global loads here).
            // Dequant/store lines identical to the global path below.
#pragma unroll 1
            for (int t = 0; t < KT_KITERS; ++t) {
                const int idx = tid + t * 128;
                const int k = idx >> 3;
                const int b = idx & 7;
                const uint16_t su = (uint16_t)(pf_ks >> (t * 16));
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
#endif
        } else
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
#if HIPFIRE_FA2_KT == 32
        // ---- F6: prefetch the next tile's raw K bytes (all four waves) ----
        // Issued immediately after the post-fill barrier so these global
        // loads overlap the current tile's QK/softmax/PV below; the next
        // iteration's K fill (above) dequantizes from these registers with
        // no fresh K global loads (V still streams in its fill slot).
        // Addressing and guards mirror the K fill loads exactly (no
        // out-of-range reads); OOB slots stage zeros so tails zero-fill
        // bit-identically to today. The QK/PV loops' existing compiler
        // barriers already fence these loads above the QK loop, so no
        // sched_barrier (and never an asm memory clobber) is used.
        {
            const int ktile_next = ktile + KT;
            if (ktile_next < seq_len) {
#if HIPFIRE_FA2_KMODE == 3
#pragma unroll 1
                for (int t = 0; t < KT_KITERS; ++t) {
                    const int s = tid + t * 128;
                    const int k = s >> 3;
                    const int b = s & 7;
                    const int gk = ktile_next + k;
                    const unsigned char* rec = (gk < seq_len)
                        ? k_cache + (unsigned long long)gk * 400ULL
                            + (unsigned long long)(kv_h * 100)
                        : nullptr;
                    float cnorm = 0.0f;
                    if (rec != nullptr)
                        __builtin_memcpy(&cnorm, rec, 4);
                    pf_kcn[t] = cnorm;
#pragma unroll 1
                    for (int m = 0; m < 4; ++m) {
                        uint32_t packed = 0;
                        if (rec != nullptr)
                            __builtin_memcpy(&packed, rec + 4 + (b * 4 + m) * 3, 3);
                        pf_kp[t * 4 + m] = packed;
                    }
                }
#else
                pf_ks = 0;
#pragma unroll 1
                for (int t = 0; t < KT_KITERS; ++t) {
                    const int idx = tid + t * 128;
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
                    pf_ks |= (uint32_t)su << (t * 16);
#pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        uint32_t u = 0;
                        if (blk != nullptr)
                            __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                        pf_kc[t * 8 + w] = u;
                    }
                }
#endif
            }
        }
#endif

        // ---- Transition 3: KT/16 QK/softmax/PV subtiles of 16 keys ----
        if (compute) {
#if HIPFIRE_FA2_KT == 32
            // qshare2 (KT32): one Q load+cvt per dim chunk feeds both
            // subtles' QK; then softmax/PV sub0 fully, then sub1.
            // Online-softmax order and PV accumulation order unchanged.
            float8_t sacc0 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc1 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const int kt_sub0 = ktile;
            const int kt_sub1 = ktile + 16;
            const bool do0 = kt_sub0 <= gmax;
            const bool do1 = kt_sub1 <= gmax;
            if (do0 || do1) {
                // QK as S^T = K*Q^T for both subtles, sharing Q fragment.
                // Q is f16 pre-converted: one aligned 16-byte half8 load per
                // 8 dims, no cvt. Element order matches the old f32 path
                // (qf[e] = dim dc*16+e), so the WMMA B operand is identical.
                for (int dc = 0; dc < 16; ++dc) {
                    half16_t qf;
                    {
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
                    }
                    if (do0) {
                        half16_t kf;
                        {
                            // sub 0: key = 0*16+ml
                            const int key = ml;
                            const int rot = fa2_swiz(key & 15);
                            const int base = key * 128;
                            // Four aligned b64 LDS loads at i={0,2,4,6}. Even
                            // swizzle makes each start 8-byte aligned; consecutive
                            // dwords i,i+1 never wrap (start never 127). Unpack
                            // low then high dword into half16 order for WMMA A.
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
                        }
                        sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            kf, qf, sacc0);
                    }
                    if (do1) {
                        half16_t kf;
                        {
                            // sub 1: key = 1*16+ml
                            const int key = 16 + ml;
                            const int rot = fa2_swiz(key & 15);
                            const int base = key * 128;
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
                        }
                        sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            kf, qf, sacc1);
                    }
                    // Bound K/Q-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }

                // Softmax + PV for sub0, then sub1 (online order unchanged).
#pragma unroll 1
                for (int sub = 0; sub < 2; ++sub) {
                    const bool dos = (sub == 0) ? do0 : do1;
                    if (!dos)
                        continue;
                    float8_t sacc = (sub == 0) ? sacc0 : sacc1;
                    const int kt_sub = ktile + sub * 16;
                    const bool full = (kt_sub + 15) <= gmin;
                    // This lane pair's softmax query is ml; its position is
                    // loaded once per subtile (masked path only).
                    const int pos_ml =
                        (!full && qok_ml) ? positions[qr_ml] : 0;

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
                        // Four aligned b64 loads at i={0,2,4,6}. Within either
                        // V-row parity the even residues cover all even b64
                        // starts once; unpack order matches producer store order.
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
                        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            vf, prow, Ofr[dc]);
                        // Bound V-load hoisting to this chunk.
                        asm volatile("" ::: "memory");
                    }
                }
            }
#else
            // KT64 path: original per-subtile Q reload (host pins KT32).
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
                        // Four aligned b64 LDS loads at i={0,2,4,6}. Even
                        // swizzle makes each start 8-byte aligned; consecutive
                        // dwords i,i+1 never wrap (start never 127). Unpack
                        // low then high dword into half16 order for WMMA A.
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
                        // Q is f16 pre-converted (same half8 pair as KT32).
                        half16_t qf;
                        {
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
                        }
                        sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            kf, qf, sacc);
                        // Bound K/Q-load hoisting to this chunk.
                        asm volatile("" ::: "memory");
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
                    // Four aligned b64 loads at i={0,2,4,6}. Within either
                    // V-row parity the even residues cover all even b64
                    // starts once; unpack order matches producer store order.
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
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        vf, prow, Ofr[dc]);
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }
            }
#endif
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

// ---- F4b: on-device Q pre-convert (f32 -> f16, optional FWHT-256) ----
// Launch once per FA2 call before the body below, on the same stream (no
// sync needed: same-stream ordering covers the q16 handoff). Reads f32 Q
// row-major [batch, 24, 256] — the exact layout/stride the body used to
// read — and writes f16 scratch row-major [batch, 24, 256] with the
// IDENTICAL `(_Float16)` cast the body used per fragment (device
// v_cvt_f16_f32, RNE), so the WMMA B operand is bit-identical.
//
// With do_rotate != 0 (fwht3 module only) applies fwht_shfl_forward_256 to
// the f32 row BEFORE the cast: same helper, same lane mapping (lane owns
// dims lane*8..+8), same op order as the old in-body prologue, so rotated
// values are bit-identical. Reads the ORIGINAL f32 Q and leaves it
// untouched — Q is now immutable across launches (replay-idempotent).
// signs1/signs2 may be null when do_rotate == 0.
//
// The entry name is KMODE-selected: the Q8 module (KMODE=0) exports
// attention_fa2_q_preconvert_gfx11, the fwht3 module (KMODE=3) exports
// attention_fa2_q_preconvert_fwht3_gfx11. The host function cache
// (rdna-compute compile_and_load_kernel) is keyed by SYMBOL, not by
// (module, symbol): one shared name across the two module builds silently
// binds whichever build registered first. The Q8 module compiles first in
// the ship gate, so a shared name made the fwht3 path launch the
// rotation-free KMODE=0 build with do_rotate compiled out — a
// deterministic near-miss on both cards (near-uniform softmax hides the
// missing rotation to ~1e-3), not an op-order race. Distinct symbols make
// each launcher resolve its own module's build.
#if HIPFIRE_FA2_KMODE == 3
#define HIPFIRE_FA2_Q_PRECONVERT_SYM attention_fa2_q_preconvert_fwht3_gfx11
#else
#define HIPFIRE_FA2_Q_PRECONVERT_SYM attention_fa2_q_preconvert_gfx11
#endif
//
// One wave32 per Q row (4 rows per 128-thread block): the row index depends
// only on (blockIdx.x, wave), so every lane of a wave takes the same early
// path and the ds_swizzle group inside fwht_shfl_forward_256 stays intact.
// Rows are disjoint across waves/blocks (no LDS, no barrier); each f16 row
// is written exactly once (one aligned 16-byte store per lane).
extern "C" __global__ __launch_bounds__(128, 1) void HIPFIRE_FA2_Q_PRECONVERT_SYM(
    const float* __restrict__ q,
    _Float16* __restrict__ q16,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int batch_size,
    int do_rotate)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 4 + wave;
    if (r >= batch_size * 24)
        return;
    const int qr = r / 24;
    const int h = r % 24;
    const float* rp = q + (unsigned)qr * 6144u + (unsigned)h * 256u;
    _Float16* wp = q16 + (unsigned)qr * 6144u + (unsigned)h * 256u;
    float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
    float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
    float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
    float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
#if HIPFIRE_FA2_KMODE == 3
    if (do_rotate)
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7,
                              signs1, signs2, lane);
#else
    (void)signs1;
    (void)signs2;
    (void)do_rotate;
#endif
    half8_t o;
    o[0] = (_Float16)v0; o[1] = (_Float16)v1;
    o[2] = (_Float16)v2; o[3] = (_Float16)v3;
    o[4] = (_Float16)v4; o[5] = (_Float16)v5;
    o[6] = (_Float16)v6; o[7] = (_Float16)v7;
    *(half8_t*)(wp + lane * 8) = o;
}

// Body entries take pre-converted f16 Q at kernarg offset 0 (same slot, same
// size as the old f32 Q pointer, so the pointer-array blob ABI is unchanged).
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_gfx11(
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
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body(q16, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}

// ---- KMODE=3 entry: fwht3 K, pre-rotated + pre-converted f16 Q ----
// K is stored FWHT-rotated, so Q must be rotated before QK (orthogonal
// transform: dot products preserved; fwht_shfl_forward_256 is normalized
// 1/sqrt(256) = 1/16 and scale_attn applies downstream exactly as in the
// Q8 path — no scale change). The rotation now lives in
// attention_fa2_q_preconvert_fwht3_gfx11 (same helper, same op order), so this
// entry takes the SAME kernarg list as the Q8 entry above (q16 at offset 0,
// no signs) and never mutates Q: replay-idempotent, capture-safe.
#if HIPFIRE_FA2_KMODE == 3
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fwht3k_gfx11(
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
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body(q16, k_cache, v_cache, out, positions, batch_size,
                 scale_attn, kv_h, q_base);
}
#endif
#if defined(HIPFIRE_FA2_I8)
// The whole F5a section is compiled only into the two i8 lab modules
// (kernels.rs prepends `#define HIPFIRE_FA2_I8 1` + turbo_common.h); the
// shipping Q8/fwht3 modules never see it.
// ---- F5a: int8-WMMA QK lab arm (KT32-only) ----
// Lab-only symbols in this section (never on a default path):
//   attention_fa2_q_preconvert_i8_gfx11  — f32 Q -> i8 codes + f32 sQ/row.
//   fa2_gqa_body_i8                     — private i8-QK body (f16 PV kept).
//   attention_q8_0_fa2_gqa_i8_gfx11      — Q8-K entry (KMODE=0 module).
//   attention_q8_0_fa2_gqa_fwht3k_i8_gfx11 — fwht3-K entry (KMODE=3 module).
//
// Layout (dynamic LDS 17,408 B, two resident workgroups per CU):
//   K codes 32 keys x 256 B raw i8 (8,192 B) + K scales f32 32x8 (1,024 B)
//   + V f16 transposed exactly as shipping (8,192 B).
// QK per 16-key subtile: 16 iu8 WMMAs (16-dim steps) into an i32 acc that is
// folded every 2 steps (one 32-dim block b) into the f32 score fragment with
// sQ*sK[key][b]; the score then flows into the shipping masking/online
// softmax/PV/epilogue unchanged. Barriers: same two per KT tile, same order.
// Q/K/V/positions are read-only in the body (replay-idempotent); Q codes and
// sQ are immutable after the pre-convert below.
typedef int32_t __attribute__((ext_vector_type(4))) fa2_i32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) fa2_i32x8_t;

// Fixed centroid-requantized fwht3 K LUT (§3.4 of the F5 plan):
// LUT8[c] = round(TURBO_C3_256[c] / gamma), gamma = 0.134860/127, and the
// fill materializes sK[k,b] = cnorm[k]*gamma per 32-dim block. The fill does
// a table lookup per code: no f16 value, no cnorm*LUT multiply per element.
__constant__ signed char FA2_FWHT3_LUT8[8] =
    {-127, -78, -44, -14, 14, 44, 78, 127};
#define FA2_FWHT3_GAMMA (0.134860f / 127.0f)

// ---- F5a: on-device Q pre-convert (f32 -> i8 + f32 row scale) ----
// Launch once per FA2 call before the i8 body below, on the same stream (no
// sync needed: same-stream ordering covers the q8 handoff). Reads f32 Q
// row-major [batch, n_heads, 256] and writes one 260-byte row per
// (token, head): bytes 0..255 are i8 codes, bytes 256..259 are the f32 sQ.
// Per row: absmax over D=256, sQ = absmax/127 (0 when absmax == 0),
// q8[d] = round-half-away (roundf) of Q/sQ clamped to [-127,127].
// With rotate_fwht3 != 0 applies fwht_shfl_forward_256 to the f32 row BEFORE
// quantization (same helper, same lane mapping as the f16 pre-convert), so
// the i8 body never rotates Q. The host passes the real 256-element sign
// tables in fwht3 mode; signs may be null when rotate_fwht3 == 0.
//
// ONE shared symbol across both i8 module builds (selected at run time, not
// by KMODE ifdefs): the body below is KMODE-independent, so either module's
// copy behaves identically and the symbol-keyed host function cache cannot
// misbind. Requires turbo_common.h prepended in BOTH i8 module concats
// (kernels.rs) for fwht_shfl_forward_256.
//
// Grid [ceil(batch*n_heads/4),1,1], block 128: one wave32 per Q row (4 rows
// per block); the row index depends only on (blockIdx.x, wave), so every
// lane of a wave takes the same early path and the ds_swizzle group inside
// fwht_shfl_forward_256 stays intact. Rows are disjoint across waves/blocks
// (no LDS, no barrier); each 260-byte row is written exactly once.
extern "C" __global__ __launch_bounds__(128, 1) void attention_fa2_q_preconvert_i8_gfx11(
    const float* __restrict__ q,
    unsigned char* __restrict__ q8,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    int rotate_fwht3)
{
    (void)n_kv_heads;
    if (n_heads != 24 || head_dim != 256)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 4 + wave;
    if (r >= batch_size * n_heads)
        return;
    const int qr = r / n_heads;
    const int h = r % n_heads;
    const float* rp = q + (unsigned)qr * 6144u + (unsigned)h * 256u;
    unsigned char* wp = q8 + (unsigned)r * 260u;
    float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
    float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
    float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
    float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
    if (rotate_fwht3)
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7,
                              signs1, signs2, lane);
    float am = fabsf(v0);
    am = fmaxf(am, fabsf(v1)); am = fmaxf(am, fabsf(v2));
    am = fmaxf(am, fabsf(v3)); am = fmaxf(am, fabsf(v4));
    am = fmaxf(am, fabsf(v5)); am = fmaxf(am, fabsf(v6));
    am = fmaxf(am, fabsf(v7));
    am = fmaxf(am, __shfl_xor(am, 16, 32));
    am = fmaxf(am, __shfl_xor(am, 8, 32));
    am = fmaxf(am, __shfl_xor(am, 4, 32));
    am = fmaxf(am, __shfl_xor(am, 2, 32));
    am = fmaxf(am, __shfl_xor(am, 1, 32));
    const float sQ = (am > 0.0f) ? am * (1.0f / 127.0f) : 0.0f;
    const float inv = (sQ > 0.0f) ? 1.0f / sQ : 0.0f;
    int q0 = (int)roundf(v0 * inv), q1 = (int)roundf(v1 * inv);
    int q2 = (int)roundf(v2 * inv), q3 = (int)roundf(v3 * inv);
    int q4 = (int)roundf(v4 * inv), q5 = (int)roundf(v5 * inv);
    int q6 = (int)roundf(v6 * inv), q7 = (int)roundf(v7 * inv);
    q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
    q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
    q2 = q2 < -127 ? -127 : (q2 > 127 ? 127 : q2);
    q3 = q3 < -127 ? -127 : (q3 > 127 ? 127 : q3);
    q4 = q4 < -127 ? -127 : (q4 > 127 ? 127 : q4);
    q5 = q5 < -127 ? -127 : (q5 > 127 ? 127 : q5);
    q6 = q6 < -127 ? -127 : (q6 > 127 ? 127 : q6);
    q7 = q7 < -127 ? -127 : (q7 > 127 ? 127 : q7);
    // Row stride 260 B is 4-aligned but not 16-aligned: memcpy emits safe
    // narrow stores (pre-convert is one pass over Q; not the hot path).
    const uint32_t wlo = (uint32_t)(q0 & 0xff) | ((uint32_t)(q1 & 0xff) << 8) |
                         ((uint32_t)(q2 & 0xff) << 16) | ((uint32_t)(q3 & 0xff) << 24);
    const uint32_t whi = (uint32_t)(q4 & 0xff) | ((uint32_t)(q5 & 0xff) << 8) |
                         ((uint32_t)(q6 & 0xff) << 16) | ((uint32_t)(q7 & 0xff) << 24);
    __builtin_memcpy(wp + (unsigned)lane * 8u, &wlo, 4);
    __builtin_memcpy(wp + (unsigned)lane * 8u + 4u, &whi, 4);
    if (lane == 0)
        __builtin_memcpy(wp + 256, &sQ, 4);
}
// Shared i8 workgroup body (direct only: writes normalized O to `out`).
// `kv_h` selects the KV head (blockIdx.y); `q_base` the 8-position tile
// (blockIdx.x). Q arrives as i8 codes row-major [batch, 24, 260 B]
// (see attention_fa2_q_preconvert_i8_gfx11); the body never touches f32 Q.
// Causal bounds come from positions[] (mailbox-reduced to gmin/gmax);
// the tile loop clamps to the derived seq_len. KT32-only: the entries
// early-out otherwise (LDS budget below assumes KT=32).
// Register prefetch of the next tile's raw K bytes mirrors F6 and can be
// compiled out with -DHIPFIRE_FA2_I8_NOPF for the with/without comparison.
__device__ __forceinline__ void fa2_gqa_body_i8(
    const unsigned char* __restrict__ q8,
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
    constexpr int KT_KITERS = KT / 16;
    constexpr int KT_VITERS = KT / 32;
    constexpr int V_ROW = KT / 2;
    constexpr int V_MASK = KT / 2 - 1;
    // KT32 LDS map (dwords): K codes 2048 + V f16 2048 + K scales 256 f32.
    constexpr int K_DWORDS = KT * 64;
    constexpr int LDS_DWORDS = KT * 128 + 256;

    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int half = lane >> 4;
    const bool compute = wave < 3;

    const int q_stride = 24 * 260;
    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + K_DWORDS;
    float* KS = (float*)(LDS + 2 * K_DWORDS);

    // ---- Transition 1: online state init (compute waves) ----
    // Q codes stream straight from q8_scratch (one 16-byte memcpy per
    // 16-dim step per lane, sQ once per row); nothing Q-side is kept across
    // tiles. K scales arrive via the LDS KS plane (see fill).
    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qq_base = 0;
    float sQ = 0.0f;
    if (compute) {
        h_ml = 6 * kv_h + (wave << 1) + (ml >> 3);
        qr_ml = q_base + (ml & 7);
        qok_ml = qr_ml < batch_size;
        qq_base = (unsigned)qr_ml * 6240u + (unsigned)h_ml * 260u;
        if (qok_ml)
            __builtin_memcpy(&sQ, q8 + qq_base + 256, 4);
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // ---- Per-WG causal bounds (positions[] is authoritative) ----
    // Identical to shipping: lanes 0..7 read the tile's eight positions;
    // wave 0 reduces; lane 0 publishes via two transient LDS mailbox dwords
    // (overwritten by the first tile's K-scale stores, so the 17,408 B
    // plane budget is unaffected).
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

    // ---- F6-style software prefetch of the next tile's raw K bytes ----
    // Per-lane raw K share at KT32/block128 (all four waves fill):
    //   Q8 K: KT*8 slots / 128 lanes = 2 slots/lane x 34 B (f16 scale +
    //         32 i8 codes) = 68 B. The f16 scales share one dword.
    //   fwht3 K: 2 slots/lane x (f32 cnorm + 12 packed bytes).
    // Staged raw (copied, not dequantized) right after the post-fill barrier
    // so the K global loads overlap this tile's QK/softmax/PV; the next fill
    // slot copies K from here with no fresh K global loads. The small staging
    // arrays scalarize to VGPRs (see resource report with/without).
#ifndef HIPFIRE_FA2_I8_NOPF
#if HIPFIRE_FA2_KMODE == 3
    float pf_kcn[KT_KITERS];
    uint32_t pf_kp[KT_KITERS * 4];
#else
    uint32_t pf_ks;
    uint32_t pf_kc[KT_KITERS * 8];
#endif
#endif
    // ---- Transitions 2-4: KT tiles ----
    for (int tile = 0; ; ++tile) {
        const int ktile = tile * KT;
        if (ktile >= seq_len)
            break;
#ifndef HIPFIRE_FA2_I8_NOPF
        // Tiles after the first consume the raw bytes prefetched into
        // registers during the previous tile's compute phase (see below);
        // tile 0 fills from global exactly as below. Uniform (tile), so no
        // divergence. tile > 0 implies the prefetch ran: the previous
        // iteration saw ktile_prev + KT < seq_len, else this iteration
        // would have broken out above.
        const bool have_pf = (tile > 0);
#else
        constexpr bool have_pf = false;
#endif

        // ---- Transition 2: cooperative KT fill (all four waves) ----
#if HIPFIRE_FA2_KMODE == 3
        // fwht3 K: per (position, kv head) record = [f32 cnorm][96 B codes].
        // KT*8 slots (k*8+b, like the Q8 path); slot (k, b) unpacks its
        // block's four 3-byte groups via the fixed centroid i8 LUT (no f16
        // value, no per-element multiply) into eight code dwords under the
        // 64-dword swizzle, and stores sK[k,b] = cnorm*gamma into KS.
        // V fill below is unchanged (Q8_0).
#ifndef HIPFIRE_FA2_I8_NOPF
        if (have_pf) {
            // Decode the prefetched raw K bytes (no global loads here).
            // Store lines are identical to the global path below; the
            // prefetch staged zeros for OOB slots, so tails zero-fill
            // bit-identically.
#pragma unroll 1
            for (int t = 0; t < KT_KITERS; ++t) {
                const int s = tid + t * 128;
                const int k = s >> 3;
                const int b = s & 7;
                const float sk = pf_kcn[t] * FA2_FWHT3_GAMMA;
                const int rot = fa2_swiz(k & 15);
                const int base = k * 64;
#pragma unroll 1
                for (int m = 0; m < 4; ++m) {
                    const uint32_t packed = pf_kp[t * 4 + m];
                    const int c00 = (packed >> 0) & 7, c01 = (packed >> 3) & 7;
                    const int c10 = (packed >> 6) & 7, c11 = (packed >> 9) & 7;
                    const int c20 = (packed >> 12) & 7, c21 = (packed >> 15) & 7;
                    const int c30 = (packed >> 18) & 7, c31 = (packed >> 21) & 7;
                    const uint32_t d0 =
                        (uint32_t)(uint8_t)FA2_FWHT3_LUT8[c00] |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c01] << 8) |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c10] << 16) |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c11] << 24);
                    const uint32_t d1 =
                        (uint32_t)(uint8_t)FA2_FWHT3_LUT8[c20] |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c21] << 8) |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c30] << 16) |
                        ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c31] << 24);
                    Kdw[base + ((b * 8 + m * 2 + rot) & 63)] = d0;
                    Kdw[base + ((b * 8 + m * 2 + 1 + rot) & 63)] = d1;
                }
                KS[k * 8 + b] = sk;
            }
        } else
#endif
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
            const float sk = cnorm * FA2_FWHT3_GAMMA;
            const int rot = fa2_swiz(k & 15);
            const int base = k * 64;
#pragma unroll 1
            for (int m = 0; m < 4; ++m) {
                uint32_t packed = 0;
                if (rec != nullptr)
                    __builtin_memcpy(&packed, rec + 4 + (b * 4 + m) * 3, 3);
                const int c00 = (packed >> 0) & 7, c01 = (packed >> 3) & 7;
                const int c10 = (packed >> 6) & 7, c11 = (packed >> 9) & 7;
                const int c20 = (packed >> 12) & 7, c21 = (packed >> 15) & 7;
                const int c30 = (packed >> 18) & 7, c31 = (packed >> 21) & 7;
                const uint32_t d0 =
                    (uint32_t)(uint8_t)FA2_FWHT3_LUT8[c00] |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c01] << 8) |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c10] << 16) |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c11] << 24);
                const uint32_t d1 =
                    (uint32_t)(uint8_t)FA2_FWHT3_LUT8[c20] |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c21] << 8) |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c30] << 16) |
                    ((uint32_t)(uint8_t)FA2_FWHT3_LUT8[c31] << 24);
                Kdw[base + ((b * 8 + m * 2 + rot) & 63)] = d0;
                Kdw[base + ((b * 8 + m * 2 + 1 + rot) & 63)] = d1;
            }
            KS[k * 8 + b] = sk;
        }
#else
        // Q8 K: KT*8 blocks (k*8+b), one block per slot. Raw i8 codes are
        // copied (not dequantized) straight into the swizzled dword plane;
        // the f16 block scale becomes the f32 KS[k*8+b] plane entry.
        // Never over-read the 34 B record (blk+2 may be 2-mod-4; memcpy
        // emits a correct unaligned load either way).
#ifndef HIPFIRE_FA2_I8_NOPF
        if (have_pf) {
            // Copy the prefetched raw K bytes (no global loads here).
            // Store lines identical to the global path below.
#pragma unroll 1
            for (int t = 0; t < KT_KITERS; ++t) {
                const int idx = tid + t * 128;
                const int k = idx >> 3;
                const int b = idx & 7;
                const uint16_t su = (uint16_t)(pf_ks >> (t * 16));
                const float sf = (float)((fa2_dw_t){.u = (uint32_t)su}).h[0];
                const int rot = fa2_swiz(k & 15);
                const int base = k * 64;
                const int pb = b * 8;
#pragma unroll 1
                for (int w = 0; w < 8; ++w)
                    Kdw[base + ((pb + w + rot) & 63)] = pf_kc[t * 8 + w];
                KS[k * 8 + b] = sf;
            }
        } else
#endif
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
            const float sf = (float)((fa2_dw_t){.u = (uint32_t)su}).h[0];
            const int rot = fa2_swiz(k & 15);
            const int base = k * 64;
            const int pb = b * 8;
#pragma unroll 1
            // 8 dwords cover the 32 codes verbatim.
            for (int w = 0; w < 8; ++w) {
                uint32_t u = 0;
                if (blk != nullptr)
                    __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                Kdw[base + ((pb + w + rot) & 63)] = u;
            }
            KS[k * 8 + b] = sf;
        }
#endif
        // V: KT*2 key-pairs (m*8+b); each slot reads the even and odd block
        // and packs (V[2m][d], V[2m+1][d]) per output dim. Source-identical
        // to shipping (f16 transpose loop); codes are loaded on demand per
        // dim (no staging array).
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
#ifndef HIPFIRE_FA2_I8_NOPF
        // ---- Prefetch the next tile's raw K bytes (all four waves) ----
        // Issued immediately after the post-fill barrier so these global
        // loads overlap the current tile's QK/softmax/PV below; the next
        // iteration's K fill (above) copies from these registers with no
        // fresh K global loads (V still streams in its fill slot).
        // Addressing and guards mirror the K fill loads exactly (no
        // out-of-range reads); OOB slots stage zeros so tails zero-fill
        // bit-identically. The QK loop's compiler barrier below already
        // fences these loads above the QK loop, so no sched_barrier (and
        // never an asm memory clobber) is used.
        {
            const int ktile_next = ktile + KT;
            if (ktile_next < seq_len) {
#if HIPFIRE_FA2_KMODE == 3
#pragma unroll 1
                for (int t = 0; t < KT_KITERS; ++t) {
                    const int s = tid + t * 128;
                    const int k = s >> 3;
                    const int b = s & 7;
                    const int gk = ktile_next + k;
                    const unsigned char* rec = (gk < seq_len)
                        ? k_cache + (unsigned long long)gk * 400ULL
                            + (unsigned long long)(kv_h * 100)
                        : nullptr;
                    float cnorm = 0.0f;
                    if (rec != nullptr)
                        __builtin_memcpy(&cnorm, rec, 4);
                    pf_kcn[t] = cnorm;
#pragma unroll 1
                    for (int m = 0; m < 4; ++m) {
                        uint32_t packed = 0;
                        if (rec != nullptr)
                            __builtin_memcpy(&packed, rec + 4 + (b * 4 + m) * 3, 3);
                        pf_kp[t * 4 + m] = packed;
                    }
                }
#else
                pf_ks = 0;
#pragma unroll 1
                for (int t = 0; t < KT_KITERS; ++t) {
                    const int idx = tid + t * 128;
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
                    pf_ks |= (uint32_t)su << (t * 16);
#pragma unroll 1
                    for (int w = 0; w < 8; ++w) {
                        uint32_t u = 0;
                        if (blk != nullptr)
                            __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                        pf_kc[t * 8 + w] = u;
                    }
                }
#endif
            }
        }
#endif

        // ---- Transition 3: KT/16 QK/softmax/PV subtiles of 16 keys ----
        // QK as S^T = K*Q^T for both subtiles, sharing one Q fragment per
        // 16-dim step. A lane pair's query row is ml; lanes ml and ml+16
        // supply identical A/B rows per the verified F5.1 map
        // (A[byte]=(row ml, k=byte), B[byte]=(k=byte, col ml)).
        // One i32 acc per subtile accumulates two K16 steps (one 32-dim
        // block b, |C_32| <= 520,192 < 2^21, clamp=false), then folds once:
        //   sacc[j] += (float)C[j] * (sQ * sK[key][b])
        // and is zeroed. Softmax/PV below are source-identical to shipping.
        if (compute) {
            float8_t sacc0 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc1 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const int kt_sub0 = ktile;
            const int kt_sub1 = ktile + 16;
            const bool do0 = kt_sub0 <= gmax;
            const bool do1 = kt_sub1 <= gmax;
            if (do0 || do1) {
                fa2_i32x8_t iacc0 = {0, 0, 0, 0, 0, 0, 0, 0};
                fa2_i32x8_t iacc1 = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll 1
                for (int b = 0; b < 8; ++b) {
                    for (int s2 = 0; s2 < 2; ++s2) {
                        const int st = b * 2 + s2;
                        // Q fragment: 16 bytes at qq_base+st*16 (memcpy:
                        // the 260 B row stride is not 16-aligned, so no
                        // wide vector load; the compiler picks safe narrows).
                        fa2_i32x4_t qf = {0, 0, 0, 0};
                        if (qok_ml)
                            __builtin_memcpy(&qf, q8 + qq_base + (unsigned)(st * 16), 16);
                        if (do0) {
                            // sub 0: key = 0*16+ml. Four code dwords at
                            // 4*st..4*st+3 read as two aligned b64s (starts
                            // are even: 4*st plus the even swizzle rot; the
                            // pair never wraps: start <= 62).
                            const int key = ml;
                            const int rot = fa2_swiz(key & 15);
                            const int base = key * 64;
                            const int32x2_t p0 = *(const int32x2_t*)(Kdw + (base + ((st * 4 + rot) & 63)));
                            const int32x2_t p1 = *(const int32x2_t*)(Kdw + (base + ((st * 4 + 2 + rot) & 63)));
                            const fa2_i32x4_t ka = {p0[0], p0[1], p1[0], p1[1]};
                            iacc0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                                true, ka, true, qf, iacc0, false);
                        }
                        if (do1) {
                            // sub 1: key = 1*16+ml.
                            const int key = 16 + ml;
                            const int rot = fa2_swiz(key & 15);
                            const int base = key * 64;
                            const int32x2_t p0 = *(const int32x2_t*)(Kdw + (base + ((st * 4 + rot) & 63)));
                            const int32x2_t p1 = *(const int32x2_t*)(Kdw + (base + ((st * 4 + 2 + rot) & 63)));
                            const fa2_i32x4_t ka = {p0[0], p0[1], p1[0], p1[1]};
                            iacc1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                                true, ka, true, qf, iacc1, false);
                        }
                        // Bound K/Q-load hoisting to this step.
                        asm volatile("" ::: "memory");
                    }
                    // Fold block b: one f32 fold per 32 dims (sK varies per
                    // block, so a single post-256 fold would be wrong). Lane
                    // (ml,half) owns keys sub*16+2*j+half; KS holds f32.
                    if (do0) {
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            sacc0[j] += (float)iacc0[j] *
                                (sQ * KS[(2 * j + half) * 8 + b]);
                        iacc0 = (fa2_i32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                    if (do1) {
#pragma unroll
                        for (int j = 0; j < 8; ++j)
                            sacc1[j] += (float)iacc1[j] *
                                (sQ * KS[(16 + 2 * j + half) * 8 + b]);
                        iacc1 = (fa2_i32x8_t){0, 0, 0, 0, 0, 0, 0, 0};
                    }
                }

                // Softmax + PV for sub0, then sub1 (online order unchanged).
#pragma unroll 1
                for (int sub = 0; sub < 2; ++sub) {
                    const bool dos = (sub == 0) ? do0 : do1;
                    if (!dos)
                        continue;
                    float8_t sacc = (sub == 0) ? sacc0 : sacc1;
                    const int kt_sub = ktile + sub * 16;
                    const bool full = (kt_sub + 15) <= gmin;
                    // This lane pair's softmax query is ml; its position is
                    // loaded once per subtile (masked path only).
                    const int pos_ml =
                        (!full && qok_ml) ? positions[qr_ml] : 0;

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
                        // Four aligned b64 loads at i={0,2,4,6}. Within either
                        // V-row parity the even residues cover all even b64
                        // starts once; unpack order matches producer store order.
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
                        Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            vf, prow, Ofr[dc]);
                        // Bound V-load hoisting to this chunk.
                        asm volatile("" ::: "memory");
                    }
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
// Body entries take the i8 Q scratch at kernarg offset 0 (same slot, same
// size as the old f32 Q / f16 q16 pointer, so the pointer-array blob ABI is
// unchanged). Grid [ceil(batch/8),4,1], block 128, dynamic LDS 17,408 B
// (host-pinned). KT32-only: early-out otherwise.
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_i8_gfx11(
    const unsigned char* __restrict__ q8_scratch,
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
    if (HIPFIRE_FA2_KT != 32)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body_i8(q8_scratch, k_cache, v_cache, out, positions, batch_size,
                    scale_attn, kv_h, q_base);
}

// ---- KMODE=3 entry: fwht3 K, pre-rotated + pre-quantized i8 Q ----
// K is stored FWHT-rotated, so Q must be rotated before QK (orthogonal
// transform: dot products preserved; fwht_shfl_forward_256 is normalized
// 1/sqrt(256) = 1/16 and scale_attn applies downstream exactly as in the
// Q8 path — no scale change). The rotation now lives in
// attention_fa2_q_preconvert_i8_gfx11 (same helper, same op order, runtime
// rotate_fwht3 select), so this entry takes the SAME kernarg list as the Q8
// i8 entry above (q8_scratch at offset 0, no signs) and never mutates Q:
// replay-idempotent, capture-safe. K codes arrive via the fixed FA2_FWHT3_LUT8
// fill above (centroid requantization error reported separately, never
// averaged with Q8 outcomes).
#if HIPFIRE_FA2_KMODE == 3
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fwht3k_i8_gfx11(
    const unsigned char* __restrict__ q8_scratch,
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
    if (HIPFIRE_FA2_KT != 32)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_gqa_body_i8(q8_scratch, k_cache, v_cache, out, positions, batch_size,
                    scale_attn, kv_h, q_base);
}
#endif

#endif // HIPFIRE_FA2_I8
