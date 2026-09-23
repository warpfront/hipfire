#define HIPFIRE_FA2_KMODE 8
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201-only GQA-fused FA2 prefill attention (research opt-in).
//
// One workgroup owns one KV head and eight token positions: 48 live rows
// (8 positions x 6 query heads, H=24/Hkv=4/D=256) packed into three 16-row
// WMMA tiles, plus one loader/helper wave. K and V are each dequantized
// exactly once per 64-key tile into two fragment-major f16 LDS planes
// (32 KiB each: 2048 aligned half8 fragments per plane, one b128 load per
// fragment at the consumers), then reused by all six query heads. Scores,
// online-softmax state (m, l), and f32 O remain in VGPRs; dynamic LDS is
// exactly 65,536 B.
//
// Grid:  [ceil(batch/8), 4, 1] direct (x=batch tiles, y=KV head),
//        [ceil(batch/8), 4, n_splits] partial.
// Block: 128 threads (four wave32s; waves 0..2 compute, wave 3 helper).
//
// Row mapping (compute wave w=0..2, packed index t=0..15):
//   head_local = 2*w + (t >> 3);  q_local = t & 7;
//   h = 6*kv_h + head_local;      q_row = 8*blockIdx.x + q_local.
//
// Swapped operand order (no P transpose exists):
// gfx12 WMMA: lane (ml=lane&15, kg=lane>>4) supplies A[row=ml][k=kg*8..+8]
// and B[row=ml][k=kg*8..+8]; acc[j] = C[8*kg+j][ml] with
// C[i][n] = sum_k A[i][k]*B[n][k].
//   QK computes S^T = K*Q^T: A = K fragment (row ml = key 0..15),
//     B = Q fragment (row ml = packed query), acc[j] = S[query=ml][key].
//     Each lane owns one query row; the other key half sits in lane ml^16,
//     combined with one shfl_xor(16). m, l, alpha are per-lane scalars.
//   PV computes O^T = V^T*P^T: A = V^T fragment (row ml = dim 0..15 of the
//     dc tile), B = the packed P half8, Ofr[dc][j] = O[query=ml][dim].
//     O rows are the lane's OWN softmax query, so rescale/normalization
// LDS fragment layout (frozen; F = 8 f16 = 16 B, lane = 16*kg+ml):
//   Kfrag(sub,dc,lane) = ((dc*4+sub)*32+lane)*F, K at byte 0;
//   Vfrag(sub,dc,lane) = ((sub*16+dc)*32+lane)*F, V at byte 32768.
//   K fragment j = K[key=sub*16+ml][dim=dc*16+kg*8+j];
//   V fragment j = V[key=sub*16+kg*8+j][dim=dc*16+ml].
// Consumers issue one aligned half8 (b128) load per fragment; no wrap.
//
// Q f16 rounding matches the incumbent (Q/K/V/P operands f16,
// accumulators/state/O f32). No slots, no windows, no FP8, no other arch.

#include <hip/hip_runtime.h>

// K-cache mode, selected at JIT time (kernels.rs concat define):
//   0 = standard Q8 K (34 B blocks, f16 scale + 32 int8 codes).
//   3 = fwht3 K (100 B/head: f32 cnorm + 96 B of 3-bit codes, K stored
//       FWHT-rotated; V stays Q8_0). KMODE=3 dequantizes fwht3 K into the
//       K plane; the Q-side signed-FWHT-256 rotation lives in the
//       pre-convert kernel below (fwht_shfl_forward_256 from
//       turbo_common.h — prepended by the launcher, NOT #included here: the
//       runtime compile has no -I to kernels/src), applied to the f32 row
//       BEFORE the cast. Everything downstream of the K plane (fragment
//       layout, QK/PV, softmax, epilogue) is untouched.
//   8 = native fp8 KV, Q0 fill-time decode (slice-B): K and V rows are
//       token-local [codes: 4*256 bytes, head-major][scales: 4 f16]
//       (stride 1032). The fill decodes f16(f32(scale)*decode_e4m3(code))
//       into the same f16 planes; everything downstream is untouched.
#ifndef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 0
#endif

// U0 scaffold: HIPFIRE_FA2_FP8 renames entry symbols only. Body stays the
// f16 kernel until Ua (planes) / Ub (QK) / Uc (PV). gfx120x-only.
#ifdef HIPFIRE_FA2_FP8
#if !defined(__gfx1200__) && !defined(__gfx1201__)
#error "HIPFIRE_FA2_FP8 is gfx120x-only"
#endif
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

typedef union {
    uint32_t u;
    _Float16 h[2];
} fa2_dw_t;
// fp8 WMMA operand (8 bytes = one fragment) and HW e4m3 decode pair. Types
// only: no code, no object impact on any translation unit.
typedef int __attribute__((ext_vector_type(2))) v2i_t;
typedef float __attribute__((ext_vector_type(2))) float2_t;

// Frozen fragment-offset helpers (half-element units; one fragment = 8).
// lane = 16*kg+ml. K plane starts at half 0, V plane at half 16384.
__device__ __forceinline__ int fa2_koff(int sub, int dc, int lane, int j)
{
    return ((dc * 4 + sub) * 32 + lane) * 8 + j;
}
__device__ __forceinline__ int fa2_voff(int sub, int dc, int lane, int j)
{
    return ((sub * 16 + dc) * 32 + lane) * 8 + j;
}
__device__ __forceinline__ int fa2_kfrag8(int sub, int dc, int lane)
{
    return (dc * 4 + sub) * 32 + lane;
}
__device__ __forceinline__ int fa2_vfrag8(int sub, int dc, int lane)
{
    return (sub * 16 + dc) * 32 + lane;
}
#if HIPFIRE_FA2_KMODE == 8
// OCP E4M3FN -> f32, bit-exact copy of hipfire_fp8_e4m3_to_f32_tile in
// attention_flash_q8_0_tile.hip (the tiled reference). The Q0 fill below
// must match the reference decode exactly at the f16 rounding point:
// f16(f32(stored_scale) * decode_e4m3(code)). Guarded so the KMODE=0/3
// translation units never see it.
static __device__ __forceinline__ float fa2_fp8_e4m3_to_f32(unsigned char b) {
    const unsigned int u = b;
    const unsigned int s = (u & 0x80u) << 24;
    const unsigned int e = (u >> 3) & 0x0fu;
    const unsigned int m = u & 0x07u;
    if (e == 0u) {
        const float f = (float)m * 1.953125e-3f;  // m * 2^-9, exact
        return s ? -f : f;
    }
    if (e == 15u) {
        const float f = (m == 7u) ? __builtin_bit_cast(float, 0x7fc00000u)
                                  : 256.0f * (1.0f + (float)m * 0.125f);
        return s ? -f : f;
    }
    return __builtin_bit_cast(float, s | ((e - 7u + 127u) << 23) | (m << 20));
}
#endif

// Shared workgroup body. PARTIAL=false writes normalized O to `out`;
// PARTIAL=true writes [m, l, O_unnormalized[256]] records (stride 258) to
// `out`. `tile0/tile1` bound the KT64 tile range (direct: whole sequence;
// partial split: this split's disjoint range). `split` selects the record
// plane for PARTIAL (ignored otherwise). Q arrives f16 pre-converted
// row-major [batch, 24, 256] (see attention_fa2_q_preconvert_gfx1201);
// the body never touches f32 Q.
template <bool PARTIAL>
__device__ __forceinline__ void fa2_gqa_body(
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base,
    int tile0,
    int tile1,
    int split,
    int n_splits)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const bool compute = wave < 3;

    const int q_stride = 24 * 256;
    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    // Fragment-major f16 planes: K at byte 0, V at byte 32768.
    _Float16* K16 = (_Float16*)LDS;
    _Float16* V16 = (_Float16*)(LDS + 8192);

    // ---- Transition 1: online state init (compute waves) ----
    // Q arrives f16 pre-converted (see attention_fa2_q_preconvert_gfx1201);
    // within a KT64 tile, each dim-chunk Q fragment is loaded once (one
    // aligned 16-byte half8 load per 8 dims, no cvt) and reused for all
    // four subtiles' QK WMMAs (F4a); softmax/PV still run sub0..sub3 in
    // order so online order and PV accumulation stay unchanged.
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
    // Wave-local: each wave's lanes 0..7 read the same eight positions and
    // reduce them with the identical integer min/max graph, so all four
    // waves derive the same gmin/gmax/seq_len with no mailbox and no
    // barrier. Other lanes keep the existing sentinel values.
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (lane < 8) {
        const int qr = q_base + lane;
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
    // Uniformize through lane 0 of this wave; every wave reduced the same
    // eight positions, so every wave's lane 0 holds the same values.
    const int gmax = __builtin_amdgcn_readfirstlane(wmx);
    const int gmin = __builtin_amdgcn_readfirstlane(wmn);
    const int seq_len = gmax + 1;

    // ---- Transitions 2-4: KT64 tiles ----
    for (int tile = tile0; tile < tile1; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Transition 2: cooperative KT64 fill (all four waves) ----
#if HIPFIRE_FA2_KMODE == 8
        // Native fp8 K (Q0 fill-time decode): per (position, kv head) the
        // token-local row is [codes: 4*256 bytes, head-major][scales: 4 f16]
        // (stride 1032 = 4*(256+2); Hkv=4/D=256 gated at every entry).
        // b*32..b*32+31 of key ktile+k and dequantizes 32 dims = 8 dwords
        // straight into the fragment-order f16 K plane: dword w covers the
        // 4-wide dim run at 4-aligned d0 = b*32+w*4, which never crosses an
        // 8-dim fragment boundary, so all four share one (dc, lane) with
        // j = d0&7 .. +3 (same addressing as the q8 arm below). V fill
        // below is the native-fp8 twin of the Q8_0 V fill.
        // Decode matches the tiled reference exactly at the f16 rounding
        // point: f16(f32(stored_scale) * decode_e4m3(code)). Out-of-range
        // rows read zero codes with a zero scale, like the Q8 path.
        (void)row_stride;
        (void)kv_blk;
        for (int t = 0; t < 4; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 3;
            const int b = idx & 7;
            const int gk = ktile + k;
            const unsigned char* row = (gk < seq_len)
                ? k_cache + (unsigned long long)gk * 1032ULL
                : nullptr;
            uint16_t su = 0;
            if (row != nullptr)
                su = (uint16_t)row[1024 + kv_h * 2] |
                    ((uint16_t)row[1024 + kv_h * 2 + 1] << 8);
            _Float16 s = ((fa2_dw_t){.u = (uint32_t)su}).h[0];
            const float sf = (float)s;
            const int sub = k >> 4;
            const int mlk = k & 15;
            const unsigned char* codes = (row != nullptr)
                ? row + (unsigned long long)kv_h * 256ULL + (unsigned long long)b * 32ULL
                : nullptr;
#pragma unroll 1
            // 8 dwords cover the 32 codes (memcpy emits a correct load at any
            // alignment; 1032 % 4 == 0 so rows stay dword-aligned anyway).
            for (int w = 0; w < 8; ++w) {
                uint32_t u = 0;
                if (codes != nullptr)
                    __builtin_memcpy(&u, codes + w * 4, 4);
                const float d0 = fa2_fp8_e4m3_to_f32((unsigned char)u);
                const float d1 = fa2_fp8_e4m3_to_f32((unsigned char)(u >> 8));
                const float d2 = fa2_fp8_e4m3_to_f32((unsigned char)(u >> 16));
                const float d3 = fa2_fp8_e4m3_to_f32((unsigned char)(u >> 24));
                // 4-wide dim run at 4-aligned dd (template K:355-365).
                const int dd = b * 32 + w * 4;
                const int dc = dd >> 4;
                const int ln = 16 * ((dd >> 3) & 1) + mlk;
                const int o = fa2_koff(sub, dc, ln, dd & 7);
                K16[o] = (_Float16)(sf * d0);
                K16[o + 1] = (_Float16)(sf * d1);
                K16[o + 2] = (_Float16)(sf * d2);
                K16[o + 3] = (_Float16)(sf * d3);
            }
        }
#else
#if HIPFIRE_FA2_KMODE == 3
        // fwht3 K: per (position, kv head) record = [f32 cnorm][96 B codes]
        // (k_bytes_per_head = 100, k_bytes_per_pos = 4*100 = 400, head offset
        // kv_h*100 — the same contract as attention_flash_fwht3_tile_batched
        // and kv_cache_write_asym_k_fwht3*). 512 slots (k*8+b, like the Q8
        // path); slot (k, b) owns lanes b*4..b*4+3 (lane l owns dims l*8..+8
        // via 3 bytes at +4+l*3, code (packed >> 3*i) & 7, value cnorm *
        // TURBO_C3_256[code]). Arithmetic is unchanged; the two halves of
        // each dword pair land in their Kfrag(sub=k>>4, dc, lane) fragment
        // via fa2_koff instead of the old swizzled dword plane, so block b
        // covers the same dims b*32..b*32+31. V fill below is unchanged.
        for (int t = 0; t < 4; ++t) {
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
            const int sub = k >> 4;
            const int mlk = k & 15;
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
                    // Pair p0+i holds dims 2*(p0+i), 2*(p0+i)+1; d0 is
                    // even, so both halves share one fragment.
                    const int d0 = (p0 + i) * 2;
                    const int dc = d0 >> 4;
                    const int ln = 16 * ((d0 >> 3) & 1) + mlk;
                    const int o = fa2_koff(sub, dc, ln, d0 & 7);
                    K16[o] = (_Float16)(cnorm * TURBO_C3_256[code0]);
                    K16[o + 1] = (_Float16)(cnorm * TURBO_C3_256[code1]);
                }
            }
        }
#else
        // K: 512 blocks (k*8+b), one block per slot. Codes are loaded on
        // demand per pair (no staging array); dequant arithmetic is
        // unchanged and the four halves land in their Kfrag(sub=k>>4, dc,
        // lane) fragment via fa2_koff. The 4-wide dim run at 4-aligned d0
        // never crosses an 8-dim fragment boundary, so all four share one
        // (dc, lane) with j = d0&7 .. +3.
        for (int t = 0; t < 4; ++t) {
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
            const int sub = k >> 4;
            const int mlk = k & 15;
#pragma unroll 1
            // 8 quads cover the 32 codes (blk+2 may be 2-mod-4;
            // memcpy emits a correct unaligned load either way).
            for (int w = 0; w < 8; ++w) {
                uint32_t u = 0;
                if (blk != nullptr)
                    __builtin_memcpy(&u, blk + 2 + w * 4, 4);
                const float c0 = (float)(int8_t)u;
                const float c1 = (float)(int8_t)(u >> 8);
                const float c2 = (float)(int8_t)(u >> 16);
                const float c3 = (float)(int8_t)(u >> 24);
                const int d0 = b * 32 + w * 4;
                const int dc = d0 >> 4;
                const int ln = 16 * ((d0 >> 3) & 1) + mlk;
                const int o = fa2_koff(sub, dc, ln, d0 & 7);
                K16[o] = (_Float16)(sf * c0);
                K16[o + 1] = (_Float16)(sf * c1);
                K16[o + 2] = (_Float16)(sf * c2);
                K16[o + 3] = (_Float16)(sf * c3);
            }
        }
#endif
#endif
#if HIPFIRE_FA2_KMODE == 8
        // Native fp8 V (Q0 fill-time decode): 256 key-pairs (m*8+b) like the
        // Q8 path; each slot reads dims b*32..b*32+31 of keys ktile+2m /
        // ktile+2m+1 from the Vfill grouped-fragment layout and packs
        // (V[2m][d], V[2m+1][d]) per output dim into the fragment-order f16
        // V plane via fa2_voff (same addressing as the q8 arm below):
        // 2m/2m+1 never straddle a sub/kg boundary, so both halves share
        // one fragment at consecutive j. Grouped codes for (head, group, dim)
        // are one contiguous b64 at group_base + kv_h*D*8 + d*8; the pair's
        // two bytes are adjacent (u16) unless 2m ends a group (j0 == 7).
        // Same decode/rounding contract as the K fill above.
        for (int t = 0; t < 2; ++t) {
            const int pair = tid + t * 128;
            const int m = pair >> 3;
            const int b = pair & 7;
            const int k0 = ktile + 2 * m;
            // Scales at the group block end (home rows are aliased by grouped
            // codes); codes via the grouped bases below.
            const int j0 = k0 & 7;
            const int j1 = (k0 + 1) & 7;
            const unsigned char* gb0 = (k0 < seq_len)
                ? v_cache + (unsigned long long)(k0 >> 3) * (1032ULL * 8ULL)
                    + (unsigned long long)kv_h * 256ULL * 8ULL
                : nullptr;
            const unsigned char* gb1 = (k0 + 1 < seq_len)
                ? v_cache + (unsigned long long)((k0 + 1) >> 3) * (1032ULL * 8ULL)
                    + (unsigned long long)kv_h * 256ULL * 8ULL
                : nullptr;
            uint16_t s0u = 0;
            uint16_t s1u = 0;
            if (gb0 != nullptr) {
                const unsigned char* s0p = gb0 + 8192ULL
                    + (unsigned long long)j0 * 8ULL
                    + (unsigned long long)kv_h * 2ULL;
                s0u = (uint16_t)s0p[0] | ((uint16_t)s0p[1] << 8);
            }
            if (gb1 != nullptr) {
                const unsigned char* s1p = gb1 + 8192ULL
                    + (unsigned long long)j1 * 8ULL
                    + (unsigned long long)kv_h * 2ULL;
                s1u = (uint16_t)s1p[0] | ((uint16_t)s1p[1] << 8);
            }
            _Float16 s0 = ((fa2_dw_t){.u = (uint32_t)s0u}).h[0];
            _Float16 s1 = ((fa2_dw_t){.u = (uint32_t)s1u}).h[0];
            const float sf0 = (float)s0;
            const float sf1 = (float)s1;
            const bool same = (j0 != 7);
            const int sub = (2 * m) >> 4;
            const int kgm = ((2 * m) >> 3) & 1;
            const int jpair = (2 * m) & 7;
#pragma unroll 1
            for (int w = 0; w < 8; ++w) {
                // 4 dims per step (same trip count as the old dword loop).
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int d = b * 32 + w * 4 + c;
                    unsigned char c0 = 0, c1 = 0;
                    if (same) {
                        // Adjacent bytes of one group b64: one u16 load.
                        unsigned short p = 0;
                        if (gb0 != nullptr)
                            __builtin_memcpy(&p, gb0 + (size_t)d * 8u + (size_t)j0, 2);
                        c0 = (unsigned char)(p & 0xffu);
                        c1 = (unsigned char)(p >> 8);
                    } else {
                        if (gb0 != nullptr)
                            c0 = *(gb0 + (size_t)d * 8u + (size_t)j0);
                        if (gb1 != nullptr)
                            c1 = *(gb1 + (size_t)d * 8u + (size_t)j1);
                    }
                    const float v0 = fa2_fp8_e4m3_to_f32(c0);
                    const float v1 = fa2_fp8_e4m3_to_f32(c1);
                    const int dc = d >> 4;
                    const int ln = 16 * kgm + (d & 15);
                    const int o = fa2_voff(sub, dc, ln, jpair);
                    V16[o] = (_Float16)(sf0 * v0);
                    V16[o + 1] = (_Float16)(sf1 * v1);
                }
            }
        }
#else
        // V: 256 key-pairs (m*8+b); each slot reads the even and odd block.
        // Dequant arithmetic is unchanged; the pair (V[2m][d], V[2m+1][d])
        // lands in its Vfrag(sub=(2m)>>4, dc, lane) fragment via fa2_voff.
        // 2m/2m+1 never straddle a sub/kg boundary, so both halves share
        // one fragment at consecutive j. Codes are loaded on demand per dim
        // (no staging array).
        for (int t = 0; t < 2; ++t) {
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
            const int sub = (2 * m) >> 4;
            const int kgm = ((2 * m) >> 3) & 1;
            const int j0 = (2 * m) & 7;
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
                    const int dc = d >> 4;
                    const int ln = 16 * kgm + (d & 15);
                    const int o = fa2_voff(sub, dc, ln, j0);
                    V16[o] = (_Float16)(sf0 * c0);
                    V16[o + 1] = (_Float16)(sf1 * c1);
                }
        }
        }
#endif
        __syncthreads();

        // ---- Transition 3: four KT16 QK/softmax/PV subtiles ----
        // F4a: each Q dim-chunk fragment is loaded once per tile and shared
        // by all four subtiles' QK WMMAs; softmax/PV still run sub0..sub3 in
        // order so online-softmax order and PV accumulation are unchanged
        // (bit-exact: every subtile accumulates dc 0..15 in the same order
        // as the old per-subtile reload).
        if (compute) {
            const bool do0 = ktile <= gmax;
            const bool do1 = ktile + 16 <= gmax;
            const bool do2 = ktile + 32 <= gmax;
            const bool do3 = ktile + 48 <= gmax;
            float8_t sacc0 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc1 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc2 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            float8_t sacc3 =
                (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (do0 || do1 || do2 || do3) {
                // QK as S^T = K*Q^T: A = K fragment (row ml = key),
                // B = Q fragment loaded once per chunk (row ml = query).
                // acc[j] = S[query=ml][key=8*kg+j].
                const unsigned qb = qq_base;
                // Unrolled by 4: each group's Q loads issue once, then each
                // live subtile's K loads + WMMAs consume the shared fragments
                // (software pipelining for LDS/L1 latency). One scheduling
                // barrier per group bounds the in-flight fragments.
#pragma unroll
                for (int dg = 0; dg < 4; ++dg) {
                    half8_t qfa[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int dc = dg * 4 + u;
                        // One aligned 16-byte half8 load per 8 dims, no cvt:
                        // the pre-convert wrote f16 with the identical cast
                        // the old f32 path applied per fragment, so the WMMA
                        // B operand is bit-identical. qd[kg] selects this
                        // lane's 8 dims (dc*16+kg*8..+8).
                        const half8_t* qd =
                            (const half8_t*)(q16 + qb + (unsigned)(dc * 16));
                        half8_t qf = {0, 0, 0, 0, 0, 0, 0, 0};
                        if (qok_ml)
                            qf = qd[kg];
                        qfa[u] = qf;
                    }
                    if (do0) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=0, dc,
                            // lane): K[key=ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(0, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc0);
                    }
                    if (do1) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=1, dc,
                            // lane): K[key=16+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(1, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc1);
                    }
                    if (do2) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=2, dc,
                            // lane): K[key=32+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(2, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc2);
                    }
                    if (do3) {
                        half8_t kfa[4];
                        const half8_t* Kf = (const half8_t*)K16;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // One aligned half8 load = Kfrag(sub=3, dc,
                            // lane): K[key=48+ml][dim=dc*16+kg*8..+8].
                            kfa[u] = Kf[fa2_kfrag8(3, dc, lane)];
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc3);
                    }
                    asm volatile("" ::: "memory");
                }
            }
#pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const bool dos = (sub == 0) ? do0
                    : (sub == 1) ? do1 : (sub == 2) ? do2 : do3;
                if (!dos)
                    continue;
                float8_t sacc = (sub == 0) ? sacc0
                    : (sub == 1) ? sacc1 : (sub == 2) ? sacc2 : sacc3;
                const int kt_sub = ktile + sub * 16;
                const bool full = (kt_sub + 15) <= gmin;
                // This lane's softmax query is ml; its position is loaded
                // once per subtile (masked path only).
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                // Mask + row max. The mx chain serializes the unrolled
                // rows; the other key half is combined with one shuffle.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int key_g = kt_sub + 8 * kg + j;
                    const bool ok = qok_ml && (full || key_g <= pos_ml);
                    const float x = ok ? sacc[j] * scale_attn : -INFINITY;
                    // NOTE: max-accumulation happens in the second pass
                    // below; this pass only masks into sc.
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

                // Probabilities pack straight into the PV B operand.
                // The sum chain likewise serializes rows.
                half8_t prow;
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY)
                        ? 0.0f
                        : __expf(sacc[j] - m_new);
                    prow[j] = (_Float16)e;
                    sum += e;
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;

                // O rescale with the lane's OWN alpha: Ofr rows are the
                // lane's own softmax query (O[query=ml][dim]).
#pragma unroll
                for (int dc = 0; dc < 16; ++dc)
                    Ofr[dc] *= alpha;

                // PV as O^T = V^T*P^T: A = V^T fragment (row ml = dim of
                // the dc tile), B = prow. Ofr[dc][j] = O[query][dim].
                // One aligned half8 load = Vfrag(sub, dc, lane):
                // V[key=sub*16+kg*8..+8][dim=dc*16+ml].
                const half8_t* Vf = (const half8_t*)V16;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const half8_t vf = Vf[fa2_vfrag8(sub, dc, lane)];
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        vf, prow, Ofr[dc]);
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }
            }
        }
        // Tile turnover: no wave may overwrite LDS before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Transition 4: completion ----
    // Ofr[dc][j] at lane (ml,kg) = O[query=ml][dim=dc*16+8*kg+j]: own
    // query rows, 8 consecutive dims per dc. Own l_val normalizes.
    if (compute) {
        if (!PARTIAL) {
            if (qok_ml) {
                const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
                // 32-bit row base (whole O tensor < 10 MiB).
                const unsigned bo =
                    (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const unsigned bd = bo + (unsigned)(dc * 16 + kg * 8);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        out[bd + (unsigned)j] = Ofr[dc][j] * inv;
                }
            }
        } else {
            // Caller-owned partial records: [m, l, O_unnormalized[256]],
            // stride 258, layout [((qr*H + h)*S + split)]. Every lane owns
            // its query ml; lanes (ml, 0) write m/l once.
            if (kg == 0) {
                if (qok_ml) {
                    // 32-bit record base (partials < 64 MiB even at S=8).
                    const unsigned rb =
                        (((unsigned)qr_ml * 24u + (unsigned)h_ml)
                             * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                    float* rec = out + rb;
                    rec[0] = m_old;
                    rec[1] = l_val;
                }
            }
            if (qok_ml) {
                const unsigned rb =
                    (((unsigned)qr_ml * 24u + (unsigned)h_ml)
                         * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                float* rec = out + rb;
                // j-outer/dc-inner, matching the direct epilogue's proven
                // structure: j selects this lane's 8 dims per chunk.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned rj = rb + 2u + (unsigned)(kg * 8 + j);
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        rec[rj + (unsigned)(dc * 16)] = Ofr[dc][j];
                }
            }
        }
    }
}

// ---- Stage-b route-N byte-copy fill + dump (B1b diagnostic) ----
// KMODE=8 + HIPFIRE_FA2_FP8 only. The fill copies native e4m3 codes verbatim
// into the fragment-order fp8 planes (F = 8 bytes, K at byte 0, V at byte
// 16384; dynamic LDS 32768) with NO decode, NO VALU, NO f16 rounding:
//   K: one lane per (key, dc16) pair (1024/tile, 256/wave, 8/lane):
//      one global_load_b128 + two ds_store_b64 (dims dc*16+0..7 and +8..15
//      land in Kfrag(sub,dc,ml) and Kfrag(sub,dc,16+ml), 128 bytes apart).
//   V: one lane per (sub, dc, lane) fragment (2048/tile, 512/wave, 16/lane):
//      one global_load_b64 (the Vfill grouped-fragment layout stores the 8
//      keys at one dim contiguously) + one ds_store_b64, mirroring the K fill.
// Out-of-range keys (gk >= seq_len) write code 0. K scales are NEVER staged:
// route N reads them header-direct through fa2_scale_n below (S3 sizes no
// scale plane; Main ruling); V scales come from the group block end via
// fa2_scale_n_v. The dump writes LDS planes + accessor-read
// scales to `out` for the host oracle, then returns: with 32768 LDS the f16
// body cannot run here (its V plane at byte 32768 is out of bounds), so B2
// replaces dump+return with the fp8 body while keeping this fill.
// Route-N V scale accessor (group block end). Vfill grouped codes alias
// home-row bytes, so V scales live past the 8192 code bytes of each 8-key
// group block: scale for (gkey, kv_h) at group_base + Hkv*D*8 + (gkey&7)*8 +
// kv_h*2 (Hkv=4/D=256 literals, same style as fa2_scale_n). OOR reads 0.
static __device__ __forceinline__ float fa2_scale_n_v(
    const unsigned char* __restrict__ cache, int gkey, int kv_h, int seq_len)
{
    uint16_t u = 0;
    if (gkey < seq_len) {
        const unsigned char* s = cache
            + (unsigned long long)(gkey >> 3) * (1032ULL * 8ULL) + 8192ULL
            + (unsigned long long)(gkey & 7) * 8ULL
            + (unsigned long long)kv_h * 2ULL;
        u = (uint16_t)s[0] | ((uint16_t)s[1] << 8);
    }
    return (float)((fa2_dw_t){.u = (uint32_t)u}).h[0];
}
// Route-N scale accessor (header-direct). B4 gives route Q the same
// signature backed by its scratch scale plane, so the B2/B3 epilogue calls
// only this and the body stays source-agnostic. OOR keys read scale 0,
// matching the code-0 rule (§2: zero fragments contribute exactly zero).
static __device__ __forceinline__ float fa2_scale_n(
    const unsigned char* __restrict__ cache, int gkey, int kv_h, int seq_len)
{
    uint16_t u = 0;
    if (gkey < seq_len) {
        const unsigned char* row =
            cache + (unsigned long long)gkey * 1032ULL;
        u = (uint16_t)row[1024 + kv_h * 2] |
            ((uint16_t)row[1024 + kv_h * 2 + 1] << 8);
    }
    return (float)((fa2_dw_t){.u = (uint32_t)u}).h[0];
}

// B1b fill+dump. Dump layout per (blockIdx.x, kv_h, split, tile), in u32
// words at out + ((((bx*4+kv_h)*S+s)*512+t)*8320): K plane (4096) | V plane
// (4096) | K scales as f32 via fa2_scale_n (64) | V scales (64). T_MAX=512
// mirrors the partial tile convention; S=1/s=0 on direct.
template <bool PARTIAL>
__device__ void fa2_stageb_nfill_dump(
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    int kv_h,
    int q_base,
    int tile0,
    int tile1,
    int split,
    int n_splits)
{
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;

    // Wave-local causal bounds (same mailbox-free graph as the body: every
    // wave reduces the same eight positions).
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (lane < 8) {
        const int qr = q_base + lane;
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
    const int gmax = __builtin_amdgcn_readfirstlane(wmx);
    const int gmin = __builtin_amdgcn_readfirstlane(wmn);
    (void)gmin;
    const int seq_len = gmax + 1;

    extern __shared__ uint32_t LDS[];
    unsigned char* K8 = (unsigned char*)LDS;
    unsigned char* V8 = (unsigned char*)LDS + 16384;

    for (int tile = tile0; tile < tile1; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Route-N K fill: 1024 (key, dc) pairs, 8 per lane ----
        // Fast path (key in range) is straight-line: one b128 + two b64
        // per pair. Edge keys take the zeroing path. Full tiles never
        // diverge, so the steady-state census sees only the fast path.
        for (int t = 0; t < 8; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 4;
            const int dc = idx & 15;
            const int gk = ktile + k;
            const int sub = k >> 4;
            const int ml = k & 15;
            unsigned char* dst =
                K8 + (unsigned)(((dc * 4 + sub) * 32 + ml) * 8);
            if (gk < seq_len) {
                const unsigned char* src = k_cache
                    + (unsigned long long)gk * 1032ULL
                    + (unsigned long long)kv_h * 256ULL
                    + (unsigned long long)dc * 16ULL;
                uint4 v = *(const uint4*)src;
                *(unsigned long long*)dst =
                    ((unsigned long long)v.y << 32) | v.x;
                *(unsigned long long*)(dst + 128) =
                    ((unsigned long long)v.w << 32) | v.z;
            } else {
                *(unsigned long long*)dst = 0ULL;
                *(unsigned long long*)(dst + 128) = 0ULL;
            }
        }

        // ---- Route-N V fill: 2048 fragments, 16 per lane ----
        // Vfill grouped-fragment layout: kbase is 8-aligned (ktile % 64 == 0,
        // sub*16, kg*8), so the fragment is one contiguous b64 at group_base
        // + kv_h*D*8 + dim*8. Fast path: one global_load_b64 + one
        // ds_store_b64. Edge fragments fall back to per-byte predication off
        // the same base; all-OOR writes zero. (gbase is address math only and
        // never dereferenced unless its group is in range.)
        for (int t = 0; t < 16; ++t) {
            const int idx = tid + t * 128;
            const int sub = idx >> 9;
            const int r = idx & 511;
            const int dc = r >> 5;
            const int vlane = r & 31;
            const int ml = vlane & 15;
            const int kg = vlane >> 4;
            const int dim = dc * 16 + ml;
            const int kbase = ktile + sub * 16 + kg * 8;
            unsigned char* dst = V8
                + (unsigned)(((sub * 16 + dc) * 32 + vlane) * 8);
            const unsigned char* gbase = v_cache
                + (unsigned long long)(kbase >> 3) * (1032ULL * 8ULL)
                + (unsigned long long)kv_h * 256ULL * 8ULL
                + (unsigned long long)dim * 8ULL;
            unsigned long long w = 0;
            if (kbase + 7 < seq_len) {
                w = *(const unsigned long long*)gbase;
            } else if (kbase < seq_len) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned char c = 0;
                    if (kbase + j < seq_len)
                        c = *(gbase + j);
                    w |= (unsigned long long)c
                        << (unsigned long long)(j * 8);
                }
            }
            *(unsigned long long*)dst = w;
        }
        __syncthreads();

        // ---- Dump: planes + accessor-read scales ----
        const unsigned long long dump =
            (((((unsigned long long)(int)blockIdx.x * 4u
                 + (unsigned long long)kv_h)
                * (unsigned long long)n_splits + (unsigned long long)split)
               * 512ULL + (unsigned long long)tile)
              * 8320ULL);
        uint32_t* D = (uint32_t*)out + dump;
        for (int i = 0; i < 16; ++i)
            ((uint4*)D)[tid * 16 + i] =
                ((const uint4*)LDS)[tid * 16 + i];
        if (tid < 64) {
            const int gk = ktile + tid;
            ((float*)D)[8192 + tid] =
                fa2_scale_n(k_cache, gk, kv_h, seq_len);
            ((float*)D)[8256 + tid] =
                fa2_scale_n_v(v_cache, gk, kv_h, seq_len);
        }
        __syncthreads();
    }
}
// ---- Stage-b route-N body with fp8 QK leg (B2) ----
// Separate template so the f16 objects stay identical. Skeleton (bounds,
// tile loop, softmax order, O/partial records, barriers) mirrors
// fa2_gqa_body; differences: route-N byte-copy fill (same loops as the B1b
// oracle fn above), QK as S^T = K8*Q8^T in fp8 WMMA (64 per compute wave
// per full tile; one b64 per fragment operand; Q streamed from the fp8
// scratch row), score[j] = (acc[j]*qs)*sk[key_j] at the old scale position
// (qs formed per tile (qs_tile); sk for the 8 OWNED keys sub*16+8*kg+j via
// fa2_scale_n, one subtile lifetime, sharing one 8-register file with sv).
// - PV in fp8 with the bounded changing-unit O (B3): w = e*sv, b_new =
//   max(wmax/448, 2^-64), p8 = E4M3_RNE(w/b_new), rho = (alpha*b_prev)/b_new
//   folded into the Ofr rescale; V fragments load codes directly (no
//   decode); completion stores the physical numerator (direct: Ofr*b_prev/l,
//   partial: b_prev*Ofr in stride-258 records, merge untouched).
template <bool PARTIAL>
__device__ void fa2_stageb_nbody(
    const unsigned char* __restrict__ q8_codes,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base,
    int tile0,
    int tile1,
    int split,
    int n_splits)
{
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int ml = lane & 15;
    const int kg = lane >> 4;
    const bool compute = wave < 3;

    extern __shared__ uint32_t LDS[];
    // fp8 fragment planes: K at byte 0, V at byte 16384 (32768 LDS).
    unsigned char* K8 = (unsigned char*)LDS;
    unsigned char* V8 = (unsigned char*)LDS + 16384;

    float8_t Ofr[16];
    float m_old = -INFINITY;
    float l_val = 0.0f;
    // Changing-unit O state (§5): O_physical = b_prev*Ofr across subtiles
    // and tiles. b_prev = 1 with zero Ofr is the correct initial unit.
    float b_prev = 1.0f;
    int h_ml = 0;
    int qr_ml = 0;
    bool qok_ml = false;
    unsigned qb = 0;
    if (compute) {
        h_ml = 6 * kv_h + (wave << 1) + (ml >> 3);
        qr_ml = q_base + (ml & 7);
        qok_ml = qr_ml < batch_size;
        qb = ((unsigned)qr_ml * 24u + (unsigned)h_ml) * 256u;
#pragma unroll
        for (int dc = 0; dc < 16; ++dc)
            Ofr[dc] = (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }

    // ---- Per-WG causal bounds (positions[] is authoritative) ----
    // Wave-local: each wave's lanes 0..7 read the same eight positions and
    // reduce them with the identical integer min/max graph, so all four
    // waves derive the same gmin/gmax/seq_len with no mailbox and no
    // barrier. Other lanes keep the existing sentinel values.
    int wmx = -1;
    int wmn = 0x7fffffff;
    if (lane < 8) {
        const int qr = q_base + lane;
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
    const int gmax = __builtin_amdgcn_readfirstlane(wmx);
    const int gmin = __builtin_amdgcn_readfirstlane(wmn);
    const int seq_len = gmax + 1;

    // ---- KT64 tiles ----
    for (int tile = tile0; tile < tile1; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Route-N byte-copy fill (same loops as the B1b oracle) ----
        // Rolled (unlike the oracle copy): rolling serializes the 24 slot
        // copies so their temps reuse instead of overlapping in scheduling
        // (saves ~80 VGPR); dynamic work is identical, plus loop latches.
#pragma unroll 1
        for (int t = 0; t < 8; ++t) {
            const int idx = tid + t * 128;
            const int k = idx >> 4;
            const int dc = idx & 15;
            const int gk = ktile + k;
            const int sub = k >> 4;
            const int mlk = k & 15;
            unsigned char* dst =
                K8 + (unsigned)(((dc * 4 + sub) * 32 + mlk) * 8);
            if (gk < seq_len) {
                const unsigned char* src = k_cache
                    + (unsigned long long)gk * 1032ULL
                    + (unsigned long long)kv_h * 256ULL
                    + (unsigned long long)dc * 16ULL;
                uint4 v = *(const uint4*)src;
                *(unsigned long long*)dst =
                    ((unsigned long long)v.y << 32) | v.x;
                *(unsigned long long*)(dst + 128) =
                    ((unsigned long long)v.w << 32) | v.z;
            } else {
                *(unsigned long long*)dst = 0ULL;
                *(unsigned long long*)(dst + 128) = 0ULL;
            }
        }
#pragma unroll 1
        for (int t = 0; t < 16; ++t) {
            const int idx = tid + t * 128;
            const int sub = idx >> 9;
            const int r = idx & 511;
            const int dc = r >> 5;
            const int vlane = r & 31;
            const int mlv = vlane & 15;
            const int kgv = vlane >> 4;
            const int dim = dc * 16 + mlv;
            const int kbase = ktile + sub * 16 + kgv * 8;
            unsigned char* dst = V8
                + (unsigned)(((sub * 16 + dc) * 32 + vlane) * 8);
            // Same grouped-fragment b64 as the oracle fill above.
            const unsigned char* gbase = v_cache
                + (unsigned long long)(kbase >> 3) * (1032ULL * 8ULL)
                + (unsigned long long)kv_h * 256ULL * 8ULL
                + (unsigned long long)dim * 8ULL;
            unsigned long long w = 0;
            if (kbase + 7 < seq_len) {
                w = *(const unsigned long long*)gbase;
            } else if (kbase < seq_len) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned char c = 0;
                    if (kbase + j < seq_len)
                        c = *(gbase + j);
                    w |= (unsigned long long)c
                        << (unsigned long long)(j * 8);
                }
            }
            *(unsigned long long*)dst = w;
        }
        __syncthreads();

        // ---- Four KT16 QK/softmax/PV subtiles (online order kept) ----
        // Fused per subtile (B2 VGPR tradeoff): each live subtile runs its
        // own QK dg groups into one sacc, then mask/softmax/PV in order, so
        // online-softmax order and PV accumulation are unchanged.
        if (compute) {
            const bool do0 = ktile <= gmax;
            const bool do1 = ktile + 16 <= gmax;
            const bool do2 = ktile + 32 <= gmax;
            const bool do3 = ktile + 48 <= gmax;
            // qs formed per tile (not per call): sq is constant for the
            // query, so holding qb/sqp/sqv across the fill only wastes
            // VGPRs. One f32 load + mul per tile; negligible packets.
            float qs_tile = 0.0f;
            {
                const unsigned qrow =
                    (unsigned)qr_ml * 24u + (unsigned)h_ml;
                const float* sqp = (const float*)(
                    q8_codes + (unsigned)batch_size * 24u * 256u);
                float sqv = 1.0f;
                if (qok_ml)
                    sqv = sqp[qrow];
                qs_tile = sqv * scale_attn;
            }
#pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const bool dos = (sub == 0) ? do0
                    : (sub == 1) ? do1 : (sub == 2) ? do2 : do3;
                if (!dos)
                    continue;
                // QK for this subtile only (fused: one sacc instead of four
                // saves 24 VGPR; Q reloaded per live subtile — L1-served —
                // while K/WMMA counts are unchanged). S^T = K8*Q8^T in fp8.
                float8_t sacc =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                // K base in v2i units (sub factored once per subtile; dc
                // below is unrolled-const, so every fragment load is a base
                // VGPR + immediate offset — the slice-A addressing idiom.
                // Dynamic fa2_kfrag8 indexing here cost ~100 packets/tile in
                // index math + spill reloads; never do that again.)
                const unsigned kbn = (unsigned)(sub * 32) + (unsigned)lane;
                const v2i_t* Kf = (const v2i_t*)K8;
                // V base in u64 units for the expand below (same idiom).
                const unsigned vbn = (unsigned)(sub * 512) + (unsigned)lane;
                // OOR-query clamp (per subtile): row 0 always exists, and
                // below are unconditionally valid.
                const unsigned qbb = qok_ml ? qb : 0u;
#pragma unroll 1
                for (int dg = 0; dg < 4; ++dg) {
                    // Q base per group (dg*64 unrolled-const → immediate).
                    // OOR lanes clamp to row 0 (valid: batch >= 1): their
                    // scores are fully discarded by the mask below, so all
                    // lanes issue straight-line loads (no diamonds).
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int dc = dg * 4 + u;
                        // One aligned 8-byte load: Q codes for this lane's
                        // query at dims dc*16+kg*8..+8.
                        const v2i_t* qd = (const v2i_t*)(
                            q8_codes + qbb + (unsigned)(dc * 16));
                        v2i_t qf = qd[kg];
                        // One aligned 8-byte load = Kfrag(sub, dc, lane):
                        // static immediate kbn + dc*128 (dc unrolled-const).
                        v2i_t kf = Kf[kbn + dc * 128];
                        sacc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            kf, qf, sacc);
                    }
                    // Bound the in-flight fragments to this group.
                    asm volatile("" ::: "memory");
                }
                const int kt_sub = ktile + sub * 16;
                const bool full = (kt_sub + 15) <= gmin;
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;
                // sk for the 8 OWNED keys (transposed ownership: this lane
                // supplies K row ml but owns acc keys sub*16+8*kg+j, never
                // ml). One subtile lifetime, sharing one 8-register file
                // with sv below (sk is dead once scores are formed).
                float sc[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    sc[j] = fa2_scale_n(
                        k_cache, kt_sub + 8 * kg + j, kv_h, seq_len);
                // Mask + row max, scale folded at the old position.
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int key_g = kt_sub + 8 * kg + j;
                    const bool ok = qok_ml && (full || key_g <= pos_ml);
                    const float x = ok ? (sacc[j] * qs_tile) * sc[j] : -INFINITY;
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

                // Weighted-P quantization (plan §5): l accumulates
                // UNWEIGHTED e; w = e*sv enters only the numerator. sacc is
                // dead after e, so w reuses its registers (no w array).
                // sv reuses the scale file (sk dead: scores are formed).
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    sc[j] = fa2_scale_n_v(
                        v_cache, kt_sub + 8 * kg + j, kv_h, seq_len);
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY)
                        ? 0.0f
                        : __expf(sacc[j] - m_new);
                    sum += e;
                    sacc[j] = e * sc[j];
                }
                sum += __shfl_xor(sum, 16, 32);
                l_val = l_val * alpha + sum;

                // Block scale over all 16 valid keys (both halves agree):
                // wmax reduces this lane's 8 w combined with the partner
                // ml^16 lane. b_new floors at 2^-64; zero-weight rows keep
                // b_prev with zero P (w = 0 encodes to code 0 exactly).
                float wmax = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    wmax = fmaxf(wmax, sacc[j]);
                wmax = fmaxf(wmax, __shfl_xor(wmax, 16, 32));
                float b_new = b_prev;
                if (wmax > 0.0f)
                    b_new = fmaxf(wmax / 448.0f, 0x1p-64f);
                // O invariant O_physical = b_prev*Ofr: fold rho into the
                // existing 128-value rescale pass (replaces *= alpha).
                const float rho = (alpha * b_prev) / b_new;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc)
                    Ofr[dc] *= rho;

                // PV as O^T = V8^T*P8^T in fp8: A = V^T fragment codes
                // (row ml = dim of the dc tile, one b64, no decode), B =
                // the packed P8 v2i. Ofr[dc][j] = O[query][dim].
                // p8 = E4M3_RNE(w/b_new): 8 weights/lane in 2 dwords,
                // packed ONCE per subtile (division matches the host model
                // exactly); the dc loop only streams V fragments. Packing
                // inside the loop would redo 8 divs + 4 cvts 16 times.
                unsigned int p0 = 0u;
                p0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                    sacc[0] / b_new, sacc[1] / b_new, p0, false);
                p0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                    sacc[2] / b_new, sacc[3] / b_new, p0, true);
                unsigned int p1 = 0u;
                p1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                    sacc[4] / b_new, sacc[5] / b_new, p1, false);
                p1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                    sacc[6] / b_new, sacc[7] / b_new, p1, true);
                const v2i_t pb = {(int)p0, (int)p1};
                const v2i_t* Vf = (const v2i_t*)V8;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const v2i_t va = Vf[vbn + dc * 32];
                    Ofr[dc] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        va, pb, Ofr[dc]);
                    // Bound V-load hoisting to this chunk.
                    asm volatile("" ::: "memory");
                }
                b_prev = b_new;
            }
        }
        // Tile turnover: no wave may overwrite LDS before every PV read
        // has finished.
        __syncthreads();
    }

    // ---- Completion (same records as the f16 body) ----
    if (compute) {
        if (!PARTIAL) {
            if (qok_ml) {
                const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
                const unsigned bo =
                    (unsigned)qr_ml * 6144u + (unsigned)h_ml * 256u;
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const unsigned bd = bo + (unsigned)(dc * 16 + kg * 8);
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        out[bd + (unsigned)j] = Ofr[dc][j] * (b_prev * inv);
                }
            }
        } else {
            if (kg == 0) {
                if (qok_ml) {
                    const unsigned rb =
                        (((unsigned)qr_ml * 24u + (unsigned)h_ml)
                             * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                    float* rec = out + rb;
                    rec[0] = m_old;
                    rec[1] = l_val;
                }
            }
            if (qok_ml) {
                const unsigned rb =
                    (((unsigned)qr_ml * 24u + (unsigned)h_ml)
                         * (unsigned)n_splits + (unsigned)split)
                        * 258u;
                float* rec = out + rb;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned rj = rb + 2u + (unsigned)(kg * 8 + j);
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        rec[rj + (unsigned)(dc * 16)] = b_prev * Ofr[dc][j];
                }
            }
        }
    }
}


// Body entries take pre-converted f16 Q at kernarg offset 0 (same slot, same
// size as the old f32 Q pointer, so the pointer-array blob ABI is unchanged).
// Q0 (slice-B) entry: KMODE=8 selects `attention_fp8_e4m3_fa2_gqa_f16_gfx1201`
// (native-fp8 fill, f16 body). Stage-b arithmetic, when it lands, gets its
// own symbol; the f16 name stays Q0-only. KMODE=0/3 arms below are untouched.
#if HIPFIRE_FA2_KMODE == 8 && defined(HIPFIRE_FA2_FP8)
// B1b diagnostic (§14.2/§14.3): route-N byte-copy fill + plane dump; the fp8
// body lands in B2. Kernargs frozen: fp8-code Q in the old Q slot (unused by
// the fill), sq at +batch*24*256, out is the plane dump buffer, 32768 LDS.
extern "C" __global__ __launch_bounds__(128, 1) void attention_fp8_e4m3_fa2_gqa_gfx1201(
    const unsigned char* __restrict__ q8_codes,
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
    // B2: the fp8-QK body consumes the planes in place (dump retired).
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    fa2_stageb_nbody<false>(q8_codes, k_cache, v_cache, out, positions,
                            batch_size, scale_attn, kv_h, q_base, 0,
                            0x3fffffff, 0, 1);
}
#else
#if HIPFIRE_FA2_KMODE == 8
extern "C" __global__ __launch_bounds__(128, 1) void attention_fp8_e4m3_fa2_gqa_f16_gfx1201(
#elif defined(HIPFIRE_FA2_FP8)
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fp8_gfx1201(
#else
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_gfx1201(
#endif
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
    fa2_gqa_body<false>(q16, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 0x3fffffff, 0, 1);
}
#endif

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
// attention_fa2_q_preconvert_gfx1201, the fwht3 module (KMODE=3) exports
// attention_fa2_q_preconvert_fwht3_gfx1201. The host function cache
// (rdna-compute compile_and_load_kernel) is keyed by SYMBOL, not by
// (module, symbol): one shared name across the two module builds silently
// binds whichever build registered first. Distinct symbols make each
// launcher resolve its own module's build.
#if HIPFIRE_FA2_KMODE == 3
#define HIPFIRE_FA2_Q_PRECONVERT_SYM attention_fa2_q_preconvert_fwht3_gfx1201
#elif HIPFIRE_FA2_KMODE == 8
// Q0 body is f16 with no rotation, same as KMODE=0, but the symbol must be
// distinct: the host function cache is keyed by symbol alone.
#define HIPFIRE_FA2_Q_PRECONVERT_SYM attention_fp8_e4m3_fa2_q_preconvert_f16_gfx1201
#else
#define HIPFIRE_FA2_Q_PRECONVERT_SYM attention_fa2_q_preconvert_gfx1201
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

#if HIPFIRE_FA2_KMODE == 8 && defined(HIPFIRE_FA2_FP8)
// ---- Stage-b Q pre-convert (f32 -> e4m3 codes + f32 sq, plan §3) ----
// One wave32 per Q row (same grid convention as the f16 pre-convert:
// r = blockIdx.x*4+wave, grid [(batch*24).div_ceil(4)], block 128, no LDS;
// the launcher in attention.rs owns the grid). amax = wave_max(|q|) over a
// 5-step shfl_xor tree; sq = (amax == 0) ? 1 : amax/448 in f32 (never
// rounded: sq is an epilogue multiplier, not a WMMA operand); cq =
// E4M3_RNE(q/sq) via __builtin_amdgcn_cvt_pk_fp8_f32 — the same instruction
// the native fp8 writer uses. 8 codes/lane in one aligned u64 store; sq by
// lane 0. Zero rows yield zero codes with sq=1. Q never mutated. B2 gates
// the numerics; B1b gates compile + resources.
extern "C" __global__ __launch_bounds__(128, 1) void attention_fp8_e4m3_fa2_q_preconvert_fp8_gfx1201(
    const float* __restrict__ q,
    unsigned char* __restrict__ codes,
    float* __restrict__ sq,
    int batch_size)
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
    const float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
    const float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
    const float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
    const float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
    float a = fabsf(v0);
    a = fmaxf(a, fabsf(v1));
    a = fmaxf(a, fabsf(v2));
    a = fmaxf(a, fabsf(v3));
    a = fmaxf(a, fabsf(v4));
    a = fmaxf(a, fabsf(v5));
    a = fmaxf(a, fabsf(v6));
    a = fmaxf(a, fabsf(v7));
    a = fmaxf(a, __shfl_xor(a, 16, 32));
    a = fmaxf(a, __shfl_xor(a, 8, 32));
    a = fmaxf(a, __shfl_xor(a, 4, 32));
    a = fmaxf(a, __shfl_xor(a, 2, 32));
    a = fmaxf(a, __shfl_xor(a, 1, 32));
    const float s = (a == 0.0f) ? 1.0f : a / 448.0f;
    unsigned int w0 = 0u;
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v0 / s, v1 / s, w0, false);
    w0 = __builtin_amdgcn_cvt_pk_fp8_f32(v2 / s, v3 / s, w0, true);
    unsigned int w1 = 0u;
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v4 / s, v5 / s, w1, false);
    w1 = __builtin_amdgcn_cvt_pk_fp8_f32(v6 / s, v7 / s, w1, true);
    unsigned char* wp = codes + ((unsigned)qr * 24u + (unsigned)h) * 256u
        + (unsigned)lane * 8u;
    *(unsigned int*)wp = w0;
    *(unsigned int*)(wp + 4) = w1;
    if (lane == 0)
        sq[(unsigned)qr * 24u + (unsigned)h] = s;
}
#endif

// ---- KMODE=3 entry: fwht3 K, pre-rotated + pre-converted f16 Q ----
// K is stored FWHT-rotated, so Q must be rotated before QK (orthogonal
// transform: dot products preserved; fwht_shfl_forward_256 is normalized
// 1/sqrt(256) = 1/16 and scale_attn applies downstream exactly as in the
// Q8 path — no scale change). The rotation now lives in
// attention_fa2_q_preconvert_fwht3_gfx1201 (same helper, same op order), so
// this entry takes the SAME kernarg list as the Q8 entry above (q16 at
// offset 0, no signs) and never mutates Q: replay-idempotent,
// capture-safe.
#if HIPFIRE_FA2_KMODE == 3
#ifdef HIPFIRE_FA2_FP8
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fwht3k_fp8_gfx1201(
#else
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fwht3k_gfx1201(
#endif
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
    fa2_gqa_body<false>(q16, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 0x3fffffff, 0, 1);
}
#endif

#if HIPFIRE_FA2_KMODE == 8 && defined(HIPFIRE_FA2_FP8)
// B1b diagnostic partial twin: same tile partition as the f16 partial; the
// dump index carries the split so split dumps never alias.
extern "C" __global__ __launch_bounds__(128, 1) void attention_fp8_e4m3_fa2_gqa_partial_gfx1201(
    const unsigned char* __restrict__ q8_codes,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn,
    int n_splits)
{
    // B2: the fp8-QK body consumes the planes in place (dump retired).
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    const int split = blockIdx.z;
    if (split >= n_splits)
        return;
    const int per = (512 + n_splits - 1) / n_splits;
    const int t0 = split * per;
    const int t1 = t0 + per;
    fa2_stageb_nbody<true>(q8_codes, k_cache, v_cache, out, positions,
                           batch_size, scale_attn, kv_h, q_base, t0, t1,
                           split, n_splits);
}
#else
#if HIPFIRE_FA2_KMODE == 8
extern "C" __global__ __launch_bounds__(128, 1) void attention_fp8_e4m3_fa2_gqa_partial_f16_gfx1201(
#elif defined(HIPFIRE_FA2_FP8)
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_partial_fp8_gfx1201(
#else
extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_partial_gfx1201(
#endif
    const _Float16* __restrict__ q16,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ partials,
    const int* __restrict__ positions,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int batch_size,
    float scale_attn,
    int n_splits)
{
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int kv_h = blockIdx.y;
    if (kv_h >= 4)
        return;
    const int q_base = blockIdx.x * 8;
    if (q_base >= batch_size)
        return;
    const int split = blockIdx.z;
    if (split >= n_splits)
        return;
    // Key-range partition needs seq_len, which lives in positions[] on
    // the host side of the launcher... instead partition by tile index
    // with an unbounded range: split s owns tiles [s*T/S,(s+1)*T/S) of
    // the clamped range. The body cannot know T yet, so approximate with
    // a fixed upper bound derived from the 32768 max_ctx_len gate:
    // T_MAX = 32768/64 = 512 tiles; per = ceil(512/S).
    const int per = (512 + n_splits - 1) / n_splits;
    const int t0 = split * per;
    const int t1 = t0 + per;
    fa2_gqa_body<true>(q16, k_cache, v_cache, partials, positions, batch_size,
                       scale_attn, kv_h, q_base, t0, t1, split, n_splits);
}
#endif

#if HIPFIRE_FA2_KMODE == 8 && defined(HIPFIRE_FA2_FP8)
// Stage-b merge (§14.3): record format is scale-free (physical numerator in
// the 258-float record), so this is the f16 merge body under the stage-b
// symbol. Duplicated rather than shared so the Q0 object stays identical.
extern "C" __global__ __launch_bounds__(256) void attention_fp8_e4m3_fa2_gqa_merge_gfx1201(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int batch_size,
    int n_heads,
    int head_dim,
    int n_splits)
{
    if (n_heads != 24 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rec = blockIdx.x * 8 + wave;
    if (rec >= batch_size * n_heads)
        return;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        const float* pr =
            partials + ((long long)rec * n_splits + s) * 258;
        m = fmaxf(m, pr[0]);
    }
    for (int d = lane; d < head_dim; d += 32) {
        float num = 0.0f;
        float den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float* pr =
                partials + ((long long)rec * n_splits + s) * 258;
            const float w =
                (pr[1] > 0.0f) ? __expf(pr[0] - m) : 0.0f;
            den += w * pr[1];
            num += w * pr[2 + d];
        }
        out[(long long)rec * head_dim + d] =
            (den > 0.0f) ? num / den : 0.0f;
    }
}
#else
#if HIPFIRE_FA2_KMODE == 8
extern "C" __global__ __launch_bounds__(256) void attention_fp8_e4m3_fa2_gqa_merge_f16_gfx1201(
#elif defined(HIPFIRE_FA2_FP8)
extern "C" __global__ __launch_bounds__(256) void attention_q8_0_fa2_gqa_merge_fp8_gfx1201(
#else
extern "C" __global__ __launch_bounds__(256) void attention_q8_0_fa2_gqa_merge_gfx1201(
#endif
    const float* __restrict__ partials,
    float* __restrict__ out,
    int batch_size,
    int n_heads,
    int head_dim,
    int n_splits)
{
    if (n_heads != 24 || head_dim != 256)
        return;
    if (n_splits < 1 || n_splits > 8)
        return;
    const int tid = (int)threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    const int rec = blockIdx.x * 8 + wave;
    if (rec >= batch_size * n_heads)
        return;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        const float* pr =
            partials + ((long long)rec * n_splits + s) * 258;
        m = fmaxf(m, pr[0]);
    }
    for (int d = lane; d < head_dim; d += 32) {
        float num = 0.0f;
        float den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float* pr =
                partials + ((long long)rec * n_splits + s) * 258;
            const float w =
                (pr[1] > 0.0f) ? __expf(pr[0] - m) : 0.0f;
            den += w * pr[1];
            num += w * pr[2 + d];
        }
        out[(long long)rec * head_dim + d] =
            (den > 0.0f) ? num / den : 0.0f;
    }
}
#endif
