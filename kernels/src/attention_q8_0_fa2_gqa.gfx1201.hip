// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// gfx1201-only GQA-fused FA2 prefill attention (research opt-in).
//
// One workgroup owns one KV head and eight token positions: 48 live rows
// (8 positions x 6 query heads, H=24/Hkv=4/D=256) packed into three 16-row
// WMMA tiles, plus one loader/helper wave. K and V are each dequantized
// exactly once per 64-key tile into two swizzled f16 LDS planes (32 KiB
// each), then reused by all six query heads. Scores, online-softmax state
// (m, l), and f32 O remain in VGPRs; dynamic LDS is exactly 65,536 B.
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
//     use the lane's own alpha/l scalars: zero shuffles anywhere.
//
// LDS swizzle (f16 pairs as dwords, S(c) = 8*(c>>2) + (c&3)):
//   K store/read: dword pair p for key k at K[k][(p + S(k&15)) & 127].
//   V store/read: key-pair p for dim d at V_T[d][(p + S(d&15)) & 31].
// Loads are four explicit uint32_t reads unpacked to half8_t; the circular
// row wrap is intentional.
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
//       BEFORE the cast. Everything downstream of the K plane (layout,
//       swizzle, QK/PV, softmax, epilogue) is untouched.
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

__device__ __forceinline__ int fa2_swiz(int c)
{
    return 8 * (c >> 2) + (c & 3);
}
#ifdef HIPFIRE_FA2_FP8
// ---- Stage-a fp8 planes (OCP e4m3, gfx120x-only) ----
// K: 64 keys x 256 dims, one byte per dim, dword-packed. Key k's byte row
// is swizzled by the 4-aligned byte swizzle S8(c) = 32*(c>>2)+4*(c&3)
// (c = k&15); dword w of key k at Kdw8[k*64 + ((b*8+w+S(c))&63)] with
// S(c) = fa2_swiz(c) (S8>>2, same spread as the f16 plane, mod 64).
// V: even/odd byte planes (2 x 256 dims x 32 pairs), addressed as bytes
// through Vedw/Vodw. Pair m at dim d lives in quad qq = ((m>>2)+S(d&15))&7,
// byte lane m&3. Reads are one dword per plane (ps is always 4-aligned);
// fills use byte stores (no RMW races).
// Total: 4096 + 2048 + 2048 = 8192 dwords = 32,768 B dynamic LDS.
// q8_0 block scales are NOT stored (no budget left): they are reloaded
// from the KV-cache block headers (2 B, L1-served) at fragment-build time
// and applied per fragment in f32, RNE to f16 — the fill-order rounding.
// (A post-accumulation scale would be inexact: scales are per 32-dim
// block, 8 distinct scales per key, so no single epilogue multiply is
// exact; per-block partial accumulators would cost +8 VGPR. The screen's
// stage-a model (per-block partial dots scaled in f32) agrees with this
// placement to f16-input rounding, inside the tolerance gate.)
typedef float __attribute__((ext_vector_type(2))) fa2_f2_t;

// K fragment: 8 dims [dc*16+kg*8, +8) of key `key`, e4m3 -> f16 in-register
// (one cvt per byte-pair) then the q8 block scale in f32. KMODE=3 bakes
// cnorm at fill, so the load is a pure cvt there. One reused f32x2 temp
// keeps the transient live set small (register budget is the gate).
//
// Address construction only: the caller passes key_row_dw = key*64 and an
// opaque group-local byte coordinate
//   group_byte = dg*64 + kg*8 + 4*fa2_swiz(ml)
// established at the start of each executed (dg,sub) group with a
// dependency on the preceding score state (see QK loop). The byte offset
//   x = (group_byte + 16*u) & 255
// reproduces the historical dc*16+kg*8+swizzle expansion (dc = dg*4+u)
// exactly, including row wrapping, but the two expanded word addresses
// are rematerialized inside the group and die with it instead of living
// across the whole KT64 tile. The scale header still uses the original
// dc = dg*4+u and dc>>1 at the call site.
__device__ __forceinline__ half8_t fa2_kfrag_fp8(const uint32_t* __restrict__ Kdw8,
    unsigned key_row_dw, unsigned group_byte, int u, float sf)
{
    const unsigned x = (group_byte + 16u * (unsigned)u) & 255u;
    const unsigned xi = x >> 2;
    const uint32_t w0 = Kdw8[key_row_dw + xi];
    const uint32_t w1 = Kdw8[key_row_dw + ((xi + 1u) & 63u)];
    half8_t kf;
    // Scheduling barriers between pairs: without them LLVM batches all
    // four decodes ahead of the converts, holding 12 decoded floats live
    // at once (measured: +11 spills). Each pair is consumed into kf before
    // the next decode begins, so at most one f32x2 is live.
    fa2_f2_t dd = __builtin_amdgcn_cvt_pk_f32_fp8(w0, 0);
#if HIPFIRE_FA2_KMODE == 3
    (void)sf;
    kf[0] = (_Float16)dd[0]; kf[1] = (_Float16)dd[1];
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w0, 1);
    kf[2] = (_Float16)dd[0]; kf[3] = (_Float16)dd[1];
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w1, 0);
    kf[4] = (_Float16)dd[0]; kf[5] = (_Float16)dd[1];
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w1, 1);
    kf[6] = (_Float16)dd[0]; kf[7] = (_Float16)dd[1];
#else
    kf[0] = (_Float16)(sf * dd[0]); kf[1] = (_Float16)(sf * dd[1]);
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w0, 1);
    kf[2] = (_Float16)(sf * dd[0]); kf[3] = (_Float16)(sf * dd[1]);
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w1, 0);
    kf[4] = (_Float16)(sf * dd[0]); kf[5] = (_Float16)(sf * dd[1]);
    asm volatile("" : "+v"(dd));
    dd = __builtin_amdgcn_cvt_pk_f32_fp8(w1, 1);
    kf[6] = (_Float16)(sf * dd[0]); kf[7] = (_Float16)(sf * dd[1]);
#endif
    return kf;
}

// V fragment: 8 keys (4 pairs ps..ps+3, even/odd) at dim `d`. se/so hold
// the 4 pairs' even/odd block scales (f16, staged by the caller).
__device__ __forceinline__ half8_t fa2_vfrag_fp8(const uint32_t* __restrict__ Vedw,
    const uint32_t* __restrict__ Vodw, int d, int ps,
    const _Float16* __restrict__ se, const _Float16* __restrict__ so)
{
    const int qq = ((ps >> 2) + fa2_swiz(d & 15)) & 7;
    const uint32_t E = Vedw[d * 8 + qq];
    const uint32_t O = Vodw[d * 8 + qq];
    half8_t vf;
    fa2_f2_t ee = __builtin_amdgcn_cvt_pk_f32_fp8(E, 0);
    fa2_f2_t oo = __builtin_amdgcn_cvt_pk_f32_fp8(O, 0);
    vf[0] = (_Float16)((float)se[0] * ee[0]); vf[1] = (_Float16)((float)so[0] * oo[0]);
    vf[2] = (_Float16)((float)se[1] * ee[1]); vf[3] = (_Float16)((float)so[1] * oo[1]);
    ee = __builtin_amdgcn_cvt_pk_f32_fp8(E, 1);
    oo = __builtin_amdgcn_cvt_pk_f32_fp8(O, 1);
    vf[4] = (_Float16)((float)se[2] * ee[0]); vf[5] = (_Float16)((float)so[2] * oo[0]);
    vf[6] = (_Float16)((float)se[3] * ee[1]); vf[7] = (_Float16)((float)so[3] * oo[1]);
    return vf;
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
#ifdef HIPFIRE_FA2_FP8
    // Three live LDS views (V byte stores address Vedw/Vodw directly;
    // separate byte pointers would cost 2 whole-function VGPRs).
    uint32_t* Kdw8 = LDS;
    uint32_t* Vedw = LDS + 4096;
    uint32_t* Vodw = LDS + 6144;
#else
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + 8192;
#endif

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
    // Lanes 0..7 read the tile's eight positions; wave 0 reduces; lane 0
    // publishes via two transient LDS mailbox dwords (overwritten by the
    // first V fill, so the 64 KiB plane budget is unaffected).
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
#ifdef HIPFIRE_FA2_FP8
        // Top 2 dwords of the odd V plane (overwritten by the first V fill).
        LDS[8190] = (uint32_t)wmx;
        LDS[8191] = (uint32_t)wmn;
#else
        LDS[16382] = (uint32_t)wmx;
        LDS[16383] = (uint32_t)wmn;
#endif
    }
    __syncthreads();
    // Uniformize through lane 0: every lane loaded the same mailbox
    // dwords, but that uniformity is invisible through LDS.
#ifdef HIPFIRE_FA2_FP8
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[8190]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[8191]);
#else
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[16382]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[16383]);
#endif
    const int seq_len = gmax + 1;

    // ---- Transitions 2-4: KT64 tiles ----
    for (int tile = tile0; tile < tile1; ++tile) {
        const int ktile = tile * 64;
        if (ktile >= seq_len)
            break;

        // ---- Transition 2: cooperative KT64 fill (all four waves) ----
#if HIPFIRE_FA2_KMODE == 3
        // fwht3 K: per (position, kv head) record = [f32 cnorm][96 B codes]
        // (k_bytes_per_head = 100, k_bytes_per_pos = 4*100 = 400, head offset
        // kv_h*100 — the same contract as attention_flash_fwht3_tile_batched
        // and kv_cache_write_asym_k_fwht3*). 512 slots (k*8+b, like the Q8
        // path); slot (k, b) owns lanes b*4..b*4+3 (lane l owns dims l*8..+8
        // via 3 bytes at +4+l*3, code (packed >> 3*i) & 7, value cnorm *
        // TURBO_C3_256[code]) and dequantizes 32 dims = 16 dwords straight
        // into the swizzled dword plane (pair p for key k at
        // K[k][(p + S(k&15)) & 127] — identical layout downstream, so block
        // b covers pairs b*16..b*16+15 exactly as in the Q8 path). V fill
        // below is unchanged (Q8_0).
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
            const int rot = fa2_swiz(k & 15);
            const int base = k * 128;
#pragma unroll 1
            for (int m = 0; m < 4; ++m) {
                uint32_t packed = 0;
                if (rec != nullptr)
                    __builtin_memcpy(&packed, rec + 4 + (b * 4 + m) * 3, 3);
#ifdef HIPFIRE_FA2_FP8
                // e4m3 bytes, cnorm baked (per-key single scale needs no
                // fragment-time scale): dims Q..Q+7, Q=(b*4+m)*8, at bytes
                // (Q+j+S8)&255, S8 4-aligned -> 2 dwords, wrap-safe via &63.
                const int Q = (b * 4 + m) * 8;
                const int s8 = 32 * ((k & 15) >> 2) + 4 * ((k & 15) & 3);
                const int dwA = ((Q + s8) & 255) >> 2;
                unsigned accA = 0, accB = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int code0 = (packed >> (6 * i)) & 7;
                    const int code1 = (packed >> (6 * i + 3)) & 7;
                    const float v0 = cnorm * TURBO_C3_256[code0];
                    const float v1 = cnorm * TURBO_C3_256[code1];
                    if (i < 2) {
                        if (i == 0)
                            accA = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, accA, false);
                        else
                            accA = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, accA, true);
                    } else {
                        if (i == 2)
                            accB = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, accB, false);
                        else
                            accB = __builtin_amdgcn_cvt_pk_fp8_f32(v0, v1, accB, true);
                    }
                }
                Kdw8[k * 64 + dwA] = accA;
                Kdw8[k * 64 + ((dwA + 1) & 63)] = accB;
#else
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
#endif
            }
        }
#else
        // K: 512 blocks (k*8+b), one block per slot. Codes are loaded on
        // demand per pair (no staging array) and dequantized straight
        // into the swizzled dword plane.
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
#ifdef HIPFIRE_FA2_FP8
                // e4m3 codes, one byte per dim (RNE of the raw int8 code,
                // exactly once); the block scale stays in the cache header
                // for fragment-time application. Dword w of slot (k,b).
                unsigned acc = 0;
                acc = __builtin_amdgcn_cvt_pk_fp8_f32(
                    (float)(int8_t)u, (float)(int8_t)(u >> 8), acc, false);
                acc = __builtin_amdgcn_cvt_pk_fp8_f32(
                    (float)(int8_t)(u >> 16), (float)(int8_t)(u >> 24), acc, true);
                Kdw8[k * 64 + ((b * 8 + w + fa2_swiz(k & 15)) & 63)] = acc;
#else
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
#endif
            }
        }
#endif
        // V: 256 key-pairs (m*8+b); each slot reads the even and odd block
        // and packs (V[2m][d], V[2m+1][d]) per output dim. Codes are loaded
        // on demand per dim (no staging array).
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
#ifdef HIPFIRE_FA2_FP8
                    // e4m3 bytes into the even/odd planes (RNE of the raw
                    // int8 code, exactly once); scales stay in the headers.
                    // Byte stores: every slot owns distinct bytes, no races.
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    unsigned e = 0, o = 0;
                    e = __builtin_amdgcn_cvt_pk_fp8_f32(c0, 0.0f, e, false);
                    o = __builtin_amdgcn_cvt_pk_fp8_f32(c1, 0.0f, o, false);
                    const int qq = (((m >> 2) + fa2_swiz(d & 15)) & 7) * 4 + (m & 3);
                    ((unsigned char*)Vedw)[d * 32 + qq] = (unsigned char)(e & 0xFFu);
                    ((unsigned char*)Vodw)[d * 32 + qq] = (unsigned char)(o & 0xFFu);
#else
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    fa2_dw_t dw;
                    dw.h[0] = (_Float16)(sf0 * c0);
                    dw.h[1] = (_Float16)(sf1 * c1);
                    Vdw[d * 32 + ((m + fa2_swiz(d & 15)) & 31)] = dw.u;
#endif
                }
            }
        }
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
#ifdef HIPFIRE_FA2_FP8
                    // fp8 path loads Q per fragment inside the fused
                    // build+WMMA loops below: staging all four here would
                    // hold 16 VGPR live across the QK peak (the spill
                    // gate). qfa is dead in this instantiation.
#else
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
#endif
                    if (do0
#ifdef HIPFIRE_FA2_FP8
                        && qok_ml
#endif
                    ) {
                        const int key = ml;
#ifdef HIPFIRE_FA2_FP8
                        // Fused build+WMMA (no kfa[4] staging): the q8 row
                        // offset is 32-bit and clamped; OOB fragments get a
                        // zero scale (softmax-masked). Accumulation order
                        // matches the f16 path exactly (u=0..3 into sacc).
                        // Fully unrolled u=0..3: each step's K addresses
                        // rematerialize from the group seed above, so only
                        // one group's addresses are live between steps.
                        const int gk = ktile + key;
                        const bool gkok = gk < seq_len;
                        const unsigned koff = gkok
                            ? (unsigned)gk * (unsigned)row_stride
                            : 0u;
                        const unsigned key_row_dw = (unsigned)key * 64u;
                        // Group-local K-address seed: opaque to
                        // loop-invariant expansion below. The tied asm
                        // input keeps this group's expanded addresses
                        // from hoisting above the preceding score state;
                        // the seed and its addresses die with the group.
                        unsigned group_byte = (unsigned)dg * 64u
                            + (unsigned)kg * 8u
                            + 4u * (unsigned)fa2_swiz(ml);
                        asm volatile("" : "+v"(group_byte)
                                         : "v"(sacc0), "v"(sacc1),
                                           "v"(sacc2), "v"(sacc3));
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // Q fragment on demand (no qfa staging here;
                            // qok_ml guards the sub-block, so the load
                            // is unconditional and always in bounds).
                            half8_t qf = ((const half8_t*)(q16 + qb + (unsigned)(dc * 16)))[kg];
                            const unsigned hbo = koff + (unsigned)(kv_blk + (dc >> 1)) * 34u;
                            const unsigned char* hb = k_cache + (unsigned long long)hbo;
                            const float ksf = gkok
                                ? (float)((fa2_dw_t){.u = (uint32_t)((uint16_t)hb[0] | ((uint16_t)hb[1] << 8))}).h[0]
                                : 0.0f;
                            half8_t kf = fa2_kfrag_fp8(Kdw8, key_row_dw, group_byte, u, ksf);
                            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kf, qf, sacc0);
                        }
#else
                        half8_t kfa[4];
                        const int rot = fa2_swiz(key & 15);
                        const int base = key * 128;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            half8_t kf;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                fa2_dw_t dw;
                                dw.u = Kdw[base
                                    + ((dc * 8 + kg * 4 + i + rot) & 127)];
                                kf[2 * i] = dw.h[0];
                                kf[2 * i + 1] = dw.h[1];
                            }
                            kfa[u] = kf;
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc0);
#endif
                    }
                    if (do1
#ifdef HIPFIRE_FA2_FP8
                        && qok_ml
#endif
                    ) {
                        const int key = 16 + ml;
#ifdef HIPFIRE_FA2_FP8
                        // Fused build+WMMA (no kfa[4] staging): the q8 row
                        // offset is 32-bit and clamped; OOB fragments get a
                        // zero scale (softmax-masked). Accumulation order
                        // matches the f16 path exactly (u=0..3 into sacc).
                        // Fully unrolled u=0..3: each step's K addresses
                        // rematerialize from the group seed above, so only
                        // one group's addresses are live between steps.
                        const int gk = ktile + key;
                        const bool gkok = gk < seq_len;
                        const unsigned koff = gkok
                            ? (unsigned)gk * (unsigned)row_stride
                            : 0u;
                        const unsigned key_row_dw = (unsigned)key * 64u;
                        // Group-local K-address seed: opaque to
                        // loop-invariant expansion below. The tied asm
                        // input keeps this group's expanded addresses
                        // from hoisting above the preceding score state;
                        // the seed and its addresses die with the group.
                        unsigned group_byte = (unsigned)dg * 64u
                            + (unsigned)kg * 8u
                            + 4u * (unsigned)fa2_swiz(ml);
                        asm volatile("" : "+v"(group_byte)
                                         : "v"(sacc0), "v"(sacc1),
                                           "v"(sacc2), "v"(sacc3));
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // Q fragment on demand (no qfa staging here;
                            // qok_ml guards the sub-block, so the load
                            // is unconditional and always in bounds).
                            half8_t qf = ((const half8_t*)(q16 + qb + (unsigned)(dc * 16)))[kg];
                            const unsigned hbo = koff + (unsigned)(kv_blk + (dc >> 1)) * 34u;
                            const unsigned char* hb = k_cache + (unsigned long long)hbo;
                            const float ksf = gkok
                                ? (float)((fa2_dw_t){.u = (uint32_t)((uint16_t)hb[0] | ((uint16_t)hb[1] << 8))}).h[0]
                                : 0.0f;
                            half8_t kf = fa2_kfrag_fp8(Kdw8, key_row_dw, group_byte, u, ksf);
                            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kf, qf, sacc1);
                        }
#else
                        half8_t kfa[4];
                        const int rot = fa2_swiz(key & 15);
                        const int base = key * 128;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            half8_t kf;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                fa2_dw_t dw;
                                dw.u = Kdw[base
                                    + ((dc * 8 + kg * 4 + i + rot) & 127)];
                                kf[2 * i] = dw.h[0];
                                kf[2 * i + 1] = dw.h[1];
                            }
                            kfa[u] = kf;
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc1);
#endif
                    }
                    if (do2
#ifdef HIPFIRE_FA2_FP8
                        && qok_ml
#endif
                    ) {
                        const int key = 32 + ml;
#ifdef HIPFIRE_FA2_FP8
                        // Fused build+WMMA (no kfa[4] staging): the q8 row
                        // offset is 32-bit and clamped; OOB fragments get a
                        // zero scale (softmax-masked). Accumulation order
                        // matches the f16 path exactly (u=0..3 into sacc).
                        // Fully unrolled u=0..3: each step's K addresses
                        // rematerialize from the group seed above, so only
                        // one group's addresses are live between steps.
                        const int gk = ktile + key;
                        const bool gkok = gk < seq_len;
                        const unsigned koff = gkok
                            ? (unsigned)gk * (unsigned)row_stride
                            : 0u;
                        const unsigned key_row_dw = (unsigned)key * 64u;
                        // Group-local K-address seed: opaque to
                        // loop-invariant expansion below. The tied asm
                        // input keeps this group's expanded addresses
                        // from hoisting above the preceding score state;
                        // the seed and its addresses die with the group.
                        unsigned group_byte = (unsigned)dg * 64u
                            + (unsigned)kg * 8u
                            + 4u * (unsigned)fa2_swiz(ml);
                        asm volatile("" : "+v"(group_byte)
                                         : "v"(sacc0), "v"(sacc1),
                                           "v"(sacc2), "v"(sacc3));
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // Q fragment on demand (no qfa staging here;
                            // qok_ml guards the sub-block, so the load
                            // is unconditional and always in bounds).
                            half8_t qf = ((const half8_t*)(q16 + qb + (unsigned)(dc * 16)))[kg];
                            const unsigned hbo = koff + (unsigned)(kv_blk + (dc >> 1)) * 34u;
                            const unsigned char* hb = k_cache + (unsigned long long)hbo;
                            const float ksf = gkok
                                ? (float)((fa2_dw_t){.u = (uint32_t)((uint16_t)hb[0] | ((uint16_t)hb[1] << 8))}).h[0]
                                : 0.0f;
                            half8_t kf = fa2_kfrag_fp8(Kdw8, key_row_dw, group_byte, u, ksf);
                            sacc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kf, qf, sacc2);
                        }
#else
                        half8_t kfa[4];
                        const int rot = fa2_swiz(key & 15);
                        const int base = key * 128;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            half8_t kf;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                fa2_dw_t dw;
                                dw.u = Kdw[base
                                    + ((dc * 8 + kg * 4 + i + rot) & 127)];
                                kf[2 * i] = dw.h[0];
                                kf[2 * i + 1] = dw.h[1];
                            }
                            kfa[u] = kf;
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc2 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc2);
#endif
                    }
                    if (do3
#ifdef HIPFIRE_FA2_FP8
                        && qok_ml
#endif
                    ) {
                        const int key = 48 + ml;
#ifdef HIPFIRE_FA2_FP8
                        // Fused build+WMMA (no kfa[4] staging): the q8 row
                        // offset is 32-bit and clamped; OOB fragments get a
                        // zero scale (softmax-masked). Accumulation order
                        // matches the f16 path exactly (u=0..3 into sacc).
                        // Fully unrolled u=0..3: each step's K addresses
                        // rematerialize from the group seed above, so only
                        // one group's addresses are live between steps.
                        const int gk = ktile + key;
                        const bool gkok = gk < seq_len;
                        const unsigned koff = gkok
                            ? (unsigned)gk * (unsigned)row_stride
                            : 0u;
                        const unsigned key_row_dw = (unsigned)key * 64u;
                        // Group-local K-address seed: opaque to
                        // loop-invariant expansion below. The tied asm
                        // input keeps this group's expanded addresses
                        // from hoisting above the preceding score state;
                        // the seed and its addresses die with the group.
                        unsigned group_byte = (unsigned)dg * 64u
                            + (unsigned)kg * 8u
                            + 4u * (unsigned)fa2_swiz(ml);
                        asm volatile("" : "+v"(group_byte)
                                         : "v"(sacc0), "v"(sacc1),
                                           "v"(sacc2), "v"(sacc3));
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            // Q fragment on demand (no qfa staging here;
                            // qok_ml guards the sub-block, so the load
                            // is unconditional and always in bounds).
                            half8_t qf = ((const half8_t*)(q16 + qb + (unsigned)(dc * 16)))[kg];
                            const unsigned hbo = koff + (unsigned)(kv_blk + (dc >> 1)) * 34u;
                            const unsigned char* hb = k_cache + (unsigned long long)hbo;
                            const float ksf = gkok
                                ? (float)((fa2_dw_t){.u = (uint32_t)((uint16_t)hb[0] | ((uint16_t)hb[1] << 8))}).h[0]
                                : 0.0f;
                            half8_t kf = fa2_kfrag_fp8(Kdw8, key_row_dw, group_byte, u, ksf);
                            sacc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kf, qf, sacc3);
                        }
#else
                        half8_t kfa[4];
                        const int rot = fa2_swiz(key & 15);
                        const int base = key * 128;
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            half8_t kf;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                fa2_dw_t dw;
                                dw.u = Kdw[base
                                    + ((dc * 8 + kg * 4 + i + rot) & 127)];
                                kf[2 * i] = dw.h[0];
                                kf[2 * i + 1] = dw.h[1];
                            }
                            kfa[u] = kf;
                        }
#pragma unroll
                        for (int u = 0; u < 4; ++u)
                            sacc3 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc3);
#endif
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
#pragma unroll
                for (int dc = 0; dc < 16; ++dc) {
                    const int d = dc * 16 + ml;
#ifdef HIPFIRE_FA2_FP8
                    // e4m3 even/odd planes -> f16 fragment in-register; the
                    // pair scales (f16, staged) multiply per element in f32.
                    const int ps = sub * 8 + kg * 4;
                    const int vbb = d >> 5;
                    _Float16 vse[4], vso[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int gk0 = ktile + 2 * (ps + i);
                        _Float16 e = (_Float16)0.0f, o = (_Float16)0.0f;
                        if (gk0 < seq_len) {
                            uint16_t t;
                            __builtin_memcpy(&t, v_cache + (unsigned long long)gk0 * (unsigned long long)row_stride + (unsigned long long)(kv_blk + vbb) * 34ULL, 2);
                            e = ((fa2_dw_t){.u = (uint32_t)t}).h[0];
                        }
                        if (gk0 + 1 < seq_len) {
                            uint16_t t;
                            __builtin_memcpy(&t, v_cache + (unsigned long long)(gk0 + 1) * (unsigned long long)row_stride + (unsigned long long)(kv_blk + vbb) * 34ULL, 2);
                            o = ((fa2_dw_t){.u = (uint32_t)t}).h[0];
                        }
                        vse[i] = e;
                        vso[i] = o;
                    }
                    half8_t vf = fa2_vfrag_fp8(Vedw, Vodw, d, ps, vse, vso);
#else
                    const int rot = fa2_swiz(d & 15);
                    const int vbase = d * 32;
                    const int ps = sub * 8 + kg * 4;
                    half8_t vf;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        fa2_dw_t dw;
                        dw.u = Vdw[vbase + ((ps + i + rot) & 31)];
                        vf[2 * i] = dw.h[0];
                        vf[2 * i + 1] = dw.h[1];
                    }
#endif
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

// Body entries take pre-converted f16 Q at kernarg offset 0 (same slot, same
// size as the old f32 Q pointer, so the pointer-array blob ABI is unchanged).
#ifdef HIPFIRE_FA2_FP8
extern "C" __global__ __launch_bounds__(128, 2) void attention_q8_0_fa2_gqa_fp8_gfx1201(
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
extern "C" __global__ __launch_bounds__(128, 2) void attention_q8_0_fa2_gqa_fwht3k_fp8_gfx1201(
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

#ifdef HIPFIRE_FA2_FP8
extern "C" __global__ __launch_bounds__(128, 2) void attention_q8_0_fa2_gqa_partial_fp8_gfx1201(
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

#ifdef HIPFIRE_FA2_FP8
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
