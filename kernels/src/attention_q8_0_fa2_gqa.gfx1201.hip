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
//       FWHT-rotated; V stays Q8_0). KMODE=3 additionally rotates this WG's
//       Q rows in place (signed FWHT-256, fwht_shfl_forward_256 from
//       turbo_common.h — prepended by the launcher, NOT #included here: the
//       runtime compile has no -I to kernels/src) before the main body, and
//       dequantizes fwht3 K into the K plane. Everything downstream of the K
//       plane (layout, swizzle, QK/PV, softmax, epilogue) is untouched.
// KMODE=0 preprocessing output is token-identical to the pre-KMODE source
// (verify: disassembly diff of the existing symbols before/after).
#ifndef HIPFIRE_FA2_KMODE
#define HIPFIRE_FA2_KMODE 0
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

// Shared workgroup body. PARTIAL=false writes normalized O to `out`.
// PARTIAL=true follows CK split-KV's accumulator contract in one caller-owned
// allocation: F32 LSE records first, then normalized F32 O accumulators. The
// record index is ((query * H + head) * n_splits + split). Each split owns a
// disjoint range of the live KT64 tiles derived from positions[], rather than
// a max-context-sized static range.
template <bool PARTIAL>
__device__ __forceinline__ void fa2_gqa_body(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ positions,
    int batch_size,
    float scale_attn,
    int kv_h,
    int q_base,
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
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + 8192;

    // ---- Transition 1: online state init (compute waves) ----
    // Q is NOT kept persistent (64 VGPR the tile loop cannot spare).
    // Instead each QK chunk reloads its B-fragment as two float4 loads
    // from L1-resident Q (the 16 KiB per-wave Q tile is read once per
    // tile into L1 and re-hit per subtile; L1 bandwidth makes the
    // re-read free). This is the authorized fallback for the register
    // gate: persistent Q left exactly 2 spills.
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
        LDS[16382] = (uint32_t)wmx;
        LDS[16383] = (uint32_t)wmn;
    }
    __syncthreads();
    // Uniformize through lane 0: every lane loaded the same mailbox
    // dwords, but that uniformity is invisible through LDS.
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[16382]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[16383]);
    const int seq_len = gmax + 1;
    const int tiles_total = (seq_len + 63) / 64;
    const int tiles_per_split = (tiles_total + n_splits - 1) / n_splits;
    const int tile0 = split * tiles_per_split;
    const int tile1 = min(tile0 + tiles_per_split, tiles_total);

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
                    const float c0 = (float)(int8_t)(u0 >> (c * 8));
                    const float c1 = (float)(int8_t)(u1 >> (c * 8));
                    fa2_dw_t dw;
                    dw.h[0] = (_Float16)(sf0 * c0);
                    dw.h[1] = (_Float16)(sf1 * c1);
                    Vdw[d * 32 + ((m + fa2_swiz(d & 15)) & 31)] = dw.u;
                }
            }
        }
        __syncthreads();

        // ---- Transition 3: four KT16 QK/softmax/PV subtiles ----
        if (compute) {
#pragma unroll 1
            for (int sub = 0; sub < 4; ++sub) {
                const int kt_sub = ktile + sub * 16;
                if (kt_sub > gmax)
                    continue;
                const bool full = (kt_sub + 15) <= gmin;
                // This lane's softmax query is ml; its position is loaded
                // once per subtile (masked path only).
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                // QK as S^T = K*Q^T: A = K fragment (row ml = key),
                // B = Q fragment reloaded per chunk (row ml = query).
                // acc[j] = S[query=ml][key=8*kg+j].
                float8_t sacc =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                {
                    const int key = sub * 16 + ml;
                    const int rot = fa2_swiz(key & 15);
                    const int base = key * 128;
                    const unsigned qb = qq_base + (unsigned)(kg * 8);
                    // Unrolled by 4: each group's K/Q loads issue before
                    // its WMMAs (software pipelining for LDS/L1 latency).
                    // One scheduling barrier per group bounds the in-flight
                    // fragments to 4 chunks (the 66 free VGPR are spent on
                    // in-flight loads, not on hoisted-all-16 pileups).
#pragma unroll
                    for (int dg = 0; dg < 4; ++dg) {
                        half8_t kfa[4];
                        half8_t qfa[4];
#pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            const int dc = dg * 4 + u;
                            float4_t lo = {0.f, 0.f, 0.f, 0.f};
                            float4_t hi = {0.f, 0.f, 0.f, 0.f};
                            if (qok_ml) {
                                const float* qd =
                                    q + qb + (unsigned)(dc * 16);
                                lo = *(const float4_t*)qd;
                                hi = *(const float4_t*)(qd + 4);
                            }
                            qfa[u][0] = (_Float16)lo[0];
                            qfa[u][1] = (_Float16)lo[1];
                            qfa[u][2] = (_Float16)lo[2];
                            qfa[u][3] = (_Float16)lo[3];
                            qfa[u][4] = (_Float16)hi[0];
                            qfa[u][5] = (_Float16)hi[1];
                            qfa[u][6] = (_Float16)hi[2];
                            qfa[u][7] = (_Float16)hi[3];
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
                            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                                kfa[u], qfa[u], sacc);
                        asm volatile("" ::: "memory");
                    }
                }

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
            // CK-style split-KV contract. Keep LSE and Oacc in separate
            // logical planes so the combine kernel has unambiguous strides:
            //   lse[record]
            //   oacc[record][256] (normalized partial output)
            const unsigned records =
                (unsigned)batch_size * 24u * (unsigned)n_splits;
            float* lse = out;
            float* oacc = out + records;
            const unsigned record =
                ((unsigned)qr_ml * 24u + (unsigned)h_ml)
                    * (unsigned)n_splits + (unsigned)split;
            const float inv = l_val > 0.0f ? 1.0f / l_val : 0.0f;
            if (kg == 0) {
                if (qok_ml) {
                    lse[record] = l_val > 0.0f
                        ? m_old + __logf(l_val)
                        : -INFINITY;
                }
            }
            if (qok_ml) {
                float* rec = oacc + record * 256u;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const unsigned rj = (unsigned)(kg * 8 + j);
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc)
                        rec[rj + (unsigned)(dc * 16)] = Ofr[dc][j] * inv;
                }
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_gfx1201(
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
    fa2_gqa_body<false>(q, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 1);
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

extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_fwht3k_gfx1201(
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
    fa2_gqa_body<false>(q, k_cache, v_cache, out, positions, batch_size,
                        scale_attn, kv_h, q_base, 0, 1);
}
#endif

extern "C" __global__ __launch_bounds__(128, 1) void attention_q8_0_fa2_gqa_partial_gfx1201(
    const float* __restrict__ q,
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
    fa2_gqa_body<true>(q, k_cache, v_cache, partials, positions, batch_size,
                       scale_attn, kv_h, q_base, split, n_splits);
}

extern "C" __global__ __launch_bounds__(256) void attention_q8_0_fa2_gqa_merge_gfx1201(
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
    const long long records =
        (long long)batch_size * n_heads * n_splits;
    const float* lse = partials;
    const float* oacc = partials + records;
    float m = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        m = fmaxf(m, lse[(long long)rec * n_splits + s]);
    }
    for (int d = lane; d < head_dim; d += 32) {
        float num = 0.0f;
        float den = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const long long record = (long long)rec * n_splits + s;
            const float split_lse = lse[record];
            const float w = isfinite(split_lse) ? __expf(split_lse - m) : 0.0f;
            den += w;
            num += w * oacc[record * head_dim + d];
        }
        out[(long long)rec * head_dim + d] =
            (den > 0.0f) ? num / den : 0.0f;
    }
}
