//! F3.0 — FA2 gfx11 fill/exp/PV attribution twins (precondition for F3).
//!
//! Diagnostic only. Does not edit production kernels.
//! Derived from kernels/src/attention_q8_0_fa2_gqa.gfx11.hip (Q8 entry).
//! Shipping shape: H24 / KV4 / D256 (not 48/8/128).
//!
//! Six twins of the Q8 entry, distinct symbols:
//!   fa2_attrib_prod    — unmodified production body
//!   fa2_attrib_nofill  — after first KT tile cooperative fill, skip later
//!                        global K/V loads + dequant + LDS stores (barriers kept)
//!   fa2_attrib_noexp   — every __expf replaced by x*0.5f (data flow live)
//!   fa2_attrib_nopv    — skip PV WMMAs; cheap P checksum keeps softmax live
//!   fa2_attrib_noqload — no global Q load/f32→f16; synth half16 Q from lane+chunk
//!   fa2_attrib_noqk    — keep Q load+convert; skip QK WMMA; synth scores from Q
//!
//! Fixture: batch 512 rows, positions at end of L ∈ {1024,8192,32768},
//! random finite Q8 K/V + Q. 100 interleaved HIP-event samples/twin/L, warm.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f3a \
//!     cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_attrib
//!
//! Metadata symbols the parent must read after first JIT (radiowave /
//! llvm-readobj notes on the cached .hsaco under HIPFIRE_KERNEL_CACHE):
//!   fa2_attrib_prod
//!   fa2_attrib_nofill
//!   fa2_attrib_noexp
//!   fa2_attrib_nopv
//!   fa2_attrib_noqload
//!   fa2_attrib_noqk
//!
//! Twins must stay within production VGPR count ±10% or attribution is
//! confounded — this binary prints that reminder; parent verifies metadata.

use hip_bridge::KernargBlob;
use rdna_compute::{DType, Gpu};
use std::time::Instant;

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32; // 8
const Q8_BLOCK: usize = 34;
const ROW_STRIDE: usize = N_KV * BPH * Q8_BLOCK; // 1088
const QO_ROW: usize = N_HEADS * HD; // 6144
const BATCH: usize = 512;
const L_VALUES: [usize; 3] = [1024, 8192, 32768];
const BLOCK: [u32; 3] = [128, 1, 1];
const LDS_BYTES: u32 = 32768; // KT32
const ATTR_ITERS: usize = 100;
const ATTR_WARMUP: usize = 5;
const SCALE: f32 = 1.0 / 16.0; // 1/sqrt(256)

const SYMS: [&str; 6] = [
    "fa2_attrib_prod",
    "fa2_attrib_nofill",
    "fa2_attrib_noexp",
    "fa2_attrib_nopv",
    "fa2_attrib_noqload",
    "fa2_attrib_noqk",
];

const TWIN_SRC: &str = r#####"
// F3.0 FA2 attribution twins — derived from attention_q8_0_fa2_gqa.gfx11.hip
// Diagnostic only. Modes: 0=prod 1=nofill 2=noexp 3=nopv 4=noqload 5=noqk.
#include <hip/hip_runtime.h>

#define HIPFIRE_FA2_KMODE 0
#define HIPFIRE_FA2_KT 32

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
    const int r = ((c & 1) << 3) | ((c & 2) << 1)
                | ((c & 4) >> 1) | ((c & 8) >> 3);
    return 2 * r;
}

template <int MODE>
__device__ __forceinline__ float fa2_attrib_exp(float x)
{
    if constexpr (MODE == 2) {
        // Cheap non-transcendental stand-in; keeps data flow live.
        return x * 0.5f;
    } else {
        return __expf(x);
    }
}

template <int MODE>
__device__ __forceinline__ void fa2_gqa_body_attrib(
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
    (void)q_stride;
    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + KT * 128;

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
    const int gmax = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 2]);
    const int gmin = __builtin_amdgcn_readfirstlane((int)LDS[LDS_DWORDS - 1]);
    const int seq_len = gmax + 1;

    for (int tile = 0; ; ++tile) {
        const int ktile = tile * KT;
        if (ktile >= seq_len)
            break;

        // nofill: first KT tile fills; later tiles skip global K/V load+dequant+LDS store
        // but keep both barriers so sync structure matches production.
        const bool do_fill = (MODE != 1) || (tile == 0);
        if (do_fill) {
            // K fill (Q8)
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
            // V fill
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
        }
        __syncthreads();

        if (compute) {
#pragma unroll 1
            for (int sub = 0; sub < KT_SUBS; ++sub) {
                const int kt_sub = ktile + sub * 16;
                if (kt_sub > gmax)
                    continue;
                const bool full = (kt_sub + 15) <= gmin;
                const int pos_ml =
                    (!full && qok_ml) ? positions[qr_ml] : 0;

                float8_t sacc =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                {
                    const int key = sub * 16 + ml;
                    const int rot = fa2_swiz(key & 15);
                    const int base = key * 128;
                    // Last converted Q fragment — used by noqk score synth.
                    half16_t qf_last = (_Float16)0;
                    for (int dc = 0; dc < 16; ++dc) {
                        half16_t kf;
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
                        half16_t qf;
                        if constexpr (MODE == 4) {
                            // noqload: no global Q read / f32->f16 convert.
                            // Synthesize half16 from lane + chunk (data-dependent).
                            const float base_q =
                                (float)(ml + 1) * 0.015625f
                                + (float)dc * 0.00390625f
                                + (float)half * 0.001f
                                + (float)(sub & 1) * 0.0005f;
#pragma unroll
                            for (int g = 0; g < 16; ++g)
                                qf[g] = (_Float16)(base_q + (float)g * 0.00025f);
                        } else {
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
                        qf_last = qf;
                        if constexpr (MODE == 5) {
                            // noqk: keep Q load+convert; skip QK WMMA.
                            // Fold a couple of halves into a live scalar so qf is not DCE'd.
                            // K fragment still loaded so LDS traffic matches prod for isolation of WMMA.
                            float keep = (float)qf[0] + (float)qf[8]
                                       + (float)kf[0] * 0.0f;
                            asm volatile("" : "+v"(keep) ::);
                            (void)keep;
                        } else {
                            sacc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                kf, qf, sacc);
                        }
                        asm volatile("" ::: "memory");
                    }
                    if constexpr (MODE == 5) {
                        // Synthesize score fragment from converted Q so softmax/PV stay live.
                        const float s0 = (float)qf_last[0] + (float)qf_last[1];
                        const float s1 = (float)qf_last[2] + (float)qf_last[3];
                        const float s2 = (float)qf_last[4] + (float)qf_last[5];
                        const float s3 = (float)qf_last[6] + (float)qf_last[7];
                        const float s4 = (float)qf_last[8] + (float)qf_last[9];
                        const float s5 = (float)qf_last[10] + (float)qf_last[11];
                        const float s6 = (float)qf_last[12] + (float)qf_last[13];
                        const float s7 = (float)qf_last[14] + (float)qf_last[15];
                        sacc = (float8_t){
                            s0 * 0.125f, s1 * 0.125f, s2 * 0.125f, s3 * 0.125f,
                            s4 * 0.125f, s5 * 0.125f, s6 * 0.125f, s7 * 0.125f};
                    }
                }

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
                    (m_old == -INFINITY) ? 0.0f : fa2_attrib_exp<MODE>(m_old - m_new);
                m_old = m_new;

                float earr[8];
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float e = (m_new == -INFINITY)
                        ? 0.0f
                        : fa2_attrib_exp<MODE>(sacc[j] - m_new);
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

                if constexpr (MODE == 3) {
                    // nopv: skip PV WMMAs; cheap checksum from P keeps flow live.
                    float pcs = 0.0f;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        pcs += earr[j] * (float)(j + 1);
                    // Fold into O so epilogue / l_val path cannot DCE the softmax work.
                    Ofr[0][0] += pcs;
                    Ofr[0][1] += pcs * 0.5f;
                    asm volatile("" ::: "memory");
                    (void)prow;
                } else {
#pragma unroll
                    for (int dc = 0; dc < 16; ++dc) {
                        const int d = dc * 16 + ml;
                        const int rot = fa2_swiz(d & 15);
                        const int vbase = d * V_ROW;
                        const int ps = sub * 8;
                        half16_t vf;
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
                        asm volatile("" ::: "memory");
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

#define FA2_ATTRIB_ENTRY(NAME, MODE) \
extern "C" __global__ __launch_bounds__(128, 1) void NAME( \
    const float* __restrict__ q, \
    const unsigned char* __restrict__ k_cache, \
    const unsigned char* __restrict__ v_cache, \
    float* __restrict__ out, \
    const int* __restrict__ positions, \
    int n_heads, \
    int n_kv_heads, \
    int head_dim, \
    int batch_size, \
    float scale_attn) \
{ \
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256) \
        return; \
    const int kv_h = blockIdx.y; \
    if (kv_h >= 4) \
        return; \
    const int q_base = blockIdx.x * 8; \
    if (q_base >= batch_size) \
        return; \
    fa2_gqa_body_attrib<MODE>(q, k_cache, v_cache, out, positions, batch_size, \
                 scale_attn, kv_h, q_base); \
}

FA2_ATTRIB_ENTRY(fa2_attrib_prod, 0)
FA2_ATTRIB_ENTRY(fa2_attrib_nofill, 1)
FA2_ATTRIB_ENTRY(fa2_attrib_noexp, 2)
FA2_ATTRIB_ENTRY(fa2_attrib_nopv, 3)
FA2_ATTRIB_ENTRY(fa2_attrib_noqload, 4)
FA2_ATTRIB_ENTRY(fa2_attrib_noqk, 5)

"#####;

fn prng_u32(i: u64, salt: u32) -> u32 {
    let mut x = (i as u32)
        .wrapping_mul(0x9e3779b1)
        .wrapping_add(salt)
        .wrapping_mul(0x85ebca6b);
    x ^= x >> 16;
    x = x.wrapping_mul(0xc2b2ae35);
    x ^ (x >> 16)
}

fn prng_f32(i: u64, salt: u32) -> f32 {
    (prng_u32(i, salt) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 0xff {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign << 15) | 0x7c00 | m;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return (sign << 15) | 0x7c00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

fn pack_q8_block(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let s = f32_to_f16_bits(scale);
    out[0] = (s & 0xff) as u8;
    out[1] = (s >> 8) as u8;
    for w in 0..8 {
        let mut u = 0u32;
        for c in 0..4 {
            let v = codes[w * 4 + c] as u8;
            u |= (v as u32) << (c * 8);
        }
        out[2 + w * 4..2 + w * 4 + 4].copy_from_slice(&u.to_le_bytes());
    }
    out
}

fn fill_q8_cache(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * ROW_STRIDE];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            for b in 0..BPH {
                let mut codes = [0i8; 32];
                for c in 0..32 {
                    let r = prng_u32((g * 64 + kv * 8 + b) as u64 * 32 + c as u64, salt);
                    codes[c] = ((r % 61) as i8).wrapping_sub(30);
                }
                let scale =
                    0.01 + (prng_u32((g * 32 + kv * 8 + b) as u64, salt ^ 0xabc) % 100) as f32 * 0.001;
                let blk = pack_q8_block(scale, &codes);
                let off = g * ROW_STRIDE + (kv * BPH + b) * Q8_BLOCK;
                buf[off..off + 34].copy_from_slice(&blk);
            }
        }
    }
    buf
}

fn fill_q(batch: usize, salt: u32) -> Vec<f32> {
    let n = batch * N_HEADS * HD;
    (0..n)
        .map(|i| prng_f32(i as u64, salt) * 0.5)
        .collect()
}

fn fill_positions_end_of_l(batch: usize, l: usize) -> Vec<i32> {
    assert!(l >= batch, "L must cover batch positions");
    let start = l - batch;
    (0..batch).map(|b| (start + b) as i32).collect()
}

fn positions_to_bytes(pos: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pos.len() * 4);
    for &p in pos {
        bytes.extend_from_slice(&p.to_le_bytes());
    }
    bytes
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn ensure_twins(gpu: &mut Gpu) {
    const MOD: &str = "halo_fa2_f30_attrib";
    for sym in SYMS {
        gpu.ensure_kernel_public(MOD, TWIN_SRC, sym)
            .unwrap_or_else(|e| panic!("compile {sym}: {e:?}"));
        eprintln!("F3.0 meta: compiled twin symbol {sym} (module {MOD})");
    }
}

fn launch_twin(
    gpu: &mut Gpu,
    name: &str,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    batch: i32,
) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    let mut args = KernargBlob::new();
    args.push_ptr(d_q.buf.as_ptr());
    args.push_ptr(d_k.buf.as_ptr());
    args.push_ptr(d_v.buf.as_ptr());
    args.push_ptr(d_out.buf.as_ptr());
    args.push_ptr(d_pos.buf.as_ptr());
    args.push_i32(N_HEADS as i32);
    args.push_i32(N_KV as i32);
    args.push_i32(HD as i32);
    args.push_i32(batch);
    args.push_f32(SCALE);
    let grid_x = (batch as u32).div_ceil(8);
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.launch_kernel_blob(
        name,
        [grid_x, 4, 1],
        BLOCK,
        LDS_BYTES,
        args.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch twin {name}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0 // µs
}

fn run_attribution(gpu: &mut Gpu) {
    eprintln!("\n=== F3.0 FA2 attribution twins (100 interleaved event samples) ===");
    eprintln!(
        "shape H{N_HEADS}/KV{N_KV}/D{HD} batch={BATCH}  L∈{L_VALUES:?}  positions at end of L"
    );
    eprintln!("symbols: {}", SYMS.join(", "));
    eprintln!(
        "WARNING: parent MUST verify all six twins stay within production VGPR ±10%; \
         otherwise attribution is confounded. Read hsaco metadata under HIPFIRE_KERNEL_CACHE."
    );

    for &l in &L_VALUES {
        eprintln!("\n-- L={l} (positions {}..{}) --", l - BATCH, l - 1);
        let t_pack = Instant::now();
        let pos_h = fill_positions_end_of_l(BATCH, l);
        let q_h = fill_q(BATCH, 0xF30A);
        let k_h = fill_q8_cache(l, 0xF30B);
        let v_h = fill_q8_cache(l, 0xF30C);
        eprintln!(
            "packed Q={} f32  K={} B  V={} B  pos={} i32 in {:.2}s",
            q_h.len(),
            k_h.len(),
            v_h.len(),
            pos_h.len(),
            t_pack.elapsed().as_secs_f64()
        );

        let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
        let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
        let d_q = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        let pos_bytes = positions_to_bytes(&pos_h);
        let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();

        // Separate outs so arms do not contend mid-interleave.
        let d_outs: Vec<_> = (0..SYMS.len())
            .map(|_| gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap())
            .collect();

        // Warmup
        for _ in 0..ATTR_WARMUP {
            for (si, sym) in SYMS.iter().enumerate() {
                let _ = launch_twin(
                    gpu,
                    sym,
                    &d_q,
                    &d_k,
                    &d_v,
                    &d_outs[si],
                    &d_pos,
                    BATCH as i32,
                );
            }
        }

        let mut samples: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::new());
        for _ in 0..ATTR_ITERS {
            for (si, sym) in SYMS.iter().enumerate() {
                let us = launch_twin(
                    gpu,
                    sym,
                    &d_q,
                    &d_k,
                    &d_v,
                    &d_outs[si],
                    &d_pos,
                    BATCH as i32,
                );
                samples[si].push(us);
            }
        }

        let mut med = [0.0f64; 6];
        for i in 0..6 {
            med[i] = median_f64(&mut samples[i]);
            let mn = samples[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = samples[i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {:<20} median_us={:.2}  min={:.2} max={:.2}",
                SYMS[i], med[i], mn, mx
            );
        }
        let prod = med[0].max(1e-30);
        let fill = (med[0] - med[1]) / prod;
        let exp_a = (med[0] - med[2]) / prod;
        let pv = (med[0] - med[3]) / prod;
        let qload = (med[0] - med[4]) / prod;
        let qk = (med[0] - med[5]) / prod;
        eprintln!(
            "  fill =(prod-nofill)/prod  = {fill:.4}  ({:.2}%)",
            fill * 100.0
        );
        eprintln!(
            "  exp  =(prod-noexp)/prod   = {exp_a:.4}  ({:.2}%)",
            exp_a * 100.0
        );
        eprintln!(
            "  pv   =(prod-nopv)/prod    = {pv:.4}  ({:.2}%)",
            pv * 100.0
        );
        eprintln!(
            "  qload=(prod-noqload)/prod = {qload:.4}  ({:.2}%)",
            qload * 100.0
        );
        eprintln!(
            "  qk   =(prod-noqk)/prod    = {qk:.4}  ({:.2}%)",
            qk * 100.0
        );
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "tmp_fa2_attrib arch={}  (F3.0 FA2 fill/exp/PV attribution)",
        gpu.arch
    );
    if !matches!(
        gpu.arch.as_str(),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151"
    ) {
        eprintln!("skip: need gfx11 family for FA2 WMMA, got {}", gpu.arch);
        eprintln!("Halo command (parent runs gates):");
        eprintln!(
            "  HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \\"
        );
        eprintln!("    HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f3a \\");
        eprintln!(
            "    cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_attrib"
        );
        eprintln!("symbols: {}", SYMS.join(", "));
        return;
    }

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    }

    ensure_twins(&mut gpu);

    eprintln!("\nF3.0: first JIT done. Parent must read code-object metadata for:");
    for s in SYMS {
        eprintln!("  {s}");
    }
    eprintln!("HIPFIRE_KERNEL_CACHE entries live under the process cache dir (see env).");
    eprintln!(
        "VGPR WARNING: twins MUST stay within production allocated VGPR ±10% or \
         attribution is confounded — parent verifies hsaco notes before trusting fractions."
    );

    run_attribution(&mut gpu);

    eprintln!("\nF3.0 complete. symbols: {}", SYMS.join(", "));
}
