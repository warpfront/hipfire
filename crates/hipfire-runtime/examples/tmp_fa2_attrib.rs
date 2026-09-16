//! F3.0 — FA2 gfx11 fill/exp/PV/Q attribution + candidate twins.
//!
//! Diagnostic only. Does not edit production kernels.
//! Derived from kernels/src/attention_q8_0_fa2_gqa.gfx11.hip (Q8 entry).
//! Shipping shape: H24 / KV4 / D256 (not 48/8/128).
//!
//! Twins of the Q8 entry, distinct symbols:
//!   fa2_attrib_prod         — unmodified production body
//!   fa2_attrib_nofill       — after first KT tile fill, skip later K/V global fill
//!   fa2_attrib_noexp        — every __expf → x*0.5f
//!   fa2_attrib_nopv         — skip PV WMMAs; P checksum
//!   fa2_attrib_noqload      — synth Q fragment (no global Q load)
//!   fa2_attrib_noqk         — keep Q load; skip QK WMMA; synth scores
//!   fa2_attrib_qshare2      — Q load once/dim chunk for both KT32 subtles (bit-exact cand)
//!   fa2_attrib_qf16         — host-pre-rounded f16 Q, aligned half8×2 load (bit-exact cand)
//!   fa2_attrib_qshare2_f16  — qshare2 + qf16 (bit-exact cand)
//!
//! Production Q f32→f16: C cast `(_Float16)f` (device v_cvt_f16_f32, RNE).
//! Host f16 Q packing matches that RNE.
//!
//! Fixture: batch 512, positions at end of L ∈ {1024,8192,32768},
//! random finite Q8 K/V + Q. 100 interleaved HIP-event samples/twin/L, warm.
//! Candidates g/h/i: bitwise_eq of out vs prod printed per L.
//!
//! Halo (hipx device 1):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f3a \
//!     cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_attrib
//!
//! Arm `ship` (argv[1]=="ship"): launch SHIPPING kernels via runtime
//! direct launchers vs frozen prod twins (Q8: fa2_attrib_prod; fwht3:
//! fa2_attrib_prod_fwht3 frozen from HEAD pre-qshare2). Prints bitwise_eq
//! out (and, for fwht3, ship-Q-unchanged vs input) + interleaved medians
//! ship vs prod per L. F4b: the shipping launchers pre-convert internally
//! and never mutate Q.
//!
//! Halo ship:
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f4a \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_fa2_attrib -- ship
//!
//! Arm `rotcheck` (argv[1]=="rotcheck"): fwht3 Q-rotation isolation (L=1024
//! fixture). Runs the frozen prod twin once and downloads the rotated f32 Q;
//! runs the shipping pre-convert entry raw (do_rotate=1) on a fresh ORIGINAL
//! Q into f16 scratch; runs the example-local debug entry
//! `attention_fa2_q_preconvert_f32_debug_gfx11` (production pre-convert copy
//! storing the rotated f32 BEFORE the cast). Prints the first 8 differing
//! f32 elements (row, head, dim, prod bits, new bits), the count, and whether
//! casting prod's rotated f32 with the same RNE cast reproduces the f16
//! scratch (isolates rotation vs cast).
//!
//! Halo rotcheck:
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f4a \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_fa2_attrib -- rotcheck
//!
//! Arm `iu8probe` (argv[1]=="iu8probe", F5.1): example-local probe of
//! `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (one-hot / signed /
//! byte-position / duplicate-lane / clamp-flag / fold cases) with a host
//! scalar reference. Prints the byte order + lane maps and pass/fail;
//! machine-readable maps go to stdout as `IU8MAP {...}` JSON lines for the
//! kernel owner. Symbols: `iu8probe_a1b1c0` (+a0/b0/c1 variants).
//!
//! Halo iu8probe (parent runs first):
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f5a \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_fa2_attrib -- iu8probe
//!
//! Arm `i8` (argv[1]=="i8", F5.5): lab i8-QK candidate (F5a) vs the shipping
//! arm on byte-identical K/V/positions/f32 Q at L1024/8192/32768, both K
//! modes. Exact gfx1151 only. Reports max abs/rel/RMS error, finite and
//! masked-row counts, immutable-Q check, and 100-interleaved HIP-event
//! medians; exports both outputs + a JSON sidecar per L/mode for offline
//! KLD-style analysis (`FA2_I8_OUTDIR`, default /tmp). Until the F5.3/F5.4
//! entries/launchers land, the arm resolves the frozen §8.2 symbols at
//! runtime and skips cleanly when absent (the example still builds).
//!
//! Halo i8:
//!   HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \
//!     HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f5a \
//!     cargo run --release -p hipfire-runtime --features lab \
//!       --example tmp_fa2_attrib -- i8
//!
//! Metadata symbols:
//!   fa2_attrib_prod
//!   fa2_attrib_nofill
//!   fa2_attrib_noexp
//!   fa2_attrib_nopv
//!   fa2_attrib_noqload
//!   fa2_attrib_noqk
//!   fa2_attrib_qshare2
//!   fa2_attrib_qf16
//!   fa2_attrib_qshare2_f16
//!   fa2_attrib_prod_fwht3   (frozen pre-change fwht3 entry)

use hip_bridge::KernargBlob;
use rdna_compute::{gen_fwht_signs, DType, Gpu};
use std::time::Instant;

const N_HEADS: usize = 24;
const N_KV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32;
const Q8_BLOCK: usize = 34;
const ROW_STRIDE: usize = N_KV * BPH * Q8_BLOCK;
const QO_ROW: usize = N_HEADS * HD;
const BATCH: usize = 512;
const L_VALUES: [usize; 3] = [1024, 8192, 32768];
const BLOCK: [u32; 3] = [128, 1, 1];
const LDS_BYTES: u32 = 32768;
const ATTR_ITERS: usize = 100;
const ATTR_WARMUP: usize = 5;
const SCALE: f32 = 1.0 / 16.0;
const FWHT3_HEAD: usize = 100;
const FWHT3_POS: usize = N_KV * FWHT3_HEAD; // 400
const PROD_Q8_SYM: &str = "fa2_attrib_prod";
const PROD_FWHT3_SYM: &str = "fa2_attrib_prod_fwht3";
const PROD_FWHT3_MOD: &str = "halo_fa2_f4a_prod_fwht3";

/// Frozen pre-qshare2 fwht3 entry (HEAD kernel, renamed symbol) + turbo_common.
const PROD_FWHT3_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KMODE 3\n",
    "#define HIPFIRE_FA2_KT 32\n",
    include_str!("../../../kernels/src/turbo_common.h"),
    include_str!("tmp_fa2_attrib_prod_fwht3.hip"),
);
/// F4b rotcheck: example-local f32 debug entry. Mirrors the production
/// pre-convert exactly (same loads, same `fwht_shfl_forward_256` call, same
/// lane mapping) but stores the rotated f32 BEFORE the cast, so the rotation
/// can be compared bit-for-bit against the frozen prod twin's in-place Q.
const PRECONVERT_F32_DEBUG_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KMODE 3\n",
    "#define HIPFIRE_FA2_KT 32\n",
    include_str!("../../../kernels/src/turbo_common.h"),
    r#####"
#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(128, 1) void attention_fa2_q_preconvert_f32_debug_gfx11(
    const float* __restrict__ q,
    float* __restrict__ q32,
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
    float* wp = q32 + (unsigned)qr * 6144u + (unsigned)h * 256u;
    float v0 = rp[lane * 8 + 0], v1 = rp[lane * 8 + 1];
    float v2 = rp[lane * 8 + 2], v3 = rp[lane * 8 + 3];
    float v4 = rp[lane * 8 + 4], v5 = rp[lane * 8 + 5];
    float v6 = rp[lane * 8 + 6], v7 = rp[lane * 8 + 7];
    if (do_rotate)
        fwht_shfl_forward_256(v0, v1, v2, v3, v4, v5, v6, v7,
                              signs1, signs2, lane);
    wp[lane * 8 + 0] = v0; wp[lane * 8 + 1] = v1;
    wp[lane * 8 + 2] = v2; wp[lane * 8 + 3] = v3;
    wp[lane * 8 + 4] = v4; wp[lane * 8 + 5] = v5;
    wp[lane * 8 + 6] = v6; wp[lane * 8 + 7] = v7;
}
"#####,
);
const DEBUG_F32_SYM: &str = "attention_fa2_q_preconvert_f32_debug_gfx11";
/// F4b rotcheck: shipping module sources with the byte-identical recipe the
/// rdna-compute launchers use (attention.rs). Q8 = KT32 + raw kernel file
/// (KMODE defaults to 0, rotation compiled out); fwht3 = KT32 + KMODE=3 +
/// turbo_common + kernel file. rotcheck ensures the Q8 module FIRST to mirror
/// the ship arm's Q8-first order.
const SHIP_Q8_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KT 32\n",
    include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"),
);
const SHIP_FWHT3_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KT 32\n",
    "#define HIPFIRE_FA2_KMODE 3\n",
    include_str!("../../../kernels/src/turbo_common.h"),
    include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"),
);
const SHIP_Q8_PRECONVERT_SYM: &str = "attention_fa2_q_preconvert_gfx11";
const SHIP_FWHT3_PRECONVERT_SYM: &str = "attention_fa2_q_preconvert_fwht3_gfx11";
const SHIP_FWHT3_BODY_SYM: &str = "attention_q8_0_fa2_gqa_fwht3k_gfx11";

const SYMS: [&str; 9] = [
    "fa2_attrib_prod",
    "fa2_attrib_nofill",
    "fa2_attrib_noexp",
    "fa2_attrib_nopv",
    "fa2_attrib_noqload",
    "fa2_attrib_noqk",
    "fa2_attrib_qshare2",
    "fa2_attrib_qf16",
    "fa2_attrib_qshare2_f16",
];

/// Candidates that must be bit-exact vs prod (indices into SYMS).
const CAND_IDX: [usize; 3] = [6, 7, 8];

const TWIN_SRC: &str = r#####"

// F3.0 FA2 attribution twins — derived from attention_q8_0_fa2_gqa.gfx11.hip
// Modes: 0=prod 1=nofill 2=noexp 3=nopv 4=noqload 5=noqk
//        6=qshare2 7=qf16 8=qshare2_f16
// Q f32→f16 conversion in production is the C cast (_Float16)f  (device v_cvt_f16_f32 RNE).
#include <hip/hip_runtime.h>

#define HIPFIRE_FA2_KMODE 0
#define HIPFIRE_FA2_KT 32

typedef _Float16 __attribute__((ext_vector_type(8))) half8_t;
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef float __attribute__((ext_vector_type(4))) float4_t;
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
    if constexpr (MODE == 2)
        return x * 0.5f;
    else
        return __expf(x);
}

// MODE bits: Q_F16 if MODE==7||MODE==8; QSHARE if MODE==6||MODE==8
template <int MODE>
__device__ __forceinline__ constexpr bool fa2_mode_qf16()
{
    return MODE == 7 || MODE == 8;
}
template <int MODE>
__device__ __forceinline__ constexpr bool fa2_mode_qshare()
{
    return MODE == 6 || MODE == 8;
}

template <int MODE>
__device__ __forceinline__ half16_t fa2_load_q_frag(
    const float* __restrict__ q_f32,
    const _Float16* __restrict__ q_f16,
    unsigned qq_base,
    int dc,
    int ml,
    int half,
    int sub,
    bool qok_ml)
{
    half16_t qf;
    if constexpr (MODE == 4) {
        const float base_q =
            (float)(ml + 1) * 0.015625f
            + (float)dc * 0.00390625f
            + (float)half * 0.001f
            + (float)(sub & 1) * 0.0005f;
#pragma unroll
        for (int g = 0; g < 16; ++g)
            qf[g] = (_Float16)(base_q + (float)g * 0.00025f);
    } else if constexpr (fa2_mode_qf16<MODE>()) {
        // Pre-rounded f16 Q: two aligned 16-byte half8 loads → half16 fragment.
        const _Float16* qd = q_f16 + qq_base + (unsigned)(dc * 16);
        if (qok_ml) {
            const half8_t h0 = *(const half8_t*)(qd + 0);
            const half8_t h1 = *(const half8_t*)(qd + 8);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                qf[i] = h0[i];
                qf[8 + i] = h1[i];
            }
        } else {
#pragma unroll
            for (int i = 0; i < 16; ++i)
                qf[i] = (_Float16)0;
        }
    } else {
        // Production: 4× float4 global loads + (_Float16) cast (RNE cvt).
        const float* qd = q_f32 + qq_base + (unsigned)(dc * 16);
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
    return qf;
}

template <int MODE>
__device__ __forceinline__ half16_t fa2_load_k_frag(
    const uint32_t* __restrict__ Kdw,
    int sub,
    int ml,
    int dc)
{
    half16_t kf;
    const int key = sub * 16 + ml;
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
    return kf;
}

// Softmax + PV for one 16-key subtile (production order). Updates m_old, l_val, Ofr.
template <int MODE>
__device__ __forceinline__ void fa2_softmax_pv_sub(
    float8_t sacc,
    int kt_sub,
    int gmin,
    int half,
    bool qok_ml,
    int pos_ml,
    float scale_attn,
    float& m_old,
    float& l_val,
    float8_t Ofr[16],
    const uint32_t* __restrict__ Vdw,
    int V_ROW,
    int V_MASK,
    int sub,
    int ml)
{
    const bool full = (kt_sub + 15) <= gmin;
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
        float pcs = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            pcs += earr[j] * (float)(j + 1);
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
                const int32x2_t pair = *(const int32x2_t*)(Vdw + off);
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

template <int MODE>
__device__ __forceinline__ void fa2_gqa_body_attrib(
    const float* __restrict__ q_f32,
    const _Float16* __restrict__ q_f16,
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

    const int row_stride = 4 * (256 / 32) * 34;
    const int kv_blk = kv_h * (256 / 32);

    extern __shared__ uint32_t LDS[];
    uint32_t* Kdw = LDS;
    uint32_t* Vdw = LDS + KT * 128;

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

        const bool do_fill = (MODE != 1) || (tile == 0);
        if (do_fill) {
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
                uint16_t s0u = 0, s1u = 0;
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
            if constexpr (fa2_mode_qshare<MODE>()) {
                // qshare2 / qshare2_f16: one Q load per dim chunk, both subtles' QK,
                // then softmax/PV sub0 then sub1 (online order unchanged).
                float8_t sacc0 =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                float8_t sacc1 =
                    (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                const int kt_sub0 = ktile;
                const int kt_sub1 = ktile + 16;
                const bool do0 = kt_sub0 <= gmax;
                const bool do1 = kt_sub1 <= gmax;
                if (do0 || do1) {
                    for (int dc = 0; dc < 16; ++dc) {
                        half16_t qf = fa2_load_q_frag<MODE>(
                            q_f32, q_f16, qq_base, dc, ml, half, 0, qok_ml);
                        if (do0) {
                            half16_t kf0 = fa2_load_k_frag<MODE>(Kdw, 0, ml, dc);
                            sacc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                kf0, qf, sacc0);
                        }
                        if (do1) {
                            half16_t kf1 = fa2_load_k_frag<MODE>(Kdw, 1, ml, dc);
                            sacc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                                kf1, qf, sacc1);
                        }
                        asm volatile("" ::: "memory");
                    }
                    if (do0) {
                        const int pos_ml =
                            (!((kt_sub0 + 15) <= gmin) && qok_ml) ? positions[qr_ml] : 0;
                        fa2_softmax_pv_sub<MODE>(
                            sacc0, kt_sub0, gmin, half, qok_ml, pos_ml, scale_attn,
                            m_old, l_val, Ofr, Vdw, V_ROW, V_MASK, 0, ml);
                    }
                    if (do1) {
                        const int pos_ml =
                            (!((kt_sub1 + 15) <= gmin) && qok_ml) ? positions[qr_ml] : 0;
                        fa2_softmax_pv_sub<MODE>(
                            sacc1, kt_sub1, gmin, half, qok_ml, pos_ml, scale_attn,
                            m_old, l_val, Ofr, Vdw, V_ROW, V_MASK, 1, ml);
                    }
                }
            } else {
#pragma unroll 1
                for (int sub = 0; sub < KT_SUBS; ++sub) {
                    const int kt_sub = ktile + sub * 16;
                    if (kt_sub > gmax)
                        continue;
                    const int pos_ml =
                        (!((kt_sub + 15) <= gmin) && qok_ml) ? positions[qr_ml] : 0;

                    float8_t sacc =
                        (float8_t){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    half16_t qf_last = (_Float16)0;
                    for (int dc = 0; dc < 16; ++dc) {
                        half16_t kf = fa2_load_k_frag<MODE>(Kdw, sub, ml, dc);
                        half16_t qf = fa2_load_q_frag<MODE>(
                            q_f32, q_f16, qq_base, dc, ml, half, sub, qok_ml);
                        qf_last = qf;
                        if constexpr (MODE == 5) {
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
                    fa2_softmax_pv_sub<MODE>(
                        sacc, kt_sub, gmin, half, qok_ml, pos_ml, scale_attn,
                        m_old, l_val, Ofr, Vdw, V_ROW, V_MASK, sub, ml);
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

// f32-Q entries (q_f16 unused; pass nullptr)
#define FA2_ATTRIB_ENTRY_F32(NAME, MODE) \
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
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256) return; \
    const int kv_h = blockIdx.y; \
    if (kv_h >= 4) return; \
    const int q_base = blockIdx.x * 8; \
    if (q_base >= batch_size) return; \
    fa2_gqa_body_attrib<MODE>(q, (_Float16*)nullptr, k_cache, v_cache, out, positions, \
        batch_size, scale_attn, kv_h, q_base); \
}

// f16-Q entries (q is half*)
#define FA2_ATTRIB_ENTRY_F16(NAME, MODE) \
extern "C" __global__ __launch_bounds__(128, 1) void NAME( \
    const _Float16* __restrict__ q, \
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
    if (n_heads != 24 || n_kv_heads != 4 || head_dim != 256) return; \
    const int kv_h = blockIdx.y; \
    if (kv_h >= 4) return; \
    const int q_base = blockIdx.x * 8; \
    if (q_base >= batch_size) return; \
    fa2_gqa_body_attrib<MODE>((const float*)nullptr, q, k_cache, v_cache, out, positions, \
        batch_size, scale_attn, kv_h, q_base); \
}

FA2_ATTRIB_ENTRY_F32(fa2_attrib_prod, 0)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_nofill, 1)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_noexp, 2)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_nopv, 3)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_noqload, 4)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_noqk, 5)
FA2_ATTRIB_ENTRY_F32(fa2_attrib_qshare2, 6)
FA2_ATTRIB_ENTRY_F16(fa2_attrib_qf16, 7)
FA2_ATTRIB_ENTRY_F16(fa2_attrib_qshare2_f16, 8)

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

/// IEEE754 f32→f16 round-to-nearest-even (matches AMD device `(_Float16)f` / v_cvt_f16_f32).
fn f32_to_f16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;

    if exp == 0xff {
        // Inf/NaN
        if mant != 0 {
            // quiet NaN with some payload bits
            return sign | 0x7e00 | ((mant >> 13) as u16 & 0x1ff);
        }
        return sign | 0x7c00;
    }

    // Rebias: f32 bias 127 → f16 bias 15
    let half_exp = exp - 127 + 15;

    if exp == 0 {
        // f32 denormal/zero → f16 zero (underflow)
        return sign;
    }

    if half_exp >= 0x1f {
        // overflow → inf
        return sign | 0x7c00;
    }

    if half_exp <= 0 {
        // f16 denormal or underflows to zero
        if half_exp < -10 {
            return sign;
        }
        // Implicit leading 1 of normal f32 becomes explicit for denormal f16.
        let mant_full = mant | 0x800000;
        // Shift so the f16 denormal exponent lands; round bit at bit (14 - half_exp) from top of 24-bit.
        let shift = (14 - half_exp) as u32; // 14..24
        let rounded = (mant_full >> (shift + 1)) as u32;
        let round_bit = (mant_full >> shift) & 1;
        let sticky = if (mant_full & ((1u32 << shift) - 1)) != 0 { 1 } else { 0 };
        let mut out_m = rounded;
        if round_bit == 1 && (sticky == 1 || (out_m & 1) == 1) {
            out_m += 1;
        }
        return sign | (out_m as u16 & 0x3ff);
    }

    // Normal: 10-bit mantissa with RNE from 23-bit
    let mant16 = mant >> 13;
    let round_bit = (mant >> 12) & 1;
    let sticky = if (mant & 0xfff) != 0 { 1 } else { 0 };
    let mut out_m = mant16;
    let mut out_e = half_exp as u32;
    if round_bit == 1 && (sticky == 1 || (out_m & 1) == 1) {
        out_m += 1;
        if out_m == 0x400 {
            // mantissa overflow → bump exp
            out_m = 0;
            out_e += 1;
            if out_e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((out_e as u16) << 10) | (out_m as u16)
}

fn pack_q8_block(scale: f32, codes: &[i8; 32]) -> [u8; 34] {
    let mut out = [0u8; 34];
    let s = f32_to_f16_rne(scale);
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
    (0..n).map(|i| prng_f32(i as u64, salt) * 0.5).collect()
}

/// Host-side f16 Q with the SAME RNE as production `(_Float16)f`.
fn pack_q_f16_rne(q_f32: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(q_f32.len() * 2);
    for &x in q_f32 {
        bytes.extend_from_slice(&f32_to_f16_rne(x).to_le_bytes());
    }
    bytes
}

fn fill_positions_end_of_l(batch: usize, l: usize) -> Vec<i32> {
    assert!(l >= batch);
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

fn is_f16_q_sym(sym: &str) -> bool {
    sym == "fa2_attrib_qf16" || sym == "fa2_attrib_qshare2_f16"
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
    gpu.launch_kernel_blob(name, [grid_x, 4, 1], BLOCK, LDS_BYTES, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch twin {name}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

fn download_f32(gpu: &Gpu, t: &rdna_compute::GpuTensor, n: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; n * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .unwrap_or_else(|e| panic!("download_f32: {e:?}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn bitwise_eq_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn run_attribution(gpu: &mut Gpu) {
    eprintln!("\n=== F3.0 FA2 attribution twins (100 interleaved event samples) ===");
    eprintln!(
        "shape H{N_HEADS}/KV{N_KV}/D{HD} batch={BATCH}  L∈{L_VALUES:?}  positions at end of L"
    );
    eprintln!("symbols: {}", SYMS.join(", "));
    eprintln!(
        "Production Q cvt: (_Float16)f  → device v_cvt_f16_f32 RNE; host pack_q_f16_rne matches."
    );
    eprintln!(
        "WARNING: parent MUST verify all twins stay within production VGPR ±10%; \
         otherwise attribution is confounded. Read hsaco metadata under HIPFIRE_KERNEL_CACHE."
    );

    for &l in &L_VALUES {
        eprintln!("\n-- L={l} (positions {}..{}) --", l - BATCH, l - 1);
        let t_pack = Instant::now();
        let pos_h = fill_positions_end_of_l(BATCH, l);
        let q_h = fill_q(BATCH, 0xF30A);
        let q_f16 = pack_q_f16_rne(&q_h);
        let k_h = fill_q8_cache(l, 0xF30B);
        let v_h = fill_q8_cache(l, 0xF30C);
        eprintln!(
            "packed Q_f32={}  Q_f16={} B  K={} B  V={} B  pos={} in {:.2}s",
            q_h.len(),
            q_f16.len(),
            k_h.len(),
            v_h.len(),
            pos_h.len(),
            t_pack.elapsed().as_secs_f64()
        );

        let d_k = gpu.upload_raw(&k_h, &[k_h.len()]).unwrap();
        let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
        let d_q = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        let d_q_f16 = gpu.upload_raw(&q_f16, &[q_f16.len()]).unwrap();
        let pos_bytes = positions_to_bytes(&pos_h);
        let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();

        let d_outs: Vec<_> = (0..SYMS.len())
            .map(|_| gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap())
            .collect();

        let q_for = |sym: &str| -> &rdna_compute::GpuTensor {
            if is_f16_q_sym(sym) {
                &d_q_f16
            } else {
                &d_q
            }
        };

        // --- bit-exact check for candidates vs prod (untimed) ---
        let _ = launch_twin(
            gpu,
            SYMS[0],
            q_for(SYMS[0]),
            &d_k,
            &d_v,
            &d_outs[0],
            &d_pos,
            BATCH as i32,
        );
        let prod_out = download_f32(gpu, &d_outs[0], BATCH * QO_ROW);
        for &ci in &CAND_IDX {
            // zero out then launch
            let z = vec![0u8; BATCH * QO_ROW * 4];
            gpu.hip
                .memcpy_htod(&d_outs[ci].buf, &z)
                .unwrap_or_else(|e| panic!("zero out: {e:?}"));
            let _ = launch_twin(
                gpu,
                SYMS[ci],
                q_for(SYMS[ci]),
                &d_k,
                &d_v,
                &d_outs[ci],
                &d_pos,
                BATCH as i32,
            );
            let cand_out = download_f32(gpu, &d_outs[ci], BATCH * QO_ROW);
            let eq = bitwise_eq_f32(&prod_out, &cand_out);
            let mut n_diff = 0usize;
            let mut max_abs = 0.0f32;
            for (a, b) in prod_out.iter().zip(cand_out.iter()) {
                if a.to_bits() != b.to_bits() {
                    n_diff += 1;
                    max_abs = max_abs.max((a - b).abs());
                }
            }
            eprintln!(
                "  bitwise_eq L={l} {sym:<24} = {eq}  n_diff={n_diff} max_abs={max_abs:.6e}",
                sym = SYMS[ci]
            );
        }

        // Warmup
        for _ in 0..ATTR_WARMUP {
            for (si, sym) in SYMS.iter().enumerate() {
                let _ = launch_twin(
                    gpu,
                    sym,
                    q_for(sym),
                    &d_k,
                    &d_v,
                    &d_outs[si],
                    &d_pos,
                    BATCH as i32,
                );
            }
        }

        let mut samples: [Vec<f64>; 9] = std::array::from_fn(|_| Vec::new());
        for _ in 0..ATTR_ITERS {
            for (si, sym) in SYMS.iter().enumerate() {
                let us = launch_twin(
                    gpu,
                    sym,
                    q_for(sym),
                    &d_k,
                    &d_v,
                    &d_outs[si],
                    &d_pos,
                    BATCH as i32,
                );
                samples[si].push(us);
            }
        }

        let mut med = [0.0f64; 9];
        for i in 0..9 {
            med[i] = median_f64(&mut samples[i]);
            let mn = samples[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = samples[i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "  {:<24} median_us={:.2}  min={:.2} max={:.2}",
                SYMS[i], med[i], mn, mx
            );
        }
        let prod = med[0].max(1e-30);
        let fill = (med[0] - med[1]) / prod;
        let exp_a = (med[0] - med[2]) / prod;
        let pv = (med[0] - med[3]) / prod;
        let qload = (med[0] - med[4]) / prod;
        let qk = (med[0] - med[5]) / prod;
        let r_qshare2 = med[6] / prod;
        let r_qf16 = med[7] / prod;
        let r_both = med[8] / prod;
        eprintln!(
            "  fill =(prod-nofill)/prod   = {fill:.4}  ({:.2}%)",
            fill * 100.0
        );
        eprintln!(
            "  exp  =(prod-noexp)/prod    = {exp_a:.4}  ({:.2}%)",
            exp_a * 100.0
        );
        eprintln!(
            "  pv   =(prod-nopv)/prod     = {pv:.4}  ({:.2}%)",
            pv * 100.0
        );
        eprintln!(
            "  qload=(prod-noqload)/prod  = {qload:.4}  ({:.2}%)",
            qload * 100.0
        );
        eprintln!(
            "  qk   =(prod-noqk)/prod     = {qk:.4}  ({:.2}%)",
            qk * 100.0
        );
        eprintln!(
            "  ratio qshare2/prod         = {r_qshare2:.4}  ({:.2}% of prod time)",
            r_qshare2 * 100.0
        );
        eprintln!(
            "  ratio qf16/prod            = {r_qf16:.4}  ({:.2}% of prod time)",
            r_qf16 * 100.0
        );
        eprintln!(
            "  ratio qshare2_f16/prod     = {r_both:.4}  ({:.2}% of prod time)",
            r_both * 100.0
        );
        eprintln!(
            "  speedup qshare2            = {:.4}  ({:.2}% faster)",
            1.0 - r_qshare2,
            (1.0 - r_qshare2) * 100.0
        );
        eprintln!(
            "  speedup qf16               = {:.4}  ({:.2}% faster)",
            1.0 - r_qf16,
            (1.0 - r_qf16) * 100.0
        );
        eprintln!(
            "  speedup qshare2_f16        = {:.4}  ({:.2}% faster)",
            1.0 - r_both,
            (1.0 - r_both) * 100.0
        );
    }
}


fn fill_fwht3_k(seq_len: usize, salt: u32) -> Vec<u8> {
    let mut buf = vec![0u8; seq_len * FWHT3_POS];
    for g in 0..seq_len {
        for kv in 0..N_KV {
            let base = g * FWHT3_POS + kv * FWHT3_HEAD;
            let cnorm = 0.05 + (prng_u32((g * 4 + kv) as u64, salt) % 200) as f32 * 0.001;
            buf[base..base + 4].copy_from_slice(&cnorm.to_le_bytes());
            for g3 in 0..32 {
                let mut packed = 0u32;
                for i in 0..8 {
                    let code =
                        (prng_u32((g * 256 + kv * 64 + g3 * 8 + i) as u64, salt) % 8) as u32;
                    packed |= code << (3 * i);
                }
                let bytes = packed.to_le_bytes();
                buf[base + 4 + g3 * 3..base + 4 + g3 * 3 + 3].copy_from_slice(&bytes[..3]);
            }
        }
    }
    buf
}

fn download_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor, n_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; n_bytes];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .unwrap_or_else(|e| panic!("download_bytes: {e:?}"));
    bytes
}

fn bitwise_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    a == b
}

fn ensure_prod_fwht3(gpu: &mut Gpu) {
    gpu.ensure_kernel_public(PROD_FWHT3_MOD, PROD_FWHT3_SRC, PROD_FWHT3_SYM)
        .unwrap_or_else(|e| panic!("compile {PROD_FWHT3_SYM}: {e:?}"));
    eprintln!("F4a meta: compiled frozen twin symbol {PROD_FWHT3_SYM} (module {PROD_FWHT3_MOD})");
}

fn launch_prod_fwht3(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    d_s1: &rdna_compute::GpuTensor,
    d_s2: &rdna_compute::GpuTensor,
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
    args.push_ptr(d_s1.buf.as_ptr());
    args.push_ptr(d_s2.buf.as_ptr());
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
        PROD_FWHT3_SYM,
        [grid_x, 4, 1],
        BLOCK,
        LDS_BYTES,
        args.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch {PROD_FWHT3_SYM}: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

fn launch_ship_q8(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.attention_q8_0_fa2_gqa_gfx11(
        d_q, d_k, d_v, d_out, d_pos, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("ship q8 launch: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

fn launch_ship_fwht3(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    d_s1: &rdna_compute::GpuTensor,
    d_s2: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) -> f64 {
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx11(
        d_q, d_k, d_v, d_out, d_pos, d_s1, d_s2, N_HEADS, N_KV, HD, seq_len, batch,
    )
    .unwrap_or_else(|e| panic!("ship fwht3 launch: {e:?}"));
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

fn zero_out(gpu: &Gpu, t: &rdna_compute::GpuTensor, n_f32: usize) {
    let z = vec![0u8; n_f32 * 4];
    gpu.hip
        .memcpy_htod(&t.buf, &z)
        .unwrap_or_else(|e| panic!("zero out: {e:?}"));
}

fn run_ship(gpu: &mut Gpu) {
    eprintln!("\n=== F4b ship arm: shipping FA2 vs frozen prod twins ===");
    eprintln!(
        "shape H{N_HEADS}/KV{N_KV}/D{HD} batch={BATCH}  L∈{L_VALUES:?}  positions at end of L"
    );
    eprintln!("Q8: ship=attention_q8_0_fa2_gqa_gfx11  prod={PROD_Q8_SYM}");
    eprintln!(
        "fwht3: ship=attention_q8_0_fa2_gqa_fwht3k_gfx11  prod={PROD_FWHT3_SYM} (frozen HEAD)"
    );

    // Ensure Q8 prod twin + frozen fwht3 twin.
    gpu.ensure_kernel_public("halo_fa2_f30_attrib", TWIN_SRC, PROD_Q8_SYM)
        .unwrap_or_else(|e| panic!("compile {PROD_Q8_SYM}: {e:?}"));
    eprintln!("F4a meta: compiled twin symbol {PROD_Q8_SYM}");
    ensure_prod_fwht3(gpu);

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&signs1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&signs2, &[256]).unwrap();

    for &l in &L_VALUES {
        eprintln!("\n-- L={l} (positions {}..{}) --", l - BATCH, l - 1);
        let t_pack = Instant::now();
        let pos_h = fill_positions_end_of_l(BATCH, l);
        let q_h = fill_q(BATCH, 0xF30A);
        let k_q8 = fill_q8_cache(l, 0xF30B);
        let v_h = fill_q8_cache(l, 0xF30C);
        let k_fw = fill_fwht3_k(l, 0xF30D);
        eprintln!(
            "packed Q={}  Kq8={}  Kfw={}  V={}  pos={} in {:.2}s",
            q_h.len(),
            k_q8.len(),
            k_fw.len(),
            v_h.len(),
            pos_h.len(),
            t_pack.elapsed().as_secs_f64()
        );

        let d_k_q8 = gpu.upload_raw(&k_q8, &[k_q8.len()]).unwrap();
        let d_k_fw = gpu.upload_raw(&k_fw, &[k_fw.len()]).unwrap();
        let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
        let pos_bytes = positions_to_bytes(&pos_h);
        let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();

        let d_out_prod_q8 = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
        let d_out_ship_q8 = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
        let d_out_prod_fw = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
        let d_out_ship_fw = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();

        // ---- Q8 bit-exact: prod twin vs shipping launcher ----
        let d_q_prod = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        let d_q_ship = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        zero_out(gpu, &d_out_prod_q8, BATCH * QO_ROW);
        zero_out(gpu, &d_out_ship_q8, BATCH * QO_ROW);
        let _ = launch_twin(
            gpu,
            PROD_Q8_SYM,
            &d_q_prod,
            &d_k_q8,
            &d_v,
            &d_out_prod_q8,
            &d_pos,
            BATCH as i32,
        );
        let _ = launch_ship_q8(
            gpu,
            &d_q_ship,
            &d_k_q8,
            &d_v,
            &d_out_ship_q8,
            &d_pos,
            BATCH,
            l,
        );
        let prod_out = download_f32(gpu, &d_out_prod_q8, BATCH * QO_ROW);
        let ship_out = download_f32(gpu, &d_out_ship_q8, BATCH * QO_ROW);
        let eq = bitwise_eq_f32(&prod_out, &ship_out);
        let mut n_diff = 0usize;
        let mut max_abs = 0.0f32;
        for (a, b) in prod_out.iter().zip(ship_out.iter()) {
            if a.to_bits() != b.to_bits() {
                n_diff += 1;
                max_abs = max_abs.max((a - b).abs());
            }
        }
        eprintln!(
            "  bitwise_eq L={l} Q8 ship_vs_prod = {eq}  n_diff={n_diff} max_abs={max_abs:.6e}"
        );

        // ---- fwht3 bit-exact: frozen prod vs shipping (out + post-Q) ----
        // F4b: the frozen prod twin still rotates Q in place (needs a fresh
        // upload before every launch); the shipping launcher pre-converts
        // internally and never mutates Q (post-Q must equal the input bytes).
        let q_bytes_host: Vec<u8> = q_h.iter().flat_map(|x| x.to_le_bytes()).collect();
        let d_q_prod_fw = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        let d_q_ship_fw = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        zero_out(gpu, &d_out_prod_fw, BATCH * QO_ROW);
        zero_out(gpu, &d_out_ship_fw, BATCH * QO_ROW);
        let _ = launch_prod_fwht3(
            gpu,
            &d_q_prod_fw,
            &d_k_fw,
            &d_v,
            &d_out_prod_fw,
            &d_pos,
            &d_s1,
            &d_s2,
            BATCH as i32,
        );
        let _ = launch_ship_fwht3(
            gpu,
            &d_q_ship_fw,
            &d_k_fw,
            &d_v,
            &d_out_ship_fw,
            &d_pos,
            &d_s1,
            &d_s2,
            BATCH,
            l,
        );
        let prod_fw_out = download_f32(gpu, &d_out_prod_fw, BATCH * QO_ROW);
        let ship_fw_out = download_f32(gpu, &d_out_ship_fw, BATCH * QO_ROW);
        let eq_o = bitwise_eq_f32(&prod_fw_out, &ship_fw_out);
        let mut n_diff_o = 0usize;
        let mut max_abs_o = 0.0f32;
        for (a, b) in prod_fw_out.iter().zip(ship_fw_out.iter()) {
            if a.to_bits() != b.to_bits() {
                n_diff_o += 1;
                max_abs_o = max_abs_o.max((a - b).abs());
            }
        }
        eprintln!(
            "  bitwise_eq L={l} fwht3 out ship_vs_prod = {eq_o}  n_diff={n_diff_o} max_abs={max_abs_o:.6e}"
        );
        // F4b: ship Q must be bit-identical to the pre-launch input (the
        // launcher pre-converts internally; Q immutable). Prod Q is rotated
        // in place by the frozen twin, so it must DIFFER (sanity: rotation
        // really happened on the prod side).
        let q_bytes = BATCH * QO_ROW * 4;
        let prod_q_post = download_bytes(gpu, &d_q_prod_fw, q_bytes);
        let ship_q_post = download_bytes(gpu, &d_q_ship_fw, q_bytes);
        let eq_q = bitwise_eq_bytes(&ship_q_post, &q_bytes_host);
        let mut n_diff_q = 0usize;
        for (a, b) in ship_q_post.iter().zip(q_bytes_host.iter()) {
            if a != b {
                n_diff_q += 1;
            }
        }
        eprintln!(
            "  bitwise_eq L={l} fwht3 postQ ship_vs_input = {eq_q}  n_diff_bytes={n_diff_q}"
        );
        let prod_rotated = !bitwise_eq_bytes(&prod_q_post, &q_bytes_host);
        eprintln!("  fwht3 prod rotated in place = {prod_rotated} (expect true)");

        // ---- Warmup ----
        for _ in 0..ATTR_WARMUP {
            let _ = launch_twin(
                gpu,
                PROD_Q8_SYM,
                &d_q_prod,
                &d_k_q8,
                &d_v,
                &d_out_prod_q8,
                &d_pos,
                BATCH as i32,
            );
            let _ = launch_ship_q8(
                gpu,
                &d_q_ship,
                &d_k_q8,
                &d_v,
                &d_out_ship_q8,
                &d_pos,
                BATCH,
                l,
            );
            // fwht3: refresh PROD Q outside timed path (in-place rotate
            // consumes it). Ship Q is immutable under F4b — no refresh.
            gpu.hip
                .memcpy_htod(&d_q_prod_fw.buf, &q_bytes_host)
                .unwrap();
            let _ = launch_prod_fwht3(
                gpu,
                &d_q_prod_fw,
                &d_k_fw,
                &d_v,
                &d_out_prod_fw,
                &d_pos,
                &d_s1,
                &d_s2,
                BATCH as i32,
            );
            let _ = launch_ship_fwht3(
                gpu,
                &d_q_ship_fw,
                &d_k_fw,
                &d_v,
                &d_out_ship_fw,
                &d_pos,
                &d_s1,
                &d_s2,
                BATCH,
                l,
            );
        }

        let mut samp_prod_q8 = Vec::with_capacity(ATTR_ITERS);
        let mut samp_ship_q8 = Vec::with_capacity(ATTR_ITERS);
        let mut samp_prod_fw = Vec::with_capacity(ATTR_ITERS);
        let mut samp_ship_fw = Vec::with_capacity(ATTR_ITERS);
        for _ in 0..ATTR_ITERS {
            let us = launch_twin(
                gpu,
                PROD_Q8_SYM,
                &d_q_prod,
                &d_k_q8,
                &d_v,
                &d_out_prod_q8,
                &d_pos,
                BATCH as i32,
            );
            samp_prod_q8.push(us);
            let us = launch_ship_q8(
                gpu,
                &d_q_ship,
                &d_k_q8,
                &d_v,
                &d_out_ship_q8,
                &d_pos,
                BATCH,
                l,
            );
            samp_ship_q8.push(us);

            // fwht3: refresh PROD Q before each launch (outside timed interval
            // is the memcpy; event timer only wraps the kernel). Ship Q is
            // immutable under F4b — launched as-is every iteration.
            gpu.hip
                .memcpy_htod(&d_q_prod_fw.buf, &q_bytes_host)
                .unwrap();
            let us = launch_prod_fwht3(
                gpu,
                &d_q_prod_fw,
                &d_k_fw,
                &d_v,
                &d_out_prod_fw,
                &d_pos,
                &d_s1,
                &d_s2,
                BATCH as i32,
            );
            samp_prod_fw.push(us);
            let us = launch_ship_fwht3(
                gpu,
                &d_q_ship_fw,
                &d_k_fw,
                &d_v,
                &d_out_ship_fw,
                &d_pos,
                &d_s1,
                &d_s2,
                BATCH,
                l,
            );
            samp_ship_fw.push(us);
        }

        let med_pq = median_f64(&mut samp_prod_q8);
        let med_sq = median_f64(&mut samp_ship_q8);
        let med_pf = median_f64(&mut samp_prod_fw);
        let med_sf = median_f64(&mut samp_ship_fw);
        let r_q8 = med_sq / med_pq.max(1e-30);
        let r_fw = med_sf / med_pf.max(1e-30);
        eprintln!(
            "  Q8   prod_median_us={med_pq:.2}  ship_median_us={med_sq:.2}  ship/prod={r_q8:.4}  ({:.2}% of prod)  speedup={:.2}%",
            r_q8 * 100.0,
            (1.0 - r_q8) * 100.0
        );
        eprintln!(
            "  fwht3 prod_median_us={med_pf:.2}  ship_median_us={med_sf:.2}  ship/prod={r_fw:.4}  ({:.2}% of prod)  speedup={:.2}%",
            r_fw * 100.0,
            (1.0 - r_fw) * 100.0
        );
    }
    eprintln!("\nF4b ship arm complete.");
}

fn launch_preconvert_raw(
    gpu: &mut Gpu,
    sym: &str,
    d_q: &rdna_compute::GpuTensor,
    d_dst: &rdna_compute::GpuTensor,
    d_s1: &rdna_compute::GpuTensor,
    d_s2: &rdna_compute::GpuTensor,
    batch: usize,
    do_rotate: i32,
) {
    // Kernargs mirror the production pre-convert ABI: q@0, dst@8, s1@16,
    // s2@24, batch@32, do_rotate@36. Grid covers batch*24 rows, 4/block.
    let mut args = KernargBlob::new();
    args.push_ptr(d_q.buf.as_ptr());
    args.push_ptr(d_dst.buf.as_ptr());
    args.push_ptr(d_s1.buf.as_ptr());
    args.push_ptr(d_s2.buf.as_ptr());
    args.push_i32(batch as i32);
    args.push_i32(do_rotate);
    let grid_x = (batch * N_HEADS).div_ceil(4) as u32;
    gpu.launch_kernel_blob(sym, [grid_x, 1, 1], BLOCK, 0, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch preconvert {sym}: {e:?}"));
    // Explicit same-stream sync: a stale dst must never masquerade as output.
    let ev = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&ev, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&ev).unwrap();
    let _ = gpu.hip.event_destroy(ev);
}

fn run_rotcheck(gpu: &mut Gpu) {
    // F4b rotcheck: fwht3 Q-rotation isolation on the L=1024 fixture (same
    // salts as the ship arm so results are comparable). Ground truth is the
    // frozen prod twin's in-place rotated f32 Q.
    let l = 1024usize;
    eprintln!("\n=== F4b rotcheck: fwht3 Q-rotation isolation (L={l} fixture) ===");
    ensure_prod_fwht3(gpu);
    gpu.ensure_kernel_public(
        "halo_fa2_f4b_rotcheck_dbg",
        PRECONVERT_F32_DEBUG_SRC,
        DEBUG_F32_SYM,
    )
    .unwrap_or_else(|e| panic!("compile {DEBUG_F32_SYM}: {e:?}"));
    // Shipping modules in ship-arm order: Q8 first, then fwht3.
    gpu.ensure_kernel_public(
        "halo_fa2_f4b_rotcheck_q8",
        SHIP_Q8_SRC,
        SHIP_Q8_PRECONVERT_SYM,
    )
    .unwrap_or_else(|e| panic!("compile {SHIP_Q8_PRECONVERT_SYM}: {e:?}"));
    gpu.ensure_kernel_public(
        "halo_fa2_f4b_rotcheck_fw3",
        SHIP_FWHT3_SRC,
        SHIP_FWHT3_BODY_SYM,
    )
    .unwrap_or_else(|e| panic!("compile {SHIP_FWHT3_BODY_SYM}: {e:?}"));
    // The fwht3 pre-convert symbol only exists once the kernel exports a
    // KMODE-selected name; without it the ship-f16 check is skipped.
    let have_fwht3_preconvert = gpu
        .ensure_kernel_public(
            "halo_fa2_f4b_rotcheck_fw3",
            SHIP_FWHT3_SRC,
            SHIP_FWHT3_PRECONVERT_SYM,
        )
        .map(|_| true)
        .unwrap_or_else(|e| {
            eprintln!("  note: {SHIP_FWHT3_PRECONVERT_SYM} absent ({e:?}); skipping ship-f16 check");
            false
        });

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&signs1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&signs2, &[256]).unwrap();
    let q_h = fill_q(BATCH, 0xF30A);
    let pos_h = fill_positions_end_of_l(BATCH, l);
    let pos_bytes = positions_to_bytes(&pos_h);
    let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
    let k_fw = fill_fwht3_k(l, 0xF30D);
    let v_h = fill_q8_cache(l, 0xF30C);
    let d_k_fw = gpu.upload_raw(&k_fw, &[k_fw.len()]).unwrap();
    let d_v = gpu.upload_raw(&v_h, &[v_h.len()]).unwrap();
    let d_out = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();

    // (a) Frozen prod twin once: rotates Q in place; download rotated f32.
    let d_q_prod = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
    let _ = launch_prod_fwht3(
        gpu, &d_q_prod, &d_k_fw, &d_v, &d_out, &d_pos, &d_s1, &d_s2,
        BATCH as i32,
    );
    let rot_h = download_f32(gpu, &d_q_prod, BATCH * QO_ROW);

    // (b) Debug f32 entry on a fresh ORIGINAL Q (do_rotate=1).
    let d_q_dbg = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
    let d_q32 = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
    launch_preconvert_raw(gpu, DEBUG_F32_SYM, &d_q_dbg, &d_q32, &d_s1, &d_s2, BATCH, 1);
    let dbg_h = download_f32(gpu, &d_q32, BATCH * QO_ROW);

    // (c) prod-rotated vs debug f32: first 8 diffs (row, head, dim, bits).
    let mut n_diff = 0usize;
    let mut first: Vec<(usize, u32, u32)> = Vec::new();
    for (i, (a, b)) in rot_h.iter().zip(dbg_h.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            if first.len() < 8 {
                first.push((i, a.to_bits(), b.to_bits()));
            }
            n_diff += 1;
        }
    }
    eprintln!("  prod_rot32 vs debug_f32: n_diff={n_diff} / {}", rot_h.len());
    for (i, pb, nb) in &first {
        let qr = i / (N_HEADS * HD);
        let h = (i / HD) % N_HEADS;
        let d = i % HD;
        eprintln!("    row={qr} head={h} dim={d} prod_bits=0x{pb:08x} new_bits=0x{nb:08x}");
    }

    // Host RNE cast of prod-rotated f32 (same cast the device applies).
    let rot16: Vec<u8> = rot_h
        .iter()
        .flat_map(|x| f32_to_f16_rne(*x).to_le_bytes())
        .collect();

    // (d) Shipping fwht3 pre-convert raw (do_rotate=1) on fresh ORIGINAL Q.
    if have_fwht3_preconvert {
        let d_q_ship = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
        let d_q16 = gpu.zeros(&[BATCH * QO_ROW], DType::F16).unwrap();
        launch_preconvert_raw(
            gpu, SHIP_FWHT3_PRECONVERT_SYM, &d_q_ship, &d_q16, &d_s1, &d_s2, BATCH, 1,
        );
        let ship16 = download_bytes(gpu, &d_q16, BATCH * QO_ROW * 2);
        let mut n16 = 0usize;
        for (a, b) in rot16.iter().zip(ship16.iter()) {
            if a != b {
                n16 += 1;
            }
        }
        eprintln!(
            "  cast(prod_rot32)==ship_f16: {} (n_diff_bytes={n16} / {})",
            n16 == 0,
            rot16.len()
        );
    }

    // (e) Legacy probe: the Q8-registered build with do_rotate=1. Its KMODE=0
    // compile has no rotation, so it must equal the plain cast of ORIGINAL Q
    // (this is the build the fwht3 path accidentally launched pre-fix).
    let d_q_leg = gpu.upload_f32(&q_h, &[BATCH, N_HEADS, HD]).unwrap();
    let d_q16leg = gpu.zeros(&[BATCH * QO_ROW], DType::F16).unwrap();
    launch_preconvert_raw(
        gpu, SHIP_Q8_PRECONVERT_SYM, &d_q_leg, &d_q16leg, &d_s1, &d_s2, BATCH, 1,
    );
    let leg16 = download_bytes(gpu, &d_q16leg, BATCH * QO_ROW * 2);
    let orig16: Vec<u8> = q_h
        .iter()
        .flat_map(|x| f32_to_f16_rne(*x).to_le_bytes())
        .collect();
    let mut nleg = 0usize;
    for (a, b) in orig16.iter().zip(leg16.iter()) {
        if a != b {
            nleg += 1;
        }
    }
    eprintln!(
        "  legacy(q8-build,do_rotate=1)==cast(orig_q): {} (n_diff_bytes={nleg} / {})",
        nleg == 0,
        orig16.len()
    );
    eprintln!("\nF4b rotcheck complete.");
}

// ==================== F5.1 iu8probe (owner O) ====================
//
// Example-local probe of `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` per
// plan §2.1. One wave (32 threads): the host supplies per-lane A/B
// fragments (16 bytes each), an optional i32 accumulator seed, runs one
// WMMA, and downloads all 32x8 accumulator words. Sign/clamp immediates
// are baked per module (clang `_Constant bool` — not runtime params).
// The VERIFIED hypothesis (maps on stdout as `IU8MAP {...}` JSON lines):
//   A[lane][byte] -> (row = lane & 15, k = byte)
//   B[lane][byte] -> (k = byte, col = lane & 15)
//   acc[lane][elem j] -> C[row = 2*j + (lane >> 4)][col = lane & 15]
// `verified` per table is true iff the corresponding sweeps passed.

const IU8PROBE_BLOCK: [u32; 3] = [32, 1, 1];

const IU8PROBE_SRC_TMPL: &str = r#####"
#include <hip/hip_runtime.h>
typedef __attribute__((ext_vector_type(4))) int int32x4_t;
typedef __attribute__((ext_vector_type(8))) int int32x8_t;
extern "C" __global__ __launch_bounds__(32, 1) void IU8SYM(
    const int* __restrict__ A,
    const int* __restrict__ B,
    const int* __restrict__ C_in,
    int use_c_in,
    int* __restrict__ C_out)
{
    int lane = (int)threadIdx.x;
    int32x4_t a;
    int32x4_t b;
    int32x8_t c;
    for (int i = 0; i < 4; i++) { a[i] = A[lane * 4 + i]; b[i] = B[lane * 4 + i]; }
    for (int j = 0; j < 8; j++) { c[j] = use_c_in ? C_in[lane * 8 + j] : 0; }
    c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(IU8A, a, IU8B, b, c, IU8C);
    for (int j = 0; j < 8; j++) { C_out[lane * 8 + j] = c[j]; }
}
"#####;

fn iu8_lit(b: bool) -> &'static str {
    if b {
        "true"
    } else {
        "false"
    }
}

fn ensure_iu8(gpu: &mut Gpu, a_sign: bool, b_sign: bool, clamp: bool) -> String {
    let sym = format!(
        "iu8probe_a{}b{}c{}",
        a_sign as u8, b_sign as u8, clamp as u8
    );
    let module = format!("halo_fa2_{sym}");
    let src = IU8PROBE_SRC_TMPL
        .replace("IU8SYM", &sym)
        .replace("IU8A", iu8_lit(a_sign))
        .replace("IU8B", iu8_lit(b_sign))
        .replace("IU8C", iu8_lit(clamp));
    gpu.ensure_kernel_public(&module, &src, &sym)
        .unwrap_or_else(|e| panic!("compile {sym}: {e:?}"));
    eprintln!("F5.1 meta: compiled {sym} (module {module})");
    sym
}

fn iu8_upload_lanes(gpu: &Gpu, lanes: &[[u8; 16]; 32]) -> rdna_compute::GpuTensor {
    // Lane fragment bytes are consecutive LE words: word w = bytes 4w..4w+3.
    let bytes: Vec<u8> = lanes.iter().flatten().copied().collect();
    gpu.upload_raw(&bytes, &[bytes.len()]).unwrap()
}

fn iu8_upload_acc_words(gpu: &Gpu, vals: &[[i32; 8]; 32]) -> rdna_compute::GpuTensor {
    let mut bytes = Vec::with_capacity(256 * 4);
    for l in 0..32 {
        for j in 0..8 {
            bytes.extend_from_slice(&vals[l][j].to_le_bytes());
        }
    }
    gpu.upload_raw(&bytes, &[bytes.len()]).unwrap()
}

fn iu8_download_acc(gpu: &Gpu, t: &rdna_compute::GpuTensor) -> Vec<i32> {
    let mut bytes = vec![0u8; 256 * 4];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &t.buf)
        .unwrap_or_else(|e| panic!("iu8 download: {e:?}"));
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn iu8_launch(
    gpu: &Gpu,
    sym: &str,
    d_a: &rdna_compute::GpuTensor,
    d_b: &rdna_compute::GpuTensor,
    d_cin: Option<&rdna_compute::GpuTensor>,
    d_out: &rdna_compute::GpuTensor,
) {
    let null: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut args = KernargBlob::new();
    args.push_ptr(d_a.buf.as_ptr());
    args.push_ptr(d_b.buf.as_ptr());
    args.push_ptr(d_cin.map(|t| t.buf.as_ptr()).unwrap_or(null));
    args.push_i32(d_cin.is_some() as i32);
    args.push_ptr(d_out.buf.as_ptr());
    gpu.launch_kernel_blob(sym, [1, 1, 1], IU8PROBE_BLOCK, 0, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("iu8 launch {sym}: {e:?}"));
    let ev = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&ev, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&ev).unwrap();
    let _ = gpu.hip.event_destroy(ev);
}

/// Host scalar reference under the VERIFIED map (gfx1151 iu8probe):
/// A[lane][byte] -> (row = lane & 15, k = byte);
/// B[lane][byte] -> (k = byte, col = lane & 15);
/// acc[lane][j] -> C[2*j + (lane >> 4)][lane & 15].
/// Canonical copies are lanes 0..15 (lanes 16..31 duplicate per docs).
fn iu8_expected(
    a_lanes: &[[u8; 16]; 32],
    b_lanes: &[[u8; 16]; 32],
    c_in: Option<&[[i32; 8]; 32]>,
    a_sign: bool,
    b_sign: bool,
    clamp: bool,
) -> [[i32; 8]; 32] {
    let mut a = [[0i32; 16]; 16];
    for l in 0..16 {
        for by in 0..16 {
            let av = a_lanes[l][by] as i32;
            // A[lane][byte] -> (row = lane, k = byte): tile row-major.
            a[l][by] = if a_sign { (av as i8) as i32 } else { av };
        }
    }
    // B[lane][byte] -> (k = byte, col = lane): tile B[K][C] = frag[C][K].
    // NOTE (probe fix): the first probe run assembled this transposed
    // (frag[K][C]), which caused every b_col/k_match/k_mismatch FAIL
    // against correct hardware; the device data matches the spec map.
    let mut b = [[0i32; 16]; 16];
    for l in 0..16 {
        for by in 0..16 {
            let bv = b_lanes[l][by] as i32;
            b[by][l] = if b_sign { (bv as i8) as i32 } else { bv };
        }
    }
    let mut d = [[0i32; 16]; 16];
    for r in 0..16 {
        for c in 0..16 {
            let mut s = 0i32;
            for k in 0..16 {
                s = s.wrapping_add(a[r][k].wrapping_mul(b[k][c]));
            }
            d[r][c] = s;
        }
    }
    let mut out = [[0i32; 8]; 32];
    for lane in 0..32 {
        for j in 0..8 {
            let r = 2 * j + (lane >> 4);
            let c = lane & 15;
            let base = c_in.map(|ci| ci[lane][j]).unwrap_or(0);
            out[lane][j] = if clamp {
                base.saturating_add(d[r][c])
            } else {
                base.wrapping_add(d[r][c])
            };
        }
    }
    out
}

/// Model-independent diagnostic: lanes whose whole accumulator equals `val`.
/// Pins the B-lane->col / acc-lane->col maps on the re-run regardless of
/// pass/fail (b_col c must show exactly [c, c+16] for val=16).
fn iu8_hotlanes(got: &[i32], val: i32) -> Vec<usize> {
    (0..32)
        .filter(|&l| (0..8).all(|j| got[l * 8 + j] == val))
        .collect()
}

fn iu8_check(name: &str, got: &[i32], exp: &[[i32; 8]; 32]) -> bool {
    let mut n = 0usize;
    let mut first: Vec<(usize, usize, i32, i32)> = Vec::new();
    for l in 0..32 {
        for j in 0..8 {
            if got[l * 8 + j] != exp[l][j] {
                if first.len() < 6 {
                    first.push((l, j, exp[l][j], got[l * 8 + j]));
                }
                n += 1;
            }
        }
    }
    let pass = n == 0;
    eprintln!(
        "  IU8 [{name}] {} (n_diff={n}/256)",
        if pass { "PASS" } else { "FAIL" }
    );
    for (l, j, e, g) in first {
        eprintln!("    lane={l} elem={j} expect={e} got={g}");
    }
    pass
}

fn iu8_half_class(
    got: &[i32],
    m5: &[[i32; 8]; 32],
    m9: &[[i32; 8]; 32],
    half: std::ops::Range<usize>,
) -> (usize, usize, usize) {
    let mut n5 = 0;
    let mut n9 = 0;
    let mut no = 0;
    for l in half {
        let ok5 = (0..8).all(|j| got[l * 8 + j] == m5[l][j]);
        let ok9 = (0..8).all(|j| got[l * 8 + j] == m9[l][j]);
        if ok5 {
            n5 += 1;
        } else if ok9 {
            n9 += 1;
        } else {
            no += 1;
        }
    }
    (n5, n9, no)
}

/// CPU-only §3.3 fold identity: exact i32 C_b, f32 fold vs f64 reference,
/// |C_32| <= 520192 bound, and a negative control (one 256-wide K scale).
fn iu8_host_fold_check() -> bool {
    let scale_attn = 1.0f64 / 16.0;
    let mut worst_rel = 0.0f64;
    let mut worst_c32 = 0i32;
    let mut neg_diff = 0usize;
    let mut neg_n = 0usize;
    let mut ok = true;
    for it in 0..2000u64 {
        let mut q8 = [0i8; 256];
        let mut k8 = [0i8; 256];
        for d in 0..256 {
            let (qv, kv) = match it % 4 {
                1 => (127i32, -128i32),
                2 => (0, 0),
                3 => (if d % 2 == 0 { 127 } else { -127 }, if d % 3 == 0 { 127 } else { -128 }),
                _ => (
                    (prng_u32(it * 256 + d as u64, 0xF501) % 256) as i32 - 128,
                    (prng_u32(it * 256 + d as u64, 0xF502) % 256) as i32 - 128,
                ),
            };
            q8[d] = qv.clamp(-127, 127) as i8;
            k8[d] = kv.clamp(-127, 127) as i8;
        }
        let sq = 0.001 + (prng_u32(it, 0xF503) % 1000) as f64 / 1e6;
        let mut sk = [0f64; 8];
        for b in 0..8 {
            sk[b] = 0.005 + (prng_u32(it * 8 + b as u64, 0xF504) % 2000) as f64 / 1e5;
        }
        let mut acc_f32 = 0f32;
        let mut acc_f64 = 0f64;
        let mut abs_terms = 0f64;
        let mut cbs = [0i32; 8];
        for b in 0..8 {
            let mut cb = 0i32;
            for d in 0..32 {
                cb += q8[b * 32 + d] as i32 * k8[b * 32 + d] as i32;
            }
            cbs[b] = cb;
            worst_c32 = worst_c32.max(cb.abs());
            if cb.abs() > 520_192 {
                ok = false;
            }
            acc_f32 += cb as f32 * (sq * sk[b]) as f32;
            acc_f64 += cb as f64 * (sq * sk[b]);
            abs_terms += (cb as f64 * (sq * sk[b])).abs();
        }
        let s_f32 = scale_attn as f32 * acc_f32;
        let s_f64 = scale_attn * acc_f64;
        // f32 accumulation of 8 signed terms: judge the error against the
        // sum of term magnitudes (cancellation makes the plain relative
        // error unbounded near zero and says nothing about the hardware).
        if abs_terms > 1e-9 {
            let rel = ((s_f32 as f64 - s_f64).abs()) / (scale_attn * abs_terms);
            worst_rel = worst_rel.max(rel);
            if rel > 1e-5 {
                ok = false;
            }
        }
        if it % 4 == 0 {
            neg_n += 1;
            let mut wrong = 0f64;
            for b in 0..8 {
                wrong += cbs[b] as f64 * (sq * sk[0]);
            }
            if (wrong - acc_f64).abs() > 1e-9 * acc_f64.abs().max(1.0) {
                neg_diff += 1;
            }
        }
    }
    eprintln!(
        "  IU8 [host_fold] {} (worst_rel={worst_rel:.3e} worst_|C32|={worst_c32} bound=520192 neg_diff={neg_diff}/{neg_n})",
        if ok && neg_diff == neg_n { "PASS" } else { "FAIL" }
    );
    ok && neg_diff == neg_n
}

fn iu8_emit_maps(arch: &str, a_ok: bool, b_ok: bool, c_ok: bool) {
    for lane in 0..32 {
        for byte in 0..16 {
            println!(
                "IU8MAP {{\"table\":\"A\",\"lane\":{lane},\"byte\":{byte},\"row\":{},\"k\":{byte},\"verified\":{a_ok}}}",
                lane & 15
            );
        }
    }
    for lane in 0..32 {
        for byte in 0..16 {
            println!(
                "IU8MAP {{\"table\":\"B\",\"lane\":{lane},\"byte\":{byte},\"k\":{byte},\"col\":{},\"verified\":{b_ok}}}",
                lane & 15
            );
        }
    }
    for lane in 0..32 {
        for j in 0..8 {
            println!(
                "IU8MAP {{\"table\":\"C\",\"lane\":{lane},\"elem\":{j},\"row\":{},\"col\":{},\"verified\":{c_ok}}}",
                2 * j + (lane >> 4),
                lane & 15
            );
        }
    }
    println!(
        "IU8MAP {{\"table\":\"meta\",\"arch\":\"{arch}\",\"builtin\":\"__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32\",\"a_sign\":true,\"b_sign\":true,\"clamp\":false,\"byte_order\":\"le-fragment-row-major\",\"a_verified\":{a_ok},\"b_verified\":{b_ok},\"c_verified\":{c_ok}}}"
    );
}

fn run_iu8probe(gpu: &mut Gpu) {
    eprintln!("\n=== F5.1 iu8probe: __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 semantics ===");
    eprintln!("cases: zero / ones / a_row x16 / b_col x16 / k_match x16 + k_mismatch x3 /");
    eprintln!("  signed x4 / dup_consistent / dup_conflict(info) / clamp x4 / fold x2 + host_fold");
    let s110 = ensure_iu8(gpu, true, true, false);
    let s010 = ensure_iu8(gpu, false, true, false);
    let s100 = ensure_iu8(gpu, true, false, false);
    let s000 = ensure_iu8(gpu, false, false, false);
    let s111 = ensure_iu8(gpu, true, true, true);

    let d_out = gpu.zeros(&[256], DType::F32).unwrap();
    let arch = gpu.arch.clone();
    let g: &Gpu = &*gpu;
    let run = |sym: &str,
               a: &[[u8; 16]; 32],
               b: &[[u8; 16]; 32],
               ci: Option<&[[i32; 8]; 32]>|
     -> Vec<i32> {
        let d_a = iu8_upload_lanes(g, a);
        let d_b = iu8_upload_lanes(g, b);
        let d_ci = ci.map(|v| iu8_upload_acc_words(g, v));
        iu8_launch(g, sym, &d_a, &d_b, d_ci.as_ref(), &d_out);
        iu8_download_acc(g, &d_out)
    };

    let mut pass_all = true;
    let ones: [[u8; 16]; 32] = [[1u8; 16]; 32];
    let zeros: [[u8; 16]; 32] = [[0u8; 16]; 32];

    // E0/E1: zero + all-ones (D=16 everywhere).
    let mut e0 = true;
    e0 &= iu8_check(
        "zero",
        &run(&s110, &zeros, &zeros, None),
        &iu8_expected(&zeros, &zeros, None, true, true, false),
    );
    e0 &= iu8_check(
        "ones",
        &run(&s110, &ones, &ones, None),
        &iu8_expected(&ones, &ones, None, true, true, false),
    );

    // E2: A row sweep — lanes holding row r0 get ones (D[r0][c]=16).
    let mut e2 = true;
    for r0 in 0..16 {
        let mut a = [[0u8; 16]; 32];
        for l in 0..32 {
            if (l & 15) == r0 {
                a[l] = [1u8; 16];
            }
        }
        e2 &= iu8_check(
            &format!("a_row r={r0}"),
            &run(&s110, &a, &ones, None),
            &iu8_expected(&a, &ones, None, true, true, false),
        );
    }

    // E3: B column sweep — lanes holding column c0 get ones.
    let mut e3 = true;
    for c0 in 0..16 {
        let mut b = [[0u8; 16]; 32];
        for l in 0..32 {
            if (l & 15) == c0 {
                b[l] = [1u8; 16];
            }
        }
        let got_bc = run(&s110, &ones, &b, None);
        eprintln!("    b_col c={c0} hot16={:?}", iu8_hotlanes(&got_bc, 16));
        e3 &= iu8_check(
            &format!("b_col c={c0}"),
            &got_bc,
            &iu8_expected(&ones, &b, None, true, true, false),
        );
    }

    // E4: byte->k — matching one-hot pairs give all-ones D; mismatches zero.
    let mut e4m = true;
    let mut e4x = true;
    for k0 in 0..16 {
        let mut a = [[0u8; 16]; 32];
        let mut b = [[0u8; 16]; 32];
        for l in 0..32 {
            a[l][k0] = 1;
            b[l][k0] = 1;
        }
        let got_km = run(&s110, &a, &b, None);
        eprintln!("    k_match k={k0} hot1={:?}", iu8_hotlanes(&got_km, 1));
        e4m &= iu8_check(
            &format!("k_match k={k0}"),
            &got_km,
            &iu8_expected(&a, &b, None, true, true, false),
        );
    }
    for (ka, kb) in [(3, 7), (0, 15), (9, 2)] {
        let mut a = [[0u8; 16]; 32];
        let mut b = [[0u8; 16]; 32];
        for l in 0..32 {
            a[l][ka] = 1;
            b[l][kb] = 1;
        }
        e4x &= iu8_check(
            &format!("k_mismatch ka={ka} kb={kb}"),
            &run(&s110, &a, &b, None),
            &iu8_expected(&a, &b, None, true, true, false),
        );
    }

    // E5: signedness — 0x80/0xFF as signed vs unsigned per operand.
    let mut e5 = true;
    let a80: [[u8; 16]; 32] = [[0x80u8; 16]; 32];
    let bff: [[u8; 16]; 32] = [[0xFFu8; 16]; 32];
    e5 &= iu8_check(
        "sign_a_neg",
        &run(&s110, &a80, &ones, None),
        &iu8_expected(&a80, &ones, None, true, true, false),
    );
    e5 &= iu8_check(
        "sign_a_uns",
        &run(&s010, &a80, &ones, None),
        &iu8_expected(&a80, &ones, None, false, true, false),
    );
    e5 &= iu8_check(
        "sign_b_neg",
        &run(&s110, &ones, &bff, None),
        &iu8_expected(&ones, &bff, None, true, true, false),
    );
    e5 &= iu8_check(
        "sign_b_uns",
        &run(&s100, &ones, &bff, None),
        &iu8_expected(&ones, &bff, None, true, false, false),
    );
    e5 &= iu8_check(
        "sign_uu",
        &run(&s000, &a80, &bff, None),
        &iu8_expected(&a80, &bff, None, false, false, false),
    );

    // E6: duplicate-lane consistency — both halves hold row 5.
    let mut ac = [[0u8; 16]; 32];
    for l in 0..32 {
        if (l & 15) == 5 {
            ac[l] = [1u8; 16];
        }
    }
    let e6 = iu8_check(
        "dup_consistent",
        &run(&s110, &ac, &ones, None),
        &iu8_expected(&ac, &ones, None, true, true, false),
    );

    // E7: duplicate-lane conflict (informational) — low half row 5, high row 9.
    let mut ax = [[0u8; 16]; 32];
    for l in 0..32 {
        let r = if l < 16 { 5 } else { 9 };
        if (l & 15) == r {
            ax[l] = [1u8; 16];
        }
    }
    let got_x = run(&s110, &ax, &ones, None);
    let mut a5 = [[0u8; 16]; 32];
    let mut a9 = [[0u8; 16]; 32];
    for l in 0..32 {
        if (l & 15) == 5 {
            a5[l] = [1u8; 16];
        }
        if (l & 15) == 9 {
            a9[l] = [1u8; 16];
        }
    }
    let m5 = iu8_expected(&a5, &ones, None, true, true, false);
    let m9 = iu8_expected(&a9, &ones, None, true, true, false);
    let (l5, l9, lo) = iu8_half_class(&got_x, &m5, &m9, 0..16);
    let (h5, h9, ho) = iu8_half_class(&got_x, &m5, &m9, 16..32);
    eprintln!(
        "  IU8 [dup_conflict] INFO low_half(row5-copy): match5={l5}/16 match9={l9}/16 other={lo} | high_half(row9-copy): match5={h5}/16 match9={h9}/16 other={ho}"
    );
    eprintln!(
        "  IU8 [dup_conflict] INFO verdict={}",
        if h5 == 16 {
            "low-copy-wins (HW broadcasts lanes 0..15; software MUST duplicate)"
        } else if h9 == 16 {
            "own-lane-wins (lanes consume their own fragment; duplicate anyway per docs)"
        } else {
            "MIXED/UNEXPECTED — stop F5, do not compose"
        }
    );

    // E8: clamp flag — c seed at i32 extremes, D= +/-16.
    let mut e8 = true;
    let cmax: [[i32; 8]; 32] = [[i32::MAX; 8]; 32];
    let cmin: [[i32; 8]; 32] = [[i32::MIN; 8]; 32];
    e8 &= iu8_check(
        "clamp_wrap_hi",
        &run(&s110, &ones, &ones, Some(&cmax)),
        &iu8_expected(&ones, &ones, Some(&cmax), true, true, false),
    );
    e8 &= iu8_check(
        "clamp_sat_hi",
        &run(&s111, &ones, &ones, Some(&cmax)),
        &iu8_expected(&ones, &ones, Some(&cmax), true, true, true),
    );
    e8 &= iu8_check(
        "clamp_wrap_lo",
        &run(&s110, &bff, &ones, Some(&cmin)),
        &iu8_expected(&bff, &ones, Some(&cmin), true, true, false),
    );
    e8 &= iu8_check(
        "clamp_sat_lo",
        &run(&s111, &bff, &ones, Some(&cmin)),
        &iu8_expected(&bff, &ones, Some(&cmin), true, true, true),
    );

    // E9: fold/additivity — const seed and chained accumulation.
    let mut e9 = true;
    let c1k: [[i32; 8]; 32] = [[1000i32; 8]; 32];
    e9 &= iu8_check(
        "fold_seed",
        &run(&s110, &ones, &ones, Some(&c1k)),
        &iu8_expected(&ones, &ones, Some(&c1k), true, true, false),
    );
    let out1 = run(&s110, &ones, &ones, None);
    let mut chained = [[0i32; 8]; 32];
    for l in 0..32 {
        for j in 0..8 {
            chained[l][j] = out1[l * 8 + j];
        }
    }
    e9 &= iu8_check(
        "fold_chain",
        &run(&s110, &ones, &ones, Some(&chained)),
        &iu8_expected(&ones, &ones, Some(&chained), true, true, false),
    );

    // E10: host fold identity (no GPU).
    let e10 = iu8_host_fold_check();

    pass_all &= e0 & e2 & e3 & e4m & e4x & e5 & e6 & e8 & e9 & e10;
    let a_ok = e2 && e4m;
    let b_ok = e3 && e4m;
    let c_ok = e0 && e2 && e3;
    eprintln!(
        "  IU8 finding: fragment byte b <-> k=b (dword w LE <-> k=4w..4w+3); A lane L <-> row L&15 (L+16 duplicates); B lane L <-> col L&15; acc[lane][j] = C[2j+(lane>>4)][lane&15]. First run FAILs were a transposed host B (frag[K][C]); device matches this map. b_col hot16 must read [c,c+16]; k_match hot1 all lanes."
    );
    iu8_emit_maps(&arch, a_ok, b_ok, c_ok);
    eprintln!(
        "F5.1 GATE: {} (proceed to F5.3 iff PASS; dup_conflict is informational)",
        if pass_all { "PASS" } else { "FAIL" }
    );
}

// ==================== F5.5 i8 arm (owner O) ====================
//
// Lab i8-QK candidate (F5a) vs the shipping arm on byte-identical
// K/V/positions/f32 Q at L1024/8192/32768, both K modes. Exact gfx1151
// only (other archs keep the exact kernel). Reports max abs/rel/RMS
// error, finite + masked-row counts, immutable-Q check, and
// 100-interleaved HIP-event medians; exports both outputs + a JSON
// sidecar per L/mode (`FA2_I8_OUTDIR`, default /tmp) for offline
// KLD-style analysis.
//
// Frozen lab launchers (HaloF5H, F5.4 — same arg order/types as shipping,
// q stays f32 with internal pre-convert to q8 scratch):
//   gpu.attention_q8_0_fa2_gqa_i8_gfx11(q,k,v,out,pos,NH,NKV,HD,L,B)
//   gpu.attention_q8_0_fa2_gqa_fwht3k_i8_gfx11(q,k,v,out,pos,s1,s2,NH,NKV,HD,L,B)
// Until those methods (and the F5.3 entries) exist, the timed path below
// drives the frozen §8.2 kernel entries directly with the same ABI and
// resolves them at runtime: the example still builds and the arm skips
// cleanly when the symbols are absent. CUTOVER is two marked call sites.

const I8_Q8_MOD: &str = "halo_fa2_f5_i8_q8";
const I8_FW_MOD: &str = "halo_fa2_f5_i8_fw3";
const I8_Q8_BODY_SYM: &str = "attention_q8_0_fa2_gqa_i8_gfx11";
const I8_FW_BODY_SYM: &str = "attention_q8_0_fa2_gqa_fwht3k_i8_gfx11";
const I8_PRECONVERT_SYM: &str = "attention_fa2_q_preconvert_i8_gfx11";
const I8_Q8_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KT 32\n",
    include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"),
);
const I8_FW_SRC: &str = concat!(
    "#define HIPFIRE_FA2_KT 32\n",
    "#define HIPFIRE_FA2_KMODE 3\n",
    include_str!("../../../kernels/src/turbo_common.h"),
    include_str!("../../../kernels/src/attention_q8_0_fa2_gqa.gfx11.hip"),
);
/// §8.2 q8 scratch: row stride 260 B (bytes 0..255 i8 codes, 256..259 f32 sQ).
const I8_Q8_ROW_STRIDE: usize = 260;

/// Runtime gate: resolve the frozen §8.2 entries; missing (F5.3 pending) is
/// a clean skip, not a build or runtime failure.
fn ensure_i8_mods(gpu: &mut Gpu) -> (bool, String, bool, String) {
    let mut q8_note = String::new();
    let q8_ok = match gpu.ensure_kernel_public(I8_Q8_MOD, I8_Q8_SRC, I8_Q8_BODY_SYM) {
        Ok(_) => match gpu.ensure_kernel_public(I8_Q8_MOD, I8_Q8_SRC, I8_PRECONVERT_SYM) {
            Ok(_) => true,
            Err(e) => {
                q8_note = format!("preconvert: {e:?}");
                false
            }
        },
        Err(e) => {
            q8_note = format!("body: {e:?}");
            false
        }
    };
    let mut fw_note = String::new();
    let fw_ok = match gpu.ensure_kernel_public(I8_FW_MOD, I8_FW_SRC, I8_FW_BODY_SYM) {
        Ok(_) => match gpu.ensure_kernel_public(I8_FW_MOD, I8_FW_SRC, I8_PRECONVERT_SYM) {
            Ok(_) => true,
            Err(e) => {
                fw_note = format!("preconvert: {e:?}");
                false
            }
        },
        Err(e) => {
            fw_note = format!("body: {e:?}");
            false
        }
    };
    (q8_ok, q8_note, fw_ok, fw_note)
}

fn i8_sync(gpu: &Gpu) {
    let ev = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&ev, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&ev).unwrap();
    let _ = gpu.hip.event_destroy(ev);
}

fn i8_preconvert_nowait(
    gpu: &Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_q8: &rdna_compute::GpuTensor,
    d_s1: Option<&rdna_compute::GpuTensor>,
    d_s2: Option<&rdna_compute::GpuTensor>,
    batch: usize,
    rotate: i32,
) {
    // Flat §8.2 order (mirrors HaloF5H launch_fa2_q_preconvert_i8_gfx11):
    // q, q8_scratch, signs1, signs2, n_heads, n_kv_heads, head_dim,
    let null: *mut std::ffi::c_void = std::ptr::null_mut();
    let mut args = KernargBlob::new();
    args.push_ptr(d_q.buf.as_ptr());
    args.push_ptr(d_q8.buf.as_ptr());
    args.push_ptr(d_s1.map(|t| t.buf.as_ptr()).unwrap_or(null));
    args.push_ptr(d_s2.map(|t| t.buf.as_ptr()).unwrap_or(null));
    args.push_i32(N_HEADS as i32);
    args.push_i32(N_KV as i32);
    args.push_i32(HD as i32);
    args.push_i32(batch as i32);
    args.push_i32(rotate);
    let grid_x = (batch * N_HEADS).div_ceil(4) as u32;
    gpu.launch_kernel_blob(
        I8_PRECONVERT_SYM,
        [grid_x, 1, 1],
        BLOCK,
        0,
        args.as_mut_slice(),
    )
    .unwrap_or_else(|e| panic!("launch {I8_PRECONVERT_SYM}: {e:?}"));
}

fn i8_body_nowait(
    gpu: &Gpu,
    body_sym: &str,
    d_q8: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    batch: usize,
) {
    // Frozen §8.2: shipping flat list with slot 0 = q8 scratch (same 8 B
    // slot). Confirmed with HaloF5H (F5.4): the fwht3 i8 body takes the same
    // ship-like 10-arg list (no signs); signs travel only to pre-convert.
    let mut args = KernargBlob::new();
    args.push_ptr(d_q8.buf.as_ptr());
    args.push_ptr(d_k.buf.as_ptr());
    args.push_ptr(d_v.buf.as_ptr());
    args.push_ptr(d_out.buf.as_ptr());
    args.push_ptr(d_pos.buf.as_ptr());
    args.push_i32(N_HEADS as i32);
    args.push_i32(N_KV as i32);
    args.push_i32(HD as i32);
    args.push_i32(batch as i32);
    args.push_f32(SCALE);
    let grid_x = (batch as u32).div_ceil(8);
    gpu.launch_kernel_blob(body_sym, [grid_x, 4, 1], BLOCK, LDS_BYTES, args.as_mut_slice())
        .unwrap_or_else(|e| panic!("launch {body_sym}: {e:?}"));
    i8_sync(gpu);
}

/// CUTOVER (F5.4): replace preconvert+body with
/// gpu.attention_q8_0_fa2_gqa_i8_gfx11(d_q,d_k,d_v,d_out,d_pos,NH,NKV,HD,seq_len,batch).
fn launch_lab_i8_q8(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    d_q8: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) -> f64 {
    let _ = seq_len; // consumed by the F5.4 cutover call above.
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    i8_preconvert_nowait(gpu, d_q, d_q8, None, None, batch, 0);
    i8_body_nowait(gpu, I8_Q8_BODY_SYM, d_q8, d_k, d_v, d_out, d_pos, batch);
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

/// CUTOVER (F5.4): replace preconvert+body with
/// gpu.attention_q8_0_fa2_gqa_fwht3k_i8_gfx11(d_q,d_k,d_v,d_out,d_pos,d_s1,d_s2,NH,NKV,HD,seq_len,batch).
fn launch_lab_i8_fwht3(
    gpu: &mut Gpu,
    d_q: &rdna_compute::GpuTensor,
    d_k: &rdna_compute::GpuTensor,
    d_v: &rdna_compute::GpuTensor,
    d_out: &rdna_compute::GpuTensor,
    d_pos: &rdna_compute::GpuTensor,
    d_s1: &rdna_compute::GpuTensor,
    d_s2: &rdna_compute::GpuTensor,
    d_q8: &rdna_compute::GpuTensor,
    batch: usize,
    seq_len: usize,
) -> f64 {
    let _ = seq_len; // consumed by the F5.4 cutover call above.
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip
        .event_record(&start, gpu.active_stream.as_ref())
        .unwrap();
    i8_preconvert_nowait(gpu, d_q, d_q8, Some(d_s1), Some(d_s2), batch, 1);
    i8_body_nowait(gpu, I8_FW_BODY_SYM, d_q8, d_k, d_v, d_out, d_pos, batch);
    gpu.hip
        .event_record(&stop, gpu.active_stream.as_ref())
        .unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let ms = gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64;
    let _ = gpu.hip.event_destroy(start);
    let _ = gpu.hip.event_destroy(stop);
    ms * 1000.0
}

struct I8Cmp {
    max_abs: f32,
    max_rel: f32,
    rms: f64,
    nf_ship: usize,
    nf_i8: usize,
    mask_ship: usize,
    mask_i8: usize,
    mask_diff: usize,
}

/// Differential: rel = |d| / max(|ship|, 1e-6); masked row = all-HD outputs
/// exactly 0.0 (fully causally masked / no-attention row).
fn i8_compare(ship: &[f32], cand: &[f32]) -> I8Cmp {
    assert_eq!(ship.len(), cand.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut ss = 0f64;
    let mut nf_ship = 0usize;
    let mut nf_i8 = 0usize;
    for (a, b) in ship.iter().zip(cand.iter()) {
        if !a.is_finite() {
            nf_ship += 1;
        }
        if !b.is_finite() {
            nf_i8 += 1;
        }
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / a.abs().max(1e-6));
        ss += (d as f64) * (d as f64);
    }
    let rms = (ss / ship.len().max(1) as f64).sqrt();
    let row_elems = N_HEADS * HD;
    let mut mask_ship = 0usize;
    let mut mask_i8 = 0usize;
    let mut mask_diff = 0usize;
    for r in 0..BATCH {
        let s = &ship[r * row_elems..(r + 1) * row_elems];
        let c = &cand[r * row_elems..(r + 1) * row_elems];
        let ms = s.iter().all(|&x| x == 0.0);
        let mc = c.iter().all(|&x| x == 0.0);
        mask_ship += ms as usize;
        mask_i8 += mc as usize;
        mask_diff += (ms != mc) as usize;
    }
    I8Cmp {
        max_abs,
        max_rel,
        rms,
        nf_ship,
        nf_i8,
        mask_ship,
        mask_i8,
        mask_diff,
    }
}

fn i8_export(tag: &str, l: usize, ship: &[f32], cand: &[f32], meta_json: &str) {
    let dir = std::env::var("FA2_I8_OUTDIR").unwrap_or_else(|_| "/tmp".to_string());
    let ps = format!("{dir}/fa2_i8_{tag}_L{l}_ship.f32");
    let pc = format!("{dir}/fa2_i8_{tag}_L{l}_i8.f32");
    let pm = format!("{dir}/fa2_i8_{tag}_L{l}.json");
    let to_le = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    };
    std::fs::write(&ps, to_le(ship)).unwrap_or_else(|e| panic!("write {ps}: {e:?}"));
    std::fs::write(&pc, to_le(cand)).unwrap_or_else(|e| panic!("write {pc}: {e:?}"));
    std::fs::write(&pm, meta_json.as_bytes()).unwrap_or_else(|e| panic!("write {pm}: {e:?}"));
    eprintln!("  export ship={ps} ({} f32)  i8={pc}  meta={pm}", ship.len());
}

fn run_i8_side(
    gpu: &mut Gpu,
    fwht3: bool,
    l: usize,
    q_h: &[f32],
    k_h: &[u8],
    v_h: &[u8],
    pos_h: &[i32],
    d_s1: &rdna_compute::GpuTensor,
    d_s2: &rdna_compute::GpuTensor,
) {
    let tag = if fwht3 { "fwht3" } else { "q8" };
    // Byte-identical inputs by construction: one K/V/pos upload shared by
    // both arms; two separate uploads of the same f32 Q bytes.
    let d_k = gpu.upload_raw(k_h, &[k_h.len()]).unwrap();
    let d_v = gpu.upload_raw(v_h, &[v_h.len()]).unwrap();
    let pos_bytes = positions_to_bytes(pos_h);
    let d_pos = gpu.upload_raw(&pos_bytes, &[pos_bytes.len()]).unwrap();
    let d_q_ship = gpu.upload_f32(q_h, &[BATCH, N_HEADS, HD]).unwrap();
    let d_q_i8 = gpu.upload_f32(q_h, &[BATCH, N_HEADS, HD]).unwrap();
    let d_q8 = gpu
        .upload_raw(
            &vec![0u8; BATCH * N_HEADS * I8_Q8_ROW_STRIDE],
            &[BATCH * N_HEADS * I8_Q8_ROW_STRIDE],
        )
        .unwrap();
    let d_out_ship = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
    let d_out_i8 = gpu.zeros(&[BATCH * QO_ROW], DType::F32).unwrap();
    let q_bytes_host: Vec<u8> = q_h.iter().flat_map(|x| x.to_le_bytes()).collect();

    // ---- correctness (untimed) ----
    zero_out(gpu, &d_out_ship, BATCH * QO_ROW);
    zero_out(gpu, &d_out_i8, BATCH * QO_ROW);
    if fwht3 {
        let _ = launch_ship_fwht3(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, d_s1, d_s2, BATCH, l);
        i8_preconvert_nowait(gpu, &d_q_i8, &d_q8, Some(d_s1), Some(d_s2), BATCH, 1);
        i8_body_nowait(gpu, I8_FW_BODY_SYM, &d_q8, &d_k, &d_v, &d_out_i8, &d_pos, BATCH);
    } else {
        let _ = launch_ship_q8(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, BATCH, l);
        i8_preconvert_nowait(gpu, &d_q_i8, &d_q8, None, None, BATCH, 0);
        i8_body_nowait(gpu, I8_Q8_BODY_SYM, &d_q8, &d_k, &d_v, &d_out_i8, &d_pos, BATCH);
    }
    let ship_out = download_f32(gpu, &d_out_ship, BATCH * QO_ROW);
    let i8_out = download_f32(gpu, &d_out_i8, BATCH * QO_ROW);
    let c = i8_compare(&ship_out, &i8_out);
    eprintln!(
        "  {tag} L={l} ship_vs_i8: max_abs={:.6e} max_rel={:.6e} rms={:.6e} n={} nonfinite(ship,i8)=({},{}) masked_rows(ship,i8,diff)=({},{},{})",
        c.max_abs, c.max_rel, c.rms, ship_out.len(),
        c.nf_ship, c.nf_i8, c.mask_ship, c.mask_i8, c.mask_diff
    );
    // Immutable-Q: neither launcher may mutate the f32 Q input (F4b ship is
    // replay-idempotent; F5 i8 reads immutable codes after pre-convert).
    let q_bytes = BATCH * QO_ROW * 4;
    let ship_q_post = download_bytes(gpu, &d_q_ship, q_bytes);
    let i8_q_post = download_bytes(gpu, &d_q_i8, q_bytes);
    let mut nd_ship = 0usize;
    let mut nd_i8 = 0usize;
    for (a, b) in ship_q_post.iter().zip(q_bytes_host.iter()) {
        nd_ship += (a != b) as usize;
    }
    for (a, b) in i8_q_post.iter().zip(q_bytes_host.iter()) {
        nd_i8 += (a != b) as usize;
    }
    eprintln!(
        "  {tag} L={l} Q_immutable_bytes: ship_diff={nd_ship} i8_diff={nd_i8} (expect 0,0)"
    );

    // ---- 100 interleaved HIP-event medians ----
    for _ in 0..ATTR_WARMUP {
        if fwht3 {
            let _ = launch_ship_fwht3(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, d_s1, d_s2, BATCH, l);
            let _ = launch_lab_i8_fwht3(gpu, &d_q_i8, &d_k, &d_v, &d_out_i8, &d_pos, d_s1, d_s2, &d_q8, BATCH, l);
        } else {
            let _ = launch_ship_q8(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, BATCH, l);
            let _ = launch_lab_i8_q8(gpu, &d_q_i8, &d_k, &d_v, &d_out_i8, &d_pos, &d_q8, BATCH, l);
        }
    }
    let mut samp_ship = Vec::with_capacity(ATTR_ITERS);
    let mut samp_i8 = Vec::with_capacity(ATTR_ITERS);
    for _ in 0..ATTR_ITERS {
        if fwht3 {
            samp_ship.push(launch_ship_fwht3(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, d_s1, d_s2, BATCH, l));
            samp_i8.push(launch_lab_i8_fwht3(gpu, &d_q_i8, &d_k, &d_v, &d_out_i8, &d_pos, d_s1, d_s2, &d_q8, BATCH, l));
        } else {
            samp_ship.push(launch_ship_q8(gpu, &d_q_ship, &d_k, &d_v, &d_out_ship, &d_pos, BATCH, l));
            samp_i8.push(launch_lab_i8_q8(gpu, &d_q_i8, &d_k, &d_v, &d_out_i8, &d_pos, &d_q8, BATCH, l));
        }
    }
    let med_s = median_f64(&mut samp_ship);
    let med_i = median_f64(&mut samp_i8);
    let ratio = med_i / med_s.max(1e-30);
    eprintln!(
        "  {tag} L={l} ship_median_us={med_s:.2} i8_median_us={med_i:.2} i8/ship={ratio:.4} speedup={:.2}%",
        (1.0 - ratio) * 100.0
    );

    // ---- export both outputs + JSON sidecar (raw samples included) ----
    let arch = gpu.arch.clone();
    let join = |v: &[f64]| v.iter().map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(",");
    let meta = format!(
        "{{\"arm\":\"i8\",\"mode\":\"{tag}\",\"L\":{l},\"arch\":\"{arch}\",\"batch\":{BATCH},\"max_abs\":{:.6e},\"max_rel\":{:.6e},\"rms\":{:.6e},\"n\":{},\"nonfinite_ship\":{},\"nonfinite_i8\":{},\"masked_rows_ship\":{},\"masked_rows_i8\":{},\"masked_rows_diff\":{},\"q_immutable_ship_diff_bytes\":{nd_ship},\"q_immutable_i8_diff_bytes\":{nd_i8},\"ship_median_us\":{med_s:.2},\"i8_median_us\":{med_i:.2},\"speedup_pct\":{:.2},\"ship_samples_us\":[{}],\"i8_samples_us\":[{}]}}",
        c.max_abs,
        c.max_rel,
        c.rms,
        ship_out.len(),
        c.nf_ship,
        c.nf_i8,
        c.mask_ship,
        c.mask_i8,
        c.mask_diff,
        (1.0 - ratio) * 100.0,
        join(&samp_ship),
        join(&samp_i8),
    );
    i8_export(tag, l, &ship_out, &i8_out, &meta);
}

fn run_i8(gpu: &mut Gpu) {
    eprintln!("\n=== F5.5 i8 arm: lab i8-QK (F5a) vs shipping arm ===");
    eprintln!(
        "shape H{N_HEADS}/KV{N_KV}/D{HD} batch={BATCH}  L∈{L_VALUES:?}  positions at end of L"
    );
    eprintln!("K/V/positions/f32 Q byte-identical across arms (shared K/V/pos uploads; dual f32 Q uploads of the same bytes)");
    if gpu.arch.as_str() != "gfx1151" {
        eprintln!(
            "skip: F5a is exact-gfx1151 only (arch={}); gfx1100/gfx1201 keep the exact kernel",
            gpu.arch
        );
        return;
    }
    if std::env::var("HIPFIRE_GRAPH")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        eprintln!("WARN: direct blob path needs eager HIPFIRE_GRAPH=0; times under capture are invalid");
    }
    let (have_q8, q8_note, have_fw, fw_note) = ensure_i8_mods(gpu);
    eprintln!("i8 entries: Q8 present={have_q8} {q8_note} | fwht3 present={have_fw} {fw_note}");
    if !have_q8 && !have_fw {
        eprintln!("SKIP: i8 entries absent (F5.3 pending) — shipping control is covered by the `ship` arm.");
        eprintln!("re-run once F5.3/F5.4 land:");
        eprintln!("  HOME=/tmp/home-lloyd-hipx HIPFIRE_GRAPH=0 ROCR_VISIBLE_DEVICES=1 \\");
        eprintln!("    HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f5a \\");
        eprintln!("    cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_attrib -- i8");
        return;
    }

    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let d_s1 = gpu.upload_f32(&signs1, &[256]).unwrap();
    let d_s2 = gpu.upload_f32(&signs2, &[256]).unwrap();

    for &l in &L_VALUES {
        eprintln!("\n-- L={l} (positions {}..{}) --", l - BATCH, l - 1);
        let t_pack = Instant::now();
        let pos_h = fill_positions_end_of_l(BATCH, l);
        let q_h = fill_q(BATCH, 0xF30A);
        let k_q8 = fill_q8_cache(l, 0xF30B);
        let v_h = fill_q8_cache(l, 0xF30C);
        let k_fw = fill_fwht3_k(l, 0xF30D);
        eprintln!(
            "packed Q={} Kq8={} Kfw={} V={} pos={} in {:.2}s",
            q_h.len(),
            k_q8.len(),
            k_fw.len(),
            v_h.len(),
            pos_h.len(),
            t_pack.elapsed().as_secs_f64()
        );
        if have_q8 {
            run_i8_side(gpu, false, l, &q_h, &k_q8, &v_h, &pos_h, &d_s1, &d_s2);
        } else {
            eprintln!("  Q8 i8 entries absent ({q8_note}); skipping Q8 side");
        }
        if have_fw {
            run_i8_side(gpu, true, l, &q_h, &k_fw, &v_h, &pos_h, &d_s1, &d_s2);
        } else {
            eprintln!("  fwht3 i8 entries absent ({fw_note}); skipping fwht3 side");
        }
    }
    eprintln!("\nF5.5 i8 arm complete.");
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arm = std::env::args().nth(1).unwrap_or_default();
    let is_ship = arm == "ship";
    let is_rotcheck = arm == "rotcheck";
    let is_iu8probe = arm == "iu8probe";
    let is_i8 = arm == "i8";
    eprintln!(
        "tmp_fa2_attrib arch={}  arm={}  (F3.0 attribution / F4b ship+rotcheck / F5.1 iu8probe+F5.5 i8)",
        gpu.arch,
        if is_ship {
            "ship"
        } else if is_rotcheck {
            "rotcheck"
        } else if is_iu8probe {
            "iu8probe"
        } else if is_i8 {
            "i8"
        } else {
            "attrib"
        }
    );
    eprintln!(
        "Production Q conversion: (_Float16)float C cast → AMD v_cvt_f16_f32 (RNE)."
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
        eprintln!("    HIPFIRE_KERNEL_CACHE=/tmp/kc-halo-f4a \\");
        eprintln!(
            "    cargo run --release -p hipfire-runtime --features lab --example tmp_fa2_attrib -- ship"
        );
        eprintln!("symbols: {} + {PROD_FWHT3_SYM}", SYMS.join(", "));
        return;
    }

    if gpu.active_stream.is_none() {
        gpu.active_stream = Some(gpu.hip.stream_create().expect("stream"));
    }

    if is_ship {
        run_ship(&mut gpu);
        return;
    }
    if is_rotcheck {
        run_rotcheck(&mut gpu);
        return;
    }
    if is_iu8probe {
        run_iu8probe(&mut gpu);
        return;
    }
    if is_i8 {
        run_i8(&mut gpu);
        return;
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
