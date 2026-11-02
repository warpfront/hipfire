// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//
// Identity: mqv2-affine-dot-v2-20260828
//
// gfx1201 HIP microbenchmark: MQ4V2/MQ6V2 affine GEMV baseline vs factored
// dot2 (v_dot2_f32_f16) and dot4 (v_dot4_i32_iu8 / sudot4 mixed-sign).
//
// Synthetic only — NOT a product admission gate. Wire layouts match production
// V2 exactly (MQ4V2 136 B/G256, MQ6V2 200 B/G256, dual-G128 fp16 s/z header).
//
// Arms (per bits ∈ {4,6}):
//   1. F32              — production-equivalent F32 affine reconstruct + F32 x
//   2. dot2             — exact q as half2, inline F32→FP16 x pack, fdot2,
//                         then sc*sum(q*x_h)+zp*sum(x_f32) in FP32 (no FP16 acc)
//   3. dot4-q8-ready    — sudot4 compute only (X pre-quantized to Q8_1 DS4)
//   4. q8-convert-only  — quantize_q8_1_mmq_ds4 alone
//   5. q8-convert+dot4  — conversion then dot4 inside one HIP-event pair
//
// Timing: BATCH_LAUNCHES=8 independent A slabs (same packed matrix copied into
// each slab). Each timed sample fires 8 launches against slab[rep] inside one
// HIP-event pair, then divides elapsed by 8 (per-launch us). Keeps the weight
// working set DRAM-resident at large shapes and amortizes event/queue floor.
// Weight GiB/s uses one matrix's bytes over that per-launch time.
// Sample arm order: cyclic rotate of the 10 arms; reverse on odd samples.
//
// Two dot4 timing interpretations:
//   - q8-ready  = pure integer-dot kernel cost once activations are Q8_1
//   - convert+dot4 = end-to-end cost including F32→Q8_1 DS4 conversion
//
// Dot4 uses __builtin_amdgcn_sudot4 (v_dot4_i32_iu8): unsigned MQ q codes
// (MQ4 q=0..15, MQ6 q=0..63 fit u8) × signed Q8 activations. No q-shift and
// no +8*sc; zero correction is zp*sum_x*0.25. Requires gfx11/gfx12 dot8-insts
// (active on gfx1201).
//
// Build:
//   /opt/rocm/core/bin/hipcc --offload-arch=gfx1201 -O3 \
//       tools/quant-design/bench_mqv2_affine_dot.hip \
//       -o /tmp/bench_mqv2_affine_dot
//
// Run:
//   /tmp/bench_mqv2_affine_dot [--help] [M] [K] [samples]
//   Defaults: M=512 K=2048 samples=1000 (require K%256==0)
//
// Correctness: host double-accumulation wire-contract reference; host Q8_1
// DS4 byte-exact vs device convert; max_abs/rel_l2/cosine; finite/checksum;
// F32 baseline tightly vs host; M-output prefix/suffix sentinels.
// PASS/FAIL per bit/arm; nonzero on fail.
// Suggested limits: F32 rel_l2<=1e-5; dot2<=2e-2; dot4<=5e-2.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

static constexpr const char* kBenchId = "mqv2-affine-dot-v2-20260828";
static constexpr int kWarmup = 50;
static constexpr int kDefaultSamples = 1000;
static constexpr int kDefaultM = 512;
static constexpr int kDefaultK = 2048;
static constexpr int kBatchLaunches = 8; // independent A slabs per timed sample
static constexpr int kQK8_1 = 32;
static constexpr float kSentinelVal = 7.654321f;
static constexpr float kLimF32RelL2 = 1e-5f;
static constexpr float kLimDot2RelL2 = 2e-2f;
static constexpr float kLimDot4RelL2 = 5e-2f;

#define CK(x)                                                                  \
    do {                                                                       \
        hipError_t e_ = (x);                                                   \
        if (e_ != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP error %s at %s:%d\n",                    \
                         hipGetErrorString(e_), __FILE__, __LINE__);           \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static inline int v2_group_bytes(int bits) { return bits * 32 + 8; }

// ─────────────────────────────────────────────────────────────────────────────
// Host helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline uint64_t lcg_u64(uint64_t* s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s;
}
static inline float lcg_unit(uint64_t* s) {
    return (float)(lcg_u64(s) >> 40) * (1.0f / (float)(1u << 24));
}
static inline float lcg_finite(uint64_t* s, float lo, float hi) {
    return lo + (hi - lo) * lcg_unit(s);
}

static inline uint16_t f32_to_f16_bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(1 - exp);
        uint32_t half = mant >> (shift + 13);
        const uint32_t rem = mant & ((1u << (shift + 13)) - 1u);
        if (rem > (1u << (shift + 12)) ||
            (rem == (1u << (shift + 12)) && (half & 1u)))
            half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t half = (uint32_t)exp << 10 | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t)(sign | half);
}

static inline float f16_bits_to_f32(uint16_t b) {
    const uint32_t sign = (uint32_t)(b & 0x8000u) << 16;
    uint32_t exp = (b >> 10) & 0x1Fu;
    uint32_t mant = b & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

static void pack_thread_qs(uint8_t* data, int bits, int tid, const uint8_t q[8]) {
    if (bits == 4) {
        for (int b = 0; b < 4; b++)
            data[tid * 4 + b] = (uint8_t)(q[2 * b] | (q[2 * b + 1] << 4));
    } else { // bits == 6
        uint8_t* d = data + tid * 6;
        for (int h = 0; h < 2; h++) {
            const uint8_t q0 = q[4 * h + 0] & 63u;
            const uint8_t q1 = q[4 * h + 1] & 63u;
            const uint8_t q2 = q[4 * h + 2] & 63u;
            const uint8_t q3 = q[4 * h + 3] & 63u;
            d[h * 3 + 0] = (uint8_t)(q0 | (q1 << 6));
            d[h * 3 + 1] = (uint8_t)((q1 >> 2) | (q2 << 4));
            d[h * 3 + 2] = (uint8_t)((q2 >> 4) | (q3 << 2));
        }
    }
}

static void pack_v2(std::vector<uint8_t>& out, int bits, int M, int K, uint64_t seed) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    out.assign((size_t)M * (size_t)gpr * (size_t)gb, 0);
    uint64_t rng = seed;
    const int nlev = 1 << bits;
    for (int r = 0; r < M; r++) {
        for (int g = 0; g < gpr; g++) {
            uint8_t* gp = out.data() + ((size_t)r * gpr + g) * gb;
            float s0 = lcg_finite(&rng, 0.01f, 2.0f);
            float z0 = lcg_finite(&rng, -1.0f, 1.0f);
            float s1 = lcg_finite(&rng, 0.01f, 2.0f);
            float z1 = lcg_finite(&rng, -1.0f, 1.0f);
            uint16_t h[4] = {
                f32_to_f16_bits(s0), f32_to_f16_bits(z0),
                f32_to_f16_bits(s1), f32_to_f16_bits(z1),
            };
            std::memcpy(gp, h, 8);
            uint8_t* data = gp + 8;
            for (int t = 0; t < 32; t++) {
                uint8_t q[8];
                for (int i = 0; i < 8; i++)
                    q[i] = (uint8_t)(lcg_u64(&rng) % (unsigned)nlev);
                pack_thread_qs(data, bits, t, q);
            }
        }
    }
}

static void host_unpack8(const uint8_t* data, int bits, int tid, uint32_t q[8]) {
    if (bits == 4) {
        for (int i = 0; i < 4; i++) {
            uint8_t by = data[tid * 4 + i];
            q[2 * i] = by & 0xF;
            q[2 * i + 1] = (by >> 4) & 0xF;
        }
    } else {
        const uint8_t* d = data + tid * 6;
        for (int h = 0; h < 2; h++) {
            const uint8_t b0 = d[h * 3 + 0], b1 = d[h * 3 + 1], b2 = d[h * 3 + 2];
            q[4 * h + 0] = b0 & 63u;
            q[4 * h + 1] = (b0 >> 6) | ((b1 & 0xFu) << 2);
            q[4 * h + 2] = (b1 >> 4) | ((b2 & 3u) << 4);
            q[4 * h + 3] = b2 >> 2;
        }
    }
}

// Host double-accumulation wire-contract reference (V2 dual-half affine).
static void host_gemv_v2_ref(const uint8_t* A, const float* x, float* y,
                             int bits, int M, int K) {
    const int gpr = K / 256;
    const int gb = v2_group_bytes(bits);
    for (int r = 0; r < M; r++) {
        double acc = 0.0;
        const uint8_t* row = A + (size_t)r * gpr * gb;
        for (int g = 0; g < gpr; g++) {
            const uint8_t* gp = row + g * gb;
            float s0 = f16_bits_to_f32(*(const uint16_t*)(gp + 0));
            float z0 = f16_bits_to_f32(*(const uint16_t*)(gp + 2));
            float s1 = f16_bits_to_f32(*(const uint16_t*)(gp + 4));
            float z1 = f16_bits_to_f32(*(const uint16_t*)(gp + 6));
            const uint8_t* data = gp + 8;
            for (int t = 0; t < 32; t++) {
                const float s = (t < 16) ? s0 : s1;
                const float z = (t < 16) ? z0 : z1;
                const int base = g * 256 + t * 8;
                uint32_t q[8];
                host_unpack8(data, bits, t, q);
                for (int i = 0; i < 8; i++)
                    acc += (double)(s * (float)q[i] + z) * (double)x[base + i];
            }
        }
        y[r] = (float)acc;
    }
}

struct CorrMetrics {
    double max_abs = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
    bool finite = true;
};

static CorrMetrics compare_vecs(const float* a, const float* ref, int n) {
    CorrMetrics m;
    double num = 0.0, den = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) {
            m.finite = false;
            continue;
        }
        const double da = (double)a[i];
        const double db = (double)ref[i];
        const double d = std::fabs(da - db);
        if (d > m.max_abs) m.max_abs = d;
        num += d * d;
        den += db * db;
        dot += da * db;
        na += da * da;
        nb += db * db;
    }
    m.rel_l2 = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
    const double denom = std::sqrt(na) * std::sqrt(nb);
    m.cosine = (denom > 0.0) ? (dot / denom) : 0.0;
    return m;
}

static bool all_finite(const float* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(p[i])) return false;
    return true;
}

static uint64_t checksum_finite(const float* p, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(p[i])) return 0;
        uint32_t bits;
        std::memcpy(&bits, &p[i], 4);
        h ^= bits + (uint64_t)i * 0x9E3779B97F4A7C15ull;
        h *= 0x100000001B3ull;
    }
    return h ? h : 1ull;
}

static float percentile_sorted(std::vector<float>& v, double p) {
    if (v.empty()) return 0.f;
    const double idx = p * (double)(v.size() - 1);
    const size_t lo = (size_t)idx;
    const size_t hi = std::min(lo + 1, v.size() - 1);
    const float t = (float)(idx - (double)lo);
    return v[lo] * (1.f - t) + v[hi] * t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Device: Q8_1 DS4 block + conversion (production layout, B=1 kblock-major)
// per-32 d and original-F32 sum
// ─────────────────────────────────────────────────────────────────────────────

struct block_q8_1_mmq {
    half2 ds4[4];
    int8_t qs[4 * kQK8_1];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "bad block_q8_1_mmq size");

// Host Q8_1 DS4 for N=1: mirrors device per-32 reduction (8 lanes × float4,
// XOR tree offsets 4/2/1), full-F32 d for q rounding, fp16 d + original-F32
// sum in ds4 metadata.
static void host_quantize_q8_1_mmq_ds4(const float* X, block_q8_1_mmq* Y,
                                      int K) {
    const int n_kblocks = K / 128;
    std::memset(Y, 0, (size_t)n_kblocks * sizeof(block_q8_1_mmq));
    for (int kb = 0; kb < n_kblocks; kb++) {
        block_q8_1_mmq& out = Y[kb];
        // One warp covers 128 floats = 4 sub-blocks × 8 lanes × float4.
        for (int grp = 0; grp < 4; grp++) {
            float amax_l[8], sum_l[8];
            float xv[8][4];
            for (int t = 0; t < 8; t++) {
                const int i0 = kb * 128 + grp * 32 + t * 4;
                float a = 0.0f, s = 0.0f;
                for (int i = 0; i < 4; i++) {
                    const float v = (i0 + i < K) ? X[i0 + i] : 0.0f;
                    xv[t][i] = v;
                    a = std::fmax(a, std::fabs(v));
                    s += v;
                }
                amax_l[t] = a;
                sum_l[t] = s;
            }
            // Device: for (offset=4; offset>0; offset>>=1) shfl_xor within warp
            // group of 8 lanes that own this 32-K sub-block.
            for (int offset = 4; offset > 0; offset >>= 1) {
                float na[8], ns[8];
                for (int t = 0; t < 8; t++) {
                    na[t] = std::fmax(amax_l[t], amax_l[t ^ offset]);
                    ns[t] = sum_l[t] + sum_l[t ^ offset];
                }
                std::memcpy(amax_l, na, sizeof(na));
                std::memcpy(sum_l, ns, sizeof(ns));
            }
            const float amax = amax_l[0];
            const float sum = sum_l[0];
            // Full-F32 d for q rounding (matches device before fp16 store).
            const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            for (int t = 0; t < 8; t++) {
                const int iqs = grp * 32 + t * 4;
                int8_t q0 = (int8_t)std::lrintf(
                    std::fmax(-127.0f, std::fmin(127.0f, xv[t][0] * id)));
                int8_t q1 = (int8_t)std::lrintf(
                    std::fmax(-127.0f, std::fmin(127.0f, xv[t][1] * id)));
                int8_t q2 = (int8_t)std::lrintf(
                    std::fmax(-127.0f, std::fmin(127.0f, xv[t][2] * id)));
                int8_t q3 = (int8_t)std::lrintf(
                    std::fmax(-127.0f, std::fmin(127.0f, xv[t][3] * id)));
                out.qs[iqs + 0] = q0;
                out.qs[iqs + 1] = q1;
                out.qs[iqs + 2] = q2;
                out.qs[iqs + 3] = q3;
            }
            // ds4: fp16(d), fp16(original F32 sum) — little-endian half2.
            const uint16_t db = f32_to_f16_bits(d);
            const uint16_t sb = f32_to_f16_bits(sum);
            const uint32_t ds_bits = (uint32_t)db | ((uint32_t)sb << 16);
            std::memcpy(&out.ds4[grp], &ds_bits, sizeof(uint32_t));
        }
    }
}

extern "C" __global__ void quantize_q8_1_mmq_ds4(
    const float* __restrict__ X,
    block_q8_1_mmq* __restrict__ Y,
    int K,
    int N
) {
    constexpr int WARP_SIZE = 32;
    const int token = blockIdx.y;
    const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (token >= N || i0 >= K) return;

    const float4 xi = (i0 + 3 < K)
        ? *(const float4*)(X + (long long)token * K + i0)
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)), fmaxf(fabsf(xi.z), fabsf(xi.w)));
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor(amax, offset, WARP_SIZE));

    // Original F32 sum over the 32-K sub-block (production contract).
    float sum = xi.x + xi.y + xi.z + xi.w;
    for (int offset = 4; offset > 0; offset >>= 1)
        sum += __shfl_xor(sum, offset, WARP_SIZE);

    const float d = amax > 1.0e-20f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    char4 q;
    q.x = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.x * id)));
    q.y = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.y * id)));
    q.z = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.z * id)));
    q.w = (char)lrintf(fmaxf(-127.0f, fminf(127.0f, xi.w * id)));

    const int kblock = i0 / 128;
    const int iqs = i0 & 127;
    block_q8_1_mmq* out = Y + (long long)kblock * N + token;
    ((char4*)out->qs)[iqs / 4] = q;

    if ((iqs & 31) == 0)
        out->ds4[iqs / 32] = make_half2((_Float16)d, (_Float16)sum);
}

static inline void launch_quantize_q8(const float* dX, block_q8_1_mmq* dXq,
                                      int K, int N) {
    // Each thread covers 4 floats; blockDim=256 → 1024 floats / block.x.
    const int threads = 256;
    const int floats_per_block = threads * 4;
    const dim3 grid((K + floats_per_block - 1) / floats_per_block, N);
    quantize_q8_1_mmq_ds4<<<grid, threads>>>(dX, dXq, K, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Device unpack helpers
// ─────────────────────────────────────────────────────────────────────────────

template <int BITS>
__device__ __forceinline__ void dev_unpack8(const unsigned char* data, int tid,
                                            unsigned q[8]) {
    if constexpr (BITS == 4) {
        const unsigned pk = *(const unsigned*)(data + tid * 4);
        q[0] = (pk      ) & 0xFu;
        q[1] = (pk >>  4) & 0xFu;
        q[2] = (pk >>  8) & 0xFu;
        q[3] = (pk >> 12) & 0xFu;
        q[4] = (pk >> 16) & 0xFu;
        q[5] = (pk >> 20) & 0xFu;
        q[6] = (pk >> 24) & 0xFu;
        q[7] = (pk >> 28) & 0xFu;
    } else {
        const int boff = tid * 6;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned b0 = data[boff + h * 3 + 0];
            const unsigned b1 = data[boff + h * 3 + 1];
            const unsigned b2 = data[boff + h * 3 + 2];
            q[4 * h + 0] = b0 & 63u;
            q[4 * h + 1] = (b0 >> 6) | ((b1 & 0xFu) << 2);
            q[4 * h + 2] = (b1 >> 4) | ((b2 & 3u) << 4);
            q[4 * h + 3] = b2 >> 2;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Device header load: dual-G128 fp16 s/z → F32; tid<16 → half0 else half1
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ float2 load_v2_half_header(const char* gp, int tid) {
    const unsigned hA = *(const unsigned*)(gp);
    const unsigned hB = *(const unsigned*)(gp + 4);
    const unsigned hs = (tid < 16) ? hA : hB;
    const float sc =
        __half2float(__ushort_as_half((unsigned short)(hs & 0xFFFFu)));
    const float zp =
        __half2float(__ushort_as_half((unsigned short)(hs >> 16)));
    return make_float2(sc, zp);
}

// ─────────────────────────────────────────────────────────────────────────────
// gemv_v2_f32 — production-equivalent F32 affine, 4-way F32 acc
// ─────────────────────────────────────────────────────────────────────────────

template <int BITS>
__device__ __forceinline__ void f32_lane_group(const char* gp, const float* x,
                                               int base, int tid, float& acc) {
    const float2 h = load_v2_half_header(gp, tid);
    unsigned qv[8];
    dev_unpack8<BITS>((const unsigned char*)(gp + 8), tid, qv);
#pragma unroll
    for (int i = 0; i < 8; i++)
        acc += (h.x * (float)qv[i] + h.y) * x[base + i];
}

template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_v2_f32(const char* __restrict__ A,
                            const float* __restrict__ x,
                            float* __restrict__ y,
                            int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int gb = BITS * 32 + 8;
    const char* row_ptr = A + (long long)row * gpr * gb;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * gb;
        const int base = g * 256 + tid * 8;
        f32_lane_group<BITS>(gp0, x, base, tid, acc0);
        f32_lane_group<BITS>(gp0 + gb, x, base + 256, tid, acc1);
        f32_lane_group<BITS>(gp0 + 2 * gb, x, base + 512, tid, acc2);
        f32_lane_group<BITS>(gp0 + 3 * gb, x, base + 768, tid, acc3);
    }
    if (tail >= 1) {
        const int g = quads << 2;
        f32_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        f32_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        f32_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// gemv_v2_dot2 — factored affine via fdot2 + F32 scale/zero correction
//   sum (sc*q + zp)*x = sc * sum(q * x_half) + zp * sum(x_f32)
// Never uses FP16 accumulation. Never dequants weights to FP16 beyond exact q.
// ─────────────────────────────────────────────────────────────────────────────

typedef _Float16 __attribute__((ext_vector_type(2))) half2_vec_t;

template <int BITS>
__device__ __forceinline__ void dot2_lane_group(const char* gp, const float* x,
                                                int base, int tid, float& acc) {
    const float2 h = load_v2_half_header(gp, tid);
    unsigned qv[8];
    dev_unpack8<BITS>((const unsigned char*)(gp + 8), tid, qv);

    const float* xp = x + base;
    // Exact unsigned q codes as half2; inline F32→FP16 activation packing.
    const half2_vec_t w01 = {(_Float16)(float)qv[0], (_Float16)(float)qv[1]};
    const half2_vec_t w23 = {(_Float16)(float)qv[2], (_Float16)(float)qv[3]};
    const half2_vec_t w45 = {(_Float16)(float)qv[4], (_Float16)(float)qv[5]};
    const half2_vec_t w67 = {(_Float16)(float)qv[6], (_Float16)(float)qv[7]};
    const half2_vec_t x01 = {(_Float16)xp[0], (_Float16)xp[1]};
    const half2_vec_t x23 = {(_Float16)xp[2], (_Float16)xp[3]};
    const half2_vec_t x45 = {(_Float16)xp[4], (_Float16)xp[5]};
    const half2_vec_t x67 = {(_Float16)xp[6], (_Float16)xp[7]};

    // Four independent fdot2 partials (ILP); F32 result each.
    float p0 = __builtin_amdgcn_fdot2(w01, x01, 0.0f, false);
    float p1 = __builtin_amdgcn_fdot2(w23, x23, 0.0f, false);
    float p2 = __builtin_amdgcn_fdot2(w45, x45, 0.0f, false);
    float p3 = __builtin_amdgcn_fdot2(w67, x67, 0.0f, false);
    const float sum_qx = (p0 + p1) + (p2 + p3);

    // Lane-local sum of original F32 activations.
    const float sum_x =
        ((xp[0] + xp[1]) + (xp[2] + xp[3])) +
        ((xp[4] + xp[5]) + (xp[6] + xp[7]));

    // Widened FP32 scale/zero affine correction + FP32 accumulation.
    acc += h.x * sum_qx + h.y * sum_x;
}

template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_v2_dot2(const char* __restrict__ A,
                             const float* __restrict__ x,
                             float* __restrict__ y,
                             int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int tid = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int gb = BITS * 32 + 8;
    const char* row_ptr = A + (long long)row * gpr * gb;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * gb;
        const int base = g * 256 + tid * 8;
        dot2_lane_group<BITS>(gp0, x, base, tid, acc0);
        dot2_lane_group<BITS>(gp0 + gb, x, base + 256, tid, acc1);
        dot2_lane_group<BITS>(gp0 + 2 * gb, x, base + 512, tid, acc2);
        dot2_lane_group<BITS>(gp0 + 3 * gb, x, base + 768, tid, acc3);
    }
    if (tail >= 1) {
        const int g = quads << 2;
        dot2_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        dot2_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        dot2_lane_group<BITS>(row_ptr + g * gb, x, g * 256 + tid * 8, tid, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (tid == 0) y[row] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// gemv_v2_dot4 — sudot4 (v_dot4_i32_iu8) against Q8_1 DS4 activations
//   MQ q packs as unsigned u8 (MQ4 0..15, MQ6 0..63) — no q-shift, no +8*sc
//   Q8 activations are signed int8
//   Per lane/sub-block: sc*d_x*sumi + zp*sum_x*0.25
// ─────────────────────────────────────────────────────────────────────────────

template <int BITS>
__device__ __forceinline__ void dot4_lane_group(
    const char* gp, const block_q8_1_mmq* Xq, int g, int lane,
    int sub_in_block, int block_within_group, int qs_byte_off, float& acc) {
    const float2 h = load_v2_half_header(gp, lane);
    const float sc = h.x;
    const float zp = h.y;

    unsigned qv[8];
    dev_unpack8<BITS>((const unsigned char*)(gp + 8), lane, qv);

    // Unsigned MQ q in low 8 bits of each lane (both MQ4 and MQ6).
    const int int_a =
        (int)((qv[0] & 0xFFu) | ((qv[1] & 0xFFu) << 8) |
              ((qv[2] & 0xFFu) << 16) | ((qv[3] & 0xFFu) << 24));
    const int int_b =
        (int)((qv[4] & 0xFFu) | ((qv[5] & 0xFFu) << 8) |
              ((qv[6] & 0xFFu) << 16) | ((qv[7] & 0xFFu) << 24));

    const int kblock = 2 * g + block_within_group;
    const block_q8_1_mmq* xb = Xq + kblock;
    const int* xqs = (const int*)(xb->qs + qs_byte_off);
    const int xq_a = xqs[0];
    const int xq_b = xqs[1];

    // sudot4: a_sign=false (unsigned MQ q), b_sign=true (signed Q8).
    int sumi = 0;
    sumi = __builtin_amdgcn_sudot4(false, int_a, true, xq_a, sumi, false);
    sumi = __builtin_amdgcn_sudot4(false, int_b, true, xq_b, sumi, false);

    const half2 ds = xb->ds4[sub_in_block];
    const float2 dsf = __half22float2(ds);
    const float d_x = dsf.x;
    const float sum_x = dsf.y;

    // 0.25: four lanes share one 32-K sub-block sum_x (original F32 sum).
    acc += sc * d_x * (float)sumi + zp * sum_x * 0.25f;
}

template <int BITS>
__launch_bounds__(32, 16)
__global__ void gemv_v2_dot4(const char* __restrict__ A,
                             const block_q8_1_mmq* __restrict__ Xq,
                             float* __restrict__ y,
                             int M, int K) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int lane = (int)threadIdx.x;
    const int gpr = K / 256;
    constexpr int gb = BITS * 32 + 8;
    const char* row_ptr = A + (long long)row * gpr * gb;

    // Lane → Q8_1 sub-block mapping (same as fused_gate_up_*_dp4a).
    const int sub_in_block = (lane / 4) % 4;
    const int block_within_group = lane / 16;
    const int qs_byte_off = 8 * (lane % 16);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    const int quads = gpr >> 2;
    const int tail = gpr & 3;

    for (int q = 0; q < quads; q++) {
        const int g = q << 2;
        const char* gp0 = row_ptr + g * gb;
        dot4_lane_group<BITS>(gp0, Xq, g + 0, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc0);
        dot4_lane_group<BITS>(gp0 + gb, Xq, g + 1, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc1);
        dot4_lane_group<BITS>(gp0 + 2 * gb, Xq, g + 2, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc2);
        dot4_lane_group<BITS>(gp0 + 3 * gb, Xq, g + 3, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc3);
    }
    if (tail >= 1) {
        const int g = quads << 2;
        dot4_lane_group<BITS>(row_ptr + g * gb, Xq, g, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc0);
    }
    if (tail >= 2) {
        const int g = (quads << 2) + 1;
        dot4_lane_group<BITS>(row_ptr + g * gb, Xq, g, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc1);
    }
    if (tail >= 3) {
        const int g = (quads << 2) + 2;
        dot4_lane_group<BITS>(row_ptr + g * gb, Xq, g, lane, sub_in_block,
                              block_within_group, qs_byte_off, acc2);
    }

    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down(acc, off);
    if (lane == 0) y[row] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
// Launch helpers
// ─────────────────────────────────────────────────────────────────────────────

using GemvF32 = void (*)(const char*, const float*, float*, int, int);
using GemvDot2 = void (*)(const char*, const float*, float*, int, int);
using GemvDot4 = void (*)(const char*, const block_q8_1_mmq*, float*, int, int);

static GemvF32 pick_f32(int bits) {
    return bits == 4 ? gemv_v2_f32<4> : gemv_v2_f32<6>;
}
static GemvDot2 pick_dot2(int bits) {
    return bits == 4 ? gemv_v2_dot2<4> : gemv_v2_dot2<6>;
}
static GemvDot4 pick_dot4(int bits) {
    return bits == 4 ? gemv_v2_dot4<4> : gemv_v2_dot4<6>;
}

enum class ArmKind : int {
    F32 = 0,
    Dot2 = 1,
    Dot4Q8Ready = 2,
    Q8ConvertOnly = 3,
    Q8ConvertDot4 = 4,
};

static const char* arm_name(ArmKind k) {
    switch (k) {
    case ArmKind::F32: return "F32";
    case ArmKind::Dot2: return "dot2";
    case ArmKind::Dot4Q8Ready: return "dot4-q8-ready";
    case ArmKind::Q8ConvertOnly: return "q8-convert-only";
    case ArmKind::Q8ConvertDot4: return "q8-convert+dot4";
    }
    return "?";
}

static const char* bits_name(int bits) {
    return bits == 4 ? "MQ4V2" : "MQ6V2";
}

// y buffer layout: [sentinel_lo | M floats | sentinel_hi]
struct YBuf {
    float* d_base = nullptr; // full allocation
    float* d_y = nullptr;    // points to M-region
    size_t full_elems = 0;
    int M = 0;
};

static YBuf alloc_y(int M) {
    YBuf b;
    b.M = M;
    b.full_elems = (size_t)M + 2;
    CK(hipMalloc(&b.d_base, b.full_elems * sizeof(float)));
    b.d_y = b.d_base + 1;
    // Seed sentinels + zero payload.
    std::vector<float> h(b.full_elems, 0.f);
    h[0] = kSentinelVal;
    h[b.full_elems - 1] = kSentinelVal;
    CK(hipMemcpy(b.d_base, h.data(), b.full_elems * sizeof(float),
                 hipMemcpyHostToDevice));
    return b;
}

static bool check_sentinels(const YBuf& b, const char* tag) {
    float lo = 0.f, hi = 0.f;
    CK(hipMemcpy(&lo, b.d_base, sizeof(float), hipMemcpyDeviceToHost));
    CK(hipMemcpy(&hi, b.d_base + b.full_elems - 1, sizeof(float),
                 hipMemcpyDeviceToHost));
    if (lo != kSentinelVal || hi != kSentinelVal) {
        std::printf("FAIL %s sentinel overrun lo=%.6f hi=%.6f (want %.6f)\n",
                    tag, lo, hi, kSentinelVal);
        return false;
    }
    return true;
}

static void free_y(YBuf& b) {
    if (b.d_base) CK(hipFree(b.d_base));
    b.d_base = nullptr;
    b.d_y = nullptr;
}

struct ArmState {
    ArmKind kind;
    int bits;
    YBuf y;
    std::vector<float> samples_us;
    float med = 0, p10 = 0, p90 = 0, gib_s = 0;
    CorrMetrics corr{};
    bool pass = false;
    uint64_t csum = 0;
};

static void print_help(const char* argv0) {
    std::printf(
        "Usage: %s [--help] [M] [K] [samples]\n"
        "\n"
        "Identity: %s\n"
        "gfx1201 synthetic microbench: MQ4V2/MQ6V2 affine F32 GEMV vs factored\n"
        "dot2 (fdot2) and dot4 (sudot4 / v_dot4_i32_iu8 + Q8_1 DS4).\n"
        "NOT a product admission gate.\n"
        "\n"
        "Wire: MQ4V2 136 B/G256, MQ6V2 200 B/G256, dual-G128 fp16 s/z header.\n"
        "Arms per bits: F32, dot2, dot4-q8-ready, q8-convert-only, q8-convert+dot4.\n"
        "\n"
        "Timing: BATCH_LAUNCHES=%d independent A slabs; each sample fires %d\n"
        "launches inside one HIP-event pair and reports per-launch us (elapsed/%d).\n"
        "Arm order: cyclic rotate of 10 arms; reverse on odd samples.\n"
        "Weight GiB/s = one matrix bytes / per-launch seconds.\n"
        "\n"
        "dot4: unsigned MQ q × signed Q8 via sudot4; no q-shift / no +8*sc.\n"
        "  q8-ready         — kernel-only once X is Q8_1 DS4\n"
        "  q8-convert+dot4  — conversion then dot4 inside one event pair\n"
        "\n"
        "Defaults: M=%d K=%d samples=%d (K must be a multiple of 256).\n"
        "Warmup=%d HIP events; report median/p10/p90 us and weight GiB/s.\n",
        argv0, kBenchId, kBatchLaunches, kBatchLaunches, kBatchLaunches,
        kDefaultM, kDefaultK, kDefaultSamples, kWarmup);
}

int main(int argc, char** argv) {
    int M = kDefaultM;
    int K = kDefaultK;
    int samples = kDefaultSamples;

    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            print_help(argv[0]);
            return 0;
        }
    }
    if (argc >= 2 && argv[1][0] != '-') M = std::atoi(argv[1]);
    if (argc >= 3 && argv[2][0] != '-') K = std::atoi(argv[2]);
    if (argc >= 4 && argv[3][0] != '-') samples = std::atoi(argv[3]);

    if (M <= 0 || K <= 0 || (K % 256) != 0 || samples <= 0) {
        std::fprintf(stderr,
                     "Need M>0, K%%256==0, samples>0 (got M=%d K=%d samples=%d)\n",
                     M, K, samples);
        return 2;
    }

    CK(hipSetDevice(0));
    hipDeviceProp_t prop{};
    CK(hipGetDeviceProperties(&prop, 0));
    std::printf("=== MQV2 affine vs factored-dot microbench (synthetic) ===\n");
    std::printf("identity: %s\n", kBenchId);
    std::printf("GPU: %s  offload: gfx1201  M=%d K=%d samples=%d warmup=%d "
                "batch_launches=%d\n",
                prop.name, M, K, samples, kWarmup, kBatchLaunches);
    std::printf("NOT product admission. dot4=sudot4 (uMQ×sQ8); "
                "q8-ready vs convert+dot4 separate.\n\n");

    // ── Host data ───────────────────────────────────────────────────────────
    std::vector<float> hX((size_t)K);
    {
        uint64_t rng = 0xA11CE5EEDull;
        for (size_t i = 0; i < hX.size(); i++)
            hX[i] = lcg_finite(&rng, -1.0f, 1.0f);
    }

    std::vector<uint8_t> pack4, pack6;
    pack_v2(pack4, 4, M, K, 0xBEEF0004ull);
    pack_v2(pack6, 6, M, K, 0xBEEF0006ull);

    std::vector<float> href4((size_t)M), href6((size_t)M);
    host_gemv_v2_ref(pack4.data(), hX.data(), href4.data(), 4, M, K);
    host_gemv_v2_ref(pack6.data(), hX.data(), href6.data(), 6, M, K);
    if (!all_finite(href4.data(), href4.size()) ||
        !all_finite(href6.data(), href6.size())) {
        std::printf("FAIL host reference non-finite\n");
        return 1;
    }

    const int n_kblocks = K / 128;
    std::vector<block_q8_1_mmq> hXq_ref((size_t)n_kblocks);
    host_quantize_q8_1_mmq_ds4(hX.data(), hXq_ref.data(), K);

    // ── Device buffers (A slabs × kBatchLaunches) ───────────────────────────
    float* dX = nullptr;
    char *dA4 = nullptr, *dA6 = nullptr;
    block_q8_1_mmq* dXq = nullptr;
    const size_t pack4_bytes = pack4.size();
    const size_t pack6_bytes = pack6.size();

    CK(hipMalloc(&dX, (size_t)K * sizeof(float)));
    CK(hipMemcpy(dX, hX.data(), (size_t)K * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMalloc(&dA4, pack4_bytes * (size_t)kBatchLaunches));
    CK(hipMalloc(&dA6, pack6_bytes * (size_t)kBatchLaunches));
    for (int r = 0; r < kBatchLaunches; r++) {
        CK(hipMemcpy(dA4 + (size_t)r * pack4_bytes, pack4.data(), pack4_bytes,
                     hipMemcpyHostToDevice));
        CK(hipMemcpy(dA6 + (size_t)r * pack6_bytes, pack6.data(), pack6_bytes,
                     hipMemcpyHostToDevice));
    }
    CK(hipMalloc(&dXq, (size_t)n_kblocks * sizeof(block_q8_1_mmq)));
    CK(hipMemset(dXq, 0, (size_t)n_kblocks * sizeof(block_q8_1_mmq)));

    // Independent Y per arm (5 arms × 2 bits).
    const ArmKind kinds[5] = {
        ArmKind::F32, ArmKind::Dot2, ArmKind::Dot4Q8Ready,
        ArmKind::Q8ConvertOnly, ArmKind::Q8ConvertDot4,
    };
    const int bits_list[2] = {4, 6};
    std::vector<ArmState> arms;
    arms.reserve(10);
    for (int bits : bits_list) {
        for (ArmKind k : kinds) {
            ArmState a;
            a.kind = k;
            a.bits = bits;
            a.y = alloc_y(M);
            a.samples_us.reserve((size_t)samples);
            arms.push_back(a);
        }
    }
    const int n_arms = (int)arms.size(); // 10

    // Correctness always uses slab 0.
    auto dA_slab = [&](int bits, int rep) -> char* {
        if (bits == 4)
            return dA4 + (size_t)rep * pack4_bytes;
        return dA6 + (size_t)rep * pack6_bytes;
    };
    auto href_for = [&](int bits) -> const float* {
        return bits == 4 ? href4.data() : href6.data();
    };
    auto wbytes_for = [&](int bits) -> size_t {
        return (size_t)M * (size_t)(K / 256) * (size_t)v2_group_bytes(bits);
    };

    auto launch_arm_rep = [&](ArmState& a, int rep) {
        char* dA = dA_slab(a.bits, rep);
        float* dy = a.y.d_y;
        switch (a.kind) {
        case ArmKind::F32:
            pick_f32(a.bits)<<<M, 32>>>(dA, dX, dy, M, K);
            break;
        case ArmKind::Dot2:
            pick_dot2(a.bits)<<<M, 32>>>(dA, dX, dy, M, K);
            break;
        case ArmKind::Dot4Q8Ready:
            pick_dot4(a.bits)<<<M, 32>>>(dA, dXq, dy, M, K);
            break;
        case ArmKind::Q8ConvertOnly:
            launch_quantize_q8(dX, dXq, K, /*N=*/1);
            break;
        case ArmKind::Q8ConvertDot4:
            launch_quantize_q8(dX, dXq, K, /*N=*/1);
            pick_dot4(a.bits)<<<M, 32>>>(dA, dXq, dy, M, K);
            break;
        }
    };

    auto launch_arm_batch = [&](ArmState& a) {
        for (int r = 0; r < kBatchLaunches; r++)
            launch_arm_rep(a, r);
    };

    // ── Correctness (slab 0 only) ───────────────────────────────────────────
    bool all_pass = true;
    std::printf("--- correctness ---\n");

    // Pre-quantize once for q8-ready / shared dXq paths.
    launch_quantize_q8(dX, dXq, K, 1);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());

    for (ArmState& a : arms) {
        if (a.kind == ArmKind::Q8ConvertOnly ||
            a.kind == ArmKind::Q8ConvertDot4) {
            CK(hipMemset(dXq, 0, (size_t)n_kblocks * sizeof(block_q8_1_mmq)));
        }
        if (a.kind == ArmKind::Dot4Q8Ready) {
            launch_quantize_q8(dX, dXq, K, 1);
            CK(hipDeviceSynchronize());
        }

        CK(hipMemset(a.y.d_y, 0, (size_t)M * sizeof(float)));
        launch_arm_rep(a, /*rep=*/0);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());

        char tag[64];
        std::snprintf(tag, sizeof(tag), "%s/%s", bits_name(a.bits),
                      arm_name(a.kind));

        if (!check_sentinels(a.y, tag)) {
            a.pass = false;
            all_pass = false;
            continue;
        }

        // convert-only: byte-exact ds4+qs vs independent host Q8_1 DS4.
        if (a.kind == ArmKind::Q8ConvertOnly) {
            std::vector<block_q8_1_mmq> hq((size_t)n_kblocks);
            CK(hipMemcpy(hq.data(), dXq, hq.size() * sizeof(block_q8_1_mmq),
                         hipMemcpyDeviceToHost));
            bool ok = true;
            int mism_kb = -1, mism_off = -1;
            for (int kb = 0; kb < n_kblocks && ok; kb++) {
                const uint8_t* d = (const uint8_t*)&hq[kb];
                const uint8_t* r = (const uint8_t*)&hXq_ref[kb];
                for (int off = 0; off < (int)sizeof(block_q8_1_mmq); off++) {
                    if (d[off] != r[off]) {
                        ok = false;
                        mism_kb = kb;
                        mism_off = off;
                        break;
                    }
                }
            }
            a.pass = ok;
            a.csum = a.pass ? 1ull : 0ull;
            a.corr.finite = ok;
            if (a.pass) {
                std::printf("PASS %-22s  q8-convert byte-exact vs host (%d kblocks)\n",
                            tag, n_kblocks);
            } else {
                std::printf("FAIL %-22s  q8 byte mismatch kb=%d off=%d\n", tag,
                            mism_kb, mism_off);
                all_pass = false;
            }
            continue;
        }

        std::vector<float> hdev((size_t)M);
        CK(hipMemcpy(hdev.data(), a.y.d_y, (size_t)M * sizeof(float),
                     hipMemcpyDeviceToHost));
        a.corr = compare_vecs(hdev.data(), href_for(a.bits), M);
        a.csum = checksum_finite(hdev.data(), hdev.size());

        float lim = kLimF32RelL2;
        if (a.kind == ArmKind::Dot2) lim = kLimDot2RelL2;
        if (a.kind == ArmKind::Dot4Q8Ready || a.kind == ArmKind::Q8ConvertDot4)
            lim = kLimDot4RelL2;

        const bool tight_f32 =
            (a.kind != ArmKind::F32) || (a.corr.rel_l2 <= kLimF32RelL2);

        a.pass = a.corr.finite && a.csum != 0 && a.corr.rel_l2 <= (double)lim &&
                 tight_f32 && std::isfinite(a.corr.cosine);

        if (a.pass) {
            std::printf(
                "PASS %-22s  max_abs=%.3e  rel_l2=%.3e  cos=%.6f  csum=0x%016llx\n",
                tag, a.corr.max_abs, a.corr.rel_l2, a.corr.cosine,
                (unsigned long long)a.csum);
        } else {
            std::printf(
                "FAIL %-22s  max_abs=%.3e  rel_l2=%.3e  cos=%.6f  finite=%d "
                "csum=0x%016llx  lim=%.1e\n",
                tag, a.corr.max_abs, a.corr.rel_l2, a.corr.cosine,
                (int)a.corr.finite, (unsigned long long)a.csum, lim);
            all_pass = false;
        }
    }

    if (!all_pass) {
        std::printf("\nCORRECTNESS: FAIL\n");
        for (ArmState& a : arms) free_y(a.y);
        CK(hipFree(dX));
        CK(hipFree(dA4));
        CK(hipFree(dA6));
        CK(hipFree(dXq));
        return 1;
    }
    std::printf("CORRECTNESS: PASS\n\n");

    // ── Warmup ──────────────────────────────────────────────────────────────
    // Deterministic convert leaves identical Q8 bytes; prequantize once.
    launch_quantize_q8(dX, dXq, K, 1);
    CK(hipDeviceSynchronize());

    for (int w = 0; w < kWarmup; w++) {
        for (ArmState& a : arms)
            launch_arm_batch(a);
    }
    CK(hipDeviceSynchronize());

    // Anti-DCE checksum after warmup.
    uint64_t sink = 0xCBF29CE484222325ull;
    for (ArmState& a : arms) {
        if (a.kind == ArmKind::Q8ConvertOnly) continue;
        std::vector<float> hy((size_t)M);
        CK(hipMemcpy(hy.data(), a.y.d_y, (size_t)M * sizeof(float),
                     hipMemcpyDeviceToHost));
        const uint64_t cs = checksum_finite(hy.data(), hy.size());
        if (cs == 0) {
            std::printf("FAIL post-warmup checksum %s/%s\n", bits_name(a.bits),
                        arm_name(a.kind));
            return 1;
        }
        sink = (sink ^ cs) * 0x100000001B3ull;
        if (!check_sentinels(a.y, "post-warmup")) return 1;
    }

    // Prequantize once before timing (no per-sample H2D restore).
    launch_quantize_q8(dX, dXq, K, 1);
    CK(hipDeviceSynchronize());

    // ── Interleaved timing (rotated order; batch launches) ──────────────────
    hipEvent_t ev0, ev1;
    CK(hipEventCreate(&ev0));
    CK(hipEventCreate(&ev1));

    for (int s = 0; s < samples; s++) {
        int order[10];
        for (int i = 0; i < n_arms; i++)
            order[i] = i;
        // Cyclic rotate by s % n_arms.
        const int rot = s % n_arms;
        std::rotate(order, order + rot, order + n_arms);
        // Reverse traversal on odd samples.
        if (s & 1)
            std::reverse(order, order + n_arms);

        for (int oi = 0; oi < n_arms; oi++) {
            ArmState& a = arms[order[oi]];
            CK(hipEventRecord(ev0, nullptr));
            launch_arm_batch(a);
            CK(hipEventRecord(ev1, nullptr));
            CK(hipEventSynchronize(ev1));
            float ms = 0.f;
            CK(hipEventElapsedTime(&ms, ev0, ev1));
            // Per-launch microseconds (batch amortizes event/queue floor).
            a.samples_us.push_back((ms * 1000.f) / (float)kBatchLaunches);
        }
    }
    CK(hipEventDestroy(ev0));
    CK(hipEventDestroy(ev1));

    // Final finite/checksum + sentinels.
    for (ArmState& a : arms) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "post-bench %s/%s", bits_name(a.bits),
                      arm_name(a.kind));
        if (a.kind == ArmKind::Q8ConvertOnly) continue;
        if (!check_sentinels(a.y, tag)) return 1;
        std::vector<float> hy((size_t)M);
        CK(hipMemcpy(hy.data(), a.y.d_y, (size_t)M * sizeof(float),
                     hipMemcpyDeviceToHost));
        const uint64_t cs = checksum_finite(hy.data(), hy.size());
        if (cs == 0) {
            std::printf("FAIL %s checksum\n", tag);
            return 1;
        }
        sink = (sink ^ cs) * 0x100000001B3ull;
    }

    // ── Stats ───────────────────────────────────────────────────────────────
    for (ArmState& a : arms) {
        std::sort(a.samples_us.begin(), a.samples_us.end());
        a.med = percentile_sorted(a.samples_us, 0.50);
        a.p10 = percentile_sorted(a.samples_us, 0.10);
        a.p90 = percentile_sorted(a.samples_us, 0.90);
        // Weight GiB/s: one matrix bytes / per-launch seconds.
        const bool compute_only =
            a.kind == ArmKind::F32 || a.kind == ArmKind::Dot2 ||
            a.kind == ArmKind::Dot4Q8Ready;
        if (compute_only) {
            const double sec = (double)a.med * 1e-6;
            a.gib_s = (sec > 0.0)
                          ? (float)((double)wbytes_for(a.bits) / sec /
                                    (1024.0 * 1024.0 * 1024.0))
                          : 0.f;
        } else {
            a.gib_s = 0.f;
        }
    }

    auto find_med = [&](int bits, ArmKind k) -> float {
        for (const ArmState& a : arms)
            if (a.bits == bits && a.kind == k) return a.med;
        return 0.f;
    };

    std::printf("--- results (per-launch median/p10/p90 us; weight GiB/s) ---\n");
    std::printf("%-6s %-18s %8s %8s %8s %10s %10s %12s\n", "fmt", "arm",
                "med_us", "p10_us", "p90_us", "GiB/s", "vs_F32", "vs_F32_e2e");
    for (const ArmState& a : arms) {
        const float f32 = find_med(a.bits, ArmKind::F32);
        float vs_f32 = 0.f, vs_e2e = 0.f;
        if (a.kind == ArmKind::Dot2 || a.kind == ArmKind::Dot4Q8Ready) {
            vs_f32 = (a.med > 0.f) ? (f32 / a.med) : 0.f;
        }
        if (a.kind == ArmKind::Q8ConvertDot4) {
            vs_e2e = (a.med > 0.f) ? (f32 / a.med) : 0.f;
        }
        std::printf("%-6s %-18s %8.2f %8.2f %8.2f %10.2f %10.3f %12.3f\n",
                    bits_name(a.bits), arm_name(a.kind), a.med, a.p10, a.p90,
                    a.gib_s, vs_f32, vs_e2e);
    }

    std::printf("\nSpeedup summary (median F32 / arm; >1 is faster):\n");
    for (int bits : bits_list) {
        const float f32 = find_med(bits, ArmKind::F32);
        const float d2 = find_med(bits, ArmKind::Dot2);
        const float d4 = find_med(bits, ArmKind::Dot4Q8Ready);
        const float e2e = find_med(bits, ArmKind::Q8ConvertDot4);
        std::printf("  %s: dot2=%.3fx  dot4-q8-ready=%.3fx  "
                    "q8-convert+dot4=%.3fx  (vs F32)\n",
                    bits_name(bits),
                    d2 > 0.f ? f32 / d2 : 0.f,
                    d4 > 0.f ? f32 / d4 : 0.f,
                    e2e > 0.f ? f32 / e2e : 0.f);
    }

    std::printf("\nchecksum_sink=0x%016llx  (anti-DCE)\n",
                (unsigned long long)sink);
    std::printf("identity: %s\n", kBenchId);
    std::printf("CORRECTNESS: PASS\n");

    for (ArmState& a : arms) free_y(a.y);
    CK(hipFree(dX));
    CK(hipFree(dA4));
    CK(hipFree(dA6));
    CK(hipFree(dXq));
    return 0;
}
