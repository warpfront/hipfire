// SPDX-License-Identifier: Apache-2.0
// hipfire — Qwen4 MoE row fold: top-10 combine + shared fold + HC write + the
// next HC read's norm/gate projection, one block per token.
// Idea from Gufo upstream (gufo-org/gufo @1071b361, MIT): row-level ownership of
// the MoE/shared fold, the HC update and the next normalization.
//
// Bytewise the chain it replaces (BF16 HC streams, 4 branches, hidden % 256 == 0,
// hidden <= 2560):
//   moe_down_combine_grouped_top10_bf16in_zinit      (routed = ten rows in order)
//   gemm_wmma_lds_128_256_32_64_k64_hcsd epilogue    (fold with sigmoid-selector * shared, HC write)
//   hyper_norm_gate_outputs                          (next read: norm, F16 row, next write's gates)
// Every expression is copied from those: combine from moe_down_combine_grouped_bf16in
// (initial_zero), fold + write from the hcw_* epilogue block (v_mul/v_add pinned for
// NaN operand order), norm + gate projection from hyper_norm_gate_outputs_body.
//
// Grid [tokens], block 256, dynamic LDS (hidden * 8 + 4 * 256 * 4 + hidden * 2 + 16)
// bytes.  `streams` is rewritten in place (a token's block only touches its row).
//
// hc_row_fold_norm_gate is the gfx1151 entry (its text, and so its code object, is
// frozen).  The gfx12 entries `hc_row_fold_moe_gfx1201` / `hc_row_write_norm_gate_gfx1201`
// at the end of the file are the same fold for the gfx1201 default chain.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}
static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float hcw_bf16f(float v) {
    return static_cast<float>(static_cast<hip_bfloat16>(v));
}
static __device__ __forceinline__ unsigned int hcw_bf16bits(float v) {
    return (unsigned int)static_cast<hip_bfloat16>(v).data;
}
static __device__ __forceinline__ float hcw_mul(float a, float b) {
    float r;
    asm("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
static __device__ __forceinline__ float hcw_add(float a, float b) {
    float r;
    asm("v_add_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
    return r;
}
// fma(a, b, c) with the operand order pinned: hyper_norm_gate_outputs' gate projection
// compiles to fma(weight, x, acc), and which NaN a mixed-NaN row propagates follows the
// operand order of every fma/add in the chain, so the chain below is pinned to that ISA.
static __device__ __forceinline__ float hcw_fma(float a, float b, float c) {
    float r;
    asm("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
    return r;
}
static __device__ __forceinline__ float hcw_gate(float logit) {
    const float gate_logit = hcw_bf16f(logit);
    const float gate_scaled = hcw_bf16f(gate_logit / 4.0f);
    return hcw_bf16f(2.0f * hcw_bf16f(1.0f / (1.0f + expf(-gate_scaled))));
}

static __device__ __forceinline__ float hc_lane0_tree_sum(float v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x108, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x104, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x102, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x101, 0xf, 0xf, true));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
#endif
    return v;
}

extern "C" __global__ __launch_bounds__(256) void hc_row_fold_norm_gate(
    const unsigned short* __restrict__ grouped_down,  // [grouped_rows, hidden] BF16 bits
    const int2* __restrict__ order,                    // [tokens * 10] moe_combine_order_top10
    const unsigned short* __restrict__ shared_bf16,   // [tokens, hidden] BF16 bits (shared down)
    const float* __restrict__ selector,                // [tokens]
    unsigned short* streams,                           // [tokens, 4 * hidden] BF16 bits, in place
    const float* __restrict__ gates,                   // [tokens, 4] this write's gate logits
    const __hip_bfloat16* __restrict__ norm_weight,   // next read's norm weight [4 * hidden]
    const unsigned short* __restrict__ gate_weight,   // next write's BF16 gate weight [4, 4 * hidden]
    float* __restrict__ next_gates,                    // [tokens, 4]
    _Float16* __restrict__ normalized16,               // [tokens, 4 * hidden] F16
    int hidden) {
    constexpr int branches = 4;
    constexpr int U = 10;
    extern __shared__ float hc_lds[];
    const int wide = branches * hidden;
    const int per = hidden / 256;
    unsigned short* xsb = (unsigned short*)hc_lds;
    float* rms_sum = hc_lds + wide / 2;
    unsigned int* mixed_lds = (unsigned int*)(rms_sum + branches * 256);  // hidden / 2 words
    float* gate_lds = (float*)(mixed_lds + hidden / 2);                   // 4 floats
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const size_t in = (size_t)row * wide;

    // --- combine (initial_zero) + shared fold: two columns per thread step -----
    {
        const size_t route_base = (size_t)row * 10;
        int rowidx[10];
        float wts[10];
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const int2 o = order[route_base + c];
            rowidx[c] = o.x;
            wts[c] = __int_as_float(o.y);
        }
        const float gate_sel = hcw_bf16f(selector[row]);
        for (int p = tid; p < hidden / 2; p += 256) {
            unsigned int w[10];
#pragma unroll
            for (int c = 0; c < 10; ++c)
                w[c] = rowidx[c] >= 0
                    ? *(const unsigned int*)(grouped_down + (size_t)rowidx[c] * hidden + 2 * p)
                    : 0u;
            const unsigned int sw = *(const unsigned int*)(shared_bf16 + (size_t)row * hidden + 2 * p);
            float routed[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                __hip_bfloat16 accumulated = bf16_round(0.0f);
#pragma unroll
                for (int c = 0; c < 10; ++c) {
                    const float raw_output = __uint_as_float((e & 1) ? (w[c] & 0xFFFF0000u) : (w[c] << 16));
                    const __hip_bfloat16 output = bf16_round(raw_output);
                    const __hip_bfloat16 weighted = bf16_round(bf16_to_f32(output) * wts[c]);
                    accumulated = bf16_round(bf16_to_f32(accumulated) + bf16_to_f32(weighted));
                }
                routed[e] = bf16_to_f32(accumulated);
            }
            const float r0 = __uint_as_float(sw << 16);
            const float r1 = __uint_as_float(sw & 0xFFFF0000u);
            const float res0 = hcw_bf16f(hcw_add(hcw_bf16f(routed[0]), hcw_bf16f(hcw_mul(r0, gate_sel))));
            const float res1 = hcw_bf16f(hcw_add(hcw_bf16f(routed[1]), hcw_bf16f(hcw_mul(r1, gate_sel))));
            mixed_lds[p] = hcw_bf16bits(res0) | (hcw_bf16bits(res1) << 16);
        }
        if (tid < branches) gate_lds[tid] = hcw_gate(gates[(size_t)row * branches + tid]);
    }
    __syncthreads();

    // --- HC write: stream += gate * mixed, 8 columns per unit, 4 * hidden / 8 units ---
    {
        const int chunks = hidden / 8;
        for (int u8 = tid; u8 < branches * chunks; u8 += 256) {
            const int br = u8 / chunks;
            const int c0 = (u8 - br * chunks) * 8;
            const uint4 mq = *(const uint4*)((const unsigned short*)mixed_lds + c0);
            const uint4 sw = *(const uint4*)(streams + in + (size_t)br * hidden + c0);
            const unsigned int mw[4] = {mq.x, mq.y, mq.z, mq.w};
            const unsigned int sin[4] = {sw.x, sw.y, sw.z, sw.w};
            const float g = gate_lds[br];
            unsigned int out[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float m0 = __uint_as_float(mw[p] << 16);
                const float m1 = __uint_as_float(mw[p] & 0xFFFF0000u);
                const float lo = hcw_add(__uint_as_float(sin[p] << 16), hcw_bf16f(hcw_mul(g, m0)));
                const float hi = hcw_add(__uint_as_float(sin[p] & 0xFFFF0000u), hcw_bf16f(hcw_mul(g, m1)));
                out[p] = hcw_bf16bits(lo) | (hcw_bf16bits(hi) << 16);
            }
            const uint4 ov = make_uint4(out[0], out[1], out[2], out[3]);
            *(uint4*)(streams + in + (size_t)br * hidden + c0) = ov;
            *(uint4*)(xsb + (size_t)br * hidden + c0) = ov;
        }
    }
    __syncthreads();

    // --- next read: hyper_norm_gate_outputs_body (state_bf16, normalized16 only) ---
    float raw[branches][U];
#pragma unroll
    for (int b = 0; b < branches; ++b)
#pragma unroll
        for (int u = 0; u < U; ++u)
            raw[b][u] = u < per
                ? __uint_as_float((unsigned int)xsb[b * hidden + tid + 256 * u] << 16)
                : 0.0f;
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        float partial = 0.0f;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            raw[b][u] = bf16_to_f32(bf16_round(raw[b][u]));
            if (u < per) partial += raw[b][u] * raw[b][u];
        }
        rms_sum[b * 256 + tid] = partial;
    }
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
#pragma unroll
            for (int b = 0; b < branches; ++b)
                rms_sum[b * 256 + tid] += rms_sum[b * 256 + tid + stride];
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < branches; ++b) {
        const float inv = rsqrtf(rms_sum[b * 256] / (float)hidden + 1.0e-6f);
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = b * hidden + tid + 256 * u;
            if (u < per)
                xsb[j] = (unsigned short)(__float_as_uint(bf16_to_f32(bf16_round(
                    raw[b][u] * inv * (1.0f + bf16_to_f32(norm_weight[j]))))) >> 16);
        }
    }
    __syncthreads();
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave >= branches) {
        const int t2 = tid - branches * 32;
        for (int v = t2; v < wide / 8; v += 4 * 32) {
            const uint4 xv = ((const uint4*)xsb)[v];
            typedef _Float16 half8v __attribute__((ext_vector_type(8)));
            half8v h;
            h[0] = (_Float16)__uint_as_float(xv.x << 16);
            h[1] = (_Float16)__uint_as_float(xv.x & 0xFFFF0000u);
            h[2] = (_Float16)__uint_as_float(xv.y << 16);
            h[3] = (_Float16)__uint_as_float(xv.y & 0xFFFF0000u);
            h[4] = (_Float16)__uint_as_float(xv.z << 16);
            h[5] = (_Float16)__uint_as_float(xv.z & 0xFFFF0000u);
            h[6] = (_Float16)__uint_as_float(xv.w << 16);
            h[7] = (_Float16)__uint_as_float(xv.w & 0xFFFF0000u);
            ((half8v*)(normalized16 + in))[v] = h;
        }
        return;
    }
    const unsigned short* w_row = gate_weight + (size_t)wave * wide;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    constexpr int BATCH = 10;
    int i = lane * 8;
    for (; i + 8 + (BATCH - 1) * 256 <= wide; i += BATCH * 256) {
        uint4 wb[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) wb[u] = *(const uint4*)(w_row + i + u * 256);
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const uint4 wv = wb[u];
            const uint4 xv = *(const uint4*)(xsb + i + u * 256);
            const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
            const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
            a0 = hcw_fma(__uint_as_float(wv.x << 16), xa.x, a0);
            a1 = hcw_fma(__uint_as_float(wv.x & 0xFFFF0000u), xa.y, a1);
            a2 = hcw_fma(__uint_as_float(wv.y << 16), xa.z, a2);
            a3 = hcw_fma(__uint_as_float(wv.y & 0xFFFF0000u), xa.w, a3);
            a0 = hcw_fma(__uint_as_float(wv.z << 16), xb.x, a0);
            a1 = hcw_fma(__uint_as_float(wv.z & 0xFFFF0000u), xb.y, a1);
            a2 = hcw_fma(__uint_as_float(wv.w << 16), xb.z, a2);
            a3 = hcw_fma(__uint_as_float(wv.w & 0xFFFF0000u), xb.w, a3);
        }
    }
    for (; i + 8 <= wide; i += 32 * 8) {
        const uint4 wv = *(const uint4*)(w_row + i);
        const uint4 xv = *(const uint4*)(xsb + i);
        const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
        const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
        a0 = hcw_fma(__uint_as_float(wv.x << 16), xa.x, a0);
        a1 = hcw_fma(__uint_as_float(wv.x & 0xFFFF0000u), xa.y, a1);
        a2 = hcw_fma(__uint_as_float(wv.y << 16), xa.z, a2);
        a3 = hcw_fma(__uint_as_float(wv.y & 0xFFFF0000u), xa.w, a3);
        a0 = hcw_fma(__uint_as_float(wv.z << 16), xb.x, a0);
        a1 = hcw_fma(__uint_as_float(wv.z & 0xFFFF0000u), xb.y, a1);
        a2 = hcw_fma(__uint_as_float(wv.w << 16), xb.z, a2);
        a3 = hcw_fma(__uint_as_float(wv.w & 0xFFFF0000u), xb.w, a3);
    }
    // hyper_norm_gate_outputs compiles (a0 + a1) + (a2 + a3) to (a3 + a2) + (a1 + a0).
    const float sum = hc_lane0_tree_sum(hcw_add(hcw_add(a3, a2), hcw_add(a1, a0)));
    if (lane == 0) next_gates[(size_t)row * branches + wave] = sum;
}

#if defined(__GFX12__)
// ---------------------------------------------------------------------------------------
// gfx1201 row folds: bytewise the gfx1201 default prefill chain (the hcw / hcsd epilogues
// stay off on gfx1201), expressions copied as SOURCE TEXT from the replaced kernels --
// which NaN a mixed-NaN row propagates follows the compiled operand order of the unfused
// kernels, which the byte oracles in rdna-compute/src/tensor_ops.rs
// (hc_row_fold_gfx1201_*) gate against the unfused launches, NaN payloads included.  The
// operand order of the adds / multiplies is pinned (hcw_*) to what those kernels compile to.
//
//   hc_row_fold_moe_gfx1201 (MOE = true), the MoE sublayer tail:
//     moe_down_combine_grouped_top10_bf16in_zinit   routed rows (F32, never stored)
//     bf16_scaled_add_batched_f32                   + sigmoid-selector * shared down (F32 rows)
//     hyper_write_bf16x8_gfx1201                    streams += gate * mixed
//     hyper_norm_gate_outputs                       next read's norm, F16 row, next gates
//   hc_row_write_norm_gate_gfx1201 (MOE = false), the mixer tail:
//     hyper_write_bf16x8_gfx1201 of the out projection's F32 `mixed`, then
//     hyper_norm_gate_outputs
//
// Grid [tokens], block 320, dynamic LDS 4 * hidden * 2 + 4 * 256 * 4 + 16 bytes; four
// branches, hidden == 2560 (the launcher's gate).  `streams` is rewritten in place.
//
// Geometry: thread t owns columns [8t, 8t + 8) of every branch (320 threads = 2560 / 8).
// Everything global a token needs -- the ten gathered expert rows, the shared / mixed row,
// the four stream chunks -- is requested before the first use, and the combine -> fold ->
// HC write chain stays in registers (no `mixed` LDS round trip, no barrier between them);
// the written chunks land in the LDS row `xsb` the norm and the gate projection read.
// The norm (waves 0..7: 256 partial sums per branch, the old stride 128..1 LDS tree) and
// the gate projection (waves 0..3, one branch each) are the replaced kernels' expressions in
// their reduction order; the tree's last five levels run in one wave instead of five
// barriers.
//
// bf16 rounding: every value that reaches a rounding below is a bf16-exact float or the
// product / sum of two (NaN payloads then carry no bits below bit 16), for which the
// reference rounding is `(u + 0x7fff + bit16) & 0xffff0000` -- no NaN / inf split.  The only
// F32 values with arbitrary payloads are the route weights (checked per token: a NaN weight
// with bits in its low half sends the block through the reference rounding), the shared /
// mixed rows, the selector and the gate logits (rounded once, by the reference rounding).
constexpr int HCF_H = 2560;
constexpr int HCF_WIDE = 4 * HCF_H;
constexpr int HCF_THREADS = HCF_H / 8;

static __device__ __forceinline__ float hcf_bf16_exact(float x) {
    const unsigned int u = __float_as_uint(x);
    return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}
static __device__ __forceinline__ float hcf_bf16_full(float x) {
    return bf16_to_f32(bf16_round(x));
}
template <bool FAST>
static __device__ __forceinline__ float hcf_rnd(float x) {
    if constexpr (FAST) return hcf_bf16_exact(x);
    else return hcf_bf16_full(x);
}

// The old stride-128..1 LDS tree over 256 partial sums, `rms_sum[t] += rms_sum[t + s]`
// (compiled other + own), for branch row `p` of 256 floats: lanes hold p[lane + 32 k];
// returns the total in lane 0.
static __device__ __forceinline__ float hcf_tree256(const float* p, int lane) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) v[k] = p[lane + 32 * k];
    const float a0 = hcw_add(v[4], v[0]);
    const float a1 = hcw_add(v[5], v[1]);
    const float a2 = hcw_add(v[6], v[2]);
    const float a3 = hcw_add(v[7], v[3]);
    const float b0 = hcw_add(a2, a0);
    const float b1 = hcw_add(a3, a1);
    float s = hcw_add(b1, b0);
    s = hcw_add(__int_as_float(__builtin_amdgcn_permlanex16(
                    __float_as_int(s), __float_as_int(s), 0x76543210u, 0xfedcba98u, false, false)), s);
    s = hcw_add(__int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(s), 0x108, 0xf, 0xf, true)), s);
    s = hcw_add(__int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(s), 0x104, 0xf, 0xf, true)), s);
    s = hcw_add(__int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(s), 0x102, 0xf, 0xf, true)), s);
    s = hcw_add(__int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(s), 0x101, 0xf, 0xf, true)), s);
    return s;
}

// Combine (initial_zero) + scaled shared add of one column: the ten expert values in
// order, then `+ rhs * gate`.  `w` holds the column's word of each expert row, `hi` selects
// the upper half.
template <bool FAST>
static __device__ __forceinline__ float hcf_moe_column(
    const unsigned int (&w)[10], const bool hi, const float (&wts)[10], const float shared_value,
    const float gate) {
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const float raw_output = __uint_as_float(hi ? (w[c] & 0xFFFF0000u) : (w[c] << 16));
        const float weighted = hcf_rnd<FAST>(hcw_mul(wts[c], raw_output));
        acc = hcf_rnd<FAST>(hcw_add(acc, weighted));
    }
    // bf16_scaled_add_batched_f32: residual = the combine's F32 output.
    const float rhs = hcf_bf16_full(shared_value);
    const float product = hcf_rnd<FAST>(hcw_mul(rhs, gate));
    return hcf_rnd<FAST>(hcw_add(acc, product));
}

template <bool MOE>
static __device__ __forceinline__ void hc_row_fold_gfx12_body(
    const unsigned short* __restrict__ grouped_down,  // MOE: [grouped_rows, hidden] BF16 bits
    const int2* __restrict__ order,                    // MOE: [tokens * 10] moe_combine_order_top10
    const float* __restrict__ shared_or_mixed,         // MOE: shared down F32 / else mixed F32 [tokens, hidden]
    const float* __restrict__ selector,                // MOE: [tokens]
    unsigned short* streams,                           // [tokens, 4 * hidden] BF16 bits, in place
    const float* __restrict__ gates,                   // [tokens, 4] this write's gate logits
    const __hip_bfloat16* __restrict__ norm_weight,   // next read's norm weight [4 * hidden]
    const unsigned short* __restrict__ gate_weight,   // next write's BF16 gate weight [4, 4 * hidden]
    float* __restrict__ next_gates,                    // [tokens, 4]
    _Float16* __restrict__ normalized16,               // [tokens, 4 * hidden] F16
    int hidden) {                                      // == HCF_H
    constexpr int branches = 4;
    constexpr int U = HCF_H / 256;
    extern __shared__ __attribute__((aligned(16))) unsigned char hcf_lds[];
    unsigned short* xsb = (unsigned short*)hcf_lds;                       // [4, hidden] BF16 bits
    float* rms_sum = (float*)(hcf_lds + (size_t)branches * HCF_H * 2);    // [4, 256]
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const size_t in = (size_t)row * HCF_WIDE;
    const int c0 = tid * 8;

    // --- everything global this token's fold reads, requested up front ------------------
    uint4 sin[branches];
#pragma unroll
    for (int br = 0; br < branches; ++br)
        sin[br] = *(const uint4*)(streams + in + (size_t)br * HCF_H + c0);
    uint4 gw[10];
    float shv[8];
    float wts[10];
    bool slow = false;
    if constexpr (MOE) {
        const size_t route_base = (size_t)row * 10;
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            const int2 o = order[route_base + c];
            wts[c] = __int_as_float(o.y);
            const unsigned int wu = (unsigned int)o.y;
            slow |= ((wu & 0x7f800000u) == 0x7f800000u) && ((wu & 0xffffu) != 0u);
            gw[c] = o.x >= 0
                ? *(const uint4*)(grouped_down + (size_t)o.x * HCF_H + c0)
                : make_uint4(0u, 0u, 0u, 0u);
        }
    }
    {
        const float4* src = (const float4*)(shared_or_mixed + (size_t)row * HCF_H + c0);
        const float4 s0 = src[0];
        const float4 s1 = src[1];
        shv[0] = s0.x; shv[1] = s0.y; shv[2] = s0.z; shv[3] = s0.w;
        shv[4] = s1.x; shv[5] = s1.y; shv[6] = s1.z; shv[7] = s1.w;
    }
    // The four HC write gates: one lane each, then broadcast.
    const float gate_lane = hcw_gate(gates[(size_t)row * branches + (lane & 3)]);
    float g[branches];
#pragma unroll
    for (int br = 0; br < branches; ++br)
        g[br] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(gate_lane), br));

    // --- mixed: MOE, combine (initial_zero) + bf16_scaled_add; else the F32 input ---------
    float m[8];
    if constexpr (MOE) {
        const float gate = hcf_bf16_full(selector[row]);
        unsigned int wcol[4][10];
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            wcol[0][c] = gw[c].x;
            wcol[1][c] = gw[c].y;
            wcol[2][c] = gw[c].z;
            wcol[3][c] = gw[c].w;
        }
        if (!slow) {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                m[2 * p] = hcf_moe_column<true>(wcol[p], false, wts, shv[2 * p], gate);
                m[2 * p + 1] = hcf_moe_column<true>(wcol[p], true, wts, shv[2 * p + 1], gate);
            }
        } else {
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                m[2 * p] = hcf_moe_column<false>(wcol[p], false, wts, shv[2 * p], gate);
                m[2 * p + 1] = hcf_moe_column<false>(wcol[p], true, wts, shv[2 * p + 1], gate);
            }
        }
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) m[e] = hcf_bf16_full(shv[e]);
    }

    // --- hyper_write_bf16x8_gfx1201: stream += gate * mixed, 8 columns per unit -----------
#pragma unroll
    for (int br = 0; br < branches; ++br) {
        const unsigned int sw[4] = {sin[br].x, sin[br].y, sin[br].z, sin[br].w};
        unsigned int out[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            // The compiled kernel adds stream + injection (src0 = the stream word).
            const float lo = hcf_bf16_exact(hcw_add(
                __uint_as_float(sw[p] << 16), hcf_bf16_exact(hcw_mul(g[br], m[2 * p]))));
            const float hi = hcf_bf16_exact(hcw_add(
                __uint_as_float(sw[p] & 0xFFFF0000u), hcf_bf16_exact(hcw_mul(g[br], m[2 * p + 1]))));
            out[p] = (__float_as_uint(lo) >> 16) | __float_as_uint(hi);
        }
        const uint4 ov = make_uint4(out[0], out[1], out[2], out[3]);
        *(uint4*)(streams + in + (size_t)br * HCF_H + c0) = ov;
        *(uint4*)(xsb + (size_t)br * HCF_H + c0) = ov;
    }
    __syncthreads();

    // --- next read: hyper_norm_gate_outputs_body (state_bf16, normalized16 only) ---------
    float raw[branches][U];
    if (tid < 256) {
#pragma unroll
        for (int b = 0; b < branches; ++b)
#pragma unroll
            for (int u = 0; u < U; ++u)
                raw[b][u] = __uint_as_float((unsigned int)xsb[b * HCF_H + tid + 256 * u] << 16);
#pragma unroll
        for (int b = 0; b < branches; ++b) {
            // The replaced kernel compiles `partial += raw * raw` to a mul, then fma(raw, raw, partial).
            float partial = hcw_mul(raw[b][0], raw[b][0]);
#pragma unroll
            for (int u = 1; u < U; ++u) partial = hcw_fma(raw[b][u], raw[b][u], partial);
            rms_sum[b * 256 + tid] = partial;
        }
    }
    __syncthreads();
    if (tid < 256) {
#pragma unroll
        for (int b = 0; b < branches; ++b) {
            const float total = __int_as_float(__builtin_amdgcn_readfirstlane(
                __float_as_int(hcf_tree256(rms_sum + b * 256, lane))));
            const float inv = rsqrtf(total / (float)hidden + 1.0e-6f);
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int j = b * HCF_H + tid + 256 * u;
                // raw * inv * (1 + weight), compiled by the replaced kernel as (inv * raw) * (1 + w).
                xsb[j] = (unsigned short)(__float_as_uint(hcf_bf16_exact(hcw_mul(
                    hcw_mul(inv, raw[b][u]), hcw_add(1.0f, bf16_to_f32(norm_weight[j]))))) >> 16);
            }
        }
    }
    __syncthreads();
    const int wave = tid >> 5;
    if (wave >= branches) {
        const int t2 = tid - branches * 32;
        for (int v = t2; v < HCF_WIDE / 8; v += (HCF_THREADS - branches * 32)) {
            const uint4 xv = ((const uint4*)xsb)[v];
            typedef _Float16 half8v __attribute__((ext_vector_type(8)));
            half8v h;
            h[0] = (_Float16)__uint_as_float(xv.x << 16);
            h[1] = (_Float16)__uint_as_float(xv.x & 0xFFFF0000u);
            h[2] = (_Float16)__uint_as_float(xv.y << 16);
            h[3] = (_Float16)__uint_as_float(xv.y & 0xFFFF0000u);
            h[4] = (_Float16)__uint_as_float(xv.z << 16);
            h[5] = (_Float16)__uint_as_float(xv.z & 0xFFFF0000u);
            h[6] = (_Float16)__uint_as_float(xv.w << 16);
            h[7] = (_Float16)__uint_as_float(xv.w & 0xFFFF0000u);
            ((half8v*)(normalized16 + in))[v] = h;
        }
        return;
    }
    const unsigned short* w_row = gate_weight + (size_t)wave * HCF_WIDE;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    constexpr int BATCH = 10;
    int i = lane * 8;
#pragma unroll 1
    for (; i + 8 + (BATCH - 1) * 256 <= HCF_WIDE; i += BATCH * 256) {
        uint4 wb[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) wb[u] = *(const uint4*)(w_row + i + u * 256);
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const uint4 wv = wb[u];
            const uint4 xv = *(const uint4*)(xsb + i + u * 256);
            const float4 xa = {__uint_as_float(xv.x << 16), __uint_as_float(xv.x & 0xFFFF0000u), __uint_as_float(xv.y << 16), __uint_as_float(xv.y & 0xFFFF0000u)};
            const float4 xb = {__uint_as_float(xv.z << 16), __uint_as_float(xv.z & 0xFFFF0000u), __uint_as_float(xv.w << 16), __uint_as_float(xv.w & 0xFFFF0000u)};
            a0 = hcw_fma(__uint_as_float(wv.x << 16), xa.x, a0);
            a1 = hcw_fma(__uint_as_float(wv.x & 0xFFFF0000u), xa.y, a1);
            a2 = hcw_fma(__uint_as_float(wv.y << 16), xa.z, a2);
            a3 = hcw_fma(__uint_as_float(wv.y & 0xFFFF0000u), xa.w, a3);
            a0 = hcw_fma(__uint_as_float(wv.z << 16), xb.x, a0);
            a1 = hcw_fma(__uint_as_float(wv.z & 0xFFFF0000u), xb.y, a1);
            a2 = hcw_fma(__uint_as_float(wv.w << 16), xb.z, a2);
            a3 = hcw_fma(__uint_as_float(wv.w & 0xFFFF0000u), xb.w, a3);
        }
    }
    const float sum = hc_lane0_tree_sum(hcw_add(hcw_add(a3, a2), hcw_add(a1, a0)));
    if (lane == 0) next_gates[(size_t)row * branches + wave] = sum;
}

extern "C" __global__ __launch_bounds__(HCF_THREADS) void hc_row_fold_moe_gfx1201(
    const unsigned short* __restrict__ grouped_down,  // [grouped_rows, hidden] BF16 bits
    const int2* __restrict__ order,                    // [tokens * 10] moe_combine_order_top10
    const float* __restrict__ shared_f32,              // [tokens, hidden] F32 (shared down)
    const float* __restrict__ selector,                // [tokens]
    unsigned short* streams,                           // [tokens, 4 * hidden] BF16 bits, in place
    const float* __restrict__ gates,                   // [tokens, 4] this write's gate logits
    const __hip_bfloat16* __restrict__ norm_weight,   // next read's norm weight [4 * hidden]
    const unsigned short* __restrict__ gate_weight,   // next write's BF16 gate weight [4, 4 * hidden]
    float* __restrict__ next_gates,                    // [tokens, 4]
    _Float16* __restrict__ normalized16,               // [tokens, 4 * hidden] F16
    int hidden) {
    hc_row_fold_gfx12_body<true>(grouped_down, order, shared_f32, selector, streams, gates,
                                 norm_weight, gate_weight, next_gates, normalized16, hidden);
}

extern "C" __global__ __launch_bounds__(HCF_THREADS) void hc_row_write_norm_gate_gfx1201(
    const float* __restrict__ mixed,                   // [tokens, hidden] F32
    unsigned short* streams,                           // [tokens, 4 * hidden] BF16 bits, in place
    const float* __restrict__ gates,                   // [tokens, 4] this write's gate logits
    const __hip_bfloat16* __restrict__ norm_weight,   // next read's norm weight [4 * hidden]
    const unsigned short* __restrict__ gate_weight,   // next write's BF16 gate weight [4, 4 * hidden]
    float* __restrict__ next_gates,                    // [tokens, 4]
    _Float16* __restrict__ normalized16,               // [tokens, 4 * hidden] F16
    int hidden) {
    hc_row_fold_gfx12_body<false>(nullptr, nullptr, mixed, nullptr, streams, gates, norm_weight,
                                  gate_weight, next_gates, normalized16, hidden);
}
#endif
