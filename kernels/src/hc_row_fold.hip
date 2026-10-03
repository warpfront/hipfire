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
    _Float16* __restrict__ normalized16,               // [tokens, ld16] F16, ld16 >= 4 * hidden
    int hidden,
    int ld16) {
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
            ((half8v*)(normalized16 + (size_t)row * ld16))[v] = h;
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
