// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX MMDiT pre-attention/pre-MLP norm: weightless LayerNorm fused with the
// adaLN-Zero modulation affine, with the f32 -> f16 cast folded into the
// store.
//
// The MMDiT forward runs this triple as three separate launches over the same
// [n_rows, 3072] activation — `layernorm_f32` (gamma = 1, beta = 0), then
// `modulate_f32`, then `cast_f32_to_f16` for the GEMM that consumes it — so
// each row is read and written three times for what is one pass of work. Here
// it is one launch, one read, one write, and the store already carries the
// dtype the next kernel wants.
//
//   out[r, i] = LN(x[r])[i] * (1 + scale[i]) + shift[i]
//   LN(x[r])[i] = (x[r,i] - mean(x[r])) * rsqrt(var(x[r]) + eps)
//
// Layouts: x [n_rows, d] f32, shift [d] f32, scale [d] f32, out [n_rows, d]
// in `OutT`, row-major. One workgroup per row; block size is chosen by the
// launcher as `min(256, d)` rounded up to a power of two, exactly as
// `Gpu::layernorm_batched` does, because the reduction tree — and hence the
// rounding — depends on it.
//
// BIT-IDENTITY. The f32 entry point is bit-identical to `layernorm_batched`
// (gamma = 1, beta = 0) followed by `modulate_f32`, not merely close. That is
// a deliberate constraint, not a happy accident: the FLUX forward is validated
// against a golden latent, so the fusion must be provably a no-op on the
// numbers. Three things keep it exact:
//
//   1. The same strided partition (`i = tid; i < n; i += blockDim.x`) and the
//      same per-thread accumulation order as `layernorm_f32`.
//   2. The same reduction TREE. See `ln_block_sum` below — a shfl butterfly
//      is not automatically the same association as an LDS halving loop, and
//      float addition is not associative.
//   3. `gamma[i] * (x - mean) * inv_std + beta[i]` with gamma = 1, beta = 0
//      collapses to `(x - mean) * inv_std + 0.0f`: `1.0f * t == t` exactly for
//      every t, and the `+ 0.0f` is kept rather than dropped so that a `-0.0`
//      product lands on `+0.0` the way the unfused pair does.
//
// The f16 entry point is the f32 result put through the same round-to-nearest-
// even `(_Float16)` conversion `cast_f32_to_f16` uses, so it is bit-identical
// to the four-launch chain too.

// Block sum reproducing `layernorm_f32`'s association exactly, with five of
// its eight barriers removed.
//
// The LDS halving loop combines LDS slots by bit: `s = 128` pairs slot `t`
// with `t ^ 128`, `s = 64` pairs `t` with `t ^ 64`, and so on down to
// `s = 1`. A `__shfl_xor` butterfly over `2^k` lanes combines by exactly the
// same bits in exactly the same order, so lane 0's accumulator visits the same
// operands, in the same groupings, as slot 0 of the halving loop — the trees
// are congruent and the rounding is identical. What a butterfly cannot do is
// reach ACROSS waves, so the levels above the wave boundary (`s >= 32`) still
// go through LDS; only the intra-wave tail becomes register traffic. Reversing
// that split — butterfly first, then combine wave sums — would build a
// differently-shaped tree and would NOT be bit-identical.
__device__ __forceinline__ float ln_block_sum(float* sdata, float v) {
    const int tid = threadIdx.x;
    sdata[tid] = v;
    __syncthreads();

    // Cross-wave levels: LDS, one barrier each.
    for (int s = (int)(blockDim.x >> 1); s >= 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Intra-wave tail: no LDS round trip, no barrier. `tail` is the number of
    // live slots left, which is 32 for any block of 32+ threads and the whole
    // block below that (d < 32 is a lab shape, not a FLUX one, but the
    // launcher permits it so the kernel must handle it).
    const int tail = (blockDim.x < 32u) ? (int)blockDim.x : 32;
    if (tid < tail) {
        float t = sdata[tid];
        for (int off = tail >> 1; off > 0; off >>= 1) {
            t += __shfl_xor(t, off, 32);
        }
        if (tid == 0) sdata[0] = t;
    }
    __syncthreads();

    const float total = sdata[0];
    __syncthreads(); // caller reuses sdata for the next reduction
    return total;
}

// Pin `v` in a VGPR as a rounded f32 before anything narrows it. Empty asm,
// zero instructions, but it stops the backend folding the modulation FMA and
// the following `f32 -> f16` conversion into one `v_fma_mixlo_f16`. That fold
// rounds ONCE, straight from the exact product-sum to f16, where the chain
// this replaces rounds twice — to f32 in `modulate_f32`, then to f16 in
// `cast_f32_to_f16`. The two disagree by 1 ULP on roughly one f16 word in
// 100k, which is harmless numerically but breaks the bit-identity the f32
// entry point already gives us, and a parity gate that has to carry an
// exception is a parity gate nobody trusts. Measured on gfx1150: without
// this, 2 of 3 shapes mismatched (e.g. c175 vs c176).
__device__ __forceinline__ float materialize_f32(float v) {
    asm volatile("" : "+v"(v));
    return v;
}

template <typename OutT>
__device__ __forceinline__ void layernorm_modulate_body(
    const float* __restrict__ x,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    OutT* __restrict__ out,
    int n, float eps,
    float* sdata) {

    const int tid = threadIdx.x;
    const float* xi = x + (size_t)blockIdx.x * n;
    OutT* oi = out + (size_t)blockIdx.x * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += xi[i];
    const float mean = ln_block_sum(sdata, sum) / (float)n;

    float var_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float d = xi[i] - mean;
        var_sum += d * d;
    }
    const float inv_std = rsqrtf(ln_block_sum(sdata, var_sum) / (float)n + eps);

    for (int i = tid; i < n; i += blockDim.x) {
        // See BIT-IDENTITY note 3 above for the trailing `+ 0.0f`.
        const float normed = (xi[i] - mean) * inv_std + 0.0f;
        oi[i] = (OutT)materialize_f32(normed * (1.0f + scale[i]) + shift[i]);
    }
}

extern "C" __global__ __launch_bounds__(256) void layernorm_modulate_f32(
    const float* __restrict__ x,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int n, float eps) {
    extern __shared__ float sdata[];
    layernorm_modulate_body<float>(x, shift, scale, out, n, eps, sdata);
}

extern "C" __global__ __launch_bounds__(256) void layernorm_modulate_f16(
    const float* __restrict__ x,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    _Float16* __restrict__ out,
    int n, float eps) {
    extern __shared__ float sdata[];
    layernorm_modulate_body<_Float16>(x, shift, scale, out, n, eps, sdata);
}
