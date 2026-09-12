// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Philipp Hug
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>

// FLUX MMDiT Q/K head prep: per-(row, head) RMSNorm fused with the 2D axial
// RoPE, with the f32 <-> f16 conversions folded into the load and the store.
//
// The forward runs this as three or four launches over the same
// [n_txt + n_img, heads*head_dim] buffer — `rmsnorm_batched`, then
// `rope_2d_flux_f32_fast`, then a `cast_f32_to_f16` for the attention kernel
// that consumes Q and K — and the QK-norm launch alone reads and writes the
// whole tensor. Here it is one launch: one read, one write, and the store
// already carries the dtype the next kernel wants. All four dtype
// combinations are instantiated so the caller never needs a separate cast.
//
//   norm[r, h, i]  = x[r, h, i] * scale[i] * rsqrt(mean(x[r,h]^2) + eps)
//   out[r, h, ...] = rope_2d(norm[r, h, ...])   for image rows only
//
// Text rows — the first `n_txt` rows in the BFL text-first concat — get the
// norm and NO rotation, exactly as the unfused pair does (there the rope
// launch is passed `row_offset = n_txt` and never touches them).
//
// The rotation math is `rope_2d_flux_f32_fast`'s, verbatim: `expf/logf`
// instead of the f64 `pow`, and the zero-position axis skipped as the exact
// identity it is. See that file for why each of those is not an accuracy
// trade.
//
// SHAPE. One wave per (row, head), 8 waves per 256-thread block. The RMS
// reduction needs the whole `head_dim` vector, so it cannot be one thread per
// pair the way the standalone rope kernel is; instead each lane owns the pairs
// `{lane, lane + 32, ...}` — for the real FLUX `head_dim = 128` that is two
// `float2`s, i.e. 4 values, held in registers across the reduction so the
// rotation pass needs no second load. Adjacent lanes still hold adjacent
// pairs, so each load is a contiguous 256-byte wave transaction. The
// reduction is a `__shfl_xor` butterfly: no LDS, no barrier, and every lane
// ends with the total.
//
// WAVE32 ONLY. The hardcoded 32s — the `lane + 32*k` pair stride, the
// `blockDim.x >> 5` wave index, the butterfly's 16-down-to-1 offsets, and
// `MAX_PAIRS_PER_LANE` sized as `head_dim / (2 * 32)` — all assume a 32-lane
// wave. That is every RDNA part hipfire targets (gfx10xx through gfx1201);
// a wave64 arch would need the strides and the butterfly reworked, not just
// a recompile.
//
// `out` MUST NOT alias `x` (both are `__restrict__`). The FLUX caller always
// has a distinct destination because the dtype changes.
//
// PARITY. Not bit-identical to `rmsnorm_batched` + `rope_2d_flux_f32_fast`:
// the reduction is 32 lanes x 4 values with a butterfly where `rmsnorm_f32`
// is 128 threads x 1 value with an LDS halving tree, and float addition is
// not associative. The rotation itself is bit-identical. Measured f32->f32
// agreement at real FLUX geometry is well inside 1e-6 relative — see
// `examples/test_qk_rmsnorm_rope_parity.rs`.

// hd <= 2 * 32 * MAX_PAIRS_PER_LANE. FLUX uses 128; the launcher rejects
// anything larger so the per-lane pair buffer stays in registers (a runtime
// bound would index it dynamically and spill it to scratch).
#define MAX_PAIRS_PER_LANE 4

typedef _Float16 __attribute__((ext_vector_type(2))) half2_t;

// Widening pair load / narrowing pair store. `base + 2*p` is always even and
// the allocation is 256-byte aligned, so both the f32 (8-byte) and the f16
// (4-byte) vector accesses are naturally aligned.
__device__ __forceinline__ float2 load_pair(const float* p) {
    return *(const float2*)p;
}
__device__ __forceinline__ float2 load_pair(const _Float16* p) {
    const half2_t h = *(const half2_t*)p;
    return make_float2((float)h.x, (float)h.y);
}
__device__ __forceinline__ void store_pair(float* p, float a, float b) {
    *(float2*)p = make_float2(a, b);
}
__device__ __forceinline__ void store_pair(_Float16* p, float a, float b) {
    half2_t h;
    h.x = (_Float16)a; // round-to-nearest-even, same as cast_f32_to_f16
    h.y = (_Float16)b;
    *(half2_t*)p = h;
}

template <typename InT, typename OutT>
__device__ __forceinline__ void qk_rmsnorm_rope_flux_body(
    const InT* __restrict__ x,
    const float* __restrict__ ids,   // nullable: [n_img][4] positions per image row
    const float* __restrict__ scale,
    OutT* __restrict__ out,
    int n_txt, int n_img, int heads, int hd,
    int grid_w,
    int ax0, int ax1, int ax2, int ax3,
    double theta, float eps) {

    const int lane = (int)(threadIdx.x & 31u);
    const int waves_per_block = (int)(blockDim.x >> 5);
    const long unit = (long)blockIdx.x * waves_per_block + (int)(threadIdx.x >> 5);
    const long units = ((long)n_txt + n_img) * heads;
    if (unit >= units) return;

    // unit == row * heads + h, so the row-major base offset is just unit * hd.
    const long row = unit / heads;
    const long base = unit * (long)hd;
    const int pairs_total = hd >> 1;

    float2 v[MAX_PAIRS_PER_LANE];
    float sum_sq = 0.0f;
#pragma unroll
    for (int k = 0; k < MAX_PAIRS_PER_LANE; ++k) {
        const int p = lane + 32 * k;
        float2 t = make_float2(0.0f, 0.0f);
        if (p < pairs_total) {
            t = load_pair(x + base + 2 * (long)p);
            sum_sq += t.x * t.x;
            sum_sq += t.y * t.y;
        }
        v[k] = t;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_sq += __shfl_xor(sum_sq, off, 32);
    }
    const float rms = rsqrtf(sum_sq / (float)hd + eps);

    // Image rows carry a position (from the id table, or the derived
    // model); text rows carry no position at all and skip the rotation
    // entirely.
    const bool is_img = row >= (long)n_txt;
    float pos0 = 0.0f, pos1 = 0.0f, pos2 = 0.0f, pos3 = 0.0f;
    float log_theta = 0.0f;
    if (is_img) {
        const long t = row - (long)n_txt;
        if (ids != nullptr) {
            pos0 = ids[t * 4 + 0]; pos1 = ids[t * 4 + 1]; pos2 = ids[t * 4 + 2]; pos3 = ids[t * 4 + 3];
        } else {
            pos1 = (float)(t / (long)grid_w);
            pos2 = (float)(t % (long)grid_w);
        }
        log_theta = logf((float)theta);
    }
    const int pairs0 = ax0 >> 1;
    const int pairs1 = ax1 >> 1;
    const int pairs2 = ax2 >> 1;

#pragma unroll
    for (int k = 0; k < MAX_PAIRS_PER_LANE; ++k) {
        const int pg = lane + 32 * k;
        if (pg >= pairs_total) continue;
        const int i = 2 * pg;

        // Same expression order as `rmsnorm_f32`: (x * weight) * rms.
        float a = v[k].x * scale[i] * rms;
        float b = v[k].y * scale[i + 1] * rms;

        if (is_img) {
            float p_axis;
            int p, d_axis;
            if (pg < pairs0) {
                p_axis = pos0;
                p = pg;
                d_axis = ax0;
            } else if (pg < pairs0 + pairs1) {
                p_axis = pos1;
                p = pg - pairs0;
                d_axis = ax1;
            } else if (pg < pairs0 + pairs1 + pairs2) {
                p_axis = pos2;
                p = pg - pairs0 - pairs1;
                d_axis = ax2;
            } else {
                p_axis = pos3;
                p = pg - pairs0 - pairs1 - pairs2;
                d_axis = ax3;
            }
            // Angle 0 is exactly the identity rotation — skipping it is
            // bit-preserving, and covers axis 0 / axis 3 in the derived-
            // position case plus row 0 / col 0.
            if (p_axis != 0.0f) {
                const float inv_freq =
                    expf(-((2.0f * (float)p) / (float)d_axis) * log_theta);
                const float angle = p_axis * inv_freq;
                float c, s;
                sincosf(angle, &s, &c);
                const float ra = a * c - b * s;
                const float rb = a * s + b * c;
                a = ra;
                b = rb;
            }
        }
        store_pair(out + base + i, a, b);
    }
}

#define QK_RMSNORM_ROPE_ENTRY(SUFFIX, IN_T, OUT_T)                             \
    extern "C" __global__ __launch_bounds__(256) void                          \
        qk_rmsnorm_rope_flux_##SUFFIX(                                         \
            const IN_T* __restrict__ x, const float* __restrict__ ids,        \
            const float* __restrict__ scale, OUT_T* __restrict__ out,         \
            int n_txt, int n_img, int heads, int hd, int grid_w, int ax0,     \
            int ax1, int ax2, int ax3, double theta, float eps) {             \
        qk_rmsnorm_rope_flux_body<IN_T, OUT_T>(x, ids, scale, out, n_txt,      \
                                               n_img, heads, hd, grid_w, ax0,  \
                                               ax1, ax2, ax3, theta, eps);     \
    }

QK_RMSNORM_ROPE_ENTRY(f32_f32, float, float)
QK_RMSNORM_ROPE_ENTRY(f32_f16, float, _Float16)
QK_RMSNORM_ROPE_ENTRY(f16_f32, _Float16, float)
QK_RMSNORM_ROPE_ENTRY(f16_f16, _Float16, _Float16)
