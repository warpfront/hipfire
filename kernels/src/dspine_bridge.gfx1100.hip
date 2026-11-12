// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

// DSpine arch-24 code-bridge stages, gfx1100-only. Five symbols backing
// `rdna_compute::dspine_bridge`; operation order mirrors the CPU oracle in
// `hipfire_runtime::dspine::cpu_oracle` exactly (F32 accumulation everywhere,
// BF16 only at the tensor-storage boundaries):
//
// - dspine_prepare_read_gfx1100: row RMS stats + F32-normalized rows. The
//   kernel replaces prepare_read's Triton pair (rms stats, x/rms); the delta
//   write-back lives in dspine_finish so the shared trunk loop needs no
//   extra residual copy.
// - dspine_normalize_codes_gfx1100: anchor rows take the codebook identity
//   path (codebook[anchor] * sqrt(C)); non-anchor rows L2-normalize their
//   projection to sqrt(C) scale. Degenerate zero projections normalize to
//   zeros (never NaN).
// - dspine_feature_message_gfx1100: gate = sigmoid(query + sender_prev +
//   bias), message = gate * value_prev, with block-first rows forced to 0.
//   Rows are block-major, so the predecessor is row-1 in memory order.
// - dspine_finish_gfx1100: x += rms * delta on non-block-first rows (block
//   first rows pass through), in place on the trunk's f32 block rows. The
//   next layer's existing pre-stage attention norm applies to the bridged
//   rows — same math as upstream fusing the next-input RMSNorm into the
//   write, one norm total.
//
// The packed GEMM (codes @ concat(gate, predecessor_gate, prev)), the
// projection GEMM (normed @ read_weight[i]) and the delta GEMM (message @
// feature_write_weights[i]) ride the existing F32 draft GEMM dispatch — the
// bridge weights are lifted BF16→F32 at load (exact), and the per-layer
// matrices are tiny ([16,5120]@[5120,1024] and friends).

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// ── prepare_read: rms stats + normalized rows ────────────────────────────
// Grid: [m, 1, 1]. Block: [256]. x: [m, H] f32 (trunk block rows);
// rms: [m] f32; normed: [m, H] f32 (x / rms).
extern "C" __launch_bounds__(256, 4)
__global__ void dspine_prepare_read_gfx1100(
    const float* __restrict__ x,
    float* __restrict__ rms,
    float* __restrict__ normed,
    int H,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* x_row = x + (long long)row * H;
    float* n_row = normed + (long long)row * H;

    // F32 accumulate sum of squares over H (strided loop).
    float acc = 0.0f;
    for (int i = tid; i < H; i += 256) {
        acc += x_row[i] * x_row[i];
    }
    // Block tree reduction (LDS).
    __shared__ float red[256];
    red[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    const float denom = sqrtf(red[0] / (float)H + eps);
    if (tid == 0) {
        rms[row] = denom;
    }
    const float inv = 1.0f / denom;
    for (int i = tid; i < H; i += 256) {
        n_row[i] = x_row[i] * inv;
    }
}

// ── normalize_codes ──────────────────────────────────────────────────────
// Grid: [m, 1, 1]. Block: [256]. projected: [m, C] f32; codebook: [V, C]
// bf16; anchor_ids: [m] u32; codes: [m, C] f32. Anchor rows (block row 0)
// read codebook[anchor_id] scaled by sqrt(C); all other rows L2-normalize
// their projection to sqrt(C) scale (zeros stay zeros).
extern "C" __launch_bounds__(256, 4)
__global__ void dspine_normalize_codes_gfx1100(
    const float* __restrict__ projected,
    const __hip_bfloat16* __restrict__ codebook,
    const unsigned int* __restrict__ anchor_ids,
    float* __restrict__ codes,
    int C,
    int T
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_row = row % T;
    float* out_row = codes + (long long)row * C;
    const float scale = sqrtf((float)C);

    if (block_row == 0) {
        // Identity path: codebook[anchor_id] * sqrt(C).
        const unsigned int anchor = anchor_ids[row / T];
        const __hip_bfloat16* code = codebook + (long long)anchor * C;
        for (int i = tid; i < C; i += 256) {
            out_row[i] = __bfloat162float(code[i]) * scale;
        }
        return;
    }
    const float* p_row = projected + (long long)row * C;
    float acc = 0.0f;
    for (int i = tid; i < C; i += 256) {
        acc += p_row[i] * p_row[i];
    }
    __shared__ float red[256];
    red[tid] = acc;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        __syncthreads();
    }
    const float norm = sqrtf(red[0]);
    if (norm > 0.0f) {
        const float k = scale / norm;
        for (int i = tid; i < C; i += 256) {
            out_row[i] = p_row[i] * k;
        }
    } else {
        for (int i = tid; i < C; i += 256) {
            out_row[i] = 0.0f;
        }
    }
}

// ── feature_message ──────────────────────────────────────────────────────
// Grid: [m, 1, 1]. Block: [256]. packed: [m, 3R] f32 with segments
// query=packed[..R], sender=packed[R..2R], value=packed[2R..3R] per row;
// gate_bias: [R] f32; message: [m, R] f32. Block-first rows (row % T == 0)
// write 0; every other row reads its predecessor row's sender/value.
extern "C" __launch_bounds__(256, 4)
__global__ void dspine_feature_message_gfx1100(
    const float* __restrict__ packed,
    const float* __restrict__ gate_bias,
    float* __restrict__ message,
    int R,
    int T
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_row = row % T;
    float* out_row = message + (long long)row * R;

    if (block_row == 0) {
        for (int i = tid; i < R; i += 256) {
            out_row[i] = 0.0f;
        }
        return;
    }
    const float* cur = packed + (long long)row * 3 * R;          // query segment
    const float* prev = packed + (long long)(row - 1) * 3 * R;   // sender/value segments
    const float* sender_prev = prev + R;
    const float* value_prev = prev + 2 * R;
    for (int i = tid; i < R; i += 256) {
        const float gate = 1.0f / (1.0f + expf(-(cur[i] + sender_prev[i] + gate_bias[i])));
        out_row[i] = gate * value_prev[i];
    }
}

// ── finish: in-place delta add on non-first rows ─────────────────────────
// Grid: [ceil(H*m / 256), 1, 1]. Block: [256]. x: [m, H] f32 (in place);
// rms: [m] f32; delta: [m, H] f32. x[row] += rms[row] * delta[row] for
// non-block-first rows.
extern "C" __launch_bounds__(256, 4)
__global__ void dspine_finish_gfx1100(
    float* __restrict__ x,
    const float* __restrict__ rms,
    const float* __restrict__ delta,
    long long H,
    int T
) {
    const long long idx = (long long)blockIdx.x * 256 + threadIdx.x;
    const long long row = idx / H;
    const int col = (int)(idx - row * H);
    if (col >= H) {
        return;
    }
    if (row % T == 0) {
        return;
    }
    x[idx] += rms[row] * delta[idx];
}
