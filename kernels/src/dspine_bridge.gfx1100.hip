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


// bf16 storage is plain u16 bit patterns: every ROCm release names the
// bf16 type differently (7.x hip_bfloat16 vs 10.x __hip_bfloat16) and the
// system header ordering differs per host, so the kernels take u16 and
// convert through the bit layout — no bf16 type anywhere.
__device__ static inline float dspine_bf16_to_f32_bits(unsigned short bits) {
    union { unsigned int u; float f; } cvt;
    cvt.u = ((unsigned int)bits) << 16;
    return cvt.f;
}

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
    const unsigned short* __restrict__ codebook,
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
        const unsigned short* code = codebook + (long long)anchor * C;
        for (int i = tid; i < C; i += 256) {
            out_row[i] = dspine_bf16_to_f32_bits(code[i]) * scale;
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

// ── serial refinement: fused score + greedy chain ────────────────────────
// Grid: [n/T blocks, 1, 1]. Block: [256] (16 candidates × 16 sub-lanes).
// Rows within a block are walked serially (each row sender = previous
// row argmax). base/cand are the per-row head-logits top-K [n, K];
// tsv_gate/tsv_val/tg are the load-time sender tables [V, R] f32;
// query [n, R] f32 is the last-write stash query segment.
// draft_ids [n] u32: the selected token id per row.
extern "C" __launch_bounds__(256, 2)
__global__ void dspine_serial_chain_gfx1100(
    const float* __restrict__ base,
    const unsigned int* __restrict__ cand,
    const float* __restrict__ rms,
    const float* __restrict__ query,
    const float* __restrict__ tsv_gate,
    const float* __restrict__ tsv_val,
    const float* __restrict__ tg,
    const float* __restrict__ gate_bias,
    unsigned int anchor_sid,
    unsigned int* __restrict__ draft_ids,
    int K,
    int R,
    int T,
    int packed_stride,
    int N
) {
    const int block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid / 16;      // candidate lane 0..15 (K <= 16)
    const int sub = tid % 16;       // sub-lane within the candidate

    __shared__ float s_sender[512];
    __shared__ float s_part[16][16];
    __shared__ float s_score[16];
    __shared__ unsigned int s_sid;

    if (tid == 0) {
        s_sid = anchor_sid;
    }
    __syncthreads();

    const int cand_sub_elems = R / 16;
    for (int r = 0; r < T; ++r) {
        const int row = block_idx * T + r;
        if (row >= N) break;  // partial tail block: all threads exit together
        // 1. sender contribution for this row: sigmoid(query + TSV_gate[sid]
        //    + bias) * TSV_val[sid], R-wide (256 threads × 2 elements).
        const unsigned int sid = s_sid;
        if (tid < 256) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int j = tid + half * 256;
                if (j < R) {
                    const long long s_off = (long long)sid * R;
                    const float gate = 1.0f /
                        (1.0f + expf(-(query[(long long)row * packed_stride + j]
                                       + tsv_gate[s_off + j]
                                       + gate_bias[j])));
                    s_sender[j] = gate * tsv_val[s_off + j];
                }
            }
        }
        __syncthreads();
        // 2. candidate score: sender · TG[cand] over R (16 sub-lanes each).
        const unsigned int cid = cand[(long long)row * K + lane];
        const long long tg_off = (long long)cid * R;
        float part = 0.0f;
        for (int j = sub * cand_sub_elems; j < (sub + 1) * cand_sub_elems; ++j) {
            part += s_sender[j] * tg[tg_off + j];
        }
        s_part[lane][sub] = part;
        __syncthreads();
        // 3. reduce each candidate lane to a score (one thread per lane).
        if (sub == 0 && lane < K) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                s += s_part[lane][i];
            }
            s_score[lane] = base[(long long)row * K + lane] + rms[row] * s;
        }
        __syncthreads();
        // 4. argmax over K candidates; advance the chain.
        if (tid == 0) {
            int best = 0;
            float best_s = s_score[0];
            for (int c = 1; c < K; ++c) {
                if (s_score[c] > best_s) {
                    best_s = s_score[c];
                    best = c;
                }
            }
            const unsigned int picked = cand[(long long)row * K + best];
            s_sid = picked;
            draft_ids[row] = picked;
        }
        __syncthreads();
    }
}

// Per-row top-16 over [rows, V] logits with the gathered base scores: the
// serial refinement's candidate shortlist. Phase 1: each of 256 threads
// scans a V/256 slice keeping a local 16-slot insertion list; phase 2:
// thread 0 merges the 4096 survivors. First-seen (lowest) index wins ties,
// matching argmax_f32_batched's first-max semantics.
__global__ void dspine_topk16_gfx1100(
    const float* __restrict__ logits,
    unsigned int* __restrict__ cand,
    float* __restrict__ base_out,
    int V)
{
    const int row = blockIdx.x;
    const float* l = logits + (long long)row * V;
    __shared__ float s_vals[256 * 16];
    __shared__ unsigned int s_ids[256 * 16];

    const int tid = threadIdx.x;
    const int slice = (V + 255) / 256;
    float vals[16];
    unsigned int ids[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { vals[i] = -3.0e38f; ids[i] = 0u; }
    const int lo = tid * slice;
    const int hi = min(lo + slice, V);
    for (int i = lo; i < hi; ++i) {
        const float v = l[i];
        if (v > vals[15]) {
            int p = 15;
            #pragma unroll
            for (int q = 0; q < 15; ++q) {
                if (!(vals[p - 1] < v)) break;
                vals[p] = vals[p - 1];
                ids[p] = ids[p - 1];
                --p;
            }
            vals[p] = v;
            ids[p] = (unsigned int)i;
        }
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        s_vals[tid * 16 + i] = vals[i];
        s_ids[tid * 16 + i] = ids[i];
    }
    __syncthreads();

    if (tid == 0) {
        float m_vals[16];
        unsigned int m_ids[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) { m_vals[i] = -3.0e38f; m_ids[i] = 0u; }
        for (int i = 0; i < 256 * 16; ++i) {
            const float v = s_vals[i];
            if (v > m_vals[15]) {
                int p = 15;
                #pragma unroll
                for (int q = 0; q < 15; ++q) {
                    if (!(m_vals[p - 1] < v)) break;
                    m_vals[p] = m_vals[p - 1];
                    m_ids[p] = m_ids[p - 1];
                    --p;
                }
                m_vals[p] = v;
                m_ids[p] = s_ids[i];
            }
        }
        for (int i = 0; i < 16; ++i) {
            cand[(long long)row * 16 + i] = m_ids[i];
            base_out[(long long)row * 16 + i] = m_vals[i];
        }
    }
}
