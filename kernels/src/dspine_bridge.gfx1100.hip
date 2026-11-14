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

// fp16 (IEEE half) bits -> f32, type-free: the gfx1100 HIP headers on target
// hosts do not expose __ushort_as_half/__half2float reliably (7.x vs 10.x
// header split), so the mq4v2 scale/zero decode goes through the bit layout.
__device__ static inline float dspine_fp16_bits_to_f32(unsigned short h) {
    union { unsigned int u; float f; } cvt;
    const unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    const unsigned int exp10 = (unsigned int)((h >> 10) & 0x1Fu);
    const unsigned int man = (unsigned int)(h & 0x03FFu);
    if (exp10 == 31u) {
        cvt.u = sign | 0x7F800000u | (man << 13);
        return cvt.f;
    }
    if (exp10 == 0u) {
        if (man == 0u) {
            cvt.u = sign;
            return cvt.f;
        }
        // subnormal: man * 2^-24 (exact in f32)
        cvt.u = 0x3B800000u; // 2^-24
        float v = (float)man * cvt.f;
        union { unsigned int u; float f; } sv;
        sv.u = sign;
        return copysignf(v, sv.f);
    }
    cvt.u = sign | ((exp10 - 15u + 127u) << 23) | (man << 13);
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
    int N,
    float rms_scale
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
            s_score[lane] = base[(long long)row * K + lane] + rms[row] * rms_scale * s;
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
extern "C" __global__ void dspine_topk16_gfx1100(
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

// out[r*H+c] = x[(row_offset+r)*H+c] * nw[c]: the pre-write hidden rows
// rescaled by the target final-norm weights, the sender-free `base` term of
// the upstream serial_table score.
extern "C" __global__ void dspine_scale_rows_gfx1100(
    const float* __restrict__ x,
    const float* __restrict__ nw,
    float* __restrict__ out,
    int row_offset,
    int rows,
    int H)
{
    const long long total = (long long)rows * H;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x)
    {
        // Upstream base: raw ⊙ draft-norm @ lm_head (serial_table takes the
        // stash raw and the draft model's norm.weight verbatim — no RMS
        // division; the draft norm's scale is what balances the term).
        // idx is already relative to row_offset: one linear base shift.
        out[idx] = x[(long long)row_offset * H + idx] * nw[(int)(idx % H)];
    }
}

// base_pre[row][lane] = src[row * V + cand[row][lane]]: the serial_table
// score's base term evaluated exactly at the post-write top-k candidate ids
// (upstream rescoring: candidates from the forward's output logits, base
// from the pre-write stash).
extern "C" __global__ void dspine_gather16_gfx1100(
    const float* __restrict__ src,
    const unsigned int* __restrict__ cand,
    float* __restrict__ out,
    int V,
    int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int row = idx / 16;
    out[idx] = src[(long long)row * V + cand[idx]];
}

// ── DSpine refinement lattice (upstream build_dspine_lattice semantics) ──
// Three stages, all independent across (row, predecessor) pairs so the GPU
// stays busy: message (this kernel), delta = write_last @ msg (one F32 GEMM
// over every pair at once, launched from `rdna_compute::dspine_bridge`), and
// the head score (the next kernel). Arithmetic per pair is unchanged from the
// single-kernel per-row walk.
extern "C" __global__ void dspine_lattice_msg_gfx1100(
    const float* __restrict__ query,      // [rows, 3R] packed (q = [0..R])
    const unsigned short* __restrict__ codebook, // [V, C] bf16 raw
    const float* __restrict__ pred_gate,  // [R, C]
    const float* __restrict__ prev_w,     // [R, C]
    const float* __restrict__ gate_bias,  // [R]
    const unsigned int* __restrict__ prev_cand, // [rows, 16] row r-1 candidates (row 0 unused)
    const float* __restrict__ anchor_code, // [C] sqrt(C)-scaled anchor row
    const float* __restrict__ write_last, // [H, R] (dbg slab only)
    float* __restrict__ msg_out,          // [rows*16, R]
    float* __restrict__ dbg_seg,          // [11776] row0 pred0: codes | msg | x | normed | write_last[0..R]
    int R, int C, int rows)
{
    const int block = blockIdx.x;
    const int row = block / 16;
    const int pred = block % 16;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    __shared__ float s_codes[1024];   // C cap
    if (C > 1024) return;

    const float* q_row = query + (long long)row * 3 * R;
    if (row == 0) {
        for (int j = tid; j < C; j += blockDim.x) s_codes[j] = anchor_code[j];
    } else {
        const unsigned int pid = prev_cand[(long long)(row - 1) * 16 + pred];
        const unsigned short* cb = codebook + (long long)pid * C;
        for (int j = tid; j < C; j += blockDim.x) {
            s_codes[j] = dspine_bf16_to_f32_bits(cb[j]) * sqrtf((float)C);
        }
    }
    __syncthreads();

    float* mrow = msg_out + (long long)block * R;
    for (int t = tid; t < R; t += blockDim.x) {
        const float* pg = pred_gate + (long long)t * C;
        const float* pv_ = prev_w + (long long)t * C;
        float pr = 0.0f, pvv = 0.0f;
        for (int j = 0; j < C; ++j) {
            pr += pg[j] * s_codes[j];
            pvv += pv_[j] * s_codes[j];
        }
        const float qt = q_row[t] + gate_bias[t];
        mrow[t] = (1.0f / (1.0f + expf(-(qt + pr)))) * pvv;
    }
    __syncthreads();

    if (dbg_seg && block == 0) {
        for (int j = tid; j < C; j += blockDim.x) dbg_seg[j] = s_codes[j];
        for (int t = tid; t < R; t += blockDim.x) dbg_seg[C + t] = mrow[t];
        for (int t = tid; t < R; t += blockDim.x) dbg_seg[1024 + 512 + 2 * 5120 + t] = write_last[t];
    }
}

// Head score for one (draft row, predecessor): fold the GEMM's delta into the
// write state, rmsnorm, apply the draft head norm, FWHT-rotate, then score
// every candidate against its decoded mq4v2 head row. Grid: [rows * 16].
extern "C" __global__ void dspine_lattice_score_gfx1100(
    const float* __restrict__ raw,        // [rows, H] pre-write hidden rows
    const float* __restrict__ rms,        // [rows]
    const float* __restrict__ delta,      // [rows*16, H] = write_last @ msg (GEMM)
    const float* __restrict__ out_norm,   // [H]
    const float* __restrict__ mq_signs1,  // [256] FWHT sign table (seed 42)
    const float* __restrict__ mq_signs2,  // [256] FWHT sign table (seed 1042)
    const float* __restrict__ lm_row_scratch, // [rows, 16, H] decoded head rows
    float* __restrict__ refined_out,      // [rows, 16, H] normed refined (debug)
    float* __restrict__ scores,           // [rows, 16, 16] f32
    float* __restrict__ dbg_seg,          // [11776] row0 pred0 slab (x at dbg_off, normed at dbg_off+H)
    int H, int rows, int dbg_off)
{
    const int block = blockIdx.x;
    const int row = block / 16;
    const int pred = block % 16;
    if (row >= rows) return;
    const int tid = threadIdx.x;

    __shared__ float s_refined[5120]; // H cap: 27B draft hidden
    __shared__ float s_part[16][16];
    __shared__ float s_red[256];
    if (H > 5120) return;

    const float* raw_row = raw + (long long)row * H;
    const float* d_row = delta + (long long)block * H;
    const float rms_r = rms[row];

    // C. pre-norm write state for this predecessor.
    for (int h = tid; h < H; h += blockDim.x) {
        s_refined[h] = raw_row[h] + rms_r * d_row[h];
    }
    __syncthreads();

    if (dbg_seg && block == 0) {
        for (int h = tid; h < H; h += blockDim.x) dbg_seg[dbg_off + h] = s_refined[h];
    }
    __syncthreads();

    // D. rmsnorm, head norm, FWHT rotation, score column `pred`.
    float ss = 0.0f;
    for (int h = tid; h < H; h += blockDim.x) ss += s_refined[h] * s_refined[h];
    s_red[tid] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 256; ++i) tot += s_red[i];
        s_red[0] = 1.0f / sqrtf(tot / (float)H + 1e-6f);
    }
    __syncthreads();
    const float inv = s_red[0];
    for (int h = tid; h < H; h += blockDim.x) {
        const float nv = s_refined[h] * inv * out_norm[h];
        s_refined[h] = nv;
        refined_out[((long long)row * 16 + pred) * H + h] = nv;
        if (dbg_seg && block == 0) dbg_seg[dbg_off + H + h] = nv;
    }
    __syncthreads();
    // FWHT-forward rotate refined in place (H/256 groups x 256): the mq4v2
    // head rows encode FWHT(W), so the score consumes rotate(refined) —
    // signs1 -> butterfly -> 0.0625*signs2.
    for (int g = 0; g < H / 256; ++g) {
        float* grp = s_refined + g * 256;
        for (int i = tid; i < 256; i += blockDim.x) grp[i] *= mq_signs1[i];
        __syncthreads();
        for (int stride = 1; stride < 256; stride <<= 1) {
            const int i = tid;
            if ((i & stride) == 0) {
                const float a = grp[i];
                const float b = grp[i | stride];
                grp[i] = a + b;
                grp[i | stride] = a - b;
            }
            __syncthreads();
        }
        for (int i = tid; i < 256; i += blockDim.x) grp[i] *= 0.0625f * mq_signs2[i];
        __syncthreads();
    }
    {
        const int c = tid / 16;
        const int lane = tid % 16;
        const float* hr = lm_row_scratch + ((long long)row * 16 + c) * H;
        float acc = 0.0f;
        for (int h = lane; h < H; h += 16) acc += hr[h] * s_refined[h];
        s_part[c][lane] = acc;
    }
    __syncthreads();
    if (tid < 16) {
        float sc = 0.0f;
        for (int l = 0; l < 16; ++l) sc += s_part[tid][l];
        scores[((long long)row * 16 + tid) * 16 + pred] = sc;
    }
}

// Decode one mq4v2 head row (256-element groups of 136 B: 8 B fp16
// scale/zero pair + 128 B nibbles; lo nibble -> even index, hi -> odd) into
// , cooperatively over  lanes.
__device__ static inline void dspine_decode_mq4v2_row(
    const unsigned char* row, int H, int R, float* out, int lane, int nlanes)
{
    const int groups = (H + 255) / 256;
    for (int g = lane; g < groups; g += nlanes) {
        const int off = g * 136;
        const float s0 = dspine_fp16_bits_to_f32((unsigned short)(row[off] | (row[off+1] << 8)));
        const float z0 = dspine_fp16_bits_to_f32((unsigned short)(row[off+2] | (row[off+3] << 8)));
        const float s1 = dspine_fp16_bits_to_f32((unsigned short)(row[off+4] | (row[off+5] << 8)));
        const float z1 = dspine_fp16_bits_to_f32((unsigned short)(row[off+6] | (row[off+7] << 8)));
        const int base = g * 256;
        for (int i = 0; i < 128; ++i) {
            const unsigned char byte = row[off + 8 + i];
            if (base + 2*i < H)     out[base + 2*i] = z0 + (byte & 0xF) * s0;
            if (base + 2*i + 1 < H) out[base + 2*i + 1] = z1 + (byte >> 4) * s1;
        }
    }
}

// Head rows: decode each draft row's 16 candidate mq4v2 head rows once into
// the per-row slab the score stage reads. Grid: [rows]. Block: [256].
extern "C" __global__ void dspine_lattice_head_rows_gfx1100(
    const unsigned int* __restrict__ cand,      // [rows, 16] post-write top-k ids
    const unsigned char* __restrict__ lm_head,  // mq4v2 rows, row_bytes stride
    float* __restrict__ lm_rows,                // [rows, 16, H] decoded head rows
    int H, int R, int rows, int row_bytes)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int c = tid / 16;
    const int lane = tid % 16;
    const unsigned int cid = cand[(long long)row * 16 + c];
    dspine_decode_mq4v2_row(lm_head + (long long)cid * row_bytes, H, R,
                            lm_rows + ((long long)row * 16 + c) * H, lane, 16);
}

// base_pre[row][c] = (raw_row * out_norm) . lm_head[cand[row][c]] — the
// serial_table score's sender-free term, evaluated against the decoded
// (FWHT-rotated) candidate head rows already in `lm_rows`.
extern "C" __global__ void dspine_lattice_base_pre_gfx1100(
    const float* __restrict__ raw,        // [rows, H] pre-write hidden rows
    const float* __restrict__ out_norm,   // [H]
    const float* __restrict__ lm_rows,    // [rows, 16, H] decoded head rows
    const float* __restrict__ mq_signs1,  // [256] FWHT sign table (seed 42)
    const float* __restrict__ mq_signs2,  // [256] FWHT sign table (seed 1042)
    float* __restrict__ base_pre,         // [rows, 16]
    int H, int rows)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    __shared__ float s_x[5120]; // H cap: 27B draft hidden
    __shared__ float s_part[16][16];
    if (H > 5120) return;

    const float* raw_row = raw + (long long)row * H;
    for (int h = tid; h < H; h += blockDim.x) s_x[h] = raw_row[h] * out_norm[h];
    __syncthreads();
    // Same orthogonal FWHT rotation the score stage applies to the refined
    // vector: the head rows encode rotate(W), so dotting them with rotate(x)
    // reproduces W . x exactly (rotate is orthogonal).
    for (int g = 0; g < H / 256; ++g) {
        float* grp = s_x + g * 256;
        for (int i = tid; i < 256; i += blockDim.x) grp[i] *= mq_signs1[i];
        __syncthreads();
        for (int stride = 1; stride < 256; stride <<= 1) {
            const int i = tid;
            if ((i & stride) == 0) {
                const float a = grp[i];
                const float b = grp[i | stride];
                grp[i] = a + b;
                grp[i | stride] = a - b;
            }
            __syncthreads();
        }
        for (int i = tid; i < 256; i += blockDim.x) grp[i] *= 0.0625f * mq_signs2[i];
        __syncthreads();
    }
    const int c = tid / 16;
    const int lane = tid % 16;
    const float* hr = lm_rows + ((long long)row * 16 + c) * H;
    float acc = 0.0f;
    for (int h = lane; h < H; h += 16) acc += hr[h] * s_x[h];
    s_part[c][lane] = acc;
    __syncthreads();
    if (tid < 16) {
        float sc = 0.0f;
        for (int l = 0; l < 16; ++l) sc += s_part[tid][l];
        base_pre[(long long)row * 16 + tid] = sc;
    }
}
