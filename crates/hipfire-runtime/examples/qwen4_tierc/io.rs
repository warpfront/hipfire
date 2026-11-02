// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C cases, family `io`: embedding, the PLE device path (gather widen,
//! gate, norm, depthwise conv + stream add) and the copy / convert / round-trip
//! kernels the gfx942 (F32 HC streams) route launches between stages.
//!
//! Every case calls the same public launcher the Qwen4 route calls
//! (`crates/hipfire-arch-qwen4/src/gpu_forward.rs`,
//! `crates/hipfire-dispatch/src/pipeline/layer_ops.rs`), so on gfx942 it
//! exercises the variant gfx942 takes: `qwen4_ple_fuse_enabled()` is exact
//! gfx1151 / gfx1201 and `qwen4_bf16_streams` needs `has_gfx11_plus_simt()`, so
//! the unfused PLE chain on F32 streams is what runs here.
//! Census of record: `local://census-io.md`.

use super::common::*;
use super::Q4Case;
use rdna_compute::grouped_ops::{GroupedDepthwiseConv, GroupedGate, GroupedNorm};
use rdna_compute::{DType, Gpu, GpuTensor};

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_embedding_bf16_batched_h2560",
        timeout_s: 240,
        run: embedding_bf16_batched,
    },
    Q4Case {
        name: "qwen4_embedding_q8_batched_h2560",
        timeout_s: 240,
        run: embedding_q8_batched,
    },
    Q4Case {
        name: "qwen4_grouped_gather_convert_bf16_ple_16x160",
        timeout_s: 60,
        run: grouped_gather_convert_bf16,
    },
    Q4Case {
        name: "qwen4_grouped_gate_bf16_ple_4x2560",
        timeout_s: 120,
        run: grouped_gate_bf16,
    },
    Q4Case {
        name: "qwen4_grouped_norm_bf16_ple_4x2560",
        timeout_s: 60,
        run: grouped_norm_bf16,
    },
    Q4Case {
        name: "qwen4_grouped_depthwise_conv_silu_add_bf16_ple_k4d3",
        timeout_s: 60,
        run: grouped_depthwise_conv_silu_add_bf16,
    },
    Q4Case {
        name: "qwen4_copy_f32_buffer_hc_wide",
        timeout_s: 60,
        run: copy_f32_buffer,
    },
    Q4Case {
        name: "qwen4_add_f32_inplace_ple_streams",
        timeout_s: 60,
        run: add_f32_inplace,
    },
    Q4Case {
        name: "qwen4_convert_f32_to_bf16",
        timeout_s: 60,
        run: convert_f32_to_bf16,
    },
    Q4Case {
        name: "qwen4_bf16_round_trip_f32",
        timeout_s: 60,
        run: bf16_round_trip_f32,
    },
    Q4Case {
        name: "qwen4_zero_f32",
        timeout_s: 60,
        run: zero_f32,
    },
];

// ── Qwen4 PLE geometry ────────────────────────────────────────────────────────
// 16 heads × 160 = 2560 = hidden (`config.rs:2774` ple_row_width 160; embed_dim
// == hidden for Flash-Next), HC streams 4, conv kernel 4 (`config.rs:1118`),
// dilation 3 / history rows 9 (`config.rs:2775-2776`).
const PLE_HEADS: usize = 16;
const PLE_ROW_WIDTH: usize = 160;
const PLE_GROUPS: usize = HC_STREAMS;
const PLE_CHANNELS: usize = HC_WIDE; // 10,240
const PLE_KERNEL: usize = 4;
const PLE_DILATION: usize = 3;
const PLE_HISTORY: usize = (PLE_KERNEL - 1) * PLE_DILATION; // 9
/// `gpu_forward.rs:68` `EPSILON`.
const PLE_EPS: f32 = 1.0e-6;
/// `grouped_depthwise_conv_silu_add_bf16` token chunk is `max(history, 32)`
/// (`grouped_ops.rs` launcher); 130 rows = 5 chunks, 33 rows = 2 chunks.
const CONV_ROW_COUNTS: [usize; 4] = [1, 7, 33, 130];

// ── Deterministic bulk fill (cheap enough for a 248,320-row table) ───────────
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `rows × dim` BF16 little-endian bytes, every value finite (exponent forced
/// off all-ones), wide exponent range, including zeros and subnormals.
fn fill_bf16_table(rows: usize, dim: usize, seed: u64) -> Vec<u8> {
    let mut bytes = vec![0u8; rows * dim * 2];
    let mut state = seed;
    for chunk in bytes.chunks_exact_mut(8) {
        let z = mix(&mut state);
        for k in 0..4 {
            let mut b = (z >> (16 * k)) as u16;
            if b & 0x7F80 == 0x7F80 {
                b ^= 0x4000;
            }
            chunk[2 * k..2 * k + 2].copy_from_slice(&b.to_le_bytes());
        }
    }
    bytes
}

/// Q8_0 table: per 32-element block a little-endian F16 scale (finite normal,
/// exponent field 5..=20) followed by 32 signed bytes (34 bytes per block).
fn fill_q8_table(rows: usize, dim: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = dim / 32;
    let mut bytes = vec![0u8; rows * blocks_per_row * 34];
    let mut state = seed;
    for block in bytes.chunks_exact_mut(34) {
        let z = mix(&mut state);
        let exp = 5 + (z % 16) as u16;
        let mant = ((z >> 8) & 0x3FF) as u16;
        let sign = ((z >> 20) & 1) as u16;
        let half = (sign << 15) | (exp << 10) | mant;
        block[..2].copy_from_slice(&half.to_le_bytes());
        for quad in block[2..].chunks_exact_mut(8) {
            quad.copy_from_slice(&mix(&mut state).to_le_bytes());
        }
    }
    bytes
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let v = mant as f32 * (1.0f32 / 16_777_216.0); // 2^-24
        return if sign == 1 { -v } else { v };
    }
    // Finite normals only (exp 31 never generated).
    f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}

/// Token-id tensor for the embedding launchers. `embedding_lookup_bf16_batched`
/// requires `dtype == DType::Raw` and `numel() >= n`; Raw is 1 byte/element, so
/// the shape is `[4n]` (numel and byte_size both cover the `n` i32 payload).
fn up_ids(gpu: &mut Gpu, ids: &[u32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let t = gpu.upload_raw(&bytes, &[bytes.len()]).map_err(e)?;
    debug_assert_eq!(t.dtype, DType::Raw);
    Ok(t)
}

/// Token ids for `n` rows: row 0 / row 1 are `0` / `VOCAB-1` (n == 1 uses
/// `VOCAB-1`), the rest are uniform over the whole vocabulary.
fn embedding_ids(n: usize, rng: &mut Rng) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..n).map(|_| rng.below(VOCAB) as u32).collect();
    if n == 1 {
        ids[0] = (VOCAB - 1) as u32;
    } else {
        ids[0] = 0;
        ids[1] = (VOCAB - 1) as u32;
    }
    ids
}

fn id_sets(rng: &mut Rng) -> Vec<Vec<u32>> {
    let mut sets: Vec<Vec<u32>> = ROW_COUNTS.iter().map(|&n| embedding_ids(n, rng)).collect();
    sets.push(vec![0]); // single-row lookup of id 0 (decode)
    sets
}

// ── 1. embedding_bf16_batched (`embedding_bf16_batched.hip:13`) ──────────────
// Route: `execute_embedding` (`layer_ops.rs:3831`) → `Gpu::embedding_lookup_bf16_batched`
// (`embedding.rs:329`), grid [n], block 256. Pure widen: exact.
fn embedding_bf16_batched(gpu: &mut Gpu) -> Result<String, String> {
    let table_bytes = fill_bf16_table(VOCAB, HIDDEN, 0xE0B0_0001);
    let mut table = gpu.upload_raw(&table_bytes, &[VOCAB, HIDDEN]).map_err(e)?;
    table.dtype = DType::BF16; // byte_size = 2 * numel = payload
    let mut rng = Rng::new(0xE001);
    let mut results = Vec::new();
    for ids in id_sets(&mut rng) {
        let n = ids.len();
        let ids_t = up_ids(gpu, &ids)?;
        let out = sentinel_f32(gpu, n * HIDDEN)?;
        gpu.embedding_lookup_bf16_batched(&table, &out, &ids_t, n, HIDDEN)
            .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &out)?;
        let mut want = Vec::with_capacity(n * HIDDEN);
        for &id in &ids {
            let row = &table_bytes[id as usize * HIDDEN * 2..(id as usize + 1) * HIDDEN * 2];
            want.extend(
                row.chunks_exact(2)
                    .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            );
        }
        results.push(exact_f32(
            &format!(
                "embedding_bf16 n={n} ids[..2]={:?}",
                &ids[..ids.len().min(2)]
            ),
            &got,
            &want,
        ));
        free_all(gpu, vec![ids_t, out])?;
    }
    free_all(gpu, vec![table])?;
    all(results)
}

// ── 2. embedding_q8_batched (`embedding_q8_batched.hip:10`) ──────────────────
// Today's recipe's embedding (Q8_0); launched through the same
// `execute_embedding` arm for a Q8_0 table. out = f16_scale * i8 (one F32 mul).
fn embedding_q8_batched(gpu: &mut Gpu) -> Result<String, String> {
    let row_bytes = HIDDEN / 32 * 34;
    let table_bytes = fill_q8_table(VOCAB, HIDDEN, 0xE0B0_0008);
    // Raw is 1 byte/element: shape = byte count.
    let table = gpu
        .upload_raw(&table_bytes, &[VOCAB * row_bytes])
        .map_err(e)?;
    let mut rng = Rng::new(0xE002);
    let mut results = Vec::new();
    for ids in id_sets(&mut rng) {
        let n = ids.len();
        let ids_t = up_ids(gpu, &ids)?;
        let out = sentinel_f32(gpu, n * HIDDEN)?;
        gpu.embedding_lookup_q8_batched(&table, &out, &ids_t, n, HIDDEN)
            .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &out)?;
        let mut want = Vec::with_capacity(n * HIDDEN);
        for &id in &ids {
            let row = &table_bytes[id as usize * row_bytes..(id as usize + 1) * row_bytes];
            for block in row.chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                want.extend(block[2..].iter().map(|&q| d * (q as i8) as f32));
            }
        }
        results.push(exact_f32(&format!("embedding_q8 n={n}"), &got, &want));
        free_all(gpu, vec![ids_t, out])?;
    }
    free_all(gpu, vec![table])?;
    all(results)
}

// ── 3. grouped_gather_convert_bf16 (`grouped_ops.hip:21`) ────────────────────
// PLE row gather: host-staged BF16 [tokens, 16, 160] widened to F32
// (`gpu_forward.rs:6293`, lanes `gpu_forward_lanes.rs:1576`). Exact.
// (`grouped_gather_convert_bf16_f16` is the fused-F16 sibling: unreachable on
// gfx942, `qwen4_ple_rows_f16` needs `qwen4_ple_fuse_enabled()` = gfx1151/1201.)
fn grouped_gather_convert_bf16(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE003);
    let mut results = Vec::new();
    for &n in &ROW_COUNTS {
        let elements = n * PLE_HEADS * PLE_ROW_WIDTH;
        let vals = rng.vec_bf16_exact(elements, -4.0, 4.0);
        let staged = up_bf16(gpu, &vals)?;
        let out = sentinel_f32(gpu, elements)?;
        gpu.grouped_gather_convert_bf16(&staged, &out, n, PLE_HEADS, PLE_ROW_WIDTH)
            .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &out)?;
        results.push(exact_f32(&format!("gather_bf16 rows={n}"), &got, &vals));
        free_all(gpu, vec![staged, out])?;
    }
    all(results)
}

// ── 4. grouped_gate_bf16 (`grouped_ops.hip:197`) ─────────────────────────────
// Unfused PLE gate (`layer_ops.rs:3820`): grid [4 branches, rows], block 256,
// LDS 8*hidden. Thread 0 runs the three reductions serially in index order; the
// CPU twin mirrors that order. rsqrtf / expf / sqrtf are device libm (and the
// compiler may contract mul+add) so the bound is the frozen F32_ACCUM.
fn sigmoid(v: f32) -> f32 {
    if v >= 0.0 {
        1.0 / (1.0 + (-v).exp())
    } else {
        let z = v.exp();
        z / (1.0 + z)
    }
}

#[allow(clippy::too_many_arguments)]
fn gate_cpu(
    key: &[f32],
    query: &[f32],
    value: &[f32],
    norm_key: &[f32],
    norm_query: &[f32],
    tokens: usize,
    groups: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let channels = groups * hidden;
    let eps = eps.max(1.0e-12);
    let mut out = vec![0.0f32; tokens * channels];
    for t in 0..tokens {
        for b in 0..groups {
            let base = t * channels + b * hidden;
            let (mut ks, mut qs) = (0.0f32, 0.0f32);
            for i in 0..hidden {
                ks += key[base + i] * key[base + i];
                qs += query[base + i] * query[base + i];
            }
            let kinv = 1.0 / (ks / hidden as f32 + eps).sqrt();
            let qinv = 1.0 / (qs / hidden as f32 + eps).sqrt();
            let mut dot = 0.0f32;
            for i in 0..hidden {
                let w = b * hidden + i;
                let kv = key[base + i] * kinv * (1.0 + norm_key[w]);
                let qv = query[base + i] * qinv * (1.0 + norm_query[w]);
                dot += kv * qv;
            }
            dot /= (hidden as f32).sqrt();
            let sign = if dot >= 0.0 { 1.0f32 } else { -1.0 };
            let gate = sigmoid(sign * dot.abs().max(1.0e-6).sqrt());
            for i in 0..hidden {
                out[base + i] = gate * value[t * hidden + i];
            }
        }
    }
    out
}

fn grouped_gate_bf16(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE004);
    let norm_key = rng.vec_bf16_exact(PLE_CHANNELS, -0.5, 0.5);
    let norm_query = rng.vec_bf16_exact(PLE_CHANNELS, -0.5, 0.5);
    let nk_t = up_bf16(gpu, &norm_key)?;
    let nq_t = up_bf16(gpu, &norm_query)?;
    let mut results = Vec::new();
    for &n in &ROW_COUNTS {
        let key = rng.vec(n * PLE_CHANNELS, -2.0, 2.0);
        let query = rng.vec(n * PLE_CHANNELS, -2.0, 2.0);
        let value = rng.vec(n * HIDDEN, -1.0, 1.0);
        let (k_t, q_t, v_t) = (
            up_f32(gpu, &key)?,
            up_f32(gpu, &query)?,
            up_f32(gpu, &value)?,
        );
        let out = sentinel_f32(gpu, n * PLE_CHANNELS)?;
        rdna_compute::grouped_ops::grouped_gate_bf16(
            gpu,
            &GroupedGate {
                key: &k_t,
                query: &q_t,
                value: &v_t,
                norm_key: &nk_t,
                norm_query: &nq_t,
                gated: &out,
                rows: n,
                groups: PLE_GROUPS,
                group_size: HIDDEN,
                epsilon: PLE_EPS,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &out)?;
        let want = gate_cpu(
            &key,
            &query,
            &value,
            &norm_key,
            &norm_query,
            n,
            PLE_GROUPS,
            HIDDEN,
            PLE_EPS,
        );
        results.push(close_f32(&format!("gate rows={n}"), &got, &want, F32_ACCUM));
        free_all(gpu, vec![k_t, q_t, v_t, out])?;
    }
    free_all(gpu, vec![nk_t, nq_t])?;
    all(results)
}

// ── 5. grouped_norm_bf16 (`grouped_ops.hip:262`) ─────────────────────────────
// Zero-centered `(1 + w)` RMSNorm over each of the 4 stream groups of 2560
// (`layer_ops.rs:3835`): grid [4, rows], block 256, `partial[256]` LDS tree
// (stride 128..1, `__syncthreads`, no shuffles). The twin mirrors the lane
// strided partial sums and the tree; rsqrtf / FMA contraction keep the bound at
// F32_ACCUM.
fn grouped_norm_cpu(
    input: &[f32],
    weight: &[f32],
    tokens: usize,
    groups: usize,
    group_size: usize,
    eps: f32,
) -> Vec<f32> {
    let eps = eps.max(1.0e-12);
    let mut out = vec![0.0f32; input.len()];
    for t in 0..tokens {
        for g in 0..groups {
            let base = t * groups * group_size + g * group_size;
            let mut partial = [0.0f32; 256];
            for (lane, slot) in partial.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                let mut i = lane;
                while i < group_size {
                    sum += input[base + i] * input[base + i];
                    i += 256;
                }
                *slot = sum;
            }
            let mut stride = 128;
            while stride > 0 {
                for lane in 0..stride {
                    partial[lane] += partial[lane + stride];
                }
                stride >>= 1;
            }
            let inv = 1.0 / (partial[0] / group_size as f32 + eps).sqrt();
            for i in 0..group_size {
                out[base + i] = input[base + i] * inv * (1.0 + weight[g * group_size + i]);
            }
        }
    }
    out
}

fn grouped_norm_bf16(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE005);
    let weight = rng.vec_bf16_exact(PLE_CHANNELS, -0.5, 0.5);
    let w_t = up_bf16(gpu, &weight)?;
    let mut results = Vec::new();
    for &n in &ROW_COUNTS {
        let input = rng.vec(n * PLE_CHANNELS, -3.0, 3.0);
        let in_t = up_f32(gpu, &input)?;
        let out = sentinel_f32(gpu, n * PLE_CHANNELS)?;
        rdna_compute::grouped_ops::grouped_norm_bf16(
            gpu,
            &GroupedNorm {
                input: &in_t,
                weight: &w_t,
                output: &out,
                tokens: n,
                groups: PLE_GROUPS,
                group_size: HIDDEN,
                epsilon: PLE_EPS,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &out)?;
        let want = grouped_norm_cpu(&input, &weight, n, PLE_GROUPS, HIDDEN, PLE_EPS);
        results.push(close_f32(&format!("norm rows={n}"), &got, &want, F32_ACCUM));
        free_all(gpu, vec![in_t, out])?;
    }
    free_all(gpu, vec![w_t])?;
    all(results)
}

// ── 6. grouped_depthwise_conv_silu_add_bf16 (`grouped_ops.hip:295`) ──────────
// Layer-2 PLE conv (`layer_ops.rs:3848`): channels 10,240, kernel 4, dilation 3,
// history 9 rows, F32 state in/out. Grid [channels/256, ceil(rows/32)]: only
// token-chunk 0 reads and rewrites `state`. out = gated + silu(Σ tap·w);
// the state update is a pure copy, so it is compared exactly.
#[allow(clippy::too_many_arguments)]
fn conv_cpu(
    gated: &[f32],
    normed: &[f32],
    weight: &[f32],
    state: &[f32],
    tokens: usize,
    channels: usize,
    kernel: usize,
    dilation: usize,
) -> (Vec<f32>, Vec<f32>) {
    let history = (kernel - 1) * dilation;
    let mut out = vec![0.0f32; tokens * channels];
    for t in 0..tokens {
        for c in 0..channels {
            let mut total = 0.0f32;
            for tap in 0..kernel {
                let source = t + tap * dilation;
                let value = if source < history {
                    state[source * channels + c]
                } else {
                    let index = source - history;
                    if index < tokens {
                        normed[index * channels + c]
                    } else {
                        0.0
                    }
                };
                total += value * weight[c * kernel + tap];
            }
            out[t * channels + c] = gated[t * channels + c] + total * sigmoid(total);
        }
    }
    let mut new_state = vec![0.0f32; history * channels];
    for row in 0..history {
        let source = tokens + row;
        for c in 0..channels {
            new_state[row * channels + c] = if source < history {
                state[source * channels + c]
            } else {
                normed[(source - history) * channels + c]
            };
        }
    }
    (out, new_state)
}

fn grouped_depthwise_conv_silu_add_bf16(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE006);
    let weight = rng.vec_bf16_exact(PLE_CHANNELS * PLE_KERNEL, -1.0, 1.0);
    let w_t = up_bf16(gpu, &weight)?;
    let mut results = Vec::new();
    for &n in &CONV_ROW_COUNTS {
        let gated = rng.vec(n * PLE_CHANNELS, -1.0, 1.0);
        let normed = rng.vec(n * PLE_CHANNELS, -1.0, 1.0);
        let state = rng.vec(PLE_HISTORY * PLE_CHANNELS, -1.0, 1.0);
        let (g_t, n_t, s_t) = (
            up_f32(gpu, &gated)?,
            up_f32(gpu, &normed)?,
            up_f32(gpu, &state)?,
        );
        let out = sentinel_f32(gpu, n * PLE_CHANNELS)?;
        rdna_compute::grouped_ops::grouped_depthwise_conv_silu_add_bf16(
            gpu,
            &GroupedDepthwiseConv {
                gated: &g_t,
                normed: &n_t,
                conv_weight: &w_t,
                state: &s_t,
                output: &out,
                tokens: n,
                channels: PLE_CHANNELS,
                kernel_size: PLE_KERNEL,
                dilation: PLE_DILATION,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got_out = down_f32(gpu, &out)?;
        let got_state = down_f32(gpu, &s_t)?;
        let (want_out, want_state) = conv_cpu(
            &gated,
            &normed,
            &weight,
            &state,
            n,
            PLE_CHANNELS,
            PLE_KERNEL,
            PLE_DILATION,
        );
        results.push(close_f32(
            &format!("conv out rows={n}"),
            &got_out,
            &want_out,
            F32_ACCUM,
        ));
        results.push(exact_f32(
            &format!("conv state rows={n}"),
            &got_state,
            &want_state,
        ));
        free_all(gpu, vec![g_t, n_t, s_t, out])?;
    }
    free_all(gpu, vec![w_t])?;
    all(results)
}

// ── Copy / convert / round-trip kernels ─────────────────────────────────────
/// Sizes in elements: tail (not a multiple of 256) and the HC-wide row counts.
fn wide_sizes() -> Vec<usize> {
    let mut sizes = vec![1usize, 257];
    sizes.extend(ROW_COUNTS.iter().map(|&r| r * HC_WIDE));
    sizes
}

/// Values that stress RNE: zeros, a tie rounding to even and to odd,
/// subnormal, near-FLT_MAX (rounds to inf), ±inf, canonical NaN.
const SPECIALS: [u32; 10] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3F80_8000, // tie, even neighbour stays
    0x3F81_8000, // tie, rounds up to even
    0x0000_7FFF, // subnormal
    0x7F7F_FFFF, // FLT_MAX -> inf
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x7FC0_0000, // canonical NaN
    0xBF80_C001, // above tie, negative
];

fn stress_values(n: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = rng.vec(n, -8.0, 8.0);
    for (slot, bits) in v.iter_mut().zip(SPECIALS) {
        *slot = f32::from_bits(bits);
    }
    v
}

// 7. copy_f32_buffer (`copy_f32_buffer.hip:4`): the unfused PLE chain's
// `copy_f32_buffer(query <- streams)` (`layer_ops.rs:3826`; F32 streams on
// gfx942) and the QSA selection copy. Exact.
fn copy_f32_buffer(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE007);
    let mut results = Vec::new();
    for n in wide_sizes() {
        let src = rng.vec(n, -1.0e3, 1.0e3);
        let src_t = up_f32(gpu, &src)?;
        let dst_t = sentinel_f32(gpu, n)?;
        gpu.copy_f32_buffer(&dst_t, &src_t, n).map_err(e)?;
        sync(gpu)?;
        results.push(exact_f32(
            &format!("copy_f32_buffer n={n}"),
            &down_f32(gpu, &dst_t)?,
            &src,
        ));
        free_all(gpu, vec![src_t, dst_t])?;
    }
    all(results)
}

// 8. add_f32 (`add.hip:4`): `gpu.add_f32(&streams, &output, &streams)` —
// the unfused PLE stream add on F32 streams (`layer_ops.rs:3879`), in place
// (`c` aliases `a`). Exact IEEE add.
fn add_f32_inplace(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE008);
    let mut results = Vec::new();
    for n in wide_sizes() {
        let a = rng.vec(n, -4.0, 4.0);
        let b = rng.vec(n, -4.0, 4.0);
        let a_t = up_f32(gpu, &a)?;
        let b_t = up_f32(gpu, &b)?;
        gpu.add_f32(&a_t, &b_t, &a_t).map_err(e)?;
        sync(gpu)?;
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        results.push(exact_f32(
            &format!("add_f32 in-place n={n}"),
            &down_f32(gpu, &a_t)?,
            &want,
        ));
        free_all(gpu, vec![a_t, b_t])?;
    }
    all(results)
}

// 9. convert_f32_to_bf16 (`convert_f32_to_bf16.hip:6`): gfx942 MFMA activation
// staging and BF16-output landing round (`layer_ops.rs:801,819`). RNE; exact.
fn convert_f32_to_bf16(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE009);
    let mut results = Vec::new();
    for n in wide_sizes() {
        let src = stress_values(n, &mut rng);
        let src_t = up_f32(gpu, &src)?;
        let dst_t = up_bf16(gpu, &vec![f32::NAN; n])?;
        gpu.convert_f32_to_bf16(&src_t, &dst_t, n).map_err(e)?;
        sync(gpu)?;
        let want: Vec<u16> = src.iter().map(|&v| f32_to_bf16_bits(v)).collect();
        results.push(exact_bf16(
            &format!("convert_f32_to_bf16 n={n}"),
            &down_bf16(gpu, &dst_t)?,
            &want,
        ));
        free_all(gpu, vec![src_t, dst_t])?;
    }
    all(results)
}

// 10. bf16_round_trip_f32 (`bf16_round_trip.hip:4`): in-place RNE to BF16 and
// back; non-finite values untouched. Used on the MoE / dense-FFN / QSA storage
// boundaries (`mod.rs:3127..4427`, `layer_ops.rs:4172..`). Exact.
fn bf16_round_trip_f32(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE00A);
    let mut results = Vec::new();
    for n in wide_sizes() {
        let src = stress_values(n, &mut rng);
        let x_t = up_f32(gpu, &src)?;
        gpu.bf16_round_trip_f32(&x_t).map_err(e)?;
        sync(gpu)?;
        let want: Vec<f32> = src
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    let u = v.to_bits();
                    f32::from_bits(u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000)
                } else {
                    v
                }
            })
            .collect();
        results.push(exact_f32(
            &format!("bf16_round_trip_f32 n={n}"),
            &down_f32(gpu, &x_t)?,
            &want,
        ));
        free_all(gpu, vec![x_t])?;
    }
    all(results)
}

// 11. zero_f32 (`zero_f32.hip:4`): `ClearOp` (`layer_ops.rs:3906`) before a
// sealed MoE execution. Every element must become +0.0 bitwise.
fn zero_f32(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0xE00B);
    let mut results = Vec::new();
    for n in wide_sizes() {
        let x_t = up_f32(gpu, &rng.vec(n, -5.0, 5.0))?;
        gpu.zero_f32(&x_t).map_err(e)?;
        sync(gpu)?;
        results.push(exact_f32(
            &format!("zero_f32 n={n}"),
            &down_f32(gpu, &x_t)?,
            &vec![0.0f32; n],
        ));
        free_all(gpu, vec![x_t])?;
    }
    all(results)
}
