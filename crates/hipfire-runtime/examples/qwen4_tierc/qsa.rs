// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 QSA (sparse/indexed attention, 12 layers) Tier C cases: GPU kernel vs
//! CPU twin at the Qwen4 shapes (24 q heads / 2 KV heads / head_dim 256,
//! indexer 4 heads x 1 KV x 128, compress 4, budget 2048 = 512 blocks,
//! selected capacity 2051, partial rotary 64, RoPE theta 1e7).
//!
//! Census of record: `local://census-qsa.md` (G942 route audit).
//!
//! gfx942 parallel route (P2). The batched QSA launchers gate on
//! `arch_caps.has_qwen4_parallel_qsa_gdn()` (`has_gfx11_plus_simt() ||
//! is_gfx942()`), so on gfx942 `indexed_attention_select_batch_mirrored`
//! launches the rows8 / rows16 block-score kernels + `select_from_scores` (F32
//! pooled keys, budget <= 512) or the fused `select_{f32,bf16}_batched`
//! (BF16 pooled keys, budget > 512, or an arena past the LDS row: scores in
//! global rows), and `indexed_attention_attention_batch_with_arena` launches
//! `attention_f32_batched` / `_hg4` / `_hg12`; the `*_serial` kernels remain
//! only as the oversized-shape fallback. The rows8 / rows16 and hg4 / hg12
//! choices follow `Gpu::route_rows(rows)`: gfx942 prefill pins
//! `prefill_invariant_scope` (every row count routes as 8192: rows16 / hg12),
//! decode and verify do not (rows8 up to 8 rows, hg4 from 2, hg12 from 16).
//! Still gfx11+ only: the exact `mov_dpp8` selector
//! (`HIPFIRE_QWEN4_QSA_SELECT_EXACT` falls back to the fused selector on
//! gfx942), the single-row `indexed_attention_select` /
//! `indexed_attention_attention`, and the gathered-WMMA / F16-arena routes. The
//! gfx942 kernels run wave64 with software 32-lane shuffle groups
//! (`__shfl_*(.., 32)`, `lane = tid & 31`, `wave = tid >> 5`), so a block's
//! eight software waves behave as on wave32.
//!
//! Serial-kernel exactness claim (gfx942 only): every parallel kernel is bitwise
//! its `*_serial` HIP twin, scores and selection (lowest-index tie rule) and the
//! attention output F32 bits. Open-gate attention batches on every architecture
//! instead assert bitwise equality with n one-row public decode launches.
//! Cases:
//! - `*_route` call the public launchers under both scope settings and every
//!   arm (`select_arm_name` / `attn_arm` label the arm the launcher's predicates
//!   pick) and compare selections as raw ints and attention as raw F32 bits
//!   against the serial HIP kernels launched directly, plus the CPU twins
//!   (plain: tolerance for attention, exact ints for selection).
//! - `qwen4_qsa_parallel_*` launch each enabled kernel directly (a private JIT
//!   module of the shipped `tensor_ops.hip`, `DIRECT_MODULE`) at the same shapes
//!   and compare against the CPU twin, the serial HIP twin and, for scores, the
//!   fused kernel's `score4` scores.
//! - Data: `plain` = dyadic select data / uniform attention operands;
//!   `realistic` = non-dyadic wide-exponent operands with cancelling pairs,
//!   duplicated rows (bitwise ties, the budget boundary forced inside a tie
//!   group) and outlier channels (`DataKind`), so equality is not proved by
//!   dyadic inputs alone. The realistic CPU score twin uses `mul_add` in d order.

use super::common::*;
use super::Q4Case;
use rdna_compute::rope::{Rope, RopeScaling, RopeYarn, Yarn};
use rdna_compute::tensor_ops::{
    indexed_attention_append_prologue, indexed_attention_attention,
    indexed_attention_attention_batch_with_arena, indexed_attention_cache_append,
    indexed_attention_cache_append_batch, indexed_attention_decode_prologue,
    indexed_attention_index_key_append_batch, indexed_attention_norm_rope,
    indexed_attention_norm_rope_batch, indexed_attention_pool_rope,
    indexed_attention_pool_rope_incremental, indexed_attention_reuse_selection,
    indexed_attention_select_batch_mirrored, IndexedAttentionAppendPrologue,
    IndexedAttentionAttention, IndexedAttentionAttentionBatch, IndexedAttentionCacheAppend,
    IndexedAttentionCacheAppendBatch, IndexedAttentionDecodePrologue,
    IndexedAttentionIndexKeyAppendBatch, IndexedAttentionNormRope, IndexedAttentionNormRopeBatch,
    IndexedAttentionPoolRope, IndexedAttentionReuseSelection, IndexedAttentionSelectBatch,
    QsaKvFormat, QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};

mod attn_open_gate;
use attn_open_gate::{attn_arm, attn_one_row_calls};

pub const CASES: &[Q4Case] = &[
    Q4Case {
        name: "qwen4_qsa_gfx942_gates_and_kv_format_refusal",
        timeout_s: 60,
        run: case_gates,
    },
    Q4Case {
        name: "qwen4_indexed_attention_norm_rope_f32_batched",
        timeout_s: 120,
        run: case_norm_rope_batched,
    },
    Q4Case {
        name: "qwen4_indexed_attention_norm_rope_f32_single",
        timeout_s: 60,
        run: case_norm_rope_single,
    },
    Q4Case {
        name: "qwen4_indexed_attention_cache_append_f32_batched",
        timeout_s: 60,
        run: case_cache_append_batched,
    },
    Q4Case {
        name: "qwen4_indexed_attention_cache_append_f32_single",
        timeout_s: 60,
        run: case_cache_append_single,
    },
    Q4Case {
        name: "qwen4_indexed_attention_index_key_append_bf16_batched",
        timeout_s: 60,
        run: case_index_key_append,
    },
    Q4Case {
        name: "qwen4_bf16_round_trip_f32_strided_qsa_index_k",
        timeout_s: 60,
        run: case_round_trip_strided,
    },
    Q4Case {
        name: "qwen4_copy_rows_strided_f32_qsa_raw_index_keys",
        timeout_s: 60,
        run: case_copy_rows_strided,
    },
    Q4Case {
        name: "qwen4_indexed_attention_decode_prologue_f32",
        timeout_s: 120,
        run: case_decode_prologue,
    },
    Q4Case {
        name: "qwen4_indexed_attention_pool_rope_f32",
        timeout_s: 120,
        run: case_pool_rope_f32,
    },
    Q4Case {
        name: "qwen4_indexed_attention_pool_rope_bf16",
        timeout_s: 120,
        run: case_pool_rope_bf16,
    },
    Q4Case {
        name: "qwen4_indexed_attention_select_batch_f32_route",
        timeout_s: 1500,
        run: case_select_f32,
    },
    Q4Case {
        name: "qwen4_indexed_attention_select_batch_bf16_route",
        timeout_s: 1500,
        run: case_select_bf16,
    },
    Q4Case {
        name: "qwen4_indexed_attention_attention_batch_f32_route",
        timeout_s: 600,
        run: case_attention_batch,
    },
    Q4Case {
        name: "qwen4_indexed_attention_attention_batch_f32_route_geometry",
        timeout_s: 600,
        run: case_attention_batch_geometry,
    },
    Q4Case {
        name: "qwen4_indexed_attention_reuse_selection",
        timeout_s: 60,
        run: case_reuse_selection,
    },
    Q4Case {
        name: "qwen4_indexed_attention_attention_f32_single",
        timeout_s: 120,
        run: case_attention_single,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_cache_append",
        timeout_s: 180,
        run: case_fp8_cache_append,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_decode_bits",
        timeout_s: 180,
        run: case_fp8_decode_bits,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_decode_prologue",
        timeout_s: 240,
        run: case_fp8_decode_prologue,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_attention_batch_route",
        timeout_s: 600,
        run: case_fp8_attention_batch,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_attention_single",
        timeout_s: 180,
        run: case_fp8_attention_single,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_select_attention_route",
        timeout_s: 600,
        run: case_fp8_select_attention,
    },
    Q4Case {
        name: "qwen4_qsa_fp8_fnuz_attention_batch_serial_equality",
        timeout_s: 600,
        run: case_fp8_attention_serial_equality,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_select_scores_rows8_f32",
        timeout_s: 900,
        run: parallel_rows8,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_select_scores_rows16_f32",
        timeout_s: 900,
        run: parallel_rows16,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_select_scores_rows32_f32",
        timeout_s: 900,
        run: parallel_rows32,
    },
    Q4Case {
        name: "qwen4_qsa_wide_decode_prologue_f32_one_row_bitwise",
        timeout_s: 1200,
        run: wide_prologue_f32,
    },
    Q4Case {
        name: "qwen4_qsa_wide_decode_prologue_fp8_one_row_bitwise",
        timeout_s: 1200,
        run: wide_prologue_fp8,
    },
    Q4Case {
        name: "qwen4_qsa_wide_pool_rope_f32_one_row_bitwise",
        timeout_s: 1200,
        run: wide_pool_f32,
    },
    Q4Case {
        name: "qwen4_qsa_wide_pool_rope_bf16_one_row_bitwise",
        timeout_s: 1200,
        run: wide_pool_bf16,
    },
    Q4Case {
        name: "qwen4_qsa_wide_attention_f32_one_row_bitwise",
        timeout_s: 1800,
        run: wide_attention_f32,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_select_f32_batched",
        timeout_s: 1200,
        run: parallel_select_f32,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_select_bf16_batched",
        timeout_s: 1200,
        run: parallel_select_bf16,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_attention_f32_batched",
        timeout_s: 600,
        run: parallel_attention_batched,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_attention_f32_batched_hg4",
        timeout_s: 600,
        run: parallel_attention_hg4,
    },
    Q4Case {
        name: "qwen4_qsa_parallel_attention_f32_batched_hg12",
        timeout_s: 600,
        run: parallel_attention_hg12,
    },
];

// ── Geometry ──────────────────────────────────────────────────────────────────
const HD: usize = QSA_HEAD_DIM; // 256
const IDX_DIM: usize = QSA_INDEX_DIM; // 128
const IDX_Q_W: usize = QSA_INDEX_HEADS * QSA_INDEX_DIM; // 512
const IDX_W: usize = IDX_Q_W + IDX_DIM; // 640: [index q | index k]
const KV_W: usize = QSA_KV_HEADS * QSA_HEAD_DIM; // 512
const Q_W: usize = QSA_Q_HEADS * QSA_HEAD_DIM; // 6144
const QG_W: usize = 2 * Q_W; // 12288: [heads][q | gate]
const ROTARY: usize = 64; // head_dim * partial_rotary_factor 0.25
const BUDGET_BLOCKS: usize = QSA_BUDGET / QSA_COMPRESS; // 512
const SEL_CAP: usize = QSA_BUDGET + QSA_COMPRESS - 1; // 2051
const POOLED_CAP: usize = 1024; // shape bound handed to the selector

// ── Small helpers ─────────────────────────────────────────────────────────────
/// Byte-addressed `DType::Raw` i32 tensor with `numel == bytes`, the shape the
/// QSA launchers check (`selected.numel() >= capacity * 4`).
fn raw_i32(gpu: &mut Gpu, data: &[i32]) -> Result<GpuTensor, String> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.upload_raw(&bytes, &[bytes.len()]).map_err(e)
}

fn bf16_vec(gpu: &mut Gpu, data: &[f32]) -> Result<GpuTensor, String> {
    up_bf16(gpu, data)
}

/// NaN-sentinel F32 tensor, shaped `[n]`.
fn nan_f32(gpu: &mut Gpu, n: usize) -> Result<GpuTensor, String> {
    sentinel_f32(gpu, n)
}

/// RNE round of a finite F32 to BF16 precision, in place on the F32 bits
/// (`bf16_round_trip_f32_strided` / the prologue's index-key round trip).
fn bf16_trunc_rne(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7FFF + ((u >> 16) & 1)) & 0xFFFF_0000)
}

fn dyadic(rng: &mut Rng, n: usize, denom: f32) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.below(17) as f32 - 8.0) / denom)
        .collect()
}

/// Compare `got` to `want`: elements where `exact_at(i)` bit-exactly, the rest
/// within `tol`.
fn cmp_split(
    label: &str,
    got: &[f32],
    want: &[f32],
    tol: Tol,
    exact_at: impl Fn(usize) -> bool,
) -> Result<String, String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    let (mut ge, mut we, mut gc, mut wc) = (vec![], vec![], vec![], vec![]);
    for i in 0..got.len() {
        if exact_at(i) {
            ge.push(got[i]);
            we.push(want[i]);
        } else {
            gc.push(got[i]);
            wc.push(want[i]);
        }
    }
    all(vec![
        exact_f32(&format!("{label}[copy]"), &ge, &we),
        close_f32(&format!("{label}[math]"), &gc, &wc, tol),
    ])
}

fn yarns() -> [(&'static str, RopeYarn); 2] {
    let rope = Rope {
        theta: 1.0e7,
        scaling: RopeScaling::Yarn(Yarn::new(4.0, 4096, None, None, None, None)),
    };
    [("plain", RopeYarn::OFF), ("yarn4", rope.yarn(ROTARY))]
}

// ── CPU twins ─────────────────────────────────────────────────────────────────
/// `indexed_attention_norm_rope_f32{,_batched}` / `qsa_norm_rope_head`
/// (`kernels/src/tensor_ops.hip:2881-2921`, `3176-3215`): serial i-order
/// sum of squares, `inv = rsqrt(sum/dim + 1e-6)`, `(1 + norm)` scale, HalfSplit
/// RoPE over the first `rotary_dim` channels with
/// `angle = position * powf(1e7, -2*base/rotary_dim)` and the YaRN `sin_cos`
/// host twin. The arch-qwen4 CPU twin is `ops.rs:464 rope_prefix_halfsplit`
/// (same angle/yarn math; it has no RMS/learned-norm stage).
fn norm_rope_head(
    x: &[f32],
    norm: &[f32],
    position: usize,
    rotary_dim: usize,
    yarn: RopeYarn,
) -> Vec<f32> {
    let head_dim = x.len();
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let inv = 1.0f32 / (sum / head_dim as f32 + 1.0e-6f32).sqrt();
    let half = rotary_dim / 2;
    let mut out = vec![0.0f32; head_dim];
    for c in 0..head_dim {
        let mut value = x[c] * inv * (1.0 + norm[c]);
        if c < rotary_dim {
            let partner = if c < half { c + half } else { c - half };
            let other = x[partner] * inv * (1.0 + norm[partner]);
            let base = if c < half { c } else { partner };
            let angle =
                position as f32 * 10_000_000.0f32.powf(-((2 * base) as f32) / rotary_dim as f32);
            let (sine, cosine) = yarn.sin_cos(angle, base);
            value = if c < half {
                value * cosine - other * sine
            } else {
                other * sine + value * cosine
            };
        }
        out[c] = value;
    }
    out
}

/// One pooled index block, `qsa_pool_rope` (`tensor_ops.hip:3471-3542`): mean of
/// `compress` raw keys (sequential `/compress`), BF16 round, BF16-rounded
/// square, the kernel's 256-slot LDS tree sum (strides 128..1), learned-norm
/// scale rounded to BF16, partner RoPE at `block * compress`, final BF16
/// round. (`ops.rs:490 qsa_pool_block` / `reference_forward.rs:946` are the
/// model-level twins without the BF16 intermediates.)
fn pool_rope_block(raw: &[f32], norm: &[f32], block: usize, yarn: RopeYarn) -> Vec<f32> {
    let dim = IDX_DIM;
    let mean_bf16: Vec<f32> = (0..dim)
        .map(|c| {
            let mut mean = 0.0f32;
            for row in 0..QSA_COMPRESS {
                mean += raw[row * dim + c] / QSA_COMPRESS as f32;
            }
            bf16_round(mean)
        })
        .collect();
    let mut partial = vec![0.0f32; 256];
    for c in 0..dim {
        partial[c] = bf16_round(mean_bf16[c] * mean_bf16[c]);
    }
    let mut stride = 128;
    while stride > 0 {
        for t in 0..stride {
            partial[t] += partial[t + stride];
        }
        stride >>= 1;
    }
    let inv = 1.0f32 / (partial[0] / dim as f32 + 1.0e-6f32).sqrt();
    let half = ROTARY / 2;
    (0..dim)
        .map(|c| {
            let mut value = bf16_round(mean_bf16[c] * inv * (1.0 + norm[c]));
            if c < ROTARY {
                let partner = if c < half { c + half } else { c - half };
                let other = bf16_round(mean_bf16[partner] * inv * (1.0 + norm[partner]));
                let pair = if c < half { c } else { partner };
                let angle = (block * QSA_COMPRESS) as f32
                    * 10_000_000.0f32.powf(-((2 * pair) as f32) / ROTARY as f32);
                let (sine, cosine) = yarn.sin_cos(angle, pair);
                value = if c < half {
                    value * cosine - other * sine
                } else {
                    other * sine + value * cosine
                };
            }
            bf16_round(value)
        })
        .collect()
}

/// `indexed_attention_select_score_f32` (`tensor_ops.hip:3570-3591`): per head
/// a serial d-order dot, `relu(dot / sqrt(128))`, heads summed in order.
/// Exact for the dyadic test data (every product and partial sum is exactly
/// representable, so FMA contraction / association cannot change a bit).
/// Model-level twin: `ops.rs:517 qsa_select_blocks`,
/// `reference_forward.rs:963 select_index_tokens`.
fn index_score(q: &[f32], key: &[f32]) -> f32 {
    let mut score = 0.0f32;
    for h in 0..QSA_INDEX_HEADS {
        let mut dot = 0.0f32;
        for d in 0..IDX_DIM {
            dot += q[h * IDX_DIM + d] * key[d];
        }
        score += (dot / (IDX_DIM as f32).sqrt()).max(0.0);
    }
    score
}

/// Blocks ordered (score desc, index asc): the stable lowest-index tie rule.
fn rank_order(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&x, &y| {
        scores[y]
            .partial_cmp(&scores[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(&y))
    });
    idx
}

/// Operand family of the select / attention exactness cases. `Plain`: dyadic
/// select data (every product and partial sum exactly representable, so the
/// CPU twin is exact whatever the kernel's contraction or association) and
/// uniform attention operands. `Realistic`: non-dyadic wide-exponent operands
/// with adjacent-pair cancellation, duplicated rows (bitwise ties) and outlier
/// channels, where only a kernel that reproduces its serial twin's exact
/// operation order passes a raw-bit comparison.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DataKind {
    Plain,
    Realistic,
}

impl DataKind {
    fn name(self) -> &'static str {
        match self {
            DataKind::Plain => "plain",
            DataKind::Realistic => "realistic",
        }
    }

    /// The CPU twin that reproduces the kernels' scores bit for bit.
    fn twin_name(self) -> &'static str {
        match self {
            DataKind::Plain => "cpu_twin",
            DataKind::Realistic => "cpu_fma_twin",
        }
    }

    fn seed(self) -> u64 {
        match self {
            DataKind::Plain => 0,
            DataKind::Realistic => 0x1_0000,
        }
    }
}

/// [`index_score`] with the kernels' fused dot chain: the device builds
/// `dot += q * p` as one FMA per term (`-ffp-contract=fast` default), so
/// `mul_add` in d order reproduces `indexed_attention_select_score_f32` /
/// `score4_f32` and the rows8/rows16 kernels for data that is not dyadic.
fn index_score_fma(q: &[f32], key: &[f32]) -> f32 {
    let mut score = 0.0f32;
    for h in 0..QSA_INDEX_HEADS {
        let mut dot = 0.0f32;
        for d in 0..IDX_DIM {
            dot = q[h * IDX_DIM + d].mul_add(key[d], dot);
        }
        score += (dot / (IDX_DIM as f32).sqrt()).max(0.0);
    }
    score
}

fn index_score_of(kind: DataKind) -> fn(&[f32], &[f32]) -> f32 {
    match kind {
        DataKind::Plain => index_score,
        DataKind::Realistic => index_score_fma,
    }
}

/// `[rows][capacity]` selection of `indexed_attention_select_*_batched*`
/// (`qsa_select_batched_serial` twin): top `min(budget, blocks)` blocks by
/// (score desc, index asc) at slot `rank * compress + r`, then the causal tail
/// tokens, `-1` elsewhere. `score` is the per-block score twin. Returns
/// `(selected, scores[row][block])`.
#[allow(clippy::too_many_arguments)]
fn select_rows_cpu_with(
    score: fn(&[f32], &[f32]) -> f32,
    query: &[f32],
    pooled: &[f32],
    rows: usize,
    blocks: usize,
    position_start: usize,
    budget: usize,
    capacity: usize,
) -> (Vec<i32>, Vec<f32>) {
    let mut selected = vec![-1i32; rows * capacity];
    let mut all_scores = vec![0.0f32; rows * blocks];
    for r in 0..rows {
        let visible = position_start + r + 1;
        let rbc = (visible / QSA_COMPRESS).min(blocks);
        let q = &query[r * IDX_W..r * IDX_W + IDX_Q_W];
        let scores: Vec<f32> = (0..rbc)
            .map(|b| score(q, &pooled[b * IDX_DIM..(b + 1) * IDX_DIM]))
            .collect();
        all_scores[r * blocks..r * blocks + rbc].copy_from_slice(&scores);
        let chosen = budget.min(rbc);
        let order = rank_order(&scores);
        let row = &mut selected[r * capacity..(r + 1) * capacity];
        for (slot, &block) in order.iter().take(chosen).enumerate() {
            for k in 0..QSA_COMPRESS {
                let index = slot * QSA_COMPRESS + k;
                let token = block * QSA_COMPRESS + k;
                if index < capacity && token < visible {
                    row[index] = token as i32;
                }
            }
        }
        let mut offset = chosen * QSA_COMPRESS;
        for token in rbc * QSA_COMPRESS..visible {
            if offset >= capacity {
                break;
            }
            row[offset] = token as i32;
            offset += 1;
        }
    }
    (selected, all_scores)
}

/// [`select_rows_cpu_with`] at the Qwen4 budget (512 blocks, capacity 2051) with
/// the dyadic score twin.
fn select_rows_cpu(
    query: &[f32],
    pooled: &[f32],
    rows: usize,
    blocks: usize,
    position_start: usize,
) -> (Vec<i32>, Vec<f32>) {
    select_rows_cpu_with(
        index_score,
        query,
        pooled,
        rows,
        blocks,
        position_start,
        BUDGET_BLOCKS,
        SEL_CAP,
    )
}

/// A `[rows][blocks]` score table from rows laid out at `stride` (the CPU twin's
/// `stride == blocks`, the fused selector's global rows
/// `blocks.div_ceil(4) * 4`), with every slot a row does not score (blocks past
/// its visible count) set to the kernels' NaN sentinel.
fn masked_scores(
    table: &[f32],
    rows: usize,
    blocks: usize,
    stride: usize,
    position_start: usize,
) -> Vec<f32> {
    let mut out = vec![f32::from_bits(0x7FC0_DEAD); rows * blocks];
    for r in 0..rows {
        let rbc = ((position_start + r + 1) / QSA_COMPRESS).min(blocks);
        out[r * blocks..r * blocks + rbc].copy_from_slice(&table[r * stride..r * stride + rbc]);
    }
    out
}

/// Wide-dynamic-range values: uniform [-1, 1] mantissas scaled by 2^-5..2^5.
fn wide_row(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let m = rng.range(-1.0, 1.0);
            m * 2.0f32.powi(rng.below(11) as i32 - 5)
        })
        .collect()
}

/// Select inputs with abundant exact ties. `Plain`: a palette of 24 repeated
/// dyadic pooled rows, all-zero rows (relu -> score 0) and fresh rows.
/// `Realistic`: the same structure over wide-exponent rows, plus pooled rows
/// whose adjacent channels are exact negations (`p[2i+1] = -p[2i]`) against
/// index heads 1 and 3 whose adjacent query channels are equal, so their dot
/// chains cancel pairwise and the result depends on the exact FMA / association
/// order. The last row's top-`min(budget, blocks)` boundary is forced to sit
/// inside a tie group when blocks exceed the budget. `bf16`: pooled values are
/// RNE-rounded to BF16 first (the BF16 arena then holds them exactly). Returns
/// `(query[rows*640], pooled[blocks*128])`.
fn gen_select_data_kind(
    seed: u64,
    rows: usize,
    blocks: usize,
    budget: usize,
    kind: DataKind,
    bf16: bool,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut rng = Rng::new(seed);
    let score = index_score_of(kind);
    let mut pooled = Vec::with_capacity(blocks * IDX_DIM);
    let mut query = Vec::with_capacity(rows * IDX_W);
    match kind {
        DataKind::Plain => {
            let palette: Vec<Vec<f32>> = (0..24).map(|_| dyadic(&mut rng, IDX_DIM, 32.0)).collect();
            for b in 0..blocks {
                if b % 5 == 0 {
                    pooled.extend(dyadic(&mut rng, IDX_DIM, 32.0));
                } else if b % 11 == 0 {
                    pooled.extend(vec![0.0f32; IDX_DIM]);
                } else {
                    pooled.extend_from_slice(&palette[rng.below(24)]);
                }
            }
            for _ in 0..rows {
                query.extend(dyadic(&mut rng, IDX_Q_W, 16.0));
                query.extend(rng.vec(IDX_DIM, -1.0, 1.0)); // index k half: ignored by select
            }
        }
        DataKind::Realistic => {
            let palette: Vec<Vec<f32>> = (0..24).map(|_| wide_row(&mut rng, IDX_DIM)).collect();
            for b in 0..blocks {
                match b % 13 {
                    0 => pooled.extend(vec![0.0f32; IDX_DIM]),
                    1 | 7 => {
                        for x in wide_row(&mut rng, IDX_DIM / 2) {
                            pooled.push(x);
                            pooled.push(-x);
                        }
                    }
                    2 | 5 | 9 => pooled.extend(wide_row(&mut rng, IDX_DIM)),
                    _ => pooled.extend_from_slice(&palette[rng.below(24)]),
                }
            }
            for _ in 0..rows {
                for head in 0..QSA_INDEX_HEADS {
                    let mut q = wide_row(&mut rng, IDX_DIM);
                    if head % 2 == 1 {
                        for i in 0..IDX_DIM / 2 {
                            q[2 * i + 1] = q[2 * i];
                        }
                    }
                    query.extend(q);
                }
                query.extend(rng.vec(IDX_DIM, -1.0, 1.0)); // index k half: ignored by select
            }
        }
    }
    if bf16 {
        for v in &mut pooled {
            *v = bf16_round(*v);
        }
    }
    let chosen = budget.min(blocks);
    if chosen < blocks {
        let last_q = &query[(rows - 1) * IDX_W..(rows - 1) * IDX_W + IDX_Q_W];
        let scores_of = |pooled: &[f32]| -> Vec<f32> {
            (0..blocks)
                .map(|b| score(last_q, &pooled[b * IDX_DIM..(b + 1) * IDX_DIM]))
                .collect()
        };
        let mut scores = scores_of(&pooled);
        let order = rank_order(&scores);
        let (a, b) = (order[chosen - 1], order[chosen]);
        if scores[a] != scores[b] {
            let row = pooled[a * IDX_DIM..(a + 1) * IDX_DIM].to_vec();
            pooled[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&row);
            scores = scores_of(&pooled);
        }
        let order = rank_order(&scores);
        if scores[order[chosen - 1]] != scores[order[chosen]] {
            return Err("select data: failed to straddle the budget boundary with a tie".into());
        }
    }
    Ok((query, pooled))
}

/// [`gen_select_data_kind`] with the dyadic family at the Qwen4 budget.
fn gen_select_data(seed: u64, rows: usize, blocks: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    gen_select_data_kind(seed, rows, blocks, BUDGET_BLOCKS, DataKind::Plain, false)
}

/// One query row of `qsa_attention_batched` / `_serial`
/// (`tensor_ops.hip:4338-4417`, `4419-4605`; model-level twin
/// `ops.rs:582 qsa_attention`, which multiplies by `sigmoid(gate)` per term
/// instead of dividing the finished sum). Invalid selected entries
/// (`< 0`, `>= full_capacity`, `>= visible`) contribute nothing.
fn attention_row_cpu(
    q_row: &[f32],
    keys: &[f32],
    values: &[f32],
    sel: &[i32],
    visible: usize,
    full_capacity: usize,
) -> Vec<f32> {
    let group = QSA_Q_HEADS / QSA_KV_HEADS;
    let mut out = vec![0.0f32; Q_W];
    for head in 0..QSA_Q_HEADS {
        let kv_head = head / group;
        let q = &q_row[head * 2 * HD..head * 2 * HD + HD];
        let gate = &q_row[head * 2 * HD + HD..head * 2 * HD + 2 * HD];
        let mut toks: Vec<usize> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut max_score = -3.402_823_5e38f32;
        for &t in sel {
            if t < 0 || t as usize >= full_capacity || t as usize >= visible {
                continue;
            }
            let t = t as usize;
            let key = &keys[t * KV_W + kv_head * HD..t * KV_W + kv_head * HD + HD];
            let mut s = 0.0f32;
            for d in 0..HD {
                s += q[d] * key[d];
            }
            let s = s / (HD as f32).sqrt();
            max_score = max_score.max(s);
            toks.push(t);
            scores.push(s);
        }
        let mut den = 0.0f32;
        let mut num = vec![0.0f32; HD];
        for (i, &t) in toks.iter().enumerate() {
            let w = (scores[i] - max_score).exp();
            den += w;
            let v = &values[t * KV_W + kv_head * HD..t * KV_W + kv_head * HD + HD];
            for c in 0..HD {
                num[c] += w * v[c];
            }
        }
        for c in 0..HD {
            out[head * HD + c] = if den > 0.0 {
                num[c] / den / (1.0 + (-gate[c]).exp())
            } else {
                0.0
            };
        }
    }
    out
}

struct AttnInputs {
    position_start: usize,
    q: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    selected: Vec<i32>,
    want: Vec<f32>,
}

/// Injected-invalid selection for `rows` rows ending at `end` tokens: the CPU
/// select twin's selection (`#blocks > 512` => blocks are dropped), plus
/// entries on row 0 (`-1` and a not-yet-visible token) the kernels must skip.
fn attn_selection(seed: u64, rows: usize, end: usize) -> Result<Vec<i32>, String> {
    let blocks = end / QSA_COMPRESS;
    let position_start = end - rows;
    let (query, pooled) = gen_select_data(seed ^ 0x5151, rows, blocks)?;
    let (mut selected, _) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
    let visible0 = position_start + 1;
    selected[3] = -1;
    selected[5] = (visible0 + 1) as i32;
    Ok(selected)
}

/// Realistic attention operands at any `heads x kv_heads x head_dim`: per-head
/// query scales spanning 2^-3..2^3 (peaked and flat softmaxes), outlier key
/// channels (every 16th x8), every fifth key duplicated from four tokens back
/// (equal scores, equal weights), and odd-token values the exact negation of
/// the previous token's (the weighted sums cancel). Layout: q `[rows][heads][q |
/// gate]`, keys / values `[end][kv_heads * head_dim]`.
fn realistic_attn_qkv(
    seed: u64,
    rows: usize,
    end: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed ^ 0xA77E);
    let kv_w = kv_heads * head_dim;
    let mut q = Vec::with_capacity(rows * 2 * heads * head_dim);
    for _ in 0..rows {
        for _ in 0..heads {
            let scale = 2.0f32.powi(rng.below(7) as i32 - 3);
            q.extend(rng.vec(head_dim, -3.0, 3.0).into_iter().map(|x| x * scale));
            q.extend(rng.vec(head_dim, -3.0, 3.0));
        }
    }
    let mut keys = rng.vec(end * kv_w, -1.0, 1.0);
    for t in 0..end {
        for c in (0..kv_w).step_by(16) {
            keys[t * kv_w + c] *= 8.0;
        }
    }
    for t in (5..end).step_by(5) {
        keys.copy_within((t - 4) * kv_w..(t - 3) * kv_w, t * kv_w);
    }
    let mut values = rng.vec(end * kv_w, -1.0, 1.0);
    for t in (1..end).step_by(2) {
        for c in 0..kv_w {
            values[t * kv_w + c] = -values[(t - 1) * kv_w + c];
        }
    }
    (q, keys, values)
}

/// Attention inputs at `end` cached tokens (uniform operands).
fn attn_inputs(seed: u64, rows: usize, end: usize) -> Result<AttnInputs, String> {
    attn_inputs_kind(seed, rows, end, DataKind::Plain)
}

/// [`attn_inputs`] with the operands of `kind`: plain = uniform q in [-3, 3] and
/// keys / values in [-1, 1]; realistic = [`realistic_attn_qkv`].
fn attn_inputs_kind(
    seed: u64,
    rows: usize,
    end: usize,
    kind: DataKind,
) -> Result<AttnInputs, String> {
    let position_start = end - rows;
    let selected = attn_selection(seed, rows, end)?;
    let (q, keys, values) = match kind {
        DataKind::Plain => {
            let mut rng = Rng::new(seed);
            let q = rng.vec(rows * QG_W, -3.0, 3.0);
            let keys = rng.vec(end * KV_W, -1.0, 1.0);
            let values = rng.vec(end * KV_W, -1.0, 1.0);
            (q, keys, values)
        }
        DataKind::Realistic => realistic_attn_qkv(seed, rows, end, QSA_Q_HEADS, QSA_KV_HEADS, HD),
    };
    let mut want = Vec::with_capacity(rows * Q_W);
    for r in 0..rows {
        let visible = position_start + r + 1;
        let block_count = visible / QSA_COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        let selected_len =
            (chosen * QSA_COMPRESS + visible - block_count * QSA_COMPRESS).min(SEL_CAP);
        want.extend(attention_row_cpu(
            &q[r * QG_W..(r + 1) * QG_W],
            &keys,
            &values,
            &selected[r * SEL_CAP..r * SEL_CAP + selected_len],
            visible,
            end,
        ));
    }
    Ok(AttnInputs {
        position_start,
        q,
        keys,
        values,
        selected,
        want,
    })
}

// ── norm + RoPE ───────────────────────────────────────────────────────────────
struct NrConfig {
    name: &'static str,
    heads: usize,
    head_dim: usize,
    head_stride: usize,
    row_stride: usize,
}
const NR_CONFIGS: [NrConfig; 3] = [
    NrConfig {
        name: "index_q",
        heads: QSA_INDEX_HEADS,
        head_dim: IDX_DIM,
        head_stride: IDX_DIM,
        row_stride: IDX_W,
    },
    NrConfig {
        name: "q",
        heads: QSA_Q_HEADS,
        head_dim: HD,
        head_stride: 2 * HD,
        row_stride: QG_W,
    },
    NrConfig {
        name: "k",
        heads: QSA_KV_HEADS,
        head_dim: HD,
        head_stride: HD,
        row_stride: KV_W,
    },
];

fn nr_want(
    cfg: &NrConfig,
    input: &[f32],
    norm: &[f32],
    rows: usize,
    pos0: usize,
    yarn: RopeYarn,
) -> (Vec<f32>, Vec<bool>) {
    let mut want = input.to_vec();
    let mut touched = vec![false; input.len()];
    for r in 0..rows {
        for h in 0..cfg.heads {
            let start = r * cfg.row_stride + h * cfg.head_stride;
            let out = norm_rope_head(
                &input[start..start + cfg.head_dim],
                norm,
                pos0 + r,
                ROTARY,
                yarn,
            );
            want[start..start + cfg.head_dim].copy_from_slice(&out);
            for t in &mut touched[start..start + cfg.head_dim] {
                *t = true;
            }
        }
    }
    (want, touched)
}

#[allow(clippy::too_many_arguments)]
fn norm_rope_launch(
    gpu: &mut Gpu,
    values: &GpuTensor,
    norm: &GpuTensor,
    rows: usize,
    position: usize,
    yarn: RopeYarn,
    heads: usize,
    head_dim: usize,
    head_stride: usize,
    row_stride: usize,
) -> Result<(), String> {
    indexed_attention_norm_rope_batch(
        gpu,
        &IndexedAttentionNormRopeBatch {
            values,
            norm,
            rows,
            row_stride,
            heads,
            head_dim,
            head_stride,
            position_start: position,
            rotary_dim: ROTARY,
            yarn,
        },
    )
    .map_err(e)
}

/// `indexed_attention_norm_rope_batch` (the route's norm+RoPE for rows > 8 in
/// decode and for every prefill chunk on gfx942, where `route_rows == 8192`).
/// Positions stay below 40: the CPU `powf`/`sin`/`cos` differ from the device
/// ocml functions by ~1 ulp of the angle, which the frozen F32_ACCUM tol only
/// absorbs at small angles.
fn case_norm_rope_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for cfg in &NR_CONFIGS {
        let mut rng = Rng::new(0x7101 + cfg.heads as u64);
        let norm = rng.vec_bf16_exact(cfg.head_dim, -0.5, 0.5);
        let norm_t = bf16_vec(gpu, &norm)?;
        for (yname, yarn) in yarns() {
            for rows in ROW_COUNTS {
                let pos0 = 3;
                let input = rng.vec(rows * cfg.row_stride, -2.0, 2.0);
                let values = up_f32(gpu, &input)?;
                norm_rope_launch(
                    gpu,
                    &values,
                    &norm_t,
                    rows,
                    pos0,
                    yarn,
                    cfg.heads,
                    cfg.head_dim,
                    cfg.head_stride,
                    cfg.row_stride,
                )?;
                sync(gpu)?;
                let got = down_f32(gpu, &values)?;
                let (want, touched) = nr_want(cfg, &input, &norm, rows, pos0, yarn);
                checks.push(cmp_split(
                    &format!("{}/{yname}/rows{rows}", cfg.name),
                    &got,
                    &want,
                    F32_ACCUM,
                    |i| !touched[i],
                ));
                free_all(gpu, vec![values])?;
            }
        }
        free_all(gpu, vec![norm_t])?;
    }
    all(checks)
}

/// `indexed_attention_norm_rope` (single row; used by parity harnesses, not by
/// the C1 route): index-q and k configs at position 5.
fn case_norm_rope_single(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for cfg in [&NR_CONFIGS[0], &NR_CONFIGS[2]] {
        let mut rng = Rng::new(0x7201 + cfg.heads as u64);
        let norm = rng.vec_bf16_exact(cfg.head_dim, -0.5, 0.5);
        let norm_t = bf16_vec(gpu, &norm)?;
        let input = rng.vec(cfg.row_stride, -2.0, 2.0);
        let values = up_f32(gpu, &input)?;
        indexed_attention_norm_rope(
            gpu,
            &IndexedAttentionNormRope {
                values: &values,
                norm: &norm_t,
                heads: cfg.heads,
                head_dim: cfg.head_dim,
                head_stride: cfg.head_stride,
                position: 5,
                rotary_dim: ROTARY,
                yarn: RopeYarn::OFF,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let got = down_f32(gpu, &values)?;
        let (want, touched) = nr_want(cfg, &input, &norm, 1, 5, RopeYarn::OFF);
        checks.push(cmp_split(cfg.name, &got, &want, F32_ACCUM, |i| !touched[i]));
        free_all(gpu, vec![values, norm_t])?;
    }
    all(checks)
}

// ── cache append / index key append ──────────────────────────────────────────
/// `indexed_attention_cache_append_f32_batched` (`tensor_ops.hip:3377`): pure
/// copy to `position_start + row`; bit-exact incl. untouched rows. 600 rows
/// exceeds every row threshold the route sees (512).
fn case_cache_append_batched(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in [1usize, 7, 33, 600] {
        let mut rng = Rng::new(0x7301 + rows as u64);
        let key = rng.vec(rows * KV_W, -2.0, 2.0);
        let value = rng.vec(rows * KV_W, -2.0, 2.0);
        let cap = (pos0 + rows) * KV_W;
        let (k_t, v_t) = (up_f32(gpu, &key)?, up_f32(gpu, &value)?);
        let (fk, fv) = (nan_f32(gpu, cap)?, nan_f32(gpu, cap)?);
        indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &k_t,
                value: &v_t,
                full_keys: &fk,
                full_values: &fv,
                rows,
                position_start: pos0,
                kv_heads: QSA_KV_HEADS,
                head_dim: HD,
                format: QsaKvFormat::F32,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let sentinel = f32::from_bits(0x7FC0_DEAD);
        let mut want_k = vec![sentinel; cap];
        let mut want_v = vec![sentinel; cap];
        want_k[pos0 * KV_W..].copy_from_slice(&key);
        want_v[pos0 * KV_W..].copy_from_slice(&value);
        checks.push(exact_f32(
            &format!("rows{rows}/keys"),
            &down_f32(gpu, &fk)?,
            &want_k,
        ));
        checks.push(exact_f32(
            &format!("rows{rows}/values"),
            &down_f32(gpu, &fv)?,
            &want_v,
        ));
        free_all(gpu, vec![k_t, v_t, fk, fv])?;
    }
    all(checks)
}

/// `indexed_attention_cache_append_f32` (single row; parity harnesses only).
fn case_cache_append_single(gpu: &mut Gpu) -> Result<String, String> {
    let mut rng = Rng::new(0x7401);
    let key = rng.vec(KV_W, -2.0, 2.0);
    let value = rng.vec(KV_W, -2.0, 2.0);
    let position = 7;
    let cap = (position + 1) * KV_W;
    let (k_t, v_t) = (up_f32(gpu, &key)?, up_f32(gpu, &value)?);
    let (fk, fv) = (nan_f32(gpu, cap)?, nan_f32(gpu, cap)?);
    indexed_attention_cache_append(
        gpu,
        &IndexedAttentionCacheAppend {
            key: &k_t,
            value: &v_t,
            full_keys: &fk,
            full_values: &fv,
            position,
            kv_width: KV_W,
        },
    )
    .map_err(e)?;
    sync(gpu)?;
    let sentinel = f32::from_bits(0x7FC0_DEAD);
    let mut want_k = vec![sentinel; cap];
    let mut want_v = vec![sentinel; cap];
    want_k[position * KV_W..].copy_from_slice(&key);
    want_v[position * KV_W..].copy_from_slice(&value);
    let r = all(vec![
        exact_f32("keys", &down_f32(gpu, &fk)?, &want_k),
        exact_f32("values", &down_f32(gpu, &fv)?, &want_v),
    ]);
    free_all(gpu, vec![k_t, v_t, fk, fv])?;
    r
}

/// Index-key specials: RNE ties both ways, just below/above a tie, ±Inf (kept
/// unrounded in place), signed zeros.
fn specials(include_nan: bool) -> Vec<f32> {
    let mut v = vec![
        f32::from_bits(0x3F80_8000),
        f32::from_bits(0x3F81_8000),
        f32::from_bits(0x3F80_7FFF),
        f32::from_bits(0x3F80_8001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
    ];
    if include_nan {
        v.push(f32::from_bits(0x7FC0_1234));
    }
    v
}

/// `indexed_attention_index_key_append_bf16_batched` (`tensor_ops.hip:3441`).
/// NOT on the gfx942 route: only a BF16 raw-key arena takes it, and the BF16
/// index-key formats (`QsaKvFormat::{Fp8,Q8}`) are RDNA-only (gfx942 refuses
/// both; its `Fp8Fnuz` keeps F32 index arenas); F32 state uses the round-trip +
/// row-copy pair (cases below). Exercised here because the launcher/kernel are
/// portable.
fn case_index_key_append(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in [1usize, 7, 33, 600] {
        let mut rng = Rng::new(0x7501 + rows as u64);
        let mut input = rng.vec(rows * IDX_W, -3.0, 3.0);
        for (i, s) in specials(false).into_iter().enumerate() {
            input[IDX_Q_W + i] = s; // row 0's index-k head
        }
        let rows_t = up_f32(gpu, &input)?;
        let raw = bf16_vec(gpu, &vec![f32::NAN; (pos0 + rows) * IDX_DIM])?;
        indexed_attention_index_key_append_batch(
            gpu,
            &IndexedAttentionIndexKeyAppendBatch {
                index_rows: &rows_t,
                raw_index_keys: &raw,
                rows,
                index_q_width: IDX_Q_W,
                index_kv_width: IDX_DIM,
                position_start: pos0,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let mut want_rows = input.clone();
        let mut want_raw = vec![f32_to_bf16_bits(f32::NAN); (pos0 + rows) * IDX_DIM];
        for r in 0..rows {
            for i in 0..IDX_DIM {
                let idx = r * IDX_W + IDX_Q_W + i;
                let v = input[idx];
                let rounded = if v.is_finite() { bf16_trunc_rne(v) } else { v };
                want_rows[idx] = rounded;
                want_raw[(pos0 + r) * IDX_DIM + i] = f32_to_bf16_bits(rounded);
            }
        }
        checks.push(exact_f32(
            &format!("rows{rows}/index_rows"),
            &down_f32(gpu, &rows_t)?,
            &want_rows,
        ));
        checks.push(exact_bf16(
            &format!("rows{rows}/raw"),
            &down_bf16(gpu, &raw)?,
            &want_raw,
        ));
        free_all(gpu, vec![rows_t, raw])?;
    }
    all(checks)
}

/// `bf16_round_trip_f32_strided` (`qwen4_bf16_round_trip.hip`), the route's F32
/// index-k boundary (offset 512, width 128, stride 640). Non-finite values
/// pass through unchanged.
fn case_round_trip_strided(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for rows in ROW_COUNTS {
        let mut rng = Rng::new(0x7601 + rows as u64);
        let mut input = rng.vec(rows * IDX_W, -3.0, 3.0);
        for (i, s) in specials(true).into_iter().enumerate() {
            input[IDX_Q_W + i] = s;
            input[i] = s; // outside the rounded window: must stay untouched
        }
        let x = up_f32(gpu, &input)?;
        gpu.bf16_round_trip_f32_strided(&x, rows, IDX_Q_W, IDX_W, IDX_DIM)
            .map_err(e)?;
        sync(gpu)?;
        let mut want = input.clone();
        for r in 0..rows {
            for i in 0..IDX_DIM {
                let idx = r * IDX_W + IDX_Q_W + i;
                if want[idx].is_finite() {
                    want[idx] = bf16_trunc_rne(want[idx]);
                }
            }
        }
        checks.push(exact_f32(
            &format!("rows{rows}"),
            &down_f32(gpu, &x)?,
            &want,
        ));
        free_all(gpu, vec![x])?;
    }
    all(checks)
}

/// `copy_rows_strided_f32` as the route calls it to append the index key rows
/// to the F32 raw-key arena (`layer_ops.rs` `execute_indexed_attention_hc_impl`): src = a view at
/// offset 512 of the `[rows][640]` index batch, dst column offset
/// `position * 128`, vec4 fast path (everything 16-byte aligned).
fn case_copy_rows_strided(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let pos0 = 2790;
    for rows in ROW_COUNTS {
        let mut rng = Rng::new(0x7701 + rows as u64);
        let input = rng.vec(rows * IDX_W, -3.0, 3.0);
        let batch = up_f32(gpu, &input)?;
        let keys = batch.sub_offset(IDX_Q_W, rows * IDX_W - IDX_Q_W);
        let raw = nan_f32(gpu, (pos0 + rows) * IDX_DIM)?;
        gpu.copy_rows_strided_f32(
            &keys,
            &raw,
            rows,
            IDX_DIM,
            IDX_W,
            IDX_DIM,
            pos0 * IDX_DIM,
            Some(IDX_DIM),
        )
        .map_err(e)?;
        sync(gpu)?;
        let mut want = vec![f32::from_bits(0x7FC0_DEAD); (pos0 + rows) * IDX_DIM];
        for r in 0..rows {
            let src = r * IDX_W + IDX_Q_W;
            want[(pos0 + r) * IDX_DIM..(pos0 + r + 1) * IDX_DIM]
                .copy_from_slice(&input[src..src + IDX_DIM]);
        }
        checks.push(exact_f32(
            &format!("rows{rows}"),
            &down_f32(gpu, &raw)?,
            &want,
        ));
        free_all(gpu, vec![batch, raw])?; // `keys` is a view: not freed
    }
    all(checks)
}

// ── decode prologue ───────────────────────────────────────────────────────────
struct PrologueWant {
    index_row: Vec<f32>,
    qgate: Vec<f32>,
    keys: Vec<f32>,
    full_keys: Vec<f32>,
    full_values: Vec<f32>,
    raw: Vec<f32>,
}

/// `qsa_decode_prologue<QSA_KV_F32>` (`tensor_ops.hip:3252-3346`): the three
/// norm+RoPE stages, K/V cache append at `position + row`, index-k BF16 round
/// trip and raw-key copy.
#[allow(clippy::too_many_arguments)]
fn prologue_want(
    index_row: &[f32],
    qgate: &[f32],
    keys: &[f32],
    values: &[f32],
    iq_norm: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    rows: usize,
    position: usize,
    yarn: RopeYarn,
) -> PrologueWant {
    let mut w = PrologueWant {
        index_row: index_row.to_vec(),
        qgate: qgate.to_vec(),
        keys: keys.to_vec(),
        full_keys: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * KV_W],
        full_values: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * KV_W],
        raw: vec![f32::from_bits(0x7FC0_DEAD); (position + rows) * IDX_DIM],
    };
    for r in 0..rows {
        let pos = position + r;
        for h in 0..QSA_INDEX_HEADS {
            let s = r * IDX_W + h * IDX_DIM;
            let o = norm_rope_head(&index_row[s..s + IDX_DIM], iq_norm, pos, ROTARY, yarn);
            w.index_row[s..s + IDX_DIM].copy_from_slice(&o);
        }
        for h in 0..QSA_Q_HEADS {
            let s = r * QG_W + h * 2 * HD;
            let o = norm_rope_head(&qgate[s..s + HD], q_norm, pos, ROTARY, yarn);
            w.qgate[s..s + HD].copy_from_slice(&o);
        }
        for h in 0..QSA_KV_HEADS {
            let s = r * KV_W + h * HD;
            let o = norm_rope_head(&keys[s..s + HD], k_norm, pos, ROTARY, yarn);
            w.keys[s..s + HD].copy_from_slice(&o);
        }
        let rolled: Vec<f32> = w.keys[r * KV_W..(r + 1) * KV_W].to_vec();
        w.full_keys[pos * KV_W..(pos + 1) * KV_W].copy_from_slice(&rolled);
        w.full_values[pos * KV_W..(pos + 1) * KV_W]
            .copy_from_slice(&values[r * KV_W..(r + 1) * KV_W]);
        for i in 0..IDX_DIM {
            let idx = r * IDX_W + IDX_Q_W + i;
            let v = index_row[idx];
            let v = if v.is_finite() { bf16_trunc_rne(v) } else { v };
            w.index_row[idx] = v;
            w.raw[pos * IDX_DIM + i] = v;
        }
    }
    w
}

struct PrologueGpu {
    index_row: GpuTensor,
    qgate: GpuTensor,
    keys: GpuTensor,
    values: GpuTensor,
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw: GpuTensor,
}

impl PrologueGpu {
    fn new(gpu: &mut Gpu, i: &[Vec<f32>; 4], rows: usize, position: usize) -> Result<Self, String> {
        Ok(Self {
            index_row: up_f32(gpu, &i[0])?,
            qgate: up_f32(gpu, &i[1])?,
            keys: up_f32(gpu, &i[2])?,
            values: up_f32(gpu, &i[3])?,
            full_keys: nan_f32(gpu, (position + rows) * KV_W)?,
            full_values: nan_f32(gpu, (position + rows) * KV_W)?,
            raw: nan_f32(gpu, (position + rows) * IDX_DIM)?,
        })
    }
    fn free(self, gpu: &mut Gpu) -> Result<(), String> {
        free_all(
            gpu,
            vec![
                self.index_row,
                self.qgate,
                self.keys,
                self.values,
                self.full_keys,
                self.full_values,
                self.raw,
            ],
        )
    }
}

/// `indexed_attention_decode_prologue` (route: decode / verify rows <= 8, via
/// `layer_ops.rs` `execute_indexed_attention_hc_impl`; rows 33 runs the same kernel past the route's
/// `route_rows <= 8` gate, which on gfx942 prefill is always false because the
/// invariant profile pins `route_rows == 8192`), the append prologue
/// (`heads == 0`, MTP) and the bitwise claim "prologue == the six launches it
/// replaces" (norm_rope_batch x3 + round trip + copy_rows + cache_append_batch,
/// i.e. the prefill chain).
fn case_decode_prologue(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    let position = 3;
    for (yname, yarn) in yarns() {
        for rows in ROW_COUNTS {
            let mut rng = Rng::new(0x7801 + rows as u64);
            let iq_norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
            let q_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let k_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let inputs = [
                rng.vec(rows * IDX_W, -2.0, 2.0),
                rng.vec(rows * QG_W, -2.0, 2.0),
                rng.vec(rows * KV_W, -2.0, 2.0),
                rng.vec(rows * KV_W, -2.0, 2.0),
            ];
            let (iq_t, q_t, k_t) = (
                bf16_vec(gpu, &iq_norm)?,
                bf16_vec(gpu, &q_norm)?,
                bf16_vec(gpu, &k_norm)?,
            );
            let label = format!("{yname}/rows{rows}");

            // Fused decode prologue.
            let p = PrologueGpu::new(gpu, &inputs, rows, position)?;
            indexed_attention_decode_prologue(
                gpu,
                &IndexedAttentionDecodePrologue {
                    index_row: &p.index_row,
                    qgate: &p.qgate,
                    keys: &p.keys,
                    values: &p.values,
                    full_keys: &p.full_keys,
                    full_values: &p.full_values,
                    raw_index_keys: &p.raw,
                    index_q_norm: &iq_t,
                    q_norm: &q_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    heads: QSA_Q_HEADS,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: QsaKvFormat::F32,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let got_index = down_f32(gpu, &p.index_row)?;
            let got_qgate = down_f32(gpu, &p.qgate)?;
            let got_keys = down_f32(gpu, &p.keys)?;
            let got_fk = down_f32(gpu, &p.full_keys)?;
            let got_fv = down_f32(gpu, &p.full_values)?;
            let got_raw = down_f32(gpu, &p.raw)?;

            // CPU twin.
            let w = prologue_want(
                &inputs[0], &inputs[1], &inputs[2], &inputs[3], &iq_norm, &q_norm, &k_norm, rows,
                position, yarn,
            );
            checks.push(cmp_split(
                &format!("{label}/index_row"),
                &got_index,
                &w.index_row,
                F32_ACCUM,
                |i| i % IDX_W >= IDX_Q_W,
            ));
            checks.push(cmp_split(
                &format!("{label}/qgate"),
                &got_qgate,
                &w.qgate,
                F32_ACCUM,
                |i| i % (2 * HD) >= HD,
            ));
            checks.push(close_f32(
                &format!("{label}/keys"),
                &got_keys,
                &w.keys,
                F32_ACCUM,
            ));
            checks.push(cmp_split(
                &format!("{label}/full_keys"),
                &got_fk,
                &w.full_keys,
                F32_ACCUM,
                |i| i < position * KV_W,
            ));
            checks.push(exact_f32(
                &format!("{label}/full_values"),
                &got_fv,
                &w.full_values,
            ));
            checks.push(exact_f32(
                &format!("{label}/raw_index_keys"),
                &got_raw,
                &w.raw,
            ));

            // The prefill chain the prologue replaces (bitwise).
            let c = PrologueGpu::new(gpu, &inputs, rows, position)?;
            norm_rope_launch(
                gpu,
                &c.index_row,
                &iq_t,
                rows,
                position,
                yarn,
                QSA_INDEX_HEADS,
                IDX_DIM,
                IDX_DIM,
                IDX_W,
            )?;
            gpu.bf16_round_trip_f32_strided(&c.index_row, rows, IDX_Q_W, IDX_W, IDX_DIM)
                .map_err(e)?;
            let keys_view = c.index_row.sub_offset(IDX_Q_W, rows * IDX_W - IDX_Q_W);
            gpu.copy_rows_strided_f32(
                &keys_view,
                &c.raw,
                rows,
                IDX_DIM,
                IDX_W,
                IDX_DIM,
                position * IDX_DIM,
                Some(IDX_DIM),
            )
            .map_err(e)?;
            norm_rope_launch(
                gpu,
                &c.qgate,
                &q_t,
                rows,
                position,
                yarn,
                QSA_Q_HEADS,
                HD,
                2 * HD,
                QG_W,
            )?;
            norm_rope_launch(
                gpu,
                &c.keys,
                &k_t,
                rows,
                position,
                yarn,
                QSA_KV_HEADS,
                HD,
                HD,
                KV_W,
            )?;
            indexed_attention_cache_append_batch(
                gpu,
                &IndexedAttentionCacheAppendBatch {
                    key: &c.keys,
                    value: &c.values,
                    full_keys: &c.full_keys,
                    full_values: &c.full_values,
                    rows,
                    position_start: position,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    format: QsaKvFormat::F32,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(exact_f32(
                &format!("{label}/chain==prologue index_row"),
                &down_f32(gpu, &c.index_row)?,
                &got_index,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue qgate"),
                &down_f32(gpu, &c.qgate)?,
                &got_qgate,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue keys"),
                &down_f32(gpu, &c.keys)?,
                &got_keys,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue full_keys"),
                &down_f32(gpu, &c.full_keys)?,
                &got_fk,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue full_values"),
                &down_f32(gpu, &c.full_values)?,
                &got_fv,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue raw"),
                &down_f32(gpu, &c.raw)?,
                &got_raw,
            ));
            c.free(gpu)?;

            // Append prologue (heads == 0): same kernel, same bytes.
            let a = PrologueGpu::new(gpu, &inputs, rows, position)?;
            indexed_attention_append_prologue(
                gpu,
                &IndexedAttentionAppendPrologue {
                    index_row: &a.index_row,
                    keys: &a.keys,
                    values: &a.values,
                    full_keys: &a.full_keys,
                    full_values: &a.full_values,
                    raw_index_keys: &a.raw,
                    index_q_norm: &iq_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: QsaKvFormat::F32,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(exact_f32(
                &format!("{label}/append==decode index_row"),
                &down_f32(gpu, &a.index_row)?,
                &got_index,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode keys"),
                &down_f32(gpu, &a.keys)?,
                &got_keys,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode full_keys"),
                &down_f32(gpu, &a.full_keys)?,
                &got_fk,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode full_values"),
                &down_f32(gpu, &a.full_values)?,
                &got_fv,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode raw"),
                &down_f32(gpu, &a.raw)?,
                &got_raw,
            ));
            a.free(gpu)?;
            p.free(gpu)?;
            free_all(gpu, vec![iq_t, q_t, k_t])?;
        }
    }
    all(checks)
}

// ── pool + RoPE ───────────────────────────────────────────────────────────────
/// Shared body of the two `pool_rope` cases. Incremental launches (the route:
/// `execute_indexed_attention_hc_impl`) pool only blocks `[position_start/4, (position_start +
/// rows)/4)` and must leave every other pooled block untouched; the full
/// (non-incremental) launch pools all 700 blocks. Context 2800 tokens ⇒ 700
/// blocks > 512 budget blocks. Output is BF16-quantised, so the frozen
/// BF16_INPUT_F32_ACCUM tol applies (it also absorbs the device `rsqrtf` /
/// `powf` vs CPU differences at positions up to 2800).
fn pool_rope_case(gpu: &mut Gpu, bf16_arena: bool) -> Result<String, String> {
    let end = 2800usize;
    let blocks = end / QSA_COMPRESS;
    let mut rng = Rng::new(if bf16_arena { 0x7A02 } else { 0x7A01 });
    let raw = rng.vec_bf16_exact(end * IDX_DIM, -2.0, 2.0);
    let norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
    let norm_t = bf16_vec(gpu, &norm)?;
    let raw_t = if bf16_arena {
        bf16_vec(gpu, &raw)?
    } else {
        up_f32(gpu, &raw)?
    };
    let sentinel = if bf16_arena {
        f32::from_bits(0x7FC0_0000)
    } else {
        f32::from_bits(0x7FC0_DEAD)
    };
    let new_pooled = |gpu: &mut Gpu| -> Result<GpuTensor, String> {
        if bf16_arena {
            bf16_vec(gpu, &vec![f32::NAN; POOLED_CAP * IDX_DIM])
        } else {
            nan_f32(gpu, POOLED_CAP * IDX_DIM)
        }
    };
    let download = |gpu: &Gpu, t: &GpuTensor| -> Result<Vec<f32>, String> {
        if bf16_arena {
            Ok(down_bf16(gpu, t)?
                .into_iter()
                .map(bf16_bits_to_f32)
                .collect())
        } else {
            down_f32(gpu, t)
        }
    };
    let mut checks = Vec::new();
    for (yname, yarn) in yarns() {
        let block_row = |b: usize| {
            pool_rope_block(
                &raw[b * QSA_COMPRESS * IDX_DIM..(b + 1) * QSA_COMPRESS * IDX_DIM],
                &norm,
                b,
                yarn,
            )
        };
        for rows in ROW_COUNTS {
            let position_start = end - rows;
            let first = position_start / QSA_COMPRESS;
            let pooled = new_pooled(gpu)?;
            indexed_attention_pool_rope_incremental(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &raw_t,
                    pooled: &pooled,
                    norm: Some(&norm_t),
                    block_count: blocks,
                    compress: QSA_COMPRESS,
                    index_dim: IDX_DIM,
                    position: Some(QsaPositionBinding {
                        position_start,
                        rows,
                    }),
                    grid_bound: POOLED_CAP,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let mut want = vec![sentinel; POOLED_CAP * IDX_DIM];
            for b in first..blocks {
                want[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&block_row(b));
            }
            let got = download(gpu, &pooled)?;
            let (lo, hi) = (first * IDX_DIM, blocks * IDX_DIM);
            checks.push(cmp_split(
                &format!("{yname}/incremental rows{rows} (blocks {first}..{blocks})"),
                &got,
                &want,
                BF16_INPUT_F32_ACCUM,
                |i| i < lo || i >= hi,
            ));
            free_all(gpu, vec![pooled])?;
        }
        let pooled = new_pooled(gpu)?;
        indexed_attention_pool_rope(
            gpu,
            &IndexedAttentionPoolRope {
                raw_keys: &raw_t,
                pooled: &pooled,
                norm: Some(&norm_t),
                block_count: blocks,
                compress: QSA_COMPRESS,
                index_dim: IDX_DIM,
                position: None,
                grid_bound: blocks,
                yarn,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let mut want = vec![sentinel; POOLED_CAP * IDX_DIM];
        for b in 0..blocks {
            want[b * IDX_DIM..(b + 1) * IDX_DIM].copy_from_slice(&block_row(b));
        }
        let got = download(gpu, &pooled)?;
        let hi = blocks * IDX_DIM;
        checks.push(cmp_split(
            &format!("{yname}/full {blocks} blocks"),
            &got,
            &want,
            BF16_INPUT_F32_ACCUM,
            |i| i >= hi,
        ));
        if !bf16_arena {
            // F32 arena values are BF16-exact by construction.
            let bad = got[..hi].iter().filter(|&&v| v != bf16_round(v)).count();
            checks.push(if bad == 0 {
                Ok("F32 pooled values are BF16-exact".to_string())
            } else {
                Err(format!("{bad} pooled F32 values are not BF16-exact"))
            });
        }
        free_all(gpu, vec![pooled])?;
    }
    free_all(gpu, vec![raw_t, norm_t])?;
    all(checks)
}

/// `indexed_attention_pool_rope_f32` — the F32 arena the gfx942 route uses.
fn case_pool_rope_f32(gpu: &mut Gpu) -> Result<String, String> {
    pool_rope_case(gpu, false)
}

/// `indexed_attention_pool_rope_bf16` — BF16 arena (quantised-KV states only;
/// not reachable on gfx942).
fn case_pool_rope_bf16(gpu: &mut Gpu) -> Result<String, String> {
    pool_rope_case(gpu, true)
}

// ── parallel route: shared constants and helpers ─────────────────────────────
/// Select row counts: 1 and 8 take rows8 at decode, 9..=16 rows16, 17..=32 the
/// wide-verify rows32 tile (gfx1151) or rows16 groups; 33 = two full 16-row
/// groups plus a one-row tail (not a verify width: no one-row oracle); the
/// invariant prefill scope sends every count to rows16. 15/16/17/31/32 straddle
/// the 16- and 32-row tile edges.
const SEL_ROWS: [usize; 9] = [1, 8, 9, 15, 16, 17, 31, 32, 33];
/// Attention widths spanning the legacy hg4/hg12 thresholds and the wide-verify
/// gate: scope-off rows 2..=32 use the decode policy, 33 is past the gate, and
/// prefill scope keeps the legacy row-routed kernels.
const ATTN_ROWS: [usize; 9] = [1, 2, 7, 15, 16, 17, 31, 32, 33];
/// A budget past `select_from_scores`' 512-block cap: the launcher keeps the
/// fused selector, and the past-radix-list-capacity rank-everything path runs.
const FUSED_BUDGET_BLOCKS: usize = BUDGET_BLOCKS + 8;
/// An arena shape bound whose score row (4 B per block) plus the selector's
/// static LDS exceeds the 64 KiB LDS limit: the global-score mode.
const GLOBAL_SHAPE_BLOCKS: usize = 20_000;

fn cap_for(budget: usize) -> usize {
    budget * QSA_COMPRESS + QSA_COMPRESS - 1
}

/// Raw-bit equality with the `*_serial` HIP kernels is the gfx942 claim; other
/// architectures keep the CPU-twin tolerance checks only.
fn claims_serial_equality(gpu: &Gpu) -> bool {
    gpu.arch == "gfx942"
}

/// Run `f` with `Gpu::prefill_invariant_scope` set (gfx942 prefill pins it on,
/// decode / verify leave it off), restoring the prior value.
fn with_scope<T>(
    gpu: &mut Gpu,
    scope: bool,
    f: impl FnOnce(&mut Gpu) -> Result<T, String>,
) -> Result<T, String> {
    let prior = std::mem::replace(&mut gpu.prefill_invariant_scope, scope);
    let result = f(gpu);
    gpu.prefill_invariant_scope = prior;
    result
}

fn scope_name(scope: bool) -> &'static str {
    if scope {
        "prefill_scope"
    } else {
        "decode"
    }
}

// ── select (route) ────────────────────────────────────────────────────────────
/// The public launcher run exactly as `execute_indexed_attention_hc_impl` calls
/// it: `(selected, mirror, persisted)`.
fn public_select(
    gpu: &mut Gpu,
    r: &SelectRun,
    shape_blocks: usize,
) -> Result<(Vec<i32>, Vec<i32>, bool), String> {
    let selected = raw_i32(gpu, &vec![-7777; r.rows * r.capacity])?;
    let mirror = raw_i32(gpu, &vec![-7777; r.capacity])?;
    // A verify forward of two or more rows runs row-exact to the widest rows;
    // a one-row call is the decode itself.
    gpu.set_wide_verify_rows(if r.rows >= 2 { WIDE_ROWS } else { 0 });
    let launched = indexed_attention_select_batch_mirrored(
        gpu,
        &IndexedAttentionSelectBatch {
            query: r.query,
            pooled: r.pooled,
            pooled_bf16: None,
            pooled_bf16_valid_blocks: 0,
            selected: &selected,
            rows: r.rows,
            query_row_stride: IDX_W,
            block_count: r.blocks,
            index_heads: QSA_INDEX_HEADS,
            index_dim: IDX_DIM,
            budget_blocks: r.budget,
            compress: QSA_COMPRESS,
            position_start: r.position_start,
            capacity: r.capacity,
            shape_blocks,
        },
        &mirror,
    );
    gpu.set_wide_verify_rows(0);
    let persisted = launched.map_err(e)?;
    sync(gpu)?;
    let out = (
        down_i32(gpu, &selected)?,
        down_i32(gpu, &mirror)?,
        persisted,
    );
    free_all(gpu, vec![selected, mirror])?;
    Ok(out)
}

/// One public-select configuration: the budget, the arena shape bound the
/// launcher is handed, and the `prefill_invariant_scope` settings run.
struct SelectArm {
    budget: usize,
    shape_blocks: usize,
    scopes: &'static [bool],
}

/// F32 pooled keys: budget <= 512 takes the rows8/rows16 score kernels +
/// `select_from_scores` (scope off: 1 and 8 rows -> rows8, 9 and 33 -> rows16;
/// scope on: rows16 throughout); budget 520 keeps the fused selector, scores in
/// LDS and, with an oversized arena bound, in global rows.
const F32_ARMS: [SelectArm; 3] = [
    SelectArm {
        budget: BUDGET_BLOCKS,
        shape_blocks: POOLED_CAP,
        scopes: &[false, true],
    },
    SelectArm {
        budget: FUSED_BUDGET_BLOCKS,
        shape_blocks: POOLED_CAP,
        scopes: &[false],
    },
    SelectArm {
        budget: FUSED_BUDGET_BLOCKS,
        shape_blocks: GLOBAL_SHAPE_BLOCKS,
        scopes: &[false],
    },
];

/// BF16 pooled keys (FP8/Q8 KV states) never take the builder score pair off
/// gfx1151: the fused selector at both budgets, LDS and global scores.
const BF16_ARMS: [SelectArm; 4] = [
    SelectArm {
        budget: BUDGET_BLOCKS,
        shape_blocks: POOLED_CAP,
        scopes: &[false],
    },
    SelectArm {
        budget: BUDGET_BLOCKS,
        shape_blocks: GLOBAL_SHAPE_BLOCKS,
        scopes: &[false],
    },
    SelectArm {
        budget: FUSED_BUDGET_BLOCKS,
        shape_blocks: POOLED_CAP,
        scopes: &[false],
    },
    SelectArm {
        budget: FUSED_BUDGET_BLOCKS,
        shape_blocks: GLOBAL_SHAPE_BLOCKS,
        scopes: &[false],
    },
];

/// The kernel arm the gfx942 launcher takes for `arm` (label only; derived from
/// the launcher's predicates, not observed).
fn select_arm_name(bf16: bool, arm: &SelectArm, rows: usize, scope: bool) -> &'static str {
    if !bf16 && arm.budget <= BUDGET_BLOCKS {
        let route_rows = if scope {
            rdna_compute::prefill_invariant::QWEN4_PREFILL_CANON_ROWS
        } else {
            rows
        };
        return if route_rows > 8 {
            "rows16+from_scores"
        } else {
            "rows8+from_scores"
        };
    }
    if arm.shape_blocks * 4 + rdna_compute::tensor_ops::QSA_SELECT_BATCHED_STATIC_LDS_BYTES
        > 64 * 1024
    {
        "fused_global_scores"
    } else {
        "fused_lds_scores"
    }
}

/// Public `indexed_attention_select_batch_mirrored`, exactly as
/// `execute_indexed_attention_hc_impl` calls it, across the gfx942 parallel arms
/// (`F32_ARMS` / `BF16_ARMS`) and both `prefill_invariant_scope` settings.
/// Contexts: 2802 tokens (700 blocks > the budget; the last row's boundary
/// sits in a tie group; partial 2-token tail) and 1602 tokens (400 blocks:
/// every block selected). Per data set:
/// - plain (dyadic): selection and the final-row mirror bit-exact against the
///   stable (score desc, block asc) CPU twin (exact regardless of FMA contraction);
/// - realistic (non-dyadic wide-exponent, cancelling, forced lowest-index ties,
///   gfx942 claim): selection raw-int equal to `*_batched_serial` HIP and to
///   the CPU `mul_add` twin.
/// On gfx942 the mirror must also be persisted by the launcher.
fn select_route(gpu: &mut Gpu, bf16: bool) -> Result<String, String> {
    let claim = claims_serial_equality(gpu);
    let arms: &[SelectArm] = if bf16 { &BF16_ARMS } else { &F32_ARMS };
    let mut checks = Vec::new();
    for end in [2802usize, 1602] {
        let blocks = end / QSA_COMPRESS;
        for rows in SEL_ROWS {
            let position_start = end - rows;
            for kind in [DataKind::Plain, DataKind::Realistic] {
                if kind == DataKind::Realistic && !claim {
                    continue;
                }
                for budget in [BUDGET_BLOCKS, FUSED_BUDGET_BLOCKS] {
                    // With every block selected the budgets give the same bytes.
                    if budget > BUDGET_BLOCKS && blocks <= BUDGET_BLOCKS {
                        continue;
                    }
                    let capacity = cap_for(budget);
                    let seed = 0x7B00 + end as u64 + rows as u64 + budget as u64 + kind.seed();
                    let (query, pooled) =
                        gen_select_data_kind(seed, rows, blocks, budget, kind, bf16)?;
                    let (want, _) = select_rows_cpu_with(
                        index_score_of(kind),
                        &query,
                        &pooled,
                        rows,
                        blocks,
                        position_start,
                        budget,
                        capacity,
                    );
                    let query_t = up_f32(gpu, &query)?;
                    let pooled_t = if bf16 {
                        bf16_vec(gpu, &pooled)?
                    } else {
                        up_f32(gpu, &pooled)?
                    };
                    let run = SelectRun {
                        query: &query_t,
                        pooled: &pooled_t,
                        rows,
                        blocks,
                        position_start,
                        budget,
                        capacity,
                    };
                    let serial = if claim && kind == DataKind::Realistic {
                        Some(serial_select(gpu, &run, bf16)?)
                    } else {
                        None
                    };
                    for arm in arms.iter().filter(|a| a.budget == budget) {
                        for &scope in arm.scopes {
                            let (selected, mirror, persisted) = with_scope(gpu, scope, |g| {
                                public_select(g, &run, arm.shape_blocks)
                            })?;
                            let label = format!(
                                "{}/end{end}/rows{rows}/b{budget}/shape{}/{}/{}",
                                kind.name(),
                                arm.shape_blocks,
                                scope_name(scope),
                                select_arm_name(bf16, arm, rows, scope)
                            );
                            checks.push(exact_ints(
                                &format!("{label}/selected=={}", kind.twin_name()),
                                &selected,
                                &want,
                            ));
                            if let Some(serial) = &serial {
                                checks.push(exact_ints(
                                    &format!("{label}/selected==serial_hip"),
                                    &selected,
                                    serial,
                                ));
                            }
                            if claim && !persisted {
                                checks.push(Err(format!(
                                    "{label}: launcher did not persist the mirror on gfx942"
                                )));
                            }
                            let want_mirror = if persisted {
                                want[(rows - 1) * capacity..].to_vec()
                            } else {
                                vec![-7777; capacity]
                            };
                            checks.push(exact_ints(
                                &format!("{label}/mirror(persisted={persisted})"),
                                &mirror,
                                &want_mirror,
                            ));
                            // Wide verify (exact gfx1151 / gfx1201): the batch's selection is
                            // bitwise the n one-row public calls at `position_start + r`.
                            if !scope && rows <= WIDE_ROWS && wide_arch(gpu) {
                                let mut one = Vec::with_capacity(rows * capacity);
                                for r in 0..rows {
                                    let row_t = up_f32(gpu, &query[r * IDX_W..(r + 1) * IDX_W])?;
                                    let run1 = SelectRun {
                                        query: &row_t,
                                        pooled: run.pooled,
                                        rows: 1,
                                        blocks: (position_start + r + 1) / QSA_COMPRESS,
                                        position_start: position_start + r,
                                        budget,
                                        capacity,
                                    };
                                    let (s1, _, _) = public_select(gpu, &run1, arm.shape_blocks)?;
                                    one.extend(s1);
                                    free_all(gpu, vec![row_t])?;
                                }
                                checks.push(exact_ints(
                                    &format!("{label}/selected==one-row public calls"),
                                    &selected,
                                    &one,
                                ));
                            }
                        }
                    }
                    free_all(gpu, vec![query_t, pooled_t])?;
                }
            }
        }
    }
    all(checks)
}

fn case_select_f32(gpu: &mut Gpu) -> Result<String, String> {
    select_route(gpu, false)
}

/// BF16 pooled arena (the explicit-FP8 K/V state's index keys on gfx1201/RDNA;
/// gfx942's FNUZ FP8 keeps F32 index arenas, so here the launcher is run
/// directly on a BF16 arena): the fused selector.
fn case_select_bf16(gpu: &mut Gpu) -> Result<String, String> {
    select_route(gpu, true)
}

// ── attention (route) ─────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
struct AttnGeom {
    name: &'static str,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

const QWEN4_ATTN: AttnGeom = AttnGeom {
    name: "qwen4_24x2x256",
    heads: QSA_Q_HEADS,
    kv_heads: QSA_KV_HEADS,
    head_dim: HD,
};

impl AttnGeom {
    fn out_w(self) -> usize {
        self.heads * self.head_dim
    }
}

struct AttnTensors {
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    sel: GpuTensor,
}

fn upload_attn(
    gpu: &mut Gpu,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[i32],
) -> Result<AttnTensors, String> {
    Ok(AttnTensors {
        q: up_f32(gpu, q)?,
        k: up_f32(gpu, keys)?,
        v: up_f32(gpu, values)?,
        sel: raw_i32(gpu, selected)?,
    })
}

fn free_attn(gpu: &mut Gpu, t: AttnTensors) -> Result<(), String> {
    free_all(gpu, vec![t.q, t.k, t.v, t.sel])
}

/// Public `indexed_attention_attention_batch_with_arena(.., None)` on F32 K/V.
fn attn_public(
    gpu: &mut Gpu,
    t: &AttnTensors,
    g: AttnGeom,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    attn_public_parts(gpu, &t.q, &t.k, &t.v, &t.sel, g, rows, position_start, end)
}

/// [`attn_public`] over separate tensors (one-row calls reuse the K/V of a batch).
#[allow(clippy::too_many_arguments)]
fn attn_public_parts(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    sel: &GpuTensor,
    g: AttnGeom,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    let out = nan_f32(gpu, rows * g.out_w())?;
    // A verify forward of two or more rows runs row-exact to the widest rows;
    // a one-row call is the decode itself.
    gpu.set_wide_verify_rows(if rows >= 2 { WIDE_ROWS } else { 0 });
    let launched = indexed_attention_attention_batch_with_arena(
        gpu,
        &IndexedAttentionAttentionBatch {
            q_with_gate: q,
            full_keys: k,
            full_values: v,
            selected: sel,
            output: &out,
            rows,
            position_start,
            n_heads: g.heads,
            n_kv_heads: g.kv_heads,
            head_dim: g.head_dim,
            budget_blocks: BUDGET_BLOCKS,
            compress: QSA_COMPRESS,
            capacity: SEL_CAP,
            full_capacity: end,
            format: QsaKvFormat::F32,
            shape_selected: SEL_CAP,
        },
        None,
    );
    gpu.set_wide_verify_rows(0);
    launched.map_err(e)?;
    sync(gpu)?;
    let got = down_f32(gpu, &out)?;
    free_all(gpu, vec![out])?;
    Ok(got)
}

/// Direct launch of one batched F32 attention kernel (all share one kernarg
/// layout: q, keys, values, selected, out, rows, position_start, heads, kv
/// heads, head_dim, budget, compress, capacity, full_capacity).
#[allow(clippy::too_many_arguments)]
fn attn_direct(
    gpu: &mut Gpu,
    sym: &str,
    grid: [u32; 3],
    shared: u32,
    t: &AttnTensors,
    g: AttnGeom,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    direct_load(gpu, sym)?;
    let out = nan_f32(gpu, rows * g.out_w())?;
    let mut a = hip_bridge::KernargBlob::new();
    for tensor in [&t.q, &t.k, &t.v, &t.sel, &out] {
        a.push_ptr(tensor.buf.as_ptr());
    }
    for x in [
        rows,
        position_start,
        g.heads,
        g.kv_heads,
        g.head_dim,
        BUDGET_BLOCKS,
        QSA_COMPRESS,
        SEL_CAP,
        end,
    ] {
        a.push_i32(x as i32);
    }
    direct_launch(gpu, sym, grid, [256, 1, 1], shared, a)?;
    sync(gpu)?;
    let got = down_f32(gpu, &out)?;
    free_all(gpu, vec![out])?;
    Ok(got)
}

/// `indexed_attention_attention_f32_batched_serial` (grid heads x ceil(hd/256)
/// x rows): the per-channel baseline the parallel kernels claim to equal bit
/// for bit.
fn attn_serial(
    gpu: &mut Gpu,
    t: &AttnTensors,
    g: AttnGeom,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<Vec<f32>, String> {
    attn_direct(
        gpu,
        "indexed_attention_attention_f32_batched_serial",
        [g.heads as u32, g.head_dim.div_ceil(256) as u32, rows as u32],
        0,
        t,
        g,
        rows,
        position_start,
        end,
    )
}

/// gfx942: raw F32 bits (a -0 / +0 or one-ulp difference fails); elsewhere the
/// frozen accumulation tolerance.
fn attn_claim(gpu: &Gpu, label: &str, got: &[f32], serial: &[f32]) -> Result<String, String> {
    if claims_serial_equality(gpu) {
        exact_f32(label, got, serial)
    } else {
        close_f32(label, got, serial, F32_ACCUM)
    }
}

/// Public `indexed_attention_attention_batch_with_arena(.., None)` as the route
/// calls it (`execute_indexed_attention_hc_impl`), under both
/// `prefill_invariant_scope` settings. Decode (scope off) with 2..=`WIDE_ROWS`
/// rows opens the verify gate (`attn_public` / `with_wide_gate`): an admitted
/// decode call takes ONE implementation for every row count (split-K on eager
/// gfx1151 F32 hd256 group-12 layouts, else the per-head kernels; never hg4 /
/// hg12), so those batches must equal the n one-row public calls at
/// `position_start + r` raw F32 bit for bit on every architecture
/// (`attn_one_row_calls`); the arm label comes from `attn_arm`. Gate-closed
/// calls keep the legacy checks: one row (decode), 33 rows (past the limit:
/// hg12 for this geometry) and the prefill scope, which routes every count as
/// 8192 rows (hg12). 2802 ctx => 700 blocks, 512 chosen + tail => selected_len
/// 2048..2050 (<= the 2304-row hg12 tile budget); 402 ctx keeps every block.
/// Row 0 carries injected `-1` / not-yet-visible entries the kernels must skip.
/// Per data set (uniform, realistic), always: `serial_hip_vs_cpu`
/// (`*_batched_serial` HIP vs the CPU twin at F32_ACCUM; the twin sums in the
/// kernels' d / selected-row order, `expf` and contraction differ in the last
/// ulps). Gate-closed public calls: the CPU twin at F32_ACCUM, and on gfx942
/// raw F32 bits equal to the serial HIP.
fn case_attention_batch(gpu: &mut Gpu) -> Result<String, String> {
    let g = QWEN4_ATTN;
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for kind in [DataKind::Plain, DataKind::Realistic] {
            for rows in ATTN_ROWS {
                let inp = attn_inputs_kind(0x7C00 + end as u64 + rows as u64, rows, end, kind)?;
                let t = upload_attn(gpu, &inp.q, &inp.keys, &inp.values, &inp.selected)?;
                let serial = attn_serial(gpu, &t, g, rows, inp.position_start, end)?;
                checks.push(close_f32(
                    &format!("{}/end{end}/rows{rows}/serial_hip_vs_cpu", kind.name()),
                    &serial,
                    &inp.want,
                    F32_ACCUM,
                ));
                for scope in [false, true] {
                    let got = with_scope(gpu, scope, |gp| {
                        attn_public(gp, &t, g, rows, inp.position_start, end)
                    })?;
                    let label = format!(
                        "{}/end{end}/rows{rows}/{}/{}",
                        kind.name(),
                        scope_name(scope),
                        attn_arm(gpu, g, rows, scope, end)?
                    );
                    if !scope && (2..=WIDE_ROWS).contains(&rows) {
                        let one = attn_one_row_calls(
                            gpu,
                            &t,
                            &inp.q,
                            &inp.selected,
                            g,
                            rows,
                            inp.position_start,
                            end,
                        )?;
                        checks.push(exact_f32(
                            &format!("{label}/bitwise_one_row_public"),
                            &got,
                            &one,
                        ));
                    } else {
                        checks.push(close_f32(
                            &format!("{label}/vs_cpu"),
                            &got,
                            &inp.want,
                            F32_ACCUM,
                        ));
                        checks.push(attn_claim(
                            gpu,
                            &format!("{label}/vs_serial_hip"),
                            &got,
                            &serial,
                        ));
                    }
                }
                free_attn(gpu, t)?;
            }
        }
    }
    all(checks)
}

/// The same public route over other geometries. Open-gate decode (scope off,
/// 2..=`WIDE_ROWS` rows: the verify gate `with_wide_gate` / `attn_public` opens)
/// is one admitted decode implementation (split-K or per-head, never hg4 / hg12)
/// whose n-row launch must be raw-F32 equal to the n one-row public calls
/// (`attn_one_row_calls`) on every architecture. Gate-closed calls (1 row, 33
/// rows, the prefill scope) keep the legacy check against `*_batched_serial`
/// HIP (the CPU twin is Qwen4-shaped): Qwen4 24x2x256 (hg4 / hg12 / split-K),
/// 8x2x256 (head groups of 4: hg4 only, never hg12), 6x2x256 (group of 3: no
/// grouped kernel) and 24x2x128 (head_dim != 256: no grouped kernel). Realistic
/// data; 2802 ctx; both scope settings.
fn case_attention_batch_geometry(gpu: &mut Gpu) -> Result<String, String> {
    let end = 2802usize;
    let geoms = [
        QWEN4_ATTN,
        AttnGeom {
            name: "group4_8x2x256",
            heads: 8,
            kv_heads: 2,
            head_dim: 256,
        },
        AttnGeom {
            name: "group3_6x2x256",
            heads: 6,
            kv_heads: 2,
            head_dim: 256,
        },
        AttnGeom {
            name: "hd128_24x2x128",
            heads: 24,
            kv_heads: 2,
            head_dim: 128,
        },
    ];
    let mut checks = Vec::new();
    for g in geoms {
        for rows in [1usize, 7, 16, 33] {
            let seed =
                0x7E00 + end as u64 + rows as u64 + g.heads as u64 * 0x100 + g.head_dim as u64;
            let position_start = end - rows;
            let selected = attn_selection(seed, rows, end)?;
            let (q, keys, values) =
                realistic_attn_qkv(seed, rows, end, g.heads, g.kv_heads, g.head_dim);
            let t = upload_attn(gpu, &q, &keys, &values, &selected)?;
            let serial = attn_serial(gpu, &t, g, rows, position_start, end)?;
            if !serial.iter().all(|v| v.is_finite()) {
                checks.push(Err(format!(
                    "{}/rows{rows}: serial HIP output not finite",
                    g.name
                )));
            }
            for scope in [false, true] {
                let got = with_scope(gpu, scope, |gp| {
                    attn_public(gp, &t, g, rows, position_start, end)
                })?;
                let label = format!(
                    "{}/rows{rows}/{}/{}",
                    g.name,
                    scope_name(scope),
                    attn_arm(gpu, g, rows, scope, end)?
                );
                if !scope && (2..=WIDE_ROWS).contains(&rows) {
                    let one =
                        attn_one_row_calls(gpu, &t, &q, &selected, g, rows, position_start, end)?;
                    checks.push(exact_f32(
                        &format!("{label}/bitwise_one_row_public"),
                        &got,
                        &one,
                    ));
                } else {
                    checks.push(attn_claim(
                        gpu,
                        &format!("{label}/vs_serial_hip"),
                        &got,
                        &serial,
                    ));
                }
            }
            free_attn(gpu, t)?;
        }
    }
    all(checks)
}

// ── reuse selection + single-row attention (MTP route) ───────────────────────
struct ReuseScenario {
    name: &'static str,
    prior: Vec<i32>,
    selected_len: usize,
    position: usize,
}

fn reuse_scenarios() -> Vec<ReuseScenario> {
    let mut rng = Rng::new(0x7D01);
    let position = 3000usize;
    let mut a: Vec<i32> = (0..600)
        .map(|_| rng.below(position + 10) as i32 - 1)
        .collect();
    a.retain(|&t| t != position as i32);
    a[5] = position as i32 + 4; // above position: dropped
    let a_len = a.len();
    let mut b: Vec<i32> = (0..500).map(|_| rng.below(position - 1) as i32).collect();
    b[123] = position as i32; // already present: not appended
    b[7] = -1;
    let b_len = b.len();
    let c: Vec<i32> = (0..SEL_CAP)
        .map(|i| (i as i32 * 3) % position as i32)
        .collect();
    vec![
        ReuseScenario {
            name: "append_current",
            prior: a,
            selected_len: a_len,
            position,
        },
        ReuseScenario {
            name: "current_present",
            prior: b,
            selected_len: b_len,
            position,
        },
        ReuseScenario {
            name: "full_capacity_no_room",
            prior: c,
            selected_len: SEL_CAP,
            position,
        },
        ReuseScenario {
            name: "empty_prior",
            prior: vec![],
            selected_len: 0,
            position,
        },
    ]
}

fn reuse_want(s: &ReuseScenario) -> (Vec<i32>, i32) {
    let n = s.selected_len.min(SEL_CAP);
    let mut kept: Vec<i32> = s.prior[..n]
        .iter()
        .copied()
        .filter(|&t| t >= 0 && t <= s.position as i32)
        .collect();
    if !kept.contains(&(s.position as i32)) && kept.len() < SEL_CAP {
        kept.push(s.position as i32);
    }
    let count = kept.len() as i32;
    kept.resize(SEL_CAP, -1);
    (kept, count)
}

/// `indexed_attention_reuse_selection` (`tensor_ops.hip:4280`): stable
/// in-place compaction (keep `0 <= t <= position`, append `position` unless
/// present or full), `-1` fill, length scalar. Width-32 `__shfl_up` scan inside
/// 8 software waves of a 256-thread block.
fn case_reuse_selection(gpu: &mut Gpu) -> Result<String, String> {
    let mut checks = Vec::new();
    for s in reuse_scenarios() {
        let mut buf = s.prior.clone();
        buf.resize(SEL_CAP, 12345); // beyond selected_len: garbage the kernel must overwrite
        let selected = raw_i32(gpu, &buf)?;
        let len_out = raw_i32(gpu, &[-9999])?;
        indexed_attention_reuse_selection(
            gpu,
            &IndexedAttentionReuseSelection {
                selected: &selected,
                selected_len: s.selected_len,
                position: s.position,
                capacity: SEL_CAP,
                selected_len_out: &len_out,
            },
        )
        .map_err(e)?;
        sync(gpu)?;
        let (want, want_len) = reuse_want(&s);
        checks.push(exact_ints(
            &format!("{}/selected", s.name),
            &down_i32(gpu, &selected)?,
            &want,
        ));
        checks.push(exact_ints(
            &format!("{}/len", s.name),
            &down_i32(gpu, &len_out)?,
            &[want_len][..],
        ));
        free_all(gpu, vec![selected, len_out])?;
    }
    all(checks)
}

/// `indexed_attention_attention` (single query row, explicit `selected_len`;
/// the MTP reuse route in `execute_indexed_attention_hc_impl`): on gfx942
/// `indexed_attention_attention_f32_serial`. Selection = scenario A of the
/// reuse case after compaction (arbitrary token order, `-1` fill beyond
/// `selected_len`), cache of 3011 rows.
fn case_attention_single(gpu: &mut Gpu) -> Result<String, String> {
    let scenarios = reuse_scenarios();
    let s = &scenarios[0];
    let (sel, count) = reuse_want(s);
    let full_capacity = s.position + 11;
    let mut rng = Rng::new(0x7E01);
    let q = rng.vec(QG_W, -3.0, 3.0);
    let keys = rng.vec(full_capacity * KV_W, -1.0, 1.0);
    let values = rng.vec(full_capacity * KV_W, -1.0, 1.0);
    let want = attention_row_cpu(
        &q,
        &keys,
        &values,
        &sel[..count as usize],
        full_capacity,
        full_capacity,
    );
    let (q_t, k_t, v_t) = (up_f32(gpu, &q)?, up_f32(gpu, &keys)?, up_f32(gpu, &values)?);
    let sel_t = raw_i32(gpu, &sel)?;
    let out = nan_f32(gpu, Q_W)?;
    indexed_attention_attention(
        gpu,
        &IndexedAttentionAttention {
            q_with_gate: &q_t,
            full_keys: &k_t,
            full_values: &v_t,
            selected: &sel_t,
            output: &out,
            n_heads: QSA_Q_HEADS,
            n_kv_heads: QSA_KV_HEADS,
            head_dim: HD,
            selected_len: count as usize,
            full_capacity,
            end_position: s.position + 1,
            format: QsaKvFormat::F32,
        },
    )
    .map_err(e)?;
    sync(gpu)?;
    let r = close_f32(
        "single-row attention",
        &down_f32(gpu, &out)?,
        &want,
        F32_ACCUM,
    );
    free_all(gpu, vec![q_t, k_t, v_t, sel_t, out])?;
    r
}

// ── FP8 (gfx942, `QsaKvFormat::Fp8Fnuz`) K/V ─────────────────────────────────
// Row layout (Qwen4: 2 KV heads x 256): `kv_heads * head_dim` OCP e4m3fn code
// bytes, then `kv_heads` f16 scales (little endian) = 516 bytes per K or V row.
// The gfx942 hardware cvt is FNUZ (0x80 is NaN there), so the stored bytes are
// the OCP e4m3 RNE bytes of the gfx1201 encoder (amax f16 scale, amax/448 with
// the one-ulp bump) with ONE sanctioned deviation: every code the OCP encoder
// would emit as 0x80 (-0.0 input AND negative values that underflow to zero)
// is stored as canonical +0 (0x00); 0x80 is NEVER stored. The decoder is the
// OCP decode of the stored bytes, so a decoded zero may differ from the OCP
// decode only in the sign of zero. The index/pooled arenas of this format are
// F32 (same as the F32 state). Case-name filter prefix: `qwen4_qsa_fp8_fnuz`.
const FP8_FMT: QsaKvFormat = QsaKvFormat::Fp8Fnuz;
const FP8_ROW: usize = QSA_KV_HEADS * (HD + 2); // 516

fn fp8_skip(gpu: &Gpu) -> Option<String> {
    (gpu.arch != "gfx942").then(|| format!("arch {} != gfx942: Fp8Fnuz cases skipped", gpu.arch))
}

/// f32 -> f16 bits, round-to-nearest-even (what `(_Float16)x` does).
fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let a = x & 0x7FFF_FFFF;
    if a >= 0x7F80_0000 {
        return sign | 0x7C00 | if a > 0x7F80_0000 { 0x200 } else { 0 };
    }
    if a >= 0x477F_F000 {
        return sign | 0x7C00; // >= 65520 rounds to inf
    }
    let (h, rem, half) = if a >= 0x3880_0000 {
        let r = a - 0x3800_0000;
        (r >> 13, r & 0x1FFF, 0x1000u32)
    } else {
        let exp = a >> 23;
        if exp < 102 {
            return sign; // below half the smallest subnormal
        }
        let mant = (a & 0x7F_FFFF) | 0x80_0000;
        let shift = 126 - exp;
        (
            mant >> shift,
            mant & ((1 << shift) - 1),
            1u32 << (shift - 1),
        )
    };
    let h = if rem > half || (rem == half && (h & 1) == 1) {
        h + 1
    } else {
        h
    };
    sign | h as u16
}

fn f16_bits_to_f32(b: u16) -> f32 {
    let sign = ((b as u32) & 0x8000) << 16;
    let e = ((b >> 10) & 31) as u32;
    let m = (b & 0x3FF) as u32;
    if e == 0 {
        let mag = m as f32 * f32::from_bits(0x3380_0000); // 2^-24
        return if sign != 0 { -mag } else { mag };
    }
    if e == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (m << 13));
    }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

/// OCP e4m3fn decode; 0x80 is -0.0, 0x7F/0xFF are NaN.
fn e4m3_decode(code: u8) -> f32 {
    let e = ((code >> 3) & 15) as u32;
    let m = (code & 7) as u32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m as f32 * f32::from_bits(0x3B00_0000) // 2^-9
    } else {
        (1.0 + m as f32 / 8.0) * f32::from_bits((e + 120) << 23) // 2^(e-7)
    };
    if code & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

fn rne_u32(t: f32) -> u32 {
    let fl = t.floor();
    let d = t - fl;
    let r = fl as u32;
    if d > 0.5 || (d == 0.5 && (r & 1) == 1) {
        r + 1
    } else {
        r
    }
}

/// OCP e4m3fn round-to-nearest-even encode of a finite f32; the sign comes
/// from the f32 sign bit (so a negative value that underflows is 0x80, and
/// -0.0 is 0x80). Magnitudes above 448 saturate to 0x7E (unreachable after the
/// scale bump).
fn e4m3_encode(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7F;
    }
    if a > 448.0 {
        return sign | 0x7E;
    }
    let code = if a < f32::from_bits(0x3C80_0000) {
        // < 2^-6: subnormal region, step 2^-9 (q == 8 is the min normal 0x08).
        rne_u32(a * 512.0) as u8
    } else {
        let e = (a.to_bits() >> 23) as i32 - 127;
        let t = a * f32::from_bits(((127 + 3 - e) as u32) << 23); // [8, 16)
        let mut q = rne_u32(t) as i32;
        let mut e = e;
        if q == 16 {
            q = 8;
            e += 1;
        }
        let ec = e + 7;
        if ec > 15 || (ec == 15 && q - 8 == 7) {
            0x7E
        } else {
            ((ec as u8) << 3) | (q - 8) as u8
        }
    };
    sign | code
}

/// The stored code of an OCP code on gfx942: 0x80 (-0) becomes +0.
fn fp8_fold(code: u8) -> u8 {
    if code == 0x80 {
        0x00
    } else {
        code
    }
}

/// Head scale as `qsa_kv_store` computes it: `(_Float16)(amax / 448)`, 1.0 for
/// an all-zero head, the smallest subnormal if it underflows, bumped by one
/// f16 ulp if `448 * scale < amax`. Returns `(f16 bits, f32 scale)`.
fn fp8_scale(amax: f32) -> (u16, f32) {
    let mut bits = if amax == 0.0 {
        0x3C00
    } else {
        f32_to_f16_bits(amax / 448.0)
    };
    if amax > 0.0 && bits == 0 {
        bits = 1;
    }
    let mut sf = f16_bits_to_f32(bits);
    if amax > 0.0 && 448.0 * sf < amax {
        bits = if bits >= 0x7BFF { 0x7BFF } else { bits + 1 };
        sf = f16_bits_to_f32(bits);
    }
    (bits, sf)
}

/// CPU twin of the K or V row store, OCP bytes (0x80 possible): `vals` =
/// `rows * KV_W` f32.
fn fp8_encode_rows(vals: &[f32]) -> Vec<u8> {
    assert_eq!(vals.len() % KV_W, 0);
    let rows = vals.len() / KV_W;
    let mut out = vec![0u8; rows * FP8_ROW];
    for r in 0..rows {
        for h in 0..QSA_KV_HEADS {
            let head = &vals[r * KV_W + h * HD..r * KV_W + (h + 1) * HD];
            let amax = head.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (bits, sf) = fp8_scale(amax);
            for c in 0..HD {
                out[r * FP8_ROW + h * HD + c] = e4m3_encode(head[c] / sf);
            }
            out[r * FP8_ROW + KV_W + 2 * h..r * FP8_ROW + KV_W + 2 * h + 2]
                .copy_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

/// Stored form of OCP rows: every code byte folded (0x80 -> 0x00), scales kept.
fn fp8_fold_rows(bytes: &[u8]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for r in 0..bytes.len() / FP8_ROW {
        for i in 0..KV_W {
            out[r * FP8_ROW + i] = fp8_fold(out[r * FP8_ROW + i]);
        }
    }
    out
}

/// CPU OCP decode of rows: `f32(e4m3(code)) * f32(f16 scale)`.
fn fp8_decode_rows(bytes: &[u8]) -> Vec<f32> {
    let rows = bytes.len() / FP8_ROW;
    let mut out = vec![0.0f32; rows * KV_W];
    for r in 0..rows {
        for h in 0..QSA_KV_HEADS {
            let o = r * FP8_ROW + KV_W + 2 * h;
            let sf = f16_bits_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]));
            for c in 0..HD {
                out[r * KV_W + h * HD + c] = e4m3_decode(bytes[r * FP8_ROW + h * HD + c]) * sf;
            }
        }
    }
    out
}

fn cycle_to(data: &[f32], n: usize) -> Vec<f32> {
    (0..n).map(|i| data[i % data.len()]).collect()
}

/// One random head: uniform values plus a log-spread population (reaches the
/// subnormal codes) with an element equal to +-amax.
fn fp8_rand_head(rng: &mut Rng, amax: f32) -> Vec<f32> {
    let mut h: Vec<f32> = (0..HD)
        .map(|_| match rng.below(3) {
            0 => amax * rng.range(-1.0, 1.0) * 0.5f32.powi(rng.below(14) as i32),
            _ => amax * rng.range(-1.0, 1.0),
        })
        .collect();
    let at = rng.below(HD);
    h[at] = if rng.below(2) == 0 { amax } else { -amax };
    h
}

/// Dataset heads: every finite rounding boundary at scale exactly 1.0 (amax
/// 448): each adjacent-code midpoint (a tie, so RNE), its f32 neighbours, every
/// code value, both signs, plus +-0, -0.0, negative underflow; then all-zero
/// and all -0.0 heads and random heads over a palette of non-power-of-two /
/// extreme amax (f16 scale underflow, scale bump, large scales).
fn fp8_heads(seed: u64, flip: bool) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let mut vals: Vec<f32> = Vec::new();
    for c in 0u8..0x7E {
        let mid = (e4m3_decode(c) + e4m3_decode(c + 1)) * 0.5;
        for v in [
            mid,
            f32::from_bits(mid.to_bits() + 1),
            f32::from_bits(mid.to_bits() - 1),
        ] {
            vals.extend([v, -v]);
        }
    }
    for c in 0u8..=0x7E {
        let v = e4m3_decode(c);
        vals.extend([v, -v]);
    }
    vals.extend([
        1e-30,
        -1e-30,
        1e-38,
        -1e-38,
        f32::from_bits(1),
        -f32::from_bits(1),
    ]);
    let mut heads: Vec<Vec<f32>> = Vec::new();
    for (i, chunk) in vals.chunks(HD - 3).enumerate() {
        let mut h = vec![if i % 2 == 0 { 448.0 } else { -448.0 }, -0.0, 0.0];
        h.extend_from_slice(chunk);
        h.resize(HD, 0.0);
        heads.push(h);
    }
    heads.push(vec![0.0; HD]);
    heads.push(vec![-0.0; HD]);
    let palette = [
        1e-9f32,
        1e-8,
        3e-5,
        1.0 / 3.0,
        0.7,
        5.3,
        17.0,
        448.0,
        1000.0,
        3.0e4,
        2.5e5,
        448.0 * 6.103_515_6e-5,
        448.0 * 5.960_464_5e-8,
        123.456,
        0.0123,
    ];
    for round in 0..3 {
        for &a in &palette {
            let amax = if round == 0 {
                a
            } else {
                a * (0.75 + 0.5 * rng.unit())
            };
            heads.push(fp8_rand_head(&mut rng, amax));
        }
    }
    if heads.len() % QSA_KV_HEADS != 0 {
        heads.push(fp8_rand_head(&mut rng, 2.0));
    }
    if flip {
        heads.reverse();
    }
    heads
}

fn fp8_data(seed: u64, flip: bool) -> Vec<f32> {
    fp8_heads(seed, flip).into_iter().flatten().collect()
}

fn fp8_cache(gpu: &mut Gpu, rows: usize) -> Result<GpuTensor, String> {
    let bytes = vec![0xA5u8; rows * FP8_ROW];
    gpu.upload_raw(&bytes, &[bytes.len()]).map_err(e)
}

/// Split of the sanctioned OCP-0x80 -> stored-0x00 deviation:
/// `(actual -0.0 inputs, negative nonzero inputs that underflowed)`.
type NegZeroSplit = (usize, usize);

/// Stored codes + scales vs the CPU OCP encoder: byte-exact except an OCP 0x80
/// code, which must be stored 0x00 (counted: actual -0.0 input vs negative
/// underflow). A stored 0x80 anywhere is a failure.
fn cmp_fp8(
    label: &str,
    got: &[u8],
    want: &[u8],
    input: &[f32],
    nz: &mut NegZeroSplit,
) -> Result<String, String> {
    if got.len() != want.len() || want.len() % FP8_ROW != 0 {
        return Err(format!(
            "{label}: byte lens got {} want {}",
            got.len(),
            want.len()
        ));
    }
    let rows = want.len() / FP8_ROW;
    if input.len() != rows * KV_W {
        return Err(format!(
            "{label}: input len {} != {}",
            input.len(),
            rows * KV_W
        ));
    }
    let (mut bad, mut first, mut split) = (0usize, None::<String>, (0usize, 0usize));
    for r in 0..rows {
        let base = r * FP8_ROW;
        for h in 0..QSA_KV_HEADS {
            for c in 0..HD {
                let (g, w) = (got[base + h * HD + c], want[base + h * HD + c]);
                let x = input[r * KV_W + h * HD + c];
                if g == 0x80 {
                    bad += 1;
                    first.get_or_insert_with(|| {
                        format!("stored 0x80 at row{r} head{h} ch{c} (input {x:?})")
                    });
                } else if g == w {
                    continue;
                } else if w == 0x80 && g == 0x00 {
                    if x.to_bits() == 0x8000_0000 {
                        split.0 += 1;
                    } else {
                        split.1 += 1;
                    }
                } else {
                    bad += 1;
                    first.get_or_insert_with(|| {
                        format!("code row{r} head{h} ch{c}: got {g:#04x} want {w:#04x} input {x:?} ({:#010x})", x.to_bits())
                    });
                }
            }
            let o = base + KV_W + 2 * h;
            if got[o..o + 2] != want[o..o + 2] {
                bad += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "scale row{r} head{h}: got {:02x?} want {:02x?}",
                        &got[o..o + 2],
                        &want[o..o + 2]
                    )
                });
            }
        }
    }
    if let Some(f) = first {
        return Err(format!(
            "{label}: {bad} byte mismatches over {rows} rows; first {f}"
        ));
    }
    nz.0 += split.0;
    nz.1 += split.1;
    Ok(format!(
        "{label}: codes+scales byte-exact over {rows} rows, no 0x80 stored (OCP 0x80 -> 0x00: {} input -0.0, {} negative underflow)",
        split.0, split.1
    ))
}

/// `(negative-underflow OCP 0x80 count, input -0.0 count)` of a dataset: the
/// populations this test must actually exercise.
fn fp8_population(input: &[f32], want: &[u8]) -> (usize, usize) {
    let rows = want.len() / FP8_ROW;
    let (mut under, mut zero) = (0usize, 0usize);
    for r in 0..rows {
        for h in 0..QSA_KV_HEADS {
            for c in 0..HD {
                if want[r * FP8_ROW + h * HD + c] == 0x80 {
                    if input[r * KV_W + h * HD + c].to_bits() == 0x8000_0000 {
                        zero += 1;
                    } else {
                        under += 1;
                    }
                }
            }
        }
    }
    (under, zero)
}

/// Decoded stored rows vs decoded OCP rows: identical except zeros, whose sign
/// may differ (OCP 0x80 -> stored +0). Returns the number of zero-sign diffs.
fn zero_sign_only(label: &str, stored: &[f32], ocp: &[f32]) -> Result<String, String> {
    if stored.len() != ocp.len() {
        return Err(format!("{label}: len {} != {}", stored.len(), ocp.len()));
    }
    let mut flips = 0usize;
    for (i, (&s, &o)) in stored.iter().zip(ocp).enumerate() {
        if s.to_bits() == o.to_bits() {
            continue;
        }
        if s == 0.0 && o == 0.0 {
            flips += 1;
            continue;
        }
        return Err(format!(
            "{label}: @{i} stored decode {s:?} vs OCP decode {o:?} differ beyond the sign of zero"
        ));
    }
    Ok(format!(
        "{label}: decoded == OCP decode except {flips} zero-sign differences"
    ))
}

fn fp8_append(
    gpu: &mut Gpu,
    key: &GpuTensor,
    value: &GpuTensor,
    fk: &GpuTensor,
    fv: &GpuTensor,
    rows: usize,
    position_start: usize,
) -> Result<(), String> {
    indexed_attention_cache_append_batch(
        gpu,
        &IndexedAttentionCacheAppendBatch {
            key,
            value,
            full_keys: fk,
            full_values: fv,
            rows,
            position_start,
            kv_heads: QSA_KV_HEADS,
            head_dim: HD,
            format: FP8_FMT,
        },
    )
    .map_err(e)
}

fn untouched(label: &str, bytes: &[u8]) -> Result<String, String> {
    match bytes.iter().position(|&b| b != 0xA5) {
        None => Ok(format!("{label}: {} untouched sentinel bytes", bytes.len())),
        Some(i) => Err(format!(
            "{label}: sentinel byte {i} overwritten ({:#04x})",
            bytes[i]
        )),
    }
}

/// `indexed_attention_cache_append_batch` with `Fp8Fnuz`: stored bytes vs the
/// CPU OCP RNE encoder (boundaries, ties, zeros, non-power-of-two scales;
/// OCP 0x80 stored 0x00, split reported), rows before `position_start`
/// untouched, 7 single-row launches byte-identical to the batch, and the
/// stored rows decoded on device (scalar `qsa_kv_value`) bit-exact against the
/// CPU OCP decode of the stored bytes and equal to the OCP-encoder decode up
/// to the sign of zero.
fn case_fp8_cache_append(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let pos0 = 3usize;
    let key = fp8_data(0x8101, false);
    let value = fp8_data(0x8102, true);
    let rows = key.len() / KV_W;
    let (want_k, want_v) = (fp8_encode_rows(&key), fp8_encode_rows(&value));
    let mut checks = Vec::new();
    for (name, input, want) in [("K", &key, &want_k), ("V", &value, &want_v)] {
        let (under, zero) = fp8_population(input, want);
        if under == 0 || zero == 0 {
            return Err(format!(
                "{name} dataset lacks negative underflow ({under}) or -0 ({zero}) codes"
            ));
        }
        checks.push(Ok(format!("{name} dataset: {rows} rows, {under} negative-underflow OCP 0x80 codes, {zero} input -0.0")));
    }
    let mut nz: NegZeroSplit = (0, 0);
    let (k_t, v_t) = (up_f32(gpu, &key)?, up_f32(gpu, &value)?);
    let (fk, fv) = (fp8_cache(gpu, pos0 + rows)?, fp8_cache(gpu, pos0 + rows)?);
    fp8_append(gpu, &k_t, &v_t, &fk, &fv, rows, pos0)?;
    sync(gpu)?;
    let (gk, gv) = (down_bytes(gpu, &fk)?, down_bytes(gpu, &fv)?);
    checks.push(untouched(
        "K rows before position_start",
        &gk[..pos0 * FP8_ROW],
    ));
    checks.push(untouched(
        "V rows before position_start",
        &gv[..pos0 * FP8_ROW],
    ));
    checks.push(cmp_fp8(
        "batch K",
        &gk[pos0 * FP8_ROW..],
        &want_k,
        &key,
        &mut nz,
    ));
    checks.push(cmp_fp8(
        "batch V",
        &gv[pos0 * FP8_ROW..],
        &want_v,
        &value,
        &mut nz,
    ));
    // Single-row launches (decode shape) == the batch bytes.
    let singles = 7usize;
    let (sk, sv) = (
        fp8_cache(gpu, pos0 + singles)?,
        fp8_cache(gpu, pos0 + singles)?,
    );
    let mut temps = Vec::new();
    for r in 0..singles {
        let (a, b) = (
            up_f32(gpu, &key[r * KV_W..(r + 1) * KV_W])?,
            up_f32(gpu, &value[r * KV_W..(r + 1) * KV_W])?,
        );
        fp8_append(gpu, &a, &b, &sk, &sv, 1, pos0 + r)?;
        temps.extend([a, b]);
    }
    sync(gpu)?;
    let lo = pos0 * FP8_ROW;
    let hi = (pos0 + singles) * FP8_ROW;
    checks.push(exact_ints(
        "single-row K == batch",
        &down_bytes(gpu, &sk)?[lo..hi],
        &gk[lo..hi],
    ));
    checks.push(exact_ints(
        "single-row V == batch",
        &down_bytes(gpu, &sv)?[lo..hi],
        &gv[lo..hi],
    ));
    temps.extend([k_t, v_t, fk, fv, sk, sv]);
    free_all(gpu, temps)?;
    // Device decode of the device-written rows.
    for (name, stored, want) in [("K", &gk[lo..], &want_k), ("V", &gv[lo..], &want_v)] {
        let (dv, _) = fp8_probe_decode(gpu, stored)?;
        let stored_dec = fp8_decode_rows(stored);
        checks.push(exact_f32(
            &format!("{name}: device decode == CPU OCP decode of stored bytes"),
            &dv,
            &stored_dec,
        ));
        checks.push(zero_sign_only(
            &format!("{name}: stored decode vs OCP-encoder decode"),
            &dv,
            &fp8_decode_rows(want),
        ));
    }
    checks.push(Ok(format!(
        "OCP 0x80 stored as 0x00 in batch: {} input -0.0, {} negative underflow",
        nz.0, nz.1
    )));
    all(checks)
}

// ── decode bit probe ────────────────────────────────────────────────────────
// Private module = tensor_ops.hip + one kernel that runs the shipped decoders
// on stored rows: `qsa_kv_value<QSA_KV_FP8_FNUZ>` (scalar bit probe) and
// `qsa_kv_key_dots<QSA_KV_FP8_FNUZ, 1>` with a one-hot query (the packed QK
// path; its dot starts at +0, so a zero channel reads +0).
const FP8_PROBE_SRC: &str = concat!(
    include_str!("../../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../../kernels/src/rope_yarn.h"),
    include_str!("../../../../kernels/src/tensor_ops.hip"),
    r#"
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx942__)
#error "qwen4_tierc_qsa_fp8_decode_bits is a gfx942-only Tier C probe"
#endif
extern "C" __global__ __launch_bounds__(256) void qsa_tierc_fp8_decode_bits(
    const unsigned char* rows, float* values, float* keys, int kv_heads, int head_dim) {
    const int row = (int)blockIdx.y;
    const int head = (int)blockIdx.x;
    const int c = (int)threadIdx.x;
    const unsigned char* r = rows + (size_t)row * qsa_kv_row_bytes<QSA_KV_FP8_FNUZ>(kv_heads, head_dim);
    const size_t o = ((size_t)row * kv_heads + head) * head_dim + c;
    values[o] = qsa_kv_value<QSA_KV_FP8_FNUZ>(r, kv_heads, head_dim, head, c);
    float q[256];
    for (int i = 0; i < 256; ++i) q[i] = i == c ? 1.0f : 0.0f;
    float score[1] = {0.0f};
    qsa_kv_key_dots<QSA_KV_FP8_FNUZ, 1>(q, head_dim, r, kv_heads, head_dim, head, score);
    keys[o] = score[0];
}
"#
);
const FP8_PROBE_MODULE: &str = "qwen4_tierc_qsa_fp8_decode_bits";
const FP8_PROBE_SYM: &str = "qsa_tierc_fp8_decode_bits";

/// `(scalar decode, one-hot QK decode)` of stored rows, `rows * KV_W` each.
/// The rows must not hold 0x80 or NaN codes.
fn fp8_probe_decode(gpu: &mut Gpu, bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>), String> {
    let rows = bytes.len() / FP8_ROW;
    gpu.ensure_kernel_public(FP8_PROBE_MODULE, FP8_PROBE_SRC, FP8_PROBE_SYM)
        .map_err(e)?;
    let rows_t = gpu.upload_raw(bytes, &[bytes.len()]).map_err(e)?;
    let values = nan_f32(gpu, rows * KV_W)?;
    let keys = nan_f32(gpu, rows * KV_W)?;
    let mut a = hip_bridge::KernargBlob::new();
    a.push_ptr(rows_t.buf.as_ptr());
    a.push_ptr(values.buf.as_ptr());
    a.push_ptr(keys.buf.as_ptr());
    a.push_i32(QSA_KV_HEADS as i32);
    a.push_i32(HD as i32);
    a.pad_to(16);
    gpu.launch_kernel_blob(
        FP8_PROBE_SYM,
        [QSA_KV_HEADS as u32, rows as u32, 1],
        [256, 1, 1],
        0,
        a.as_mut_slice(),
    )
    .map_err(e)?;
    sync(gpu)?;
    let out = (down_f32(gpu, &values)?, down_f32(gpu, &keys)?);
    free_all(gpu, vec![rows_t, values, keys])?;
    Ok(out)
}

/// Decoded stored rows (scalar path and packed one-hot QK path) bit-exact
/// against the CPU OCP decode (zeros: the packed path reads -0 as +0, and
/// stored rows hold no -0 at all). Inputs: CPU-encoded dataset rows folded to
/// the stored form (0x80 -> 0x00), and rows holding every finite non-0x80
/// code under non-power-of-two / extreme f16 scales.
fn case_fp8_decode_bits(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let mut sets: Vec<(String, Vec<u8>)> = vec![
        (
            "dataset K".into(),
            fp8_fold_rows(&fp8_encode_rows(&fp8_data(0x8101, false))),
        ),
        (
            "dataset V".into(),
            fp8_fold_rows(&fp8_encode_rows(&fp8_data(0x8102, true))),
        ),
    ];
    let scales: [u16; 8] = [
        0x3C00, 0x3555, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x1234, 0x5A5A,
    ];
    let mut codes = Vec::new();
    for r in 0..scales.len() {
        let mut head0: Vec<u8> = (0..=255u8).collect();
        head0.rotate_left(r * 17);
        for b in head0.iter_mut() {
            if *b & 0x7F == 0x7F {
                *b ^= 0x01; // NaN codes 0x7F/0xFF -> 0x7E/0xFE
            }
            *b = fp8_fold(*b);
        }
        let mut head1 = head0.clone();
        head1.reverse();
        codes.extend_from_slice(&head0);
        codes.extend_from_slice(&head1);
        codes.extend_from_slice(&scales[r].to_le_bytes());
        codes.extend_from_slice(&scales[(r + 3) % scales.len()].to_le_bytes());
    }
    sets.push(("all finite codes".into(), codes));
    let mut checks = Vec::new();
    for (name, bytes) in &sets {
        if (0..bytes.len() / FP8_ROW)
            .any(|r| bytes[r * FP8_ROW..r * FP8_ROW + KV_W].contains(&0x80))
        {
            return Err(format!("{name}: probe rows hold a 0x80 code"));
        }
        let want = fp8_decode_rows(bytes);
        let (got_v, got_k) = fp8_probe_decode(gpu, bytes)?;
        checks.push(exact_f32(&format!("{name}: scalar decode"), &got_v, &want));
        let want_k: Vec<f32> = want
            .iter()
            .map(|&v| if v == 0.0 { 0.0 } else { v })
            .collect();
        checks.push(exact_f32(
            &format!("{name}: one-hot QK decode"),
            &got_k,
            &want_k,
        ));
    }
    all(checks)
}

// ── prologue (decode + append) ──────────────────────────────────────────────
fn fp8_prologue_gpu(
    gpu: &mut Gpu,
    i: &[Vec<f32>; 4],
    rows: usize,
    position: usize,
) -> Result<PrologueGpu, String> {
    Ok(PrologueGpu {
        index_row: up_f32(gpu, &i[0])?,
        qgate: up_f32(gpu, &i[1])?,
        keys: up_f32(gpu, &i[2])?,
        values: up_f32(gpu, &i[3])?,
        full_keys: fp8_cache(gpu, position + rows)?,
        full_values: fp8_cache(gpu, position + rows)?,
        raw: nan_f32(gpu, (position + rows) * IDX_DIM)?,
    })
}

/// `indexed_attention_decode_prologue` / `_append_prologue` with `Fp8Fnuz`
/// (F32 raw index arena): the in-place norm+RoPE outputs follow the F32
/// prologue's CPU twin; the K cache rows are the CPU encoding of the device's
/// own post-RoPE keys, the V rows the CPU encoding of the inputs (byte-exact,
/// OCP 0x80 -> 0x00 split reported); the six-launch prefill chain (norm_rope
/// x3 + round trip + copy_rows + `cache_append_batch`) and the append prologue
/// are byte-identical to the fused kernel.
fn case_fp8_decode_prologue(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let mut checks = Vec::new();
    let position = 3;
    let mut nz: NegZeroSplit = (0, 0);
    for (yname, yarn) in yarns() {
        for rows in ROW_COUNTS {
            let mut rng = Rng::new(0x8201 + rows as u64);
            let iq_norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
            let q_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let k_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
            let inputs = [
                rng.vec(rows * IDX_W, -2.0, 2.0),
                rng.vec(rows * QG_W, -2.0, 2.0),
                rng.vec(rows * KV_W, -2.0, 2.0),
                cycle_to(&fp8_data(0x8203, true), rows * KV_W),
            ];
            let (iq_t, q_t, k_t) = (
                bf16_vec(gpu, &iq_norm)?,
                bf16_vec(gpu, &q_norm)?,
                bf16_vec(gpu, &k_norm)?,
            );
            let label = format!("{yname}/rows{rows}");

            let p = fp8_prologue_gpu(gpu, &inputs, rows, position)?;
            indexed_attention_decode_prologue(
                gpu,
                &IndexedAttentionDecodePrologue {
                    index_row: &p.index_row,
                    qgate: &p.qgate,
                    keys: &p.keys,
                    values: &p.values,
                    full_keys: &p.full_keys,
                    full_values: &p.full_values,
                    raw_index_keys: &p.raw,
                    index_q_norm: &iq_t,
                    q_norm: &q_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    heads: QSA_Q_HEADS,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: FP8_FMT,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            let got_index = down_f32(gpu, &p.index_row)?;
            let got_qgate = down_f32(gpu, &p.qgate)?;
            let got_keys = down_f32(gpu, &p.keys)?;
            let got_fk = down_bytes(gpu, &p.full_keys)?;
            let got_fv = down_bytes(gpu, &p.full_values)?;
            let got_raw = down_f32(gpu, &p.raw)?;

            let w = prologue_want(
                &inputs[0], &inputs[1], &inputs[2], &inputs[3], &iq_norm, &q_norm, &k_norm, rows,
                position, yarn,
            );
            checks.push(cmp_split(
                &format!("{label}/index_row"),
                &got_index,
                &w.index_row,
                F32_ACCUM,
                |i| i % IDX_W >= IDX_Q_W,
            ));
            checks.push(cmp_split(
                &format!("{label}/qgate"),
                &got_qgate,
                &w.qgate,
                F32_ACCUM,
                |i| i % (2 * HD) >= HD,
            ));
            checks.push(close_f32(
                &format!("{label}/keys"),
                &got_keys,
                &w.keys,
                F32_ACCUM,
            ));
            checks.push(exact_f32(
                &format!("{label}/raw_index_keys (F32 arena)"),
                &got_raw,
                &w.raw,
            ));
            checks.push(untouched(
                &format!("{label}/K rows before position"),
                &got_fk[..position * FP8_ROW],
            ));
            checks.push(untouched(
                &format!("{label}/V rows before position"),
                &got_fv[..position * FP8_ROW],
            ));
            checks.push(cmp_fp8(
                &format!("{label}/K bytes"),
                &got_fk[position * FP8_ROW..],
                &fp8_encode_rows(&got_keys),
                &got_keys,
                &mut nz,
            ));
            checks.push(cmp_fp8(
                &format!("{label}/V bytes"),
                &got_fv[position * FP8_ROW..],
                &fp8_encode_rows(&inputs[3]),
                &inputs[3],
                &mut nz,
            ));

            // Prefill chain the prologue replaces.
            let c = fp8_prologue_gpu(gpu, &inputs, rows, position)?;
            norm_rope_launch(
                gpu,
                &c.index_row,
                &iq_t,
                rows,
                position,
                yarn,
                QSA_INDEX_HEADS,
                IDX_DIM,
                IDX_DIM,
                IDX_W,
            )?;
            gpu.bf16_round_trip_f32_strided(&c.index_row, rows, IDX_Q_W, IDX_W, IDX_DIM)
                .map_err(e)?;
            let keys_view = c.index_row.sub_offset(IDX_Q_W, rows * IDX_W - IDX_Q_W);
            gpu.copy_rows_strided_f32(
                &keys_view,
                &c.raw,
                rows,
                IDX_DIM,
                IDX_W,
                IDX_DIM,
                position * IDX_DIM,
                Some(IDX_DIM),
            )
            .map_err(e)?;
            norm_rope_launch(
                gpu,
                &c.qgate,
                &q_t,
                rows,
                position,
                yarn,
                QSA_Q_HEADS,
                HD,
                2 * HD,
                QG_W,
            )?;
            norm_rope_launch(
                gpu,
                &c.keys,
                &k_t,
                rows,
                position,
                yarn,
                QSA_KV_HEADS,
                HD,
                HD,
                KV_W,
            )?;
            fp8_append(
                gpu,
                &c.keys,
                &c.values,
                &c.full_keys,
                &c.full_values,
                rows,
                position,
            )?;
            sync(gpu)?;
            checks.push(exact_f32(
                &format!("{label}/chain==prologue index_row"),
                &down_f32(gpu, &c.index_row)?,
                &got_index,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue qgate"),
                &down_f32(gpu, &c.qgate)?,
                &got_qgate,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue keys"),
                &down_f32(gpu, &c.keys)?,
                &got_keys,
            ));
            checks.push(exact_ints(
                &format!("{label}/chain==prologue K bytes"),
                &down_bytes(gpu, &c.full_keys)?,
                &got_fk,
            ));
            checks.push(exact_ints(
                &format!("{label}/chain==prologue V bytes"),
                &down_bytes(gpu, &c.full_values)?,
                &got_fv,
            ));
            checks.push(exact_f32(
                &format!("{label}/chain==prologue raw"),
                &down_f32(gpu, &c.raw)?,
                &got_raw,
            ));
            c.free(gpu)?;

            // Append prologue (heads == 0): same kernel, same bytes.
            let a = fp8_prologue_gpu(gpu, &inputs, rows, position)?;
            indexed_attention_append_prologue(
                gpu,
                &IndexedAttentionAppendPrologue {
                    index_row: &a.index_row,
                    keys: &a.keys,
                    values: &a.values,
                    full_keys: &a.full_keys,
                    full_values: &a.full_values,
                    raw_index_keys: &a.raw,
                    index_q_norm: &iq_t,
                    k_norm: &k_t,
                    index_heads: QSA_INDEX_HEADS,
                    index_dim: IDX_DIM,
                    index_kv_width: IDX_DIM,
                    kv_heads: QSA_KV_HEADS,
                    head_dim: HD,
                    position,
                    rows,
                    format: FP8_FMT,
                    yarn,
                },
            )
            .map_err(e)?;
            sync(gpu)?;
            checks.push(exact_f32(
                &format!("{label}/append==decode index_row"),
                &down_f32(gpu, &a.index_row)?,
                &got_index,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode keys"),
                &down_f32(gpu, &a.keys)?,
                &got_keys,
            ));
            checks.push(exact_ints(
                &format!("{label}/append==decode K bytes"),
                &down_bytes(gpu, &a.full_keys)?,
                &got_fk,
            ));
            checks.push(exact_ints(
                &format!("{label}/append==decode V bytes"),
                &down_bytes(gpu, &a.full_values)?,
                &got_fv,
            ));
            checks.push(exact_f32(
                &format!("{label}/append==decode raw"),
                &down_f32(gpu, &a.raw)?,
                &got_raw,
            ));
            a.free(gpu)?;
            p.free(gpu)?;
            free_all(gpu, vec![iq_t, q_t, k_t])?;
        }
    }
    checks.push(Ok(format!(
        "OCP 0x80 stored as 0x00 across the case: {} input -0.0, {} negative underflow",
        nz.0, nz.1
    )));
    all(checks)
}

// ── attention / select on the FP8 cache ─────────────────────────────────────
/// CPU reference over DECODED K/V (the OCP decode of the stored bytes): the
/// F32 attention twin with the F32 selected-length rule.
fn attn_want_rows(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    selected: &[i32],
    rows: usize,
    position_start: usize,
    end: usize,
) -> Vec<f32> {
    let mut want = Vec::with_capacity(rows * Q_W);
    for r in 0..rows {
        let visible = position_start + r + 1;
        let block_count = visible / QSA_COMPRESS;
        let chosen = BUDGET_BLOCKS.min(block_count);
        let selected_len =
            (chosen * QSA_COMPRESS + visible - block_count * QSA_COMPRESS).min(SEL_CAP);
        want.extend(attention_row_cpu(
            &q[r * QG_W..(r + 1) * QG_W],
            keys,
            values,
            &selected[r * SEL_CAP..r * SEL_CAP + selected_len],
            visible,
            end,
        ));
    }
    want
}

#[allow(clippy::too_many_arguments)]
fn fp8_attention_batch(
    gpu: &mut Gpu,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    sel: &GpuTensor,
    out: &GpuTensor,
    rows: usize,
    position_start: usize,
    end: usize,
) -> Result<(), String> {
    indexed_attention_attention_batch_with_arena(
        gpu,
        &IndexedAttentionAttentionBatch {
            q_with_gate: q,
            full_keys: k,
            full_values: v,
            selected: sel,
            output: out,
            rows,
            position_start,
            n_heads: QSA_Q_HEADS,
            n_kv_heads: QSA_KV_HEADS,
            head_dim: HD,
            budget_blocks: BUDGET_BLOCKS,
            compress: QSA_COMPRESS,
            capacity: SEL_CAP,
            full_capacity: end,
            format: FP8_FMT,
            shape_selected: SEL_CAP,
        },
        None,
    )
    .map_err(e)
}

/// Stored-form (folded) FP8 rows of `vals`, uploaded, plus their CPU decode.
fn fp8_cache_from(gpu: &mut Gpu, vals: &[f32]) -> Result<(GpuTensor, Vec<f32>), String> {
    let bytes = fp8_fold_rows(&fp8_encode_rows(vals));
    let dec = fp8_decode_rows(&bytes);
    Ok((gpu.upload_raw(&bytes, &[bytes.len()]).map_err(e)?, dec))
}

/// Public batched attention (arena `None`) on a stored-form FP8 cache vs the
/// CPU attention over the decoded cache (F32_ACCUM, the F32 route tolerance:
/// the kernels sum `scale * (q . code)`, the twin `q . (scale * code)`, plus
/// `expf`/FMA ulps). Same contexts / rows / injected invalid entries as the F32
/// route case.
fn case_fp8_attention_batch(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for rows in ROW_COUNTS {
            let inp = attn_inputs(0x8400 + end as u64 + rows as u64, rows, end)?;
            let (k, kd) = fp8_cache_from(gpu, &inp.keys)?;
            let (v, vd) = fp8_cache_from(gpu, &inp.values)?;
            let want = attn_want_rows(
                &inp.q,
                &kd,
                &vd,
                &inp.selected,
                rows,
                inp.position_start,
                end,
            );
            let q = up_f32(gpu, &inp.q)?;
            let sel = raw_i32(gpu, &inp.selected)?;
            let out = nan_f32(gpu, rows * Q_W)?;
            fp8_attention_batch(gpu, &q, &k, &v, &sel, &out, rows, inp.position_start, end)?;
            sync(gpu)?;
            checks.push(close_f32(
                &format!("end{end}/rows{rows}"),
                &down_f32(gpu, &out)?,
                &want,
                F32_ACCUM,
            ));
            free_all(gpu, vec![q, k, v, sel, out])?;
        }
    }
    all(checks)
}

/// FP8 (`Fp8Fnuz`) batched attention equality: the public route (decode and
/// prefill scope; quantized K/V never take hg12, so hg4 from 2 route rows, the
/// per-head parallel kernel at 1) and the directly launched
/// `attention_fp8_fnuz_batched` / `_batched_hg4` kernels must equal
/// `indexed_attention_attention_fp8_fnuz_batched_serial` HIP as raw F32 bits
/// (gfx942 claim; the dot chains are over decoded codes, so this also pins the
/// quantized hg4 dot against the serial single-head dot). Uniform and
/// realistic operands, both contexts, rows either side of the hg4 minimum.
fn case_fp8_attention_serial_equality(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let g = QWEN4_ATTN;
    let hg4_lds = SEL_CAP * (4 * 4 + 4) + 4 * (4 * 256 + 4);
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for kind in [DataKind::Plain, DataKind::Realistic] {
            for rows in ATTN_ROWS {
                let inp = attn_inputs_kind(0x8800 + end as u64 + rows as u64, rows, end, kind)?;
                let (k, _) = fp8_cache_from(gpu, &inp.keys)?;
                let (v, _) = fp8_cache_from(gpu, &inp.values)?;
                let t = AttnTensors {
                    q: up_f32(gpu, &inp.q)?,
                    k,
                    v,
                    sel: raw_i32(gpu, &inp.selected)?,
                };
                let pos = inp.position_start;
                let serial = attn_direct(
                    gpu,
                    "indexed_attention_attention_fp8_fnuz_batched_serial",
                    [g.heads as u32, 1, rows as u32],
                    0,
                    &t,
                    g,
                    rows,
                    pos,
                    end,
                )?;
                let label = format!("{}/end{end}/rows{rows}", kind.name());
                let generic = attn_direct(
                    gpu,
                    "indexed_attention_attention_fp8_fnuz_batched",
                    [g.heads as u32, 1, rows as u32],
                    (SEL_CAP * 8) as u32,
                    &t,
                    g,
                    rows,
                    pos,
                    end,
                )?;
                checks.push(exact_f32(
                    &format!("{label}/direct_generic==serial_hip"),
                    &generic,
                    &serial,
                ));
                let hg4 = attn_direct(
                    gpu,
                    "indexed_attention_attention_fp8_fnuz_batched_hg4",
                    [(g.heads / 4) as u32, 1, rows as u32],
                    hg4_lds as u32,
                    &t,
                    g,
                    rows,
                    pos,
                    end,
                )?;
                checks.push(exact_f32(
                    &format!("{label}/direct_hg4==serial_hip"),
                    &hg4,
                    &serial,
                ));
                for scope in [false, true] {
                    let out = nan_f32(gpu, rows * Q_W)?;
                    with_scope(gpu, scope, |gp| {
                        fp8_attention_batch(gp, &t.q, &t.k, &t.v, &t.sel, &out, rows, pos, end)
                    })?;
                    sync(gpu)?;
                    let got = down_f32(gpu, &out)?;
                    free_all(gpu, vec![out])?;
                    let route_rows = if scope {
                        rdna_compute::prefill_invariant::QWEN4_PREFILL_CANON_ROWS
                    } else {
                        rows
                    };
                    checks.push(exact_f32(
                        &format!(
                            "{label}/{}/{}==serial_hip",
                            scope_name(scope),
                            if route_rows >= 2 {
                                "hg4"
                            } else {
                                "generic_parallel"
                            }
                        ),
                        &got,
                        &serial,
                    ));
                }
                free_attn(gpu, t)?;
            }
        }
    }
    all(checks)
}

/// Single-row `indexed_attention_attention` on the FP8 cache (MTP reuse
/// route): scenario A of the reuse case, 3011-row cache.
fn case_fp8_attention_single(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let scenarios = reuse_scenarios();
    let s = &scenarios[0];
    let (sel, count) = reuse_want(s);
    let full_capacity = s.position + 11;
    let mut rng = Rng::new(0x8501);
    let q = rng.vec(QG_W, -3.0, 3.0);
    let (k_t, kd) = fp8_cache_from(gpu, &rng.vec(full_capacity * KV_W, -1.0, 1.0))?;
    let (v_t, vd) = fp8_cache_from(gpu, &rng.vec(full_capacity * KV_W, -1.0, 1.0))?;
    let want = attention_row_cpu(
        &q,
        &kd,
        &vd,
        &sel[..count as usize],
        full_capacity,
        full_capacity,
    );
    let q_t = up_f32(gpu, &q)?;
    let sel_t = raw_i32(gpu, &sel)?;
    let out = nan_f32(gpu, Q_W)?;
    indexed_attention_attention(
        gpu,
        &IndexedAttentionAttention {
            q_with_gate: &q_t,
            full_keys: &k_t,
            full_values: &v_t,
            selected: &sel_t,
            output: &out,
            n_heads: QSA_Q_HEADS,
            n_kv_heads: QSA_KV_HEADS,
            head_dim: HD,
            selected_len: count as usize,
            full_capacity,
            end_position: s.position + 1,
            format: FP8_FMT,
        },
    )
    .map_err(e)?;
    sync(gpu)?;
    let r = close_f32(
        "single-row fp8 attention",
        &down_f32(gpu, &out)?,
        &want,
        F32_ACCUM,
    );
    free_all(gpu, vec![q_t, k_t, v_t, sel_t, out])?;
    r
}

/// Select (public `select_batch_mirrored` over the F32 pooled arena the FP8
/// state uses) feeding FP8 attention: device selection bit-exact vs the
/// stable-tie CPU twin, then attention over that device selection vs the CPU
/// reference on the decoded cache. 2802-token context (700 blocks > budget).
fn case_fp8_select_attention(gpu: &mut Gpu) -> Result<String, String> {
    if let Some(s) = fp8_skip(gpu) {
        return Ok(s);
    }
    let end = 2802usize;
    let blocks = end / QSA_COMPRESS;
    let mut checks = Vec::new();
    for rows in ROW_COUNTS {
        let position_start = end - rows;
        let seed = 0x8600 + rows as u64;
        let (query, pooled) = gen_select_data(seed, rows, blocks)?;
        let (sel_cpu, _) = select_rows_cpu(&query, &pooled, rows, blocks, position_start);
        let query_t = up_f32(gpu, &query)?;
        let pooled_t = up_f32(gpu, &pooled)?;
        let selected = raw_i32(gpu, &vec![-7777; rows * SEL_CAP])?;
        let mirror = raw_i32(gpu, &vec![-7777; SEL_CAP])?;
        indexed_attention_select_batch_mirrored(
            gpu,
            &IndexedAttentionSelectBatch {
                query: &query_t,
                pooled: &pooled_t,
                pooled_bf16: None,
                pooled_bf16_valid_blocks: 0,
                selected: &selected,
                rows,
                query_row_stride: IDX_W,
                block_count: blocks,
                index_heads: QSA_INDEX_HEADS,
                index_dim: IDX_DIM,
                budget_blocks: BUDGET_BLOCKS,
                compress: QSA_COMPRESS,
                position_start,
                capacity: SEL_CAP,
                shape_blocks: POOLED_CAP,
            },
            &mirror,
        )
        .map_err(e)?;
        sync(gpu)?;
        let label = format!("rows{rows}");
        checks.push(exact_ints(
            &format!("{label}/selected"),
            &down_i32(gpu, &selected)?,
            &sel_cpu,
        ));

        let mut rng = Rng::new(seed ^ 0x77);
        let q = rng.vec(rows * QG_W, -3.0, 3.0);
        let (k_t, kd) = fp8_cache_from(gpu, &rng.vec(end * KV_W, -1.0, 1.0))?;
        let (v_t, vd) = fp8_cache_from(gpu, &rng.vec(end * KV_W, -1.0, 1.0))?;
        let want = attn_want_rows(&q, &kd, &vd, &sel_cpu, rows, position_start, end);
        let q_t = up_f32(gpu, &q)?;
        let out = nan_f32(gpu, rows * Q_W)?;
        fp8_attention_batch(
            gpu,
            &q_t,
            &k_t,
            &v_t,
            &selected,
            &out,
            rows,
            position_start,
            end,
        )?;
        sync(gpu)?;
        checks.push(close_f32(
            &format!("{label}/attention over device selection"),
            &down_f32(gpu, &out)?,
            &want,
            F32_ACCUM,
        ));
        free_all(
            gpu,
            vec![query_t, pooled_t, selected, mirror, q_t, k_t, v_t, out],
        )?;
    }
    all(checks)
}

/// CPU codec unit tests (no GPU): `cargo test -p hipfire-runtime --example
/// <this example> fp8_fnuz_codec`.
#[cfg(test)]
mod fp8_fnuz_codec_tests {
    use super::*;

    /// Finite positive code table value and nearest-even brute force.
    fn brute_encode(a: f32) -> u8 {
        let mut best = (f64::INFINITY, 0u8);
        for c in 0u8..=0x7E {
            let d = (e4m3_decode(c) as f64 - a as f64).abs();
            if d < best.0 || (d == best.0 && c % 2 == 0 && best.1 % 2 == 1) {
                best = (d, c);
            }
        }
        best.1
    }

    #[test]
    fn every_finite_code_round_trips_except_80_to_00_and_no_80_is_stored() {
        for c in 0u8..=255 {
            if c & 0x7F == 0x7F {
                assert!(e4m3_decode(c).is_nan(), "{c:#04x}");
                continue;
            }
            let v = e4m3_decode(c);
            let stored = fp8_fold(e4m3_encode(v));
            assert_ne!(stored, 0x80, "stored 0x80 from code {c:#04x}");
            let expect = if c == 0x80 { 0x00 } else { c };
            assert_eq!(stored, expect, "code {c:#04x} (value {v:?})");
            // Decode of the stored code equals the OCP decode up to zero sign.
            let back = e4m3_decode(stored);
            assert!(back == v, "{c:#04x}: {back:?} vs {v:?}");
            if c != 0x80 {
                assert_eq!(back.to_bits(), v.to_bits(), "{c:#04x}");
            }
        }
    }

    #[test]
    fn negative_underflow_and_negative_zero_fold_to_plus_zero() {
        for x in [
            -0.0f32,
            -1e-30,
            -f32::from_bits(1),
            -2f32.powi(-10),
            -(2f32.powi(-10) * 0.99),
        ] {
            assert_eq!(e4m3_encode(x), 0x80, "{x:?}");
            assert_eq!(fp8_fold(e4m3_encode(x)), 0x00, "{x:?}");
        }
        // Just above the half-subnormal tie rounds to the first negative code.
        assert_eq!(
            e4m3_encode(-f32::from_bits(2f32.powi(-10).to_bits() + 1)),
            0x81
        );
    }

    #[test]
    fn encoder_matches_nearest_even_brute_force() {
        let mut rng = Rng::new(0xC0DEC);
        let mut xs: Vec<f32> = Vec::new();
        for c in 0u8..0x7E {
            let mid = (e4m3_decode(c) + e4m3_decode(c + 1)) * 0.5;
            xs.extend([
                mid,
                f32::from_bits(mid.to_bits() + 1),
                f32::from_bits(mid.to_bits() - 1),
            ]);
        }
        for _ in 0..20000 {
            let k = rng.below(18) as i32 - 12;
            xs.push(rng.range(0.0, 448.0) * 2f32.powi(k).min(1.0));
        }
        for a in xs {
            if a > 448.0 {
                continue;
            }
            assert_eq!(
                e4m3_encode(a),
                brute_encode(a),
                "{a:?} ({:#010x})",
                a.to_bits()
            );
            assert_eq!(e4m3_encode(-a), brute_encode(a) | 0x80, "{:?}", -a);
        }
    }

    #[test]
    fn f16_helpers_round_trip_and_round_to_nearest_even() {
        for b in 0u16..=0xFFFF {
            if (b >> 10) & 31 == 31 {
                continue;
            }
            assert_eq!(f32_to_f16_bits(f16_bits_to_f32(b)), b, "{b:#06x}");
        }
        // Ties between adjacent f16 values go to even.
        for b in [0x0001u16, 0x0002, 0x03FF, 0x0400, 0x3C00, 0x3C01, 0x7BFE] {
            let mid = (f16_bits_to_f32(b) + f16_bits_to_f32(b + 1)) * 0.5;
            let want = if b % 2 == 0 { b } else { b + 1 };
            assert_eq!(f32_to_f16_bits(mid), want, "tie above {b:#06x}");
        }
    }

    #[test]
    fn scale_follows_amax_over_448_with_one_ulp_bump() {
        assert_eq!(fp8_scale(0.0), (0x3C00, 1.0));
        assert_eq!(fp8_scale(448.0), (0x3C00, 1.0));
        // Underflow lands on the smallest subnormal.
        assert_eq!(fp8_scale(1e-9).0, 0x0001);
        for i in 1..20000u32 {
            let amax = i as f32 * 0.37;
            let (bits, sf) = fp8_scale(amax);
            assert!(448.0 * sf >= amax, "amax {amax}: scale {bits:#06x}");
        }
    }
}

// ── gates / unreachable-kernel proofs ────────────────────────────────────────
/// Pure + arch-conditional evidence for the gfx942 KV formats: `kv_mode::
/// resolve_qwen4` (auto/bf16 = F32 state, explicit fp8 accepted, q8 refused),
/// the `Fp8Fnuz` row geometry (516 B, Raw, F32 index arenas), `has_gfx11_plus_simt`
/// / `has_wmma_w32` / `is_gfx1151` false, the F16-arena and gathered-WMMA
/// predicates false, the RDNA `Fp8` and `Q8` K/V launchers returning `Err`
/// (`check_qsa_format`) before any launch, and the `Fp8Fnuz` launcher accepted
/// (row stored as the CPU encoder says).
fn case_gates(gpu: &mut Gpu) -> Result<String, String> {
    use hipfire_runtime::kv_mode::{resolve_qwen4, KvMode};
    for raw in ["", "auto", "bf16"] {
        let r = resolve_qwen4(raw, "gfx942")?;
        if !matches!(r.mode, KvMode::Bf16) {
            return Err(format!(
                "resolve_qwen4({raw:?}, gfx942) is not the F32 (Bf16) state"
            ));
        }
    }
    let r = resolve_qwen4("fp8", "gfx942")
        .map_err(|err| format!("resolve_qwen4(\"fp8\", gfx942) must be accepted: {err}"))?;
    if !matches!(r.mode, KvMode::Fp8) {
        return Err("resolve_qwen4(\"fp8\", gfx942) is not the Fp8 mode".into());
    }
    if resolve_qwen4("q8", "gfx942").is_ok() {
        return Err("resolve_qwen4(\"q8\", gfx942) must be refused".into());
    }
    let mut notes =
        vec!["resolve_qwen4(gfx942): auto/bf16 -> F32 state, fp8 accepted, q8 refused".to_string()];
    let f = QsaKvFormat::Fp8Fnuz;
    if !f.supports(QSA_KV_HEADS, HD)
        || f.kv_row_units(QSA_KV_HEADS, HD) != FP8_ROW
        || f.kv_row_bytes(QSA_KV_HEADS, HD) != FP8_ROW
        || FP8_ROW != 516
        || f.kv_dtype() != DType::Raw
        || f.index_dtype() != DType::F32
    {
        return Err(format!(
            "Fp8Fnuz geometry: supports={} row_units={} row_bytes={} kv_dtype={:?} index_dtype={:?}; want 516 B Raw rows + F32 index",
            f.supports(QSA_KV_HEADS, HD),
            f.kv_row_units(QSA_KV_HEADS, HD),
            f.kv_row_bytes(QSA_KV_HEADS, HD),
            f.kv_dtype(),
            f.index_dtype()
        ));
    }
    notes.push("Fp8Fnuz: 516 B Raw K/V rows, F32 index arenas".into());
    if gpu.arch != "gfx942" {
        notes.push(format!(
            "arch {} != gfx942: arch-conditional checks skipped",
            gpu.arch
        ));
        return Ok(notes.join("; "));
    }
    {
        let caps = &gpu.arch_caps;
        if caps.has_gfx11_plus_simt() || caps.has_wmma_w32() || caps.is_gfx1151() {
            return Err("gfx942 reports gfx11 SIMT / WMMA / gfx1151 capability".into());
        }
    }
    for fmt in [
        QsaKvFormat::F32,
        QsaKvFormat::Q8,
        QsaKvFormat::Fp8,
        QsaKvFormat::Fp8Fnuz,
    ] {
        if rdna_compute::tensor_ops::qsa_gathered_wmma_enabled(gpu, fmt) {
            return Err(format!("gathered WMMA enabled for {fmt:?} on gfx942"));
        }
    }
    for fmt in [QsaKvFormat::F32, QsaKvFormat::Fp8Fnuz] {
        if rdna_compute::tensor_ops::qsa_f16_arena_enabled(gpu, fmt, HD) {
            return Err(format!("QSA F16 arena enabled for {fmt:?} on gfx942"));
        }
    }
    notes.push(
        "has_gfx11_plus_simt/has_wmma_w32/is_gfx1151 false; gathered-WMMA + F16-arena predicates false"
            .into(),
    );
    for fmt in [QsaKvFormat::Fp8, QsaKvFormat::Q8, QsaKvFormat::Fp8Fnuz] {
        let row_units = fmt.kv_row_units(QSA_KV_HEADS, HD);
        let cache_k = gpu.zeros(&[row_units], DType::Raw).map_err(e)?;
        let cache_v = gpu.zeros(&[row_units], DType::Raw).map_err(e)?;
        let key = up_f32(gpu, &vec![0.5; KV_W])?;
        let value = up_f32(gpu, &vec![0.5; KV_W])?;
        let append = indexed_attention_cache_append_batch(
            gpu,
            &IndexedAttentionCacheAppendBatch {
                key: &key,
                value: &value,
                full_keys: &cache_k,
                full_values: &cache_v,
                rows: 1,
                position_start: 0,
                kv_heads: QSA_KV_HEADS,
                head_dim: HD,
                format: fmt,
            },
        );
        let stored = if append.is_ok() {
            sync(gpu)?;
            Some((down_bytes(gpu, &cache_k)?, down_bytes(gpu, &cache_v)?))
        } else {
            None
        };
        free_all(gpu, vec![cache_k, cache_v, key, value])?;
        match (fmt, append, stored) {
            (QsaKvFormat::Fp8Fnuz, Ok(()), Some((k, v))) => {
                let want = fp8_fold_rows(&fp8_encode_rows(&vec![0.5; KV_W]));
                if k != want || v != want {
                    return Err("Fp8Fnuz cache_append_batch accepted but stored bytes differ from the CPU encoder".into());
                }
                notes.push("cache_append_batch Fp8Fnuz accepted (bytes == CPU encoder)".into());
            }
            (QsaKvFormat::Fp8Fnuz, Err(err), _) => {
                return Err(format!(
                    "cache_append_batch Fp8Fnuz refused on gfx942: {err}"
                ));
            }
            (_, Ok(()), _) => {
                return Err(format!("cache_append_batch accepted {fmt:?} K/V on gfx942"))
            }
            (_, Err(err), _) => notes.push(format!("cache_append_batch {fmt:?} refused: {err}")),
        }
    }
    Ok(notes.join("; "))
}

// ── direct launches: each enabled parallel kernel on its own ─────────────────
// A private module name makes the JIT compile the shipped `tensor_ops.hip`
// source (same concatenation as `tensor_ops.rs` `TENSOR_OPS_SRC`) apart from
// the launchers' module, so one kernel runs in isolation with a chosen grid /
// LDS and next to its `*_serial` HIP twin (the exactness baseline; the public
// launchers keep the serial kernels only as the oversized-shape fallback).
// Kernarg layouts follow the launchers (`tensor_ops.rs`).
const DIRECT_SRC: &str = concat!(
    include_str!("../../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../../kernels/src/rope_yarn.h"),
    include_str!("../../../../kernels/src/tensor_ops.hip")
);
const DIRECT_MODULE: &str = "qwen4_tierc_qsa_parallel_tensor_ops";

fn direct_load(gpu: &mut Gpu, sym: &str) -> Result<(), String> {
    gpu.ensure_kernel_public(DIRECT_MODULE, DIRECT_SRC, sym)
        .map_err(e)
}

fn direct_launch(
    gpu: &mut Gpu,
    sym: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared: u32,
    mut args: hip_bridge::KernargBlob,
) -> Result<(), String> {
    args.pad_to(16);
    gpu.launch_kernel_blob(sym, grid, block, shared, args.as_mut_slice())
        .map_err(e)
}

/// One select launch's shared arguments (query rows are `IDX_W` wide, four
/// 128-wide index heads).
struct SelectRun<'a> {
    query: &'a GpuTensor,
    pooled: &'a GpuTensor,
    rows: usize,
    blocks: usize,
    position_start: usize,
    budget: usize,
    capacity: usize,
}

/// `indexed_attention_select_{f32,bf16}_batched_serial` (grid `rows`, one
/// thread per row): the selection-sort baseline every parallel selector must
/// reproduce byte for byte, whatever the scores' rounding.
fn serial_select(gpu: &mut Gpu, r: &SelectRun, bf16: bool) -> Result<Vec<i32>, String> {
    let sym = if bf16 {
        "indexed_attention_select_bf16_batched_serial"
    } else {
        "indexed_attention_select_f32_batched_serial"
    };
    direct_load(gpu, sym)?;
    let selected = raw_i32(gpu, &vec![-7777; r.rows * r.capacity])?;
    let mut a = hip_bridge::KernargBlob::new();
    a.push_ptr(r.query.buf.as_ptr());
    a.push_ptr(r.pooled.buf.as_ptr());
    a.push_ptr(selected.buf.as_ptr());
    for v in [
        r.rows,
        IDX_W,
        r.blocks,
        QSA_INDEX_HEADS,
        IDX_DIM,
        r.budget,
        QSA_COMPRESS,
        r.position_start,
        r.capacity,
    ] {
        a.push_i32(v as i32);
    }
    direct_launch(gpu, sym, [r.rows as u32, 1, 1], [1, 1, 1], 0, a)?;
    sync(gpu)?;
    let out = down_i32(gpu, &selected)?;
    free_all(gpu, vec![selected])?;
    Ok(out)
}

struct FusedOut {
    selected: Vec<i32>,
    mirror: Vec<i32>,
    /// `[rows][blocks.div_ceil(4) * 4]` score rows (global-score mode only).
    scores: Option<Vec<f32>>,
}

/// `indexed_attention_select_{f32,bf16}_batched` (the rank-parallel fused
/// selector, block 256) with a non-null mirror. `global_scores`: score rows in
/// a global buffer (the arena-past-LDS mode, no dynamic LDS); otherwise the
/// scores live in `POOLED_CAP * 4` bytes of dynamic LDS.
fn fused_select(
    gpu: &mut Gpu,
    r: &SelectRun,
    bf16: bool,
    global_scores: bool,
) -> Result<FusedOut, String> {
    let sym = if bf16 {
        "indexed_attention_select_bf16_batched"
    } else {
        "indexed_attention_select_f32_batched"
    };
    direct_load(gpu, sym)?;
    let stride = r.blocks.div_ceil(4) * 4;
    let selected = raw_i32(gpu, &vec![-7777; r.rows * r.capacity])?;
    let mirror = raw_i32(gpu, &vec![-7777; r.capacity])?;
    let scores_t = if global_scores {
        Some(nan_f32(gpu, r.rows * stride)?)
    } else {
        None
    };
    let mut a = hip_bridge::KernargBlob::new();
    a.push_ptr(r.query.buf.as_ptr());
    a.push_ptr(r.pooled.buf.as_ptr());
    a.push_ptr(selected.buf.as_ptr());
    for v in [
        r.rows,
        IDX_W,
        r.blocks,
        QSA_INDEX_HEADS,
        IDX_DIM,
        r.budget,
        QSA_COMPRESS,
        r.position_start,
        r.capacity,
    ] {
        a.push_i32(v as i32);
    }
    a.push_ptr(mirror.buf.as_ptr());
    match &scores_t {
        Some(t) => {
            a.push_ptr(t.buf.as_ptr());
            a.push_i32(stride as i32);
        }
        None => {
            a.push_ptr(std::ptr::null::<std::ffi::c_void>());
            a.push_i32(0);
        }
    }
    let shared = if global_scores {
        0
    } else {
        (POOLED_CAP * 4) as u32
    };
    direct_launch(gpu, sym, [r.rows as u32, 1, 1], [256, 1, 1], shared, a)?;
    sync(gpu)?;
    let out = FusedOut {
        selected: down_i32(gpu, &selected)?,
        mirror: down_i32(gpu, &mirror)?,
        scores: match &scores_t {
            Some(t) => Some(down_f32(gpu, t)?),
            None => None,
        },
    };
    let mut free = vec![selected, mirror];
    free.extend(scores_t);
    free_all(gpu, free)?;
    Ok(out)
}

/// rows8 / rows16 block-score kernel then `indexed_attention_select_from_scores`
/// (the pair the public F32 route runs for budgets <= 512). Returns the score
/// table (`[rows][blocks]`, unwritten slots stay NaN) and the selection.
fn rows_scores_and_select(
    gpu: &mut Gpu,
    r: &SelectRun,
    group_rows: usize,
    score_sym: &str,
) -> Result<(Vec<f32>, Vec<i32>), String> {
    let select_sym = "indexed_attention_select_from_scores";
    direct_load(gpu, score_sym)?;
    direct_load(gpu, select_sym)?;
    let scores_t = nan_f32(gpu, r.rows * r.blocks)?;
    let selected = raw_i32(gpu, &vec![-7777; r.rows * r.capacity])?;
    let mut a = hip_bridge::KernargBlob::new();
    a.push_ptr(r.query.buf.as_ptr());
    a.push_ptr(r.pooled.buf.as_ptr());
    a.push_ptr(scores_t.buf.as_ptr());
    for v in [
        r.rows,
        IDX_W,
        r.blocks,
        IDX_DIM,
        QSA_COMPRESS,
        r.position_start,
        r.blocks,
    ] {
        a.push_i32(v as i32);
    }
    direct_launch(
        gpu,
        score_sym,
        [
            r.blocks.div_ceil(256) as u32,
            r.rows.div_ceil(group_rows) as u32,
            1,
        ],
        [256, 1, 1],
        0,
        a,
    )?;
    let mut s = hip_bridge::KernargBlob::new();
    s.push_ptr(scores_t.buf.as_ptr());
    s.push_i32(r.blocks as i32); // score_stride
    s.push_ptr(selected.buf.as_ptr());
    for v in [
        r.rows,
        r.blocks,
        r.budget,
        QSA_COMPRESS,
        r.position_start,
        r.capacity,
    ] {
        s.push_i32(v as i32);
    }
    s.push_ptr(std::ptr::null::<std::ffi::c_void>());
    direct_launch(gpu, select_sym, [r.rows as u32, 1, 1], [256, 1, 1], 0, s)?;
    sync(gpu)?;
    let out = (down_f32(gpu, &scores_t)?, down_i32(gpu, &selected)?);
    free_all(gpu, vec![scores_t, selected])?;
    Ok(out)
}

/// `indexed_attention_select_scores_rows{8,16}_f32` + `select_from_scores`
/// (`tensor_ops.hip`), the decode (rows8) / prefill (rows16) pair. Context 2802
/// => 700 blocks, the last row's budget boundary inside a tie group (the radix
/// select's width-32 `__shfl_up` equal-count scan). Per data set:
/// - plain (dyadic): scores and selection bit-exact against the CPU twin;
/// - realistic (non-dyadic, cancelling; gfx942 claim): scores bit-exact
///   against the CPU `mul_add` twin AND against the fused kernel's `score4`
///   scores, selection bit-exact against `*_batched_serial` HIP and the twin.
fn parallel_rows(gpu: &mut Gpu, group_rows: usize, score_sym: &str) -> Result<String, String> {
    let claim = claims_serial_equality(gpu);
    let end = 2802usize;
    let blocks = end / QSA_COMPRESS;
    let mut checks = Vec::new();
    for kind in [DataKind::Plain, DataKind::Realistic] {
        if kind == DataKind::Realistic && !claim {
            continue;
        }
        for rows in SEL_ROWS {
            let position_start = end - rows;
            let (query, pooled) = gen_select_data_kind(
                0x7F00 + rows as u64 + kind.seed(),
                rows,
                blocks,
                BUDGET_BLOCKS,
                kind,
                false,
            )?;
            let (want_sel, want_scores) = select_rows_cpu_with(
                index_score_of(kind),
                &query,
                &pooled,
                rows,
                blocks,
                position_start,
                BUDGET_BLOCKS,
                SEL_CAP,
            );
            let want_masked = masked_scores(&want_scores, rows, blocks, blocks, position_start);
            let query_t = up_f32(gpu, &query)?;
            let pooled_t = up_f32(gpu, &pooled)?;
            let run = SelectRun {
                query: &query_t,
                pooled: &pooled_t,
                rows,
                blocks,
                position_start,
                budget: BUDGET_BLOCKS,
                capacity: SEL_CAP,
            };
            let (scores, selected) = rows_scores_and_select(gpu, &run, group_rows, score_sym)?;
            let label = format!("{}/rows{rows}", kind.name());
            let twin = kind.twin_name();
            checks.push(exact_f32(
                &format!("{label}/scores=={twin}"),
                &scores,
                &want_masked,
            ));
            checks.push(exact_ints(
                &format!("{label}/selected=={twin}"),
                &selected,
                &want_sel,
            ));
            // Wide verify: every row of the batch is bitwise the one-row launch of
            // the same kernel at `position_start + r` (its own block count), scores
            // and selection.
            if rows <= WIDE_ROWS {
                for r in 0..rows {
                    let b_r = (position_start + r + 1) / QSA_COMPRESS;
                    let row_t = up_f32(gpu, &query[r * IDX_W..(r + 1) * IDX_W])?;
                    let run1 = SelectRun {
                        query: &row_t,
                        pooled: &pooled_t,
                        rows: 1,
                        blocks: b_r,
                        position_start: position_start + r,
                        budget: BUDGET_BLOCKS,
                        capacity: SEL_CAP,
                    };
                    let (scores1, selected1) =
                        rows_scores_and_select(gpu, &run1, group_rows, score_sym)?;
                    checks.push(exact_f32(
                        &format!("{label}/row{r} scores==one-row launch"),
                        &scores[r * blocks..r * blocks + b_r],
                        &scores1[..b_r],
                    ));
                    checks.push(exact_ints(
                        &format!("{label}/row{r} selected==one-row launch"),
                        &selected[r * SEL_CAP..(r + 1) * SEL_CAP],
                        &selected1,
                    ));
                    free_all(gpu, vec![row_t])?;
                }
            }
            if claim && kind == DataKind::Realistic {
                let serial = serial_select(gpu, &run, false)?;
                checks.push(exact_ints(
                    &format!("{label}/selected==serial_hip"),
                    &selected,
                    &serial,
                ));
                let fused = fused_select(gpu, &run, false, true)?;
                let fused_scores = masked_scores(
                    fused
                        .scores
                        .as_deref()
                        .ok_or("fused select returned no scores")?,
                    rows,
                    blocks,
                    blocks.div_ceil(4) * 4,
                    position_start,
                );
                checks.push(exact_f32(
                    &format!("{label}/scores==fused_score4_hip"),
                    &scores,
                    &fused_scores,
                ));
            }
            free_all(gpu, vec![query_t, pooled_t])?;
        }
    }
    all(checks)
}

fn parallel_rows8(gpu: &mut Gpu) -> Result<String, String> {
    parallel_rows(gpu, 8, "indexed_attention_select_scores_rows8_f32")
}

fn parallel_rows16(gpu: &mut Gpu) -> Result<String, String> {
    parallel_rows(gpu, 16, "indexed_attention_select_scores_rows16_f32")
}

fn parallel_rows32(gpu: &mut Gpu) -> Result<String, String> {
    parallel_rows(gpu, 32, "indexed_attention_select_scores_rows32_f32")
}

// ── wide verify (n <= 32 rows): batched == n one-row calls, bitwise ──────────
/// Widest verify block (`Gpu::set_wide_verify_rows`).
const WIDE_ROWS: usize = rdna_compute::prefill_invariant::WIDE_VERIFY_MAX_ROWS;
/// End depths in tokens (8K / 32K / 128K / 256K at compress 4).
const WIDE_DEPTHS: [usize; 4] = [8192, 32768, 131072, 262144];

/// The arches whose exact-width gate the wide verify work raises.
fn wide_arch(gpu: &Gpu) -> bool {
    gpu.arch == "gfx1151" || gpu.arch == "gfx1201"
}

/// Every n at displacement 0, ragged and tile-edge n at displacements 1..3.
fn wide_window_ns(displacement: usize) -> Vec<usize> {
    if displacement == 0 {
        (1..=WIDE_ROWS).collect()
    } else {
        vec![1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 31, 32]
    }
}

/// Runs `f` as a verify forward of `rows` rows would: the exact-width gate on
/// for two or more rows, off for one.
fn with_wide_gate<T>(gpu: &mut Gpu, rows: usize, f: impl FnOnce(&mut Gpu) -> T) -> T {
    gpu.set_wide_verify_rows(if rows >= 2 { WIDE_ROWS } else { 0 });
    let out = f(gpu);
    gpu.set_wide_verify_rows(0);
    out
}

fn cmp_bytes(fails: &mut Vec<String>, label: &str, got: &[u8], want: &[u8]) {
    if got.len() != want.len() {
        fails.push(format!(
            "{label}: byte lens {} != {}",
            got.len(),
            want.len()
        ));
    } else if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        fails.push(format!(
            "{label}: first differing byte @{i}: batched {:#04x} one-row {:#04x}",
            got[i], want[i]
        ));
    }
}

fn wide_verdict(name: &str, windows: usize, fails: Vec<String>) -> Result<String, String> {
    if fails.is_empty() {
        Ok(format!(
            "{name}: {windows} windows bit-exact vs n one-row calls"
        ))
    } else {
        let shown: Vec<String> = fails.iter().take(40).cloned().collect();
        Err(format!(
            "{name}: {} failure(s) in {windows} windows: {}",
            fails.len(),
            shown.join(" | ")
        ))
    }
}

/// `indexed_attention_decode_prologue` (grid.y = rows) over a window of n rows
/// at position P versus n one-row prologues at P + r on views of identical
/// inputs: normalized index query, Q with the untouched gate, normalized K, the
/// V source, every K/V cache byte of rows P..P+n (fp8: codes and scales) and
/// the raw index keys, bit for bit; the poisoned rows P-1 and P+n must stay
/// poisoned. End depths 8K..256K minus displacement 0..3; both RoPE settings.
fn wide_prologue(gpu: &mut Gpu, fp8: bool) -> Result<String, String> {
    if !wide_arch(gpu) {
        return Ok(format!(
            "skip: the wide verify gate is exact gfx1151/gfx1201 only (got {})",
            gpu.arch
        ));
    }
    if fp8 && gpu.arch != "gfx1201" {
        return Ok(format!(
            "skip: OCP FP8 K/V is gfx1201 only (got {})",
            gpu.arch
        ));
    }
    let format = if fp8 {
        QsaKvFormat::Fp8
    } else {
        QsaKvFormat::F32
    };
    // The deepest end plus the canary row P+n after a window that ends there.
    let depth_max = WIDE_DEPTHS[WIDE_DEPTHS.len() - 1] + 1;
    let (row_bytes, raw_unit) = if fp8 {
        (FP8_ROW, IDX_DIM * 2)
    } else {
        (KV_W * 4, IDX_DIM * 4)
    };
    let (k_cache, v_cache, raw) = if fp8 {
        (
            fp8_cache(gpu, depth_max)?,
            fp8_cache(gpu, depth_max)?,
            bf16_vec(gpu, &vec![f32::NAN; depth_max * IDX_DIM])?,
        )
    } else {
        (
            nan_f32(gpu, depth_max * KV_W)?,
            nan_f32(gpu, depth_max * KV_W)?,
            nan_f32(gpu, depth_max * IDX_DIM)?,
        )
    };
    let mut fails = Vec::new();
    let mut windows = 0usize;
    for (yname, yarn) in yarns() {
        let mut rng = Rng::new((if fp8 { 0x9A11u64 } else { 0x9A01u64 }) + windows as u64);
        let iq_norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
        let q_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
        let k_norm = rng.vec_bf16_exact(HD, -0.5, 0.5);
        let (iq_t, q_t, k_t) = (
            bf16_vec(gpu, &iq_norm)?,
            bf16_vec(gpu, &q_norm)?,
            bf16_vec(gpu, &k_norm)?,
        );
        let launch = |gpu: &mut Gpu,
                      t: [&GpuTensor; 4],
                      position: usize,
                      rows: usize|
         -> Result<(), String> {
            with_wide_gate(gpu, rows, |g| {
                indexed_attention_decode_prologue(
                    g,
                    &IndexedAttentionDecodePrologue {
                        index_row: t[0],
                        qgate: t[1],
                        keys: t[2],
                        values: t[3],
                        full_keys: &k_cache,
                        full_values: &v_cache,
                        raw_index_keys: &raw,
                        index_q_norm: &iq_t,
                        q_norm: &q_t,
                        k_norm: &k_t,
                        index_heads: QSA_INDEX_HEADS,
                        index_dim: IDX_DIM,
                        index_kv_width: IDX_DIM,
                        heads: QSA_Q_HEADS,
                        kv_heads: QSA_KV_HEADS,
                        head_dim: HD,
                        position,
                        rows,
                        format,
                        yarn,
                    },
                )
                .map_err(e)
            })
        };
        for depth in WIDE_DEPTHS {
            for disp in 0..4usize {
                let end = depth - disp;
                for n in wide_window_ns(disp) {
                    windows += 1;
                    let ps = end - n;
                    let label = format!("{yname}/{}/E{end}/n{n}", if fp8 { "fp8" } else { "f32" });
                    let inputs = [
                        rng.vec(n * IDX_W, -2.0, 2.0),
                        rng.vec(n * QG_W, -2.0, 2.0),
                        rng.vec(n * KV_W, -2.0, 2.0),
                        rng.vec(n * KV_W, -2.0, 2.0),
                    ];
                    let a = [
                        up_f32(gpu, &inputs[0])?,
                        up_f32(gpu, &inputs[1])?,
                        up_f32(gpu, &inputs[2])?,
                        up_f32(gpu, &inputs[3])?,
                    ];
                    let b = [
                        up_f32(gpu, &inputs[0])?,
                        up_f32(gpu, &inputs[1])?,
                        up_f32(gpu, &inputs[2])?,
                        up_f32(gpu, &inputs[3])?,
                    ];
                    let caches = [
                        (&k_cache, row_bytes),
                        (&v_cache, row_bytes),
                        (&raw, raw_unit),
                    ];
                    let poison = |gpu: &Gpu| -> Result<(), String> {
                        for (cache, unit) in caches {
                            gpu.hip
                                .memcpy_htod_offset(
                                    &cache.buf,
                                    (ps - 1) * unit,
                                    &vec![0xA5u8; (n + 2) * unit],
                                )
                                .map_err(e)?;
                        }
                        Ok(())
                    };
                    let observe = |gpu: &Gpu, t: &[GpuTensor; 4]| -> Result<Vec<Vec<u8>>, String> {
                        let mut out = Vec::new();
                        for tensor in t {
                            out.push(down_bytes(gpu, tensor)?);
                        }
                        for (cache, unit) in caches {
                            let mut rows = vec![0u8; (n + 2) * unit];
                            gpu.hip
                                .memcpy_dtoh_at(&mut rows, &cache.buf, (ps - 1) * unit)
                                .map_err(e)?;
                            out.push(rows);
                        }
                        Ok(out)
                    };
                    poison(gpu)?;
                    launch(gpu, [&a[0], &a[1], &a[2], &a[3]], ps, n)?;
                    sync(gpu)?;
                    let batched = observe(gpu, &a)?;
                    poison(gpu)?;
                    for r in 0..n {
                        let views = [
                            b[0].sub_offset(r * IDX_W, IDX_W),
                            b[1].sub_offset(r * QG_W, QG_W),
                            b[2].sub_offset(r * KV_W, KV_W),
                            b[3].sub_offset(r * KV_W, KV_W),
                        ];
                        launch(gpu, [&views[0], &views[1], &views[2], &views[3]], ps + r, 1)?;
                    }
                    sync(gpu)?;
                    let one = observe(gpu, &b)?;
                    for (i, what) in [
                        "index_row",
                        "qgate",
                        "keys",
                        "values",
                        "K cache rows",
                        "V cache rows",
                        "raw index key rows",
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        cmp_bytes(&mut fails, &format!("{label}/{what}"), &batched[i], &one[i]);
                    }
                    let values_in: Vec<u8> =
                        inputs[3].iter().flat_map(|v| v.to_le_bytes()).collect();
                    cmp_bytes(
                        &mut fails,
                        &format!("{label}/values untouched"),
                        &batched[3],
                        &values_in,
                    );
                    for (i, unit) in [(4usize, row_bytes), (5, row_bytes), (6, raw_unit)] {
                        let kept = batched[i][..unit].iter().all(|x| *x == 0xA5)
                            && batched[i][(n + 1) * unit..].iter().all(|x| *x == 0xA5);
                        if !kept {
                            fails.push(format!("{label}: wrote outside rows [P, P+n) (cache {i})"));
                        }
                    }
                    free_all(gpu, a.into_iter().chain(b).collect())?;
                }
            }
        }
        free_all(gpu, vec![iq_t, q_t, k_t])?;
    }
    free_all(gpu, vec![k_cache, v_cache, raw])?;
    wide_verdict(
        if fp8 {
            "wide prologue fp8"
        } else {
            "wide prologue f32"
        },
        windows,
        fails,
    )
}

fn wide_prologue_f32(gpu: &mut Gpu) -> Result<String, String> {
    wide_prologue(gpu, false)
}

fn wide_prologue_fp8(gpu: &mut Gpu) -> Result<String, String> {
    wide_prologue(gpu, true)
}

/// `indexed_attention_pool_rope_incremental` of a window `[P, P+n)` (blocks
/// `P/4 .. (P+n)/4`, `ceil(n/4)` grid) versus the n one-row pools at P + r
/// (each pooling the block its token completes, if any): the pooled bytes of
/// blocks `P/4-1 ..= (P+n)/4` (one canary block each side) must be identical.
fn wide_pool(gpu: &mut Gpu, bf16_arena: bool) -> Result<String, String> {
    if !wide_arch(gpu) {
        return Ok(format!(
            "skip: the wide verify gate is exact gfx1151/gfx1201 only (got {})",
            gpu.arch
        ));
    }
    let depth_max = WIDE_DEPTHS[WIDE_DEPTHS.len() - 1];
    let max_blocks = depth_max / QSA_COMPRESS + 8;
    let unit = IDX_DIM * if bf16_arena { 2 } else { 4 };
    let mut rng = Rng::new(if bf16_arena { 0x9B02 } else { 0x9B01 });
    let raw = rng.vec_bf16_exact(depth_max * IDX_DIM, -2.0, 2.0);
    let norm = rng.vec_bf16_exact(IDX_DIM, -0.5, 0.5);
    let norm_t = bf16_vec(gpu, &norm)?;
    let raw_t = if bf16_arena {
        bf16_vec(gpu, &raw)?
    } else {
        up_f32(gpu, &raw)?
    };
    drop(raw);
    let new_arena = |gpu: &mut Gpu| -> Result<GpuTensor, String> {
        if bf16_arena {
            bf16_vec(gpu, &vec![f32::NAN; max_blocks * IDX_DIM])
        } else {
            nan_f32(gpu, max_blocks * IDX_DIM)
        }
    };
    let (pooled_a, pooled_b) = (new_arena(gpu)?, new_arena(gpu)?);
    let mut fails = Vec::new();
    let mut windows = 0usize;
    for (yname, yarn) in yarns() {
        let pool = |gpu: &mut Gpu,
                    pooled: &GpuTensor,
                    position_start: usize,
                    rows: usize|
         -> Result<(), String> {
            with_wide_gate(gpu, rows, |g| {
                indexed_attention_pool_rope_incremental(
                    g,
                    &IndexedAttentionPoolRope {
                        raw_keys: &raw_t,
                        pooled,
                        norm: Some(&norm_t),
                        block_count: (position_start + rows) / QSA_COMPRESS,
                        compress: QSA_COMPRESS,
                        index_dim: IDX_DIM,
                        position: Some(QsaPositionBinding {
                            position_start,
                            rows,
                        }),
                        grid_bound: max_blocks,
                        yarn,
                    },
                )
                .map_err(e)
            })
        };
        for depth in WIDE_DEPTHS {
            for disp in 0..4usize {
                let end = depth - disp;
                for n in wide_window_ns(disp) {
                    windows += 1;
                    let ps = end - n;
                    let (first, last) = (ps / QSA_COMPRESS, end / QSA_COMPRESS);
                    let lo = first - 1;
                    let span = (last + 1 - lo + 1) * unit;
                    let label = format!(
                        "{yname}/{}/E{end}/n{n}",
                        if bf16_arena { "bf16" } else { "f32" }
                    );
                    for arena in [&pooled_a, &pooled_b] {
                        gpu.hip
                            .memcpy_htod_offset(&arena.buf, lo * unit, &vec![0xA5u8; span])
                            .map_err(e)?;
                    }
                    pool(gpu, &pooled_a, ps, n)?;
                    for r in 0..n {
                        pool(gpu, &pooled_b, ps + r, 1)?;
                    }
                    sync(gpu)?;
                    let (mut got, mut want) = (vec![0u8; span], vec![0u8; span]);
                    gpu.hip
                        .memcpy_dtoh_at(&mut got, &pooled_a.buf, lo * unit)
                        .map_err(e)?;
                    gpu.hip
                        .memcpy_dtoh_at(&mut want, &pooled_b.buf, lo * unit)
                        .map_err(e)?;
                    cmp_bytes(
                        &mut fails,
                        &format!("{label}/pooled blocks {lo}..={}", last + 1),
                        &got,
                        &want,
                    );
                    // The canary blocks outside [first, last) must be untouched.
                    let touched_outside = got[..unit].iter().any(|x| *x != 0xA5)
                        || got[(last - lo + 1) * unit..].iter().any(|x| *x != 0xA5);
                    if touched_outside {
                        fails.push(format!("{label}: pooled a block outside [{first}, {last})"));
                    }
                }
            }
        }
    }
    free_all(gpu, vec![pooled_a, pooled_b, raw_t, norm_t])?;
    wide_verdict(
        if bf16_arena {
            "wide pool bf16"
        } else {
            "wide pool f32"
        },
        windows,
        fails,
    )
}

fn wide_pool_f32(gpu: &mut Gpu) -> Result<String, String> {
    wide_pool(gpu, false)
}

fn wide_pool_bf16(gpu: &mut Gpu) -> Result<String, String> {
    wide_pool(gpu, true)
}

/// Public F32 attention (`indexed_attention_attention_batch_with_arena`, the
/// route the verify forward calls) over n = 1..=32 rows versus the n one-row
/// public calls at `position_start + r` on the same K/V, raw F32 bits: the
/// split-K route on gfx1151, the per-head kernels on gfx1201. Row 0 carries
/// injected `-1` / not-yet-visible entries. Plain and realistic operands at
/// 2802 and 8192 cached tokens.
fn wide_attention_f32(gpu: &mut Gpu) -> Result<String, String> {
    if !wide_arch(gpu) {
        return Ok(format!(
            "skip: the wide verify gate is exact gfx1151/gfx1201 only (got {})",
            gpu.arch
        ));
    }
    let g = QWEN4_ATTN;
    let mut fails = Vec::new();
    let mut windows = 0usize;
    for end in [2802usize, 8192] {
        for kind in [DataKind::Plain, DataKind::Realistic] {
            for rows in 1..=WIDE_ROWS {
                windows += 1;
                let seed = 0x7D00 + end as u64 + rows as u64 + kind.seed();
                let position_start = end - rows;
                let selected = attn_selection(seed, rows, end)?;
                let (q, keys, values) = match kind {
                    DataKind::Plain => {
                        let mut rng = Rng::new(seed);
                        (
                            rng.vec(rows * QG_W, -3.0, 3.0),
                            rng.vec(end * KV_W, -1.0, 1.0),
                            rng.vec(end * KV_W, -1.0, 1.0),
                        )
                    }
                    DataKind::Realistic => {
                        realistic_attn_qkv(seed, rows, end, QSA_Q_HEADS, QSA_KV_HEADS, HD)
                    }
                };
                let t = upload_attn(gpu, &q, &keys, &values, &selected)?;
                let batched = attn_public(gpu, &t, g, rows, position_start, end)?;
                let one = attn_one_row_calls(gpu, &t, &q, &selected, g, rows, position_start, end)?;
                if let Err(msg) = exact_f32(
                    &format!("{}/E{end}/rows{rows}", kind.name()),
                    &batched,
                    &one,
                ) {
                    fails.push(msg);
                }
                free_attn(gpu, t)?;
            }
        }
    }
    wide_verdict("wide attention f32 (public route)", windows, fails)
}

/// `indexed_attention_select_{f32,bf16}_batched` launched directly (dynamic LDS
/// and global-score modes, non-null mirror, atomics): budget 512 takes the
/// past-budget radix path (700 blocks), budget 520 (> the radix list capacity
/// of 512) the rank-everything path. Selection and mirror are bit-exact against
/// the CPU twin; the global mode's score rows too. On gfx942 the realistic data
/// set (non-dyadic, cancelling, forced lowest-index ties) additionally requires
/// selection bits equal to `*_batched_serial` HIP.
fn parallel_select_fused(gpu: &mut Gpu, bf16: bool) -> Result<String, String> {
    let claim = claims_serial_equality(gpu);
    let end = 2802usize;
    let blocks = end / QSA_COMPRESS;
    let mut checks = Vec::new();
    for kind in [DataKind::Plain, DataKind::Realistic] {
        if kind == DataKind::Realistic && !claim {
            continue;
        }
        for budget in [BUDGET_BLOCKS, FUSED_BUDGET_BLOCKS] {
            let capacity = cap_for(budget);
            for rows in SEL_ROWS {
                let position_start = end - rows;
                let (query, pooled) = gen_select_data_kind(
                    0x7F80 + rows as u64 + budget as u64 + kind.seed(),
                    rows,
                    blocks,
                    budget,
                    kind,
                    bf16,
                )?;
                let (want, want_scores) = select_rows_cpu_with(
                    index_score_of(kind),
                    &query,
                    &pooled,
                    rows,
                    blocks,
                    position_start,
                    budget,
                    capacity,
                );
                let want_masked = masked_scores(&want_scores, rows, blocks, blocks, position_start);
                let query_t = up_f32(gpu, &query)?;
                let pooled_t = if bf16 {
                    bf16_vec(gpu, &pooled)?
                } else {
                    up_f32(gpu, &pooled)?
                };
                let run = SelectRun {
                    query: &query_t,
                    pooled: &pooled_t,
                    rows,
                    blocks,
                    position_start,
                    budget,
                    capacity,
                };
                let serial = if claim && kind == DataKind::Realistic {
                    Some(serial_select(gpu, &run, bf16)?)
                } else {
                    None
                };
                for global in [false, true] {
                    let out = fused_select(gpu, &run, bf16, global)?;
                    let label = format!(
                        "{}/b{budget}/rows{rows}/{}",
                        kind.name(),
                        if global {
                            "global_scores"
                        } else {
                            "lds_scores"
                        }
                    );
                    let twin = kind.twin_name();
                    checks.push(exact_ints(
                        &format!("{label}/selected=={twin}"),
                        &out.selected,
                        &want,
                    ));
                    checks.push(exact_ints(
                        &format!("{label}/mirror"),
                        &out.mirror,
                        &want[(rows - 1) * capacity..],
                    ));
                    if let Some(serial) = &serial {
                        checks.push(exact_ints(
                            &format!("{label}/selected==serial_hip"),
                            &out.selected,
                            serial,
                        ));
                    }
                    if let Some(scores) = &out.scores {
                        let got = masked_scores(
                            scores,
                            rows,
                            blocks,
                            blocks.div_ceil(4) * 4,
                            position_start,
                        );
                        checks.push(exact_f32(
                            &format!("{label}/scores=={twin}"),
                            &got,
                            &want_masked,
                        ));
                    }
                }
                free_all(gpu, vec![query_t, pooled_t])?;
            }
        }
    }
    all(checks)
}

fn parallel_select_f32(gpu: &mut Gpu) -> Result<String, String> {
    parallel_select_fused(gpu, false)
}

fn parallel_select_bf16(gpu: &mut Gpu) -> Result<String, String> {
    parallel_select_fused(gpu, true)
}

/// One batched F32 attention kernel launched directly over the shared inputs
/// (same grid/LDS as `indexed_attention_attention_batch_arena`), against the CPU
/// twin (`F32_ACCUM`) and `indexed_attention_attention_f32_batched_serial`
/// (raw F32 bits on gfx942). Both the uniform and the realistic data set (wide
/// score range, outlier key channels, duplicated keys, sign-cancelling values),
/// at row counts either side of the hg4 (2) and hg12 (16) route thresholds.
fn parallel_attention(
    gpu: &mut Gpu,
    sym: &str,
    grid_x: u32,
    shared: u32,
) -> Result<String, String> {
    let g = QWEN4_ATTN;
    let mut checks = Vec::new();
    for end in [2802usize, 402] {
        for kind in [DataKind::Plain, DataKind::Realistic] {
            for rows in ATTN_ROWS {
                let inp = attn_inputs_kind(0x7C00 + end as u64 + rows as u64, rows, end, kind)?;
                let t = upload_attn(gpu, &inp.q, &inp.keys, &inp.values, &inp.selected)?;
                let got = attn_direct(
                    gpu,
                    sym,
                    [grid_x, 1, rows as u32],
                    shared,
                    &t,
                    g,
                    rows,
                    inp.position_start,
                    end,
                )?;
                let serial = attn_serial(gpu, &t, g, rows, inp.position_start, end)?;
                let label = format!("{}/end{end}/rows{rows}", kind.name());
                checks.push(close_f32(
                    &format!("{label}/vs_cpu"),
                    &got,
                    &inp.want,
                    F32_ACCUM,
                ));
                checks.push(attn_claim(
                    gpu,
                    &format!("{label}/vs_serial_hip"),
                    &got,
                    &serial,
                ));
                free_attn(gpu, t)?;
            }
        }
    }
    all(checks)
}

/// `indexed_attention_attention_f32_batched`: one 256-thread block per head,
/// scores/tokens in dynamic LDS (8 B per selected row), width-32 `__shfl_xor`.
fn parallel_attention_batched(gpu: &mut Gpu) -> Result<String, String> {
    parallel_attention(
        gpu,
        "indexed_attention_attention_f32_batched",
        QSA_Q_HEADS as u32,
        (SEL_CAP * 8) as u32,
    )
}

/// `indexed_attention_attention_f32_batched_hg4`: four heads per block, with the
/// gfx942 buffer descriptor for the value loads (`0a739489d`).
fn parallel_attention_hg4(gpu: &mut Gpu) -> Result<String, String> {
    let lds = SEL_CAP * (4 * 4 + 4) + 4 * (4 * 256 + 4);
    parallel_attention(
        gpu,
        "indexed_attention_attention_f32_batched_hg4",
        (QSA_Q_HEADS / 4) as u32,
        lds as u32,
    )
}

/// `indexed_attention_attention_f32_batched_hg12`: one block per KV group (12
/// heads), scores in registers (QSA_T = 9 tiles of 256 rows), same buffer
/// descriptor fix.
fn parallel_attention_hg12(gpu: &mut Gpu) -> Result<String, String> {
    parallel_attention(
        gpu,
        "indexed_attention_attention_f32_batched_hg12",
        QSA_KV_HEADS as u32,
        0,
    )
}
