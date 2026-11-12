// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Tier C oracle of the gfx942 batched QSA decode core
//! (`rdna_compute::tensor_ops::qsa_core_lanes_gfx942_min` + `Gfx942Device::qwen4_qsa_lanes_core`,
//! kernels `qwen4_qsa_lanes_*_gfx942`, `kernels/src/qwen4_qsa_lanes_f32.gfx942.hip`) against the
//! per-lane launches it replaces, exact gfx942, F32 and E4M3-FNUZ K/V state, F32 index keys.
//!
//! Synthetic Flash-Next QSA layer: 24 query heads, 2 KV heads, head_dim 256, indexer 4 heads x
//! 128 + 1 KV head, compress 4, budget 2048 tokens (512 blocks), selected capacity 2051, yarn
//! RoPE. N lanes (1 / 8 / 64) with the same or DIFFERENT contexts per lane (uniform 64 / 2048 /
//! 16K, a mixed ragged set that holds 16K lanes next to 255, a block-boundary set, tie-heavy
//! pooled keys) and 1 row per lane (AR decode) or 2..=8 rows per lane (verify windows).
//!
//! - reference (per lane): exactly the five launches `layer_ops.rs`
//!   `execute_indexed_attention_hc_impl` issues for a Core step on gfx942:
//!   `indexed_attention_decode_prologue`, `indexed_attention_pool_rope_incremental`,
//!   `indexed_attention_select_batch_mirrored` (the live rows8 score + select-from-scores pair),
//!   `indexed_attention_attention_batch_with_arena` (the per-head parallel kernel), on the
//!   lane's rows of one shared row scratch and one shared selected scratch (row 0, as the lane
//!   forward does);
//! - candidate (grouped): `qsa_core_lanes_gfx942_min` + `Gfx942Device::qwen4_qsa_lanes_core`
//!   over the same arguments (each lane's selected rows in its own range).
//!
//! Compared as BYTES after every one of `--steps` consecutive steps with the state carried
//! (positions advance by the lane's rows): every lane's whole K and V arena (written rows AND
//! untouched prefix / tail), raw index keys, pooled keys, persistent selected indices, the
//! shared index / q+gate / K / V / output row scratch (sentinel-initialised, with guard rows
//! around the lanes' rows), and each lane's selected scratch rows (the tail of the grouped
//! scratch must stay sentinel). Negative controls the oracle MUST detect, after step 0 on a
//! fresh third arm: one flipped bit (exactly one differing byte in each of the pooled keys, the
//! persistent selection and the output), two lanes' state tensors swapped, a wrong position on
//! one lane, two lanes' scratch rows swapped, the last lane dropped from the launch. Also: a
//! set with a lane that completes no pooled block is DECLINED by the admission (the per-lane
//! path would take the fused select), and the reference moved the state (no sentinel left in a
//! lane's output rows, the persistent selection overwritten).
//!
//! `--time N,..` (default off): one full-model QSA pass of a step (12 layers, each with its own
//! state) at `--time-ctxs` contexts, E4M3-FNUZ state: the per-lane launch chain (N x 5 launches
//! a layer) against the batched launches (5 a layer, admission included), median and min..max
//! over `--reps` timed passes after `--warm` warm-up passes, 1 GiB cache scrub between reps.
//!
//! The QSA lanes route is on by arch on gfx942; the oracle sets the flag in its own `Gpu` regardless.

//!
//! `cargo run --release -p rdna-compute --example qsa_lanes_gfx942_oracle --
//!    [--formats f32,fnuz] [--lanes 1,8,64] [--ctxs 64,2048,16384] [--steps 2] [--seed U64]
//!    [--quick] [--only SUBSTR] [--skip-controls] [--skip-correctness]
//!    [--time 8,64] [--time-ctxs 2048,16384] [--reps 9] [--warm 3]`

use rdna_compute::rope::RopeYarn;
use rdna_compute::tensor_ops::{
    indexed_attention_attention_batch_with_arena, indexed_attention_decode_prologue,
    indexed_attention_pool_rope_incremental, indexed_attention_select_batch_mirrored,
    qsa_core_lanes_gfx942_min, IndexedAttentionAttentionBatch, IndexedAttentionDecodePrologue,
    IndexedAttentionPoolRope, IndexedAttentionSelectBatch, QsaCoreLane, QsaKvFormat,
    QsaPositionBinding,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::Arc;

const NH: usize = 24;
const NKV: usize = 2;
const HD: usize = 256;
const IDX_HEADS: usize = 4;
const IDX_DIM: usize = 128;
const COMPRESS: usize = 4;
const BUDGET: usize = 2048;
const BUDGET_BLOCKS: usize = BUDGET / COMPRESS;
/// `config.qsa_selected_capacity()`: budget + compress - 1.
const CAP: usize = BUDGET + COMPRESS - 1;
/// Index row: `[q (4 x 128) | k (1 x 128)]`.
const IDXW: usize = (IDX_HEADS + 1) * IDX_DIM;
const QGW: usize = 2 * NH * HD;
const KVW: usize = NKV * HD;
const OUTW: usize = NH * HD;
/// Guard rows around the lanes' rows in every row scratch.
const GUARD: usize = 2;
const QSA_LAYERS: usize = 12;
const SENTINEL_F32: u32 = 0x7fc0_dead;
const SENTINEL_I32: u32 = 0x7f7f_7f7e;
const SENTINEL_PERSIST: u32 = 0x7e7e_7e7d;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fmt {
    F32,
    Fnuz,
}

impl Fmt {
    fn kv(self) -> QsaKvFormat {
        match self {
            Fmt::F32 => QsaKvFormat::F32,
            Fmt::Fnuz => QsaKvFormat::Fp8Fnuz,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Fmt::F32 => "f32",
            Fmt::Fnuz => "fnuz",
        }
    }
    fn row_bytes(self) -> usize {
        self.kv().kv_row_bytes(NKV, HD)
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

fn bf16_exact(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

fn bf16_vec(rng: &mut Rng, n: usize, amp: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16_bits(rng.sym() * amp).to_le_bytes())
        .collect()
}

/// `n` random F32 words in `[-amp, amp)` as little-endian bytes.
fn f32_rand_bytes(rng: &mut Rng, n: usize, amp: f32, bf16: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        let v = rng.sym() * amp;
        out.extend_from_slice(&(if bf16 { bf16_exact(v) } else { v }).to_le_bytes());
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Host initial state of one lane (the arenas at `max_seq` rows; rows `[0, ctx)` filled).
struct LaneHost {
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    raw: Vec<u8>,
    pooled: Vec<u8>,
}

/// Random valid FNUZ rows: E4M3-FNUZ codes without the NaN code (0x80) and f16 scales in
/// [2^-7, 2^-3).
fn fnuz_cache(rng: &mut Rng, filled: usize, max_seq: usize) -> Vec<u8> {
    let row_bytes = Fmt::Fnuz.row_bytes();
    let mut c = vec![0u8; max_seq * row_bytes];
    for pos in 0..filled {
        let row = &mut c[pos * row_bytes..(pos + 1) * row_bytes];
        let (codes, scales) = row.split_at_mut(NKV * HD);
        for chunk in codes.chunks_mut(8) {
            let w = rng.next().to_le_bytes();
            for (b, r) in chunk.iter_mut().zip(w) {
                *b = if r == 0x80 { 0x00 } else { r };
            }
        }
        for h in 0..NKV {
            let exp = 8 + (rng.next() % 5) as u16;
            let mant = (rng.next() & 0x3ff) as u16;
            scales[h * 2..h * 2 + 2].copy_from_slice(&((exp << 10) | mant).to_le_bytes());
        }
    }
    c
}

fn f32_cache(rng: &mut Rng, filled: usize, max_seq: usize) -> Vec<u8> {
    let row = KVW * 4;
    let mut c = f32_rand_bytes(rng, filled * KVW, 1.0, false);
    c.resize(max_seq * row, 0);
    c
}

fn lane_host(
    fmt: Fmt,
    rng: &mut Rng,
    ctx: usize,
    max_seq: usize,
    pooled_cap: usize,
    ties: bool,
) -> LaneHost {
    let blocks = ctx / COMPRESS;
    let mut raw = f32_rand_bytes(rng, ctx * IDX_DIM, 1.0, true);
    raw.resize(max_seq * IDX_DIM * 4, 0);
    let mut pooled = f32_rand_bytes(rng, blocks * IDX_DIM, 1.0, true);
    pooled.resize(pooled_cap * IDX_DIM * 4, 0);
    if ties {
        // Equal-score blocks exercise the (score desc, index asc) rank ties.
        let row = IDX_DIM * 4;
        for b in 5..blocks {
            let (head, tail) = pooled.split_at_mut(b * row);
            tail[..row].copy_from_slice(&head[(b % 5) * row..(b % 5 + 1) * row]);
        }
    }
    let (k_cache, v_cache) = match fmt {
        Fmt::F32 => (f32_cache(rng, ctx, max_seq), f32_cache(rng, ctx, max_seq)),
        Fmt::Fnuz => (fnuz_cache(rng, ctx, max_seq), fnuz_cache(rng, ctx, max_seq)),
    };
    LaneHost {
        k_cache,
        v_cache,
        raw,
        pooled,
    }
}

fn weights(rng: &mut Rng) -> [Vec<u8>; 4] {
    [
        bf16_vec(rng, IDX_DIM, 0.3),
        bf16_vec(rng, HD, 0.3),
        bf16_vec(rng, HD, 0.3),
        bf16_vec(rng, IDX_DIM, 0.3),
    ]
}

fn upload_typed(gpu: &mut Gpu, bytes: &[u8], elements: usize, dtype: DType) -> GpuTensor {
    let t = gpu.alloc_tensor(&[elements], dtype).expect("alloc");
    assert_eq!(t.byte_size(), bytes.len(), "upload size");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
    t
}

/// Weights shared by every lane of an arm.
struct Weights {
    index_q_norm: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    pool_norm: GpuTensor,
}

/// One lane's device state.
struct LaneState {
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw: GpuTensor,
    pooled: GpuTensor,
    persistent: GpuTensor,
}

/// Geometry of a case.
#[derive(Clone, Copy)]
struct Geo {
    fmt: Fmt,
    max_seq: usize,
    pooled_cap: usize,
    /// Rows of the shared row scratches (the lanes' rows, without guards).
    scratch_rows: usize,
}

fn sentinel_f32_rows(rows: usize, width: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL_F32); rows * width]
}

fn upload_lane(gpu: &mut Gpu, g: &Geo, h: &LaneHost) -> LaneState {
    let (kv_elems, dtype) = match g.fmt {
        Fmt::F32 => (g.max_seq * KVW, DType::F32),
        Fmt::Fnuz => (g.max_seq * g.fmt.row_bytes(), DType::Raw),
    };
    LaneState {
        full_keys: upload_typed(gpu, &h.k_cache, kv_elems, dtype),
        full_values: upload_typed(gpu, &h.v_cache, kv_elems, dtype),
        raw: upload_typed(gpu, &h.raw, g.max_seq * IDX_DIM, DType::F32),
        pooled: upload_typed(gpu, &h.pooled, g.pooled_cap * IDX_DIM, DType::F32),
        // The persistent selection starts as a recognisable non-(-1) pattern.
        persistent: upload_typed(
            gpu,
            &(0..CAP)
                .flat_map(|_| SENTINEL_PERSIST.to_le_bytes())
                .collect::<Vec<_>>(),
            CAP * 4,
            DType::Raw,
        ),
    }
}

/// One arm's device buffers: the lanes' states and the shared scratches.
struct Arm {
    lanes: Vec<LaneState>,
    weights: Weights,
    index: GpuTensor,
    qgate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    out: GpuTensor,
    selected: GpuTensor,
}

fn make_scratch(gpu: &mut Gpu, g: &Geo) -> [GpuTensor; 5] {
    let rows = g.scratch_rows + 2 * GUARD;
    let scratch = |gpu: &mut Gpu, w: usize| {
        gpu.upload_f32(&sentinel_f32_rows(rows, w), &[rows * w])
            .expect("scratch")
    };
    [
        scratch(gpu, IDXW),
        scratch(gpu, QGW),
        scratch(gpu, KVW),
        scratch(gpu, KVW),
        scratch(gpu, OUTW),
    ]
}

fn make_selected(gpu: &mut Gpu, g: &Geo) -> GpuTensor {
    upload_typed(
        gpu,
        &(0..g.scratch_rows * CAP)
            .flat_map(|_| SENTINEL_I32.to_le_bytes())
            .collect::<Vec<_>>(),
        g.scratch_rows * CAP * 4,
        DType::Raw,
    )
}

fn make_weights(gpu: &mut Gpu, w: &[Vec<u8>; 4]) -> Weights {
    Weights {
        index_q_norm: upload_typed(gpu, &w[0], IDX_DIM, DType::BF16),
        q_norm: upload_typed(gpu, &w[1], HD, DType::BF16),
        k_norm: upload_typed(gpu, &w[2], HD, DType::BF16),
        pool_norm: upload_typed(gpu, &w[3], IDX_DIM, DType::BF16),
    }
}

fn make_arm(gpu: &mut Gpu, g: &Geo, lanes: Vec<LaneState>, w: &[Vec<u8>; 4]) -> Arm {
    let [index, qgate, k, v, out] = make_scratch(gpu, g);
    Arm {
        lanes,
        weights: make_weights(gpu, w),
        index,
        qgate,
        k,
        v,
        out,
        selected: make_selected(gpu, g),
    }
}

fn free_arm(gpu: &mut Gpu, arm: Arm) {
    for l in arm.lanes {
        for t in [l.full_keys, l.full_values, l.raw, l.pooled, l.persistent] {
            let _ = gpu.free_tensor(t);
        }
    }
    for t in [
        arm.weights.index_q_norm,
        arm.weights.q_norm,
        arm.weights.k_norm,
        arm.weights.pool_norm,
        arm.index,
        arm.qgate,
        arm.k,
        arm.v,
        arm.out,
        arm.selected,
    ] {
        let _ = gpu.free_tensor(t);
    }
}

/// Host-side step inputs (the projections' outputs), shared by the arms.
struct StepInputs {
    index: Vec<f32>,
    qgate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn step_inputs(rng: &mut Rng, g: &Geo, total_rows: usize) -> StepInputs {
    let fill = |rng: &mut Rng, w: usize| -> Vec<f32> {
        let mut v = sentinel_f32_rows(g.scratch_rows + 2 * GUARD, w);
        for x in &mut v[GUARD * w..(GUARD + total_rows) * w] {
            *x = rng.sym();
        }
        v
    };
    StepInputs {
        index: fill(rng, IDXW),
        qgate: fill(rng, QGW),
        k: fill(rng, KVW),
        v: fill(rng, KVW),
    }
}

fn upload_step(gpu: &mut Gpu, arm: &Arm, s: &StepInputs) {
    for (t, d) in [
        (&arm.index, &s.index),
        (&arm.qgate, &s.qgate),
        (&arm.k, &s.k),
        (&arm.v, &s.v),
    ] {
        gpu.hip
            .memcpy_htod(&t.buf, &f32_bytes(d))
            .expect("step htod");
    }
    // The output scratch restarts as sentinel every step: a lane that did not run shows.
    let rows = arm.out.numel() / OUTW;
    gpu.hip
        .memcpy_htod(&arm.out.buf, &f32_bytes(&sentinel_f32_rows(rows, OUTW)))
        .expect("out sentinel");
}

/// Where one lane sits in a step.
#[derive(Clone, Copy)]
struct Spec {
    pos: usize,
    rows: usize,
    /// First row of the lane in the shared row scratches.
    row0: usize,
    /// First row of the lane in the selected scratch (0 for the per-lane arm).
    sel0: usize,
}

struct Views {
    index: GpuTensor,
    qgate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    out: GpuTensor,
    selected: GpuTensor,
}

fn views(arm: &Arm, s: &Spec) -> Views {
    let g = GUARD + s.row0;
    Views {
        index: arm.index.sub_offset(g * IDXW, s.rows * IDXW),
        qgate: arm.qgate.sub_offset(g * QGW, s.rows * QGW),
        k: arm.k.sub_offset(g * KVW, s.rows * KVW),
        v: arm.v.sub_offset(g * KVW, s.rows * KVW),
        out: arm.out.sub_offset(g * OUTW, s.rows * OUTW),
        selected: arm.selected.sub_offset(s.sel0 * CAP * 4, s.rows * CAP * 4),
    }
}

fn yarn() -> RopeYarn {
    // A scaled (non-OFF) yarn over the 64 rotated channels (32 pairs).
    RopeYarn {
        freq_scale: 0.25,
        corr_low: 4.0,
        corr_high: 20.0,
        mscale: 1.0693,
    }
}

/// The lane's launch argument sets, as the interpreter builds them.
#[allow(clippy::too_many_arguments)]
fn core_lane<'a>(
    g: &Geo,
    w: &'a Weights,
    state: &'a LaneState,
    v: &'a Views,
    pos: usize,
    rows: usize,
) -> QsaCoreLane<'a> {
    let complete = (pos + rows) / COMPRESS;
    let format = g.fmt.kv();
    QsaCoreLane {
        prologue: IndexedAttentionDecodePrologue {
            index_row: &v.index,
            qgate: &v.qgate,
            keys: &v.k,
            values: &v.v,
            full_keys: &state.full_keys,
            full_values: &state.full_values,
            raw_index_keys: &state.raw,
            index_q_norm: &w.index_q_norm,
            q_norm: &w.q_norm,
            k_norm: &w.k_norm,
            index_heads: IDX_HEADS,
            index_dim: IDX_DIM,
            index_kv_width: IDX_DIM,
            heads: NH,
            kv_heads: NKV,
            head_dim: HD,
            position: pos,
            rows,
            format,
            yarn: yarn(),
        },
        pool: (complete > 0).then(|| IndexedAttentionPoolRope {
            raw_keys: &state.raw,
            pooled: &state.pooled,
            norm: Some(&w.pool_norm),
            block_count: complete,
            compress: COMPRESS,
            index_dim: IDX_DIM,
            position: Some(QsaPositionBinding {
                position_start: pos,
                rows,
            }),
            grid_bound: g.pooled_cap,
            yarn: yarn(),
        }),
        select: IndexedAttentionSelectBatch {
            query: &v.index,
            pooled: &state.pooled,
            pooled_bf16: None,
            pooled_bf16_valid_blocks: 0,
            selected: &v.selected,
            rows,
            query_row_stride: IDXW,
            block_count: complete,
            index_heads: IDX_HEADS,
            index_dim: IDX_DIM,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            position_start: pos,
            capacity: CAP,
            shape_blocks: g.pooled_cap,
        },
        persistent_selection: &state.persistent,
        attention: IndexedAttentionAttentionBatch {
            q_with_gate: &v.qgate,
            full_keys: &state.full_keys,
            full_values: &state.full_values,
            selected: &v.selected,
            output: &v.out,
            rows,
            position_start: pos,
            n_heads: NH,
            n_kv_heads: NKV,
            head_dim: HD,
            budget_blocks: BUDGET_BLOCKS,
            compress: COMPRESS,
            capacity: CAP,
            full_capacity: g.max_seq,
            format,
            shape_selected: CAP,
        },
    }
}

/// Per-lane launches of one step; with `capture`, also each lane's selected rows right after
/// its select (the shared scratch is overwritten by the next lane; the download synchronises,
/// so timing runs pass `false`). Returns the captures and the launch count.
fn run_per_lane(
    gpu: &mut Gpu,
    g: &Geo,
    arm: &Arm,
    specs: &[Spec],
    capture: bool,
) -> (Vec<Vec<u8>>, usize) {
    let mut launches = 0;
    let mut selected = Vec::new();
    for (i, s) in specs.iter().enumerate() {
        let s0 = Spec { sel0: 0, ..*s };
        let v = views(arm, &s0);
        let lane = core_lane(g, &arm.weights, &arm.lanes[i], &v, s.pos, s.rows);
        indexed_attention_decode_prologue(gpu, &lane.prologue).expect("prologue");
        launches += 1;
        if let Some(pool) = &lane.pool {
            indexed_attention_pool_rope_incremental(gpu, pool).expect("pool");
            launches += 1;
        }
        let persisted =
            indexed_attention_select_batch_mirrored(gpu, &lane.select, lane.persistent_selection)
                .expect("select");
        assert!(
            persisted,
            "the per-lane select must mirror the persistent selection"
        );
        // The live score + select-from-scores pair.
        launches += 2;
        indexed_attention_attention_batch_with_arena(gpu, &lane.attention, None)
            .expect("attention");
        launches += 1;
        if capture {
            selected.push(gpu.download_raw_bytes(&v.selected).expect("selected"));
        }
    }
    (selected, launches)
}

/// Negative controls applied to the grouped arm's launch arguments.
#[derive(Clone, Copy, Debug)]
enum Tamper {
    None,
    /// Lanes `a` and `b` launch on each other's state tensors.
    SwapState(usize, usize),
    /// Lane `a` launches at position + 1 everywhere (admission still passes).
    WrongPosition(usize),
    /// Lanes `a` and `b` launch on each other's scratch rows.
    SwapScratch(usize, usize),
    /// The last lane is left out of the launch.
    DropLast,
}

impl Tamper {
    fn name(self) -> String {
        match self {
            Tamper::None => "none".into(),
            Tamper::SwapState(a, b) => format!("lanes {a} and {b} launched on each other's state"),
            Tamper::WrongPosition(a) => format!("lane {a} launched at position + 1"),
            Tamper::SwapScratch(a, b) => {
                format!("lanes {a} and {b} launched on each other's scratch rows")
            }
            Tamper::DropLast => "last lane dropped from the launch".into(),
        }
    }
}

/// Grouped launch of one step. `Ok(launches)`; `Err` when admission declined.
fn run_grouped(
    gpu: &mut Gpu,
    g: &Geo,
    arm: &Arm,
    specs: &[Spec],
    tamper: Tamper,
) -> Result<usize, String> {
    let vs: Vec<Views> = specs.iter().map(|s| views(arm, s)).collect();
    // Scratch views a lane launches on (a control swaps two lanes').
    let scratch_of = |i: usize| match tamper {
        Tamper::SwapScratch(a, b) if i == a => b,
        Tamper::SwapScratch(a, b) if i == b => a,
        _ => i,
    };
    let state_of = |i: usize| match tamper {
        Tamper::SwapState(a, b) if i == a => b,
        Tamper::SwapState(a, b) if i == b => a,
        _ => i,
    };
    let n = match tamper {
        Tamper::DropLast => specs.len() - 1,
        _ => specs.len(),
    };
    let lanes: Vec<QsaCoreLane<'_>> = specs[..n]
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let pos = match tamper {
                Tamper::WrongPosition(a) if a == i => s.pos + 1,
                _ => s.pos,
            };
            // The tampered scratch lane keeps its own selected range.
            let mut lane = core_lane(
                g,
                &arm.weights,
                &arm.lanes[state_of(i)],
                &vs[scratch_of(i)],
                pos,
                s.rows,
            );
            if scratch_of(i) != i {
                lane.select.selected = &vs[i].selected;
                lane.attention.selected = &vs[i].selected;
            }
            lane
        })
        .collect();
    let Some(plan) = qsa_core_lanes_gfx942_min(gpu, &lanes, 1).map_err(|e| e.to_string())? else {
        return Err("qsa_core_lanes_gfx942 declined the set".into());
    };
    let launches = plan.launch_count();
    gpu.try_gfx942()
        .ok_or("not exact gfx942")?
        .qwen4_qsa_lanes_core(&plan)
        .map_err(|e| e.to_string())?;
    Ok(launches)
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.download_raw_bytes(t).expect("download")
}

fn diff_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Compare every compared tensor of the two arms; `(name, differing bytes)` of the ones that
/// differ.
fn compare(
    gpu: &Gpu,
    reference: &Arm,
    candidate: &Arm,
    specs: &[Spec],
    ref_selected: &[Vec<u8>],
    scratch_rows: usize,
) -> Vec<(String, usize)> {
    let mut bad = Vec::new();
    let mut check = |name: String, a: &[u8], b: &[u8]| {
        let d = diff_bytes(a, b);
        if d != 0 {
            bad.push((name, d));
        }
    };
    for (i, (r, c)) in reference.lanes.iter().zip(&candidate.lanes).enumerate() {
        for (what, a, b) in [
            ("full_keys", &r.full_keys, &c.full_keys),
            ("full_values", &r.full_values, &c.full_values),
            ("raw_index_keys", &r.raw, &c.raw),
            ("pooled_keys", &r.pooled, &c.pooled),
            ("selected_indices", &r.persistent, &c.persistent),
        ] {
            check(
                format!("lane{i}.{what}"),
                &download(gpu, a),
                &download(gpu, b),
            );
        }
    }
    for (what, a, b) in [
        ("index_scratch", &reference.index, &candidate.index),
        ("qgate_scratch", &reference.qgate, &candidate.qgate),
        ("k_scratch", &reference.k, &candidate.k),
        ("v_scratch", &reference.v, &candidate.v),
        ("qsa_output", &reference.out, &candidate.out),
    ] {
        check(what.to_string(), &download(gpu, a), &download(gpu, b));
    }
    // Selected scratch rows: the reference's capture right after each lane's select against
    // the candidate's own range; the unused tail stays sentinel.
    let cand = download(gpu, &candidate.selected);
    let mut used = 0;
    for (i, s) in specs.iter().enumerate() {
        let a = &ref_selected[i];
        let b = &cand[s.sel0 * CAP * 4..(s.sel0 + s.rows) * CAP * 4];
        check(format!("lane{i}.selected_scratch"), a, b);
        used = used.max(s.sel0 + s.rows);
    }
    let sentinel: Vec<u8> = (0..(scratch_rows - used) * CAP)
        .flat_map(|_| SENTINEL_I32.to_le_bytes())
        .collect();
    check(
        "selected_scratch_tail".into(),
        &sentinel,
        &cand[used * CAP * 4..],
    );
    bad
}

/// The reference really ran: no sentinel word left in a lane's output rows, the persistent
/// selection overwritten, and every lane's K row at its position written (nonzero bytes).
fn reference_ran(gpu: &Gpu, reference: &Arm, specs: &[Spec]) -> Result<(), String> {
    let out = download(gpu, &reference.out);
    for (i, s) in specs.iter().enumerate() {
        let lo = (GUARD + s.row0) * OUTW * 4;
        let rows = &out[lo..lo + s.rows * OUTW * 4];
        let left = rows
            .chunks_exact(4)
            .filter(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == SENTINEL_F32)
            .count();
        if left != 0 {
            return Err(format!(
                "lane {i}: {left} sentinel words left in its output rows"
            ));
        }
        let persistent = download(gpu, &reference.lanes[i].persistent);
        if persistent
            .chunks_exact(4)
            .any(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) == SENTINEL_PERSIST)
        {
            return Err(format!(
                "lane {i}: the persistent selection was not written"
            ));
        }
    }
    Ok(())
}

#[derive(Clone)]
struct Case {
    name: String,
    fmt: Fmt,
    ctxs: Vec<usize>,
    rows: Vec<usize>,
    ties: bool,
    controls: bool,
}

const MIXED: [usize; 18] = [
    16384, 64, 2048, 65, 2047, 4095, 255, 1000, 63, 4093, 16383, 3, 7, 2051, 513, 12000, 64, 2048,
];
const BOUNDARY: [usize; 8] = [3, 7, 63, 64, 65, 255, 2047, 2051];

fn cases(fmts: &[Fmt], lanes: &[usize], ctxs: &[usize], quick: bool) -> Vec<Case> {
    let mut out = Vec::new();
    for &fmt in fmts {
        for &n in lanes {
            let mk =
                |name: String, c: Vec<usize>, rows: Vec<usize>, ties: bool, controls: bool| Case {
                    name: format!("{}/{name}", fmt.name()),
                    fmt,
                    ctxs: c,
                    rows,
                    ties,
                    controls,
                };
            for &ctx in ctxs {
                let controls = (n == 1 && ctx == 64) || (n == 64 && ctx == 2048);
                out.push(mk(
                    format!("u{ctx}/N{n}/r1"),
                    vec![ctx; n],
                    vec![1; n],
                    false,
                    controls,
                ));
            }
            if n >= 2 {
                let mixed: Vec<usize> = (0..n).map(|l| MIXED[(l * 7) % MIXED.len()]).collect();
                let boundary: Vec<usize> = (0..n).map(|l| BOUNDARY[l % BOUNDARY.len()]).collect();
                out.push(mk(
                    format!("mixed/N{n}/r1"),
                    mixed.clone(),
                    vec![1; n],
                    false,
                    n == 8 || n == 64,
                ));
                out.push(mk(
                    format!("mixed/N{n}/ragged-rows"),
                    mixed.clone(),
                    (0..n).map(|l| 1 + (l * 3) % 8).collect(),
                    false,
                    false,
                ));
                out.push(mk(
                    format!("boundary/N{n}/r1"),
                    boundary.clone(),
                    vec![1; n],
                    false,
                    false,
                ));
                if !quick {
                    out.push(mk(
                        format!("u2048/N{n}/r4"),
                        vec![2048; n],
                        vec![4; n],
                        false,
                        false,
                    ));
                    out.push(mk(
                        format!("boundary/N{n}/r8"),
                        boundary,
                        vec![8; n],
                        false,
                        false,
                    ));
                    out.push(mk(
                        format!("ties-u2048/N{n}/r1"),
                        vec![2048; n],
                        vec![1; n],
                        true,
                        false,
                    ));
                    out.push(mk(
                        format!("ties-mixed/N{n}/r1"),
                        mixed,
                        vec![1; n],
                        true,
                        false,
                    ));
                }
            }
        }
    }
    out
}

fn specs_for(rows: &[usize], pos: &[usize]) -> Vec<Spec> {
    let mut row0 = 0;
    pos.iter()
        .zip(rows)
        .map(|(&p, &r)| {
            let s = Spec {
                pos: p,
                rows: r,
                row0,
                sel0: row0,
            };
            row0 += r;
            s
        })
        .collect()
}

struct CaseResult {
    name: String,
    steps: usize,
    mismatches: Vec<(usize, String, usize)>,
    declined: Option<usize>,
    ran: Result<(), String>,
    controls: Vec<(String, bool, usize)>,
    launches_per_lane: usize,
    launches_grouped: usize,
}

fn host_state(
    fmt: Fmt,
    rng: &mut Rng,
    ctxs: &[usize],
    max_seq: usize,
    pooled_cap: usize,
    ties: bool,
) -> Vec<LaneHost> {
    ctxs.iter()
        .map(|&c| lane_host(fmt, rng, c, max_seq, pooled_cap, ties))
        .collect()
}

/// `arms` fresh arms from the case's deterministic host state (identical bytes in every arm),
/// the RNG positioned for the step inputs, and the weights.
fn build_arms(gpu: &mut Gpu, g: &Geo, case: &Case, seed: u64, arms: usize) -> (Vec<Arm>, Rng) {
    let mut rng = Rng(seed);
    let host = host_state(
        case.fmt,
        &mut rng,
        &case.ctxs,
        g.max_seq,
        g.pooled_cap,
        case.ties,
    );
    let w = weights(&mut rng);
    let built = (0..arms)
        .map(|_| {
            let lanes = host.iter().map(|h| upload_lane(gpu, g, h)).collect();
            make_arm(gpu, g, lanes, &w)
        })
        .collect();
    (built, rng)
}

fn geo_for(case: &Case, steps: usize) -> Geo {
    let max_ctx = *case.ctxs.iter().max().unwrap();
    let max_rows = *case.rows.iter().max().unwrap();
    let max_seq = (max_ctx + steps * max_rows + 8).next_multiple_of(COMPRESS);
    Geo {
        fmt: case.fmt,
        max_seq,
        pooled_cap: max_seq / COMPRESS,
        scratch_rows: case.rows.iter().sum(),
    }
}

fn run_case(gpu: &mut Gpu, case: &Case, steps: usize, seed: u64, controls: bool) -> CaseResult {
    let g = geo_for(case, steps);
    let total_rows: usize = case.rows.iter().sum();
    let (mut arms, mut rng) = build_arms(gpu, &g, case, seed, 2);
    let candidate = arms.pop().unwrap();
    let reference = arms.pop().unwrap();
    let mut pos = case.ctxs.clone();
    let mut result = CaseResult {
        name: case.name.clone(),
        steps,
        mismatches: Vec::new(),
        declined: None,
        ran: Ok(()),
        controls: Vec::new(),
        launches_per_lane: 0,
        launches_grouped: 0,
    };
    for step in 0..steps {
        let inputs = step_inputs(&mut rng, &g, total_rows);
        upload_step(gpu, &reference, &inputs);
        upload_step(gpu, &candidate, &inputs);
        let specs = specs_for(&case.rows, &pos);
        let (ref_selected, launches) = run_per_lane(gpu, &g, &reference, &specs, true);
        result.launches_per_lane = launches;
        match run_grouped(gpu, &g, &candidate, &specs, Tamper::None) {
            Ok(n) => result.launches_grouped = n,
            Err(e) => {
                eprintln!("  {}: step {step}: {e}", case.name);
                result.declined = Some(step);
                break;
            }
        }
        gpu.hip.device_synchronize().expect("sync");
        if let Err(e) = reference_ran(gpu, &reference, &specs) {
            result.ran = Err(format!("step {step}: {e}"));
        }
        for (what, d) in compare(
            gpu,
            &reference,
            &candidate,
            &specs,
            &ref_selected,
            g.scratch_rows,
        ) {
            result.mismatches.push((step, what, d));
        }
        if step == 0 && controls {
            run_controls(
                gpu,
                &g,
                case,
                seed,
                &reference,
                &candidate,
                &specs,
                &inputs,
                &ref_selected,
                &mut result,
            );
        }
        for (p, r) in pos.iter_mut().zip(&case.rows) {
            *p += r;
        }
    }
    free_arm(gpu, reference);
    free_arm(gpu, candidate);
    result
}

/// The negative controls of a case, right after step 0 (the reference is at its after-step-0
/// state): a flipped bit in the candidate's downloaded bytes must be exactly one differing
/// byte, and every tampered grouped launch on a fresh arm must differ from the reference.
#[allow(clippy::too_many_arguments)]
fn run_controls(
    gpu: &mut Gpu,
    g: &Geo,
    case: &Case,
    seed: u64,
    reference: &Arm,
    candidate: &Arm,
    specs: &[Spec],
    inputs: &StepInputs,
    ref_selected: &[Vec<u8>],
    result: &mut CaseResult,
) {
    let n = specs.len();
    let last = n - 1;
    // Control 1: one flipped bit, per tensor family of the candidate's last lane.
    let flip = |bytes: &[u8], at: usize| {
        let mut c = bytes.to_vec();
        c[at] ^= 0x04;
        c
    };
    for (what, r, c) in [
        (
            "pooled keys",
            &reference.lanes[last].pooled,
            &candidate.lanes[last].pooled,
        ),
        (
            "persistent selection",
            &reference.lanes[last].persistent,
            &candidate.lanes[last].persistent,
        ),
        ("qsa output", &reference.out, &candidate.out),
        (
            "K cache",
            &reference.lanes[last].full_keys,
            &candidate.lanes[last].full_keys,
        ),
    ] {
        let rb = download(gpu, r);
        let cb = download(gpu, c);
        assert_eq!(
            diff_bytes(&rb, &cb),
            0,
            "the control starts from equal bytes ({what})"
        );
        let at = if what == "qsa output" {
            (GUARD + specs[last].row0) * OUTW * 4 + 4097
        } else if what == "K cache" {
            specs[last].pos * g.fmt.row_bytes() + 7
        } else {
            rb.len() / 3
        };
        let d = diff_bytes(&rb, &flip(&cb, at));
        result
            .controls
            .push((format!("one flipped bit in the {what}"), d == 1, d));
    }
    // Controls 2..: the tampered grouped launch on a fresh arm.
    let mut tampers = vec![Tamper::WrongPosition(last)];
    if n >= 2 {
        tampers.push(Tamper::SwapState(0, last));
        tampers.push(Tamper::SwapScratch(0, last));
        tampers.push(Tamper::DropLast);
    }
    for tamper in tampers {
        let (mut fresh, _) = build_arms(gpu, g, case, seed, 1);
        let fresh = fresh.pop().unwrap();
        upload_step(gpu, &fresh, inputs);
        let differing = match run_grouped(gpu, g, &fresh, specs, tamper) {
            Ok(_) => {
                gpu.hip.device_synchronize().expect("sync");
                compare(gpu, reference, &fresh, specs, ref_selected, g.scratch_rows)
                    .iter()
                    .map(|(_, d)| d)
                    .sum()
            }
            // A declined tampered set is not a detection by the oracle.
            Err(_) => 0,
        };
        result
            .controls
            .push((tamper.name(), differing > 0, differing));
        free_arm(gpu, fresh);
    }
}

/// A set holding a lane that completes no pooled block is declined (the per-lane path takes
/// the fused select there), and so is a set with a rows-9 lane; nothing launched.
fn declines(gpu: &mut Gpu, fmt: Fmt, seed: u64) -> Vec<(String, bool)> {
    let mut out = Vec::new();
    for (name, ctxs, rows) in [
        (
            "a lane completing no block (ctx 1, 1 row)",
            vec![100usize, 1],
            vec![1usize, 1],
        ),
        ("a lane of more than 8 rows", vec![100, 100], vec![1, 9]),
    ] {
        let case = Case {
            name: name.into(),
            fmt,
            ctxs: ctxs.clone(),
            rows: rows.clone(),
            ties: false,
            controls: false,
        };
        let g = geo_for(&case, 1);
        let (mut arms, mut rng) = build_arms(gpu, &g, &case, seed, 1);
        let arm = arms.pop().unwrap();
        upload_step(gpu, &arm, &step_inputs(&mut rng, &g, rows.iter().sum()));
        let specs = specs_for(&rows, &ctxs);
        let declined = run_grouped(gpu, &g, &arm, &specs, Tamper::None).is_err();
        out.push((format!("{}: {name} is declined", fmt.name()), declined));
        free_arm(gpu, arm);
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.len() % 2 == 1 {
        v[v.len() / 2]
    } else {
        0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2])
    }
}

/// One full-model QSA pass of a step (12 layers, each with its own state), E4M3-FNUZ state:
/// the per-lane launch chain against the batched launches, admission included. A pass re-runs
/// the same step at the same positions (idempotent: same inputs, same bytes).
fn time_case(gpu: &mut Gpu, lanes: usize, ctx: usize, reps: usize, warm: usize, seed: u64) {
    let fmt = Fmt::Fnuz;
    let case = Case {
        name: "time".into(),
        fmt,
        ctxs: vec![ctx; lanes],
        rows: vec![1; lanes],
        ties: false,
        controls: false,
    };
    let g = geo_for(&case, 1);
    let mut rng = Rng(seed ^ 0x7131 ^ ((lanes as u64) << 24) ^ ((ctx as u64) << 40));
    // One random host state per lane, uploaded into every layer's own tensors.
    let host = host_state(fmt, &mut rng, &case.ctxs, g.max_seq, g.pooled_cap, false);
    let w = weights(&mut rng);
    let layers: Vec<Vec<LaneState>> = (0..QSA_LAYERS)
        .map(|_| host.iter().map(|h| upload_lane(gpu, &g, h)).collect())
        .collect();
    drop(host);
    let mut arm = make_arm(gpu, &g, Vec::new(), &w);
    let inputs = step_inputs(&mut rng, &g, lanes);
    upload_step(gpu, &arm, &inputs);
    let specs = specs_for(&case.rows, &case.ctxs);
    let vs: Vec<Views> = specs.iter().map(|s| views(&arm, s)).collect();
    let vs_per_lane: Vec<Views> = specs
        .iter()
        .map(|s| views(&arm, &Spec { sel0: 0, ..*s }))
        .collect();
    let scrub = gpu.alloc_tensor(&[1usize << 28], DType::F32).unwrap();
    let e0 = gpu.hip.event_create().unwrap();
    let e1 = gpu.hip.event_create().unwrap();
    let pass = |gpu: &mut Gpu, batched: bool| -> (f64, f64) {
        gpu.bind_thread().unwrap();
        gpu.hip.memset(&scrub.buf, 0x5A, scrub.byte_size()).unwrap();
        gpu.hip.device_synchronize().unwrap();
        let wall = std::time::Instant::now();
        gpu.hip.event_record(&e0, None).unwrap();
        for layer in &layers {
            if batched {
                let lane_args: Vec<QsaCoreLane<'_>> = (0..lanes)
                    .map(|l| core_lane(&g, &arm.weights, &layer[l], &vs[l], specs[l].pos, 1))
                    .collect();
                let plan = qsa_core_lanes_gfx942_min(gpu, &lane_args, 1)
                    .unwrap()
                    .expect("admitted");
                gpu.try_gfx942()
                    .unwrap()
                    .qwen4_qsa_lanes_core(&plan)
                    .unwrap();
            } else {
                for l in 0..lanes {
                    let lane = core_lane(
                        &g,
                        &arm.weights,
                        &layer[l],
                        &vs_per_lane[l],
                        specs[l].pos,
                        1,
                    );
                    indexed_attention_decode_prologue(gpu, &lane.prologue).unwrap();
                    if let Some(pool) = &lane.pool {
                        indexed_attention_pool_rope_incremental(gpu, pool).unwrap();
                    }
                    indexed_attention_select_batch_mirrored(
                        gpu,
                        &lane.select,
                        lane.persistent_selection,
                    )
                    .unwrap();
                    indexed_attention_attention_batch_with_arena(gpu, &lane.attention, None)
                        .unwrap();
                }
            }
        }
        gpu.hip.event_record(&e1, None).unwrap();
        gpu.hip.event_synchronize(&e1).unwrap();
        let event_ms = gpu.hip.event_elapsed_ms(&e0, &e1).unwrap() as f64;
        (event_ms, wall.elapsed().as_secs_f64() * 1e3)
    };
    let mut results = Vec::new();
    for batched in [false, true] {
        for _ in 0..warm {
            pass(gpu, batched);
        }
        let samples: Vec<(f64, f64)> = (0..reps).map(|_| pass(gpu, batched)).collect();
        let mut ev: Vec<f64> = samples.iter().map(|s| s.0).collect();
        let mut wl: Vec<f64> = samples.iter().map(|s| s.1).collect();
        let (mn, mx) = (
            ev.iter().cloned().fold(f64::MAX, f64::min),
            ev.iter().cloned().fold(0.0, f64::max),
        );
        results.push((median(&mut ev), mn, mx, median(&mut wl)));
    }
    println!(
        "[time lanes={lanes} ctx={ctx} fnuz layers={QSA_LAYERS}] full-model QSA pass ms median \
         [min..max] over {reps} reps (warm {warm}, 1 GiB scrub per rep), event / host wall: \
         per-lane launches {:.3} [{:.3}..{:.3}] / {:.3} ({} launches) | batched launches \
         {:.3} [{:.3}..{:.3}] / {:.3} ({} launches, admission included) | speedup {:.2}x (event)",
        results[0].0,
        results[0].1,
        results[0].2,
        results[0].3,
        QSA_LAYERS * lanes * 5,
        results[1].0,
        results[1].1,
        results[1].2,
        results[1].3,
        QSA_LAYERS * 5,
        results[0].0 / results[1].0,
    );
    gpu.hip.event_destroy(e0).ok();
    gpu.hip.event_destroy(e1).ok();
    drop(vs);
    drop(vs_per_lane);
    for layer in layers {
        for l in layer {
            for t in [l.full_keys, l.full_values, l.raw, l.pooled, l.persistent] {
                let _ = gpu.free_tensor(t);
            }
        }
    }
    arm.lanes.clear();
    free_arm(gpu, arm);
    gpu.free_tensor(scrub).ok();
}

struct Cli {
    formats: Vec<Fmt>,
    lanes: Vec<usize>,
    ctxs: Vec<usize>,
    steps: usize,
    seed: u64,
    quick: bool,
    only: Option<String>,
    controls: bool,
    correctness: bool,
    time: Vec<usize>,
    time_ctxs: Vec<usize>,
    reps: usize,
    warm: usize,
}

fn csv(v: &str) -> Vec<usize> {
    v.split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("integer list"))
        .collect()
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        formats: vec![Fmt::F32, Fmt::Fnuz],
        lanes: vec![1, 8, 64],
        ctxs: vec![64, 2048, 16384],
        steps: 2,
        seed: 0x5153_4139_3432_4c33,
        quick: false,
        only: None,
        controls: true,
        correctness: true,
        time: vec![],
        time_ctxs: vec![2048, 16384],
        reps: 9,
        warm: 3,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--formats" => {
                cli.formats = value()
                    .split(',')
                    .map(|f| match f {
                        "f32" => Fmt::F32,
                        "fnuz" => Fmt::Fnuz,
                        other => panic!("unknown format {other}"),
                    })
                    .collect()
            }
            "--lanes" => cli.lanes = csv(&value()),
            "--ctxs" => cli.ctxs = csv(&value()),
            "--steps" => cli.steps = value().parse().expect("steps"),
            "--seed" => cli.seed = value().parse().expect("seed"),
            "--quick" => cli.quick = true,
            "--only" => cli.only = Some(value()),
            "--skip-controls" => cli.controls = false,
            "--skip-correctness" => cli.correctness = false,
            "--time" => cli.time = csv(&value()),
            "--time-ctxs" => cli.time_ctxs = csv(&value()),
            "--reps" => cli.reps = value().parse().expect("reps"),
            "--warm" => cli.warm = value().parse().expect("warm"),
            other => panic!("unknown argument {other}"),
        }
    }
    cli
}

fn main() {
    let cli = parse_cli();
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.try_gfx942().is_none() {
        eprintln!(
            "qsa_lanes_gfx942_oracle: needs exact gfx942, this device is {}",
            gpu.arch
        );
        std::process::exit(2);
    }
    // The opt-in flag, for this process only.
    let mut flags = (*gpu.flags).clone();
    flags.qwen4_qsa_lanes = true;
    gpu.flags = Arc::new(flags);
    assert!(gpu.flags.qwen4_qsa_lanes_enabled());
    println!(
        "device {} | QSA lanes flag on | {NH} heads x {NKV} KV x {HD}, indexer {IDX_HEADS} x {IDX_DIM}, \
         compress {COMPRESS}, budget {BUDGET}, capacity {CAP} | steps {}",
        gpu.arch, cli.steps
    );
    let mut failures = 0usize;
    if cli.correctness {
        for &fmt in &cli.formats {
            for (what, ok) in declines(&mut gpu, fmt, cli.seed) {
                println!("{} {what}", if ok { "ok  " } else { "FAIL" });
                failures += usize::from(!ok);
            }
        }
        let all: Vec<Case> = cases(&cli.formats, &cli.lanes, &cli.ctxs, cli.quick)
            .into_iter()
            .filter(|c| cli.only.as_ref().is_none_or(|o| c.name.contains(o)))
            .collect();
        println!("{} cases x {} steps", all.len(), cli.steps);
        for (i, case) in all.iter().enumerate() {
            let t = std::time::Instant::now();
            let r = run_case(
                &mut gpu,
                case,
                cli.steps,
                cli.seed ^ (i as u64) << 8,
                cli.controls && case.controls,
            );
            let ok = r.mismatches.is_empty() && r.declined.is_none() && r.ran.is_ok();
            failures += usize::from(!ok);
            println!(
                "{} {} (steps {}, launches per-lane {} -> grouped {}, state diff bytes {}, {:.1}s){}{}",
                if ok { "ok  " } else { "FAIL" },
                r.name,
                r.steps,
                r.launches_per_lane,
                r.launches_grouped,
                r.mismatches.iter().map(|m| m.2).sum::<usize>(),
                t.elapsed().as_secs_f64(),
                match (&r.declined, r.mismatches.first()) {
                    (Some(s), _) => format!(" declined at step {s}"),
                    (None, Some((s, what, d))) => format!(" first mismatch: step {s} {what} ({d} bytes)"),
                    _ => String::new(),
                },
                match &r.ran {
                    Err(e) => format!(" reference did not run: {e}"),
                    Ok(()) => String::new(),
                },
            );
            for (name, detected, d) in &r.controls {
                println!(
                    "{}   control: {name}: {d} differing bytes ({})",
                    if *detected { "ok  " } else { "FAIL" },
                    if name.starts_with("one flipped") {
                        "want exactly 1"
                    } else {
                        "must be detected, want > 0"
                    }
                );
                failures += usize::from(!detected);
            }
        }
    }
    for &lanes in &cli.time {
        for &ctx in &cli.time_ctxs {
            time_case(&mut gpu, lanes, ctx, cli.reps, cli.warm, cli.seed);
        }
    }
    println!(
        "qsa_lanes_gfx942_oracle: {}",
        if failures == 0 { "PASS" } else { "FAIL" }
    );
    std::process::exit(i32::from(failures != 0));
}
