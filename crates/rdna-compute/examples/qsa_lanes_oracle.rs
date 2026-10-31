// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the multi-lane QSA decode core (`rdna_compute::tensor_ops::
//! qsa_core_lanes`, kernels `qsa_lanes.gfx1201.hip`) against the per-lane
//! singleton launches, exact gfx1201, fp8 K/V.
//!
//! Synthetic Flash-Next QSA layer: 24 query heads, 2 KV heads, head_dim 256,
//! indexer 4 heads x 128 + 1 KV head, compress 4, budget 2048 tokens (512
//! blocks), selected capacity 2051, fp8 KV rows of 516 B per token, yarn RoPE.
//! N lanes (1..=8) with DIFFERENT contexts per lane (a `complete == 0` lane,
//! lanes right before / after a 4-token block boundary, ctx 61 / 64 / 65, ctx
//! 1000, budget-saturating 2000..2102 and 3000, lanes in different orders) and
//! 1 row per lane (AR decode) or 2..=8 rows per lane (verify windows, uniform
//! and ragged).
//!
//! - reference (per lane): exactly the four launches `layer_ops.rs`
//!   `execute_indexed_attention_hc_impl` issues for a Core step:
//!   `indexed_attention_decode_prologue`, `indexed_attention_pool_rope_incremental`
//!   (when `(position + rows) / compress > 0`), `indexed_attention_select_batch_mirrored`
//!   and `indexed_attention_attention_batch_with_arena`, on the lane's rows of
//!   one shared row scratch and one shared selected scratch (row 0, as the
//!   lane forward does);
//! - candidate (grouped): `qsa_core_lanes` + `QsaCoreLanesPlan::launch` over
//!   the same arguments (each lane's selected rows in its own range).
//!
//! Compared as BYTES after every step of several consecutive steps with the
//! state carried (positions advance by the lane's rows): every lane's whole fp8
//! K and V arena (written rows AND untouched prefix / tail), raw index keys,
//! pooled keys, persistent selected indices, the shared index / q+gate / K / V /
//! output row scratch (sentinel-initialised, with guard rows around the lanes'
//! rows), and each lane's selected scratch rows (the tail of the grouped scratch
//! must stay sentinel). Negative controls the oracle MUST detect: two lanes'
//! state pointers swapped, a wrong position on one lane, two lanes' scratch rows
//! swapped. Reports launch counts (per-lane vs grouped) and the event time per
//! replay of an N = 8 step per variant (50 replays).
//!
//! `HIPFIRE_QWEN4_QSA_LANES` need not be set: the oracle turns the flag on in
//! its own `Gpu`. Exit 0 and `PASS qsa_lanes_oracle ...` only if every case and
//! every control pass; `--out DIR` writes `qsa_lanes_oracle.json`.
//!
//! `cargo run --release -p rdna-compute --features lab --example qsa_lanes_oracle -- [--out DIR] [--steps N] [--replays N] [--quick]`

use rdna_compute::rope::RopeYarn;
use rdna_compute::tensor_ops::{
    indexed_attention_attention_batch_with_arena, indexed_attention_decode_prologue,
    indexed_attention_pool_rope_incremental, indexed_attention_select_batch_mirrored,
    qsa_core_lanes_min, IndexedAttentionAttentionBatch, IndexedAttentionDecodePrologue,
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
const MAX_SEQ: usize = 4096;
const POOLED_CAP: usize = MAX_SEQ / COMPRESS;
const ROW_BYTES: usize = NKV * (HD + 2);
/// Index row: `[q (4 x 128) | k (1 x 128)]`.
const IDXW: usize = (IDX_HEADS + 1) * IDX_DIM;
const QGW: usize = 2 * NH * HD;
const KVW: usize = NKV * HD;
const OUTW: usize = NH * HD;
/// Guard rows around the lanes' rows in every row scratch.
const GUARD: usize = 2;
/// Row scratch capacity in rows (8 lanes x 8 rows) beyond the guards.
const SCRATCH_ROWS: usize = 64;
const SENTINEL_F32: u32 = 0x7fc0_dead;
const SENTINEL_I32: u32 = 0x7f7f_7f7e;
const SENTINEL_PERSIST: u32 = 0x7e7e_7e7d;

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

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) as u16
}

fn bf16_vec(rng: &mut Rng, n: usize, amp: f32, offset: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16_bits(offset + rng.sym() * amp).to_le_bytes())
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Random valid fp8 rows: E4M3 codes without the NaN codes and f16 scales in
/// [2^-7, 2^-3) for rows `[0, filled)`; zero elsewhere.
fn kv_cache(rng: &mut Rng, filled: usize) -> Vec<u8> {
    let mut c = vec![0u8; MAX_SEQ * ROW_BYTES];
    for pos in 0..filled {
        let row = &mut c[pos * ROW_BYTES..(pos + 1) * ROW_BYTES];
        let (codes, scales) = row.split_at_mut(NKV * HD);
        for chunk in codes.chunks_mut(8) {
            let w = rng.next().to_le_bytes();
            for (b, r) in chunk.iter_mut().zip(w) {
                *b = if r & 0x7f == 0x7f { r & 0xbf } else { r };
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

/// Host initial state of one lane.
struct LaneHost {
    k_cache: Vec<u8>,
    v_cache: Vec<u8>,
    raw: Vec<u8>,
    pooled: Vec<u8>,
}

fn lane_host(rng: &mut Rng, ctx: usize, ties: bool) -> LaneHost {
    let blocks = ctx / COMPRESS;
    let mut raw = vec![0u8; MAX_SEQ * IDX_DIM * 2];
    raw[..ctx * IDX_DIM * 2].copy_from_slice(&bf16_vec(rng, ctx * IDX_DIM, 1.0, 0.0));
    let mut pooled = vec![0u8; POOLED_CAP * IDX_DIM * 2];
    pooled[..blocks * IDX_DIM * 2].copy_from_slice(&bf16_vec(rng, blocks * IDX_DIM, 1.0, 0.0));
    if ties {
        // Equal-score blocks exercise the (score desc, index asc) rank ties.
        let row = IDX_DIM * 2;
        for b in 5..blocks {
            let (head, tail) = pooled.split_at_mut(b * row);
            tail[..row].copy_from_slice(&head[(b % 5) * row..(b % 5 + 1) * row]);
        }
    }
    LaneHost {
        k_cache: kv_cache(rng, ctx),
        v_cache: kv_cache(rng, ctx),
        raw,
        pooled,
    }
}

/// Weights shared by every lane of an arm.
struct Weights {
    index_q_norm: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
    pool_norm: GpuTensor,
}

fn upload_typed(gpu: &mut Gpu, bytes: &[u8], elements: usize, dtype: DType) -> GpuTensor {
    let t = gpu.alloc_tensor(&[elements], dtype).expect("alloc");
    assert_eq!(t.byte_size(), bytes.len(), "upload size");
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
    t
}

/// One lane's device state.
struct LaneState {
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw: GpuTensor,
    pooled: GpuTensor,
    persistent: GpuTensor,
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

fn sentinel_f32_rows(rows: usize, width: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL_F32); rows * width]
}

fn make_arm(gpu: &mut Gpu, host: &[LaneHost], weights: &[Vec<u8>; 4]) -> Arm {
    let lanes = host
        .iter()
        .map(|h| LaneState {
            full_keys: upload_typed(gpu, &h.k_cache, h.k_cache.len(), DType::Raw),
            full_values: upload_typed(gpu, &h.v_cache, h.v_cache.len(), DType::Raw),
            raw: upload_typed(gpu, &h.raw, MAX_SEQ * IDX_DIM, DType::BF16),
            pooled: upload_typed(gpu, &h.pooled, POOLED_CAP * IDX_DIM, DType::BF16),
            // The persistent selection starts as a recognisable non-(-1) pattern.
            persistent: upload_typed(
                gpu,
                &(0..CAP)
                    .flat_map(|_| SENTINEL_PERSIST.to_le_bytes())
                    .collect::<Vec<_>>(),
                CAP * 4,
                DType::Raw,
            ),
        })
        .collect::<Vec<_>>();
    let rows = SCRATCH_ROWS + 2 * GUARD;
    let scratch = |gpu: &mut Gpu, w: usize| {
        gpu.upload_f32(&sentinel_f32_rows(rows, w), &[rows * w])
            .expect("scratch")
    };
    let selected = upload_typed(
        gpu,
        &(0..SCRATCH_ROWS * CAP)
            .flat_map(|_| SENTINEL_I32.to_le_bytes())
            .collect::<Vec<_>>(),
        SCRATCH_ROWS * CAP * 4,
        DType::Raw,
    );
    Arm {
        lanes,
        weights: Weights {
            index_q_norm: upload_typed(gpu, &weights[0], IDX_DIM, DType::BF16),
            q_norm: upload_typed(gpu, &weights[1], HD, DType::BF16),
            k_norm: upload_typed(gpu, &weights[2], HD, DType::BF16),
            pool_norm: upload_typed(gpu, &weights[3], IDX_DIM, DType::BF16),
        },
        index: scratch(gpu, IDXW),
        qgate: scratch(gpu, QGW),
        k: scratch(gpu, KVW),
        v: scratch(gpu, KVW),
        out: scratch(gpu, OUTW),
        selected,
    }
}

/// Host-side step inputs (the projections' outputs), shared by both arms.
struct StepInputs {
    index: Vec<f32>,
    qgate: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn step_inputs(rng: &mut Rng, total_rows: usize) -> StepInputs {
    let fill = |rng: &mut Rng, w: usize| -> Vec<f32> {
        let mut v = sentinel_f32_rows(SCRATCH_ROWS + 2 * GUARD, w);
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
        qgate: arm.qgate.sub_offset(g * QGW, s.rows * 2 * NH * HD),
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

/// The lane's four launch argument sets, as the interpreter builds them.
fn core_lane<'a>(
    arm: &'a Arm,
    state: &'a LaneState,
    v: &'a Views,
    pos: usize,
    rows: usize,
) -> QsaCoreLane<'a> {
    let complete = (pos + rows) / COMPRESS;
    let w = &arm.weights;
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
            format: QsaKvFormat::Fp8,
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
            grid_bound: POOLED_CAP,
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
            shape_blocks: POOLED_CAP,
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
            full_capacity: MAX_SEQ,
            format: QsaKvFormat::Fp8,
            shape_selected: CAP,
        },
    }
}

/// Per-lane launches of one step; with `capture`, also each lane's selected
/// rows right after its select (the shared scratch is overwritten by the next
/// lane; the download synchronises, so timing runs pass `false`). Returns the
/// captures and the launch count.
fn run_per_lane(gpu: &mut Gpu, arm: &Arm, specs: &[Spec], capture: bool) -> (Vec<Vec<u8>>, usize) {
    let mut launches = 0;
    let mut selected = Vec::new();
    for (i, s) in specs.iter().enumerate() {
        let s0 = Spec { sel0: 0, ..*s };
        let v = views(arm, &s0);
        let lane = core_lane(arm, &arm.lanes[i], &v, s.pos, s.rows);
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
        launches += 1;
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
}

/// Grouped launch of one step. `Ok(launches)`; `Err` when admission declined.
fn run_grouped(gpu: &mut Gpu, arm: &Arm, specs: &[Spec], tamper: Tamper) -> Result<usize, String> {
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
    let lanes: Vec<QsaCoreLane<'_>> = specs
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let pos = match tamper {
                Tamper::WrongPosition(a) if a == i => s.pos + 1,
                _ => s.pos,
            };
            let rows = s.rows;
            // The tampered scratch lane keeps its own selected range.
            let mut lane = core_lane(arm, &arm.lanes[state_of(i)], &vs[scratch_of(i)], pos, rows);
            if scratch_of(i) != i {
                lane.select.selected = &vs[i].selected;
                lane.attention.selected = &vs[i].selected;
            }
            lane
        })
        .collect();
    let Some(plan) = qsa_core_lanes_min(gpu, &lanes, 1).map_err(|e| e.to_string())? else {
        return Err("qsa_core_lanes declined the set".into());
    };
    let launches = plan.launch_count();
    plan.launch(gpu).map_err(|e| e.to_string())?;
    Ok(launches)
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.download_raw_bytes(t).expect("download")
}

fn diff_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Compare every compared tensor of the two arms; `(name, differing bytes)` of
/// the ones that differ.
fn compare(
    gpu: &Gpu,
    reference: &Arm,
    candidate: &Arm,
    specs: &[Spec],
    ref_selected: &[Vec<u8>],
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
    // Selected scratch rows: the reference's capture right after each lane's
    // select against the candidate's own range; the unused tail stays sentinel.
    let cand = download(gpu, &candidate.selected);
    let mut used = 0;
    for (i, s) in specs.iter().enumerate() {
        let a = &ref_selected[i];
        let b = &cand[s.sel0 * CAP * 4..(s.sel0 + s.rows) * CAP * 4];
        check(format!("lane{i}.selected_scratch"), a, b);
        used = used.max(s.sel0 + s.rows);
    }
    let sentinel: Vec<u8> = (0..(SCRATCH_ROWS - used) * CAP)
        .flat_map(|_| SENTINEL_I32.to_le_bytes())
        .collect();
    check(
        "selected_scratch_tail".into(),
        &sentinel,
        &cand[used * CAP * 4..],
    );
    bad
}

#[derive(Clone)]
struct Case {
    name: String,
    ctxs: Vec<usize>,
    rows: Vec<usize>,
    ties: bool,
}

fn cases(quick: bool) -> Vec<Case> {
    let mixed = vec![1000, 2, 63, 64, 65, 61, 2100, 2000];
    let reversed: Vec<usize> = mixed.iter().rev().copied().collect();
    let rotated: Vec<usize> = (0..8).map(|i| mixed[(i + 3) % 8]).collect();
    let boundary = vec![3, 7, 0, 63, 1, 64, 65, 2];
    let budget = vec![2043, 2044, 2047, 2048, 2051, 2100, 2101, 3000];
    let zero = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let long = vec![3990, 3000, 2500, 1500, 1200, 3500, 3900, 4000];
    // Rows of lane `l`: uniform decode / verify windows and two ragged mixes.
    type RowsMode = (&'static str, fn(usize) -> usize);
    let rows_modes: [RowsMode; 6] = [
        ("r1", |_| 1),
        ("r2", |_| 2),
        ("r4", |_| 4),
        ("r8", |_| 8),
        ("ragged", |l| 1 + (l * 3) % 8),
        ("ragged2", |l| 8 - (l * 5) % 8),
    ];
    let mut out = Vec::new();
    let mut add = |set: &str, ctxs: &Vec<usize>, ns: &[usize], ties: bool, modes: &[usize]| {
        for &n in ns {
            for &m in modes {
                let (mode, f) = &rows_modes[m];
                out.push(Case {
                    name: format!("{set}/N{n}/{mode}{}", if ties { "/ties" } else { "" }),
                    ctxs: ctxs[..n].to_vec(),
                    rows: (0..n).map(f).collect(),
                    ties,
                });
            }
        }
    };
    let all_modes: Vec<usize> = (0..rows_modes.len()).collect();
    if quick {
        add("mixed", &mixed, &[1, 2, 8], false, &[0, 1, 4]);
        add("budget", &budget, &[8], true, &[0, 3]);
    } else {
        add(
            "mixed",
            &mixed,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            false,
            &all_modes,
        );
        add("mixed-reversed", &reversed, &[2, 5, 8], false, &all_modes);
        add("mixed-rotated", &rotated, &[3, 8], false, &all_modes);
        add("boundary", &boundary, &[2, 5, 8], false, &all_modes);
        add("budget", &budget, &[2, 8], false, &all_modes);
        add("budget", &budget, &[8], true, &[0, 1, 4]);
        add("zero", &zero, &[2, 4, 8], false, &[0, 1, 4]);
        add("long", &long, &[2, 8], false, &[0, 3, 5]);
    }
    out
}

struct CaseResult {
    name: String,
    steps: usize,
    mismatches: Vec<(usize, String, usize)>,
    declined: Option<usize>,
    launches_per_lane: usize,
    launches_grouped: usize,
}

fn weights(rng: &mut Rng) -> [Vec<u8>; 4] {
    [
        bf16_vec(rng, IDX_DIM, 0.3, 0.0),
        bf16_vec(rng, HD, 0.3, 0.0),
        bf16_vec(rng, HD, 0.3, 0.0),
        bf16_vec(rng, IDX_DIM, 0.3, 0.0),
    ]
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

fn specs_for(rows: &[usize], pos: &[usize]) -> Vec<Spec> {
    let mut row0 = 0;
    let mut sel0 = 0;
    pos.iter()
        .zip(rows)
        .map(|(&p, &r)| {
            let s = Spec {
                pos: p,
                rows: r,
                row0,
                sel0,
            };
            row0 += r;
            sel0 += r;
            s
        })
        .collect()
}

fn run_case(gpu: &mut Gpu, case: &Case, steps: usize, seed: u64) -> CaseResult {
    let mut rng = Rng(seed);
    let host: Vec<LaneHost> = case
        .ctxs
        .iter()
        .map(|&c| lane_host(&mut rng, c, case.ties))
        .collect();
    let w = weights(&mut rng);
    let reference = make_arm(gpu, &host, &w);
    let candidate = make_arm(gpu, &host, &w);
    let mut pos = case.ctxs.clone();
    let total_rows: usize = case.rows.iter().sum();
    let mut result = CaseResult {
        name: case.name.clone(),
        steps,
        mismatches: Vec::new(),
        declined: None,
        launches_per_lane: 0,
        launches_grouped: 0,
    };
    for step in 0..steps {
        if pos.iter().zip(&case.rows).any(|(&p, &r)| p + r > MAX_SEQ) {
            result.steps = step;
            break;
        }
        let inputs = step_inputs(&mut rng, total_rows);
        upload_step(gpu, &reference, &inputs);
        upload_step(gpu, &candidate, &inputs);
        let specs = specs_for(&case.rows, &pos);
        let (ref_selected, launches) = run_per_lane(gpu, &reference, &specs, true);
        result.launches_per_lane = launches;
        match run_grouped(gpu, &candidate, &specs, Tamper::None) {
            Ok(n) => result.launches_grouped = n,
            Err(e) => {
                eprintln!("  {}: step {step}: {e}", case.name);
                result.declined = Some(step);
                break;
            }
        }
        for (what, d) in compare(gpu, &reference, &candidate, &specs, &ref_selected) {
            result.mismatches.push((step, what, d));
        }
        for (p, r) in pos.iter_mut().zip(&case.rows) {
            *p += r;
        }
    }
    free_arm(gpu, reference);
    free_arm(gpu, candidate);
    result
}

struct ControlResult {
    name: String,
    detected: bool,
    differing: usize,
}

/// One tampered grouped step on a fresh pair of arms: must differ from the
/// per-lane reference.
fn run_control(gpu: &mut Gpu, name: &str, tamper: Tamper, seed: u64) -> ControlResult {
    let ctxs = vec![1000usize, 2, 63, 64, 65, 61, 2100, 2000];
    let rows = vec![1usize; 8];
    let mut rng = Rng(seed);
    let host: Vec<LaneHost> = ctxs
        .iter()
        .map(|&c| lane_host(&mut rng, c, false))
        .collect();
    let w = weights(&mut rng);
    let reference = make_arm(gpu, &host, &w);
    let candidate = make_arm(gpu, &host, &w);
    let inputs = step_inputs(&mut rng, 8);
    upload_step(gpu, &reference, &inputs);
    upload_step(gpu, &candidate, &inputs);
    let specs = specs_for(&rows, &ctxs);
    let (ref_selected, _) = run_per_lane(gpu, &reference, &specs, true);
    let differing = match run_grouped(gpu, &candidate, &specs, tamper) {
        Ok(_) => compare(gpu, &reference, &candidate, &specs, &ref_selected)
            .iter()
            .map(|(_, d)| d)
            .sum(),
        // A declined tampered set is not a detection of the oracle.
        Err(_) => 0,
    };
    free_arm(gpu, reference);
    free_arm(gpu, candidate);
    ControlResult {
        name: name.to_string(),
        detected: differing > 0,
        differing,
    }
}

struct Timing {
    ctx: usize,
    per_lane_us: f64,
    grouped_us: f64,
    per_lane_launches: usize,
    grouped_launches: usize,
}

fn time_us(gpu: &mut Gpu, replays: usize, mut body: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..5 {
        body(gpu);
    }
    let start = gpu.hip.event_create().expect("start event");
    let stop = gpu.hip.event_create().expect("stop event");
    gpu.hip.event_record(&start, None).expect("record start");
    for _ in 0..replays {
        body(gpu);
    }
    gpu.hip.event_record(&stop, None).expect("record stop");
    gpu.hip.event_synchronize(&stop).expect("sync");
    gpu.hip.event_elapsed_ms(&start, &stop).expect("elapsed") as f64 * 1_000.0 / replays as f64
}

/// N = 8 one-row lanes around `ctx`: event time per replay of the per-lane
/// step vs the grouped step. A replay re-runs the same step at the same
/// positions (idempotent: same inputs, same bytes).
fn run_timing(gpu: &mut Gpu, ctx: usize, replays: usize) -> Timing {
    let ctxs: Vec<usize> = (0..8).map(|l| ctx + l).collect();
    let rows = vec![1usize; 8];
    let mut rng = Rng(0x71_4D_1A + ctx as u64);
    let host: Vec<LaneHost> = ctxs
        .iter()
        .map(|&c| lane_host(&mut rng, c, false))
        .collect();
    let w = weights(&mut rng);
    let arm = make_arm(gpu, &host, &w);
    let inputs = step_inputs(&mut rng, 8);
    upload_step(gpu, &arm, &inputs);
    let specs = specs_for(&rows, &ctxs);
    let (_, per_lane_launches) = run_per_lane(gpu, &arm, &specs, false);
    let grouped_launches = run_grouped(gpu, &arm, &specs, Tamper::None).expect("grouped");
    let per_lane_us = time_us(gpu, replays, |gpu| {
        run_per_lane(gpu, &arm, &specs, false);
    });
    let grouped_us = time_us(gpu, replays, |gpu| {
        run_grouped(gpu, &arm, &specs, Tamper::None).expect("grouped");
    });
    free_arm(gpu, arm);
    Timing {
        ctx,
        per_lane_us,
        grouped_us,
        per_lane_launches,
        grouped_launches,
    }
}

fn main() {
    let mut out_dir: Option<String> = None;
    let mut steps = 5usize;
    let mut replays = 50usize;
    let mut quick = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out_dir = args.next(),
            "--steps" => steps = args.next().and_then(|v| v.parse().ok()).expect("--steps N"),
            "--replays" => {
                replays = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--replays N")
            }
            "--quick" => quick = true,
            other => {
                eprintln!("qsa_lanes_oracle: unknown argument {other}");
                std::process::exit(2);
            }
        }
    }
    let mut gpu = Gpu::init().expect("Gpu::init");
    if gpu.arch != "gfx1201" {
        eprintln!("qsa_lanes_oracle: needs exact gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    // The opt-in flag, for this process only.
    let mut flags = (*gpu.flags).clone();
    flags.qwen4_qsa_lanes = true;
    gpu.flags = Arc::new(flags);
    assert!(gpu.flags.qwen4_qsa_lanes_enabled());
    println!("arch={} qsa lanes flag on", gpu.arch);

    let mut failures = 0usize;
    let mut case_json = Vec::new();
    let all = cases(quick);
    println!("{} cases x {steps} steps", all.len());
    for (i, case) in all.iter().enumerate() {
        let r = run_case(&mut gpu, case, steps, 0x5153_4100 + i as u64);
        let ok = r.mismatches.is_empty() && r.declined.is_none();
        if !ok {
            failures += 1;
        }
        println!(
            "{} {} (steps {}, launches per-lane {} -> grouped {}){}",
            if ok { "ok  " } else { "FAIL" },
            r.name,
            r.steps,
            r.launches_per_lane,
            r.launches_grouped,
            match (&r.declined, r.mismatches.first()) {
                (Some(s), _) => format!(" declined at step {s}"),
                (None, Some((s, w, d))) => {
                    format!(
                        " first mismatch step {s} {w}: {d} bytes ({} in all)",
                        r.mismatches.len()
                    )
                }
                _ => String::new(),
            }
        );
        case_json.push(serde_json::json!({
            "name": r.name,
            "steps": r.steps,
            "pass": ok,
            "declined_at_step": r.declined,
            "launches_per_lane": r.launches_per_lane,
            "launches_grouped": r.launches_grouped,
            "mismatches": r.mismatches.iter().map(|(s, w, d)| serde_json::json!({"step": s, "tensor": w, "bytes": d})).collect::<Vec<_>>(),
        }));
    }

    let controls = [
        ("swap-state(0,6)", Tamper::SwapState(0, 6)),
        ("wrong-position(3)", Tamper::WrongPosition(3)),
        ("swap-scratch(1,2)", Tamper::SwapScratch(1, 2)),
    ];
    let mut control_json = Vec::new();
    let mut controls_ok = true;
    for (i, (name, tamper)) in controls.iter().enumerate() {
        let c = run_control(&mut gpu, name, *tamper, 0xC0DE_0000 + i as u64);
        println!(
            "{} control {}: {} differing bytes",
            if c.detected { "ok  " } else { "FAIL" },
            c.name,
            c.differing
        );
        controls_ok &= c.detected;
        control_json.push(serde_json::json!({"name": c.name, "detected": c.detected, "differing_bytes": c.differing}));
    }

    let mut timing_json = Vec::new();
    for ctx in [64usize, 1024, 2100] {
        let t = run_timing(&mut gpu, ctx, replays);
        println!(
            "timing N=8 ctx~{}: per-lane {:.1} us ({} launches) vs grouped {:.1} us ({} launches), {:.2}x",
            t.ctx,
            t.per_lane_us,
            t.per_lane_launches,
            t.grouped_us,
            t.grouped_launches,
            t.per_lane_us / t.grouped_us
        );
        timing_json.push(serde_json::json!({
            "ctx": t.ctx,
            "lanes": 8,
            "replays": replays,
            "per_lane_us": t.per_lane_us,
            "grouped_us": t.grouped_us,
            "per_lane_launches": t.per_lane_launches,
            "grouped_launches": t.grouped_launches,
        }));
    }

    let pass = failures == 0 && controls_ok;
    println!(
        "{} qsa_lanes_oracle: {} / {} cases pass, controls {}",
        if pass { "PASS" } else { "FAIL" },
        all.len() - failures,
        all.len(),
        if controls_ok {
            "detected"
        } else {
            "NOT DETECTED"
        }
    );
    if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir).expect("create --out dir");
        let report = serde_json::json!({
            "oracle": "qsa_lanes_oracle",
            "arch": gpu.arch,
            "pass": pass,
            "steps": steps,
            "cases": case_json,
            "controls": control_json,
            "timing": timing_json,
        });
        std::fs::write(
            std::path::Path::new(&dir).join("qsa_lanes_oracle.json"),
            serde_json::to_string_pretty(&report).unwrap(),
        )
        .expect("write json");
    }
    std::process::exit(if pass { 0 } else { 1 });
}
