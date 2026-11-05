// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle of the split MTP draft attention (exact gfx1201, Q8_0 projections) against the
//! per-lane whole op.
//!
//! A batched MTP draft round of `n` lanes used to emit one `Step::IndexedAttention` op with
//! `MixerPhase::Whole` per lane, each running its five Q8_0 projections as rows == 1
//! `gemv_q8_0` singletons. The split emits ONE `MixerPhase::InProj` op over all `n` rows (Q8_0
//! `gemv_q8_0_rows_exact`, bitwise the singleton), then one `MixerPhase::Core` op per lane (one
//! row, the lane's own state / cache, `Full` or `ReuseSelection`) on that lane's row of the
//! `n`-row projection scratch, then ONE `MixerPhase::OutProj` op over all `n` rows. This example
//! drives both through the real dispatch entry points `validate_steps` /
//! `execute_validated_steps` with `DispatchCtx::new(&gpu)` and compares BYTES after every round:
//!
//! * reference: per lane ONE `Whole` op (rows 1) on ONE shared one-row scratch set, the lane's
//!   activation row in and its row of an `[n x 2560]` output out, all in one step list;
//! * candidate: `[InProj(n rows), Core x n, OutProj(n rows)]` in one step list; the Core ops'
//!   index / q+gate / K / V / attention-output scratch are `sub_offset` views at the lane's row
//!   of the `n`-row buffers, their selection scratch is one shared one-row buffer (like
//!   `mtp_gpu/rows.rs`; `--wide-selected-proj` gives the InProj / OutProj ops an `n`-row
//!   selection scratch instead of the shared one-row buffer).
//!
//! Synthetic Flash-Next MTP head QSA geometry: input width 2560, 24 query heads, 2 KV heads,
//! head_dim 256, indexer 4 heads + 1 KV head x 128, compress 4, budget 2048, selected capacity
//! 2051, rotary_dim 64 with a yarn rope, full capacity 4096, exact F32 K/V (`QsaKvFormat::F32`),
//! no F16 arena, no pooled BF16 mirror, `trunk_a4 = 0`. Weights are random raw Q8_0 (34-byte
//! blocks), norms random F32-around-1.0 stored as BF16. Every lane holds random finite caches
//! over its committed prefix and a fixed NaN sentinel past it; each arm gets its OWN identical
//! copy of every lane state. Every lane's `full_keys`, `full_values`, `raw_index_keys`,
//! `pooled_keys`, `selected_indices` (whole buffers, sentinel tails included),
//! `selected_len_out`, the committed host lengths and the `[n x 2560]` attention output are
//! compared as bytes after every round.
//!
//! Cases: (a) n=2 at the SAME position with different caches / activations, round 1 Full then 4
//! ReuseSelection rounds (a 4-token block boundary is crossed); (b) n=2 equal positions with
//! mixed rounds (lane 0 Full / lane 1 Reuse and the converse); (c) n=8 at positions
//! {1001,1001,1002,63,64,65,3000,1001}, round 1 Full then 3 Reuse rounds; (d) n=3 at positions
//! 3, 4, 5, Full only.
//!
//! Negative controls (each MUST produce a mismatch or a launch error, run once on case (a)):
//! (1) candidate lane 1's Core op bound to lane 0's state; (2) candidate lanes 0 / 1 Core
//! scratch row views swapped; (3) candidate lane 1 position (and lengths) off by one.
//!
//! ```text
//! cargo run --release -p hipfire-dispatch --features test-utils \
//!   --example mtp_proj_batch_oracle -- [--out DIR] [--wide-selected-proj]
//! ```
//!
//! Prints per case / round PASS / FAIL lines and a final `PASS mtp_proj_batch_oracle` only if
//! every comparison is equal and every control was detected. Exit 1 on any mismatch or
//! undetected control, exit 2 on a non-gfx1201 device or a launch error. `--out DIR` writes the
//! lines as `mtp_proj_batch_oracle.txt`.

use std::fmt::Debug;
use std::path::PathBuf;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_validated_steps, validate_steps, IndexedAttentionMode, IndexedAttentionOp,
    IndexedAttentionState, MixerPhase, Step,
};
use rdna_compute::rope::{Rope, RopeScaling, Yarn};
use rdna_compute::tensor_ops::QsaKvFormat;
use rdna_compute::{DType, Gpu, GpuTensor};

type R<T> = Result<T, String>;

// ---- geometry ------------------------------------------------------------------------------

const HIDDEN: usize = 2560;
const HEADS: usize = 24;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const INDEX_HEADS: usize = 4;
const INDEX_KV_HEADS: usize = 1;
const INDEX_DIM: usize = 128;
const COMPRESS: usize = 4;
const BUDGET: usize = 2048;
const ROTARY_DIM: usize = 64;
const SELECTED_CAP: usize = BUDGET + COMPRESS - 1;
const FULL_CAP: usize = 4096;
const POOLED_CAP: usize = FULL_CAP / COMPRESS;
const MAX_LANES: usize = 8;

const INDEX_W: usize = (INDEX_HEADS + INDEX_KV_HEADS) * INDEX_DIM;
const INDEX_KV_W: usize = INDEX_KV_HEADS * INDEX_DIM;
const Q_W: usize = HEADS * HEAD_DIM;
const QGATE_W: usize = 2 * Q_W;
const KV_W: usize = KV_HEADS * HEAD_DIM;

const SENTINEL: u32 = 0x7fc0_beef;
const SEL_SENTINEL: i32 = -1;
const SEL_LEN_SENTINEL: [u8; 4] = [0xef, 0xbe, 0xad, 0xde];

fn err<E: Debug>(what: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("{what}: {e:?}")
}

fn dbg_err<E: Debug>(e: E) -> String {
    format!("{e:?}")
}

// ---- deterministic data --------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Unit-variance bell (sum of four uniforms).
    fn normal(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_050_8
    }
}

/// The `q8_weight` generator of `rdna-compute/tests/gemv_q8_0_rows_exact.rs`.
fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 24
}

/// Raw Q8_0 weight, `m` rows of `k / 32` 34-byte blocks (f16 scale + 32 int8).
fn q8_weight(seed: &mut u64, m: usize, k: usize) -> Vec<u8> {
    let mut w = Vec::with_capacity(m * (k / 32) * 34);
    for _ in 0..m * (k / 32) {
        let sign = (lcg(seed) & 1) as u16;
        let exp = 9 + (lcg(seed) % 9) as u16; // 2^-6 ..= 2^2
        let mant = (lcg(seed) & 0x3ff) as u16;
        w.extend_from_slice(&((sign << 15) | (exp << 10) | mant).to_le_bytes());
        w.extend((0..32).map(|_| lcg(seed) as u8));
    }
    w
}

/// Activations: unit-range values scaled by 2^0..2^4.
fn activations(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let u = rng.unit() * 2.0 - 1.0;
            u * (1u32 << (rng.next() % 5)) as f32
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// BF16 norm weight around 1.0.
fn bf16_norm(rng: &mut Rng, n: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|_| {
            let v = 0.75 + 0.5 * rng.unit();
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// `total` F32 words: `live` random finite words, then the sentinel.
fn filled(rng: &mut Rng, total: usize, live: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(total * 4);
    for i in 0..total {
        let bits = if i < live {
            rng.normal().to_bits()
        } else {
            SENTINEL
        };
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

// ---- device buffers ------------------------------------------------------------------------

fn upload_bytes(gpu: &mut Gpu, bytes: &[u8], dtype: DType) -> R<GpuTensor> {
    let t = gpu
        .alloc_tensor(&[bytes.len() / dtype.size()], dtype)
        .map_err(err("alloc"))?;
    gpu.hip.memcpy_htod(&t.buf, bytes).map_err(err("upload"))?;
    Ok(t)
}

fn q8_ref(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype: DType::Q8_0,
        m,
        k,
        row_stride: k / 32 * 34,
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

struct Weights {
    indexer_qk: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    output: GpuTensor,
    indexer_q_norm: GpuTensor,
    indexer_k_norm: GpuTensor,
    q_norm: GpuTensor,
    k_norm: GpuTensor,
}

impl Weights {
    fn new(gpu: &mut Gpu) -> R<Self> {
        let mut seed = 0x0d10_5eed_u64;
        let mut raw = |gpu: &mut Gpu, m: usize, k: usize| -> R<GpuTensor> {
            let bytes = q8_weight(&mut seed, m, k);
            gpu.upload_raw(&bytes, &[bytes.len()])
                .map_err(err("weight"))
        };
        let indexer_qk = raw(gpu, INDEX_W, HIDDEN)?;
        let q = raw(gpu, QGATE_W, HIDDEN)?;
        let k = raw(gpu, KV_W, HIDDEN)?;
        let v = raw(gpu, KV_W, HIDDEN)?;
        let output = raw(gpu, HIDDEN, Q_W)?;
        let mut rng = Rng::new(0x4e04);
        let mut norm =
            |gpu: &mut Gpu, n: usize| upload_bytes(gpu, &bf16_norm(&mut rng, n), DType::BF16);
        Ok(Self {
            indexer_qk,
            q,
            k,
            v,
            output,
            indexer_q_norm: norm(gpu, INDEX_DIM)?,
            indexer_k_norm: norm(gpu, INDEX_DIM)?,
            q_norm: norm(gpu, HEAD_DIM)?,
            k_norm: norm(gpu, HEAD_DIM)?,
        })
    }
}

fn rope() -> Rope {
    Rope {
        theta: 1_000_000.0,
        scaling: RopeScaling::Yarn(Yarn::new(4.0, 4096, None, None, None, None)),
    }
}

/// Host image of one lane's committed cache.
struct LaneHost {
    full_keys: Vec<u8>,
    full_values: Vec<u8>,
    raw_index_keys: Vec<u8>,
    pooled_keys: Vec<u8>,
}

fn lane_host(seed: u64, position: usize) -> LaneHost {
    let mut rng = Rng::new(seed);
    LaneHost {
        full_keys: filled(&mut rng, FULL_CAP * KV_W, position * KV_W),
        full_values: filled(&mut rng, FULL_CAP * KV_W, position * KV_W),
        raw_index_keys: filled(&mut rng, FULL_CAP * INDEX_KV_W, position * INDEX_KV_W),
        pooled_keys: filled(
            &mut rng,
            POOLED_CAP * INDEX_KV_W,
            position / COMPRESS * INDEX_KV_W,
        ),
    }
}

/// One lane's device state in one arm.
struct LaneDev {
    full_keys: GpuTensor,
    full_values: GpuTensor,
    raw_index_keys: GpuTensor,
    pooled_keys: GpuTensor,
    selected_indices: GpuTensor,
    selected_len_out: GpuTensor,
}

impl LaneDev {
    fn new(gpu: &mut Gpu, host: &LaneHost) -> R<Self> {
        let sel: Vec<u8> = (0..SELECTED_CAP)
            .flat_map(|_| SEL_SENTINEL.to_le_bytes())
            .collect();
        Ok(Self {
            full_keys: upload_bytes(gpu, &host.full_keys, DType::F32)?,
            full_values: upload_bytes(gpu, &host.full_values, DType::F32)?,
            raw_index_keys: upload_bytes(gpu, &host.raw_index_keys, DType::F32)?,
            pooled_keys: upload_bytes(gpu, &host.pooled_keys, DType::F32)?,
            selected_indices: upload_bytes(gpu, &sel, DType::Raw)?,
            selected_len_out: upload_bytes(gpu, &SEL_LEN_SENTINEL, DType::Raw)?,
        })
    }
}

/// Host lengths of one lane: `(full_len, raw_len, pooled_len, selected_len, position)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lens {
    full: usize,
    raw: usize,
    pooled: usize,
    selected: usize,
    position: usize,
}

impl Lens {
    fn at(position: usize) -> Self {
        Self {
            full: position,
            raw: position,
            pooled: position / COMPRESS,
            selected: 0,
            position,
        }
    }
    fn from_tuple(t: (usize, usize, usize, usize, usize)) -> Self {
        Self {
            full: t.0,
            raw: t.1,
            pooled: t.2,
            selected: t.3,
            position: t.4,
        }
    }
}

/// Scratch of one arm: `rows` rows of every projection buffer, one shared selection row.
struct RowScratch {
    index: GpuTensor,
    qgate: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    qsa_output: GpuTensor,
    selected: GpuTensor,
    /// `rows`-row selection scratch (only read under `--wide-selected-proj`).
    selected_wide: GpuTensor,
    bf16: GpuTensor,
    rotation: GpuTensor,
}

impl RowScratch {
    fn new(gpu: &mut Gpu, rows: usize) -> R<Self> {
        let mut f32s = |n: usize| gpu.zeros(&[n], DType::F32).map_err(err("scratch"));
        let index = f32s(rows * INDEX_W)?;
        let qgate = f32s(rows * QGATE_W)?;
        let k = f32s(rows * KV_W)?;
        let v = f32s(rows * KV_W)?;
        let qsa_output = f32s(rows * Q_W)?;
        let rotation = f32s(MAX_LANES * Q_W)?;
        Ok(Self {
            index,
            qgate,
            k,
            v,
            qsa_output,
            rotation,
            selected: gpu
                .zeros(&[SELECTED_CAP * 4], DType::Raw)
                .map_err(err("selected"))?,
            selected_wide: gpu
                .zeros(&[rows * SELECTED_CAP * 4], DType::Raw)
                .map_err(err("selected wide"))?,
            bf16: gpu
                .zeros(&[Q_W.max(KV_W)], DType::BF16)
                .map_err(err("bf16"))?,
        })
    }
}

/// One arm of a case: its own lane states, committed lengths, activations, output, scratch.
struct Arm {
    lanes: Vec<LaneDev>,
    lens: Vec<Lens>,
    input: GpuTensor,
    output: GpuTensor,
    scratch: RowScratch,
}

impl Arm {
    fn new(gpu: &mut Gpu, hosts: &[LaneHost], positions: &[usize], scratch_rows: usize) -> R<Self> {
        let n = positions.len();
        Ok(Self {
            lanes: hosts
                .iter()
                .map(|h| LaneDev::new(gpu, h))
                .collect::<R<Vec<_>>>()?,
            lens: positions.iter().map(|&p| Lens::at(p)).collect(),
            input: gpu.zeros(&[n * HIDDEN], DType::F32).map_err(err("input"))?,
            output: gpu
                .zeros(&[n * HIDDEN], DType::F32)
                .map_err(err("output"))?,
            scratch: RowScratch::new(gpu, scratch_rows)?,
        })
    }
}

// ---- op construction -----------------------------------------------------------------------

/// The scratch tensors one op binds.
#[derive(Clone, Copy)]
struct Sc<'a> {
    index: &'a GpuTensor,
    qgate: &'a GpuTensor,
    k: &'a GpuTensor,
    v: &'a GpuTensor,
    qsa_output: &'a GpuTensor,
    selected: &'a GpuTensor,
    bf16: &'a GpuTensor,
    rotation: &'a GpuTensor,
}

/// `head_attention_op` of `mtp_gpu.rs`, with the phase, rows and scratch of the call.
#[allow(clippy::too_many_arguments)]
fn make_op<'a>(
    w: &'a Weights,
    st: &'a LaneDev,
    l: Lens,
    reuse: bool,
    phase: MixerPhase,
    rows: usize,
    input: &'a GpuTensor,
    s: Sc<'a>,
    attention_output: &'a GpuTensor,
) -> IndexedAttentionOp<'a> {
    IndexedAttentionOp {
        indexer_qk: q8_ref(&w.indexer_qk, INDEX_W, HIDDEN),
        indexer_q_norm: &w.indexer_q_norm,
        indexer_k_norm: &w.indexer_k_norm,
        q: q8_ref(&w.q, QGATE_W, HIDDEN),
        k: q8_ref(&w.k, KV_W, HIDDEN),
        v: q8_ref(&w.v, KV_W, HIDDEN),
        q_norm: &w.q_norm,
        k_norm: &w.k_norm,
        output: q8_ref(&w.output, HIDDEN, Q_W),
        state: IndexedAttentionState {
            format: QsaKvFormat::F32,
            full_keys: &st.full_keys,
            full_values: &st.full_values,
            raw_index_keys: &st.raw_index_keys,
            pooled_keys: &st.pooled_keys,
            selected_indices: &st.selected_indices,
            f16_arena: None,
            pooled_bf16: None,
            full_capacity: FULL_CAP,
            raw_capacity: FULL_CAP,
            pooled_capacity: POOLED_CAP,
            selected_capacity: SELECTED_CAP,
            position_capacity: FULL_CAP,
            full_len: l.full,
            raw_len: l.raw,
            pooled_len: l.pooled,
            selected_len: l.selected,
            position: l.position,
        },
        input,
        index_scratch: s.index,
        qgate_scratch: s.qgate,
        k_scratch: s.k,
        v_scratch: s.v,
        qsa_output: s.qsa_output,
        selected_scratch: s.selected,
        attention_output,
        bf16_scratch: s.bf16,
        rows,
        index_heads: INDEX_HEADS,
        index_kv_heads: INDEX_KV_HEADS,
        index_dim: INDEX_DIM,
        budget: BUDGET,
        compress: COMPRESS,
        heads: HEADS,
        kv_heads: KV_HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: ROTARY_DIM,
        rope: rope(),
        input_width: HIDDEN,
        rotation: s.rotation,
        mode: if reuse {
            IndexedAttentionMode::ReuseSelection {
                selected_len_out: &st.selected_len_out,
            }
        } else {
            IndexedAttentionMode::Full
        },
        trunk_a4: 0,
        phase,
    }
}

/// Lane `row`'s one-row views of an `n`-row scratch: index, q+gate, K, V, attention output.
fn row_views(s: &RowScratch, row: usize) -> [GpuTensor; 5] {
    [
        s.index.sub_offset(row * INDEX_W, INDEX_W),
        s.qgate.sub_offset(row * QGATE_W, QGATE_W),
        s.k.sub_offset(row * KV_W, KV_W),
        s.v.sub_offset(row * KV_W, KV_W),
        s.qsa_output.sub_offset(row * Q_W, Q_W),
    ]
}

// ---- one round -----------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    /// Candidate lane 1's Core op bound to lane 0's state.
    CrossState,
    /// Candidate lanes 0 / 1 Core scratch row views swapped.
    SwappedScratch,
    /// Candidate lane 1 position (and lengths) off by one.
    OffByOne,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Control::None => "none",
            Control::CrossState => "lane-1-core-on-lane-0-state",
            Control::SwappedScratch => "swapped-core-scratch-rows",
            Control::OffByOne => "lane-1-position-off-by-one",
        }
    }
}

fn run_steps(gpu: &mut Gpu, ctx: &DispatchCtx, steps: &[Step<'_>]) -> R<()> {
    validate_steps(gpu, steps).map_err(|e| format!("validate: {}", dbg_err(e)))?;
    execute_validated_steps(gpu, ctx, steps).map_err(|e| format!("execute: {}", dbg_err(e)))
}

/// The reference arm: one whole op per lane, executed lane after lane.
fn run_reference(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &Weights,
    arm: &mut Arm,
    reuse: &[bool],
) -> R<()> {
    let n = reuse.len();
    let ins: Vec<GpuTensor> = (0..n)
        .map(|i| arm.input.sub_offset(i * HIDDEN, HIDDEN))
        .collect();
    let outs: Vec<GpuTensor> = (0..n)
        .map(|i| arm.output.sub_offset(i * HIDDEN, HIDDEN))
        .collect();
    let s = &arm.scratch;
    let sc = Sc {
        index: &s.index,
        qgate: &s.qgate,
        k: &s.k,
        v: &s.v,
        qsa_output: &s.qsa_output,
        selected: &s.selected,
        bf16: &s.bf16,
        rotation: &s.rotation,
    };
    let mut steps = Vec::with_capacity(n);
    let mut next = Vec::with_capacity(n);
    for i in 0..n {
        let op = make_op(
            w,
            &arm.lanes[i],
            arm.lens[i],
            reuse[i],
            MixerPhase::Whole,
            1,
            &ins[i],
            sc,
            &outs[i],
        );
        next.push(Lens::from_tuple(op.next_lengths().map_err(dbg_err)?));
        steps.push(Step::IndexedAttention(op));
    }
    run_steps(gpu, ctx, &steps)?;
    drop(steps);
    arm.lens = next;
    Ok(())
}

/// The candidate arm: `[InProj(n rows), Core x n, OutProj(n rows)]` in one step list.
fn run_candidate(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &Weights,
    arm: &mut Arm,
    reuse: &[bool],
    control: Control,
    wide_selected: bool,
) -> R<()> {
    let n = reuse.len();
    let ins: Vec<GpuTensor> = (0..n)
        .map(|i| arm.input.sub_offset(i * HIDDEN, HIDDEN))
        .collect();
    let outs: Vec<GpuTensor> = (0..n)
        .map(|i| arm.output.sub_offset(i * HIDDEN, HIDDEN))
        .collect();
    let views: Vec<[GpuTensor; 5]> = (0..n).map(|i| row_views(&arm.scratch, i)).collect();
    let s = &arm.scratch;
    let batch = Sc {
        index: &s.index,
        qgate: &s.qgate,
        k: &s.k,
        v: &s.v,
        qsa_output: &s.qsa_output,
        selected: if wide_selected {
            &s.selected_wide
        } else {
            &s.selected
        },
        bf16: &s.bf16,
        rotation: &s.rotation,
    };
    let mut steps = Vec::with_capacity(n + 2);
    let mut next = Vec::with_capacity(n);
    // Lane 0's state only fills the op: the projection phases never read it.
    steps.push(Step::IndexedAttention(make_op(
        w,
        &arm.lanes[0],
        arm.lens[0],
        false,
        MixerPhase::InProj,
        n,
        &arm.input,
        batch,
        &arm.output,
    )));
    for i in 0..n {
        let v = &views[if control == Control::SwappedScratch && i < 2 {
            1 - i
        } else {
            i
        }];
        let sc = Sc {
            index: &v[0],
            qgate: &v[1],
            k: &v[2],
            v: &v[3],
            qsa_output: &v[4],
            selected: &s.selected,
            bf16: &s.bf16,
            rotation: &s.rotation,
        };
        let st = if control == Control::CrossState && i == 1 {
            &arm.lanes[0]
        } else {
            &arm.lanes[i]
        };
        let op = make_op(
            w,
            st,
            arm.lens[i],
            reuse[i],
            MixerPhase::Core,
            1,
            &ins[i],
            sc,
            &outs[i],
        );
        next.push(Lens::from_tuple(op.next_lengths().map_err(dbg_err)?));
        steps.push(Step::IndexedAttention(op));
    }
    steps.push(Step::IndexedAttention(make_op(
        w,
        &arm.lanes[0],
        arm.lens[0],
        false,
        MixerPhase::OutProj,
        n,
        &arm.input,
        batch,
        &arm.output,
    )));
    run_steps(gpu, ctx, &steps)?;
    drop(steps);
    arm.lens = next;
    Ok(())
}

// ---- comparison ----------------------------------------------------------------------------

fn word_mismatches(a: &[u8], b: &[u8]) -> usize {
    if a.len() != b.len() {
        return a.len().max(b.len()) / 4 + 1;
    }
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| x != y)
        .count()
}

fn bytes_of(gpu: &Gpu, t: &GpuTensor) -> R<Vec<u8>> {
    gpu.download_raw_bytes(t).map_err(err("download"))
}

/// Mismatching word counts per tensor after one round (`pos_before`: each lane's position
/// before the round, for the liveness checks).
fn compare_round(gpu: &Gpu, rf: &Arm, cd: &Arm, pos_before: &[usize]) -> R<Vec<(String, usize)>> {
    let mut bad = Vec::new();
    let mut push = |name: String, words: usize| {
        if words != 0 {
            bad.push((name, words));
        }
    };
    for (i, (a, b)) in rf.lanes.iter().zip(&cd.lanes).enumerate() {
        let pairs = [
            ("full_keys", &a.full_keys, &b.full_keys),
            ("full_values", &a.full_values, &b.full_values),
            ("raw_index_keys", &a.raw_index_keys, &b.raw_index_keys),
            ("pooled_keys", &a.pooled_keys, &b.pooled_keys),
            ("selected_indices", &a.selected_indices, &b.selected_indices),
            ("selected_len_out", &a.selected_len_out, &b.selected_len_out),
        ];
        for (name, ta, tb) in pairs {
            let ba = bytes_of(gpu, ta)?;
            let bb = bytes_of(gpu, tb)?;
            push(format!("lane{i}.{name}"), word_mismatches(&ba, &bb));
            if name == "full_keys" {
                // Liveness: the reference wrote this lane's new K row over the sentinel.
                let at = pos_before[i] * KV_W * 4;
                let word = u32::from_le_bytes(ba[at..at + 4].try_into().expect("word"));
                push(
                    format!("lane{i}.reference_wrote_k_row"),
                    usize::from(word == SENTINEL),
                );
            }
        }
        push(
            format!("lane{i}.host_lengths"),
            usize::from(rf.lens[i] != cd.lens[i]),
        );
    }
    let out_a = bytes_of(gpu, &rf.output)?;
    let out_b = bytes_of(gpu, &cd.output)?;
    push("attention_output".into(), word_mismatches(&out_a, &out_b));
    push(
        "reference_output_all_zero".into(),
        usize::from(out_a.iter().all(|&b| b == 0)),
    );
    Ok(bad)
}

// ---- cases ---------------------------------------------------------------------------------

struct Case {
    name: &'static str,
    seed: u64,
    positions: Vec<usize>,
    /// Per round, per lane: ReuseSelection (true) or Full (false).
    rounds: Vec<Vec<bool>>,
}

fn cases() -> Vec<Case> {
    let same = |n: usize, reuse: bool| vec![reuse; n];
    let a = {
        let mut rounds = vec![same(2, false)];
        rounds.extend((0..4).map(|_| same(2, true)));
        Case {
            name: "a: n=2 same position 1001, Full then Reuse x4",
            seed: 0xa1,
            positions: vec![1001, 1001],
            rounds,
        }
    };
    let b = Case {
        name: "b: n=2 equal positions, mixed Full / Reuse",
        seed: 0xb2,
        positions: vec![1001, 1001],
        rounds: vec![
            vec![false, false],
            vec![false, true],
            vec![true, false],
            vec![true, true],
        ],
    };
    let c = {
        let mut rounds = vec![same(8, false)];
        rounds.extend((0..3).map(|_| same(8, true)));
        Case {
            name: "c: n=8 ragged positions, Full then Reuse x3",
            seed: 0xc3,
            positions: vec![1001, 1001, 1002, 63, 64, 65, 3000, 1001],
            rounds,
        }
    };
    let d = Case {
        name: "d: n=3 positions 3,4,5, Full only",
        seed: 0xd4,
        positions: vec![3, 4, 5],
        rounds: (0..3).map(|_| same(3, false)).collect(),
    };
    vec![a, b, c, d]
}

struct Outcome {
    /// Rounds with at least one mismatching tensor.
    failed_rounds: usize,
    /// Total mismatching words over every round and tensor.
    words: usize,
    /// A launch / validation error (the failing arm and message).
    error: Option<String>,
}

fn run_case(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &Weights,
    case: &Case,
    control: Control,
    wide_selected: bool,
    verbose: bool,
    log: &mut Vec<String>,
) -> R<Outcome> {
    let n = case.positions.len();
    let hosts: Vec<LaneHost> = case
        .positions
        .iter()
        .enumerate()
        .map(|(i, &p)| lane_host(case.seed * 1000 + i as u64, p))
        .collect();
    let mut rf = Arm::new(gpu, &hosts, &case.positions, 1)?;
    let mut cd = Arm::new(gpu, &hosts, &case.positions, n)?;
    if control == Control::OffByOne {
        cd.lens[1] = Lens::at(case.positions[1] + 1);
    }
    let mut rng = Rng::new(case.seed ^ 0x5eed);
    let mut outcome = Outcome {
        failed_rounds: 0,
        words: 0,
        error: None,
    };
    let emit = |line: String, log: &mut Vec<String>| {
        println!("{line}");
        log.push(line);
    };
    for (r, reuse) in case.rounds.iter().enumerate() {
        let acts = f32_bytes(&activations(&mut rng, n * HIDDEN));
        gpu.hip
            .memcpy_htod(&rf.input.buf, &acts)
            .map_err(err("activations"))?;
        gpu.hip
            .memcpy_htod(&cd.input.buf, &acts)
            .map_err(err("activations"))?;
        let pos_before: Vec<usize> = rf.lens.iter().map(|l| l.position).collect();
        if let Err(e) = run_reference(gpu, ctx, w, &mut rf, reuse) {
            return Err(format!("reference arm, round {}: {e}", r + 1));
        }
        if let Err(e) = run_candidate(gpu, ctx, w, &mut cd, reuse, control, wide_selected) {
            outcome.error = Some(format!("candidate arm, round {}: {e}", r + 1));
            break;
        }
        let bad = compare_round(gpu, &rf, &cd, &pos_before)?;
        let modes: String = reuse.iter().map(|&u| if u { 'R' } else { 'F' }).collect();
        if bad.is_empty() {
            if verbose {
                emit(
                    format!("PASS case {} round {} [{modes}]", case.name, r + 1),
                    log,
                );
            }
        } else {
            outcome.failed_rounds += 1;
            outcome.words += bad.iter().map(|b| b.1).sum::<usize>();
            if verbose {
                let detail: Vec<String> = bad.iter().map(|(n, c)| format!("{n}={c}")).collect();
                emit(
                    format!(
                        "FAIL case {} round {} [{modes}]: mismatching words: {}",
                        case.name,
                        r + 1,
                        detail.join(" ")
                    ),
                    log,
                );
            }
        }
    }
    Ok(outcome)
}

// ---- driver --------------------------------------------------------------------------------

struct Cli {
    out: Option<PathBuf>,
    wide_selected: bool,
}

fn parse_cli() -> R<Cli> {
    let mut cli = Cli {
        out: None,
        wide_selected: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                cli.out = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out needs a directory".to_string())?,
                ));
            }
            "--wide-selected-proj" => cli.wide_selected = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(cli)
}

fn run() -> R<bool> {
    let cli = parse_cli()?;
    let mut gpu = Gpu::init().map_err(err("Gpu::init"))?;
    println!(
        "mtp_proj_batch_oracle: arch={} few_row_limit={} qsa_lanes={} wide_selected_proj={}",
        gpu.arch,
        gpu.few_row_limit(),
        gpu.flags.qwen4_qsa_lanes_enabled(),
        cli.wide_selected
    );
    if gpu.arch != "gfx1201" {
        return Err(format!("needs exact gfx1201, got {}", gpu.arch));
    }
    let w = Weights::new(&mut gpu)?;
    let ctx = DispatchCtx::new(&gpu);
    let mut log: Vec<String> = Vec::new();
    let mut ok = true;

    let all = cases();
    for case in &all {
        let n = case.positions.len();
        let route = gpu.route_rows(n);
        let line = format!(
            "case {}: n={n} route_rows={route} (few-row gate 2..={})",
            case.name,
            gpu.few_row_limit()
        );
        println!("{line}");
        log.push(line);
        let o = run_case(
            &mut gpu,
            &ctx,
            &w,
            case,
            Control::None,
            cli.wide_selected,
            true,
            &mut log,
        )?;
        if let Some(e) = o.error {
            let line = format!("FAIL case {}: {e}", case.name);
            println!("{line}");
            log.push(line);
            ok = false;
        } else if o.failed_rounds != 0 {
            ok = false;
        }
    }

    let case_a = &all[0];
    for control in [
        Control::CrossState,
        Control::SwappedScratch,
        Control::OffByOne,
    ] {
        let o = run_case(
            &mut gpu,
            &ctx,
            &w,
            case_a,
            control,
            cli.wide_selected,
            false,
            &mut log,
        )?;
        let detected = o.error.is_some() || o.words != 0;
        let line = format!(
            "{} control {}: failed_rounds={} mismatching_words={} error={}",
            if detected { "DETECTED" } else { "UNDETECTED" },
            control.name(),
            o.failed_rounds,
            o.words,
            o.error.as_deref().unwrap_or("none")
        );
        println!("{line}");
        log.push(line);
        ok &= detected;
    }

    let verdict = if ok {
        "PASS mtp_proj_batch_oracle".to_string()
    } else {
        "FAIL mtp_proj_batch_oracle".to_string()
    };
    println!("{verdict}");
    log.push(verdict);
    if let Some(dir) = &cli.out {
        std::fs::create_dir_all(dir).map_err(err("create --out"))?;
        std::fs::write(dir.join("mtp_proj_batch_oracle.txt"), log.join("\n") + "\n")
            .map_err(err("write receipt"))?;
    }
    Ok(ok)
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("mtp_proj_batch_oracle: {e}");
            std::process::exit(2);
        }
    }
}
