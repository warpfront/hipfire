// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G0 dispatch-level byte oracle of the HC8 (`.hc8` sidecar, `DType::HC8E4M3Row`) lane
//! row-twins: the `HcRead` and `FinalHyper` stages of `gpu_forward_lanes` with the
//! `StageRoute::HyperRead([BF16, HC8E4M3Row, HC8E4M3Row])` classification (BF16 `hc_norm`,
//! HC8 down and up mixers; `HcWrite` keeps its BF16 `block_inject`).
//!
//! ```text
//! lane_hc8_rows_oracle <BASE.mq4-xts> <BASE.mq4-xts.hc8> [--out DIR] [--rows-max 64] [--seeds 2]
//! ```
//!
//! The sidecar is bound to its base and every payload sha256-verified (`bind_hc8_sidecar` +
//! `read_verified`); each payload is uploaded as `DType::HC8E4M3Row` exactly as the loader's
//! `upload_hc8` does, the BF16 `hc_norm` / `block_inject` come from the base. Every real HC
//! read instance the sidecar replaces (48 layers x attn / mlp, the MTP layer's, the final
//! mixer) is one stage:
//!
//! * `read`: the lane step's `HcRead` (`execute_hyper_read`) for every sidecar connection;
//! * `final`: `FinalHyper` (`execute_final_hyper`) for every connection that is not a trunk
//!   layer (the final mixer, the MTP layer's);
//! * `chain`: the paired path the lanes use at a layer boundary,
//!   `W(attn) R(mlp) Clear W(mlp) R(attn') W(attn') R(mlp') Clear` through
//!   `execute_validated_steps` (the fused write -> read launch) for every consecutive trunk
//!   layer pair.
//!
//! For `R` in `1..=rows-max` (every value) and both row orders (forward and reversed), the
//! shared launch over `R` rows at row 0 (the all-rows launch: under
//! `Gpu::set_wide_verify_rows(R)` for `R > 8`, exactly the lane step's all-rows segment) is
//! compared word for word with `R` separate `rows == 1` launches (scope off), row by row,
//! over every buffer the stage writes (`normalized`, `low`, `up`, `mixed`; the chain also
//! `streams`, `gates`, `clear`). Buffers hold 72 rows, NaN-sentinel prefilled; words past
//! the `R` live rows of a written buffer and every read-only buffer must keep their upload
//! in both arms. Inputs are seeded BF16-exact stream rows (uniform, heavy-tailed, zeros and
//! signed zeros, subnormals, bell, sparse spikes at scales 0.25..=4) plus eight fixed
//! real-magnitude edge rows (zeros, constants, 1e4-class, 1e-12-class, one-hot spike,
//! subnormals, per-branch scale split).
//!
//! Last result line (the card script greps it):
//! `lane_hc8_rows_oracle: <C> cells, <W> bytes compared, <M> mismatching bytes`.
//! Exit 0 pass, 1 mismatch, 2 environment. Needs exact gfx1201 and
//! `HIPFIRE_QWEN4_LANE_HC_ALLROWS` on (its default).

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_final_hyper, execute_hyper_read, execute_validated_steps, validate_steps, ClearOp,
    HyperReadOp, HyperWriteOp, Step,
};
use hipfire_runtime::hc8_sidecar::bind_hc8_sidecar;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

type R<T> = Result<T, String>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const RANK: usize = 320;
const LANE_ROWS: usize = 8;
/// Rows every scratch buffer holds: the widest step (64) plus canary rows.
const CAP: usize = 72;
const BASE_ROWS: usize = 64;
const GATES_PER_ROW_FULL: usize = BRANCHES + 32;
const GATES_WORDS: usize = CAP * GATES_PER_ROW_FULL;
const SENTINEL: u32 = 0x7fc0_beef;

const NBUF: usize = 8;
const STREAMS: usize = 0;
const NORMALIZED: usize = 1;
const LOW: usize = 2;
const UP: usize = 3;
const MIXED: usize = 4;
const GATES: usize = 5;
const WMIX: usize = 6;
const CLEAR: usize = 7;
const NAMES: [&str; NBUF] = [
    "streams",
    "normalized",
    "low",
    "up",
    "mixed",
    "gates",
    "wmix",
    "clear",
];
const WIDTH: [usize; NBUF] = [WIDE, WIDE, RANK, WIDE, HIDDEN, BRANCHES, HIDDEN, HIDDEN];

fn err<E: Debug>(what: &'static str) -> impl FnOnce(E) -> String {
    move |e| format!("{what}: {e:?}")
}

fn dbg_err<E: Debug>(e: E) -> String {
    format!("{e:?}")
}

// ---- deterministic data ------------------------------------------------------------------

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
    fn normal(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_050_8
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    f32::from_bits((b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) << 16)
}

/// One activation row: a pattern family at a scale in `0.25..=4`.
fn gen_row(rng: &mut Rng, width: usize, bf16: bool) -> Vec<f32> {
    let scale = [0.25f32, 0.5, 1.0, 2.0, 4.0][rng.below(5) as usize];
    let pattern = rng.below(6);
    (0..width)
        .map(|_| {
            let u = rng.unit() * 2.0 - 1.0;
            let v = match pattern {
                0 => u,
                1 => {
                    let mut v = u * u * u;
                    if rng.below(40) == 0 {
                        v *= 64.0;
                    }
                    v
                }
                2 => 0.0,
                3 => match rng.below(24) {
                    0 => 0.0,
                    1 => -0.0,
                    2 => f32::from_bits(rng.next() as u32 & 0x807f_ffff),
                    _ => u,
                },
                4 => rng.normal(),
                _ => {
                    if rng.below(64) == 0 {
                        u * 32.0
                    } else {
                        0.0
                    }
                }
            } * scale;
            if bf16 {
                bf16_round(v)
            } else {
                v
            }
        })
        .collect()
}

/// A fixed real-magnitude edge stream row (BF16-exact), `which` in `0..8`.
fn edge_row(rng: &mut Rng, which: usize) -> Vec<f32> {
    let sign = |rng: &mut Rng| if rng.below(2) == 0 { 1.0f32 } else { -1.0 };
    let v: Vec<f32> = match which {
        0 => vec![0.0; WIDE],
        1 => vec![1.0; WIDE],
        2 => (0..WIDE)
            .map(|i| if i % 2 == 0 { 3.0e4 } else { -3.0e4 })
            .collect(),
        3 => (0..WIDE)
            .map(|_| sign(rng) * 1.0e4 * (0.5 + rng.unit()))
            .collect(),
        4 => (0..WIDE)
            .map(|_| sign(rng) * 1.0e-12 * (0.5 + rng.unit()))
            .collect(),
        5 => {
            let mut v = vec![0.0; WIDE];
            v[rng.below(WIDE as u64) as usize] = 256.0;
            v
        }
        6 => (0..WIDE)
            .map(|i| match i % 4 {
                0 => -0.0,
                1 => f32::from_bits(0x0000_8000 | (rng.next() as u32 & 0x7f) << 16),
                2 => f32::from_bits(0x8000_0000 | 0x0001_0000),
                _ => 0.0,
            })
            .collect(),
        _ => (0..WIDE)
            .map(|i| {
                let branch = i / HIDDEN;
                let scale = if branch == 0 { 100.0 } else { 0.01 };
                rng.normal() * scale
            })
            .collect(),
    };
    v.into_iter().map(bf16_round).collect()
}

/// Host images (words) of every buffer for the 64 base rows; rows past `rows` hold the
/// sentinel in every buffer but `clear` (prefilled nonzero for live rows).
struct Init {
    words: Vec<Vec<u32>>,
}

struct BaseRows {
    streams: Vec<Vec<f32>>,
    wmix: Vec<Vec<f32>>,
    clear: Vec<Vec<f32>>,
}

fn base_rows(seed: u64) -> BaseRows {
    let mut rng = Rng::new(0xC0FF_EE00 ^ (seed << 20));
    let mut streams = Vec::new();
    let mut wmix = Vec::new();
    let mut clear = Vec::new();
    for r in 0..BASE_ROWS {
        streams.push(if r >= BASE_ROWS - 8 {
            edge_row(&mut rng, r - (BASE_ROWS - 8))
        } else {
            gen_row(&mut rng, WIDE, true)
        });
        wmix.push(gen_row(&mut rng, HIDDEN, false));
        clear.push((0..HIDDEN).map(|_| 1.0 + rng.unit()).collect());
    }
    BaseRows {
        streams,
        wmix,
        clear,
    }
}

/// An init whose row `i` is base row `order[i]`.
fn make_init(base: &BaseRows, order: &[usize]) -> Init {
    let mut words: Vec<Vec<u32>> = (0..NBUF)
        .map(|b| {
            let len = if b == GATES {
                GATES_WORDS
            } else {
                CAP * WIDTH[b]
            };
            vec![SENTINEL; len]
        })
        .collect();
    for (r, &src) in order.iter().enumerate() {
        for (i, v) in base.streams[src].iter().enumerate() {
            words[STREAMS][r * WIDE + i] = v.to_bits();
        }
        for (i, v) in base.wmix[src].iter().enumerate() {
            words[WMIX][r * HIDDEN + i] = v.to_bits();
        }
        for (i, v) in base.clear[src].iter().enumerate() {
            words[CLEAR][r * HIDDEN + i] = v.to_bits();
        }
    }
    Init { words }
}

// ---- GPU helpers -------------------------------------------------------------------------

fn htod_words(gpu: &Gpu, t: &GpuTensor, words: &[u32]) -> R<()> {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.hip.device_synchronize().map_err(err("sync"))?;
    gpu.hip.memcpy_htod(&t.buf, &bytes).map_err(err("htod"))
}

fn dtoh_words(gpu: &Gpu, t: &GpuTensor) -> R<Vec<u32>> {
    gpu.hip.device_synchronize().map_err(err("sync"))?;
    Ok(gpu
        .download_f32(t)
        .map_err(err("download"))?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

fn upload_bf16(gpu: &mut Gpu, bytes: &[u8], numel: usize) -> R<GpuTensor> {
    if bytes.len() != numel * 2 {
        return Err(format!(
            "BF16 tensor has {} bytes, expected {}",
            bytes.len(),
            numel * 2
        ));
    }
    let t = gpu
        .alloc_tensor(&[numel], DType::BF16)
        .map_err(err("alloc bf16"))?;
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(err("upload bf16"))?;
    Ok(t)
}

fn wref(buf: &GpuTensor, dtype: DType, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype,
        m,
        k,
        row_stride: dtype.row_bytes(k).unwrap_or(k * dtype.size()),
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

struct Bufs {
    t: Vec<GpuTensor>,
    rotation: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu) -> R<Self> {
        let mut t = Vec::with_capacity(NBUF);
        for b in 0..NBUF {
            let len = if b == GATES {
                GATES_WORDS
            } else {
                CAP * WIDTH[b]
            };
            t.push(
                gpu.alloc_tensor(&[len], DType::F32)
                    .map_err(err("alloc scratch"))?,
            );
        }
        let rotation = gpu
            .alloc_tensor(&[CAP * WIDE], DType::F32)
            .map_err(err("alloc rotation"))?;
        Ok(Bufs { t, rotation })
    }

    fn views(&self, begin: usize) -> Vec<GpuTensor> {
        (0..NBUF)
            .map(|b| {
                let skip = begin * WIDTH[b];
                self.t[b].sub_offset(skip, self.t[b].numel() - skip)
            })
            .collect()
    }

    fn reset(&self, gpu: &Gpu, init: &Init) -> R<()> {
        for b in 0..NBUF {
            htod_words(gpu, &self.t[b], &init.words[b])?;
        }
        Ok(())
    }

    fn snapshot(&self, gpu: &Gpu) -> R<Vec<Vec<u32>>> {
        self.t.iter().map(|t| dtoh_words(gpu, t)).collect()
    }
}

// ---- weights -----------------------------------------------------------------------------

struct Conn {
    name: String,
    norm: GpuTensor,
    down: GpuTensor,
    up: GpuTensor,
    inject: Option<GpuTensor>,
}

fn trunk_layer(name: &str) -> Option<(usize, bool)> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (id, tail) = rest.split_once('.')?;
    let id = id.parse().ok()?;
    match tail {
        "attn_hyper_connection" => Some((id, false)),
        "mlp_hyper_connection" => Some((id, true)),
        _ => None,
    }
}

// ---- stages ------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Read,
    Final,
    Chain,
}

#[derive(Clone, Copy)]
enum Stage {
    Read(usize),
    Final(usize),
    /// Trunk layers `first`, `second` as indexes into the layer table.
    Chain(usize, usize),
}

impl Stage {
    fn kind(self) -> Kind {
        match self {
            Stage::Read(_) => Kind::Read,
            Stage::Final(_) => Kind::Final,
            Stage::Chain(..) => Kind::Chain,
        }
    }
}

struct Model {
    conns: Vec<Conn>,
    /// Trunk layer `i`: (attn conn index, mlp conn index), ascending layer id.
    layers: Vec<(usize, usize, usize)>,
}

fn read_op<'a>(
    c: &'a Conn,
    v: &'a [GpuTensor],
    rot: &'a GpuTensor,
    rows: usize,
) -> HyperReadOp<'a> {
    HyperReadOp {
        state_bf16: false,
        input: &v[STREAMS],
        norm_weight: &c.norm,
        input_mix_down: wref(&c.down, DType::HC8E4M3Row, RANK, WIDE),
        input_mix_up: wref(&c.up, DType::HC8E4M3Row, WIDE, RANK),
        normalized: &v[NORMALIZED],
        low: &v[LOW],
        up: &v[UP],
        mixed: &v[MIXED],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        low_rank: RANK,
        rotation: rot,
    }
}

fn write_op<'a>(
    c: &'a Conn,
    v: &'a [GpuTensor],
    rot: &'a GpuTensor,
    rows: usize,
) -> HyperWriteOp<'a> {
    HyperWriteOp {
        state_bf16: false,
        input: &v[STREAMS],
        norm_weight: &c.norm,
        block_inject: wref(
            c.inject.as_ref().expect("chain connection has an inject"),
            DType::BF16,
            BRANCHES,
            WIDE,
        ),
        normalized: &v[NORMALIZED],
        mixed: &v[WMIX],
        gates: &v[GATES],
        output: &v[STREAMS],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        rotation: rot,
        ar: None,
    }
}

fn label(m: &Model, stage: Stage) -> String {
    match stage {
        Stage::Read(c) => format!("read {}", m.conns[c].name),
        Stage::Final(c) => format!("final {}", m.conns[c].name),
        Stage::Chain(a, b) => format!(
            "chain {} -> {}",
            m.conns[m.layers[a].1].name, m.conns[m.layers[b].1].name
        ),
    }
}

/// Run `stage` over `rows` rows of views `v`; `scope` opens the all-rows wide scope.
fn exec_stage(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    m: &Model,
    stage: Stage,
    v: &[GpuTensor],
    rot: &GpuTensor,
    rows: usize,
    scope: bool,
) -> R<()> {
    let prior = gpu.wide_verify_rows;
    if scope {
        gpu.set_wide_verify_rows(rows);
        if rows > LANE_ROWS && !gpu.lane_hc_allrows_applies(rows) {
            gpu.set_wide_verify_rows(prior);
            return Err(format!(
                "all-rows arm not engaged for {rows} rows (lane_hc_allrows_applies false)"
            ));
        }
    }
    let result = (|| -> R<()> {
        match stage {
            Stage::Read(c) => {
                let op = read_op(&m.conns[c], v, rot, rows);
                op.validate_for_gpu(gpu).map_err(dbg_err)?;
                execute_hyper_read(gpu, &op).map_err(dbg_err)
            }
            Stage::Final(c) => {
                let op = read_op(&m.conns[c], v, rot, rows);
                op.validate_for_gpu(gpu).map_err(dbg_err)?;
                execute_final_hyper(gpu, &op).map_err(dbg_err)
            }
            Stage::Chain(a, b) => {
                let conn = |l: usize, mlp: bool| {
                    let (_, attn, mlp_i) = m.layers[l];
                    &m.conns[if mlp { mlp_i } else { attn }]
                };
                let clear = || {
                    Step::Clear(ClearOp {
                        tensor: &v[CLEAR],
                        elements: rows * HIDDEN,
                    })
                };
                let steps = vec![
                    Step::HyperWrite(write_op(conn(a, false), v, rot, rows)),
                    Step::HyperRead(read_op(conn(a, true), v, rot, rows)),
                    clear(),
                    Step::HyperWrite(write_op(conn(a, true), v, rot, rows)),
                    Step::HyperRead(read_op(conn(b, false), v, rot, rows)),
                    Step::HyperWrite(write_op(conn(b, false), v, rot, rows)),
                    Step::HyperRead(read_op(conn(b, true), v, rot, rows)),
                    clear(),
                ];
                let ctx_steps = validate_steps(gpu, &steps).map_err(dbg_err);
                ctx_steps?;
                execute_validated_steps(gpu, ctx, &steps).map_err(dbg_err)
            }
        }
    })();
    gpu.set_wide_verify_rows(prior);
    result
}

/// Whether the stage writes each buffer.
fn written(kind: Kind) -> [bool; NBUF] {
    match kind {
        Kind::Read | Kind::Final => [false, true, true, true, true, false, false, false],
        Kind::Chain => [true, true, true, true, true, true, false, true],
    }
}

/// One arm: reset, run, snapshot.
fn run_arm(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    m: &Model,
    stage: Stage,
    bufs: &Bufs,
    init: &Init,
    rows: usize,
    singleton: bool,
) -> R<Vec<Vec<u32>>> {
    bufs.reset(gpu, init)?;
    let prior = gpu.set_wide_verify_rows(0);
    let result = (|| -> R<()> {
        if singleton {
            for i in 0..rows {
                let v = bufs.views(i);
                exec_stage(gpu, ctx, m, stage, &v, &bufs.rotation, 1, false)?;
            }
            Ok(())
        } else {
            let v = bufs.views(0);
            exec_stage(gpu, ctx, m, stage, &v, &bufs.rotation, rows, true)
        }
    })();
    gpu.set_wide_verify_rows(prior);
    result?;
    bufs.snapshot(gpu)
}

// ---- report ------------------------------------------------------------------------------

#[derive(Default)]
struct Stats {
    cells: u64,
    bytes: u64,
    mismatching_bytes: u64,
    violation_words: u64,
    per_buffer_bytes: BTreeMap<&'static str, u64>,
    per_buffer_mismatch: BTreeMap<&'static str, u64>,
    first: Vec<String>,
}

impl Stats {
    fn note(&mut self, msg: String) {
        if self.first.len() < 20 {
            self.first.push(msg);
        }
    }
}

/// Compare one all-rows cell against the singleton reference rows.
#[allow(clippy::too_many_arguments)]
fn compare_cell(
    st: &mut Stats,
    tag: &str,
    kind: Kind,
    rows: usize,
    order: &[usize],
    all: &[Vec<u32>],
    single: &[Vec<u32>],
    init: &Init,
) {
    st.cells += 1;
    let wr = written(kind);
    for b in 0..NBUF {
        if !wr[b] {
            // Read-only buffers must equal their upload everywhere.
            let bad = all[b]
                .iter()
                .zip(&init.words[b])
                .filter(|(x, y)| x != y)
                .count();
            if bad != 0 {
                st.violation_words += bad as u64;
                st.note(format!(
                    "{tag}: readonly {} changed in {bad} words",
                    NAMES[b]
                ));
            }
            continue;
        }
        let w = WIDTH[b];
        // Words past the live rows of a written buffer keep their upload (the chain's
        // gates also hold the fused writes' quarter slots there).
        if b != GATES {
            let from = rows * w;
            let bad = all[b][from..]
                .iter()
                .zip(&init.words[b][from..])
                .filter(|(x, y)| x != y)
                .count();
            if bad != 0 {
                st.violation_words += bad as u64;
                st.note(format!(
                    "{tag}: tail of {} touched in {bad} words",
                    NAMES[b]
                ));
            }
        }
        for (i, &src) in order.iter().enumerate() {
            let a = &all[b][i * w..(i + 1) * w];
            let s = &single[b][src * w..(src + 1) * w];
            st.bytes += (w * 4) as u64;
            *st.per_buffer_bytes.entry(NAMES[b]).or_default() += (w * 4) as u64;
            if a != s {
                let mut bad_bytes = 0u64;
                let mut first_word = None;
                for (j, (x, y)) in a.iter().zip(s).enumerate() {
                    if x != y {
                        bad_bytes += x
                            .to_le_bytes()
                            .iter()
                            .zip(y.to_le_bytes())
                            .filter(|(p, q)| **p != *q)
                            .count() as u64;
                        first_word.get_or_insert((j, *x, *y));
                    }
                }
                st.mismatching_bytes += bad_bytes;
                *st.per_buffer_mismatch.entry(NAMES[b]).or_default() += bad_bytes;
                let (j, x, y) = first_word.expect("a differing slice has a differing word");
                st.note(format!(
                    "{tag}: R={rows} row {i} (base row {src}) buffer {} word offset {j} \
                     (byte offset {}) {x:#010x} vs singleton {y:#010x}; {bad_bytes} bytes differ",
                    NAMES[b],
                    j * 4
                ));
            }
        }
    }
}

fn parse_args() -> R<(PathBuf, PathBuf, PathBuf, usize, u64)> {
    let mut args = std::env::args().skip(1);
    let mut pos = Vec::new();
    let mut out = PathBuf::from(".");
    let mut rows_max = 64usize;
    let mut seeds = 2u64;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out needs a value")?),
            "--rows-max" => {
                rows_max = args
                    .next()
                    .ok_or("--rows-max needs a value")?
                    .parse()
                    .map_err(err("--rows-max"))?;
                if !(1..=64).contains(&rows_max) {
                    return Err("--rows-max must be 1..=64".into());
                }
            }
            "--seeds" => {
                seeds = args
                    .next()
                    .ok_or("--seeds needs a value")?
                    .parse()
                    .map_err(err("--seeds"))?;
            }
            _ => pos.push(PathBuf::from(a)),
        }
    }
    if pos.len() != 2 {
        return Err(
            "usage: lane_hc8_rows_oracle <BASE> <SIDECAR.hc8> [--out DIR] [--rows-max N] [--seeds N]"
                .into(),
        );
    }
    let sidecar = pos.pop().unwrap();
    let base = pos.pop().unwrap();
    Ok((base, sidecar, out, rows_max, seeds))
}

fn run() -> R<bool> {
    let (base_path, sidecar_path, out_dir, rows_max, seeds) = parse_args()?;
    let base = HfqFile::open(Path::new(&base_path)).map_err(|e| format!("base: {e}"))?;
    let bound = bind_hc8_sidecar(&base, &sidecar_path)?;
    let payloads = bound.read_verified()?;
    println!(
        "hc8 sidecar: {} tensors, {} payload bytes (every payload sha256-verified)",
        payloads.len(),
        bound.payload_bytes()
    );
    let mut gpu = Gpu::init().map_err(err("Gpu::init"))?;
    println!("device arch: {}", gpu.arch);
    if gpu.arch != "gfx1201" {
        return Err(format!("needs exact gfx1201, got {}", gpu.arch));
    }
    if !gpu.flags.qwen4_lane_hc_allrows_enabled() {
        return Err(
            "HIPFIRE_QWEN4_LANE_HC_ALLROWS is off: the all-rows arm would not engage".into(),
        );
    }

    // Group the payloads by connection prefix.
    let mut by_prefix: BTreeMap<String, [Option<usize>; 2]> = BTreeMap::new();
    for (i, p) in payloads.iter().enumerate() {
        let (prefix, slot) = if let Some(x) = p.name.strip_suffix(".input_mix_weight_down.weight") {
            (x, 0)
        } else if let Some(x) = p.name.strip_suffix(".input_mix_weight_up.weight") {
            (x, 1)
        } else {
            return Err(format!("unexpected sidecar tensor {}", p.name));
        };
        let want = if slot == 0 {
            [RANK, WIDE]
        } else {
            [WIDE, RANK]
        };
        if p.shape != want {
            return Err(format!(
                "{}: shape {:?}, expected {want:?}",
                p.name, p.shape
            ));
        }
        let slots = by_prefix.entry(prefix.to_string()).or_default();
        if slots[slot].replace(i).is_some() {
            return Err(format!("duplicate sidecar tensor {}", p.name));
        }
    }
    let mut conns = Vec::new();
    for (prefix, slots) in &by_prefix {
        let (Some(d), Some(u)) = (slots[0], slots[1]) else {
            return Err(format!("{prefix}: down/up payloads incomplete"));
        };
        let upload_hc8 = |gpu: &mut Gpu, bytes: &[u8]| -> R<GpuTensor> {
            let mut t = gpu
                .upload_raw(bytes, &[bytes.len()])
                .map_err(err("upload hc8"))?;
            t.dtype = DType::HC8E4M3Row;
            Ok(t)
        };
        let bf16 = |gpu: &mut Gpu, name: &str, numel: usize| -> R<Option<GpuTensor>> {
            match base.tensor_data_vec(name) {
                Some((_, bytes)) => Ok(Some(upload_bf16(gpu, &bytes, numel)?)),
                None => Ok(None),
            }
        };
        let norm = bf16(&mut gpu, &format!("{prefix}.hc_norm.weight"), WIDE)?
            .ok_or_else(|| format!("{prefix}.hc_norm.weight missing from the base"))?;
        let inject = bf16(
            &mut gpu,
            &format!("{prefix}.block_inject_weight.weight"),
            BRANCHES * WIDE,
        )?;
        conns.push(Conn {
            name: prefix.clone(),
            norm,
            down: upload_hc8(&mut gpu, &payloads[d].bytes)?,
            up: upload_hc8(&mut gpu, &payloads[u].bytes)?,
            inject,
        });
    }
    let mut trunk: BTreeMap<usize, [Option<usize>; 2]> = BTreeMap::new();
    for (i, c) in conns.iter().enumerate() {
        if let Some((id, mlp)) = trunk_layer(&c.name) {
            trunk.entry(id).or_default()[usize::from(mlp)] = Some(i);
        }
    }
    let mut layers = Vec::new();
    for (id, s) in &trunk {
        let (Some(a), Some(m)) = (s[0], s[1]) else {
            return Err(format!("trunk layer {id}: attn/mlp connection missing"));
        };
        layers.push((*id, a, m));
    }
    let model = Model { conns, layers };
    println!(
        "connections: {} ({} trunk layers)",
        model.conns.len(),
        model.layers.len()
    );

    let mut stages = Vec::new();
    for c in 0..model.conns.len() {
        stages.push(Stage::Read(c));
        if trunk_layer(&model.conns[c].name).is_none() {
            stages.push(Stage::Final(c));
        }
    }
    for l in 0..model.layers.len().saturating_sub(1) {
        let (a, b) = (&model.layers[l], &model.layers[l + 1]);
        if b.0 == a.0 + 1
            && model.conns[a.1].inject.is_some()
            && model.conns[a.2].inject.is_some()
            && model.conns[b.1].inject.is_some()
            && model.conns[b.2].inject.is_some()
        {
            stages.push(Stage::Chain(l, l + 1));
        }
    }
    let n_read = stages.iter().filter(|s| s.kind() == Kind::Read).count();
    let n_final = stages.iter().filter(|s| s.kind() == Kind::Final).count();
    let n_chain = stages.iter().filter(|s| s.kind() == Kind::Chain).count();
    println!("stages: {n_read} read, {n_final} final, {n_chain} chain; rows 1..={rows_max}; {seeds} seeds");

    let ctx = DispatchCtx::new(&gpu);
    let bufs = Bufs::new(&mut gpu)?;
    let mut st = Stats::default();
    let mut per_kind: BTreeMap<&'static str, (u64, u64, u64)> = BTreeMap::new();
    for seed in 0..seeds {
        let base = base_rows(seed);
        let ident: Vec<usize> = (0..BASE_ROWS).collect();
        let single_init = make_init(&base, &ident);
        for &stage in &stages {
            let tag = format!("seed {seed} {}", label(&model, stage));
            let kind = stage.kind();
            let single = run_arm(
                &mut gpu,
                &ctx,
                &model,
                stage,
                &bufs,
                &single_init,
                BASE_ROWS,
                true,
            )?;
            let before = (
                st.cells,
                st.bytes,
                st.mismatching_bytes + st.violation_words,
            );
            for rows in 1..=rows_max {
                for reversed in [false, true] {
                    let order: Vec<usize> = if reversed {
                        (0..rows).rev().collect()
                    } else {
                        (0..rows).collect()
                    };
                    let init = make_init(&base, &order);
                    let all = run_arm(&mut gpu, &ctx, &model, stage, &bufs, &init, rows, false)?;
                    let cell = format!("{tag} rev={reversed}");
                    compare_cell(&mut st, &cell, kind, rows, &order, &all, &single, &init);
                }
            }
            let name = match kind {
                Kind::Read => "read",
                Kind::Final => "final",
                Kind::Chain => "chain",
            };
            let e = per_kind.entry(name).or_default();
            e.0 += st.cells - before.0;
            e.1 += st.bytes - before.1;
            e.2 += st.mismatching_bytes + st.violation_words - before.2;
            if st.mismatching_bytes + st.violation_words != before.2 {
                eprintln!("MISMATCH {tag}");
            }
        }
    }

    let bad = st.mismatching_bytes + st.violation_words;
    for line in &st.first {
        eprintln!("first mismatch: {line}");
    }
    let json = serde_json::json!({
        "oracle": "lane_hc8_rows_oracle",
        "arch": gpu.arch,
        "base": base_path.display().to_string(),
        "sidecar": sidecar_path.display().to_string(),
        "sidecar_tensors": payloads.len(),
        "connections": model.conns.len(),
        "trunk_layers": model.layers.len(),
        "stages": { "read": n_read, "final": n_final, "chain": n_chain },
        "rows": format!("1..={rows_max}"),
        "orders": ["forward", "reversed"],
        "seeds": seeds,
        "cells": st.cells,
        "bytes_compared": st.bytes,
        "mismatching_bytes": st.mismatching_bytes,
        "violation_words": st.violation_words,
        "per_kind": per_kind.iter().map(|(k, (c, b, m))| (k.to_string(), serde_json::json!({"cells": c, "bytes": b, "mismatches": m}))).collect::<BTreeMap<_, _>>(),
        "per_buffer_bytes": st.per_buffer_bytes,
        "per_buffer_mismatching_bytes": st.per_buffer_mismatch,
        "first_mismatches": st.first,
        "pass": bad == 0,
    });
    std::fs::create_dir_all(&out_dir).map_err(err("create out dir"))?;
    std::fs::write(
        out_dir.join("oracle.json"),
        serde_json::to_string_pretty(&json).map_err(err("json"))?,
    )
    .map_err(err("write oracle.json"))?;
    println!(
        "lane_hc8_rows_oracle: {} cells, {} bytes compared, {} mismatching bytes",
        st.cells, st.bytes, bad
    );
    Ok(bad == 0)
}

fn main() {
    match run() {
        Ok(true) => println!("PASS"),
        Ok(false) => {
            println!("FAIL");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("lane_hc8_rows_oracle: {e}");
            std::process::exit(2);
        }
    }
}
