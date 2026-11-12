// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle, launch census and bench of the Flash-Next lane step's all-rows HC arm
//! (`HIPFIRE_QWEN4_LANE_HC_ALLROWS=1`, exact gfx1201).
//!
//! A lane step of more than eight rows runs its HC stages (HcRead / HcWrite / FinalHyper)
//! once per group of at most eight rows (`group_segments`); with the flag they run ONE
//! launch per kernel over every row, under `Gpu::set_wide_verify_rows(n)` set only around
//! those steps. This example drives the real dispatch entry points
//! (`execute_hyper_read`, `execute_hyper_write`, `execute_final_hyper`, and the write -> read
//! fusion of `execute_validated_steps`) on three arms and compares every buffer word for
//! word:
//!
//! * `groups`    today: the same op once per lane group on row-offset views, scope off;
//! * `all`       one call over all `n` rows at row 0 under the wide scope;
//! * `singleton` one `rows == 1` call per row, scope off.
//!
//! Stages: (a) read, (b) write (gates computed inside), (c) the layer-boundary chain
//! `W(attn) R(mlp) Clear W(mlp) R(attn') W(attn') R(mlp') Clear` that the fused
//! write+read launch, its `Clear` absorption and the gate-quarter hand-off produce (run
//! once with room for both quarter slots in `gates`, once with a tight `gates` that holds
//! none), (d) the final read. Every buffer has 72 rows of capacity and is
//! sentinel-prefilled with a NaN pattern; rows past `n` must stay untouched in every arm,
//! read-only buffers must equal their upload, the chain's clear target must be +0.0, and
//! the quarters the fused writes leave in `gates` are compared per row and slot (their
//! layout depends on the group, their content does not).
//!
//! Negative controls (each must be DETECTED): wrong group offsets, a shifted activation
//! view, an off-by-one row count, an over-wide tiled launch over poisoned pad rows, and
//! bit flips of a live word, a tail word and a quarter word.
//!
//! ```text
//! HIPFIRE_QWEN4_LANE_HC_ALLROWS=1 cargo run --release -p hipfire-dispatch \
//!   --features test-utils --example hc_allrows_oracle -- \
//!   [--out DIR] [--model PATH] [--layers 0,1,..] [--seeds N] [--rows-max 64] \
//!   [--bench | --bench-only] [--iters N]
//! ```
//!
//! Exit nonzero on a non-gfx1201 device, a missing flag, a launch error, any mismatch,
//! census failure or undetected control. Receipts: `oracle.json`, `bench.json` in `--out`.

use std::collections::BTreeMap;
use std::fmt::{Debug, Write as _};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_final_hyper, execute_hyper_read, execute_hyper_write,
    execute_validated_steps_with_moe_hooks, validate_steps, ClearOp, HyperReadOp, HyperWriteOp,
    MoeStepHooks, Step,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{DType, Gpu, GpuTensor, HipResult, LabLaunchWitness};

type R<T> = Result<T, String>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const RANK: usize = 320;
/// Lane group width (`LANE_MAX_ROWS`).
const LANE_ROWS: usize = 8;
/// Rows every scratch buffer holds: the widest step (64) plus canary rows.
const CAP: usize = 72;
/// Per-row gate words: the live gates plus two quarter slots of 16 words.
const GATES_PER_ROW_FULL: usize = BRANCHES + 32;
const GATES_FULL: usize = CAP * GATES_PER_ROW_FULL;
const GATES_TIGHT: usize = CAP * BRANCHES;
const SENTINEL: u32 = 0x7fc0_beef;
const POISON_NAN: u32 = 0x7fc0_dead;
const DEFAULT_LAYERS: [usize; 4] = [0, 1, 23, 47];
const CENSUS_ROWS: [usize; 4] = [8, 16, 24, 32];
const EXPLICIT_ROWS: [usize; 6] = [9, 16, 24, 31, 32, 48];

// ---- buffer table ------------------------------------------------------------------------

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
/// Words per row of each buffer.
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
    /// Unit-variance bell (sum of four uniforms).
    fn normal(&mut self) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.732_050_8
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_round(x: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(x)) << 16)
}

fn synth_bf16(rng: &mut Rng, count: usize, mut value: impl FnMut(&mut Rng) -> f32) -> Vec<u8> {
    (0..count)
        .flat_map(|_| bf16_bits(value(rng)).to_le_bytes())
        .collect()
}

/// One activation row: a pattern family (uniform, heavy-tailed, zeros, signed zeros and
/// subnormals, bell, sparse spikes) at a scale in `0.25..=4`.
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

/// A pad row's poison: finite garbage on even rows, a NaN with a foreign payload on odd.
fn garbage_row(rng: &mut Rng, width: usize, odd: bool) -> Vec<u32> {
    if odd {
        return vec![POISON_NAN; width];
    }
    (0..width)
        .map(|_| ((rng.unit() * 2.0 - 1.0) * 10f32.powi(rng.below(4) as i32) + 0.5).to_bits())
        .collect()
}

/// Host images of every buffer an arm starts from.
struct Init {
    words: Vec<Vec<u32>>,
}

/// Streams and block outputs of rows `0..n`, the clear target prefilled with nonzero data,
/// everything else the NaN sentinel; `poison` fills the pad rows `n..ceil(n/8)*8` of every
/// row buffer but `gates` with garbage instead.
fn make_init(n: usize, seed: u64, gates_words: usize, poison: bool) -> Init {
    let mut rng = Rng::new(0xC0FF_EE00 ^ (seed << 20) ^ n as u64);
    let mut words: Vec<Vec<u32>> = (0..NBUF)
        .map(|b| {
            let len = if b == GATES {
                gates_words
            } else {
                CAP * WIDTH[b]
            };
            vec![SENTINEL; len]
        })
        .collect();
    for r in 0..n {
        let streams = gen_row(&mut rng, WIDE, true);
        let wmix = gen_row(&mut rng, HIDDEN, false);
        for (i, v) in streams.iter().enumerate() {
            words[STREAMS][r * WIDE + i] = v.to_bits();
        }
        for (i, v) in wmix.iter().enumerate() {
            words[WMIX][r * HIDDEN + i] = v.to_bits();
        }
        for i in 0..HIDDEN {
            words[CLEAR][r * HIDDEN + i] = (1.0 + rng.unit()).to_bits();
        }
    }
    if poison {
        for r in n..n.div_ceil(LANE_ROWS) * LANE_ROWS {
            for b in [STREAMS, NORMALIZED, LOW, UP, MIXED, WMIX, CLEAR] {
                let row = garbage_row(&mut rng, WIDTH[b], r % 2 == 1);
                words[b][r * WIDTH[b]..(r + 1) * WIDTH[b]].copy_from_slice(&row);
            }
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
    let t = gpu
        .alloc_tensor(&[numel], DType::BF16)
        .map_err(err("alloc bf16"))?;
    gpu.hip
        .memcpy_htod(&t.buf, bytes)
        .map_err(err("upload bf16"))?;
    Ok(t)
}

fn wref(buf: &GpuTensor, m: usize, k: usize) -> WeightRef<'_> {
    WeightRef {
        buf,
        dtype: DType::BF16,
        m,
        k,
        row_stride: 2 * k,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

/// The scratch of one arm: every buffer at `CAP` rows of its width (`gates` at the
/// variant's size) plus the BF16 path's unread rotation scratch.
struct Bufs {
    t: Vec<GpuTensor>,
    rotation: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu, gates_words: usize) -> R<Self> {
        let mut t = Vec::with_capacity(NBUF);
        for b in 0..NBUF {
            let len = if b == GATES {
                gates_words
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

    /// Views of every buffer from `begin` to the end of its allocation: what a group of
    /// the lane step binds.
    fn views(&self, begin: usize) -> Views {
        Views(
            (0..NBUF)
                .map(|b| {
                    let skip = begin * WIDTH[b];
                    self.t[b].sub_offset(skip, self.t[b].numel() - skip)
                })
                .collect(),
        )
    }

    fn reset(&self, gpu: &Gpu, init: &Init) -> R<()> {
        for b in 0..NBUF {
            debug_assert_eq!(init.words[b].len(), self.t[b].numel());
            htod_words(gpu, &self.t[b], &init.words[b])?;
        }
        Ok(())
    }

    fn snapshot(&self, gpu: &Gpu) -> R<Vec<Vec<u32>>> {
        self.t.iter().map(|t| dtoh_words(gpu, t)).collect()
    }
}

struct Views(Vec<GpuTensor>);

// ---- weights -----------------------------------------------------------------------------

struct ReadW {
    norm: GpuTensor,
    down: GpuTensor,
    up: GpuTensor,
}

struct Conn {
    read: ReadW,
    inject: GpuTensor,
}

struct Layer {
    id: usize,
    attn: Conn,
    mlp: Conn,
}

struct WeightSet {
    label: String,
    layers: Vec<Layer>,
    fin: ReadW,
}

fn synth_read(gpu: &mut Gpu, rng: &mut Rng) -> R<ReadW> {
    let norm = synth_bf16(rng, WIDE, |r| 0.5 + r.unit());
    let down = synth_bf16(rng, RANK * WIDE, |r| r.normal() * 0.02);
    let up = synth_bf16(rng, WIDE * RANK, |r| r.normal() * 0.02);
    Ok(ReadW {
        norm: upload_bf16(gpu, &norm, WIDE)?,
        down: upload_bf16(gpu, &down, RANK * WIDE)?,
        up: upload_bf16(gpu, &up, WIDE * RANK)?,
    })
}

fn synth_conn(gpu: &mut Gpu, rng: &mut Rng) -> R<Conn> {
    let read = synth_read(gpu, rng)?;
    let inject = synth_bf16(rng, BRANCHES * WIDE, |r| r.normal() * 0.05);
    Ok(Conn {
        read,
        inject: upload_bf16(gpu, &inject, BRANCHES * WIDE)?,
    })
}

fn synthetic_weights(gpu: &mut Gpu) -> R<WeightSet> {
    let mut rng = Rng::new(0x5EED_0001);
    let mut layers = Vec::new();
    for id in 0..2 {
        layers.push(Layer {
            id,
            attn: synth_conn(gpu, &mut rng)?,
            mlp: synth_conn(gpu, &mut rng)?,
        });
    }
    Ok(WeightSet {
        label: "synthetic".into(),
        layers,
        fin: synth_read(gpu, &mut rng)?,
    })
}

/// Minimal HFQM index reader (tensor name -> byte range), as `otile_bf16_rows_oracle`.
struct Hfq {
    file: File,
    tensors: BTreeMap<String, (u64, u64, Vec<u32>, u8)>,
}

impl Hfq {
    fn open(path: &str) -> R<Self> {
        let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
        let mut hdr = [0u8; 32];
        file.read_exact_at(&mut hdr, 0).map_err(err("header"))?;
        if &hdr[..4] != b"HFQM" {
            return Err(format!("{path}: not an HFQM container"));
        }
        let n = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        let meta_off = u64::from_le_bytes(hdr[16..24].try_into().unwrap());
        let data_off = u64::from_le_bytes(hdr[24..32].try_into().unwrap());
        let mut meta = vec![0u8; (data_off - meta_off) as usize];
        file.read_exact_at(&mut meta, meta_off)
            .map_err(err("index"))?;
        // Metadata JSON ends at its closing top-level brace; the tensor index follows.
        let (mut depth, mut in_str, mut esc, mut end) = (0i32, false, false, 0usize);
        for (i, &b) in meta.iter().enumerate() {
            if esc {
                esc = false;
                continue;
            }
            if b == b'\\' && in_str {
                esc = true;
                continue;
            }
            if b == b'"' {
                in_str = !in_str;
                continue;
            }
            if !in_str {
                if b == b'{' {
                    depth += 1;
                } else if b == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
            }
        }
        if end == 0 {
            return Err(format!("{path}: metadata JSON not terminated"));
        }
        let mut pos = end;
        let idx_n = u32::from_le_bytes(meta[pos..pos + 4].try_into().unwrap()) as usize;
        if idx_n != n {
            return Err(format!("{path}: tensor count {idx_n} != {n}"));
        }
        pos += 4;
        let mut off = data_off;
        let mut tensors = BTreeMap::new();
        for _ in 0..n {
            let nl = u16::from_le_bytes(meta[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            let name = String::from_utf8_lossy(&meta[pos..pos + nl]).to_string();
            pos += nl;
            let qt = meta[pos];
            let nd = meta[pos + 1] as usize;
            pos += 2;
            let shape: Vec<u32> = (0..nd)
                .map(|d| u32::from_le_bytes(meta[pos + 4 * d..pos + 4 * d + 4].try_into().unwrap()))
                .collect();
            pos += 4 * nd;
            pos += 4; // group size
            let size = u64::from_le_bytes(meta[pos..pos + 8].try_into().unwrap());
            pos += 8;
            tensors.insert(name, (off, size, shape, qt));
            off += size;
        }
        Ok(Hfq { file, tensors })
    }

    /// BF16 tensor `name` of exactly `shape`, as raw little-endian u16 bytes.
    fn bf16(&self, name: &str, shape: &[usize]) -> R<Vec<u8>> {
        let (off, size, have, qt) = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name} missing"))?;
        let want: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
        let numel: usize = shape.iter().product();
        if *qt != 16 || *have != want || *size as usize != numel * 2 {
            return Err(format!(
                "{name}: expected BF16 {shape:?}, container has quant type {qt} shape {have:?} \
                 ({size} bytes)"
            ));
        }
        let mut buf = vec![0u8; *size as usize];
        self.file
            .read_exact_at(&mut buf, *off)
            .map_err(err("tensor read"))?;
        Ok(buf)
    }
}

fn real_read(gpu: &mut Gpu, hfq: &Hfq, prefix: &str) -> R<ReadW> {
    let norm = hfq.bf16(&format!("{prefix}.hc_norm.weight"), &[WIDE])?;
    let down = hfq.bf16(
        &format!("{prefix}.input_mix_weight_down.weight"),
        &[RANK, WIDE],
    )?;
    let up = hfq.bf16(
        &format!("{prefix}.input_mix_weight_up.weight"),
        &[WIDE, RANK],
    )?;
    Ok(ReadW {
        norm: upload_bf16(gpu, &norm, WIDE)?,
        down: upload_bf16(gpu, &down, RANK * WIDE)?,
        up: upload_bf16(gpu, &up, WIDE * RANK)?,
    })
}

fn real_conn(gpu: &mut Gpu, hfq: &Hfq, prefix: &str) -> R<Conn> {
    let read = real_read(gpu, hfq, prefix)?;
    let inject = hfq.bf16(
        &format!("{prefix}.block_inject_weight.weight"),
        &[BRANCHES, WIDE],
    )?;
    Ok(Conn {
        read,
        inject: upload_bf16(gpu, &inject, BRANCHES * WIDE)?,
    })
}

fn real_weights(gpu: &mut Gpu, path: &str, layers: &[usize]) -> R<WeightSet> {
    let hfq = Hfq::open(path)?;
    let mut out = Vec::new();
    for &id in layers {
        let p = format!("model.language_model.layers.{id}");
        out.push(Layer {
            id,
            attn: real_conn(gpu, &hfq, &format!("{p}.attn_hyper_connection"))?,
            mlp: real_conn(gpu, &hfq, &format!("{p}.mlp_hyper_connection"))?,
        });
    }
    Ok(WeightSet {
        label: format!("real:{path}"),
        layers: out,
        fin: real_read(gpu, &hfq, "model.language_model.hyper_connection_mixer")?,
    })
}

// ---- stages and ops ----------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Read,
    Write,
    Chain,
    Final,
}

#[derive(Clone, Copy)]
enum Stage {
    Read {
        layer: usize,
        mlp: bool,
    },
    Write {
        layer: usize,
        mlp: bool,
    },
    /// `W(attn first) R(mlp first) Clear W(mlp first) R(attn second) W(attn second)
    /// R(mlp second) Clear`; `tight`: `gates` holds no quarter slot.
    Chain {
        first: usize,
        second: usize,
        tight: bool,
    },
    Final,
}

impl Stage {
    fn kind(self) -> Kind {
        match self {
            Stage::Read { .. } => Kind::Read,
            Stage::Write { .. } => Kind::Write,
            Stage::Chain { .. } => Kind::Chain,
            Stage::Final => Kind::Final,
        }
    }
    fn tight(self) -> bool {
        matches!(self, Stage::Chain { tight: true, .. })
    }
    /// Whether the fused writes' gate quarters are compared (room for both slots).
    fn quarters(self) -> bool {
        matches!(self, Stage::Chain { tight: false, .. })
    }
    fn gates_words(self) -> usize {
        if self.tight() {
            GATES_TIGHT
        } else {
            GATES_FULL
        }
    }
    fn label(self, set: &WeightSet) -> String {
        let conn = |mlp: bool| if mlp { "mlp" } else { "attn" };
        match self {
            Stage::Read { layer, mlp } => format!("read L{} {}", set.layers[layer].id, conn(mlp)),
            Stage::Write { layer, mlp } => {
                format!("write L{} {}", set.layers[layer].id, conn(mlp))
            }
            Stage::Chain {
                first,
                second,
                tight,
            } => format!(
                "chain L{}->L{}{}",
                set.layers[first].id,
                set.layers[second].id,
                if tight { " tight-gates" } else { "" }
            ),
            Stage::Final => "final".into(),
        }
    }
}

fn read_op<'a>(
    r: &'a ReadW,
    v: &'a Views,
    rotation: &'a GpuTensor,
    rows: usize,
) -> HyperReadOp<'a> {
    HyperReadOp {
        state_bf16: false,
        input: &v.0[STREAMS],
        norm_weight: &r.norm,
        input_mix_down: wref(&r.down, RANK, WIDE),
        input_mix_up: wref(&r.up, WIDE, RANK),
        normalized: &v.0[NORMALIZED],
        low: &v.0[LOW],
        up: &v.0[UP],
        mixed: &v.0[MIXED],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        low_rank: RANK,
        rotation,
    }
}

fn write_op<'a>(
    conn: &'a Conn,
    v: &'a Views,
    rotation: &'a GpuTensor,
    rows: usize,
) -> HyperWriteOp<'a> {
    HyperWriteOp {
        state_bf16: false,
        input: &v.0[STREAMS],
        norm_weight: &conn.read.norm,
        block_inject: wref(&conn.inject, BRANCHES, WIDE),
        normalized: &v.0[NORMALIZED],
        mixed: &v.0[WMIX],
        gates: &v.0[GATES],
        output: &v.0[STREAMS],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        rotation,
        ar: None,
    }
}

/// What one group (or the all-rows segment, or one row) executes.
enum Plan<'a> {
    Read(HyperReadOp<'a>),
    Write(HyperWriteOp<'a>),
    Final(HyperReadOp<'a>),
    Steps(Vec<Step<'a>>),
}

fn build_plan<'a>(
    gpu: &Gpu,
    set: &'a WeightSet,
    stage: Stage,
    v: &'a Views,
    rotation: &'a GpuTensor,
    rows: usize,
) -> R<Plan<'a>> {
    let conn = |layer: usize, mlp: bool| {
        let l = &set.layers[layer];
        if mlp {
            &l.mlp
        } else {
            &l.attn
        }
    };
    Ok(match stage {
        Stage::Read { layer, mlp } => {
            let op = read_op(&conn(layer, mlp).read, v, rotation, rows);
            op.validate_for_gpu(gpu).map_err(dbg_err)?;
            Plan::Read(op)
        }
        Stage::Write { layer, mlp } => {
            let op = write_op(conn(layer, mlp), v, rotation, rows);
            op.validate_for_gpu(gpu).map_err(dbg_err)?;
            Plan::Write(op)
        }
        Stage::Final => {
            let op = read_op(&set.fin, v, rotation, rows);
            op.validate_for_gpu(gpu).map_err(dbg_err)?;
            Plan::Final(op)
        }
        Stage::Chain { first, second, .. } => {
            let clear = || {
                Step::Clear(ClearOp {
                    tensor: &v.0[CLEAR],
                    elements: rows * HIDDEN,
                })
            };
            let steps = vec![
                Step::HyperWrite(write_op(conn(first, false), v, rotation, rows)),
                Step::HyperRead(read_op(&conn(first, true).read, v, rotation, rows)),
                clear(),
                Step::HyperWrite(write_op(conn(first, true), v, rotation, rows)),
                Step::HyperRead(read_op(&conn(second, false).read, v, rotation, rows)),
                Step::HyperWrite(write_op(conn(second, false), v, rotation, rows)),
                Step::HyperRead(read_op(&conn(second, true).read, v, rotation, rows)),
                clear(),
            ];
            validate_steps(gpu, &steps).map_err(dbg_err)?;
            Plan::Steps(steps)
        }
    })
}

// ---- launch log and execution ------------------------------------------------------------

#[derive(Clone)]
struct Launch {
    symbol: String,
    grid: [u32; 3],
    block: [u32; 3],
}

#[derive(Default)]
struct Log {
    on: AtomicBool,
    launches: Mutex<Vec<Launch>>,
}

impl Log {
    fn len(&self) -> usize {
        self.launches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
    fn start(&self) {
        self.launches
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.on.store(true, Ordering::Relaxed);
    }
    fn stop(&self) -> Vec<Launch> {
        self.on.store(false, Ordering::Relaxed);
        std::mem::take(&mut *self.launches.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// The interpreter's step announcements: records where each announced step starts in the
/// launch log and opens the all-rows wide scope (`HcAllRowsScope` of the lane forward) for
/// the HC steps of `scope` rows only.
struct Probe<'l> {
    scope: Option<usize>,
    prior: usize,
    log: &'l Log,
    marks: Vec<usize>,
}

impl MoeStepHooks for Probe<'_> {
    fn before_moe(&mut self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        Ok(())
    }
    fn after_moe(&mut self, _gpu: &mut Gpu) -> Result<(), DispatchError> {
        Ok(())
    }
    fn before_step(&mut self, gpu: &mut Gpu, step: &Step<'_>) -> Result<(), DispatchError> {
        self.marks.push(self.log.len());
        let rows = match step {
            Step::HyperRead(op) => Some(op.rows),
            Step::HyperWrite(op) => Some(op.rows),
            _ => None,
        };
        match (rows, self.scope) {
            (Some(rows), Some(scope)) if rows == scope => {
                gpu.set_wide_verify_rows(scope);
                if scope > LANE_ROWS && !gpu.lane_hc_allrows_applies(scope) {
                    return Err(DispatchError::Hip(format!(
                        "all-rows arm not engaged for {scope} rows (lane_hc_allrows_applies false)"
                    )));
                }
            }
            _ => {
                gpu.set_wide_verify_rows(self.prior);
            }
        }
        Ok(())
    }
}

struct CallRec {
    marks: Vec<usize>,
    end: usize,
}

fn with_scope<T>(gpu: &mut Gpu, scope: Option<usize>, f: impl FnOnce(&mut Gpu) -> R<T>) -> R<T> {
    let prior = gpu.wide_verify_rows;
    let result = match scope {
        Some(rows) => {
            gpu.set_wide_verify_rows(rows);
            if rows > LANE_ROWS && !gpu.lane_hc_allrows_applies(rows) {
                Err(format!(
                    "all-rows arm not engaged for {rows} rows (lane_hc_allrows_applies false)"
                ))
            } else {
                f(gpu)
            }
        }
        None => f(gpu),
    };
    gpu.set_wide_verify_rows(prior);
    result
}

fn exec(
    plan: &Plan<'_>,
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    scope: Option<usize>,
    log: &Log,
) -> R<CallRec> {
    let mut marks = Vec::new();
    match plan {
        Plan::Read(op) => with_scope(gpu, scope, |g| execute_hyper_read(g, op).map_err(dbg_err))?,
        Plan::Write(op) => with_scope(gpu, scope, |g| execute_hyper_write(g, op).map_err(dbg_err))?,
        Plan::Final(op) => with_scope(gpu, scope, |g| execute_final_hyper(g, op).map_err(dbg_err))?,
        Plan::Steps(steps) => {
            let mut probe = Probe {
                scope,
                prior: gpu.wide_verify_rows,
                log,
                marks: Vec::new(),
            };
            let result = execute_validated_steps_with_moe_hooks(gpu, ctx, steps, &mut probe);
            gpu.set_wide_verify_rows(probe.prior);
            result.map_err(dbg_err)?;
            marks = probe.marks;
        }
    }
    Ok(CallRec {
        marks,
        end: log.len(),
    })
}

// ---- arms --------------------------------------------------------------------------------

struct Env {
    gpu: Gpu,
    ctx: DispatchCtx,
    full: Bufs,
    tight: Bufs,
    log: Arc<Log>,
}

#[derive(Clone)]
enum Arm {
    /// One call over `rows` rows at row 0 under the wide scope.
    AllRows(usize),
    /// One call per `(begin, rows)` group, scope off.
    Groups(Vec<(usize, usize)>),
    /// One `rows == 1` call per row, scope off.
    Singleton,
}

struct Outcome {
    snap: Vec<Vec<u32>>,
    /// Per quarter slot the fused writes left in `gates`, 16 words per row, global row order.
    quarters: [Vec<u32>; 2],
    calls: Vec<CallRec>,
    launches: Vec<Launch>,
}

fn segments(arm: &Arm, n: usize) -> (Vec<(usize, usize)>, Option<usize>) {
    match arm {
        Arm::AllRows(rows) => (vec![(0, *rows)], Some(*rows)),
        Arm::Groups(groups) => (groups.clone(), None),
        Arm::Singleton => ((0..n).map(|i| (i, 1)).collect(), None),
    }
}

fn run_arm(
    env: &mut Env,
    set: &WeightSet,
    stage: Stage,
    arm: &Arm,
    n: usize,
    init: &Init,
) -> R<Outcome> {
    let Env {
        gpu,
        ctx,
        full,
        tight,
        log,
    } = env;
    let bufs: &Bufs = if stage.tight() { tight } else { full };
    bufs.reset(gpu, init)?;
    let (segs, scope) = segments(arm, n);
    let mut calls = Vec::new();
    let mut quarters: [Vec<u32>; 2] = Default::default();
    let prior = gpu.set_wide_verify_rows(0);
    log.start();
    let result = (|| -> R<()> {
        for &(begin, rows) in &segs {
            let views = bufs.views(begin);
            let plan = build_plan(gpu, set, stage, &views, &bufs.rotation, rows)?;
            calls.push(exec(&plan, gpu, ctx, scope, log)?);
            if stage.quarters() {
                let gates = dtoh_words(gpu, &bufs.t[GATES])?;
                for (slot, out) in quarters.iter_mut().enumerate() {
                    let start = BRANCHES * (begin + rows) + 16 * rows * slot;
                    out.extend_from_slice(&gates[start..start + 16 * rows]);
                }
            }
        }
        Ok(())
    })();
    let launches = log.stop();
    gpu.set_wide_verify_rows(prior);
    result?;
    Ok(Outcome {
        snap: bufs.snapshot(gpu)?,
        quarters,
        calls,
        launches,
    })
}

// ---- lane partitions ---------------------------------------------------------------------

/// `group_segments` of the lane forward: consecutive lanes packed into groups of at most
/// eight rows; `(begin, rows)` per group.
fn group_segments(lanes: &[usize]) -> Vec<(usize, usize)> {
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut begin = 0;
    for &rows in lanes {
        match groups.last_mut() {
            Some((_, group_rows)) if *group_rows + rows <= LANE_ROWS => *group_rows += rows,
            _ => groups.push((begin, rows)),
        }
        begin += rows;
    }
    groups
}

fn uniform_lanes(n: usize, rows: usize) -> Vec<usize> {
    let mut lanes = vec![rows; n / rows];
    if n % rows != 0 {
        lanes.push(n % rows);
    }
    lanes
}

/// The lane partitions of `n` rows: all one-row lanes, uniform two-, three- and four-row
/// lanes, and a seeded random mix of 1..=4-row lanes.
fn lane_partitions(n: usize, seed: u64) -> Vec<(String, Vec<usize>)> {
    let mut rng = Rng::new(0x1A4E ^ (seed << 12) ^ n as u64);
    let mut mix = Vec::new();
    let mut sum = 0;
    while sum < n {
        let rows = (1 + rng.below(4) as usize).min(n - sum);
        mix.push(rows);
        sum += rows;
    }
    vec![
        ("1row".into(), uniform_lanes(n, 1)),
        ("2row".into(), uniform_lanes(n, 2)),
        ("3row".into(), uniform_lanes(n, 3)),
        ("4row".into(), uniform_lanes(n, 4)),
        ("mix".into(), mix),
    ]
}

/// The distinct group lists of the partitions, each named by the partitions that produce it.
fn distinct_groups(parts: &[(String, Vec<usize>)]) -> Vec<(String, Vec<(usize, usize)>)> {
    let mut out: Vec<(String, Vec<(usize, usize)>)> = Vec::new();
    for (name, lanes) in parts {
        let groups = group_segments(lanes);
        match out.iter_mut().find(|(_, g)| *g == groups) {
            Some((joined, _)) => {
                joined.push('+');
                joined.push_str(name);
            }
            None => out.push((name.clone(), groups)),
        }
    }
    if let Some((name, groups)) = out.first_mut() {
        if groups.len() == 1 {
            *name = "single".into();
        }
    }
    out
}

// ---- comparison --------------------------------------------------------------------------

/// Whether the stage writes each buffer; the others must equal their upload everywhere.
fn written(kind: Kind) -> [bool; NBUF] {
    match kind {
        Kind::Read | Kind::Final => [false, true, true, true, true, false, false, false],
        Kind::Write => [true, true, false, false, false, true, false, false],
        Kind::Chain => [true, true, true, true, true, true, false, true],
    }
}

/// Words of `buf` rows `0..n` (`gates`: the live gates).
fn live_end(buf: usize, n: usize) -> usize {
    n * WIDTH[buf]
}

/// First word of a written buffer that must still equal its upload: past the live rows,
/// and for the chain's `gates` past the quarter slots.
fn tail_start(kind: Kind, buf: usize, n: usize, gates_words: usize) -> usize {
    if buf == GATES && kind == Kind::Chain {
        (n * GATES_PER_ROW_FULL).min(gates_words)
    } else {
        live_end(buf, n)
    }
}

/// Canary and invariant violations of one arm: read-only buffers changed, words past the
/// live rows touched, the chain's clear target not +0.0.
fn integrity(
    kind: Kind,
    n: usize,
    gates_words: usize,
    out: &Outcome,
    init: &Init,
) -> Vec<(String, usize)> {
    let mut bad = Vec::new();
    for b in 0..NBUF {
        let is_written = written(kind)[b];
        let from = if is_written {
            tail_start(kind, b, n, gates_words)
        } else {
            0
        };
        let count = out.snap[b][from..]
            .iter()
            .zip(&init.words[b][from..])
            .filter(|(x, y)| x != y)
            .count();
        if count != 0 {
            let what = if is_written { "tail" } else { "readonly" };
            bad.push((format!("{what}:{}", NAMES[b]), count));
        }
    }
    if kind == Kind::Chain {
        let nonzero = out.snap[CLEAR][..n * HIDDEN]
            .iter()
            .filter(|&&w| w != 0)
            .count();
        if nonzero != 0 {
            bad.push(("clear:not-zero".into(), nonzero));
        }
    }
    bad
}

/// Mismatching live words between two arms (written buffers, and the quarters when both
/// arms produced them), with the first one located. `up` is never compared: only the
/// `rows == 1` route writes it (its up GEMV output, consumed inside the same read), the
/// multi-row routes fuse the up projection into the mix and leave it alone. With
/// `skip_gates` (a chain whose `gates` have no room for the quarter slots) the final
/// `gates` words are not compared either: nothing reads them once the chain's writes ran,
/// and the per-group views' overlapping quarters leave a different dead residue there.
fn compare(kind: Kind, n: usize, a: &Outcome, b: &Outcome) -> (usize, Option<String>) {
    compare_with(kind, false, n, a, b)
}

fn compare_with(
    kind: Kind,
    skip_gates: bool,
    n: usize,
    a: &Outcome,
    b: &Outcome,
) -> (usize, Option<String>) {
    let mut total = 0;
    let mut first = None;
    for buf in 0..NBUF {
        if !written(kind)[buf] || buf == UP || (skip_gates && buf == GATES) {
            continue;
        }
        for i in 0..live_end(buf, n) {
            let (x, y) = (a.snap[buf][i], b.snap[buf][i]);
            if x != y {
                total += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "{}[row {} col {}] {x:#010x} vs {y:#010x}",
                        NAMES[buf],
                        i / WIDTH[buf],
                        i % WIDTH[buf]
                    )
                });
            }
        }
    }
    for slot in 0..2 {
        let (qa, qb) = (&a.quarters[slot], &b.quarters[slot]);
        if qa.is_empty() || qb.is_empty() {
            continue;
        }
        if qa.len() != qb.len() {
            total += qa.len().abs_diff(qb.len());
            first.get_or_insert_with(|| {
                format!("quarters{slot}: {} vs {} words", qa.len(), qb.len())
            });
            continue;
        }
        for (i, (x, y)) in qa.iter().zip(qb).enumerate() {
            if x != y {
                total += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "quarters{slot}[row {} col {}] {x:#010x} vs {y:#010x}",
                        i / 16,
                        i % 16
                    )
                });
            }
        }
    }
    (total, first)
}

/// Per pair-launch counts of a chain: the launches between consecutive announced steps,
/// summed over the arm's calls.
fn pair_launches(out: &Outcome) -> Vec<usize> {
    let mut sums: Vec<usize> = Vec::new();
    for call in &out.calls {
        for (i, &start) in call.marks.iter().enumerate() {
            let end = call.marks.get(i + 1).copied().unwrap_or(call.end);
            if sums.len() <= i {
                sums.resize(i + 1, 0);
            }
            sums[i] += end - start;
        }
    }
    sums
}

/// Launch counts keyed by `symbol[grid]/block`.
fn symbol_counts(launches: &[Launch]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for l in launches {
        let key = format!(
            "{}[{}x{}x{}]/{}x{}x{}",
            l.symbol, l.grid[0], l.grid[1], l.grid[2], l.block[0], l.block[1], l.block[2]
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn fmt_symbols(counts: &BTreeMap<String, usize>) -> String {
    counts
        .iter()
        .map(|(s, c)| format!("{s}x{c}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---- report ------------------------------------------------------------------------------

struct CaseRec {
    weights: String,
    seed: u64,
    n: usize,
    partition: String,
    stage: String,
    comparison: &'static str,
    mismatch_words: usize,
    first_mismatch: Option<String>,
    violations: Vec<(String, usize)>,
    launches_ref: usize,
    launches_all: usize,
}

struct CensusRow {
    n: usize,
    partition: String,
    groups: usize,
    stage: String,
    ref_launches: usize,
    all_launches: usize,
    ref_symbols: BTreeMap<String, usize>,
    all_symbols: BTreeMap<String, usize>,
    /// Chain only: launches of each fused pair (steady state: the middle one).
    ref_pairs: Vec<usize>,
    all_pairs: Vec<usize>,
}

#[derive(Default)]
struct Report {
    cases: Vec<CaseRec>,
    controls: Vec<(String, bool)>,
    census: Vec<CensusRow>,
    census_failures: Vec<String>,
}

impl Report {
    fn mismatches(&self) -> usize {
        self.cases
            .iter()
            .map(|c| c.mismatch_words + c.violations.iter().map(|(_, n)| n).sum::<usize>())
            .sum()
    }

    fn detected(&self) -> usize {
        self.controls.iter().filter(|(_, d)| *d).count()
    }

    fn passed(&self) -> bool {
        self.mismatches() == 0
            && self.detected() == self.controls.len()
            && self.census_failures.is_empty()
    }
}

struct CaseCtx<'a> {
    set: &'a WeightSet,
    stage: Stage,
    seed: u64,
    n: usize,
}

#[allow(clippy::too_many_arguments)]
fn record(
    rep: &mut Report,
    cc: &CaseCtx<'_>,
    partition: &str,
    comparison: &'static str,
    a: &Outcome,
    b: &Outcome,
    inits: (&Init, &Init),
) {
    let kind = cc.stage.kind();
    let gates = cc.stage.gates_words();
    let (mismatch_words, first) = compare_with(kind, cc.stage.tight(), cc.n, a, b);
    let mut violations = integrity(kind, cc.n, gates, a, inits.0);
    violations.extend(integrity(kind, cc.n, gates, b, inits.1));
    let rec = CaseRec {
        weights: cc.set.label.clone(),
        seed: cc.seed,
        n: cc.n,
        partition: partition.into(),
        stage: cc.stage.label(cc.set),
        comparison,
        mismatch_words,
        first_mismatch: first,
        violations,
        launches_ref: b.launches.len(),
        launches_all: a.launches.len(),
    };
    if rec.mismatch_words != 0 || !rec.violations.is_empty() {
        eprintln!(
            "MISMATCH {} seed={} n={} {} {} {}: {} words {:?} {}",
            rec.weights,
            rec.seed,
            rec.n,
            rec.stage,
            rec.partition,
            rec.comparison,
            rec.mismatch_words,
            rec.violations,
            rec.first_mismatch.as_deref().unwrap_or("")
        );
    }
    rep.cases.push(rec);
}

// ---- oracle ------------------------------------------------------------------------------

fn stages(set: &WeightSet) -> Vec<Stage> {
    let mut out = Vec::new();
    let layers = set.layers.len();
    for layer in 0..layers {
        for mlp in [false, true] {
            out.push(Stage::Read { layer, mlp });
            out.push(Stage::Write { layer, mlp });
        }
        out.push(Stage::Chain {
            first: layer,
            second: (layer + 1) % layers,
            tight: false,
        });
    }
    out.push(Stage::Chain {
        first: 0,
        second: 1 % layers,
        tight: true,
    });
    out.push(Stage::Final);
    out
}

fn run_case(
    env: &mut Env,
    set: &WeightSet,
    stage: Stage,
    n: usize,
    seed: u64,
    groups: &[(String, Vec<(usize, usize)>)],
    rep: &mut Report,
) -> R<()> {
    let cc = CaseCtx {
        set,
        stage,
        seed,
        n,
    };
    let gates = stage.gates_words();
    let init = make_init(n, seed, gates, false);
    let all = run_arm(env, set, stage, &Arm::AllRows(n), n, &init)?;
    let single = run_arm(env, set, stage, &Arm::Singleton, n, &init)?;
    record(
        rep,
        &cc,
        "singleton",
        "vs_singleton",
        &all,
        &single,
        (&init, &init),
    );
    for (name, g) in groups {
        let reference = run_arm(env, set, stage, &Arm::Groups(g.clone()), n, &init)?;
        record(
            rep,
            &cc,
            name,
            "vs_groups",
            &all,
            &reference,
            (&init, &init),
        );
    }
    if n % LANE_ROWS != 0 {
        let poisoned = make_init(n, seed, gates, true);
        let bad = run_arm(env, set, stage, &Arm::AllRows(n), n, &poisoned)?;
        record(
            rep,
            &cc,
            "pad-poison",
            "pad_poison",
            &all,
            &bad,
            (&init, &poisoned),
        );
    }
    Ok(())
}

fn oracle(
    env: &mut Env,
    sets: &[WeightSet],
    rows: &[usize],
    seeds: u64,
    rep: &mut Report,
) -> R<()> {
    for set in sets {
        for seed in 0..seeds {
            for &n in rows {
                let before = rep.mismatches();
                let count = rep.cases.len();
                let groups = distinct_groups(&lane_partitions(n, seed));
                for stage in stages(set) {
                    run_case(env, set, stage, n, seed, &groups, rep)?;
                }
                println!(
                    "{} seed={seed} n={n}: cases={} mismatches={}",
                    set.label,
                    rep.cases.len() - count,
                    rep.mismatches() - before
                );
            }
        }
    }
    Ok(())
}

// ---- negative controls -------------------------------------------------------------------

fn control(rep: &mut Report, name: String, detected: bool) {
    println!(
        "control {name} {}",
        if detected { "DETECTED" } else { "MISSED" }
    );
    rep.controls.push((name, detected));
}

/// A comparison of `bad` against the reference `good` is detected when the live words differ
/// or `bad` breaks an invariant.
fn detects(kind: Kind, n: usize, gates: usize, good: &Outcome, bad: &Outcome, init: &Init) -> bool {
    compare(kind, n, good, bad).0 != 0 || !integrity(kind, n, gates, bad, init).is_empty()
}

fn gemv_launch(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    rows: usize,
    tiled: bool,
) -> R<()> {
    if tiled {
        return gpu
            .gemv_bf16_xf32_k4_rows_tiled(w, x, y, RANK, WIDE, Some(0.25), rows)
            .map_err(err("tiled gemv"));
    }
    with_scope(gpu, Some(rows), |g| {
        g.gemv_bf16_xf32_k4_rows(w, x, y, RANK, WIDE, Some(0.25), rows)
            .map_err(err("all-rows gemv"))
    })
}

fn words_differ(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Controls (i) shifted activation view and (iii) pad rows read by an over-wide launch, on
/// the all-rows HC down GEMV itself. The unaffected-by-poison half of (iii) is an oracle case.
fn gemv_controls(env: &mut Env, set: &WeightSet, rep: &mut Report) -> R<()> {
    let gpu = &mut env.gpu;
    let down = &set.layers[0].attn.read.down;
    let x = gpu
        .alloc_tensor(&[CAP * WIDE], DType::F32)
        .map_err(err("alloc x"))?;
    let ys: Vec<GpuTensor> = (0..3)
        .map(|_| {
            gpu.alloc_tensor(&[CAP * RANK], DType::F32)
                .map_err(err("alloc y"))
        })
        .collect::<R<_>>()?;
    let sentinel = vec![SENTINEL; CAP * RANK];
    let fetch_y = |gpu: &Gpu, y: &GpuTensor| dtoh_words(gpu, y);
    let reset_y = |gpu: &Gpu| -> R<()> {
        for y in &ys {
            htod_words(gpu, y, &sentinel)?;
        }
        Ok(())
    };

    // (i) the all-rows down GEMV with its activation view shifted by one row.
    let n = 24;
    let init = make_init(n, 100, GATES_FULL, false);
    htod_words(gpu, &x, &init.words[STREAMS])?;
    reset_y(gpu)?;
    gemv_launch(gpu, down, &x, &ys[0], n, false)?;
    let shifted = x.sub_offset(WIDE, CAP * WIDE - WIDE);
    gemv_launch(gpu, down, &shifted, &ys[1], n, false)?;
    let good = fetch_y(gpu, &ys[0])?;
    let bad = fetch_y(gpu, &ys[1])?;
    control(
        rep,
        "wrong_tile_partition_x_shift".into(),
        words_differ(&good[..n * RANK], &bad[..n * RANK]) != 0,
    );

    // (iii) poisoned pad rows [n, ceil(n/8)*8).
    for n in [13usize, 30] {
        let pad_end = n.div_ceil(LANE_ROWS) * LANE_ROWS;
        let clean = make_init(n, 101, GATES_FULL, false);
        let poisoned = make_init(n, 101, GATES_FULL, true);
        reset_y(gpu)?;
        htod_words(gpu, &x, &clean.words[STREAMS])?;
        gemv_launch(gpu, down, &x, &ys[0], n, false)?;
        htod_words(gpu, &x, &poisoned.words[STREAMS])?;
        gemv_launch(gpu, down, &x, &ys[1], n, false)?;
        gemv_launch(gpu, down, &x, &ys[2], pad_end, true)?;
        let clean_y = fetch_y(gpu, &ys[0])?;
        let poison_y = fetch_y(gpu, &ys[1])?;
        let over_y = fetch_y(gpu, &ys[2])?;
        let live = n * RANK;
        // The real all-rows arm is unaffected: live rows equal the clean run, y pad rows
        // and everything past them still the sentinel.
        let mismatch_words = words_differ(&clean_y[..live], &poison_y[..live])
            + words_differ(&poison_y[live..], &sentinel[live..]);
        rep.cases.push(CaseRec {
            weights: set.label.clone(),
            seed: 101,
            n,
            partition: "pad-poison".into(),
            stage: "gemv down 320x10240".into(),
            comparison: "pad_poison",
            mismatch_words,
            first_mismatch: None,
            violations: Vec::new(),
            launches_ref: 1,
            launches_all: 1,
        });
        // The over-wide tiled launch reads the poisoned pad and writes y's pad rows.
        let detected = words_differ(&clean_y[..live], &over_y[..live]) != 0
            || words_differ(&over_y[live..], &sentinel[live..]) != 0;
        control(rep, format!("poisoned_pad_overwide_n{n}"), detected);
    }
    Ok(())
}

fn controls(env: &mut Env, sets: &[WeightSet], rep: &mut Report) -> R<()> {
    let set = &sets[0];
    let n = 24;
    let (lanes, seed) = (uniform_lanes(n, 4), 100);
    let shifted: Vec<(usize, usize)> = group_segments(&lanes)
        .iter()
        .map(|&(begin, rows)| (begin + 1, rows))
        .collect();
    let chain = Stage::Chain {
        first: 0,
        second: 1,
        tight: false,
    };
    let all_stages = [
        Stage::Read {
            layer: 0,
            mlp: true,
        },
        Stage::Write {
            layer: 0,
            mlp: false,
        },
        chain,
        Stage::Final,
    ];
    for stage in all_stages {
        let (kind, gates) = (stage.kind(), stage.gates_words());
        let init = make_init(n, seed, gates, false);
        let good = run_arm(env, set, stage, &Arm::AllRows(n), n, &init)?;
        // (i) the reference groups' offsets shifted by one row.
        let bad = run_arm(env, set, stage, &Arm::Groups(shifted.clone()), n, &init)?;
        control(
            rep,
            format!("wrong_group_offsets_{kind:?}").to_lowercase(),
            detects(kind, n, gates, &good, &bad, &init),
        );
        // (ii) the all-rows arm launched for n - 1 rows, judged against the n-row reference.
        let bad = run_arm(env, set, stage, &Arm::AllRows(n - 1), n, &init)?;
        control(
            rep,
            format!("off_by_one_rows_{kind:?}").to_lowercase(),
            detects(kind, n, gates, &good, &bad, &init),
        );
        // (iv) the comparator itself: a flipped bit in one live word, one tail word, and
        // (chain) one quarter word.
        let mut flipped = Outcome {
            snap: good.snap.clone(),
            quarters: good.quarters.clone(),
            calls: Vec::new(),
            launches: Vec::new(),
        };
        let live_buf = if kind == Kind::Write {
            STREAMS
        } else {
            NORMALIZED
        };
        flipped.snap[live_buf][5 * WIDTH[live_buf] + 7] ^= 1 << 13;
        control(
            rep,
            format!("bitflip_live_{kind:?}").to_lowercase(),
            compare(kind, n, &good, &flipped).0 != 0,
        );
        flipped.snap[live_buf][5 * WIDTH[live_buf] + 7] ^= 1 << 13;
        flipped.snap[live_buf][live_end(live_buf, n) + 3] ^= 1 << 5;
        control(
            rep,
            format!("bitflip_tail_{kind:?}").to_lowercase(),
            !integrity(kind, n, gates, &flipped, &init).is_empty(),
        );
        if stage.quarters() {
            flipped.snap[live_buf][live_end(live_buf, n) + 3] ^= 1 << 5;
            flipped.quarters[1][3 * 16 + 2] ^= 1 << 9;
            control(
                rep,
                "bitflip_quarter_chain".into(),
                compare(kind, n, &good, &flipped).0 != 0,
            );
        }
    }
    gemv_controls(env, set, rep)
}

// ---- launch census -----------------------------------------------------------------------

fn census(env: &mut Env, set: &WeightSet, rep: &mut Report) -> R<()> {
    let census_stages = [
        Stage::Read {
            layer: 0,
            mlp: true,
        },
        Stage::Write {
            layer: 0,
            mlp: false,
        },
        Stage::Chain {
            first: 0,
            second: 1 % set.layers.len(),
            tight: false,
        },
        Stage::Final,
    ];
    let mut baseline: BTreeMap<String, usize> = BTreeMap::new();
    for &n in &CENSUS_ROWS {
        for stage in census_stages {
            let label = stage.label(set);
            let init = make_init(n, 7, stage.gates_words(), false);
            let all = run_arm(env, set, stage, &Arm::AllRows(n), n, &init)?;
            // The all-rows arm does not grow with the row count: its launch count equals
            // the eight-row single-group count.
            if n == CENSUS_ROWS[0] {
                baseline.insert(label.clone(), all.launches.len());
            } else if baseline[&label] != all.launches.len() {
                rep.census_failures.push(format!(
                    "{label}: all-rows launches {} at n={n} != {} at n={}",
                    all.launches.len(),
                    baseline[&label],
                    CENSUS_ROWS[0]
                ));
            }
            for (name, lanes) in lane_partitions(n, 7) {
                let groups = group_segments(&lanes);
                let reference = run_arm(env, set, stage, &Arm::Groups(groups.clone()), n, &init)?;
                rep.census.push(CensusRow {
                    n,
                    partition: name,
                    groups: groups.len(),
                    stage: label.clone(),
                    ref_launches: reference.launches.len(),
                    all_launches: all.launches.len(),
                    ref_symbols: symbol_counts(&reference.launches),
                    all_symbols: symbol_counts(&all.launches),
                    ref_pairs: pair_launches(&reference),
                    all_pairs: pair_launches(&all),
                });
            }
        }
    }
    println!(
        "launch census (synthetic weights): launches per stage, group-major reference vs all-rows"
    );
    println!(
        "| rows | partition | groups | stage | launches ref | launches all | all-rows symbols |"
    );
    println!("|---:|---|---:|---|---:|---:|---|");
    for row in &rep.census {
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            row.n,
            row.partition,
            row.groups,
            row.stage,
            row.ref_launches,
            row.all_launches,
            fmt_symbols(&row.all_symbols)
        );
    }
    for row in rep
        .census
        .iter()
        .filter(|r| r.stage.starts_with("chain") && r.partition == "4row")
    {
        println!(
            "chain n={} pairs (launches per fused W+R, middle = steady state with quarters): ref {:?} all {:?}",
            row.n, row.ref_pairs, row.all_pairs
        );
    }
    for failure in &rep.census_failures {
        eprintln!("CENSUS FAIL {failure}");
    }
    Ok(())
}

// ---- bench -------------------------------------------------------------------------------

struct BenchRow {
    n: usize,
    op: String,
    groups: usize,
    launches_ref: usize,
    launches_all: usize,
    ref_us: f64,
    all_us: f64,
}

const WARMUPS: usize = 5;

/// Median wall time (us) of executing every segment's plan in order between two device
/// syncs; the scratch is reset outside the timed region.
fn time_arm(
    env: &mut Env,
    set: &WeightSet,
    stage: Stage,
    arm: &Arm,
    n: usize,
    init: &Init,
    iters: usize,
) -> R<f64> {
    let Env {
        gpu,
        ctx,
        full,
        tight,
        log,
    } = env;
    let bufs: &Bufs = if stage.tight() { tight } else { full };
    let (segs, scope) = segments(arm, n);
    let views: Vec<Views> = segs.iter().map(|&(begin, _)| bufs.views(begin)).collect();
    let plans: Vec<Plan<'_>> = segs
        .iter()
        .zip(&views)
        .map(|(&(_, rows), v)| build_plan(gpu, set, stage, v, &bufs.rotation, rows))
        .collect::<R<_>>()?;
    let mut samples = Vec::with_capacity(iters);
    for i in 0..WARMUPS + iters {
        bufs.reset(gpu, init)?;
        gpu.hip.device_synchronize().map_err(err("sync"))?;
        let t0 = Instant::now();
        for plan in &plans {
            exec(plan, gpu, ctx, scope, log)?;
        }
        gpu.hip.device_synchronize().map_err(err("sync"))?;
        if i >= WARMUPS {
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn bench(env: &mut Env, set: &WeightSet, iters: usize) -> R<Vec<BenchRow>> {
    let mut rows = Vec::new();
    println!("| rows | op | groups | launches ref | launches all | ref us | all us | speedup |");
    println!("|---:|---|---:|---:|---:|---:|---:|---:|");
    for (n, per_lane) in [(8usize, 1usize), (16, 2), (32, 4)] {
        let groups = group_segments(&vec![per_lane; LANE_ROWS]);
        let layers = set.layers.len();
        let ops = [
            (
                "read",
                Stage::Read {
                    layer: 0,
                    mlp: true,
                },
            ),
            (
                "chain",
                Stage::Chain {
                    first: 0,
                    second: 1 % layers,
                    tight: false,
                },
            ),
            ("final", Stage::Final),
        ];
        for (op, stage) in ops {
            let init = make_init(n, 11, stage.gates_words(), false);
            let all_arm = Arm::AllRows(n);
            let ref_arm = Arm::Groups(groups.clone());
            let launches_all = run_arm(env, set, stage, &all_arm, n, &init)?.launches.len();
            let launches_ref = run_arm(env, set, stage, &ref_arm, n, &init)?.launches.len();
            let ref_us = time_arm(env, set, stage, &ref_arm, n, &init, iters)?;
            let all_us = time_arm(env, set, stage, &all_arm, n, &init, iters)?;
            println!(
                "| {n} | {op} | {} | {launches_ref} | {launches_all} | {ref_us:.1} | {all_us:.1} | {:.3} |",
                groups.len(),
                ref_us / all_us
            );
            rows.push(BenchRow {
                n,
                op: op.into(),
                groups: groups.len(),
                launches_ref,
                launches_all,
                ref_us,
                all_us,
            });
        }
    }
    Ok(rows)
}

// ---- JSON receipts -----------------------------------------------------------------------

enum J {
    Bool(bool),
    Int(usize),
    Num(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(&'static str, J)>),
    /// An object with runtime keys (kernel symbols, violation names).
    Map(Vec<(String, J)>),
}

fn jstr(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

impl J {
    fn render(&self, out: &mut String) {
        match self {
            J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            J::Int(i) => {
                let _ = write!(out, "{i}");
            }
            J::Num(f) if f.is_finite() => {
                let _ = write!(out, "{f}");
            }
            J::Num(_) => out.push_str("null"),
            J::Str(s) => jstr(s, out),
            J::Arr(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    item.render(out);
                }
                out.push(']');
            }
            J::Obj(fields) => render_object(out, fields.iter().map(|(k, v)| (*k, v))),
            J::Map(fields) => render_object(out, fields.iter().map(|(k, v)| (k.as_str(), v))),
        }
    }
}

fn render_object<'a>(out: &mut String, fields: impl Iterator<Item = (&'a str, &'a J)>) {
    out.push('{');
    for (i, (key, value)) in fields.enumerate() {
        if i != 0 {
            out.push(',');
        }
        jstr(key, out);
        out.push(':');
        value.render(out);
    }
    out.push('}');
}

fn j_symbols(counts: &BTreeMap<String, usize>) -> J {
    J::Map(
        counts
            .iter()
            .map(|(s, c)| (s.clone(), J::Int(*c)))
            .collect(),
    )
}

fn j_ints(v: &[usize]) -> J {
    J::Arr(v.iter().map(|&i| J::Int(i)).collect())
}

fn oracle_json(gpu: &Gpu, cli: &Cli, rows: &[usize], rep: &Report) -> J {
    let cases = rep
        .cases
        .iter()
        .map(|c| {
            J::Obj(vec![
                ("weights", J::Str(c.weights.clone())),
                ("seed", J::Int(c.seed as usize)),
                ("n", J::Int(c.n)),
                ("partition", J::Str(c.partition.clone())),
                ("stage", J::Str(c.stage.clone())),
                ("comparison", J::Str(c.comparison.into())),
                (
                    "verdict",
                    J::Str(
                        if c.mismatch_words == 0 && c.violations.is_empty() {
                            "PASS"
                        } else {
                            "FAIL"
                        }
                        .into(),
                    ),
                ),
                ("mismatch_words", J::Int(c.mismatch_words)),
                (
                    "first_mismatch",
                    c.first_mismatch
                        .clone()
                        .map_or(J::Str(String::new()), J::Str),
                ),
                (
                    "violations",
                    J::Map(
                        c.violations
                            .iter()
                            .map(|(k, v)| (k.clone(), J::Int(*v)))
                            .collect(),
                    ),
                ),
                ("launches_ref", J::Int(c.launches_ref)),
                ("launches_all", J::Int(c.launches_all)),
            ])
        })
        .collect();
    let census = rep
        .census
        .iter()
        .map(|r| {
            J::Obj(vec![
                ("n", J::Int(r.n)),
                ("partition", J::Str(r.partition.clone())),
                ("groups", J::Int(r.groups)),
                ("stage", J::Str(r.stage.clone())),
                ("launches_ref", J::Int(r.ref_launches)),
                ("launches_all", J::Int(r.all_launches)),
                ("symbols_ref", j_symbols(&r.ref_symbols)),
                ("symbols_all", j_symbols(&r.all_symbols)),
                ("pairs_ref", j_ints(&r.ref_pairs)),
                ("pairs_all", j_ints(&r.all_pairs)),
            ])
        })
        .collect();
    J::Obj(vec![
        ("tool", J::Str("hc_allrows_oracle".into())),
        ("arch", J::Str(gpu.arch.clone())),
        (
            "lane_hc_allrows",
            J::Bool(gpu.flags.qwen4_lane_hc_allrows_enabled()),
        ),
        (
            "hc_fuse_level",
            J::Int(usize::from(gpu.flags.qwen4_hc_fuse_level())),
        ),
        ("model", J::Str(cli.model.clone().unwrap_or_default())),
        ("seeds", J::Int(cli.seeds as usize)),
        ("rows", j_ints(rows)),
        ("cases", J::Arr(cases)),
        (
            "controls",
            J::Arr(
                rep.controls
                    .iter()
                    .map(|(name, detected)| {
                        J::Obj(vec![
                            ("name", J::Str(name.clone())),
                            ("detected", J::Bool(*detected)),
                        ])
                    })
                    .collect(),
            ),
        ),
        ("census", J::Arr(census)),
        (
            "census_failures",
            J::Arr(
                rep.census_failures
                    .iter()
                    .map(|s| J::Str(s.clone()))
                    .collect(),
            ),
        ),
        (
            "summary",
            J::Obj(vec![
                ("cases", J::Int(rep.cases.len())),
                ("mismatches", J::Int(rep.mismatches())),
                ("controls_detected", J::Int(rep.detected())),
                ("controls_total", J::Int(rep.controls.len())),
                ("pass", J::Bool(rep.passed())),
            ]),
        ),
    ])
}

fn bench_json(gpu: &Gpu, iters: usize, rows: &[BenchRow]) -> J {
    J::Obj(vec![
        ("tool", J::Str("hc_allrows_oracle".into())),
        ("arch", J::Str(gpu.arch.clone())),
        ("iters", J::Int(iters)),
        ("warmups", J::Int(WARMUPS)),
        (
            "rows",
            J::Arr(
                rows.iter()
                    .map(|r| {
                        J::Obj(vec![
                            ("rows", J::Int(r.n)),
                            ("op", J::Str(r.op.clone())),
                            ("groups", J::Int(r.groups)),
                            ("launches_ref", J::Int(r.launches_ref)),
                            ("launches_all", J::Int(r.launches_all)),
                            ("ref_us", J::Num(r.ref_us)),
                            ("all_us", J::Num(r.all_us)),
                            ("speedup", J::Num(r.ref_us / r.all_us)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])
}

fn write_json(dir: &Option<PathBuf>, name: &str, json: &J) -> R<()> {
    let Some(dir) = dir else { return Ok(()) };
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let mut text = String::new();
    json.render(&mut text);
    text.push('\n');
    let path = dir.join(name);
    std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

// ---- driver ------------------------------------------------------------------------------

struct Cli {
    out: Option<PathBuf>,
    model: Option<String>,
    layers: Vec<usize>,
    seeds: u64,
    rows_max: usize,
    bench: bool,
    bench_only: bool,
    iters: usize,
}

fn parse_cli() -> R<Cli> {
    let mut cli = Cli {
        out: None,
        model: None,
        layers: DEFAULT_LAYERS.to_vec(),
        seeds: 1,
        rows_max: 64,
        bench: false,
        bench_only: false,
        iters: 21,
    };
    let mut args = std::env::args().skip(1);
    let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    };
    let number = |text: String, flag: &str| {
        text.parse::<usize>()
            .map_err(|_| format!("{flag}: not a number: {text}"))
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => cli.out = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--model" => cli.model = Some(value(&mut args, "--model")?),
            "--layers" => {
                cli.layers = value(&mut args, "--layers")?
                    .split(',')
                    .map(|s| number(s.to_string(), "--layers"))
                    .collect::<R<_>>()?
            }
            "--seeds" => cli.seeds = number(value(&mut args, "--seeds")?, "--seeds")? as u64,
            "--rows-max" => {
                cli.rows_max = number(value(&mut args, "--rows-max")?, "--rows-max")?;
                if !(1..=64).contains(&cli.rows_max) {
                    return Err("--rows-max must be 1..=64".into());
                }
            }
            "--bench" => cli.bench = true,
            "--bench-only" => cli.bench_only = true,
            "--iters" => {
                cli.iters = number(value(&mut args, "--iters")?, "--iters")?;
                if cli.iters < 5 {
                    return Err("--iters must be at least 5".into());
                }
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(cli)
}

fn run() -> R<bool> {
    let cli = parse_cli()?;
    let mut gpu = Gpu::init().map_err(err("Gpu::init"))?;
    println!("hc_allrows_oracle: arch={} model={:?}", gpu.arch, cli.model);
    if gpu.arch != "gfx1201" {
        return Err(format!("needs exact gfx1201, got {}", gpu.arch));
    }
    if !gpu.flags.qwen4_lane_hc_allrows_enabled() {
        return Err(
            "HIPFIRE_QWEN4_LANE_HC_ALLROWS=1 is required: the all-rows arm is off, the oracle \
             would compare the incumbent with itself"
                .into(),
        );
    }
    let log = Arc::new(Log::default());
    let sink = log.clone();
    let observer = Arc::new(
        move |_gpu: &Gpu, w: &LabLaunchWitness<'_>| -> HipResult<()> {
            if sink.on.load(Ordering::Relaxed) {
                sink.launches
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push(Launch {
                        symbol: w.symbol.to_string(),
                        grid: w.grid,
                        block: w.block,
                    });
            }
            Ok(())
        },
    );
    gpu.set_lab_launch_observer(Some(observer));

    let mut sets = vec![synthetic_weights(&mut gpu)?];
    if let Some(path) = &cli.model {
        sets.push(real_weights(&mut gpu, path, &cli.layers)?);
    }
    let full = Bufs::new(&mut gpu, GATES_FULL)?;
    let tight = Bufs::new(&mut gpu, GATES_TIGHT)?;
    let ctx = DispatchCtx::new(&gpu);
    let mut env = Env {
        gpu,
        ctx,
        full,
        tight,
        log,
    };

    let mut rows: Vec<usize> = (1..=cli.rows_max).collect();
    rows.extend(
        EXPLICIT_ROWS
            .iter()
            .filter(|n| !rows.contains(n))
            .copied()
            .collect::<Vec<_>>(),
    );
    let mut rep = Report::default();
    if !cli.bench_only {
        oracle(&mut env, &sets, &rows, cli.seeds, &mut rep)?;
        controls(&mut env, &sets, &mut rep)?;
        census(&mut env, &sets[0], &mut rep)?;
        write_json(
            &cli.out,
            "oracle.json",
            &oracle_json(&env.gpu, &cli, &rows, &rep),
        )?;
        let line = format!(
            "cases={} mismatches={} controls={}/{}",
            rep.cases.len(),
            rep.mismatches(),
            rep.detected(),
            rep.controls.len()
        );
        if rep.passed() {
            println!("ORACLE PASS {line}");
        } else {
            println!(
                "ORACLE FAIL {line} census_failures={}",
                rep.census_failures.len()
            );
        }
    }
    if (cli.bench || cli.bench_only) && (cli.bench_only || rep.passed()) {
        let bench_rows = bench(&mut env, sets.last().expect("a weight set"), cli.iters)?;
        write_json(
            &cli.out,
            "bench.json",
            &bench_json(&env.gpu, cli.iters, &bench_rows),
        )?;
    }
    Ok(cli.bench_only || rep.passed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("hc_allrows_oracle: {e}");
            std::process::exit(2);
        }
    }
}
