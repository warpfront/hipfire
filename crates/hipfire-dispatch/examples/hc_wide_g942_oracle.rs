// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle, route census and bench of the exact-gfx942 lane step's wide HC stages
//! (`hc_allrows_oracle`'s counterpart: that oracle is exact gfx1201 only).
//!
//! A gfx942 lane step of more than eight rows runs its HC reads / writes once over every row
//! (the all-rows segment, `LaneAllRowsRule`). Every row of that segment must be bitwise the row
//! the one-row AR singleton computes: the write's gates take the four-quarter `(q0 + q1) +
//! (q2 + q3)` arithmetic (not the ordinary chunk's BF16 MFMA GEMV) and the read's down
//! projection + activation is the fused rows16 entry in 16-row slices. This example drives the real
//! dispatch entry points (`execute_hyper_read`, `execute_hyper_write`, `execute_final_hyper`, and
//! the `execute_steps` write -> read chain) on three arms and compares every live buffer word for
//! word:
//!
//! * `single`  one `rows == 1` call per row on row-offset views, scope off: the pinned singleton
//!             route (fused quarters chain through `execute_hyper_write_then_read`);
//! * `lane`    one call over all `n` rows at row 0 with `Gpu::lane_rows_scope` and the wide
//!             verify scope open (what `forward_lanes` sets): the lane-wide route under test;
//! * `chunk`   the same call with both scopes closed: the ordinary prefill chunk's route, whose
//!             bytes and launch census must not change (reported as a hash and a symbol list so
//!             a before / after build can be diffed; the oracle itself requires only that its
//!             gates still differ from the singleton's, i.e. that the comparison has power).
//!
//! Stages: read, write (gates + norm + streams), chain `W(A) R(A) W(B) R(B)` through
//! `execute_steps` and the final read. Rows past `n` must stay untouched in every arm.
//!
//! ```text
//! cargo run --release -p hipfire-dispatch --features test-utils --example hc_wide_g942_oracle -- \
//!   [--out DIR] [--model PATH] [--layers 0,1,..] [--rows 9,16,17,32,33,64 | --sweep] \
//!   [--seeds N] [--bench] [--bench-only] [--iters N] [--bench-rows N]
//! ```
//!
//! Exit nonzero on a non-gfx942 device, a launch error, any mismatch, census failure or an
//! undetected control.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{
    execute_final_hyper, execute_hyper_read, execute_hyper_write, execute_steps, project_weights,
    validate_steps, HyperReadOp, HyperWriteOp, Step,
};
use rdna_compute::{DType, Gpu, GpuTensor, HipResult, LabLaunchWitness};

type R<T> = Result<T, String>;

const HIDDEN: usize = 2560;
const BRANCHES: usize = 4;
const WIDE: usize = BRANCHES * HIDDEN;
const RANK: usize = 320;
/// Rows every scratch buffer holds: the widest step (64) plus canary rows.
const CAP: usize = 72;
const SENTINEL: u32 = 0x7fc0_beef;
const DEFAULT_LAYERS: [usize; 4] = [0, 1, 23, 47];
const DEFAULT_ROWS: [usize; 6] = [9, 16, 17, 32, 33, 64];

// ---- buffer table ------------------------------------------------------------------------

const NBUF: usize = 10;
const STREAMS: usize = 0;
const NORMALIZED: usize = 1;
const LOW: usize = 2;
const UP: usize = 3;
const MIXED: usize = 4;
const MIXED2: usize = 5;
const GATES: usize = 6;
const GATES2: usize = 7;
const WMIX: usize = 8;
const WMIX2: usize = 9;
const NAMES: [&str; NBUF] = [
    "streams",
    "normalized",
    "low",
    "up",
    "mixed",
    "mixed2",
    "gates",
    "gates2",
    "wmix",
    "wmix2",
];
/// Words per row of each buffer; the gate buffers hold the chain's two quarter slots per row.
const GATES_PER_ROW: usize = BRANCHES + 32;
const WIDTH: [usize; NBUF] = [
    WIDE,
    WIDE,
    RANK,
    WIDE,
    HIDDEN,
    HIDDEN,
    GATES_PER_ROW,
    GATES_PER_ROW,
    HIDDEN,
    HIDDEN,
];

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

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn synth_bf16(rng: &mut Rng, count: usize, mut value: impl FnMut(&mut Rng) -> f32) -> Vec<u8> {
    (0..count)
        .flat_map(|_| bf16_bits(value(rng)).to_le_bytes())
        .collect()
}

/// One F32 stream / mixed row: a pattern family at a scale in `0.25..=4`.
fn gen_row(rng: &mut Rng, width: usize) -> Vec<f32> {
    let scale = [0.25f32, 0.5, 1.0, 2.0, 4.0][rng.below(5) as usize];
    let pattern = rng.below(5);
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
                2 => rng.normal(),
                3 => match rng.below(24) {
                    0 => 0.0,
                    1 => -0.0,
                    _ => u,
                },
                _ => {
                    if rng.below(64) == 0 {
                        u * 32.0
                    } else {
                        u * 0.05
                    }
                }
            };
            v * scale
        })
        .collect()
}

/// Host images of every buffer an arm starts from: streams and write inputs of rows `0..n`,
/// everything else the NaN sentinel.
struct Init {
    words: Vec<Vec<u32>>,
}

fn make_init(n: usize, seed: u64) -> Init {
    let mut rng = Rng::new(0xC0FF_EE42 ^ (seed << 20) ^ n as u64);
    let mut words: Vec<Vec<u32>> = (0..NBUF)
        .map(|b| vec![SENTINEL; CAP * WIDTH[b]])
        .collect();
    for r in 0..n {
        for (b, width) in [(STREAMS, WIDE), (WMIX, HIDDEN), (WMIX2, HIDDEN)] {
            let row = gen_row(&mut rng, width);
            for (i, v) in row.iter().enumerate() {
                words[b][r * width + i] = v.to_bits();
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
        rotation: None,
        awq_scale: None,
        lloyd_lut_e4m3: None,
        lloyd_lut_f16: None,
        lloyd_lut_c16: None,
    }
}

fn fnv64(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

struct Bufs {
    t: Vec<GpuTensor>,
    rotation: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu) -> R<Self> {
        let mut t = Vec::with_capacity(NBUF);
        for b in 0..NBUF {
            t.push(
                gpu.alloc_tensor(&[CAP * WIDTH[b]], DType::F32)
                    .map_err(err("alloc scratch"))?,
            );
        }
        let rotation = gpu
            .alloc_tensor(&[CAP * WIDE], DType::F32)
            .map_err(err("alloc rotation"))?;
        Ok(Bufs { t, rotation })
    }

    /// Views of every buffer from row `begin` to the end of its allocation. The gate buffers
    /// are packed at four words per row for single-write stages and at `GATES_PER_ROW` for
    /// the chain (its fused writes leave quarter slots after the gates).
    fn views(&self, begin: usize, gates_stride: usize) -> Vec<GpuTensor> {
        (0..NBUF)
            .map(|b| {
                let stride = if b == GATES || b == GATES2 {
                    gates_stride
                } else {
                    WIDTH[b]
                };
                let skip = begin * stride;
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
    let mut rng = Rng::new(0x5EED_0942);
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

/// Minimal HFQM index reader (tensor name -> byte range), as `hc_allrows_oracle`.
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
            pos += 4;
            let size = u64::from_le_bytes(meta[pos..pos + 8].try_into().unwrap());
            pos += 8;
            tensors.insert(name, (off, size, shape, qt));
            off += size;
        }
        Ok(Hfq { file, tensors })
    }

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
    Read { layer: usize, mlp: bool },
    Write { layer: usize, mlp: bool },
    /// `W(attn first) R(attn first) W(mlp first) R(mlp first)` over layer `first`.
    Chain { first: usize },
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
    /// Words of the gate buffers per row in the views this stage binds.
    fn gates_stride(self) -> usize {
        match self.kind() {
            Kind::Chain => GATES_PER_ROW,
            _ => BRANCHES,
        }
    }
    /// The buffers whose live rows are compared, and the words per live row of each.
    fn compared(self) -> Vec<(usize, usize)> {
        match self.kind() {
            Kind::Read | Kind::Final => vec![
                (NORMALIZED, WIDE),
                (LOW, RANK),
                (UP, WIDE),
                (MIXED, HIDDEN),
            ],
            Kind::Write => vec![(NORMALIZED, WIDE), (GATES, BRANCHES), (STREAMS, WIDE)],
            Kind::Chain => vec![(STREAMS, WIDE), (MIXED, HIDDEN), (MIXED2, HIDDEN)],
        }
    }
    fn label(self, set: &WeightSet) -> String {
        let conn = |mlp: bool| if mlp { "mlp" } else { "attn" };
        match self {
            Stage::Read { layer, mlp } => format!("read L{} {}", set.layers[layer].id, conn(mlp)),
            Stage::Write { layer, mlp } => {
                format!("write L{} {}", set.layers[layer].id, conn(mlp))
            }
            Stage::Chain { first } => format!("chain L{}", set.layers[first].id),
            Stage::Final => "final".into(),
        }
    }
}

fn stages(set: &WeightSet) -> Vec<Stage> {
    let mut out = Vec::new();
    for layer in 0..set.layers.len() {
        for mlp in [false, true] {
            out.push(Stage::Read { layer, mlp });
            out.push(Stage::Write { layer, mlp });
        }
        out.push(Stage::Chain { first: layer });
    }
    out.push(Stage::Final);
    out
}

fn read_op<'a>(r: &'a ReadW, v: &'a [GpuTensor], rotation: &'a GpuTensor, rows: usize, mixed: usize) -> HyperReadOp<'a> {
    HyperReadOp {
        state_bf16: false,
        input: &v[STREAMS],
        norm_weight: &r.norm,
        input_mix_down: wref(&r.down, RANK, WIDE),
        input_mix_up: wref(&r.up, WIDE, RANK),
        normalized: &v[NORMALIZED],
        low: &v[LOW],
        up: &v[UP],
        mixed: &v[mixed],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        low_rank: RANK,
        rotation,
    }
}

#[allow(clippy::too_many_arguments)]
fn write_op<'a>(
    conn: &'a Conn,
    v: &'a [GpuTensor],
    rotation: &'a GpuTensor,
    rows: usize,
    gates: usize,
    wmix: usize,
) -> HyperWriteOp<'a> {
    HyperWriteOp {
        state_bf16: false,
        input: &v[STREAMS],
        norm_weight: &conn.read.norm,
        block_inject: wref(&conn.inject, BRANCHES, WIDE),
        normalized: &v[NORMALIZED],
        mixed: &v[wmix],
        gates: &v[gates],
        output: &v[STREAMS],
        rows,
        branches: BRANCHES,
        hidden: HIDDEN,
        rotation,
        ar: None,
    }
}

// ---- launch log and execution ------------------------------------------------------------

#[derive(Default)]
struct Log {
    on: AtomicBool,
    symbols: Mutex<Vec<String>>,
}

impl Log {
    fn start(&self) {
        self.symbols.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.on.store(true, Ordering::Relaxed);
    }
    fn stop(&self) -> Vec<String> {
        self.on.store(false, Ordering::Relaxed);
        std::mem::take(&mut *self.symbols.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

/// Scope an arm runs under: what `forward_lanes_with` (lane scope + the HC all-rows wide scope)
/// sets, or nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scope {
    Off,
    Lane(usize),
}

fn exec_stage(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    set: &WeightSet,
    stage: Stage,
    v: &[GpuTensor],
    rotation: &GpuTensor,
    rows: usize,
    scope: Scope,
) -> R<()> {
    let prior_lane = gpu.lane_rows_scope;
    let prior_wide = gpu.wide_verify_rows;
    if let Scope::Lane(n) = scope {
        gpu.lane_rows_scope = true;
        gpu.set_wide_verify_rows(n);
    }
    let conn = |layer: usize, mlp: bool| {
        let l = &set.layers[layer];
        if mlp {
            &l.mlp
        } else {
            &l.attn
        }
    };
    let result = (|| -> R<()> {
        match stage {
            Stage::Read { layer, mlp } => {
                let op = read_op(&conn(layer, mlp).read, v, rotation, rows, MIXED);
                op.validate_for_gpu(gpu).map_err(dbg_err)?;
                execute_hyper_read(gpu, &op).map_err(dbg_err)
            }
            Stage::Write { layer, mlp } => {
                let op = write_op(conn(layer, mlp), v, rotation, rows, GATES, WMIX);
                op.validate_for_gpu(gpu).map_err(dbg_err)?;
                execute_hyper_write(gpu, &op).map_err(dbg_err)
            }
            Stage::Final => {
                let op = read_op(&set.fin, v, rotation, rows, MIXED);
                op.validate_for_gpu(gpu).map_err(dbg_err)?;
                execute_final_hyper(gpu, &op).map_err(dbg_err)
            }
            Stage::Chain { first } => {
                let steps = vec![
                    Step::HyperWrite(write_op(conn(first, false), v, rotation, rows, GATES, WMIX)),
                    Step::HyperRead(read_op(&conn(first, false).read, v, rotation, rows, MIXED)),
                    Step::HyperWrite(write_op(conn(first, true), v, rotation, rows, GATES2, WMIX2)),
                    Step::HyperRead(read_op(&conn(first, true).read, v, rotation, rows, MIXED2)),
                ];
                validate_steps(gpu, &steps).map_err(dbg_err)?;
                execute_steps(gpu, ctx, &steps).map_err(dbg_err)
            }
        }
    })();
    gpu.lane_rows_scope = prior_lane;
    gpu.set_wide_verify_rows(prior_wide);
    result
}

// ---- arms --------------------------------------------------------------------------------

struct Env {
    gpu: Gpu,
    ctx: DispatchCtx,
    bufs: Bufs,
    log: Arc<Log>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Single,
    Lane,
    Chunk,
}

struct Outcome {
    snap: Vec<Vec<u32>>,
    symbols: Vec<String>,
}

fn run_arm(env: &mut Env, set: &WeightSet, stage: Stage, arm: Arm, n: usize, init: &Init) -> R<Outcome> {
    let Env { gpu, ctx, bufs, log } = env;
    bufs.reset(gpu, init)?;
    log.start();
    let result = (|| -> R<()> {
        match arm {
            Arm::Single => {
                for r in 0..n {
                    let v = bufs.views(r, stage.gates_stride());
                    exec_stage(gpu, ctx, set, stage, &v, &bufs.rotation, 1, Scope::Off)?;
                }
                Ok(())
            }
            Arm::Lane | Arm::Chunk => {
                let v = bufs.views(0, stage.gates_stride());
                let scope = if arm == Arm::Lane {
                    Scope::Lane(n)
                } else {
                    Scope::Off
                };
                exec_stage(gpu, ctx, set, stage, &v, &bufs.rotation, n, scope)
            }
        }
    })();
    let symbols = log.stop();
    result?;
    Ok(Outcome {
        snap: bufs.snapshot(gpu)?,
        symbols,
    })
}

/// Words that differ between two arms over the live rows of every compared buffer; the first
/// differing `(buffer, row, word)` as a message.
fn compare(stage: Stage, n: usize, a: &Outcome, b: &Outcome) -> (usize, Option<String>) {
    let mut diff = 0usize;
    let mut first = None;
    for (buf, width) in stage.compared() {
        let live = n * width;
        for i in 0..live {
            let (x, y) = (a.snap[buf][i], b.snap[buf][i]);
            if x != y {
                diff += 1;
                first.get_or_insert_with(|| {
                    format!(
                        "{} row {} word {}: {x:#010x} vs {y:#010x}",
                        NAMES[buf],
                        i / width,
                        i % width
                    )
                });
            }
        }
    }
    (diff, first)
}

/// Rows past `n` of every compared buffer keep the init's bytes.
fn tail_clean(stage: Stage, n: usize, out: &Outcome, init: &Init) -> Option<String> {
    for (buf, width) in stage.compared() {
        // The gate buffer's tail in the packed stages starts at the live end.
        let from = n * width;
        let to = CAP * if buf == GATES || buf == GATES2 {
            WIDTH[buf]
        } else {
            width
        };
        for i in from..to.min(out.snap[buf].len()) {
            if out.snap[buf][i] != init.words[buf][i] {
                return Some(format!("{} word {i} past row {n} was written", NAMES[buf]));
            }
        }
    }
    None
}

fn count(symbols: &[String], needle: &str) -> usize {
    symbols.iter().filter(|s| s.contains(needle)).count()
}

fn census_line(symbols: &[String]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for s in symbols {
        *counts.entry(s.as_str()).or_default() += 1;
    }
    counts
        .iter()
        .map(|(s, c)| format!("{s}x{c}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---- oracle ------------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    cases: usize,
    mismatches: Vec<String>,
    census_failures: Vec<String>,
    controls: Vec<(String, bool)>,
    /// `set stage rows seed -> fnv64 of the chunk arm's compared words` (before / after diff).
    chunk_hashes: Vec<String>,
}

impl Report {
    fn passed(&self) -> bool {
        self.mismatches.is_empty()
            && self.census_failures.is_empty()
            && self.controls.iter().all(|(_, d)| *d)
    }
}

fn oracle(env: &mut Env, sets: &[WeightSet], rows: &[usize], seeds: u64, rep: &mut Report) -> R<()> {
    for set in sets {
        for stage in stages(set) {
            for &n in rows {
                for seed in 0..seeds {
                    let init = make_init(n, seed);
                    let single = run_arm(env, set, stage, Arm::Single, n, &init)?;
                    let lane = run_arm(env, set, stage, Arm::Lane, n, &init)?;
                    let chunk = run_arm(env, set, stage, Arm::Chunk, n, &init)?;
                    let label = format!("{} {} n={n} seed={seed}", set.label, stage.label(set));
                    rep.cases += 1;

                    // The lane arm is bitwise the singleton row by row.
                    let (diff, first) = compare(stage, n, &single, &lane);
                    if diff != 0 {
                        rep.mismatches.push(format!(
                            "{label}: lane vs single {diff} words differ, first {}",
                            first.unwrap_or_default()
                        ));
                    }
                    if let Some(why) = tail_clean(stage, n, &lane, &init) {
                        rep.mismatches.push(format!("{label}: lane {why}"));
                    }
                    if let Some(why) = tail_clean(stage, n, &chunk, &init) {
                        rep.mismatches.push(format!("{label}: chunk {why}"));
                    }

                    // The chunk route's bytes: a hash to diff across builds.
                    let mut words = Vec::new();
                    for (buf, width) in stage.compared() {
                        words.extend_from_slice(&chunk.snap[buf][..n * width]);
                    }
                    rep.chunk_hashes
                        .push(format!("{label}: chunk fnv64={:016x}", fnv64(&words)));

                    // Census: which arithmetic ran where.
                    match stage.kind() {
                        Kind::Write => {
                            let l = &lane.symbols;
                            if count(l, "k4_rows_tiled") == 0 || count(l, "mfma") != 0 {
                                rep.census_failures.push(format!(
                                    "{label}: lane write must run k4_rows_tiled and no MFMA: {}",
                                    census_line(l)
                                ));
                            }
                            let c = &chunk.symbols;
                            if count(c, "mfma") == 0 || count(c, "k4_rows_tiled") != 0 {
                                rep.census_failures.push(format!(
                                    "{label}: chunk write must keep the MFMA gates: {}",
                                    census_line(c)
                                ));
                            }
                        }
                        Kind::Read | Kind::Final => {
                            let l = &lane.symbols;
                            let slices = n.div_ceil(16);
                            if count(l, "hc_act") != slices || count(l, "roundtrip") != 0 {
                                rep.census_failures.push(format!(
                                    "{label}: lane read wants {slices} fused hc_act slices and no \
                                     unfused activation chain: {}",
                                    census_line(l)
                                ));
                            }
                            let c = &chunk.symbols;
                            let want_fold = n <= 16;
                            if (count(c, "hc_act") != 0) == !want_fold
                                || (!want_fold && count(c, "roundtrip") == 0)
                            {
                                rep.census_failures.push(format!(
                                    "{label}: chunk read route changed: {}",
                                    census_line(c)
                                ));
                            }
                        }
                        Kind::Chain => {}
                    }

                    // Control: the chunk's gates differ from the singleton's, so the
                    // comparison has the power to see the arithmetic difference.
                    if stage.kind() == Kind::Write && seed == 0 {
                        let (d, _) = compare(stage, n, &single, &chunk);
                        rep.controls.push((
                            format!("{label}: chunk MFMA gates differ from the singleton"),
                            d != 0,
                        ));
                    }
                    // Control: a flipped live word is detected.
                    if seed == 0 {
                        let mut bad = Outcome {
                            snap: lane.snap.clone(),
                            symbols: Vec::new(),
                        };
                        let (buf, _) = stage.compared()[0];
                        bad.snap[buf][n / 2] ^= 1;
                        let (d, _) = compare(stage, n, &single, &bad);
                        rep.controls
                            .push((format!("{label}: one flipped word detected"), d == 1));
                    }
                }
            }
        }
    }
    Ok(())
}

// ---- bench -------------------------------------------------------------------------------

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn sync(gpu: &Gpu) -> R<()> {
    gpu.hip.device_synchronize().map_err(err("sync"))
}

/// Median wall (ms, device-synchronised) of `reps` calls of `f`.
fn time_ms(env: &mut Env, iters: usize, mut f: impl FnMut(&mut Env) -> R<()>) -> R<f64> {
    f(env)?;
    sync(&env.gpu)?;
    let mut ms = Vec::new();
    for _ in 0..iters {
        sync(&env.gpu)?;
        let t0 = Instant::now();
        f(env)?;
        sync(&env.gpu)?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    Ok(median(ms))
}

/// HC wall of one forward's worth of reads + writes at `n` rows: 48 layers of the
/// `W(attn) R(attn) W(mlp) R(mlp)` chain, then the final read.
fn bench_hc(env: &mut Env, set: &WeightSet, n: usize, scope: Scope, iters: usize) -> R<f64> {
    let init = make_init(n, 0);
    env.bufs.reset(&env.gpu, &init)?;
    time_ms(env, iters, |env| {
        let Env { gpu, ctx, bufs, .. } = env;
        let v = bufs.views(0, GATES_PER_ROW);
        for layer in 0..48 {
            let first = layer % set.layers.len();
            exec_stage(gpu, ctx, set, Stage::Chain { first }, &v, &bufs.rotation, n, scope)?;
        }
        exec_stage(gpu, ctx, set, Stage::Final, &v, &bufs.rotation, n, scope)
    })
}

/// Wall of 96 calls of one stage kind (a forward's reads, or its writes) at `n` rows.
fn bench_stage(env: &mut Env, set: &WeightSet, n: usize, scope: Scope, iters: usize, write: bool) -> R<f64> {
    let init = make_init(n, 0);
    env.bufs.reset(&env.gpu, &init)?;
    time_ms(env, iters, |env| {
        let Env { gpu, ctx, bufs, .. } = env;
        let v = bufs.views(0, BRANCHES);
        for i in 0..96 {
            let layer = (i / 2) % set.layers.len();
            let mlp = i % 2 == 1;
            let stage = if write {
                Stage::Write { layer, mlp }
            } else {
                Stage::Read { layer, mlp }
            };
            exec_stage(gpu, ctx, set, stage, &v, &bufs.rotation, n, scope)?;
        }
        Ok(())
    })
}

/// The trunk's dense projection census (shapes and per-forward counts of
/// `mi300x/beat/P/tilechain.cpp`, grouped as the dispatch groups them) at `n` rows through the
/// real `project_weights` route, each shape cycling enough weight instances to exceed the 256 MiB
/// infinity cache.
fn bench_dense(env: &mut Env, n: usize, iters: usize) -> R<f64> {
    struct Shape {
        group: &'static str,
        m: usize,
        k: usize,
        count: usize,
    }
    // (group, m, k, calls per forward); members of a group read one input, so one call.
    let shapes = [
        Shape { group: "gdn", m: 10240, k: 2560, count: 36 },
        Shape { group: "gdn", m: 6144, k: 2560, count: 36 },
        Shape { group: "gdn", m: 48, k: 2560, count: 36 },
        Shape { group: "gdn", m: 48, k: 2560, count: 36 },
        Shape { group: "out", m: 2560, k: 6144, count: 48 },
        Shape { group: "qsa", m: 12288, k: 2560, count: 12 },
        Shape { group: "qsa", m: 512, k: 2560, count: 12 },
        Shape { group: "qsa", m: 512, k: 2560, count: 12 },
        Shape { group: "qsa", m: 640, k: 2560, count: 12 },
        Shape { group: "sh", m: 640, k: 2560, count: 48 },
        Shape { group: "sh", m: 640, k: 2560, count: 48 },
        Shape { group: "shd", m: 2560, k: 640, count: 48 },
        Shape { group: "router", m: 512, k: 2560, count: 48 },
        Shape { group: "egate", m: 1, k: 2560, count: 48 },
        Shape { group: "ple", m: 10240, k: 2560, count: 1 },
        Shape { group: "ple2", m: 2560, k: 2560, count: 1 },
    ];
    let mut rng = Rng::new(0xD3_5E);
    // Instances per shape.
    const CACHE_BYTES: usize = 320 << 20;
    let mut weights: Vec<Vec<GpuTensor>> = Vec::new();
    for s in &shapes {
        let bytes = s.m * s.k * 2;
        let inst = CACHE_BYTES.div_ceil(bytes).clamp(1, 48);
        let host = synth_bf16(&mut rng, s.m * s.k, |r| r.normal() * 0.02);
        let mut v = Vec::new();
        for _ in 0..inst {
            v.push(upload_bf16(&mut env.gpu, &host, s.m * s.k)?);
        }
        weights.push(v);
    }
    let max_k = 6144usize;
    let x_host: Vec<u32> = {
        let mut r = Rng::new(7);
        (0..n * max_k).map(|_| (r.normal()).to_bits()).collect()
    };
    let x = env
        .gpu
        .alloc_tensor(&[n * max_k], DType::F32)
        .map_err(err("alloc x"))?;
    htod_words(&env.gpu, &x, &x_host)?;
    let rotation = env
        .gpu
        .alloc_tensor(&[n * 12288], DType::F32)
        .map_err(err("alloc rotation"))?;
    let mut outs: Vec<GpuTensor> = Vec::new();
    for s in &shapes {
        outs.push(
            env.gpu
                .alloc_tensor(&[n * s.m], DType::F32)
                .map_err(err("alloc out"))?,
        );
    }
    // Group index runs.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < shapes.len() {
        let mut j = i + 1;
        while j < shapes.len() && shapes[j].group == shapes[i].group && shapes[i].group != "shd" {
            j += 1;
        }
        groups.push((i, j));
        i = j;
    }
    let mut call = 0usize;
    time_ms(env, iters, |env| {
        for &(a, b) in &groups {
            let reps = shapes[a].count;
            let k = shapes[a].k;
            for rep in 0..reps {
                call += 1;
                let refs: Vec<WeightRef<'_>> = (a..b)
                    .map(|s| {
                        let w = &weights[s][(call + rep) % weights[s].len()];
                        wref(w, shapes[s].m, shapes[s].k)
                    })
                    .collect();
                let projections: Vec<(&WeightRef<'_>, &GpuTensor)> =
                    refs.iter().zip(a..b).map(|(r, s)| (r, &outs[s])).collect();
                let input = x.sub_offset(0, n * k);
                project_weights(&mut env.gpu, &input, n, Some(&rotation), &projections)
                    .map_err(dbg_err)?;
            }
        }
        Ok(())
    })
}

// ---- cli ---------------------------------------------------------------------------------

struct Cli {
    out: Option<PathBuf>,
    model: Option<String>,
    layers: Vec<usize>,
    rows: Vec<usize>,
    seeds: u64,
    bench: bool,
    bench_only: bool,
    iters: usize,
    bench_rows: usize,
}

fn parse_cli() -> R<Cli> {
    let mut cli = Cli {
        out: None,
        model: None,
        layers: DEFAULT_LAYERS.to_vec(),
        rows: DEFAULT_ROWS.to_vec(),
        seeds: 2,
        bench: false,
        bench_only: false,
        iters: 15,
        bench_rows: 32,
    };
    let mut args = std::env::args().skip(1);
    let value = |args: &mut dyn Iterator<Item = String>, flag: &str| {
        args.next().ok_or_else(|| format!("{flag} needs a value"))
    };
    let number = |text: String, flag: &str| {
        text.parse::<usize>()
            .map_err(|_| format!("{flag}: not a number: {text}"))
    };
    let list = |text: String, flag: &str| -> R<Vec<usize>> {
        text.split(',').map(|s| number(s.to_string(), flag)).collect()
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => cli.out = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--model" => cli.model = Some(value(&mut args, "--model")?),
            "--layers" => cli.layers = list(value(&mut args, "--layers")?, "--layers")?,
            "--rows" => cli.rows = list(value(&mut args, "--rows")?, "--rows")?,
            "--sweep" => cli.rows = (9..=64).collect(),
            "--seeds" => cli.seeds = number(value(&mut args, "--seeds")?, "--seeds")? as u64,
            "--bench" => cli.bench = true,
            "--bench-only" => cli.bench_only = true,
            "--iters" => cli.iters = number(value(&mut args, "--iters")?, "--iters")?,
            "--bench-rows" => cli.bench_rows = number(value(&mut args, "--bench-rows")?, "--bench-rows")?,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if cli.rows.iter().any(|n| !(1..=64).contains(n)) {
        return Err("--rows must be 1..=64".into());
    }
    Ok(cli)
}

fn run() -> R<bool> {
    let cli = parse_cli()?;
    let mut gpu = Gpu::init().map_err(err("Gpu::init"))?;
    println!("hc_wide_g942_oracle: arch={} model={:?}", gpu.arch, cli.model);
    if !gpu.arch_caps.is_gfx942() {
        return Err(format!("needs exact gfx942, got {}", gpu.arch));
    }
    let log = Arc::new(Log::default());
    let sink = log.clone();
    let observer = Arc::new(move |_gpu: &Gpu, w: &LabLaunchWitness<'_>| -> HipResult<()> {
        if sink.on.load(Ordering::Relaxed) {
            sink.symbols
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(w.symbol.to_string());
        }
        Ok(())
    });
    gpu.set_lab_launch_observer(Some(observer));

    let mut sets = vec![synthetic_weights(&mut gpu)?];
    if let Some(path) = &cli.model {
        sets.push(real_weights(&mut gpu, path, &cli.layers)?);
    }
    let bufs = Bufs::new(&mut gpu)?;
    let ctx = DispatchCtx::new(&gpu);
    let mut env = Env { gpu, ctx, bufs, log };

    let mut rep = Report::default();
    if !cli.bench_only {
        oracle(&mut env, &sets, &cli.rows, cli.seeds, &mut rep)?;
        for h in &rep.chunk_hashes {
            println!("CHUNK {h}");
        }
        for m in &rep.mismatches {
            println!("MISMATCH {m}");
        }
        for m in &rep.census_failures {
            println!("CENSUS-FAIL {m}");
        }
        let detected = rep.controls.iter().filter(|(_, d)| *d).count();
        for (name, d) in rep.controls.iter().filter(|(_, d)| !*d) {
            println!("CONTROL-UNDETECTED {name} ({d})");
        }
        let line = format!(
            "cases={} mismatches={} census_failures={} controls={}/{} rows={:?} seeds={}",
            rep.cases,
            rep.mismatches.len(),
            rep.census_failures.len(),
            detected,
            rep.controls.len(),
            cli.rows,
            cli.seeds
        );
        println!("{} {line}", if rep.passed() { "ORACLE PASS" } else { "ORACLE FAIL" });
    }
    if (cli.bench || cli.bench_only) && (cli.bench_only || rep.passed()) {
        let n = cli.bench_rows;
        let set = sets.last().expect("a weight set");
        let hc_lane = bench_hc(&mut env, set, n, Scope::Lane(n), cli.iters)?;
        let hc_chunk = bench_hc(&mut env, set, n, Scope::Off, cli.iters)?;
        let dense = bench_dense(&mut env, n, cli.iters)?;
        for (name, scope) in [("lane", Scope::Lane(n)), ("chunk", Scope::Off)] {
            let reads = bench_stage(&mut env, set, n, scope, cli.iters, false)?;
            let writes = bench_stage(&mut env, set, n, scope, cli.iters, true)?;
            println!("BENCH-STAGE n={n} route={name} reads96_ms={reads:.3} writes96_ms={writes:.3}");
        }
        println!(
            "BENCH n={n} iters={} weights={} hc_lane_scope_ms={hc_lane:.3} hc_chunk_route_ms={hc_chunk:.3} \
             dense_census_ms={dense:.3} (dense route per HIPFIRE_G942_BF16_WIDE / chain flags of this process)",
            cli.iters, set.label
        );
    }
    if let Some(dir) = &cli.out {
        std::fs::create_dir_all(dir).map_err(err("mkdir out"))?;
    }
    Ok(cli.bench_only || rep.passed())
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("hc_wide_g942_oracle: {e}");
            std::process::exit(2);
        }
    }
}
