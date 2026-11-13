// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5-family prefill GDN recurrence / chunk-boundary oracle.
//!
//! Frozen contract of slice A of `reliability/CHUNK-KL.md`: drive the
//! PRODUCTION `Gpu::gated_delta_net_q8_batch_seq` (repeated / pre-expanded
//! Q/K) and `Gpu::gated_delta_net_q8_batch_seq_indexed` (compact Q/K, runtime
//! head ratio) wrappers, the same launches the Qwen35 prefill dispatcher makes
//! for the `gated_delta_net_q8_fast` kernel, with a seeded fixture and compare
//! the output rows and the final reconstructed recurrent state against a CPU
//! recurrence, for whole-call versus split-call execution.
//!
//! ## Fixture (all planes seeded from independent sub-streams of `--seed`)
//!
//! * Q, K: L2-normalized per (token, Q/K head), F32 (`qk_heads = heads / ratio`).
//!   The repeated arm uses the `repeat_interleave_qk` expansion of the SAME
//!   compact planes (`kh = h / ratio`); both planes are hashed separately.
//! * V in [-1, 1]; log-decay `gate` in [-0.3, -0.02] (kernel `alpha = expf(gate)`);
//!   `beta = sigmoid(u)` in [0.1, 0.9].
//! * Entering state: nonzero F32 `S0` (row amplitude varies per row) quantized
//!   to the production state format: i8 codes + one F32 scale per
//!   (head, value row) (`scale = max|S0|/127`, round-nearest-even) and the F16
//!   error-feedback residual `E = f16(S0 - q*scale)`. State layout is
//!   `[head][row][key]`, `row` = value channel, scale layout `[head][row]`.
//!
//! ## References (CPU, F64; an F32 pass bounds the FP32-recurrence noise floor)
//!
//! * `continuous`: one uninterrupted recurrence from the effective entering
//!   state `q*scale + E` (F64). This is the chunk-invariant ground truth: the
//!   recurrence transform `F` applied to the whole state, `F(q*scale + E)`.
//! * `legacy (ef-at-end)` boundary model: per call, recurrence from
//!   `q*scale` ONLY, then the incoming `E` is added immediately before the
//!   final quantization (`F(q*scale) + E`), exactly what
//!   `gated_delta_net_q8_fast.hip` does at the base commit. Output rows and
//!   state are quantized at every call boundary like the kernel.
//! * `corrected (ef-at-start)` boundary model: `F(q*scale + E)` per call with
//!   the same boundary quantization.
//!
//! The pass/fail verdict is against `continuous`; the two boundary models only
//! attribute a failure to the EF placement (kernel == legacy model to
//! quantization-level noise proves the old placement, not a misreading of the
//! kernel math).
//!
//! ## Cells
//!
//! `{EF mode: zero (buffer present, zero entering) | nonzero | none (no EF
//! buffer, stochastic rounding)} x {Q/K: indexed | repeated} x {split}` where
//! `split` is the rows per call (`--splits`, default `1,25,63,64,511,512,513,
//! 1024,3072,8192`; the whole call is always run first), eager or, with
//! `--graph 1`, captured into a hipGraph (all calls of the cell in ONE graph),
//! instantiated and replayed twice with the state reset between replays; the
//! eager run, replay 1 and replay 2 must be bit-identical.
//!
//! ## Tolerances (fixed a priori, never loosened by the harness)
//!
//! * EF modes: reconstructed-state error against `continuous` must be
//!   <= 0.02 LSB of the row's final scale (the F16 residual rounding bound is
//!   ~1e-3 LSB); output rows relative error <= 1e-4 of the global max |ref|.
//! * `none` (stochastic rounding, error < 1 LSB per quantization): applied only
//!   to whole-call cells (state <= 1.001 LSB, output <= 1e-4). Split `none`
//!   cells are reported but not judged (inter-call quantization noise is not a
//!   defect).
//!
//! ## Receipts
//!
//! `--json <path>` writes input plane hashes (host planes and device
//! read-back after every run), launched kernel names, graph/replay counts,
//! per-row output max-abs errors and per-row output hashes, whole-vs-split and
//! indexed-vs-repeated comparisons, reconstructed-state hashes/errors, and an
//! input-LSB negative-control block (one LSB flipped per plane: exactly that
//! plane's hash must change, no earlier output row may change; output
//! sensitivity is reported, since FP32 arithmetic may absorb one LSB). Exit 0:
//! every judged cell and control passed; 1: at least one failed (an unmodified
//! base kernel fails its nonzero-EF cells); 2: usage / setup error.
//!
//! `cargo run --release -p rdna-compute --features lab,deltanet --example \
//!   qwen35_prefill_chunk_oracle -- --seed 1 --rows 8192 --graph 0 --json out.json`

use half::f16;
use rayon::prelude::*;
use rdna_compute::norm::restore_gdn_requant_frame_checkpoint;
use rdna_compute::{Gpu, GpuTensor, LabLaunchWitness};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

const HD: usize = 128;
const DEFAULT_SEED: u64 = 0x5133_5f43_4b4c_0001;
const DEFAULT_ROWS: usize = 8192;
const DEFAULT_SPLITS: &[usize] = &[1, 25, 63, 64, 511, 512, 513, 1024, 3072, 8192];
const DEFAULT_HEADS: usize = 4;
const DEFAULT_RATIO: usize = 2;
const FRAME_BASE: u32 = 0;
const ROW_BLOCK: usize = 8;

const STATE_TOL_LSB_EF: f64 = 0.02;
const STATE_TOL_LSB_STOCH: f64 = 1.001;
const OUT_TOL_REL: f64 = 1e-4;

const USAGE: &str = "usage: qwen35_prefill_chunk_oracle [--seed <u64>] [--rows <N>] \
[--splits <csv>] [--graph 0|1] [--json <path>] [--heads <N>] [--qk-ratio <R>]";

type Res<T> = Result<T, String>;

fn ctx<T, E: std::fmt::Display>(r: Result<T, E>, what: &str) -> Res<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

// ═══════════════════════════ CLI ══════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
struct Cli {
    seed: u64,
    rows: usize,
    splits: Vec<usize>,
    graph: bool,
    json: Option<String>,
    heads: usize,
    ratio: usize,
}

fn parse_u64(flag: &str, v: &str) -> Res<u64> {
    let parsed = match v.strip_prefix("0x").or_else(|| v.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(&h.replace('_', ""), 16),
        None => v.replace('_', "").parse::<u64>(),
    };
    parsed.map_err(|e| format!("{flag}: bad number '{v}': {e}"))
}

fn parse_cli(args: &[String]) -> Res<Cli> {
    let mut cli = Cli {
        seed: DEFAULT_SEED,
        rows: DEFAULT_ROWS,
        splits: DEFAULT_SPLITS.to_vec(),
        graph: false,
        json: None,
        heads: DEFAULT_HEADS,
        ratio: DEFAULT_RATIO,
    };
    let mut i = 0;
    let value = |i: &mut usize| -> Res<String> {
        *i += 1;
        args.get(*i)
            .cloned()
            .ok_or_else(|| format!("missing value for {}\n{USAGE}", args[*i - 1]))
    };
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => cli.seed = parse_u64("--seed", &value(&mut i)?)?,
            "--rows" => cli.rows = parse_u64("--rows", &value(&mut i)?)? as usize,
            "--splits" => {
                let v = value(&mut i)?;
                cli.splits = v
                    .split(',')
                    .map(|s| parse_u64("--splits", s.trim()).map(|x| x as usize))
                    .collect::<Res<Vec<_>>>()?;
            }
            "--graph" => {
                cli.graph = match value(&mut i)?.as_str() {
                    "0" => false,
                    "1" => true,
                    o => return Err(format!("--graph must be 0 or 1, got '{o}'")),
                }
            }
            "--json" => cli.json = Some(value(&mut i)?),
            "--heads" => cli.heads = parse_u64("--heads", &value(&mut i)?)? as usize,
            "--qk-ratio" => cli.ratio = parse_u64("--qk-ratio", &value(&mut i)?)? as usize,
            "-h" | "--help" => return Err(USAGE.to_string()),
            o => return Err(format!("unknown argument '{o}'\n{USAGE}")),
        }
        i += 1;
    }
    if cli.rows == 0 {
        return Err("--rows must be >= 1".into());
    }
    if cli.splits.is_empty() || cli.splits.contains(&0) {
        return Err("--splits must be a nonempty csv of values >= 1".into());
    }
    if cli.heads == 0 || cli.ratio == 0 || cli.heads % cli.ratio != 0 {
        return Err("--heads must be >= 1 and divisible by --qk-ratio (>= 1)".into());
    }
    Ok(cli)
}

// ═══════════════════════════ fixture ══════════════════════════════════════════

struct Rng(u64);

impl Rng {
    /// Independent sub-stream: plane `tag` of `seed` never depends on how many
    /// values another plane drew.
    fn new(seed: u64, tag: u64) -> Self {
        Rng(seed ^ tag.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn sym(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

#[derive(Clone)]
struct Planes {
    rows: usize,
    n_heads: usize,
    qk_heads: usize,
    ratio: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    q_rep: Vec<f32>,
    k_rep: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    /// F32 entering state `[head][row][key]` the Q8+EF planes were made from.
    s0: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    ef_nonzero: Vec<u16>,
    ef_zero: Vec<u16>,
}

/// `repeat_interleave_qk_f32_batched`: state head `h` reads Q/K head `h / ratio`.
fn expand_qk(src: &[f32], rows: usize, n_heads: usize, ratio: usize) -> Vec<f32> {
    let qk_heads = n_heads / ratio;
    let mut out = Vec::with_capacity(rows * n_heads * HD);
    for t in 0..rows {
        for h in 0..n_heads {
            let b = (t * qk_heads + h / ratio) * HD;
            out.extend_from_slice(&src[b..b + HD]);
        }
    }
    out
}

/// One row quantization exactly as the kernel's EF boundary (all F32 ops).
fn quant_row_ef(s: &[f32]) -> (Vec<i8>, f32, Vec<u16>) {
    let my_max = s.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let inv = if my_max > 0.0 { 127.0 / my_max } else { 0.0 };
    let scale = if my_max > 0.0 { my_max / 127.0 } else { 1.0 };
    let mut codes = Vec::with_capacity(s.len());
    let mut ef = Vec::with_capacity(s.len());
    for &x in s {
        let qf = (x * inv).round_ties_even().clamp(-128.0, 127.0);
        codes.push(qf as i8);
        ef.push(f16::from_f32(x - qf * scale).to_bits());
    }
    (codes, scale, ef)
}

fn gen_planes(seed: u64, rows: usize, n_heads: usize, ratio: usize) -> Planes {
    let qk_heads = n_heads / ratio;
    let unit_vectors = |tag: u64, n: usize| -> Vec<f32> {
        let mut rng = Rng::new(seed, tag);
        let mut out = Vec::with_capacity(n * HD);
        for _ in 0..n {
            let v: Vec<f32> = (0..HD).map(|_| rng.sym()).collect();
            let norm = v
                .iter()
                .map(|x| (*x as f64) * (*x as f64))
                .sum::<f64>()
                .sqrt();
            out.extend(v.iter().map(|x| (*x as f64 / norm) as f32));
        }
        out
    };
    let q = unit_vectors(1, rows * qk_heads);
    let k = unit_vectors(2, rows * qk_heads);
    let mut rv = Rng::new(seed, 3);
    let v: Vec<f32> = (0..rows * n_heads * HD).map(|_| rv.sym()).collect();
    let mut rg = Rng::new(seed, 4);
    let gate: Vec<f32> = (0..rows * n_heads)
        .map(|_| (-(0.02 + 0.28 * rg.unit() as f64)) as f32)
        .collect();
    let mut rb = Rng::new(seed, 5);
    let ln9 = 9.0f64.ln();
    let beta: Vec<f32> = (0..rows * n_heads)
        .map(|_| {
            let u = (rb.unit() as f64 * 2.0 - 1.0) * ln9;
            ((1.0 / (1.0 + (-u).exp())) as f32).clamp(0.1, 0.9)
        })
        .collect();
    let mut rs = Rng::new(seed, 6);
    let mut s0 = Vec::with_capacity(n_heads * HD * HD);
    let mut codes = Vec::with_capacity(n_heads * HD * HD);
    let mut scales = Vec::with_capacity(n_heads * HD);
    let mut ef_nonzero = Vec::with_capacity(n_heads * HD * HD);
    for _h in 0..n_heads {
        for _r in 0..HD {
            let amp = 0.25 + 0.75 * rs.unit();
            let row: Vec<f32> = (0..HD).map(|_| rs.sym() * amp).collect();
            let (c, sc, e) = quant_row_ef(&row);
            s0.extend_from_slice(&row);
            codes.extend(c.iter().map(|x| *x as u8));
            scales.push(sc);
            ef_nonzero.extend(e);
        }
    }
    let q_rep = expand_qk(&q, rows, n_heads, ratio);
    let k_rep = expand_qk(&k, rows, n_heads, ratio);
    Planes {
        rows,
        n_heads,
        qk_heads,
        ratio,
        q,
        k,
        q_rep,
        k_rep,
        v,
        gate,
        beta,
        s0,
        codes,
        scales,
        ef_zero: vec![0u16; ef_nonzero.len()],
        ef_nonzero,
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn sha(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}
fn fnv32(bytes: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for b in bytes {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Per-plane SHA-256 of the exact bytes handed to the device.
fn plane_hashes(p: &Planes) -> BTreeMap<&'static str, String> {
    let mut m = BTreeMap::new();
    m.insert("q", sha(&f32_bytes(&p.q)));
    m.insert("k", sha(&f32_bytes(&p.k)));
    m.insert("q_expanded", sha(&f32_bytes(&p.q_rep)));
    m.insert("k_expanded", sha(&f32_bytes(&p.k_rep)));
    m.insert("v", sha(&f32_bytes(&p.v)));
    m.insert("gate", sha(&f32_bytes(&p.gate)));
    m.insert("beta", sha(&f32_bytes(&p.beta)));
    m.insert("state_f32_s0", sha(&f32_bytes(&p.s0)));
    m.insert("q8_codes", sha(&p.codes));
    m.insert("f32_scales", sha(&f32_bytes(&p.scales)));
    m.insert("f16_ef_nonzero", sha(&u16_bytes(&p.ef_nonzero)));
    m.insert("f16_ef_zero", sha(&u16_bytes(&p.ef_zero)));
    m
}

// ═══════════════════════════ modes ════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum EfMode {
    Zero,
    Nonzero,
    None,
}

impl EfMode {
    fn label(self) -> &'static str {
        match self {
            EfMode::Zero => "zero",
            EfMode::Nonzero => "nonzero",
            EfMode::None => "none",
        }
    }
    /// Entering EF plane as the kernel will see it (`None`: no buffer).
    fn entering<'a>(self, p: &'a Planes) -> &'a [u16] {
        match self {
            EfMode::Nonzero => &p.ef_nonzero,
            EfMode::Zero | EfMode::None => &p.ef_zero,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Qk {
    Indexed,
    Repeated,
}

impl Qk {
    fn label(self) -> &'static str {
        match self {
            Qk::Indexed => "indexed",
            Qk::Repeated => "repeated",
        }
    }
}

// ═══════════════════════════ CPU models ═══════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Placement {
    /// One uninterrupted F64 recurrence from `q*scale + E`.
    Continuous,
    /// Base kernel: recurrence from `q*scale`, incoming `E` added before the
    /// final quantization of every call.
    EfAtEnd,
    /// Corrected: recurrence from `q*scale + E`.
    EfAtStart,
}

struct ModelOut {
    /// `[t][head][row]`.
    out: Vec<f64>,
    /// Final reconstructed state `[head][row][key]`.
    recon: Vec<f64>,
}

struct UnitOut {
    h: usize,
    r0: usize,
    out: Vec<f64>,
    recon: Vec<f64>,
}

#[inline]
fn recon_f32(code: u8, scale: f32, ef: u16) -> f32 {
    (code as i8 as f32) * scale + f16::from_bits(ef).to_f32()
}

/// Recurrence of `nr` rows of head `h` over tokens `[t0, t1)`; the same
/// per-row math as the kernel (F64). `out[(t - t0) * nr + r]`.
#[allow(clippy::too_many_arguments)]
fn recur_unit(
    p: &Planes,
    qd: &[f64],
    kd: &[f64],
    h: usize,
    r0: usize,
    nr: usize,
    state: &mut [f64],
    t0: usize,
    t1: usize,
    out: &mut [f64],
) {
    let nh = p.n_heads;
    let qkh = h / p.ratio;
    for t in t0..t1 {
        let alpha = (p.gate[t * nh + h] as f64).exp();
        let beta = p.beta[t * nh + h] as f64;
        let kv = &kd[(t * p.qk_heads + qkh) * HD..][..HD];
        let qv = &qd[(t * p.qk_heads + qkh) * HD..][..HD];
        for r in 0..nr {
            let s = &mut state[r * HD..(r + 1) * HD];
            let mut dot = 0.0f64;
            for c in 0..HD {
                dot += s[c] * kv[c];
            }
            let vr = p.v[(t * nh + h) * HD + r0 + r] as f64;
            let delta = (vr - alpha * dot) * beta;
            let mut o = 0.0f64;
            for c in 0..HD {
                s[c] = alpha * s[c] + kv[c] * delta;
                o += s[c] * qv[c];
            }
            out[(t - t0) * nr + r] = o;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn model_unit(
    p: &Planes,
    qd: &[f64],
    kd: &[f64],
    ef_start: &[u16],
    placement: Placement,
    split: usize,
    h: usize,
    r0: usize,
) -> UnitOut {
    let nr = ROW_BLOCK;
    let base = h * HD + r0;
    let mut out = vec![0.0f64; p.rows * nr];
    let mut codes: Vec<u8> = p.codes[base * HD..(base + nr) * HD].to_vec();
    let mut scales: Vec<f32> = p.scales[base..base + nr].to_vec();
    let mut ef: Vec<u16> = ef_start[base * HD..(base + nr) * HD].to_vec();
    if placement == Placement::Continuous {
        let mut state = vec![0.0f64; nr * HD];
        for r in 0..nr {
            for c in 0..HD {
                state[r * HD + c] = (codes[r * HD + c] as i8) as f64 * scales[r] as f64
                    + f16::from_bits(ef[r * HD + c]).to_f64();
            }
        }
        recur_unit(p, qd, kd, h, r0, nr, &mut state, 0, p.rows, &mut out);
        return UnitOut {
            h,
            r0,
            out,
            recon: state,
        };
    }
    let mut t0 = 0;
    while t0 < p.rows {
        let t1 = (t0 + split).min(p.rows);
        let mut state = vec![0.0f64; nr * HD];
        for r in 0..nr {
            for c in 0..HD {
                let qs = (codes[r * HD + c] as i8 as f32) * scales[r];
                let s_in = if placement == Placement::EfAtStart {
                    qs + f16::from_bits(ef[r * HD + c]).to_f32()
                } else {
                    qs
                };
                state[r * HD + c] = s_in as f64;
            }
        }
        let ef_in = ef.clone();
        recur_unit(
            p,
            qd,
            kd,
            h,
            r0,
            nr,
            &mut state,
            t0,
            t1,
            &mut out[t0 * nr..t1 * nr],
        );
        for r in 0..nr {
            let mut s = [0.0f32; HD];
            for c in 0..HD {
                s[c] = state[r * HD + c] as f32;
                if placement == Placement::EfAtEnd {
                    s[c] += f16::from_bits(ef_in[r * HD + c]).to_f32();
                }
            }
            let (c8, sc, e) = quant_row_ef(&s);
            for c in 0..HD {
                codes[r * HD + c] = c8[c] as u8;
                ef[r * HD + c] = e[c];
            }
            scales[r] = sc;
        }
        t0 = t1;
    }
    let mut recon = vec![0.0f64; nr * HD];
    for r in 0..nr {
        for c in 0..HD {
            recon[r * HD + c] = recon_f32(codes[r * HD + c], scales[r], ef[r * HD + c]) as f64;
        }
    }
    UnitOut { h, r0, out, recon }
}

fn to_f64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| *x as f64).collect()
}

fn model(
    p: &Planes,
    qd: &[f64],
    kd: &[f64],
    ef_start: &[u16],
    placement: Placement,
    split: usize,
) -> ModelOut {
    let nh = p.n_heads;
    let units: Vec<(usize, usize)> = (0..nh)
        .flat_map(|h| (0..HD / ROW_BLOCK).map(move |b| (h, b * ROW_BLOCK)))
        .collect();
    let results: Vec<UnitOut> = units
        .par_iter()
        .map(|&(h, r0)| model_unit(p, qd, kd, ef_start, placement, split, h, r0))
        .collect();
    let mut out = vec![0.0f64; p.rows * nh * HD];
    let mut recon = vec![0.0f64; nh * HD * HD];
    for u in results {
        for t in 0..p.rows {
            for r in 0..ROW_BLOCK {
                out[(t * nh + u.h) * HD + u.r0 + r] = u.out[t * ROW_BLOCK + r];
            }
        }
        let b = (u.h * HD + u.r0) * HD;
        recon[b..b + ROW_BLOCK * HD].copy_from_slice(&u.recon);
    }
    ModelOut { out, recon }
}

/// FP32 sequential recurrence (kernel summation order is not reproducible on
/// the CPU; this bounds the FP32 noise floor against the F64 reference).
fn f32_floor(p: &Planes, ef_start: &[u16], f64_ref: &ModelOut) -> (f64, f64) {
    let nh = p.n_heads;
    let qd: Vec<f32> = p.q.clone();
    let kd: Vec<f32> = p.k.clone();
    let units: Vec<(usize, usize)> = (0..nh)
        .flat_map(|h| (0..HD / ROW_BLOCK).map(move |b| (h, b * ROW_BLOCK)))
        .collect();
    let res: Vec<(usize, usize, Vec<f32>, Vec<f32>)> = units
        .par_iter()
        .map(|&(h, r0)| {
            let nr = ROW_BLOCK;
            let base = h * HD + r0;
            let mut st = vec![0.0f32; nr * HD];
            for r in 0..nr {
                for c in 0..HD {
                    st[r * HD + c] = (p.codes[(base + r) * HD + c] as i8 as f32)
                        * p.scales[base + r]
                        + f16::from_bits(ef_start[(base + r) * HD + c]).to_f32();
                }
            }
            let mut out = vec![0.0f32; p.rows * nr];
            let qkh = h / p.ratio;
            for t in 0..p.rows {
                let alpha = p.gate[t * nh + h].exp();
                let beta = p.beta[t * nh + h];
                let kv = &kd[(t * p.qk_heads + qkh) * HD..][..HD];
                let qv = &qd[(t * p.qk_heads + qkh) * HD..][..HD];
                for r in 0..nr {
                    let s = &mut st[r * HD..(r + 1) * HD];
                    let mut dot = 0.0f32;
                    for c in 0..HD {
                        dot += s[c] * kv[c];
                    }
                    let vr = p.v[(t * nh + h) * HD + r0 + r];
                    let delta = (vr - alpha * dot) * beta;
                    let mut o = 0.0f32;
                    for c in 0..HD {
                        s[c] = alpha * s[c] + kv[c] * delta;
                        o += s[c] * qv[c];
                    }
                    out[t * nr + r] = o;
                }
            }
            (h, r0, out, st)
        })
        .collect();
    let mut out_err = 0.0f64;
    let mut st_err = 0.0f64;
    for (h, r0, out, st) in res {
        for t in 0..p.rows {
            for r in 0..ROW_BLOCK {
                let e =
                    (out[t * ROW_BLOCK + r] as f64 - f64_ref.out[(t * nh + h) * HD + r0 + r]).abs();
                out_err = out_err.max(e);
            }
        }
        for r in 0..ROW_BLOCK {
            for c in 0..HD {
                let e = (st[r * HD + c] as f64 - f64_ref.recon[((h * HD + r0 + r) * HD) + c]).abs();
                st_err = st_err.max(e);
            }
        }
    }
    (out_err, st_err)
}

// ═══════════════════════════ device side ══════════════════════════════════════

struct Dev {
    q: GpuTensor,
    k: GpuTensor,
    q_rep: GpuTensor,
    k_rep: GpuTensor,
    v: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    out: GpuTensor,
    codes: GpuTensor,
    scales: GpuTensor,
    ef: GpuTensor,
    poison: Vec<u8>,
}

fn upload_dev(gpu: &mut Gpu, p: &Planes) -> Res<Dev> {
    let up = |gpu: &mut Gpu, v: &[f32]| ctx(gpu.upload_f32(v, &[v.len()]), "upload f32");
    let out_n = p.rows * p.n_heads * HD;
    Ok(Dev {
        q: up(gpu, &p.q)?,
        k: up(gpu, &p.k)?,
        q_rep: up(gpu, &p.q_rep)?,
        k_rep: up(gpu, &p.k_rep)?,
        v: up(gpu, &p.v)?,
        gate: up(gpu, &p.gate)?,
        beta: up(gpu, &p.beta)?,
        out: up(gpu, &vec![f32::from_bits(0xFFFF_FFFF); out_n])?,
        codes: ctx(gpu.upload_raw(&p.codes, &[p.codes.len()]), "upload codes")?,
        scales: up(gpu, &p.scales)?,
        ef: {
            let b = u16_bytes(&p.ef_zero);
            ctx(gpu.upload_raw(&b, &[b.len()]), "upload ef")?
        },
        poison: vec![0xFFu8; out_n * 4],
    })
}

fn free_dev(gpu: &mut Gpu, d: Dev) {
    for t in [
        d.q, d.k, d.q_rep, d.k_rep, d.v, d.gate, d.beta, d.out, d.codes, d.scales, d.ef,
    ] {
        let _ = gpu.free_tensor(t);
    }
}

fn sync(gpu: &Gpu) -> Res<()> {
    ctx(gpu.hip.device_synchronize(), "device synchronize")
}

fn reset_state(gpu: &Gpu, d: &Dev, p: &Planes, ef: EfMode) -> Res<()> {
    sync(gpu)?;
    ctx(gpu.hip.memcpy_htod(&d.codes.buf, &p.codes), "reset codes")?;
    ctx(
        gpu.hip.memcpy_htod(&d.scales.buf, &f32_bytes(&p.scales)),
        "reset scales",
    )?;
    ctx(
        gpu.hip.memcpy_htod(&d.ef.buf, &u16_bytes(ef.entering(p))),
        "reset ef",
    )?;
    ctx(gpu.hip.memcpy_htod(&d.out.buf, &d.poison), "poison out")?;
    Ok(())
}

/// The production launch sequence of one cell: `calls` consecutive wrapper
/// calls of `split` rows (last one shorter).
fn launch_calls(
    gpu: &mut Gpu,
    d: &Dev,
    p: &Planes,
    ef: EfMode,
    qk: Qk,
    split: usize,
) -> Res<usize> {
    restore_gdn_requant_frame_checkpoint(FRAME_BASE);
    let nh = p.n_heads;
    let ef_t = match ef {
        EfMode::None => None,
        _ => Some(&d.ef),
    };
    let (qs, ks, qw, ratio) = match qk {
        Qk::Indexed => (&d.q, &d.k, p.qk_heads * HD, p.ratio),
        Qk::Repeated => (&d.q_rep, &d.k_rep, nh * HD, 1),
    };
    let (mut t0, mut calls) = (0usize, 0usize);
    while t0 < p.rows {
        let n = split.min(p.rows - t0);
        let qv = qs.sub_offset(t0 * qw, n * qw);
        let kv = ks.sub_offset(t0 * qw, n * qw);
        let vv = d.v.sub_offset(t0 * nh * HD, n * nh * HD);
        let gv = d.gate.sub_offset(t0 * nh, n * nh);
        let bv = d.beta.sub_offset(t0 * nh, n * nh);
        let ov = d.out.sub_offset(t0 * nh * HD, n * nh * HD);
        let r = match qk {
            Qk::Indexed => gpu.gated_delta_net_q8_batch_seq_indexed(
                &qv, &kv, &vv, &gv, &bv, &d.codes, &d.scales, &ov, n, nh, HD, ef_t, ratio,
            ),
            Qk::Repeated => gpu.gated_delta_net_q8_batch_seq(
                &qv, &kv, &vv, &gv, &bv, &d.codes, &d.scales, &ov, n, nh, HD, ef_t,
            ),
        };
        ctx(r, "gated_delta_net_q8_batch_seq")?;
        calls += 1;
        t0 += n;
    }
    Ok(calls)
}

struct KernelResult {
    out: Vec<f32>,
    codes: Vec<u8>,
    scales: Vec<f32>,
    ef: Vec<u16>,
}

impl KernelResult {
    fn bits_eq(&self, o: &KernelResult) -> bool {
        self.out.len() == o.out.len()
            && self
                .out
                .iter()
                .zip(&o.out)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.codes == o.codes
            && self
                .scales
                .iter()
                .zip(&o.scales)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && self.ef == o.ef
    }
    /// Reconstructed state (F32 ops as the kernel's own dequantization).
    fn recon(&self) -> Vec<f32> {
        (0..self.codes.len())
            .map(|i| recon_f32(self.codes[i], self.scales[i / HD], self.ef[i]))
            .collect()
    }
}

fn read_result(gpu: &Gpu, d: &Dev) -> Res<KernelResult> {
    sync(gpu)?;
    let ef_bytes = ctx(gpu.download_raw_bytes(&d.ef), "download ef")?;
    Ok(KernelResult {
        out: ctx(gpu.download_f32(&d.out), "download out")?,
        codes: ctx(gpu.download_raw_bytes(&d.codes), "download codes")?,
        scales: ctx(gpu.download_f32(&d.scales), "download scales")?,
        ef: ef_bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
    })
}

/// Device input planes still equal the host planes (kernel must never write
/// its inputs; also proves no plane was regenerated between arms).
fn inputs_unchanged(gpu: &Gpu, d: &Dev, p: &Planes) -> Res<bool> {
    sync(gpu)?;
    let mut ok = true;
    for (t, host) in [
        (&d.q, &p.q),
        (&d.k, &p.k),
        (&d.q_rep, &p.q_rep),
        (&d.k_rep, &p.k_rep),
        (&d.v, &p.v),
        (&d.gate, &p.gate),
        (&d.beta, &p.beta),
    ] {
        ok &= ctx(gpu.download_f32(t), "readback input")?
            .iter()
            .zip(host)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    }
    Ok(ok)
}

/// Hash the actual entering bytes, including all three mutable state planes,
/// before recurrence. A boolean alone would not attest state-input equality.
fn entering_hashes(gpu: &Gpu, d: &Dev) -> Res<BTreeMap<&'static str, String>> {
    sync(gpu)?;
    let mut hashes = BTreeMap::new();
    for (name, tensor) in [
        ("q", &d.q),
        ("k", &d.k),
        ("q_expanded", &d.q_rep),
        ("k_expanded", &d.k_rep),
        ("v", &d.v),
        ("gate", &d.gate),
        ("beta", &d.beta),
        ("q8_codes", &d.codes),
        ("f32_scales", &d.scales),
        ("f16_ef", &d.ef),
    ] {
        hashes.insert(
            name,
            sha(&ctx(gpu.download_raw_bytes(tensor), "hash entering plane")?),
        );
    }
    Ok(hashes)
}

fn expected_entering_hashes(p: &Planes, ef: EfMode) -> BTreeMap<&'static str, String> {
    let mut hashes = plane_hashes(p);
    hashes.remove("state_f32_s0");
    hashes.remove("f16_ef_zero");
    hashes.remove("f16_ef_nonzero");
    hashes.insert("f16_ef", sha(&u16_bytes(ef.entering(p))));
    hashes
}

type KernelLog = Arc<Mutex<BTreeMap<String, u64>>>;

fn log_take(log: &KernelLog) -> BTreeMap<String, u64> {
    std::mem::take(&mut *log.lock().unwrap())
}

fn capture_graph(
    gpu: &mut Gpu,
    expected: usize,
    body: impl FnOnce(&mut Gpu) -> Res<()>,
) -> Res<()> {
    sync(gpu)?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    {
        let stream = gpu
            .active_stream
            .as_ref()
            .ok_or("graph capture: no active stream")?;
        ctx(
            gpu.graphs
                .begin_graph_capture(&gpu.hip, gpu.device_id, stream),
            "begin graph capture",
        )?;
    }
    let result = body(gpu);
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or("graph capture: stream vanished")?;
    if let Err(e) = result {
        gpu.graphs
            .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
        return Err(format!("graph capture body: {e}"));
    }
    ctx(
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, stream),
        "end graph capture",
    )?;
    let n = gpu.graphs.ar_forward_blobs.len();
    if n != expected || gpu.graphs.graph_exec.is_none() {
        return Err(format!(
            "graph capture recorded {n} launches, expected {expected} (or no instantiated graph)"
        ));
    }
    Ok(())
}

fn replay(gpu: &Gpu) -> Res<()> {
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or("graph replay: no active stream")?;
    ctx(
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stream),
        "graph replay",
    )?;
    sync(gpu)
}

/// Execute one cell; with `graph` the REPORTED result is graph replay 1 and the
/// eager run / replay 2 are bit-compared against it.
fn exec_cell(
    gpu: &mut Gpu,
    d: &Dev,
    p: &Planes,
    ef: EfMode,
    qk: Qk,
    split: usize,
    graph: bool,
    log: &KernelLog,
) -> Res<(KernelResult, Value, bool)> {
    let _ = log_take(log);
    reset_state(gpu, d, p, ef)?;
    let entering = entering_hashes(gpu, d)?;
    let entering_ok = entering == expected_entering_hashes(p, ef);
    let calls = launch_calls(gpu, d, p, ef, qk, split)?;
    let eager_kernels = log_take(log);
    let eager = read_result(gpu, d)?;
    let in_ok = entering_ok && inputs_unchanged(gpu, d, p)?;
    if !graph {
        let info = json!({
            "mode": "eager",
            "calls": calls,
            "kernels_launched": eager_kernels,
            "entering_plane_sha256": entering,
            "entering_hashes_match_host": entering_ok,
        });
        return Ok((eager, info, in_ok));
    }
    reset_state(gpu, d, p, ef)?;
    let capture_inputs_ok = entering_hashes(gpu, d)? == entering;
    capture_graph(gpu, calls, |g| {
        launch_calls(g, d, p, ef, qk, split).map(|_| ())
    })?;
    let capture_kernels = log_take(log);
    let captured_launches = gpu.graphs.ar_forward_blobs.len();
    replay(gpu)?;
    let r1 = read_result(gpu, d)?;
    reset_state(gpu, d, p, ef)?;
    let replay2_inputs_ok = entering_hashes(gpu, d)? == entering;
    replay(gpu)?;
    let r2 = read_result(gpu, d)?;
    let in_ok_after = inputs_unchanged(gpu, d, p)?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    let e_eq_r1 = eager.bits_eq(&r1);
    let r1_eq_r2 = r1.bits_eq(&r2);
    let info = json!({
        "mode": "hipgraph",
        "calls": calls,
        "kernels_launched_eager": eager_kernels,
        "kernels_launched_capture": capture_kernels,
        "graphs_captured": 1,
        "captured_kernel_nodes": captured_launches,
        "replay_count": 2,
        "entering_plane_sha256": entering,
        "entering_hashes_match_host": entering_ok && capture_inputs_ok && replay2_inputs_ok,
        "eager_vs_replay1_bit_identical": e_eq_r1,
        "replay1_vs_replay2_bit_identical": r1_eq_r2,
    });
    Ok((
        r1,
        info,
        in_ok && in_ok_after && capture_inputs_ok && replay2_inputs_ok && e_eq_r1 && r1_eq_r2,
    ))
}

// ═══════════════════════════ metrics ══════════════════════════════════════════

struct Metrics {
    row_err: Vec<f32>,
    out_err_max: f64,
    out_ref_max: f64,
    nonfinite: usize,
    state_max_lsb: f64,
    state_rms_lsb: f64,
    state_max_abs: f64,
    state_arg: (usize, usize, usize),
}

fn compare_to(kr: &KernelResult, refm: &ModelOut, p: &Planes) -> Metrics {
    let nh = p.n_heads;
    let rw = nh * HD;
    let mut row_err = vec![0.0f32; p.rows];
    let mut nonfinite = 0;
    let mut out_err_max = 0.0f64;
    let mut out_ref_max = 0.0f64;
    for t in 0..p.rows {
        let mut m = 0.0f64;
        for i in 0..rw {
            let kv = kr.out[t * rw + i];
            let rv = refm.out[t * rw + i];
            out_ref_max = out_ref_max.max(rv.abs());
            if !kv.is_finite() {
                nonfinite += 1;
                m = f64::INFINITY;
            } else {
                m = m.max((kv as f64 - rv).abs());
            }
        }
        row_err[t] = m as f32;
        out_err_max = out_err_max.max(m);
    }
    let recon = kr.recon();
    let (mut mx, mut ss, mut mabs, mut arg) = (0.0f64, 0.0f64, 0.0f64, (0, 0, 0));
    for h in 0..nh {
        for r in 0..HD {
            let sc = kr.scales[h * HD + r] as f64;
            for c in 0..HD {
                let i = (h * HD + r) * HD + c;
                let e = if !recon[i].is_finite() || !sc.is_finite() || sc <= 0.0 {
                    nonfinite += 1;
                    f64::INFINITY
                } else {
                    (recon[i] as f64 - refm.recon[i]).abs()
                };
                mabs = mabs.max(e);
                let lsb = e / sc;
                ss += lsb * lsb;
                if lsb > mx {
                    mx = lsb;
                    arg = (h, r, c);
                }
            }
        }
    }
    Metrics {
        row_err,
        out_err_max,
        out_ref_max,
        nonfinite,
        state_max_lsb: mx,
        state_rms_lsb: (ss / (nh * HD * HD) as f64).sqrt(),
        state_max_abs: mabs,
        state_arg: arg,
    }
}

fn row_hashes(kr: &KernelResult, p: &Planes) -> String {
    let rw = p.n_heads * HD;
    let mut s = String::with_capacity(p.rows * 8);
    for t in 0..p.rows {
        s.push_str(&format!(
            "{:08x}",
            fnv32(&f32_bytes(&kr.out[t * rw..(t + 1) * rw]))
        ));
    }
    s
}

fn state_hashes(kr: &KernelResult) -> Value {
    json!({
        "q8_codes_sha256": sha(&kr.codes),
        "f32_scales_sha256": sha(&f32_bytes(&kr.scales)),
        "f16_ef_sha256": sha(&u16_bytes(&kr.ef)),
        "reconstructed_f32_sha256": sha(&f32_bytes(&kr.recon())),
    })
}

/// Kernel against a boundary model (same quantization schedule): output
/// max-abs error and reconstructed-state error in kernel-scale LSBs.
fn compare_boundary(kr: &KernelResult, m: &ModelOut, p: &Planes) -> Value {
    let mt = compare_to(kr, m, p);
    json!({
        "out_max_abs_err": mt.out_err_max,
        "out_rel_err": mt.out_err_max / mt.out_ref_max.max(f64::MIN_POSITIVE),
        "state_max_lsb": mt.state_max_lsb,
        "state_rms_lsb": mt.state_rms_lsb,
    })
}

// ═══════════════════════════ negative controls ═══════════════════════════════

fn flip_f32(v: &mut [f32], i: usize) {
    v[i] = f32::from_bits(v[i].to_bits() ^ 1);
}

/// One LSB flipped in one plane: exactly that plane's hash must change and
/// no earlier output row may change. Report output sensitivity separately:
/// expf/FP32 may absorb the bit; the old kernel ignores EF until its commit.
fn lsb_controls(
    gpu: &mut Gpu,
    base: &Planes,
    base_result: &KernelResult,
    log: &KernelLog,
) -> Res<(Value, bool)> {
    let fr = base.rows / 2;
    let nh = base.n_heads;
    let rw = nh * HD;
    let h0 = plane_hashes(base);
    let mut all_ok = true;
    let mut arr = Vec::new();
    // (plane, expected first affected row)
    for (plane, first_row) in [
        ("q", fr),
        ("k", fr),
        ("v", fr),
        ("gate", fr),
        ("beta", fr),
        ("q8_codes", 0),
        ("f32_scales", 0),
        ("f16_ef_nonzero", 0),
    ] {
        let mut m = base.clone();
        match plane {
            "q" => {
                flip_f32(&mut m.q, fr * m.qk_heads * HD);
                m.q_rep = expand_qk(&m.q, m.rows, m.n_heads, m.ratio);
            }
            "k" => {
                flip_f32(&mut m.k, fr * m.qk_heads * HD);
                m.k_rep = expand_qk(&m.k, m.rows, m.n_heads, m.ratio);
            }
            "v" => flip_f32(&mut m.v, fr * rw),
            "gate" => flip_f32(&mut m.gate, fr * nh),
            "beta" => flip_f32(&mut m.beta, fr * nh),
            "q8_codes" => m.codes[0] ^= 1,
            "f32_scales" => flip_f32(&mut m.scales, 0),
            _ => m.ef_nonzero[0] ^= 1,
        }
        let h1 = plane_hashes(&m);
        let changed: Vec<&str> = h0
            .iter()
            .filter(|(k, v)| h1[**k] != **v)
            .map(|(k, _)| *k)
            .collect();
        // Planes that legitimately co-change with the flipped one.
        let expected_changed: Vec<&str> = match plane {
            "q" => vec!["q", "q_expanded"],
            "k" => vec!["k", "k_expanded"],
            p => vec![p],
        };
        let hash_ok = changed == expected_changed;
        let dev = upload_dev(gpu, &m)?;
        reset_state(gpu, &dev, &m, EfMode::Nonzero)?;
        launch_calls(gpu, &dev, &m, EfMode::Nonzero, Qk::Indexed, m.rows)?;
        let _ = log_take(log);
        let r = read_result(gpu, &dev)?;
        free_dev(gpu, dev);
        let before_equal = (0..first_row).all(|t| {
            (0..rw).all(|i| r.out[t * rw + i].to_bits() == base_result.out[t * rw + i].to_bits())
        });
        let diff_in_row = (0..rw)
            .filter(|i| {
                r.out[first_row * rw + i].to_bits() != base_result.out[first_row * rw + i].to_bits()
            })
            .count();
        let first_diff_row = (0..m.rows).find(|t| {
            (0..rw).any(|i| r.out[t * rw + i].to_bits() != base_result.out[t * rw + i].to_bits())
        });
        let state_bits_equal = r.codes == base_result.codes
            && r.scales
                .iter()
                .zip(&base_result.scales)
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && r.ef == base_result.ef;
        // q only touches its own row's output: state must stay bit-identical.
        let state_ok = if plane == "q" { state_bits_equal } else { true };
        let ok = hash_ok && before_equal && state_ok;
        all_ok &= ok;
        arr.push(json!({
            "plane": plane,
            "flipped": "one LSB of one element",
            "expected_first_affected_row": first_row,
            "hash_changed_planes": changed,
            "hash_change_exactly_expected": hash_ok,
            "rows_before_bit_identical": before_equal,
            "differing_elements_in_first_affected_row": diff_in_row,
            "output_sensitive_to_lsb": diff_in_row > 0,
            "first_differing_row": first_diff_row,
            "final_state_bits_identical_to_baseline": state_bits_equal,
            "detected": ok,
        }));
    }
    Ok((json!({ "all_detected": all_ok, "controls": arr }), all_ok))
}

// ═══════════════════════════ main ═════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}

fn run(args: &[String]) -> Res<i32> {
    let cli = parse_cli(args)?;
    if hipfire_env_per_token() {
        return Err(
            "HIPFIRE_DN_REQUANT_PER_TOKEN is set: the oracle targets gated_delta_net_q8_fast"
                .into(),
        );
    }
    let p = gen_planes(cli.seed, cli.rows, cli.heads, cli.ratio);
    let hashes = plane_hashes(&p);
    let mut gpu = ctx(Gpu::init(), "Gpu::init")?;
    if cli.graph {
        ctx(gpu.ensure_capture_stream(), "graph route: capture stream")?;
    }
    let log: KernelLog = Arc::new(Mutex::new(BTreeMap::new()));
    {
        let l = log.clone();
        gpu.set_lab_launch_observer(Some(Arc::new(
            move |_g: &Gpu, w: &LabLaunchWitness<'_>| -> rdna_compute::HipResult<()> {
                *l.lock()
                    .unwrap()
                    .entry(format!(
                        "{}[grid={:?},block={:?}]",
                        w.symbol, w.grid, w.block
                    ))
                    .or_insert(0) += 1;
                Ok(())
            },
        )));
    }
    let arch = gpu.arch.clone();
    eprintln!(
        "[chunk-oracle] arch={arch} seed={:#x} rows={} heads={} qk_ratio={} graph={} splits={:?}",
        cli.seed, cli.rows, cli.heads, cli.ratio, cli.graph, cli.splits
    );

    // Whole call first, then the requested splits (clamped to `rows`, deduped).
    let mut splits: Vec<usize> = vec![cli.rows];
    for s in &cli.splits {
        let s = (*s).min(cli.rows);
        if !splits.contains(&s) {
            splits.push(s);
        }
    }

    let qd = to_f64(&p.q);
    let kd = to_f64(&p.k);
    let dev = upload_dev(&mut gpu, &p)?;
    let mut cells: Vec<Value> = Vec::new();
    let mut whole: HashMap<(EfMode, Qk), KernelResult> = HashMap::new();
    let mut failed = 0usize;
    let mut judged = 0usize;
    let mut kernel_names: BTreeMap<String, u64> = BTreeMap::new();
    let mut f32_floor_by_ef: HashMap<&'static str, (f64, f64)> = HashMap::new();

    for ef in [EfMode::Zero, EfMode::Nonzero, EfMode::None] {
        let ef_start = ef.entering(&p);
        let cont = model(&p, &qd, &kd, ef_start, Placement::Continuous, cli.rows);
        let floor = f32_floor(&p, ef_start, &cont);
        f32_floor_by_ef.insert(ef.label(), floor);
        eprintln!(
            "[chunk-oracle] ef={} continuous F64 ref ready (F32-vs-F64 floor: out {:.3e} state {:.3e})",
            ef.label(),
            floor.0,
            floor.1
        );
        for &split in &splits {
            let (old_m, new_m) = if ef == EfMode::None {
                (None, None)
            } else {
                (
                    Some(model(&p, &qd, &kd, ef_start, Placement::EfAtEnd, split)),
                    Some(model(&p, &qd, &kd, ef_start, Placement::EfAtStart, split)),
                )
            };
            let mut qk_results: Vec<(Qk, KernelResult)> = Vec::new();
            for qk in [Qk::Indexed, Qk::Repeated] {
                let (kr, info, exec_ok) =
                    exec_cell(&mut gpu, &dev, &p, ef, qk, split, cli.graph, &log)?;
                if let Some(ks) = info
                    .get("kernels_launched")
                    .or_else(|| info.get("kernels_launched_eager"))
                    .and_then(|v| v.as_object())
                {
                    for (k, v) in ks {
                        *kernel_names.entry(k.clone()).or_insert(0) += v.as_u64().unwrap_or(0);
                    }
                }
                let calls = info["calls"].as_u64().unwrap() as usize;
                let mt = compare_to(&kr, &cont, &p);
                let out_rel = mt.out_err_max / mt.out_ref_max.max(f64::MIN_POSITIVE);
                let judged_cell = ef != EfMode::None || calls == 1;
                let state_tol = if ef == EfMode::None {
                    STATE_TOL_LSB_STOCH
                } else {
                    STATE_TOL_LSB_EF
                };
                let out_ok = !judged_cell || (mt.nonfinite == 0 && out_rel <= OUT_TOL_REL);
                let state_ok = !judged_cell || mt.state_max_lsb <= state_tol;
                let finite_ok = mt.nonfinite == 0;
                let pass = exec_ok && finite_ok && out_ok && state_ok;
                if judged_cell {
                    judged += 1;
                    if !pass {
                        failed += 1;
                    }
                } else if !(exec_ok && finite_ok) {
                    judged += 1;
                    failed += 1;
                }
                // Whole-vs-split against the same arm's whole call.
                let whole_cmp = if split == cli.rows {
                    Value::Null
                } else {
                    let w = &whole[&(ef, qk)];
                    let rw = p.n_heads * HD;
                    let rows_equal = (0..p.rows)
                        .filter(|t| {
                            (0..rw).all(|i| {
                                kr.out[t * rw + i].to_bits() == w.out[t * rw + i].to_bits()
                            })
                        })
                        .count();
                    let first_diff = (0..p.rows).find(|t| {
                        (0..rw).any(|i| kr.out[t * rw + i].to_bits() != w.out[t * rw + i].to_bits())
                    });
                    let max_abs = (0..p.rows * rw)
                        .map(|i| (kr.out[i] as f64 - w.out[i] as f64).abs())
                        .fold(0.0f64, f64::max);
                    let (a, b) = (kr.recon(), w.recon());
                    let mut st_lsb = 0.0f64;
                    for i in 0..a.len() {
                        st_lsb =
                            st_lsb.max((a[i] as f64 - b[i] as f64).abs() / w.scales[i / HD] as f64);
                    }
                    json!({
                        "rows_bit_identical_to_whole": rows_equal,
                        "rows_total": p.rows,
                        "first_row_differing_from_whole": first_diff,
                        "out_max_abs_diff_vs_whole": max_abs,
                        "recon_state_max_diff_lsb_vs_whole": st_lsb,
                        "state_bytes_identical_to_whole": kr.codes == w.codes && kr.ef == w.ef
                            && kr.scales.iter().zip(&w.scales).all(|(x, y)| x.to_bits() == y.to_bits()),
                    })
                };
                let cell = json!({
                    "ef_mode": ef.label(),
                    "qk_mode": qk.label(),
                    "split": split,
                    "calls": calls,
                    "judged": judged_cell,
                    "execution": info,
                    "input_planes_unchanged_on_device_and_graph_ok": exec_ok,
                    "output_nonfinite": mt.nonfinite,
                    "vs_continuous_f64_reference": {
                        "out_max_abs_err": mt.out_err_max,
                        "out_ref_max_abs": mt.out_ref_max,
                        "out_rel_err": out_rel,
                        "out_tol_rel": OUT_TOL_REL,
                        "out_ok": out_ok,
                        "state_max_err_lsb": mt.state_max_lsb,
                        "state_rms_err_lsb": mt.state_rms_lsb,
                        "state_max_err_abs": mt.state_max_abs,
                        "state_max_err_at_head_row_key": [mt.state_arg.0, mt.state_arg.1, mt.state_arg.2],
                        "state_tol_lsb": state_tol,
                        "state_ok": state_ok,
                        "per_row_out_max_abs_err": mt.row_err,
                    },
                    "vs_legacy_ef_at_end_model": old_m.as_ref().map(|m| compare_boundary(&kr, m, &p)),
                    "vs_corrected_ef_at_start_model": new_m.as_ref().map(|m| compare_boundary(&kr, m, &p)),
                    "whole_vs_split": whole_cmp,
                    "output_sha256": sha(&f32_bytes(&kr.out)),
                    "per_row_output_fnv32_hex": row_hashes(&kr, &p),
                    "final_state": state_hashes(&kr),
                    "pass": pass,
                });
                eprintln!(
                    "[chunk-oracle] ef={:<7} qk={:<8} split={:<5} calls={:<5} out_rel={:.3e} state_max={:.3e}LSB legacy_model_state={} -> {}{}",
                    ef.label(),
                    qk.label(),
                    split,
                    calls,
                    out_rel,
                    mt.state_max_lsb,
                    old_m
                        .as_ref()
                        .map(|m| format!("{:.3e}", compare_to(&kr, m, &p).state_max_lsb))
                        .unwrap_or_else(|| "-".into()),
                    if pass { "PASS" } else { "FAIL" },
                    if judged_cell { "" } else { " (unjudged)" }
                );
                cells.push(cell);
                qk_results.push((qk, kr));
            }
            // Indexed vs repeated bit identity of this (ef, split).
            let same = qk_results[0].1.bits_eq(&qk_results[1].1);
            let n = cells.len();
            cells[n - 2]["indexed_vs_repeated_bit_identical"] = json!(same);
            cells[n - 1]["indexed_vs_repeated_bit_identical"] = json!(same);
            if !same {
                failed += 1;
                cells[n - 2]["pass"] = json!(false);
                cells[n - 1]["pass"] = json!(false);
            }
            if split == cli.rows {
                for (qk, kr) in qk_results {
                    whole.insert((ef, qk), kr);
                }
            }
        }
    }

    // Negative controls (indexed, nonzero EF, whole call).
    let (controls, controls_ok) =
        lsb_controls(&mut gpu, &p, &whole[&(EfMode::Nonzero, Qk::Indexed)], &log)?;
    if !controls_ok {
        failed += 1;
    }
    eprintln!("[chunk-oracle] input-LSB negative controls detected: {controls_ok}");

    // First kill gate: nonzero-EF whole-call state must FAIL, zero-EF whole must
    // PASS, input hashes identical across all cells.
    let whole_cell = |ef: &str, qk: &str| -> Option<&Value> {
        cells.iter().find(|c| {
            c["ef_mode"] == ef && c["qk_mode"] == qk && c["split"].as_u64() == Some(cli.rows as u64)
        })
    };
    let state_fails =
        |c: Option<&Value>| c.map(|c| c["vs_continuous_f64_reference"]["state_ok"] == false);
    let nz_i = state_fails(whole_cell("nonzero", "indexed"));
    let nz_r = state_fails(whole_cell("nonzero", "repeated"));
    let z_i = state_fails(whole_cell("zero", "indexed"));
    let z_r = state_fails(whole_cell("zero", "repeated"));
    let all_inputs_ok = cells
        .iter()
        .all(|c| c["input_planes_unchanged_on_device_and_graph_ok"] == true);
    let kill = json!({
        "definition": "unmodified base kernel: nonzero-EF whole-call reconstructed state FAILS the continuous reference while input hashes match; zero-EF whole-call PASSES",
        "nonzero_ef_whole_call_state_fails": {"indexed": nz_i, "repeated": nz_r},
        "zero_ef_whole_call_state_fails": {"indexed": z_i, "repeated": z_r},
        "input_hashes_match_in_every_cell": all_inputs_ok,
        "kill_gate_confirms_ef_defect": nz_i == Some(true) && nz_r == Some(true)
            && z_i == Some(false) && z_r == Some(false) && all_inputs_ok,
    });
    eprintln!("[chunk-oracle] first kill gate: {kill}");

    let receipt = json!({
        "tool": "qwen35_prefill_chunk_oracle",
        "schema": 1,
        "arch": arch,
        "seed": cli.seed,
        "rows": cli.rows,
        "n_heads": cli.heads,
        "qk_heads": cli.heads / cli.ratio,
        "qk_head_ratio": cli.ratio,
        "head_dim": HD,
        "graph": cli.graph,
        "splits_run": splits,
        "fixture": {
            "q_k": "L2-normalized per (token, Q/K head), F32",
            "v": "uniform [-1, 1)",
            "gate_log_decay": "[-0.3, -0.02] (alpha = expf(gate))",
            "beta": "sigmoid, [0.1, 0.9]",
            "state": "nonzero F32 S0 -> i8 codes + F32 scale (max/127, RNE) + F16 residual",
            "ef_modes": {"zero": "EF buffer present, zero entering", "nonzero": "F16 residual of S0 quantization", "none": "no EF buffer (stochastic rounding, frame base 0)"},
        },
        "input_plane_sha256": hashes,
        "tolerances": {
            "out_rel_err": OUT_TOL_REL,
            "state_lsb_ef_modes": STATE_TOL_LSB_EF,
            "state_lsb_none_whole_call": STATE_TOL_LSB_STOCH,
        },
        "f32_vs_f64_reference_floor": f32_floor_by_ef.iter()
            .map(|(k, v)| (k.to_string(), json!({"out_max_abs": v.0, "state_max_abs": v.1})))
            .collect::<Map<String, Value>>(),
        "kernels_launched_aggregate": kernel_names,
        "cells": cells,
        "input_lsb_negative_controls": controls,
        "first_kill_gate": kill,
        "summary": {"judged_cells": judged, "failed": failed},
    });
    if let Some(path) = &cli.json {
        let text = ctx(serde_json::to_string_pretty(&receipt), "serialize receipt")?;
        ctx(std::fs::write(path, text), "write receipt")?;
        eprintln!("[chunk-oracle] receipt: {path}");
    }
    free_dev(&mut gpu, dev);
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    println!(
        "qwen35_prefill_chunk_oracle: arch={arch} graph={} judged={judged} failed={failed} \
         kill_gate_confirms_ef_defect={}",
        cli.graph, receipt["first_kill_gate"]["kill_gate_confirms_ef_defect"]
    );
    Ok(if failed == 0 { 0 } else { 1 })
}

fn hipfire_env_per_token() -> bool {
    std::env::var("HIPFIRE_DN_REQUANT_PER_TOKEN")
        .map(|v| {
            let v = v.trim();
            !v.is_empty() && v != "0"
        })
        .unwrap_or(false)
}

// ═══════════════════════════ CPU-only regression tests ════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Planes {
        gen_planes(0xC0FFEE, 96, 2, 2)
    }

    fn run_model(p: &Planes, ef: &[u16], pl: Placement, split: usize) -> ModelOut {
        model(p, &to_f64(&p.q), &to_f64(&p.k), ef, pl, split)
    }

    fn state_lsb(a: &ModelOut, b: &ModelOut, p: &Planes) -> f64 {
        // LSB of the corrected model's own final row scale is not exposed by
        // `ModelOut`; use the fixed row scale of the entering state instead
        // (same order of magnitude, ~max|S|/127).
        let mut worst = 0.0f64;
        for h in 0..p.n_heads {
            for r in 0..HD {
                let lsb = p.scales[h * HD + r] as f64;
                for c in 0..HD {
                    let i = (h * HD + r) * HD + c;
                    worst = worst.max((a.recon[i] - b.recon[i]).abs() / lsb);
                }
            }
        }
        worst
    }

    #[test]
    fn fixture_is_seeded_in_range_and_plane_independent() {
        let a = gen_planes(7, 64, 4, 2);
        let b = gen_planes(7, 64, 4, 2);
        assert_eq!(plane_hashes(&a), plane_hashes(&b));
        let c = gen_planes(8, 64, 4, 2);
        assert_ne!(plane_hashes(&a)["q"], plane_hashes(&c)["q"]);
        // More rows changes V but must not change the (independent) state planes.
        let d = gen_planes(7, 65, 4, 2);
        assert_eq!(plane_hashes(&a)["q8_codes"], plane_hashes(&d)["q8_codes"]);
        assert_eq!(
            plane_hashes(&a)["f16_ef_nonzero"],
            plane_hashes(&d)["f16_ef_nonzero"]
        );
        for row in a.q.chunks(HD).chain(a.k.chunks(HD)) {
            let n: f64 = row.iter().map(|x| (*x as f64).powi(2)).sum();
            assert!((n - 1.0).abs() < 1e-6, "Q/K row not unit norm: {n}");
        }
        assert!(a.v.iter().all(|x| (-1.0..=1.0).contains(x)));
        assert!(a.gate.iter().all(|x| (-0.3..=-0.02).contains(x)));
        assert!(a.beta.iter().all(|x| (0.1..=0.9).contains(x)));
        assert!(
            a.ef_nonzero.iter().any(|e| *e != 0),
            "EF residual must be nonzero"
        );
        assert!(a.ef_zero.iter().all(|e| *e == 0));
        // Entering state is quantized to <= 0.5 LSB (+ f16 rounding) of its own scale.
        for (i, s0) in a.s0.iter().enumerate() {
            let sc = a.scales[i / HD];
            let rec = recon_f32(a.codes[i], sc, a.ef_nonzero[i]);
            assert!(
                (rec - s0).abs() <= 1e-3 * sc.max(f32::MIN_POSITIVE) + 1e-6,
                "q+ef != S0"
            );
        }
        // Repeated Q/K is the interleaved expansion of the compact plane.
        for t in 0..a.rows {
            for h in 0..a.n_heads {
                let rep = &a.q_rep[(t * a.n_heads + h) * HD..][..HD];
                let src = &a.q[(t * a.qk_heads + h / a.ratio) * HD..][..HD];
                assert_eq!(rep, src);
            }
        }
    }

    #[test]
    fn ef_at_start_is_chunk_invariant_and_ef_at_end_is_not() {
        let p = small();
        let cont = run_model(&p, &p.ef_nonzero, Placement::Continuous, p.rows);
        for split in [1usize, 7, 25, 64, 96] {
            let good = run_model(&p, &p.ef_nonzero, Placement::EfAtStart, split);
            let lsb = state_lsb(&good, &cont, &p);
            assert!(
                lsb <= STATE_TOL_LSB_EF,
                "corrected model drifts {lsb} LSB at split {split}"
            );
        }
        // The legacy placement leaves the (undecayed) residual: decisively off.
        let bad = run_model(&p, &p.ef_nonzero, Placement::EfAtEnd, p.rows);
        assert!(
            state_lsb(&bad, &cont, &p) > 10.0 * STATE_TOL_LSB_EF,
            "legacy placement must fail the nonzero-EF state check"
        );
        // With zero entering EF the whole call is identical for both placements.
        let z_old = run_model(&p, &p.ef_zero, Placement::EfAtEnd, p.rows);
        let z_new = run_model(&p, &p.ef_zero, Placement::EfAtStart, p.rows);
        assert!(z_old.recon.iter().zip(&z_new.recon).all(|(a, b)| a == b));
        let z_cont = run_model(&p, &p.ef_zero, Placement::Continuous, p.rows);
        assert!(state_lsb(&z_old, &z_cont, &p) <= STATE_TOL_LSB_EF);
        // ... and for split calls the legacy placement diverges (chunk dependence).
        let z_split = run_model(&p, &p.ef_zero, Placement::EfAtEnd, 1);
        assert!(state_lsb(&z_split, &z_cont, &p) > STATE_TOL_LSB_EF);
    }

    #[test]
    fn f32_recurrence_stays_within_noise_floor_of_f64() {
        let p = small();
        let cont = run_model(&p, &p.ef_nonzero, Placement::Continuous, p.rows);
        let (out_err, st_err) = f32_floor(&p, &p.ef_nonzero, &cont);
        assert!(out_err < 1e-4 && st_err < 1e-4, "{out_err} {st_err}");
    }

    #[test]
    fn lsb_flip_changes_exactly_the_flipped_plane_hash() {
        let base = small();
        let h0 = plane_hashes(&base);
        let mut m = base.clone();
        flip_f32(&mut m.v, 5);
        let h1 = plane_hashes(&m);
        let changed: Vec<_> = h0
            .iter()
            .filter(|(k, v)| h1[**k] != **v)
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(changed, vec!["v"]);
        let mut m = base.clone();
        m.q8_flip();
        let h2 = plane_hashes(&m);
        let changed: Vec<_> = h0
            .iter()
            .filter(|(k, v)| h2[**k] != **v)
            .map(|(k, _)| *k)
            .collect();
        assert_eq!(changed, vec!["q8_codes"]);
    }

    #[test]
    fn cli_parses_frozen_flags_and_rejects_bad_input() {
        let a = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let c = parse_cli(&a(
            "--seed 0x10 --rows 100 --splits 1,25,63 --graph 1 --json x.json",
        ))
        .unwrap();
        assert_eq!((c.seed, c.rows, c.graph), (16, 100, true));
        assert_eq!(c.splits, vec![1, 25, 63]);
        assert_eq!(c.json.as_deref(), Some("x.json"));
        assert!(parse_cli(&a("--graph 2")).is_err());
        assert!(parse_cli(&a("--splits 1,0")).is_err());
        assert!(parse_cli(&a("--rows 0")).is_err());
        assert!(parse_cli(&a("--heads 3 --qk-ratio 2")).is_err());
        assert!(parse_cli(&a("--nope")).is_err());
    }

    impl Planes {
        fn q8_flip(&mut self) {
            self.codes[0] ^= 1;
        }
    }
}
