// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the A3B shared-expert down fold (H3, gfx1201,
//! lab): `Gpu::try_gemm_mq4g256v2_shared_down_sigmoid_epi` (sigmoid of the
//! per-token shared gate fused into the final residual store), against the
//! incumbent two-launch sequence. The fold has two launch targets with the same
//! kernarg layout and bytes: the builder `_b1` bundle entry
//! `gemm_mq4g256v2_residual_mmq_iu4_full_add_sigmoid_b1` (selected when
//! `HIPFIRE_G12_IU4_ISA` holds and the shape/pointer are eligible) and the hipcc
//! V3 `full_add_v3` kernel launched with runtime mode `add == 2` (the anchor).
//!
//! ## Arms (A3B geometry: hidden M = 2048, shared intermediate K = 512)
//!
//! * incumbent (the actual shared-down stage): `Gpu::gemm_mq4g256v2(w_down,
//!   x_rot, tmp, 2048, 512, rows)` (zero `tmp`, ONE standalone
//!   `quantize_int4_mmq_ds128`, MMQ ADD) followed by
//!   `Gpu::sigmoid_scaled_residual_add_batched_f32(residual, tmp, gate, rows,
//!   2048)`.
//! * candidate: `Gpu::try_gemm_mq4g256v2_shared_down_sigmoid_epi(w_down, x_rot,
//!   None, residual, gate, 2048, 512, rows)`: one standalone quantize from the
//!   already-rotated `x_rot`, the `_b1` sigmoid ADD straight into the residual,
//!   no `tmp` memset or write. The oracle asserts the `_b1` symbol was the
//!   arm's last launch.
//! * V3 anchor: the same call with `Gpu::flags.g12_iu4_isa` cleared for the
//!   call, which selects the hipcc V3 ADD mode 2 (the pre-`_b1` selected entry);
//!   the oracle asserts the V3 symbol was its last launch. Its bytes must equal both the
//!   incumbent and the candidate, and it is timed as a third arm.
//!
//! `x_rot` (SwiGLU + FWHT, AWQ variant when AWQ is `some`) is produced once per
//! case by the same F32 producer for every arm and sits outside every timer.
//!
//! ## What is compared
//!
//! The FINAL residual bytes `[rows, 2048]` F32 of each arm pair, bit for bit
//! (signed zeros and NaN payloads included). The initial residual is a pristine
//! device buffer restored into each arm's residual before every run (outside the
//! timer) and contains `-0.0` and `+0.0` elements so the candidate's
//! `RN(+0 + acc)` zero normalisation and the final fused multiply-add are
//! visible. The gate logits are finite, random in `[-8, 8)` plus saturating
//! `-120` / `+120` entries.
//!
//! The summary carries the key `all_byte_identical` (value `true`) ONLY when
//! every case passed on the first run, again after the timing loop, and the
//! matrix finished inside the deadline. A failing, incomplete, refused or
//! unsupported run never prints that key; the first mismatch is printed as
//! `first_mismatch` instead. Byte identity is a GPU outcome: `--plan` and
//! `--self-check` never initialise HIP and never print a pass.
//!
//! ## Matrix
//!
//! rows {512, 4096, 7808, 8192} x data {zero, random} x AWQ {none, some} = 16
//! cases (7808 and 8192 are the A3B pp16000 prefill chunks).
//! `zero` activations (gate = up = 0 -> `x_rot` = 0, amax 0 / `d = 1` path);
//! the residual and gate logits stay the nonzero pattern above so the epilogue
//! is still exercised. `random` activations are finite, per-row scaled
//! `0.25 * (1 + row % 5)`. AWQ scales are positive F32 `[K]` in `[0.5, 2.0)`.
//!
//! Weights are random but valid SYMMETRIC MQ4G256V2: 136 B per 256-group, two
//! fp16 `{scale, zero}` words (zero = -8 * scale exactly, finite scale) + 128 B
//! random packed nibbles (same stride and header order as
//! `qwen4_trunk_iu4::synth_weights`). The oracle sets `Gpu::mq4v2_symmetric`
//! itself (the loader does it for real artifacts).
//!
//! ## Admission and refusal
//!
//! The oracle refuses (exit 2, error on stderr and one `error` JSON line, no
//! comparison) on any arch except exact `gfx1201`, or when
//! `Gpu::mq4g256v2_shared_down_sigmoid_epi_supported(2048, 512, rows)` is false
//! for any selected case, and when the candidate returns `Ok(false)` after the
//! support check. It never compares a fallback against the incumbent. It does
//! not read `HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI`. The V3 module is only
//! reached through the host executor's support predicate (exact gfx1201 IU4
//! prefill route, symmetric MQ4V2, `HIPFIRE_G12_IU4_ISA` or V3 enabled); the
//! incumbent must also take the standalone-quantizer IU4 route; this is checked
//! per case through the scratch generation bump, and a miss aborts as
//! unsupported.
//!
//! ## Timing
//!
//! `--reps N` (default 30, minimum 20) + warmups (default 3), incumbent and
//! candidate interleaved. Each rep: restore the residual from the pristine
//! buffer + `device_synchronize` (untimed), then `Instant::now()`, the arm's
//! whole sequence (incumbent: tmp memset + quantizer + ADD + sigmoid add;
//! candidate: quantizer + sigmoid-epilogue ADD), `device_synchronize`, elapsed.
//! Reported as median / p10 / p90 microseconds (nearest-rank percentiles on the
//! sorted samples, median = mean of the two middle samples).
//!
//! ## Deadline
//!
//! One process, at most 120 s (`--deadline-secs N`, 1..=120): cooperative
//! checks between reps/cases plus a watchdog thread that exits 3 even if a GPU
//! call hangs.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab \
//!     --example shared_down_sigmoid_epi_oracle -- --plan
//! cargo run --release -p rdna-compute --features lab \
//!     --example shared_down_sigmoid_epi_oracle -- --self-check
//! cargo run --release -p rdna-compute --features lab \
//!     --example shared_down_sigmoid_epi_oracle
//! # optional: --rows 512,4096,7808,8192 --data zero,random --awq none,some
//! #           --reps N>=20 --warmups N --deadline-secs 1..=120
//! ```
//!
//! stdout (GPU mode) = one JSON line per case then the summary line. Diagnostics
//! go to stderr. Exit 0 when every case is byte-identical, 1 on any mismatch
//! (after the summary), 2 on bad arguments / refusal / unsupported (before any
//! comparison), 3 on deadline, 4 on a HIP launch or allocation error.

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const DEFAULT_REPS: usize = 30;
const MIN_REPS: usize = 20;
const DEFAULT_WARMUPS: usize = 3;
const MAX_DEADLINE_SECS: u64 = 120;
/// A3B hidden size: output channels of the shared down projection.
const M: usize = 2048;
/// A3B shared-expert intermediate size: reduction K of the shared down.
const K: usize = 512;
const DEFAULT_ROWS: [usize; 4] = [512, 4096, 7808, 8192];
/// The incumbent IU4 route (`mq4g256v2_shared_iu4_supported`) needs rows >= 64.
const MIN_ROWS: usize = 64;
const MAX_ROWS: usize = 16384;
/// MQ4G256V2 bytes per 256-wide group: 8 B fp16 header + 128 B nibbles.
const GROUP_BYTES: usize = 136;
/// `block_i4_128` record bytes (f32 d, f32 s, 64 B nibbles).
const RECORD_BYTES: usize = 72;
/// `full_add_v3` runtime `add` argument: 1 = plain ADD, 2 = sigmoid residual.
const ADD_MODE_PLAIN: i32 = 1;
const ADD_MODE_SIGMOID: i32 = 2;
/// `full_add_v3` kernarg fields in declaration order: (name, size == align).
const V3_ABI: [(&str, usize); 8] = [
    ("A", 8),
    ("Xq", 8),
    ("Y", 8),
    ("M", 4),
    ("K", 4),
    ("N", 4),
    ("add", 4),
    ("shared_gate", 8),
];
const V3_SYMBOL: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_v3";
const B1_SIGMOID_SYMBOL: &str = "gemm_mq4g256v2_residual_mmq_iu4_full_add_sigmoid_b1";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Data {
    Zero,
    Random,
}

impl Data {
    fn name(self) -> &'static str {
        match self {
            Data::Zero => "zero",
            Data::Random => "random",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    rows: usize,
    data: Data,
    awq: bool,
}

impl Case {
    fn name(&self) -> String {
        format!(
            "shared_down_m{M}_k{K}_rows{}_{}_{}",
            self.rows,
            self.data.name(),
            if self.awq { "awq" } else { "noawq" }
        )
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Run,
    Plan,
    SelfCheck,
}

struct Args {
    mode: Mode,
    rows: Vec<usize>,
    datas: Vec<Data>,
    awqs: Vec<bool>,
    reps: usize,
    warmups: usize,
    deadline_secs: u64,
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut a = Args {
        mode: Mode::Run,
        rows: DEFAULT_ROWS.to_vec(),
        datas: vec![Data::Zero, Data::Random],
        awqs: vec![false, true],
        reps: DEFAULT_REPS,
        warmups: DEFAULT_WARMUPS,
        deadline_secs: MAX_DEADLINE_SECS,
    };
    let mut it = raw.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String, String> {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "--plan" => a.mode = Mode::Plan,
            "--self-check" => a.mode = Mode::SelfCheck,
            "--rows" => {
                a.rows = value("--rows")?
                    .split(',')
                    .map(|s| match s.parse::<usize>() {
                        Ok(r) if (MIN_ROWS..=MAX_ROWS).contains(&r) => Ok(r),
                        _ => Err(format!(
                            "rows {s:?} not an integer in {MIN_ROWS}..={MAX_ROWS}"
                        )),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--data" => {
                a.datas = value("--data")?
                    .split(',')
                    .map(|s| match s {
                        "zero" => Ok(Data::Zero),
                        "random" => Ok(Data::Random),
                        _ => Err(format!("data {s:?} not zero|random")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--awq" => {
                a.awqs = value("--awq")?
                    .split(',')
                    .map(|s| match s {
                        "none" => Ok(false),
                        "some" => Ok(true),
                        _ => Err(format!("awq {s:?} not none|some")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--reps" => {
                a.reps = value("--reps")?
                    .parse()
                    .map_err(|e| format!("--reps: {e}"))?;
                if a.reps < MIN_REPS {
                    return Err(format!("--reps must be >= {MIN_REPS}"));
                }
            }
            "--warmups" => {
                a.warmups = value("--warmups")?
                    .parse()
                    .map_err(|e| format!("--warmups: {e}"))?;
            }
            "--deadline-secs" => {
                a.deadline_secs = value("--deadline-secs")?
                    .parse()
                    .map_err(|e| format!("--deadline-secs: {e}"))?;
                if a.deadline_secs == 0 || a.deadline_secs > MAX_DEADLINE_SECS {
                    return Err(format!(
                        "--deadline-secs must be in 1..={MAX_DEADLINE_SECS}"
                    ));
                }
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if a.rows.is_empty() || a.datas.is_empty() || a.awqs.is_empty() {
        return Err("empty --rows / --data / --awq list".into());
    }
    Ok(a)
}

fn case_matrix(a: &Args) -> Vec<Case> {
    let mut out = Vec::new();
    for &rows in &a.rows {
        for &data in &a.datas {
            for &awq in &a.awqs {
                out.push(Case { rows, data, awq });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU-side admission / dimension / ABI logic (covered by --self-check)
// ---------------------------------------------------------------------------

/// Shape rules this oracle requires before it will touch a GPU buffer.
fn validate_dims(m: usize, k: usize, rows: usize) -> Result<(), String> {
    if m == 0 || k == 0 || rows == 0 {
        return Err(format!("zero dimension m={m} k={k} rows={rows}"));
    }
    if k % 256 != 0 {
        return Err(format!("k={k} is not a multiple of 256 (MQ4G256V2 group)"));
    }
    if rows < MIN_ROWS {
        return Err(format!(
            "rows={rows} < {MIN_ROWS}: the incumbent IU4 route is not taken below {MIN_ROWS} rows"
        ));
    }
    let elems = rows
        .checked_mul(m)
        .ok_or_else(|| "rows * m overflows".to_string())?;
    if elems.checked_mul(4).map_or(true, |b| b > u32::MAX as usize) {
        return Err(format!("residual bytes {elems} * 4 exceed 32-bit offsets"));
    }
    if rdna_compute::scratch::int4_mmq_reservation_bytes(k, rows).is_none() {
        return Err(format!("no IU4 sidecar reservation for k={k} rows={rows}"));
    }
    Ok(())
}

/// Why the oracle must refuse to compare, if it must. `supported` is the
/// executor's `mq4g256v2_shared_down_sigmoid_epi_supported(m, k, rows)`.
fn refuse_reason(arch: &str, supported: bool) -> Option<String> {
    if arch != "gfx1201" {
        return Some(format!("arch {arch} is not exact gfx1201"));
    }
    if !supported {
        return Some("mq4g256v2_shared_down_sigmoid_epi_supported is false".into());
    }
    None
}

/// Natural-alignment offsets of the `full_add_v3` kernargs and the padded total.
fn abi_layout() -> (Vec<(&'static str, usize)>, usize) {
    let mut off = 0usize;
    let mut out = Vec::new();
    let mut align = 1usize;
    for (name, size) in V3_ABI {
        off = off.next_multiple_of(size);
        out.push((name, off));
        off += size;
        align = align.max(size);
    }
    (out, off.next_multiple_of(align))
}

/// CPU model of the unfused tail: `tmp = RN(0 + acc)` (the zeroed temp after
/// ADD), then `y = fma(sig, tmp, y)` (contracted `y += sig * tmp`).
fn cpu_unfused(old: f32, acc: f32, sig: f32) -> f32 {
    let tmp = 0.0f32 + acc;
    sig.mul_add(tmp, old)
}

/// CPU model of the fused epilogue: `down = 0 + acc; fmaf(sig, down, old)`.
fn cpu_fused(old: f32, acc: f32, sig: f32) -> f32 {
    let down = 0.0f32 + acc;
    sig.mul_add(down, old)
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 24
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xff_ffff) as f32 / 16_777_216.0
    }
}

/// Random valid SYMMETRIC MQ4G256V2 rows: `m` rows of `k / 256` groups of
/// 136 B; per half `{scale f16, zero f16 = -8 * scale}` then 128 B nibbles.
fn synth_mq4g256v2_sym(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = f16::from_f32(0.004 + (rng.next() & 0x3f) as f32 * 2.0e-4);
            let zero = f16::from_f32(-8.0 * scale.to_f32());
            out.extend_from_slice(&scale.to_bits().to_le_bytes());
            out.extend_from_slice(&zero.to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    assert_eq!(out.len(), m * (k / 256) * GROUP_BYTES);
    out
}

/// SwiGLU inputs `(gate, up)` `[rows, k]`: all zero for [`Data::Zero`], else
/// finite gate in `[-4, 4)` and up in `[-1, 1)`, both times `0.25 * (1 + row % 5)`.
fn synth_gate_up(rows: usize, k: usize, seed: u64, data: Data) -> (Vec<f32>, Vec<f32>) {
    if data == Data::Zero {
        return (vec![0.0; rows * k], vec![0.0; rows * k]);
    }
    let mut rng = Rng(seed);
    let mut gate = Vec::with_capacity(rows * k);
    let mut up = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..k {
            gate.push((rng.unit() * 8.0 - 4.0) * scale);
            up.push((rng.unit() * 2.0 - 1.0) * scale);
        }
    }
    (gate, up)
}

/// Positive F32 vector in `[lo, lo + span)` (AWQ scales).
fn synth_vec(len: usize, seed: u64, lo: f32, span: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len).map(|_| lo + rng.unit() * span).collect()
}

/// Initial residual `[rows, m]` in `[-2, 2)` with `-0.0` (every 97th, phase 11)
/// and `+0.0` (phase 53) elements.
fn synth_residual(rows: usize, m: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows * m)
        .map(|i| match i % 97 {
            11 => -0.0,
            53 => 0.0,
            _ => rng.unit() * 4.0 - 2.0,
        })
        .collect()
}

/// Pre-sigmoid shared-gate logits `[rows]` in `[-8, 8)` with saturating
/// `-120` (i % 211 == 5) and `+120` (i % 223 == 7) entries and a zero at 0.
fn synth_logits(rows: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..rows)
        .map(|i| {
            if i == 0 {
                0.0
            } else if i % 211 == 5 {
                -120.0
            } else if i % 223 == 7 {
                120.0
            } else {
                rng.unit() * 16.0 - 8.0
            }
        })
        .collect()
}

fn case_seed(c: &Case) -> u64 {
    0x5d51_9000 ^ ((c.rows as u64) << 8)
}

// ---------------------------------------------------------------------------
// Compare / stats helpers (CPU, covered by --self-check)
// ---------------------------------------------------------------------------

fn out_locate(byte: usize, m: usize) -> Value {
    let elem = byte / 4;
    json!({
        "byte": byte,
        "elem": elem,
        "token": elem / m,
        "channel": elem % m,
        "byte_in_f32": byte % 4,
    })
}

struct Cmp {
    identical: bool,
    json: Value,
    first: Option<Value>,
}

/// Bitwise compare of two `[rows, m]` F32 residual byte buffers.
fn compare_bytes(inc: &[u8], cand: &[u8], m: usize) -> Cmp {
    if inc.len() != cand.len() {
        let first = json!({ "length_mismatch": [inc.len(), cand.len()] });
        return Cmp {
            identical: false,
            json: json!({ "identical": false, "first_mismatch": first.clone() }),
            first: Some(first),
        };
    }
    let first_byte = inc.iter().zip(cand.iter()).position(|(a, b)| a != b);
    let bad_elems = inc
        .chunks_exact(4)
        .zip(cand.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    let first = first_byte.map(|byte| {
        let e = byte / 4 * 4;
        let ib = u32::from_le_bytes(inc[e..e + 4].try_into().unwrap());
        let cb = u32::from_le_bytes(cand[e..e + 4].try_into().unwrap());
        let mut v = out_locate(byte, m);
        v["incumbent_bits"] = json!(format!("0x{ib:08x}"));
        v["candidate_bits"] = json!(format!("0x{cb:08x}"));
        v["incumbent"] = json!(f32::from_bits(ib));
        v["candidate"] = json!(f32::from_bits(cb));
        v
    });
    let identical = first.is_none();
    Cmp {
        identical,
        json: json!({
            "identical": identical,
            "bytes": inc.len(),
            "mismatching_f32": bad_elems,
            "first_mismatch": first.clone(),
        }),
        first,
    }
}

fn stats(us: &[f64]) -> Value {
    let mut us = us.to_vec();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = us.len();
    let pct = |p: f64| us[((n - 1) as f64 * p).round() as usize];
    let median = (us[(n - 1) / 2] + us[n / 2]) / 2.0;
    json!({ "median": median, "p10": pct(0.1), "p90": pct(0.9), "samples": n })
}

// ---------------------------------------------------------------------------
// GPU fixture and arms
// ---------------------------------------------------------------------------

enum Fail {
    /// Refusal / unsupported: exit 2, never a comparison.
    Unsupported(String),
    /// HIP launch / allocation error: exit 4.
    Launch(String),
}

fn hip<T, E: std::fmt::Debug>(r: Result<T, E>, what: &str) -> Result<T, Fail> {
    r.map_err(|e| Fail::Launch(format!("{what}: {e:?}")))
}

fn sync(gpu: &Gpu) -> Result<(), Fail> {
    hip(gpu.hip.device_synchronize(), "device_synchronize")
}

struct Fx<'a> {
    rows: usize,
    w: &'a GpuTensor,
    awq: Option<&'a GpuTensor>,
    gate: &'a GpuTensor,
    up: &'a GpuTensor,
    x_rot: &'a GpuTensor,
    logits: &'a GpuTensor,
    tmp: &'a GpuTensor,
}

struct Bufs {
    /// Pristine initial residual.
    res0: GpuTensor,
    inc: GpuTensor,
    cand: GpuTensor,
    v3: GpuTensor,
}

fn reset(gpu: &Gpu, dst: &GpuTensor, src: &GpuTensor) -> Result<(), Fail> {
    hip(
        gpu.hip.memcpy_dtod(&dst.buf, &src.buf, dst.byte_size()),
        "restore residual",
    )
}

/// Common F32 producer of every arm (writes `x_rot`).
fn produce_x_rot(gpu: &mut Gpu, f: &Fx) -> Result<(), Fail> {
    match f.awq {
        Some(a) => hip(
            gpu.fused_silu_mul_rotate_mq_awq_batched(f.gate, f.up, a, f.x_rot, K, f.rows),
            "fused_silu_mul_rotate_mq_awq_batched",
        ),
        None => hip(
            gpu.fused_silu_mul_rotate_mq_batched(f.gate, f.up, f.x_rot, K, f.rows),
            "fused_silu_mul_rotate_mq_batched",
        ),
    }
}

fn sigmoid_add(gpu: &mut Gpu, f: &Fx, residual: &GpuTensor) -> Result<(), Fail> {
    hip(
        gpu.sigmoid_scaled_residual_add_batched_f32(residual, f.tmp, f.logits, f.rows, M),
        "sigmoid_scaled_residual_add_batched_f32",
    )
}

/// Incumbent: zeroed F32 GEMM output, then the gated residual add.
fn run_incumbent(gpu: &mut Gpu, f: &Fx, residual: &GpuTensor) -> Result<(), Fail> {
    hip(
        gpu.gemm_mq4g256v2(f.w, f.x_rot, f.tmp, M, K, f.rows),
        "gemm_mq4g256v2",
    )?;
    sigmoid_add(gpu, f, residual)
}

fn run_candidate(gpu: &mut Gpu, f: &Fx, residual: &GpuTensor) -> Result<(), Fail> {
    match gpu.try_gemm_mq4g256v2_shared_down_sigmoid_epi(
        f.w, f.x_rot, None, residual, f.logits, M, K, f.rows,
    ) {
        Ok(true) => Ok(()),
        Ok(false) => Err(Fail::Unsupported(
            "try_gemm_mq4g256v2_shared_down_sigmoid_epi returned false although \
             mq4g256v2_shared_down_sigmoid_epi_supported held; refusing to compare a fallback"
                .into(),
        )),
        Err(e) => Err(Fail::Launch(format!(
            "try_gemm_mq4g256v2_shared_down_sigmoid_epi: {e:?}"
        ))),
    }
}

/// The candidate launch with the builder bundle switched off, which selects the
/// hipcc V3 ADD mode 2; the flag is restored on every path.
fn run_v3_anchor(gpu: &mut Gpu, f: &Fx, residual: &GpuTensor) -> Result<(), Fail> {
    let isa = gpu.flags.g12_iu4_isa;
    std::sync::Arc::make_mut(&mut gpu.flags).g12_iu4_isa = false;
    let r = run_candidate(gpu, f, residual);
    std::sync::Arc::make_mut(&mut gpu.flags).g12_iu4_isa = isa;
    r
}

/// Fail unless the arm's final launch was `symbol` (the arm really ran the entry it claims).
fn require_last_launch(gpu: &Gpu, symbol: &str, arm: &str) -> Result<(), Fail> {
    match gpu.last_launched_kernel() {
        Some(last) if last == symbol => Ok(()),
        last => Err(Fail::Unsupported(format!(
            "{arm}: last launch was {last:?}, expected {symbol}; the arm did not take its intended route"
        ))),
    }
}

struct Pass {
    json: Value,
    identical: bool,
    first: Option<Value>,
}

/// One verification pass: incumbent vs candidate over one standalone quantize each.
fn verify_pass(gpu: &mut Gpu, f: &Fx, b: &Bufs) -> Result<Pass, Fail> {
    // 1. incumbent vs candidate(None): each quantizes from x_rot itself.
    reset(gpu, &b.inc, &b.res0)?;
    reset(gpu, &b.cand, &b.res0)?;
    sync(gpu)?;
    let gen0 = gpu.scratch.int4_mmq_live().0;
    run_incumbent(gpu, f, &b.inc)?;
    sync(gpu)?;
    if gpu.scratch.int4_mmq_live().0 == gen0 {
        return Err(Fail::Unsupported(
            "the incumbent gemm_mq4g256v2 did not take the standalone-quantizer IU4 route \
             (scratch generation unchanged)"
                .into(),
        ));
    }
    run_candidate(gpu, f, &b.cand)?;
    sync(gpu)?;
    require_last_launch(gpu, B1_SIGMOID_SYMBOL, "candidate")?;
    reset(gpu, &b.v3, &b.res0)?;
    sync(gpu)?;
    run_v3_anchor(gpu, f, &b.v3)?;
    sync(gpu)?;
    require_last_launch(gpu, V3_SYMBOL, "v3 anchor")?;
    let inc = hip(
        gpu.download_raw_bytes(&b.inc),
        "download incumbent residual",
    )?;
    let cand = hip(
        gpu.download_raw_bytes(&b.cand),
        "download candidate residual",
    )?;
    let v3 = hip(gpu.download_raw_bytes(&b.v3), "download v3 anchor residual")?;
    let quant = compare_bytes(&inc, &cand, M);
    let v3_vs_inc = compare_bytes(&inc, &v3, M);
    let v3_vs_cand = compare_bytes(&cand, &v3, M);

    let identical = quant.identical && v3_vs_inc.identical && v3_vs_cand.identical;
    let first = quant
        .first
        .clone()
        .map(|m| json!({ "arm": "standalone_quantize", "mismatch": m }))
        .or_else(|| {
            v3_vs_inc
                .first
                .clone()
                .map(|m| json!({ "arm": "v3_anchor_vs_incumbent", "mismatch": m }))
        })
        .or_else(|| {
            v3_vs_cand
                .first
                .clone()
                .map(|m| json!({ "arm": "v3_anchor_vs_candidate", "mismatch": m }))
        });
    Ok(Pass {
        json: json!({
            "standalone_quantize": quant.json,
            "v3_anchor_vs_incumbent": v3_vs_inc.json,
            "v3_anchor_vs_candidate": v3_vs_cand.json,
        }),
        identical,
        first,
    })
}

struct Timing {
    inc: Vec<f64>,
    cand: Vec<f64>,
    v3: Vec<f64>,
    deadline_hit: bool,
}

fn time_arms(
    gpu: &mut Gpu,
    f: &Fx,
    b: &Bufs,
    reps: usize,
    warmups: usize,
    deadline: Instant,
) -> Result<Timing, Fail> {
    let mut t = Timing {
        inc: Vec::with_capacity(reps),
        cand: Vec::with_capacity(reps),
        v3: Vec::with_capacity(reps),
        deadline_hit: false,
    };
    for rep in 0..warmups + reps {
        if Instant::now() >= deadline {
            t.deadline_hit = true;
            break;
        }
        reset(gpu, &b.inc, &b.res0)?;
        sync(gpu)?;
        let t0 = Instant::now();
        run_incumbent(gpu, f, &b.inc)?;
        sync(gpu)?;
        let inc_us = t0.elapsed().as_secs_f64() * 1e6;

        reset(gpu, &b.cand, &b.res0)?;
        sync(gpu)?;
        let t0 = Instant::now();
        run_candidate(gpu, f, &b.cand)?;
        sync(gpu)?;
        let cand_us = t0.elapsed().as_secs_f64() * 1e6;

        reset(gpu, &b.v3, &b.res0)?;
        sync(gpu)?;
        let t0 = Instant::now();
        run_v3_anchor(gpu, f, &b.v3)?;
        sync(gpu)?;
        let v3_us = t0.elapsed().as_secs_f64() * 1e6;

        if rep >= warmups {
            t.inc.push(inc_us);
            t.cand.push(cand_us);
            t.v3.push(v3_us);
        }
    }
    if t.inc.len() < reps {
        t.deadline_hit = true;
    }
    Ok(t)
}

struct CaseOut {
    line: Value,
    identical: bool,
    first: Option<Value>,
    deadline_hit: bool,
}

struct Shared {
    w: GpuTensor,
    awq: GpuTensor,
}

fn run_case(
    gpu: &mut Gpu,
    case: &Case,
    sh: &Shared,
    a: &Args,
    deadline: Instant,
) -> Result<CaseOut, Fail> {
    let rows = case.rows;
    let seed = case_seed(case);
    let (gate_h, up_h) = synth_gate_up(rows, K, seed ^ 0x11, case.data);
    let gate = hip(gpu.upload_f32(&gate_h, &[rows, K]), "upload gate")?;
    let up = hip(gpu.upload_f32(&up_h, &[rows, K]), "upload up")?;
    let x_rot = hip(gpu.zeros(&[rows, K], DType::F32), "alloc x_rot")?;
    let logits = hip(
        gpu.upload_f32(&synth_logits(rows, seed ^ 0x22), &[rows]),
        "upload logits",
    )?;
    let res0 = hip(
        gpu.upload_f32(&synth_residual(rows, M, seed ^ 0x33), &[rows, M]),
        "upload residual",
    )?;
    let tmp = hip(gpu.alloc_tensor(&[rows, M], DType::F32), "alloc tmp")?;
    let bufs = Bufs {
        res0,
        inc: hip(gpu.alloc_tensor(&[rows, M], DType::F32), "alloc residual")?,
        cand: hip(gpu.alloc_tensor(&[rows, M], DType::F32), "alloc residual")?,
        v3: hip(gpu.alloc_tensor(&[rows, M], DType::F32), "alloc residual")?,
    };
    let fx = Fx {
        rows,
        w: &sh.w,
        awq: case.awq.then_some(&sh.awq),
        gate: &gate,
        up: &up,
        x_rot: &x_rot,
        logits: &logits,
        tmp: &tmp,
    };
    produce_x_rot(gpu, &fx)?;
    sync(gpu)?;

    let first = verify_pass(gpu, &fx, &bufs)?;
    let timing = time_arms(gpu, &fx, &bufs, a.reps, a.warmups, deadline)?;
    let post = verify_pass(gpu, &fx, &bufs)?;

    let identical = first.identical && post.identical && !timing.deadline_hit;
    let first_mismatch = first.first.clone().or_else(|| post.first.clone());
    let timing_json = if timing.inc.len() >= MIN_REPS
        && timing.cand.len() >= MIN_REPS
        && timing.v3.len() >= MIN_REPS
    {
        let inc = stats(&timing.inc);
        let cand = stats(&timing.cand);
        let v3 = stats(&timing.v3);
        let speedup = inc["median"].as_f64().unwrap() / cand["median"].as_f64().unwrap();
        let v3_speedup = v3["median"].as_f64().unwrap() / cand["median"].as_f64().unwrap();
        json!({
            "incumbent_us": inc,
            "candidate_us": cand,
            "v3_anchor_us": v3,
            "candidate_speedup_vs_v3_median": v3_speedup,
            "speedup_median": speedup,
            "saving_vs_incumbent_pct": (1.0 - 1.0 / speedup) * 100.0,
        })
    } else {
        json!({ "incomplete": true, "samples": timing.inc.len() })
    };
    let line = json!({
        "case": case.name(),
        "m": M, "k": K, "rows": rows,
        "data": case.data.name(), "awq": case.awq,
        "residual_bytes": rows * M * 4,
        "sidecar_bytes": (K / 128) * rows * RECORD_BYTES,
        "reps": a.reps, "warmups": a.warmups,
        "first_run": first.json,
        "post_timing_run": post.json,
        "timing": timing_json,
        "deadline_exceeded": timing.deadline_hit,
        "case_byte_identical": identical,
    });

    for t in [
        gate, up, x_rot, logits, tmp, bufs.res0, bufs.inc, bufs.cand, bufs.v3,
    ] {
        let _ = gpu.free_tensor(t);
    }
    Ok(CaseOut {
        line,
        identical,
        first: first_mismatch,
        deadline_hit: timing.deadline_hit,
    })
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

fn incumbent_sequence() -> Vec<&'static str> {
    vec![
        "fused_silu_mul_rotate_mq[_awq]_batched (common producer, untimed): F32 x_rot",
        "gemm_mq4g256v2 (memset tmp 0, standalone quantize_int4_mmq_ds128, MMQ ADD mode 1 into tmp)",
        "sigmoid_scaled_residual_add_batched_f32 (residual += sigmoid(logit[t]) * tmp)",
    ]
}

fn candidate_sequence() -> Vec<&'static str> {
    vec![
        "same F32 x_rot (common producer, untimed)",
        "try_gemm_mq4g256v2_shared_down_sigmoid_epi(prepared = None): one standalone quantize from x_rot, the _b1 full_add_sigmoid_b1 ADD straight into the residual; no tmp memset, no intermediate write",
        "V3 anchor (same call, Gpu::flags.g12_iu4_isa cleared): one standalone quantize from x_rot, V3 full_add_v3 ADD mode 2 straight into the residual",
    ]
}

fn prerequisites() -> Vec<&'static str> {
    vec![
        "exact gfx1201 with WMMA (any other arch: exit 2, nothing compared)",
        "Gpu::mq4g256v2_shared_down_sigmoid_epi_supported(2048, 512, rows) true for every selected rows (IU4 prefill, symmetric MQ4V2, V3 route, no slab sidecars); false is an error, never a pass",
        "the _b1 sigmoid entry must be the launch target of the candidate and the V3 module that of the anchor (support inherits IU4 prefill + symmetric MQ4V2 + HIPFIRE_G12_IU4_ISA or HIPFIRE_G12_IU4_V3); the oracle only sets Gpu::mq4v2_symmetric = true for its symmetric synthetic weights",
        "the incumbent gemm_mq4g256v2 must bump the IU4 scratch generation (standalone-quantizer route); checked per case",
        "one GPU, single process, <= 120 s; weights are synthetic valid symmetric MQ4G256V2, not model weights",
    ]
}

fn limitations() -> Vec<&'static str> {
    vec![
        "launcher-level bytes only: does not read HIPFIRE_MOE_SHARED_DOWN_SIGMOID_EPI and does not exercise PrefillRoute::MoeSharedDownSigmoidEpi or the recording.rs/dispatch wiring",
        "only the prepared = None candidate (one standalone quantize from the rotated x) is compared; Some(token) consumption, stale-token refusal and the calibration tap are not exercised here",
        "sigmoid uses the GPU __expf; the CPU modes model only the fused-multiply-add / signed-zero structure, never GPU bytes",
        "zero data keeps the residual and gate logits nonzero so the epilogue still runs on an all-zero product",
        "timing includes host launch overhead with device_synchronize bracketing, not isolated kernel time",
    ]
}

fn plan(a: &Args) {
    let cases = case_matrix(a);
    let (layout, total) = abi_layout();
    println!(
        "{}",
        json!({
            "plan": true,
            "oracle": "shared_down_sigmoid_epi",
            "gpu_initialized": false,
            "byte_result": null,
            "note": "CPU plan only: no GPU was touched and nothing was compared; byte identity is a GPU outcome",
            "m": M, "k": K,
            "cases": cases.len(),
            "reps": a.reps, "min_reps": MIN_REPS, "warmups": a.warmups,
            "deadline_secs": a.deadline_secs,
            "runtime_mode": {
                "kernel": V3_SYMBOL,
                "add_plain": ADD_MODE_PLAIN,
                "add_sigmoid_epi": ADD_MODE_SIGMOID,
                "kernarg_offsets": layout.iter().map(|(n, o)| json!({"field": n, "offset": o})).collect::<Vec<_>>(),
                "kernarg_bytes": total,
            },
            "prerequisites": prerequisites(),
            "limitations": limitations(),
        })
    );
    for c in &cases {
        let dims = validate_dims(M, K, c.rows);
        println!(
            "{}",
            json!({
                "plan_case": c.name(),
                "m": M, "k": K, "rows": c.rows,
                "data": c.data.name(), "awq": c.awq,
                "dims_ok": dims.is_ok(),
                "dims_error": dims.err(),
                "residual_bytes": c.rows * M * 4,
                "tmp_bytes_incumbent_only": c.rows * M * 4,
                "sidecar_bytes": (K / 128) * c.rows * RECORD_BYTES,
                "incumbent_sequence": incumbent_sequence(),
                "candidate_sequence": candidate_sequence(),
            })
        );
    }
    println!(
        "{}",
        json!({ "plan_summary": true, "cases": cases.len(), "gpu_initialized": false })
    );
}

fn self_check() {
    let mut checks = 0usize;
    let mut failures: Vec<String> = Vec::new();
    macro_rules! check {
        ($cond:expr, $($msg:tt)*) => {{
            checks += 1;
            if !$cond {
                failures.push(format!($($msg)*));
            }
        }};
    }

    // Case matrix.
    let all = parse_args(&[]).expect("default args");
    let cases = case_matrix(&all);
    check!(
        cases.len() == 4 * 2 * 2,
        "default matrix has {} cases",
        cases.len()
    );
    check!(
        all.rows == vec![512, 4096, 7808, 8192],
        "default rows {:?}",
        all.rows
    );
    let mut names: Vec<String> = cases.iter().map(Case::name).collect();
    names.sort();
    names.dedup();
    check!(names.len() == cases.len(), "case names are not unique");
    for c in &cases {
        check!(validate_dims(M, K, c.rows).is_ok(), "{}: dims", c.name());
        check!(
            rdna_compute::scratch::int4_mmq_reservation_bytes(K, c.rows)
                == Some((K / 128) * c.rows * RECORD_BYTES),
            "{}: reservation bytes",
            c.name()
        );
    }
    check!(
        M == 2048 && K == 512,
        "A3B geometry hidden 2048 / shared intermediate 512"
    );

    // Dimension / admission logic.
    check!(validate_dims(2048, 512, 8192).is_ok(), "A3B shape accepted");
    check!(validate_dims(2048, 500, 512).is_err(), "k % 256 rejected");
    check!(validate_dims(2048, 512, 63).is_err(), "rows < 64 rejected");
    check!(validate_dims(2048, 512, 0).is_err(), "rows 0 rejected");
    check!(validate_dims(0, 512, 512).is_err(), "m 0 rejected");
    check!(validate_dims(2048, 0, 512).is_err(), "k 0 rejected");
    check!(
        validate_dims(1 << 20, 512, 8192).is_err(),
        "32-bit residual offsets enforced"
    );
    check!(
        refuse_reason("gfx1100", true).is_some(),
        "non-gfx1201 refused"
    );
    check!(
        refuse_reason("gfx1201", false).is_some(),
        "unsupported refused"
    );
    check!(
        refuse_reason("gfx1201", true).is_none(),
        "gfx1201 + supported admitted"
    );

    // Runtime mode / kernarg ABI.
    check!(ADD_MODE_PLAIN == 1 && ADD_MODE_SIGMOID == 2, "add modes");
    let (layout, total) = abi_layout();
    let off = |n: &str| layout.iter().find(|(f, _)| *f == n).map(|(_, o)| *o);
    check!(
        off("A") == Some(0) && off("Xq") == Some(8) && off("Y") == Some(16),
        "pointer offsets {layout:?}"
    );
    check!(
        off("M") == Some(24)
            && off("K") == Some(28)
            && off("N") == Some(32)
            && off("add") == Some(36),
        "scalar offsets {layout:?}"
    );
    check!(
        off("shared_gate") == Some(40),
        "shared_gate at offset 40: {layout:?}"
    );
    check!(total == 48, "kernarg block is 48 bytes, got {total}");

    // Epilogue arithmetic structure (CPU model; the GPU computes the sigmoid).
    check!(
        (0.0f32 + -0.0f32).to_bits() == 0 && cpu_fused(-0.0, -0.0, 0.5).to_bits() == 0,
        "RN(+0 + -0) is +0 and a -0 residual absorbs to +0"
    );
    check!(
        (-0.0f32).to_bits() != 0.0f32.to_bits()
            && cpu_fused(-0.0, -0.0, 0.5).to_bits() != (-0.0f32).to_bits(),
        "signed zeros differ bytewise"
    );
    let mut rng = Rng(0xfeed);
    let (mut split, mut tail_equal) = (0usize, true);
    for _ in 0..4096 {
        let old = rng.unit() * 4.0 - 2.0;
        let acc = rng.unit() * 2.0 - 1.0;
        let sig = 1.0 / (1.0 + (-(rng.unit() * 16.0 - 8.0)).exp());
        if cpu_unfused(old, acc, sig).to_bits() != cpu_fused(old, acc, sig).to_bits() {
            tail_equal = false;
        }
        if (old + sig * (0.0f32 + acc)).to_bits() != cpu_fused(old, acc, sig).to_bits() {
            split += 1;
        }
    }
    check!(
        tail_equal,
        "unfused tail model and fused epilogue model agree bitwise"
    );
    check!(
        split > 0,
        "a separately rounded multiply+add is distinguishable from the fused FMA"
    );

    // Synthetic data.
    let wb = synth_mq4g256v2_sym(3, 512, 1);
    check!(wb.len() == 3 * 2 * GROUP_BYTES, "synth weight size");
    check!(
        wb == synth_mq4g256v2_sym(3, 512, 1),
        "synth weight determinism"
    );
    check!(wb != synth_mq4g256v2_sym(3, 512, 2), "synth weight seeds");
    let sym = wb.chunks_exact(GROUP_BYTES).all(|g| {
        (0..2).all(|h| {
            let sc = f16::from_bits(u16::from_le_bytes([g[4 * h], g[4 * h + 1]]));
            let zp = f16::from_bits(u16::from_le_bytes([g[4 * h + 2], g[4 * h + 3]]));
            sc.is_finite() && sc.to_f32() > 0.0 && zp.to_f32() == -8.0 * sc.to_f32()
        })
    });
    check!(
        sym,
        "every group half is symmetric: zero == -8 * scale, finite scale"
    );
    let (g0, u0) = synth_gate_up(4, 8, 1, Data::Zero);
    let (g1, u1) = synth_gate_up(4, 8, 1, Data::Random);
    check!(
        g0.iter().chain(u0.iter()).all(|v| *v == 0.0)
            && g1.iter().chain(u1.iter()).all(|v| v.is_finite())
            && g1.iter().any(|v| *v != 0.0)
            && g1 != u1,
        "synth_gate_up data modes"
    );
    check!(
        synth_gate_up(4, 8, 1, Data::Random) == (g1, u1),
        "synth_gate_up determinism"
    );
    let awq = synth_vec(512, 3, 0.5, 1.5);
    check!(
        awq.iter().all(|v| (0.5..2.0).contains(v)),
        "awq scales in [0.5, 2)"
    );
    let res = synth_residual(64, M, 9);
    check!(
        res.len() == 64 * M && res.iter().all(|v| v.is_finite()),
        "residual shape/finite"
    );
    check!(
        res.iter().any(|v| v.to_bits() == (-0.0f32).to_bits())
            && res.iter().any(|v| v.to_bits() == 0.0f32.to_bits())
            && res.iter().any(|v| *v != 0.0),
        "residual carries -0, +0 and nonzero values"
    );
    check!(
        res == synth_residual(64, M, 9) && res != synth_residual(64, M, 10),
        "residual determinism / seeds"
    );
    let lg = synth_logits(512, 5);
    check!(
        lg.len() == 512
            && lg.iter().all(|v| v.is_finite())
            && lg.contains(&-120.0)
            && lg.contains(&120.0)
            && lg[0] == 0.0
            && lg.iter().any(|v| v.abs() < 8.0 && *v != 0.0),
        "logits: finite, saturating entries, zero at 0"
    );

    // Compare / locate / stats helpers.
    let a: Vec<u8> = 0.0f32
        .to_le_bytes()
        .into_iter()
        .chain(1.0f32.to_le_bytes())
        .collect();
    check!(
        compare_bytes(&a, &a.clone(), 2).identical,
        "identical buffers compare equal"
    );
    let mut b = a.clone();
    b[6] ^= 0x10;
    let c = compare_bytes(&a, &b, 2);
    let f = c.first.clone().unwrap_or(Value::Null);
    check!(
        !c.identical
            && f["byte"] == 6
            && f["elem"] == 1
            && f["token"] == 0
            && f["channel"] == 1
            && f["byte_in_f32"] == 2,
        "locate first mismatch: {f}"
    );
    check!(c.json["mismatching_f32"] == 1, "mismatch count");
    let nz: Vec<u8> = (-0.0f32).to_le_bytes().to_vec();
    let pz: Vec<u8> = 0.0f32.to_le_bytes().to_vec();
    check!(
        !compare_bytes(&nz, &pz, 1).identical,
        "-0 vs +0 is a byte mismatch"
    );
    check!(
        !compare_bytes(&[1, 2, 3, 4], &[1, 2, 3, 4, 5, 6, 7, 8], 1).identical,
        "length mismatch"
    );
    let l = out_locate(4 * (3 * 2048 + 17) + 2, 2048);
    check!(
        l["token"] == 3 && l["channel"] == 17 && l["byte_in_f32"] == 2,
        "out_locate {l}"
    );
    let st = stats(&[7.0, 1.0, 11.0, 3.0, 9.0, 2.0, 10.0, 4.0, 6.0, 8.0, 5.0]);
    check!(
        st["median"] == 6.0 && st["p10"] == 2.0 && st["p90"] == 10.0 && st["samples"] == 11,
        "stats {st}"
    );

    // Arg parsing.
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    check!(
        parse_args(&s(&["--reps", "19"])).is_err(),
        "reps < 20 rejected"
    );
    check!(
        parse_args(&s(&["--reps", "20"])).is_ok(),
        "reps 20 accepted"
    );
    check!(
        parse_args(&s(&["--deadline-secs", "121"])).is_err(),
        "deadline > 120 rejected"
    );
    check!(
        parse_args(&s(&["--deadline-secs", "0"])).is_err(),
        "deadline 0 rejected"
    );
    check!(
        parse_args(&s(&["--rows", "63"])).is_err(),
        "rows < 64 rejected"
    );
    check!(
        parse_args(&s(&["--rows", "x"])).is_err(),
        "rows non-numeric rejected"
    );
    check!(
        parse_args(&s(&["--data", "ones"])).is_err(),
        "bad data rejected"
    );
    check!(
        parse_args(&s(&["--awq", "all"])).is_err(),
        "bad awq rejected"
    );
    check!(
        parse_args(&s(&["--bogus"])).is_err(),
        "unknown arg rejected"
    );
    check!(
        parse_args(&s(&["--rows"])).is_err(),
        "missing value rejected"
    );
    check!(
        case_matrix(
            &parse_args(&s(&["--rows", "512", "--data", "random", "--awq", "some"])).unwrap()
        )
        .len()
            == 1,
        "filters narrow the matrix"
    );
    check!(
        parse_args(&s(&["--plan"]))
            .map(|a| a.mode == Mode::Plan)
            .unwrap_or(false)
            && parse_args(&s(&["--self-check"]))
                .map(|a| a.mode == Mode::SelfCheck)
                .unwrap_or(false),
        "CPU modes parse"
    );

    let failed = failures.len();
    for f in &failures {
        eprintln!("self-check FAIL: {f}");
    }
    println!(
        "{}",
        json!({
            "self_check": true,
            "oracle": "shared_down_sigmoid_epi",
            "gpu_initialized": false,
            "checks": checks,
            "failures": failed,
            "byte_result": null,
            "note": "CPU self-check validates the case matrix, dimension/admission rules, the kernarg ABI, the epilogue arithmetic structure, synthetic data and compare helpers only; it is NOT a PM certification or emulator proof of the kernel and proves nothing about GPU bytes: evidence is the kernel source identity plus the GPU owner's byte oracle run",
        })
    );
    if failed != 0 {
        std::process::exit(1);
    }
}

fn refuse(msg: &str, detail: Value) -> ! {
    eprintln!("error: shared_down_sigmoid_epi_oracle: {msg}; refusing to compare a fallback");
    println!(
        "{}",
        json!({ "error": "unsupported", "message": msg, "detail": detail })
    );
    std::process::exit(2);
}

fn fail_exit(f: Fail) -> ! {
    match f {
        Fail::Unsupported(m) => refuse(&m, Value::Null),
        Fail::Launch(m) => {
            eprintln!("error: shared_down_sigmoid_epi_oracle: {m}");
            println!("{}", json!({ "error": "launch", "message": m }));
            std::process::exit(4);
        }
    }
}

fn run(a: &Args) {
    let start = Instant::now();
    let deadline = start + Duration::from_secs(a.deadline_secs);
    let secs = a.deadline_secs;
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(secs));
        eprintln!(
            "shared_down_sigmoid_epi_oracle: {secs}s deadline exceeded (watchdog); exiting 3"
        );
        std::process::exit(3);
    });

    let cases = case_matrix(a);
    for c in &cases {
        if let Err(e) = validate_dims(M, K, c.rows) {
            refuse(&format!("{}: {e}", c.name()), Value::Null);
        }
    }

    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        let arch = gpu.arch.clone();
        refuse(
            &refuse_reason(&arch, true).unwrap_or_default(),
            json!({ "arch": arch }),
        );
    }
    // The synthetic weights are symmetric; real artifacts set this in the loader.
    gpu.mq4v2_symmetric = true;
    let unsupported: Vec<String> = cases
        .iter()
        .filter(|c| !gpu.mq4g256v2_shared_down_sigmoid_epi_supported(M, K, c.rows))
        .map(Case::name)
        .collect();
    if !unsupported.is_empty() {
        refuse(
            &format!(
                "mq4g256v2_shared_down_sigmoid_epi_supported({M}, {K}, rows) is false on {} for {} case(s); \
                 symmetric MQ4V2, V3 module (HIPFIRE_G12_IU4_ISA / HIPFIRE_G12_IU4_V3), \
                 IU4 prefill, no A8 prefill / FP8 verify scope, eager stream",
                gpu.arch,
                unsupported.len()
            ),
            json!({ "cases": unsupported, "g12_iu4_isa": gpu.flags.g12_iu4_isa }),
        );
    }
    let hipcc_extra_flags = gpu.flags.hipcc_extra_flags.clone();

    let w_bytes = synth_mq4g256v2_sym(M, K, 0x5d51_0001);
    let sh = Shared {
        w: gpu
            .upload_raw(&w_bytes, &[w_bytes.len()])
            .unwrap_or_else(|e| fail_exit(Fail::Launch(format!("upload weight: {e:?}")))),
        awq: gpu
            .upload_f32(&synth_vec(K, 0x5d51_0002, 0.5, 1.5), &[K])
            .unwrap_or_else(|e| fail_exit(Fail::Launch(format!("upload awq: {e:?}")))),
    };

    let mut identical_cases = 0usize;
    let mut completed = 0usize;
    let mut first_mismatch: Option<Value> = None;
    let mut deadline_hit = false;
    for case in &cases {
        if Instant::now() >= deadline {
            deadline_hit = true;
            break;
        }
        let out = run_case(&mut gpu, case, &sh, a, deadline).unwrap_or_else(|f| fail_exit(f));
        println!("{}", out.line);
        completed += 1;
        if out.identical {
            identical_cases += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = out
                .first
                .map(|m| json!({ "case": case.name(), "mismatch": m }));
        }
        if out.deadline_hit {
            deadline_hit = true;
            break;
        }
    }
    let all_identical = !deadline_hit && completed == cases.len() && identical_cases == completed;
    let mut summary = json!({
        "summary": true,
        "oracle": "shared_down_sigmoid_epi",
        "arch": gpu.arch,
        "m": M, "k": K,
        "cases_planned": cases.len(),
        "cases_completed": completed,
        "cases_identical": identical_cases,
        "reps": a.reps, "warmups": a.warmups,
        "deadline_secs": a.deadline_secs,
        "elapsed_secs": start.elapsed().as_secs_f64(),
        "deadline_exceeded": deadline_hit,
        "hipcc_extra_flags": hipcc_extra_flags,
        "first_mismatch": first_mismatch,
    });
    if all_identical {
        summary["all_byte_identical"] = json!(true);
    } else {
        summary["byte_result"] = json!(if deadline_hit { "INCOMPLETE" } else { "FAIL" });
    }
    println!("{summary}");
    if deadline_hit {
        eprintln!("shared_down_sigmoid_epi_oracle: deadline reached before the matrix completed");
        std::process::exit(3);
    }
    if !all_identical {
        eprintln!(
            "shared_down_sigmoid_epi_oracle: {} of {} case(s) not byte-identical",
            completed - identical_cases,
            completed
        );
        std::process::exit(1);
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&raw) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("shared_down_sigmoid_epi_oracle: {e}");
            eprintln!(
                "usage: [--plan|--self-check] [--rows 512,4096,7808,8192] [--data zero,random] \
                 [--awq none,some] [--reps N>=20] [--warmups N] [--deadline-secs 1..=120]"
            );
            std::process::exit(2);
        }
    };
    match args.mode {
        Mode::Plan => plan(&args),
        Mode::SelfCheck => self_check(),
        Mode::Run => run(&args),
    }
}
