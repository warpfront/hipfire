// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the H1 PM C8 IU4 producer twin (gfx1201, lab):
//! the peacemaker-ISA exports `iu4_c8_rotate_pm_gfx1201` and
//! `iu4_c8_rmsnorm_rotate_pm_gfx1201` (launched through
//! `Gpu::mq_rotate_x_i4_c8_pm` / `Gpu::fused_rmsnorm_mq_rotate_i4_c8_pm`)
//! against the incumbent HIP C8 producers
//! `Gpu::rotate_x_mq_i4_gfx12_exact_batched` /
//! `Gpu::fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched` (`cand_mode = 8`,
//! `quantize_block_i4_128_wave<false>`, the 8-candidate "C8" search).
//!
//! L4 folds (same module, 80-byte kernarg block): `iu4_c8_sigmoid_rotate_pm_gfx1201`
//! (`Gpu::mq_sigmoid_rotate_x_i4_c8_pm`) against the unfused chain
//! `Gpu::sigmoid_mul_f32` (in place on a copy of the attention input) followed by
//! `Gpu::rotate_x_mq_i4_gfx12_exact_batched`, and `iu4_c8_rmsnorm_out_rotate_pm_gfx1201`
//! (`Gpu::fused_rmsnorm_out_mq_rotate_i4_c8_pm`) against `Gpu::rmsnorm_batched` (the standalone
//! `rmsnorm_f32`) followed by `Gpu::rotate_x_mq_i4_gfx12_exact_batched` on its output. Besides the
//! sidecar and the rotated f32, the norm fold's normalized f32 row (`norm_out`) is compared
//! bytewise, and the sigmoid fold must leave its attention input unmodified.
//!
//! ## What is compared
//!
//! For every case TWO things are compared bytewise, never one alone:
//!
//! 1. the whole `block_i4_128` sidecar (72 B record per `[K/128 block, row]`:
//!    f32 `d` at 0..4, i32 exact integer sum `s` at 4..8, 64 B packed nibbles
//!    ("payload", low nibble first) at 8..72), snapshotted from the live
//!    `int4_mmq_x_scratch` right after each arm. All 72 bytes of every record
//!    are compared; `d`, `s` and `payload` are reported separately and the
//!    first mismatching record is located (`kblock`, `token`, `field`, byte
//!    offset, and for payload the first K element);
//! 2. the f32 rotated side-output (`x_rot`), bit-for-bit, first mismatching
//!    byte located (`row`, `col`, `group`, producer `lane`).
//!
//! Both arms start from a `0xA5` sentinel in the sidecar region and in
//! `x_rot`, so a PM launch that skips any byte cannot match by accident.
//! `all_byte_identical` is true only when both comparisons hold on the first
//! run AND again after the timing loop (one more untimed pass of both arms).
//! Byte identity is a GPU outcome: nothing on the CPU (`--plan`,
//! `--self-check`) prints a pass, and a refused / aborted run never reports
//! one.
//!
//! The PM exports are launched explicitly through the low-level PM launchers,
//! independent of the production route: the oracle never reads the
//! `PrefillIu4C8Pm` capability / `developer.prefill_iu4_c8_pm`. The route is
//! qualified on exact gfx1201, so an unset flag would let the HIP incumbent
//! arm (`*_exact_batched`) take the PM twin; the oracle therefore requires the
//! explicit kill `HIPFIRE_PREFILL_IU4_C8_PM=0` and refuses to run otherwise.
//!
//! CPU diagnostic (never contributes to `all_byte_identical`): the first 64
//! rows of the incumbent's rotated f32 are quantized by the CPU C8 model in
//! this file and compared with the incumbent's own sidecar
//! (`cpu_model_matches_incumbent`), which validates the CPU model used by
//! `--self-check` against real hardware bytes.
//!
//! ## Matrix
//!
//! | export | K | rows | data | AWQ |
//! |---|---|---|---|---|
//! | rotate (`iu4_c8_rotate_pm_gfx1201`, block 32, grid rows*K/256) | 2048, 4096 | 512, 4096, 8192 | zero, random, large | none, some |
//! | norm (`iu4_c8_rmsnorm_rotate_pm_gfx1201`, block 256, grid rows) | 2048 | 512, 4096, 8192 | zero, random, large | none, some |
//! | sigmoid_rotate (`iu4_c8_sigmoid_rotate_pm_gfx1201`, block 32, grid rows*K/256) | 2048, 4096 | 512, 4096, 7808, 8192 | zero, random, large | none, some |
//! | norm_out (`iu4_c8_rmsnorm_out_rotate_pm_gfx1201`, block 256, grid rows) | 2048 | 512, 4096, 7808, 8192 | zero, random, large | none, some |
//!
//! Rows are 512, 4096, 7808 and 8192 for every export (7808 and 8192 are the A3B prefill chunks);
//! the default matrix is 48 + 24 + 48 + 24 = 144 cases (`--exports` filters, so one export is
//! 24 or 48 cases). `zero` makes every amax 0 (d = 1 branch); `random`
//! is uniform in `[-1, 1)` times a per-row scale `0.25 * (1 + row % 5)`;
//! `large` spans ~2^-40..2^40 per element/row, with all-zero rows
//! (`row % 7 == 3`), single-spike rows (`row % 13 == 0`) and rows with
//! negative zeros (`row % 5 == 2`). AWQ scales and norm weights are finite
//! positive F32 (large data uses wide exponent spreads); no reciprocal planes.
//!
//! ## Timing
//!
//! `--reps N` (default 20, minimum 20) + 3 warmups per arm. The arm order
//! alternates each rep (HIP, PM, PM, HIP, ...). Each rep: `x_rot` sentinel
//! memset + `device_synchronize` (untimed), then `Instant::now()`, the arm's
//! reservation + producer launch, `device_synchronize`, elapsed. Median / p10
//! / p90 in microseconds for both arms. Timing includes host launch overhead.
//!
//! ## Deadline
//!
//! One process, at most 120 s (`--deadline-secs N`, 1..=120): cooperative
//! checks between reps/cases plus a watchdog thread that exits 3 even if a
//! GPU call hangs.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab \
//!     --example iu4_c8_pm_oracle -- --plan
//! cargo run --release -p rdna-compute --features lab \
//!     --example iu4_c8_pm_oracle -- --self-check
//! cargo run --release -p rdna-compute --features lab \
//!     --example iu4_c8_pm_oracle                       # GPU, single process
//! # filters: --exports rotate,norm --k 2048,4096 --rows 512 --data zero,random,large --awq none,some
//! ```
//!
//! stdout (GPU mode) = one JSON line per case then the summary line
//! `{"summary":true,...,"all_byte_identical":bool,"first_mismatch":{...}|null}`.
//! Exit 0 when every case is byte-identical, 1 on any mismatch (after the
//! summary), 2 on bad arguments / unsupported configuration (before any case),
//! 3 on deadline.

use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor, Int4MmqPrepared};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const DEFAULT_REPS: usize = 20;
const MIN_REPS: usize = 20;
const DEFAULT_WARMUPS: usize = 3;
const MAX_DEADLINE_SECS: u64 = 120;
/// `block_i4_128` record: f32 d, i32 s, 64 B nibbles.
const RECORD_BYTES: usize = 72;
const D_RANGE: (usize, usize) = (0, 4);
const S_RANGE: (usize, usize) = (4, 8);
const QS_RANGE: (usize, usize) = (8, 72);
/// Nonzero fill for sidecar and rotated outputs before each verified launch.
const SENTINEL: u8 = 0xA5;
const EPS: f32 = 1.0e-6;
const ROWS: [usize; 4] = [512, 4096, 7808, 8192];
const KS: [usize; 2] = [2048, 4096];
/// Raw env key of the production flag; the oracle must not run with it on.
const FLAG_ENV: &str = "HIPFIRE_PREFILL_IU4_C8_PM";
const FLAG_TYPED_KEY: &str = "developer.prefill_iu4_c8_pm";
const PM_MODULE: &str = "iu4_c8_producer_pm_gfx1201";
/// Candidate count of the C8 search and the first-rows CPU model span.
const C8_CANDIDATES: usize = 8;
const CPU_MODEL_ROWS: usize = 64;

/// Frozen PM kernarg ABI (72 bytes): (name, offset, size).
const PM_KERNARG: [(&str, usize, usize); 10] = [
    ("x", 0, 8),
    ("rotated_out", 8, 8),
    ("sidecar", 16, 8),
    ("signs1", 24, 8),
    ("signs2", 32, 8),
    ("weight", 40, 8),
    ("awq", 48, 8),
    ("rows", 56, 4),
    ("k", 60, 4),
    ("eps", 64, 4),
];
const PM_KERNARG_BYTES: usize = 72;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Export {
    Rotate,
    Norm,
    SigmoidRotate,
    NormOut,
}

impl Export {
    const ALL: [Export; 4] = [
        Export::Rotate,
        Export::Norm,
        Export::SigmoidRotate,
        Export::NormOut,
    ];

    fn name(self) -> &'static str {
        match self {
            Export::Rotate => "rotate",
            Export::Norm => "norm",
            Export::SigmoidRotate => "sigmoid_rotate",
            Export::NormOut => "norm_out",
        }
    }

    fn from_name(s: &str) -> Option<Export> {
        Export::ALL.into_iter().find(|e| e.name() == s)
    }

    fn symbol(self) -> &'static str {
        match self {
            Export::Rotate => "iu4_c8_rotate_pm_gfx1201",
            Export::Norm => "iu4_c8_rmsnorm_rotate_pm_gfx1201",
            Export::SigmoidRotate => "iu4_c8_sigmoid_rotate_pm_gfx1201",
            Export::NormOut => "iu4_c8_rmsnorm_out_rotate_pm_gfx1201",
        }
    }

    fn supports_k(self, k: usize) -> bool {
        match self {
            Export::Rotate | Export::SigmoidRotate => k == 2048 || k == 4096,
            Export::Norm | Export::NormOut => k == 2048,
        }
    }

    /// The RMSNorm-fronted exports (take a norm weight).
    fn is_norm(self) -> bool {
        matches!(self, Export::Norm | Export::NormOut)
    }

    fn hip_incumbent(self) -> &'static str {
        match self {
            Export::Rotate => "rotate_x_mq_i4_gfx12_exact_batched",
            Export::Norm => "fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched",
            Export::SigmoidRotate => {
                "sigmoid_mul_f32 + rotate_x_mq_i4_gfx12_exact_batched (unfused chain)"
            }
            Export::NormOut => {
                "rmsnorm_batched + rotate_x_mq_i4_gfx12_exact_batched (unfused chain)"
            }
        }
    }

    fn pm_launcher(self) -> &'static str {
        match self {
            Export::Rotate => "mq_rotate_x_i4_c8_pm",
            Export::Norm => "fused_rmsnorm_mq_rotate_i4_c8_pm",
            Export::SigmoidRotate => "mq_sigmoid_rotate_x_i4_c8_pm",
            Export::NormOut => "fused_rmsnorm_out_mq_rotate_i4_c8_pm",
        }
    }

    /// (grid, block) of the PM export.
    fn geometry(self, k: usize, rows: usize) -> (usize, usize) {
        match self {
            Export::Rotate | Export::SigmoidRotate => (rows * (k / 256), 32),
            Export::Norm | Export::NormOut => (rows, 256),
        }
    }

    /// The byte streams compared for this export.
    fn compared(self) -> Vec<&'static str> {
        let mut v = vec![
            "sidecar d@0..4",
            "sidecar s@4..8",
            "sidecar payload@8..72",
            "rotated f32",
        ];
        match self {
            Export::NormOut => v.push("normalized f32 row (norm_out)"),
            Export::SigmoidRotate => v.push("attention input untouched"),
            Export::Rotate | Export::Norm => {}
        }
        v
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Data {
    Zero,
    Random,
    Large,
}

impl Data {
    const ALL: [Data; 3] = [Data::Zero, Data::Random, Data::Large];

    fn name(self) -> &'static str {
        match self {
            Data::Zero => "zero",
            Data::Random => "random",
            Data::Large => "large",
        }
    }

    fn from_name(s: &str) -> Option<Data> {
        Data::ALL.into_iter().find(|d| d.name() == s)
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    export: Export,
    k: usize,
    rows: usize,
    data: Data,
    awq: bool,
}

impl Case {
    fn name(&self) -> String {
        format!(
            "{}_k{}_rows{}_{}_{}",
            self.export.name(),
            self.k,
            self.rows,
            self.data.name(),
            if self.awq { "awq" } else { "noawq" }
        )
    }

    fn sidecar_bytes(&self) -> usize {
        (self.k / 128) * self.rows * RECORD_BYTES
    }

    fn rot_bytes(&self) -> usize {
        self.rows * self.k * 4
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
    exports: Vec<Export>,
    ks: Vec<usize>,
    rows: Vec<usize>,
    datas: Vec<Data>,
    awqs: Vec<bool>,
    reps: usize,
    warmups: usize,
    deadline_secs: u64,
}

fn parse_list<T>(
    name: &str,
    v: &str,
    f: impl Fn(&str) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let out: Vec<T> = v.split(',').map(&f).collect::<Result<_, _>>()?;
    if out.is_empty() {
        return Err(format!("{name} list is empty"));
    }
    Ok(out)
}

fn parse_args(raw: &[String]) -> Result<Args, String> {
    let mut a = Args {
        mode: Mode::Run,
        exports: Export::ALL.to_vec(),
        ks: KS.to_vec(),
        rows: ROWS.to_vec(),
        datas: Data::ALL.to_vec(),
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
            "--exports" => {
                a.exports = parse_list("--exports", &value("--exports")?, |s| {
                    Export::from_name(s).ok_or_else(|| {
                        format!("export {s:?} not rotate|norm|sigmoid_rotate|norm_out")
                    })
                })?;
            }
            "--k" => {
                a.ks = parse_list("--k", &value("--k")?, |s| match s.parse::<usize>() {
                    Ok(k) if KS.contains(&k) => Ok(k),
                    _ => Err(format!("k {s:?} not in {KS:?}")),
                })?;
            }
            "--rows" => {
                a.rows = parse_list("--rows", &value("--rows")?, |s| match s.parse::<usize>() {
                    Ok(r) if ROWS.contains(&r) => Ok(r),
                    _ => Err(format!("rows {s:?} not in {ROWS:?}")),
                })?;
            }
            "--data" => {
                a.datas = parse_list("--data", &value("--data")?, |s| {
                    Data::from_name(s).ok_or_else(|| format!("data {s:?} not zero|random|large"))
                })?;
            }
            "--awq" => {
                a.awqs = parse_list("--awq", &value("--awq")?, |s| match s {
                    "none" => Ok(false),
                    "some" => Ok(true),
                    _ => Err(format!("awq {s:?} not none|some")),
                })?;
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
    if case_matrix(&a).is_empty() {
        return Err("filters select no case (norm supports only K=2048)".to_string());
    }
    Ok(a)
}

/// Ordered so consecutive cases share one `(rows, data, k)` input group:
/// rows -> data -> k -> export -> awq.
fn case_matrix(a: &Args) -> Vec<Case> {
    let mut out = Vec::new();
    for &rows in &a.rows {
        for &data in &a.datas {
            for &k in &a.ks {
                for &export in &a.exports {
                    if !export.supports_k(k) {
                        continue;
                    }
                    for &awq in &a.awqs {
                        out.push(Case {
                            export,
                            k,
                            rows,
                            data,
                            awq,
                        });
                    }
                }
            }
        }
    }
    out
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

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0
    }
}

/// Exact `2^e` for `e` in `-126..=127`.
fn pow2(e: i32) -> f32 {
    assert!((-126..=127).contains(&e));
    f32::from_bits(((e + 127) as u32) << 23)
}

fn group_seed(k: usize, rows: usize, data: Data) -> u64 {
    0xc8_0000_0000 + ((k as u64) << 20) + ((rows as u64) << 4) + data as u64
}

/// Finite F32 activations; see the module docs for the three data modes.
fn synth_x(rows: usize, k: usize, seed: u64, data: Data) -> Vec<f32> {
    match data {
        Data::Zero => vec![0.0; rows * k],
        Data::Random => {
            let mut rng = Rng(seed);
            let mut out = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let scale = 0.25 * (1 + r % 5) as f32;
                for _ in 0..k {
                    out.push(rng.unit() * scale);
                }
            }
            out
        }
        Data::Large => {
            let mut rng = Rng(seed);
            let mut out = Vec::with_capacity(rows * k);
            for r in 0..rows {
                let start = out.len();
                if r % 7 == 3 {
                    out.resize(start + k, 0.0);
                    continue;
                }
                let row_off = (r % 5) as i32 * 8 - 16;
                for _ in 0..k {
                    // Magnitude in [0.5, 1) with a random sign, exponent in
                    // [-24, 24] plus the row offset in [-16, 16]: never
                    // subnormal, squares stay finite for K <= 4096.
                    let u = rng.unit();
                    let m = (0.5 + 0.5 * u.abs().min(0.999_999)) * u.signum();
                    let e = (rng.next() % 49) as i32 - 24 + row_off;
                    out.push(m * pow2(e));
                }
                if r % 13 == 0 {
                    let keep = (r * 37) % k;
                    for v in &mut out[start..start + k] {
                        *v = 0.0;
                    }
                    out[start + keep] = pow2((r % 61) as i32 - 30);
                } else if r % 5 == 2 {
                    for i in (0..k).step_by(16) {
                        out[start + i] = -0.0;
                    }
                }
            }
            out
        }
    }
}

/// Positive finite F32 vector: `[lo, lo + span)`, or for [`Data::Large`] a
/// mantissa in `[0.5, 1)` times `2^e`, `e` in `-spread..=spread`.
fn synth_pos(len: usize, seed: u64, lo: f32, span: f32, data: Data, spread: i32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|_| {
            let u = (rng.next() & 0xff_ffff) as f32 / 16_777_216.0;
            if data == Data::Large {
                let e = (rng.next() % (2 * spread as u64 + 1)) as i32 - spread;
                (0.5 + 0.5 * u) * pow2(e)
            } else {
                lo + u * span
            }
        })
        .collect()
}

/// Gate rows of the sigmoid fold. `zero`: all zero (`sigmoid = 0.5`). `random`: uniform in `[-30, 30)` with
/// one element in 32 forced to `+-[80, 110)`, which straddles the `expf(-g)` range selects (`-g` below
/// -103.27 / above 88.72). `large`: the wide-exponent activation synthesis (magnitudes up to ~2^40 of both signs).
fn synth_gate(rows: usize, k: usize, seed: u64, data: Data) -> Vec<f32> {
    match data {
        Data::Zero => vec![0.0; rows * k],
        Data::Random => {
            let mut rng = Rng(seed);
            (0..rows * k)
                .map(|_| {
                    let u = rng.unit();
                    if rng.next() % 32 == 0 {
                        let m = 80.0 + 30.0 * rng.unit().abs();
                        if u < 0.0 {
                            -m
                        } else {
                            m
                        }
                    } else {
                        u * 30.0
                    }
                })
                .collect()
        }
        Data::Large => synth_x(rows, k, seed, Data::Large),
    }
}

// ---------------------------------------------------------------------------
// CPU model of the C8 quantizer / remap / packing (covered by --self-check)
// ---------------------------------------------------------------------------

/// `rintf` (ties to even).
fn rint(v: f32) -> f32 {
    v.round_ties_even()
}

/// `dj = (amax/7)*0.5*(1+(float)j/7)`, C evaluation order, no contraction.
fn c8_candidate(amax: f32, j: usize) -> f32 {
    (amax / 7.0) * 0.5 * (1.0 + (j as f32) / 7.0)
}

/// `q = clamp(rint(x/dj), -8, 7)` as float.
fn c8_code(x: f32, d: f32) -> f32 {
    rint(x / d).max(-8.0).min(7.0)
}

/// Per-lane 4-element mse: `err = fma(-q,dj,x)`, `mse = fma(err,err,mse)` from +0.
fn lane_mse(xv: &[f32], dj: f32) -> f32 {
    let mut mse = 0.0f32;
    for &x in xv {
        let q = c8_code(x, dj);
        let err = (-q).mul_add(dj, x);
        mse = err.mul_add(err, mse);
    }
    mse
}

/// Wave32 xor-butterfly sum, offsets 16, 8, 4, 2, 1 (all lanes end equal).
fn xor_sum32(mut v: [f32; 32]) -> [f32; 32] {
    for off in [16usize, 8, 4, 2, 1] {
        let mut n = v;
        for l in 0..32 {
            n[l] = v[l] + v[l ^ off];
        }
        v = n;
    }
    v
}

/// Selected C8 candidate index (None when amax == 0) and `d`.
fn c8_select(x: &[f32; 128]) -> (Option<usize>, f32) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return (None, 1.0);
    }
    let mut best_mse = 1.0e30f32;
    let mut best_d = 1.0f32;
    let mut best_j = None;
    for j in 0..C8_CANDIDATES {
        let dj = c8_candidate(amax, j);
        let mut lanes = [0.0f32; 32];
        for (l, m) in lanes.iter_mut().enumerate() {
            *m = lane_mse(&x[4 * l..4 * l + 4], dj);
        }
        let mse = xor_sum32(lanes)[0];
        if mse < best_mse {
            best_mse = mse;
            best_d = dj;
            best_j = Some(j);
        }
    }
    (best_j, best_d)
}

/// One 72 B `block_i4_128` record for 128 rotated floats (element `i` = K
/// position `128*kblock + i`).
fn quantize_block_c8(x: &[f32; 128]) -> [u8; RECORD_BYTES] {
    let (j, d) = c8_select(x);
    let mut q4 = [0i32; 128];
    if j.is_some() {
        for (q, &v) in q4.iter_mut().zip(x.iter()) {
            *q = c8_code(v, d) as i32;
        }
    }
    let s: i32 = q4.iter().sum();
    let mut rec = [0u8; RECORD_BYTES];
    rec[..4].copy_from_slice(&d.to_le_bytes());
    rec[4..8].copy_from_slice(&s.to_le_bytes());
    for lane in 0..32 {
        let q = &q4[4 * lane..4 * lane + 4];
        rec[8 + lane * 2] = ((q[0] & 15) | ((q[1] & 15) << 4)) as u8;
        rec[8 + lane * 2 + 1] = ((q[2] & 15) | ((q[3] & 15) << 4)) as u8;
    }
    rec
}

/// Producer-ownership remap: for half `h` and quantizer lane `l` float `e` of
/// the float4, returns (source producer lane, float index 0..8, element within
/// the 256-group).
fn remap(h: usize, l: usize, e: usize) -> (usize, usize, usize) {
    let src = 16 * h + (l >> 1);
    let fidx = (l & 1) * 4 + e;
    (src, fidx, 8 * src + fidx)
}

/// Token-order record index of `(group, half)` for `token` of `n` rows.
fn record_index(group: usize, h: usize, token: usize, n: usize) -> usize {
    (2 * group + h) * n + token
}

fn le_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// CPU C8 model over the first `take` tokens of `rot` (`[rows, k]` f32 bytes)
/// against a token-order sidecar of `n` rows: first mismatching
/// `(kblock, token)` or None.
fn cpu_model_vs_sidecar(
    rot: &[u8],
    sidecar: &[u8],
    k: usize,
    n: usize,
    take: usize,
) -> Option<(usize, usize)> {
    for token in 0..take.min(n) {
        let row = le_f32s(&rot[token * k * 4..(token + 1) * k * 4]);
        for kb in 0..k / 128 {
            let mut blk = [0.0f32; 128];
            blk.copy_from_slice(&row[kb * 128..kb * 128 + 128]);
            let want = quantize_block_c8(&blk);
            let at = (kb * n + token) * RECORD_BYTES;
            if sidecar[at..at + RECORD_BYTES] != want {
                return Some((kb, token));
            }
        }
    }
    None
}

/// Flag expectation for `HIPFIRE_PREFILL_IU4_C8_PM`: explicit 1 opts in,
/// 0 kills, unset follows the exact-gfx1201 qualification (on there).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlagState {
    Default,
    On,
    Killed,
}

fn flag_state(env: Option<&str>) -> FlagState {
    match env {
        Some("1") => FlagState::On,
        Some("0") => FlagState::Killed,
        _ => FlagState::Default,
    }
}

/// The oracle's HIP incumbent must not be re-routed to PM: only the explicit
/// kill keeps the `*_exact_batched` methods on the HIP export.
fn env_isolated(env: Option<&str>) -> bool {
    flag_state(env) == FlagState::Killed
}

// ---------------------------------------------------------------------------
// Byte compare / locate helpers (CPU, covered by --self-check)
// ---------------------------------------------------------------------------

/// First differing byte; when one slice is a strict prefix of the other, the
/// shorter length.
fn first_mismatch(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn record_field(offset_in_record: usize) -> &'static str {
    if offset_in_record < D_RANGE.1 {
        "d"
    } else if offset_in_record < S_RANGE.1 {
        "s"
    } else {
        "payload"
    }
}

/// Locate a sidecar byte in the token-order `[kblock, token]` record stream.
fn sidecar_locate(byte: usize, n: usize) -> Value {
    let record = byte / RECORD_BYTES;
    let off = byte % RECORD_BYTES;
    let mut v = json!({
        "byte": byte,
        "record": record,
        "kblock": record / n,
        "token": record % n,
        "offset_in_record": off,
        "field": record_field(off),
    });
    if off >= QS_RANGE.0 {
        // byte j of qs holds elements 2j (low nibble) and 2j+1 (high nibble)
        v["k_element_in_block"] = json!(2 * (off - QS_RANGE.0));
    }
    v
}

/// Locate a rotated f32 byte in a `[rows, k]` buffer.
fn rot_locate(byte: usize, k: usize) -> Value {
    let elem = byte / 4;
    let col = elem % k;
    json!({
        "byte": byte,
        "elem": elem,
        "row": elem / k,
        "col": col,
        "group256": col / 256,
        "producer_lane": (col % 256) / 8,
        "byte_in_f32": byte % 4,
    })
}

struct SidecarCmp {
    equal: bool,
    json: Value,
}

/// Compare two token-order sidecars over all 72 bytes of every record:
/// whole-stream equality, per-field equality and the first mismatch.
fn compare_sidecar(inc: &[u8], new: &[u8], n: usize) -> SidecarCmp {
    let first = first_mismatch(inc, new);
    let mut d_eq = true;
    let mut s_eq = true;
    let mut qs_eq = true;
    let mut bad_records = 0usize;
    for (a, b) in inc
        .chunks_exact(RECORD_BYTES)
        .zip(new.chunks_exact(RECORD_BYTES))
    {
        let d = a[D_RANGE.0..D_RANGE.1] == b[D_RANGE.0..D_RANGE.1];
        let s = a[S_RANGE.0..S_RANGE.1] == b[S_RANGE.0..S_RANGE.1];
        let q = a[QS_RANGE.0..QS_RANGE.1] == b[QS_RANGE.0..QS_RANGE.1];
        d_eq &= d;
        s_eq &= s;
        qs_eq &= q;
        bad_records += usize::from(!(d && s && q));
    }
    let equal = first.is_none();
    SidecarCmp {
        equal,
        json: json!({
            "equal": equal,
            "bytes_incumbent": inc.len(),
            "bytes_pm": new.len(),
            "d_equal": d_eq,
            "s_equal": s_eq,
            "payload_equal": qs_eq,
            "mismatching_records": bad_records,
            "first_mismatch": first.map(|b| sidecar_locate(b, n)),
        }),
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

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("device synchronize");
}

type R = rdna_compute::HipResult<()>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Hip,
    Pm,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Hip => "hip_c8",
            Arm::Pm => "pm_c8",
        }
    }
}

/// Per `(rows, data, k)` buffers shared by every export/AWQ case of the group.
struct Group {
    key: (usize, Data, usize),
    x: GpuTensor,
    rot_hip: GpuTensor,
    rot_pm: GpuTensor,
    /// Sigmoid fold: the gate rows, and the copy of `x` the unfused chain gates in place.
    gate: GpuTensor,
    attn_work: GpuTensor,
    /// Norm fold: the normalized f32 row of each arm.
    norm_hip: GpuTensor,
    norm_pm: GpuTensor,
}

fn make_group(gpu: &mut Gpu, case: &Case) -> Group {
    let (k, rows) = (case.k, case.rows);
    let host = synth_x(rows, k, group_seed(k, rows, case.data), case.data);
    let x = gpu.upload_f32(&host, &[rows, k]).expect("upload x");
    let gate_host = synth_gate(rows, k, group_seed(k, rows, case.data) + 0x400, case.data);
    let gate = gpu.upload_f32(&gate_host, &[rows, k]).expect("upload gate");
    Group {
        key: (case.rows, case.data, case.k),
        x,
        rot_hip: gpu.zeros(&[rows, k], DType::F32).expect("alloc rot_hip"),
        rot_pm: gpu.zeros(&[rows, k], DType::F32).expect("alloc rot_pm"),
        gate,
        attn_work: gpu.zeros(&[rows, k], DType::F32).expect("alloc attn_work"),
        norm_hip: gpu.zeros(&[rows, k], DType::F32).expect("alloc norm_hip"),
        norm_pm: gpu.zeros(&[rows, k], DType::F32).expect("alloc norm_pm"),
    }
}

fn free_group(gpu: &mut Gpu, g: Group) {
    for t in [
        g.x,
        g.rot_hip,
        g.rot_pm,
        g.gate,
        g.attn_work,
        g.norm_hip,
        g.norm_pm,
    ] {
        let _ = gpu.free_tensor(t);
    }
}

struct Ctx<'a> {
    case: &'a Case,
    x: &'a GpuTensor,
    weight: Option<&'a GpuTensor>,
    awq: Option<&'a GpuTensor>,
    g: &'a Group,
}

fn fill_device(gpu: &Gpu, ptr: *mut std::ffi::c_void, bytes: usize, val: u8) {
    let view = unsafe { DeviceBuffer::from_raw(ptr, bytes) };
    gpu.hip
        .memset(&view, val as i32, bytes)
        .expect("sentinel memset");
    std::mem::forget(view);
}

/// Reserve the sidecar, optionally sentinel-fill it, then launch one arm's
/// producer writing `x_rot` and the sidecar.
fn launch(
    gpu: &mut Gpu,
    c: &Ctx,
    arm: Arm,
    x_rot: &GpuTensor,
    fill_sidecar: bool,
) -> rdna_compute::HipResult<Int4MmqPrepared> {
    let (k, n) = (c.case.k, c.case.rows);
    let res = gpu.reserve_int4_mmq(k, n)?;
    if fill_sidecar {
        let (_, ptr) = gpu.scratch.int4_mmq_live();
        assert!(!ptr.is_null(), "no live IU4 sidecar scratch");
        fill_device(gpu, ptr, c.case.sidecar_bytes(), SENTINEL);
    }
    match (c.case.export, arm) {
        (Export::Rotate, Arm::Hip) => {
            gpu.rotate_x_mq_i4_gfx12_exact_batched(c.x, c.awq, Some(x_rot), res, k, n)
        }
        (Export::Rotate, Arm::Pm) => gpu.mq_rotate_x_i4_c8_pm(c.x, c.awq, Some(x_rot), res, k, n),
        (Export::Norm, Arm::Hip) => gpu.fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched(
            c.x,
            c.weight.expect("norm weight"),
            c.awq,
            Some(x_rot),
            res,
            k,
            EPS,
            n,
        ),
        (Export::Norm, Arm::Pm) => gpu.fused_rmsnorm_mq_rotate_i4_c8_pm(
            c.x,
            c.weight.expect("norm weight"),
            c.awq,
            Some(x_rot),
            res,
            k,
            EPS,
            n,
        ),
        // The unfused chain of the sigmoid fold: `sigmoid_mul_f32` in place on the attention copy, then the
        // HIP exact rotate of it (the flag is off, so the exact wrapper is the HIP producer).
        (Export::SigmoidRotate, Arm::Hip) => {
            gpu.sigmoid_mul_f32(&c.g.attn_work, &c.g.gate)?;
            gpu.rotate_x_mq_i4_gfx12_exact_batched(&c.g.attn_work, c.awq, Some(x_rot), res, k, n)
        }
        (Export::SigmoidRotate, Arm::Pm) => {
            gpu.mq_sigmoid_rotate_x_i4_c8_pm(c.x, &c.g.gate, c.awq, Some(x_rot), res, k, n)
        }
        // The unfused chain of the norm fold: the standalone `rmsnorm_f32` (as `rmsnorm_batched`) into
        // `norm_hip`, then the HIP exact rotate of it.
        (Export::NormOut, Arm::Hip) => {
            gpu.rmsnorm_batched(
                c.x,
                c.weight.expect("norm weight"),
                &c.g.norm_hip,
                n,
                k,
                EPS,
            )?;
            gpu.rotate_x_mq_i4_gfx12_exact_batched(&c.g.norm_hip, c.awq, Some(x_rot), res, k, n)
        }
        (Export::NormOut, Arm::Pm) => gpu.fused_rmsnorm_out_mq_rotate_i4_c8_pm(
            c.x,
            c.weight.expect("norm weight"),
            c.awq,
            &c.g.norm_pm,
            Some(x_rot),
            res,
            k,
            EPS,
            n,
        ),
    }
}

/// The normalized-row tensor of an arm (norm fold only).
fn norm_tensor<'a>(c: &Ctx<'a>, arm: Arm) -> &'a GpuTensor {
    if arm == Arm::Hip {
        &c.g.norm_hip
    } else {
        &c.g.norm_pm
    }
}

/// Untimed per-launch setup: the unfused sigmoid chain gates a fresh copy of the attention input in place.
fn prepare_arm(gpu: &mut Gpu, c: &Ctx, arm: Arm) {
    if c.case.export == Export::SigmoidRotate && arm == Arm::Hip {
        gpu.hip
            .memcpy_dtod_at(&c.g.attn_work.buf, 0, &c.x.buf, 0, c.x.byte_size())
            .expect("copy attention input");
    }
}

/// Download the live IU4 sidecar of `(k, n)`; the C8 producers emit the AoS
/// token-order layout, a slab layout is refused.
fn snapshot_sidecar(gpu: &Gpu, k: usize, n: usize) -> Vec<u8> {
    let nbytes = (k / 128) * n * RECORD_BYTES;
    let (_, ptr) = gpu.scratch.int4_mmq_live();
    assert!(!ptr.is_null(), "no live IU4 sidecar scratch");
    assert!(
        !gpu.scratch.int4_mmq_slab_at(ptr),
        "C8 sidecar must be the AoS token-order layout, found slab"
    );
    let mut bytes = vec![0u8; nbytes];
    let view = unsafe { DeviceBuffer::from_raw(ptr, nbytes) };
    gpu.hip
        .memcpy_dtoh(&mut bytes, &view)
        .expect("download sidecar");
    std::mem::forget(view);
    bytes
}

struct ArmOut {
    sidecar: Vec<u8>,
    rot: Vec<u8>,
    /// The arm's normalized f32 row (norm fold only, else empty).
    norm: Vec<u8>,
    /// The attention input after the PM sigmoid arm (sigmoid fold only, else empty).
    attn_after: Vec<u8>,
}

fn run_arm_verified(gpu: &mut Gpu, c: &Ctx, arm: Arm, x_rot: &GpuTensor) -> ArmOut {
    let (k, n) = (c.case.k, c.case.rows);
    fill_device(gpu, x_rot.buf.as_ptr(), x_rot.byte_size(), SENTINEL);
    if c.case.export == Export::NormOut {
        let t = norm_tensor(c, arm);
        fill_device(gpu, t.buf.as_ptr(), t.byte_size(), SENTINEL);
    }
    prepare_arm(gpu, c, arm);
    sync(gpu);
    let gen_before = gpu.scratch.int4_mmq_live().0;
    let prep = launch(gpu, c, arm, x_rot, true)
        .unwrap_or_else(|e| panic!("{} launch failed: {e:?}", arm.name()));
    sync(gpu);
    drop(prep);
    assert_ne!(
        gen_before,
        gpu.scratch.int4_mmq_live().0,
        "{} did not reserve a fresh sidecar generation",
        arm.name()
    );
    ArmOut {
        sidecar: snapshot_sidecar(gpu, k, n),
        rot: gpu.download_raw_bytes(x_rot).expect("download x_rot"),
        norm: if c.case.export == Export::NormOut {
            gpu.download_raw_bytes(norm_tensor(c, arm))
                .expect("download norm_out")
        } else {
            Vec::new()
        },
        attn_after: if c.case.export == Export::SigmoidRotate && arm == Arm::Pm {
            gpu.download_raw_bytes(c.x)
                .expect("download attention input")
        } else {
            Vec::new()
        },
    }
}

struct Pass {
    json: Value,
    identical: bool,
    first_mismatch: Option<Value>,
}

/// One verification pass: HIP incumbent then PM, both sentinel-initialized.
fn verify_pass(gpu: &mut Gpu, c: &Ctx, g: &Group) -> Pass {
    let (k, n) = (c.case.k, c.case.rows);
    let attn_before = (c.case.export == Export::SigmoidRotate).then(|| {
        gpu.download_raw_bytes(c.x)
            .expect("download attention input")
    });
    let inc = run_arm_verified(gpu, c, Arm::Hip, &g.rot_hip);
    let pm = run_arm_verified(gpu, c, Arm::Pm, &g.rot_pm);

    let side = compare_sidecar(&inc.sidecar, &pm.sidecar, n);
    let rot_first = first_mismatch(&inc.rot, &pm.rot);
    let rot_equal = rot_first.is_none();
    // Norm fold: the normalized f32 row. Sigmoid fold: the PM arm leaves its attention input unmodified.
    let norm_first = (c.case.export == Export::NormOut)
        .then(|| first_mismatch(&inc.norm, &pm.norm))
        .flatten();
    let norm_equal = norm_first.is_none();
    let attn_first = attn_before
        .as_ref()
        .and_then(|before| first_mismatch(before, &pm.attn_after));
    let attn_equal = attn_first.is_none();

    // Diagnostic: CPU C8 model on the incumbent's rotated f32 vs the
    // incumbent's own sidecar (first CPU_MODEL_ROWS rows).
    let model_bad = cpu_model_vs_sidecar(&inc.rot, &inc.sidecar, k, n, CPU_MODEL_ROWS);

    let identical = side.equal && rot_equal && norm_equal && attn_equal;
    let first_mismatch = if !side.equal {
        Some(json!({ "stage": "sidecar", "at": side.json["first_mismatch"].clone() }))
    } else if let Some(b) = rot_first {
        Some(json!({ "stage": "rotated_f32", "at": rot_locate(b, k) }))
    } else if let Some(b) = norm_first {
        Some(json!({ "stage": "norm_out_f32", "at": rot_locate(b, k) }))
    } else {
        attn_first.map(|b| json!({ "stage": "attention_input_modified", "at": rot_locate(b, k) }))
    };
    Pass {
        json: json!({
            "sidecar": side.json,
            "rotated_f32": {
                "equal": rot_equal,
                "bytes": inc.rot.len(),
                "first_mismatch": rot_first.map(|b| rot_locate(b, k)),
            },
            "norm_out_f32": (c.case.export == Export::NormOut).then(|| json!({
                "equal": norm_equal,
                "bytes": inc.norm.len(),
                "first_mismatch": norm_first.map(|b| rot_locate(b, k)),
            })),
            "attention_input_untouched": (c.case.export == Export::SigmoidRotate).then_some(attn_equal),
            "cpu_model_matches_incumbent": model_bad.is_none(),
            "cpu_model_first_mismatch": model_bad.map(|(kb, t)| json!({"kblock": kb, "token": t})),
            "cpu_model_rows": CPU_MODEL_ROWS.min(n),
            "byte_identical": identical,
        }),
        identical,
        first_mismatch,
    }
}

struct TimingOut {
    hip: Vec<f64>,
    pm: Vec<f64>,
    deadline_hit: bool,
}

fn time_arms(
    gpu: &mut Gpu,
    c: &Ctx,
    g: &Group,
    reps: usize,
    warmups: usize,
    deadline: Instant,
) -> TimingOut {
    let mut t = TimingOut {
        hip: Vec::with_capacity(reps),
        pm: Vec::with_capacity(reps),
        deadline_hit: false,
    };
    for rep in 0..warmups + reps {
        if Instant::now() >= deadline {
            t.deadline_hit = true;
            break;
        }
        // Alternate the arm order every rep so neither arm always runs first.
        let order = if rep % 2 == 0 {
            [Arm::Hip, Arm::Pm]
        } else {
            [Arm::Pm, Arm::Hip]
        };
        let mut us = [0.0f64; 2];
        for arm in order {
            let x_rot = if arm == Arm::Hip {
                &g.rot_hip
            } else {
                &g.rot_pm
            };
            fill_device(gpu, x_rot.buf.as_ptr(), x_rot.byte_size(), SENTINEL);
            prepare_arm(gpu, c, arm);
            sync(gpu);
            let t0 = Instant::now();
            let r: R = launch(gpu, c, arm, x_rot, false).map(drop);
            r.unwrap_or_else(|e| panic!("timed {} launch failed: {e:?}", arm.name()));
            sync(gpu);
            us[arm as usize] = t0.elapsed().as_secs_f64() * 1e6;
        }
        if rep >= warmups {
            t.hip.push(us[Arm::Hip as usize]);
            t.pm.push(us[Arm::Pm as usize]);
        }
    }
    if t.hip.len() < reps {
        t.deadline_hit = true;
    }
    t
}

struct CaseOut {
    line: Value,
    identical: bool,
    first_mismatch: Option<Value>,
    deadline_hit: bool,
}

fn run_case(gpu: &mut Gpu, case: &Case, g: &Group, a: &Args, deadline: Instant) -> CaseOut {
    let k = case.k;
    let seed = group_seed(k, case.rows, case.data);
    let weight = case.export.is_norm().then(|| {
        let h = synth_pos(k, seed + 0x200, 0.5, 1.0, case.data, 8);
        gpu.upload_f32(&h, &[k]).expect("upload norm weight")
    });
    let awq = case.awq.then(|| {
        let h = synth_pos(k, seed + 0x300, 0.5, 1.5, case.data, 6);
        gpu.upload_f32(&h, &[k]).expect("upload awq scale")
    });
    let ctx = Ctx {
        case,
        x: &g.x,
        weight: weight.as_ref(),
        awq: awq.as_ref(),
        g,
    };

    let first = verify_pass(gpu, &ctx, g);
    let timing = time_arms(gpu, &ctx, g, a.reps, a.warmups, deadline);
    let post = verify_pass(gpu, &ctx, g);

    let identical = first.identical && post.identical && !timing.deadline_hit;
    let first_mismatch = first.first_mismatch.clone().or(post.first_mismatch.clone());
    let timing_json = if timing.hip.len() >= MIN_REPS && timing.pm.len() >= MIN_REPS {
        let hip = stats(&timing.hip);
        let pm = stats(&timing.pm);
        let speedup = hip["median"].as_f64().unwrap() / pm["median"].as_f64().unwrap();
        json!({
            "incumbent_hip_us": hip,
            "pm_us": pm,
            "speedup_median": speedup,
            "saving_vs_incumbent_pct": (1.0 - 1.0 / speedup) * 100.0,
            "order": "alternating",
        })
    } else {
        json!({ "incomplete": true, "samples": timing.hip.len() })
    };
    let (grid, block) = case.export.geometry(k, case.rows);
    let line = json!({
        "case": case.name(),
        "export": case.export.name(),
        "symbol": case.export.symbol(),
        "k": k, "rows": case.rows,
        "data": case.data.name(), "awq": case.awq,
        "pm_grid": grid, "pm_block": block,
        "sidecar_bytes": case.sidecar_bytes(),
        "rotated_f32_bytes": case.rot_bytes(),
        "reps": a.reps, "warmups": a.warmups,
        "first_run": first.json,
        "post_timing_run": post.json,
        "timing": timing_json,
        "deadline_exceeded": timing.deadline_hit,
        "all_byte_identical": identical,
    });
    if let Some(t) = weight {
        let _ = gpu.free_tensor(t);
    }
    if let Some(t) = awq {
        let _ = gpu.free_tensor(t);
    }
    CaseOut {
        line,
        identical,
        first_mismatch,
        deadline_hit: timing.deadline_hit,
    }
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

fn prerequisites() -> Vec<String> {
    vec![
        "exact gfx1201 (the oracle refuses any other arch with exit 2)".to_string(),
        format!("{FLAG_ENV}=0 is required: the route is qualified on exact gfx1201, the oracle launches the PM launchers explicitly and the HIP incumbent through the *_exact_batched methods; exit 2 otherwise"),
        format!("module {PM_MODULE} is embedded in the build (kernels/{PM_MODULE}.hxaco, include_bytes! in rdna-compute src/iu4_c8_pm.rs)"),
        "one GPU, single process, watchdog <= 120 s".to_string(),
    ]
}

fn limitations() -> Vec<&'static str> {
    vec![
        "launcher-level bytes only: does not read the PrefillIu4C8Pm capability, the production flag or the prefill.rs wiring",
        "the norm fold's normalized f32 row is compared bytewise; the plain norm twin emits none, so only the f32 rotated side-output and the sidecar exist for it",
        "inputs are finite f32 (no inf/NaN); weights, signs and AWQ scales are finite, AWQ positive without reciprocal planes",
        "the shared-down SwiGLU producer is not part of this twin (incumbent HIP stays); the sigmoid and RMSNorm folds are compared against the unfused chain on the HIP exact rotate",
        "timing includes host launch overhead with device_synchronize bracketing, not isolated kernel time",
        "the CPU model (--self-check) covers quantizer/index/pack only; it is cross-checked against the incumbent on 64 rows per case as a diagnostic",
    ]
}

fn plan(a: &Args) {
    let cases = case_matrix(a);
    println!(
        "{}",
        json!({
            "plan": true,
            "oracle": "iu4_c8_pm",
            "gpu_initialized": false,
            "byte_result": null,
            "note": "CPU plan only: no GPU was touched and nothing was compared; byte identity is a GPU outcome",
            "cases": cases.len(),
            "reps": a.reps, "min_reps": MIN_REPS, "warmups": a.warmups,
            "deadline_secs": a.deadline_secs,
            "module": PM_MODULE,
            "flag_env": FLAG_ENV,
            "flag_typed_key": FLAG_TYPED_KEY,
            "flag_env_current": std::env::var(FLAG_ENV).ok(),
            "flag_state_current": format!("{:?}", flag_state(std::env::var(FLAG_ENV).ok().as_deref())),
            "env_isolated": env_isolated(std::env::var(FLAG_ENV).ok().as_deref()),
            "kernarg_abi_bytes": PM_KERNARG_BYTES,
            "kernarg_abi": PM_KERNARG.iter().map(|(n, o, s)| json!({"name": n, "offset": o, "size": s})).collect::<Vec<_>>(),
            "prerequisites": prerequisites(),
            "limitations": limitations(),
        })
    );
    for c in &cases {
        let (grid, block) = c.export.geometry(c.k, c.rows);
        println!(
            "{}",
            json!({
                "plan_case": c.name(),
                "export": c.export.name(),
                "symbol": c.export.symbol(),
                "incumbent": c.export.hip_incumbent(),
                "pm_launcher": c.export.pm_launcher(),
                "k": c.k, "rows": c.rows,
                "data": c.data.name(), "awq": c.awq,
                "pm_grid": grid, "pm_block": block,
                "sidecar_bytes": c.sidecar_bytes(),
                "rotated_f32_bytes": c.rot_bytes(),
                "compared": c.export.compared(),
            })
        );
    }
    println!(
        "{}",
        json!({ "plan_summary": true, "cases": cases.len(), "all_byte_identical": null })
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

    // ---- Case matrix -------------------------------------------------
    let all = parse_args(&[]).expect("default args");
    let cases = case_matrix(&all);
    check!(
        cases.len() == 144,
        "default matrix has {} cases",
        cases.len()
    );
    let mut names: Vec<String> = cases.iter().map(Case::name).collect();
    names.sort();
    names.dedup();
    check!(names.len() == cases.len(), "case names are not unique");
    let count = |f: &dyn Fn(&Case) -> bool| cases.iter().filter(|c| f(c)).count();
    check!(count(&|c| c.export == Export::Rotate) == 48, "rotate cases");
    check!(count(&|c| c.export == Export::Norm) == 24, "norm cases");
    check!(
        count(&|c| c.export == Export::SigmoidRotate) == 48,
        "sigmoid_rotate cases"
    );
    check!(
        count(&|c| c.export == Export::NormOut) == 24,
        "norm_out cases"
    );
    check!(
        count(&|c| c.export == Export::Rotate && c.k == 4096) == 24,
        "rotate K4096 cases"
    );
    check!(
        count(&|c| c.export == Export::SigmoidRotate && c.k == 4096) == 24,
        "sigmoid_rotate K4096 cases"
    );
    check!(
        count(&|c| c.rows == 7808) == 36 && count(&|c| c.rows == 8192) == 36,
        "the two A3B prefill chunk row counts"
    );
    check!(
        cases.iter().all(|c| !c.export.is_norm() || c.k == 2048),
        "norm exports only K2048"
    );
    check!(
        cases
            .iter()
            .all(|c| c.export.supports_k(c.k) && c.k % 256 == 0),
        "export/K support"
    );
    for r in ROWS {
        for d in Data::ALL {
            for aw in [false, true] {
                check!(
                    count(&|c| c.rows == r && c.data == d && c.awq == aw) == 6,
                    "rows {r} {} awq {aw}: rotate and sigmoid_rotate K2048/K4096 + norm and norm_out K2048",
                    d.name()
                );
            }
        }
    }
    // Group contiguity: consecutive cases with one (rows,data,k) key.
    let mut seen = std::collections::HashSet::new();
    let mut last = None;
    for c in &cases {
        let key = (c.rows, c.data as usize, c.k);
        if last != Some(key) {
            check!(seen.insert(key), "input group {key:?} not contiguous");
            last = Some(key);
        }
    }
    check!(
        rdna_compute::scratch::int4_mmq_reservation_bytes(4096, 8192)
            == Some(cases.iter().map(Case::sidecar_bytes).max().unwrap()),
        "max sidecar reservation bytes"
    );
    for c in &cases {
        check!(
            rdna_compute::scratch::int4_mmq_reservation_bytes(c.k, c.rows)
                == Some(c.sidecar_bytes()),
            "{}: reservation bytes",
            c.name()
        );
    }

    // ---- Geometry / ABI ----------------------------------------------
    check!(
        Export::Rotate.geometry(2048, 512) == (512 * 8, 32)
            && Export::Rotate.geometry(4096, 8192) == (8192 * 16, 32)
            && Export::Norm.geometry(2048, 4096) == (4096, 256)
            && Export::SigmoidRotate.geometry(4096, 7808) == (7808 * 16, 32)
            && Export::NormOut.geometry(2048, 7808) == (7808, 256),
        "launch geometry"
    );
    check!(
        Export::Rotate.symbol() == "iu4_c8_rotate_pm_gfx1201"
            && Export::Norm.symbol() == "iu4_c8_rmsnorm_rotate_pm_gfx1201"
            && Export::SigmoidRotate.symbol() == "iu4_c8_sigmoid_rotate_pm_gfx1201"
            && Export::NormOut.symbol() == "iu4_c8_rmsnorm_out_rotate_pm_gfx1201",
        "export symbols"
    );
    {
        let mut end = 0usize;
        let mut contiguous = true;
        for (name, off, size) in PM_KERNARG {
            contiguous &= off == end || (name == "eps" && off == 64);
            contiguous &= off % size == 0;
            end = off + size;
        }
        check!(
            contiguous,
            "kernarg fields are naturally aligned and contiguous"
        );
        check!(
            end == 68 && PM_KERNARG_BYTES == 72,
            "kernarg ends at 68 + 4 padding = 72"
        );
        let names: Vec<&str> = PM_KERNARG.iter().map(|f| f.0).collect();
        check!(
            names
                == [
                    "x",
                    "rotated_out",
                    "sidecar",
                    "signs1",
                    "signs2",
                    "weight",
                    "awq",
                    "rows",
                    "k",
                    "eps"
                ],
            "kernarg order"
        );
    }

    // ---- Flag expectations -------------------------------------------
    check!(
        flag_state(None) == FlagState::Default,
        "unset follows qualification"
    );
    check!(flag_state(Some("1")) == FlagState::On, "1 opts in");
    check!(flag_state(Some("0")) == FlagState::Killed, "0 kills");
    check!(
        flag_state(Some("")) == FlagState::Default,
        "empty is the default"
    );
    check!(
        !env_isolated(None) && env_isolated(Some("0")) && !env_isolated(Some("1")),
        "oracle runs only with the explicit kill"
    );
    check!(
        FLAG_ENV == "HIPFIRE_PREFILL_IU4_C8_PM" && FLAG_TYPED_KEY == "developer.prefill_iu4_c8_pm",
        "flag names"
    );

    // ---- Record layout -----------------------------------------------
    check!(
        D_RANGE.1 - D_RANGE.0 == 4 && S_RANGE.1 - S_RANGE.0 == 4 && QS_RANGE.1 - QS_RANGE.0 == 64,
        "record field widths"
    );
    check!(
        D_RANGE.1 == S_RANGE.0 && S_RANGE.1 == QS_RANGE.0 && QS_RANGE.1 == RECORD_BYTES,
        "record fields tile the 72 B record"
    );

    // ---- Remap / ownership / record index (full) ---------------------
    let mut covered = [[false; 256]; 1];
    let mut remap_ok = true;
    for h in 0..2 {
        for l in 0..32 {
            for e in 0..4 {
                let (src, fidx, elem) = remap(h, l, e);
                remap_ok &= src < 32 && fidx < 8 && elem < 256;
                // quantizer lane l of half h owns 128h + 4l + e
                remap_ok &= elem == 128 * h + 4 * l + e;
                // producer lane p owns 8p..8p+7
                remap_ok &= elem / 8 == src && elem % 8 == fidx;
                remap_ok &= src / 16 == h;
                // parity: even lane first float4, odd lane second
                remap_ok &= (fidx >= 4) == (l & 1 == 1);
                remap_ok &= !covered[0][elem];
                covered[0][elem] = true;
            }
        }
    }
    check!(
        remap_ok && covered[0].iter().all(|c| *c),
        "remap is a bijection onto the 256-group"
    );
    for n in [1usize, 3, 512] {
        for k in KS {
            let groups = k / 256;
            let mut seen = vec![false; (k / 128) * n];
            let mut ok = true;
            for g in 0..groups {
                for h in 0..2 {
                    for t in 0..n {
                        let r = record_index(g, h, t, n);
                        ok &= r < seen.len() && !seen[r];
                        // token-order stream: kblock = 2*group+half.
                        ok &= r / n == 2 * g + h && r % n == t;
                        seen[r] = true;
                    }
                }
            }
            check!(
                ok && seen.iter().all(|s| *s),
                "record index bijection k={k} n={n}"
            );
        }
    }

    // ---- Candidate constants -----------------------------------------
    check!(C8_CANDIDATES == 8, "eight candidates");
    for amax in [1.0f32, 0.375, 17.0, 1.0e-3, 12345.0] {
        let ds: Vec<f32> = (0..8).map(|j| c8_candidate(amax, j)).collect();
        check!(
            ds.windows(2).all(|w| w[0] < w[1]),
            "candidates strictly increase for amax {amax}"
        );
        check!(
            ds[0] == (amax / 7.0) * 0.5,
            "candidate 0 = amax/14 for {amax}"
        );
        check!(ds[7] == amax / 7.0, "candidate 7 = amax/7 for {amax}");
        for (j, d) in ds.iter().enumerate() {
            let want = (amax / 7.0) * 0.5 * (1.0 + (j as f32) / 7.0);
            check!(
                d.to_bits() == want.to_bits(),
                "candidate {j} formation for {amax}"
            );
        }
    }
    check!(
        rint(0.5) == 0.0
            && rint(1.5) == 2.0
            && rint(2.5) == 2.0
            && rint(-0.5) == 0.0
            && rint(-1.5) == -2.0,
        "rint ties-to-even"
    );
    check!(
        c8_code(100.0, 1.0) == 7.0 && c8_code(-100.0, 1.0) == -8.0 && c8_code(-7.9, 1.0) == -8.0,
        "clamp -8..7"
    );
    // xor butterfly: every lane ends with the same total, order-stable.
    let mut v = [0.0f32; 32];
    for (i, x) in v.iter_mut().enumerate() {
        *x = (i as f32 + 1.0) * 0.25;
    }
    let r = xor_sum32(v);
    check!(
        r.iter().all(|x| *x == r[0]) && r[0] == 132.0,
        "xor butterfly sum {}",
        r[0]
    );

    // ---- Quantizer model ---------------------------------------------
    let zero = [0.0f32; 128];
    let rz = quantize_block_c8(&zero);
    check!(
        f32::from_le_bytes([rz[0], rz[1], rz[2], rz[3]]) == 1.0
            && i32::from_le_bytes([rz[4], rz[5], rz[6], rz[7]]) == 0
            && rz[8..].iter().all(|b| *b == 0),
        "amax 0 forces d=1, s=0, codes 0"
    );
    let mut negz = [0.0f32; 128];
    negz[3] = -0.0;
    check!(quantize_block_c8(&negz) == rz, "negative zero is amax 0");
    // Low nibble = even element.
    let mut one = [0.0f32; 128];
    one[0] = 7.0;
    one[1] = -8.0;
    one[2] = 3.0;
    one[127] = -1.0;
    let r1 = quantize_block_c8(&one);
    let d1 = f32::from_le_bytes([r1[0], r1[1], r1[2], r1[3]]);
    check!(d1 > 0.0 && d1.is_finite(), "d positive finite");
    let dec = |rec: &[u8; RECORD_BYTES], i: usize| -> i32 {
        let byte = rec[8 + i / 2];
        let nib = (if i % 2 == 0 { byte & 15 } else { byte >> 4 }) as i32;
        if nib >= 8 {
            nib - 16
        } else {
            nib
        }
    };
    let s1 = i32::from_le_bytes([r1[4], r1[5], r1[6], r1[7]]);
    check!(
        s1 == (0..128).map(|i| dec(&r1, i)).sum::<i32>(),
        "s is the exact integer sum of decoded nibbles"
    );
    check!(
        (0..128).all(|i| dec(&r1, i) == c8_code(one[i], d1) as i32),
        "codes requantized from best_d"
    );
    check!(
        dec(&r1, 0) != 0 && dec(&r1, 127) <= 0 && (0..128).all(|i| (-8..=7).contains(&dec(&r1, i))),
        "decoded range"
    );
    // Pack positions: byte lane*2 = elements 4l (low),4l+1 (high); byte lane*2+1 = 4l+2,4l+3.
    let mut ramp = [0.0f32; 128];
    for (i, x) in ramp.iter_mut().enumerate() {
        *x = (i % 16) as f32 - 8.0 + 0.0;
    }
    let rr = quantize_block_c8(&ramp);
    let dr = f32::from_le_bytes([rr[0], rr[1], rr[2], rr[3]]);
    check!(
        (0..128).all(|i| dec(&rr, i) == c8_code(ramp[i], dr) as i32),
        "pack/decode round trip over every lane"
    );
    for l in 0..32 {
        let q: Vec<i32> = (0..4).map(|e| dec(&rr, 4 * l + e)).collect();
        check!(
            rr[8 + 2 * l] == ((q[0] & 15) | ((q[1] & 15) << 4)) as u8
                && rr[9 + 2 * l] == ((q[2] & 15) | ((q[3] & 15) << 4)) as u8,
            "lane {l} nibble packing"
        );
    }
    // Selection: first-argmin with strict <, d is one of the eight candidates.
    let mut rng = Rng(7);
    let mut sel_ok = true;
    let mut seen_j = [false; 8];
    for t in 0..400 {
        let mut x = [0.0f32; 128];
        for v in x.iter_mut() {
            *v = rng.unit() * (1.0 + (t % 9) as f32);
        }
        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let (j, d) = c8_select(&x);
        let j = j.expect("nonzero block selects a candidate");
        sel_ok &= d.to_bits() == c8_candidate(amax, j).to_bits();
        let mse_of = |jj: usize| {
            let dj = c8_candidate(amax, jj);
            let mut lanes = [0.0f32; 32];
            for (l, m) in lanes.iter_mut().enumerate() {
                *m = lane_mse(&x[4 * l..4 * l + 4], dj);
            }
            xor_sum32(lanes)[0]
        };
        let best = mse_of(j);
        sel_ok &= (0..j).all(|e| mse_of(e) > best) && (j + 1..8).all(|e| mse_of(e) >= best);
        seen_j[j] = true;
    }
    check!(sel_ok, "first strict argmin over the eight candidates");
    check!(
        seen_j.iter().filter(|s| **s).count() >= 3,
        "random blocks exercise several candidates"
    );
    // All-equal tie: identical candidates would retain the first; here the
    // amax=1 constant block has distinct candidates, so just require
    // determinism of the full record.
    check!(quantize_block_c8(&ramp) == rr, "model is deterministic");

    // ---- Synthetic data ----------------------------------------------
    check!(
        pow2(0) == 1.0 && pow2(-3) == 0.125 && pow2(10) == 1024.0,
        "pow2"
    );
    check!(
        synth_x(4, 8, 1, Data::Zero)
            .iter()
            .all(|v| v.to_bits() == 0),
        "zero data is +0.0"
    );
    let rnd = synth_x(16, 64, 1, Data::Random);
    check!(
        rnd.iter().any(|v| *v != 0.0) && rnd.iter().all(|v| v.is_finite()),
        "random data"
    );
    let lg = synth_x(40, 256, 9, Data::Large);
    check!(lg.iter().all(|v| v.is_finite()), "large data finite");
    check!(
        lg == synth_x(40, 256, 9, Data::Large),
        "large data deterministic"
    );
    check!(
        lg[3 * 256..4 * 256].iter().all(|v| v.to_bits() == 0),
        "large row 3 is all +0.0"
    );
    check!(
        lg[26 * 256..27 * 256].iter().filter(|v| **v != 0.0).count() == 1,
        "large row 26 (r%13==0) is a single spike"
    );
    check!(
        lg[7 * 256..8 * 256]
            .iter()
            .any(|v| v.to_bits() == (-0.0f32).to_bits()),
        "large row 7 (r%5==2) carries negative zeros"
    );
    let nz: Vec<f32> = lg.iter().filter(|v| **v != 0.0).map(|v| v.abs()).collect();
    let (mx, mn) = nz
        .iter()
        .fold((0.0f32, f32::MAX), |(a, b), v| (a.max(*v), b.min(*v)));
    check!(
        mx / mn > pow2(40) && mx < pow2(41) && mn >= pow2(-41),
        "large dynamic range {mn:e}..{mx:e}"
    );
    // Squares of the largest values over K=4096 stay finite.
    check!((mx * mx * 4096.0).is_finite(), "large data squares finite");
    for d in Data::ALL {
        let w = synth_pos(512, 5, 0.5, 1.0, d, 8);
        check!(
            w.iter().all(|v| v.is_finite() && *v > 0.0),
            "{} positive vec",
            d.name()
        );
        let a = synth_pos(512, 5, 0.5, 1.5, d, 6);
        check!(
            a.iter().all(|v| v.is_finite() && *v > 0.0),
            "{} awq positive",
            d.name()
        );
    }
    check!(
        group_seed(2048, 512, Data::Zero) != group_seed(4096, 512, Data::Zero)
            && group_seed(2048, 512, Data::Zero) != group_seed(2048, 4096, Data::Zero)
            && group_seed(2048, 512, Data::Zero) != group_seed(2048, 512, Data::Random),
        "group seeds"
    );

    // ---- CPU model vs sidecar helper ---------------------------------
    {
        let (k, n) = (256usize, 3usize);
        let mut rng = Rng(11);
        let rot: Vec<f32> = (0..n * k).map(|_| rng.unit()).collect();
        let rot_bytes: Vec<u8> = rot.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut side = vec![0u8; (k / 128) * n * RECORD_BYTES];
        for t in 0..n {
            for kb in 0..k / 128 {
                let mut blk = [0.0f32; 128];
                blk.copy_from_slice(&rot[t * k + kb * 128..t * k + kb * 128 + 128]);
                let at = (kb * n + t) * RECORD_BYTES;
                side[at..at + RECORD_BYTES].copy_from_slice(&quantize_block_c8(&blk));
            }
        }
        check!(
            cpu_model_vs_sidecar(&rot_bytes, &side, k, n, 64).is_none(),
            "model matches own sidecar"
        );
        side[(n + 2) * RECORD_BYTES + 20] ^= 1;
        check!(
            cpu_model_vs_sidecar(&rot_bytes, &side, k, n, 64) == Some((1, 2)),
            "model locates kblock 1 token 2"
        );
        check!(
            cpu_model_vs_sidecar(&rot_bytes, &side, k, n, 2).is_none(),
            "take limit excludes token 2"
        );
    }

    // ---- First-mismatch / locate / compare ---------------------------
    check!(
        first_mismatch(&[1, 2, 3], &[1, 2, 3]).is_none(),
        "equal slices"
    );
    check!(
        first_mismatch(&[1, 2, 3], &[1, 9, 3]) == Some(1),
        "first diff index"
    );
    check!(
        first_mismatch(&[1, 2], &[1, 2, 3]) == Some(2),
        "prefix length mismatch"
    );
    check!(first_mismatch(&[], &[]).is_none(), "empty slices");
    let n = 4usize;
    let mut a = vec![0u8; 2 * n * RECORD_BYTES];
    for (field_off, field) in [
        (0usize, "d"),
        (3, "d"),
        (4, "s"),
        (7, "s"),
        (8, "payload"),
        (71, "payload"),
    ] {
        let mut b = a.clone();
        let rec = n + 2; // kblock 1, token 2
        b[rec * RECORD_BYTES + field_off] ^= 0x10;
        let cmp = compare_sidecar(&a, &b, n);
        let fm = &cmp.json["first_mismatch"];
        check!(!cmp.equal, "{field}@{field_off}: must differ");
        check!(
            fm["field"] == field
                && fm["kblock"] == 1
                && fm["token"] == 2
                && fm["offset_in_record"] == field_off,
            "locate {field}@{field_off}: {fm}"
        );
        check!(
            cmp.json["d_equal"] == (field != "d")
                && cmp.json["s_equal"] == (field != "s")
                && cmp.json["payload_equal"] == (field != "payload")
                && cmp.json["mismatching_records"] == 1,
            "per-field flags {field}@{field_off}"
        );
        if field == "payload" {
            check!(
                fm["k_element_in_block"] == 2 * (field_off - 8),
                "payload element locate {fm}"
            );
        }
    }
    a[0] = 1;
    check!(
        compare_sidecar(&a, &a.clone(), n).equal,
        "identical sidecars compare equal"
    );
    let o = rot_locate(4 * (3 * 2048 + 8 * 5 + 2) + 2, 2048);
    check!(
        o["row"] == 3
            && o["col"] == 42
            && o["byte_in_f32"] == 2
            && o["group256"] == 0
            && o["producer_lane"] == 5,
        "rot_locate {o}"
    );
    let o = rot_locate(4 * (1 * 4096 + 3000), 4096);
    check!(
        o["row"] == 1 && o["col"] == 3000 && o["group256"] == 11 && o["producer_lane"] == 23,
        "rot_locate group {o}"
    );

    // ---- Stats --------------------------------------------------------
    let st = stats(&[7.0, 1.0, 11.0, 3.0, 9.0, 2.0, 10.0, 4.0, 6.0, 8.0, 5.0]);
    check!(
        st["median"] == 6.0 && st["p10"] == 2.0 && st["p90"] == 10.0 && st["samples"] == 11,
        "stats {st}"
    );

    // ---- Arg parsing --------------------------------------------------
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
        parse_args(&s(&["--rows", "64"])).is_err(),
        "rows outside matrix rejected"
    );
    check!(
        parse_args(&s(&["--k", "512"])).is_err(),
        "k outside matrix rejected"
    );
    check!(
        parse_args(&s(&["--bogus"])).is_err(),
        "unknown arg rejected"
    );
    check!(
        parse_args(&s(&["--exports", "norm", "--k", "4096"])).is_err(),
        "norm K4096 selects nothing"
    );
    check!(
        case_matrix(
            &parse_args(&s(&[
                "--exports",
                "rotate",
                "--k",
                "4096",
                "--rows",
                "512",
                "--data",
                "large",
                "--awq",
                "some"
            ]))
            .unwrap()
        )
        .len()
            == 1,
        "filters narrow the matrix"
    );
    check!(
        parse_args(&s(&["--data", "large", "--exports", "norm"]))
            .map(|a| case_matrix(&a).len())
            .ok()
            == Some(ROWS.len() * 2),
        "norm large = every row count x 2 awq"
    );

    let failed = failures.len();
    for f in &failures {
        eprintln!("self-check FAIL: {f}");
    }
    println!(
        "{}",
        json!({
            "self_check": true,
            "gpu_initialized": false,
            "checks": checks,
            "failures": failed,
            "byte_result": null,
            "note": "CPU self-check validates the case matrix, ABI table, flag expectations, ownership remap, record index, C8 candidate constants, packing and the compare helpers only; it proves nothing about GPU bytes",
        })
    );
    if failed != 0 {
        std::process::exit(1);
    }
}

fn run(a: &Args) {
    let start = Instant::now();
    let deadline = start + Duration::from_secs(a.deadline_secs);
    let secs = a.deadline_secs;
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(secs));
        eprintln!("iu4_c8_pm_oracle: {secs}s deadline exceeded (watchdog); exiting 3");
        std::process::exit(3);
    });

    let env = std::env::var(FLAG_ENV).ok();
    if !env_isolated(env.as_deref()) {
        eprintln!(
            "iu4_c8_pm_oracle: {FLAG_ENV} must be 0 (the route is qualified on exact gfx1201, \
             so unset or 1 lets the HIP incumbent route to PM; the oracle launches PM explicitly)"
        );
        std::process::exit(2);
    }

    let mut gpu = Gpu::init().expect("gpu init");
    let cases = case_matrix(a);
    if gpu.arch != "gfx1201" {
        eprintln!(
            "iu4_c8_pm_oracle: arch {} is not gfx1201; refusing to compare",
            gpu.arch
        );
        std::process::exit(2);
    }

    let mut group: Option<Group> = None;
    let mut identical_cases = 0usize;
    let mut completed = 0usize;
    let mut first_mismatch: Option<Value> = None;
    let mut deadline_hit = false;
    for case in &cases {
        if Instant::now() >= deadline {
            deadline_hit = true;
            break;
        }
        let key = (case.rows, case.data, case.k);
        if group.as_ref().map(|g| g.key) != Some(key) {
            if let Some(old) = group.take() {
                free_group(&mut gpu, old);
            }
            group = Some(make_group(&mut gpu, case));
        }
        let out = run_case(&mut gpu, case, group.as_ref().unwrap(), a, deadline);
        println!("{}", out.line);
        completed += 1;
        if out.identical {
            identical_cases += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = out
                .first_mismatch
                .map(|m| json!({ "case": case.name(), "mismatch": m }));
        }
        if out.deadline_hit {
            deadline_hit = true;
            break;
        }
    }
    if let Some(old) = group.take() {
        free_group(&mut gpu, old);
    }
    let all_identical = !deadline_hit && completed == cases.len() && identical_cases == completed;
    println!(
        "{}",
        json!({
            "summary": true,
            "oracle": "iu4_c8_pm",
            "arch": gpu.arch,
            "module": PM_MODULE,
            "flag_env": env,
            "cases_planned": cases.len(),
            "cases_completed": completed,
            "cases_identical": identical_cases,
            "reps": a.reps, "warmups": a.warmups,
            "deadline_secs": a.deadline_secs,
            "elapsed_secs": start.elapsed().as_secs_f64(),
            "deadline_exceeded": deadline_hit,
            "first_mismatch": first_mismatch,
            "all_byte_identical": all_identical,
        })
    );
    if deadline_hit {
        eprintln!("iu4_c8_pm_oracle: deadline reached before the matrix completed");
        std::process::exit(3);
    }
    if !all_identical {
        eprintln!(
            "iu4_c8_pm_oracle: {} of {} case(s) not byte-identical",
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
            eprintln!("iu4_c8_pm_oracle: {e}");
            eprintln!(
                "usage: [--plan|--self-check] [--exports rotate,norm] [--k 2048,4096] \
                 [--rows 512,4096,8192] [--data zero,random,large] [--awq none,some] \
                 [--reps N>=20] [--warmups N] [--deadline-secs 1..=120]"
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
