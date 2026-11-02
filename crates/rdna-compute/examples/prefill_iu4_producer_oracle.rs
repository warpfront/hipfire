// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Byte oracle and timing for the A3B prefill IU4 producer fuse (F3, gfx1201,
//! lab): the reference8 ("exact") gfx12 producers
//! `Gpu::fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched`,
//! `Gpu::rotate_x_mq_i4_gfx12_exact_batched` and
//! `Gpu::fused_silu_mul_rotate_mq_i4_gfx12_exact_batched` feeding the existing
//! prepared IU4 consumers, against the incumbent F32 producer + the GEMM
//! wrapper that runs the standalone `quantize_int4_mmq_ds128` internally.
//!
//! ## What is compared
//!
//! For every case TWO things are compared bytewise, never one alone:
//!
//! 1. the IU4 GEMM INPUT: the `block_i4_128` sidecar (72 B record per
//!    `[K/128 block, row]`: f32 `d` scale at 0..4, `s` sum at 4..8, 64 B packed
//!    nibbles at 8..72) the consumer actually reads. It is snapshotted from the
//!    live `int4_mmq_x_scratch` right after each arm's whole sequence:
//!    after the incumbent the scratch holds what the incumbent GEMM wrapper's
//!    own `ensure_int4_mmq_x` wrote (the incumbent must have bumped the scratch
//!    generation, else it did not take the IU4 route and the oracle aborts);
//!    after the new arm it holds the producer's records. `d`, `s` and `qs` are
//!    reported separately and the first mismatching record is located
//!    (`kblock`, `token`, `field`, byte offset). A slab-layout sidecar (the
//!    `HIPFIRE_A4_SLAB` twins; `Gpu::scratch.int4_mmq_slab_at`) is converted to
//!    token-order records before the compare and reported as `layout:"slab"`.
//! 2. the GEMM OUTPUT of the SAME unchanged consumer family: F32 output buffers,
//!    bit-for-bit.
//!
//! `all_byte_identical` is true only when the input sidecar AND every output
//! buffer match, both on the first run and again after the timing loop (one
//! more untimed pass of both arms, so the shared scratch is re-snapshotted).
//! Byte identity is a GPU outcome: nothing on the CPU (`--plan`,
//! `--self-check`) prints a pass, and a refused / aborted run never reports
//! one.
//!
//! The diagnostic field `legacy_input_identical` re-runs the legacy compiled
//! (`IU4_A4_CANDIDATES`, default C2) producer on the same input and compares
//! its sidecar to the standalone quantizer. It is information only (the legacy
//! producer is C2/RTN by construction and is expected NOT to match on
//! gfx1201); it never contributes to `all_byte_identical`.
//!
//! ## Sites (A3B / Qwen3.6-35B-A3B geometry; hidden 2048, 16 FA heads of 256,
//! 2 KV heads, 16 K / 32 V linear heads of 128, shared expert 512)
//!
//! | site | producer | consumer (incumbent -> new) |
//! |---|---|---|
//! | `gdn_qkvza` K2048 | RMSNorm+FWHT | `gemm_qkvza_hfq4g256_mq4v2` (qkv 8192, z 4096, beta 32, alpha 32, SET) -> `gemm_qkvza_mq4g256v2_wmma_iu4_prepared` |
//! | `fa_qkv` K2048 | RMSNorm+FWHT | `gemm_qkv_hfq4g256_mq4v2` (q 8192, k 512, v 512, SET) -> `gemm_qkv_mq4g256v2_wmma_iu4_prepared` |
//! | `fa_out` K4096 M2048 | FWHT rotate | `gemm_hfq4g256_residual_mq4v2` (ADD into a nonzero residual) -> `gemm_mq4g256v2_residual_wmma_iu4_prepared` |
//! | `shared_down` K512 M2048 | SwiGLU+FWHT | `gemm_mq4g256v2` (zero + ADD) -> `gemm_mq4g256v2_prepared_iu4` (zero + ADD on the token) |
//! | `router_shared` K2048 | FWHT rotate of the normalized f32 | existing one-quantizer route `try_gemm_mq4g256v2_shared_router_iu4` (router zero + ADD) + `try_gemm_mq4g256v2_shared_iu4(.., Some(token))` (selector zero + ADD, gate/up SET) -> exact rotate, router `gemm_mq4g256v2_prepared_iu4`, then `try_gemm_mq4g256v2_shared_iu4(.., Some(prep))` |
//!
//! `router_shared`'s incumbent is the EXISTING one-quantizer shared prepared
//! route (`try_gemm_mq4g256v2_shared_router_iu4` + group helper: ONE standalone
//! quantizer), the profile baseline. The generic three-quantizer sequence (router
//! `gemm_mq4g256v2`, selector `gemm_mq4g256v2`, `gemm_gate_up_hfq4g256_mq4v2`) is
//! an extra diagnostic arm: byte-compared and timed against that baseline, and
//! reported as `three_quantizer_over_one_quantizer_median`; the reported saving
//! (`saving_vs_incumbent_pct`) is always against the one-quantizer baseline.
//!
//! Matrix: 5 sites x rows {512, 4096} x data {zero, random} x AWQ {none, some}
//! = 40 cases. `zero` activations make every amax 0 (d = 1 path); `random`
//! activations use a per-row scale `0.25 * (1 + row % 5)`. AWQ scales are
//! length-K positive F32 WITHOUT reciprocal planes (AoS, non-slab twins).
//!
//! ## Scope limits (see `--plan`)
//!
//! * The oracle calls the launchers directly. It does not read
//!   `HIPFIRE_PREFILL_IU4_PRODUCER_FUSED` and does not exercise the
//!   `PrefillIu4ProducerFused` capability or the prefill.rs wiring.
//! * The new `gdn_qkvza` consumer is the existing prepared qkvza consumer; on
//!   exact gfx1201 it runs beta/alpha as plain MMQ SET launches, whereas the
//!   incumbent wrapper first calls `gemm_mq4g256v2_small_tail_set_iu4` (zeroed
//!   Y + the gfx11 MW4 tail attempt, then the same MMQ SET fallback). The byte
//!   comparison of the beta/alpha outputs is what proves them equal.
//! * Weights are random but valid MQ4G256V2 (same generator as
//!   `moe_shared_prepared_iu4_oracle`).
//!
//! ## Timing
//!
//! `--reps N` (default 30, minimum 20) + 3 warmups per arm, arms interleaved.
//! Each rep: output init (sentinel `0xA5`, or the pristine residual for
//! `fa_out`) + `device_synchronize` (untimed), then `Instant::now()`, the arm's
//! WHOLE sequence (F32/legacy producer or fused producer, standalone quantizer
//! included in the incumbent, every consumer launch), `device_synchronize`,
//! elapsed. Median / p10 / p90 in microseconds (nearest-rank percentiles on
//! the sorted samples; median = mean of the two middle samples).
//!
//! ## Deadline
//!
//! One process, at most 120 s (`--deadline-secs N`, 1..=120): cooperative
//! checks between reps/cases plus a watchdog thread that exits 3 even if a GPU
//! call hangs. A run that hit the deadline prints the summary with
//! `deadline_exceeded:true` and `all_byte_identical:false`.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab \
//!     --example prefill_iu4_producer_oracle -- --plan
//! cargo run --release -p rdna-compute --features lab \
//!     --example prefill_iu4_producer_oracle -- --self-check
//! cargo run --release -p rdna-compute --features lab \
//!     --example prefill_iu4_producer_oracle
//! # filters (optional): --sites gdn_qkvza,fa_out --rows 512 --data random --awq some
//! ```
//!
//! stdout (GPU mode) = one JSON line per case then the summary line
//! `{"summary":true,...,"all_byte_identical":bool,"first_mismatch":{...}|null}`.
//! Diagnostics go to stderr. Exit 0 when every case is byte-identical, 1 on
//! any mismatch (after the summary), 2 on bad arguments / unsupported
//! configuration (before any case), 3 on deadline.

use half::f16;
use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor, Int4MmqPrepared};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const DEFAULT_REPS: usize = 30;
const MIN_REPS: usize = 20;
const DEFAULT_WARMUPS: usize = 3;
const MAX_DEADLINE_SECS: u64 = 120;
/// MQ4G256V2 bytes per 256-wide group: 8 B fp16 header + 128 B nibbles.
const GROUP_BYTES: usize = 136;
/// `block_i4_128` record: f32 d, f32 s, 64 B nibbles.
const RECORD_BYTES: usize = 72;
const D_RANGE: (usize, usize) = (0, 4);
const S_RANGE: (usize, usize) = (4, 8);
const QS_RANGE: (usize, usize) = (8, 72);
/// Nonzero fill for every output before each launch.
const SENTINEL: u8 = 0xA5;
const EPS: f32 = 1.0e-6;
const ROWS: [usize; 2] = [512, 4096];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Site {
    GdnQkvza,
    FaQkv,
    FaOut,
    SharedDown,
    RouterShared,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Producer {
    /// RMSNorm + FWHT of the residual `x`.
    Norm,
    /// FWHT of an already formed row (FA attention output / normalized x).
    Rotate,
    /// SwiGLU(gate, up) + FWHT.
    Silu,
}

impl Site {
    const ALL: [Site; 5] = [
        Site::GdnQkvza,
        Site::FaQkv,
        Site::FaOut,
        Site::SharedDown,
        Site::RouterShared,
    ];

    fn name(self) -> &'static str {
        match self {
            Site::GdnQkvza => "gdn_qkvza",
            Site::FaQkv => "fa_qkv",
            Site::FaOut => "fa_out",
            Site::SharedDown => "shared_down",
            Site::RouterShared => "router_shared",
        }
    }

    fn from_name(s: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.name() == s)
    }

    /// GEMM reduction width (the producer width).
    fn k(self) -> usize {
        match self {
            Site::GdnQkvza | Site::FaQkv | Site::RouterShared => 2048,
            Site::FaOut => 4096,
            Site::SharedDown => 512,
        }
    }

    fn producer(self) -> Producer {
        match self {
            Site::GdnQkvza | Site::FaQkv => Producer::Norm,
            Site::FaOut | Site::RouterShared => Producer::Rotate,
            Site::SharedDown => Producer::Silu,
        }
    }

    /// `(name, m)` per weight / output buffer, in the index order the slot
    /// constants below use.
    fn outputs(self) -> &'static [(&'static str, usize)] {
        match self {
            Site::GdnQkvza => &[("qkv", 8192), ("z", 4096), ("beta", 32), ("alpha", 32)],
            Site::FaQkv => &[("q_full", 8192), ("k", 512), ("v", 512)],
            Site::FaOut => &[("out_residual", 2048)],
            Site::SharedDown => &[("down", 2048)],
            Site::RouterShared => &[
                ("shared_gate", 512),
                ("shared_up", 512),
                ("shared_selector", 1),
                ("router", 256),
            ],
        }
    }

    fn incumbent_sequence(self) -> &'static [&'static str] {
        match self {
            Site::GdnQkvza => &[
                "fused_rmsnorm_rotate_mq[_awq]_batched (f32 x_rot)",
                "gemm_qkvza_hfq4g256_mq4v2 (standalone quantize_int4_mmq_ds128, 4x MMQ SET; beta/alpha via small_tail_set_iu4)",
            ],
            Site::FaQkv => &[
                "fused_rmsnorm_rotate_mq[_awq]_batched (f32 x_rot)",
                "gemm_qkv_hfq4g256_mq4v2 (standalone quantize_int4_mmq_ds128, 3x MMQ SET)",
            ],
            Site::FaOut => &[
                "rotate_x_mq[_awq]_batched (f32 x_rot)",
                "gemm_hfq4g256_residual_mq4v2 (standalone quantize_int4_mmq_ds128, MMQ ADD into residual y)",
            ],
            Site::SharedDown => &[
                "fused_silu_mul_rotate_mq[_awq]_batched (f32 x_rot)",
                "gemm_mq4g256v2 (memset y, standalone quantize_int4_mmq_ds128, MMQ ADD)",
            ],
            Site::RouterShared => &[
                "rotate_x_mq[_awq]_batched (f32 x_rot from the normalized x)",
                "try_gemm_mq4g256v2_shared_router_iu4 (router memset + ONE standalone quantizer + ADD, returns token)",
                "try_gemm_mq4g256v2_shared_iu4([selector, gate, up], .., Some(token)) (selector zero+ADD, gate/up SET)",
                "diagnostic only, not the baseline: 3-quantizer generic stage (gemm_mq4g256v2 router, gemm_mq4g256v2 selector, gemm_gate_up_hfq4g256_mq4v2)",
            ],
        }
    }

    fn new_sequence(self) -> &'static [&'static str] {
        match self {
            Site::GdnQkvza => &[
                "reserve_int4_mmq",
                "fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched (reference8 sidecar, no f32 store)",
                "gemm_qkvza_mq4g256v2_wmma_iu4_prepared (4x MMQ SET on the sidecar)",
            ],
            Site::FaQkv => &[
                "reserve_int4_mmq",
                "fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched (reference8 sidecar, no f32 store)",
                "gemm_qkv_mq4g256v2_wmma_iu4_prepared (3x MMQ SET on the sidecar)",
            ],
            Site::FaOut => &[
                "reserve_int4_mmq",
                "rotate_x_mq_i4_gfx12_exact_batched (reference8 sidecar, no f32 store)",
                "gemm_mq4g256v2_residual_wmma_iu4_prepared (MMQ ADD into residual y)",
            ],
            Site::SharedDown => &[
                "reserve_int4_mmq",
                "fused_silu_mul_rotate_mq_i4_gfx12_exact_batched (reference8 sidecar, no f32 store)",
                "gemm_mq4g256v2_prepared_iu4 (verify receipt, fp16 cache invalidate, memset y, MMQ ADD, token validation)",
            ],
            Site::RouterShared => &[
                "reserve_int4_mmq",
                "rotate_x_mq_i4_gfx12_exact_batched (reference8 sidecar, no f32 store)",
                "gemm_mq4g256v2_prepared_iu4 (router: memset + MMQ ADD on the token)",
                "try_gemm_mq4g256v2_shared_iu4([selector, gate, up], .., Some(prep)) (selector zero+ADD, gate/up SET)",
            ],
        }
    }
}

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
    site: Site,
    rows: usize,
    data: Data,
    awq: bool,
}

impl Case {
    fn name(&self) -> String {
        format!(
            "{}_k{}_rows{}_{}_{}",
            self.site.name(),
            self.site.k(),
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
    sites: Vec<Site>,
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
        sites: Site::ALL.to_vec(),
        rows: ROWS.to_vec(),
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
            "--sites" => {
                a.sites = value("--sites")?
                    .split(',')
                    .map(|s| Site::from_name(s).ok_or_else(|| format!("unknown site {s:?}")))
                    .collect::<Result<_, _>>()?;
            }
            "--rows" => {
                a.rows = value("--rows")?
                    .split(',')
                    .map(|s| match s.parse::<usize>() {
                        Ok(r) if ROWS.contains(&r) => Ok(r),
                        _ => Err(format!("rows {s:?} not in {ROWS:?}")),
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
    Ok(a)
}

fn case_matrix(a: &Args) -> Vec<Case> {
    let mut out = Vec::new();
    for &site in &a.sites {
        for &rows in &a.rows {
            for &data in &a.datas {
                for &awq in &a.awqs {
                    out.push(Case {
                        site,
                        rows,
                        data,
                        awq,
                    });
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
}

/// Random valid MQ4G256V2 rows: `m` rows of `k / 256` groups of 136 B.
fn synth_mq4g256v2(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert!(k % 256 == 0);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(m * (k / 256) * GROUP_BYTES);
    for _ in 0..m * (k / 256) {
        for _half in 0..2 {
            let scale = 0.004 + (rng.next() & 0x3f) as f32 * 2.0e-4;
            let zero = -7.5 * scale + ((rng.next() & 0xf) as f32 - 8.0) * 1.0e-4;
            out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
            out.extend_from_slice(&f16::from_f32(zero).to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            out.extend_from_slice(&(rng.next() as u32).to_le_bytes());
        }
    }
    assert_eq!(out.len(), m * (k / 256) * GROUP_BYTES);
    out
}

/// Finite F32 activations in `[-1, 1)` times a per-row scale
/// `0.25 * (1 + row % 5)`; all zeros for [`Data::Zero`].
fn synth_x(rows: usize, k: usize, seed: u64, data: Data) -> Vec<f32> {
    if data == Data::Zero {
        return vec![0.0; rows * k];
    }
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = 0.25 * (1 + r % 5) as f32;
        for _ in 0..k {
            out.push(((rng.next() & 0xff_ffff) as f32 / 8_388_608.0 - 1.0) * scale);
        }
    }
    out
}

/// Positive F32 vector in `[lo, lo + span)` (norm weights, AWQ scales,
/// residual stream init when `signed`).
fn synth_vec(len: usize, seed: u64, lo: f32, span: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..len)
        .map(|_| lo + (rng.next() & 0xff_ffff) as f32 / 16_777_216.0 * span)
        .collect()
}

fn site_seed(site: Site) -> u64 {
    0x5eed_f300 + site as u64 * 0x1000
}

/// Host activations per site: Norm / Rotate -> one `[rows, k]` tensor;
/// Silu -> gate and up `[rows, k]`.
fn synth_inputs(site: Site, rows: usize, data: Data) -> Vec<Vec<f32>> {
    let k = site.k();
    let base = site_seed(site) ^ (rows as u64) << 8;
    match site.producer() {
        Producer::Silu => vec![
            synth_x(rows, k, base ^ 0x11, data),
            synth_x(rows, k, base ^ 0x22, data),
        ],
        _ => vec![synth_x(rows, k, base ^ 0x33, data)],
    }
}

// ---------------------------------------------------------------------------
// Byte compare / locate helpers (CPU, covered by --self-check)
// ---------------------------------------------------------------------------

/// First differing byte; when one slice is a strict prefix of the other, the
/// shorter length.
fn first_mismatch(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn iu4_field(offset_in_record: usize) -> &'static str {
    if offset_in_record < D_RANGE.1 {
        "d"
    } else if offset_in_record < S_RANGE.1 {
        "s"
    } else {
        "qs"
    }
}

/// Locate a sidecar byte in the token-order `[kblock, token]` record stream.
fn iu4_locate(byte: usize, n: usize) -> Value {
    let record = byte / RECORD_BYTES;
    let off = byte % RECORD_BYTES;
    json!({
        "byte": byte,
        "record": record,
        "kblock": record / n,
        "token": record % n,
        "offset_in_record": off,
        "field": iu4_field(off),
    })
}

/// Locate an F32 output byte in a `[rows, m]` buffer.
fn out_locate(byte: usize, m: usize) -> Value {
    let elem = byte / 4;
    json!({
        "byte": byte,
        "elem": elem,
        "row": elem / m,
        "col": elem % m,
        "byte_in_f32": byte % 4,
    })
}

/// `HIPFIRE_A4_SLAB` planes per K128 block of `n` tokens:
/// `[d 4n][s 4n][qs 0..31: 32n][qs 32..63: 32n]` -> token-order records.
fn slab_to_token_i4(slab: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; slab.len()];
    for (src, dst) in slab
        .chunks_exact(n * RECORD_BYTES)
        .zip(out.chunks_exact_mut(n * RECORD_BYTES))
    {
        for t in 0..n {
            let r = &mut dst[t * RECORD_BYTES..t * RECORD_BYTES + RECORD_BYTES];
            r[..4].copy_from_slice(&src[t * 4..t * 4 + 4]);
            r[4..8].copy_from_slice(&src[4 * n + t * 4..4 * n + t * 4 + 4]);
            r[8..40].copy_from_slice(&src[8 * n + t * 32..8 * n + t * 32 + 32]);
            r[40..].copy_from_slice(&src[40 * n + t * 32..40 * n + t * 32 + 32]);
        }
    }
    out
}

/// Inverse of [`slab_to_token_i4`] (self-check only).
fn token_to_slab_i4(tok: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; tok.len()];
    for (src, dst) in tok
        .chunks_exact(n * RECORD_BYTES)
        .zip(out.chunks_exact_mut(n * RECORD_BYTES))
    {
        for t in 0..n {
            let r = &src[t * RECORD_BYTES..t * RECORD_BYTES + RECORD_BYTES];
            dst[t * 4..t * 4 + 4].copy_from_slice(&r[..4]);
            dst[4 * n + t * 4..4 * n + t * 4 + 4].copy_from_slice(&r[4..8]);
            dst[8 * n + t * 32..8 * n + t * 32 + 32].copy_from_slice(&r[8..40]);
            dst[40 * n + t * 32..40 * n + t * 32 + 32].copy_from_slice(&r[40..]);
        }
    }
    out
}

struct SidecarCmp {
    equal: bool,
    json: Value,
}

/// Compare two token-order sidecars: whole-stream equality, per-field
/// equality and the first mismatching record.
fn compare_sidecar(inc: &[u8], new: &[u8], n: usize) -> SidecarCmp {
    let first = first_mismatch(inc, new);
    let mut d_eq = true;
    let mut s_eq = true;
    let mut qs_eq = true;
    for (a, b) in inc
        .chunks_exact(RECORD_BYTES)
        .zip(new.chunks_exact(RECORD_BYTES))
    {
        d_eq &= a[D_RANGE.0..D_RANGE.1] == b[D_RANGE.0..D_RANGE.1];
        s_eq &= a[S_RANGE.0..S_RANGE.1] == b[S_RANGE.0..S_RANGE.1];
        qs_eq &= a[QS_RANGE.0..QS_RANGE.1] == b[QS_RANGE.0..QS_RANGE.1];
    }
    let equal = first.is_none();
    SidecarCmp {
        equal,
        json: json!({
            "equal": equal,
            "bytes_incumbent": inc.len(),
            "bytes_new": new.len(),
            "d_equal": d_eq,
            "s_equal": s_eq,
            "qs_equal": qs_eq,
            "first_mismatch": first.map(|b| iu4_locate(b, n)),
        }),
    }
}

/// Compare per-output F32 buffers; returns JSON per output, whether every
/// output is equal, and the first mismatch (output name + location).
fn compare_outputs(
    site: Site,
    inc: &[Vec<u8>],
    new: &[Vec<u8>],
) -> (Vec<Value>, bool, Option<Value>) {
    let mut all = true;
    let mut first_overall = None;
    let mut outs = Vec::new();
    for (i, (name, m)) in site.outputs().iter().enumerate() {
        let (a, b) = (&inc[i], &new[i]);
        let first = first_mismatch(a, b);
        let equal = first.is_none();
        if !equal {
            all = false;
            if first_overall.is_none() {
                let mut loc = out_locate(first.unwrap(), *m);
                loc["output"] = json!(name);
                first_overall = Some(loc);
            }
        }
        outs.push(json!({
            "name": name, "m": m, "bytes": a.len(),
            "equal": equal,
            "first_mismatch": first.map(|b| out_locate(b, *m)),
        }));
    }
    (outs, all, first_overall)
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

struct SiteRes {
    w: Vec<GpuTensor>,
    norm: Option<GpuTensor>,
    awq: GpuTensor,
}

fn upload_site(gpu: &mut Gpu, site: Site) -> SiteRes {
    let k = site.k();
    let w = site
        .outputs()
        .iter()
        .enumerate()
        .map(|(i, (_, m))| {
            let bytes = synth_mq4g256v2(*m, k, site_seed(site) + 0x100 + i as u64);
            gpu.upload_raw(&bytes, &[bytes.len()])
                .expect("upload weight")
        })
        .collect();
    let norm = (site.producer() == Producer::Norm).then(|| {
        gpu.upload_f32(&synth_vec(k, site_seed(site) + 0x200, 0.5, 1.0), &[k])
            .expect("upload norm weight")
    });
    let awq = gpu
        .upload_f32(&synth_vec(k, site_seed(site) + 0x300, 0.5, 1.5), &[k])
        .expect("upload awq scale");
    SiteRes { w, norm, awq }
}

struct Ctx<'a> {
    site: Site,
    k: usize,
    rows: usize,
    w: &'a [GpuTensor],
    norm: Option<&'a GpuTensor>,
    awq: Option<&'a GpuTensor>,
    x: &'a [GpuTensor],
    x_rot: &'a GpuTensor,
    /// Pristine residual for `fa_out` (ADD into a nonzero Y).
    y0: Option<&'a GpuTensor>,
}

const GATE: usize = 0;
const UP: usize = 1;
const SELECTOR: usize = 2;
const ROUTER: usize = 3;

fn sync(gpu: &Gpu) {
    gpu.hip.device_synchronize().expect("device synchronize");
}

fn alloc_ys(gpu: &mut Gpu, site: Site, rows: usize) -> Vec<GpuTensor> {
    site.outputs()
        .iter()
        .map(|(_, m)| gpu.alloc_tensor(&[rows, *m], DType::F32).expect("alloc y"))
        .collect()
}

fn init_ys(gpu: &Gpu, c: &Ctx, ys: &[GpuTensor]) {
    match c.y0 {
        Some(y0) => gpu
            .hip
            .memcpy_dtod(&ys[0].buf, &y0.buf, ys[0].byte_size())
            .expect("restore residual"),
        None => {
            for y in ys {
                gpu.hip
                    .memset(&y.buf, SENTINEL as i32, y.byte_size())
                    .expect("sentinel memset");
            }
        }
    }
}

type R = rdna_compute::HipResult<()>;

/// F32 producer of the incumbent sequence (writes `x_rot`).
fn incumbent_producer(gpu: &mut Gpu, c: &Ctx) -> R {
    let (k, n) = (c.k, c.rows);
    match (c.site.producer(), c.awq) {
        (Producer::Norm, Some(a)) => gpu.fused_rmsnorm_rotate_mq_awq_batched(
            &c.x[0],
            c.norm.expect("norm weight"),
            a,
            c.x_rot,
            k,
            EPS,
            n,
        ),
        (Producer::Norm, None) => gpu.fused_rmsnorm_rotate_mq_batched(
            &c.x[0],
            c.norm.expect("norm weight"),
            c.x_rot,
            k,
            EPS,
            n,
        ),
        (Producer::Rotate, Some(a)) => gpu.rotate_x_mq_awq_batched(&c.x[0], a, c.x_rot, k, n),
        (Producer::Rotate, None) => gpu.rotate_x_mq_batched(&c.x[0], c.x_rot, k, n),
        (Producer::Silu, Some(a)) => {
            gpu.fused_silu_mul_rotate_mq_awq_batched(&c.x[0], &c.x[1], a, c.x_rot, k, n)
        }
        (Producer::Silu, None) => {
            gpu.fused_silu_mul_rotate_mq_batched(&c.x[0], &c.x[1], c.x_rot, k, n)
        }
    }
}

/// Incumbent: F32 producer, then the unchanged GEMM wrapper(s) that run the
/// standalone quantizer internally.
fn run_incumbent(gpu: &mut Gpu, c: &Ctx, ys: &[GpuTensor]) -> R {
    incumbent_producer(gpu, c)?;
    let (k, n, w, xr) = (c.k, c.rows, c.w, c.x_rot);
    match c.site {
        Site::GdnQkvza => gpu.gemm_qkvza_hfq4g256_mq4v2(
            &w[0], &w[1], &w[2], &w[3], xr, &ys[0], &ys[1], &ys[2], &ys[3], 8192, 4096, 32, 32, k,
            n,
        ),
        Site::FaQkv => gpu.gemm_qkv_hfq4g256_mq4v2(
            &w[0], &w[1], &w[2], xr, &ys[0], &ys[1], &ys[2], 8192, 512, 512, k, n,
        ),
        Site::FaOut => gpu.gemm_hfq4g256_residual_mq4v2(&w[0], xr, &ys[0], 2048, k, n),
        Site::SharedDown => gpu.gemm_mq4g256v2(&w[0], xr, &ys[0], 2048, k, n),
        Site::RouterShared => {
            // Baseline: the EXISTING shared prepared route (ONE standalone
            // quantizer in the router method, group helper consumes its token).
            let prep = gpu
                .try_gemm_mq4g256v2_shared_router_iu4(&w[ROUTER], xr, &ys[ROUTER], 256, k, n)?
                .expect("shared router method returned None after the support check passed");
            let launched = gpu.try_gemm_mq4g256v2_shared_iu4(
                [
                    (&w[SELECTOR], &ys[SELECTOR], 1),
                    (&w[GATE], &ys[GATE], 512),
                    (&w[UP], &ys[UP], 512),
                ],
                xr,
                k,
                n,
                Some(&prep),
            )?;
            assert!(
                launched,
                "shared IU4 group returned Ok(false) after the support check passed"
            );
            Ok(())
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProdMode {
    /// reference8 runtime-candidate producers (the F3 candidates).
    Exact,
    /// Legacy compiled-candidate (`IU4_A4_CANDIDATES`) producers: diagnostic.
    Legacy,
}

/// IU4 sidecar producer into a fresh reservation (no f32 store).
fn produce_i4(gpu: &mut Gpu, c: &Ctx, mode: ProdMode) -> rdna_compute::HipResult<Int4MmqPrepared> {
    let (k, n) = (c.k, c.rows);
    let res = gpu.reserve_int4_mmq(k, n)?;
    match (c.site.producer(), mode) {
        (Producer::Norm, ProdMode::Exact) => gpu.fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched(
            &c.x[0],
            c.norm.expect("norm weight"),
            c.awq,
            None,
            res,
            k,
            EPS,
            n,
        ),
        (Producer::Norm, ProdMode::Legacy) => gpu.fused_rmsnorm_rotate_mq_i4_gfx12_batched(
            &c.x[0],
            c.norm.expect("norm weight"),
            c.awq,
            None,
            res,
            k,
            EPS,
            n,
        ),
        (Producer::Rotate, ProdMode::Exact) => {
            gpu.rotate_x_mq_i4_gfx12_exact_batched(&c.x[0], c.awq, None, res, k, n)
        }
        (Producer::Rotate, ProdMode::Legacy) => {
            gpu.rotate_x_mq_i4_gfx12_batched(&c.x[0], c.awq, None, res, k, n)
        }
        (Producer::Silu, ProdMode::Exact) => gpu.fused_silu_mul_rotate_mq_i4_gfx12_exact_batched(
            &c.x[0], &c.x[1], c.awq, None, res, k, n,
        ),
        (Producer::Silu, ProdMode::Legacy) => {
            gpu.fused_silu_mul_rotate_mq_i4_gfx12_batched(&c.x[0], &c.x[1], c.awq, None, res, k, n)
        }
    }
}

/// New arm: fused producer, then the existing prepared consumers with the same
/// SET / ADD dispatch as the incumbent wrapper.
fn run_new(gpu: &mut Gpu, c: &Ctx, ys: &[GpuTensor]) -> R {
    let prep = produce_i4(gpu, c, ProdMode::Exact)?;
    let (k, n, w, xr) = (c.k, c.rows, c.w, c.x_rot);
    match c.site {
        Site::GdnQkvza => gpu.gemm_qkvza_mq4g256v2_wmma_iu4_prepared(
            &w[0], &w[1], &w[2], &w[3], xr, &prep, &ys[0], &ys[1], &ys[2], &ys[3], 8192, 4096, 32,
            32, k, n,
        ),
        Site::FaQkv => gpu.gemm_qkv_mq4g256v2_wmma_iu4_prepared(
            &w[0], &w[1], &w[2], &prep, &ys[0], &ys[1], &ys[2], 8192, 512, 512, k, n,
        ),
        Site::FaOut => {
            gpu.gemm_mq4g256v2_residual_wmma_iu4_prepared(&w[0], &prep, &ys[0], 2048, k, n)
        }
        Site::SharedDown => gpu.gemm_mq4g256v2_prepared_iu4(&w[0], &prep, &ys[0], 2048, k, n),
        Site::RouterShared => {
            gpu.gemm_mq4g256v2_prepared_iu4(&w[ROUTER], &prep, &ys[ROUTER], 256, k, n)?;
            let launched = gpu.try_gemm_mq4g256v2_shared_iu4(
                [
                    (&w[SELECTOR], &ys[SELECTOR], 1),
                    (&w[GATE], &ys[GATE], 512),
                    (&w[UP], &ys[UP], 512),
                ],
                xr,
                k,
                n,
                Some(&prep),
            )?;
            assert!(
                launched,
                "shared IU4 group returned Ok(false) after the support check passed"
            );
            Ok(())
        }
    }
}

/// Diagnostic arm (router site only): the generic A3B shared stage with THREE
/// standalone quantizers (router GEMM, selector GEMM, fused gate/up), after the
/// same F32 rotate. Its bytes are compared with the one-quantizer baseline and
/// its time is printed relative to that baseline; it is never the incumbent.
fn run_three_quant_router(gpu: &mut Gpu, c: &Ctx, ys: &[GpuTensor]) -> R {
    assert_eq!(c.site, Site::RouterShared);
    incumbent_producer(gpu, c)?;
    let (k, n, w, xr) = (c.k, c.rows, c.w, c.x_rot);
    gpu.gemm_mq4g256v2(&w[ROUTER], xr, &ys[ROUTER], 256, k, n)?;
    gpu.gemm_mq4g256v2(&w[SELECTOR], xr, &ys[SELECTOR], 1, k, n)?;
    gpu.gemm_gate_up_hfq4g256_mq4v2(&w[GATE], &w[UP], xr, &ys[GATE], &ys[UP], 512, 512, k, n)
}

/// Download the live IU4 sidecar of `(k, n)` as token-order records.
/// Returns `(bytes, was_slab)`.
fn snapshot_sidecar(gpu: &Gpu, k: usize, n: usize) -> (Vec<u8>, bool) {
    let nbytes = (k / 128) * n * RECORD_BYTES;
    let (_, ptr) = gpu.scratch.int4_mmq_live();
    assert!(!ptr.is_null(), "no live IU4 sidecar scratch");
    let mut bytes = vec![0u8; nbytes];
    let view = unsafe { DeviceBuffer::from_raw(ptr, nbytes) };
    gpu.hip
        .memcpy_dtoh(&mut bytes, &view)
        .expect("download sidecar");
    std::mem::forget(view);
    let slab = gpu.scratch.int4_mmq_slab_at(ptr);
    if slab {
        (slab_to_token_i4(&bytes, n), true)
    } else {
        (bytes, false)
    }
}

fn download(gpu: &Gpu, ys: &[GpuTensor]) -> Vec<Vec<u8>> {
    ys.iter()
        .map(|y| gpu.download_raw_bytes(y).expect("download y"))
        .collect()
}

struct Pass {
    json: Value,
    identical: bool,
    first_mismatch: Option<Value>,
}

/// One verification pass: incumbent then new (then the legacy diagnostic
/// producer and, for the router site, the existing shared route). Each arm's
/// sidecar is snapshotted right after its sequence (the scratch is shared).
fn verify_pass(
    gpu: &mut Gpu,
    c: &Ctx,
    ys_inc: &[GpuTensor],
    ys_new: &[GpuTensor],
    ys_ex: Option<&[GpuTensor]>,
) -> Pass {
    let (k, n) = (c.k, c.rows);

    init_ys(gpu, c, ys_inc);
    sync(gpu);
    let gen_before = gpu.scratch.int4_mmq_live().0;
    run_incumbent(gpu, c, ys_inc).expect("incumbent sequence");
    sync(gpu);
    let gen_after = gpu.scratch.int4_mmq_live().0;
    assert_ne!(
        gen_before, gen_after,
        "incumbent did not run the standalone IU4 quantizer (scratch generation unchanged): \
         the IU4 route was not taken, refusing to compare a fallback"
    );
    let (inc_side, inc_slab) = snapshot_sidecar(gpu, k, n);
    assert!(
        !inc_slab,
        "standalone quantizer sidecar must be token order"
    );
    let inc_out = download(gpu, ys_inc);

    init_ys(gpu, c, ys_new);
    sync(gpu);
    let gen_before = gpu.scratch.int4_mmq_live().0;
    run_new(gpu, c, ys_new).expect("new sequence");
    sync(gpu);
    assert_ne!(
        gen_before,
        gpu.scratch.int4_mmq_live().0,
        "new arm did not reserve a fresh sidecar generation"
    );
    let (new_side, new_slab) = snapshot_sidecar(gpu, k, n);
    let new_out = download(gpu, ys_new);

    let side = compare_sidecar(&inc_side, &new_side, n);
    let (outs, outs_equal, first_out) = compare_outputs(c.site, &inc_out, &new_out);

    // Diagnostic: legacy compiled-candidate producer vs the standalone sidecar.
    let prep = produce_i4(gpu, c, ProdMode::Legacy).expect("legacy producer");
    sync(gpu);
    let _ = prep;
    let (leg_side, leg_slab) = snapshot_sidecar(gpu, k, n);
    let legacy = compare_sidecar(&inc_side, &leg_side, n);

    // Router site, diagnostic: the generic 3-quantizer stage vs the
    // one-quantizer baseline (incumbent).
    let three_quant = ys_ex.map(|ys_ex| {
        init_ys(gpu, c, ys_ex);
        sync(gpu);
        run_three_quant_router(gpu, c, ys_ex).expect("three-quantizer diagnostic");
        sync(gpu);
        let ex_out = download(gpu, ys_ex);
        let (outs, eq, _) = compare_outputs(c.site, &inc_out, &ex_out);
        json!({ "outputs_equal": eq, "outputs": outs })
    });

    let identical = side.equal && outs_equal;
    let first_mismatch = if !side.equal {
        side.json["first_mismatch"]
            .as_object()
            .map(|_| json!({ "stage": "iu4_input", "at": side.json["first_mismatch"].clone() }))
    } else {
        first_out.map(|at| json!({ "stage": "gemm_output", "at": at }))
    };
    Pass {
        json: json!({
            "iu4_input": side.json,
            "new_sidecar_layout": if new_slab { "slab" } else { "aos" },
            "outputs": outs,
            "outputs_equal": outs_equal,
            "legacy_input_identical": legacy.equal,
            "legacy_input": legacy.json,
            "legacy_sidecar_layout": if leg_slab { "slab" } else { "aos" },
            "three_quantizer_diagnostic": three_quant,
            "byte_identical": identical,
        }),
        identical,
        first_mismatch,
    }
}

struct TimingOut {
    inc: Vec<f64>,
    new: Vec<f64>,
    ex: Vec<f64>,
    deadline_hit: bool,
}

fn time_arms(
    gpu: &mut Gpu,
    c: &Ctx,
    ys: [&[GpuTensor]; 3],
    with_existing: bool,
    reps: usize,
    warmups: usize,
    deadline: Instant,
) -> TimingOut {
    let mut t = TimingOut {
        inc: Vec::with_capacity(reps),
        new: Vec::with_capacity(reps),
        ex: Vec::with_capacity(reps),
        deadline_hit: false,
    };
    for rep in 0..warmups + reps {
        if Instant::now() >= deadline {
            t.deadline_hit = true;
            break;
        }
        let arms: &[usize] = if with_existing { &[0, 1, 2] } else { &[0, 1] };
        let mut us = [0.0f64; 3];
        for &arm in arms {
            init_ys(gpu, c, ys[arm]);
            sync(gpu);
            let t0 = Instant::now();
            let r = match arm {
                0 => run_incumbent(gpu, c, ys[0]),
                1 => run_new(gpu, c, ys[1]),
                _ => run_three_quant_router(gpu, c, ys[2]),
            };
            r.expect("timed arm");
            sync(gpu);
            us[arm] = t0.elapsed().as_secs_f64() * 1e6;
        }
        if rep >= warmups {
            t.inc.push(us[0]);
            t.new.push(us[1]);
            if with_existing {
                t.ex.push(us[2]);
            }
        }
    }
    if t.inc.len() < reps {
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

fn run_case(gpu: &mut Gpu, case: &Case, sr: &SiteRes, a: &Args, deadline: Instant) -> CaseOut {
    let site = case.site;
    let (k, rows) = (site.k(), case.rows);
    let inputs: Vec<GpuTensor> = synth_inputs(site, rows, case.data)
        .iter()
        .map(|h| gpu.upload_f32(h, &[rows, k]).expect("upload input"))
        .collect();
    let x_rot = gpu.zeros(&[rows, k], DType::F32).expect("alloc x_rot");
    let y0 = (site == Site::FaOut).then(|| {
        let h = synth_vec(
            rows * 2048,
            site_seed(site) + 0x400 + rows as u64,
            -1.0,
            2.0,
        );
        gpu.upload_f32(&h, &[rows, 2048]).expect("upload residual")
    });
    let ys_inc = alloc_ys(gpu, site, rows);
    let ys_new = alloc_ys(gpu, site, rows);
    let with_existing = site == Site::RouterShared;
    let ys_ex = with_existing.then(|| alloc_ys(gpu, site, rows));
    let ctx = Ctx {
        site,
        k,
        rows,
        w: &sr.w,
        norm: sr.norm.as_ref(),
        awq: case.awq.then_some(&sr.awq),
        x: &inputs,
        x_rot: &x_rot,
        y0: y0.as_ref(),
    };

    let first = verify_pass(gpu, &ctx, &ys_inc, &ys_new, ys_ex.as_deref());
    let timing = time_arms(
        gpu,
        &ctx,
        [&ys_inc, &ys_new, ys_ex.as_deref().unwrap_or(&ys_new)],
        with_existing,
        a.reps,
        a.warmups,
        deadline,
    );
    let post = verify_pass(gpu, &ctx, &ys_inc, &ys_new, ys_ex.as_deref());

    let identical = first.identical && post.identical && !timing.deadline_hit;
    let first_mismatch = first.first_mismatch.clone().or(post.first_mismatch.clone());
    let timing_json = if timing.inc.len() >= MIN_REPS && timing.new.len() >= MIN_REPS {
        let inc = stats(&timing.inc);
        let new = stats(&timing.new);
        let speedup = inc["median"].as_f64().unwrap() / new["median"].as_f64().unwrap();
        let three = with_existing.then(|| stats(&timing.ex));
        let three_ratio = three
            .as_ref()
            .map(|t| t["median"].as_f64().unwrap() / inc["median"].as_f64().unwrap());
        json!({
            "incumbent_us": inc,
            "new_us": new,
            "three_quantizer_diagnostic_us": three,
            "three_quantizer_over_one_quantizer_median": three_ratio,
            "speedup_median": speedup,
            "saving_vs_incumbent_pct": (1.0 - 1.0 / speedup) * 100.0,
        })
    } else {
        json!({ "incomplete": true, "samples": timing.inc.len() })
    };
    let line = json!({
        "case": case.name(),
        "site": site.name(), "k": k, "rows": rows,
        "data": case.data.name(), "awq": case.awq,
        "outputs": site.outputs().iter().map(|(n, m)| json!({"name": n, "m": m})).collect::<Vec<_>>(),
        "sidecar_bytes": (k / 128) * rows * RECORD_BYTES,
        "reps": a.reps, "warmups": a.warmups,
        "first_run": first.json,
        "post_timing_run": post.json,
        "timing": timing_json,
        "a4_slab_active": gpu.a4_slab_active(),
        "deadline_exceeded": timing.deadline_hit,
        "all_byte_identical": identical,
    });

    for t in inputs {
        let _ = gpu.free_tensor(t);
    }
    let _ = gpu.free_tensor(x_rot);
    if let Some(t) = y0 {
        let _ = gpu.free_tensor(t);
    }
    for t in ys_inc
        .into_iter()
        .chain(ys_new)
        .chain(ys_ex.into_iter().flatten())
    {
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

fn prerequisites() -> Vec<&'static str> {
    vec![
        "exact gfx1201 with WMMA (the oracle refuses any other arch with exit 2)",
        "IU4 prefill enabled (HIPFIRE_IU4_PREFILL not 0), A8 prefill not active, not in FP8 verify scope: Gpu::mq4g256v2_shared_iu4_supported(k, rows) must hold for every case k/rows; the incumbent GEMM wrappers then take the standalone-quantizer IU4 route (checked per case via the scratch generation bump)",
        "the three reference8 exact producers exist: fused_rmsnorm_rotate_mq_i4_gfx12_exact_batched, rotate_x_mq_i4_gfx12_exact_batched, fused_silu_mul_rotate_mq_i4_gfx12_exact_batched (no new GPU kernels in this oracle)",
        "the standalone quantizer is always the 8-candidate reference; the exact producers select it at runtime (cand_mode 8) so no HIPFIRE_G12_A4C2 / -DIU4_A4_CANDIDATES setting is required for the new arm",
        "the legacy compiled-candidate producer arm is diagnostic only (default C2; HIPFIRE_G12_A4C2=0 gives C1/RTN, neither equals the standalone 8-candidate quantizer) and never contributes to all_byte_identical",
        "one GPU, single process; weights are synthetic valid MQ4G256V2, not model weights",
    ]
}

fn limitations() -> Vec<&'static str> {
    vec![
        "does not read HIPFIRE_PREFILL_IU4_PRODUCER_FUSED and does not exercise the PrefillIu4ProducerFused capability or the prefill.rs/dispatch wiring: it proves launcher-level bytes only",
        "sigmoid_mul_f32 ahead of the FA out rotate and rmsnorm_f32 ahead of the router rotate are upstream of the compared sequences and common to both arms (the oracle feeds their outputs directly)",
        "AWQ scales carry no reciprocal planes (AoS, non-slab twins); a slab sidecar from a future producer would be converted and reported as layout:slab",
        "gdn_qkvza: the new prepared consumer runs beta/alpha as plain MMQ SET on gfx1201; the incumbent wrapper goes through small_tail_set_iu4 (zero Y + gfx11 MW4 attempt, MMQ SET fallback); equality is proven only by the output byte compare",
        "random data only covers synthetic activations in [-1,1) with per-row scale; zero data covers the amax=0 branch; no inf/NaN rows",
        "timing includes the whole sequence incl. host launch overhead with device_synchronize bracketing, not isolated kernel time",
    ]
}

fn plan(a: &Args) {
    let cases = case_matrix(a);
    println!(
        "{}",
        json!({
            "plan": true,
            "oracle": "prefill_iu4_producer",
            "gpu_initialized": false,
            "byte_result": null,
            "note": "CPU plan only: no GPU was touched and nothing was compared; byte identity is a GPU outcome",
            "cases": cases.len(),
            "reps": a.reps, "min_reps": MIN_REPS, "warmups": a.warmups,
            "deadline_secs": a.deadline_secs,
            "scope_sites": Site::ALL.iter().map(|s| s.name()).collect::<Vec<_>>(),
            "prerequisites": prerequisites(),
            "limitations": limitations(),
        })
    );
    for c in &cases {
        println!(
            "{}",
            json!({
                "plan_case": c.name(),
                "site": c.site.name(),
                "k": c.site.k(), "rows": c.rows,
                "data": c.data.name(), "awq": c.awq,
                "outputs": c.site.outputs().iter().map(|(n, m)| json!({"name": n, "m": m, "bytes": c.rows * m * 4})).collect::<Vec<_>>(),
                "sidecar_bytes": (c.site.k() / 128) * c.rows * RECORD_BYTES,
                "incumbent_sequence": c.site.incumbent_sequence(),
                "new_sequence": c.site.new_sequence(),
                "legacy_diagnostic_producer": true,
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

    // Case matrix.
    let all = parse_args(&[]).expect("default args");
    let cases = case_matrix(&all);
    check!(
        cases.len() == 5 * 2 * 2 * 2,
        "default matrix has {} cases",
        cases.len()
    );
    let mut names: Vec<String> = cases.iter().map(Case::name).collect();
    names.sort();
    names.dedup();
    check!(names.len() == cases.len(), "case names are not unique");
    for c in &cases {
        let k = c.site.k();
        check!(k % 256 == 0, "{}: k % 256", c.name());
        check!(ROWS.contains(&c.rows), "{}: rows", c.name());
        check!(
            rdna_compute::scratch::int4_mmq_reservation_bytes(k, c.rows)
                == Some((k / 128) * c.rows * RECORD_BYTES),
            "{}: reservation bytes",
            c.name()
        );
    }
    let sites_k: Vec<(&str, usize)> = Site::ALL.iter().map(|s| (s.name(), s.k())).collect();
    check!(
        sites_k
            == vec![
                ("gdn_qkvza", 2048),
                ("fa_qkv", 2048),
                ("fa_out", 4096),
                ("shared_down", 512),
                ("router_shared", 2048),
            ],
        "site k table {sites_k:?}"
    );
    let ms = |s: Site| s.outputs().iter().map(|o| o.1).collect::<Vec<_>>();
    check!(
        ms(Site::GdnQkvza) == vec![8192, 4096, 32, 32],
        "gdn outputs"
    );
    check!(ms(Site::FaQkv) == vec![8192, 512, 512], "fa qkv outputs");
    check!(ms(Site::FaOut) == vec![2048], "fa out outputs");
    check!(ms(Site::SharedDown) == vec![2048], "shared down outputs");
    check!(
        ms(Site::RouterShared) == vec![512, 512, 1, 256],
        "router shared outputs"
    );
    check!(
        Site::from_name("fa_out") == Some(Site::FaOut) && Site::from_name("nope").is_none(),
        "site names"
    );
    for s in Site::ALL {
        check!(
            !s.incumbent_sequence().is_empty() && !s.new_sequence().is_empty(),
            "{} sequences",
            s.name()
        );
    }

    // Layout sizes.
    check!(
        D_RANGE.1 - D_RANGE.0 == 4 && S_RANGE.1 - S_RANGE.0 == 4 && QS_RANGE.1 - QS_RANGE.0 == 64,
        "record field widths"
    );
    check!(
        D_RANGE.1 == S_RANGE.0 && S_RANGE.1 == QS_RANGE.0 && QS_RANGE.1 == RECORD_BYTES,
        "record fields tile the 72 B record"
    );
    check!(
        synth_mq4g256v2(3, 512, 1).len() == 3 * 2 * GROUP_BYTES,
        "synth weight size"
    );
    check!(
        synth_mq4g256v2(3, 512, 1) == synth_mq4g256v2(3, 512, 1),
        "synth weight determinism"
    );
    check!(
        synth_mq4g256v2(3, 512, 1) != synth_mq4g256v2(3, 512, 2),
        "synth weight seeds"
    );
    check!(
        synth_x(4, 8, 1, Data::Zero).iter().all(|v| *v == 0.0)
            && synth_x(4, 8, 1, Data::Random).iter().any(|v| *v != 0.0)
            && synth_x(4, 8, 1, Data::Random).iter().all(|v| v.is_finite()),
        "synth_x data modes"
    );
    for s in Site::ALL {
        let ins = synth_inputs(s, 64, Data::Random);
        let want = if s.producer() == Producer::Silu { 2 } else { 1 };
        check!(
            ins.len() == want && ins.iter().all(|v| v.len() == 64 * s.k()),
            "{} synth_inputs shape",
            s.name()
        );
        check!(
            s.producer() != Producer::Silu || ins[0] != ins[1],
            "{} gate/up differ",
            s.name()
        );
    }

    // Slab conversion round trip and locate.
    let n = 5usize;
    let tok: Vec<u8> = (0..2 * n * RECORD_BYTES)
        .map(|i| (i * 31 + 7) as u8)
        .collect();
    check!(
        slab_to_token_i4(&token_to_slab_i4(&tok, n), n) == tok,
        "slab <-> token round trip"
    );
    let slab = token_to_slab_i4(&tok, n);
    check!(slab != tok, "slab layout differs from token order");

    // First-mismatch helper.
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
        (8, "qs"),
        (71, "qs"),
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
                && cmp.json["qs_equal"] == (field != "qs"),
            "per-field flags {field}@{field_off}"
        );
    }
    a[0] = 1;
    check!(
        compare_sidecar(&a, &a.clone(), n).equal,
        "identical sidecars compare equal"
    );
    let o = out_locate(4 * (3 * 256 + 17) + 2, 256);
    check!(
        o["row"] == 3 && o["col"] == 17 && o["byte_in_f32"] == 2 && o["elem"] == 3 * 256 + 17,
        "out_locate {o}"
    );
    let (outs, all_eq, first) = compare_outputs(
        Site::GdnQkvza,
        &[vec![0u8; 8], vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]],
        &[
            vec![0u8; 8],
            vec![1u8; 8],
            vec![2, 0, 0, 0, 0, 0, 9, 0],
            vec![3u8; 8],
        ],
    );
    check!(
        outs.len() == 4
            && !all_eq
            && first.as_ref().map(|f| f["output"].clone()) == Some(json!("beta")),
        "compare_outputs names the first mismatching output: {first:?}"
    );

    // Stats.
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
        parse_args(&s(&["--rows", "64"])).is_err(),
        "rows outside matrix rejected"
    );
    check!(
        parse_args(&s(&["--bogus"])).is_err(),
        "unknown arg rejected"
    );
    check!(
        case_matrix(
            &parse_args(&s(&[
                "--sites", "fa_out", "--rows", "512", "--data", "random", "--awq", "some"
            ]))
            .unwrap()
        )
        .len()
            == 1,
        "filters narrow the matrix"
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
            "note": "CPU self-check validates the case matrix, layout sizes and the first-mismatch helper only; it proves nothing about GPU bytes",
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
        eprintln!("prefill_iu4_producer_oracle: {secs}s deadline exceeded (watchdog); exiting 3");
        std::process::exit(3);
    });

    let mut gpu = Gpu::init().expect("gpu init");
    let cases = case_matrix(a);
    if gpu.arch != "gfx1201" {
        eprintln!(
            "prefill_iu4_producer_oracle: arch {} is not gfx1201; refusing to compare",
            gpu.arch
        );
        std::process::exit(2);
    }
    let unsupported: Vec<String> = cases
        .iter()
        .filter(|c| !gpu.mq4g256v2_shared_iu4_supported(c.site.k(), c.rows))
        .map(Case::name)
        .collect();
    if !unsupported.is_empty() {
        eprintln!(
            "prefill_iu4_producer_oracle: mq4g256v2_shared_iu4_supported is false on {} for {} case(s) \
             (needs IU4 prefill enabled, no A8 prefill / FP8 verify scope, eager stream, k % 256 == 0, \
             rows >= 64); refusing to compare a fallback",
            gpu.arch,
            unsupported.len()
        );
        std::process::exit(2);
    }
    let candidates = gpu
        .flags
        .hipcc_extra_flags
        .split_whitespace()
        .filter_map(|f| f.strip_prefix("-DIU4_A4_CANDIDATES="))
        .last()
        .map(str::to_string);

    let mut res: HashMap<Site, SiteRes> = HashMap::new();
    let mut identical_cases = 0usize;
    let mut completed = 0usize;
    let mut first_mismatch: Option<Value> = None;
    let mut deadline_hit = false;
    for case in &cases {
        if Instant::now() >= deadline {
            deadline_hit = true;
            break;
        }
        let sr = res
            .entry(case.site)
            .or_insert_with(|| upload_site(&mut gpu, case.site));
        let out = run_case(&mut gpu, case, sr, a, deadline);
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
    let all_identical = !deadline_hit && completed == cases.len() && identical_cases == completed;
    println!(
        "{}",
        json!({
            "summary": true,
            "oracle": "prefill_iu4_producer",
            "arch": gpu.arch,
            "cases_planned": cases.len(),
            "cases_completed": completed,
            "cases_identical": identical_cases,
            "reps": a.reps, "warmups": a.warmups,
            "deadline_secs": a.deadline_secs,
            "elapsed_secs": start.elapsed().as_secs_f64(),
            "deadline_exceeded": deadline_hit,
            "legacy_compiled_candidates": candidates,
            "first_mismatch": first_mismatch,
            "all_byte_identical": all_identical,
        })
    );
    if deadline_hit {
        eprintln!("prefill_iu4_producer_oracle: deadline reached before the matrix completed");
        std::process::exit(3);
    }
    if !all_identical {
        eprintln!(
            "prefill_iu4_producer_oracle: {} of {} case(s) not byte-identical",
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
            eprintln!("prefill_iu4_producer_oracle: {e}");
            eprintln!(
                "usage: [--plan|--self-check] [--sites a,b] [--rows 512,4096] [--data zero,random] \
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
