// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G0 GDN recurrent-state precision oracle: native-MTP verify capture and
//! rollback against single-row autoregressive (AR) decode, per state format.
//!
//! Geometry is the Qwen4 tier-C GDN layer: 16 key heads, 48 value heads,
//! key = value = 128, packed q|k|v width 10240. For every window of
//! `rows = 1..=max_rows` consecutive rows the oracle compares
//!
//! * the **reference**: the source state copied to a separate AR state and
//!   advanced by `rows` single-row `gated_delta_step` launches (the production
//!   unfused recurrent output), saving the state bytes after every row, with
//! * the **candidate**: one `gated_delta_step_batched` launch with
//!   `row_states = Some(ring half)` (the verify capture), and
//! * the **rollback**: `gated_delta_rollback_layers` re-running the first
//!   `keep` captured rows from the packed recurrence input
//!   (`[projection rows | gate rows | beta rows]`, one `[input, ring]` pointer
//!   pair in the table).
//!
//! The ring holds `2R` slots (`R = max(max_rows, 8)`, the production capture
//! capacity), the source state lives in slot 0 and is the production
//! `capture_base` rule's "live" slot: the capture writes the half that does not
//! hold the source (`base = R` when `source < R`, else `0`), the candidate view
//! is `rows` slots at `base`, its final state lands in `base + rows - 1`, and
//! after every keep of the window the source advances to that final slot so the
//! half alternates. Windows are consecutive (`>= 2`), each continuing from the
//! previous window's final state at the next absolute positions.
//!
//! Per window the candidate runs **alone first**, twice, and its outputs, final
//! slot and source slot are byte-checked before any rollback can rewrite the
//! final slot (a full-keep rollback would otherwise mask a capture bug). Each
//! of those findings is aggregated into every keep case of the window. Then, for
//! each keep, candidate + rollback run twice (`keep = 0` is the candidate-only
//! result: no rollback, source untouched). The checks per case:
//!
//! * batched outputs equal AR row by row (bitwise);
//! * the final capture slot equals AR's state after the last row;
//! * the rollback target slot `base + keep - 1` equals AR's state after row
//!   `keep - 1`;
//! * the rollback's discarded outputs equal AR's first `keep` rows;
//! * the source slot is byte-identical to its pre-window bytes;
//! * the second identical run reproduces every compared region byte for byte
//!   (same-route determinism);
//! * no output or reference state is non-finite.
//!
//! Output, discard, final slot and rollback target slot are poisoned before each
//! run (outside any graph) so unwritten bytes cannot pass; the source is never
//! reset. With `--graph 1` the candidate-only check uses a candidate-only
//! hipGraph and every keep case a candidate+rollback hipGraph (two kernels
//! captured, instantiated, replayed twice; kernels are warmed eagerly before
//! capture, but every measured candidate and rollback launch is a graph
//! replay, never an eager fallback). The `GraphState` owns one live graph, so
//! each is dropped before the next capture and before any buffer is freed.
//!
//! Route flags set like the production verify forward: `qwen4_scope = true`,
//! `prefill_invariant_scope = false`, `wide_verify_rows = R` for the batched
//! launches and `0` for AR. The eager few-row limit (`max(8, R)`) and the limit
//! inside a graph capture (stays 8) are both printed and written to the JSON.
//!
//! Inputs copy the tier-C generator ranges (`qwen4_tierc/gdn.rs::make_steps`):
//! q|k uniform in [-2, 2], v in [-1, 1] (BF16-rounded), gate in [-0.3, -0.02],
//! beta in [0.1, 0.9]; the initial F32 state is small uniform noise, converted
//! to Q8 by the production `gdn_state_f32_to_q8` kernel (launched exactly like
//! the private `gdn_state_convert` wrapper). The same data feeds every format.
//! `max|state|` per format is taken over the decoded AR snapshots (Q8 decoded
//! by `gdn_state_q8_to_f32`); candidate states are bytewise equal to them
//! whenever the case passes.
//!
//! **`q8_ef`** (explicit-only error-feedback state, option (a)) is plain Q8
//! plus a separate live F16 residual per owner (`GdnStateFormat::Q8Ef`). The
//! ring, the source slot, the candidate final slot and the rollback target
//! are plain-Q8 bytes (`format.storage()`, ring dtype/extent of Q8); the
//! residual (`format.residual_units`, element `(head*128+row)*128+column`)
//! lives in two further F16 tensors, one per live owner: the AR reference
//! owner and the candidate owner. For every `rows` the source slot is first
//! advanced by `SEED_STEPS` single-row EF AR steps from a zero residual so
//! the initial residual is nonzero (its max is recorded and printed); a zero
//! seed residual is a hard failure. Both owners start every window from the
//! same source slot and the same residual bytes (the candidate owner
//! continues from the previous window's capture final slot and final
//! residual; the AR owner is re-seeded from that same state so the diagnostic
//! comparison never accumulates drift). The AR reference is `rows` single-row
//! `q8_ef` steps (residual kept row to row). The candidate is the same
//! capture launch (it also writes the final residual into the candidate
//! owner's residual). Cases per keep:
//!
//! * `keep == rows`: all-accept commit, candidate alone, the residual is
//!   kept (no reset);
//! * `keep == 0`: candidate, then the live-residual reset
//!   `gdn_q8ef_residual_zero` (the source slot is the restore target);
//! * `0 < keep < rows`: candidate, `gated_delta_rollback_layers` with format
//!   `q8_ef` (the plain-Q8 replay of the ring), then the residual reset.
//!
//! HARD checks for `q8_ef` (any failure exits 1): (a) the candidate run twice
//! from the identical source slot and residual is byte-identical in outputs,
//! final ring slot, rollback target and live residual (same-route
//! determinism); (b) the live source slot is byte-unchanged; (c) no ring slot
//! of the candidate view other than the final slot (and the rollback target)
//! is written (poison check of the whole view); (d) after keep `< rows` the
//! live residual is exactly zero and the rollback target equals the second
//! identical run; (e) after keep `== rows` the live residual equals the
//! capture's final residual (kept, not zeroed); (f) no non-finite output,
//! state (decoded) or residual; (g) with `--graph 1` the captured and
//! replayed graph (candidate; candidate + reset; candidate + rollback +
//! reset, the reset dispatch inside the capture) reproduces the eager bytes.
//! Every case also requires the candidate outputs and final slot to equal the
//! window's candidate-alone capture (same route, same history) and the
//! capture to have written the residual.
//!
//! Cross-route equality is a DIAGNOSTIC and never fails: AR-with-EF against
//! the candidate (outputs, final slot, final residual and the effective
//! state `code*scale + residual`, attached to the `keep == rows` case) and
//! the AR state after row `keep-1` against the rollback-plus-reset slot, the
//! rollback's discarded outputs and the effective states (attached to every
//! `0 < keep < rows` case). Each is printed as a `diag:` line (equal flag,
//! first differing byte offset / row, max abs decoded difference; for the
//! effective-state entries the offset indexes the F32 effective state) and
//! written to the JSON `diagnostics`.
//!
//! `q8` and `fp32` checks are unchanged and entirely hard.
//!
//! ```text
//! cargo run --release -p rdna-compute --features lab --example \
//!     gdn_state_precision_oracle -- --formats q8,fp32 --graph 1 \
//!     --max-rows 32 --all-keeps [--seed N] [--windows W] [--json PATH]
//! ```
//!
//! `--formats` takes `q8`, `fp32` and `q8_ef` (default `q8,fp32`; `q8_ef` is
//! explicit-only); `--graph 0|1` (default 0); `--max-rows 1..=32` (default 32,
//! the production wide-verify maximum); `--windows >= 2` (default 3); without
//! `--all-keeps` only keep 0 and the full keep run per window.
//!
//! Exit 0 = every hard check matched; 1 = at least one hard mismatch (one
//! stdout line per `(format, graph, rows, keep, window)` with the first
//! differing byte offset and kind/row; `q8_ef` diagnostics never count); 2 =
//! bad usage or an unsupported GPU/graph/setup, named on stderr. Per format a
//! final line prints
//! `oracle: format=<f> graph=<g> cases=<n> mismatches=<m> kernels=<names>`;
//! `q8_ef` appends ` hard_failures=<h> diagnostic_diffs=<d>` (`h` counts the
//! hard findings over all cases, window-level findings repeated in every
//! case of their window; `d` counts the differing diagnostics).

use hip_bridge::KernargBlob;
use rdna_compute::prefill_invariant::{FEW_ROW_ROUTE_ROWS, WIDE_VERIFY_MAX_ROWS};
use rdna_compute::tensor_ops::{
    gated_delta_rollback_layers, gated_delta_step, gated_delta_step_batched,
    gdn_q8ef_residual_zero, GatedDeltaRollbackLayers, GatedDeltaStep, GatedDeltaStepBatched,
    GdnStateFormat,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const KH: usize = 16;
const VH: usize = 48;
const HD: usize = 128;
const QK: usize = KH * HD;
const VW: usize = VH * HD;
const QKV: usize = 2 * QK + VW;
const STATE_ELEMS: usize = VH * HD * HD;
/// `qwen4_tierc/gdn.rs::START_POSITION`.
const START_POSITION: usize = 5;
/// `qwen4_tierc/gdn.rs` step seed.
const DEFAULT_SEED: u64 = 0x6764_6e04;
const DEFAULT_WINDOWS: usize = 3;
const POISON_OUTPUT: u8 = 0xFF;
const POISON_STATE: u8 = 0xA5;
/// Single-row EF AR steps that seed the initial q8_ef residual (`< START_POSITION`).
const SEED_STEPS: usize = 4;
/// Bytes of one plain-Q8 slot (`value_heads * 128 * (128 + 4)`).
const SLOT_BYTES: usize = VH * HD * (HD + 4);
/// Bytes of the code plane of a Q8 slot; the F32 scales follow.
const Q8_CODE_BYTES: usize = VH * HD * HD;
/// Bytes of one live F16 residual.
const RESIDUAL_BYTES: usize = 2 * VH * HD * HD;
const _: () = {
    assert!(SEED_STEPS < START_POSITION);
    assert!(GdnStateFormat::Q8.state_bytes(VH, HD, HD) == SLOT_BYTES);
    assert!(GdnStateFormat::Q8Ef.state_bytes(VH, HD, HD) == SLOT_BYTES);
    assert!(GdnStateFormat::Q8Ef.residual_bytes(VH, HD, HD) == RESIDUAL_BYTES);
};

/// `rdna-compute/src/tensor_ops.rs::TENSOR_OPS_SRC`, same concatenation so the
/// kernel module is the one the production launchers load.
const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/rope_yarn.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);

const USAGE: &str = "usage: gdn_state_precision_oracle [--formats q8,fp32,q8_ef] [--graph 0|1] \
[--max-rows 1..=32] [--all-keeps] [--seed N] [--windows W>=2] [--json PATH]";

type Res<T> = Result<T, String>;

fn ctx<T, E: std::fmt::Display>(result: Result<T, E>, what: &str) -> Res<T> {
    result.map_err(|e| format!("{what}: {e}"))
}

// ═══════════════════════════ CLI ══════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq)]
struct Cli {
    formats: Vec<GdnStateFormat>,
    graph: bool,
    max_rows: usize,
    all_keeps: bool,
    seed: u64,
    windows: usize,
    json: Option<PathBuf>,
}

fn label(format: GdnStateFormat) -> &'static str {
    match format {
        GdnStateFormat::F32 => "fp32",
        GdnStateFormat::Q8 => "q8",
        GdnStateFormat::Q8Ef => "q8_ef",
    }
}

fn parse_format(name: &str) -> Res<GdnStateFormat> {
    match name {
        "q8" => Ok(GdnStateFormat::Q8),
        "fp32" | "f32" => Ok(GdnStateFormat::F32),
        "q8_ef" => Ok(GdnStateFormat::Q8Ef),
        other => Err(format!(
            "unknown format '{other}' (expected q8, fp32 or q8_ef)"
        )),
    }
}

fn parse_number(flag: &str, value: &str) -> Res<u64> {
    let parsed = match value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some(hex) => u64::from_str_radix(&hex.replace('_', ""), 16),
        None => value.replace('_', "").parse::<u64>(),
    };
    parsed.map_err(|_| format!("{flag}: '{value}' is not an unsigned integer"))
}

fn parse_args(args: &[String]) -> Res<Cli> {
    let mut formats: Option<Vec<GdnStateFormat>> = None;
    let mut graph = false;
    let mut max_rows = WIDE_VERIFY_MAX_ROWS;
    let mut all_keeps = false;
    let mut seed = DEFAULT_SEED;
    let mut windows = DEFAULT_WINDOWS;
    let mut json = None;
    let mut index = 0;
    while index < args.len() {
        let (flag, inline) = match args[index].split_once('=') {
            Some((flag, value)) => (flag.to_string(), Some(value.to_string())),
            None => (args[index].clone(), None),
        };
        index += 1;
        let value = |index: &mut usize| -> Res<String> {
            if let Some(v) = inline.clone() {
                return Ok(v);
            }
            let v = args
                .get(*index)
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))?;
            *index += 1;
            Ok(v)
        };
        match flag.as_str() {
            "--all-keeps" => {
                if inline.is_some() {
                    return Err("--all-keeps takes no value".to_string());
                }
                all_keeps = true;
            }
            "--formats" => {
                let raw = value(&mut index)?;
                let mut parsed = Vec::new();
                for name in raw.split(',') {
                    let format = parse_format(name.trim())?;
                    if parsed.contains(&format) {
                        return Err(format!("format '{}' listed twice", name.trim()));
                    }
                    parsed.push(format);
                }
                formats = Some(parsed);
            }
            "--graph" => {
                graph = match value(&mut index)?.as_str() {
                    "0" => false,
                    "1" => true,
                    other => return Err(format!("--graph must be 0 or 1, got '{other}'")),
                };
            }
            "--max-rows" => {
                let raw = value(&mut index)?;
                max_rows = parse_number("--max-rows", &raw)? as usize;
                if max_rows == 0 || max_rows > WIDE_VERIFY_MAX_ROWS {
                    return Err(format!(
                        "--max-rows {raw} is outside 1..={WIDE_VERIFY_MAX_ROWS} \
                         (the production wide-verify maximum)"
                    ));
                }
            }
            "--seed" => {
                let raw = value(&mut index)?;
                seed = parse_number("--seed", &raw)?;
            }
            "--windows" => {
                let raw = value(&mut index)?;
                windows = parse_number("--windows", &raw)? as usize;
                if windows < 2 {
                    return Err(format!(
                        "--windows {raw} is below 2: the oracle needs consecutive windows"
                    ));
                }
            }
            "--json" => json = Some(PathBuf::from(value(&mut index)?)),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Cli {
        formats: formats.unwrap_or_else(|| vec![GdnStateFormat::Q8, GdnStateFormat::F32]),
        graph,
        max_rows,
        all_keeps,
        seed,
        windows,
        json,
    })
}

// ═══════════════════════════ pure helpers ═════════════════════════════════════

/// `GdnStateOwner::capture_base`: the ring half that does not hold `source`.
fn capture_base(source: usize, ring_rows: usize) -> usize {
    if source < ring_rows {
        ring_rows
    } else {
        0
    }
}

fn keeps_for(rows: usize, all_keeps: bool) -> Vec<usize> {
    if all_keeps {
        (0..=rows).collect()
    } else {
        vec![0, rows]
    }
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

/// Byte offset of the first non-finite F32 in `bytes`.
fn first_nonfinite(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(4)
        .position(|c| !f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite())
        .map(|i| i * 4)
}

/// Largest finite `|x|` of the F32 `bytes`, and the offset of the first
/// non-finite value.
fn state_stats(bytes: &[u8]) -> (f64, Option<usize>) {
    let mut max = 0.0f64;
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if value.is_finite() {
            max = max.max((value as f64).abs());
        }
    }
    (max, first_nonfinite(bytes))
}

/// Exact F16 bits -> F32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x3ff) as u32;
    let out = match (exponent, fraction) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: value = fraction * 2^-24, renormalized.
            let mut exponent = 113u32;
            let mut fraction = fraction;
            while fraction & 0x400 == 0 {
                fraction <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((fraction & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => sign | 0x7fc0_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(out)
}

/// Byte offset of the first non-finite F16 (little-endian pairs) in `bytes`.
fn first_nonfinite_f16(bytes: &[u8]) -> Option<usize> {
    bytes
        .chunks_exact(2)
        .position(|c| u16::from_le_bytes([c[0], c[1]]) & 0x7c00 == 0x7c00)
        .map(|i| i * 2)
}

/// Largest finite `|x|` of the F16 `bytes`, and the offset of the first
/// non-finite value.
fn f16_stats(bytes: &[u8]) -> (f64, Option<usize>) {
    let mut max = 0.0f64;
    for chunk in bytes.chunks_exact(2) {
        let value = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        if value.is_finite() {
            max = max.max((value as f64).abs());
        }
    }
    (max, first_nonfinite_f16(bytes))
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_f16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Host decode of a plain-Q8 slot (`code * scale` per `(head*128 + row)*128 +
/// column`, the layout of the residual and of `gdn_state_q8_to_f32` up to its
/// transpose); empty when `slot` is not a whole slot.
fn decode_q8_slot(slot: &[u8]) -> Vec<f32> {
    if slot.len() != SLOT_BYTES {
        return Vec::new();
    }
    let (codes, scales) = slot.split_at(Q8_CODE_BYTES);
    codes
        .iter()
        .enumerate()
        .map(|(index, code)| {
            let at = index / HD * 4;
            let scale =
                f32::from_le_bytes([scales[at], scales[at + 1], scales[at + 2], scales[at + 3]]);
            (*code as i8 as f32) * scale
        })
        .collect()
}

/// Largest finite decoded `|x|` of a Q8 slot and the code index of the first
/// non-finite decoded value.
fn q8_slot_stats(slot: &[u8]) -> (f64, Option<usize>) {
    let mut max = 0.0f64;
    let mut bad = None;
    for (index, value) in decode_q8_slot(slot).into_iter().enumerate() {
        if value.is_finite() {
            max = max.max((value as f64).abs());
        } else if bad.is_none() {
            bad = Some(index);
        }
    }
    (max, bad)
}

/// The state the kernels read (`code * scale + residual`, both rounded once).
fn effective_state(slot: &[u8], residual: Option<&[u8]>) -> Vec<f32> {
    let mut state = decode_q8_slot(slot);
    if let Some(residual) = residual {
        for (value, extra) in state.iter_mut().zip(decode_f16(residual)) {
            *value += extra;
        }
    }
    state
}

/// Largest `|a - b|` over the common prefix.
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |max, (x, y)| max.max((*x as f64 - *y as f64).abs()))
}

fn first_nonzero(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|b| *b != 0)
}

/// A non-empty run of nothing but the state poison byte: the slot was never written.
fn is_poison(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|b| *b == POISON_STATE)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// RNE F32 -> BF16 bits (`qwen4_tierc/common.rs::f32_to_bf16_bits`).
fn f32_to_bf16_bits(value: f32) -> u16 {
    let u = value.to_bits();
    if value.is_nan() {
        return ((u >> 16) as u16) | 0x0040;
    }
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

fn bf16_round(value: f32) -> f32 {
    f32::from_bits((f32_to_bf16_bits(value) as u32) << 16)
}

/// SplitMix64 (`qwen4_tierc/common.rs::Rng`).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.range(lo, hi)).collect()
    }
}

struct Steps {
    proj: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

/// `qwen4_tierc/gdn.rs::make_steps` ranges (no zeroed gates); the projection
/// is BF16-rounded like the convolution output the route reads.
fn make_steps(rows: usize, seed: u64) -> Steps {
    let mut rng = Rng::new(seed);
    let mut proj = Vec::with_capacity(rows * QKV);
    for _ in 0..rows {
        proj.extend(rng.vec(2 * QK, -2.0, 2.0));
        proj.extend(rng.vec(VW, -1.0, 1.0));
    }
    for value in &mut proj {
        *value = bf16_round(*value);
    }
    let mut gate = Vec::with_capacity(rows * VH);
    for _ in 0..rows * VH {
        gate.push(rng.range(-0.3, -0.02));
    }
    let beta = rng.vec(rows * VH, 0.1, 0.9);
    Steps { proj, gate, beta }
}

/// Seed of one `(rows, window)` input block, shared by every format.
fn window_seed(seed: u64, rows: usize, window: usize) -> u64 {
    seed ^ ((rows as u64) << 32) ^ ((window as u64 + 1) << 8) ^ DEFAULT_SEED
}

fn state_seed(seed: u64, rows: usize) -> u64 {
    seed.rotate_left(17) ^ ((rows as u64) << 40) ^ 0x5747_5354_4154_4530
}

// ═══════════════════════════ findings ═════════════════════════════════════════

#[derive(Debug, Clone)]
struct Failure {
    kind: String,
    row: Option<usize>,
    byte_offset: usize,
    detail: String,
}

impl Failure {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "kind": self.kind,
            "row": self.row,
            "byte_offset": self.byte_offset,
            "detail": self.detail,
        })
    }
}

/// First difference between `observed` and `expected`, if any.
fn compare(
    kind: &str,
    observed: &[u8],
    expected: &[u8],
    row_bytes: Option<usize>,
) -> Option<Failure> {
    if observed.len() != expected.len() {
        return Some(Failure {
            kind: kind.to_string(),
            row: None,
            byte_offset: observed.len().min(expected.len()),
            detail: format!("length {} != expected {}", observed.len(), expected.len()),
        });
    }
    first_diff(observed, expected).map(|offset| Failure {
        kind: kind.to_string(),
        row: row_bytes.map(|bytes| offset / bytes),
        byte_offset: offset,
        detail: format!(
            "observed 0x{:02x} expected 0x{:02x}",
            observed[offset], expected[offset]
        ),
    })
}

fn nonfinite(kind: &str, bytes: &[u8], row_bytes: usize) -> Option<Failure> {
    first_nonfinite(bytes).map(|offset| Failure {
        kind: kind.to_string(),
        row: Some(offset / row_bytes),
        byte_offset: offset,
        detail: "non-finite F32".to_string(),
    })
}

struct CaseResult {
    rows: usize,
    keep: usize,
    window: usize,
    failures: Vec<Failure>,
    /// q8_ef only: reported, never failing.
    diagnostics: Vec<Diag>,
}

/// A cross-route comparison that is reported but never fails (q8_ef).
#[derive(Debug, Clone)]
struct Diag {
    kind: String,
    equal: bool,
    row: Option<usize>,
    byte_offset: Option<usize>,
    max_abs_diff: Option<f64>,
    detail: String,
}

impl Diag {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "kind": self.kind,
            "equal": self.equal,
            "row": self.row,
            "byte_offset": self.byte_offset,
            "max_abs_diff": self.max_abs_diff,
            "detail": self.detail,
        })
    }
}

/// Diagnostic of a byte comparison; `max_abs_diff` is the caller's decoded
/// difference (0 when the bytes are equal).
fn diag_bytes(
    kind: &str,
    observed: &[u8],
    expected: &[u8],
    row_bytes: Option<usize>,
    max_abs_diff: Option<f64>,
) -> Diag {
    match compare(kind, observed, expected, row_bytes) {
        None => Diag {
            kind: kind.to_string(),
            equal: true,
            row: None,
            byte_offset: None,
            max_abs_diff: max_abs_diff.map(|_| 0.0),
            detail: "bytes equal".to_string(),
        },
        Some(f) => Diag {
            kind: kind.to_string(),
            equal: false,
            row: f.row,
            byte_offset: Some(f.byte_offset),
            max_abs_diff,
            detail: f.detail,
        },
    }
}

fn diag_f32(kind: &str, observed: &[u8], expected: &[u8], row_bytes: Option<usize>) -> Diag {
    let diff = max_abs_diff(&decode_f32(observed), &decode_f32(expected));
    diag_bytes(kind, observed, expected, row_bytes, Some(diff))
}

fn diag_q8(kind: &str, observed: &[u8], expected: &[u8]) -> Diag {
    let diff = max_abs_diff(&decode_q8_slot(observed), &decode_q8_slot(expected));
    diag_bytes(kind, observed, expected, None, Some(diff))
}

fn diag_f16(kind: &str, observed: &[u8], expected: &[u8]) -> Diag {
    let diff = max_abs_diff(&decode_f16(observed), &decode_f16(expected));
    diag_bytes(kind, observed, expected, Some(HD * 2), Some(diff))
}

/// Effective-state diagnostic; the offset indexes the F32 effective state.
fn diag_effective(kind: &str, observed: &[f32], expected: &[f32]) -> Diag {
    diag_bytes(
        kind,
        &f32_bytes(observed),
        &f32_bytes(expected),
        None,
        Some(max_abs_diff(observed, expected)),
    )
}

struct FormatReport {
    format: GdnStateFormat,
    cases: Vec<CaseResult>,
    max_abs_state: f64,
    kernels: BTreeSet<String>,
    /// q8_ef: largest `|residual|` over the AR snapshots and capture finals.
    max_abs_residual: f64,
    /// q8_ef: `(max, min)` over the `rows` values of the seed residual's max `|x|`.
    seed_residual: Option<(f64, f64)>,
}

impl FormatReport {
    fn new(format: GdnStateFormat) -> Self {
        Self {
            format,
            cases: Vec::new(),
            max_abs_state: 0.0,
            kernels: BTreeSet::new(),
            max_abs_residual: 0.0,
            seed_residual: None,
        }
    }

    /// Cases with at least one hard failure.
    fn mismatches(&self) -> usize {
        self.cases.iter().filter(|c| !c.failures.is_empty()).count()
    }

    /// Hard findings over all cases (window-level findings repeat per case).
    fn hard_failures(&self) -> usize {
        self.cases.iter().map(|c| c.failures.len()).sum()
    }

    fn diagnostic_diffs(&self) -> usize {
        self.cases
            .iter()
            .flat_map(|c| &c.diagnostics)
            .filter(|d| !d.equal)
            .count()
    }

    fn note_seed_residual(&mut self, max: f64) {
        self.seed_residual = Some(match self.seed_residual {
            None => (max, max),
            Some((hi, lo)) => (hi.max(max), lo.min(max)),
        });
    }

    /// Per diagnostic kind: `(total, differing, largest max_abs_diff)`.
    fn diag_summary(&self) -> BTreeMap<String, (usize, usize, f64)> {
        let mut summary: BTreeMap<String, (usize, usize, f64)> = BTreeMap::new();
        for d in self.cases.iter().flat_map(|c| &c.diagnostics) {
            let entry = summary.entry(d.kind.clone()).or_insert((0, 0, 0.0));
            entry.0 += 1;
            entry.1 += usize::from(!d.equal);
            entry.2 = entry.2.max(d.max_abs_diff.unwrap_or(0.0));
        }
        summary
    }
}

// ═══════════════════════════ device plumbing ══════════════════════════════════

fn observe(gpu: &Gpu, kernels: &mut BTreeSet<String>) {
    if let Some(name) = gpu.last_launched_kernel() {
        kernels.insert(name.to_string());
    }
}

fn sync(gpu: &Gpu) -> Res<()> {
    ctx(gpu.hip.device_synchronize(), "device synchronize")
}

fn download(gpu: &Gpu, tensor: &GpuTensor) -> Res<Vec<u8>> {
    sync(gpu)?;
    ctx(gpu.download_raw_bytes(tensor), "download")
}

fn fill(gpu: &Gpu, tensor: &GpuTensor, byte: u8) -> Res<()> {
    sync(gpu)?;
    ctx(
        gpu.hip
            .memcpy_htod(&tensor.buf, &vec![byte; tensor.byte_size()]),
        "poison",
    )
}

fn upload(gpu: &Gpu, tensor: &GpuTensor, data: &[f32]) -> Res<()> {
    sync(gpu)?;
    ctx(
        gpu.hip
            .memcpy_htod(&tensor.sub_offset(0, data.len()).buf, &f32_bytes(data)),
        "upload",
    )
}

/// The production `gdn_state_convert` launch (`tensor_ops.rs`): `gdn_state_q8_to_f32`,
/// or `gdn_state_f32_to_q8` with its requantization `frame`; grid
/// `value_heads * 128` value rows, block 128.
fn gdn_state_convert(
    gpu: &mut Gpu,
    kernel: &str,
    source: *mut std::ffi::c_void,
    destination: *mut std::ffi::c_void,
    frame: Option<u32>,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    ctx(
        gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel),
        kernel,
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(source);
    args.push_ptr(destination);
    args.push_i32(VH as i32);
    if let Some(frame) = frame {
        args.push_u32(frame);
    }
    args.pad_to(16);
    ctx(
        gpu.launch_blob_recorded(
            kernel,
            [(VH * 128) as u32, 1, 1],
            [128, 1, 1],
            0,
            args.as_mut_slice(),
            Default::default(),
        ),
        kernel,
    )?;
    observe(gpu, kernels);
    Ok(())
}

/// Capture `body` into the `GraphState`'s one live graph (dropping the previous
/// one first). Returns the few-row limit seen while capturing. The launches must
/// have been warmed eagerly: nothing executes during capture.
fn capture_graph(
    gpu: &mut Gpu,
    expected_launches: usize,
    body: impl FnOnce(&mut Gpu) -> Res<()>,
) -> Res<usize> {
    sync(gpu)?;
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    {
        let stream = gpu
            .active_stream
            .as_ref()
            .ok_or_else(|| "graph capture: no active stream".to_string())?;
        ctx(
            gpu.graphs
                .begin_graph_capture(&gpu.hip, gpu.device_id, stream),
            "begin graph capture",
        )?;
    }
    let result = body(gpu);
    let limit = gpu.few_row_limit();
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "graph capture: stream vanished".to_string())?;
    if let Err(error) = result {
        gpu.graphs
            .abort_graph_capture(&gpu.hip, gpu.device_id, stream);
        return Err(format!("graph capture body: {error}"));
    }
    ctx(
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, stream),
        "end graph capture",
    )?;
    let launches = gpu.graphs.ar_forward_blobs.len();
    if launches != expected_launches || gpu.graphs.graph_exec.is_none() {
        return Err(format!(
            "graph capture recorded {launches} launches, expected {expected_launches} \
             (or no instantiated graph): not a real capture"
        ));
    }
    Ok(limit)
}

fn replay(gpu: &Gpu) -> Res<()> {
    let stream = gpu
        .active_stream
        .as_ref()
        .ok_or_else(|| "graph replay: no active stream".to_string())?;
    ctx(
        gpu.graphs.graph_launch(&gpu.hip, gpu.device_id, stream),
        "graph replay",
    )?;
    sync(gpu)
}

// ═══════════════════════════ rig ══════════════════════════════════════════════

struct Rig {
    format: GdnStateFormat,
    /// `R`: slots per ring half.
    ring_rows: usize,
    /// `format.storage().state_units` (dtype units per state slot).
    units: usize,
    slot_bytes: usize,
    ring: GpuTensor,
    ar_state: GpuTensor,
    proj: GpuTensor,
    gate: GpuTensor,
    beta: GpuTensor,
    out: GpuTensor,
    ar_out: GpuTensor,
    discard: GpuTensor,
    recurrence: GpuTensor,
    table: GpuTensor,
    /// F32 `[value_head][key][value]` scratch: Q8 init source and decode target.
    dec: GpuTensor,
    /// q8_ef: the AR reference owner's live F16 residual.
    ef_ar: Option<GpuTensor>,
    /// q8_ef: the candidate owner's live F16 residual (a capture leaves its
    /// final residual here; the reset kernel zeroes it).
    ef_cand: Option<GpuTensor>,
}

impl Rig {
    fn new(gpu: &mut Gpu, format: GdnStateFormat, ring_rows: usize) -> Res<Self> {
        // The ring, the source slot and the AR state hold the STORAGE format
        // (plain Q8 for q8_ef); the residual is separate.
        let storage = format.storage();
        let units = storage.state_units(VH, HD, HD);
        let slot_bytes = storage.state_bytes(VH, HD, HD);
        let dtype = storage.dtype();
        let mut specs = vec![
            (2 * ring_rows * units, dtype),
            (units, dtype),
            (ring_rows * QKV, DType::F32),
            (ring_rows * VH, DType::F32),
            (ring_rows * VH, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * VW, DType::F32),
            (ring_rows * (QKV + 2 * VH), DType::F32),
            (STATE_ELEMS, DType::F32),
        ];
        if format.has_residual() {
            let residual_units = format.residual_units(VH, HD, HD);
            specs.push((residual_units, DType::F16));
            specs.push((residual_units, DType::F16));
        }
        let mut owned: Vec<GpuTensor> = Vec::with_capacity(specs.len() + 1);
        for (elements, dtype) in specs {
            match gpu.zeros(&[elements], dtype) {
                Ok(tensor) => owned.push(tensor),
                Err(error) => {
                    free_all(gpu, owned);
                    return Err(format!("allocate {elements} x {dtype:?}: {error}"));
                }
            }
        }
        // `[recurrence input, ring]` native-endian u64 pointer pair (one layer).
        let pointers: Vec<u8> = [owned[8].buf.as_ptr(), owned[0].buf.as_ptr()]
            .iter()
            .flat_map(|p| (*p as usize as u64).to_ne_bytes())
            .collect();
        match gpu.upload_raw(&pointers, &[pointers.len()]) {
            Ok(table) => owned.push(table),
            Err(error) => {
                free_all(gpu, owned);
                return Err(format!("upload rollback table: {error}"));
            }
        }
        let mut it = owned.into_iter();
        let mut next = || it.next().expect("rig tensor");
        let ring = next();
        let ar_state = next();
        let proj = next();
        let gate = next();
        let beta = next();
        let out = next();
        let ar_out = next();
        let discard = next();
        let recurrence = next();
        let dec = next();
        let ef_ar = format.has_residual().then(&mut next);
        let ef_cand = format.has_residual().then(&mut next);
        let table = next();
        Ok(Self {
            format,
            ring_rows,
            units,
            slot_bytes,
            ring,
            ar_state,
            proj,
            gate,
            beta,
            out,
            ar_out,
            discard,
            recurrence,
            table,
            dec,
            ef_ar,
            ef_cand,
        })
    }

    fn slot(&self, index: usize) -> GpuTensor {
        self.ring.sub_offset(index * self.units, self.units)
    }

    /// The AR owner's live residual (q8_ef only).
    fn ef_ar_tensor(&self) -> Res<&GpuTensor> {
        self.ef_ar
            .as_ref()
            .ok_or_else(|| format!("format {} has no AR residual", label(self.format)))
    }

    /// The candidate owner's live residual (q8_ef only).
    fn ef_cand_tensor(&self) -> Res<&GpuTensor> {
        self.ef_cand
            .as_ref()
            .ok_or_else(|| format!("format {} has no candidate residual", label(self.format)))
    }

    fn free(self, gpu: &mut Gpu) -> Res<()> {
        let mut first = None;
        let residuals = [self.ef_ar, self.ef_cand].into_iter().flatten();
        for tensor in [
            self.ring,
            self.ar_state,
            self.proj,
            self.gate,
            self.beta,
            self.out,
            self.ar_out,
            self.discard,
            self.recurrence,
            self.dec,
            self.table,
        ]
        .into_iter()
        .chain(residuals)
        {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("free: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn free_all(gpu: &mut Gpu, tensors: Vec<GpuTensor>) {
    for tensor in tensors {
        let _ = gpu.free_tensor(tensor);
    }
}

// ═══════════════════════════ launches ═════════════════════════════════════════

fn ar_step(
    gpu: &mut Gpu,
    rig: &Rig,
    row: usize,
    position: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    ctx(
        gated_delta_step(
            gpu,
            &GatedDeltaStep {
                q: &rig.proj.sub_offset(row * QKV, QK),
                k: &rig.proj.sub_offset(row * QKV + QK, QK),
                v: &rig.proj.sub_offset(row * QKV + 2 * QK, VW),
                gate: &rig.gate.sub_offset(row * VH, VH),
                beta: &rig.beta.sub_offset(row * VH, VH),
                state: &rig.ar_state,
                format: rig.format,
                residual: rig.ef_ar.as_ref(),
                output: &rig.ar_out.sub_offset(row * VW, VW),
                key_heads: KH,
                value_heads: VH,
                key_dim: HD,
                value_dim: HD,
                position,
            },
        ),
        "gated_delta_step",
    )?;
    observe(gpu, kernels);
    Ok(())
}

fn launch_candidate(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    base: usize,
    rows: usize,
    start: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    gpu.set_wide_verify_rows(rig.ring_rows);
    ctx(
        gated_delta_step_batched(
            gpu,
            &GatedDeltaStepBatched {
                projection: &rig.proj.sub_offset(0, rows * QKV),
                gate: &rig.gate.sub_offset(0, rows * VH),
                beta: &rig.beta.sub_offset(0, rows * VH),
                state: &rig.slot(source),
                format: rig.format,
                residual: rig.ef_cand.as_ref(),
                output: &rig.out.sub_offset(0, rows * VW),
                row_states: Some(&rig.ring.sub_offset(base * rig.units, rows * rig.units)),
                rows,
                qkv_width: QKV,
                key_heads: KH,
                value_heads: VH,
                key_dim: HD,
                value_dim: HD,
                position: start,
            },
        ),
        "gated_delta_step_batched (capture)",
    )?;
    observe(gpu, kernels);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_rollback(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    base: usize,
    rows: usize,
    keep: usize,
    start: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    gpu.set_wide_verify_rows(rig.ring_rows);
    ctx(
        gated_delta_rollback_layers(
            gpu,
            &GatedDeltaRollbackLayers {
                table: &rig.table,
                discard: &rig.discard.sub_offset(0, keep * VW),
                format: rig.format,
                layers: 1,
                rows,
                keep,
                from: source,
                to: base,
                qkv_width: QKV,
                key_heads: KH,
                value_heads: VH,
                position: start + keep - 1,
            },
        ),
        "gated_delta_rollback_layers",
    )?;
    observe(gpu, kernels);
    Ok(())
}

// ═══════════════════════════ reference ════════════════════════════════════════

struct Reference {
    out: Vec<u8>,
    states: Vec<Vec<u8>>,
    /// q8_ef: the AR owner's live residual after every row (empty otherwise).
    residuals: Vec<Vec<u8>>,
    max_abs: f64,
    max_abs_residual: f64,
    failures: Vec<Failure>,
}

/// AR decode of `rows` rows from the source slot, row `r` at position
/// `start + r`, saving every row's state bytes (and, for q8_ef, residual
/// bytes: the AR owner starts from `residual_in`).
fn run_reference(
    gpu: &mut Gpu,
    rig: &Rig,
    source: usize,
    rows: usize,
    start: usize,
    residual_in: Option<&[u8]>,
    kernels: &mut BTreeSet<String>,
) -> Res<Reference> {
    if rig.format.has_residual() != residual_in.is_some() {
        return Err(format!(
            "reference for {}: the incoming residual must be given exactly for a residual format",
            label(rig.format)
        ));
    }
    gpu.set_wide_verify_rows(0);
    fill(gpu, &rig.ar_out, POISON_OUTPUT)?;
    sync(gpu)?;
    ctx(
        gpu.copy_d2d(&rig.slot(source), &rig.ar_state, rig.slot_bytes),
        "copy source to AR state",
    )?;
    if let Some(residual) = residual_in {
        upload_bytes(gpu, rig.ef_ar_tensor()?, residual)?;
    }
    let mut states = Vec::with_capacity(rows);
    let mut residuals = Vec::new();
    let mut max_abs = 0.0f64;
    let mut max_abs_residual = 0.0f64;
    let mut failures = Vec::new();
    for row in 0..rows {
        ar_step(gpu, rig, row, start + row, kernels)?;
        let bytes = download(gpu, &rig.ar_state)?;
        let (max, bad) = match rig.format {
            GdnStateFormat::F32 => state_stats(&bytes),
            GdnStateFormat::Q8 => {
                gdn_state_convert(
                    gpu,
                    "gdn_state_q8_to_f32",
                    rig.ar_state.buf.as_ptr(),
                    rig.dec.buf.as_ptr(),
                    None,
                    kernels,
                )?;
                state_stats(&download(gpu, &rig.dec)?)
            }
            GdnStateFormat::Q8Ef => q8_slot_stats(&bytes),
        };
        max_abs = max_abs.max(max);
        if let Some(offset) = bad {
            failures.push(Failure {
                kind: "nonfinite_reference_state".to_string(),
                row: Some(row),
                byte_offset: offset,
                detail: "non-finite AR state (decoded F32 offset)".to_string(),
            });
        }
        if rig.format.has_residual() {
            let residual = download(gpu, rig.ef_ar_tensor()?)?;
            let (max, bad) = f16_stats(&residual);
            max_abs_residual = max_abs_residual.max(max);
            if let Some(offset) = bad {
                failures.push(Failure {
                    kind: "nonfinite_reference_residual".to_string(),
                    row: Some(row),
                    byte_offset: offset,
                    detail: "non-finite AR residual (F16 byte offset)".to_string(),
                });
            }
            residuals.push(residual);
        }
        states.push(bytes);
    }
    let out = download(gpu, &rig.ar_out.sub_offset(0, rows * VW))?;
    if let Some(f) = nonfinite("nonfinite_reference_output", &out, VW * 4) {
        failures.push(f);
    }
    Ok(Reference {
        out,
        states,
        residuals,
        max_abs,
        max_abs_residual,
        failures,
    })
}

// ═══════════════════════════ snapshots and checks ═════════════════════════════

struct Snap {
    out: Vec<u8>,
    discard: Vec<u8>,
    final_state: Vec<u8>,
    kept: Vec<u8>,
    source: Vec<u8>,
}

fn snapshot(
    gpu: &Gpu,
    rig: &Rig,
    rows: usize,
    keep: usize,
    source: usize,
    base: usize,
) -> Res<Snap> {
    let out = download(gpu, &rig.out.sub_offset(0, rows * VW))?;
    let discard = if keep > 0 {
        download(gpu, &rig.discard.sub_offset(0, keep * VW))?
    } else {
        Vec::new()
    };
    let final_state = download(gpu, &rig.slot(base + rows - 1))?;
    let kept = if keep > 0 {
        download(gpu, &rig.slot(base + keep - 1))?
    } else {
        Vec::new()
    };
    let source = download(gpu, &rig.slot(source))?;
    Ok(Snap {
        out,
        discard,
        final_state,
        kept,
        source,
    })
}

fn check_determinism(prefix: &str, a: &Snap, b: &Snap) -> Vec<Failure> {
    [
        compare(
            &format!("{prefix}nondeterministic_output"),
            &b.out,
            &a.out,
            Some(VW * 4),
        ),
        compare(
            &format!("{prefix}nondeterministic_discard"),
            &b.discard,
            &a.discard,
            Some(VW * 4),
        ),
        compare(
            &format!("{prefix}nondeterministic_final_state"),
            &b.final_state,
            &a.final_state,
            None,
        ),
        compare(
            &format!("{prefix}nondeterministic_kept_state"),
            &b.kept,
            &a.kept,
            None,
        ),
        compare(
            &format!("{prefix}nondeterministic_source"),
            &b.source,
            &a.source,
            None,
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn check_candidate_only(
    snap: &Snap,
    reference: &Reference,
    rows: usize,
    source_bytes: &[u8],
) -> Vec<Failure> {
    [
        compare(
            "candidate_only_output",
            &snap.out,
            &reference.out,
            Some(VW * 4),
        ),
        nonfinite("candidate_only_nonfinite_output", &snap.out, VW * 4),
        compare(
            "candidate_only_final_state",
            &snap.final_state,
            &reference.states[rows - 1],
            None,
        ),
        compare(
            "candidate_only_source_modified",
            &snap.source,
            source_bytes,
            None,
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn check_case(
    snap: &Snap,
    reference: &Reference,
    rows: usize,
    keep: usize,
    source_bytes: &[u8],
) -> Vec<Failure> {
    [
        compare("candidate_output", &snap.out, &reference.out, Some(VW * 4)),
        nonfinite("nonfinite_output", &snap.out, VW * 4),
        compare(
            "rollback_discard",
            &snap.discard,
            &reference.out[..keep * VW * 4],
            Some(VW * 4),
        ),
        nonfinite("nonfinite_discard", &snap.discard, VW * 4),
        compare(
            "final_state",
            &snap.final_state,
            &reference.states[rows - 1],
            None,
        ),
        compare("kept_state", &snap.kept, &reference.states[keep - 1], None),
        compare("source_modified", &snap.source, source_bytes, None),
    ]
    .into_iter()
    .flatten()
    .collect()
}

// ═══════════════════════════ case runners ═════════════════════════════════════

struct Window<'a> {
    graph: bool,
    source: usize,
    base: usize,
    rows: usize,
    start: usize,
    source_bytes: &'a [u8],
    reference: &'a Reference,
}

/// Candidate alone, twice: byte-checked before any rollback can rewrite the
/// final slot.
fn run_candidate_only(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &Window<'_>,
    kernels: &mut BTreeSet<String>,
    capture_limit: &mut Option<usize>,
) -> Res<Vec<Failure>> {
    if w.graph {
        let limit = capture_graph(gpu, 1, |g| {
            launch_candidate(g, rig, w.source, w.base, w.rows, w.start, kernels)
        })?;
        capture_limit.get_or_insert(limit);
    }
    let mut snaps = Vec::with_capacity(2);
    for _ in 0..2 {
        fill(gpu, &rig.out, POISON_OUTPUT)?;
        fill(gpu, &rig.slot(w.base + w.rows - 1), POISON_STATE)?;
        if w.graph {
            replay(gpu)?;
        } else {
            launch_candidate(gpu, rig, w.source, w.base, w.rows, w.start, kernels)?;
            sync(gpu)?;
        }
        snaps.push(snapshot(gpu, rig, w.rows, 0, w.source, w.base)?);
    }
    let mut failures = check_candidate_only(&snaps[0], w.reference, w.rows, w.source_bytes);
    failures.extend(check_determinism("candidate_only_", &snaps[0], &snaps[1]));
    Ok(failures)
}

/// Candidate + rollback for one `keep >= 1`, twice.
fn run_keep(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &Window<'_>,
    keep: usize,
    kernels: &mut BTreeSet<String>,
    capture_limit: &mut Option<usize>,
) -> Res<Vec<Failure>> {
    if w.graph {
        let limit = capture_graph(gpu, 2, |g| {
            launch_candidate(g, rig, w.source, w.base, w.rows, w.start, kernels)?;
            launch_rollback(g, rig, w.source, w.base, w.rows, keep, w.start, kernels)
        })?;
        capture_limit.get_or_insert(limit);
    }
    let mut snaps = Vec::with_capacity(2);
    for _ in 0..2 {
        fill(gpu, &rig.out, POISON_OUTPUT)?;
        fill(gpu, &rig.discard, POISON_OUTPUT)?;
        fill(gpu, &rig.slot(w.base + w.rows - 1), POISON_STATE)?;
        fill(gpu, &rig.slot(w.base + keep - 1), POISON_STATE)?;
        if w.graph {
            replay(gpu)?;
        } else {
            launch_candidate(gpu, rig, w.source, w.base, w.rows, w.start, kernels)?;
            launch_rollback(gpu, rig, w.source, w.base, w.rows, keep, w.start, kernels)?;
            sync(gpu)?;
        }
        snaps.push(snapshot(gpu, rig, w.rows, keep, w.source, w.base)?);
    }
    let mut failures = check_case(&snaps[0], w.reference, w.rows, keep, w.source_bytes);
    failures.extend(check_determinism("", &snaps[0], &snaps[1]));
    Ok(failures)
}

// ═══════════════════════════ q8_ef cases ══════════════════════════════════════

/// The launch sequence of one q8_ef case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EfMode {
    /// Candidate alone: the window's capture and `keep == rows` (all-accept
    /// commit: the live residual is kept).
    Capture,
    /// `keep == 0`: candidate, then the live-residual reset (the source slot
    /// is the restore target and stays untouched).
    Restore,
    /// `0 < keep < rows`: candidate, the plain-Q8 rollback, the reset.
    Rollback(usize),
}

impl EfMode {
    /// Kernel launches of the sequence (the graph capture's expected count).
    const fn launches(self) -> usize {
        match self {
            Self::Capture => 1,
            Self::Restore => 2,
            Self::Rollback(_) => 3,
        }
    }
}

fn ef_mode(keep: usize, rows: usize) -> EfMode {
    if keep == rows {
        EfMode::Capture
    } else if keep == 0 {
        EfMode::Restore
    } else {
        EfMode::Rollback(keep)
    }
}

struct EfWindow<'a> {
    graph: bool,
    source: usize,
    base: usize,
    rows: usize,
    start: usize,
    /// Pre-window bytes of the source slot.
    source_bytes: &'a [u8],
    /// Live residual both owners start the window with.
    residual: &'a [u8],
}

struct EfSnap {
    out: Vec<u8>,
    discard: Vec<u8>,
    final_slot: Vec<u8>,
    kept: Vec<u8>,
    source: Vec<u8>,
    /// The candidate owner's live residual after the sequence.
    residual: Vec<u8>,
    /// Offset (within the candidate view) of the first byte of a slot that
    /// must not have been written but no longer holds the poison.
    stray: Option<usize>,
}

fn failure(kind: &str, row: Option<usize>, byte_offset: usize, detail: &str) -> Failure {
    Failure {
        kind: kind.to_string(),
        row,
        byte_offset,
        detail: detail.to_string(),
    }
}

fn upload_bytes(gpu: &Gpu, tensor: &GpuTensor, data: &[u8]) -> Res<()> {
    sync(gpu)?;
    ctx(gpu.hip.memcpy_htod(&tensor.buf, data), "upload bytes")
}

/// The production residual reset (`gdn_q8ef_residual_zero`) of the candidate
/// owner's live residual.
fn ef_zero(gpu: &mut Gpu, rig: &Rig, kernels: &mut BTreeSet<String>) -> Res<()> {
    ctx(
        gdn_q8ef_residual_zero(gpu, rig.ef_cand_tensor()?),
        "gdn_q8ef_residual_zero",
    )?;
    observe(gpu, kernels);
    Ok(())
}

fn ef_launch(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &EfWindow<'_>,
    mode: EfMode,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    launch_candidate(gpu, rig, w.source, w.base, w.rows, w.start, kernels)?;
    match mode {
        EfMode::Capture => Ok(()),
        EfMode::Restore => ef_zero(gpu, rig, kernels),
        EfMode::Rollback(keep) => {
            launch_rollback(gpu, rig, w.source, w.base, w.rows, keep, w.start, kernels)?;
            ef_zero(gpu, rig, kernels)
        }
    }
}

fn snapshot_ef(gpu: &Gpu, rig: &Rig, w: &EfWindow<'_>, mode: EfMode) -> Res<EfSnap> {
    let rows = w.rows;
    let keep = match mode {
        EfMode::Rollback(keep) => keep,
        EfMode::Capture | EfMode::Restore => 0,
    };
    let out = download(gpu, &rig.out.sub_offset(0, rows * VW))?;
    let discard = if keep > 0 {
        download(gpu, &rig.discard.sub_offset(0, keep * VW))?
    } else {
        Vec::new()
    };
    let view = download(
        gpu,
        &rig.ring.sub_offset(w.base * rig.units, rows * rig.units),
    )?;
    let slot = |index: usize| &view[index * rig.slot_bytes..(index + 1) * rig.slot_bytes];
    let kept_index = (keep > 0).then(|| keep - 1);
    let stray = (0..rows)
        .filter(|index| *index != rows - 1 && Some(*index) != kept_index)
        .find_map(|index| {
            slot(index)
                .iter()
                .position(|b| *b != POISON_STATE)
                .map(|offset| index * rig.slot_bytes + offset)
        });
    let final_slot = slot(rows - 1).to_vec();
    let kept = kept_index
        .map(|index| slot(index).to_vec())
        .unwrap_or_default();
    let source = download(gpu, &rig.slot(w.source))?;
    let residual = download(gpu, rig.ef_cand_tensor()?)?;
    Ok(EfSnap {
        out,
        discard,
        final_slot,
        kept,
        source,
        residual,
        stray,
    })
}

/// One execution of a case sequence from the window's source slot and
/// residual: poison the outputs and the whole candidate view, load the
/// residual into the candidate owner (outside any graph), run eagerly or by
/// graph replay.
fn ef_run_once(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &EfWindow<'_>,
    mode: EfMode,
    replayed: bool,
    kernels: &mut BTreeSet<String>,
) -> Res<EfSnap> {
    fill(gpu, &rig.out, POISON_OUTPUT)?;
    fill(gpu, &rig.discard, POISON_OUTPUT)?;
    fill(
        gpu,
        &rig.ring.sub_offset(w.base * rig.units, w.rows * rig.units),
        POISON_STATE,
    )?;
    upload_bytes(gpu, rig.ef_cand_tensor()?, w.residual)?;
    if replayed {
        replay(gpu)?;
    } else {
        ef_launch(gpu, rig, w, mode, kernels)?;
        sync(gpu)?;
    }
    snapshot_ef(gpu, rig, w, mode)
}

/// Byte equality of two executions of one sequence (`label` names the pair:
/// `nondeterministic`, `graph_nondeterministic`, `graph_vs_eager`).
fn check_ef_same(label: &str, a: &EfSnap, b: &EfSnap) -> Vec<Failure> {
    let mut failures: Vec<Failure> = [
        compare(&format!("{label}_output"), &b.out, &a.out, Some(VW * 4)),
        compare(
            &format!("{label}_discard"),
            &b.discard,
            &a.discard,
            Some(VW * 4),
        ),
        compare(
            &format!("{label}_final_state"),
            &b.final_slot,
            &a.final_slot,
            None,
        ),
        compare(&format!("{label}_kept_state"), &b.kept, &a.kept, None),
        compare(&format!("{label}_source"), &b.source, &a.source, None),
        compare(
            &format!("{label}_residual"),
            &b.residual,
            &a.residual,
            Some(HD * 2),
        ),
    ]
    .into_iter()
    .flatten()
    .collect();
    if a.stray != b.stray {
        failures.push(failure(
            &format!("{label}_ring_stray_write"),
            None,
            b.stray.or(a.stray).unwrap_or(0),
            &format!("stray ring write {:?} != {:?}", b.stray, a.stray),
        ));
    }
    failures
}

/// Run a sequence twice eagerly (determinism) and, with `--graph 1`, capture
/// it into one graph and replay it twice (graph determinism and graph versus
/// eager). The eager first run is returned for the per-case checks.
fn ef_exec(
    gpu: &mut Gpu,
    rig: &Rig,
    w: &EfWindow<'_>,
    mode: EfMode,
    kernels: &mut BTreeSet<String>,
    capture_limit: &mut Option<usize>,
) -> Res<(EfSnap, Vec<Failure>)> {
    let first = ef_run_once(gpu, rig, w, mode, false, kernels)?;
    let second = ef_run_once(gpu, rig, w, mode, false, kernels)?;
    let mut failures = check_ef_same("nondeterministic", &first, &second);
    if w.graph {
        let limit = capture_graph(gpu, mode.launches(), |g| {
            ef_launch(g, rig, w, mode, kernels)
        })?;
        capture_limit.get_or_insert(limit);
        let replay_a = ef_run_once(gpu, rig, w, mode, true, kernels)?;
        let replay_b = ef_run_once(gpu, rig, w, mode, true, kernels)?;
        failures.extend(check_ef_same(
            "graph_nondeterministic",
            &replay_a,
            &replay_b,
        ));
        failures.extend(check_ef_same("graph_vs_eager", &first, &replay_a));
    }
    Ok((first, failures))
}

/// The HARD per-execution checks of a q8_ef case. `cap` is the window's
/// candidate-alone capture (`None` while checking the capture itself).
fn check_ef_snap(
    snap: &EfSnap,
    source_bytes: &[u8],
    residual_in: &[u8],
    mode: EfMode,
    cap: Option<&EfSnap>,
) -> Vec<Failure> {
    let mut failures = Vec::new();
    failures.extend(compare("source_modified", &snap.source, source_bytes, None));
    if let Some(offset) = snap.stray {
        failures.push(failure(
            "ring_stray_write",
            Some(offset / SLOT_BYTES),
            offset,
            "a candidate-view slot other than the final slot / rollback target was written",
        ));
    }
    failures.extend(nonfinite("nonfinite_output", &snap.out, VW * 4));
    if is_poison(&snap.final_slot) {
        failures.push(failure(
            "final_slot_unwritten",
            None,
            0,
            "final slot still poison",
        ));
    }
    if let Some(index) = q8_slot_stats(&snap.final_slot).1 {
        failures.push(failure(
            "nonfinite_final_state",
            None,
            index,
            "non-finite decoded state",
        ));
    }
    if let Some(offset) = first_nonfinite_f16(&snap.residual) {
        failures.push(failure(
            "nonfinite_residual",
            Some(offset / (HD * 2)),
            offset,
            "non-finite F16 residual",
        ));
    }
    if snap.residual.len() != RESIDUAL_BYTES {
        failures.push(failure(
            "residual_length",
            None,
            snap.residual.len().min(RESIDUAL_BYTES),
            &format!(
                "residual is {} bytes, expected {RESIDUAL_BYTES}",
                snap.residual.len()
            ),
        ));
    }
    if let EfMode::Rollback(_) = mode {
        failures.extend(nonfinite("nonfinite_discard", &snap.discard, VW * 4));
        if is_poison(&snap.kept) {
            failures.push(failure(
                "rollback_target_unwritten",
                None,
                0,
                "rollback target still poison",
            ));
        }
        if let Some(index) = q8_slot_stats(&snap.kept).1 {
            failures.push(failure(
                "nonfinite_kept_state",
                None,
                index,
                "non-finite decoded state",
            ));
        }
    }
    if matches!(mode, EfMode::Restore | EfMode::Rollback(_)) {
        if let Some(offset) = first_nonzero(&snap.residual) {
            failures.push(failure(
                "residual_not_zero",
                Some(offset / (HD * 2)),
                offset,
                "live residual not exactly zero after the reset event",
            ));
        }
    }
    if mode == EfMode::Capture
        && first_nonzero(residual_in).is_some()
        && snap.residual == residual_in
    {
        failures.push(failure(
            "capture_residual_unwritten",
            None,
            0,
            "the capture left the incoming residual unchanged",
        ));
    }
    if let Some(cap) = cap {
        failures.extend(compare(
            "output_vs_capture",
            &snap.out,
            &cap.out,
            Some(VW * 4),
        ));
        failures.extend(compare(
            "final_state_vs_capture",
            &snap.final_slot,
            &cap.final_slot,
            None,
        ));
        if mode == EfMode::Capture {
            failures.extend(compare(
                "residual_not_kept",
                &snap.residual,
                &cap.residual,
                Some(HD * 2),
            ));
        }
    }
    failures
}

/// AR-with-EF against the candidate capture (never failing).
fn ef_capture_diags(cap: &EfSnap, reference: &Reference, rows: usize) -> Vec<Diag> {
    let last = rows - 1;
    vec![
        diag_f32(
            "ar_vs_capture_output",
            &cap.out,
            &reference.out,
            Some(VW * 4),
        ),
        diag_q8(
            "ar_vs_capture_final_slot",
            &cap.final_slot,
            &reference.states[last],
        ),
        diag_f16(
            "ar_vs_capture_final_residual",
            &cap.residual,
            &reference.residuals[last],
        ),
        diag_effective(
            "ar_vs_capture_effective_state",
            &effective_state(&cap.final_slot, Some(&cap.residual)),
            &effective_state(&reference.states[last], Some(&reference.residuals[last])),
        ),
    ]
}

/// AR-with-EF after row `keep - 1` against the rollback-plus-reset state
/// (never failing; expected to differ).
fn ef_rollback_diags(snap: &EfSnap, reference: &Reference, keep: usize) -> Vec<Diag> {
    let row = keep - 1;
    vec![
        diag_q8("ar_vs_rollback_slot", &snap.kept, &reference.states[row]),
        diag_effective(
            "ar_vs_rollback_effective_state",
            &effective_state(&snap.kept, None),
            &effective_state(&reference.states[row], Some(&reference.residuals[row])),
        ),
        diag_f32(
            "ar_vs_rollback_discard",
            &snap.discard,
            &reference.out[..keep * VW * 4],
            Some(VW * 4),
        ),
    ]
}

/// Source slot 0 := the shared Q8 init, then `SEED_STEPS` EF AR steps from a
/// zero residual on the AR owner; the stepped state becomes slot 0 and the
/// residual they leave (returned) is the first window's incoming residual.
fn init_ef_source(
    gpu: &mut Gpu,
    rig: &Rig,
    seed: u64,
    rows: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<Vec<u8>> {
    init_source(gpu, rig, seed, rows, kernels)?;
    gpu.set_wide_verify_rows(0);
    sync(gpu)?;
    ctx(
        gpu.copy_d2d(&rig.slot(0), &rig.ar_state, rig.slot_bytes),
        "copy source to AR state (seed)",
    )?;
    fill(gpu, rig.ef_ar_tensor()?, 0)?;
    let steps = make_steps(SEED_STEPS, state_seed(seed, rows) ^ 0x4546_5345_4544_0000);
    upload(gpu, &rig.proj, &steps.proj)?;
    upload(gpu, &rig.gate, &steps.gate)?;
    upload(gpu, &rig.beta, &steps.beta)?;
    for row in 0..SEED_STEPS {
        ar_step(gpu, rig, row, START_POSITION - SEED_STEPS + row, kernels)?;
    }
    sync(gpu)?;
    ctx(
        gpu.copy_d2d(&rig.ar_state, &rig.slot(0), rig.slot_bytes),
        "commit the seeded state to the source slot",
    )?;
    download(gpu, rig.ef_ar_tensor()?)
}

fn print_mismatch(
    format_label: &str,
    graph: u8,
    rows: usize,
    keep: usize,
    window: usize,
    failures: &[Failure],
) {
    if let Some(first) = failures.first() {
        println!(
            "MISMATCH format={format_label} graph={graph} rows={rows} keep={keep} \
             window={window} first_kind={} byte_offset={} row={} failures={} \
             kinds={} detail=\"{}\"",
            first.kind,
            first.byte_offset,
            first.row.map_or("-".to_string(), |r| r.to_string()),
            failures.len(),
            failures
                .iter()
                .map(|f| f.kind.as_str())
                .collect::<Vec<_>>()
                .join("|"),
            first.detail,
        );
    }
}

fn print_diags(
    format_label: &str,
    graph: u8,
    rows: usize,
    keep: usize,
    window: usize,
    diags: &[Diag],
) {
    for d in diags {
        println!(
            "diag: format={format_label} graph={graph} rows={rows} keep={keep} window={window} \
             kind={} equal={} first_byte_offset={} row={} max_abs_diff={} detail=\"{}\"",
            d.kind,
            u8::from(d.equal),
            d.byte_offset.map_or("-".to_string(), |o| o.to_string()),
            d.row.map_or("-".to_string(), |r| r.to_string()),
            d.max_abs_diff.map_or("-".to_string(), |m| format!("{m:e}")),
            d.detail,
        );
    }
}

/// The q8_ef window loop (see the module docs for the checks).
fn run_cases_ef(
    gpu: &mut Gpu,
    cli: &Cli,
    rig: &Rig,
    report: &mut FormatReport,
    capture_limit: &mut Option<usize>,
) -> Res<()> {
    let format_label = label(rig.format);
    let graph = u8::from(cli.graph);
    let mut kernels = BTreeSet::new();
    prewarm(gpu, rig, &mut kernels)?;
    for rows in 1..=cli.max_rows {
        let seed_residual = init_ef_source(gpu, rig, cli.seed, rows, &mut kernels)?;
        let (seed_max, seed_bad) = f16_stats(&seed_residual);
        report.note_seed_residual(seed_max);
        let mut seed_failures = Vec::new();
        if seed_max == 0.0 {
            seed_failures.push(failure(
                "seed_residual_zero",
                None,
                0,
                "the EF seed steps left an all-zero residual",
            ));
        }
        if let Some(offset) = seed_bad {
            seed_failures.push(failure(
                "nonfinite_seed_residual",
                None,
                offset,
                "non-finite F16",
            ));
        }
        let mut residual = seed_residual;
        let mut source = 0usize;
        for window in 0..cli.windows {
            let start = START_POSITION + window * rows;
            upload_window_inputs(gpu, rig, rows, cli.seed, window)?;
            let base = capture_base(source, rig.ring_rows);
            let source_bytes = download(gpu, &rig.slot(source))?;
            let reference =
                run_reference(gpu, rig, source, rows, start, Some(&residual), &mut kernels)?;
            report.max_abs_state = report.max_abs_state.max(reference.max_abs);
            report.max_abs_residual = report.max_abs_residual.max(reference.max_abs_residual);
            let w = EfWindow {
                graph: cli.graph,
                source,
                base,
                rows,
                start,
                source_bytes: &source_bytes,
                residual: &residual,
            };
            let mut window_failures = reference.failures.clone();
            if window == 0 {
                window_failures.extend(seed_failures.iter().cloned());
            }
            let (cap, capture_failures) =
                ef_exec(gpu, rig, &w, EfMode::Capture, &mut kernels, capture_limit)?;
            window_failures.extend(capture_failures);
            window_failures.extend(check_ef_snap(
                &cap,
                &source_bytes,
                &residual,
                EfMode::Capture,
                None,
            ));
            report.max_abs_state = report.max_abs_state.max(q8_slot_stats(&cap.final_slot).0);
            report.max_abs_residual = report.max_abs_residual.max(f16_stats(&cap.residual).0);
            let capture_diags = ef_capture_diags(&cap, &reference, rows);
            for keep in keeps_for(rows, cli.all_keeps) {
                let mode = ef_mode(keep, rows);
                let mut failures = window_failures.clone();
                let (snap, run_failures) =
                    ef_exec(gpu, rig, &w, mode, &mut kernels, capture_limit)?;
                failures.extend(run_failures);
                failures.extend(check_ef_snap(
                    &snap,
                    &source_bytes,
                    &residual,
                    mode,
                    Some(&cap),
                ));
                let diagnostics = match mode {
                    EfMode::Capture => capture_diags.clone(),
                    EfMode::Rollback(keep) => ef_rollback_diags(&snap, &reference, keep),
                    EfMode::Restore => Vec::new(),
                };
                print_mismatch(format_label, graph, rows, keep, window, &failures);
                print_diags(format_label, graph, rows, keep, window, &diagnostics);
                report.cases.push(CaseResult {
                    rows,
                    keep,
                    window,
                    failures,
                    diagnostics,
                });
            }
            // The capture's final slot and residual continue into the next window.
            upload_bytes(gpu, &rig.slot(base + rows - 1), &cap.final_slot)?;
            residual = cap.residual;
            source = base + rows - 1;
        }
        eprintln!(
            "{format_label} graph={graph} rows={rows}/{} done",
            cli.max_rows
        );
    }
    report.kernels.extend(kernels);
    Ok(())
}

/// Load the AR, capture, rollback, residual-reset and converter kernels
/// before any capture.
fn prewarm(gpu: &mut Gpu, rig: &Rig, kernels: &mut BTreeSet<String>) -> Res<()> {
    gpu.set_wide_verify_rows(0);
    ar_step(gpu, rig, 0, START_POSITION, kernels)?;
    launch_candidate(gpu, rig, 0, rig.ring_rows, 1, START_POSITION, kernels)?;
    launch_rollback(gpu, rig, 0, rig.ring_rows, 1, 1, START_POSITION, kernels)?;
    if rig.format.has_residual() {
        ef_zero(gpu, rig, kernels)?;
    }
    if rig.format.storage() == GdnStateFormat::Q8 {
        gdn_state_convert(
            gpu,
            "gdn_state_f32_to_q8",
            rig.dec.buf.as_ptr(),
            rig.ar_state.buf.as_ptr(),
            Some(START_POSITION as u32),
            kernels,
        )?;
        gdn_state_convert(
            gpu,
            "gdn_state_q8_to_f32",
            rig.ar_state.buf.as_ptr(),
            rig.dec.buf.as_ptr(),
            None,
            kernels,
        )?;
    }
    sync(gpu)
}

/// One `(rows, window)` input block (projection, gate, beta and the packed
/// rollback recurrence input).
fn upload_window_inputs(gpu: &Gpu, rig: &Rig, rows: usize, seed: u64, window: usize) -> Res<()> {
    let steps = make_steps(rows, window_seed(seed, rows, window));
    upload(gpu, &rig.proj, &steps.proj)?;
    upload(gpu, &rig.gate, &steps.gate)?;
    upload(gpu, &rig.beta, &steps.beta)?;
    let packed: Vec<f32> = steps
        .proj
        .iter()
        .chain(&steps.gate)
        .chain(&steps.beta)
        .copied()
        .collect();
    upload(gpu, &rig.recurrence, &packed)
}

/// Source slot 0 := small random F32 state, as Q8 through the production
/// converter for the Q8 formats (plain Q8 and the plain-Q8 slot of q8_ef).
fn init_source(
    gpu: &mut Gpu,
    rig: &Rig,
    seed: u64,
    rows: usize,
    kernels: &mut BTreeSet<String>,
) -> Res<()> {
    let state = Rng::new(state_seed(seed, rows)).vec(STATE_ELEMS, -0.1, 0.1);
    let slot0 = rig.slot(0);
    match rig.format {
        GdnStateFormat::F32 => upload(gpu, &slot0, &state),
        GdnStateFormat::Q8 | GdnStateFormat::Q8Ef => {
            upload(gpu, &rig.dec, &state)?;
            gdn_state_convert(
                gpu,
                "gdn_state_f32_to_q8",
                rig.dec.buf.as_ptr(),
                slot0.buf.as_ptr(),
                Some((START_POSITION - 1) as u32),
                kernels,
            )?;
            sync(gpu)
        }
    }
}

fn run_cases(
    gpu: &mut Gpu,
    cli: &Cli,
    rig: &Rig,
    report: &mut FormatReport,
    capture_limit: &mut Option<usize>,
) -> Res<()> {
    let format_label = label(rig.format);
    let graph = u8::from(cli.graph);
    let mut kernels = BTreeSet::new();
    prewarm(gpu, rig, &mut kernels)?;
    for rows in 1..=cli.max_rows {
        init_source(gpu, rig, cli.seed, rows, &mut kernels)?;
        let mut source = 0usize;
        for window in 0..cli.windows {
            let start = START_POSITION + window * rows;
            upload_window_inputs(gpu, rig, rows, cli.seed, window)?;
            let base = capture_base(source, rig.ring_rows);
            let source_bytes = download(gpu, &rig.slot(source))?;
            let reference = run_reference(gpu, rig, source, rows, start, None, &mut kernels)?;
            report.max_abs_state = report.max_abs_state.max(reference.max_abs);
            let w = Window {
                graph: cli.graph,
                source,
                base,
                rows,
                start,
                source_bytes: &source_bytes,
                reference: &reference,
            };
            let mut window_failures = reference.failures.clone();
            window_failures.extend(run_candidate_only(
                gpu,
                rig,
                &w,
                &mut kernels,
                capture_limit,
            )?);
            for keep in keeps_for(rows, cli.all_keeps) {
                let mut failures = window_failures.clone();
                if keep > 0 {
                    failures.extend(run_keep(gpu, rig, &w, keep, &mut kernels, capture_limit)?);
                }
                if let Some(first) = failures.first() {
                    println!(
                        "MISMATCH format={format_label} graph={graph} rows={rows} keep={keep} \
                         window={window} first_kind={} byte_offset={} row={} failures={} \
                         kinds={} detail=\"{}\"",
                        first.kind,
                        first.byte_offset,
                        first.row.map_or("-".to_string(), |r| r.to_string()),
                        failures.len(),
                        failures
                            .iter()
                            .map(|f| f.kind.as_str())
                            .collect::<Vec<_>>()
                            .join("|"),
                        first.detail,
                    );
                }
                report.cases.push(CaseResult {
                    rows,
                    keep,
                    window,
                    failures,
                    diagnostics: Vec::new(),
                });
            }
            // The committed final slot is the next window's source.
            source = base + rows - 1;
        }
        eprintln!(
            "{format_label} graph={graph} rows={rows}/{} done",
            cli.max_rows
        );
    }
    report.kernels.extend(kernels);
    Ok(())
}

fn run_format(
    gpu: &mut Gpu,
    cli: &Cli,
    format: GdnStateFormat,
    ring_rows: usize,
    capture_limit: &mut Option<usize>,
) -> Res<FormatReport> {
    let rig = Rig::new(gpu, format, ring_rows)?;
    let mut report = FormatReport::new(format);
    let result = if format.has_residual() {
        run_cases_ef(gpu, cli, &rig, &mut report, capture_limit)
    } else {
        run_cases(gpu, cli, &rig, &mut report, capture_limit)
    };
    gpu.set_wide_verify_rows(0);
    gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
    let freed = rig.free(gpu);
    result?;
    freed?;
    Ok(report)
}

// ═══════════════════════════ main ═════════════════════════════════════════════

fn run(cli: &Cli) -> Res<i32> {
    let mut gpu = ctx(Gpu::init(), "Gpu::init")?;
    if !gpu.arch_caps.has_gfx11_plus_simt() {
        return Err(format!(
            "arch {} has no gfx11+ SIMT row-capture GDN route (the production verify capture \
             is unavailable here)",
            gpu.arch
        ));
    }
    if cli.graph {
        ctx(gpu.ensure_capture_stream(), "graph route: capture stream")?;
    }
    let arch = gpu.arch.clone();
    // The production verify forward's scope (`gpu_forward.rs`).
    gpu.qwen4_scope = true;
    gpu.prefill_invariant_scope = false;
    let ring_rows = cli.max_rows.max(FEW_ROW_ROUTE_ROWS);
    gpu.set_wide_verify_rows(ring_rows);
    let eager_limit = gpu.few_row_limit();
    gpu.set_wide_verify_rows(0);
    println!(
        "gdn_state_precision_oracle arch={arch} geometry=KH{KH}/VH{VH}/K{HD}/V{HD}/QKV{QKV} \
         graph={} max_rows={} ring_rows={ring_rows} windows={} all_keeps={} seed={} \
         qwen4_scope=true prefill_invariant_scope=false wide_verify_rows={ring_rows} \
         few_row_limit_eager={eager_limit}",
        u8::from(cli.graph),
        cli.max_rows,
        cli.windows,
        cli.all_keeps,
        cli.seed,
    );
    let mut capture_limit: Option<usize> = None;
    let mut reports = Vec::new();
    let mut failed = None;
    for &format in &cli.formats {
        match run_format(&mut gpu, cli, format, ring_rows, &mut capture_limit) {
            Ok(report) => {
                println!(
                    "state format={} max_abs_state={:e} state_units={} state_bytes={}{}",
                    label(format),
                    report.max_abs_state,
                    format.state_units(VH, HD, HD),
                    format.state_bytes(VH, HD, HD),
                    if format.has_residual() {
                        let (seed_max, seed_min) = report.seed_residual.unwrap_or((0.0, 0.0));
                        format!(
                            " max_abs_residual={:e} seed_residual_max_abs={seed_max:e} \
                             seed_residual_min_abs={seed_min:e} residual_bytes={}",
                            report.max_abs_residual,
                            format.residual_bytes(VH, HD, HD),
                        )
                    } else {
                        String::new()
                    },
                );
                println!(
                    "oracle: format={} graph={} cases={} mismatches={} kernels={}{}",
                    label(format),
                    u8::from(cli.graph),
                    report.cases.len(),
                    report.mismatches(),
                    report.kernels.iter().cloned().collect::<Vec<_>>().join(","),
                    if format.has_residual() {
                        format!(
                            " hard_failures={} diagnostic_diffs={}",
                            report.hard_failures(),
                            report.diagnostic_diffs(),
                        )
                    } else {
                        String::new()
                    },
                );
                for (kind, (total, differing, max_diff)) in report.diag_summary() {
                    println!(
                        "diag-summary: format={} graph={} kind={kind} total={total} \
                         differing={differing} max_abs_diff={max_diff:e}",
                        label(format),
                        u8::from(cli.graph),
                    );
                }
                reports.push(report);
            }
            Err(error) => {
                failed = Some(format!("format {}: {error}", label(format)));
                break;
            }
        }
    }
    gpu.qwen4_scope = false;
    if let Some(error) = failed {
        return Err(error);
    }
    if let Some(limit) = capture_limit {
        println!("few_row_limit_in_capture={limit}");
    }
    let mismatches: usize = reports.iter().map(FormatReport::mismatches).sum();
    if let Some(path) = &cli.json {
        let formats: Vec<serde_json::Value> = reports
            .iter()
            .map(|r| {
                let dtype = format!("{:?}", r.format.dtype()).to_ascii_lowercase();
                let kernels: Vec<&String> = r.kernels.iter().collect();
                let results: Vec<serde_json::Value> = r
                    .cases
                    .iter()
                    .map(|c| {
                        let failures: Vec<serde_json::Value> =
                            c.failures.iter().map(Failure::to_json).collect();
                        let mut entry = json!({
                            "rows": c.rows,
                            "keep": c.keep,
                            "window": c.window,
                            "ok": c.failures.is_empty(),
                            "failures": failures,
                        });
                        if !c.diagnostics.is_empty() {
                            entry["diagnostics"] = serde_json::Value::Array(
                                c.diagnostics.iter().map(Diag::to_json).collect(),
                            );
                        }
                        entry
                    })
                    .collect();
                let state_bytes = r.format.state_bytes(VH, HD, HD);
                let mut block = json!({
                    "format": label(r.format),
                    "dtype": dtype,
                    "state_units": r.format.state_units(VH, HD, HD),
                    "state_bytes": state_bytes,
                    "ring_bytes": 2 * ring_rows * state_bytes,
                    "cases": r.cases.len(),
                    "mismatches": r.mismatches(),
                    "max_abs_state": r.max_abs_state,
                    "max_abs_state_source": "decoded AR reference snapshots",
                    "kernels": kernels,
                    "results": results,
                });
                if r.format.has_residual() {
                    let by_kind: serde_json::Map<String, serde_json::Value> = r
                        .diag_summary()
                        .into_iter()
                        .map(|(kind, (total, differing, max_diff))| {
                            (
                                kind,
                                json!({
                                    "total": total,
                                    "differing": differing,
                                    "max_abs_diff": max_diff,
                                }),
                            )
                        })
                        .collect();
                    let (seed_max, seed_min) = r.seed_residual.unwrap_or((0.0, 0.0));
                    block["storage_format"] = json!(label(r.format.storage()));
                    block["residual_bytes"] = json!(r.format.residual_bytes(VH, HD, HD));
                    block["max_abs_residual"] = json!(r.max_abs_residual);
                    block["seed_residual"] = json!({
                        "steps": SEED_STEPS,
                        "max_abs_max_over_rows": seed_max,
                        "max_abs_min_over_rows": seed_min,
                    });
                    block["hard"] = json!({
                        "cases": r.cases.len(),
                        "failing_cases": r.mismatches(),
                        "failures": r.hard_failures(),
                    });
                    block["diagnostics"] = json!({
                        "policy": "cross-route q8_ef comparisons are reported, never failing",
                        "differing": r.diagnostic_diffs(),
                        "by_kind": by_kind,
                    });
                }
                block
            })
            .collect();
        let doc = json!({
            "tool": "gdn_state_precision_oracle",
            "gpu_arch": arch,
            "geometry": {
                "key_heads": KH, "value_heads": VH, "key_dim": HD, "value_dim": HD,
                "qkv_width": QKV, "start_position": START_POSITION,
            },
            "seed": cli.seed,
            "windows": cli.windows,
            "graph": cli.graph,
            "max_rows": cli.max_rows,
            "ring_rows": ring_rows,
            "all_keeps": cli.all_keeps,
            "route_flags": {
                "qwen4_scope": true,
                "prefill_invariant_scope": false,
                "wide_verify_rows_batched": ring_rows,
                "wide_verify_rows_ar": 0,
                "wide_verify_max_rows": WIDE_VERIFY_MAX_ROWS,
                "few_row_limit_eager": eager_limit,
                "few_row_limit_in_capture": capture_limit,
            },
            "total_cases": reports.iter().map(|r| r.cases.len()).sum::<usize>(),
            "total_mismatches": mismatches,
            "total_diagnostic_diffs": reports.iter().map(FormatReport::diagnostic_diffs).sum::<usize>(),
            "formats": formats,
        });
        let text = ctx(serde_json::to_string_pretty(&doc), "serialize json")?;
        ctx(
            std::fs::write(path, text),
            &format!("write {}", path.display()),
        )?;
    }
    Ok(i32::from(mismatches > 0))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_args(&args) {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("gdn_state_precision_oracle: {message}\n{USAGE}");
            std::process::exit(2);
        }
    };
    match run(&cli) {
        Ok(code) => std::process::exit(code),
        Err(message) => {
            eprintln!("gdn_state_precision_oracle: unsupported or failed setup: {message}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn cli_defaults() {
        let cli = parse_args(&[]).unwrap();
        assert_eq!(cli.formats, vec![GdnStateFormat::Q8, GdnStateFormat::F32]);
        assert!(!cli.graph && !cli.all_keeps);
        assert_eq!(cli.max_rows, 32);
        assert_eq!(cli.windows, 3);
        assert_eq!(cli.seed, DEFAULT_SEED);
        assert_eq!(cli.json, None);
    }

    #[test]
    fn cli_parses_full_command_line() {
        let cli = parse_args(&args(
            "--formats fp32,q8 --graph 1 --max-rows=12 --all-keeps --seed 0x10 --windows 4 --json /tmp/o.json",
        ))
        .unwrap();
        assert_eq!(cli.formats, vec![GdnStateFormat::F32, GdnStateFormat::Q8]);
        assert!(cli.graph && cli.all_keeps);
        assert_eq!((cli.max_rows, cli.seed, cli.windows), (12, 16, 4));
        assert_eq!(cli.json, Some(PathBuf::from("/tmp/o.json")));
    }

    #[test]
    fn cli_parses_q8_ef() {
        let cli = parse_args(&args("--formats q8,q8_ef,fp32 --graph 1")).unwrap();
        assert_eq!(
            cli.formats,
            vec![
                GdnStateFormat::Q8,
                GdnStateFormat::Q8Ef,
                GdnStateFormat::F32
            ]
        );
        assert_eq!(parse_format("q8_ef").unwrap(), GdnStateFormat::Q8Ef);
        assert_eq!(label(GdnStateFormat::Q8Ef), "q8_ef");
        assert_eq!(label(GdnStateFormat::Q8), "q8");
        assert!(parse_args(&args("--formats q8_ef,q8_ef")).is_err());
        let error = parse_args(&args("--formats q8ef")).unwrap_err();
        assert!(error.contains("q8_ef"), "{error}");
    }

    #[test]
    fn cli_rejects_invalid_values() {
        for bad in [
            "--windows 1",
            "--windows 0",
            "--max-rows 0",
            "--max-rows 33",
            "--graph 2",
            "--formats q8,q8",
            "--formats bf16",
            "--seed nope",
            "--max-rows",
            "--bogus",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn capture_base_alternates_halves() {
        let r = 8;
        let mut source = 0;
        let mut bases = Vec::new();
        for rows in [8, 3, 5, 1, 8] {
            let base = capture_base(source, r);
            bases.push(base);
            source = base + rows - 1;
        }
        assert_eq!(bases, vec![8, 0, 8, 0, 8]);
    }

    #[test]
    fn keeps_selection() {
        assert_eq!(keeps_for(3, true), vec![0, 1, 2, 3]);
        assert_eq!(keeps_for(3, false), vec![0, 3]);
        assert_eq!(keeps_for(1, false), vec![0, 1]);
    }

    #[test]
    fn compare_reports_first_differing_byte_and_row() {
        let a = vec![0u8; 16];
        let mut b = a.clone();
        assert!(compare("k", &a, &b, Some(8)).is_none());
        b[11] = 1;
        b[14] = 2;
        let f = compare("k", &a, &b, Some(8)).unwrap();
        assert_eq!((f.byte_offset, f.row), (11, Some(1)));
        assert_eq!(first_diff(&a, &a[..12]), Some(12));
        assert!(compare("k", &a, &a[..12], None).is_some());
    }

    #[test]
    fn nonfinite_and_state_stats() {
        let bytes = f32_bytes(&[1.0, -3.5, 2.0]);
        assert_eq!(state_stats(&bytes), (3.5, None));
        let bad = f32_bytes(&[1.0, f32::NAN, f32::INFINITY]);
        assert_eq!(first_nonfinite(&bad), Some(4));
        assert_eq!(state_stats(&bad).0, 1.0);
    }

    #[test]
    fn bf16_rounding_is_rne_and_inputs_stay_in_range() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3F80_8000)), 1.0);
        assert_eq!(
            bf16_round(f32::from_bits(0x3F81_8000)),
            f32::from_bits(0x3F82_0000)
        );
        let steps = make_steps(2, window_seed(1, 2, 0));
        assert_eq!(
            (steps.proj.len(), steps.gate.len(), steps.beta.len()),
            (2 * QKV, 2 * VH, 2 * VH)
        );
        assert!(steps
            .proj
            .iter()
            .all(|v| bf16_round(*v) == *v && v.abs() <= 2.0));
        assert!(steps.gate.iter().all(|g| (-0.3..-0.02).contains(g)));
        assert!(steps.beta.iter().all(|b| (0.1..0.9).contains(b)));
    }

    #[test]
    fn q8_ef_keeps_select_the_reset_or_commit_sequence() {
        let modes = |rows: usize, all: bool| -> Vec<EfMode> {
            keeps_for(rows, all)
                .into_iter()
                .map(|keep| ef_mode(keep, rows))
                .collect()
        };
        assert_eq!(
            modes(4, true),
            vec![
                EfMode::Restore,
                EfMode::Rollback(1),
                EfMode::Rollback(2),
                EfMode::Rollback(3),
                EfMode::Capture
            ]
        );
        assert_eq!(modes(4, false), vec![EfMode::Restore, EfMode::Capture]);
        assert_eq!(modes(1, true), vec![EfMode::Restore, EfMode::Capture]);
        assert_eq!(
            [
                EfMode::Capture.launches(),
                EfMode::Restore.launches(),
                EfMode::Rollback(2).launches()
            ],
            [1, 2, 3]
        );
    }

    #[test]
    fn f16_conversion_and_stats() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        let bytes = [0x00, 0x3c, 0x00, 0x7c, 0x00, 0xc0];
        assert_eq!(first_nonfinite_f16(&bytes), Some(2));
        assert_eq!(f16_stats(&bytes), (2.0, Some(2)));
        assert_eq!(f16_stats(&bytes[..2]), (1.0, None));
        assert_eq!(first_nonfinite_f16(&[0x00, 0x7e]), Some(0));
    }

    fn q8_slot(code: i8, scale: f32) -> Vec<u8> {
        let mut slot = vec![code as u8; Q8_CODE_BYTES];
        for _ in 0..VH * HD {
            slot.extend(scale.to_le_bytes());
        }
        assert_eq!(slot.len(), SLOT_BYTES);
        slot
    }

    fn f16_plane(bits: u16) -> Vec<u8> {
        bits.to_le_bytes()
            .iter()
            .copied()
            .cycle()
            .take(RESIDUAL_BYTES)
            .collect()
    }

    #[test]
    fn q8_slot_decode_and_stats() {
        let slot = q8_slot(-2, 0.5);
        let decoded = decode_q8_slot(&slot);
        assert_eq!(decoded.len(), VH * HD * HD);
        assert!(decoded.iter().all(|v| *v == -1.0));
        assert_eq!(q8_slot_stats(&slot), (1.0, None));
        let mut bad = q8_slot(1, 0.5);
        let at = Q8_CODE_BYTES + 4 * 3;
        bad[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        // Row 3's 128 codes decode to NaN; the first is code index 3 * 128.
        assert_eq!(q8_slot_stats(&bad), (0.5, Some(3 * HD)));
        assert!(decode_q8_slot(&slot[..SLOT_BYTES - 1]).is_empty());
        let effective = effective_state(&q8_slot(2, 0.5), Some(&f16_plane(0x3800)));
        assert!(effective.iter().all(|v| *v == 1.5));
        assert_eq!(effective_state(&q8_slot(2, 0.5), None)[0], 1.0);
    }

    #[test]
    fn diagnostics_report_first_difference_and_decoded_distance() {
        let a = q8_slot(1, 0.5);
        let same = diag_q8("k", &a, &a);
        assert!(same.equal && same.byte_offset.is_none());
        assert_eq!(same.max_abs_diff, Some(0.0));
        let mut b = q8_slot(1, 0.5);
        b[130] = 3;
        let d = diag_q8("k", &b, &a);
        assert!(!d.equal);
        assert_eq!(d.byte_offset, Some(130));
        assert_eq!(d.max_abs_diff, Some(1.0));
        // 2.5 = 0x40200000 vs 2.0 = 0x40000000: byte 2 of element 1.
        let eff = diag_effective("e", &[1.0, 2.5, 3.0], &[1.0, 2.0, 3.0]);
        assert!(!eff.equal);
        assert_eq!((eff.byte_offset, eff.max_abs_diff), (Some(6), Some(0.5)));
        let f = diag_f32(
            "o",
            &f32_bytes(&[1.0, 2.0]),
            &f32_bytes(&[1.0, 2.0]),
            Some(4),
        );
        assert!(f.equal);
        let r = diag_f16("r", &f16_plane(0x3c00), &f16_plane(0x3800));
        assert!(!r.equal);
        assert_eq!(r.row, Some(0));
        assert_eq!(r.max_abs_diff, Some(0.5));
    }

    fn kinds(failures: &[Failure]) -> Vec<&str> {
        failures.iter().map(|f| f.kind.as_str()).collect()
    }

    const RESIDUAL_IN: u16 = 0x3c00;

    fn clean_snap(mode: EfMode) -> EfSnap {
        let rollback = matches!(mode, EfMode::Rollback(_));
        EfSnap {
            out: f32_bytes(&vec![0.5; VW]),
            discard: if rollback {
                f32_bytes(&vec![0.25; VW])
            } else {
                Vec::new()
            },
            final_slot: q8_slot(1, 0.5),
            kept: if rollback {
                q8_slot(2, 0.5)
            } else {
                Vec::new()
            },
            source: q8_slot(3, 0.25),
            residual: if mode == EfMode::Capture {
                f16_plane(0x3800)
            } else {
                vec![0; RESIDUAL_BYTES]
            },
            stray: None,
        }
    }

    fn check(snap: &EfSnap, mode: EfMode, cap: Option<&EfSnap>) -> Vec<Failure> {
        check_ef_snap(snap, &q8_slot(3, 0.25), &f16_plane(RESIDUAL_IN), mode, cap)
    }

    #[test]
    fn q8_ef_hard_checks_pass_clean_snapshots() {
        let cap = clean_snap(EfMode::Capture);
        assert!(check(&cap, EfMode::Capture, None).is_empty());
        assert!(check(&clean_snap(EfMode::Capture), EfMode::Capture, Some(&cap)).is_empty());
        let restore = clean_snap(EfMode::Restore);
        assert!(check(&restore, EfMode::Restore, Some(&cap)).is_empty());
        let rollback = clean_snap(EfMode::Rollback(2));
        assert!(check(&rollback, EfMode::Rollback(2), Some(&cap)).is_empty());
    }

    #[test]
    fn q8_ef_hard_checks_flag_each_violation() {
        let cap = clean_snap(EfMode::Capture);
        let mut snap = clean_snap(EfMode::Capture);
        snap.source = q8_slot(4, 0.25);
        assert_eq!(
            kinds(&check(&snap, EfMode::Capture, None)),
            ["source_modified"]
        );

        let mut snap = clean_snap(EfMode::Capture);
        snap.stray = Some(SLOT_BYTES + 7);
        let failures = check(&snap, EfMode::Capture, None);
        assert_eq!(kinds(&failures), ["ring_stray_write"]);
        assert_eq!(failures[0].row, Some(1));

        let mut snap = clean_snap(EfMode::Capture);
        snap.final_slot = vec![POISON_STATE; SLOT_BYTES];
        assert!(kinds(&check(&snap, EfMode::Capture, None)).contains(&"final_slot_unwritten"));

        let mut snap = clean_snap(EfMode::Capture);
        snap.residual = f16_plane(0x7e00);
        assert!(kinds(&check(&snap, EfMode::Capture, None)).contains(&"nonfinite_residual"));

        let mut snap = clean_snap(EfMode::Capture);
        snap.out = f32_bytes(&vec![f32::NAN; VW]);
        assert_eq!(
            kinds(&check(&snap, EfMode::Capture, None)),
            ["nonfinite_output"]
        );

        // The capture must write the residual.
        let mut snap = clean_snap(EfMode::Capture);
        snap.residual = f16_plane(RESIDUAL_IN);
        assert_eq!(
            kinds(&check(&snap, EfMode::Capture, None)),
            ["capture_residual_unwritten"]
        );

        // All-accept: the residual must be the capture's, not zeroed.
        let mut snap = clean_snap(EfMode::Capture);
        snap.residual = vec![0; RESIDUAL_BYTES];
        assert_eq!(
            kinds(&check(&snap, EfMode::Capture, Some(&cap))),
            ["residual_not_kept"]
        );

        // Reset events leave exactly zero.
        let mut restore = clean_snap(EfMode::Restore);
        restore.residual[RESIDUAL_BYTES - 1] = 1;
        let failures = check(&restore, EfMode::Restore, Some(&cap));
        assert_eq!(kinds(&failures), ["residual_not_zero"]);
        assert_eq!(failures[0].byte_offset, RESIDUAL_BYTES - 1);
        let mut rollback = clean_snap(EfMode::Rollback(2));
        rollback.residual = f16_plane(0x3800);
        assert_eq!(
            kinds(&check(&rollback, EfMode::Rollback(2), Some(&cap))),
            ["residual_not_zero"]
        );
        let mut rollback = clean_snap(EfMode::Rollback(2));
        rollback.kept = vec![POISON_STATE; SLOT_BYTES];
        assert_eq!(
            kinds(&check(&rollback, EfMode::Rollback(2), Some(&cap))),
            ["rollback_target_unwritten"]
        );

        // Same route, same history: the case equals the window capture.
        let mut snap = clean_snap(EfMode::Capture);
        snap.final_slot = q8_slot(5, 0.5);
        assert_eq!(
            kinds(&check(&snap, EfMode::Capture, Some(&cap))),
            ["final_state_vs_capture"]
        );
    }

    #[test]
    fn q8_ef_same_route_comparison_names_each_region() {
        let a = clean_snap(EfMode::Rollback(2));
        assert!(check_ef_same("nondeterministic", &a, &clean_snap(EfMode::Rollback(2))).is_empty());
        let mut b = clean_snap(EfMode::Rollback(2));
        b.residual[5] = 1;
        b.kept[0] ^= 1;
        b.stray = Some(3);
        assert_eq!(
            kinds(&check_ef_same("graph_vs_eager", &a, &b)),
            [
                "graph_vs_eager_kept_state",
                "graph_vs_eager_residual",
                "graph_vs_eager_ring_stray_write"
            ]
        );
    }

    #[test]
    fn format_report_separates_hard_failures_from_diagnostics() {
        let diag = |kind: &str, equal: bool, max: f64| Diag {
            kind: kind.to_string(),
            equal,
            row: None,
            byte_offset: None,
            max_abs_diff: Some(max),
            detail: String::new(),
        };
        let mut report = FormatReport::new(GdnStateFormat::Q8Ef);
        report.cases.push(CaseResult {
            rows: 2,
            keep: 1,
            window: 0,
            failures: vec![failure("a", None, 0, ""), failure("b", None, 0, "")],
            diagnostics: vec![diag("x", false, 0.5), diag("y", true, 0.0)],
        });
        report.cases.push(CaseResult {
            rows: 2,
            keep: 2,
            window: 0,
            failures: Vec::new(),
            diagnostics: vec![diag("x", false, 0.75)],
        });
        assert_eq!(report.mismatches(), 1);
        assert_eq!(report.hard_failures(), 2);
        assert_eq!(report.diagnostic_diffs(), 2);
        let summary: BTreeMap<String, (usize, usize, f64)> = report.diag_summary();
        assert_eq!(summary["x"], (2, 2, 0.75));
        assert_eq!(summary["y"], (1, 0, 0.0));
        report.note_seed_residual(0.5);
        report.note_seed_residual(2.0);
        assert_eq!(report.seed_residual, Some((2.0, 0.5)));
    }
}
