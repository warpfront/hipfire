// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Trunk-shard layer oracle: the engine behind `examples/trunk_shard_layer_oracle.rs`
//! (lab only; never part of a serving build).
//!
//! The example documents WHAT the oracle proves and does not prove; this module holds the
//! parts that need crate-internal access plus the whole CPU-side analysis, so the analysis
//! is unit-tested on any box:
//!
//! * **core** (no GPU): the deterministic HC-stream generator, the per-element contract
//!   checks, the difference statistics, the case/dump file format and the reference-vs-TP2
//!   comparison ([`evaluate_block`]);
//! * **plan** (no GPU): which step-graph ops of the real forward program a layer's mixer /
//!   FFN block is ([`layer_plan`]);
//! * **engine** (GPU): runs ONE block of the REAL forward program on injected HC streams by
//!   executing a segment of the program through `forward_mesh_chunk` (the same entry the
//!   TP mesh uses), after localising a freshly assembled bundle to a trunk-shard rank by
//!   hand ([`localize_bundle`]).

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config};
use hipfire_runtime::ar_hc_exchange::ArHcExchange;
use rdna_compute::ar_hc_write::{
    bf16_bits, bf16_round, hc_gate, ArHcPartial, StreamKind, AR_HC_HIDDEN,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// `2^-8`: the bf16 unit roundoff (round to nearest even at 8 significant bits).
pub const EPS8: f64 = 1.0 / 256.0;
/// `2^-7`: one bf16 ulp relative to the bottom of its binade, the largest change a bf16
/// rounding can make to a result when its input moved by less than the rounding step.
pub const ROUND_FLIP: f64 = 1.0 / 128.0;
/// HC branches of the Flash-Next trunk.
pub const HC_BRANCHES: usize = 4;

// ============================================================================================
// Blocks, cases, tolerances
// ============================================================================================

/// One half of a trunk layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Block {
    /// HC read, GDN/QSA mixer (+ out/o projection), attention HC write.
    Mixer,
    /// MLP HC read, MoE FFN, MLP HC write.
    Ffn,
}

impl Block {
    pub fn name(self) -> &'static str {
        match self {
            Block::Mixer => "mixer",
            Block::Ffn => "ffn",
        }
    }
}

/// The routing class of a case (the mesh route the program runs under).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RouteSel {
    /// One row, the AR decode shape.
    Decode,
    /// 2..=few rows outside the prefill scope (speculative verify, lane rows).
    Verify,
    /// A prefill chunk (the prefill scope; BF16 streams from `QWEN4_F16_WMMA_MIN_TOKENS`).
    Prefill,
}

impl RouteSel {
    pub fn name(self) -> &'static str {
        match self {
            RouteSel::Decode => "decode",
            RouteSel::Verify => "verify",
            RouteSel::Prefill => "prefill",
        }
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        match text {
            "decode" => Ok(Self::Decode),
            "verify" => Ok(Self::Verify),
            "prefill" => Ok(Self::Prefill),
            other => Err(format!("route `{other}` is not decode|verify|prefill")),
        }
    }

    /// The route `auto` picks for `rows` rows: 1 decode, 2..=16 verify, more prefill.
    pub fn auto(rows: usize) -> Self {
        match rows {
            0 | 1 => Self::Decode,
            2..=16 => Self::Verify,
            _ => Self::Prefill,
        }
    }
}

/// One (layer, rows, route) point; the blocks to run are chosen separately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseSpec {
    pub layer: usize,
    pub rows: usize,
    pub route: RouteSel,
}

impl CaseSpec {
    /// Directory name of the case inside an arm dump.
    pub fn dir_name(&self) -> String {
        format!("L{:02}-r{}-{}", self.layer, self.rows, self.route.name())
    }
}

/// Tolerances of the reference-vs-TP2 comparison (everything else is exact or contractual).
///
/// `acc` is the fp32 accumulation-order allowance of the K-split GEMV relative to
/// `|p0| + |p1|`; `row_rms` is an absolute allowance relative to the RMS of the reference
/// block output row (cancellation floor). Both are far below the bf16 rounding terms the
/// contract already allows, so only a real cut/kernel error breaks them. `fold` is the wire
/// format of the mixer partials under test (`topology.ep_ar_wire`; default bf16, the product
/// default): it sets the fold model and its bound.
#[derive(Clone, Copy, Debug)]
pub struct Tolerances {
    pub fold: ArHcPartial,
    pub acc: f64,
    pub row_rms: f64,
    /// Multiplies every bound (1.0 = the documented contract).
    pub scale: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            fold: ArHcPartial::Bf16,
            acc: 1.0 / 16_384.0,
            row_rms: 1.0 / 4_096.0,
            scale: 1.0,
        }
    }
}

// ============================================================================================
// Stream bytes <-> values
// ============================================================================================

/// The `rows * 4 * 2560` element values of HC streams stored as `kind`.
pub fn stream_values(kind: StreamKind, bytes: &[u8]) -> Result<Vec<f32>, String> {
    let width = kind.element_bytes();
    if bytes.len() % width != 0 {
        return Err(format!(
            "{} stream bytes {} not a multiple of {width}",
            kind.name(),
            bytes.len()
        ));
    }
    Ok(match kind {
        StreamKind::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        StreamKind::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    })
}

/// Inverse of [`stream_values`] for BF16-exact values.
pub fn stream_bytes(kind: StreamKind, values: &[f32]) -> Vec<u8> {
    match kind {
        StreamKind::Bf16 => values
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect(),
        StreamKind::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ============================================================================================
// Deterministic HC stream input
// ============================================================================================

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // splitmix64 seeding so nearby seeds do not correlate.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Self((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `(0, 1)`.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Per-branch scales of the synthetic streams: the four HC branches of a real trunk carry
/// residual-stream magnitudes of different order (branch 0 dominant), plus rare large
/// outliers (hidden-state outlier channels), so the HC write, the gates and the projection
/// inputs are exercised away from the unit-normal comfort zone.
const BRANCH_SCALE: [f64; HC_BRANCHES] = [3.0, 1.5, 0.75, 0.3];

/// `rows * 4 * 2560` deterministic HC-stream values (bf16-exact), a pure function of
/// `(seed, rows)` (row `r` of a longer run equals row `r` of a shorter one).
pub fn synthetic_stream_values(seed: u64, rows: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * HC_BRANCHES * AR_HC_HIDDEN);
    for row in 0..rows {
        let mut rng = Rng::new(seed ^ (row as u64).wrapping_mul(0x2545_F491_4F6C_DD1D));
        for scale in BRANCH_SCALE {
            for _ in 0..AR_HC_HIDDEN {
                let mut v = rng.normal() * scale;
                if rng.next() % 1024 == 0 {
                    v *= 16.0;
                }
                out.push(bf16_round(v as f32));
            }
        }
    }
    out
}

/// [`synthetic_stream_values`] stored as `kind`.
pub fn synthetic_stream(seed: u64, rows: usize, kind: StreamKind) -> Vec<u8> {
    stream_bytes(kind, &synthetic_stream_values(seed, rows))
}

// ============================================================================================
// Captures
// ============================================================================================

/// What one rank (or the single reference rank) recorded of one block.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockCapture {
    pub rows: usize,
    pub kind: StreamKind,
    /// HC streams entering the block (injected; identical on every arm and rank).
    pub input: Vec<u8>,
    /// HC streams after the block's HC write.
    pub output: Vec<u8>,
    /// The `[rows * 4]` gate logits the block's HC write used.
    pub gates: Vec<f32>,
    /// `[rows * 2560]` f32: on a TP rank its block-output PARTIAL (the fused AR's input), on
    /// the reference the full block output `m_ref` when the production route materialised
    /// it (`None` when the HC write was fused into the projection epilogue).
    pub partial: Option<Vec<f32>>,
}

/// The pass/fail verdict of one named check with its worst-case evidence.
#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub pass: bool,
    /// Elements (or rows/bytes, see the detail) beyond the bound.
    pub violations: usize,
    /// Elements examined.
    pub n: usize,
    /// Worst `error / bound` over the examined elements (<= 1 passes; 0 when exact).
    pub worst_ratio: f64,
    pub detail: String,
}

impl Check {
    fn skipped(name: &'static str, why: impl Into<String>) -> Self {
        Self {
            name,
            pass: true,
            violations: 0,
            n: 0,
            worst_ratio: 0.0,
            detail: format!("SKIPPED: {}", why.into()),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "pass": self.pass,
            "violations": self.violations,
            "n": self.n,
            "worst_ratio": self.worst_ratio,
            "detail": self.detail,
        })
    }
}

/// Difference statistics of `a` (candidate) against `b` (reference), values as f32.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DiffStats {
    pub n: usize,
    /// Elements that differ in value.
    pub ne: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    /// RMS of the reference values.
    pub rms_ref: f64,
    /// `max |a - b| / rms_ref` (0 when the reference is all zero and equal).
    pub max_rel_rms: f64,
    /// `mean |a - b| / rms_ref`.
    pub mean_rel_rms: f64,
    /// Largest bf16 ulp distance (only meaningful for bf16-exact values).
    pub max_ulp: u32,
}

impl DiffStats {
    pub fn to_json(&self) -> Value {
        json!({
            "n": self.n,
            "ne": self.ne,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "rms_ref": self.rms_ref,
            "max_rel_rms": self.max_rel_rms,
            "mean_rel_rms": self.mean_rel_rms,
            "max_ulp": self.max_ulp,
        })
    }
}

/// Distance in bf16 ulps between two bf16-rounded values (sign-magnitude ordered).
pub fn bf16_ulp_distance(a: f32, b: f32) -> u32 {
    fn ordered(x: f32) -> i32 {
        let bits = i32::from(bf16_bits(x) as i16);
        if bits < 0 {
            i32::from(i16::MIN) - bits
        } else {
            bits
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

pub fn diff_stats(candidate: &[f32], reference: &[f32]) -> DiffStats {
    assert_eq!(candidate.len(), reference.len());
    let n = candidate.len();
    if n == 0 {
        return DiffStats::default();
    }
    let (mut ne, mut max_abs, mut sum_abs, mut sum_sq, mut max_ulp) =
        (0usize, 0f64, 0f64, 0f64, 0u32);
    for (a, b) in candidate.iter().zip(reference) {
        let d = (f64::from(*a) - f64::from(*b)).abs();
        if a.to_bits() != b.to_bits() && d != 0.0 {
            ne += 1;
        }
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += f64::from(*b) * f64::from(*b);
        max_ulp = max_ulp.max(bf16_ulp_distance(*a, *b));
    }
    let rms_ref = (sum_sq / n as f64).sqrt();
    let rel = |x: f64| if rms_ref > 0.0 { x / rms_ref } else { x };
    DiffStats {
        n,
        ne,
        max_abs,
        mean_abs: sum_abs / n as f64,
        rms_ref,
        max_rel_rms: rel(max_abs),
        mean_rel_rms: rel(sum_abs / n as f64),
        max_ulp,
    }
}

/// Per-branch statistics of two stream sets of `rows * 4 * 2560` values.
pub fn branch_stats(candidate: &[f32], reference: &[f32], rows: usize) -> Vec<DiffStats> {
    (0..HC_BRANCHES)
        .map(|br| {
            let pick = |v: &[f32]| -> Vec<f32> {
                (0..rows)
                    .flat_map(|row| {
                        let at = (row * HC_BRANCHES + br) * AR_HC_HIDDEN;
                        v[at..at + AR_HC_HIDDEN].iter().copied()
                    })
                    .collect()
            };
            diff_stats(&pick(candidate), &pick(reference))
        })
        .collect()
}

// ============================================================================================
// Contract checks
// ============================================================================================

fn finish(name: &'static str, n: usize, violations: usize, worst: f64, detail: String) -> Check {
    Check {
        name,
        pass: violations == 0,
        violations,
        n,
        worst_ratio: worst,
        detail,
    }
}

/// C1: the two byte sets are identical (`what` names them in the detail).
pub fn check_bytes_equal(name: &'static str, a: &[u8], b: &[u8], what: &str) -> Check {
    if a.len() != b.len() {
        return finish(
            name,
            a.len().max(b.len()),
            1,
            f64::INFINITY,
            format!("{what}: lengths {} vs {}", a.len(), b.len()),
        );
    }
    let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
    let first = a.iter().zip(b).position(|(x, y)| x != y);
    finish(
        name,
        a.len(),
        bad,
        if bad == 0 { 0.0 } else { f64::INFINITY },
        match first {
            None => format!("{what}: byte-identical ({} bytes)", a.len()),
            Some(at) => format!("{what}: {bad} bytes differ, first at byte {at}"),
        },
    )
}

/// C2: the fold contract of the fused all-reduce on the REAL partials of the layer:
/// `|m - (p0 + p1)| <= tol.fold.fold_bound(p0, p1)` (f32 partials `2^-24 (|p0| + |p1|)`, bf16
/// `(2^-8 + 2^-23)(|p0| + |p1|)`) for `m = tol.fold.fold(p0, p1)` (the host model of the device fold; the kernel's own bytes are
/// checked by C3).
pub fn check_fold_contract(p0: &[f32], p1: &[f32], tol: &Tolerances) -> Check {
    assert_eq!(p0.len(), p1.len());
    let mut worst = 0f64;
    let mut bad = 0usize;
    for (a, b) in p0.iter().zip(p1) {
        let m = f64::from(tol.fold.fold(*a, *b));
        let exact = f64::from(*a) + f64::from(*b);
        let bound = tol.fold.fold_bound(*a, *b) * tol.scale;
        let err = (m - exact).abs();
        let ratio = if bound > 0.0 {
            err / bound
        } else if err == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if ratio > 1.0 {
            bad += 1;
        }
        worst = worst.max(ratio);
    }
    finish(
        "C2 fold contract |m-(p0+p1)|<=2^-24(|p0|+|p1|)",
        p0.len(),
        bad,
        worst,
        format!("worst error/bound {worst:.4}"),
    )
}

/// Host model of the fused write for ONE (row, branch) word: the stream update the kernel
/// applies with gate value `g`.
fn model_word(kind: StreamKind, g: f32, m: f32, s: f32) -> f32 {
    match kind {
        StreamKind::Bf16 => {
            let inj = bf16_round(g * bf16_round(m));
            bf16_round(inj + s)
        }
        StreamKind::F32 => {
            let inj = bf16_round(g * bf16_round(m));
            bf16_round(bf16_round(s) + inj)
        }
    }
}

/// Neighbouring bf16 values of `g` (one ulp each way): the device `expf` and the host libm
/// may round the sigmoid differently by one bf16 step.
fn gate_neighbours(g: f32) -> [f32; 2] {
    let bits = bf16_bits(g);
    let up = f32::from_bits(u32::from(bits.wrapping_add(1)) << 16);
    let down = f32::from_bits(u32::from(bits.wrapping_sub(1)) << 16);
    [down, up]
}

/// C3: the device output of the fused write equals the HOST MODEL of the write applied to
/// the captured input, partials and gates (`ArHcPartial::fold` + `hc_inject[_f32]`). A (row,
/// branch) whose every word is reproduced by a one-bf16-ulp neighbour of the host gate is
/// counted as a benign gate-rounding difference (host libm vs device `expf`), not a
/// violation; anything else is.
#[allow(clippy::too_many_arguments)]
pub fn check_kernel_vs_model(
    kind: StreamKind,
    fold: ArHcPartial,
    input: &[u8],
    output: &[u8],
    gates: &[f32],
    p0: &[f32],
    p1: &[f32],
    rows: usize,
) -> Result<Check, String> {
    let s_in = stream_values(kind, input)?;
    let s_out = stream_values(kind, output)?;
    let want_len = rows * HC_BRANCHES * AR_HC_HIDDEN;
    if s_in.len() != want_len || s_out.len() != want_len || gates.len() != rows * HC_BRANCHES {
        return Err("check_kernel_vs_model: geometry mismatch".into());
    }
    let (mut hard, mut benign, mut compared) = (0usize, 0usize, 0usize);
    for row in 0..rows {
        for br in 0..HC_BRANCHES {
            let g = hc_gate(gates[row * HC_BRANCHES + br]);
            let at = (row * HC_BRANCHES + br) * AR_HC_HIDDEN;
            let m_at = row * AR_HC_HIDDEN;
            let apply = |gate: f32| -> Vec<f32> {
                (0..AR_HC_HIDDEN)
                    .map(|c| {
                        model_word(
                            kind,
                            gate,
                            fold.fold(p0[m_at + c], p1[m_at + c]),
                            s_in[at + c],
                        )
                    })
                    .collect()
            };
            let want = apply(g);
            let same = |w: &[f32]| {
                w.iter()
                    .zip(&s_out[at..at + AR_HC_HIDDEN])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            };
            compared += AR_HC_HIDDEN;
            if same(&want) {
                continue;
            }
            if gate_neighbours(g).iter().any(|n| same(&apply(*n))) {
                benign += AR_HC_HIDDEN;
                continue;
            }
            hard += want
                .iter()
                .zip(&s_out[at..at + AR_HC_HIDDEN])
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
    }
    Ok(finish(
        "C3 device fused write == host model (fold + HC write)",
        compared,
        hard,
        if hard == 0 { 0.0 } else { f64::INFINITY },
        format!("{hard} words differ, {benign} words in gate-ulp-benign (row,branch) groups"),
    ))
}

/// The per-element allowance on `|p0 + p1 - m_ref|`: the fold's own fp32 add, the fp32
/// accumulation-order noise and an RMS floor for cancellation.
fn cut_tolerance(p0: f32, p1: f32, row_rms: f64, tol: &Tolerances) -> f64 {
    (tol.fold.fold_bound(p0, p1)
        + tol.acc * (f64::from(p0).abs() + f64::from(p1).abs())
        + tol.row_rms * row_rms)
        * tol.scale
}

fn row_rms(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values
        .iter()
        .map(|v| f64::from(*v) * f64::from(*v))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

/// C4: the K-split partials sum to the reference block output within the contract:
/// `|(p0 + p1) - m_ref| <= (2^-24 + acc)(|p0| + |p1|) + row_rms * rms(m_ref row)`.
/// This is the check that catches a wrong weight cut / head order / kernel shape: an error
/// of one head's contribution is O(|m|), orders of magnitude above this allowance.
pub fn check_cut_vs_reference(
    p0: &[f32],
    p1: &[f32],
    m_ref: &[f32],
    rows: usize,
    tol: &Tolerances,
) -> Check {
    assert_eq!(p0.len(), rows * AR_HC_HIDDEN);
    let (mut bad, mut worst) = (0usize, 0f64);
    for row in 0..rows {
        let range = row * AR_HC_HIDDEN..(row + 1) * AR_HC_HIDDEN;
        let rms = row_rms(&m_ref[range.clone()]);
        for i in range {
            let err = (f64::from(p0[i]) + f64::from(p1[i]) - f64::from(m_ref[i])).abs();
            let bound = cut_tolerance(p0[i], p1[i], rms, tol);
            let ratio = if bound > 0.0 {
                err / bound
            } else if err == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            if ratio > 1.0 {
                bad += 1;
            }
            worst = worst.max(ratio);
        }
    }
    finish(
        "C4 K-split partials sum to the reference block output",
        p0.len(),
        bad,
        worst,
        format!("worst error/bound {worst:.4}"),
    )
}

/// C5: every HC stream element of the TP2 result is within the documented envelope of the
/// reference result (same injected input, both arms):
///
/// `|s_tp - s_ref| <= (1 + 2^-8) |g| (dm + 2^-7 |m|) + 2^-7 |g| |m| + 2^-7 max(|s_tp|, |s_ref|)`
///
/// where `dm = 2^-24 (|p0| + |p1|) + cutTol` bounds `|m_fold - m_ref|` (the fold's own
/// fp32 add plus the C4 allowance), `m` is the reference block output (the fold when the
/// reference did not materialise it) and `g` the row gate; the `2^-7` terms are one bf16 ulp of
/// slack for each of the three roundings of the HC write (`bf16(m)`, `bf16(g*bf16(m))` and the
/// final sum), which can flip when their input moves.
#[allow(clippy::too_many_arguments)]
pub fn check_stream_bound(
    kind: StreamKind,
    s_tp: &[u8],
    s_ref: &[u8],
    gates: &[f32],
    p0: &[f32],
    p1: &[f32],
    m_ref: Option<&[f32]>,
    rows: usize,
    tol: &Tolerances,
) -> Result<Check, String> {
    let (a, b) = (stream_values(kind, s_tp)?, stream_values(kind, s_ref)?);
    if a.len() != b.len() || a.len() != rows * HC_BRANCHES * AR_HC_HIDDEN {
        return Err("check_stream_bound: geometry mismatch".into());
    }
    let (mut bad, mut worst) = (0usize, 0f64);
    for row in 0..rows {
        let m_at = row * AR_HC_HIDDEN;
        let m_row: Vec<f32> = match m_ref {
            Some(m) => m[m_at..m_at + AR_HC_HIDDEN].to_vec(),
            None => (0..AR_HC_HIDDEN)
                .map(|c| tol.fold.fold(p0[m_at + c], p1[m_at + c]))
                .collect(),
        };
        let rms = row_rms(&m_row);
        for br in 0..HC_BRANCHES {
            let g = f64::from(hc_gate(gates[row * HC_BRANCHES + br])).abs();
            let at = (row * HC_BRANCHES + br) * AR_HC_HIDDEN;
            for c in 0..AR_HC_HIDDEN {
                let m = f64::from(m_row[c]).abs();
                let cut = cut_tolerance(p0[m_at + c], p1[m_at + c], rms, tol);
                // `scale` is already inside `fold` and `cut`; the rounding-flip terms are the
                // pure envelope of three bf16 roundings and are not scaled.
                let fold = tol.fold.fold_bound(p0[m_at + c], p1[m_at + c]) * tol.scale;
                let dm = fold + cut;
                let bound = (1.0 + EPS8) * g * (dm + ROUND_FLIP * m)
                    + ROUND_FLIP * g * m
                    + ROUND_FLIP * f64::from(a[at + c]).abs().max(f64::from(b[at + c]).abs());
                let err = (f64::from(a[at + c]) - f64::from(b[at + c])).abs();
                let ratio = if bound > 0.0 {
                    err / bound
                } else if err == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                };
                if ratio > 1.0 {
                    bad += 1;
                }
                worst = worst.max(ratio);
            }
        }
    }
    Ok(finish(
        "C5 every HC stream element within the contract envelope of the reference",
        a.len(),
        bad,
        worst,
        format!("worst error/bound {worst:.4}"),
    ))
}

// ============================================================================================
// Evaluating one block
// ============================================================================================

/// The verdict of one block of one case.
#[derive(Clone, Debug)]
pub struct BlockReport {
    pub case: String,
    pub block: Block,
    pub layer_type: &'static str,
    pub kind: StreamKind,
    pub rows: usize,
    pub checks: Vec<Check>,
    /// Per-branch stream statistics of the TP2 result against the reference (empty without a
    /// reference).
    pub stream_vs_ref: Vec<DiffStats>,
    /// Statistics of `p0 + p1` against `m_ref` (None without a materialised `m_ref`).
    pub block_output_vs_ref: Option<DiffStats>,
    /// Statistics of the fold against the exact sum `p0 + p1` (the contract's own term).
    pub fold_vs_exact: Option<DiffStats>,
}

impl BlockReport {
    pub fn pass(&self) -> bool {
        self.checks.iter().all(|c| c.pass)
    }

    pub fn to_json(&self) -> Value {
        json!({
            "case": self.case,
            "block": self.block.name(),
            "layer_type": self.layer_type,
            "stream_kind": self.kind.name(),
            "rows": self.rows,
            "pass": self.pass(),
            "checks": self.checks.iter().map(Check::to_json).collect::<Vec<_>>(),
            "stream_vs_ref_per_branch": self.stream_vs_ref.iter().map(DiffStats::to_json).collect::<Vec<_>>(),
            "block_output_vs_ref": self.block_output_vs_ref.as_ref().map(DiffStats::to_json),
            "fold_vs_exact_sum": self.fold_vs_exact.as_ref().map(DiffStats::to_json),
        })
    }

    /// Human-readable lines (one block).
    pub fn render(&self) -> String {
        let mut out = format!(
            "[{}] {} {} {} rows={} stream={}  {}\n",
            if self.pass() { "PASS" } else { "FAIL" },
            self.case,
            self.block.name(),
            self.layer_type,
            self.rows,
            self.kind.name(),
            if self.pass() { "" } else { "<-- violations" }
        );
        for check in &self.checks {
            out.push_str(&format!(
                "    {} {:<66} viol={} worst={:.4}  {}\n",
                if check.pass { "ok  " } else { "FAIL" },
                check.name,
                check.violations,
                check.worst_ratio,
                check.detail
            ));
        }
        if let Some(stats) = &self.block_output_vs_ref {
            out.push_str(&format!(
                "    block output (p0+p1) vs reference m: max|d|={:.3e} max/rms={:.3e} mean/rms={:.3e}\n",
                stats.max_abs, stats.max_rel_rms, stats.mean_rel_rms
            ));
        }
        if let Some(stats) = &self.fold_vs_exact {
            out.push_str(&format!(
                "    fold vs exact sum: max|d|={:.3e} max/rms={:.3e}\n",
                stats.max_abs, stats.max_rel_rms
            ));
        }
        for (br, stats) in self.stream_vs_ref.iter().enumerate() {
            out.push_str(&format!(
                "    stream {br} vs reference: ne={}/{} max|d|={:.3e} max/rms={:.3e} mean/rms={:.3e} max_ulp={}\n",
                stats.ne, stats.n, stats.max_abs, stats.max_rel_rms, stats.mean_rel_rms, stats.max_ulp
            ));
        }
        out
    }
}

/// Evaluate one block of one case.
///
/// `tp` holds the TP2 captures of rank 0 and rank 1 (`partial` = each rank's partial);
/// `reference` is the unsharded capture of the same input, if available. The checks:
///
/// * **C0** the two ranks saw the same input bytes (and the reference the same too);
/// * **C1** rank 0 == rank 1: output streams and gate logits byte-equal;
/// * **C2** fold contract on the real partials;
/// * **C3** device output == host model of the fused write (the kernel on real data);
/// * **C4** `p0 + p1` vs the reference block output (when materialised);
/// * **C5** every stream element vs the reference within the contract envelope;
/// * **C6** gate logits equal to the reference's (replicated HC weights, same input).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_block(
    case: &str,
    block: Block,
    layer_type: &'static str,
    tp: [&BlockCapture; 2],
    reference: Option<&BlockCapture>,
    tol: &Tolerances,
) -> Result<BlockReport, String> {
    let [r0, r1] = tp;
    if r0.rows != r1.rows || r0.kind != r1.kind {
        return Err(format!("{case}: ranks disagree on rows/kind"));
    }
    let (rows, kind) = (r0.rows, r0.kind);
    let (p0, p1) = match (&r0.partial, &r1.partial) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(format!("{case}: TP captures carry no partials")),
    };
    let mut checks = Vec::new();
    let mut input_check = check_bytes_equal(
        "C0 identical block input",
        &r0.input,
        &r1.input,
        "rank0 vs rank1 input",
    );
    if let Some(reference) = reference {
        if reference.rows != rows || reference.kind != kind {
            return Err(format!(
                "{case}: reference rows/kind ({}, {}) differ from TP ({rows}, {})",
                reference.rows,
                reference.kind.name(),
                kind.name()
            ));
        }
        let vs_ref = check_bytes_equal(
            "C0 identical block input",
            &reference.input,
            &r0.input,
            "reference vs TP input",
        );
        if !vs_ref.pass {
            input_check = vs_ref;
        }
    }
    checks.push(input_check);
    checks.push(check_bytes_equal(
        "C1 rank0 == rank1 output streams",
        &r0.output,
        &r1.output,
        "output streams",
    ));
    checks.push(check_bytes_equal(
        "C1 rank0 == rank1 gate logits",
        &f32_bytes(&r0.gates),
        &f32_bytes(&r1.gates),
        "gate logits",
    ));
    checks.push(check_fold_contract(p0, p1, tol));
    checks.push(check_kernel_vs_model(
        kind, tol.fold, &r0.input, &r0.output, &r0.gates, p0, p1, rows,
    )?);

    let folded: Vec<f32> = p0
        .iter()
        .zip(p1)
        .map(|(a, b)| tol.fold.fold(*a, *b))
        .collect();
    let exact: Vec<f32> = p0
        .iter()
        .zip(p1)
        .map(|(a, b)| (f64::from(*a) + f64::from(*b)) as f32)
        .collect();
    let fold_vs_exact = Some(diff_stats(&folded, &exact));

    let mut stream_vs_ref = Vec::new();
    let mut block_output_vs_ref = None;
    match reference {
        None => {
            checks.push(Check::skipped(
                "C4 K-split partials sum to the reference block output",
                "no reference capture",
            ));
            checks.push(Check::skipped(
                "C5 every HC stream element within the contract envelope of the reference",
                "no reference capture",
            ));
            checks.push(Check::skipped(
                "C6 gate logits equal the reference's",
                "no reference capture",
            ));
        }
        Some(reference) => {
            match &reference.partial {
                Some(m_ref) => {
                    checks.push(check_cut_vs_reference(p0, p1, m_ref, rows, tol));
                    block_output_vs_ref = Some(diff_stats(&exact, m_ref));
                }
                None => checks.push(Check::skipped(
                    "C4 K-split partials sum to the reference block output",
                    "the reference fused its HC write into the projection (run it with HIPFIRE_QWEN4_HC_FUSE=1)",
                )),
            }
            checks.push(check_stream_bound(
                kind,
                &r0.output,
                &reference.output,
                &r0.gates,
                p0,
                p1,
                reference.partial.as_deref(),
                rows,
                tol,
            )?);
            checks.push(check_bytes_equal(
                "C6 gate logits equal the reference's",
                &f32_bytes(&r0.gates),
                &f32_bytes(&reference.gates),
                "gate logits",
            ));
            stream_vs_ref = branch_stats(
                &stream_values(kind, &r0.output)?,
                &stream_values(kind, &reference.output)?,
                rows,
            );
        }
    }
    Ok(BlockReport {
        case: case.to_string(),
        block,
        layer_type,
        kind,
        rows,
        checks,
        stream_vs_ref,
        block_output_vs_ref,
        fold_vs_exact,
    })
}

// ============================================================================================
// Dump format
// ============================================================================================

/// `OUT/<arm>/<case>/<block>/`: `meta.json`, `in.bin`, and per rank `rank<r>-out.bin`,
/// `rank<r>-gates.f32`, `rank<r>-partial.f32` (absent when the capture has none).
pub fn block_dir(arm_dir: &Path, case: &CaseSpec, block: Block) -> PathBuf {
    arm_dir.join(case.dir_name()).join(block.name())
}

pub fn write_capture(dir: &Path, rank: usize, capture: &BlockCapture) -> Result<(), String> {
    let io = |what: &str, e: std::io::Error| format!("{what}: {e}");
    std::fs::create_dir_all(dir).map_err(|e| io(&dir.display().to_string(), e))?;
    let put = |name: String, bytes: &[u8]| {
        std::fs::write(dir.join(&name), bytes).map_err(|e| io(&name, e))
    };
    put("in.bin".into(), &capture.input)?;
    put(format!("rank{rank}-out.bin"), &capture.output)?;
    put(format!("rank{rank}-gates.f32"), &f32_bytes(&capture.gates))?;
    if let Some(partial) = &capture.partial {
        put(format!("rank{rank}-partial.f32"), &f32_bytes(partial))?;
    }
    let meta = json!({
        "rows": capture.rows,
        "stream_kind": capture.kind.name(),
        "partial": capture.partial.is_some(),
    });
    std::fs::write(
        dir.join(format!("rank{rank}-meta.json")),
        serde_json::to_vec_pretty(&meta).map_err(|e| e.to_string())?,
    )
    .map_err(|e| io("meta", e))
}

pub fn read_capture(dir: &Path, rank: usize) -> Result<Option<BlockCapture>, String> {
    let meta_path = dir.join(format!("rank{rank}-meta.json"));
    if !meta_path.exists() {
        return Ok(None);
    }
    let meta: Value = serde_json::from_slice(
        &std::fs::read(&meta_path).map_err(|e| format!("{}: {e}", meta_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", meta_path.display()))?;
    let rows = meta["rows"].as_u64().ok_or("meta: rows")? as usize;
    let kind = match meta["stream_kind"].as_str() {
        Some("bf16") => StreamKind::Bf16,
        Some("f32") => StreamKind::F32,
        other => return Err(format!("meta: stream_kind {other:?}")),
    };
    let read = |name: String| std::fs::read(dir.join(&name)).map_err(|e| format!("{name}: {e}"));
    let partial_path = dir.join(format!("rank{rank}-partial.f32"));
    Ok(Some(BlockCapture {
        rows,
        kind,
        input: read("in.bin".into())?,
        output: read(format!("rank{rank}-out.bin"))?,
        gates: f32_values(&read(format!("rank{rank}-gates.f32"))?),
        partial: if partial_path.exists() {
            Some(f32_values(&read(format!("rank{rank}-partial.f32"))?))
        } else {
            None
        },
    }))
}

// ============================================================================================
// Plan: which ops of the real program are a layer's blocks
// ============================================================================================

/// Op-id spans of the real forward program (trunk step graph, `mesh_trunk_op_range`
/// numbering) of one layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    pub layer: usize,
    pub layer_type: LayerType,
    /// Ops before the layer's first HC read (`Embed`, every earlier layer, the PLE depthwise
    /// op when the layer is the PLE layer): run as the PREFIX segment.
    pub prefix: std::ops::Range<u32>,
    /// HC read, mixer, attention HC write.
    pub mixer: std::ops::Range<u32>,
    /// MLP HC read, [clear, MoE | dense MLP], MLP HC write.
    pub ffn: std::ops::Range<u32>,
}

/// The op spans of `layer` from the step graph the forward builds for `config`.
pub fn layer_plan(config: &Qwen4Config, layer: usize) -> Result<LayerPlan, String> {
    use hipfire_dispatch::pipeline::OpKind;
    let graph = crate::program::qwen4_step_graph_with_output(
        config,
        hipfire_runtime::multi_gpu::executor::MeshRoute::Decode,
        false,
        false,
    )?;
    let layer_type = *config.layer_types.get(layer).ok_or_else(|| {
        format!(
            "layer {layer} is outside the {}-layer trunk",
            config.layer_types.len()
        )
    })?;
    let ops = graph.ops();
    let mine: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| op.layer == Some(layer))
        .map(|(index, _)| index)
        .collect();
    let pick = |kind: OpKind, nth: usize| -> Result<usize, String> {
        mine.iter()
            .copied()
            .filter(|&i| ops[i].kind == kind)
            .nth(nth)
            .ok_or_else(|| {
                format!("layer {layer}: no op #{nth} of the expected kind in the step graph")
            })
    };
    let read_a = pick(OpKind::HyperRead, 0)?;
    let write_a = pick(OpKind::HyperWrite, 0)?;
    let read_m = pick(OpKind::HyperRead, 1)?;
    let write_m = pick(OpKind::HyperWrite, 1)?;
    if !(read_a < write_a && write_a + 1 == read_m && read_m < write_m) {
        return Err(format!(
            "layer {layer}: unexpected step-graph shape (read {read_a}, write {write_a}, read {read_m}, write {write_m})"
        ));
    }
    Ok(LayerPlan {
        layer,
        layer_type,
        prefix: 0..read_a as u32,
        mixer: read_a as u32..write_a as u32 + 1,
        ffn: write_a as u32 + 1..write_m as u32 + 1,
    })
}

pub fn layer_type_name(kind: LayerType) -> &'static str {
    match kind {
        LayerType::LinearAttention => "gdn",
        LayerType::FullAttention => "qsa",
    }
}

/// A CPU-constructible Flash-Next config (the compact test geometry) for `plan` runs
/// without an artifact.
pub fn synthetic_flash_next_config() -> Qwen4Config {
    crate::config::compact_test_config()
}

// ============================================================================================
// Synthetic TP2 / reference captures (self-test of the analysis, CPU only)
// ============================================================================================

/// A defect the self-test injects into otherwise exact synthetic captures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Defect {
    None,
    /// Rank 0's partial is wrong over a head-sized column slice (a bad cut / head order).
    CutSlice,
    /// Rank 1's output stream differs from rank 0's in one word.
    RankSkew,
    /// The device output differs from the host model in one word (a kernel error).
    KernelWord,
    /// One (row, branch) was written with a one-bf16-ulp different gate (benign libm skew).
    GateUlp,
}

/// Synthetic (reference, rank 0, rank 1) captures of one block with the exact arithmetic
/// of the contract: K-split partials of a random block output, the reference writing the
/// full `m` through the production-style HC write and the TP2 pair through the fused model.
pub fn synthetic_tp_case(
    seed: u64,
    rows: usize,
    kind: StreamKind,
    fold: ArHcPartial,
    defect: Defect,
) -> (BlockCapture, BlockCapture, BlockCapture) {
    let mut rng = Rng::new(seed ^ 0xA5A5);
    let n = rows * AR_HC_HIDDEN;
    let input = synthetic_stream(seed, rows, kind);
    let s_in = stream_values(kind, &input).expect("synthetic stream");
    let gates: Vec<f32> = (0..rows * HC_BRANCHES)
        .map(|_| (rng.unit() * 6.0 - 3.0) as f32)
        .collect();
    let m_true: Vec<f32> = (0..n).map(|_| (rng.normal() * 2.0) as f32).collect();
    let split: Vec<f32> = (0..n).map(|_| (rng.unit() * 2.0 - 0.5) as f32).collect();
    let p0: Vec<f32> = m_true.iter().zip(&split).map(|(m, s)| m * s).collect();
    let mut p1: Vec<f32> = m_true.iter().zip(&p0).map(|(m, a)| m - a).collect();
    // The reference accumulates in another order: perturb by an fp32 ulp now and then.
    let m_ref: Vec<f32> = m_true
        .iter()
        .map(|m| {
            if rng.next() % 7 == 0 {
                f32::from_bits(m.to_bits() ^ 1)
            } else {
                *m
            }
        })
        .collect();
    let mut p0 = p0;
    if defect == Defect::CutSlice {
        // One head's worth of columns replaced: the classic wrong-cut signature.
        for row in 0..rows {
            for c in 128..384 {
                p0[row * AR_HC_HIDDEN + c] = (rng.normal() * 2.0) as f32;
            }
        }
    }
    let apply = |m_of: &dyn Fn(usize) -> f32, gate_ulp: Option<usize>| -> Vec<f32> {
        let mut out = s_in.clone();
        for row in 0..rows {
            for br in 0..HC_BRANCHES {
                let mut g = hc_gate(gates[row * HC_BRANCHES + br]);
                if gate_ulp == Some(row * HC_BRANCHES + br) {
                    g = gate_neighbours(g)[1];
                }
                let at = (row * HC_BRANCHES + br) * AR_HC_HIDDEN;
                for c in 0..AR_HC_HIDDEN {
                    out[at + c] = model_word(kind, g, m_of(row * AR_HC_HIDDEN + c), s_in[at + c]);
                }
            }
        }
        out
    };
    let ref_out = apply(&|i| m_ref[i], None);
    let gate_ulp = (defect == Defect::GateUlp).then_some(1usize);
    let mut tp_out = apply(&|i| fold.fold(p0[i], p1[i]), gate_ulp);
    if defect == Defect::KernelWord {
        tp_out[5] = bf16_round(tp_out[5] * 1.5 + 1.0);
    }
    let out_bytes = stream_bytes(kind, &tp_out);
    let mut out1 = out_bytes.clone();
    if defect == Defect::RankSkew {
        out1[10] ^= 1;
    }
    p1.shrink_to_fit();
    let reference = BlockCapture {
        rows,
        kind,
        input: input.clone(),
        output: stream_bytes(kind, &ref_out),
        gates: gates.clone(),
        partial: Some(m_ref),
    };
    let rank0 = BlockCapture {
        rows,
        kind,
        input: input.clone(),
        output: out_bytes,
        gates: gates.clone(),
        partial: Some(p0),
    };
    let rank1 = BlockCapture {
        rows,
        kind,
        input,
        output: out1,
        gates,
        partial: Some(p1),
    };
    (reference, rank0, rank1)
}

/// Run the whole analysis on synthetic data: exact arithmetic must pass every check and each
/// injected defect must be caught by the check that owns it. Returns the lines of the
/// receipt, `Err` naming the first expectation that failed.
pub fn selftest(rows_list: &[usize]) -> Result<Vec<String>, String> {
    let mut lines = Vec::new();
    for fold in ArHcPartial::ALL {
        let tol = Tolerances {
            fold,
            ..Tolerances::default()
        };
        for &rows in rows_list {
            for kind in StreamKind::ALL {
                for defect in [
                    Defect::None,
                    Defect::GateUlp,
                    Defect::CutSlice,
                    Defect::RankSkew,
                    Defect::KernelWord,
                ] {
                    let (reference, r0, r1) =
                        synthetic_tp_case(0x5EED + rows as u64, rows, kind, fold, defect);
                    let report = evaluate_block(
                        "selftest",
                        Block::Mixer,
                        "gdn",
                        [&r0, &r1],
                        Some(&reference),
                        &tol,
                    )?;
                    let failed: Vec<&str> = report
                        .checks
                        .iter()
                        .filter(|c| !c.pass)
                        .map(|c| c.name.split(' ').next().unwrap_or(""))
                        .collect();
                    let expect: &[&str] = match defect {
                        Defect::None | Defect::GateUlp => &[],
                        Defect::CutSlice => &["C4", "C5"],
                        Defect::RankSkew => &["C1"],
                        Defect::KernelWord => &["C3", "C5"],
                    };
                    // A defect must be caught by (at least) the checks that own it; a clean
                    // case by none.
                    let ok = if expect.is_empty() {
                        failed.is_empty()
                    } else {
                        expect.iter().all(|e| failed.contains(e))
                    };
                    lines.push(format!(
                    "selftest fold={} rows={rows} stream={} defect={defect:?}: failed checks {failed:?} (expected {expect:?}) -> {}",
                    fold.name(),
                    kind.name(),
                    if ok { "ok" } else { "WRONG" }
                ));
                    if !ok {
                        return Err(format!("{}\n{}", lines.last().unwrap(), report.render()));
                    }
                }
            }
        }
    }
    Ok(lines)
}

// ============================================================================================
// Engine (GPU)
// ============================================================================================

mod engine;
pub use engine::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_stream_is_deterministic_bf16_exact_and_row_stable() {
        let a = synthetic_stream_values(7, 3);
        let b = synthetic_stream_values(7, 5);
        assert_eq!(a.len(), 3 * 4 * AR_HC_HIDDEN);
        assert_eq!(&a[..], &b[..a.len()]);
        assert!(a.iter().all(|v| bf16_round(*v) == *v));
        assert_ne!(synthetic_stream_values(8, 1), synthetic_stream_values(7, 1));
        let rms = row_rms(&a);
        assert!(rms > 0.5 && rms < 20.0, "rms {rms}");
    }

    #[test]
    fn stream_bytes_round_trip() {
        let v = synthetic_stream_values(3, 2);
        for kind in StreamKind::ALL {
            assert_eq!(stream_values(kind, &stream_bytes(kind, &v)).unwrap(), v);
        }
    }

    #[test]
    fn bf16_ulp_distance_is_symmetric_and_crosses_zero() {
        assert_eq!(bf16_ulp_distance(1.0, 1.0), 0);
        let up = f32::from_bits(0x3F81_0000);
        assert_eq!(bf16_ulp_distance(1.0, up), 1);
        assert_eq!(bf16_ulp_distance(up, 1.0), 1);
        assert_eq!(bf16_ulp_distance(-0.0, 0.0), 0);
        let tiny = f32::from_bits(0x0001_0000);
        assert_eq!(bf16_ulp_distance(tiny, -tiny), 2);
    }

    #[test]
    fn analysis_passes_exact_arithmetic_and_catches_every_defect() {
        let lines = selftest(&[1, 3]).unwrap_or_else(|e| panic!("{e}"));
        assert!(lines.iter().all(|l| l.ends_with("ok")), "{lines:#?}");
    }

    #[test]
    fn dump_round_trips() {
        let dir = std::env::temp_dir().join(format!("tslo-{}", std::process::id()));
        let case = CaseSpec {
            layer: 3,
            rows: 2,
            route: RouteSel::Verify,
        };
        let (reference, r0, r1) =
            synthetic_tp_case(1, 2, StreamKind::F32, ArHcPartial::Bf16, Defect::None);
        let block = block_dir(&dir, &case, Block::Mixer);
        write_capture(&block, 0, &r0).unwrap();
        write_capture(&block, 1, &r1).unwrap();
        write_capture(
            &block_dir(&dir.join("ref"), &case, Block::Mixer),
            0,
            &reference,
        )
        .unwrap();
        assert_eq!(read_capture(&block, 0).unwrap().unwrap(), r0);
        assert_eq!(read_capture(&block, 1).unwrap().unwrap(), r1);
        assert!(read_capture(&block, 2).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn layer_plan_segments_tile_the_layer_ops() {
        let config = synthetic_flash_next_config();
        let layers = config.layer_types.len();
        for layer in [0, layers - 1] {
            let plan = layer_plan(&config, layer).unwrap();
            assert_eq!(plan.mixer.end, plan.ffn.start);
            assert!(plan.prefix.end == plan.mixer.start);
            assert!(plan.mixer.len() == 3, "{plan:?}");
            // clear + moe + read + write
            assert!(plan.ffn.len() == 4, "{plan:?}");
        }
        assert!(layer_plan(&config, layers).is_err());
    }

    #[test]
    fn route_auto_maps_rows() {
        assert_eq!(RouteSel::auto(1), RouteSel::Decode);
        assert_eq!(RouteSel::auto(8), RouteSel::Verify);
        assert_eq!(RouteSel::auto(32), RouteSel::Prefill);
        assert!(RouteSel::parse("nope").is_err());
    }
}

/// Keep the engine's exchange type nameable from the example without a second import.
pub type Exchange = Arc<ArHcExchange>;
/// The bundle type the engine drives.
pub type Bundle = Qwen4Bundle;
