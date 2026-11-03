// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! fn_wide_kl_oracle: numerical closeness of the EP2 sampled WIDE verify pass
//! (lab). S2 "wide pass" kill gate: per-row KL(singleton || wide) <= 1e-4.
//!
//! What it measures, per verify row, on a real EP mesh with the native-MTP
//! lanes (`--lanes` concurrent prompts, window depth `--depths`):
//!
//! 1. **Reference**: for every prompt the serialized singleton greedy run
//!    (`reset`, `prefill_final`, then `tokens + 1` x
//!    `forward_token_or_argmax(None, logits)`, the `fn_mesh_lane_oracle`
//!    reference loop). Its token trajectory `c[0..=tokens]` and the logits row
//!    left by every forward (`rows[pos]` = logits after processing `c[pos]`,
//!    i.e. the distribution that predicts `c[pos + 1]`) are kept on the host as
//!    f32 (`tokens x vocab x 4` bytes per prompt).
//! 2. **Teacher-forced replay** through the mesh lane protocol
//!    ([`EpMeshLanes::step`]) with an `AcceptRule` that forces the trajectory:
//!    every draft the head proposes is replaced by the singleton token at that
//!    position (`AcceptRule::draft_token`) and every window commits
//!    `accepted = drafts` plus the singleton's next token (`AcceptRule::accept`),
//!    so every verify row has exactly the token context of its reference
//!    position, whatever the head or the verify pass produced. The rule reads
//!    every resident verify logits row through the sampler owner's `RowHandle`
//!    (`SamplingMode::NaiveSampled` lanes: ranks replay the root's decision, the
//!    handle is exposed; no sampler is drawn from) before the step's prepare,
//!    and scores it against the reference row in full vocabulary.
//!    Two arms on freshly prefilled lanes:
//!    * `exact`: `LaneRequest::wide_verify = false`, the row-exact groups of at
//!      most 8 rows (the greedy/seeded path) = the control, expected
//!      bit-identical to the singleton;
//!    * `wide`: `wide_verify = true` with `HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY=1`
//!      (set in-process when absent): consecutive lanes pack into one
//!      prefill-class group of up to 32 rows, verify scope closed.
//!      `HIPFIRE_QWEN4_EP_WIDE_VERIFY_ALL_LANES` must NOT be set (it would make
//!      the control arm wide too; refused). The wide class actually running is
//!      checked from the merged rank evidence (`counts.wide_groups`).
//! 3. Per row: KL(ref || arm) in f64 over the whole vocabulary (softmax from
//!    the f32 rows, `sum p_ref (log p_ref - log p_arm)`), max |delta logit|,
//!    top-1 agreement (host argmax and the device pick), the reference top-1
//!    margin of every disagreement. Reported as mean / p50 / p90 / p99 / max,
//!    per row index within the window, per lane, per trajectory third, and the
//!    fraction of rows above `--kl-threshold`.
//!
//! Because the replay commits the verify pass's own KV/GDN state, the wide arm's
//! error accumulates along the trajectory exactly as in serving (the control arm
//! shows the singleton-vs-lane baseline). Windows can be shorter than `depth + 1`
//! rows: the head's early stop (`prefix * ROW_WORTH`) is real and cannot be
//! overridden through the rule; the per-row-index tables and the pass-row
//! histogram say what the wide pass actually carried.
//!
//! Gate (exit 1 when any (set, lanes, depth) config fails):
//! mean per-row KL(ref || wide) <= `--gate-mean` (default 1e-4); with
//! `--gate-p99/--gate-max/--gate-frac/--gate-top1` the other statistics gate too
//! (reported only otherwise); the control arm's mean KL must also be <=
//! `--gate-mean`; every position of every lane must be covered once; no
//! non-finite logits. Exit 2: usage / runtime error.
//!
//! ```text
//! HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY=1 fn_wide_kl_oracle --model M.mq4 --max-seq 4096 \
//!     --tokens 96 --lanes 4,8 --depths 3,5 --prompt-sets mixed,story,code \
//!     --reserve-lanes 8 --out receipt.json
//! ```
//! Writes `receipt.json`, `receipt.json.rows.jsonl` (one line per scored row) and
//! `receipt.json.summary.txt`.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::ep_mesh_lanes::{EpMeshLanes, MeshLaneAdmission, MeshStepError};
use hipfire_arch_qwen4::lane::{LabRoutes, LaneSpecConfig};
use hipfire_arch_qwen4::mesh_lane_protocol::{
    pack_classed_row_groups, AcceptRule, LaneKind, LanePlan, LaneRequest, SamplerOutput,
    SamplingMode, WindowBudgets,
};
use hipfire_arch_qwen4::sampled_lane::{LaneRng, RowHandle};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use hipfire_runtime::spec::GreedyAccept;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

const TOOL: &str = "fn_wide_kl_oracle";
const SAMPLED_WIDE_ENV: &str = "HIPFIRE_QWEN4_EP_SAMPLED_WIDE_VERIFY";
const ALL_LANES_ENV: &str = "HIPFIRE_QWEN4_EP_WIDE_VERIFY_ALL_LANES";
const DEFAULT_PROMPT_ROOT: &str = "/home/kaden/qcal/release-0.4.3/mgmtp/ms5/prompts";
/// Deepest window the lane store stages (`LANE_MAX_ROWS - 1`).
const MAX_DEPTH: usize = 7;

fn e<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}

// ------------------------------------------------------------------ math

/// `log(sum(exp(row)))` in f64 (max-shifted). `-inf` for an empty row, `NaN`
/// when the row holds a `NaN`, `+inf` when it holds `+inf`.
fn log_sum_exp(row: &[f32]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &x in row {
        let x = f64::from(x);
        if x.is_nan() {
            return f64::NAN;
        }
        if x > max {
            max = x;
        }
    }
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = row.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    max + sum.ln()
}

/// Index of the first maximum (`0` for an empty row).
fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best
}

/// `top1 - top2` logit margin (`+inf` for fewer than two entries).
fn top1_margin(row: &[f32]) -> f32 {
    let top = argmax(row);
    let second = row
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != top)
        .map(|(_, &x)| x)
        .fold(f32::NEG_INFINITY, f32::max);
    if row.len() < 2 {
        f32::INFINITY
    } else {
        row[top] - second
    }
}

/// `KL(ref || cand)` over the full vocabulary in nats, from f32 logit rows
/// (`ref_lse` = `log_sum_exp(ref_row)`, passed because it is shared by every
/// candidate of a reference row). A non-finite candidate row gives `+inf`.
/// Tiny negative sums (rounding) are clamped to 0.
fn kl_full(ref_row: &[f32], ref_lse: f64, cand: &[f32]) -> f64 {
    debug_assert_eq!(ref_row.len(), cand.len());
    let cand_lse = log_sum_exp(cand);
    if !cand_lse.is_finite() || !ref_lse.is_finite() {
        return f64::INFINITY;
    }
    let mut kl = 0f64;
    for (&a, &b) in ref_row.iter().zip(cand) {
        let log_p = f64::from(a) - ref_lse;
        let p = log_p.exp();
        if p == 0.0 {
            continue;
        }
        let log_q = f64::from(b) - cand_lse;
        kl += p * (log_p - log_q);
    }
    if kl < 0.0 {
        0.0
    } else {
        kl
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(
            0.0f32,
            |m, d| if d.is_nan() { f32::INFINITY } else { m.max(d) },
        )
}

/// Nearest-rank percentile of an ascending slice (`NaN` when empty).
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Distribution of a per-row statistic.
#[derive(Clone, Debug, PartialEq)]
struct Dist {
    n: usize,
    mean: f64,
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
    /// Fraction of values strictly above the threshold given to [`Dist::of`].
    frac_gt: f64,
}

impl Dist {
    fn of(values: &[f64], threshold: f64) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let mean = if n == 0 {
            f64::NAN
        } else {
            sorted.iter().sum::<f64>() / n as f64
        };
        let above = sorted.iter().filter(|&&v| v > threshold).count();
        Self {
            n,
            mean,
            p50: percentile_sorted(&sorted, 0.50),
            p90: percentile_sorted(&sorted, 0.90),
            p99: percentile_sorted(&sorted, 0.99),
            max: sorted.last().copied().unwrap_or(f64::NAN),
            frac_gt: if n == 0 {
                f64::NAN
            } else {
                above as f64 / n as f64
            },
        }
    }

    fn json(&self, threshold: f64) -> Value {
        json!({
            "n": self.n, "mean": self.mean, "p50": self.p50, "p90": self.p90,
            "p99": self.p99, "max": self.max,
            "frac_gt": self.frac_gt, "threshold": threshold,
        })
    }
}

// ------------------------------------------------------ teacher forcing

/// Singleton logits row index (`rows[pos]`) of the verify row that processes
/// trajectory token `pos`: forward `pos + 1` consumed `c[pos]`.
const fn ref_row_of(pos: usize) -> usize {
    pos
}

/// The draft the replay forces at draft `index` of a window whose seed is
/// trajectory token `t`: `c[t + 1 + index]`.
fn forced_draft(ids: &[u32], t: usize, index: usize) -> R<u32> {
    ids.get(t + 1 + index)
        .copied()
        .ok_or_else(|| format!("forced draft {index} at seed {t} is past the trajectory"))
}

/// What a window with seed `t` and `d` drafts commits when the replay accepts
/// every draft: the drafts, then the singleton's next token (`d + 1` ids).
fn forced_commit(ids: &[u32], t: usize, d: usize) -> R<Vec<u32>> {
    ids.get(t + 1..=t + d + 1)
        .map(<[u32]>::to_vec)
        .ok_or_else(|| {
            format!(
                "forced window t={t} d={d} is past the trajectory ({} ids)",
                ids.len()
            )
        })
}

/// `LaneRequest::remaining` of a window at seed `t`: the trajectory tokens still to emit.
/// The depth is the lane's admitted `spec_k`; `remaining` caps the drafts at
/// `tokens - t - 1` so the bonus never passes the last trajectory id
/// (`ids.len() - 1 == tokens`), and makes the window that reaches it, and only that one,
/// terminal (a window emitting `remaining` tokens ends the request).
fn window_remaining(tokens: usize, t: usize) -> usize {
    tokens - t
}

/// One lane's singleton reference.
struct RefLane {
    prompt_len: usize,
    /// `c[0..=tokens]`: `c[0]` is the pick of the prefill logits.
    ids: Vec<u32>,
    /// `rows[pos]`: f32 logits left by the forward that consumed `c[pos]`.
    rows: Vec<Vec<f32>>,
    lse: Vec<f64>,
    top1: Vec<u32>,
    margin: Vec<f32>,
}

impl RefLane {
    fn new(prompt_len: usize, ids: Vec<u32>, rows: Vec<Vec<f32>>) -> Self {
        let lse = rows.iter().map(|r| log_sum_exp(r)).collect();
        let top1 = rows.iter().map(|r| argmax(r) as u32).collect();
        let margin = rows.iter().map(|r| top1_margin(r)).collect();
        Self {
            prompt_len,
            ids,
            rows,
            lse,
            top1,
            margin,
        }
    }
}

/// One scored verify row.
#[derive(Clone, Debug)]
struct RowObs {
    lane: usize,
    /// Trajectory index of the token this row processed.
    pos: usize,
    /// Row index within its window (0 = the seed row).
    row: usize,
    win_rows: usize,
    /// Rows of the once-over-all-rows group the row's lane ran in.
    group_rows: usize,
    kl: f64,
    max_abs: f32,
    top1_agree: bool,
    pick_agree: bool,
    cand_top1: u32,
    ref_top1: u32,
    ref_margin: f32,
    finite: bool,
}

struct RowMetrics {
    kl: f64,
    max_abs: f32,
    top1_agree: bool,
    cand_top1: u32,
    finite: bool,
}

fn measure(reference: &RefLane, pos: usize, cand: &[f32]) -> RowMetrics {
    let r = ref_row_of(pos);
    let ref_row = &reference.rows[r];
    let cand_top1 = argmax(cand) as u32;
    RowMetrics {
        kl: kl_full(ref_row, reference.lse[r], cand),
        max_abs: max_abs_diff(ref_row, cand),
        top1_agree: cand_top1 == reference.top1[r],
        cand_top1,
        finite: cand.iter().all(|x| x.is_finite()),
    }
}

/// Shape of one verify pass.
#[derive(Clone, Debug)]
struct StepShape {
    pass_rows: usize,
    /// `(rows, lanes)` of each once-over-all-rows group.
    groups: Vec<(usize, usize)>,
}

/// The acceptance rule of the replay: forces the singleton trajectory and
/// scores every resident verify row.
struct TeacherRule<'a> {
    refs: &'a [RefLane],
    epochs: &'a [RequestEpoch],
    wide: bool,
    rows: Option<RowHandle>,
    group_rows: Vec<(hipfire_arch_qwen4::mesh_lane_protocol::LaneWindowKey, usize)>,
    obs: Vec<RowObs>,
    steps: Vec<StepShape>,
}

impl TeacherRule<'_> {
    fn lane_of(&self, plan: &LanePlan) -> R<(usize, usize)> {
        let lane = self
            .epochs
            .iter()
            .position(|e| *e == plan.key.request_epoch)
            .ok_or("window of a lane outside this run")?;
        let t = plan
            .position
            .checked_sub(self.refs[lane].prompt_len)
            .ok_or("lane position below its prompt length")?;
        Ok((lane, t))
    }
}

impl AcceptRule for TeacherRule<'_> {
    fn draft_token(
        &mut self,
        lane: &LanePlan,
        index: usize,
        _prior_drafts: &[u32],
        _proposal: &hipfire_arch_qwen4::mesh_lane_protocol::DraftProposal,
        _rng: &mut LaneRng,
    ) -> R<u32> {
        let (idx, t) = self.lane_of(lane)?;
        forced_draft(&self.refs[idx].ids, t, index)
    }

    fn observe_verify(&mut self, out: &SamplerOutput) {
        self.rows = out.rows.clone();
        let rows: Vec<usize> = out.picks.iter().map(|(_, p)| p.len()).collect();
        let wide = vec![self.wide; rows.len()];
        let groups = pack_classed_row_groups(&rows, &wide);
        self.group_rows.clear();
        for g in &groups {
            for l in g.first_lane..g.first_lane + g.lanes {
                self.group_rows.push((out.picks[l].0, g.rows));
            }
        }
        self.steps.push(StepShape {
            pass_rows: rows.iter().sum(),
            groups: groups.iter().map(|g| (g.rows, g.lanes)).collect(),
        });
    }

    fn accept(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        picks: &[u32],
        _rng: &mut LaneRng,
    ) -> R<GreedyAccept> {
        if !matches!(lane.kind, LaneKind::Decode) {
            return Err("the replay only runs decode windows".into());
        }
        let (idx, t) = self.lane_of(lane)?;
        let reference = &self.refs[idx];
        let d = drafts.len();
        let committed = forced_commit(&reference.ids, t, d)?;
        if drafts != &committed[..d] {
            return Err(format!(
                "lane {idx}: window drafts {drafts:?} are not the forced trajectory {:?}",
                &committed[..d]
            ));
        }
        if picks.len() != d + 1 || t + d >= reference.rows.len() {
            return Err(format!(
                "lane {idx}: {} picks for {d} drafts at seed {t} ({} reference rows)",
                picks.len(),
                reference.rows.len()
            ));
        }
        let handle = self
            .rows
            .as_ref()
            .ok_or("the sampler owner returned no verify row handle")?;
        let mut cands: Vec<Vec<f32>> = Vec::with_capacity(d + 1);
        for r in 0..=d {
            let mut host = Vec::new();
            handle.row_logits(&lane.key, r, &mut host)?;
            if host.len() != reference.rows[0].len() {
                return Err(format!(
                    "verify row of {} logits, vocabulary {}",
                    host.len(),
                    reference.rows[0].len()
                ));
            }
            cands.push(host);
        }
        let metrics: Vec<RowMetrics> = std::thread::scope(|s| {
            let handles: Vec<_> = cands
                .iter()
                .enumerate()
                .map(|(r, cand)| s.spawn(move || measure(reference, t + r, cand)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("metric thread"))
                .collect()
        });
        let group_rows = self
            .group_rows
            .iter()
            .find(|(k, _)| *k == lane.key)
            .map_or(0, |(_, g)| *g);
        for (r, m) in metrics.into_iter().enumerate() {
            let pos = t + r;
            self.obs.push(RowObs {
                lane: idx,
                pos,
                row: r,
                win_rows: d + 1,
                group_rows,
                kl: m.kl,
                max_abs: m.max_abs,
                top1_agree: m.top1_agree,
                pick_agree: picks[r] == reference.top1[ref_row_of(pos)],
                cand_top1: m.cand_top1,
                ref_top1: reference.top1[ref_row_of(pos)],
                ref_margin: reference.margin[ref_row_of(pos)],
                finite: m.finite,
            });
        }
        Ok(GreedyAccept {
            committed,
            accepted: d,
            hit_eos: false,
        })
    }
}

// ----------------------------------------------------------- aggregation

fn rows_json(rows: &[&RowObs], threshold: f64) -> Value {
    let kl: Vec<f64> = rows.iter().map(|o| o.kl).collect();
    let abs: Vec<f64> = rows.iter().map(|o| f64::from(o.max_abs)).collect();
    let n = rows.len().max(1) as f64;
    json!({
        "rows": rows.len(),
        "kl": Dist::of(&kl, threshold).json(threshold),
        "max_abs_dlogit": Dist::of(&abs, 0.0).json(0.0),
        "top1_agree": rows.iter().filter(|o| o.top1_agree).count() as f64 / n,
        "pick_agree": rows.iter().filter(|o| o.pick_agree).count() as f64 / n,
        "bit_identical_rows": rows.iter().filter(|o| o.max_abs == 0.0).count(),
    })
}

/// Trajectory third of a position (0, 1, 2).
fn third(pos: usize, tokens: usize) -> usize {
    (pos * 3 / tokens.max(1)).min(2)
}

fn hist(values: impl Iterator<Item = usize>) -> Value {
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    for v in values {
        *map.entry(v).or_default() += 1;
    }
    json!(map
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect::<BTreeMap<_, _>>())
}

/// One arm's receipt. `kl` is the overall distribution (also needed by the
/// gate).
fn summarize_arm(
    obs: &[RowObs],
    steps: &[StepShape],
    lanes: usize,
    tokens: usize,
    thr: f64,
) -> (Value, Dist) {
    let all: Vec<&RowObs> = obs.iter().collect();
    let kl_all = Dist::of(&all.iter().map(|o| o.kl).collect::<Vec<_>>(), thr);
    let mut value = rows_json(&all, thr);
    let mut by_row = serde_json::Map::new();
    let max_row = obs.iter().map(|o| o.row).max().unwrap_or(0);
    for r in 0..=max_row {
        let sel: Vec<&RowObs> = obs.iter().filter(|o| o.row == r).collect();
        if !sel.is_empty() {
            by_row.insert(r.to_string(), rows_json(&sel, thr));
        }
    }
    let by_lane: Vec<Value> = (0..lanes)
        .map(|l| {
            let sel: Vec<&RowObs> = obs.iter().filter(|o| o.lane == l).collect();
            let mut v = rows_json(&sel, thr);
            v["lane"] = json!(l);
            v
        })
        .collect();
    let by_third: Vec<Value> = (0..3)
        .map(|th| {
            let sel: Vec<&RowObs> = obs.iter().filter(|o| third(o.pos, tokens) == th).collect();
            let mut v = rows_json(&sel, thr);
            v["third"] = json!(th);
            v
        })
        .collect();
    // Disagreements with the singleton's top-1, worst reference margin first
    // is the safe direction (a flip at a large margin is a real error).
    let mut flips: Vec<&RowObs> = obs.iter().filter(|o| !o.top1_agree).collect();
    flips.sort_by(|a, b| b.ref_margin.total_cmp(&a.ref_margin));
    value["by_row_index"] = Value::Object(by_row);
    value["by_lane"] = json!(by_lane);
    value["by_trajectory_third"] = json!(by_third);
    value["top1_flips"] = json!(flips
        .iter()
        .take(40)
        .map(|o| json!({
            "lane": o.lane, "pos": o.pos, "row": o.row,
            "ref_top1": o.ref_top1, "cand_top1": o.cand_top1,
            "ref_margin": o.ref_margin, "kl": o.kl,
        }))
        .collect::<Vec<_>>());
    value["top1_flip_count"] = json!(flips.len());
    value["nonfinite_rows"] = json!(obs.iter().filter(|o| !o.finite).count());
    value["group_rows_hist"] = hist(obs.iter().map(|o| o.group_rows));
    value["window_rows_hist"] = hist(obs.iter().filter(|o| o.row == 0).map(|o| o.win_rows));
    value["pass_rows_hist"] = hist(steps.iter().map(|s| s.pass_rows));
    value["group_size_hist"] = hist(steps.iter().flat_map(|s| s.groups.iter().map(|g| g.0)));
    value["lanes_per_group_hist"] = hist(steps.iter().flat_map(|s| s.groups.iter().map(|g| g.1)));
    value["steps"] = json!(steps.len());
    value["expected_rows"] = json!(lanes * tokens);
    let mut covered = vec![0u8; lanes * tokens];
    for o in obs {
        if o.lane < lanes && o.pos < tokens {
            covered[o.lane * tokens + o.pos] = covered[o.lane * tokens + o.pos].saturating_add(1);
        }
    }
    value["coverage_exact"] = json!(covered.iter().all(|&c| c == 1) && obs.len() == lanes * tokens);
    (value, kl_all)
}

/// Gate thresholds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Gate {
    mean: f64,
    p99: Option<f64>,
    max: Option<f64>,
    frac: Option<f64>,
    top1: Option<f64>,
}

/// Failure reasons of a config; empty = PASS. `wide`/`control` are the arms'
/// overall KL distributions; `top1` is the wide arm's top-1 agreement.
fn judge(
    gate: &Gate,
    wide: &Dist,
    wide_top1: f64,
    wide_clean: bool,
    control: Option<&Dist>,
    control_clean: bool,
) -> Vec<String> {
    let mut fails = Vec::new();
    if !wide.mean.is_finite() || wide.mean > gate.mean {
        fails.push(format!(
            "wide mean KL {:.3e} > gate {:.3e}",
            wide.mean, gate.mean
        ));
    }
    if let Some(g) = gate.p99 {
        if !(wide.p99 <= g) {
            fails.push(format!("wide p99 KL {:.3e} > {g:.3e}", wide.p99));
        }
    }
    if let Some(g) = gate.max {
        if !(wide.max <= g) {
            fails.push(format!("wide max KL {:.3e} > {g:.3e}", wide.max));
        }
    }
    if let Some(g) = gate.frac {
        if !(wide.frac_gt <= g) {
            fails.push(format!(
                "wide fraction of rows over the threshold {:.4} > {g}",
                wide.frac_gt
            ));
        }
    }
    if let Some(g) = gate.top1 {
        if !(wide_top1 >= g) {
            fails.push(format!("wide top-1 agreement {wide_top1:.4} < {g}"));
        }
    }
    if !wide_clean {
        fails.push(
            "wide arm: incomplete coverage, non-finite logits, or the wide class did not run"
                .into(),
        );
    }
    if let Some(c) = control {
        if !c.mean.is_finite() || c.mean > gate.mean {
            fails.push(format!(
                "control (row-exact) mean KL {:.3e} > gate {:.3e}: the singleton baseline itself is off",
                c.mean, gate.mean
            ));
        }
        if !control_clean {
            fails.push(
                "control arm: incomplete coverage, non-finite logits, or a wide group ran".into(),
            );
        }
    }
    fails
}

// ------------------------------------------------------------------ args

#[derive(Debug, Clone, PartialEq)]
struct Args {
    model: String,
    max_seq: usize,
    tokens: usize,
    lanes: Vec<usize>,
    depths: Vec<usize>,
    sets: Vec<String>,
    prompt_files: Vec<String>,
    prompt_root: String,
    reserve_lanes: Option<usize>,
    skip_control: bool,
    gate: Gate,
    kl_threshold: f64,
    out: PathBuf,
}

fn parse_list(value: &str) -> R<Vec<usize>> {
    let mut v: Vec<usize> = value
        .split(',')
        .filter(|p| !p.trim().is_empty())
        .map(|p| p.trim().parse::<usize>().map_err(|x| format!("{p:?}: {x}")))
        .collect::<R<_>>()?;
    v.sort_unstable();
    v.dedup();
    if v.is_empty() {
        return Err("empty list".into());
    }
    Ok(v)
}

fn parse_args_from(mut it: impl Iterator<Item = String>) -> R<Args> {
    let mut a = Args {
        model: String::new(),
        max_seq: 4096,
        tokens: 96,
        lanes: vec![4, 8],
        depths: vec![3, 5],
        sets: Vec::new(),
        prompt_files: Vec::new(),
        prompt_root: DEFAULT_PROMPT_ROOT.to_string(),
        reserve_lanes: None,
        skip_control: false,
        gate: Gate {
            mean: 1e-4,
            p99: None,
            max: None,
            frac: None,
            top1: None,
        },
        kl_threshold: 1e-4,
        out: PathBuf::new(),
    };
    while let Some(flag) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        let float = |name: &str, v: String| -> R<f64> {
            v.parse::<f64>().map_err(|x| format!("{name}: {x}"))
        };
        match flag.as_str() {
            "--model" => a.model = value("--model")?,
            "--max-seq" => a.max_seq = value("--max-seq")?.parse().map_err(e)?,
            "--tokens" => a.tokens = value("--tokens")?.parse().map_err(e)?,
            "--lanes" => a.lanes = parse_list(&value("--lanes")?)?,
            "--depths" => a.depths = parse_list(&value("--depths")?)?,
            "--prompt-sets" => {
                a.sets = value("--prompt-sets")?
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().to_string())
                    .collect()
            }
            "--prompt-file" => a.prompt_files.push(value("--prompt-file")?),
            "--prompt-root" => a.prompt_root = value("--prompt-root")?,
            "--reserve-lanes" => {
                a.reserve_lanes = Some(value("--reserve-lanes")?.parse().map_err(e)?)
            }
            "--skip-control" => a.skip_control = true,
            "--gate-mean" => a.gate.mean = float("--gate-mean", value("--gate-mean")?)?,
            "--gate-p99" => a.gate.p99 = Some(float("--gate-p99", value("--gate-p99")?)?),
            "--gate-max" => a.gate.max = Some(float("--gate-max", value("--gate-max")?)?),
            "--gate-frac" => a.gate.frac = Some(float("--gate-frac", value("--gate-frac")?)?),
            "--gate-top1" => a.gate.top1 = Some(float("--gate-top1", value("--gate-top1")?)?),
            "--kl-threshold" => a.kl_threshold = float("--kl-threshold", value("--kl-threshold")?)?,
            "--out" => a.out = PathBuf::from(value("--out")?),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.model.is_empty() {
        return Err("--model is required".into());
    }
    if a.out.as_os_str().is_empty() {
        return Err("--out is required".into());
    }
    if a.tokens < 8 {
        return Err("--tokens must be at least 8".into());
    }
    if a.lanes.iter().any(|&n| n == 0 || n > 8) {
        return Err("--lanes values must be 1..=8".into());
    }
    if a.depths.iter().any(|&k| k == 0 || k > MAX_DEPTH) {
        return Err(format!("--depths values must be 1..={MAX_DEPTH}"));
    }
    if a.sets.is_empty() && a.prompt_files.is_empty() {
        a.sets.push("mixed".to_string());
    }
    if !a.prompt_files.is_empty() && !a.sets.is_empty() {
        return Err("--prompt-file and --prompt-sets are exclusive".into());
    }
    if a.prompt_files.len() < *a.lanes.last().expect("non-empty") && !a.prompt_files.is_empty() {
        return Err(format!(
            "{} --prompt-file given, --lanes needs {}",
            a.prompt_files.len(),
            a.lanes.last().expect("non-empty")
        ));
    }
    if let Some(r) = a.reserve_lanes {
        if r < *a.lanes.last().expect("non-empty") {
            return Err("--reserve-lanes must cover the largest --lanes".into());
        }
    }
    Ok(a)
}

/// The `count` prompt files of a named set under `root`: `mixed` = `mixed/m{n}-*.txt`,
/// `story`/`code` (any other name) = `w1/{name}-{n}.txt`, n = 1..=count.
fn prompt_paths(set: &str, root: &Path, count: usize) -> R<Vec<String>> {
    let mut paths = Vec::with_capacity(count);
    for n in 1..=count {
        let path = if set == "mixed" {
            let dir = root.join("mixed");
            let prefix = format!("m{n}-");
            let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                .map_err(|x| format!("read {}: {x}", dir.display()))?
                .filter_map(|ent| ent.ok().map(|ent| ent.path()))
                .filter(|p| {
                    p.file_name()
                        .and_then(|f| f.to_str())
                        .is_some_and(|f| f.starts_with(&prefix) && f.ends_with(".txt"))
                })
                .collect();
            found.sort();
            found
                .into_iter()
                .next()
                .ok_or_else(|| format!("no {}/{prefix}*.txt", dir.display()))?
        } else {
            let p = root.join("w1").join(format!("{set}-{n}.txt"));
            if !p.is_file() {
                return Err(format!("{} does not exist", p.display()));
            }
            p
        };
        paths.push(path.display().to_string());
    }
    Ok(paths)
}

// ------------------------------------------------------------ device I/O

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn ids_fnv(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    format!("{:016x}", fnv1a(&bytes))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_logits(gpu: &mut Gpu, logits: &GpuTensor) -> R<Vec<f32>> {
    gpu.bind_thread().map_err(e)?;
    gpu.hip.device_synchronize().map_err(e)?;
    Ok(f32s(&gpu.download_raw_bytes(logits).map_err(e)?))
}

/// One singleton greedy run: `tokens + 1` forwards, `c[0..=tokens]` and the
/// logits of the first `tokens` of them.
fn singleton_reference(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    logits: &GpuTensor,
    prompt: &[u32],
    tokens: usize,
) -> R<RefLane> {
    bundle.reset(gpu).map_err(e)?;
    bundle.prefill_final(gpu, prompt, 0, logits).map_err(e)?;
    let mut ids = Vec::with_capacity(tokens + 1);
    let mut rows = Vec::with_capacity(tokens);
    for step in 1..=tokens + 1 {
        let id = bundle
            .forward_token_or_argmax(gpu, None, logits)
            .map_err(|x| format!("reference step {step}: {x}"))?;
        ids.push(id);
        if step <= tokens {
            rows.push(read_logits(gpu, logits)?);
        }
    }
    Ok(RefLane::new(prompt.len(), ids, rows))
}

struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    start: usize,
}

/// One prefill step's plan: per-slot rows for every slot `0..max_lanes`.
fn build_prefill_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> =
        (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let mut plan = BatchStepPlan {
        batch: SlotBatch::build(&triples),
        ..Default::default()
    };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows {
            epoch: r.epoch,
            rows: RowRange {
                begin,
                len: r.tokens.len(),
            },
            kind: RequestStepKind::Prefill,
        });
        plan.prefill_rows += r.tokens.len();
        begin += r.tokens.len();
    }
    plan
}

/// provision -> forward -> commit of one plan (no logits read).
fn lane_step(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    plan: &BatchStepPlan,
) -> R<Vec<hipfire_runtime::slot_batch::RequestAdvance>> {
    let mut ex = bundle.lane_executor().map_err(e)?;
    ex.provision_step(gpu, plan)
        .map_err(|m| format!("provision_step failed: {m}"))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(out) => out,
        Err(m) => {
            ex.abort_step(plan);
            return Err(format!("forward_step failed: {m}"));
        }
    };
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(advances),
        Err(m) => {
            ex.abort_step(plan);
            Err(format!("commit_step failed: {m}"))
        }
    }
}

fn mesh_step_error(error: MeshStepError) -> String {
    match error {
        MeshStepError::Preflight(v) => format!("mesh preflight rejected: {v:?}"),
        MeshStepError::Aborted(record) => format!("mesh step aborted: {record}"),
        MeshStepError::Protocol(p) => format!("mesh protocol: {p}"),
    }
}

// ------------------------------------------------------------------ arms

struct MeshLane {
    epoch: RequestEpoch,
    slot: usize,
    admitted: bool,
    in_mesh: bool,
}

struct ArmRun {
    obs: Vec<RowObs>,
    steps: Vec<StepShape>,
    /// `(row_groups, wide_groups)` per merged-evidence step of this arm.
    evidence: BTreeMap<u64, (usize, usize)>,
    wall_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_arm_inner(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    budgets: &WindowBudgets,
    refs: &[RefLane],
    prompts: &[&[u32]],
    depth: usize,
    wide: bool,
    tokens: usize,
    max_seq: usize,
    max_lanes: usize,
    next_tag: &mut u64,
    lanes: &mut Vec<MeshLane>,
) -> R<ArmRun> {
    let n = prompts.len();
    for prompt in prompts {
        *next_tag += 1;
        let epoch = RequestEpoch {
            request_tag: *next_tag,
            owner_generation: 1,
        };
        let slot = bundle.admit_lane(gpu, epoch, prompt.to_vec()).map_err(e)?;
        lanes.push(MeshLane {
            epoch,
            slot,
            admitted: true,
            in_mesh: false,
        });
    }
    // Prefill: one exact tile per Prefilling lane per step, through the lane
    // executor (identical on every rank); a lane enters the mesh when done.
    let mut prefilled = vec![0usize; n];
    while (0..n).any(|i| prefilled[i] < prompts[i].len()) {
        let mut specs = Vec::new();
        for i in (0..n).filter(|&i| prefilled[i] < prompts[i].len()) {
            let len = bundle
                .lane_executor()
                .map_err(e)?
                .exact_prefill_chunk_len(&lanes[i].epoch)?;
            if len == 0 || prefilled[i] + len > prompts[i].len() {
                return Err(format!(
                    "lane {i}: prefill tile of {len} rows at {}/{}",
                    prefilled[i],
                    prompts[i].len()
                ));
            }
            specs.push(RowSpec {
                slot: lanes[i].slot,
                epoch: lanes[i].epoch,
                tokens: prompts[i][prefilled[i]..prefilled[i] + len].to_vec(),
                start: prefilled[i],
            });
        }
        let plan = build_prefill_plan(max_lanes, specs);
        let advances = lane_step(bundle, gpu, &plan)?;
        mesh.note_prefill();
        for (request, advance) in plan.requests.iter().zip(&advances) {
            let i = lanes
                .iter()
                .position(|l| l.epoch == request.epoch)
                .ok_or("plan epoch is not one of this run's lanes")?;
            prefilled[i] += request.rows.len;
            if prefilled[i] < prompts[i].len() {
                continue;
            }
            let [seed] = advance.committed_ids.as_slice() else {
                return Err(format!(
                    "lane {i}: prompt-completing tile committed {:?}",
                    advance.committed_ids
                ));
            };
            if *seed != refs[i].ids[0] {
                return Err(format!(
                    "lane {i}: the lane prefill picked {seed}, the singleton picked {} (prefill identity broken)",
                    refs[i].ids[0]
                ));
            }
            mesh.admit_lane(
                bundle,
                gpu,
                MeshLaneAdmission {
                    epoch: lanes[i].epoch,
                    position: prompts[i].len(),
                    seed: *seed,
                    eos: None,
                    rng_seed: lanes[i].epoch.request_tag,
                    prompt: prompts[i].to_vec(),
                    spec_k: Some(depth),
                },
            )?;
            lanes[i].in_mesh = true;
        }
    }

    let epochs: Vec<RequestEpoch> = lanes.iter().map(|l| l.epoch).collect();
    let mut rule = TeacherRule {
        refs,
        epochs: &epochs,
        wide,
        rows: None,
        group_rows: Vec::new(),
        obs: Vec::new(),
        steps: Vec::new(),
    };
    let mut t = vec![0usize; n];
    let mut first_seq: Option<u64> = None;
    let mut evidence: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
    let started = Instant::now();
    loop {
        let active: Vec<usize> = (0..n).filter(|&i| t[i] < tokens).collect();
        if active.is_empty() {
            break;
        }
        let mut requests = Vec::with_capacity(active.len());
        for &i in &active {
            let position = prompts[i].len() + t[i];
            match mesh.root().lane_position(lanes[i].epoch) {
                Some((p, _)) if p == position => {}
                other => {
                    return Err(format!(
                        "lane {i}: root position {other:?}, replay expects {position}"
                    ))
                }
            }
            requests.push(LaneRequest {
                wide_verify: wide,
                epoch: lanes[i].epoch,
                remaining: window_remaining(tokens, t[i]),
                context_left: max_seq.saturating_sub(position),
                forced: None,
                sampling: SamplingMode::NaiveSampled,
            });
        }
        let report = mesh
            .step(bundle, gpu, &requests, budgets, &mut rule)
            .map_err(mesh_step_error)?;
        first_seq.get_or_insert(report.key.step_seq);
        for out in &report.output.lanes {
            let i = lanes
                .iter()
                .position(|l| l.epoch == out.key.request_epoch)
                .ok_or("mesh output epoch is not one of this run's lanes")?;
            let want = forced_commit(&refs[i].ids, t[i], out.consumed_rows - 1)?;
            if out.emitted != want {
                return Err(format!(
                    "lane {i}: window emitted {:?}, the forced trajectory is {want:?}",
                    out.emitted
                ));
            }
            t[i] += out.consumed_rows;
        }
        if let (Some(ev), Some(first)) = (mesh.last_evidence(), first_seq) {
            if ev.key.step_seq >= first {
                evidence.insert(
                    ev.key.step_seq,
                    (ev.counts.row_groups, ev.counts.wide_groups),
                );
            }
        }
    }
    Ok(ArmRun {
        obs: rule.obs,
        steps: rule.steps,
        evidence,
        wall_s: started.elapsed().as_secs_f64(),
    })
}

/// One arm on freshly admitted lanes; every lane is released (also on error).
#[allow(clippy::too_many_arguments)]
fn run_arm(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    mesh: &mut EpMeshLanes,
    budgets: &WindowBudgets,
    refs: &[RefLane],
    prompts: &[&[u32]],
    depth: usize,
    wide: bool,
    tokens: usize,
    max_seq: usize,
    max_lanes: usize,
    next_tag: &mut u64,
) -> R<ArmRun> {
    let mut lanes: Vec<MeshLane> = Vec::new();
    let result = run_arm_inner(
        bundle, gpu, mesh, budgets, refs, prompts, depth, wide, tokens, max_seq, max_lanes,
        next_tag, &mut lanes,
    );
    let mut release_error = None;
    for lane in lanes.iter_mut() {
        if lane.in_mesh {
            lane.in_mesh = false;
            if let Err(x) = mesh.release_lane(bundle, gpu, lane.epoch) {
                release_error.get_or_insert(format!("mesh release {:?}: {x}", lane.epoch));
            }
        }
        if lane.admitted {
            lane.admitted = false;
            if let Err(x) = bundle.release_lane(&lane.epoch) {
                release_error.get_or_insert(format!("release {:?}: {x}", lane.epoch));
            }
        }
    }
    match (result, release_error) {
        (Err(x), _) => Err(x),
        (Ok(_), Some(x)) => Err(x),
        (Ok(run), None) => Ok(run),
    }
}

// ------------------------------------------------------------------ load

struct Prompt {
    path: String,
    ids: Vec<u32>,
}

fn load_prompt(path: &str, tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> R<Prompt> {
    let bytes = std::fs::read(path).map_err(|x| format!("read {path}: {x}"))?;
    let text = std::str::from_utf8(&bytes).map_err(|x| format!("{path}: not UTF-8: {x}"))?;
    let ids = tokenizer.encode(text);
    if ids.is_empty() {
        return Err(format!("{path}: empty prompt"));
    }
    Ok(Prompt {
        path: path.to_string(),
        ids,
    })
}

/// `load_model`'s admitted route with the per-load lane reservation and the
/// native MTP head armed (`fn_mesh_lane_oracle --spec mtp --reserve-lanes`).
fn load_model(
    args: &Args,
    lanes: usize,
    spec_k: usize,
    gpu: &mut Gpu,
) -> R<hipfire_loader::LoadedModel> {
    use hipfire_loader::admission::{
        admit_source_with_options, qwen4_kv_adaptive_requested, qwen_default_q8_enabled,
        BatchLoadRequest, KvBackendHints, KvBackendRequest, SourceAdmissionOptions,
    };
    let cask = hipfire_runtime::loader_api::CaskConfig::default();
    let spec = hipfire_runtime::loader_api::SpecLoadCfg {
        mtp: Some(true),
        ..Default::default()
    };
    let mut admission = admit_source_with_options(
        &args.model,
        1,
        1,
        KvBackendRequest::Automatic,
        None,
        gpu.arch.as_str(),
        None,
        "auto",
        None,
        args.max_seq,
        KvBackendHints {
            kv_mode: None,
            kv_k: None,
            kv_v: None,
            qwen_default_q8: qwen_default_q8_enabled(),
            kv_adaptive: None,
            cask: Some(&cask),
            deepseek4_heterogeneous: false,
            vmm_runtime_available: gpu.vmm_recommended_granularity().is_ok(),
            free_vram_bytes: gpu.device_mem_info().ok().map(|(free, _)| free),
        },
        SourceAdmissionOptions {
            spec,
            kv_adaptive: qwen4_kv_adaptive_requested(None),
            cask: cask.sidecar.is_some(),
            ..Default::default()
        },
    )?;
    admission.batch_request = BatchLoadRequest {
        vmm_batch: true,
        continuous_batch_size: lanes,
        row_budget: lanes,
        spec_k: Some(spec_k),
        spec_copy: false,
    };
    hipfire_loader::load_admitted_with_gemma4_drafter(
        admission,
        &args.model,
        args.max_seq,
        None,
        hipfire_config::Deepseek4ComputePlacement::Single,
        None,
        None,
        hipfire_loader::GEMMA4_EAGLE_DRAFT_LEN,
        None,
        None,
        None,
        None,
        None,
        &cask,
        1,
        spec,
        gpu,
    )
}

// ------------------------------------------------------------------- run

fn write_text(path: &Path, text: &str) -> R<()> {
    std::fs::write(path, text).map_err(|x| format!("write {}: {x}", path.display()))
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    PathBuf::from(s)
}

fn main() {
    // The wide class and the 32-row gather capacity are process-snapshot flags:
    // set before anything reads the configuration.
    if std::env::var_os(SAMPLED_WIDE_ENV).is_none() {
        std::env::set_var(SAMPLED_WIDE_ENV, "1");
    }
    let code = match parse_args_from(std::env::args().skip(1)).and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{TOOL}: {error}");
            2
        }
    };
    std::process::exit(code);
}

fn run(args: Args) -> R<i32> {
    let mut report = json!({
        "tool": TOOL,
        "model": args.model,
        "tokens": args.tokens,
        "lanes": args.lanes,
        "depths": args.depths,
        "kl_threshold": args.kl_threshold,
        "gate": {
            "mean": args.gate.mean, "p99": args.gate.p99, "max": args.gate.max,
            "frac": args.gate.frac, "top1": args.gate.top1,
        },
        "control_arm": !args.skip_control,
        "env": std::env::vars()
            .filter(|(k, _)| k.starts_with("HIPFIRE_"))
            .collect::<BTreeMap<_, _>>(),
        "configs": [],
    });
    let result = run_inner(&args, &mut report);
    if let Err(error) = &result {
        report["error"] = json!(error);
        report["verdict"] = json!("ERROR");
    }
    let rendered = serde_json::to_vec_pretty(&report).map_err(e)?;
    std::fs::write(&args.out, rendered)
        .map_err(|x| format!("write {}: {x}", args.out.display()))?;
    result
}

fn run_inner(args: &Args, report: &mut Value) -> R<i32> {
    if hipfire_config::qwen4_ep_wide_verify_all_lanes() {
        return Err(format!(
            "{ALL_LANES_ENV} is set: it would make the row-exact control arm wide too; unset it"
        ));
    }
    if !hipfire_config::qwen4_ep_sampled_wide_verify() {
        return Err(format!(
            "{SAMPLED_WIDE_ENV} did not read as 1 through the process configuration snapshot"
        ));
    }
    let max_lanes = *args.lanes.last().expect("non-empty");
    let max_depth = *args.depths.last().expect("non-empty");
    let reserve = args.reserve_lanes.unwrap_or(max_lanes);
    eprintln!("[{TOOL}] loading {} (wide verify flag on)", args.model);
    let mut gpu = Gpu::init().map_err(|x| format!("Gpu::init: {x:?}"))?;
    let mut model = load_model(args, reserve, max_depth, &mut gpu)?;
    let mut set_prompts: Vec<(String, Vec<Prompt>)> = Vec::new();
    {
        let tokenizer = model
            .tokenizer
            .as_ref()
            .ok_or("loaded model has no tokenizer")?;
        if args.prompt_files.is_empty() {
            for set in &args.sets {
                let paths = prompt_paths(set, Path::new(&args.prompt_root), max_lanes)?;
                let prompts = paths
                    .iter()
                    .map(|p| load_prompt(p, tokenizer))
                    .collect::<R<Vec<_>>>()?;
                set_prompts.push((set.clone(), prompts));
            }
        } else {
            let prompts = args
                .prompt_files
                .iter()
                .take(max_lanes)
                .map(|p| load_prompt(p, tokenizer))
                .collect::<R<Vec<_>>>()?;
            set_prompts.push(("custom".to_string(), prompts));
        }
    }
    for (set, prompts) in &set_prompts {
        for p in prompts {
            if p.ids.len() + args.tokens + 2 > args.max_seq {
                return Err(format!(
                    "{set}: {} has {} prompt tokens + {} decode tokens, over --max-seq {}",
                    p.path,
                    p.ids.len(),
                    args.tokens,
                    args.max_seq
                ));
            }
        }
    }
    let bundle = model
        .qwen4_mut()
        .ok_or("loaded model is not a Qwen4 (Flash-Next) bundle")?;
    if bundle.ep_mesh().is_none() {
        return Err("no EP mesh attached (set topology.ep / HIPFIRE_TOPOLOGY_EP=2)".into());
    }
    let vocab = bundle.config.vocab_size;
    gpu.bind_thread().map_err(e)?;
    let logits = gpu.zeros(&[vocab], DType::F32).map_err(e)?;
    report["vocab"] = json!(vocab);
    report["ranks"] = json!(bundle.ranks());
    report["gather_rows"] = json!(hipfire_arch_qwen4::gpu_forward::qwen4_ep_gather_rows());

    // Singleton references (before the lane store exists), then stage.
    let mut refs_by_set: Vec<Vec<RefLane>> = Vec::new();
    for (set, prompts) in &set_prompts {
        let mut refs = Vec::with_capacity(prompts.len());
        for (i, p) in prompts.iter().enumerate() {
            eprintln!(
                "[{TOOL}] reference {set}/{i}: {} ({} tokens)",
                p.path,
                p.ids.len()
            );
            refs.push(singleton_reference(
                bundle,
                &mut gpu,
                &logits,
                &p.ids,
                args.tokens,
            )?);
        }
        refs_by_set.push(refs);
    }
    report["prompts"] = json!(set_prompts
        .iter()
        .zip(&refs_by_set)
        .map(|((set, prompts), refs)| json!({
            "set": set,
            "prompts": prompts.iter().zip(refs).enumerate().map(|(i, (p, r))| json!({
                "lane": i, "path": p.path, "prompt_tokens": p.ids.len(),
                "prompt_ids_fnv": ids_fnv(&p.ids), "trajectory_ids_fnv": ids_fnv(&r.ids),
            })).collect::<Vec<_>>(),
        }))
        .collect::<Vec<_>>());

    bundle
        .stage_lanes_with_copy(
            &mut gpu,
            max_lanes,
            max_lanes,
            Some(LaneSpecConfig {
                k: max_depth,
                // The trajectory ignores EOS (the singleton reference does).
                end_of_turn: Some(u32::MAX),
                spec_lanes: None,
            }),
            None,
        )
        .map_err(e)?;
    let mut mesh = EpMeshLanes::new(bundle, 1, false)?;
    let budgets = mesh.window_budgets(bundle, LabRoutes::Batched)?;
    let mut rows_file = std::io::BufWriter::new(
        std::fs::File::create(sibling(&args.out, ".rows.jsonl")).map_err(e)?,
    );
    let mut summary = String::new();
    let mut next_tag = 0u64;
    let mut any_fail = false;
    let thr = args.kl_threshold;
    let mut configs = Vec::new();
    for ((set, prompts), refs) in set_prompts.iter().zip(&refs_by_set) {
        for &n in &args.lanes {
            for &depth in &args.depths {
                let prompt_ids: Vec<&[u32]> =
                    prompts[..n].iter().map(|p| p.ids.as_slice()).collect();
                let mut arm_values = serde_json::Map::new();
                let mut dists: BTreeMap<&str, (Dist, f64, bool)> = BTreeMap::new();
                let arms: &[(&str, bool)] = if args.skip_control {
                    &[("wide", true)]
                } else {
                    &[("exact", false), ("wide", true)]
                };
                for &(arm, wide) in arms {
                    eprintln!("[{TOOL}] {set} lanes={n} depth={depth} arm={arm}");
                    let run = run_arm(
                        bundle,
                        &mut gpu,
                        &mut mesh,
                        &budgets,
                        &refs[..n],
                        &prompt_ids,
                        depth,
                        wide,
                        args.tokens,
                        args.max_seq,
                        max_lanes,
                        &mut next_tag,
                    )
                    .map_err(|x| format!("{set} lanes={n} depth={depth} arm={arm}: {x}"))?;
                    for o in &run.obs {
                        let line = json!({
                            "set": set, "lanes": n, "depth": depth, "arm": arm,
                            "lane": o.lane, "pos": o.pos, "row": o.row,
                            "win_rows": o.win_rows, "group_rows": o.group_rows,
                            "kl": o.kl, "max_abs": o.max_abs,
                            "top1_agree": o.top1_agree, "pick_agree": o.pick_agree,
                        });
                        writeln!(rows_file, "{line}").map_err(e)?;
                    }
                    let (mut value, kl) = summarize_arm(&run.obs, &run.steps, n, args.tokens, thr);
                    let wide_groups: usize = run.evidence.values().map(|v| v.1).sum();
                    let row_groups: usize = run.evidence.values().map(|v| v.0).sum();
                    value["evidence"] = json!({
                        "steps": run.evidence.len(), "row_groups": row_groups,
                        "wide_groups": wide_groups,
                    });
                    value["wall_s"] = json!(run.wall_s);
                    // The merged evidence lags one step; an arm that ran more than one
                    // step must show its class there.
                    let class_ok = run.evidence.is_empty()
                        || (wide && wide_groups > 0)
                        || (!wide && wide_groups == 0);
                    let clean = value["coverage_exact"] == json!(true)
                        && value["nonfinite_rows"] == json!(0)
                        && class_ok;
                    value["clean"] = json!(clean);
                    let top1 = value["top1_agree"].as_f64().unwrap_or(0.0);
                    dists.insert(arm, (kl, top1, clean));
                    arm_values.insert(arm.to_string(), value);
                    let (kl, top1, _) = &dists[arm];
                    summary.push_str(&format!(
                        "{set:6} lanes={n} K={depth} {arm:5}: rows={} KL mean {:.3e} p50 {:.3e} p99 {:.3e} max {:.3e} frac>{thr:.0e} {:.4} top1 {:.4} wall {:.1}s\n",
                        kl.n, kl.mean, kl.p50, kl.p99, kl.max, kl.frac_gt, top1, run.wall_s
                    ));
                }
                let (wide_kl, wide_top1, wide_clean) = dists.get("wide").expect("wide arm ran");
                let control = dists.get("exact");
                let fails = judge(
                    &args.gate,
                    wide_kl,
                    *wide_top1,
                    *wide_clean,
                    control.map(|c| &c.0),
                    control.map_or(true, |c| c.2),
                );
                let verdict = if fails.is_empty() { "PASS" } else { "FAIL" };
                any_fail |= !fails.is_empty();
                summary.push_str(&format!(
                    "{set:6} lanes={n} K={depth} => {verdict}{}\n",
                    if fails.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", fails.join("; "))
                    }
                ));
                configs.push(json!({
                    "set": set, "lanes": n, "depth": depth,
                    "arms": Value::Object(arm_values),
                    "verdict": verdict, "failures": fails,
                }));
                report["configs"] = json!(configs);
                rows_file.flush().map_err(e)?;
            }
        }
    }
    report["verdict"] = json!(if any_fail { "FAIL" } else { "PASS" });
    report["completed"] = json!(true);
    write_text(&sibling(&args.out, ".summary.txt"), &summary)?;
    print!("{summary}");
    println!("{} {TOOL}", if any_fail { "FAIL" } else { "PASS" });
    Ok(if any_fail { 1 } else { 0 })
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    fn lnrow(p: &[f64]) -> Vec<f32> {
        p.iter().map(|x| x.ln() as f32).collect()
    }

    #[test]
    fn log_sum_exp_matches_naive_and_handles_edges() {
        let row = [0.5f32, -1.0, 2.0, 0.0];
        let naive: f64 = row.iter().map(|&x| f64::from(x).exp()).sum::<f64>().ln();
        assert!((log_sum_exp(&row) - naive).abs() < 1e-12);
        // large offsets do not overflow
        let big = [1000.0f32, 1000.0];
        assert!((log_sum_exp(&big) - (1000.0 + 2f64.ln())).abs() < 1e-9);
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert!(log_sum_exp(&[0.0, f32::NAN]).is_nan());
    }

    #[test]
    fn kl_identical_rows_is_exactly_zero() {
        let row = [3.0f32, -2.0, 0.25, 7.5, 7.5];
        let lse = log_sum_exp(&row);
        assert_eq!(kl_full(&row, lse, &row), 0.0);
    }

    #[test]
    fn kl_matches_a_known_two_class_value() {
        // ref = (0.5, 0.5), cand = (0.25, 0.75)
        let r = lnrow(&[0.5, 0.5]);
        let c = lnrow(&[0.25, 0.75]);
        let want = 0.5 * (0.5f64 / 0.25).ln() + 0.5 * (0.5f64 / 0.75).ln();
        let got = kl_full(&r, log_sum_exp(&r), &c);
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        assert!(got > 0.0);
    }

    #[test]
    fn kl_is_shift_invariant_and_direction_sensitive() {
        let r = [1.0f32, 0.0, -1.0, 2.0];
        let c = [0.5f32, 0.5, -0.5, 1.0];
        let shifted: Vec<f32> = c.iter().map(|x| x + 8.0).collect();
        let a = kl_full(&r, log_sum_exp(&r), &c);
        let b = kl_full(&r, log_sum_exp(&r), &shifted);
        assert!((a - b).abs() < 1e-9);
        let rev = kl_full(&c, log_sum_exp(&c), &r);
        assert!((a - rev).abs() > 1e-4, "KL is not symmetric");
    }

    #[test]
    fn kl_nonfinite_candidate_is_infinite() {
        let r = [1.0f32, 2.0];
        let lse = log_sum_exp(&r);
        assert_eq!(kl_full(&r, lse, &[f32::NAN, 0.0]), f64::INFINITY);
        assert_eq!(kl_full(&r, lse, &[f32::INFINITY, 0.0]), f64::INFINITY);
    }

    #[test]
    fn argmax_margin_and_diff() {
        let row = [0.1f32, 2.0, 2.0, -1.0];
        assert_eq!(argmax(&row), 1, "first maximum wins");
        assert_eq!(top1_margin(&row), 0.0);
        assert!((top1_margin(&[1.0, 4.0, 3.0]) - 1.0).abs() < 1e-7);
        assert_eq!(max_abs_diff(&[1.0, 2.0], &[1.5, 0.0]), 2.0);
        assert_eq!(max_abs_diff(&[1.0], &[f32::NAN]), f32::INFINITY);
    }

    #[test]
    fn dist_percentiles_are_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        let d = Dist::of(&v, 90.0);
        assert_eq!(d.n, 100);
        assert_eq!(d.p50, 50.0);
        assert_eq!(d.p90, 90.0);
        assert_eq!(d.p99, 99.0);
        assert_eq!(d.max, 100.0);
        assert!((d.mean - 50.5).abs() < 1e-12);
        assert!((d.frac_gt - 0.10).abs() < 1e-12);
        let empty = Dist::of(&[], 1.0);
        assert!(empty.mean.is_nan() && empty.max.is_nan());
    }

    #[test]
    fn forced_windows_follow_the_trajectory() {
        // c[0..=6]
        let ids = [10u32, 11, 12, 13, 14, 15, 16];
        assert_eq!(forced_draft(&ids, 0, 0).unwrap(), 11);
        assert_eq!(forced_draft(&ids, 2, 1).unwrap(), 14);
        assert!(forced_draft(&ids, 5, 1).is_err());
        // seed t=1 with 3 drafts: drafts c[2..=4], bonus c[5]
        assert_eq!(forced_commit(&ids, 1, 3).unwrap(), vec![12, 13, 14, 15]);
        // an AR window (no drafts) commits the next token only
        assert_eq!(forced_commit(&ids, 6 - 1, 0).unwrap(), vec![16]);
        assert!(forced_commit(&ids, 6, 0).is_err());
        assert!(forced_commit(&ids, 4, 2).is_err());
    }

    #[test]
    fn window_remaining_ends_a_lane_exactly_at_the_trajectory_end() {
        // The window depth is the lane's admitted spec_k (`depth`); `remaining` only bounds
        // the drafts so the bonus stays inside c[0..=tokens]. A window is terminal (the lane
        // finishes) iff it emits `remaining` tokens, i.e. only the last one.
        let (tokens, depth) = (10usize, 3usize);
        let (mut t, mut rows, mut terminal_windows) = (0, 0, 0);
        while t < tokens {
            let rem = window_remaining(tokens, t);
            let d = depth.min(rem - 1);
            assert!(t + d + 1 <= tokens);
            let emitted = d + 1;
            if emitted == rem {
                terminal_windows += 1;
                assert_eq!(t + emitted, tokens);
            } else {
                assert!(emitted < rem, "a non-final window must not be terminal");
            }
            rows += emitted;
            t += emitted;
        }
        // every position 0..tokens is processed exactly once, one terminal window
        assert_eq!((t, rows, terminal_windows), (tokens, tokens, 1));
    }

    #[test]
    fn reference_row_index_is_the_processed_token() {
        // forward s consumes c[s-1] and leaves rows[s-1]
        assert_eq!(ref_row_of(0), 0);
        assert_eq!(ref_row_of(95), 95);
    }

    #[test]
    fn thirds_split_the_trajectory() {
        assert_eq!((third(0, 96), third(31, 96), third(32, 96)), (0, 0, 1));
        assert_eq!((third(63, 96), third(64, 96), third(95, 96)), (1, 2, 2));
    }

    fn obs(lane: usize, pos: usize, row: usize, kl: f64) -> RowObs {
        RowObs {
            lane,
            pos,
            row,
            win_rows: 4,
            group_rows: 16,
            kl,
            max_abs: 0.5,
            top1_agree: true,
            pick_agree: true,
            cand_top1: 1,
            ref_top1: 1,
            ref_margin: 2.0,
            finite: true,
        }
    }

    #[test]
    fn summary_tracks_coverage_rows_and_index_tables() {
        let mut rows = Vec::new();
        for lane in 0..2 {
            for pos in 0..6 {
                rows.push(obs(
                    lane,
                    pos,
                    pos % 3,
                    if pos % 3 == 2 { 2e-4 } else { 0.0 },
                ));
            }
        }
        let steps = [StepShape {
            pass_rows: 12,
            groups: vec![(12, 2)],
        }];
        let (v, kl) = summarize_arm(&rows, &steps, 2, 6, 1e-4);
        assert_eq!(v["coverage_exact"], json!(true));
        assert_eq!(kl.n, 12);
        assert!((kl.frac_gt - 4.0 / 12.0).abs() < 1e-12);
        assert_eq!(v["by_row_index"]["2"]["rows"], json!(4));
        assert_eq!(v["by_lane"].as_array().unwrap().len(), 2);
        assert_eq!(v["pass_rows_hist"]["12"], json!(1));
        assert_eq!(v["group_size_hist"]["12"], json!(1));
        assert_eq!(v["lanes_per_group_hist"]["2"], json!(1));
        // a missing position breaks coverage
        let (v, _) = summarize_arm(&rows[..11], &steps, 2, 6, 1e-4);
        assert_eq!(v["coverage_exact"], json!(false));
    }

    #[test]
    fn judge_gates_mean_optional_stats_and_the_control() {
        let gate = Gate {
            mean: 1e-4,
            p99: Some(1e-3),
            max: None,
            frac: Some(0.05),
            top1: Some(0.99),
        };
        let mk = |mean: f64, p99: f64, frac: f64| Dist {
            n: 100,
            mean,
            p50: mean,
            p90: p99,
            p99,
            max: p99,
            frac_gt: frac,
        };
        let zero = mk(0.0, 0.0, 0.0);
        assert!(judge(&gate, &mk(5e-5, 5e-4, 0.01), 0.995, true, Some(&zero), true).is_empty());
        let f = judge(&gate, &mk(2e-4, 5e-4, 0.01), 0.995, true, Some(&zero), true);
        assert_eq!(f.len(), 1);
        assert!(f[0].contains("wide mean KL"));
        let f = judge(&gate, &mk(5e-5, 2e-3, 0.2), 0.5, true, None, true);
        assert_eq!(f.len(), 3, "p99, frac and top-1 gates: {f:?}");
        let f = judge(
            &gate,
            &mk(5e-5, 5e-4, 0.01),
            0.995,
            true,
            Some(&mk(3e-4, 1e-3, 0.0)),
            true,
        );
        assert!(f.iter().any(|m| m.contains("control")));
        let f = judge(
            &gate,
            &mk(5e-5, 5e-4, 0.01),
            0.995,
            false,
            Some(&zero),
            true,
        );
        assert!(f.iter().any(|m| m.contains("wide arm")));
        // NaN mean never passes
        let f = judge(&gate, &mk(f64::NAN, 0.0, 0.0), 1.0, true, None, true);
        assert!(!f.is_empty());
    }

    fn argv(line: &str) -> impl Iterator<Item = String> + '_ {
        line.split_whitespace().map(str::to_owned)
    }

    #[test]
    fn args_defaults_and_validation() {
        let a = parse_args_from(argv("--model m.mq4 --out r.json")).unwrap();
        assert_eq!((a.tokens, a.max_seq), (96, 4096));
        assert_eq!(
            (a.lanes.clone(), a.depths.clone()),
            (vec![4, 8], vec![3, 5])
        );
        assert_eq!(a.sets, vec!["mixed".to_string()]);
        assert_eq!(a.gate.mean, 1e-4);
        assert!(a.gate.p99.is_none() && !a.skip_control);
        let a = parse_args_from(argv(
            "--model m --out r --lanes 8,4,4 --depths 5,3 --prompt-sets mixed,story,code --gate-p99 1e-3 --tokens 128 --reserve-lanes 8",
        ))
        .unwrap();
        assert_eq!(a.lanes, vec![4, 8]);
        assert_eq!(a.depths, vec![3, 5]);
        assert_eq!(a.sets.len(), 3);
        assert_eq!(
            (a.gate.p99, a.tokens, a.reserve_lanes),
            (Some(1e-3), 128, Some(8))
        );
        assert!(parse_args_from(argv("--out r")).is_err());
        assert!(parse_args_from(argv("--model m --out r --lanes 9")).is_err());
        assert!(parse_args_from(argv("--model m --out r --depths 8")).is_err());
        assert!(parse_args_from(argv("--model m --out r --tokens 4")).is_err());
        assert!(parse_args_from(argv("--model m --out r --reserve-lanes 4")).is_err());
        assert!(parse_args_from(argv(
            "--model m --out r --prompt-file a --prompt-sets mixed"
        ))
        .is_err());
        assert!(parse_args_from(argv("--model m --out r --prompt-file a")).is_err());
        assert!(parse_args_from(argv("--model m --out r --bogus")).is_err());
    }

    #[test]
    fn prompt_paths_resolve_mixed_and_w1_sets() {
        let root = tempfile::tempdir().unwrap();
        let mixed = root.path().join("mixed");
        let w1 = root.path().join("w1");
        std::fs::create_dir_all(&mixed).unwrap();
        std::fs::create_dir_all(&w1).unwrap();
        for n in 1..=3 {
            std::fs::write(mixed.join(format!("m{n}-code-1.txt")), "x").unwrap();
            std::fs::write(w1.join(format!("story-{n}.txt")), "x").unwrap();
        }
        std::fs::write(mixed.join("ORDER.tsv"), "x").unwrap();
        let m = prompt_paths("mixed", root.path(), 3).unwrap();
        assert!(m[0].ends_with("mixed/m1-code-1.txt") && m[2].ends_with("m3-code-1.txt"));
        let s = prompt_paths("story", root.path(), 2).unwrap();
        assert!(s[1].ends_with("w1/story-2.txt"));
        assert!(prompt_paths("mixed", root.path(), 4).is_err());
        assert!(prompt_paths("code", root.path(), 1).is_err());
    }

    #[test]
    fn fnv_is_stable() {
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(ids_fnv(&[1, 2]).len(), 16);
        assert_eq!(f32s(&1.5f32.to_le_bytes()), vec![1.5]);
    }
}
