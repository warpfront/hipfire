// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! ABBA hipEvent timing core shared by `pm_decode_twins time` and `pm_loop`:
//! the allocation-free ABBA schedule, sample statistics and the separability
//! rule. `Incumbent` is arm A (the shipped/baseline kernel), `Candidate` is
//! arm B. Included with `#[path = "support/abba.rs"] mod abba;`.
#![allow(dead_code)]

use serde_json::{json, Value};

type Result<T, E = String> = std::result::Result<T, E>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

pub fn positive_count(s: &str) -> Result<usize> {
    let n = s.parse::<usize>().map_err(err)?;
    if n == 0 {
        return Err("counts must be positive".into());
    }
    Ok(n)
}

pub fn timing_counts(n: usize, rounds: usize) -> Result<(usize, usize)> {
    if n == 0 || rounds == 0 {
        return Err("counts must be positive".into());
    }
    let per_arm = n
        .checked_mul(rounds)
        .and_then(|v| v.checked_mul(2))
        .ok_or("timing call count overflow")?;
    let total = per_arm.checked_mul(2).ok_or("timing call count overflow")?;
    per_arm
        .checked_mul(std::mem::size_of::<f64>())
        .filter(|&v| v <= isize::MAX as usize)
        .ok_or("sample storage overflow")?;
    Ok((per_arm, total))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimingArm {
    Incumbent,
    Candidate,
}

/// Pure, allocation-free ABBA iterator; construction checks all count arithmetic.
pub fn timing_schedule(n: usize, rounds: usize) -> Result<impl Iterator<Item = TimingArm>> {
    let (_, total) = timing_counts(n, rounds)?;
    Ok((0..total).map(move |i| match (i / n) % 4 {
        0 | 3 => TimingArm::Incumbent,
        _ => TimingArm::Candidate,
    }))
}

#[derive(Debug)]
pub struct TimingStats {
    pub count: usize,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub p90: f64,
    pub mean: f64,
}

impl TimingStats {
    pub fn spread(&self) -> f64 {
        self.max - self.min
    }
    pub fn json(&self) -> Value {
        json!({"call_count": self.count, "median_ns": self.median, "min_ns": self.min,
            "max_ns": self.max, "p90_ns": self.p90, "mean_ns": self.mean,
            "spread_ns": self.spread()})
    }
}

pub fn timing_stats(samples: &[f64]) -> Result<TimingStats> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("timing samples must be nonempty, positive and finite".into());
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n.is_multiple_of(2) {
        sorted[n / 2 - 1] / 2.0 + sorted[n / 2] / 2.0
    } else {
        sorted[n / 2]
    };
    // ceil(0.9*n), without overflowing integer multiplication.
    let rank = n - n / 10;
    let mean = sorted.iter().map(|v| v / n as f64).sum::<f64>();
    if !mean.is_finite() {
        return Err("nonfinite timing mean".into());
    }
    Ok(TimingStats {
        count: n,
        median,
        min: sorted[0],
        max: sorted[n - 1],
        p90: sorted[rank - 1],
        mean,
    })
}

pub fn timing_relation(a: &TimingStats, b: &TimingStats) -> (f64, f64, f64, bool) {
    let spread = a.spread().max(b.spread());
    let delta = b.median - a.median;
    (
        b.median / a.median,
        spread,
        delta,
        delta.abs() > spread && (a.max < b.min || b.max < a.min),
    )
}

// ---------------------------------------------------------------------------
// Block-median rule (pm_loop). The legacy rule above (`timing_relation`, kept
// unchanged for `pm_decode_twins`) compares full sample ranges, so one hiccup
// launch (a 7 ms outlier among 300 us samples) vetoes separability and more
// samples only make that worse. This rule works on per-block medians instead.

/// Medians of consecutive `n`-launch blocks, in schedule order. Each arm's
/// samples are pushed in schedule order, so every consecutive `n` samples are
/// exactly one ABBA block of that arm.
pub fn block_medians(samples: &[f64], n: usize) -> Result<Vec<f64>> {
    if n == 0 || samples.is_empty() || samples.len() % n != 0 {
        return Err(format!(
            "{} samples are not a whole number of {n}-launch blocks",
            samples.len()
        ));
    }
    samples
        .chunks_exact(n)
        .map(|c| timing_stats(c).map(|s| s.median))
        .collect()
}

fn min_max(v: &[f64]) -> (f64, f64) {
    (
        v.iter().copied().fold(f64::INFINITY, f64::min),
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    )
}

/// Two-sided sign-test p-value of a `majority` count out of `n` rounds where
/// every round that is not in the majority (the other sign AND ties) counts
/// against: `min(1, 2 * P(X <= n - majority))`, X ~ Binomial(n, 1/2).
pub fn majority_sign_p(majority: usize, n: usize) -> f64 {
    if n == 0 || majority > n {
        return 1.0;
    }
    let tail = n - majority;
    let mut term = 0.5f64.powi(n as i32);
    let mut sum = term;
    for i in 0..tail {
        term *= (n - i) as f64 / (i + 1) as f64;
        sum += term;
    }
    (2.0 * sum).min(1.0)
}

/// `separable` of [`block_rule`]: the sign-test p must be at most this.
pub const PAIRED_P_MAX: f64 = 1e-3;

/// Outcome of [`block_rule`]. Ratios are candidate / baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockRule {
    /// PRIMARY rule: exact two-sided sign test over the per-round paired
    /// ratios (each ABBA round compares the candidate's and the baseline's
    /// block medians of that same round) has `sign_p <= PAIRED_P_MAX`; ties
    /// count against it.
    pub separable: bool,
    /// `"faster"` / `"slower"` from the majority sign of the rounds, `"tie"`
    /// when the signs are equal in number.
    pub direction: &'static str,
    pub sign_p: f64,
    pub rounds_faster: usize,
    pub rounds_slower: usize,
    pub rounds_tied: usize,
    /// Every candidate block median is below every baseline block median, or
    /// every one is above (secondary rule; between-block clock drift defeats it).
    pub block_disjoint: bool,
    /// `median(candidate block medians) / median(baseline block medians)`.
    pub block_ratio: f64,
    /// Per ABBA round, the mean of the candidate's two block medians over the
    /// mean of the baseline's; min and max over rounds.
    pub round_ratio_min: f64,
    pub round_ratio_max: f64,
    pub baseline_block_medians: Vec<f64>,
    pub candidate_block_medians: Vec<f64>,
}

/// Separability of two arms' samples (nanoseconds) taken with `n`-launch ABBA
/// blocks: whole rounds only (`2n` samples per arm per round, two blocks each).
pub fn block_rule(base: &[f64], cand: &[f64], n: usize) -> Result<BlockRule> {
    if base.len() != cand.len() || base.len() % (2 * n.max(1)) != 0 {
        return Err("block rule needs equal whole ABBA rounds of both arms".into());
    }
    let bm = block_medians(base, n)?;
    let cm = block_medians(cand, n)?;
    let med = |v: &[f64]| timing_stats(v).map(|s| s.median);
    let (lo, hi) = min_max(&bm);
    let (clo, chi) = min_max(&cm);
    let block_disjoint = chi < lo || clo > hi;
    let rr = round_ratios_from_blocks(&bm, &cm);
    let faster = rr.iter().filter(|r| **r < 1.0).count();
    let slower = rr.iter().filter(|r| **r > 1.0).count();
    let tied = rr.len() - faster - slower;
    let majority = faster.max(slower);
    let sign_p = majority_sign_p(majority, rr.len());
    Ok(BlockRule {
        separable: sign_p <= PAIRED_P_MAX,
        direction: match faster.cmp(&slower) {
            std::cmp::Ordering::Greater => "faster",
            std::cmp::Ordering::Less => "slower",
            std::cmp::Ordering::Equal => "tie",
        },
        sign_p,
        rounds_faster: faster,
        rounds_slower: slower,
        rounds_tied: tied,
        block_disjoint,
        block_ratio: med(&cm)? / med(&bm)?,
        round_ratio_min: min_max(&rr).0,
        round_ratio_max: min_max(&rr).1,
        baseline_block_medians: bm,
        candidate_block_medians: cm,
    })
}

/// Per ABBA round, the mean of the candidate's two block medians over the mean
/// of the baseline's (candidate / baseline) — the exact per-round ratios that
/// [`block_rule`] runs its sign test over. Same input contract as
/// [`block_rule`]: equal whole ABBA rounds of both arms.
pub fn round_ratios(base: &[f64], cand: &[f64], n: usize) -> Result<Vec<f64>> {
    if base.len() != cand.len() || base.len() % (2 * n.max(1)) != 0 {
        return Err("block rule needs equal whole ABBA rounds of both arms".into());
    }
    let bm = block_medians(base, n)?;
    let cm = block_medians(cand, n)?;
    Ok(round_ratios_from_blocks(&bm, &cm))
}

/// The single implementation of the per-round ratio, shared by [`block_rule`]
/// and [`round_ratios`]. Block medians come two per round.
fn round_ratios_from_blocks(bm: &[f64], cm: &[f64]) -> Vec<f64> {
    bm.chunks_exact(2)
        .zip(cm.chunks_exact(2))
        .map(|(b, c)| (c[0] + c[1]) / (b[0] + b[1]))
        .collect()
}

/// Linear-interpolation quantile (`q` in `[0, 1]`) of an already ascending
/// sorted, nonempty slice: position `q * (n - 1)`.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Dispersion summary of positive finite values (block medians, ratios).
#[derive(Clone, Debug, PartialEq)]
pub struct Dispersion {
    pub n: usize,
    pub median: f64,
    /// Median absolute deviation from the median, unscaled.
    pub mad: f64,
    /// `p75 - p25`.
    pub iqr: f64,
    pub p05: f64,
    pub p95: f64,
    pub min: f64,
    pub max: f64,
    /// Sample standard deviation over the mean (0 for a single value).
    pub cv: f64,
    /// Sample standard deviation of `ln(v)` (0 for a single value).
    pub log_sd: f64,
}

impl Dispersion {
    pub fn json(&self) -> Value {
        json!({"n": self.n, "median": self.median, "mad": self.mad, "iqr": self.iqr,
            "p05": self.p05, "p95": self.p95, "min": self.min, "max": self.max,
            "cv": self.cv, "log_sd": self.log_sd})
    }
}

/// [`Dispersion`] of `v`; quantiles use linear interpolation between order
/// statistics. Values must be nonempty, positive and finite.
pub fn dispersion(v: &[f64]) -> Result<Dispersion> {
    if v.is_empty() || v.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err("dispersion values must be nonempty, positive and finite".into());
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = quantile_sorted(&sorted, 0.5);
    let mut dev: Vec<f64> = sorted.iter().map(|x| (x - median).abs()).collect();
    dev.sort_by(f64::total_cmp);
    let sample_sd = |xs: &[f64]| {
        if xs.len() < 2 {
            return 0.0;
        }
        let m = xs.iter().sum::<f64>() / xs.len() as f64;
        (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (xs.len() - 1) as f64).sqrt()
    };
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let logs: Vec<f64> = sorted.iter().map(|x| x.ln()).collect();
    let d = Dispersion {
        n,
        median,
        mad: quantile_sorted(&dev, 0.5),
        iqr: quantile_sorted(&sorted, 0.75) - quantile_sorted(&sorted, 0.25),
        p05: quantile_sorted(&sorted, 0.05),
        p95: quantile_sorted(&sorted, 0.95),
        min: sorted[0],
        max: sorted[n - 1],
        cv: sample_sd(&sorted) / mean,
        log_sd: sample_sd(&logs),
    };
    if !(d.cv.is_finite() && d.log_sd.is_finite() && d.median.is_finite()) {
        return Err("nonfinite dispersion".into());
    }
    Ok(d)
}

/// Relative drift per block: the OLS slope of the block median against the
/// block index, divided by the mean block median. 0 for fewer than 2 blocks.
pub fn drift_per_round(block_medians: &[f64]) -> f64 {
    let n = block_medians.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let xm = (nf - 1.0) / 2.0;
    let ym = block_medians.iter().sum::<f64>() / nf;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (i, y) in block_medians.iter().enumerate() {
        let dx = i as f64 - xm;
        sxy += dx * (y - ym);
        sxx += dx * dx;
    }
    (sxy / sxx) / ym
}

/// Why auto-extension stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtendStop {
    Separable,
    MaxRounds,
    TimeBudget,
}

impl ExtendStop {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Separable => "separable",
            Self::MaxRounds => "max_rounds",
            Self::TimeBudget => "time_budget",
        }
    }
}

/// How many more ABBA rounds to run after `done` rounds took `elapsed_s`
/// seconds: `Ok(extra)` doubles the run (`extra == done`) capped by
/// `max_rounds` and by what still fits the budget at the observed per-round
/// time; `Err(why)` stops. `extra` is always at least 1.
pub fn extend_rounds(
    done: usize,
    max_rounds: usize,
    elapsed_s: f64,
    budget_s: f64,
    separable: bool,
) -> std::result::Result<usize, ExtendStop> {
    if separable {
        return Err(ExtendStop::Separable);
    }
    if done >= max_rounds {
        return Err(ExtendStop::MaxRounds);
    }
    let per_round = elapsed_s / done.max(1) as f64;
    let fit = if per_round > 0.0 {
        ((budget_s - elapsed_s) / per_round).floor()
    } else {
        f64::INFINITY
    };
    if fit < 1.0 {
        return Err(ExtendStop::TimeBudget);
    }
    let want = done.min(max_rounds - done);
    Ok(want.min(fit as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ratios_match_block_rule() {
        // n=2: 4 samples per arm per round, two 2-sample blocks per round.
        let base = [10.0, 10.0, 10.0, 20.0, 10.0, 10.0, 10.0, 20.0];
        let cand = [5.0, 5.0, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0];
        let rr = round_ratios(&base, &cand, 2).unwrap();
        let br = block_rule(&base, &cand, 2).unwrap();
        assert_eq!(rr.len(), 2);
        assert_eq!(
            rr.iter().cloned().fold(f64::INFINITY, f64::min),
            br.round_ratio_min
        );
        assert_eq!(
            rr.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            br.round_ratio_max
        );
        assert!(round_ratios(&base, &cand[..4], 2).is_err());
    }
    #[test]
    fn round_ratios_exact_values() {
        let base = [10.0, 10.0, 20.0, 20.0];
        let cand = [5.0, 5.0, 30.0, 30.0];
        // n=1: blocks are single samples; round = (c0 + c1) / (b0 + b1).
        let rr = round_ratios(&base, &cand, 1).unwrap();
        assert_eq!(rr, vec![10.0 / 20.0, 60.0 / 40.0]);
    }
    #[test]
    fn dispersion_known_values() {
        let d = dispersion(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(d.n, 5);
        assert_eq!(d.median, 3.0);
        assert_eq!(d.mad, 1.0);
        assert_eq!(d.iqr, 2.0);
        assert!((d.p05 - 1.2).abs() < 1e-12);
        assert!((d.p95 - 4.8).abs() < 1e-12);
        assert_eq!((d.min, d.max), (1.0, 5.0));
        assert!((d.cv - 2.5f64.sqrt() / 3.0).abs() < 1e-12);
        let logs: Vec<f64> = [1.0f64, 2.0, 3.0, 4.0, 5.0]
            .iter()
            .map(|x| x.ln())
            .collect();
        let m = logs.iter().sum::<f64>() / 5.0;
        let sd = (logs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / 4.0).sqrt();
        assert!((d.log_sd - sd).abs() < 1e-12);
    }
    #[test]
    fn dispersion_single_and_invalid() {
        let d = dispersion(&[2.0]).unwrap();
        assert_eq!(
            (d.median, d.mad, d.iqr, d.cv, d.log_sd),
            (2.0, 0.0, 0.0, 0.0, 0.0)
        );
        assert!(dispersion(&[]).is_err());
        assert!(dispersion(&[1.0, 0.0]).is_err());
        assert!(dispersion(&[1.0, f64::NAN]).is_err());
    }
    #[test]
    fn drift_is_relative_ols_slope() {
        assert_eq!(drift_per_round(&[5.0]), 0.0);
        assert_eq!(drift_per_round(&[7.0, 7.0, 7.0, 7.0]), 0.0);
        // y = 10 + i: slope 1, mean 11.5.
        let d = drift_per_round(&[10.0, 11.0, 12.0, 13.0]);
        assert!((d - 1.0 / 11.5).abs() < 1e-12);
    }

    #[test]
    fn exact_abba_two_rounds() {
        use TimingArm::{Candidate as B, Incumbent as A};
        let got: Vec<_> = timing_schedule(2, 2).unwrap().collect();
        assert_eq!(got, vec![A, A, B, B, B, B, A, A, A, A, B, B, B, B, A, A]);
        assert_eq!(got.iter().filter(|&&v| v == A).count(), 8);
        assert_eq!(got.iter().filter(|&&v| v == B).count(), 8);
    }
    #[test]
    fn identical_not_separable() {
        let s = timing_stats(&[3.0, 3.0]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&s, &s);
        assert_eq!(ratio, 1.0);
        assert!(!separable);
    }
    #[test]
    fn constant_double_separable() {
        let a = timing_stats(&[2.0; 4]).unwrap();
        let b = timing_stats(&[4.0; 4]).unwrap();
        let (ratio, _, _, separable) = timing_relation(&a, &b);
        assert_eq!(ratio, 2.0);
        assert!(separable);
    }
    #[test]
    fn noisy_overlap_not_separable() {
        let a = timing_stats(&[1.0, 5.0, 9.0]).unwrap();
        let b = timing_stats(&[8.0, 10.0, 12.0]).unwrap();
        assert!(!timing_relation(&a, &b).3);
    }
    #[test]
    fn statistics_centers_and_rank() {
        let s = timing_stats(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(
            (s.count, s.median, s.min, s.max, s.p90, s.mean, s.spread()),
            (4, 2.5, 1.0, 4.0, 4.0, 2.5, 3.0)
        );
        assert_eq!(timing_stats(&[3.0, 1.0, 2.0]).unwrap().median, 2.0);
        assert_eq!(
            timing_stats(&(1..=10).map(f64::from).collect::<Vec<_>>())
                .unwrap()
                .p90,
            9.0
        );
    }
    #[test]
    fn bad_samples_and_counts() {
        for values in [&[][..], &[0.0], &[-1.0], &[f64::NAN], &[f64::INFINITY]] {
            assert!(timing_stats(values).is_err());
        }
        assert!(positive_count("0").is_err());
        assert!(positive_count("-1").is_err());
        assert!(timing_counts(usize::MAX, 1).is_err());
        assert!(timing_counts(1, usize::MAX).is_err());
        assert!(timing_schedule(0, 1).is_err());
    }
    #[test]
    fn block_medians_are_per_block() {
        let s = [1.0, 3.0, 2.0, 10.0, 30.0, 20.0];
        assert_eq!(block_medians(&s, 3).unwrap(), vec![2.0, 20.0]);
        assert!(block_medians(&s, 4).is_err());
        assert!(block_medians(&[], 1).is_err());
    }

    /// Samples of `n`-launch blocks, one `[baseline, candidate]` value pair per
    /// block in ABBA order: blocks A B B A of each round give the baseline
    /// blocks 0 and 3 and the candidate blocks 1 and 2.
    fn arms(n: usize, rounds: &[[f64; 4]]) -> (Vec<f64>, Vec<f64>) {
        let (mut base, mut cand) = (Vec::new(), Vec::new());
        for r in rounds {
            base.extend(std::iter::repeat(r[0]).take(n));
            cand.extend(std::iter::repeat(r[1]).take(n));
            cand.extend(std::iter::repeat(r[2]).take(n));
            base.extend(std::iter::repeat(r[3]).take(n));
        }
        (base, cand)
    }

    /// One 7 ms hiccup among ~300 us samples: the legacy full-range rule is
    /// vetoed, the paired-round rule is not.
    #[test]
    fn one_hiccup_does_not_flip_the_paired_rule() {
        let n = 4;
        let mut base: Vec<f64> = (0..8 * 12).map(|i| 300.0 + f64::from(i % 3)).collect();
        let cand: Vec<f64> = (0..8 * 12).map(|i| 290.0 + f64::from(i % 3)).collect();
        base[5] = 7003.0;
        let r = block_rule(&base, &cand, n).unwrap();
        assert!(r.separable, "{r:?}");
        assert_eq!(r.direction, "faster");
        assert_eq!(
            (r.rounds_faster, r.rounds_slower, r.rounds_tied),
            (12, 0, 0)
        );
        assert!(r.sign_p <= PAIRED_P_MAX && r.block_ratio < 1.0 && r.round_ratio_max < 1.0);
        let (a, b) = (timing_stats(&base).unwrap(), timing_stats(&cand).unwrap());
        assert!(!timing_relation(&a, &b).3, "legacy rule sees the hiccup");
        // ... and a hiccup in a candidate block does not flip it either.
        let mut slow = cand.clone();
        slow[2] = 9000.0;
        let r = block_rule(&base, &slow, n).unwrap();
        assert!(r.separable && r.direction == "faster", "{r:?}");
    }

    /// Between-round clock drift (+-8%) larger than the effect (3%): the
    /// block-disjoint rule fails, the within-round pairing sees every round.
    #[test]
    fn drift_defeats_block_disjoint_but_not_the_paired_rule() {
        let rounds: Vec<[f64; 4]> = (0..16)
            .map(|r| {
                let drift = 1.0 + 0.08 * if r % 2 == 0 { 1.0 } else { -1.0 };
                [300.0 * drift, 291.0 * drift, 291.0 * drift, 300.0 * drift]
            })
            .collect();
        let (base, cand) = arms(3, &rounds);
        let r = block_rule(&base, &cand, 3).unwrap();
        assert!(!r.block_disjoint);
        assert!(r.separable && r.direction == "faster", "{r:?}");
        assert!(
            (r.round_ratio_min - 0.97).abs() < 1e-12 && (r.round_ratio_max - 0.97).abs() < 1e-12
        );
    }

    #[test]
    fn slower_candidate_reports_slower() {
        let rounds = vec![[100.0, 120.0, 118.0, 101.0]; 12];
        let (base, cand) = arms(2, &rounds);
        let r = block_rule(&base, &cand, 2).unwrap();
        assert!(
            r.separable && r.direction == "slower" && r.block_disjoint,
            "{r:?}"
        );
        assert_eq!(r.rounds_slower, 12);
    }

    /// Dense-vs-dense control: equal arms tie every round, and arms that win
    /// alternate rounds split the signs; neither is ever separable.
    #[test]
    fn controls_stay_non_separable() {
        let (base, cand) = arms(2, &vec![[100.0, 100.0, 100.0, 100.0]; 64]);
        let r = block_rule(&base, &cand, 2).unwrap();
        assert!(!r.separable && !r.block_disjoint);
        assert_eq!((r.rounds_tied, r.direction, r.sign_p), (64, "tie", 1.0));
        let rounds: Vec<[f64; 4]> = (0..64)
            .map(|r| {
                let c = if r % 2 == 0 { 99.0 } else { 101.0 };
                [100.0, c, c, 100.0]
            })
            .collect();
        let (base, cand) = arms(2, &rounds);
        let r = block_rule(&base, &cand, 2).unwrap();
        assert!(!r.separable, "{r:?}");
        assert_eq!((r.rounds_faster, r.rounds_slower), (32, 32));
        assert_eq!(r.sign_p, 1.0);
    }

    /// Ties count against separability: needs 11 clean rounds, and a tie
    /// among 12 faster-or-tied rounds costs it.
    #[test]
    fn ties_count_against_and_eleven_rounds_is_the_minimum() {
        let faster = [100.0, 99.0, 99.0, 100.0];
        let tie = [100.0; 4];
        for (rounds, want) in [(10, false), (11, true), (12, true)] {
            let (b, c) = arms(1, &vec![faster; rounds]);
            assert_eq!(block_rule(&b, &c, 1).unwrap().separable, want, "{rounds}");
        }
        let mut mixed = vec![faster; 11];
        mixed.push(tie);
        let (b, c) = arms(1, &mixed);
        let r = block_rule(&b, &c, 1).unwrap();
        assert_eq!((r.rounds_faster, r.rounds_tied), (11, 1));
        assert!(!r.separable && r.sign_p > PAIRED_P_MAX, "{r:?}");
    }

    #[test]
    fn block_disjoint_rule_is_kept_and_strict() {
        let n = 2;
        let (base, cand) = arms(n, &[[105.0, 100.0, 100.0, 115.0]]);
        // candidate blocks 100/100 vs baseline blocks 105/115: disjoint.
        assert!(block_rule(&base, &cand, n).unwrap().block_disjoint);
        let (base, cand) = arms(n, &[[105.0, 100.0, 110.0, 115.0]]);
        let r = block_rule(&base, &cand, n).unwrap();
        assert!(!r.block_disjoint);
        assert_eq!(r.baseline_block_medians, vec![105.0, 115.0]);
        assert_eq!(r.candidate_block_medians, vec![100.0, 110.0]);
        assert!((r.block_ratio - 105.0 / 110.0).abs() < 1e-12);
    }

    #[test]
    fn block_rule_rejects_partial_rounds() {
        assert!(block_rule(&[1.0; 6], &[1.0; 6], 2).is_err());
        assert!(block_rule(&[1.0; 8], &[1.0; 4], 2).is_err());
    }

    #[test]
    fn round_ratios_are_paired_per_round() {
        // round 1: cand blocks (5, 5) over base (10, 10) -> 0.5; round 2: 20s -> 2.0.
        let (base, cand) = arms(1, &[[10.0, 5.0, 5.0, 10.0], [10.0, 20.0, 20.0, 10.0]]);
        let r = block_rule(&base, &cand, 1).unwrap();
        assert_eq!((r.round_ratio_min, r.round_ratio_max), (0.5, 2.0));
        assert_eq!((r.rounds_faster, r.rounds_slower), (1, 1));
        assert!(!r.separable);
    }

    #[test]
    fn majority_sign_p_matches_exact_binomial() {
        assert_eq!(majority_sign_p(0, 0), 1.0);
        assert_eq!(majority_sign_p(3, 6), 1.0);
        // 6/6: 2 * (1/64).
        assert!((majority_sign_p(6, 6) - 2.0 / 64.0).abs() < 1e-15);
        // 5/6: 2 * (1 + 6) / 64.
        assert!((majority_sign_p(5, 6) - 14.0 / 64.0).abs() < 1e-15);
        assert!(majority_sign_p(64, 64) < 1e-18);
        assert!(majority_sign_p(34, 64) > 0.5);
        // A majority below n/2 (ties dominating) is never evidence.
        assert_eq!(majority_sign_p(2, 10), 1.0);
        assert!(majority_sign_p(11, 11) <= PAIRED_P_MAX && majority_sign_p(10, 10) > PAIRED_P_MAX);
    }

    #[test]
    fn extension_doubles_until_a_stop() {
        use ExtendStop::*;
        assert_eq!(extend_rounds(6, 64, 10.0, 240.0, true), Err(Separable));
        assert_eq!(extend_rounds(6, 64, 10.0, 240.0, false), Ok(6));
        assert_eq!(extend_rounds(12, 64, 20.0, 240.0, false), Ok(12));
        // Capped by max_rounds.
        assert_eq!(extend_rounds(48, 64, 80.0, 240.0, false), Ok(16));
        assert_eq!(extend_rounds(64, 64, 80.0, 240.0, false), Err(MaxRounds));
        // Capped by what fits the budget at the observed per-round time (2 s/round).
        assert_eq!(extend_rounds(6, 64, 12.0, 20.0, false), Ok(4));
        assert_eq!(extend_rounds(6, 64, 12.0, 13.0, false), Err(TimeBudget));
        assert_eq!(extend_rounds(6, 64, 300.0, 240.0, false), Err(TimeBudget));
    }

    #[test]
    fn extension_schedule_terminates_within_bounds() {
        // Simulate a never-separable run: rounds 6 -> 12 -> 24 -> 48 -> 64.
        let (mut done, mut t) = (6usize, 6.0f64);
        let mut steps = vec![done];
        while let Ok(extra) = extend_rounds(done, 64, t, 1e9, false) {
            done += extra;
            t += extra as f64;
            steps.push(done);
        }
        assert_eq!(steps, vec![6, 12, 24, 48, 64]);
    }
}
