// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! R9 token-level KLD statistic.
//!
//! The R7 gate judged a Student-t interval over the per-chunk mean-KLD
//! differences. On the 27B WT2 reference (24 chunks x 1023 scored tokens) the
//! paired per-chunk sd is 0.00405, so 24 chunks cannot resolve a 0.0005 margin
//! (see [`kld_power`]). This module reads the evaluator's HFKTOK v1 per-token
//! output and judges a stratified chunk block bootstrap plus a normal-theory
//! bound over the per-token paired differences, and prints the sample-size
//! arithmetic that says how many chunks a margin needs.
//!
//! * [`parse_hfktok`] / [`read_hfktok`]: HFKTOK v1 (`HFKTOK\0\0`, u32 version=1,
//!   u32 n_chunk, u32 scored_per_chunk, u32 reserved=0, then `n_chunk *
//!   scored_per_chunk` little-endian f64 per-token KLDs, chunk major).
//! * [`build_token_evidence`]: strata (one incumbent/candidate pair per
//!   reference corpus) -> [`KldTokenEvidenceV2`]; every file is re-read and
//!   hash-verified, shapes and finiteness are checked and each chunk's mean is
//!   recomputed from the per-token values and compared bit for bit with the
//!   kldseq `mean_kld`.
//! * [`judge_bounds`]: the numeric gate (both upper bounds below the margin,
//!   every stratum at least [`STRATUM_MIN_CHUNKS`] chunks).
//! * [`kld_power`], [`norm_quantile`]: `n_required` and the minimum resolvable
//!   margin.
//!
//! # The statistic (`pm-census tune kld --stat token-bootstrap`)
//!
//! Input: one or more strata, each an incumbent/candidate `eval_hipfire
//! --per-token-output` pair on one reference corpus (`--stratum
//! LABEL=INC_RUN.json,CAND_RUN.json`, repeated; the run JSON carries
//! `ktok_path`/`ktok_sha256` next to `kldseq_path`/`kldseq_sha256`). Per token
//! `d = candidate - incumbent`.
//!
//! * point = token-weighted mean of `d` over every stratum.
//! * bootstrap bound: stratified chunk block bootstrap, `B = 20000` replicates
//!   from a SplitMix64 stream (seed recorded in the evidence); chunks resample
//!   with replacement inside each stratum, a chunk carrying all its tokens
//!   (tokens inside a chunk share context and may be correlated, so the chunk,
//!   not the token, is the unit). The one-sided 95% upper bound is the
//!   `ceil(0.95 B)`-th smallest replicate.
//! * normal bound: `point + 1.645 * SE`, with the stratified chunk-level
//!   `SE^2 = sum_s w_s^2 sd_chunk_s^2 / n_s`, `w_s` = stratum token share.
//! * diagnostics: per-token sd, per-chunk sd, design effect
//!   `deff = sd_chunk^2 * tokens_per_chunk / sd_token^2` (1 = tokens behave as
//!   independent, >> 1 = a per-token interval would be far too narrow), the share
//!   of `sum |d|` carried by the top 1% `|d|` tokens, `exact` (every `d` is 0).
//!
//! Gate (stage 6 of `tune promote --kld-token EV.json`; `kld_token` wins over
//! the R7 `kld_state` per-chunk evidence when both are attached): promotion
//! grade, the R7 run-consistency rules for every stratum (plus: same model,
//! evaluator, KV flags and selected code object across strata, distinct
//! references), at least [`STRATUM_MIN_CHUNKS`] chunks in every stratum, and
//! BOTH upper bounds strictly below `kld_margin` (0.0005). Refusal is the
//! default and reports the numbers. An all-zero evidence passes as in R7.
//!
//! Refused at evidence time: a hash that differs from the run JSON, a shape
//! mismatch between the two runs, a non-finite value, and a chunk whose mean
//! recomputed from its HFKTOK values (the evaluator's sequential f64 sum / count)
//! differs bitwise from the kldseq `mean_kld`.
//!
//! # Power (`pm-census tune kld-power`)
//!
//! A one-sided test of "mean diff < M" at level `alpha` with power `1 - beta`
//! against a true mean difference `delta` (default 0) needs
//! `n = ceil(((z_{1-alpha} + z_{power}) * SD / (M - delta))^2)` chunks, where
//! `SD` is the per-chunk sd of the paired differences. With `N` chunks the
//! smallest margin resolvable at `delta = 0` is
//! `MDM = (z_{1-alpha} + z_{power}) * SD / sqrt(N)`. `tune kld` prints the same
//! calculation at the measured stratified chunk sd (`sd_chunk` of the stat).
//!
//! Worked example (R8, 27B WT2, 24 chunks x 1023 tokens, paired per-chunk sd
//! 0.00405, margin 0.0005, `alpha = 0.05`, power 0.8, so
//! `z_{0.95} + z_{0.8} = 1.6449 + 0.8416 = 2.4865`):
//!
//! * `n_required = ceil((2.4865 * 0.00405 / 0.0005)^2) = ceil(405.6) = 406`
//!   chunks (415,338 scored tokens);
//! * `MDM(24) = 2.4865 * 0.00405 / sqrt(24) = 0.002056`: four times the margin;
//! * `MDM(48) = 0.001454` (WT2 + AG together): still nearly three times the margin;
//! * even at the measured point (-0.000337), the normal bound
//!   `point + 1.645 * 0.00405 / sqrt(N)` is 0.00102 at N = 24 and 0.00062 at
//!   N = 48, both above 0.0005, so the gate keeps refusing until either the
//!   measured `sd_chunk` of the token statistic is smaller or more chunks exist.
//!   Counting on the true gain (`--delta -0.000337`) the need is 145 chunks.
//!
//! # Measured (R9, 27B, `fold_op muladd` trial vs the shipped object)
//!
//! | stratum | point | sd_token | sd_chunk | deff | top-1% share of sum abs(d) | boot / normal hi95 | n_required |
//! |---|---|---|---|---|---|---|---|
//! | WT2 24x1023 | -0.000337 | 0.1448 | 0.00405 | 0.80 | 26% | 0.00104 / 0.00102 | 406 |
//! | AG 24x1023 | +0.00819 | 0.949 | 0.0272 | 0.84 | 59% | 0.0173 / 0.0173 | 18,267 |
//! | WT2 + AG | +0.00393 | 0.679 | 0.0194 | - | 58% | 0.00846 / 0.00854 | 9,342 |
//!
//! `deff ~ 0.8`: tokens inside a chunk are close to independent, so the chunk
//! bootstrap and the per-chunk t interval carry the same information and no
//! blocking scheme shrinks the interval of the MEAN; only more chunks do. The
//! per-token differences are heavy-tailed (1% of the tokens carry a quarter to
//! more than half of `sum |d|`): a trimmed or median statistic is 3-100x tighter
//! (5%-trimmed WT2 se 0.00026, 39 chunks for the margin) but it does not estimate
//! the mean KLD and on AG it would have hidden the tail-driven +0.0082 shift
//! (5%-trimmed +0.00026). The gate therefore keeps the mean as its estimand.

use super::kld::{parse_kldseq, read_verified, sha256_hex};
use super::{
    KldRunV1, KldStratumV2, KldTokenEvidenceV2, TokenBootstrapV2, KLD_TOKEN_SCHEMA, KLD_TOKEN_STAT,
};
use serde::Serialize;
use std::collections::BTreeSet;

const MAGIC: &[u8; 8] = b"HFKTOK\0\0";
const HEADER: usize = 24;

/// Bootstrap replicates of the gate statistic.
pub const BOOT_B: usize = 20_000;
/// Default SplitMix64 seed of the bootstrap stream (recorded in the evidence).
pub const DEFAULT_SEED: u64 = 0x4B4C_4454_4F4B_0001;
/// Every stratum needs at least this many chunks for the gate.
pub const STRATUM_MIN_CHUNKS: usize = 20;
/// One-sided 95% normal quantile used by the normal-theory bound.
pub const Z95: f64 = 1.645;

/// Parsed HFKTOK v1: `values` is chunk major, `scored_per_chunk` per chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct Ktok {
    pub n_chunk: usize,
    pub scored_per_chunk: usize,
    pub values: Vec<f64>,
}

impl Ktok {
    pub fn chunk(&self, c: usize) -> &[f64] {
        &self.values[c * self.scored_per_chunk..(c + 1) * self.scored_per_chunk]
    }
}

pub fn parse_hfktok(b: &[u8]) -> Result<Ktok, String> {
    if b.len() < HEADER {
        return Err(format!("hfktok too short ({} bytes)", b.len()));
    }
    if &b[..8] != MAGIC {
        return Err("hfktok bad magic (want HFKTOK\\0\\0)".to_owned());
    }
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
    let version = u32_at(8);
    if version != 1 {
        return Err(format!("hfktok version {version}, want 1"));
    }
    let n_chunk = u32_at(12) as usize;
    let scored_per_chunk = u32_at(16) as usize;
    if u32_at(20) != 0 {
        return Err(format!("hfktok reserved {} != 0", u32_at(20)));
    }
    let want = n_chunk
        .checked_mul(scored_per_chunk)
        .and_then(|x| x.checked_mul(8))
        .and_then(|x| x.checked_add(HEADER))
        .ok_or("hfktok n_chunk * scored_per_chunk overflows")?;
    if b.len() != want {
        return Err(format!(
            "hfktok length {} != {want} for n_chunk {n_chunk} x scored_per_chunk {scored_per_chunk}",
            b.len()
        ));
    }
    let values = b[HEADER..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
        .collect();
    Ok(Ktok {
        n_chunk,
        scored_per_chunk,
        values,
    })
}

pub fn read_hfktok(path: &std::path::Path) -> Result<Ktok, String> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_hfktok(&b).map_err(|e| format!("{}: {e}", path.display()))
}

/// Mean of one chunk's per-token KLDs in the evaluator's exact order
/// (`hipfire_runtime::eval_common::chunk_mean_kld`).
pub fn chunk_mean(chunk: &[f64]) -> f64 {
    chunk.iter().copied().sum::<f64>() / chunk.len() as f64
}

/// SplitMix64 (Steele/Lea/Flood); the bootstrap's only randomness.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform index in `0..n` (multiply-shift, no modulo bias worth speaking of).
    pub fn below(&mut self, n: usize) -> usize {
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

/// Per-token paired differences `d = candidate - incumbent` of one stratum.
#[derive(Clone, Debug)]
pub struct StratumDiffs {
    pub label: String,
    pub n_chunks: usize,
    pub tokens_per_chunk: usize,
    /// Chunk major, `n_chunks * tokens_per_chunk`.
    pub d: Vec<f64>,
}

struct StratumStats {
    mean_diff: f64,
    sd_token: f64,
    sd_chunk: f64,
    deff: f64,
    top1pct_abs_share: f64,
    exact: bool,
    chunk_sums: Vec<f64>,
}

fn sample_sd(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// Share of `sum |d|` carried by the `ceil(1%)` largest `|d|` (0 when all are 0).
fn top1pct_abs_share(d: &[f64]) -> f64 {
    let mut a: Vec<f64> = d.iter().map(|x| x.abs()).collect();
    a.sort_by(|x, y| y.total_cmp(x));
    let total: f64 = a.iter().sum();
    if total == 0.0 {
        return 0.0;
    }
    let k = a.len().div_ceil(100).max(1);
    a[..k].iter().sum::<f64>() / total
}

fn stratum_stats(s: &StratumDiffs) -> Result<StratumStats, String> {
    if s.n_chunks < 2 {
        return Err(format!(
            "stratum {}: need at least 2 chunks, got {}",
            s.label, s.n_chunks
        ));
    }
    if s.tokens_per_chunk < 2 {
        return Err(format!(
            "stratum {}: need at least 2 scored tokens per chunk, got {}",
            s.label, s.tokens_per_chunk
        ));
    }
    if s.d.len() != s.n_chunks * s.tokens_per_chunk {
        return Err(format!(
            "stratum {}: diff count != chunks x tokens",
            s.label
        ));
    }
    if s.d.iter().any(|x| !x.is_finite()) {
        return Err(format!(
            "stratum {}: non-finite per-token difference",
            s.label
        ));
    }
    let t = s.tokens_per_chunk as f64;
    let chunk_sums: Vec<f64> =
        s.d.chunks_exact(s.tokens_per_chunk)
            .map(|c| c.iter().sum::<f64>())
            .collect();
    let chunk_means: Vec<f64> = chunk_sums.iter().map(|x| x / t).collect();
    let sd_token = sample_sd(&s.d);
    let sd_chunk = sample_sd(&chunk_means);
    let deff = if sd_token > 0.0 {
        sd_chunk * sd_chunk * t / (sd_token * sd_token)
    } else {
        0.0
    };
    Ok(StratumStats {
        mean_diff: chunk_sums.iter().sum::<f64>() / s.d.len() as f64,
        sd_token,
        sd_chunk,
        deff,
        top1pct_abs_share: top1pct_abs_share(&s.d),
        exact: s.d.iter().all(|x| *x == 0.0),
        chunk_sums,
    })
}

/// The pooled statistic and the per-stratum diagnostics of already-validated
/// per-token differences. Deterministic for a given `(b, seed)`.
pub fn analyze(
    strata: &[StratumDiffs],
    b: usize,
    seed: u64,
) -> Result<(TokenBootstrapV2, Vec<KldStratumDiag>), String> {
    if strata.is_empty() {
        return Err("no strata".to_owned());
    }
    if b == 0 {
        return Err("bootstrap replicate count is 0".to_owned());
    }
    let stats: Vec<StratumStats> = strata.iter().map(stratum_stats).collect::<Result<_, _>>()?;
    let n_tokens: usize = strata.iter().map(|s| s.d.len()).sum();
    let n_chunks: usize = strata.iter().map(|s| s.n_chunks).sum();
    let nt = n_tokens as f64;
    let total: f64 = stats.iter().map(|s| s.chunk_sums.iter().sum::<f64>()).sum();
    let point = total / nt;

    // Stratified chunk-level SE of the token-weighted point estimate:
    // sum_s w_s^2 * sd_chunk_s^2 / n_s with w_s = tokens_s / tokens.
    let var: f64 = strata
        .iter()
        .zip(&stats)
        .map(|(s, st)| {
            let w = s.d.len() as f64 / nt;
            w * w * st.sd_chunk * st.sd_chunk / s.n_chunks as f64
        })
        .sum();
    let se = var.sqrt();

    // Stratified chunk block bootstrap: chunks (all their tokens at once) are
    // resampled with replacement inside each stratum; every replicate keeps
    // each stratum's chunk count, so its token total stays `n_tokens`.
    let mut rng = SplitMix64::new(seed);
    let mut reps = Vec::with_capacity(b);
    for _ in 0..b {
        let mut sum = 0.0f64;
        for st in &stats {
            let n = st.chunk_sums.len();
            for _ in 0..n {
                sum += st.chunk_sums[rng.below(n)];
            }
        }
        reps.push(sum / nt);
    }
    reps.sort_by(f64::total_cmp);
    // One-sided 95% percentile: the ceil(0.95 * B)-th smallest replicate.
    let idx = (95 * b).div_ceil(100) - 1;
    let boot_hi95 = reps[idx];

    let all_d: Vec<f64> = strata.iter().flat_map(|s| s.d.iter().copied()).collect();
    let stat = TokenBootstrapV2 {
        stat: KLD_TOKEN_STAT.to_owned(),
        n_strata: strata.len(),
        n_chunks,
        n_tokens,
        point,
        se,
        boot_hi95,
        normal_hi95: point + Z95 * se,
        boot_b: b,
        seed,
        sd_token: sample_sd(&all_d),
        sd_chunk: se * (n_chunks as f64).sqrt(),
        top1pct_abs_share: top1pct_abs_share(&all_d),
        exact: stats.iter().all(|s| s.exact),
    };
    let diags = strata
        .iter()
        .zip(stats)
        .map(|(s, st)| KldStratumDiag {
            label: s.label.clone(),
            n_chunks: s.n_chunks,
            tokens_per_chunk: s.tokens_per_chunk,
            mean_diff: st.mean_diff,
            sd_token: st.sd_token,
            sd_chunk: st.sd_chunk,
            deff: st.deff,
            top1pct_abs_share: st.top1pct_abs_share,
            exact: st.exact,
        })
        .collect();
    Ok((stat, diags))
}

/// Per-stratum diagnostics (the run-free half of [`KldStratumV2`]).
#[derive(Clone, Debug, PartialEq)]
pub struct KldStratumDiag {
    pub label: String,
    pub n_chunks: usize,
    pub tokens_per_chunk: usize,
    pub mean_diff: f64,
    pub sd_token: f64,
    pub sd_chunk: f64,
    pub deff: f64,
    pub top1pct_abs_share: f64,
    pub exact: bool,
}

fn read_verified_ktok(run: &KldRunV1) -> Result<Vec<u8>, String> {
    let (Some(path), Some(sha)) = (&run.ktok_path, &run.ktok_sha256) else {
        return Err(format!(
            "{} run has no ktok_path/ktok_sha256 (run eval_hipfire with --per-token-output)",
            run.role
        ));
    };
    let b = std::fs::read(path).map_err(|e| format!("read {path} ({}): {e}", run.role))?;
    let got = sha256_hex(&b);
    if &got != sha {
        return Err(format!("{} ktok sha256 {got} != recorded {sha}", run.role));
    }
    Ok(b)
}

/// One run's per-token KLDs, hash-verified and checked against its kldseq:
/// same chunk count, finite, and each chunk's recomputed mean equal to the
/// kldseq `mean_kld` bit for bit.
fn load_run(run: &KldRunV1) -> Result<Ktok, String> {
    let seq = parse_kldseq(&read_verified(run)?).map_err(|e| format!("{}: {e}", run.role))?;
    let kt = parse_hfktok(&read_verified_ktok(run)?).map_err(|e| format!("{}: {e}", run.role))?;
    if kt.n_chunk != seq.len() {
        return Err(format!(
            "{}: ktok has {} chunks, kldseq has {}",
            run.role,
            kt.n_chunk,
            seq.len()
        ));
    }
    if kt.values.iter().any(|x| !x.is_finite()) {
        return Err(format!("{}: non-finite per-token KLD in ktok", run.role));
    }
    if kt.scored_per_chunk == 0 {
        return Err(format!("{}: ktok scored_per_chunk is 0", run.role));
    }
    for (c, s) in seq.iter().enumerate() {
        let m = chunk_mean(kt.chunk(c));
        if m.to_bits() != s.0.to_bits() {
            return Err(format!(
                "{}: chunk {c} mean recomputed from ktok {m:e} != kldseq mean_kld {:e} (bit mismatch)",
                run.role, s.0
            ));
        }
    }
    Ok(kt)
}

/// `d = candidate - incumbent` per token for one run pair; refuses any shape
/// mismatch.
fn pair_diffs(label: &str, inc: &Ktok, cand: &Ktok) -> Result<StratumDiffs, String> {
    if inc.n_chunk != cand.n_chunk || inc.scored_per_chunk != cand.scored_per_chunk {
        return Err(format!(
            "stratum {label}: shape mismatch incumbent {}x{} vs candidate {}x{} (chunks x tokens)",
            inc.n_chunk, inc.scored_per_chunk, cand.n_chunk, cand.scored_per_chunk
        ));
    }
    Ok(StratumDiffs {
        label: label.to_owned(),
        n_chunks: inc.n_chunk,
        tokens_per_chunk: inc.scored_per_chunk,
        d: cand
            .values
            .iter()
            .zip(&inc.values)
            .map(|(c, i)| c - i)
            .collect(),
    })
}

/// Evidence over `strata` = `(label, incumbent run, candidate run)`, one per
/// reference corpus. Strata must use distinct references (a repeated reference
/// would count the same tokens twice).
pub fn build_token_evidence(
    grade: &str,
    strata: Vec<(String, KldRunV1, KldRunV1)>,
    b: usize,
    seed: u64,
) -> Result<KldTokenEvidenceV2, String> {
    if grade != "screen" && grade != "promotion" {
        return Err(format!("grade {grade:?}, want screen|promotion"));
    }
    if strata.is_empty() {
        return Err("at least one stratum is required".to_owned());
    }
    let mut labels = BTreeSet::new();
    let mut refs = BTreeSet::new();
    for (label, inc, cand) in &strata {
        if label.is_empty() || !labels.insert(label.clone()) {
            return Err(format!("stratum label {label:?} is empty or repeated"));
        }
        for r in [inc, cand] {
            if r.reference_sha256 != inc.reference_sha256 {
                return Err(format!(
                    "stratum {label}: incumbent and candidate use different references"
                ));
            }
        }
        if !refs.insert(inc.reference_sha256.clone()) {
            return Err(format!(
                "stratum {label}: reference_sha256 repeats another stratum's"
            ));
        }
    }
    let mut diffs = Vec::new();
    for (label, inc, cand) in &strata {
        let ki = load_run(inc).map_err(|e| format!("stratum {label}: {e}"))?;
        let kc = load_run(cand).map_err(|e| format!("stratum {label}: {e}"))?;
        diffs.push(pair_diffs(label, &ki, &kc)?);
    }
    let (stat, diags) = analyze(&diffs, b, seed)?;
    let strata = strata
        .into_iter()
        .zip(diags)
        .map(|((label, incumbent, candidate), d)| KldStratumV2 {
            label,
            incumbent,
            candidate,
            n_chunks: d.n_chunks,
            tokens_per_chunk: d.tokens_per_chunk,
            mean_diff: d.mean_diff,
            sd_token: d.sd_token,
            sd_chunk: d.sd_chunk,
            deff: d.deff,
            top1pct_abs_share: d.top1pct_abs_share,
            exact: d.exact,
        })
        .collect();
    Ok(KldTokenEvidenceV2 {
        schema: KLD_TOKEN_SCHEMA.to_owned(),
        grade: grade.to_owned(),
        strata,
        stat,
    })
}

/// The numeric gate over a token-bootstrap evidence: every stratum has at least
/// [`STRATUM_MIN_CHUNKS`] chunks and both one-sided 95% upper bounds are below
/// `margin`. An all-zero evidence (`exact`) satisfies the bounds trivially (both
/// are 0) and still needs the chunk floor. The refusal carries the numbers.
pub fn judge_bounds(ev: &KldTokenEvidenceV2, margin: f64) -> Result<(), String> {
    if ev.schema != KLD_TOKEN_SCHEMA {
        return Err(format!("schema {:?}, want {KLD_TOKEN_SCHEMA}", ev.schema));
    }
    let s = &ev.stat;
    if s.stat != KLD_TOKEN_STAT {
        return Err(format!("stat {:?}, want {KLD_TOKEN_STAT}", s.stat));
    }
    if ev.strata.is_empty() || s.n_strata != ev.strata.len() {
        return Err(format!(
            "stat.n_strata {} != {} strata",
            s.n_strata,
            ev.strata.len()
        ));
    }
    if s.n_chunks != ev.strata.iter().map(|x| x.n_chunks).sum::<usize>() {
        return Err("stat.n_chunks != sum of stratum n_chunks".to_owned());
    }
    if s.boot_b < BOOT_B {
        return Err(format!("bootstrap replicates {} < {BOOT_B}", s.boot_b));
    }
    for st in &ev.strata {
        if st.n_chunks < STRATUM_MIN_CHUNKS {
            return Err(format!(
                "stratum {} has {} chunks < {STRATUM_MIN_CHUNKS} (per-stratum floor)",
                st.label, st.n_chunks
            ));
        }
    }
    let numbers = || {
        let need = kld_power(s.sd_chunk, margin, 0.05, 0.8, 0.0, 1, None)
            .map(|p| p.n_required.to_string())
            .unwrap_or_else(|_| "?".to_owned());
        format!(
            "point {:e}, boot_hi95 {:e}, normal_hi95 {:e}, se {:e}, sd_chunk {:e}, n_chunks {} \
             (about {need} chunks resolve this margin at 95% confidence / 80% power)",
            s.point, s.boot_hi95, s.normal_hi95, s.se, s.sd_chunk, s.n_chunks
        )
    };
    if !(s.boot_hi95.is_finite() && s.normal_hi95.is_finite()) {
        return Err(format!("non-finite upper bound: {}", numbers()));
    }
    if !(s.boot_hi95 < margin && s.normal_hi95 < margin) {
        return Err(format!(
            "token-bootstrap upper bounds not both below kld_margin {margin}: {}",
            numbers()
        ));
    }
    Ok(())
}

// ----- power -----

fn erf(x: f64) -> f64 {
    // Maclaurin series; |x| <= ~3.2 keeps the cancellation below 1e-11.
    let mut term = x;
    let mut sum = x;
    let x2 = x * x;
    for n in 1..200 {
        term *= -x2 / n as f64;
        let add = term / (2 * n + 1) as f64;
        sum += add;
        if add.abs() < 1e-18 * sum.abs() {
            break;
        }
    }
    2.0 / std::f64::consts::PI.sqrt() * sum
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Standard normal quantile `z_p` for `p` in `[1e-4, 1 - 1e-4]` (bisection on
/// the series CDF; accurate to ~1e-10).
pub fn norm_quantile(p: f64) -> Result<f64, String> {
    if !(1e-4..=1.0 - 1e-4).contains(&p) {
        return Err(format!("probability {p} outside [1e-4, 1-1e-4]"));
    }
    let (mut lo, mut hi) = (-4.5f64, 4.5f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if norm_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Sample-size arithmetic for a one-sided test of "mean diff < margin".
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PowerCalc {
    /// Per-chunk sd of the paired differences.
    pub sd: f64,
    pub margin: f64,
    pub alpha: f64,
    pub power: f64,
    /// Assumed true mean difference.
    pub delta: f64,
    pub z_alpha: f64,
    pub z_power: f64,
    /// `ceil(((z_{1-alpha} + z_power) * sd / (margin - delta))^2)` chunks.
    pub n_required: usize,
    pub tokens_per_chunk: usize,
    pub tokens_required: usize,
    pub available: Option<usize>,
    /// `(z_{1-alpha} + z_power) * sd / sqrt(available)`: the smallest margin
    /// resolvable (at `delta` = 0) with the available chunks.
    pub min_resolvable_margin: Option<f64>,
}

pub fn kld_power(
    sd: f64,
    margin: f64,
    alpha: f64,
    power: f64,
    delta: f64,
    tokens_per_chunk: usize,
    available: Option<usize>,
) -> Result<PowerCalc, String> {
    if !(sd.is_finite() && sd >= 0.0) {
        return Err(format!("--sd {sd} must be finite and >= 0"));
    }
    if !(margin.is_finite() && margin > 0.0) {
        return Err(format!("--margin {margin} must be finite and > 0"));
    }
    if !delta.is_finite() || delta >= margin {
        return Err(format!(
            "--delta {delta} must be finite and below --margin {margin}"
        ));
    }
    if tokens_per_chunk == 0 {
        return Err("--tokens-per-chunk must be > 0".to_owned());
    }
    if available == Some(0) {
        return Err("--available must be > 0".to_owned());
    }
    let z_alpha = norm_quantile(1.0 - alpha).map_err(|e| format!("--alpha: {e}"))?;
    let z_power = norm_quantile(power).map_err(|e| format!("--power: {e}"))?;
    let zs = z_alpha + z_power;
    let n = (zs * sd / (margin - delta)).powi(2).ceil();
    if !n.is_finite() || n > 1e12 {
        return Err("n_required is not representable".to_owned());
    }
    let n_required = n as usize;
    Ok(PowerCalc {
        sd,
        margin,
        alpha,
        power,
        delta,
        z_alpha,
        z_power,
        n_required,
        tokens_per_chunk,
        tokens_required: n_required.saturating_mul(tokens_per_chunk),
        available,
        min_resolvable_margin: available.map(|a| zs * sd / (a as f64).sqrt()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    pub(crate) fn hfktok_bytes(n_chunk: usize, scored: usize, vals: &[f64]) -> Vec<u8> {
        assert_eq!(vals.len(), n_chunk * scored);
        let mut v = Vec::new();
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&(n_chunk as u32).to_le_bytes());
        v.extend_from_slice(&(scored as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        for x in vals {
            v.extend_from_slice(&x.to_le_bytes());
        }
        v
    }

    /// kldseq v2 whose mean is recomputed from `vals` exactly as the evaluator does.
    pub(crate) fn kldseq_from_tokens(n_chunk: usize, scored: usize, vals: &[f64]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"HFKSEQ\0\0");
        v.extend_from_slice(&2u32.to_le_bytes());
        v.extend_from_slice(&(n_chunk as u32).to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        for c in 0..n_chunk {
            let m = chunk_mean(&vals[c * scored..(c + 1) * scored]);
            v.extend_from_slice(&m.to_le_bytes());
            v.extend_from_slice(&(m * 2.0).to_le_bytes());
            v.extend_from_slice(&1.5f64.to_le_bytes());
        }
        v
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "pm-census-ktok-{}-{}-{:?}",
            name,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub(crate) fn run_with_ktok(
        role: &str,
        dir: &Path,
        refsha: &str,
        n: usize,
        t: usize,
        vals: &[f64],
    ) -> KldRunV1 {
        let seq = kldseq_from_tokens(n, t, vals);
        let tok = hfktok_bytes(n, t, vals);
        let sp = dir.join(format!("{role}.kldseq"));
        let tp = dir.join(format!("{role}.ktok"));
        std::fs::write(&sp, &seq).unwrap();
        std::fs::write(&tp, &tok).unwrap();
        KldRunV1 {
            role: role.into(),
            model_path: "m.hfq".into(),
            model_sha256: "m".repeat(64),
            reference_path: "r.bin".into(),
            reference_sha256: refsha.repeat(64),
            evaluator_path: "eval".into(),
            evaluator_sha256: "e".repeat(64),
            hipfire_commit: "c".repeat(40),
            scoring_mode: "prefill".into(),
            max_chunks: n,
            kv_mode: "q8".into(),
            kv_v: "q8".into(),
            env: BTreeMap::from([
                ("HIPFIRE_GRAPH".to_owned(), "0".to_owned()),
                ("HIPFIRE_NORMALIZE_PROMPT".to_owned(), "0".to_owned()),
            ]),
            fresh_home: true,
            tune_table_path: None,
            tune_table_sha256: None,
            selected_code_object_sha256: None,
            kldseq_path: sp.display().to_string(),
            kldseq_sha256: sha256_hex(&seq),
            ktok_path: Some(tp.display().to_string()),
            ktok_sha256: Some(sha256_hex(&tok)),
            device_uuid: "u".into(),
            fabric_job: None,
            slice_mean_kld: 0.01,
        }
    }

    /// Deterministic synthetic per-token KLDs (>= 0) with a per-chunk shared
    /// component so chunk means vary.
    pub(crate) fn synth(seed: u64, n: usize, t: usize) -> Vec<f64> {
        let mut r = SplitMix64::new(seed);
        let mut u = move || (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let mut v = Vec::with_capacity(n * t);
        for _ in 0..n {
            let shared = u() * 0.02;
            for _ in 0..t {
                v.push(0.05 + shared + u() * u() * 0.1);
            }
        }
        v
    }

    #[test]
    fn splitmix64_reference_vector() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        let mut r = SplitMix64::new(DEFAULT_SEED);
        let mut r2 = SplitMix64::new(DEFAULT_SEED);
        assert_eq!(r.next_u64(), r2.next_u64());
        let mut counts = [0usize; 7];
        for _ in 0..70_000 {
            counts[r.below(7)] += 1;
        }
        assert!(
            counts.iter().all(|c| (9_000..11_000).contains(c)),
            "{counts:?}"
        );
    }

    #[test]
    fn hfktok_parse_round_trip_and_corruption() {
        let vals: Vec<f64> = (0..6).map(|i| i as f64 * 0.25).collect();
        let b = hfktok_bytes(2, 3, &vals);
        let k = parse_hfktok(&b).unwrap();
        assert_eq!((k.n_chunk, k.scored_per_chunk), (2, 3));
        assert_eq!(k.values, vals);
        assert_eq!(k.chunk(1), &vals[3..]);
        let d = tmp("rt");
        std::fs::write(d.join("a.ktok"), &b).unwrap();
        assert_eq!(read_hfktok(&d.join("a.ktok")).unwrap(), k);
        assert!(read_hfktok(&d.join("missing")).is_err());
        let mut bad = b.clone();
        bad[0] = b'X';
        assert!(parse_hfktok(&bad).unwrap_err().contains("magic"));
        let mut bad = b.clone();
        bad[8] = 2;
        assert!(parse_hfktok(&bad).unwrap_err().contains("version"));
        let mut bad = b.clone();
        bad[20] = 1;
        assert!(parse_hfktok(&bad).unwrap_err().contains("reserved"));
        assert!(parse_hfktok(&b[..b.len() - 1])
            .unwrap_err()
            .contains("length"));
        assert!(parse_hfktok(&b[..10]).unwrap_err().contains("short"));
        let mut huge = b.clone();
        huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        huge[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_hfktok(&huge).is_err());
    }

    #[test]
    fn hand_computed_single_stratum() {
        // chunk sums 4,4,4,6 over T=2 -> point 2.25, chunk means 2,2,2,3 (sd 0.5),
        // se = 0.5/sqrt(4) = 0.25, token sd sqrt(19.5/7), top 1% = |5|/18.
        let s = StratumDiffs {
            label: "a".into(),
            n_chunks: 4,
            tokens_per_chunk: 2,
            d: vec![1.0, 3.0, 2.0, 2.0, 0.0, 4.0, 5.0, 1.0],
        };
        let (st, dg) = analyze(&[s], 2000, 7).unwrap();
        assert!((st.point - 2.25).abs() < 1e-12);
        assert!((st.se - 0.25).abs() < 1e-12);
        assert!((st.normal_hi95 - (2.25 + 1.645 * 0.25)).abs() < 1e-12);
        assert!((st.sd_chunk - 0.5).abs() < 1e-12);
        assert!((st.sd_token - (19.5f64 / 7.0).sqrt()).abs() < 1e-12);
        assert!((st.top1pct_abs_share - 5.0 / 18.0).abs() < 1e-12);
        assert!(!st.exact);
        assert_eq!(
            (st.n_chunks, st.n_tokens, st.boot_b, st.seed),
            (4, 8, 2000, 7)
        );
        let d = &dg[0];
        assert!((d.deff - 0.25 * 2.0 / (19.5 / 7.0)).abs() < 1e-12);
        // 4 chunks: the replicate stat is one of a few discrete values; its 95th
        // percentile lies within the data range.
        assert!(st.boot_hi95 >= 2.0 && st.boot_hi95 <= 3.0);
    }

    #[test]
    fn bootstrap_is_deterministic_stratified_and_tracks_the_normal_bound() {
        let mk = |label: &str, seed: u64, n: usize| {
            let a = synth(seed, n, 64);
            let b: Vec<f64> = synth(seed + 1, n, 64);
            // candidate = incumbent + small shift + token noise.
            let d: Vec<f64> = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (y - x) * 0.05 - 0.0003)
                .collect();
            StratumDiffs {
                label: label.into(),
                n_chunks: n,
                tokens_per_chunk: 64,
                d,
            }
        };
        let strata = [mk("wt2", 11, 24), mk("ag", 99, 30)];
        let (s1, d1) = analyze(&strata, BOOT_B, DEFAULT_SEED).unwrap();
        let (s2, d2) = analyze(&strata, BOOT_B, DEFAULT_SEED).unwrap();
        assert_eq!(s1, s2);
        assert_eq!(d1, d2);
        let (s3, _) = analyze(&strata, BOOT_B, DEFAULT_SEED + 1).unwrap();
        assert_ne!(s1.boot_hi95, s3.boot_hi95);
        assert_eq!((s1.n_strata, s1.n_chunks, s1.n_tokens), (2, 54, 54 * 64));
        // Point is the token-weighted mean over both strata.
        let all: f64 = strata.iter().flat_map(|s| s.d.iter()).sum::<f64>() / (54.0 * 64.0);
        assert!((s1.point - all).abs() < 1e-15);
        // Bootstrap and normal bounds agree to a few percent of the SE here.
        assert!(s1.boot_hi95 > s1.point && s1.normal_hi95 > s1.point);
        assert!(
            (s1.boot_hi95 - s1.normal_hi95).abs() < 0.5 * s1.se,
            "{s1:?}"
        );
        // sd_chunk is the single-stratum-equivalent: se * sqrt(n_chunks).
        assert!((s1.sd_chunk - s1.se * 54f64.sqrt()).abs() < 1e-18);
    }

    #[test]
    fn exact_identity_has_zero_bounds() {
        let s = StratumDiffs {
            label: "a".into(),
            n_chunks: 3,
            tokens_per_chunk: 4,
            d: vec![0.0; 12],
        };
        let (st, dg) = analyze(&[s], 100, 1).unwrap();
        assert!(st.exact && dg[0].exact);
        assert_eq!(
            (st.point, st.se, st.boot_hi95, st.normal_hi95),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert_eq!((st.top1pct_abs_share, dg[0].deff), (0.0, 0.0));
    }

    #[test]
    fn analyze_refuses_bad_input() {
        let ok = |d: Vec<f64>, n, t| StratumDiffs {
            label: "a".into(),
            n_chunks: n,
            tokens_per_chunk: t,
            d,
        };
        assert!(analyze(&[], 10, 1).is_err());
        assert!(analyze(&[ok(vec![0.0; 4], 2, 2)], 0, 1).is_err());
        assert!(analyze(&[ok(vec![0.0; 2], 1, 2)], 10, 1).is_err());
        assert!(analyze(&[ok(vec![0.0; 2], 2, 1)], 10, 1).is_err());
        assert!(analyze(&[ok(vec![0.0; 3], 2, 2)], 10, 1).is_err());
        let e = analyze(&[ok(vec![0.0, f64::NAN, 0.0, 0.0], 2, 2)], 10, 1).unwrap_err();
        assert!(e.contains("non-finite"), "{e}");
    }

    #[test]
    fn evidence_from_files_and_every_refusal() {
        let d = tmp("ev");
        let (n, t) = (24usize, 50usize);
        let inc_a = synth(1, n, t);
        let cand_a: Vec<f64> = inc_a.iter().map(|x| x + 0.0001).collect();
        let (inc_b, cand_b) = (synth(2, n, t), synth(2, n, t));
        let mk = |tag: &str, refsha: &str, i: &[f64], c: &[f64]| {
            let dir = d.join(tag);
            std::fs::create_dir_all(&dir).unwrap();
            (
                tag.to_owned(),
                run_with_ktok("incumbent", &dir, refsha, n, t, i),
                run_with_ktok("candidate", &dir, refsha, n, t, c),
            )
        };
        let a = mk("wt2", "a", &inc_a, &cand_a);
        let bb = mk("ag", "b", &inc_b, &cand_b);
        let ev = build_token_evidence("promotion", vec![a.clone(), bb.clone()], 2000, 3).unwrap();
        assert_eq!(ev.schema, KLD_TOKEN_SCHEMA);
        assert_eq!(ev.strata.len(), 2);
        assert!(ev.strata[1].exact && !ev.strata[0].exact && !ev.stat.exact);
        assert_eq!((ev.stat.n_chunks, ev.stat.n_tokens), (48, 48 * 50));
        // d over stratum 1 is exactly the +0.0001 shift (up to rounding).
        assert!((ev.strata[0].mean_diff - 1e-4).abs() < 1e-12);
        let json = serde_json::to_string(&ev).unwrap();
        // serde_json's default float parser may differ in the last ulp; the gate
        // judges the recorded numbers, so compare them to a tight tolerance.
        let back: KldTokenEvidenceV2 = serde_json::from_str(&json).unwrap();
        assert_eq!(back.strata[0].incumbent, ev.strata[0].incumbent);
        assert_eq!(back.strata.len(), 2);
        assert!((back.stat.boot_hi95 - ev.stat.boot_hi95).abs() < 1e-15);
        assert!((back.stat.normal_hi95 - ev.stat.normal_hi95).abs() < 1e-15);
        assert_eq!((back.stat.seed, back.stat.boot_b), (3, 2000));

        // Unknown grade / empty / repeated labels / shared reference.
        assert!(build_token_evidence("bogus", vec![a.clone()], 10, 1).is_err());
        assert!(build_token_evidence("promotion", vec![], 10, 1).is_err());
        let mut dup = bb.clone();
        dup.0 = "wt2".into();
        assert!(build_token_evidence("promotion", vec![a.clone(), dup], 10, 1).is_err());
        let mut sameref = bb.clone();
        sameref.1.reference_sha256 = a.1.reference_sha256.clone();
        sameref.2.reference_sha256 = a.1.reference_sha256.clone();
        let e = build_token_evidence("promotion", vec![a.clone(), sameref], 10, 1).unwrap_err();
        assert!(e.contains("repeats"), "{e}");

        // Missing ktok fields.
        let mut no = a.clone();
        no.1.ktok_path = None;
        let e = build_token_evidence("promotion", vec![no], 10, 1).unwrap_err();
        assert!(e.contains("--per-token-output"), "{e}");

        // Tampered ktok no longer matches its hash.
        let mut tamper = a.clone();
        let p = tamper.2.ktok_path.clone().unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&p, &bytes).unwrap();
        let e = build_token_evidence("promotion", vec![tamper.clone()], 10, 1).unwrap_err();
        assert!(e.contains("sha256"), "{e}");
        // ... and with the hash re-recorded, the chunk mean no longer reproduces the kldseq.
        tamper.2.ktok_sha256 = Some(sha256_hex(&bytes));
        let e = build_token_evidence("promotion", vec![tamper], 10, 1).unwrap_err();
        assert!(e.contains("bit mismatch"), "{e}");

        // Non-finite per-token value (kldseq built from the same values: NaN mean
        // would also differ bitwise, so the finiteness check must fire first).
        let mut nan_vals = inc_b.clone();
        nan_vals[5] = f64::NAN;
        let dir = d.join("nan");
        std::fs::create_dir_all(&dir).unwrap();
        let nan_run = run_with_ktok("incumbent", &dir, "c", n, t, &nan_vals);
        let e = build_token_evidence(
            "promotion",
            vec![("nan".into(), nan_run.clone(), nan_run)],
            10,
            1,
        )
        .unwrap_err();
        assert!(e.contains("non-finite"), "{e}");

        // Shape mismatch between incumbent and candidate.
        let dir = d.join("shape");
        std::fs::create_dir_all(&dir).unwrap();
        let si = run_with_ktok("incumbent", &dir, "d", n, t, &synth(5, n, t));
        let sc = run_with_ktok("candidate", &dir, "d", n - 1, t, &synth(5, n - 1, t));
        let e = build_token_evidence("promotion", vec![("s".into(), si, sc)], 10, 1).unwrap_err();
        assert!(e.contains("shape mismatch"), "{e}");
        let dir = d.join("shape2");
        std::fs::create_dir_all(&dir).unwrap();
        let si = run_with_ktok("incumbent", &dir, "d", n, t, &synth(5, n, t));
        let sc = run_with_ktok("candidate", &dir, "d", n, t - 1, &synth(5, n, t - 1));
        let e = build_token_evidence("promotion", vec![("s".into(), si, sc)], 10, 1).unwrap_err();
        assert!(e.contains("shape mismatch"), "{e}");

        // ktok chunk count != kldseq chunk count.
        let mut mism = a.clone();
        let p = mism.1.ktok_path.clone().unwrap();
        let short = hfktok_bytes(n - 1, t, &inc_a[..(n - 1) * t]);
        std::fs::write(&p, &short).unwrap();
        mism.1.ktok_sha256 = Some(sha256_hex(&short));
        let e = build_token_evidence("promotion", vec![mism], 10, 1).unwrap_err();
        assert!(e.contains("chunks"), "{e}");
    }

    fn evidence(n_per: usize, shift: f64, strata: usize) -> KldTokenEvidenceV2 {
        let d = tmp(&format!("gate-{n_per}-{strata}-{}", shift.to_bits()));
        let t = 40usize;
        let mut v = Vec::new();
        for s in 0..strata {
            let dir = d.join(s.to_string());
            std::fs::create_dir_all(&dir).unwrap();
            let refsha = ["a", "b", "c"][s];
            let inc = synth(10 + s as u64, n_per, t);
            let cand: Vec<f64> = inc.iter().map(|x| x + shift).collect();
            v.push((
                format!("s{s}"),
                run_with_ktok("incumbent", &dir, refsha, n_per, t, &inc),
                run_with_ktok("candidate", &dir, refsha, n_per, t, &cand),
            ));
        }
        build_token_evidence("promotion", v, BOOT_B, DEFAULT_SEED).unwrap()
    }

    #[test]
    fn judge_bounds_gate() {
        // Exact identity with enough chunks passes; with too few it does not.
        let ex = evidence(24, 0.0, 2);
        assert!(ex.stat.exact);
        judge_bounds(&ex, 0.0005).unwrap();
        let e = judge_bounds(&evidence(19, 0.0, 1), 0.0005).unwrap_err();
        assert!(e.contains("per-stratum floor"), "{e}");
        // A shift well below the margin (the data's chunk noise is ~0 for a
        // constant shift) passes; at/above the margin it is refused with numbers.
        judge_bounds(&evidence(24, 0.0001, 2), 0.0005).unwrap();
        let e = judge_bounds(&evidence(24, 0.0006, 2), 0.0005).unwrap_err();
        assert!(e.contains("boot_hi95") && e.contains("normal_hi95"), "{e}");
        // One bound failing is enough.
        let mut ev = evidence(24, 0.0001, 1);
        ev.stat.normal_hi95 = 0.0005;
        assert!(judge_bounds(&ev, 0.0005).is_err());
        ev.stat.normal_hi95 = 0.0;
        ev.stat.boot_hi95 = f64::NAN;
        assert!(judge_bounds(&ev, 0.0005)
            .unwrap_err()
            .contains("non-finite"));
        ev.stat.boot_hi95 = 0.0;
        ev.stat.boot_b = 100;
        assert!(judge_bounds(&ev, 0.0005).is_err());
        ev.stat.boot_b = BOOT_B;
        ev.stat.n_chunks += 1;
        assert!(judge_bounds(&ev, 0.0005).is_err());
        ev.stat.n_chunks -= 1;
        ev.schema = "x".into();
        assert!(judge_bounds(&ev, 0.0005).is_err());
    }

    #[test]
    fn normal_quantiles() {
        assert!((norm_quantile(0.95).unwrap() - 1.6449).abs() < 5e-5);
        assert!((norm_quantile(0.8).unwrap() - 0.8416).abs() < 5e-5);
        assert!((norm_quantile(0.975).unwrap() - 1.9600).abs() < 5e-5);
        assert!((norm_quantile(0.5).unwrap()).abs() < 1e-9);
        assert!((norm_quantile(0.05).unwrap() + 1.6449).abs() < 5e-5);
        assert!((norm_quantile(0.9999).unwrap() - 3.7190).abs() < 5e-4);
        assert!(norm_quantile(0.0).is_err() && norm_quantile(1.0).is_err());
        assert!(norm_quantile(f64::NAN).is_err());
    }

    #[test]
    fn power_formula_and_r8_worked_example() {
        let p = kld_power(0.00405, 0.0005, 0.05, 0.8, 0.0, 1023, Some(24)).unwrap();
        assert_eq!(p.n_required, 406);
        assert_eq!(p.tokens_required, 406 * 1023);
        assert!((p.z_alpha - 1.6449).abs() < 5e-5 && (p.z_power - 0.8416).abs() < 5e-5);
        let zs = p.z_alpha + p.z_power;
        assert!((p.min_resolvable_margin.unwrap() - zs * 0.00405 / 24f64.sqrt()).abs() < 1e-15);
        assert!((p.min_resolvable_margin.unwrap() - 0.002056).abs() < 5e-6);
        let p48 = kld_power(0.00405, 0.0005, 0.05, 0.8, 0.0, 1023, Some(48)).unwrap();
        assert!((p48.min_resolvable_margin.unwrap() - 0.001454).abs() < 5e-6);
        // An assumed true harm of 0.0003 leaves 0.0002 of headroom: ((2.48647*.00405)/0.0002)^2 = 2535.2.
        assert_eq!(
            kld_power(0.00405, 0.0005, 0.05, 0.8, 0.0003, 1023, None)
                .unwrap()
                .n_required,
            2536
        );
        // delta negative (a real improvement) needs fewer chunks.
        assert!(
            kld_power(0.00405, 0.0005, 0.05, 0.8, -0.0003, 1023, None)
                .unwrap()
                .n_required
                < 406
        );
        let zero = kld_power(0.0, 0.0005, 0.05, 0.8, 0.0, 1, Some(10)).unwrap();
        assert_eq!(
            (zero.n_required, zero.min_resolvable_margin),
            (0, Some(0.0))
        );
        assert!(
            p.available == Some(24)
                && kld_power(0.1, 0.0005, 0.05, 0.8, 0.0, 1, None)
                    .unwrap()
                    .available
                    .is_none()
        );
    }

    #[test]
    fn power_refuses_bad_input() {
        let f = |sd, m, a, pw, d, t, av| kld_power(sd, m, a, pw, d, t, av);
        assert!(f(-1.0, 0.0005, 0.05, 0.8, 0.0, 1, None).is_err());
        assert!(f(f64::NAN, 0.0005, 0.05, 0.8, 0.0, 1, None).is_err());
        assert!(f(0.004, 0.0, 0.05, 0.8, 0.0, 1, None).is_err());
        assert!(f(0.004, 0.0005, 0.05, 0.8, 0.0005, 1, None).is_err());
        assert!(f(0.004, 0.0005, 0.0, 0.8, 0.0, 1, None).is_err());
        assert!(f(0.004, 0.0005, 0.05, 1.0, 0.0, 1, None).is_err());
        assert!(f(0.004, 0.0005, 0.05, 0.8, 0.0, 0, None).is_err());
        assert!(f(0.004, 0.0005, 0.05, 0.8, 0.0, 1, Some(0)).is_err());
    }
}
