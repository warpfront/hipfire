// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Lossless sampled speculative verification over sparse truncated
//! distributions (Leviathan et al. 2023, Chen et al. 2023).
//!
//! A draft token `x` drawn from the draft distribution `q` is accepted with
//! probability `min(1, p(x) / q(x))`; a rejection emits a draw from the
//! normalized residual `(p - q)+`, and a window whose drafts were all accepted
//! emits its bonus from `p`. Every emitted token is then distributed exactly
//! as `p`, whatever `q` is, as long as `x` really was drawn from `q`.
//!
//! Both sides are [`SparseDist`]s truncated by one [`SampleSpec`]: `p` from a
//! full target logit row with exactly the arithmetic of the host AR sampler
//! ([`crate::llama::sample_top_k_p`]), `q` from the draft's candidates. Each
//! side keeps its own nucleus (the DFlash convention,
//! `docs/plans/mtp-sampled-tighten-design-2026-06-23.md` §(a)), and the
//! residual is taken across the two supports. `p` is the distribution the
//! autoregressive producer samples, so the speculative stream is the AR
//! stream in distribution.
//!
//! [`accept_naive_prefix`] is the alternative verifier (SpecInfer naive
//! sampling): drafts stay the draft head's argmax, each verify row's target
//! token is drawn with the host AR sampler itself ([`naive_target_sampler`],
//! shared request-seeded RNG), and a draft is accepted iff it equals its
//! row's draw. Every emitted token is one AR draw in AR's order, so a seeded
//! request emits AR's exact tokens wherever the verify logits equal AR's.

use crate::llama::{CPU_SAMPLE_LEGACY_POOL, CPU_SAMPLE_WIDE_POOL};
use crate::sampler::SamplerConfig;
use crate::spec::{request_rng_state, GreedyAccept, SpecRequestConfig};

/// A request's truncation, resolved as the host AR sampler
/// (`sampler::sample_cpu` → [`crate::llama::sample_top_k_p`]) resolves it:
///
/// 1. `pool`: the highest finite logits gathered first — 20 when `top_k` is
///    absent or `1..=20`, 64 otherwise.
/// 2. `cap`: candidates kept — 20 when `top_k` is absent, `k` for `1..=64`,
///    64 for `0` or above 64 (`top_k = 0` is the 64-wide pool, not the whole
///    vocabulary).
/// 3. `min_p` (`0` disables): the cap shrinks to the first rank whose
///    probability is below `min_p` times the most probable token's.
/// 4. Nucleus at `top_p` over the capped candidates, boundary token kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleSpec {
    pub temperature: f32,
    /// Clamped to `[0, 1]`.
    pub top_p: f32,
    pub pool: usize,
    /// In `1..=pool`.
    pub cap: usize,
    /// In `[0, 1]`.
    pub min_p: f32,
}

impl SampleSpec {
    /// Widest candidate pool any request gathers.
    pub const MAX_POOL: usize = CPU_SAMPLE_WIDE_POOL;

    /// The distribution the host AR sampler draws from for these request
    /// controls. `top_k` is the request's value as sent: `None` (20
    /// candidates) and `Some(0)` (64) differ. A non-finite or non-positive
    /// `min_p` disables the cut, as in the AR sampler.
    pub fn cpu_ar(temperature: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> Self {
        let (pool, cap) = match top_k {
            None => (CPU_SAMPLE_LEGACY_POOL, CPU_SAMPLE_LEGACY_POOL),
            Some(k) if (1..=CPU_SAMPLE_LEGACY_POOL as u32).contains(&k) => {
                (CPU_SAMPLE_LEGACY_POOL, k as usize)
            }
            Some(0) => (CPU_SAMPLE_WIDE_POOL, CPU_SAMPLE_WIDE_POOL),
            Some(k) => (CPU_SAMPLE_WIDE_POOL, (k as usize).min(CPU_SAMPLE_WIDE_POOL)),
        };
        Self {
            temperature,
            top_p: top_p.clamp(0.0, 1.0),
            pool,
            cap,
            min_p: if min_p.is_finite() && min_p > 0.0 {
                min_p.min(1.0)
            } else {
                0.0
            },
        }
    }

    fn check_temperature(self) -> Result<(), String> {
        if self.temperature > 0.0 && self.temperature.is_finite() {
            Ok(())
        } else {
            Err(format!(
                "sampled verification needs a positive temperature, got {}",
                self.temperature
            ))
        }
    }
}

/// A truncated distribution: `(token, probability)` in descending
/// probability order, probabilities summing to one. Rebuilt in place so a
/// caller can reuse its buffers across rows.
#[derive(Clone, Debug, Default)]
pub struct SparseDist {
    entries: Vec<(u32, f32)>,
}

impl SparseDist {
    pub fn entries(&self) -> &[(u32, f32)] {
        &self.entries
    }

    /// Probability of `token` (zero outside the support).
    pub fn prob(&self, token: u32) -> f32 {
        self.entries
            .iter()
            .find(|&&(t, _)| t == token)
            .map_or(0.0, |&(_, p)| p)
    }

    /// Inverse-CDF draw for `u` in `[0, 1)`.
    pub fn sample(&self, u: f32) -> u32 {
        let mut acc = 0.0f32;
        for &(token, p) in &self.entries {
            acc += p;
            if u < acc {
                return token;
            }
        }
        self.entries.last().map_or(0, |&(token, _)| token)
    }

    /// Build the target distribution `p` from a full logit row (row index =
    /// token id) with exactly the arithmetic of `llama::sample_top_k_p`: the
    /// same pool gather (and tie order), f32 softmax against the row maximum,
    /// stable descending sort, cap and min-p cut, summation order and nucleus
    /// boundary. Its two-pass draw (over the cut mass, then again within the
    /// nucleus) picks nucleus token `k` with probability `p_k / mass`, which
    /// is what this stores. A row with no finite mass is the AR sampler's
    /// argmax fallback, a point mass. `scratch` holds the pool between calls.
    pub fn build_from_logits(
        &mut self,
        logits: &[f32],
        spec: SampleSpec,
        scratch: &mut Vec<(u32, f32)>,
    ) -> Result<(), String> {
        spec.check_temperature()?;
        let pool = spec.pool.clamp(1, SampleSpec::MAX_POOL);
        // `sample_pool`'s fixed slots and replacement order (`gather_pool`).
        let mut vals = [f32::NEG_INFINITY; SampleSpec::MAX_POOL];
        let mut idx = [0u32; SampleSpec::MAX_POOL];
        let max_logit = crate::llama::gather_pool(logits, &mut vals[..pool], &mut idx[..pool]);
        scratch.clear();
        scratch.extend(
            idx[..pool]
                .iter()
                .copied()
                .zip(vals[..pool].iter().copied()),
        );
        let inv_temp = 1.0 / spec.temperature;
        let entries = &mut self.entries;
        entries.clear();
        let mut sum = 0.0f32;
        for &(token, l) in scratch.iter() {
            let p = if l.is_finite() {
                let p = ((l - max_logit) * inv_temp).exp();
                if p.is_finite() {
                    p
                } else {
                    0.0
                }
            } else {
                0.0
            };
            entries.push((token, p));
            sum += p;
        }
        if sum <= 0.0 || !sum.is_finite() {
            entries.clear();
            entries.push((crate::llama::argmax(logits), 1.0));
            return Ok(());
        }
        // Stable insertion sort, descending: equal probabilities keep slot
        // order, as in the AR sampler.
        for i in 1..pool {
            let mut j = i;
            while j > 0 && entries[j].1 > entries[j - 1].1 {
                entries.swap(j, j - 1);
                j -= 1;
            }
        }
        let mut cap = spec.cap.clamp(1, pool);
        if spec.min_p > 0.0 {
            let floor = spec.min_p * entries[0].1;
            if let Some(cut) = (1..cap).find(|&i| entries[i].1 < floor) {
                cap = cut;
            }
        }
        // The uncut pool keeps the slot-order sum; a cut sums the kept prefix.
        if cap < pool {
            sum = entries[..cap].iter().map(|&(_, p)| p).sum();
        }
        let threshold = spec.top_p * sum;
        let mut cumulative = 0.0f32;
        let mut boundary = None;
        for (i, &(_, p)) in entries[..cap].iter().enumerate() {
            cumulative += p;
            if cumulative >= threshold {
                boundary = Some(i + 1);
                break;
            }
        }
        match boundary {
            Some(len) => {
                entries.truncate(len);
                for entry in entries.iter_mut() {
                    entry.1 /= cumulative;
                }
            }
            None => {
                // Rounding left the prefix short of the threshold: the AR
                // draw is `p_k / sum`, and its fall-through returns the top
                // token.
                entries.truncate(cap);
                for entry in entries.iter_mut() {
                    entry.1 /= sum;
                }
                entries[0].1 += (1.0 - cumulative / sum).max(0.0);
            }
        }
        entries.retain(|&(_, p)| p > 0.0);
        Ok(())
    }

    /// Build a draft distribution `q` from `(token, logit)` candidates (any
    /// order; non-finite logits are dropped; `candidates` is reordered) by
    /// the same steps over the candidates alone: sort by logit, keep
    /// `spec.cap`, softmax at the temperature, min-p, nucleus. `q` only has
    /// to be the distribution the draft is drawn from; exactness rests on `p`.
    /// Fails when no candidate is finite.
    pub fn build_from_candidates(
        &mut self,
        candidates: &mut Vec<(u32, f32)>,
        spec: SampleSpec,
    ) -> Result<(), String> {
        spec.check_temperature()?;
        candidates.retain(|(_, l)| l.is_finite());
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        candidates.truncate(spec.cap.max(1));
        let Some(&(_, max)) = candidates.first() else {
            return Err("sampled draft has no finite candidate logit".to_string());
        };
        let inv_temp = 1.0 / spec.temperature;
        self.entries.clear();
        self.entries.extend(
            candidates
                .iter()
                .map(|&(token, l)| (token, ((l - max) * inv_temp).exp())),
        );
        if spec.min_p > 0.0 {
            let floor = spec.min_p * self.entries[0].1;
            self.entries.retain(|&(_, p)| p >= floor);
        }
        let sum: f32 = self.entries.iter().map(|&(_, p)| p).sum();
        if !(sum > 0.0) || !sum.is_finite() {
            return Err("sampled draft has no finite mass".to_string());
        }
        let threshold = spec.top_p * sum;
        let mut kept = 0.0f32;
        let mut len = self.entries.len();
        for (i, &(_, p)) in self.entries.iter().enumerate() {
            kept += p;
            if kept >= threshold {
                len = i + 1;
                break;
            }
        }
        self.entries.truncate(len);
        let mass: f32 = self.entries.iter().map(|&(_, p)| p).sum();
        for entry in &mut self.entries {
            entry.1 /= mass;
        }
        Ok(())
    }

    /// The point mass `δ(token)`: the draft distribution of a deterministic
    /// proposal (an n-gram candidate). With it, [`verify_sampled_draft`]
    /// accepts iff `u < p(token)` and a rejection draws from `p` with
    /// `token` removed, which is still exactly `p` overall.
    pub fn set_point_mass(&mut self, token: u32) {
        self.entries.clear();
        self.entries.push((token, 1.0));
    }
}

/// Write the last `min(prompt.len() + emitted.len(), window)` tokens of
/// `prompt ‖ emitted` into `out` (cleared, allocation reused) without copying
/// the whole conversation. `emitted` already holds the pending seed; it is
/// never appended a second time.
pub fn fill_penalty_history(out: &mut Vec<u32>, prompt: &[u32], emitted: &[u32], window: usize) {
    out.clear();
    if window == 0 {
        return;
    }
    if emitted.len() >= window {
        out.extend_from_slice(&emitted[emitted.len() - window..]);
        return;
    }
    let from_prompt = (window - emitted.len()).min(prompt.len());
    out.extend_from_slice(&prompt[prompt.len() - from_prompt..]);
    out.extend_from_slice(emitted);
}

/// The penalty history of one speculative request, shared by every
/// penalty-capable drafter (Qwen3.x MTP, Qwen4 native MTP, n-gram takeover).
///
/// AR penalizes each token against the trailing `window` tokens of the full
/// rendered prompt followed by everything generated so far, so verify row /
/// draft index `i` of a window must see `suffix_W(P ‖ E ‖ drafts[..i])`,
/// where `E` is the loop's authoritative committed generation (pending seed
/// included once). The history is rebuilt from `E` at every window, so
/// rejected drafts, pruned proposals, forced suffixes, semantic clipping and
/// rollback never leak into a later window. Holds at most `window` prompt
/// tokens and `window` + the window's drafts.
///
/// Window 0 (penalties inactive, see
/// [`SpecRequestConfig::penalty_window`]) makes every method a no-op on
/// empty buffers, so a neutral request does no history work.
#[derive(Clone, Debug, Default)]
pub struct PenaltyHistory {
    prompt_tail: Vec<u32>,
    tokens: Vec<u32>,
    window: usize,
    base: usize,
}

impl PenaltyHistory {
    pub fn new(window: usize) -> Self {
        Self {
            prompt_tail: Vec::with_capacity(window),
            tokens: Vec::new(),
            window,
            base: 0,
        }
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// Keep the trailing `window` tokens of the full rendered prompt. Call
    /// at every prefill, cold or warm (cache hit / realignment), with the
    /// full `prompt_tokens`, never the cache-fill suffix. Also starts the
    /// prefill window: row 0 is `suffix_W(P)`, the first token's history.
    pub fn set_prompt(&mut self, prompt: &[u32]) {
        self.prompt_tail.clear();
        self.prompt_tail
            .extend_from_slice(&prompt[prompt.len().saturating_sub(self.window)..]);
        self.begin_window(&[]);
    }

    /// Rebuild from the prompt tail and the loop's authoritative `emitted`
    /// (which already contains the pending seed). Returns the base length:
    /// the history of row 0.
    pub fn begin_window(&mut self, emitted: &[u32]) -> usize {
        fill_penalty_history(&mut self.tokens, &self.prompt_tail, emitted, self.window);
        self.base = self.tokens.len();
        self.base
    }

    /// Append a kept draft so row `i + 1` sees `drafts[..=i]`. Never push a
    /// pruned proposal or a rejected draft.
    pub fn push_draft(&mut self, token: u32) {
        if self.window > 0 {
            self.tokens.push(token);
        }
    }

    /// Drop every draft pushed since [`Self::begin_window`].
    pub fn rewind_drafts(&mut self) {
        self.tokens.truncate(self.base);
    }

    /// History of verify row / draft index `row`: the trailing
    /// `min(base + row, window)` tokens of `base + drafts[..row]`, exactly
    /// the scope AR penalizes against (and what a GPU repeat buffer holds).
    /// Panics if fewer than `row` drafts were pushed.
    pub fn row(&self, row: usize) -> &[u32] {
        &self.tokens[self.row_range(row)]
    }

    /// [`Self::row`] as a range of [`Self::tokens`]: a GPU consumer uploads
    /// `tokens()` once and penalizes row `i` against that sub-range.
    pub fn row_range(&self, row: usize) -> std::ops::Range<usize> {
        if self.window == 0 {
            return 0..0;
        }
        let end = self.base + row;
        assert!(
            end <= self.tokens.len(),
            "penalty history row {row} needs {row} pushed drafts, has {}",
            self.tokens.len() - self.base
        );
        end.saturating_sub(self.window)..end
    }

    /// The window's whole buffer: base history then every pushed draft.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

/// Draw from the normalized residual `(p - q)+`. When `p <= q` everywhere
/// (equal distributions up to rounding) the residual is empty and the draw
/// falls back to `p`, which a rejection then had probability zero to reach.
pub fn sample_residual(p: &SparseDist, q: &SparseDist, u: f32) -> u32 {
    let residual = |&(token, pt): &(u32, f32)| (token, (pt - q.prob(token)).max(0.0));
    let mass: f32 = p.entries.iter().map(|e| residual(e).1).sum();
    if !(mass > 0.0) {
        return p.sample(u);
    }
    let target = u * mass;
    let mut acc = 0.0f32;
    let mut last = None;
    for (token, r) in p.entries.iter().map(residual) {
        if r <= 0.0 {
            continue;
        }
        acc += r;
        if target < acc {
            return token;
        }
        last = Some(token);
    }
    last.unwrap_or_else(|| p.sample(u))
}

/// Outcome of verifying one sampled draft.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftVerdict {
    Accept,
    /// Rejected; the token is the residual draw that replaces the draft.
    Reject(u32),
}

/// Accept `draft` (drawn from `q`) with probability `min(1, p/q)`, else draw
/// its replacement from the residual. Consumes one uniform on accept, two
/// on reject.
pub fn verify_sampled_draft(
    p: &SparseDist,
    q: &SparseDist,
    draft: u32,
    rng: &mut SpecRng,
) -> DraftVerdict {
    let qx = q.prob(draft);
    let px = p.prob(draft);
    let u = rng.next_f32();
    // Strict: a token with p(x) = 0 is never accepted, and p >= q accepts
    // for every u in [0, 1).
    if qx > 0.0 && u * qx < px {
        DraftVerdict::Accept
    } else {
        DraftVerdict::Reject(sample_residual(p, q, rng.next_f32()))
    }
}

/// The sampled counterpart of [`crate::spec::accept_greedy_prefix`], with the
/// same result shape and EOS rule: accepted drafts in order, stopping (no
/// bonus) at an accepted EOS draft, then either the residual replacement of
/// the first rejected draft or the bonus drawn from `p` at row
/// `drafts.len()`. `target(row, out)` fills the target distribution of a
/// verify row; it is called only for rows the verdict reads, in order.
pub fn accept_sampled_prefix<F>(
    drafts: &[u32],
    draft_dists: &[SparseDist],
    eos: Option<u32>,
    rng: &mut SpecRng,
    target: &mut SparseDist,
    mut fill_target: F,
) -> Result<GreedyAccept, String>
where
    F: FnMut(usize, &mut SparseDist) -> Result<(), String>,
{
    if draft_dists.len() != drafts.len() {
        return Err(format!(
            "sampled verify has {} draft distributions for {} drafts",
            draft_dists.len(),
            drafts.len()
        ));
    }
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    for (row, (&draft, q)) in drafts.iter().zip(draft_dists).enumerate() {
        fill_target(row, target)?;
        match verify_sampled_draft(target, q, draft, rng) {
            DraftVerdict::Accept => {
                committed.push(draft);
                accepted += 1;
                if eos == Some(draft) {
                    return Ok(GreedyAccept {
                        committed,
                        accepted,
                        hit_eos: true,
                    });
                }
            }
            DraftVerdict::Reject(token) => {
                committed.push(token);
                return Ok(GreedyAccept {
                    committed,
                    accepted,
                    hit_eos: eos == Some(token),
                });
            }
        }
    }
    fill_target(drafts.len(), target)?;
    let bonus = target.sample(rng.next_f32());
    committed.push(bonus);
    Ok(GreedyAccept {
        committed,
        accepted,
        hit_eos: eos == Some(bonus),
    })
}

/// The host AR producer's sampler for a sampled request: what
/// `sampler::sample_cpu` draws a Qwen4 AR token with, request penalties
/// included. It is the target policy of both verifiers: Leviathan builds `p`
/// from the row it penalizes, naive draws with it. `top_k` passes through as
/// sent (absent and `Some(0)` differ); a non-positive `min_p` is the AR
/// sampler's absent.
pub fn naive_target_sampler(cfg: &SpecRequestConfig) -> SamplerConfig {
    SamplerConfig {
        temperature: cfg.temp,
        top_p: cfg.top_p,
        top_k: cfg.top_k,
        min_p: (cfg.min_p > 0.0).then_some(cfg.min_p),
        repeat_penalty: cfg.repeat_penalty,
        repeat_window: cfg.repeat_window,
        presence_penalty: cfg.presence_penalty,
        frequency_penalty: cfg.frequency_penalty,
        ..SamplerConfig::greedy()
    }
}

/// SpecInfer naive sampled verification, with the result shape and EOS rule
/// of [`crate::spec::accept_greedy_prefix`]. `draw(row)` returns verify row
/// `row`'s target draw; it is called in row order and only up to the row the
/// verdict ends on: the first draft that differs from its draw (the draw
/// replaces it), an accepted EOS draft (no bonus), or the bonus row
/// `drafts.len()`. Each emitted token is therefore exactly one draw, in
/// emission order.
pub fn accept_naive_prefix<F>(
    drafts: &[u32],
    eos: Option<u32>,
    mut draw: F,
) -> Result<GreedyAccept, String>
where
    F: FnMut(usize) -> Result<u32, String>,
{
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    for (row, &draft) in drafts.iter().enumerate() {
        let token = draw(row)?;
        committed.push(token);
        if token != draft || eos == Some(token) {
            return Ok(GreedyAccept {
                committed,
                accepted: row + usize::from(token == draft),
                hit_eos: eos == Some(token),
            });
        }
    }
    let bonus = draw(drafts.len())?;
    committed.push(bonus);
    Ok(GreedyAccept {
        committed,
        accepted: drafts.len(),
        hit_eos: eos == Some(bonus),
    })
}

/// Request-seeded xorshift64* stream for draft draws and verdicts. The same
/// seed replays the same draws.
#[derive(Clone, Copy, Debug)]
pub struct SpecRng(u64);

impl SpecRng {
    pub fn new(rng_seed: u64) -> Self {
        Self(
            request_rng_state(rng_seed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .max(1),
        )
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on the 2^-24 grid of `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pearson chi-square of `counts` (total `n`) against `expected`
    /// probabilities, cells with expected count under 5 pooled into one.
    /// Returns `(statistic, degrees of freedom)`.
    fn chi_square(counts: &[u64], expected: &[f64], n: u64) -> (f64, usize) {
        let mut stat = 0.0;
        let mut cells = 0usize;
        let (mut pooled_obs, mut pooled_exp) = (0.0, 0.0);
        for (&c, &e) in counts.iter().zip(expected) {
            let e = e * n as f64;
            if e < 5.0 {
                pooled_obs += c as f64;
                pooled_exp += e;
                assert!(
                    e > 0.0 || c == 0,
                    "token outside the target support was emitted"
                );
                continue;
            }
            stat += (c as f64 - e).powi(2) / e;
            cells += 1;
        }
        if pooled_exp >= 5.0 {
            stat += (pooled_obs - pooled_exp).powi(2) / pooled_exp;
            cells += 1;
        }
        (stat, cells.saturating_sub(1).max(1))
    }

    /// Wilson–Hilferty chi-square quantile at standard-normal `z`; z = 4.75
    /// is the 1 - 1e-6 quantile.
    fn chi_square_critical(df: usize, z: f64) -> f64 {
        let df = df as f64;
        let a = 2.0 / (9.0 * df);
        df * (1.0 - a + z * a.sqrt()).powi(3)
    }

    fn total_variation(counts: &[u64], expected: &[f64], n: u64) -> f64 {
        counts
            .iter()
            .zip(expected)
            .map(|(&c, &e)| (c as f64 / n as f64 - e).abs())
            .sum::<f64>()
            / 2.0
    }

    /// Deterministic pseudo-logits.
    fn logits(vocab: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = SpecRng::new(seed);
        (0..vocab)
            .map(|_| {
                // Sum of uniforms: a bell-ish spread with a few strong heads.
                let g: f32 = (0..4).map(|_| rng.next_f32()).sum::<f32>() - 2.0;
                g * scale
            })
            .collect()
    }

    /// The draft side the Qwen4 MTP drafter builds: a noisy ranking of the
    /// target's logits picks 8 candidates, which are then "re-scored" with
    /// their exact draft logits (themselves a perturbation of the target).
    fn rescored_draft(target: &[f32], seed: u64, spec: SampleSpec) -> SparseDist {
        let noise = logits(target.len(), seed, 1.5);
        let exact = logits(target.len(), seed ^ 0xABCD, 0.7);
        let mut ranked: Vec<(u32, f32)> = target
            .iter()
            .zip(&noise)
            .enumerate()
            .map(|(i, (t, n))| (i as u32, t + n))
            .collect();
        ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut candidates: Vec<(u32, f32)> = ranked[..8]
            .iter()
            .map(|&(i, _)| (i, target[i as usize] + exact[i as usize]))
            .collect();
        let mut q = SparseDist::default();
        q.build_from_candidates(&mut candidates, spec).unwrap();
        q
    }

    fn dense(dist: &SparseDist, vocab: usize) -> Vec<f64> {
        let mut out = vec![0.0; vocab];
        for &(t, p) in dist.entries() {
            out[t as usize] = p as f64;
        }
        out
    }

    const TRIALS: u64 = 200_000;

    /// Request `top_k`: absent, inside the legacy 20-wide pool, its edge,
    /// the 64-wide pool, its edge, and 0 (the 64-wide pool, not the vocab).
    const TOP_KS: [Option<u32>; 6] = [None, Some(5), Some(20), Some(40), Some(64), Some(0)];
    const MIN_PS: [f32; 3] = [0.0, 0.05, 0.1];

    struct Case {
        name: String,
        row: Vec<f32>,
        temp: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: f32,
    }

    impl Case {
        fn new(row: Vec<f32>, temp: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> Self {
            Self {
                name: format!("T{temp} top_p {top_p} top_k {top_k:?} min_p {min_p}"),
                row,
                temp,
                top_p,
                top_k,
                min_p,
            }
        }

        fn spec(&self) -> SampleSpec {
            SampleSpec::cpu_ar(self.temp, self.top_p, self.top_k, self.min_p)
        }

        fn target(&self) -> SparseDist {
            let mut p = SparseDist::default();
            p.build_from_logits(&self.row, self.spec(), &mut Vec::new())
                .unwrap();
            p
        }
    }

    /// The full `TOP_KS` x `MIN_PS` grid at T1.0 / top_p 0.95, other
    /// temperatures and nuclei, and a tied row on which the AR pool evicts a
    /// non-lowest id (20 equal logits fill the legacy pool, then a larger one
    /// replaces slot 0, so id 0 drops out while ids 20.. never enter).
    fn cases() -> Vec<Case> {
        let vocab = 300;
        let mut out = Vec::new();
        for (i, &top_k) in TOP_KS.iter().enumerate() {
            for (j, &min_p) in MIN_PS.iter().enumerate() {
                let row = logits(vocab, 1000 + (i * MIN_PS.len() + j) as u64, 2.5);
                out.push(Case::new(row, 1.0, 0.95, top_k, min_p));
            }
        }
        for (n, (temp, top_p, top_k, min_p)) in [
            (0.7, 0.8, None, 0.0),
            (1.0, 1.0, None, 0.0),
            (1.0, 1.0, Some(0), 0.0),
            (0.7, 0.9, Some(40), 0.05),
            (1.3, 1.0, Some(64), 0.1),
        ]
        .into_iter()
        .enumerate()
        {
            out.push(Case::new(
                logits(vocab, 2000 + n as u64, 2.5),
                temp,
                top_p,
                top_k,
                min_p,
            ));
        }
        let mut tied = vec![-4.0f32; vocab];
        tied[..30].fill(1.0);
        tied[25] = 2.0;
        for top_k in [None, Some(5)] {
            let mut case = Case::new(tied.clone(), 1.0, 1.0, top_k, 0.0);
            case.name.push_str(" (tied row)");
            out.push(case);
        }
        out
    }

    /// Chi-square (alpha 1e-6) and total-variation check of `counts`
    /// against `p`.
    fn assert_follows(name: &str, counts: &[u64], p: &SparseDist) {
        let expected = dense(p, counts.len());
        let (stat, df) = chi_square(counts, &expected, TRIALS);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(counts, &expected, TRIALS);
        eprintln!(
            "{name}: support={} chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4}",
            p.entries().len()
        );
        assert!(stat < crit, "{name}: chi2 {stat} >= {crit} (df {df})");
        assert!(tv < 0.01, "{name}: total variation {tv}");
    }

    /// `p` is the host AR sampler's distribution: 200k
    /// `llama::sample_top_k_p` draws per case follow it, over every request
    /// `top_k` (absent and 0 included) and `min_p`.
    #[test]
    fn target_is_the_host_ar_sampler_distribution() {
        // Draws from the process-global AR RNG; hold it for the whole run.
        let _rng = crate::llama::sampler_rng_test_guard();
        for (n, case) in cases().iter().enumerate() {
            let p = case.target();
            // Well-spread seeds: the AR xorshift32 maps close seeds to close
            // first draws.
            let seed = (SpecRng::new(n as u64 + 1).next_u64() >> 32) as u32;
            crate::llama::reset_cpu_sampler_rng(seed);
            let min_p = (case.min_p > 0.0).then_some(case.min_p);
            let mut counts = vec![0u64; case.row.len()];
            for _ in 0..TRIALS {
                let token = crate::llama::sample_top_k_p(
                    &case.row, case.temp, case.top_p, case.top_k, min_p,
                );
                counts[token as usize] += 1;
            }
            assert_follows(&format!("ar {}", case.name), &counts, &p);
        }
    }

    /// One-position exactness: draft from the 8-candidate re-scored `q`
    /// (built by the same truncation), verify against `p`, and the emitted
    /// token must follow `p` for every case; the acceptance rate must be
    /// its expectation `sum_x min(p, q)`.
    #[test]
    fn single_position_emits_target_distribution() {
        for (n, case) in cases().iter().enumerate() {
            let p = case.target();
            let q = rescored_draft(&case.row, 100 + n as u64, case.spec());
            let mut rng = SpecRng::new(7 + n as u64);
            let mut counts = vec![0u64; case.row.len()];
            let mut accepted = 0u64;
            for _ in 0..TRIALS {
                let x = q.sample(rng.next_f32());
                let token = match verify_sampled_draft(&p, &q, x, &mut rng) {
                    DraftVerdict::Accept => {
                        accepted += 1;
                        x
                    }
                    DraftVerdict::Reject(t) => t,
                };
                counts[token as usize] += 1;
            }
            assert_follows(&format!("spec {}", case.name), &counts, &p);
            let overlap: f64 = q
                .entries()
                .iter()
                .map(|&(t, qt)| (qt as f64).min(p.prob(t) as f64))
                .sum();
            let rate = accepted as f64 / TRIALS as f64;
            eprintln!("spec {}: accept={rate:.4} overlap={overlap:.4}", case.name);
            assert!(
                (rate - overlap).abs() < 0.01,
                "{}: accept {rate} vs {overlap}",
                case.name
            );
        }
    }

    /// Negative control: the verify of `single_position_emits_target_distribution`
    /// with the rejection replacement drawn from `p` instead of the residual
    /// `(p - q)+` must fail the same chi-square.
    #[test]
    fn residual_drawn_from_target_is_detected() {
        for (n, case) in cases().iter().enumerate().step_by(4) {
            let p = case.target();
            let q = rescored_draft(&case.row, 100 + n as u64, case.spec());
            let mut rng = SpecRng::new(7 + n as u64);
            let mut counts = vec![0u64; case.row.len()];
            for _ in 0..TRIALS {
                let x = q.sample(rng.next_f32());
                let qx = q.prob(x);
                let token = if qx > 0.0 && rng.next_f32() * qx < p.prob(x) {
                    x
                } else {
                    p.sample(rng.next_f32())
                };
                counts[token as usize] += 1;
            }
            let expected = dense(&p, counts.len());
            let (stat, df) = chi_square(&counts, &expected, TRIALS);
            let crit = chi_square_critical(df, 4.75);
            eprintln!(
                "control {}: chi2={stat:.1} df={df} crit(1e-6)={crit:.1}",
                case.name
            );
            assert!(
                stat > crit,
                "{}: a residual drawn from p went undetected (chi2 {stat} < {crit})",
                case.name
            );
        }
    }

    /// Two chained positions through `accept_sampled_prefix`, with the
    /// second target row conditioned on the first token and an EOS token in
    /// the vocabulary: the emitted (t1, t2) pairs must follow
    /// p1(t1) p2(t2 | t1), with an emitted EOS ending the window.
    #[test]
    fn chained_window_emits_target_joint_distribution() {
        let vocab = 8usize;
        let eos = 3u32;
        let spec = SampleSpec::cpu_ar(1.0, 0.95, None, 0.0);
        let p1_logits = logits(vocab, 21, 1.5);
        let p2_logits: Vec<Vec<f32>> = (0..vocab)
            .map(|t| logits(vocab, 40 + t as u64, 1.5))
            .collect();
        let mut p1 = SparseDist::default();
        p1.build_from_logits(&p1_logits, spec, &mut Vec::new())
            .unwrap();
        let p2: Vec<SparseDist> = p2_logits
            .iter()
            .map(|l| {
                let mut d = SparseDist::default();
                d.build_from_logits(l, spec, &mut Vec::new()).unwrap();
                d
            })
            .collect();
        // Category index: t1 * (vocab + 1) + t2, t2 == vocab = "window ended".
        let cells = vocab * (vocab + 1);
        let mut expected = vec![0.0f64; cells];
        for &(t1, a) in p1.entries() {
            if t1 == eos {
                expected[t1 as usize * (vocab + 1) + vocab] += a as f64;
                continue;
            }
            for &(t2, b) in p2[t1 as usize].entries() {
                expected[t1 as usize * (vocab + 1) + t2 as usize] += a as f64 * b as f64;
            }
        }
        let mut rng = SpecRng::new(99);
        let mut counts = vec![0u64; cells];
        let mut target = SparseDist::default();
        for trial in 0..TRIALS {
            // Draft depth 1 or 2 so both the bonus and the residual reach t2.
            let depth = 1 + (trial % 2) as usize;
            let q1 = rescored_draft(&p1_logits, 500, spec);
            let x1 = q1.sample(rng.next_f32());
            let mut drafts = vec![x1];
            let mut qs = vec![q1];
            if depth == 2 {
                let q2 = rescored_draft(&p2_logits[x1 as usize], 600 + x1 as u64, spec);
                drafts.push(q2.sample(rng.next_f32()));
                qs.push(q2);
            }
            let window = accept_sampled_prefix(
                &drafts,
                &qs,
                Some(eos),
                &mut rng,
                &mut target,
                |row, out| {
                    let src = if row == 0 {
                        &p1
                    } else {
                        &p2[drafts[row - 1] as usize]
                    };
                    out.clone_from(src);
                    Ok(())
                },
            )
            .unwrap();
            let t1 = window.committed[0] as usize;
            let t2 = if window.committed[0] == eos {
                assert!(window.hit_eos);
                vocab
            } else if let Some(&t2) = window.committed.get(1) {
                t2 as usize
            } else {
                // Rejected at row 0: the second token comes from the next
                // window, whose seed row is p2(. | t1).
                let mut next = SparseDist::default();
                next.clone_from(&p2[t1]);
                next.sample(rng.next_f32()) as usize
            };
            counts[t1 * (vocab + 1) + t2] += 1;
        }
        let (stat, df) = chi_square(&counts, &expected, TRIALS);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(&counts, &expected, TRIALS);
        eprintln!("joint: chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4}");
        assert!(stat < crit, "joint chi2 {stat} >= {crit} (df {df})");
        assert!(tv < 0.01, "joint total variation {tv}");
    }

    #[test]
    fn truncation_keeps_boundary_order_and_request_top_k() {
        let logits = [0.0f32, 3.0, f32::NAN, 2.0, 1.0, f32::NEG_INFINITY];
        let mut d = SparseDist::default();
        let spec = SampleSpec::cpu_ar(1.0, 0.7, Some(3), 0.0);
        d.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        // softmax over {1: 3, 3: 2, 4: 1} = .665/.245/.090; top_p 0.7 keeps
        // token 1 and the boundary token 3.
        let tokens: Vec<u32> = d.entries().iter().map(|e| e.0).collect();
        assert_eq!(tokens, [1, 3]);
        let sum: f32 = d.entries().iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        let mut d2 = SparseDist::default();
        d2.build_from_logits(
            &logits,
            SampleSpec::cpu_ar(1.0, 1.0, Some(3), 0.3),
            &mut Vec::new(),
        )
        .unwrap();
        // e^-1 = .37 >= .3 keeps token 3; e^-2 = .135 < .3 drops token 4.
        assert_eq!(d2.entries().iter().map(|e| e.0).collect::<Vec<_>>(), [1, 3]);
        // No finite logit: the AR sampler's argmax fallback, a point mass.
        let nan = [f32::NAN; 4];
        d.build_from_logits(&nan, spec, &mut Vec::new()).unwrap();
        assert_eq!(d.entries(), [(crate::llama::argmax(&nan), 1.0)]);
        // A flat 100-token row at top_p 1 keeps exactly the request's cap:
        // absent is 20, 0 is the 64-wide pool, larger values clamp to it.
        let row: Vec<f32> = (0..100).map(|i| -(i as f32) * 0.01).collect();
        for (top_k, kept) in [
            (None, 20),
            (Some(7), 7),
            (Some(20), 20),
            (Some(21), 21),
            (Some(40), 40),
            (Some(64), 64),
            (Some(0), 64),
            (Some(1000), 64),
        ] {
            let spec = SampleSpec::cpu_ar(1.0, 1.0, top_k, 0.0);
            d.build_from_logits(&row, spec, &mut Vec::new()).unwrap();
            assert_eq!(d.entries().len(), kept, "top_k {top_k:?}");
        }
    }

    #[test]
    fn same_seed_replays_the_same_stream() {
        let mut a = SpecRng::new(42);
        let mut b = SpecRng::new(42);
        let mut c = SpecRng::new(43);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        assert_eq!(xs, (0..8).map(|_| b.next_u64()).collect::<Vec<_>>());
        assert_ne!(xs, (0..8).map(|_| c.next_u64()).collect::<Vec<_>>());
        // Seed 0 maps onto the request sentinel instead of the stuck state.
        assert_eq!(
            SpecRng::new(0).next_u64(),
            SpecRng::new(0x1357_9BDF).next_u64()
        );
    }

    /// Toy target for the naive-verify stream test: each context's logit
    /// row is a hash of its last two tokens, and EOS gains mass late so
    /// some streams end on it (as an accepted draft, a replacement or a
    /// bonus).
    const TOY_VOCAB: usize = 300;
    const TOY_EOS: u32 = 5;

    fn toy_row(ctx: &[u32]) -> Vec<f32> {
        let n = ctx.len();
        let key = ((ctx[n - 1] as u64) << 20) | ctx[n - 2] as u64;
        let mut row = logits(TOY_VOCAB, key ^ 0x70E5, 2.5);
        if n >= 40 {
            row[TOY_EOS as usize] += 4.0;
        }
        row
    }

    /// The draft head's argmax, wrong on about a quarter of the contexts
    /// (it picks the runner-up there).
    fn toy_draft(ctx: &[u32]) -> u32 {
        let row = toy_row(ctx);
        let mut ranked: Vec<u32> = (0..TOY_VOCAB as u32).collect();
        ranked.sort_unstable_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
        let salt = ctx.len() as u64 ^ ((ctx[ctx.len() - 1] as u64) << 8);
        let miss = SpecRng::new(salt).next_u64() % 4 == 0;
        ranked[usize::from(miss)]
    }

    fn toy_draw(ctx: &[u32], sampler: &SamplerConfig) -> u32 {
        crate::sampler::sample_cpu(&mut toy_row(ctx), &[], sampler)
    }

    /// Naive verification emits the AR producer's exact stream at the same
    /// seed, over windows of every depth 1..=4 (the interleaved route is the
    /// one-draft case), under each request truncation the AR sampler honours.
    #[test]
    fn naive_verify_emits_the_seeded_ar_stream() {
        const TOKENS: usize = 64;
        let _rng = crate::llama::sampler_rng_test_guard();
        let prompt = [11u32, 42, 7];
        let (mut accepted, mut rejected, mut eos_ends) = (0usize, 0usize, 0usize);
        for (temp, top_p, top_k, min_p) in [
            (0.7, 0.8, Some(20), 0.0),
            (1.0, 0.95, None, 0.0),
            (0.8, 0.9, Some(0), 0.0),
            (0.7, 0.9, Some(40), 0.05),
        ] {
            let sampler = naive_target_sampler(&SpecRequestConfig {
                temp,
                top_p,
                top_k,
                min_p,
                ..SpecRequestConfig::default()
            });
            for seed in 1..=24u32 {
                let seed = (SpecRng::new(seed as u64).next_u64() >> 32) as u32;
                crate::llama::reset_cpu_sampler_rng(seed);
                let mut ar = prompt.to_vec();
                while ar.len() - prompt.len() < TOKENS {
                    let token = toy_draw(&ar, &sampler);
                    ar.push(token);
                    if token == TOY_EOS {
                        break;
                    }
                }
                let ar = &ar[prompt.len()..];
                eos_ends += usize::from(ar.last() == Some(&TOY_EOS));

                for depths in [[1usize, 1, 1, 1], [1, 3, 2, 4], [4, 4, 4, 4]] {
                    crate::llama::reset_cpu_sampler_rng(seed);
                    let mut ctx = prompt.to_vec();
                    // The prefill seed: the last prompt row's draw.
                    ctx.push(toy_draw(&ctx, &sampler));
                    let mut window = 0usize;
                    while ctx.len() - prompt.len() < TOKENS && ctx.last() != Some(&TOY_EOS) {
                        let mut block = ctx.clone();
                        let mut drafts = Vec::new();
                        for _ in 0..depths[window % depths.len()] {
                            let draft = toy_draft(&block);
                            drafts.push(draft);
                            block.push(draft);
                        }
                        window += 1;
                        let verdict = accept_naive_prefix(&drafts, Some(TOY_EOS), |row| {
                            let mut rows = ctx.clone();
                            rows.extend_from_slice(&drafts[..row]);
                            Ok(toy_draw(&rows, &sampler))
                        })
                        .unwrap();
                        accepted += verdict.accepted;
                        rejected += usize::from(
                            verdict.accepted < drafts.len()
                                && !(verdict.hit_eos
                                    && verdict.committed.len() == verdict.accepted),
                        );
                        ctx.extend_from_slice(&verdict.committed);
                    }
                    let mut spec = ctx[prompt.len()..].to_vec();
                    spec.truncate(TOKENS);
                    assert_eq!(
                        spec, ar,
                        "T{temp} top_p {top_p} top_k {top_k:?} min_p {min_p} seed {seed} depths {depths:?}"
                    );
                }
            }
        }
        assert!(
            accepted > 0 && rejected > 0,
            "accepted {accepted}, rejected {rejected}"
        );
        assert!(eos_ends > 0, "no stream reached EOS");
    }

    // ------------------------------------------------------------------
    // Penalty history, penalty-aware targets and lossless penalized windows
    // ------------------------------------------------------------------

    use crate::sampler::{
        apply_logit_policy_candidates_cpu, apply_logit_policy_cpu, sample_cpu,
    };
    use std::collections::HashMap;

    fn splitmix64(x: u64) -> u64 {
        let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Independent, well-spread AR RNG seed for trial `i` (the AR xorshift32
    /// maps close seeds to close first draws).
    fn scrambled_seed(i: u64) -> u32 {
        (splitmix64(i) >> 32) as u32
    }

    fn penalty_cfg(
        repeat_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        repeat_window: usize,
    ) -> SpecRequestConfig {
        SpecRequestConfig {
            temp: 0.7,
            top_p: 0.8,
            top_k: Some(20),
            min_p: 0.0,
            repeat_penalty,
            repeat_window,
            presence_penalty,
            frequency_penalty,
            ..SpecRequestConfig::default()
        }
    }

    /// `(repeat, presence, frequency)`: presence only, and all three.
    const PENALTY_CONFIGS: [(f32, f32, f32); 2] = [(1.0, 1.5, 0.0), (1.1, 1.5, 0.5)];

    fn tv_between(a: &SparseDist, b: &SparseDist, vocab: usize) -> f64 {
        dense(a, vocab)
            .iter()
            .zip(dense(b, vocab))
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
            / 2.0
    }

    /// `assert_follows` for an arbitrary trial count `n`: chi-square at
    /// alpha 1e-6, and a total-variation bound scaled to the support size.
    fn assert_follows_n(name: &str, counts: &[u64], p: &SparseDist, n: u64) {
        let expected = dense(p, counts.len());
        let (stat, df) = chi_square(counts, &expected, n);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(counts, &expected, n);
        let bound = (p.entries().len() as f64 / n as f64).sqrt() + 0.005;
        eprintln!(
            "{name}: support={} chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4} (<{bound:.4})",
            p.entries().len()
        );
        assert!(stat < crit, "{name}: chi2 {stat} >= {crit} (df {df})");
        assert!(tv < bound, "{name}: total variation {tv} >= {bound}");
    }

    // ---- fill_penalty_history / PenaltyHistory ----

    fn suffix(tokens: &[u32], window: usize) -> &[u32] {
        &tokens[tokens.len().saturating_sub(window)..]
    }

    #[test]
    fn fill_penalty_history_takes_the_trailing_window_of_prompt_and_emitted() {
        let fill = |prompt: &[u32], emitted: &[u32], window: usize| {
            // A dirty buffer: the helper must clear it.
            let mut out = vec![99u32; 3];
            fill_penalty_history(&mut out, prompt, emitted, window);
            out
        };
        let p10: Vec<u32> = (1..=10).collect();
        // Window 0: nothing, whatever the inputs.
        assert!(fill(&p10, &[11, 12], 0).is_empty());
        assert!(fill(&[], &[], 0).is_empty());
        // Nothing at all.
        assert!(fill(&[], &[], 8).is_empty());
        // Prompt shorter than W, no emitted tokens yet.
        assert_eq!(fill(&[1, 2, 3], &[], 8), [1, 2, 3]);
        // The pending seed appears exactly once.
        assert_eq!(fill(&[1, 2, 3], &[4], 8), [1, 2, 3, 4]);
        // Prompt + emitted shorter than W: everything.
        assert_eq!(fill(&[1, 2, 3], &[4, 5], 8), [1, 2, 3, 4, 5]);
        // Prompt longer than W, no emitted.
        assert_eq!(fill(&p10, &[], 4), [7, 8, 9, 10]);
        // Prompt longer than W, emitted shorter than W: prompt tail first.
        assert_eq!(fill(&p10, &[11, 12], 5), [8, 9, 10, 11, 12]);
        assert_eq!(fill(&p10, &[11], 5), [7, 8, 9, 10, 11]);
        // Emitted alone fills W (prompt contributes nothing).
        assert_eq!(fill(&p10, &[11, 12, 13, 14, 15], 5), [11, 12, 13, 14, 15]);
        assert_eq!(
            fill(&p10, &[11, 12, 13, 14, 15, 16, 17], 5),
            [13, 14, 15, 16, 17]
        );
        // W far above everything: the whole conversation.
        assert_eq!(fill(&[1, 2], &[3, 4], 1000), [1, 2, 3, 4]);
        // Always the suffix of the concatenation, for every shape.
        for window in 0..=14usize {
            for plen in 0..=12u32 {
                for elen in 0..=12u32 {
                    let prompt: Vec<u32> = (0..plen).map(|i| 1000 + i).collect();
                    let emitted: Vec<u32> = (0..elen).map(|i| 2000 + i).collect();
                    let concat = [prompt.clone(), emitted.clone()].concat();
                    assert_eq!(
                        fill(&prompt, &emitted, window),
                        suffix(&concat, window),
                        "window {window} plen {plen} elen {elen}"
                    );
                }
            }
        }
    }

    #[test]
    fn penalty_history_window_zero_is_a_no_op() {
        let mut h = PenaltyHistory::new(0);
        assert_eq!(h.window(), 0);
        h.set_prompt(&[1, 2, 3]);
        assert_eq!(h.begin_window(&[4, 5]), 0);
        h.push_draft(6);
        h.push_draft(7);
        assert!(h.tokens().is_empty(), "window 0 must buffer nothing");
        // Rows past the (ignored) pushes never panic: there is no history.
        for i in 0..6 {
            assert!(h.row(i).is_empty());
            assert_eq!(h.row_range(i), 0..0);
        }
        h.rewind_drafts();
        assert!(h.tokens().is_empty());
        // Default is window 0 too.
        let d = PenaltyHistory::default();
        assert_eq!(d.window(), 0);
        assert!(d.row(3).is_empty());
        assert!(d.tokens().is_empty());
    }

    #[test]
    fn penalty_history_rows_track_the_trailing_window() {
        let prompt: Vec<u32> = (1..=8).collect();
        let mut h = PenaltyHistory::new(5);
        assert_eq!(h.window(), 5);
        // set_prompt keeps the trailing W of the full prompt AND starts the
        // prefill window: row 0 is suffix_W(P), the first token's history.
        h.set_prompt(&prompt);
        assert_eq!(h.row(0), [4, 5, 6, 7, 8]);
        assert_eq!(h.tokens(), [4, 5, 6, 7, 8]);
        // A window: emitted holds the pending seed once.
        assert_eq!(h.begin_window(&[100]), 5);
        assert_eq!(h.tokens(), [5, 6, 7, 8, 100]);
        h.push_draft(101);
        h.push_draft(102);
        assert_eq!(h.tokens(), [5, 6, 7, 8, 100, 101, 102]);
        assert_eq!(h.row(0), [5, 6, 7, 8, 100]);
        assert_eq!(h.row(1), [6, 7, 8, 100, 101]);
        assert_eq!(h.row(2), [7, 8, 100, 101, 102]);
        assert_eq!(h.row_range(0), 0..5);
        assert_eq!(h.row_range(1), 1..6);
        assert_eq!(h.row_range(2), 2..7);
        for i in 0..=2 {
            assert_eq!(&h.tokens()[h.row_range(i)], h.row(i));
        }
        // rewind_drafts drops the drafts, keeps the base.
        h.rewind_drafts();
        assert_eq!(h.tokens(), [5, 6, 7, 8, 100]);
        assert_eq!(h.row(0), [5, 6, 7, 8, 100]);
        // ... and fresh drafts replace them cleanly.
        h.push_draft(200);
        assert_eq!(h.row(1), [6, 7, 8, 100, 200]);
        // begin_window discards prior drafts and rebuilds from `emitted`.
        h.push_draft(201);
        assert_eq!(h.begin_window(&[100, 103]), 5);
        assert_eq!(h.tokens(), [6, 7, 8, 100, 103]);
        assert_eq!(h.row(0), [6, 7, 8, 100, 103]);
        // Emitted alone fills W: the prompt no longer contributes.
        assert_eq!(h.begin_window(&[10, 11, 12, 13, 14, 15]), 5);
        assert_eq!(h.row(0), [11, 12, 13, 14, 15]);
        h.push_draft(16);
        assert_eq!(h.row(1), [12, 13, 14, 15, 16]);
        assert_eq!(h.row_range(1), 1..6);
        // A new prompt (the next request's prefill) restarts the window and
        // drops the previous request's drafts and emitted tokens.
        h.set_prompt(&[50, 51]);
        assert_eq!(h.row(0), [50, 51]);
        assert_eq!(h.tokens(), [50, 51]);
        // A prompt shorter than W is kept whole and joined by emitted tokens.
        assert_eq!(h.begin_window(&[60]), 3);
        assert_eq!(h.row(0), [50, 51, 60]);
        h.push_draft(61);
        assert_eq!(h.row(1), [50, 51, 60, 61]);
    }

    #[test]
    fn penalty_history_matches_the_reference_suffix_for_every_shape() {
        for window in 1..=9usize {
            for plen in 0..=12u32 {
                for elen in 0..=12u32 {
                    for k in 0..=4u32 {
                        let prompt: Vec<u32> = (0..plen).map(|i| 1000 + i).collect();
                        let emitted: Vec<u32> = (0..elen).map(|i| 2000 + i).collect();
                        let drafts: Vec<u32> = (0..k).map(|i| 3000 + i).collect();
                        let concat = [prompt.clone(), emitted.clone()].concat();
                        let mut h = PenaltyHistory::new(window);
                        h.set_prompt(&prompt);
                        assert_eq!(h.row(0), suffix(&prompt, window));
                        assert_eq!(h.begin_window(&emitted), concat.len().min(window));
                        for &d in &drafts {
                            h.push_draft(d);
                        }
                        for i in 0..=k as usize {
                            let full = [concat.clone(), drafts[..i].to_vec()].concat();
                            let want = suffix(&full, window);
                            let ctx = format!("W{window} plen {plen} elen {elen} k {k} row {i}");
                            assert_eq!(h.row(i), want, "{ctx}");
                            assert_eq!(&h.tokens()[h.row_range(i)], want, "{ctx}");
                        }
                        // The pending seed (last emitted) enters exactly once.
                        if let Some(&seed) = emitted.last() {
                            assert_eq!(h.row(0).iter().filter(|&&t| t == seed).count(), 1);
                        }
                        // The buffer never exceeds base + the window's drafts.
                        assert!(h.tokens().len() <= window + k as usize);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "penalty history row 2")]
    fn penalty_history_row_past_pushed_drafts_panics() {
        let mut h = PenaltyHistory::new(4);
        h.set_prompt(&[1, 2, 3]);
        h.begin_window(&[4]);
        h.push_draft(5);
        // Row 1 is fine (one draft); row 2 would need a second.
        assert_eq!(h.row(1), [2, 3, 4, 5]);
        let _ = h.row(2);
    }

    #[test]
    fn naive_target_sampler_carries_all_penalty_controls() {
        let cfg = SpecRequestConfig {
            temp: 0.7,
            top_p: 0.8,
            top_k: Some(20),
            min_p: 0.0,
            repeat_penalty: 1.1,
            repeat_window: 64,
            presence_penalty: 1.5,
            frequency_penalty: 0.5,
            ..SpecRequestConfig::default()
        };
        let s = naive_target_sampler(&cfg);
        assert_eq!(s.temperature, 0.7);
        assert_eq!(s.top_p, 0.8);
        assert_eq!(s.top_k, Some(20));
        assert_eq!(s.min_p, None);
        assert_eq!(s.repeat_penalty, 1.1);
        assert_eq!(s.repeat_window, 64);
        assert_eq!(s.presence_penalty, 1.5);
        assert_eq!(s.frequency_penalty, 0.5);
        assert!(s.blocked_tokens.is_empty());
        assert_eq!(cfg.penalty_window(), 64);
        // A positive min_p passes through; the defaults stay neutral.
        let s = naive_target_sampler(&SpecRequestConfig {
            min_p: 0.05,
            ..cfg
        });
        assert_eq!(s.min_p, Some(0.05));
        let neutral = naive_target_sampler(&SpecRequestConfig::default());
        assert_eq!(neutral.repeat_penalty, 1.0);
        assert_eq!(neutral.repeat_window, 0);
        assert_eq!(neutral.presence_penalty, 0.0);
        assert_eq!(neutral.frequency_penalty, 0.0);
        assert_eq!(SpecRequestConfig::default().penalty_window(), 0);
    }

    // ---- the penalized target is the host AR sampler's distribution ----

    /// `apply_logit_policy_cpu` + `SparseDist::build_from_logits` over a
    /// non-empty history (prompt + emitted, repeated ids including the
    /// pending seed, crossing the window) follows `sampler::sample_cpu`'s
    /// empirical distribution over independent scrambled seeds.
    fn assert_penalized_target_is_sample_cpu(cfg: &SpecRequestConfig, label: &str) {
        const N: u64 = 40_000;
        // Draws from the process-global AR RNG; hold it for the whole run.
        let _rng = crate::llama::sampler_rng_test_guard();
        let sampler = naive_target_sampler(cfg);
        let spec = SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p);
        let prompt: Vec<u32> = (0..100).map(|i| (splitmix64(i) % 70) as u32).collect();
        let mut emitted: Vec<u32> = (0..60)
            .map(|i| (splitmix64(1_000 + i) % 70) as u32)
            .collect();
        // The pending seed repeats a prompt token.
        *emitted.last_mut().unwrap() = prompt[10];
        let seed = *emitted.last().unwrap();
        let full = [prompt.clone(), emitted.clone()].concat();
        assert!(full.len() > cfg.repeat_window, "history must cross W");
        assert!(full.iter().filter(|&&t| t == seed).count() >= 2);

        let mut row = logits(300, 31_337, 2.5);
        for l in &mut row[..40] {
            // Make the history ids competitive so the penalties bite.
            *l += 1.5;
        }
        let mut penalized = row.clone();
        apply_logit_policy_cpu(&mut penalized, &full, &sampler);
        // The windowed history a speculative loop hands over is bit-identical.
        let mut windowed = Vec::new();
        fill_penalty_history(&mut windowed, &prompt, &emitted, cfg.repeat_window);
        assert_eq!(windowed.len(), cfg.repeat_window);
        let mut via_window = row.clone();
        apply_logit_policy_cpu(&mut via_window, &windowed, &sampler);
        assert!(
            penalized
                .iter()
                .zip(&via_window)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label}: windowed history diverges from the full one"
        );

        let mut p = SparseDist::default();
        p.build_from_logits(&penalized, spec, &mut Vec::new())
            .unwrap();
        let mut neutral = SparseDist::default();
        neutral
            .build_from_logits(&row, spec, &mut Vec::new())
            .unwrap();
        let moved = tv_between(&p, &neutral, row.len());
        assert!(moved > 0.05, "{label}: penalties barely move p ({moved})");

        let mut counts = vec![0u64; row.len()];
        let mut buf = row.clone();
        for trial in 0..N {
            crate::llama::reset_cpu_sampler_rng(scrambled_seed(trial));
            buf.copy_from_slice(&row);
            counts[sample_cpu(&mut buf, &full, &sampler) as usize] += 1;
        }
        assert_follows_n(label, &counts, &p, N);
    }

    #[test]
    fn presence_penalized_target_is_the_sample_cpu_distribution() {
        let (r, pr, f) = PENALTY_CONFIGS[0];
        assert_penalized_target_is_sample_cpu(
            &penalty_cfg(r, pr, f, 128),
            "presence 1.5 repeat 1 frequency 0",
        );
    }

    #[test]
    fn combined_penalized_target_is_the_sample_cpu_distribution() {
        let (r, pr, f) = PENALTY_CONFIGS[1];
        assert_penalized_target_is_sample_cpu(
            &penalty_cfg(r, pr, f, 128),
            "repeat 1.1 presence 1.5 frequency 0.5",
        );
    }

    // ---- penalized speculative windows are lossless against AR ----

    const WIN_VOCAB: usize = 24;
    const WIN_EOS: u32 = 3;
    const WIN_W: usize = 16;
    /// `(prompt, emitted)`; `emitted` ends with the pending seed (7, also in
    /// the prompt). 16 + 5 tokens cross `WIN_W`, and the first five prompt
    /// tokens (including the only 14 and 15) fall out of the window.
    fn win_context() -> (Vec<u32>, Vec<u32>) {
        (
            vec![15, 14, 7, 1, 9, 2, 7, 4, 1, 11, 2, 7, 9, 5, 13, 1],
            vec![8, 1, 9, 2, 7],
        )
    }

    /// Raw logit row after context token `last` (the toy model depends on
    /// the last token only; the penalties supply the history dependence).
    /// Boosted ids are the ones the history repeats, plus EOS.
    fn win_raw(last: u32, salt: u64) -> Vec<f32> {
        let mut row = logits(WIN_VOCAB, ((last as u64) << 16) ^ salt, 1.5);
        for id in [1usize, 3, 7, 14, 15] {
            row[id] += 1.2;
        }
        row
    }

    /// Target row `policy(raw(last), hist)` truncated as AR truncates it, and
    /// the 3-candidate draft distributions, memoized by `(last, history)`.
    struct WinWorld {
        sampler: SamplerConfig,
        spec: SampleSpec,
        q_spec: SampleSpec,
        p_cache: HashMap<(u32, Vec<u32>), SparseDist>,
        q_cache: HashMap<(u32, Vec<u32>), SparseDist>,
    }

    impl WinWorld {
        fn new(cfg: &SpecRequestConfig) -> Self {
            Self {
                sampler: naive_target_sampler(cfg),
                spec: SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p),
                // Three candidates: a support far narrower than p's.
                q_spec: SampleSpec::cpu_ar(1.0, 1.0, Some(3), 0.0),
                p_cache: HashMap::new(),
                q_cache: HashMap::new(),
            }
        }

        fn target(&mut self, last: u32, hist: &[u32]) -> &SparseDist {
            let Self {
                sampler,
                spec,
                p_cache,
                ..
            } = self;
            p_cache.entry((last, hist.to_vec())).or_insert_with(|| {
                let mut row = win_raw(last, 0x71);
                apply_logit_policy_cpu(&mut row, hist, sampler);
                let mut d = SparseDist::default();
                d.build_from_logits(&row, *spec, &mut Vec::new()).unwrap();
                d
            })
        }

        /// Draft distribution: a perturbed copy of the target's raw row,
        /// penalized against `hist` or not.
        fn draft(&mut self, last: u32, hist: &[u32], penalized: bool) -> &SparseDist {
            let Self {
                sampler,
                q_spec,
                q_cache,
                ..
            } = self;
            let key = (last, if penalized { hist.to_vec() } else { Vec::new() });
            q_cache.entry(key).or_insert_with(|| {
                let target_raw = win_raw(last, 0x71);
                let noise = win_raw(last, 0xD7);
                let mut row: Vec<f32> = target_raw
                    .iter()
                    .zip(&noise)
                    .map(|(t, n)| t + 0.8 * n)
                    .collect();
                if penalized {
                    apply_logit_policy_cpu(&mut row, hist, sampler);
                }
                let mut cands: Vec<(u32, f32)> = row
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i as u32, l))
                    .collect();
                let mut d = SparseDist::default();
                d.build_from_candidates(&mut cands, *q_spec).unwrap();
                d
            })
        }
    }

    /// AR distribution of the first two tokens after the pending seed, from
    /// the FULL (unwindowed) history: cell `t1 * (V + 1) + t2`, with
    /// `t2 == V` = "ended on EOS at t1".
    fn win_joint_expected(cfg: &SpecRequestConfig) -> Vec<f64> {
        let (prompt, emitted) = win_context();
        let seed = *emitted.last().unwrap();
        let full = [prompt, emitted].concat();
        let mut world = WinWorld::new(cfg);
        let mut expected = vec![0.0f64; WIN_VOCAB * (WIN_VOCAB + 1)];
        let p0 = world.target(seed, &full).clone();
        for &(t1, a) in p0.entries() {
            if t1 == WIN_EOS {
                expected[t1 as usize * (WIN_VOCAB + 1) + WIN_VOCAB] += a as f64;
                continue;
            }
            let mut hist = full.clone();
            hist.push(t1);
            for &(t2, b) in world.target(t1, &hist).entries() {
                expected[t1 as usize * (WIN_VOCAB + 1) + t2 as usize] += a as f64 * b as f64;
            }
        }
        expected
    }

    fn assert_joint_follows(name: &str, counts: &[u64], expected: &[f64], n: u64) {
        let (stat, df) = chi_square(counts, expected, n);
        let crit = chi_square_critical(df, 4.75);
        let tv = total_variation(counts, expected, n);
        let support = expected.iter().filter(|&&e| e > 0.0).count();
        let bound = (support as f64 / n as f64).sqrt() + 0.005;
        eprintln!(
            "{name}: support={support} chi2={stat:.1} df={df} crit(1e-6)={crit:.1} tv={tv:.4} (<{bound:.4})"
        );
        assert!(stat < crit, "{name}: chi2 {stat} >= {crit} (df {df})");
        assert!(tv < bound, "{name}: total variation {tv} >= {bound}");
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum DraftKind {
        /// q drawn from the penalized draft row.
        Penalized,
        /// q drawn from the unpenalized draft row: it must not matter.
        Unpenalized,
        /// q = delta(argmax of the raw target row), via `set_point_mass`.
        PointMass,
    }

    #[derive(Default, Debug)]
    struct WindowCoverage {
        /// Windows whose first rejection was at row 0, 1, 2.
        rejected_at: [u64; 3],
        /// Windows that accepted all `k >= 1` drafts and drew the bonus.
        all_accept: u64,
        /// Windows that ended on an accepted EOS draft.
        eos_accepted: u64,
        /// Windows with no drafts: the bonus is the only token.
        k0: u64,
        /// Rejections whose residual token is outside q's support.
        residual_outside_q: u64,
    }

    /// Run `trials` windows (depth `trial % 4`, so k = 0..=3) through
    /// `accept_sampled_prefix` with `PenaltyHistory` rows as the target fill,
    /// and tally the first two emitted tokens of the stream. When the window
    /// emits one token, the second is drawn at the next window's row 0 from a
    /// history rebuilt out of `emitted + t1` (the loop's rebuild-per-window).
    /// `use_history == false` is the negative control: unpenalized targets.
    fn run_windows(
        cfg: &SpecRequestConfig,
        kind: DraftKind,
        use_history: bool,
        trials: u64,
        rng_seed: u64,
    ) -> (Vec<u64>, WindowCoverage) {
        let (prompt, emitted0) = win_context();
        let seed_tok = *emitted0.last().unwrap();
        let mut world = WinWorld::new(cfg);
        let mut rng = SpecRng::new(rng_seed);
        let mut hist = PenaltyHistory::new(cfg.penalty_window());
        hist.set_prompt(&prompt);
        let mut next = PenaltyHistory::new(cfg.penalty_window());
        next.set_prompt(&prompt);
        let mut counts = vec![0u64; WIN_VOCAB * (WIN_VOCAB + 1)];
        let mut cov = WindowCoverage::default();
        let mut target = SparseDist::default();
        for trial in 0..trials {
            let k = (trial % 4) as usize;
            hist.begin_window(&emitted0);
            let mut drafts: Vec<u32> = Vec::with_capacity(3);
            let mut qs: Vec<SparseDist> = Vec::with_capacity(3);
            let mut last = seed_tok;
            for i in 0..k {
                let q = match kind {
                    DraftKind::PointMass => {
                        let mut d = SparseDist::default();
                        d.set_point_mass(crate::llama::argmax(&win_raw(last, 0x71)));
                        d
                    }
                    DraftKind::Penalized => world.draft(last, hist.row(i), true).clone(),
                    DraftKind::Unpenalized => world.draft(last, &[], false).clone(),
                };
                let x = q.sample(rng.next_f32());
                // Kept drafts only; the next row's history includes it.
                hist.push_draft(x);
                drafts.push(x);
                qs.push(q);
                last = x;
            }
            let res = accept_sampled_prefix(
                &drafts,
                &qs,
                Some(WIN_EOS),
                &mut rng,
                &mut target,
                |row, out| {
                    let last = if row == 0 { seed_tok } else { drafts[row - 1] };
                    let h: &[u32] = if use_history { hist.row(row) } else { &[] };
                    out.clone_from(world.target(last, h));
                    Ok(())
                },
            )
            .unwrap();

            if k == 0 {
                cov.k0 += 1;
            }
            let rejected = res.accepted < k && res.committed.len() == res.accepted + 1;
            if rejected {
                cov.rejected_at[res.accepted] += 1;
                if qs[res.accepted].prob(*res.committed.last().unwrap()) == 0.0 {
                    cov.residual_outside_q += 1;
                }
            } else if k > 0 && res.accepted == k && res.committed.len() == k + 1 {
                cov.all_accept += 1;
            }
            if res.accepted > 0 && res.hit_eos && res.committed.len() == res.accepted {
                cov.eos_accepted += 1;
            }

            let t1 = res.committed[0];
            let t2 = if t1 == WIN_EOS {
                WIN_VOCAB
            } else if let Some(&t2) = res.committed.get(1) {
                t2 as usize
            } else {
                let mut e = emitted0.clone();
                e.push(t1);
                next.begin_window(&e);
                let h: &[u32] = if use_history { next.row(0) } else { &[] };
                world.target(t1, h).sample(rng.next_f32()) as usize
            };
            counts[t1 as usize * (WIN_VOCAB + 1) + t2] += 1;
        }
        (counts, cov)
    }

    fn assert_window_is_lossless(kind: DraftKind) {
        const N: u64 = 120_000;
        for (n, &(r, pr, f)) in PENALTY_CONFIGS.iter().enumerate() {
            let cfg = penalty_cfg(r, pr, f, WIN_W);
            let expected = win_joint_expected(&cfg);
            let (counts, cov) = run_windows(&cfg, kind, true, N, 5 + n as u64);
            let name = format!("{kind:?} repeat {r} presence {pr} frequency {f}");
            assert_joint_follows(&name, &counts, &expected, N);
            eprintln!("{name}: {cov:?}");
            // Depth-2 rejections need two accepted drafts first: q far from p
            // (unpenalized / point mass) may never get there; the penalized
            // kind covers it, the shallower depths are required of every kind.
            let depths = if kind == DraftKind::Penalized { 3 } else { 2 };
            assert!(
                cov.rejected_at[..depths].iter().all(|&c| c > 0),
                "{name}: rejection never reached depths 0..{depths}: {cov:?}"
            );
            assert!(cov.all_accept > 0, "{name}: no all-accept bonus: {cov:?}");
            assert!(cov.eos_accepted > 0, "{name}: no accepted EOS draft: {cov:?}");
            assert!(cov.k0 > 0, "{name}: no k = 0 window: {cov:?}");
            assert!(
                cov.residual_outside_q > 0,
                "{name}: no residual outside q's support: {cov:?}"
            );
        }
    }

    /// The AR reference itself: sequential `sample_cpu` over the growing,
    /// FULL history (independently scrambled seed per draw) follows the
    /// analytic joint every window test compares against.
    #[test]
    fn sample_cpu_ar_stream_follows_the_penalized_joint() {
        const N: u64 = 60_000;
        let _rng = crate::llama::sampler_rng_test_guard();
        let (prompt, emitted) = win_context();
        let seed = *emitted.last().unwrap();
        let full = [prompt, emitted].concat();
        for (n, &(r, pr, f)) in PENALTY_CONFIGS.iter().enumerate() {
            let cfg = penalty_cfg(r, pr, f, WIN_W);
            let sampler = naive_target_sampler(&cfg);
            let expected = win_joint_expected(&cfg);
            let mut counts = vec![0u64; WIN_VOCAB * (WIN_VOCAB + 1)];
            for trial in 0..N {
                crate::llama::reset_cpu_sampler_rng(scrambled_seed(2 * trial));
                let t1 = sample_cpu(&mut win_raw(seed, 0x71), &full, &sampler);
                let t2 = if t1 == WIN_EOS {
                    WIN_VOCAB
                } else {
                    crate::llama::reset_cpu_sampler_rng(scrambled_seed(2 * trial + 1));
                    let mut ctx = full.clone();
                    ctx.push(t1);
                    sample_cpu(&mut win_raw(t1, 0x71), &ctx, &sampler) as usize
                };
                counts[t1 as usize * (WIN_VOCAB + 1) + t2] += 1;
            }
            assert_joint_follows(
                &format!("sample_cpu AR #{n} repeat {r} presence {pr} frequency {f}"),
                &counts,
                &expected,
                N,
            );
        }
    }

    #[test]
    fn penalized_window_with_penalized_q_is_lossless() {
        assert_window_is_lossless(DraftKind::Penalized);
    }

    /// q need not be penalized: acceptance uses the q the draft came from.
    #[test]
    fn penalized_window_with_unpenalized_q_is_lossless() {
        assert_window_is_lossless(DraftKind::Unpenalized);
    }

    #[test]
    fn penalized_window_with_point_mass_q_is_lossless() {
        assert_window_is_lossless(DraftKind::PointMass);
    }

    /// Negative control: verifying against unpenalized target rows is far
    /// from the penalized AR joint, so the lossless tests above can fail.
    #[test]
    fn unpenalized_target_rows_are_detected() {
        const N: u64 = 60_000;
        let (r, pr, f) = PENALTY_CONFIGS[1];
        let cfg = penalty_cfg(r, pr, f, WIN_W);
        let expected = win_joint_expected(&cfg);
        let (counts, _) = run_windows(&cfg, DraftKind::Penalized, false, N, 99);
        let tv = total_variation(&counts, &expected, N);
        eprintln!("unpenalized control: tv={tv:.4}");
        assert!(tv > 0.05, "unpenalized rows went undetected (tv {tv})");
    }

    // ---- point-mass drafts ----

    #[test]
    fn point_mass_draft_accepts_iff_u_below_p_and_never_resamples_itself() {
        let spec = SampleSpec::cpu_ar(0.7, 1.0, Some(20), 0.0);
        let mut p = SparseDist::default();
        p.build_from_logits(&logits(WIN_VOCAB, 77, 1.5), spec, &mut Vec::new())
            .unwrap();
        assert!(p.entries().len() > 3);
        let c = p.entries()[1].0;
        let pc = p.prob(c);
        assert!(pc > 0.0 && pc < 1.0);
        let mut delta = SparseDist::default();
        delta.set_point_mass(c);
        assert_eq!(delta.entries(), [(c, 1.0)]);

        let mut rng = SpecRng::new(4242);
        let mut counts = vec![0u64; WIN_VOCAB];
        let (mut accepts, mut rejects) = (0u64, 0u64);
        for _ in 0..100_000 {
            let mut probe = rng;
            let u = probe.next_f32();
            match verify_sampled_draft(&p, &delta, c, &mut rng) {
                DraftVerdict::Accept => {
                    assert!(u < pc, "accepted with u {u} >= p(c) {pc}");
                    accepts += 1;
                }
                DraftVerdict::Reject(t) => {
                    assert!(u >= pc, "rejected with u {u} < p(c) {pc}");
                    assert_ne!(t, c, "the residual returned the point-mass token");
                    assert!(p.prob(t) > 0.0, "residual left p's support");
                    counts[t as usize] += 1;
                    rejects += 1;
                }
            }
        }
        // Accept rate = p(c); the rejection draw is p with c removed.
        let rate = accepts as f64 / (accepts + rejects) as f64;
        assert!((rate - pc as f64).abs() < 0.01, "accept {rate} vs {pc}");
        let mut rest = p.clone();
        let kept: Vec<(u32, f32)> = p
            .entries()
            .iter()
            .filter(|&&(t, _)| t != c)
            .map(|&(t, v)| (t, v / (1.0 - pc)))
            .collect();
        rest.entries.clear();
        rest.entries.extend(kept);
        assert_follows_n("point-mass residual", &counts, &rest, rejects);

        // A token outside p's support is never accepted.
        let outside = (0..WIN_VOCAB as u32).find(|&t| p.prob(t) == 0.0).unwrap();
        let mut delta_out = SparseDist::default();
        delta_out.set_point_mass(outside);
        for _ in 0..10_000 {
            match verify_sampled_draft(&p, &delta_out, outside, &mut rng) {
                DraftVerdict::Accept => panic!("accepted a zero-probability draft"),
                DraftVerdict::Reject(t) => assert!(p.prob(t) > 0.0),
            }
        }
        // p itself a point mass on c: always accepted.
        let mut pm = SparseDist::default();
        pm.set_point_mass(c);
        for _ in 0..10_000 {
            assert_eq!(
                verify_sampled_draft(&pm, &delta, c, &mut rng),
                DraftVerdict::Accept
            );
        }
    }

    // ---- why the policy must precede the pool gather ----

    /// Counterexample: gathering the raw top-20 and penalizing only those
    /// candidates is NOT the AR distribution. The penalty demotes ids that
    /// were in the raw top-20, and ids outside it (which AR's policy-first
    /// gather promotes into the pool) can never enter.
    #[test]
    fn policy_must_precede_the_pool_gather() {
        let vocab = 40usize;
        let mut raw = vec![9.0f32; vocab];
        for (i, l) in raw[..20].iter_mut().enumerate() {
            *l = 10.0 - 0.05 * i as f32;
        }
        let history = [0u32, 1, 2, 3, 4];
        let cfg = SpecRequestConfig {
            temp: 0.7,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            presence_penalty: 3.0,
            repeat_window: 128,
            ..SpecRequestConfig::default()
        };
        let sampler = naive_target_sampler(&cfg);
        let spec = SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p);

        // Policy, then gather (what AR does).
        let mut policy_first = raw.clone();
        apply_logit_policy_cpu(&mut policy_first, &history, &sampler);
        let mut right = SparseDist::default();
        right
            .build_from_logits(&policy_first, spec, &mut Vec::new())
            .unwrap();

        // Gather the raw top-20, then penalize only the gathered candidates.
        let mut ranked: Vec<u32> = (0..vocab as u32).collect();
        ranked.sort_by(|&a, &b| {
            raw[b as usize]
                .total_cmp(&raw[a as usize])
                .then(a.cmp(&b))
        });
        let ids = &ranked[..20];
        assert!(ids.iter().all(|&t| t < 20), "raw top-20 is ids 0..20");
        let mut values: Vec<f32> = ids.iter().map(|&t| raw[t as usize]).collect();
        apply_logit_policy_candidates_cpu(ids, &mut values, &history, &sampler);
        // The per-candidate arithmetic is exact...
        for (&id, &v) in ids.iter().zip(&values) {
            assert_eq!(v.to_bits(), policy_first[id as usize].to_bits());
        }
        // ... but the pool it feeds is the wrong one.
        let mut cands: Vec<(u32, f32)> = ids.iter().copied().zip(values).collect();
        let mut wrong = SparseDist::default();
        wrong.build_from_candidates(&mut cands, spec).unwrap();

        assert!(
            right.entries().iter().any(|&(t, _)| t >= 20),
            "policy-first must promote ids outside the raw top-20"
        );
        assert!(wrong.entries().iter().all(|&(t, _)| t < 20));
        let tv = tv_between(&right, &wrong, vocab);
        assert!(tv > 0.05, "gather-first matched policy-first (tv {tv})");
    }
}
