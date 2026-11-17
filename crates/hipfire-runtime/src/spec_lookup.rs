// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure chooser for the lookup/learned-draft verify window (wide-lookup PLAN §3).
//!
//! A verify window is one of three kinds:
//!
//! * **Native** — the incumbent window: the learned drafter's candidates only.
//! * **Copy** — a pre-draft decision (`model_drafts == 0`): lookup candidates replace every
//!   learned candidate.
//! * **Tail** — a post-draft decision (`model_drafts == K > 0`): the first `K` learned candidates
//!   are kept and a lookup tail is appended (`1 + K + L <= LOOKUP_MAX_ROWS`).
//!
//! Everything here is plain CPU arithmetic: no GPU types, no config, no global clock. Measured
//! prices and measured per-depth acceptance are the only inputs; an unmeasured price or an
//! unobserved source refuses (`Native`), it is never extrapolated or invented.
//!
//! ## Acceptance evidence ([`DepthTable`])
//!
//! A [`DepthTable`] is one source's per-depth conditional acceptance: `conditional(d)` is
//! P(candidate `d` accepted | candidates `1..d` accepted). The update is the production Qwen4
//! copy-lane update, replicated verbatim so the two controllers measure the same thing:
//!
//! * `crates/hipfire-arch-qwen4/src/mtp_spec.rs:1740-1770` `observe_agreement_counts`: depths
//!   `< compared = min(matched + 1, horizon)` decay by `decay` and take the observation
//!   (`depth < matched` accepted); the remaining depths relax toward the already-updated
//!   next-shallower depth at the prior's weight — `MTP_AGREEMENT_RELAX_UP = 0.8` when the target
//!   is above the depth's own ratio, `MTP_AGREEMENT_RELAX = 0.95` otherwise
//!   (`mtp_spec.rs:1442-1475`).
//! * `mtp_spec.rs:5132-5165` `observe_ngram_shadow`: `ngram_last_first_accepted = matched >= 1`,
//!   `horizon = min(proposal_len, committed_len)`.
//! * `/home/kaden/qcal/release-0.4.3/ngram-copy-research/tune_grid.py:33-40` `update`: the same
//!   recurrence as the reference simulation (prior `(1.6, 2.0)`).
//!
//! The prior is deliberately weak (weight 2): the chooser's evidence gate (first-depth acceptance
//! with decayed weight `>= min_evidence`, plus `last_first_accepted`) cannot be satisfied by the
//! prior alone, so a wide window is only ever chosen on observed windows.

/// Verify rows of one window including the seed.
pub const LOOKUP_MAX_ROWS: usize = 16;
/// Candidates of one window (rows minus the seed).
pub const LOOKUP_MAX_CANDIDATES: usize = LOOKUP_MAX_ROWS - 1;

/// Samples kept per price cell (median of the last `PRICE_WINDOW`).
const PRICE_WINDOW: usize = 8;
/// Context buckets of the price book: `<=2K, <=8K, <=16K, <=32K, >32K` (the H0 trace buckets).
pub const PRICE_BUCKET_EDGES: [usize; 4] = [2048, 8192, 16384, 32768];
const PRICE_BUCKETS: usize = PRICE_BUCKET_EDGES.len() + 1;
const PRICE_KINDS: usize = 3;
const PRICE_ROWS: usize = LOOKUP_MAX_ROWS + 1;

/// Relax rates of depths a window did not compare (`mtp_spec.rs:1465,1475`).
const AGREEMENT_RELAX: f32 = 0.95;
const AGREEMENT_RELAX_UP: f32 = 0.8;
/// Relative tolerance under which two rates tie (shorter wins).
const TIE_EPS: f64 = 1e-9;

/// Prior of both sources (accepted mass, total weight): ratio 0.8 at weight 2, the Qwen4 prior
/// (`MTP_AGREEMENT_PRIOR`, `mtp_spec.rs:1451`).
pub const LOOKUP_PRIOR: (f32, f32) = (1.6, 2.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupAction {
    Native,
    Copy { take: usize },
    Tail { take: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupKind {
    Native,
    Copy,
    Tail,
}

impl LookupKind {
    fn index(self) -> usize {
        match self {
            LookupKind::Native => 0,
            LookupKind::Copy => 1,
            LookupKind::Tail => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LookupBudget {
    /// Learned candidates the incumbent window would verify (its `b - 1`).
    pub native_candidates: usize,
    /// `0` = pre-draft Copy decision; `K > 0` = post-draft Tail decision after `K` kept learned candidates.
    pub model_drafts: usize,
    /// Lookup candidates offered at this decision.
    pub lookup_available: usize,
    /// Verify rows allowed including the seed (`<= LOOKUP_MAX_ROWS`).
    pub max_rows: usize,
    /// Tokens this window may still emit (accepted + bonus).
    pub remaining_emit: usize,
    pub context_rows_left: usize,
}

/// Measured window prices in microseconds. Arrays are indexed by verify rows (including the
/// seed), `0..=LOOKUP_MAX_ROWS`; `None` / non-finite / `<= 0` means unavailable.
#[derive(Clone, Copy, Debug)]
pub struct LookupCosts<'a> {
    pub native_us: Option<f64>,
    pub copy_us: &'a [Option<f64>; LOOKUP_MAX_ROWS + 1],
    pub tail_us: &'a [Option<f64>; LOOKUP_MAX_ROWS + 1],
}

/// Per-depth conditional acceptance of one source (see the module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct DepthTable {
    prior: (f32, f32),
    depth: [(f32, f32); LOOKUP_MAX_CANDIDATES],
    last_first: bool,
}

impl DepthTable {
    /// `prior_accept` is the prior accepted mass, `prior_weight` the prior total weight (ratio
    /// `prior_accept / prior_weight`), every depth starting at that prior. An unusable prior
    /// (non-finite or non-positive weight) degrades to a zero-acceptance prior.
    pub fn new(prior_accept: f32, prior_weight: f32) -> Self {
        let weight = if prior_weight.is_finite() && prior_weight > 0.0 {
            prior_weight
        } else {
            1.0
        };
        let accept = if prior_accept.is_finite() {
            prior_accept.clamp(0.0, weight)
        } else {
            0.0
        };
        let prior = (accept, weight);
        Self {
            prior,
            depth: [prior; LOOKUP_MAX_CANDIDATES],
            last_first: false,
        }
    }

    /// One window's observation: candidate depths `1..=min(matched + 1, compared)` were compared
    /// (the first `matched` accepted, the next one — if compared — rejected); deeper depths were
    /// never reached and are censored: they relax toward the next-shallower depth instead of
    /// taking an observation. `compared == 0` observes nothing.
    pub fn observe(&mut self, matched: usize, compared: usize, decay: f32) {
        let compared = compared
            .min(matched.saturating_add(1))
            .min(LOOKUP_MAX_CANDIDATES);
        if compared == 0 {
            return;
        }
        let matched = matched.min(compared);
        let decay = if decay.is_nan() {
            1.0
        } else {
            decay.clamp(0.0, 1.0)
        };
        let prior_total = self.prior.1;
        let mut shallower = self.prior;
        for (index, entry) in self.depth.iter_mut().enumerate() {
            let (accepted, total) = entry;
            if index < compared {
                *accepted *= decay;
                *total *= decay;
                *total += 1.0;
                *accepted += f32::from(u8::from(index < matched));
            } else {
                let target = shallower.0 / shallower.1.max(f32::MIN_POSITIVE);
                let relax = if target > *accepted / total.max(f32::MIN_POSITIVE) {
                    AGREEMENT_RELAX_UP
                } else {
                    AGREEMENT_RELAX
                };
                *accepted = *accepted * relax + prior_total * target * (1.0 - relax);
                *total = *total * relax + prior_total * (1.0 - relax);
            }
            shallower = *entry;
        }
        self.last_first = matched >= 1;
    }

    /// P(candidate `depth` accepted | its prefix accepted), `depth` 1-based; `0.0` out of range.
    pub fn conditional(&self, depth: usize) -> f32 {
        match depth.checked_sub(1).and_then(|i| self.depth.get(i)) {
            Some(&(accepted, total)) => (accepted / total.max(f32::MIN_POSITIVE)).clamp(0.0, 1.0),
            None => 0.0,
        }
    }

    /// Decayed observation weight behind `depth` (1-based); `0.0` out of range.
    pub fn weight(&self, depth: usize) -> f32 {
        match depth.checked_sub(1).and_then(|i| self.depth.get(i)) {
            Some(&(_, total)) => total,
            None => 0.0,
        }
    }

    /// Whether the last observed window accepted its first candidate (`false` before any
    /// observation: the prior never counts as evidence).
    pub fn last_first_accepted(&self) -> bool {
        self.last_first
    }

    /// Expected tokens of a window of `n` candidates: `1 + sum_{d<=n} prod_{i<=d} a_i`.
    pub fn expected(&self, n: usize) -> f32 {
        let mut product = 1.0f64;
        let mut expected = 1.0f64;
        for depth in 1..=n.min(LOOKUP_MAX_CANDIDATES) {
            product *= f64::from(self.conditional(depth));
            expected += product;
        }
        expected as f32
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LookupAcceptance<'a> {
    pub model: &'a DepthTable,
    pub lookup: &'a DepthTable,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookupPolicy {
    /// A wide window is taken only if its rate is `>= margin x` the native window's rate.
    pub margin: f32,
    /// Lookup first-depth acceptance required to take a wide window.
    pub min_first_accept: f32,
    /// Decayed observation weight of the lookup's first depth required to take a wide window.
    /// Must exceed the prior weight, or the prior alone could satisfy the gate.
    pub min_evidence: f32,
    /// Fewest lookup candidates of a wide window.
    pub n_min: usize,
    /// Per-window decay of the acceptance tables.
    pub decay: f32,
}

impl Default for LookupPolicy {
    /// Calibrated on the real dense R9700 H0 trace (grid + declared selection rule in
    /// `wide-lookup/policy-calibration.md`): margin 1.25, first-depth acceptance 0.9, evidence
    /// weight 3 (> the prior's 2, so two observed windows are the minimum), `n_min` 1, decay
    /// 0.875 (the Qwen4 gfx1201 value; dense has no measured decay). Simulated on one diagnostic
    /// pass and tuned on the corpus it is measured on: H2 must validate on fresh prompts.
    fn default() -> Self {
        Self {
            margin: 1.25,
            min_first_accept: 0.9,
            min_evidence: 3.0,
            n_min: 1,
            decay: 0.875,
        }
    }
}

fn valid_price(us: Option<f64>) -> Option<f64> {
    us.filter(|v| v.is_finite() && *v > 0.0)
}

/// Candidates a wide window may take after every budget clamp, or `None` when fewer than
/// `n_min` fit. Clamps: offered, rows (seed + `K` kept learned candidates), `remaining_emit`
/// (accepted candidates + bonus), `context_rows_left`.
fn take_limit(budget: &LookupBudget, policy: &LookupPolicy) -> Option<usize> {
    let prefix = 1 + budget.model_drafts; // seed + kept learned candidates
    let by_rows = budget.max_rows.min(LOOKUP_MAX_ROWS).checked_sub(prefix)?;
    let by_emit = budget.remaining_emit.checked_sub(prefix)?; // `remaining_emit - 1 - K`
    let by_context = budget.context_rows_left.checked_sub(prefix)?;
    let take = budget
        .lookup_available
        .min(by_rows)
        .min(by_emit)
        .min(by_context);
    (take >= policy.n_min.max(1)).then_some(take)
}

/// The lookup source's evidence gate: first-depth acceptance with enough decayed weight, and the
/// last observed window accepted its first candidate. Written as positive comparisons so NaN
/// policy values refuse.
fn evidence_ok(lookup: &DepthTable, policy: &LookupPolicy) -> bool {
    lookup.last_first_accepted()
        && lookup.conditional(1) >= policy.min_first_accept
        && lookup.weight(1) >= policy.min_evidence
}

/// Expected tokens of a wide window with `take` lookup candidates (PLAN §3.2):
/// Copy `= lookup.expected(take)`; Tail `= E_model(K) + S_K * sum_{j<=L} prod_{i<=j} a_lookup_i`
/// with `S_K` the model-prefix survival. Returns `(expected, lookup_gain)`; a Tail whose lookup
/// gain is zero (first model prefix certain to fail) never beats the native window.
fn wide_expected(
    budget: &LookupBudget,
    acceptance: &LookupAcceptance<'_>,
    take: usize,
) -> (f64, f64) {
    let k = budget.model_drafts;
    if k == 0 {
        let e = f64::from(acceptance.lookup.expected(take));
        return (e, e - 1.0);
    }
    let model_e = f64::from(acceptance.model.expected(k));
    let mut survival = 1.0f64;
    for depth in 1..=k {
        survival *= f64::from(acceptance.model.conditional(depth));
    }
    let mut product = 1.0f64;
    let mut tail_sum = 0.0f64;
    for depth in 1..=take {
        product *= f64::from(acceptance.lookup.conditional(depth));
        tail_sum += product;
    }
    let gain = survival * tail_sum;
    (model_e + gain, gain)
}

/// Native rate (expected tokens per microsecond) of the incumbent window.
fn native_rate(budget: &LookupBudget, native_us: f64, acceptance: &LookupAcceptance<'_>) -> f64 {
    f64::from(acceptance.model.expected(budget.native_candidates)) / native_us
}

/// Choose the window. Pure: clamps first, then the evidence gate, then the best-rate take per
/// the priced rows; the wide window is taken only if its rate is `>= margin x` the native
/// rate. Any absent / non-finite / `<= 0` price refuses that row (and the native price refuses
/// everything); a zero proposal is `Native`. `Copy` only when `model_drafts == 0`, `Tail` only
/// when `> 0`; the shorter take wins ties.
pub fn choose_lookup(
    budget: LookupBudget,
    costs: &LookupCosts<'_>,
    acceptance: &LookupAcceptance<'_>,
    policy: &LookupPolicy,
) -> LookupAction {
    let Some(take_max) = take_limit(&budget, policy) else {
        return LookupAction::Native;
    };
    let Some(native_us) = valid_price(costs.native_us) else {
        return LookupAction::Native;
    };
    if !evidence_ok(acceptance.lookup, policy) {
        return LookupAction::Native;
    }
    let native = native_rate(&budget, native_us, acceptance);
    let prices = if budget.model_drafts == 0 {
        costs.copy_us
    } else {
        costs.tail_us
    };
    let mut best: Option<(usize, f64)> = None;
    for take in policy.n_min.max(1)..=take_max {
        let rows = 1 + budget.model_drafts + take;
        let Some(us) = valid_price(prices[rows]) else {
            continue;
        };
        let (expected, gain) = wide_expected(&budget, acceptance, take);
        if !(gain > 0.0) {
            continue;
        }
        let rate = expected / us;
        if !rate.is_finite() {
            continue;
        }
        if best.map_or(true, |(_, best_rate)| rate > best_rate * (1.0 + TIE_EPS)) {
            best = Some((take, rate));
        }
    }
    match best {
        Some((take, rate)) if rate >= f64::from(policy.margin) * native => {
            if budget.model_drafts == 0 {
                LookupAction::Copy { take }
            } else {
                LookupAction::Tail { take }
            }
        }
        _ => LookupAction::Native,
    }
}

#[derive(Clone, Copy, Debug)]
struct PriceCell {
    discarded_first: bool,
    len: usize,
    head: usize,
    samples: [f64; PRICE_WINDOW],
}

impl PriceCell {
    const EMPTY: Self = Self {
        discarded_first: false,
        len: 0,
        head: 0,
        samples: [0.0; PRICE_WINDOW],
    };

    fn observe(&mut self, us: f64) {
        if !self.discarded_first {
            // First observation of a cell carries first-bind / graph-capture cost.
            self.discarded_first = true;
            return;
        }
        self.samples[self.head] = us;
        self.head = (self.head + 1) % PRICE_WINDOW;
        self.len = (self.len + 1).min(PRICE_WINDOW);
    }

    fn median(&self) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let mut sorted = self.samples;
        let live = &mut sorted[..self.len];
        live.sort_by(f64::total_cmp);
        Some(if self.len % 2 == 1 {
            live[self.len / 2]
        } else {
            (live[self.len / 2 - 1] + live[self.len / 2]) / 2.0
        })
    }
}

fn price_bucket(ctx: usize) -> usize {
    PRICE_BUCKET_EDGES
        .iter()
        .position(|&edge| ctx <= edge)
        .unwrap_or(PRICE_BUCKET_EDGES.len())
}

/// Measured full-window prices, one cell per `(kind, rows, context bucket)`. The first
/// observation of each cell is discarded (first-bind / capture); the price is the median of the
/// last `PRICE_WINDOW` kept observations. Fixed storage, load-scoped.
#[derive(Clone, Debug)]
pub struct LookupPriceBook {
    cells: [[[PriceCell; PRICE_BUCKETS]; PRICE_ROWS]; PRICE_KINDS],
}

impl Default for LookupPriceBook {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupPriceBook {
    pub fn new() -> Self {
        Self {
            cells: [[[PriceCell::EMPTY; PRICE_BUCKETS]; PRICE_ROWS]; PRICE_KINDS],
        }
    }

    /// Record one window's wall time. Rows outside `1..=LOOKUP_MAX_ROWS` and non-finite or
    /// non-positive times are ignored (and never consume a cell's discarded first observation).
    pub fn observe(&mut self, kind: LookupKind, rows: usize, ctx: usize, us: f64) {
        if !(1..=LOOKUP_MAX_ROWS).contains(&rows) || !us.is_finite() || us <= 0.0 {
            return;
        }
        self.cells[kind.index()][rows][price_bucket(ctx)].observe(us);
    }

    pub fn price(&self, kind: LookupKind, rows: usize, ctx: usize) -> Option<f64> {
        if rows == 0 || rows > LOOKUP_MAX_ROWS {
            return None;
        }
        self.cells[kind.index()][rows][price_bucket(ctx)].median()
    }

    /// Copy/Tail prices of every row count at `ctx`. A row without its own price takes the
    /// Native price at the same rows and bucket iff `share_native` (dense: identical draft and
    /// verify program per row count).
    pub fn fill(
        &self,
        ctx: usize,
        share_native: bool,
        copy: &mut [Option<f64>; LOOKUP_MAX_ROWS + 1],
        tail: &mut [Option<f64>; LOOKUP_MAX_ROWS + 1],
    ) {
        copy[0] = None;
        tail[0] = None;
        for rows in 1..=LOOKUP_MAX_ROWS {
            let native = if share_native {
                self.price(LookupKind::Native, rows, ctx)
            } else {
                None
            };
            copy[rows] = self.price(LookupKind::Copy, rows, ctx).or(native);
            tail[rows] = self.price(LookupKind::Tail, rows, ctx).or(native);
        }
    }
}

/// One finished window, as the controller learns from it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowOutcome {
    pub kind: LookupKind,
    /// Verify rows including the seed.
    pub rows: usize,
    /// Learned candidates verified first (Native: `rows - 1`; Copy: `0`; Tail: `K`).
    pub model_drafts: usize,
    pub lookup_len: usize,
    /// Raw accepted candidates.
    pub accepted: usize,
    /// Native/Tail-unused: `(matched, available)` of the pre-window copy proposal against the
    /// window's committed tokens. Only a Native window's shadow is observed.
    pub shadow: Option<(usize, usize)>,
    pub ctx: usize,
    pub wall_us: f64,
}

/// Request-local acceptance tables + load-scoped price book + policy.
#[derive(Clone, Debug)]
pub struct LookupController {
    model: DepthTable,
    lookup: DepthTable,
    book: LookupPriceBook,
    policy: LookupPolicy,
    share_native: bool,
}

impl LookupController {
    pub fn new(policy: LookupPolicy, share_native_prices: bool) -> Self {
        Self {
            model: DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1),
            lookup: DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1),
            book: LookupPriceBook::new(),
            policy,
            share_native: share_native_prices,
        }
    }

    /// Reset the acceptance tables for a new request; prices are load-scoped and kept.
    pub fn begin_request(&mut self) {
        self.model = DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1);
        self.lookup = DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1);
    }

    fn costs_at(
        &self,
        budget: &LookupBudget,
        ctx: usize,
    ) -> (
        [Option<f64>; PRICE_ROWS],
        [Option<f64>; PRICE_ROWS],
        Option<f64>,
    ) {
        let mut copy = [None; PRICE_ROWS];
        let mut tail = [None; PRICE_ROWS];
        self.book.fill(ctx, self.share_native, &mut copy, &mut tail);
        let native = self
            .book
            .price(LookupKind::Native, budget.native_candidates + 1, ctx);
        (copy, tail, native)
    }

    /// The window to run: prices from the book, then [`choose_lookup`].
    pub fn plan(&self, budget: LookupBudget, ctx: usize) -> LookupAction {
        let (copy, tail, native_us) = self.costs_at(&budget, ctx);
        let costs = LookupCosts {
            native_us,
            copy_us: &copy,
            tail_us: &tail,
        };
        let acceptance = LookupAcceptance {
            model: &self.model,
            lookup: &self.lookup,
        };
        choose_lookup(budget, &costs, &acceptance, &self.policy)
    }

    /// `Some(rows)` iff a Copy would pass every gate under `share_native` pricing — the clamps,
    /// the lookup evidence gate, the native window priced, and (a price-free necessary
    /// condition) the longest legal Copy's expected tokens `>= margin x` the native window's —
    /// but its row count is unpriced at this bucket. The caller then runs THIS native window at
    /// `rows`, which prices that cell for later Copy decisions. `None` when `share_native` is
    /// off (a native window would not price a Copy), for a Tail decision, or when nothing is
    /// missing.
    pub fn price_probe_rows(&self, budget: LookupBudget, ctx: usize) -> Option<usize> {
        if !self.share_native || budget.model_drafts != 0 {
            return None;
        }
        let take = take_limit(&budget, &self.policy)?;
        let (copy, _tail, native_us) = self.costs_at(&budget, ctx);
        valid_price(native_us)?;
        if !evidence_ok(&self.lookup, &self.policy) {
            return None;
        }
        let rows = 1 + take;
        if valid_price(copy[rows]).is_some() {
            return None;
        }
        let wide = f64::from(self.lookup.expected(take));
        let native = f64::from(self.model.expected(budget.native_candidates));
        (wide >= f64::from(self.policy.margin) * native).then_some(rows)
    }

    /// Learn from one finished window (the shared observation rules; S3b must feed exactly
    /// these): every window feeds the price book; acceptance tables are source-separated and
    /// censored at the first rejected depth.
    ///
    /// * Native: `model.observe(accepted, min(accepted+1, rows-1))`; with `shadow = (m, avail)`,
    ///   `lookup.observe(m, min(m+1, avail, accepted+1))`.
    /// * Copy: `lookup.observe(accepted, min(accepted+1, lookup_len))`; model unobserved.
    /// * Tail(K, L): `model.observe(min(accepted, K), min(accepted+1, K))`; if `accepted >= K`,
    ///   `lookup.observe(accepted-K, min(accepted-K+1, L))`.
    pub fn observe(&mut self, o: &WindowOutcome) {
        let decay = self.policy.decay;
        match o.kind {
            LookupKind::Native => {
                let drafts = o.rows.saturating_sub(1);
                self.model
                    .observe(o.accepted, (o.accepted + 1).min(drafts), decay);
                if let Some((matched, available)) = o.shadow {
                    self.lookup.observe(
                        matched,
                        (matched + 1).min(available).min(o.accepted + 1),
                        decay,
                    );
                }
            }
            LookupKind::Copy => {
                self.lookup
                    .observe(o.accepted, (o.accepted + 1).min(o.lookup_len), decay);
            }
            LookupKind::Tail => {
                let k = o.model_drafts;
                self.model
                    .observe(o.accepted.min(k), (o.accepted + 1).min(k), decay);
                if o.accepted >= k {
                    let matched = o.accepted - k;
                    self.lookup
                        .observe(matched, (matched + 1).min(o.lookup_len), decay);
                }
            }
        }
        self.book.observe(o.kind, o.rows, o.ctx, o.wall_us);
    }

    pub fn model(&self) -> &DepthTable {
        &self.model
    }

    pub fn lookup(&self) -> &DepthTable {
        &self.lookup
    }

    pub fn price_book(&self) -> &LookupPriceBook {
        &self.book
    }

    pub fn policy(&self) -> &LookupPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = LOOKUP_MAX_ROWS + 1;

    /// Explicit gate constants: tests must not depend on the calibrated `Default`.
    fn policy() -> LookupPolicy {
        LookupPolicy {
            margin: 1.1,
            min_first_accept: 0.9,
            min_evidence: 4.0,
            n_min: 1,
            decay: 0.875,
        }
    }

    /// Uniform-conditional table with a settable `last_first`.
    fn table(a: f32, weight: f32, last_first: bool) -> DepthTable {
        let mut t = DepthTable::new(a * weight, weight);
        t.last_first = last_first;
        t
    }

    fn flat(us: f64) -> [Option<f64>; N] {
        let mut a = [Some(us); N];
        a[0] = None;
        a
    }

    fn by_rows(f: impl Fn(usize) -> f64) -> [Option<f64>; N] {
        let mut a = [None; N];
        for (rows, slot) in a.iter_mut().enumerate().skip(1) {
            *slot = Some(f(rows));
        }
        a
    }

    fn budget(k: usize, available: usize) -> LookupBudget {
        LookupBudget {
            native_candidates: 15,
            model_drafts: k,
            lookup_available: available,
            max_rows: 16,
            remaining_emit: 1000,
            context_rows_left: 1000,
        }
    }

    fn choose(
        b: LookupBudget,
        native_us: Option<f64>,
        copy: &[Option<f64>; N],
        tail: &[Option<f64>; N],
        model: &DepthTable,
        lookup: &DepthTable,
        p: &LookupPolicy,
    ) -> LookupAction {
        choose_lookup(
            b,
            &LookupCosts {
                native_us,
                copy_us: copy,
                tail_us: tail,
            },
            &LookupAcceptance { model, lookup },
            p,
        )
    }

    fn geo(a: f64, n: i32) -> f64 {
        // 1 + a + a^2 + ... + a^n
        1.0 + a * (1.0 - a.powi(n)) / (1.0 - a)
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-4 * b.abs().max(1.0)
    }

    #[test]
    fn lookup_copy_choice_matches_hand_computed_rates() {
        let model = table(0.5, 8.0, true);
        let lookup = table(0.9, 8.0, true);
        // E_native(15) = 1 + sum .5^d, E_copy(15) = 1 + sum .9^d, equal prices (native 100 us).
        let e_native = geo(0.5, 15);
        let e_copy = geo(0.9, 15);
        assert!(close(f64::from(model.expected(15)), e_native));
        assert!(close(f64::from(lookup.expected(15)), e_copy));
        assert!(e_copy / 100.0 >= 1.1 * e_native / 100.0);
        let copy = flat(100.0);
        let tail = flat(100.0);
        assert_eq!(
            choose(
                budget(0, 15),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Copy { take: 15 }
        );
        // Copy priced at 4.2x native: rate ratio e_copy/(4.2 e_native) = 0.97 < margin -> native.
        let pricey = flat(420.0);
        assert!(e_copy / 420.0 < 1.1 * e_native / 100.0);
        assert_eq!(
            choose(
                budget(0, 15),
                Some(100.0),
                &pricey,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Native
        );
        // Priced just inside the margin: copy 3.5x -> ratio 1.17 >= 1.1 -> Copy.
        let ok = flat(350.0);
        assert!(e_copy / 350.0 >= 1.1 * e_native / 100.0);
        assert_eq!(
            choose(
                budget(0, 15),
                Some(100.0),
                &ok,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Copy { take: 15 }
        );
    }

    #[test]
    fn lookup_tail_choice_matches_hand_computed_rates() {
        let model = table(0.8, 8.0, true);
        let lookup = table(0.9, 8.0, true);
        // K=3: E_model(3) = 1 + .8 + .64 + .512 = 2.952, S_3 = .512; L=12:
        // E_tail = 2.952 + .512 * sum_{j<=12} .9^j, E_native(15) = 1 + sum_{d<=15} .8^d.
        let s3 = 0.512f64;
        let sum12: f64 = (1..=12).map(|j| 0.9f64.powi(j)).sum();
        let e_tail = 2.952 + s3 * sum12;
        let e_native = geo(0.8, 15);
        assert!(close(e_tail, 6.2585) && close(e_native, 4.8593));
        let copy = flat(100.0);
        let tail = flat(100.0);
        assert_eq!(
            choose(
                budget(3, 12),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Tail { take: 12 }
        );
        // ratio e_tail/e_native = 1.288: margin 1.25 passes, 1.3 refuses.
        let mut p = policy();
        p.margin = 1.25;
        assert_eq!(
            choose(
                budget(3, 12),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &p
            ),
            LookupAction::Tail { take: 12 }
        );
        p.margin = 1.3;
        assert_eq!(
            choose(
                budget(3, 12),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &p
            ),
            LookupAction::Native
        );
        // A Tail decision never answers Copy and a Copy decision never answers Tail.
        assert!(matches!(
            choose(
                budget(3, 12),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Tail { .. }
        ));
        assert!(matches!(
            choose(
                budget(0, 12),
                Some(100.0),
                &copy,
                &tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Copy { .. } | LookupAction::Native
        ));
    }

    #[test]
    fn lookup_missing_nonfinite_or_nonpositive_price_refuses() {
        let model = table(0.5, 8.0, true);
        let lookup = table(0.99, 8.0, true);
        let good = flat(100.0);
        for native in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(0.0),
            Some(-5.0),
        ] {
            assert_eq!(
                choose(
                    budget(0, 15),
                    native,
                    &good,
                    &good,
                    &model,
                    &lookup,
                    &policy()
                ),
                LookupAction::Native
            );
        }
        for bad in [
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(0.0),
            Some(-1.0),
        ] {
            let mut copy = flat(100.0);
            copy[16] = bad; // only the 15-take row is bad: the next-best priced row wins
            assert_eq!(
                choose(
                    budget(0, 15),
                    Some(100.0),
                    &copy,
                    &good,
                    &model,
                    &lookup,
                    &policy()
                ),
                LookupAction::Copy { take: 14 },
                "bad={bad:?}"
            );
            let all_bad = [bad; N];
            assert_eq!(
                choose(
                    budget(0, 15),
                    Some(100.0),
                    &all_bad,
                    &good,
                    &model,
                    &lookup,
                    &policy()
                ),
                LookupAction::Native,
                "bad={bad:?}"
            );
            assert_eq!(
                choose(
                    budget(4, 11),
                    Some(100.0),
                    &good,
                    &all_bad,
                    &model,
                    &lookup,
                    &policy()
                ),
                LookupAction::Native,
                "bad={bad:?}"
            );
        }
    }

    #[test]
    fn lookup_zero_proposal_is_native() {
        let model = table(0.5, 8.0, true);
        let lookup = table(0.99, 8.0, true);
        let good = flat(100.0);
        assert_eq!(
            choose(
                budget(0, 0),
                Some(100.0),
                &good,
                &good,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Native
        );
        assert_eq!(
            choose(
                budget(5, 0),
                Some(100.0),
                &good,
                &good,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Native
        );
        // n_min above the offered length is also no proposal.
        let mut p = policy();
        p.n_min = 4;
        assert_eq!(
            choose(budget(0, 3), Some(100.0), &good, &good, &model, &lookup, &p),
            LookupAction::Native
        );
        assert!(matches!(
            choose(budget(0, 4), Some(100.0), &good, &good, &model, &lookup, &p),
            LookupAction::Copy { take: 4 }
        ));
    }

    #[test]
    fn lookup_budget_clamps_context_emit_and_seed_bonus() {
        let model = table(0.3, 8.0, true); // poor learned drafter: E_native(15) = 1.43
        let lookup = table(1.0, 8.0, true); // a = 1: longer is always better at equal price
        let good = flat(100.0);
        let mut p = policy();
        p.margin = 1.0; // the clamps, not the margin, are under test (smallest windows gain little)
        let run = |b: LookupBudget| choose(b, Some(100.0), &good, &good, &model, &lookup, &p);
        // Offered candidates clamp.
        assert_eq!(run(budget(0, 6)), LookupAction::Copy { take: 6 });
        // Rows clamp: 8 rows = seed + 7; rows above the contract maximum clamp to 16.
        assert_eq!(
            run(LookupBudget {
                max_rows: 8,
                ..budget(0, 15)
            }),
            LookupAction::Copy { take: 7 }
        );
        assert_eq!(
            run(LookupBudget {
                max_rows: 40,
                ..budget(0, 30)
            }),
            LookupAction::Copy { take: 15 }
        );
        // Context clamp: rows <= context_rows_left.
        assert_eq!(
            run(LookupBudget {
                context_rows_left: 6,
                ..budget(0, 15)
            }),
            LookupAction::Copy { take: 5 }
        );
        assert_eq!(
            run(LookupBudget {
                context_rows_left: 1,
                ..budget(0, 15)
            }),
            LookupAction::Native
        );
        assert_eq!(
            run(LookupBudget {
                context_rows_left: 0,
                ..budget(0, 15)
            }),
            LookupAction::Native
        );
        // Emit clamp: seed is already emitted, accepted + bonus <= remaining_emit -> candidates <= remaining-1.
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 5,
                ..budget(0, 15)
            }),
            LookupAction::Copy { take: 4 }
        );
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 2,
                ..budget(0, 15)
            }),
            LookupAction::Copy { take: 1 }
        );
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 1,
                ..budget(0, 15)
            }),
            LookupAction::Native
        );
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 0,
                ..budget(0, 15)
            }),
            LookupAction::Native
        );
        // Tail clamps count the K kept learned candidates too.
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 9,
                ..budget(3, 12)
            }),
            LookupAction::Tail { take: 5 }
        );
        assert_eq!(
            run(LookupBudget {
                context_rows_left: 9,
                ..budget(3, 12)
            }),
            LookupAction::Tail { take: 5 }
        );
        assert_eq!(
            run(LookupBudget {
                remaining_emit: 4,
                ..budget(3, 12)
            }),
            LookupAction::Native
        );
        assert_eq!(
            run(LookupBudget {
                max_rows: 1,
                ..budget(0, 15)
            }),
            LookupAction::Native
        );
    }

    #[test]
    fn lookup_tail_k7_l8_fits_sixteen_rows_and_l9_refuses() {
        let model = table(0.9, 8.0, true);
        let lookup = table(1.0, 8.0, true);
        let good = flat(100.0);
        let mut p = policy();
        p.margin = 1.0; // K=14,L=1 gains only ~0.3% over native: the clamp, not the margin, is under test
        let run = |b: LookupBudget| choose(b, Some(100.0), &good, &good, &model, &lookup, &p);
        // 1 seed + 7 kept + 8 tail = 16 rows.
        assert_eq!(run(budget(7, 8)), LookupAction::Tail { take: 8 });
        // 9 offered: the ninth would make 17 rows -> clamped to 8, never 9.
        assert_eq!(run(budget(7, 9)), LookupAction::Tail { take: 8 });
        assert_eq!(run(budget(7, 15)), LookupAction::Tail { take: 8 });
        // K=8 leaves 7; K=14 leaves 1; K=15 leaves none.
        assert_eq!(run(budget(8, 15)), LookupAction::Tail { take: 7 });
        assert_eq!(run(budget(14, 15)), LookupAction::Tail { take: 1 });
        assert_eq!(run(budget(15, 15)), LookupAction::Native);
        // No price at rows 16 -> the 8-take row is unavailable, the 7-take row is taken.
        let mut tail = flat(100.0);
        tail[16] = None;
        assert_eq!(
            choose(budget(7, 9), Some(100.0), &good, &tail, &model, &lookup, &p),
            LookupAction::Tail { take: 7 }
        );
    }

    #[test]
    fn lookup_first_prefix_failure_prices_zero_tail_gain() {
        // Model first-depth acceptance 0: the tail is never reached, E_tail = E_model(K).
        let mut model = table(0.9, 8.0, true);
        model.depth[0] = (0.0, 8.0);
        let lookup = table(1.0, 8.0, true);
        assert_eq!(
            wide_expected(
                &budget(5, 10),
                &LookupAcceptance {
                    model: &model,
                    lookup: &lookup
                },
                10
            )
            .1,
            0.0
        );
        let cheap_tail = flat(1.0); // even a nearly free tail window cannot buy a zero gain
        assert_eq!(
            choose(
                budget(5, 10),
                Some(100.0),
                &flat(100.0),
                &cheap_tail,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Native
        );
        // Same table with first-depth acceptance restored: the tail pays.
        let model = table(0.9, 8.0, true);
        assert_eq!(
            choose(
                budget(5, 10),
                Some(100.0),
                &flat(100.0),
                &flat(100.0),
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Tail { take: 10 }
        );
        // And the closed form: E = E_model(K) + S_K * sum.
        let acc = LookupAcceptance {
            model: &model,
            lookup: &lookup,
        };
        let (e, gain) = wide_expected(&budget(2, 4), &acc, 4);
        // model .9: E(2) = 1 + .9 + .81 = 2.71, S_2 = .81; lookup a=1, L=4 -> sum = 4.
        assert!(close(e, 2.71 + 0.81 * 4.0) && close(gain, 0.81 * 4.0));
    }

    #[test]
    fn lookup_shorter_wins_ties() {
        let model = table(0.5, 8.0, true);
        let lookup = table(1.0, 8.0, true);
        // a = 1 -> E(take) = rows; price = 100 * rows -> every take has the same rate (0.01).
        let prop = by_rows(|rows| 100.0 * rows as f64);
        assert_eq!(
            choose(
                budget(0, 15),
                Some(1.0e6),
                &prop,
                &prop,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Copy { take: 1 }
        );
        let mut p = policy();
        p.n_min = 3;
        assert_eq!(
            choose(
                budget(0, 15),
                Some(1.0e6),
                &prop,
                &prop,
                &model,
                &lookup,
                &p
            ),
            LookupAction::Copy { take: 3 }
        );
        // Tail: E_tail = E_model(2) + S_2 * L with a_model = 1 too -> rows again; same tie rule.
        let model1 = table(1.0, 8.0, true);
        assert_eq!(
            choose(
                budget(2, 13),
                Some(1.0e6),
                &prop,
                &prop,
                &model1,
                &lookup,
                &policy()
            ),
            LookupAction::Tail { take: 1 }
        );
        // A strictly better longer take still wins over the tie rule: rate rises with rows.
        let rising = by_rows(|rows| 100.0 + 10.0 * rows as f64);
        assert_eq!(
            choose(
                budget(0, 15),
                Some(1.0e6),
                &rising,
                &rising,
                &model,
                &lookup,
                &policy()
            ),
            LookupAction::Copy { take: 15 }
        );
    }

    #[test]
    fn lookup_depth_table_update_matches_production_recurrence() {
        // observe(matched=2, compared=3), decay .875, prior (1.6, 2.0).
        let mut t = DepthTable::new(1.6, 2.0);
        assert!(!t.last_first_accepted());
        t.observe(2, 3, 0.875);
        assert!(t.last_first_accepted());
        // depths 1,2 accepted: acc = 1.6*.875 + 1 = 2.4, weight = 2*.875 + 1 = 2.75.
        for d in [1, 2] {
            assert!(
                close(f64::from(t.depth[d - 1].0), 2.4) && close(f64::from(t.depth[d - 1].1), 2.75)
            );
        }
        // depth 3 compared (rejected): acc = 1.6*.875 = 1.4, weight 2.75.
        assert!(close(f64::from(t.depth[2].0), 1.4) && close(f64::from(t.depth[2].1), 2.75));
        // depth 4 not compared: target = 1.4/2.75 = .50909 < own ratio .8 -> relax .95:
        // acc = 1.6*.95 + 2*.50909*.05 = 1.570909, weight = 2*.95 + 2*.05 = 2.0.
        assert!(close(f64::from(t.depth[3].0), 1.570909) && close(f64::from(t.depth[3].1), 2.0));
        // depth 5 chases the already-updated depth 4: target = 1.570909/2 = .785454 < .8 -> relax .95.
        assert!(close(
            f64::from(t.depth[4].0),
            1.6 * 0.95 + 2.0 * 0.785454 * 0.05
        ));
        // compared is censored: observe(0, 5) compares only depth 1 (matched + 1 = 1).
        let mut u = DepthTable::new(1.6, 2.0);
        u.observe(0, 5, 0.875);
        assert!(!u.last_first_accepted());
        assert!(close(f64::from(u.depth[0].0), 1.4) && close(f64::from(u.depth[0].1), 2.75));
        assert!(close(f64::from(u.weight(2)), 2.0)); // depth 2 untouched in weight (relaxed at prior weight)
                                                     // compared == 0 observes nothing.
        let mut z = DepthTable::new(1.6, 2.0);
        z.observe(3, 0, 0.875);
        assert_eq!(z, DepthTable::new(1.6, 2.0));
        // Relax up (.8) when the shallower target is above the depth's own ratio.
        let mut v = DepthTable::new(1.6, 2.0);
        v.depth[1] = (0.2, 2.0); // depth 2 own ratio .1
        v.observe(1, 1, 1.0); // depth 1 -> (2.6, 3.0): ratio .8667; depth 2 relaxes up toward it
        assert!(close(
            f64::from(v.depth[1].0),
            0.2 * 0.8 + 2.0 * (2.6 / 3.0) * 0.2
        ));
    }

    #[test]
    fn lookup_depth_table_decay_follows_drift() {
        let mut t = DepthTable::new(1.6, 2.0);
        for _ in 0..40 {
            t.observe(15, 15, 0.875);
        }
        assert!(t.conditional(1) > 0.99 && t.conditional(15) > 0.99);
        assert!(
            f64::from(t.weight(1)) < 8.0 + 1e-3 && t.weight(1) > 7.9,
            "weight saturates at 1/(1-decay)"
        );
        assert!(t.expected(15) > 15.8);
        // Drift: the source stops matching; first-depth acceptance collapses within a few windows.
        for _ in 0..12 {
            t.observe(0, 1, 0.875);
        }
        assert!(t.conditional(1) < 0.25, "{}", t.conditional(1));
        assert!(!t.last_first_accepted());
        // Unreached depths relax slowly (at the prior's weight): they stay above the collapsed depth 1.
        assert!(
            t.conditional(4) > t.conditional(1),
            "{} {}",
            t.conditional(4),
            t.conditional(1)
        );
        // Out-of-range lookups are 0, never a panic or a prior.
        assert_eq!(t.conditional(0), 0.0);
        assert_eq!(t.conditional(LOOKUP_MAX_CANDIDATES + 1), 0.0);
        assert_eq!(t.weight(0), 0.0);
        assert_eq!(t.weight(LOOKUP_MAX_CANDIDATES + 1), 0.0);
    }

    fn outcome(
        kind: LookupKind,
        rows: usize,
        k: usize,
        l: usize,
        accepted: usize,
        shadow: Option<(usize, usize)>,
    ) -> WindowOutcome {
        WindowOutcome {
            kind,
            rows,
            model_drafts: k,
            lookup_len: l,
            accepted,
            shadow,
            ctx: 5000,
            wall_us: 50_000.0,
        }
    }

    #[test]
    fn lookup_observation_is_source_separated_and_censored() {
        let fresh = || LookupController::new(policy(), true);
        let prior = DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1);

        // Native, no shadow: only the model table moves.
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Native, 16, 15, 0, 3, None));
        let mut want_model = prior.clone();
        want_model.observe(3, 4, 0.875);
        assert_eq!(c.model(), &want_model);
        assert_eq!(c.lookup(), &prior);

        // Native with a shadow: lookup.observe(m, min(m+1, avail, accepted+1)).
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Native, 16, 15, 0, 2, Some((3, 15))));
        let mut want_lookup = prior.clone();
        want_lookup.observe(3, 3, 0.875); // min(3+1, 15, 2+1) = 3
        assert_eq!(c.lookup(), &want_lookup);
        let mut want_model = prior.clone();
        want_model.observe(2, 3, 0.875);
        assert_eq!(c.model(), &want_model);

        // Copy: only the lookup table moves; the model is never observed.
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, Some((9, 9))));
        let mut want_lookup = prior.clone();
        want_lookup.observe(15, 15, 0.875);
        assert_eq!(c.lookup(), &want_lookup);
        assert_eq!(c.model(), &prior);
        // Copy first-reject: min(0+1, 15) = 1 compared, none accepted.
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 0, None));
        let mut want_lookup = prior.clone();
        want_lookup.observe(0, 1, 0.875);
        assert_eq!(c.lookup(), &want_lookup);
        assert_eq!(c.model(), &prior);

        // Tail K=4, L=8: prefix failed (accepted 2 < 4): model sees (2, min(3, 4)=3), lookup untouched.
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Tail, 13, 4, 8, 2, Some((7, 15))));
        let mut want_model = prior.clone();
        want_model.observe(2, 3, 0.875);
        assert_eq!(c.model(), &want_model);
        assert_eq!(
            c.lookup(),
            &prior,
            "tail depths past a failed prefix are censored, not rejected"
        );
        // Tail prefix survived (accepted 4+5): model (4, 4); lookup (5, min(6, 8) = 6).
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Tail, 13, 4, 8, 9, None));
        let mut want_model = prior.clone();
        want_model.observe(4, 4, 0.875);
        let mut want_lookup = prior.clone();
        want_lookup.observe(5, 6, 0.875);
        assert_eq!(c.model(), &want_model);
        assert_eq!(c.lookup(), &want_lookup);
        // Full tail accept: min(L+1, L) = L compared, all accepted.
        let mut c = fresh();
        c.observe(&outcome(LookupKind::Tail, 13, 4, 8, 12, None));
        let mut want_lookup = prior.clone();
        want_lookup.observe(8, 8, 0.875);
        assert_eq!(c.lookup(), &want_lookup);
        // Every window feeds the price book.
        assert_eq!(
            c.price_book().price(LookupKind::Tail, 13, 5000),
            None,
            "first observation of a cell is discarded"
        );
    }

    #[test]
    fn lookup_weak_prior_alone_never_triggers_wide() {
        let mut c = LookupController::new(policy(), true);
        // Price every row so only evidence can decide; native priced, copy/tail cells priced.
        for rows in 1..=LOOKUP_MAX_ROWS {
            for kind in [LookupKind::Native, LookupKind::Copy, LookupKind::Tail] {
                c.book.observe(kind, rows, 5000, 50_000.0);
                c.book.observe(kind, rows, 5000, 50_000.0);
            }
        }
        // A model that is genuinely poor and a prior-only lookup: the prior ratio (.8) would win
        // on rate, but it is not evidence.
        for _ in 0..8 {
            c.model.observe(0, 1, 0.875);
        }
        assert!(c.model().expected(15) < 1.8);
        assert!(c.lookup().expected(15) > 3.0, "the prior alone would pay");
        let b = budget(0, 15);
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        assert_eq!(c.plan(budget(3, 12), 5000), LookupAction::Native);
        // Even a prior as heavy and as high as the gate asks for is not evidence: no observed
        // first-depth acceptance yet.
        c.lookup = DepthTable::new(99.0, 100.0);
        assert!(c.lookup().conditional(1) >= 0.9 && c.lookup().weight(1) >= 4.0);
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        // The default policy's evidence weight exceeds the weak prior's weight.
        assert!(LookupPolicy::default().min_evidence > LOOKUP_PRIOR.1);
    }

    #[test]
    fn lookup_evidence_gate_needs_weight_first_accept_and_last_first() {
        let mut c = LookupController::new(policy(), true);
        for rows in 1..=LOOKUP_MAX_ROWS {
            c.book.observe(LookupKind::Native, rows, 5000, 50_000.0);
            c.book.observe(LookupKind::Native, rows, 5000, 50_000.0);
        }
        for _ in 0..8 {
            c.model.observe(0, 1, 0.875);
        }
        let b = budget(0, 15);
        // weight(1): 2 -> 2.75 -> 3.406 -> 3.980 (< 4) -> 4.483 (>= 4).
        for n in 1..=5 {
            c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
            let want = if n >= 4 {
                LookupAction::Copy { take: 15 }
            } else {
                LookupAction::Native
            };
            assert_eq!(
                c.plan(b, 5000),
                want,
                "after {n} fully accepted copy windows (weight {})",
                c.lookup().weight(1)
            );
        }
        // A last-window first-depth reject closes the gate immediately.
        c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 0, None));
        assert!(!c.lookup().last_first_accepted());
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        // First-depth acceptance below the threshold refuses even with weight and last_first.
        let mut d = c.clone();
        d.lookup = table(0.6, 8.0, true);
        assert_eq!(d.plan(b, 5000), LookupAction::Native);
        // NaN policy values refuse rather than pass.
        let mut e = c.clone();
        e.lookup = table(1.0, 8.0, true);
        assert!(matches!(e.plan(b, 5000), LookupAction::Copy { .. }));
        e.policy.margin = f32::NAN;
        assert_eq!(e.plan(b, 5000), LookupAction::Native);
    }

    #[test]
    fn lookup_price_book_discards_first_observation_then_median_of_eight() {
        let mut book = LookupPriceBook::new();
        let ctx = 1000;
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), None);
        book.observe(LookupKind::Copy, 16, ctx, 9_000_000.0); // first-bind / capture: discarded
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), None);
        book.observe(LookupKind::Copy, 16, ctx, 100.0);
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(100.0));
        book.observe(LookupKind::Copy, 16, ctx, 300.0);
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(200.0)); // even count: mean of the middle two
        book.observe(LookupKind::Copy, 16, ctx, 200.0);
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(200.0));
        // Fill to 8 samples, then slide: only the last 8 count.
        for us in [400.0, 500.0, 600.0, 700.0, 800.0] {
            book.observe(LookupKind::Copy, 16, ctx, us);
        }
        // samples: 100 300 200 400 500 600 700 800 -> sorted 100 200 300 400 500 600 700 800 -> 450.
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(450.0));
        book.observe(LookupKind::Copy, 16, ctx, 900.0); // evicts 100
                                                        // 300 200 400 500 600 700 800 900 -> sorted 200..900 -> (500+600)/2 = 550.
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(550.0));
        for _ in 0..8 {
            book.observe(LookupKind::Copy, 16, ctx, 1000.0); // evicts everything older
        }
        assert_eq!(book.price(LookupKind::Copy, 16, ctx), Some(1000.0));
        // Cells are independent: kind, rows and context bucket each discard their own first.
        assert_eq!(book.price(LookupKind::Tail, 16, ctx), None);
        assert_eq!(book.price(LookupKind::Copy, 15, ctx), None);
        assert_eq!(book.price(LookupKind::Copy, 16, 20_000), None);
        book.observe(LookupKind::Copy, 16, 20_000, 1.0);
        assert_eq!(book.price(LookupKind::Copy, 16, 20_000), None);
        // Bucket edges: <=2048 | <=8192 | <=16384 | <=32768 | above.
        assert_eq!(price_bucket(2048), 0);
        assert_eq!(price_bucket(2049), 1);
        assert_eq!(price_bucket(8192), 1);
        assert_eq!(price_bucket(16384), 2);
        assert_eq!(price_bucket(32768), 3);
        assert_eq!(price_bucket(32769), 4);
        // Invalid observations are ignored and never consume the discarded first.
        let mut b2 = LookupPriceBook::new();
        for bad in [f64::NAN, f64::INFINITY, 0.0, -3.0] {
            b2.observe(LookupKind::Native, 8, 100, bad);
        }
        b2.observe(LookupKind::Native, 0, 100, 5.0);
        b2.observe(LookupKind::Native, 17, 100, 5.0);
        b2.observe(LookupKind::Native, 8, 100, 7.0); // first valid: discarded
        assert_eq!(b2.price(LookupKind::Native, 8, 100), None);
        b2.observe(LookupKind::Native, 8, 100, 9.0);
        assert_eq!(b2.price(LookupKind::Native, 8, 100), Some(9.0));
        assert_eq!(b2.price(LookupKind::Native, 0, 100), None);
        assert_eq!(b2.price(LookupKind::Native, 17, 100), None);
    }

    fn seeded(book: &mut LookupPriceBook, kind: LookupKind, rows: usize, ctx: usize, us: f64) {
        book.observe(kind, rows, ctx, us);
        book.observe(kind, rows, ctx, us);
    }

    #[test]
    fn lookup_price_book_fill_shares_native_only_when_asked() {
        let mut book = LookupPriceBook::new();
        seeded(&mut book, LookupKind::Native, 16, 100, 40_000.0);
        seeded(&mut book, LookupKind::Native, 8, 100, 25_000.0);
        seeded(&mut book, LookupKind::Copy, 16, 100, 41_000.0); // own copy price wins over native
        seeded(&mut book, LookupKind::Tail, 12, 100, 33_000.0);
        let (mut copy, mut tail) = ([Some(1.0); N], [Some(1.0); N]);
        book.fill(100, true, &mut copy, &mut tail);
        assert_eq!(copy[0], None);
        assert_eq!(copy[16], Some(41_000.0));
        assert_eq!(copy[8], Some(25_000.0));
        assert_eq!(copy[7], None);
        assert_eq!(tail[16], Some(40_000.0));
        assert_eq!(tail[12], Some(33_000.0));
        assert_eq!(tail[8], Some(25_000.0));
        // Not shared: only own prices.
        book.fill(100, false, &mut copy, &mut tail);
        assert_eq!(copy[16], Some(41_000.0));
        assert_eq!(copy[8], None);
        assert_eq!(tail[16], None);
        assert_eq!(tail[12], Some(33_000.0));
        // Other bucket: nothing.
        book.fill(50_000, true, &mut copy, &mut tail);
        assert!(copy.iter().all(Option::is_none) && tail.iter().all(Option::is_none));
    }

    #[test]
    fn lookup_price_probe_rows_semantics() {
        let mut c = LookupController::new(policy(), true);
        // Established evidence: lookup first depth high, weight enough, last first accepted.
        for _ in 0..6 {
            c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
        }
        // Model poor (the native window pays little).
        for _ in 0..8 {
            c.model.observe(0, 1, 0.875);
        }
        let b = budget(0, 9); // 9 candidates -> 10 rows
                              // Native itself is unpriced: nothing to compare against -> no probe.
        assert_eq!(c.price_probe_rows(b, 5000), None);
        seeded(&mut c.book, LookupKind::Native, 16, 5000, 50_000.0);
        // Native priced, 10-row cell unpriced: probe the longest legal copy's row count.
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        assert_eq!(c.price_probe_rows(b, 5000), Some(10));
        // Probing prices it: discarded first, then usable (shared Native price) -> Copy chosen, no probe.
        seeded(&mut c.book, LookupKind::Native, 10, 5000, 50_000.0);
        assert_eq!(c.price_probe_rows(b, 5000), None);
        assert!(matches!(c.plan(b, 5000), LookupAction::Copy { take: 9 }));
        // The clamps move the probed row count: remaining_emit 5 -> 4 candidates -> 5 rows.
        let clamped = LookupBudget {
            remaining_emit: 5,
            ..b
        };
        assert_eq!(c.price_probe_rows(clamped, 5000), Some(5));
        // A Tail decision never probes.
        assert_eq!(c.price_probe_rows(budget(3, 9), 5000), None);
        // A different context bucket is unpriced even natively -> no probe (nothing to compare).
        assert_eq!(c.price_probe_rows(b, 20_000), None);
        // Evidence gate closed -> no probe.
        let mut closed = c.clone();
        closed.observe(&outcome(LookupKind::Copy, 16, 0, 15, 0, None));
        assert_eq!(closed.price_probe_rows(clamped, 5000), None);
        // No clamp room -> no probe.
        assert_eq!(
            c.price_probe_rows(
                LookupBudget {
                    remaining_emit: 1,
                    ..b
                },
                5000
            ),
            None
        );
        assert_eq!(c.price_probe_rows(budget(0, 0), 5000), None);
        // share_native off: a native window would not price a Copy -> no probe.
        let mut own = LookupController::new(policy(), false);
        for _ in 0..6 {
            own.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
        }
        seeded(&mut own.book, LookupKind::Native, 16, 5000, 50_000.0);
        assert_eq!(own.price_probe_rows(b, 5000), None);
        // Price-free necessary condition: a model that already accepts everything leaves no room.
        let mut full = c.clone();
        full.model = table(1.0, 8.0, true);
        full.lookup = table(0.95, 8.0, true);
        assert_eq!(
            full.price_probe_rows(
                LookupBudget {
                    remaining_emit: 5,
                    ..b
                },
                5000
            ),
            None
        );
    }

    #[test]
    fn lookup_begin_request_resets_tables_and_keeps_prices() {
        let mut c = LookupController::new(policy(), true);
        for _ in 0..6 {
            c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
            c.observe(&outcome(LookupKind::Native, 16, 15, 0, 2, Some((1, 15))));
        }
        let (model_before, lookup_before) = (c.model().clone(), c.lookup().clone());
        let prior = DepthTable::new(LOOKUP_PRIOR.0, LOOKUP_PRIOR.1);
        assert_ne!(model_before, prior);
        assert_ne!(lookup_before, prior);
        let copy_price = c.price_book().price(LookupKind::Copy, 16, 5000);
        let native_price = c.price_book().price(LookupKind::Native, 16, 5000);
        assert_eq!(copy_price, Some(50_000.0));
        assert_eq!(native_price, Some(50_000.0));
        c.begin_request();
        assert_eq!(c.model(), &prior);
        assert_eq!(c.lookup(), &prior);
        assert!(!c.lookup().last_first_accepted());
        assert_eq!(c.price_book().price(LookupKind::Copy, 16, 5000), copy_price);
        assert_eq!(
            c.price_book().price(LookupKind::Native, 16, 5000),
            native_price
        );
        // A fresh request cannot go wide on the previous request's evidence.
        assert_eq!(c.plan(budget(0, 15), 5000), LookupAction::Native);
    }

    #[test]
    fn lookup_controller_plan_prices_come_from_the_book() {
        let mut c = LookupController::new(policy(), true);
        for _ in 0..6 {
            c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
        }
        for _ in 0..8 {
            c.model.observe(0, 1, 0.875);
        }
        let b = budget(0, 15);
        // Nothing priced: refuse.
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        // Native 16-row price present; the Copy@16 cell was priced by the six observed Copy windows.
        seeded(&mut c.book, LookupKind::Native, 16, 5000, 50_000.0);
        assert_eq!(c.plan(b, 5000), LookupAction::Copy { take: 15 });
        // Only 16-row prices exist, so a shorter proposal has no priced row: refuse.
        assert_eq!(c.plan(budget(0, 9), 5000), LookupAction::Native);
        assert!(c.book.price(LookupKind::Copy, 16, 5000).is_some());
        // Copy window measured at 4x native: rate = E_copy/(4 x native_us) is below native's.
        for _ in 0..8 {
            c.book.observe(LookupKind::Copy, 16, 5000, 1_000_000.0);
        }
        assert_eq!(c.plan(b, 5000), LookupAction::Native);
        // Another context bucket has no price at all.
        assert_eq!(c.plan(b, 40_000), LookupAction::Native);
    }

    #[test]
    fn lookup_default_policy_is_the_calibrated_point_and_needs_two_observed_windows() {
        let p = LookupPolicy::default();
        assert_eq!(
            (
                p.margin,
                p.min_first_accept,
                p.min_evidence,
                p.n_min,
                p.decay
            ),
            (1.25, 0.9, 3.0, 1, 0.875)
        );
        assert!(
            p.min_evidence > LOOKUP_PRIOR.1,
            "the weak prior's weight must not satisfy the evidence gate"
        );
        let mut c = LookupController::new(p, true);
        for rows in 1..=LOOKUP_MAX_ROWS {
            c.book.observe(LookupKind::Native, rows, 5000, 50_000.0);
            c.book.observe(LookupKind::Native, rows, 5000, 50_000.0);
        }
        for _ in 0..8 {
            c.model.observe(0, 1, 0.875);
        }
        // one fully accepted window: weight 2.75 < 3 and ratio .873 < .9; two: weight 3.41, ratio .91.
        c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
        assert_eq!(c.plan(budget(0, 15), 5000), LookupAction::Native);
        c.observe(&outcome(LookupKind::Copy, 16, 0, 15, 15, None));
        assert_eq!(c.plan(budget(0, 15), 5000), LookupAction::Copy { take: 15 });
    }
}
