// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Root-side sampled acceptance of the native-MTP mesh lanes (EP2 / PP2).
//!
//! The singleton's sampled machinery (`mtp_spec::SampledVerify`:
//! `sample_draft`, `accept_leviathan`, `bundle::mtp_draft_dist`) is bound to
//! one `Gpu`, one request and a process-global RNG. This module is its
//! lane-owned, host-only counterpart: every statistical step is the same
//! `hipfire_runtime::spec_sampling` arithmetic (`SampleSpec`, `SparseDist`,
//! `accept_sampled_prefix`, `accept_naive_prefix`, `SpecRng`), driven by
//! the mesh root's [`AcceptRule`] hooks, so a lane window is the singleton's
//! batched window in distribution and, for the same draws, in value.
//!
//! # Who draws what
//!
//! Only the **root** draws. A lane's RNG ([`LaneRng`]) is the request's
//! staged stream the reducer hands to the rule; ranks never see it.
//!
//! * [`SamplingMode::Leviathan`]: the root draws each native proposal from the
//!   head's distribution `q` ([`AcceptRule::draft_token`], one uniform per
//!   head step, *before* the verdict, exactly the singleton's order), then
//!   accepts each draft with `min(1, p/q)`, draws the residual `(p-q)+` on the
//!   first rejection and the bonus from `p` when every draft was accepted
//!   ([`AcceptRule::accept`]). A copy proposal is the point mass `δ(draft)`.
//! * [`SamplingMode::NaiveSampled`]: the existing same-seed AR contract. Drafts
//!   stay the head's argmax; each verify row is one draw of the AR sampler
//!   (`llama::sample_top_k_p_with_state`) on the lane's xorshift32 stream
//!   (the state `reset_cpu_sampler_rng(seed)` starts), and a draft is accepted
//!   iff it equals its row's draw, so every emitted token is exactly one AR
//!   draw. The draw authority moved from "every rank draws the same stream"
//!   to "the root draws, ranks replay the committed decision".
//! * [`SamplingMode::Greedy`]: the shared accept-prefix rule, no draw.
//!
//! # q lifetime
//!
//! The head's exact rescored candidates live in a buffer the *next* Draft
//! overwrites (the operator is shared by every lane). The sampler owner copies
//! them into the proposal ([`DraftRead`]) right after each Draft op, so the
//! root can build `q` for exactly the distribution it draws from, with that
//! lane's temperature/top-p/top-k/min-p and penalty history, however the lanes
//! interleave.
//!
//! # State and abort
//!
//! RNG draws advance the reducer's *staged* RNG; the committed RNG moves only at
//! `publish_commit`. Rule-side state is limited to the penalty history and
//! the current window's `q`s: the history changes only in
//! [`AcceptRule::on_publish`] (after the all-prepared boundary), the `q`s are
//! rebuilt at draft index 0 of every window. An aborted step therefore leaves
//! the RNG and the history untouched.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use hipfire_runtime::llama::sample_top_k_p_with_state;
use hipfire_runtime::sampler::{
    apply_logit_policy_candidates_cpu, apply_logit_policy_cpu, SamplerConfig,
};
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::{accept_greedy_prefix, GreedyAccept, SpecRequestConfig};
use hipfire_runtime::spec_sampling::{
    accept_naive_prefix, accept_sampled_prefix, naive_target_sampler, PenaltyHistory, SampleSpec,
    SparseDist, SpecRng,
};

use crate::mesh_lane_protocol::{
    AcceptRule, DraftProposal, LanePlan, LaneWindowKey, ProposalSource, SamplerOutput,
    SamplingMode, StepOutput,
};

// ---------------------------------------------------------------------------
// Lane RNG
// ---------------------------------------------------------------------------

/// A request's root-owned random streams.
///
/// `spec` is the [`SpecRng`] (xorshift64*) that supplies Leviathan draws, in
/// the singleton's order; `naive` is the xorshift32 state of the AR sampler
/// that [`SamplingMode::NaiveSampled`] draws from (what
/// `llama::reset_cpu_sampler_rng(seed)` starts: the seed itself, `0 -> 1`).
/// A request uses one of them; both are seeded from the same `rng_seed` so a
/// lane's mode never changes its seed identity. `Copy`: the reducer stages a
/// copy per step and publishes it only after all ranks prepared.
#[derive(Clone, Copy, Debug)]
pub struct LaneRng {
    spec: SpecRng,
    naive: u32,
}

impl LaneRng {
    /// The streams of a request admitted with `rng_seed`.
    pub fn new(rng_seed: u64) -> Self {
        Self::from_parts(SpecRng::new(rng_seed), rng_seed as u32)
    }

    /// Explicit streams (a prefill that already drew the first token from the
    /// naive stream resumes it here). A zero naive state is the sampler's
    /// stuck state, mapped to `1` exactly as `reset_cpu_sampler_rng` does.
    pub fn from_parts(spec: SpecRng, naive: u32) -> Self {
        Self {
            spec,
            naive: if naive == 0 { 1 } else { naive },
        }
    }

    /// Next raw draw of the Leviathan stream.
    pub fn next_u64(&mut self) -> u64 {
        self.spec.next_u64()
    }

    /// Next uniform in `[0, 1)` of the Leviathan stream.
    pub fn next_f32(&mut self) -> f32 {
        self.spec.next_f32()
    }

    /// The naive (AR sampler) xorshift32 state.
    pub fn naive_state(&self) -> u32 {
        self.naive
    }

    fn naive_mut(&mut self) -> &mut u32 {
        &mut self.naive
    }

    /// What the streams would draw next, without advancing them: equal
    /// fingerprints mean equal futures.
    pub fn fingerprint(&self) -> (u64, u32) {
        let mut spec = self.spec;
        (spec.next_u64(), self.naive)
    }
}

// ---------------------------------------------------------------------------
// Draft read: the head's q material
// ---------------------------------------------------------------------------

/// Widest rescored candidate set a draft head emits (`[(u32, f32); 8]`).
pub const DRAFT_CANDIDATES: usize = 8;

/// What the sampler owner read from the head right after one Draft op, before
/// another lane's Draft can overwrite the shared buffers.
#[derive(Clone, Debug)]
pub enum DraftRead {
    /// The head's exact rescored candidates `(token, exact logit)`: the
    /// support a sampled draft is drawn from (the approximate MQ2 ranking is
    /// only a candidate selector; the distribution is over these exact logits).
    Candidates(Vec<(u32, f32)>),
    /// A head without re-scoring: the whole draft logit row in token order.
    FullRow(Vec<f32>),
}

/// Bitwise equality: logits are compared as the exact words that were read.
impl PartialEq for DraftRead {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Candidates(a), Self::Candidates(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b)
                        .all(|(x, y)| x.0 == y.0 && x.1.to_bits() == y.1.to_bits())
            }
            (Self::FullRow(a), Self::FullRow(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            _ => false,
        }
    }
}

impl Eq for DraftRead {}

/// The draft distribution `q` of a rescored candidate set under `spec`:
/// `policy` (penalties over `history`, blocked tokens) is applied to the exact
/// candidate logits by token id, then truncation
/// ([`SparseDist::build_from_candidates`]). Tokens at or above `vocab` are
/// dropped. This is the pure half of `Qwen4Bundle::mtp_draft_dist` (the
/// bundle reads the device; this builds the distribution).
pub fn draft_dist_from_candidates(
    candidates: &[(u32, f32)],
    vocab: u32,
    spec: SampleSpec,
    history: &[u32],
    policy: &SamplerConfig,
    scratch: &mut Vec<(u32, f32)>,
    out: &mut SparseDist,
) -> Result<(), String> {
    scratch.clear();
    scratch.extend(
        candidates
            .iter()
            .copied()
            .filter(|&(token, _)| token < vocab),
    );
    let n = scratch.len();
    if n > DRAFT_CANDIDATES {
        return Err(format!(
            "draft head returned {n} candidates, at most {DRAFT_CANDIDATES} are re-scored"
        ));
    }
    let mut ids = [0u32; DRAFT_CANDIDATES];
    let mut values = [0f32; DRAFT_CANDIDATES];
    for (slot, &(token, logit)) in scratch.iter().enumerate() {
        ids[slot] = token;
        values[slot] = logit;
    }
    apply_logit_policy_candidates_cpu(&ids[..n], &mut values[..n], history, policy);
    for (entry, &logit) in scratch.iter_mut().zip(&values[..n]) {
        entry.1 = logit;
    }
    out.build_from_candidates(scratch, spec)
}

/// The draft distribution of a whole draft logit row (a head that does not
/// re-score): `policy` is applied to `row` in place, then `build_from_logits`.
pub fn draft_dist_from_row(
    row: &mut [f32],
    spec: SampleSpec,
    history: &[u32],
    policy: &SamplerConfig,
    scratch: &mut Vec<(u32, f32)>,
    out: &mut SparseDist,
) -> Result<(), String> {
    apply_logit_policy_cpu(row, history, policy);
    out.build_from_logits(row, spec, scratch)
}

impl DraftRead {
    /// Build `q` for this read. `host` is a reused buffer (only a
    /// [`Self::FullRow`] read uses it).
    #[allow(clippy::too_many_arguments)]
    pub fn build_dist(
        &self,
        vocab: u32,
        spec: SampleSpec,
        history: &[u32],
        policy: &SamplerConfig,
        host: &mut Vec<f32>,
        scratch: &mut Vec<(u32, f32)>,
        out: &mut SparseDist,
    ) -> Result<(), String> {
        match self {
            Self::Candidates(candidates) => {
                draft_dist_from_candidates(candidates, vocab, spec, history, policy, scratch, out)
            }
            Self::FullRow(row) => {
                host.clear();
                host.extend_from_slice(row);
                draft_dist_from_row(host, spec, history, policy, scratch, out)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Target rows: lazy access to the sampler owner's verify logits
// ---------------------------------------------------------------------------

/// The sampler owner's resident verify logits, read lazily by the root.
///
/// Rows are fetched one at a time and only up to the row a verdict reads
/// (`accept_sampled_prefix` / `accept_naive_prefix` call for rows in order), the
/// singleton's `spec_row_logits` discipline: no full-window download.
pub trait TargetRowSource: Send {
    /// Host copy of the full-vocabulary, **unpenalized** logits of verify row
    /// `row` of lane window `key` (`0` = the seed row), into `host`.
    fn row_logits(
        &mut self,
        key: &LaneWindowKey,
        row: usize,
        host: &mut Vec<f32>,
    ) -> Result<(), String>;
}

/// A shareable handle to a [`TargetRowSource`]: the sampler owner's executor
/// returns it in [`SamplerOutput::rows`] and keeps serving rows until the next
/// step. Equality is handle identity.
#[derive(Clone)]
pub struct RowHandle(Arc<Mutex<dyn TargetRowSource>>);

impl RowHandle {
    pub fn new(source: impl TargetRowSource + 'static) -> Self {
        Self(Arc::new(Mutex::new(source)))
    }

    /// Wrap an already shared source.
    pub fn shared(source: Arc<Mutex<dyn TargetRowSource>>) -> Self {
        Self(source)
    }

    fn row_logits(
        &self,
        key: &LaneWindowKey,
        row: usize,
        host: &mut Vec<f32>,
    ) -> Result<(), String> {
        self.0
            .lock()
            .map_err(|_| "target row source poisoned".to_string())?
            .row_logits(key, row, host)
    }
}

impl std::fmt::Debug for RowHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RowHandle(..)")
    }
}

impl PartialEq for RowHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RowHandle {}

// ---------------------------------------------------------------------------
// Per-request sampler registry + AcceptRule
// ---------------------------------------------------------------------------

/// The sampler of one request: the AR producer's policy
/// ([`naive_target_sampler`]: temperature, top-p, top-k, min-p, penalties) and
/// the truncation spec of its distributions.
#[derive(Clone, Debug)]
pub struct LaneSamplerParams {
    spec: SampleSpec,
    policy: SamplerConfig,
    window: usize,
}

impl LaneSamplerParams {
    /// The request's sampler. A sampled lane needs a positive finite
    /// temperature (greedy lanes use [`SamplingMode::Greedy`] and no params).
    pub fn from_request(cfg: &SpecRequestConfig) -> Result<Self, String> {
        if !(cfg.temp.is_finite() && cfg.temp > 0.0) {
            return Err(format!(
                "sampled lane needs a positive finite temperature, got {}",
                cfg.temp
            ));
        }
        Ok(Self {
            spec: SampleSpec::cpu_ar(cfg.temp, cfg.top_p, cfg.top_k, cfg.min_p),
            policy: naive_target_sampler(cfg),
            window: cfg.penalty_window(),
        })
    }

    pub fn spec(&self) -> SampleSpec {
        self.spec
    }
}

struct LaneSamplerState {
    params: LaneSamplerParams,
    /// Penalty history: prompt tail + the window's kept drafts. Window 0 (no
    /// active penalty) does no history work.
    history: PenaltyHistory,
    /// The last `window` tokens emitted so far, the pending seed included
    /// (`E` of the singleton's `H_i = suffix_W(P ‖ E ‖ drafts[..i])`).
    emitted: Vec<u32>,
    /// `q` of each kept-or-dropped draft step of the window `q_key`.
    q: Vec<SparseDist>,
    q_key: Option<LaneWindowKey>,
    /// Point-mass `q`s of a copy window, reused across windows.
    points: Vec<SparseDist>,
}

impl LaneSamplerState {
    fn trim_emitted(&mut self) {
        let window = self.params.window;
        if self.emitted.len() > window {
            self.emitted.drain(..self.emitted.len() - window);
        }
    }
}

/// The root's lane sampling context and the sampled [`AcceptRule`].
///
/// The driver registers each sampled request ([`Self::admit`]) next to
/// `MeshRoot::admit_lane`, passes `&mut LaneSampler` as the `rule` of every
/// step (greedy lanes in the same step are served by
/// `accept_greedy_prefix`, never drawing), and the reducer calls the hooks:
/// [`AcceptRule::draft_token`] per head proposal, [`AcceptRule::observe_verify`],
/// [`AcceptRule::accept`] per lane and [`AcceptRule::on_publish`] after the
/// step published.
pub struct LaneSampler {
    vocab: u32,
    lanes: HashMap<RequestEpoch, LaneSamplerState>,
    rows: Option<RowHandle>,
    host: Vec<f32>,
    scratch: Vec<(u32, f32)>,
    target: SparseDist,
}

impl LaneSampler {
    /// `vocab`: the target vocabulary (head candidate ids at or above it are
    /// dropped, as `Qwen4Bundle::mtp_draft_dist` does).
    pub fn new(vocab: usize) -> Self {
        Self {
            vocab: vocab as u32,
            lanes: HashMap::new(),
            rows: None,
            host: Vec::new(),
            scratch: Vec::with_capacity(SampleSpec::MAX_POOL),
            target: SparseDist::default(),
        }
    }

    /// Register a sampled request: its sampler, the full rendered `prompt`
    /// (the penalty history's prompt tail) and the pending `seed` (the first
    /// token the prefill emitted; part of every window's history).
    pub fn admit(
        &mut self,
        epoch: RequestEpoch,
        cfg: &SpecRequestConfig,
        prompt: &[u32],
        seed: u32,
    ) -> Result<(), String> {
        if self.lanes.contains_key(&epoch) {
            return Err(format!("sampled lane {epoch:?} is already registered"));
        }
        let params = LaneSamplerParams::from_request(cfg)?;
        let mut history = PenaltyHistory::new(params.window);
        history.set_prompt(prompt);
        let mut state = LaneSamplerState {
            params,
            history,
            emitted: vec![seed],
            q: Vec::new(),
            q_key: None,
            points: Vec::new(),
        };
        state.trim_emitted();
        self.lanes.insert(epoch, state);
        Ok(())
    }

    /// Forget a request (released lane).
    pub fn release(&mut self, epoch: &RequestEpoch) {
        self.lanes.remove(epoch);
    }

    /// Whether `epoch` has a registered sampler.
    pub fn is_registered(&self, epoch: &RequestEpoch) -> bool {
        self.lanes.contains_key(epoch)
    }

    /// Leviathan acceptance of one lane window.
    fn accept_leviathan(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String> {
        let rows = self
            .rows
            .clone()
            .ok_or("Leviathan lane: the sampler owner returned no target rows")?;
        let epoch = lane.key.request_epoch;
        let Self {
            lanes,
            host,
            scratch,
            target,
            ..
        } = self;
        let st = lanes
            .get_mut(&epoch)
            .ok_or_else(|| format!("sampled lane {epoch:?} has no registered sampler"))?;
        let LaneSamplerState {
            params,
            history,
            emitted,
            q,
            q_key,
            points,
        } = st;
        history.begin_window(emitted);
        for &draft in drafts {
            history.push_draft(draft);
        }
        let qs: &[SparseDist] = match lane.source {
            ProposalSource::Copy => {
                if points.len() < drafts.len() {
                    points.resize_with(drafts.len(), SparseDist::default);
                }
                for (point, &draft) in points.iter_mut().zip(drafts) {
                    point.set_point_mass(draft);
                }
                &points[..drafts.len()]
            }
            _ if drafts.is_empty() => &[],
            _ => {
                if *q_key != Some(lane.key) {
                    return Err(format!(
                        "Leviathan lane {:?}: no draft distributions for this window",
                        lane.key
                    ));
                }
                q.get(..drafts.len()).ok_or_else(|| {
                    format!(
                        "Leviathan lane: {} draft distributions for {} drafts",
                        q.len(),
                        drafts.len()
                    )
                })?
            }
        };
        let spec = params.spec;
        let policy = &params.policy;
        let history = &*history;
        let key = lane.key;
        accept_sampled_prefix(drafts, qs, lane.eos, &mut rng.spec, target, |row, out| {
            rows.row_logits(&key, row, host)?;
            apply_logit_policy_cpu(host, history.row(row), policy);
            out.build_from_logits(host, spec, scratch)
        })
    }

    /// Naive (SpecInfer) acceptance of one lane window on the lane's AR
    /// xorshift32 stream: the root draws every verify row in emission order,
    /// up to the row the verdict ends on.
    fn accept_naive(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String> {
        let rows = self
            .rows
            .clone()
            .ok_or("naive lane: the sampler owner returned no target rows")?;
        let epoch = lane.key.request_epoch;
        let Self { lanes, host, .. } = self;
        let st = lanes
            .get_mut(&epoch)
            .ok_or_else(|| format!("sampled lane {epoch:?} has no registered sampler"))?;
        st.history.begin_window(&st.emitted);
        for &draft in drafts {
            st.history.push_draft(draft);
        }
        let policy = &st.params.policy;
        let history = &st.history;
        let key = lane.key;
        accept_naive_prefix(drafts, lane.eos, |row| {
            rows.row_logits(&key, row, host)?;
            apply_logit_policy_cpu(host, history.row(row), policy);
            Ok(sample_top_k_p_with_state(
                host,
                policy.temperature,
                policy.top_p,
                policy.top_k,
                policy.min_p,
                rng.naive_mut(),
            ))
        })
    }
}

impl AcceptRule for LaneSampler {
    fn accept(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        picks: &[u32],
        rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String> {
        match lane.sampling {
            SamplingMode::Greedy => Ok(accept_greedy_prefix(drafts, picks, lane.eos)),
            SamplingMode::NaiveSampled => self.accept_naive(lane, drafts, rng),
            SamplingMode::Leviathan => self.accept_leviathan(lane, drafts, rng),
        }
    }

    fn draft_token(
        &mut self,
        lane: &LanePlan,
        index: usize,
        prior_drafts: &[u32],
        proposal: &DraftProposal,
        rng: &mut LaneRng,
    ) -> Result<u32, String> {
        // Greedy and naive lanes propose the head's argmax; a copy lane has no
        // head rounds.
        if lane.sampling != SamplingMode::Leviathan || lane.source != ProposalSource::Native {
            return Ok(proposal.token);
        }
        let read = proposal
            .read
            .as_ref()
            .ok_or("Leviathan lane: the head proposal carries no candidate read")?;
        let vocab = self.vocab;
        let Self {
            lanes,
            host,
            scratch,
            ..
        } = self;
        let st = lanes
            .get_mut(&lane.key.request_epoch)
            .ok_or_else(|| format!("sampled lane {:?} has no registered sampler", lane.key))?;
        if index == 0 {
            st.q.clear();
            st.q_key = Some(lane.key);
        } else if st.q_key != Some(lane.key) || st.q.len() != index {
            return Err(format!(
                "Leviathan lane {:?}: draft {index} arrived with {} earlier distributions",
                lane.key,
                st.q.len()
            ));
        }
        st.history.begin_window(&st.emitted);
        for &draft in prior_drafts {
            st.history.push_draft(draft);
        }
        let mut q = SparseDist::default();
        read.build_dist(
            vocab,
            st.params.spec,
            st.history.row(index),
            &st.params.policy,
            host,
            scratch,
            &mut q,
        )?;
        // One uniform per head step, drawn before the verdict (the
        // singleton's order); a pruned proposal still consumed it.
        let token = q.sample(rng.next_f32());
        st.q.push(q);
        Ok(token)
    }

    fn observe_verify(&mut self, out: &SamplerOutput) {
        self.rows = out.rows.clone();
    }

    fn on_publish(&mut self, out: &StepOutput) {
        self.rows = None;
        for lane in &out.lanes {
            let epoch = lane.key.request_epoch;
            if lane.cancelled || lane.terminal {
                self.lanes.remove(&epoch);
                continue;
            }
            if let Some(st) = self.lanes.get_mut(&epoch) {
                st.emitted.extend_from_slice(&lane.emitted);
                st.trim_emitted();
                st.q.clear();
                st.q_key = None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (DESIGN §5.1 / §10.1 / §10.3: `cargo test -p hipfire-arch-qwen4 sampled_lane`)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::LabRoutes;
    use crate::lane_exec::LaneRoute;
    use crate::mesh_lane_protocol::{
        run_mesh_step, CommitDecision, DraftRound, HeadState, LaneAdmission, LaneAdmit, LaneKind,
        LaneMeshExecutor, LaneOutput, LaneRequest, MeshKind, MeshRoot, MeshStepKey,
        PreparedReceipt, ProtocolError, RankCostEvidence, RankDraftResult, RankError,
        RankOwnership, RankReady, RankSession, RankVerifyResult, RootConfig, SourceAccounting,
        StagePolicy, StepPhase, VerifyPlan, WindowBudgets, WindowStart,
    };
    use hipfire_runtime::llama::sample_top_k_p_with_state;

    // ---- statistics ------------------------------------------------------

    /// Pearson chi-square of `counts` (total `n`) against `expected`
    /// probabilities; cells with expected count under 5 are pooled. Returns
    /// `(statistic, degrees of freedom)`.
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
                    "a token outside the target support was emitted"
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
    fn critical(df: usize) -> f64 {
        let df = df as f64;
        let a = 2.0 / (9.0 * df);
        df * (1.0 - a + 4.75 * a.sqrt()).powi(3)
    }

    fn assert_law(what: &str, counts: &[u64], expected: &[f64]) {
        let n: u64 = counts.iter().sum();
        let (stat, df) = chi_square(counts, expected, n);
        let limit = critical(df);
        assert!(
            stat < limit,
            "{what}: chi-square {stat:.2} over {df} dof exceeds {limit:.2} (n = {n})"
        );
    }

    // ---- fixtures --------------------------------------------------------

    const V: usize = 10;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    fn key(tag: u64, window_seq: u64) -> LaneWindowKey {
        LaneWindowKey {
            request_epoch: epoch(tag),
            lane_generation: 1,
            window_seq,
        }
    }

    fn lane_plan(
        tag: u64,
        sampling: SamplingMode,
        source: ProposalSource,
        eos: Option<u32>,
    ) -> LanePlan {
        LanePlan {
            key: key(tag, 0),
            kind: LaneKind::Decode,
            position: 10,
            seed: 1,
            route: LaneRoute::Batched { k_max: 3 },
            source,
            max_depth: 3,
            copy_drafts: Vec::new(),
            sampling,
            eos,
            remaining: 1_000_000,
            budget: 3,
            head: HeadState::Aligned,
        }
    }

    fn request_cfg(
        temp: f32,
        top_p: f32,
        top_k: Option<u32>,
        min_p: f32,
        seed: u64,
    ) -> SpecRequestConfig {
        SpecRequestConfig {
            temp,
            top_p,
            top_k,
            min_p,
            rng_seed: seed,
            ..SpecRequestConfig::default()
        }
    }

    /// Target logits of request `tag` (the same for every verify row: a
    /// stationary model, so every emitted token is an iid draw of `p`).
    fn target_logits(tag: u64) -> Vec<f32> {
        let base = [2.0f32, 1.5, 1.2, 0.9, 0.5, 0.1, -0.3, -0.8, -1.5, -2.5];
        (0..V).map(|i| base[(i + tag as usize) % V]).collect()
    }

    /// The head's 8 rescored candidates for request `tag`: a different
    /// support and weighting than `p` (q != p, tokens of p missing from q).
    fn head_candidates(tag: u64) -> Vec<(u32, f32)> {
        let toks = [0usize, 2, 1, 5, 3, 7, 4, 6];
        let logits = [1.8f32, 1.7, 0.2, 1.0, 0.1, 0.9, -0.5, -1.0];
        toks.iter()
            .zip(logits)
            .map(|(&t, l)| (((t + tag as usize) % V) as u32, l))
            .collect()
    }

    fn argmax_of(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &l) in logits.iter().enumerate() {
            if l > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Independent f64 reference of the AR sampler's truncated distribution
    /// (`SampleSpec::cpu_ar` semantics, vocabulary below the 20-wide pool).
    fn ref_dist(logits: &[f32], t: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> Vec<f64> {
        let v = logits.len();
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        let max = logits[idx[0]] as f64;
        let w: Vec<f64> = idx
            .iter()
            .map(|&i| ((logits[i] as f64 - max) / t as f64).exp())
            .collect();
        let mut cap = match top_k {
            None => 20,
            Some(0) => 64,
            Some(k) => (k as usize).min(64),
        }
        .min(v);
        if min_p > 0.0 {
            let floor = min_p as f64 * w[0];
            if let Some(cut) = (1..cap).find(|&i| w[i] < floor) {
                cap = cut;
            }
        }
        let sum: f64 = w[..cap].iter().sum();
        let threshold = top_p as f64 * sum;
        let (mut cumulative, mut len) = (0.0, cap);
        for (i, &x) in w[..cap].iter().enumerate() {
            cumulative += x;
            if cumulative >= threshold {
                len = i + 1;
                break;
            }
        }
        let kept: f64 = w[..len].iter().sum();
        let mut out = vec![0.0; v];
        for i in 0..len {
            out[idx[i]] = w[i] / kept;
        }
        out
    }

    fn dense(dist: &SparseDist) -> Vec<f64> {
        let mut out = vec![0.0; V];
        for &(t, p) in dist.entries() {
            out[t as usize] = p as f64;
        }
        out
    }

    /// The target logits of every row of every lane, by request tag.
    struct Synth {
        target: HashMap<u64, Vec<f32>>,
        head: HashMap<u64, Vec<(u32, f32)>>,
    }

    impl Synth {
        fn new(tags: &[u64]) -> Arc<Self> {
            Arc::new(Self {
                target: tags.iter().map(|&t| (t, target_logits(t))).collect(),
                head: tags.iter().map(|&t| (t, head_candidates(t))).collect(),
            })
        }
    }

    struct SynthRows {
        synth: Arc<Synth>,
        fail: Arc<std::sync::atomic::AtomicBool>,
    }

    impl TargetRowSource for SynthRows {
        fn row_logits(
            &mut self,
            key: &LaneWindowKey,
            row: usize,
            host: &mut Vec<f32>,
        ) -> Result<(), String> {
            if self.fail.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(format!("injected row {row} read failure"));
            }
            let tag = key.request_epoch.request_tag;
            host.clear();
            host.extend_from_slice(self.synth.target.get(&tag).ok_or("no logits for lane")?);
            Ok(())
        }
    }

    /// A direct-rule harness: the sampler plus constant rows per tag.
    fn rule_sampler(tags: &[u64]) -> LaneSampler {
        let mut s = LaneSampler::new(V);
        s.rows = Some(RowHandle::new(SynthRows {
            synth: Synth::new(tags),
            fail: Arc::default(),
        }));
        s
    }

    fn proposal(tag: u64, window: u64, head: &[(u32, f32)]) -> DraftProposal {
        DraftProposal {
            key: key(tag, window),
            token: head
                .iter()
                .copied()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0,
            keep_going: true,
            drop_last: false,
            read: Some(DraftRead::Candidates(head.to_vec())),
        }
    }

    /// One Leviathan window of `k` native drafts at the rule level.
    fn leviathan_window(
        s: &mut LaneSampler,
        lane: &LanePlan,
        rng: &mut LaneRng,
        head: &[(u32, f32)],
        k: usize,
    ) -> (Vec<u32>, GreedyAccept) {
        let tag = lane.key.request_epoch.request_tag;
        let mut drafts = Vec::new();
        for i in 0..k {
            let p = proposal(tag, lane.key.window_seq, head);
            let t = s.draft_token(lane, i, &drafts, &p, rng).unwrap();
            drafts.push(t);
        }
        let picks = vec![0u32; k + 1];
        let acc = s.accept(lane, &drafts, &picks, rng).unwrap();
        (drafts, acc)
    }

    // ---- LaneRng ---------------------------------------------------------

    #[test]
    fn sampled_lane_rng_seeds_match_the_two_existing_contracts() {
        // Leviathan stream: exactly SpecRng::new(seed).
        let mut lane = LaneRng::new(0xDEAD_BEEF_1234);
        let mut spec = SpecRng::new(0xDEAD_BEEF_1234);
        for _ in 0..8 {
            assert_eq!(lane.next_u64(), spec.next_u64());
        }
        // Naive stream: reset_cpu_sampler_rng(seed as u32): the seed, 0 -> 1.
        assert_eq!(LaneRng::new(5).naive_state(), 5);
        assert_eq!(LaneRng::new(0).naive_state(), 1);
        assert_eq!(LaneRng::new((1u64 << 32) | 7).naive_state(), 7);
        assert_eq!(LaneRng::from_parts(SpecRng::new(1), 0).naive_state(), 1);
        // The fingerprint does not advance either stream.
        let r = LaneRng::new(42);
        assert_eq!(r.fingerprint(), r.fingerprint());
        let mut r2 = r;
        r2.next_u64();
        assert_ne!(r.fingerprint(), r2.fingerprint());
    }

    #[test]
    fn sampled_lane_draft_read_and_row_handle_equality() {
        let a = DraftRead::Candidates(vec![(1, 0.5), (2, f32::NAN)]);
        assert_eq!(a, a.clone(), "bitwise: NaN equals itself");
        assert_ne!(a, DraftRead::Candidates(vec![(1, 0.5), (2, 1.0)]));
        assert_ne!(a, DraftRead::FullRow(vec![0.5]));
        let h = RowHandle::new(SynthRows {
            synth: Synth::new(&[1]),
            fail: Arc::default(),
        });
        assert_eq!(h, h.clone());
        let other = RowHandle::new(SynthRows {
            synth: Synth::new(&[1]),
            fail: Arc::default(),
        });
        assert_ne!(h, other);
    }

    // ---- q: candidates, policy, truncation -------------------------------

    #[test]
    fn sampled_lane_draft_dist_filters_applies_policy_and_matches_runtime_builder() {
        let spec = SampleSpec::cpu_ar(0.9, 0.95, Some(5), 0.02);
        let policy = naive_target_sampler(&SpecRequestConfig {
            temp: 0.9,
            repeat_penalty: 1.3,
            repeat_window: 8,
            presence_penalty: 0.2,
            ..SpecRequestConfig::default()
        });
        let history = [3u32, 3, 1, 7];
        // Token 99 is outside the vocabulary: dropped.
        let cands = [(0u32, 1.8f32), (3, 1.7), (1, 0.4), (99, 5.0), (5, 1.0)];
        let mut scratch = Vec::new();
        let mut q = SparseDist::default();
        draft_dist_from_candidates(
            &cands,
            V as u32,
            spec,
            &history,
            &policy,
            &mut scratch,
            &mut q,
        )
        .unwrap();
        assert_eq!(q.prob(99), 0.0);
        // Manual: policy over exact candidate logits by token id, then build.
        let ids = [0u32, 3, 1, 5];
        let mut vals = [1.8f32, 1.7, 0.4, 1.0];
        apply_logit_policy_candidates_cpu(&ids, &mut vals, &history, &policy);
        let mut manual: Vec<(u32, f32)> = ids.iter().copied().zip(vals).collect();
        let mut want = SparseDist::default();
        want.build_from_candidates(&mut manual, spec).unwrap();
        assert_eq!(q.entries(), want.entries());
        let mass: f32 = q.entries().iter().map(|e| e.1).sum();
        assert!((mass - 1.0).abs() < 1e-5);

        // A head that does not re-score: the whole row through build_from_logits.
        let row: Vec<f32> = (0..V).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let mut host = Vec::new();
        let mut from_read = SparseDist::default();
        DraftRead::FullRow(row.clone())
            .build_dist(
                V as u32,
                spec,
                &history,
                &policy,
                &mut host,
                &mut scratch,
                &mut from_read,
            )
            .unwrap();
        let mut penalized = row;
        apply_logit_policy_cpu(&mut penalized, &history, &policy);
        let mut want = SparseDist::default();
        want.build_from_logits(&penalized, spec, &mut scratch)
            .unwrap();
        assert_eq!(from_read.entries(), want.entries());

        // No finite candidate: an error, never a greedy fallback.
        let err = draft_dist_from_candidates(
            &[(1, f32::NEG_INFINITY)],
            V as u32,
            spec,
            &[],
            &policy,
            &mut scratch,
            &mut q,
        );
        assert!(err.is_err());
        // More than the head's 8 candidates is a protocol violation.
        let many: Vec<(u32, f32)> = (0..9).map(|i| (i, i as f32)).collect();
        assert!(draft_dist_from_candidates(
            &many,
            V as u32,
            spec,
            &[],
            &policy,
            &mut scratch,
            &mut q
        )
        .is_err());
    }

    // ---- Leviathan law ---------------------------------------------------

    /// Acceptance law on synthetic p/q: the first token of a Leviathan window
    /// is distributed as p whatever q is, and the single-draft acceptance rate
    /// is sum_x min(p, q).
    #[test]
    fn sampled_lane_leviathan_acceptance_law_chi_square() {
        const N: u64 = 120_000;
        let spec = SampleSpec::cpu_ar(1.0, 1.0, None, 0.0);
        let logits = target_logits(1);
        let head = head_candidates(1);
        let mut p = SparseDist::default();
        p.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        let p_dense = dense(&p);
        let mut q = SparseDist::default();
        q.build_from_candidates(&mut head.clone(), spec).unwrap();
        let q_dense = dense(&q);
        assert!(q_dense
            .iter()
            .zip(&p_dense)
            .any(|(q, p)| *q == 0.0 && *p > 0.0));
        let accept_rate: f64 = p_dense.iter().zip(&q_dense).map(|(p, q)| p.min(*q)).sum();

        for k in [1usize, 2, 3] {
            let lane = lane_plan(1, SamplingMode::Leviathan, ProposalSource::Native, None);
            let mut s = rule_sampler(&[1]);
            s.admit(epoch(1), &request_cfg(1.0, 1.0, None, 0.0, 7), &[4, 5], 1)
                .unwrap();
            let mut rng = LaneRng::new(0x5EED + k as u64);
            let mut first = vec![0u64; V];
            let mut accepted_first = 0u64;
            for _ in 0..N {
                let (drafts, acc) = leviathan_window(&mut s, &lane, &mut rng, &head, k);
                first[acc.committed[0] as usize] += 1;
                if acc.accepted >= 1 {
                    accepted_first += 1;
                    assert_eq!(acc.committed[0], drafts[0]);
                }
            }
            assert_law(&format!("k={k} first emitted token ~ p"), &first, &p_dense);
            let rate = accepted_first as f64 / N as f64;
            let sigma = (accept_rate * (1.0 - accept_rate) / N as f64).sqrt();
            assert!(
                (rate - accept_rate).abs() < 5.0 * sigma,
                "k={k}: first-draft acceptance {rate:.4} vs sum min(p,q) {accept_rate:.4}"
            );
        }
    }

    /// The residual `(p - q)+` incl. support truncation: p keeps 3 tokens
    /// (top-k), q's support only partly overlaps it. Conditioned on the first
    /// draft's rejection the emitted token follows the normalized residual and
    /// never leaves p's support.
    #[test]
    fn sampled_lane_residual_support_truncation_and_conditional_law() {
        const N: u64 = 300_000;
        let spec = SampleSpec::cpu_ar(1.0, 1.0, Some(3), 0.0);
        let cfg = request_cfg(1.0, 1.0, Some(3), 0.0, 9);
        let logits = target_logits(2);
        let head = head_candidates(2);
        let mut p = SparseDist::default();
        p.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        let mut q = SparseDist::default();
        q.build_from_candidates(&mut head.clone(), spec).unwrap();
        let p_dense = dense(&p);
        let q_dense = dense(&q);
        assert_eq!(
            p_dense.iter().filter(|&&x| x > 0.0).count(),
            3,
            "top-k=3 truncation"
        );
        assert!(
            q_dense
                .iter()
                .zip(&p_dense)
                .any(|(q, p)| *q > 0.0 && *p == 0.0),
            "q has tokens outside p's support"
        );
        let residual: Vec<f64> = p_dense
            .iter()
            .zip(&q_dense)
            .map(|(p, q)| (p - q).max(0.0))
            .collect();
        let z: f64 = residual.iter().sum();
        assert!(z > 0.05, "a non-trivial residual");
        let expected: Vec<f64> = residual.iter().map(|r| r / z).collect();

        let lane = lane_plan(2, SamplingMode::Leviathan, ProposalSource::Native, None);
        let mut s = rule_sampler(&[2]);
        s.admit(epoch(2), &cfg, &[], 1).unwrap();
        let mut rng = LaneRng::new(0xBEEF);
        let mut rejected = vec![0u64; V];
        for _ in 0..N {
            let (_, acc) = leviathan_window(&mut s, &lane, &mut rng, &head, 1);
            assert!(p_dense[acc.committed[0] as usize] > 0.0, "left p's support");
            if acc.accepted == 0 {
                rejected[acc.committed[0] as usize] += 1;
            }
        }
        assert_law("residual after rejection", &rejected, &expected);
    }

    /// Top-p / top-k / min-p / temperature: the truncated target is the
    /// independent f64 reference, and a Leviathan window reproduces it.
    #[test]
    fn sampled_lane_top_p_top_k_min_p_temperature() {
        const N: u64 = 80_000;
        let cases: [(f32, f32, Option<u32>, f32); 6] = [
            (1.0, 1.0, None, 0.0),
            (0.7, 0.9, None, 0.0),
            (1.3, 1.0, Some(4), 0.0),
            (0.9, 0.8, Some(6), 0.1),
            (1.0, 1.0, None, 0.2),
            (0.5, 0.95, Some(3), 0.05),
        ];
        for (ci, &(t, top_p, top_k, min_p)) in cases.iter().enumerate() {
            let spec = SampleSpec::cpu_ar(t, top_p, top_k, min_p);
            let logits = target_logits(3);
            let want = ref_dist(&logits, t, top_p, top_k, min_p);
            let mut p = SparseDist::default();
            p.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
            for (tok, (&got, &w)) in dense(&p).iter().zip(&want).enumerate() {
                assert!(
                    (got - w).abs() < 2e-5,
                    "case {ci}: token {tok} p {got} vs reference {w}"
                );
            }
            let cfg = request_cfg(t, top_p, top_k, min_p, 100 + ci as u64);
            let lane = lane_plan(3, SamplingMode::Leviathan, ProposalSource::Native, None);
            let mut s = rule_sampler(&[3]);
            s.admit(epoch(3), &cfg, &[], 1).unwrap();
            let head = head_candidates(3);
            let mut rng = LaneRng::new(0xA11CE + ci as u64);
            let mut counts = vec![0u64; V];
            for _ in 0..N {
                let (_, acc) = leviathan_window(&mut s, &lane, &mut rng, &head, 2);
                counts[acc.committed[0] as usize] += 1;
            }
            assert_law(
                &format!("case {ci} (T={t} p={top_p} k={top_k:?} min_p={min_p})"),
                &counts,
                &want,
            );
        }
    }

    /// A copy proposal's q is the point mass delta(candidate): accept iff
    /// u < p(candidate); a rejection draws from p with the candidate removed;
    /// the emitted token is p either way. No head round, no draw for the
    /// proposal itself.
    #[test]
    fn sampled_lane_copy_point_mass_q() {
        const N: u64 = 150_000;
        let spec = SampleSpec::cpu_ar(1.0, 1.0, Some(4), 0.0);
        let cfg = request_cfg(1.0, 1.0, Some(4), 0.0, 3);
        let logits = target_logits(4);
        let want = ref_dist(&logits, 1.0, 1.0, Some(4), 0.0);
        let mut p = SparseDist::default();
        p.build_from_logits(&logits, spec, &mut Vec::new()).unwrap();
        let top = argmax_of(&logits);
        let outside = (0..V as u32).find(|&t| want[t as usize] == 0.0).unwrap();
        for (name, cand) in [("in-support", top), ("out-of-support", outside)] {
            let mut lane = lane_plan(4, SamplingMode::Leviathan, ProposalSource::Copy, None);
            lane.copy_drafts = vec![cand];
            let mut s = rule_sampler(&[4]);
            s.admit(epoch(4), &cfg, &[9], 1).unwrap();
            let mut rng = LaneRng::new(77);
            let before = rng.fingerprint();
            // The copy lane never draws a proposal.
            let prop = DraftProposal::greedy_for_test(lane.key, cand);
            assert_eq!(s.draft_token(&lane, 0, &[], &prop, &mut rng).unwrap(), cand);
            assert_eq!(rng.fingerprint(), before);
            let mut counts = vec![0u64; V];
            let mut accepted = 0u64;
            for _ in 0..N {
                let acc = s.accept(&lane, &[cand], &[0, 0], &mut rng).unwrap();
                counts[acc.committed[0] as usize] += 1;
                accepted += acc.accepted as u64;
            }
            assert_law(&format!("copy {name}: emitted ~ p"), &counts, &want);
            let pc = want[cand as usize];
            let rate = accepted as f64 / N as f64;
            let sigma = (pc * (1.0 - pc) / N as f64).sqrt().max(1e-9);
            assert!(
                (rate - pc).abs() < 5.0 * sigma,
                "copy {name}: acceptance {rate:.4} vs p(candidate) {pc:.4}"
            );
            if name == "out-of-support" {
                assert_eq!(accepted, 0, "p(x) = 0 is never accepted");
            }
        }
    }

    impl DraftProposal {
        fn greedy_for_test(key: LaneWindowKey, token: u32) -> Self {
            Self {
                key,
                token,
                keep_going: true,
                drop_last: false,
                read: None,
            }
        }
    }

    /// An accepted EOS draft ends the window: no bonus, `hit_eos`.
    #[test]
    fn sampled_lane_accepted_eos_has_no_bonus() {
        // p = q = point mass on the EOS token (nucleus 0.01 keeps the top).
        let cfg = request_cfg(1.0, 0.01, None, 0.0, 5);
        let logits = target_logits(1);
        let eos = argmax_of(&logits);
        let lane = lane_plan(
            1,
            SamplingMode::Leviathan,
            ProposalSource::Native,
            Some(eos),
        );
        let mut s = rule_sampler(&[1]);
        s.admit(epoch(1), &cfg, &[], 1).unwrap();
        let mut rng = LaneRng::new(11);
        let head = vec![(eos, 3.0f32), ((eos + 1) % V as u32, -3.0)];
        let (drafts, acc) = leviathan_window(&mut s, &lane, &mut rng, &head, 3);
        assert_eq!(drafts[0], eos);
        assert_eq!(acc.committed, vec![eos], "accepted EOS: no bonus");
        assert_eq!((acc.accepted, acc.hit_eos), (1, true));
    }

    // ---- Naive root ------------------------------------------------------

    /// Naive mode: every emitted token is one draw of the AR sampler on the
    /// lane's xorshift32 stream, in emission order; the committed stream equals
    /// the AR stream however the head's drafts happen to agree.
    #[test]
    fn sampled_lane_naive_root_matches_ar_draw_stream() {
        let (temp, top_p, top_k, min_p) = (0.8f32, 0.9f32, Some(6u32), 0.02f32);
        let cfg = request_cfg(temp, top_p, top_k, min_p, 4242);
        let logits = target_logits(5);
        let ar = |state: &mut u32| {
            sample_top_k_p_with_state(
                &logits,
                temp,
                top_p,
                top_k,
                (min_p > 0.0).then_some(min_p),
                state,
            )
        };
        let mut ar_state = 4242u32;
        let mut s = rule_sampler(&[5]);
        s.admit(epoch(5), &cfg, &[], 1).unwrap();
        let lane = lane_plan(5, SamplingMode::NaiveSampled, ProposalSource::Native, None);
        let mut rng = LaneRng::new(4242);
        let spec_before = rng.fingerprint().0;
        let mut head_rng = SpecRng::new(1);
        let (mut emitted, mut reference) = (Vec::new(), Vec::new());
        for window in 0..400 {
            // Head drafts: the next AR draws, each corrupted a third of the time.
            let mut peek = ar_state;
            let k = 1 + window % 3;
            let drafts: Vec<u32> = (0..k)
                .map(|_| {
                    let t = ar(&mut peek);
                    if head_rng.next_u64() % 3 == 0 {
                        (t + 1) % V as u32
                    } else {
                        t
                    }
                })
                .collect();
            let acc = s.accept(&lane, &drafts, &vec![0; k + 1], &mut rng).unwrap();
            for &tok in &acc.committed {
                emitted.push(tok);
                reference.push(ar(&mut ar_state));
            }
            assert_eq!(
                rng.naive_state(),
                ar_state,
                "window {window}: one draw per emitted token"
            );
        }
        assert_eq!(emitted, reference);
        assert_eq!(
            rng.fingerprint().0,
            spec_before,
            "naive mode never draws the Leviathan stream"
        );
        assert!(emitted.len() > 400);
    }

    // ---- Singleton identity incl. penalty history -----------------------

    /// Same seed, same distributions: the lane stack (root-drawn proposals,
    /// history from `on_publish`) equals the singleton's batched window
    /// (`sample_draft` + `accept_sampled_prefix` over `PenaltyHistory`),
    /// token for token and draw for draw, across windows.
    #[test]
    fn sampled_lane_leviathan_equals_singleton_batched_window_with_penalties() {
        let cfg = SpecRequestConfig {
            temp: 0.9,
            top_p: 0.95,
            top_k: Some(8),
            min_p: 0.01,
            rng_seed: 31337,
            repeat_penalty: 1.25,
            repeat_window: 6,
            presence_penalty: 0.15,
            frequency_penalty: 0.1,
            ..SpecRequestConfig::default()
        };
        let prompt = [3u32, 4, 4, 9, 1];
        let seed_token = 2u32;
        let eos = Some(8u32);
        let params = LaneSamplerParams::from_request(&cfg).unwrap();
        let logits = target_logits(6);
        let head = head_candidates(6);

        let mut s = rule_sampler(&[6]);
        let rows = s.rows.clone();
        s.admit(epoch(6), &cfg, &prompt, seed_token).unwrap();
        let mut rng = LaneRng::new(cfg.rng_seed);

        // Singleton reference state.
        let mut ref_rng = SpecRng::new(cfg.rng_seed);
        let mut ref_hist = PenaltyHistory::new(cfg.penalty_window());
        ref_hist.set_prompt(&prompt);
        let mut ref_emitted = vec![seed_token];
        let mut scratch = Vec::new();
        let mut host = Vec::new();

        for window in 0..60u64 {
            // The sampler owner re-publishes its rows every step.
            s.rows = rows.clone();
            let k = 1 + (window as usize % 3);
            let mut lane = lane_plan(6, SamplingMode::Leviathan, ProposalSource::Native, eos);
            lane.key = key(6, window);
            let (drafts, acc) = leviathan_window(&mut s, &lane, &mut rng, &head, k);

            // Singleton: sample_draft per step, then accept_leviathan.
            ref_hist.begin_window(&ref_emitted);
            let mut qs = Vec::new();
            let mut ref_drafts = Vec::new();
            for i in 0..k {
                let mut q = SparseDist::default();
                draft_dist_from_candidates(
                    &head,
                    V as u32,
                    params.spec,
                    ref_hist.row(i),
                    &params.policy,
                    &mut scratch,
                    &mut q,
                )
                .unwrap();
                let t = q.sample(ref_rng.next_f32());
                qs.push(q);
                ref_drafts.push(t);
                ref_hist.push_draft(t);
            }
            let mut target = SparseDist::default();
            let ref_acc = accept_sampled_prefix(
                &ref_drafts,
                &qs,
                eos,
                &mut ref_rng,
                &mut target,
                |row, out| {
                    host.clear();
                    host.extend_from_slice(&logits);
                    apply_logit_policy_cpu(&mut host, ref_hist.row(row), &params.policy);
                    out.build_from_logits(&host, params.spec, &mut scratch)
                },
            )
            .unwrap();
            assert_eq!(drafts, ref_drafts, "window {window}: proposals");
            assert_eq!(acc, ref_acc, "window {window}: verdict");
            assert_eq!(
                rng.fingerprint().0,
                {
                    let mut r = ref_rng;
                    r.next_u64()
                },
                "window {window}: stream position"
            );
            ref_emitted.extend_from_slice(&ref_acc.committed);

            // Publish: the rule folds the emitted tokens into its history.
            let out = StepOutput {
                key: MeshStepKey {
                    topology_generation: 1,
                    step_seq: window + 1,
                    lanes: vec![lane.key],
                    policy_digest: 0,
                },
                lanes: vec![LaneOutput {
                    key: lane.key,
                    emitted: acc.committed.clone(),
                    forced: false,
                    terminal: false,
                    cancelled: false,
                    consumed_rows: 1 + acc.accepted,
                    next_seed: *acc.committed.last().unwrap(),
                    source: SourceAccounting {
                        source: ProposalSource::Native,
                        offered: k,
                        accepted: acc.accepted,
                    },
                }],
            };
            s.on_publish(&out);
            let keep = cfg.penalty_window();
            let tail: Vec<u32> = ref_emitted[ref_emitted.len().saturating_sub(keep)..].to_vec();
            assert_eq!(
                s.lanes[&epoch(6)].emitted,
                tail,
                "window {window}: history tail"
            );
            if acc.hit_eos {
                // EOS ends the request in production; the singleton identity
                // continues on a fresh stream, so stop here.
                return;
            }
        }
    }

    // ---- Protocol-level arms --------------------------------------------

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Fail {
        Draft(u32),
        Verify,
        Prepare,
    }

    /// An EP/PP sampler-owner rank over the synthetic model. It copies the
    /// head's candidates out of ONE shared draft buffer right after each op,
    /// as the real operator forces (the buffer is overwritten by the next
    /// lane's Draft).
    struct SynthExec {
        session: RankSession,
        synth: Arc<Synth>,
        rows: RowHandle,
        row_fail: Arc<std::sync::atomic::AtomicBool>,
        modes: HashMap<LaneWindowKey, SamplingMode>,
        buffer: Vec<(u32, f32)>,
        fail: Option<Fail>,
    }

    impl SynthExec {
        fn new(own: RankOwnership, synth: Arc<Synth>) -> Self {
            let row_fail: Arc<std::sync::atomic::AtomicBool> = Arc::default();
            Self {
                session: RankSession::new(own),
                rows: RowHandle::new(SynthRows {
                    synth: synth.clone(),
                    fail: row_fail.clone(),
                }),
                synth,
                row_fail,
                modes: HashMap::new(),
                buffer: Vec::new(),
                fail: None,
            }
        }

        fn rank(&self) -> usize {
            self.session.ownership().rank
        }

        fn err(&self, why: &str) -> RankError {
            RankError {
                rank: self.rank(),
                reason: why.to_string(),
            }
        }
    }

    impl LaneMeshExecutor for SynthExec {
        fn ownership(&self) -> RankOwnership {
            self.session.ownership()
        }

        fn admit_lane(&mut self, admit: &LaneAdmit) -> Result<(), RankError> {
            self.session.admit(admit)
        }

        fn release_lane(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
            self.session.release(epoch)
        }

        fn begin_window(&mut self, start: &WindowStart) -> RankReady {
            self.modes = start.lanes.iter().map(|l| (l.key, l.sampling)).collect();
            self.session.begin(start, Ok(()))
        }

        fn reject_window(&mut self, key: &MeshStepKey) {
            self.session.reject(key);
        }

        fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
            if self.fail == Some(Fail::Draft(round.round)) {
                return Err(self.err("injected head failure"));
            }
            let mut res = self.session.on_draft_round(round)?;
            if self.session.ownership().sampler {
                let mut props = Vec::new();
                for op in &round.ops {
                    let tag = op.key.request_epoch.request_tag;
                    // The shared operator's draft buffer: the next lane's op
                    // overwrites it.
                    self.buffer = self.synth.head[&tag].clone();
                    let token = self
                        .buffer
                        .iter()
                        .copied()
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap()
                        .0;
                    let read = (self.modes[&op.key] == SamplingMode::Leviathan)
                        .then(|| DraftRead::Candidates(self.buffer.clone()));
                    props.push(DraftProposal {
                        key: op.key,
                        token,
                        keep_going: true,
                        drop_last: false,
                        read,
                    });
                }
                res.proposals = Some(props);
            }
            Ok(res)
        }

        fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
            if self.fail == Some(Fail::Verify) {
                return Err(self.err("injected target failure"));
            }
            let mut res = self.session.on_verify(plan)?;
            if self.session.ownership().sampler {
                let picks = plan
                    .lanes
                    .iter()
                    .map(|l| {
                        let tag = l.key.request_epoch.request_tag;
                        let pick = argmax_of(&self.synth.target[&tag]);
                        (l.key, vec![pick; l.rows.len()])
                    })
                    .collect();
                res.sampler = Some(SamplerOutput {
                    picks,
                    rows: Some(self.rows.clone()),
                });
            }
            Ok(res)
        }

        fn prepare_commit(
            &mut self,
            decision: &CommitDecision,
        ) -> Result<PreparedReceipt, RankError> {
            if self.fail == Some(Fail::Prepare) {
                return Err(self.err("injected repair failure"));
            }
            self.session.on_prepare(decision)
        }

        fn publish_commit(&mut self, decision: &CommitDecision) {
            self.session.on_publish(decision);
        }

        fn abort_step(&mut self, _key: &MeshStepKey, _phase: StepPhase, _reason: &str) {
            self.session.on_abort();
        }

        fn evidence(&mut self, _key: &MeshStepKey) -> Option<RankCostEvidence> {
            None
        }
    }

    struct Mesh {
        root: MeshRoot,
        ranks: Vec<SynthExec>,
        sampler: LaneSampler,
    }

    #[derive(Clone)]
    struct Arm {
        name: &'static str,
        lab: LabRoutes,
        spec_k: Option<usize>,
        sampling: SamplingMode,
        copy: Vec<u32>,
    }

    impl Mesh {
        fn new(kind: MeshKind, tags: &[u64]) -> Self {
            Self::with_floor(kind, tags, false)
        }

        fn with_floor(kind: MeshKind, tags: &[u64], floor: bool) -> Self {
            let world = match kind {
                MeshKind::Ep => vec![RankOwnership::ep(0), RankOwnership::ep(1)],
                MeshKind::Pp => vec![RankOwnership::pp(0, false), RankOwnership::pp(1, true)],
            };
            let own = *world.iter().find(|r| r.root).unwrap();
            let root = MeshRoot::new(
                own,
                &world,
                RootConfig {
                    topology_generation: 7,
                    stage: StagePolicy {
                        moe_layers: 48,
                        moe_once: true,
                        stream_bf16: false,
                    },
                    floor_enabled: floor,
                },
            )
            .unwrap();
            let synth = Synth::new(tags);
            let ranks = world
                .iter()
                .map(|&o| SynthExec::new(o, synth.clone()))
                .collect();
            let mut sampler = LaneSampler::new(V);
            // The sampler owner's rank is the one that returns rows; the rule
            // learns the handle from `observe_verify`.
            sampler.rows = None;
            Self {
                root,
                ranks,
                sampler,
            }
        }

        fn admit(
            &mut self,
            tag: u64,
            cfg: &SpecRequestConfig,
            spec_k: Option<usize>,
            sampled: bool,
        ) {
            let prompt = vec![1u32, 2, 3, tag as u32];
            let a = self
                .root
                .admit_lane(LaneAdmission {
                    epoch: epoch(tag),
                    position: 100,
                    seed: 1,
                    eos: None,
                    rng_seed: cfg.rng_seed,
                    prompt: prompt.clone(),
                    spec_k,
                    head: HeadState::Aligned,
                })
                .unwrap();
            for r in &mut self.ranks {
                r.admit_lane(&a).unwrap();
            }
            if sampled {
                self.sampler.admit(epoch(tag), cfg, &prompt, 1).unwrap();
            }
        }

        fn step(&mut self, arm: &Arm, tags: &[u64]) -> Result<StepOutput, ProtocolError> {
            let lanes: Vec<(u64, SamplingMode)> = tags.iter().map(|&t| (t, arm.sampling)).collect();
            self.step_modes(arm, &lanes)
        }

        /// One step whose lanes carry their own sampling modes.
        fn step_modes(
            &mut self,
            arm: &Arm,
            lanes: &[(u64, SamplingMode)],
        ) -> Result<StepOutput, ProtocolError> {
            let k = arm.spec_k.unwrap_or(0).max(3);
            let budgets = WindowBudgets {
                max_lanes: 8,
                verify_rows_per_lane: k + 1,
                max_rows: 8 * (k + 1),
                copy_take_cap: 7,
                lab: arm.lab,
                pp_microbatch_lanes: None,
            };
            let reqs: Vec<LaneRequest> = lanes
                .iter()
                .map(|&(t, mode)| LaneRequest {
                    epoch: epoch(t),
                    remaining: 1_000_000,
                    context_left: 1_000_000,
                    copy_candidates: arm.copy.clone(),
                    forced: None,
                    sampling: mode,
                })
                .collect();
            let mut refs: Vec<&mut dyn LaneMeshExecutor> = self
                .ranks
                .iter_mut()
                .map(|r| r as &mut dyn LaneMeshExecutor)
                .collect();
            run_mesh_step(
                &mut self.root,
                &mut refs,
                None,
                &reqs,
                &budgets,
                &mut self.sampler,
            )
        }
    }

    struct ArmRun {
        emitted: HashMap<u64, Vec<u32>>,
        fingerprints: HashMap<u64, (u64, u32)>,
        sources: HashMap<usize, usize>,
        accepted: usize,
    }

    fn run_arm(kind: MeshKind, arm: &Arm, tags: &[u64], steps: usize, order: &[u64]) -> ArmRun {
        let mut m = Mesh::with_floor(kind, tags, arm.lab == LabRoutes::Floor);
        for &t in tags {
            m.admit(
                t,
                &request_cfg(1.0, 1.0, None, 0.0, 9000 + t * 7919),
                arm.spec_k,
                arm.sampling != SamplingMode::Greedy,
            );
        }
        let mut run = ArmRun {
            emitted: tags.iter().map(|&t| (t, Vec::new())).collect(),
            fingerprints: HashMap::new(),
            sources: HashMap::new(),
            accepted: 0,
        };
        for _ in 0..steps {
            let out = m.step(arm, order).unwrap();
            for l in &out.lanes {
                run.emitted
                    .get_mut(&l.key.request_epoch.request_tag)
                    .unwrap()
                    .extend_from_slice(&l.emitted);
                *run.sources.entry(l.source.source as usize).or_default() += 1;
                run.accepted += l.source.accepted;
            }
        }
        for &t in tags {
            run.fingerprints
                .insert(t, m.root.rng_fingerprint(epoch(t)).unwrap());
        }
        run
    }

    fn arms() -> Vec<Arm> {
        vec![
            Arm {
                name: "lane_ar",
                lab: LabRoutes::Batched,
                spec_k: None,
                sampling: SamplingMode::Leviathan,
                copy: Vec::new(),
            },
            Arm {
                name: "lane_batched",
                lab: LabRoutes::Batched,
                spec_k: Some(3),
                sampling: SamplingMode::Leviathan,
                copy: Vec::new(),
            },
            Arm {
                name: "lane_batched_k1",
                lab: LabRoutes::Batched,
                spec_k: Some(1),
                sampling: SamplingMode::Leviathan,
                copy: Vec::new(),
            },
            Arm {
                // Cycle: native windows, shallow native, and ArSynced windows
                // that take the root's copy candidates (point-mass q).
                name: "lane_copy",
                lab: LabRoutes::Cycle,
                spec_k: Some(3),
                sampling: SamplingMode::Leviathan,
                copy: vec![0, 1, 2],
            },
            Arm {
                // The root with the floor enabled but no priced evidence
                // supplied (merge_evidence/pricing is the protocol owner's;
                // the policy then drafts its budget): the law is route-blind.
                name: "lane_adaptive",
                lab: LabRoutes::Floor,
                spec_k: Some(3),
                sampling: SamplingMode::Leviathan,
                copy: vec![0, 1, 2],
            },
        ]
    }

    /// Per-lane marginal and non-overlapping pair law of the emitted stream
    /// (the stationary target makes every emitted token an iid draw of p).
    fn assert_stream_law(what: &str, tokens: &[u32], p: &[f64]) {
        let mut marginal = vec![0u64; V];
        for &t in tokens {
            marginal[t as usize] += 1;
        }
        assert_law(&format!("{what}: marginal"), &marginal, p);
        let mut pairs = vec![0u64; V * V];
        for w in tokens.chunks_exact(2) {
            pairs[w[0] as usize * V + w[1] as usize] += 1;
        }
        let joint: Vec<f64> = (0..V * V).map(|i| p[i / V] * p[i % V]).collect();
        assert_law(&format!("{what}: pairs"), &pairs, &joint);
    }

    /// Lane oracle arms (lane_ar / lane_batched / lane_copy) through the whole
    /// protocol on both topologies: each lane's emitted stream is iid p with
    /// its own p, lanes carry different p and q (per-lane laws, shared draft
    /// buffer), every proposal source is exercised.
    #[test]
    fn sampled_lane_oracle_arms_match_the_target_per_lane() {
        let tags = [1u64, 2, 3];
        for kind in [MeshKind::Ep, MeshKind::Pp] {
            for arm in arms() {
                let run = run_arm(kind, &arm, &tags, 9_000, &tags);
                for &t in &tags {
                    let want = ref_dist(&target_logits(t), 1.0, 1.0, None, 0.0);
                    assert_stream_law(
                        &format!("{kind:?} {} lane {t}", arm.name),
                        &run.emitted[&t],
                        &want,
                    );
                }
                match arm.name {
                    "lane_ar" => {
                        assert!(!run.sources.contains_key(&(ProposalSource::Native as usize)))
                    }
                    "lane_copy" => {
                        assert!(
                            run.sources.contains_key(&(ProposalSource::Copy as usize)),
                            "{:?}",
                            run.sources
                        );
                        assert!(
                            run.sources.contains_key(&(ProposalSource::Native as usize)),
                            "{:?}",
                            run.sources
                        );
                    }
                    _ => assert!(run.accepted > 0, "{}: no draft was ever accepted", arm.name),
                }
            }
        }
    }

    /// The statistic has power: accepting every draft (a biased verifier)
    /// fails the same law check.
    #[test]
    fn sampled_lane_negative_control_biased_acceptance_is_detected() {
        struct AcceptAll;
        impl AcceptRule for AcceptAll {
            fn accept(
                &mut self,
                _lane: &LanePlan,
                drafts: &[u32],
                picks: &[u32],
                _rng: &mut LaneRng,
            ) -> Result<GreedyAccept, String> {
                Ok(accept_greedy_prefix(drafts, picks, None))
            }
        }
        // Greedy accept-prefix over argmax picks emits argmax(target) forever.
        let arm = Arm {
            name: "biased",
            lab: LabRoutes::Batched,
            spec_k: Some(3),
            sampling: SamplingMode::Greedy,
            copy: Vec::new(),
        };
        let mut m = Mesh::new(MeshKind::Ep, &[1]);
        m.admit(1, &request_cfg(1.0, 1.0, None, 0.0, 1), arm.spec_k, false);
        let mut tokens = Vec::new();
        for _ in 0..600 {
            let k = 3usize;
            let budgets = WindowBudgets {
                max_lanes: 8,
                verify_rows_per_lane: k + 1,
                max_rows: 8 * (k + 1),
                copy_take_cap: 7,
                lab: arm.lab,
                pp_microbatch_lanes: None,
            };
            let reqs = [LaneRequest {
                epoch: epoch(1),
                remaining: 1_000_000,
                context_left: 1_000_000,
                copy_candidates: Vec::new(),
                forced: None,
                sampling: SamplingMode::Greedy,
            }];
            let mut refs: Vec<&mut dyn LaneMeshExecutor> = m
                .ranks
                .iter_mut()
                .map(|r| r as &mut dyn LaneMeshExecutor)
                .collect();
            let out = run_mesh_step(
                &mut m.root,
                &mut refs,
                None,
                &reqs,
                &budgets,
                &mut AcceptAll,
            )
            .unwrap();
            tokens.extend_from_slice(&out.lanes[0].emitted);
        }
        let want = ref_dist(&target_logits(1), 1.0, 1.0, None, 0.0);
        let mut marginal = vec![0u64; V];
        for &t in &tokens {
            marginal[t as usize] += 1;
        }
        let (stat, df) = chi_square(&marginal, &want, tokens.len() as u64);
        assert!(
            stat > critical(df),
            "the law check must reject a greedy stream ({stat})"
        );
    }

    /// Lane scheduling never perturbs a request's stream: alone, co-scheduled,
    /// permuted, EP or PP, and a late arrival all give the same per-request
    /// tokens and the same final RNG state.
    #[test]
    fn sampled_lane_rng_invariant_to_lane_scheduling() {
        let arm = arms().remove(1);
        let steps = 300;
        let alone = run_arm(MeshKind::Ep, &arm, &[2], steps, &[2]);
        let together = run_arm(MeshKind::Ep, &arm, &[1, 2, 3], steps, &[1, 2, 3]);
        let permuted = run_arm(MeshKind::Ep, &arm, &[1, 2, 3], steps, &[3, 2, 1]);
        let pp = run_arm(MeshKind::Pp, &arm, &[1, 2, 3], steps, &[2, 3, 1]);
        for other in [&together, &permuted, &pp] {
            assert_eq!(other.emitted[&2], alone.emitted[&2]);
            assert_eq!(other.fingerprints[&2], alone.fingerprints[&2]);
        }
        for t in [1u64, 3] {
            assert_eq!(together.emitted[&t], permuted.emitted[&t]);
            assert_eq!(together.fingerprints[&t], pp.fingerprints[&t]);
        }
        // A late arrival: lane 2 joins after 11 steps of lanes 1 and 3 and runs
        // `steps` windows: identical to its solo run.
        let mut m = Mesh::new(MeshKind::Ep, &[1, 2, 3]);
        for t in [1u64, 3] {
            m.admit(
                t,
                &request_cfg(1.0, 1.0, None, 0.0, 9000 + t * 7919),
                arm.spec_k,
                true,
            );
        }
        for _ in 0..11 {
            m.step(&arm, &[1, 3]).unwrap();
        }
        m.admit(
            2,
            &request_cfg(1.0, 1.0, None, 0.0, 9000 + 2 * 7919),
            arm.spec_k,
            true,
        );
        let mut late = Vec::new();
        for _ in 0..steps {
            let out = m.step(&arm, &[3, 2, 1]).unwrap();
            late.extend_from_slice(
                &out.lanes
                    .iter()
                    .find(|l| l.key.request_epoch.request_tag == 2)
                    .unwrap()
                    .emitted,
            );
        }
        assert_eq!(late, alone.emitted[&2]);
        assert_eq!(
            m.root.rng_fingerprint(epoch(2)).unwrap(),
            alone.fingerprints[&2]
        );
    }

    #[test]
    fn sampled_lane_deterministic_replay() {
        for arm in arms() {
            let a = run_arm(MeshKind::Ep, &arm, &[1, 2], 400, &[1, 2]);
            let b = run_arm(MeshKind::Ep, &arm, &[1, 2], 400, &[1, 2]);
            assert_eq!(a.emitted, b.emitted, "{}", arm.name);
            assert_eq!(a.fingerprints, b.fingerprints, "{}", arm.name);
        }
        // A different seed is a different stream.
        let arm = arms().remove(1);
        let a = run_arm(MeshKind::Ep, &arm, &[1], 200, &[1]);
        let b = run_arm(MeshKind::Ep, &arm, &[2], 200, &[2]);
        assert_ne!(a.emitted[&1], b.emitted[&2]);
    }

    /// The staged RNG successor is published exactly once, by exactly the
    /// draws the window consumed: one per head step, then (per accepted draft
    /// 1) + 2 on a rejection or 1 for the bonus.
    #[test]
    fn sampled_lane_publish_advances_by_the_window_draws() {
        let arm = arms().remove(1);
        let mut m = Mesh::new(MeshKind::Ep, &[1]);
        let cfg = request_cfg(1.0, 1.0, None, 0.0, 777);
        m.admit(1, &cfg, arm.spec_k, true);
        let mut reference = SpecRng::new(777);
        let mut saw = [false; 2];
        for _ in 0..200 {
            let out = m.step(&arm, &[1]).unwrap();
            let l = &out.lanes[0];
            let (offered, accepted) = (l.source.offered, l.source.accepted);
            let verdict = if accepted < offered {
                accepted + 2
            } else {
                offered + 1
            };
            saw[usize::from(accepted < offered)] = true;
            for _ in 0..offered + verdict {
                reference.next_u64();
            }
            let mut r = reference;
            assert_eq!(m.root.rng_probe(epoch(1)).unwrap(), r.next_u64());
        }
        assert!(
            saw[0] && saw[1],
            "both a full acceptance and a rejection occurred"
        );
    }

    /// Abort leaves the RNG, the history and the generated IDs unchanged,
    /// wherever it lands (head draws already consumed staged uniforms before
    /// the failure at verify / prepare / row read).
    #[test]
    fn sampled_lane_abort_does_not_advance_rng() {
        let arm = arms().remove(1);
        for fail in [
            Some(Fail::Draft(1)),
            Some(Fail::Verify),
            Some(Fail::Prepare),
            None,
        ] {
            let mut m = Mesh::new(MeshKind::Ep, &[1, 2]);
            for t in [1u64, 2] {
                m.admit(
                    t,
                    &request_cfg(1.0, 1.0, None, 0.0, 50 + t),
                    arm.spec_k,
                    true,
                );
            }
            // A published window first, so the stream is mid-flight.
            m.step(&arm, &[1, 2]).unwrap();
            let before: Vec<_> = [1u64, 2]
                .iter()
                .map(|&t| {
                    (
                        m.root.rng_fingerprint(epoch(t)),
                        m.root.generated(epoch(t)).map(<[u32]>::to_vec),
                        m.sampler.lanes[&epoch(t)].emitted.clone(),
                    )
                })
                .collect();
            match fail {
                Some(f) => m.ranks[1].fail = Some(f),
                // The rule cannot read a verify row: the acceptance itself fails.
                None => m.ranks[0]
                    .row_fail
                    .store(true, std::sync::atomic::Ordering::Relaxed),
            }
            let err = m.step(&arm, &[1, 2]).unwrap_err();
            assert!(
                matches!(err, ProtocolError::Aborted(_)),
                "{fail:?}: {err:?}"
            );
            let after: Vec<_> = [1u64, 2]
                .iter()
                .map(|&t| {
                    (
                        m.root.rng_fingerprint(epoch(t)),
                        m.root.generated(epoch(t)).map(<[u32]>::to_vec),
                        m.sampler.lanes[&epoch(t)].emitted.clone(),
                    )
                })
                .collect();
            assert_eq!(before, after, "{fail:?}");
            assert_eq!(m.root.phase(), StepPhase::Aborted);
        }
    }

    /// A Leviathan lane's proposal without the candidate read, or a rule
    /// without rows, aborts the step (never a greedy fallback).
    #[test]
    fn sampled_lane_missing_distribution_aborts_never_falls_back() {
        let lane = lane_plan(1, SamplingMode::Leviathan, ProposalSource::Native, None);
        let mut s = rule_sampler(&[1]);
        s.admit(epoch(1), &request_cfg(1.0, 1.0, None, 0.0, 1), &[], 1)
            .unwrap();
        let mut rng = LaneRng::new(1);
        let bare = DraftProposal::greedy_for_test(lane.key, 3);
        assert!(s.draft_token(&lane, 0, &[], &bare, &mut rng).is_err());
        // Accept with a draft but no recorded q.
        assert!(s.accept(&lane, &[3], &[0, 0], &mut rng).is_err());
        // No rows observed.
        s.rows = None;
        assert!(s.accept(&lane, &[], &[0], &mut rng).is_err());
        // An unregistered lane.
        let stray = lane_plan(9, SamplingMode::Leviathan, ProposalSource::Native, None);
        s.rows = Some(RowHandle::new(SynthRows {
            synth: Synth::new(&[9]),
            fail: Arc::default(),
        }));
        assert!(s.accept(&stray, &[], &[0], &mut rng).is_err());
        // Greedy lanes in the same step need nothing.
        let greedy = lane_plan(8, SamplingMode::Greedy, ProposalSource::Native, None);
        let acc = s.accept(&greedy, &[5, 6], &[5, 7, 9], &mut rng).unwrap();
        assert_eq!(acc.committed, vec![5, 7]);
    }

    /// A follower returning a distribution read is a protocol violation:
    /// followers neither propose nor draw.
    #[test]
    fn sampled_lane_followers_cannot_propose_or_draw() {
        let arm = arms().remove(1);
        let mut m = Mesh::new(MeshKind::Ep, &[1]);
        m.admit(1, &request_cfg(1.0, 1.0, None, 0.0, 1), arm.spec_k, true);
        // Make rank 1 (a follower) act as a sampler owner.
        let synth = m.ranks[1].synth.clone();
        let mut rogue = SynthExec::new(RankOwnership::ep(1), synth);
        rogue.session = RankSession::new(RankOwnership {
            sampler: true,
            ..RankOwnership::ep(1)
        });
        // Re-admit on the rogue session.
        let admit = LaneAdmit {
            epoch: epoch(1),
            lane_generation: 1,
            position: 100,
            spec: true,
        };
        rogue.admit_lane(&admit).unwrap();
        m.ranks[1] = rogue;
        let before = m.root.rng_fingerprint(epoch(1));
        let err = m.step(&arm, &[1]).unwrap_err();
        assert!(matches!(err, ProtocolError::Aborted(_)), "{err:?}");
        assert_eq!(m.root.rng_fingerprint(epoch(1)), before);
    }

    /// Greedy, naive-root and Leviathan lanes share a step: the greedy lane
    /// never draws, the naive lane emits exactly the AR sampler's stream (one
    /// xorshift32 draw per emitted token), the Leviathan lane is unperturbed
    /// by its neighbours.
    #[test]
    fn sampled_lane_mixed_modes_share_a_step() {
        let arm = arms().remove(1);
        let modes = [
            (1u64, SamplingMode::Greedy),
            (2, SamplingMode::Leviathan),
            (3, SamplingMode::NaiveSampled),
        ];
        let seed = |t: u64| 9000 + t * 7919;
        let mut m = Mesh::new(MeshKind::Ep, &[1, 2, 3]);
        for &(t, mode) in &modes {
            m.admit(
                t,
                &request_cfg(1.0, 1.0, None, 0.0, seed(t)),
                arm.spec_k,
                mode != SamplingMode::Greedy,
            );
        }
        let mut emitted: HashMap<u64, Vec<u32>> = HashMap::new();
        for _ in 0..500 {
            let out = m.step_modes(&arm, &modes).unwrap();
            for l in &out.lanes {
                emitted
                    .entry(l.key.request_epoch.request_tag)
                    .or_default()
                    .extend_from_slice(&l.emitted);
            }
        }
        // Greedy: the target argmax, no draw.
        let top = argmax_of(&target_logits(1));
        assert!(emitted[&1].iter().all(|&t| t == top));
        assert_eq!(
            m.root.rng_fingerprint(epoch(1)).unwrap(),
            LaneRng::new(seed(1)).fingerprint()
        );
        // Naive: the AR stream, one draw per emitted token.
        let mut state = seed(3) as u32;
        let logits = target_logits(3);
        for (i, &t) in emitted[&3].iter().enumerate() {
            let want = sample_top_k_p_with_state(&logits, 1.0, 1.0, None, None, &mut state);
            assert_eq!(t, want, "naive token {i}");
        }
        assert_eq!(m.root.rng_fingerprint(epoch(3)).unwrap().1, state);
        // Leviathan: identical to its solo run.
        let lev = Arm {
            sampling: SamplingMode::Leviathan,
            ..arm.clone()
        };
        let solo = run_arm(MeshKind::Ep, &lev, &[2], 500, &[2]);
        assert_eq!(emitted[&2], solo.emitted[&2]);
        assert_eq!(
            m.root.rng_fingerprint(epoch(2)).unwrap(),
            solo.fingerprints[&2]
        );
    }
}
