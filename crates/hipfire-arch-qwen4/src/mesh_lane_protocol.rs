// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host-only speculative-window transaction of the multicard (EP2 / PP2)
//! native-MTP lanes (`release-0.4.3/mgmtp/DESIGN.md` §3.2, §4).
//!
//! One authoritative root ([`MeshRoot`]) chooses route, depth, proposal
//! source and accepted prefix for every request; every other participant
//! ([`RankSession`], behind a [`LaneMeshExecutor`]) replays the root's
//! messages. Nothing in this module launches device work: topology owners
//! (S3 EP, S5 PP) implement [`LaneMeshExecutor`] and call the reducer in the
//! order [`run_mesh_step`] shows.
//!
//! ```text
//!  Idle ─plan_window→ Planned ─collect_ready→ Armed ─draft rounds→ Drafted
//!   ▲                                                                │ seal_verify
//!   │                                                                ▼ collect_verify
//!  Published ←publish_commit─ Prepared ←collect_prepared─ Decided ←decide_commit─ Verified
//!                         (Aborted is terminal and reachable from any phase)
//! ```
//!
//! Invariants enforced here:
//!
//! - **Root-only decisions.** Policy (route/depth/source), the per-request
//!   RNG, acceptance and `consumed_rows` exist only in [`MeshRoot`], which
//!   can only be built for the root/sampler rank. A follower result that
//!   carries proposals or sampler output aborts the step.
//! - **Plan purity.** [`MeshRoot::plan_window`] reads only root-owned state
//!   and the *merged* evidence of the previous step; rank-local timings
//!   reach it solely through [`merge_evidence`].
//! - **Two-phase publish.** Every mutation of root state (generated IDs,
//!   RNG successor, floor/agreement, cost fit, copy history, positions) is
//!   staged inside the step and applied by [`MeshRoot::publish_commit`],
//!   which is infallible and reachable only from `Prepared`. An abort at
//!   any earlier phase drops the staged state: nothing was published.
//! - **Consumed rows.** A lane consumes `1 + target_commit_accept_len`
//!   rows of its verify block; the bonus/correction token is predicted but
//!   not consumed, and an accepted EOS draft stays the pending seed.
//! - **Identity.** Slot numbers never identify a request: every message is
//!   keyed by [`LaneWindowKey`] (full [`RequestEpoch`], lane generation,
//!   window sequence) inside a [`MeshStepKey`].

use crate::gpu_forward_lanes::LANE_MAX_ROWS;
use crate::lane::LabRoutes;
use crate::lane_exec::LaneRoute;
use crate::mtp_lanes::{LaneCostModel, LaneSpecPolicy};
use crate::mtp_spec::target_commit_accept_len;
use crate::sampled_lane::{DraftRead, LaneRng, RowHandle};
use hipfire_runtime::slot_batch::RequestEpoch;
use hipfire_runtime::spec::{accept_greedy_prefix, GreedyAccept, MtpWindow};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

/// FNV-1a: deterministic across processes and toolchains (unlike
/// `DefaultHasher`), so root and ranks can compare plan digests.
struct Digest(u64);

impl Digest {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn u64(&mut self, v: u64) -> &mut Self {
        for b in v.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self
    }
    fn usize(&mut self, v: usize) -> &mut Self {
        self.u64(v as u64)
    }
    fn u32(&mut self, v: u32) -> &mut Self {
        self.u64(u64::from(v))
    }
    fn bool(&mut self, v: bool) -> &mut Self {
        self.u64(u64::from(v))
    }
    fn slice(&mut self, v: &[u32]) -> &mut Self {
        self.usize(v.len());
        for &t in v {
            self.u32(t);
        }
        self
    }
    fn lane_key(&mut self, k: &LaneWindowKey) -> &mut Self {
        self.u64(k.request_epoch.request_tag)
            .u64(k.request_epoch.owner_generation)
            .u64(k.lane_generation)
            .u64(k.window_seq)
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Keys and phases
// ---------------------------------------------------------------------------

/// One lane's window identity. `request_epoch` is the full
/// (tag, owner generation); `lane_generation` is the root's monotonic
/// admission counter (so a re-admitted epoch can never match a stale
/// message); `window_seq` counts the lane's published windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneWindowKey {
    pub request_epoch: RequestEpoch,
    pub lane_generation: u64,
    pub window_seq: u64,
}

/// One mesh step: topology generation, step sequence, the ordered lane keys
/// and the stage-policy digest. Every rank message echoes it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeshStepKey {
    pub topology_generation: u64,
    pub step_seq: u64,
    pub lanes: Vec<LaneWindowKey>,
    pub policy_digest: u64,
}

/// Where a step is. `Aborted` is terminal and reachable from every other
/// phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StepPhase {
    Idle,
    Planned,
    Armed,
    Drafted,
    Verified,
    Decided,
    Prepared,
    Published,
    Aborted,
}

/// Which topology a mesh is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshKind {
    Ep,
    Pp,
    /// One rank that is root, target, head and sampler (the single-card
    /// lanes). No gathers, no pipeline edges.
    Single,
}

/// What a rank owns. Device ordinal and layer order are NOT identity:
/// `rank` indexes the mesh's ranks/bands, and the PP root is the *output*
/// band wherever it sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RankOwnership {
    pub rank: usize,
    pub kind: MeshKind,
    /// Owns target state (every EP rank; every PP band for its layers).
    pub target: bool,
    /// Owns the MTP head operator and per-lane head state.
    pub head: bool,
    /// Returns target picks/probabilities to the root.
    pub sampler: bool,
    /// Runs the chooser, RNG and acceptance (exactly one per mesh).
    pub root: bool,
}

impl RankOwnership {
    /// EP: every rank owns the replicated target and a head share; rank 0 is
    /// the root and the only sampler.
    pub const fn ep(rank: usize) -> Self {
        Self {
            rank,
            kind: MeshKind::Ep,
            target: true,
            head: true,
            sampler: rank == 0,
            root: rank == 0,
        }
    }

    /// PP: every band owns target layers; only the output band owns the
    /// head, samples, and is the root.
    pub const fn pp(rank: usize, output_band: bool) -> Self {
        Self {
            rank,
            kind: MeshKind::Pp,
            target: true,
            head: output_band,
            sampler: output_band,
            root: output_band,
        }
    }
}

impl RankOwnership {
    /// Single card: the one rank owns target, head and sampling and is the
    /// root.
    pub const fn single() -> Self {
        Self {
            rank: 0,
            kind: MeshKind::Single,
            target: true,
            head: true,
            sampler: true,
            root: true,
        }
    }
}

fn validate_world(world: &[RankOwnership]) -> Result<RankOwnership, ProtocolError> {
    let bad = |m: &str| Err(ProtocolError::Config(m.to_string()));
    if world.is_empty() {
        return bad("empty rank set");
    }
    let mut seen = HashSet::new();
    if !world.iter().all(|r| seen.insert(r.rank)) {
        return bad("duplicate rank id");
    }
    if world.iter().any(|r| r.kind != world[0].kind) {
        return bad("mixed EP/PP ranks");
    }
    let roots: Vec<_> = world.iter().filter(|r| r.root).collect();
    if roots.len() != 1 {
        return bad("a mesh needs exactly one root rank");
    }
    let root = *roots[0];
    if !root.sampler || !root.head || !root.target {
        return bad("the root must own target, head and sampler");
    }
    if world.iter().filter(|r| r.sampler).count() != 1 {
        return bad("exactly one sampler owner");
    }
    if world.iter().any(|r| !r.target) {
        return bad("every rank owns target state");
    }
    if world[0].kind == MeshKind::Ep && world.iter().any(|r| !r.head) {
        return bad("EP: every rank runs the matched head calls");
    }
    if world[0].kind == MeshKind::Pp && world.iter().any(|r| r.head && !r.root) {
        return bad("PP: only the output band owns the head");
    }
    if world[0].kind == MeshKind::Single && world.len() != 1 {
        return bad("Single: exactly one rank");
    }
    Ok(root)
}

// ---------------------------------------------------------------------------
// Errors and abort record
// ---------------------------------------------------------------------------

/// The single failure record of an aborted step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbortRecord {
    pub key: Option<MeshStepKey>,
    pub phase: StepPhase,
    pub rank: Option<usize>,
    pub reason: String,
}

impl fmt::Display for AbortRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mesh step aborted in {:?}{}: {}",
            self.phase,
            self.rank
                .map(|r| format!(" (rank {r})"))
                .unwrap_or_default(),
            self.reason
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvidenceError {
    /// The receipts belong to another step (or topology generation).
    Stale(String),
    /// A rank of the mesh sent no receipt (or an unknown rank did).
    Missing(String),
    /// Operation counts disagree between ranks or with the decision.
    Incoherent(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    Config(String),
    /// A call out of phase order (a caller bug; state is unchanged).
    Phase {
        op: &'static str,
        actual: StepPhase,
    },
    /// The step was aborted (this call or earlier); the mesh is quarantined.
    Aborted(AbortRecord),
    UnknownLane(RequestEpoch),
    StaleEpoch(RequestEpoch),
    LaneNotLive(RequestEpoch),
    Budget(String),
    /// One or more ranks refused the plan before any device work.
    Preflight(Vec<(usize, String)>),
    StaleEvidence(String),
    Evidence(EvidenceError),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(m) => write!(f, "mesh lane protocol config: {m}"),
            Self::Phase { op, actual } => write!(f, "{op}: wrong phase {actual:?}"),
            Self::Aborted(r) => write!(f, "{r}"),
            Self::UnknownLane(e) => write!(f, "unknown lane {e:?}"),
            Self::StaleEpoch(e) => write!(f, "stale request epoch {e:?}"),
            Self::LaneNotLive(e) => write!(f, "lane {e:?} is terminal/cancelled"),
            Self::Budget(m) => write!(f, "plan exceeds admitted budget: {m}"),
            Self::Preflight(v) => write!(f, "preflight rejected by {v:?}"),
            Self::StaleEvidence(m) => write!(f, "stale evidence: {m}"),
            Self::Evidence(e) => write!(f, "evidence: {e:?}"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// A rank-side failure (device error, ownership violation, bad message).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankError {
    pub rank: usize,
    pub reason: String,
}

impl RankError {
    fn new(rank: usize, reason: impl Into<String>) -> Self {
        Self {
            rank,
            reason: reason.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Plan-side messages
// ---------------------------------------------------------------------------

/// How a lane's next token is produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingMode {
    Greedy,
    /// Same-seed naive lane sampling (existing AR contract); the draw
    /// authority is the root.
    NaiveSampled,
    /// Leviathan accept/residual (S4); root-only draws.
    Leviathan,
}

/// Where a window's proposals come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalSource {
    /// No proposals: an AR window (or forced rows).
    None,
    /// The native MTP head drafts (`Draft` ops in head rounds).
    Native,
    /// Root-known copy candidates (point-mass q); no head rounds.
    Copy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneKind {
    Decode,
    /// Emitter-forced tokens: not proposals, no RNG, no policy, no
    /// acceptance. Rows are `[seed, f1..f_{n-1}]`; `f_n` becomes the seed.
    Forced(Vec<u32>),
}

/// The head's alignment with the target, root-known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadState {
    /// Head position equals target position.
    Aligned,
    /// Head lags the target from `from` (retired lane keeping copy).
    Lag { from: usize },
    /// Truly head-free (no copy future).
    Free,
}

/// Where the head ends after the commit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadEnd {
    Aligned { end: usize },
    Lag { from: usize, to: usize },
    Free,
}

/// A request's admission at the root (mirrored to ranks as [`LaneAdmit`]).
#[derive(Clone, Debug)]
pub struct LaneAdmission {
    pub epoch: RequestEpoch,
    /// Target position of the seed (prompt length after prefill).
    pub position: usize,
    pub seed: u32,
    pub eos: Option<u32>,
    pub rng_seed: u64,
    /// Full prompt (seeds the request's copy history).
    pub prompt: Vec<u32>,
    /// Draft capacity of the lane (`None`: a non-spec AR lane).
    pub spec_k: Option<usize>,
    pub head: HeadState,
}

/// What a rank learns of an admission: identity only. No RNG, no policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneAdmit {
    pub epoch: RequestEpoch,
    pub lane_generation: u64,
    pub position: usize,
    pub spec: bool,
}

/// One lane's per-step request from the driver.
#[derive(Clone, Debug)]
pub struct LaneRequest {
    pub epoch: RequestEpoch,
    /// Tokens the request may still emit (> 0).
    pub remaining: usize,
    /// Positions left before `max_seq_len` (rows this window may consume).
    pub context_left: usize,
    /// Copy candidates the root's copy chooser offers this window (empty:
    /// none). Point-mass proposals; truncated at the first EOS.
    pub copy_candidates: Vec<u32>,
    /// Forced control tokens (not proposals).
    pub forced: Option<Vec<u32>>,
    pub sampling: SamplingMode,
}

/// The admitted per-step limits (the `LaneSpecBudget` slice the protocol
/// needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowBudgets {
    pub max_lanes: usize,
    /// `1 + max(native_k, armed_copy_take_cap)`, at most [`LANE_MAX_ROWS`].
    pub verify_rows_per_lane: usize,
    /// Total rows this step may carry (`lanes × verify_rows_per_lane` when
    /// capacity is sized per lane).
    pub max_rows: usize,
    /// Copy proposals' depth cap (`min(n_max, row-exact rows − 1)`).
    pub copy_take_cap: usize,
    pub lab: LabRoutes,
    /// PP: lanes per pipeline microbatch (`None`: one whole-step batch).
    pub pp_microbatch_lanes: Option<usize>,
}

/// The stage policy every rank must agree on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePolicy {
    pub moe_layers: usize,
    /// EP `moe=once` (one exchange per layer per row group) vs per-lane.
    pub moe_once: bool,
    pub stream_bf16: bool,
}

/// Immutable root configuration.
#[derive(Clone, Copy, Debug)]
pub struct RootConfig {
    pub topology_generation: u64,
    pub stage: StagePolicy,
    pub floor_enabled: bool,
}

/// One lane of a [`WindowStart`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanePlan {
    pub key: LaneWindowKey,
    pub kind: LaneKind,
    /// Absolute position of the seed (the window's first row).
    pub position: usize,
    pub seed: u32,
    /// The native chooser's route (floor accounting); `Ar` on non-spec and
    /// forced lanes. A copy window keeps the native route it overrides.
    pub route: LaneRoute,
    pub source: ProposalSource,
    /// Proposals this window verifies at most (native `k_max`, or the copy
    /// candidates kept; 0 for AR/forced).
    pub max_depth: usize,
    /// Copy candidates (only `ProposalSource::Copy`).
    pub copy_drafts: Vec<u32>,
    pub sampling: SamplingMode,
    pub eos: Option<u32>,
    pub remaining: usize,
    /// The drafts budget the floor observes.
    pub budget: usize,
    pub head: HeadState,
}

impl LanePlan {
    /// Rows this lane's verify block carries at most (exact once drafting
    /// ends).
    pub fn max_rows(&self) -> usize {
        match &self.kind {
            LaneKind::Forced(t) => t.len(),
            LaneKind::Decode => 1 + self.max_depth,
        }
    }
}

/// The root's immutable plan of one step (`plan_window` output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowStart {
    pub key: MeshStepKey,
    pub lanes: Vec<LanePlan>,
    pub budgets: WindowBudgets,
    pub stage: StagePolicy,
    pub digest: u64,
}

fn lane_plan_digest(d: &mut Digest, l: &LanePlan) {
    d.lane_key(&l.key).usize(l.position).u32(l.seed);
    match l.route {
        LaneRoute::Ar => d.usize(0),
        LaneRoute::ArSynced => d.usize(1),
        LaneRoute::ArRetired => d.usize(2),
        LaneRoute::Batched { k_max } => d.usize(3).usize(k_max),
    };
    d.usize(l.source as usize)
        .usize(l.max_depth)
        .slice(&l.copy_drafts)
        .usize(l.sampling as usize)
        .usize(l.remaining)
        .usize(l.budget);
    match &l.kind {
        LaneKind::Decode => d.usize(0),
        LaneKind::Forced(t) => d.usize(1).slice(t),
    };
    d.u64(l.eos.map_or(u64::MAX, u64::from));
}

fn start_digest(key: &MeshStepKey, lanes: &[LanePlan], b: &WindowBudgets, s: &StagePolicy) -> u64 {
    let mut d = Digest::new();
    d.u64(key.topology_generation)
        .u64(key.step_seq)
        .u64(key.policy_digest)
        .usize(b.max_lanes)
        .usize(b.verify_rows_per_lane)
        .usize(b.max_rows)
        .usize(s.moe_layers);
    for l in lanes {
        lane_plan_digest(&mut d, l);
    }
    d.finish()
}

/// A rank's answer to `begin_window`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankReady {
    pub key: MeshStepKey,
    pub rank: usize,
    /// The digest of the plan the rank validated.
    pub start_digest: u64,
    pub status: ReadyStatus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadyStatus {
    /// Capacity mapped; tickets reserved where owned.
    Ready {
        tickets: Vec<LaneTickets>,
    },
    Rejected {
        reason: String,
    },
}

/// Opaque rank-local ticket identities reserved at `begin_window`. A
/// prepared receipt must echo them lane by lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TicketId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneTickets {
    pub key: LaneWindowKey,
    pub target: Option<TicketId>,
    pub head: Option<TicketId>,
}

// ---------------------------------------------------------------------------
// Draft rounds
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftOpKind {
    /// Head Draft: runs the head FFN/expert gather and predicts.
    Draft,
}

/// One ordered head call: all EP ranks execute the whole list; a PP output
/// band alone does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftOp {
    pub key: LaneWindowKey,
    pub position: usize,
    pub input: u32,
    pub kind: DraftOpKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftRound {
    pub key: MeshStepKey,
    pub round: u32,
    pub ops: Vec<DraftOp>,
}

/// What the sampler owner read for one op: the head's argmax and keep-going
/// hint, and (Leviathan lanes) the exact candidate logits `q` is built from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftProposal {
    pub key: LaneWindowKey,
    /// The head's argmax token. Greedy/naive lanes propose exactly this; a
    /// Leviathan lane's token is drawn by the root from `read`
    /// ([`AcceptRule::draft_token`]) and this value is only a hint.
    pub token: u32,
    /// The head's own hint that deeper drafting is worthwhile; the root
    /// combines it with depth/EOS and may still stop.
    pub keep_going: bool,
    /// The singleton rule: a token after the first whose prefix acceptance
    /// fell below the row worth was drafted (the head step ran and counts in
    /// the ledger) but is NOT kept; drafting ends. Never set on the first
    /// draft of a lane.
    pub drop_last: bool,
    /// The head's exact rescored candidate logits (or the full draft row),
    /// copied out before another lane's Draft overwrites the shared draft
    /// buffers. `Some` only for a Leviathan lane (S4); the root builds `q`
    /// from it under the lane's sampler. A dropped proposal still draws
    /// (the singleton draws before it prunes).
    pub read: Option<DraftRead>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankDraftResult {
    pub key: MeshStepKey,
    pub round: u32,
    pub rank: usize,
    pub ops_executed: usize,
    /// Head gathers this rank crossed in the round.
    pub head_gathers: usize,
    /// `Some` only from the sampler owner.
    pub proposals: Option<Vec<DraftProposal>>,
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/// One lane's sealed verify block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyLane {
    pub key: LaneWindowKey,
    pub position: usize,
    /// `[seed, drafts..]` (forced: `[seed, f1..f_{n-1}]`).
    pub rows: Vec<u32>,
    /// Arm the per-lane capture ring (a window with more than one row).
    pub capture: bool,
}

/// One packed once-over-all-rows group: consecutive whole lanes, at most
/// [`LANE_MAX_ROWS`] rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowGroup {
    pub first_lane: usize,
    pub lanes: usize,
    pub rows: usize,
}

/// The expected per-rank operation ledger of a verify (counts only; no
/// timing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpLedger {
    /// Row groups, summed over microbatches.
    pub row_groups: usize,
    /// EP target row-gather exchanges (`layers × groups`, or per lane when
    /// the stage policy is per-lane MoE); zero on PP.
    pub target_exchanges: usize,
    /// PP pipeline microbatches (0 on EP).
    pub microbatches: usize,
    /// Head Draft invocations over the step's draft rounds.
    pub head_draft_calls: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyPlan {
    pub key: MeshStepKey,
    pub lanes: Vec<VerifyLane>,
    /// Row groups, packed per microbatch.
    pub groups: Vec<RowGroup>,
    /// Lanes per PP microbatch boundary (`[lanes]` for EP / unsplit).
    pub microbatches: Vec<usize>,
    pub ledger: OpLedger,
    pub digest: u64,
}

impl VerifyPlan {
    pub fn total_rows(&self) -> usize {
        self.lanes.iter().map(|l| l.rows.len()).sum()
    }
}

/// Consecutive whole lanes packed into groups of at most `max_rows` rows
/// (a wider lane is its own group) — the same rule as
/// `gpu_forward_lanes::group_segments`.
pub fn pack_row_groups(rows: &[usize], max_rows: usize) -> Vec<RowGroup> {
    let mut groups: Vec<RowGroup> = Vec::new();
    for (lane, &r) in rows.iter().enumerate() {
        match groups.last_mut() {
            Some(g) if g.rows + r <= max_rows => {
                g.rows += r;
                g.lanes += 1;
            }
            _ => groups.push(RowGroup {
                first_lane: lane,
                lanes: 1,
                rows: r,
            }),
        }
    }
    groups
}

/// The sampler owner's target output for the sealed rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplerOutput {
    /// Per lane (plan order): the target's pick at every row.
    pub picks: Vec<(LaneWindowKey, Vec<u32>)>,
    /// Lazy access to the sampler owner's resident verify logits, for the
    /// root's sampled acceptance (S4). `None` on greedy-only steps.
    pub rows: Option<RowHandle>,
}

impl SamplerOutput {
    /// Greedy picks only (no distribution rows).
    pub fn picks_only(picks: Vec<(LaneWindowKey, Vec<u32>)>) -> Self {
        Self { picks, rows: None }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankVerifyResult {
    pub key: MeshStepKey,
    pub rank: usize,
    pub plan_digest: u64,
    /// Transient target end position per lane (`position + rows`).
    pub lane_ends: Vec<(LaneWindowKey, usize)>,
    pub ledger: OpLedger,
    /// `Some` only from the sampler owner.
    pub sampler: Option<SamplerOutput>,
}

// ---------------------------------------------------------------------------
// Decision
// ---------------------------------------------------------------------------

/// Proposal accounting of a window (source-attributed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceAccounting {
    pub source: ProposalSource,
    pub offered: usize,
    pub accepted: usize,
}

/// The root's decision for one lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneDecision {
    pub key: LaneWindowKey,
    /// Accepted drafts then the bonus/correction (forced: the forced
    /// tokens). Never includes the seed.
    pub emitted: Vec<u32>,
    pub accepted: usize,
    pub hit_eos: bool,
    pub forced: bool,
    /// Rows of the verify block the target keeps:
    /// `1 + target_commit_accept_len`.
    pub consumed_rows: usize,
    /// The next pending seed: the last emitted token (not yet consumed).
    pub next_seed: u32,
    /// EOS or length termination; `next_seed` must not be consumed.
    pub terminal: bool,
    /// `position + consumed_rows`.
    pub target_end: usize,
    /// The row whose target hidden is the new pending hidden.
    pub pending_hidden_row: usize,
    pub head_end: HeadEnd,
    pub source: SourceAccounting,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitDecision {
    pub key: MeshStepKey,
    pub lanes: Vec<LaneDecision>,
    pub digest: u64,
}

fn decision_digest(key: &MeshStepKey, lanes: &[LaneDecision]) -> u64 {
    let mut d = Digest::new();
    d.u64(key.topology_generation).u64(key.step_seq);
    for l in lanes {
        d.lane_key(&l.key)
            .slice(&l.emitted)
            .usize(l.accepted)
            .bool(l.hit_eos)
            .bool(l.forced)
            .usize(l.consumed_rows)
            .u32(l.next_seed)
            .bool(l.terminal)
            .usize(l.target_end)
            .usize(l.pending_hidden_row);
        match l.head_end {
            HeadEnd::Aligned { end } => d.usize(0).usize(end),
            HeadEnd::Lag { from, to } => d.usize(1).usize(from).usize(to),
            HeadEnd::Free => d.usize(2),
        };
    }
    d.finish()
}

/// The root-only acceptance rule (greedy, naive sampled, Leviathan). It
/// receives the lane's *staged* RNG; nothing else may draw.
pub trait AcceptRule {
    /// `drafts` are the verified proposals; `picks` the target's pick per
    /// verify row (`drafts.len() + 1` long). Returns the committed tail:
    /// accepted drafts then the bonus, unless an accepted draft was EOS.
    /// `Err` (a row could not be read, a distribution is missing) aborts the
    /// step; the staged RNG is then discarded.
    fn accept(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        picks: &[u32],
        rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String>;

    /// The token of native proposal `index` of `lane` (`prior_drafts` are
    /// the `index` tokens already kept). Called by the root for every head
    /// proposal with the lane's staged RNG. The default keeps the head's
    /// argmax; a Leviathan rule draws from `q` here.
    fn draft_token(
        &mut self,
        _lane: &LanePlan,
        _index: usize,
        _prior_drafts: &[u32],
        proposal: &DraftProposal,
        _rng: &mut LaneRng,
    ) -> Result<u32, String> {
        Ok(proposal.token)
    }

    /// The sampler owner's verify output, before any `accept` of the step.
    fn observe_verify(&mut self, _out: &SamplerOutput) {}

    /// The step published: fold its emitted tokens into rule-side history.
    fn on_publish(&mut self, _out: &StepOutput) {}
}

/// The shared greedy accept-prefix rule; never draws.
pub struct GreedyRule;

impl AcceptRule for GreedyRule {
    fn accept(
        &mut self,
        lane: &LanePlan,
        drafts: &[u32],
        picks: &[u32],
        _rng: &mut LaneRng,
    ) -> Result<GreedyAccept, String> {
        Ok(accept_greedy_prefix(drafts, picks, lane.eos))
    }
}

/// A rank's proof it repaired/truncated its owned state to the decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedLane {
    pub key: LaneWindowKey,
    /// `Some` on target owners.
    pub target_end: Option<usize>,
    /// `Some` on head owners.
    pub head_end: Option<HeadEnd>,
    pub tickets: LaneTickets,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedReceipt {
    pub key: MeshStepKey,
    pub rank: usize,
    pub decision_digest: u64,
    pub lanes: Vec<PreparedLane>,
    /// Head append rows run during repair (teacher-forced; no FFN gather).
    pub head_append_rows: usize,
}

/// A lane's published result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneOutput {
    pub key: LaneWindowKey,
    /// Tokens to stream (empty for a cancelled lane).
    pub emitted: Vec<u32>,
    pub forced: bool,
    pub terminal: bool,
    pub cancelled: bool,
    pub consumed_rows: usize,
    pub next_seed: u32,
    pub source: SourceAccounting,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepOutput {
    pub key: MeshStepKey,
    pub lanes: Vec<LaneOutput>,
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimedPhase {
    Draft,
    Verify,
    Prepare,
    /// Host coordination / transport wait, counted once (root).
    Coordination,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseTiming {
    pub phase: TimedPhase,
    pub us: f32,
    /// A synchronized (device-event) measurement; otherwise diagnostic.
    pub synchronized: bool,
}

/// One `(band, microbatch)` node of a PP wavefront, on the shared host
/// monotonic clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpan {
    pub band: usize,
    pub microbatch: usize,
    pub start_us: f64,
    pub end_us: f64,
}

/// Counts every rank must report (mandatory evidence).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpCounts {
    pub target_exchanges: usize,
    pub gather_bytes: u64,
    pub head_draft_calls: usize,
    pub head_append_rows: usize,
    pub row_groups: usize,
    pub edge_count: usize,
    pub edge_bytes: u64,
    pub drafts_generated: usize,
    pub accepted: usize,
    pub consumed_rows: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RankCostEvidence {
    pub key: MeshStepKey,
    pub rank: usize,
    /// The last phase the rank completed successfully.
    pub completed: StepPhase,
    pub phases: Vec<PhaseTiming>,
    /// PP wavefront nodes of the verify (empty on EP).
    pub nodes: Vec<NodeSpan>,
    pub counts: OpCounts,
    /// A warm kernel bind or first-seen depth: stays unpriced.
    pub unpriced: bool,
}

/// Previous-window-start → next-plan interval measured by the controller:
/// the primary realized wall price.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControllerInterval {
    pub wall_us: f64,
    /// A prefill tile ran in the interval (the wall is not a decode price).
    pub prefill_ran: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MergedCostEvidence {
    pub key: MeshStepKey,
    pub ranks: usize,
    pub counts: OpCounts,
    pub wall_us: f64,
    /// Diagnostic critical path of the device phases (EP: Σ phase
    /// max-over-ranks; PP: wavefront interval + non-pipeline phases).
    pub critical_path_us: f64,
    /// Every contributing phase timing was synchronized.
    pub synchronized: bool,
    /// Whether the controller wall may train the fit.
    pub priced: bool,
    /// Whether the floor may price this step's windows at the cost model's
    /// units. `false` (NaN units: probe budget spent unpriced) only when a
    /// rank did not complete or a warm bind ran; a prefill tile in the
    /// interval leaves it `true` while `priced` is `false`.
    pub units_valid: bool,
    pub unpriced_reason: Option<&'static str>,
}

/// What a published step promised; evidence is merged against it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceExpectation {
    pub key: MeshStepKey,
    pub world: Vec<RankOwnership>,
    pub ledger: OpLedger,
    pub head_append_rows: usize,
    pub drafts_generated: usize,
    pub accepted: usize,
    pub consumed_rows: usize,
    /// PP overlap (wavefront) vs the sequential kill-switch arm.
    pub overlap: bool,
    pub decode_lanes: usize,
}

/// Merge every rank's receipts with the controller interval.
///
/// Rejects stale keys, missing/duplicate/unknown ranks and incoherent
/// counts. A phase an incomplete rank never finished, a warm bind, a
/// prefill in the interval or an unusable wall only make the result
/// unpriced.
pub fn merge_evidence(
    expect: &EvidenceExpectation,
    receipts: &[RankCostEvidence],
    interval: ControllerInterval,
) -> Result<MergedCostEvidence, EvidenceError> {
    let mut by_rank: HashMap<usize, &RankCostEvidence> = HashMap::new();
    for r in receipts {
        if r.key != expect.key {
            return Err(EvidenceError::Stale(format!(
                "rank {} receipt is for step {} (gen {}), expected step {} (gen {})",
                r.rank,
                r.key.step_seq,
                r.key.topology_generation,
                expect.key.step_seq,
                expect.key.topology_generation
            )));
        }
        if !expect.world.iter().any(|w| w.rank == r.rank) {
            return Err(EvidenceError::Missing(format!(
                "receipt from unknown rank {}",
                r.rank
            )));
        }
        if by_rank.insert(r.rank, r).is_some() {
            return Err(EvidenceError::Incoherent(format!(
                "duplicate receipt from rank {}",
                r.rank
            )));
        }
    }
    for w in &expect.world {
        if !by_rank.contains_key(&w.rank) {
            return Err(EvidenceError::Missing(format!(
                "no receipt from rank {}",
                w.rank
            )));
        }
    }
    let kind = expect.world[0].kind;
    let bands = expect.world.len();
    let incoherent = |m: String| Err(EvidenceError::Incoherent(m));
    let mut edge = None::<(usize, u64)>;
    for w in &expect.world {
        let c = &by_rank[&w.rank].counts;
        let l = &expect.ledger;
        if c.row_groups != l.row_groups {
            return incoherent(format!(
                "rank {} row groups {} != expected {}",
                w.rank, c.row_groups, l.row_groups
            ));
        }
        if c.target_exchanges != l.target_exchanges {
            return incoherent(format!(
                "rank {} target exchanges {} != expected {}",
                w.rank, c.target_exchanges, l.target_exchanges
            ));
        }
        let want_head = if w.head { l.head_draft_calls } else { 0 };
        if c.head_draft_calls != want_head {
            return incoherent(format!(
                "rank {} head draft calls {} != expected {}",
                w.rank, c.head_draft_calls, want_head
            ));
        }
        let want_append = if w.head { expect.head_append_rows } else { 0 };
        if c.head_append_rows != want_append {
            return incoherent(format!(
                "rank {} head append rows {} != expected {}",
                w.rank, c.head_append_rows, want_append
            ));
        }
        if (c.drafts_generated, c.accepted, c.consumed_rows)
            != (
                expect.drafts_generated,
                expect.accepted,
                expect.consumed_rows,
            )
        {
            return incoherent(format!(
                "rank {} drafts/accepted/consumed {:?} != decision {:?}",
                w.rank,
                (c.drafts_generated, c.accepted, c.consumed_rows),
                (
                    expect.drafts_generated,
                    expect.accepted,
                    expect.consumed_rows
                )
            ));
        }
        match kind {
            MeshKind::Ep | MeshKind::Single => {
                if c.edge_count != 0 || c.edge_bytes != 0 {
                    return incoherent(format!("EP rank {} reports pipeline edges", w.rank));
                }
            }
            MeshKind::Pp => {
                if c.edge_count != l.microbatches * (bands - 1) {
                    return incoherent(format!(
                        "PP rank {} edge count {} != {} microbatches × {} boundaries",
                        w.rank,
                        c.edge_count,
                        l.microbatches,
                        bands - 1
                    ));
                }
                match edge {
                    None => edge = Some((c.edge_count, c.edge_bytes)),
                    Some(e) if e != (c.edge_count, c.edge_bytes) => {
                        return incoherent(format!("PP rank {} edge payload disagrees", w.rank))
                    }
                    _ => {}
                }
            }
        }
    }
    let counts = by_rank[&expect.world[0].rank].counts;
    let mut counts = OpCounts {
        // Head work is counted once at the head owners (they agree).
        head_draft_calls: expect.ledger.head_draft_calls,
        head_append_rows: expect.head_append_rows,
        ..counts
    };
    counts.gather_bytes = by_rank
        .values()
        .map(|r| r.counts.gather_bytes)
        .max()
        .unwrap_or(0);

    // Diagnostic critical path.
    let mut synchronized = true;
    let mut critical = 0.0f64;
    let mut phase_max: HashMap<TimedPhase, f64> = HashMap::new();
    let pipeline_nodes: Vec<&NodeSpan> = by_rank.values().flat_map(|r| r.nodes.iter()).collect();
    let use_nodes = kind == MeshKind::Pp && !pipeline_nodes.is_empty();
    for r in by_rank.values() {
        for p in &r.phases {
            if use_nodes && p.phase == TimedPhase::Verify {
                continue;
            }
            synchronized &= p.synchronized;
            let e = phase_max.entry(p.phase).or_insert(0.0);
            *e = e.max(f64::from(p.us));
        }
    }
    critical += phase_max.values().sum::<f64>();
    if use_nodes {
        let span = if expect.overlap {
            let lo = pipeline_nodes
                .iter()
                .map(|n| n.start_us)
                .fold(f64::INFINITY, f64::min);
            let hi = pipeline_nodes
                .iter()
                .map(|n| n.end_us)
                .fold(f64::NEG_INFINITY, f64::max);
            (hi - lo).max(0.0)
        } else {
            pipeline_nodes
                .iter()
                .map(|n| (n.end_us - n.start_us).max(0.0))
                .sum()
        };
        critical += span;
    }

    let mut reason = None;
    if by_rank.values().any(|r| r.completed < StepPhase::Prepared) {
        reason = Some("a rank did not complete the step");
    } else if by_rank.values().any(|r| r.unpriced) {
        reason = Some("warm bind or first-seen depth");
    } else if interval.prefill_ran {
        reason = Some("prefill ran in the interval");
    } else if !(interval.wall_us.is_finite() && interval.wall_us > 0.0) {
        reason = Some("unusable controller interval");
    }
    Ok(MergedCostEvidence {
        key: expect.key.clone(),
        ranks: bands,
        counts,
        wall_us: interval.wall_us,
        critical_path_us: critical,
        synchronized,
        priced: reason.is_none(),
        // The floor's model units stay valid unless a rank is incomplete or
        // a warm bind polluted the step: a prefill tile or an unusable wall
        // only keeps the wall out of the fit (the singleton lane rule).
        units_valid: !matches!(
            reason,
            Some("a rank did not complete the step" | "warm bind or first-seen depth")
        ),
        unpriced_reason: reason,
    })
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

/// Epoch-scoped authoritative token history the copy chooser reads
/// (`prompt ++ retained output`; candidates and rejected tails never enter).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyHistory {
    epoch: RequestEpoch,
    tokens: Vec<u32>,
}

impl CopyHistory {
    pub fn epoch(&self) -> RequestEpoch {
        self.epoch
    }
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LaneStatus {
    Live,
    Terminal,
}

#[derive(Clone)]
struct RootLane {
    epoch: RequestEpoch,
    lane_generation: u64,
    window_seq: u64,
    position: usize,
    seed: u32,
    eos: Option<u32>,
    rng: LaneRng,
    policy: Option<LaneSpecPolicy>,
    generated: Vec<u32>,
    copy: CopyHistory,
    head: HeadState,
    status: LaneStatus,
}

/// What a finished window leaves for the next step's price.
#[derive(Clone)]
struct PendingLaneObs {
    epoch: RequestEpoch,
    route: LaneRoute,
    source: ProposalSource,
    window: MtpWindow,
    budget: usize,
}

#[derive(Clone)]
struct PendingObservation {
    key: MeshStepKey,
    decode_lanes: usize,
    spill: usize,
    verified: usize,
    verify_windows: usize,
    lanes: Vec<PendingLaneObs>,
    expect: EvidenceExpectation,
}

#[derive(Clone)]
struct StagedLane {
    epoch: RequestEpoch,
    rng: LaneRng,
    policy: Option<LaneSpecPolicy>,
    window: Option<WindowUpdate>,
}

#[derive(Clone)]
struct WindowUpdate {
    emitted: Vec<u32>,
    position: usize,
    seed: u32,
    terminal: bool,
}

/// Mutable per-draft state of one lane inside a step.
#[derive(Clone)]
struct DraftLane {
    drafts: Vec<u32>,
    active: bool,
}

struct StepState {
    start: WindowStart,
    phase: StepPhase,
    staged_cost: LaneCostModel,
    staged: Vec<StagedLane>,
    obs_applied: bool,
    obs_dropped: bool,
    tickets: HashMap<usize, Vec<LaneTickets>>,
    draft: Vec<DraftLane>,
    round: u32,
    head_calls: usize,
    verify: Option<VerifyPlan>,
    picks: Option<SamplerOutput>,
    decision: Option<CommitDecision>,
    cancelled: HashSet<RequestEpoch>,
    pending_obs: Option<PendingObservation>,
}

/// Root counters (diagnostic).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RootCounters {
    pub steps_published: u64,
    pub observations_applied: u64,
    pub observations_dropped: u64,
}

/// The root coordinator: the only owner of policy, RNG, acceptance and
/// authoritative generated IDs.
pub struct MeshRoot {
    own: RankOwnership,
    world: Vec<RankOwnership>,
    cfg: RootConfig,
    lanes: HashMap<RequestEpoch, RootLane>,
    next_lane_generation: u64,
    step_seq: u64,
    cost: LaneCostModel,
    pending_obs: Option<PendingObservation>,
    step: Option<StepState>,
    phase: StepPhase,
    aborted: Option<AbortRecord>,
    counters: RootCounters,
    /// Single-card legacy: the driver owns request termination (see
    /// [`MeshRoot::set_driver_terminates`]).
    driver_terminates: bool,
}

impl MeshRoot {
    /// A root for the mesh `world`; `own` must be the world's root rank.
    /// A follower (or a PP non-output band) cannot construct one.
    pub fn new(
        own: RankOwnership,
        world: &[RankOwnership],
        cfg: RootConfig,
    ) -> Result<Self, ProtocolError> {
        let root = validate_world(world)?;
        if !own.root || own != root {
            return Err(ProtocolError::Config(format!(
                "rank {} is not the root: only the root chooses, draws and accepts",
                own.rank
            )));
        }
        Ok(Self {
            own,
            world: world.to_vec(),
            cfg,
            lanes: HashMap::new(),
            next_lane_generation: 1,
            step_seq: 0,
            cost: LaneCostModel::new(),
            pending_obs: None,
            step: None,
            phase: StepPhase::Idle,
            aborted: None,
            counters: RootCounters::default(),
            driver_terminates: false,
        })
    }

    /// The root rank's ownership (the only rank that chooses, draws, accepts).
    pub fn ownership(&self) -> RankOwnership {
        self.own
    }

    pub fn phase(&self) -> StepPhase {
        self.phase
    }

    pub fn world(&self) -> &[RankOwnership] {
        &self.world
    }

    pub fn counters(&self) -> RootCounters {
        self.counters
    }

    /// The abort record when the mesh is quarantined.
    pub fn quarantined(&self) -> Option<&AbortRecord> {
        self.aborted.as_ref()
    }

    /// The cost model's `(F, a, r, G)` fit, for tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn cost_fit(&self) -> (f64, f64, f64, f64) {
        self.cost.fit()
    }

    fn check_live(&self) -> Result<(), ProtocolError> {
        match &self.aborted {
            Some(r) => Err(ProtocolError::Aborted(r.clone())),
            None => Ok(()),
        }
    }

    fn check_phase(&self, op: &'static str, want: StepPhase) -> Result<(), ProtocolError> {
        self.check_live()?;
        if self.phase != want {
            return Err(ProtocolError::Phase {
                op,
                actual: self.phase,
            });
        }
        Ok(())
    }

    fn fail(&mut self, rank: Option<usize>, reason: String) -> ProtocolError {
        let key = self.step.as_ref().map(|s| s.start.key.clone());
        let phase = self.phase;
        ProtocolError::Aborted(self.record_abort(key, phase, rank, reason))
    }

    fn record_abort(
        &mut self,
        key: Option<MeshStepKey>,
        phase: StepPhase,
        rank: Option<usize>,
        reason: String,
    ) -> AbortRecord {
        if let Some(r) = &self.aborted {
            return r.clone();
        }
        let record = AbortRecord {
            key,
            phase,
            rank,
            reason,
        };
        // Staged state is dropped: nothing of this step was published.
        self.step = None;
        self.pending_obs = None;
        self.phase = StepPhase::Aborted;
        self.aborted = Some(record.clone());
        record
    }

    /// `abort_step(key, phase, reason)`: idempotent terminal transition.
    /// The first record wins; later calls (any phase/reason) return it. A
    /// key that is neither the in-flight step's nor absent is stale and
    /// changes nothing.
    pub fn abort_step(
        &mut self,
        key: Option<&MeshStepKey>,
        phase: StepPhase,
        reason: &str,
    ) -> Result<AbortRecord, ProtocolError> {
        if let Some(r) = &self.aborted {
            return Ok(r.clone());
        }
        let in_flight = self.step.as_ref().map(|s| &s.start.key);
        if let (Some(k), Some(f)) = (key, in_flight) {
            if k != f {
                return Err(ProtocolError::StaleEvidence(format!(
                    "abort for step {} while step {} is in flight",
                    k.step_seq, f.step_seq
                )));
            }
        } else if key.is_some() && in_flight.is_none() {
            return Err(ProtocolError::StaleEvidence(
                "abort for a step that is not in flight".to_string(),
            ));
        }
        Ok(self.record_abort(key.cloned(), phase, None, reason.to_string()))
    }

    // ---- lifecycle ------------------------------------------------------

    /// Admit a request at Idle/Published. Returns what ranks mirror.
    pub fn admit_lane(&mut self, a: LaneAdmission) -> Result<LaneAdmit, ProtocolError> {
        self.admit_with(a, None)
    }

    fn admit_with(
        &mut self,
        a: LaneAdmission,
        policy_seed: Option<LaneSpecPolicy>,
    ) -> Result<LaneAdmit, ProtocolError> {
        self.check_live()?;
        if !matches!(self.phase, StepPhase::Idle | StepPhase::Published) {
            return Err(ProtocolError::Phase {
                op: "admit_lane",
                actual: self.phase,
            });
        }
        if !a.epoch.is_admitted() {
            return Err(ProtocolError::StaleEpoch(a.epoch));
        }
        if self.lanes.contains_key(&a.epoch) {
            return Err(ProtocolError::Config(format!(
                "epoch {:?} already admitted",
                a.epoch
            )));
        }
        // An older generation of the same tag is stale, never reused.
        if self
            .lanes
            .keys()
            .any(|e| e.request_tag == a.epoch.request_tag)
        {
            return Err(ProtocolError::Config(format!(
                "request {} still holds a live lane of another generation",
                a.epoch.request_tag
            )));
        }
        let generation = self.next_lane_generation;
        self.next_lane_generation += 1;
        let policy = policy_seed.or_else(|| {
            a.spec_k
                .map(|k| LaneSpecPolicy::with_floor(k, self.cfg.floor_enabled))
        });
        self.lanes.insert(
            a.epoch,
            RootLane {
                epoch: a.epoch,
                lane_generation: generation,
                window_seq: 0,
                position: a.position,
                seed: a.seed,
                eos: a.eos,
                rng: LaneRng::new(a.rng_seed),
                policy,
                generated: Vec::new(),
                copy: CopyHistory {
                    epoch: a.epoch,
                    tokens: a.prompt,
                },
                head: if a.spec_k.is_some() {
                    a.head
                } else {
                    HeadState::Free
                },
                status: LaneStatus::Live,
            },
        );
        Ok(LaneAdmit {
            epoch: a.epoch,
            lane_generation: generation,
            position: a.position,
            spec: a.spec_k.is_some(),
        })
    }

    /// Drop a lane (terminal, or cancelled while idle). Refused while a
    /// step is in flight over it.
    pub fn release_lane(&mut self, epoch: RequestEpoch) -> Result<(), ProtocolError> {
        self.check_live()?;
        if self.in_flight(epoch) {
            return Err(ProtocolError::Phase {
                op: "release_lane (in-flight step)",
                actual: self.phase,
            });
        }
        self.lanes
            .remove(&epoch)
            .map(|_| ())
            .ok_or(ProtocolError::UnknownLane(epoch))
    }

    fn in_flight(&self, epoch: RequestEpoch) -> bool {
        matches!(&self.step, Some(s) if s.start.lanes.iter().any(|l| l.key.request_epoch == epoch))
    }

    /// Cancel a request at the coordinated boundary. Between steps the lane
    /// is released at once; during a step it stays in the schedule (the
    /// ordered collectives are unchanged) and its output is withheld, its
    /// RNG/policy discarded, at publish. Immediate interruption is
    /// [`Self::abort_step`], not this.
    pub fn cancel_lane(&mut self, epoch: RequestEpoch) -> Result<CancelOutcome, ProtocolError> {
        self.check_live()?;
        if !self.lanes.contains_key(&epoch) {
            return Err(ProtocolError::UnknownLane(epoch));
        }
        if self.in_flight(epoch) {
            let phase = self.phase;
            if let Some(s) = self.step.as_mut() {
                s.cancelled.insert(epoch);
            }
            Ok(CancelOutcome::Deferred { phase })
        } else {
            self.lanes.remove(&epoch);
            Ok(CancelOutcome::Released)
        }
    }

    // ---- diagnostics ----------------------------------------------------

    /// The request's next RNG draw without advancing it (diagnostic).
    pub fn rng_probe(&self, epoch: RequestEpoch) -> Option<u64> {
        self.lanes.get(&epoch).map(|l| {
            let mut r = l.rng;
            r.next_u64()
        })
    }

    /// Both of the request's RNG streams as `(next Leviathan draw, naive
    /// xorshift32 state)`, without advancing them (diagnostic).
    pub fn rng_fingerprint(&self, epoch: RequestEpoch) -> Option<(u64, u32)> {
        self.lanes.get(&epoch).map(|l| l.rng.fingerprint())
    }

    pub fn generated(&self, epoch: RequestEpoch) -> Option<&[u32]> {
        self.lanes.get(&epoch).map(|l| l.generated.as_slice())
    }

    pub fn copy_history(&self, epoch: RequestEpoch) -> Option<&CopyHistory> {
        self.lanes.get(&epoch).map(|l| &l.copy)
    }

    pub fn lane_position(&self, epoch: RequestEpoch) -> Option<(usize, u32)> {
        self.lanes.get(&epoch).map(|l| (l.position, l.seed))
    }

    pub fn lane_retired(&self, epoch: RequestEpoch) -> Option<bool> {
        self.lanes
            .get(&epoch)
            .map(|l| l.policy.as_ref().is_some_and(LaneSpecPolicy::retired))
    }

    // ---- 1. plan_window -------------------------------------------------

    /// Root-only, pure host policy transition. `prior_evidence` is the
    /// merged evidence of the previous published step (or `None`: that
    /// step's price is dropped, spending probe budget unpriced). Stale
    /// evidence is refused and changes nothing. No GPU operation.
    pub fn plan_window(
        &mut self,
        prior_evidence: Option<&MergedCostEvidence>,
        requests: &[LaneRequest],
        budgets: &WindowBudgets,
    ) -> Result<WindowStart, ProtocolError> {
        self.check_live()?;
        if !matches!(self.phase, StepPhase::Idle | StepPhase::Published) {
            return Err(ProtocolError::Phase {
                op: "plan_window",
                actual: self.phase,
            });
        }
        if budgets.verify_rows_per_lane == 0 || budgets.verify_rows_per_lane > LANE_MAX_ROWS {
            return Err(ProtocolError::Budget(format!(
                "verify_rows_per_lane {} outside 1..={LANE_MAX_ROWS} (row-exact width)",
                budgets.verify_rows_per_lane
            )));
        }
        if requests.is_empty() || requests.len() > budgets.max_lanes {
            return Err(ProtocolError::Budget(format!(
                "{} lanes outside 1..={}",
                requests.len(),
                budgets.max_lanes
            )));
        }
        // Evidence validity first: a stale object changes nothing.
        match (&self.pending_obs, prior_evidence) {
            (Some(p), Some(m)) if m.key != p.key => {
                return Err(ProtocolError::StaleEvidence(format!(
                    "evidence is for step {} (gen {}), pending is step {} (gen {})",
                    m.key.step_seq,
                    m.key.topology_generation,
                    p.key.step_seq,
                    p.key.topology_generation
                )))
            }
            (None, Some(m)) => {
                return Err(ProtocolError::StaleEvidence(format!(
                    "evidence for step {} but no step awaits a price",
                    m.key.step_seq
                )))
            }
            _ => {}
        }

        // Resolve lanes in request order; duplicates and stale epochs fail.
        let mut seen = HashSet::new();
        let mut resolved: Vec<&RootLane> = Vec::with_capacity(requests.len());
        for r in requests {
            if !seen.insert(r.epoch) {
                return Err(ProtocolError::Config(format!(
                    "epoch {:?} planned twice in one step",
                    r.epoch
                )));
            }
            let lane = match self.lanes.get(&r.epoch) {
                Some(l) => l,
                None if self
                    .lanes
                    .keys()
                    .any(|e| e.request_tag == r.epoch.request_tag) =>
                {
                    return Err(ProtocolError::StaleEpoch(r.epoch))
                }
                None => return Err(ProtocolError::UnknownLane(r.epoch)),
            };
            if lane.status != LaneStatus::Live {
                return Err(ProtocolError::LaneNotLive(r.epoch));
            }
            if r.remaining == 0 {
                return Err(ProtocolError::Budget(format!(
                    "{:?}: no tokens remaining",
                    r.epoch
                )));
            }
            resolved.push(lane);
        }

        // Stage: evidence first, then routes (all on clones).
        let mut staged_cost = self.cost.clone();
        let mut staged: HashMap<RequestEpoch, StagedLane> = HashMap::new();
        for l in &resolved {
            staged.insert(
                l.epoch,
                StagedLane {
                    epoch: l.epoch,
                    rng: l.rng,
                    policy: l.policy.clone(),
                    window: None,
                },
            );
        }
        let mut applied = false;
        let mut dropped = false;
        if let Some(p) = &self.pending_obs {
            match prior_evidence {
                Some(m) => {
                    applied = true;
                    apply_evidence(&mut staged_cost, &mut staged, &self.lanes, p, m);
                }
                None => dropped = true,
            }
        }

        let mut plans: Vec<LanePlan> = Vec::with_capacity(requests.len());
        let mut total_rows = 0usize;
        for (r, lane) in requests.iter().zip(&resolved) {
            let key = LaneWindowKey {
                request_epoch: lane.epoch,
                lane_generation: lane.lane_generation,
                window_seq: lane.window_seq,
            };
            let max_drafts = (budgets.verify_rows_per_lane - 1)
                .min(r.remaining.saturating_sub(1))
                .min(r.context_left.saturating_sub(1));
            let mut plan = LanePlan {
                key,
                kind: LaneKind::Decode,
                position: lane.position,
                seed: lane.seed,
                route: LaneRoute::Ar,
                source: ProposalSource::None,
                max_depth: 0,
                copy_drafts: Vec::new(),
                sampling: r.sampling,
                eos: lane.eos,
                remaining: r.remaining,
                budget: 0,
                head: lane.head,
            };
            if let Some(forced) = &r.forced {
                if forced.is_empty() || forced.len() > budgets.verify_rows_per_lane {
                    return Err(ProtocolError::Budget(format!(
                        "{:?}: {} forced tokens outside 1..={}",
                        lane.epoch,
                        forced.len(),
                        budgets.verify_rows_per_lane
                    )));
                }
                if forced.len() > r.context_left {
                    return Err(ProtocolError::Budget(format!(
                        "{:?}: forced rows exceed context",
                        lane.epoch
                    )));
                }
                plan.kind = LaneKind::Forced(forced.clone());
            } else if let Some(sl) = staged.get_mut(&lane.epoch) {
                if let Some(policy) = sl.policy.as_mut() {
                    let budget = max_drafts.min(policy.max_k());
                    plan.budget = budget;
                    let route = policy.route(max_drafts, budgets.lab);
                    plan.route = route;
                    match route {
                        LaneRoute::Batched { k_max } => {
                            plan.source = ProposalSource::Native;
                            plan.max_depth = k_max.min(max_drafts);
                        }
                        _ => {
                            if budget > 0 && !r.copy_candidates.is_empty() {
                                let cap = r
                                    .copy_candidates
                                    .len()
                                    .min(budgets.copy_take_cap)
                                    .min(max_drafts);
                                let mut take = cap;
                                if let Some(eos) = lane.eos {
                                    if let Some(i) =
                                        r.copy_candidates[..cap].iter().position(|&t| t == eos)
                                    {
                                        take = i + 1;
                                    }
                                }
                                if take > 0 {
                                    plan.source = ProposalSource::Copy;
                                    plan.max_depth = take;
                                    plan.copy_drafts = r.copy_candidates[..take].to_vec();
                                }
                            }
                        }
                    }
                    // A window of a retired lane never touches the head
                    // (copy is the only reason a retired lane keeps one).
                    if route == LaneRoute::ArRetired
                        && plan.source == ProposalSource::None
                        && plan.head == HeadState::Aligned
                    {
                        plan.head = HeadState::Free;
                    }
                }
            }
            total_rows += plan.max_rows();
            plans.push(plan);
        }
        if total_rows > budgets.max_rows {
            return Err(ProtocolError::Budget(format!(
                "{total_rows} rows exceed the admitted {}",
                budgets.max_rows
            )));
        }

        // Commit the plan.
        self.step_seq += 1;
        let policy_digest = {
            let mut d = Digest::new();
            d.u64(self.cfg.topology_generation)
                .usize(self.cfg.stage.moe_layers)
                .bool(self.cfg.stage.moe_once)
                .bool(self.cfg.stage.stream_bf16)
                .bool(self.cfg.floor_enabled)
                .usize(budgets.max_lanes)
                .usize(budgets.verify_rows_per_lane)
                .usize(budgets.copy_take_cap)
                .usize(budgets.lab as usize)
                .usize(budgets.pp_microbatch_lanes.unwrap_or(0))
                .usize(self.world.len());
            d.finish()
        };
        let key = MeshStepKey {
            topology_generation: self.cfg.topology_generation,
            step_seq: self.step_seq,
            lanes: plans.iter().map(|p| p.key).collect(),
            policy_digest,
        };
        let digest = start_digest(&key, &plans, budgets, &self.cfg.stage);
        let start = WindowStart {
            key,
            lanes: plans,
            budgets: *budgets,
            stage: self.cfg.stage,
            digest,
        };
        // Counters move at publish, with the rest of the staged state.
        let draft = start
            .lanes
            .iter()
            .map(|l| DraftLane {
                drafts: Vec::new(),
                active: l.source == ProposalSource::Native && l.max_depth > 0,
            })
            .collect();
        let staged_vec: Vec<StagedLane> = start
            .lanes
            .iter()
            .map(|l| staged.remove(&l.key.request_epoch).expect("staged lane"))
            .collect();
        self.step = Some(StepState {
            start: start.clone(),
            phase: StepPhase::Planned,
            staged_cost,
            staged: staged_vec,
            obs_applied: applied,
            obs_dropped: dropped,
            tickets: HashMap::new(),
            draft,
            round: 0,
            head_calls: 0,
            verify: None,
            picks: None,
            decision: None,
            cancelled: HashSet::new(),
            pending_obs: None,
        });
        self.phase = StepPhase::Planned;
        Ok(start)
    }

    fn step_ref(&self) -> &StepState {
        self.step.as_ref().expect("a phase past Idle holds a step")
    }

    fn step_mut(&mut self) -> &mut StepState {
        self.step.as_mut().expect("a phase past Idle holds a step")
    }

    // ---- 2. collect_ready ----------------------------------------------

    /// Aggregate every rank's `begin_window` answer before any device work.
    /// One rejection rejects the whole step: the root returns to Idle with
    /// nothing staged ([`ProtocolError::Preflight`]); the caller must tell
    /// the ready ranks (`reject_window`). A malformed answer (wrong key,
    /// digest, missing rank) aborts.
    pub fn collect_ready(&mut self, ready: &[RankReady]) -> Result<(), ProtocolError> {
        self.check_phase("collect_ready", StepPhase::Planned)?;
        let (key, digest) = {
            let s = self.step_ref();
            (s.start.key.clone(), s.start.digest)
        };
        let mut rejected = Vec::new();
        let mut by_rank: HashMap<usize, &RankReady> = HashMap::new();
        for r in ready {
            if r.key != key {
                return Err(self.fail(Some(r.rank), "begin_window answer for another step".into()));
            }
            if !self.world.iter().any(|w| w.rank == r.rank) {
                return Err(self.fail(
                    Some(r.rank),
                    "begin_window answer from an unknown rank".into(),
                ));
            }
            if by_rank.insert(r.rank, r).is_some() {
                return Err(self.fail(Some(r.rank), "duplicate begin_window answer".into()));
            }
            match &r.status {
                ReadyStatus::Rejected { reason } => rejected.push((r.rank, reason.clone())),
                ReadyStatus::Ready { .. } if r.start_digest != digest => {
                    return Err(self.fail(Some(r.rank), "rank validated a different plan".into()))
                }
                ReadyStatus::Ready { .. } => {}
            }
        }
        let missing: Vec<usize> = self
            .world
            .iter()
            .map(|w| w.rank)
            .filter(|rk| !by_rank.contains_key(rk))
            .collect();
        if !missing.is_empty() {
            return Err(self.fail(
                None,
                format!("no begin_window answer from ranks {missing:?}"),
            ));
        }
        if !rejected.is_empty() {
            // Preflight reject: no device work began; drop the staged plan.
            self.step = None;
            self.phase = StepPhase::Idle;
            return Err(ProtocolError::Preflight(rejected));
        }
        // Ticket coherence: lane keys echo the plan's; owners present.
        let lanes = self.step_ref().start.lanes.clone();
        let mut tickets = HashMap::new();
        for w in self.world.clone().iter() {
            let ReadyStatus::Ready { tickets: t } = &by_rank[&w.rank].status else {
                unreachable!("rejections returned above")
            };
            let ok = t.len() == lanes.len()
                && t.iter().zip(&lanes).all(|(t, l)| {
                    t.key == l.key
                        && t.target.is_some() == w.target
                        && t.head.is_some() == (w.head && lane_needs_head(l))
                });
            if !ok {
                return Err(self.fail(
                    Some(w.rank),
                    "ticket reservation does not match ownership".into(),
                ));
            }
            tickets.insert(w.rank, t.clone());
        }
        // Tickets of different lanes on one rank must be distinct.
        for (rank, t) in &tickets {
            let mut ids = HashSet::new();
            for lt in t {
                for id in [lt.target, lt.head].into_iter().flatten() {
                    if !ids.insert(id) {
                        return Err(self.fail(Some(*rank), "ticket reused across lanes".into()));
                    }
                }
            }
        }
        let s = self.step_mut();
        s.tickets = tickets;
        s.phase = StepPhase::Armed;
        self.phase = StepPhase::Armed;
        Ok(())
    }

    // ---- 3. draft rounds ------------------------------------------------

    /// The next ordered head round, or `None` once drafting ended
    /// (Armed → Drafted). Rounds are built from root-owned draft state
    /// only; no rank-local early exit exists.
    pub fn next_draft_round(&mut self) -> Result<Option<DraftRound>, ProtocolError> {
        self.check_live()?;
        if !matches!(self.phase, StepPhase::Armed) {
            return Err(ProtocolError::Phase {
                op: "next_draft_round",
                actual: self.phase,
            });
        }
        let s = self.step_ref();
        let mut ops = Vec::new();
        for (i, l) in s.start.lanes.iter().enumerate() {
            let d = &s.draft[i];
            if !d.active {
                continue;
            }
            let j = d.drafts.len();
            ops.push(DraftOp {
                key: l.key,
                position: l.position + j,
                input: if j == 0 { l.seed } else { d.drafts[j - 1] },
                kind: DraftOpKind::Draft,
            });
        }
        if ops.is_empty() {
            let s = self.step_mut();
            s.phase = StepPhase::Drafted;
            self.phase = StepPhase::Drafted;
            return Ok(None);
        }
        let round = DraftRound {
            key: s.start.key.clone(),
            round: s.round,
            ops,
        };
        Ok(Some(round))
    }

    /// Fold the head owners' answers to `round`. Followers acknowledge; only
    /// the sampler owner reads proposals. A proposal from anyone else, a
    /// missing/duplicate answer or a count mismatch aborts.
    pub fn collect_draft(
        &mut self,
        round: &DraftRound,
        results: &[RankDraftResult],
    ) -> Result<(), ProtocolError> {
        self.collect_draft_with(round, results, &mut GreedyRule)
    }

    /// [`Self::collect_draft`] with the acceptance rule that draws sampled
    /// proposals: for every proposal the root asks `rule.draft_token` with the
    /// lane's *staged* RNG (a Leviathan lane's token is drawn from its `q`
    /// here; nothing else draws). A rule error aborts the step and the staged
    /// RNG is discarded with it.
    pub fn collect_draft_with(
        &mut self,
        round: &DraftRound,
        results: &[RankDraftResult],
        rule: &mut dyn AcceptRule,
    ) -> Result<(), ProtocolError> {
        self.check_phase("collect_draft", StepPhase::Armed)?;
        {
            let s = self.step_ref();
            if round.key != s.start.key || round.round != s.round {
                return Err(self.fail(None, "draft answer for a stale round".into()));
            }
        }
        let owners: Vec<RankOwnership> = self.world.iter().copied().filter(|w| w.head).collect();
        let mut by_rank: HashMap<usize, &RankDraftResult> = HashMap::new();
        for r in results {
            if r.key != round.key || r.round != round.round {
                return Err(self.fail(Some(r.rank), "draft answer for a stale key/round".into()));
            }
            let Some(owner) = owners.iter().find(|w| w.rank == r.rank) else {
                return Err(self.fail(
                    Some(r.rank),
                    "draft answer from a rank without the head".into(),
                ));
            };
            if by_rank.insert(r.rank, r).is_some() {
                return Err(self.fail(Some(r.rank), "duplicate draft answer".into()));
            }
            if r.ops_executed != round.ops.len() || r.head_gathers != round.ops.len() {
                return Err(self.fail(
                    Some(r.rank),
                    format!(
                        "executed {} ops/{} gathers, scheduled {}",
                        r.ops_executed,
                        r.head_gathers,
                        round.ops.len()
                    ),
                ));
            }
            if r.proposals.is_some() != owner.sampler {
                return Err(self.fail(
                    Some(r.rank),
                    if owner.sampler {
                        "the sampler owner returned no proposals".to_string()
                    } else {
                        "a follower returned proposals: only the root reads/samples q".to_string()
                    },
                ));
            }
        }
        if let Some(rk) = owners
            .iter()
            .map(|w| w.rank)
            .find(|rk| !by_rank.contains_key(rk))
        {
            return Err(self.fail(Some(rk), "no draft answer".into()));
        }
        let sampler = owners.iter().find(|w| w.sampler).expect("validated world");
        let proposals = by_rank[&sampler.rank]
            .proposals
            .as_ref()
            .expect("checked above")
            .clone();
        if proposals.len() != round.ops.len()
            || proposals
                .iter()
                .zip(&round.ops)
                .any(|(p, o)| p.key != o.key)
        {
            return Err(self.fail(
                Some(sampler.rank),
                "proposals do not match the scheduled ops".into(),
            ));
        }
        let first_dropped = proposals.iter().any(|p| {
            p.drop_last
                && self
                    .step_ref()
                    .start
                    .lanes
                    .iter()
                    .position(|l| l.key == p.key)
                    .is_some_and(|i| self.step_ref().draft[i].drafts.is_empty())
        });
        if first_dropped {
            return Err(self.fail(
                Some(sampler.rank),
                "the first draft of a lane cannot be dropped".into(),
            ));
        }
        let n_ops = round.ops.len();
        let driver_terminates = self.driver_terminates;
        let mut failure: Option<String> = None;
        {
            let s = self.step_mut();
            for p in &proposals {
                let i = s
                    .start
                    .lanes
                    .iter()
                    .position(|l| l.key == p.key)
                    .expect("keys checked against the round");
                let index = s.draft[i].drafts.len();
                let token = match rule.draft_token(
                    &s.start.lanes[i],
                    index,
                    &s.draft[i].drafts,
                    p,
                    &mut s.staged[i].rng,
                ) {
                    Ok(token) => token,
                    Err(e) => {
                        failure = Some(format!("proposal {index} of lane {i}: {e}"));
                        break;
                    }
                };
                let max_depth = s.start.lanes[i].max_depth;
                let eos = s.start.lanes[i].eos;
                let past_eos = driver_terminates;
                let d = &mut s.draft[i];
                if p.drop_last {
                    // A dropped token never enters the window.
                    d.active = false;
                } else {
                    d.drafts.push(token);
                    d.active = p.keep_going
                        && d.drafts.len() < max_depth
                        && (past_eos || eos != Some(token));
                }
            }
            if failure.is_none() {
                s.round += 1;
                s.head_calls += n_ops;
            }
        }
        if let Some(reason) = failure {
            return Err(self.fail(None, reason));
        }
        Ok(())
    }

    // ---- 4. seal_verify -------------------------------------------------

    /// Seal the verify schedule after data-dependent draft truncation/EOS:
    /// final rows, row groups, the expected operation ledger and a digest.
    pub fn seal_verify(&mut self) -> Result<VerifyPlan, ProtocolError> {
        self.check_phase("seal_verify", StepPhase::Drafted)?;
        let s = self.step_ref();
        let mut lanes = Vec::with_capacity(s.start.lanes.len());
        for (i, l) in s.start.lanes.iter().enumerate() {
            let rows: Vec<u32> = match &l.kind {
                LaneKind::Forced(t) => std::iter::once(l.seed)
                    .chain(t[..t.len() - 1].iter().copied())
                    .collect(),
                LaneKind::Decode => {
                    let drafts: &[u32] = match l.source {
                        ProposalSource::Native => &s.draft[i].drafts,
                        ProposalSource::Copy => &l.copy_drafts,
                        ProposalSource::None => &[],
                    };
                    std::iter::once(l.seed)
                        .chain(drafts.iter().copied())
                        .collect()
                }
            };
            lanes.push(VerifyLane {
                key: l.key,
                position: l.position,
                capture: rows.len() > 1,
                rows,
            });
        }
        let b = &s.start.budgets;
        let mb_lanes = match (self.world[0].kind, b.pp_microbatch_lanes) {
            (MeshKind::Pp, Some(n)) if n > 0 => n,
            _ => lanes.len(),
        };
        let mut microbatches = Vec::new();
        let mut groups = Vec::new();
        let mut first = 0;
        while first < lanes.len() {
            let n = mb_lanes.min(lanes.len() - first);
            let rows: Vec<usize> = lanes[first..first + n]
                .iter()
                .map(|l| l.rows.len())
                .collect();
            for mut g in pack_row_groups(&rows, LANE_MAX_ROWS) {
                g.first_lane += first;
                groups.push(g);
            }
            microbatches.push(n);
            first += n;
        }
        let ledger = OpLedger {
            row_groups: groups.len(),
            target_exchanges: match self.world[0].kind {
                MeshKind::Pp | MeshKind::Single => 0,
                MeshKind::Ep if s.start.stage.moe_once => s.start.stage.moe_layers * groups.len(),
                MeshKind::Ep => s.start.stage.moe_layers * lanes.len(),
            },
            microbatches: match self.world[0].kind {
                MeshKind::Pp => microbatches.len(),
                MeshKind::Ep | MeshKind::Single => 0,
            },
            head_draft_calls: s.head_calls,
        };
        let mut d = Digest::new();
        d.u64(s.start.key.step_seq)
            .usize(ledger.row_groups)
            .usize(ledger.target_exchanges);
        for l in &lanes {
            d.lane_key(&l.key)
                .usize(l.position)
                .slice(&l.rows)
                .bool(l.capture);
        }
        for g in &groups {
            d.usize(g.first_lane).usize(g.lanes).usize(g.rows);
        }
        let plan = VerifyPlan {
            key: s.start.key.clone(),
            lanes,
            groups,
            microbatches,
            ledger,
            digest: d.finish(),
        };
        self.step_mut().verify = Some(plan.clone());
        Ok(plan)
    }

    // ---- 5. collect_verify ----------------------------------------------

    /// Fold every target owner's verify answer. Workers return state
    /// positions and the operation ledger; only the sampler owner returns
    /// picks. Any disagreement aborts.
    pub fn collect_verify(&mut self, results: &[RankVerifyResult]) -> Result<(), ProtocolError> {
        self.check_phase("collect_verify", StepPhase::Drafted)?;
        let plan = match self.step_ref().verify.clone() {
            Some(p) => p,
            None => {
                return Err(ProtocolError::Phase {
                    op: "collect_verify (verify not sealed)",
                    actual: self.phase,
                })
            }
        };
        let mut by_rank: HashMap<usize, &RankVerifyResult> = HashMap::new();
        let expected_ends: Vec<(LaneWindowKey, usize)> = plan
            .lanes
            .iter()
            .map(|l| (l.key, l.position + l.rows.len()))
            .collect();
        let owners: Vec<RankOwnership> = self.world.iter().copied().filter(|w| w.target).collect();
        for r in results {
            let Some(owner) = owners.iter().find(|w| w.rank == r.rank) else {
                return Err(self.fail(
                    Some(r.rank),
                    "verify answer from a rank without target state".into(),
                ));
            };
            if r.key != plan.key || r.plan_digest != plan.digest {
                return Err(self.fail(Some(r.rank), "verify answer for another plan".into()));
            }
            if by_rank.insert(r.rank, r).is_some() {
                return Err(self.fail(Some(r.rank), "duplicate verify answer".into()));
            }
            if r.lane_ends != expected_ends {
                return Err(self.fail(
                    Some(r.rank),
                    "target positions disagree with the sealed plan".into(),
                ));
            }
            if r.ledger.row_groups != plan.ledger.row_groups
                || r.ledger.target_exchanges != plan.ledger.target_exchanges
                || r.ledger.microbatches != plan.ledger.microbatches
            {
                return Err(self.fail(
                    Some(r.rank),
                    format!(
                        "operation ledger {:?} differs from the sealed {:?}",
                        r.ledger, plan.ledger
                    ),
                ));
            }
            if r.sampler.is_some() != owner.sampler {
                return Err(self.fail(
                    Some(r.rank),
                    if owner.sampler {
                        "the sampler owner returned no target output".to_string()
                    } else {
                        "a non-sampler returned target output: only the root samples".to_string()
                    },
                ));
            }
        }
        if let Some(w) = owners.iter().find(|w| !by_rank.contains_key(&w.rank)) {
            return Err(self.fail(Some(w.rank), "no verify answer".into()));
        }
        let sampler = owners.iter().find(|w| w.sampler).expect("validated world");
        let out = by_rank[&sampler.rank]
            .sampler
            .clone()
            .expect("checked above");
        if out.picks.len() != plan.lanes.len()
            || out
                .picks
                .iter()
                .zip(&plan.lanes)
                .any(|((k, p), l)| *k != l.key || p.len() != l.rows.len())
        {
            return Err(self.fail(
                Some(sampler.rank),
                "picks do not cover the sealed rows".into(),
            ));
        }
        let s = self.step_mut();
        s.picks = Some(out);
        s.phase = StepPhase::Verified;
        self.phase = StepPhase::Verified;
        Ok(())
    }

    // ---- 6. decide_commit -----------------------------------------------

    /// Root-only acceptance and RNG decision, once per lane. The RNG
    /// successor, policy successor and host-ID updates are *staged*.
    pub fn decide_commit(
        &mut self,
        rule: &mut dyn AcceptRule,
    ) -> Result<CommitDecision, ProtocolError> {
        self.check_phase("decide_commit", StepPhase::Verified)?;
        let (start, plan, picks) = {
            let s = self.step_ref();
            (
                s.start.clone(),
                s.verify.clone().expect("verified phase holds a plan"),
                s.picks.clone().expect("verified phase holds picks"),
            )
        };
        rule.observe_verify(&picks);
        let mut decisions = Vec::with_capacity(start.lanes.len());
        let mut updates: Vec<(usize, WindowUpdate, Option<MtpWindow>)> = Vec::new();
        let mut rng_out: Vec<LaneRng> = Vec::new();
        for (i, l) in start.lanes.iter().enumerate() {
            let vl = &plan.lanes[i];
            let lane_picks = &picks.picks[i].1;
            let mut rng = self.step_ref().staged[i].rng;
            let (emitted, accepted, hit_eos, forced, consumed, offered) = match &l.kind {
                LaneKind::Forced(t) => (t.clone(), t.len() - 1, false, true, vl.rows.len(), 0),
                LaneKind::Decode => {
                    let drafts = &vl.rows[1..];
                    let acc = match rule.accept(l, drafts, lane_picks, &mut rng) {
                        Ok(acc) => acc,
                        Err(e) => {
                            return Err(
                                self.fail(None, format!("acceptance of lane {i} failed: {e}"))
                            )
                        }
                    };
                    let accepted_eos = acc.hit_eos && acc.committed.len() == acc.accepted;
                    let shape_ok = acc.accepted <= drafts.len()
                        && !acc.committed.is_empty()
                        && acc.committed.len() >= acc.accepted
                        && acc.committed.len() <= acc.accepted + 1
                        && acc.committed.len() == acc.accepted + usize::from(!accepted_eos)
                        && acc.committed[..acc.accepted] == drafts[..acc.accepted];
                    if !shape_ok {
                        return Err(self.fail(
                            None,
                            format!("acceptance of lane {i} violates the window shape: {acc:?}"),
                        ));
                    }
                    let consumed = 1 + target_commit_accept_len(&acc);
                    let n = drafts.len();
                    (
                        acc.committed.clone(),
                        acc.accepted,
                        acc.hit_eos,
                        false,
                        consumed,
                        n,
                    )
                }
            };
            debug_assert!(consumed >= 1 && consumed <= vl.rows.len());
            let target_end = l.position + consumed;
            let next_seed = *emitted.last().expect("non-empty emission");
            let terminal = hit_eos || emitted.len() >= l.remaining;
            let head_end = match l.head {
                HeadState::Aligned => HeadEnd::Aligned { end: target_end },
                HeadState::Lag { from } => HeadEnd::Lag {
                    from,
                    to: target_end,
                },
                HeadState::Free => HeadEnd::Free,
            };
            decisions.push(LaneDecision {
                key: l.key,
                emitted: emitted.clone(),
                accepted,
                hit_eos,
                forced,
                consumed_rows: consumed,
                next_seed,
                terminal,
                target_end,
                pending_hidden_row: consumed - 1,
                head_end,
                source: SourceAccounting {
                    source: l.source,
                    offered,
                    accepted: if forced { 0 } else { accepted },
                },
            });
            let window = (!forced).then(|| MtpWindow {
                committed: emitted.clone(),
                accepted,
                drafts_generated: offered,
            });
            updates.push((
                i,
                WindowUpdate {
                    emitted,
                    position: target_end,
                    seed: next_seed,
                    terminal,
                },
                window,
            ));
            rng_out.push(rng);
        }
        let digest = decision_digest(&start.key, &decisions);
        let decision = CommitDecision {
            key: start.key.clone(),
            lanes: decisions,
            digest,
        };

        // Pending price observation for the next plan.
        let decode_lanes = start.lanes.len();
        let verified: usize = decision.lanes.iter().map(|d| d.source.offered).sum();
        let verify_windows = decision
            .lanes
            .iter()
            .filter(|d| d.source.offered > 0)
            .count();
        let spill = plan.ledger.row_groups.saturating_sub(1);
        let obs_lanes = start
            .lanes
            .iter()
            .zip(&updates)
            .filter_map(|(l, (_, _, w))| {
                let w = w.clone()?;
                // Only native windows (and the synced AR row of a spec lane)
                // price the floor; copy windows are root copy accounting.
                (l.source != ProposalSource::Copy
                    && matches!(
                        l.route,
                        LaneRoute::Batched { .. } | LaneRoute::ArSynced | LaneRoute::ArRetired
                    ))
                .then_some(PendingLaneObs {
                    epoch: l.key.request_epoch,
                    route: l.route,
                    source: l.source,
                    window: w,
                    budget: l.budget,
                })
            })
            .collect();
        let expect = EvidenceExpectation {
            key: start.key.clone(),
            world: self.world.clone(),
            ledger: plan.ledger,
            head_append_rows: decision
                .lanes
                .iter()
                .zip(&start.lanes)
                .filter(|(_, l)| l.head != HeadState::Free)
                .map(|(d, _)| d.consumed_rows)
                .sum(),
            drafts_generated: verified,
            accepted: decision.lanes.iter().map(|d| d.source.accepted).sum(),
            consumed_rows: decision.lanes.iter().map(|d| d.consumed_rows).sum(),
            overlap: start.budgets.pp_microbatch_lanes.is_some(),
            decode_lanes,
        };
        let s = self.step_mut();
        for ((i, up, _), rng) in updates.into_iter().zip(rng_out) {
            s.staged[i].rng = rng;
            s.staged[i].window = Some(up);
        }
        s.pending_obs = Some(PendingObservation {
            key: start.key.clone(),
            decode_lanes,
            spill,
            verified,
            verify_windows,
            lanes: obs_lanes,
            expect,
        });
        s.decision = Some(decision.clone());
        s.phase = StepPhase::Decided;
        self.phase = StepPhase::Decided;
        Ok(decision)
    }

    // ---- 7. collect_prepared --------------------------------------------

    /// Every owner must prove it repaired to exactly the decision (target
    /// and head positions, the reserved ticket identities, the decision
    /// digest). Any failure or disagreement poisons the step/mesh.
    pub fn collect_prepared(&mut self, receipts: &[PreparedReceipt]) -> Result<(), ProtocolError> {
        self.check_phase("collect_prepared", StepPhase::Decided)?;
        let (decision, tickets) = {
            let s = self.step_ref();
            (s.decision.clone().expect("decided"), s.tickets.clone())
        };
        let lanes = self.step_ref().start.lanes.clone();
        let mut seen = HashSet::new();
        let world = self.world.clone();
        for r in receipts {
            let Some(w) = world.iter().find(|w| w.rank == r.rank) else {
                return Err(self.fail(Some(r.rank), "prepared receipt from an unknown rank".into()));
            };
            if !seen.insert(r.rank) {
                return Err(self.fail(Some(r.rank), "duplicate prepared receipt".into()));
            }
            if r.key != decision.key || r.decision_digest != decision.digest {
                return Err(self.fail(Some(r.rank), "prepared for a different decision".into()));
            }
            if r.lanes.len() != decision.lanes.len() {
                return Err(self.fail(Some(r.rank), "prepared lane count differs".into()));
            }
            let reserved = &tickets[&r.rank];
            for ((pl, d), (rt, plan)) in r
                .lanes
                .iter()
                .zip(&decision.lanes)
                .zip(reserved.iter().zip(&lanes))
            {
                if pl.key != d.key || pl.tickets != *rt {
                    return Err(self.fail(
                        Some(r.rank),
                        "ticket identity differs from the reservation".into(),
                    ));
                }
                let want_target = w.target.then_some(d.target_end);
                let want_head = (w.head && lane_needs_head(plan)).then_some(d.head_end);
                if pl.target_end != want_target || pl.head_end != want_head {
                    return Err(self.fail(
                        Some(r.rank),
                        format!(
                            "positions {:?}/{:?} differ from the decision {:?}/{:?}",
                            pl.target_end, pl.head_end, want_target, want_head
                        ),
                    ));
                }
            }
        }
        if let Some(w) = world.iter().find(|w| !seen.contains(&w.rank)) {
            return Err(self.fail(Some(w.rank), "no prepared receipt".into()));
        }
        let s = self.step_mut();
        s.phase = StepPhase::Prepared;
        self.phase = StepPhase::Prepared;
        Ok(())
    }

    // ---- 8. publish_commit ----------------------------------------------

    /// Apply the staged successors and return the step's output. Reachable
    /// only from `Prepared`; no fallible operation remains, so it returns
    /// no error beyond the phase check. Cancelled lanes publish nothing and
    /// are released.
    pub fn publish_commit(&mut self) -> Result<StepOutput, ProtocolError> {
        self.check_phase("publish_commit", StepPhase::Prepared)?;
        let s = self.step.take().expect("prepared phase holds a step");
        let decision = s.decision.expect("prepared");
        let mut out = Vec::with_capacity(decision.lanes.len());
        self.cost = s.staged_cost;
        if s.obs_applied {
            self.counters.observations_applied += 1;
        }
        if s.obs_dropped {
            self.counters.observations_dropped += 1;
        }
        for (st, d) in s.staged.into_iter().zip(&decision.lanes) {
            let cancelled = s.cancelled.contains(&st.epoch);
            if cancelled {
                self.lanes.remove(&st.epoch);
            } else {
                let lane = self
                    .lanes
                    .get_mut(&st.epoch)
                    .expect("lane held through the step");
                let up = st.window.expect("decided lane");
                lane.rng = st.rng;
                lane.policy = st.policy;
                lane.window_seq += 1;
                lane.position = up.position;
                lane.seed = up.seed;
                lane.generated.extend_from_slice(&up.emitted);
                lane.copy.tokens.extend_from_slice(&up.emitted);
                if up.terminal && !self.driver_terminates {
                    lane.status = LaneStatus::Terminal;
                }
            }
            out.push(LaneOutput {
                key: d.key,
                emitted: if cancelled {
                    Vec::new()
                } else {
                    d.emitted.clone()
                },
                forced: d.forced,
                terminal: d.terminal && !cancelled,
                cancelled,
                consumed_rows: d.consumed_rows,
                next_seed: d.next_seed,
                source: d.source,
            });
        }
        self.pending_obs = s.pending_obs;
        self.counters.steps_published += 1;
        self.phase = StepPhase::Published;
        Ok(StepOutput {
            key: decision.key,
            lanes: out,
        })
    }

    /// What the published step promised, for [`merge_evidence`].
    pub fn evidence_expectation(&self) -> Option<&EvidenceExpectation> {
        self.pending_obs.as_ref().map(|p| &p.expect)
    }

    /// Merge the previous step's receipts (root convenience over
    /// [`merge_evidence`]); stale receipts are rejected.
    pub fn merge_evidence(
        &self,
        receipts: &[RankCostEvidence],
        interval: ControllerInterval,
    ) -> Result<MergedCostEvidence, EvidenceError> {
        let expect = self
            .evidence_expectation()
            .ok_or_else(|| EvidenceError::Stale("no published step awaits evidence".to_string()))?;
        merge_evidence(expect, receipts, interval)
    }
}

/// The single-card route's additions (a [`MeshKind::Single`] root).
impl MeshRoot {
    /// The driver owns request termination and drafting rules of the
    /// singleton lane route: an EOS or length completion never closes a
    /// lane at the root (the driver releases it, and today still forwards
    /// an EOS pending seed once), and a native draft chain runs through an
    /// EOS draft (the verify truncates at it). Off by default (mesh
    /// topologies close lanes at the root and stop drafting at EOS).
    pub fn set_driver_terminates(&mut self, on: bool) {
        self.driver_terminates = on;
    }

    /// Whether `HIPFIRE_MTP_AR_FLOOR` is on for the lanes admitted from now
    /// on (the singleton lane route reads it at every admission).
    pub fn set_floor_enabled(&mut self, on: bool) {
        self.cfg.floor_enabled = on;
    }

    /// Whether the root holds `epoch`'s lane.
    pub fn has_lane(&self, epoch: RequestEpoch) -> bool {
        self.lanes.contains_key(&epoch)
    }

    /// Admit a promoted singleton as an already-decoding lane: its drafter's
    /// per-depth `agreement` seeds the lane policy (`retired` keeps it
    /// `ArRetired` for good). The floor is not reset.
    pub fn admit_promoted(
        &mut self,
        a: LaneAdmission,
        agreement: &[(f32, f32)],
        retired: bool,
    ) -> Result<LaneAdmit, ProtocolError> {
        let policy = a.spec_k.map(|k| {
            LaneSpecPolicy::promoted_with_floor(k, agreement, retired, self.cfg.floor_enabled)
        });
        self.admit_with(a, policy)
    }

    /// Replace a live lane's pending seed (the emitter's forced close
    /// tokens): not a proposal, no RNG draw, no policy observation; the
    /// generated IDs and the position are untouched.
    pub fn force_seed(&mut self, epoch: RequestEpoch, token: u32) -> Result<(), ProtocolError> {
        self.check_live()?;
        if self.in_flight(epoch) {
            return Err(ProtocolError::Phase {
                op: "force_seed (in-flight step)",
                actual: self.phase,
            });
        }
        let lane = self
            .lanes
            .get_mut(&epoch)
            .ok_or(ProtocolError::UnknownLane(epoch))?;
        if lane.status != LaneStatus::Live {
            return Err(ProtocolError::LaneNotLive(epoch));
        }
        lane.seed = token;
        Ok(())
    }

    /// Apply the previous published step's price now, to the cost model and
    /// to every lane still held (not only the next step's lanes: a lane
    /// that skips the next step, or ends, is still observed exactly as the
    /// singleton lane route observes at the end of its forward). `None`
    /// drops the price (probe budget spent unpriced). Idle/Published only;
    /// stale evidence changes nothing.
    pub fn settle_evidence(
        &mut self,
        merged: Option<&MergedCostEvidence>,
    ) -> Result<(), ProtocolError> {
        self.check_live()?;
        if !matches!(self.phase, StepPhase::Idle | StepPhase::Published) {
            return Err(ProtocolError::Phase {
                op: "settle_evidence",
                actual: self.phase,
            });
        }
        match (&self.pending_obs, merged) {
            (None, None) => return Ok(()),
            (None, Some(m)) => {
                return Err(ProtocolError::StaleEvidence(format!(
                    "evidence for step {} but no step awaits a price",
                    m.key.step_seq
                )))
            }
            (Some(p), Some(m)) if m.key != p.key => {
                return Err(ProtocolError::StaleEvidence(format!(
                    "evidence is for step {} (gen {}), pending is step {} (gen {})",
                    m.key.step_seq,
                    m.key.topology_generation,
                    p.key.step_seq,
                    p.key.topology_generation
                )))
            }
            _ => {}
        }
        let p = self.pending_obs.take().expect("checked above");
        let Some(m) = merged else {
            self.counters.observations_dropped += 1;
            return Ok(());
        };
        if m.priced {
            self.cost
                .observe(p.decode_lanes, p.spill, p.verified, m.wall_us as f32);
        }
        let spill_per_window = p.spill as f64 / p.verify_windows.max(1) as f64;
        for o in &p.lanes {
            let Some(policy) = self.lanes.get_mut(&o.epoch).and_then(|l| l.policy.as_mut()) else {
                continue; // released since: nothing to teach
            };
            observe_lane(&self.cost, policy, &p, o, m, spill_per_window);
        }
        self.counters.observations_applied += 1;
        Ok(())
    }

    /// Single-rank recovery from a step that failed (a device error, a
    /// cancelled participant): drop the in-flight step, its staged state and
    /// its pending price, forget `epochs` (the poisoned participants) and
    /// return to Idle. Other lanes keep their state: on one card nothing
    /// but the participants' own device state is uncertain, so the mesh-wide
    /// quarantine of [`Self::abort_step`] does not apply. Refused on a mesh
    /// topology (use `abort_step` and quarantine there).
    pub fn discard_step(&mut self, epochs: &[RequestEpoch]) -> Result<(), ProtocolError> {
        if self.own.kind != MeshKind::Single {
            return Err(ProtocolError::Config(
                "discard_step is the single-rank recovery; a mesh quarantines".to_string(),
            ));
        }
        self.step = None;
        self.pending_obs = None;
        self.aborted = None;
        self.phase = StepPhase::Idle;
        for e in epochs {
            self.lanes.remove(e);
        }
        Ok(())
    }
}

impl RankSession {
    /// Single-rank recovery: drop the in-flight step and clear quarantine
    /// (see [`MeshRoot::discard_step`]). Refused unless this rank is a
    /// [`MeshKind::Single`] rank.
    pub fn discard_step(&mut self) -> Result<(), RankError> {
        if self.own.kind != MeshKind::Single {
            return Err(RankError::new(
                self.own.rank,
                "discard_step is the single-rank recovery",
            ));
        }
        self.step = None;
        self.aborted = false;
        Ok(())
    }

    /// Forget a lane without the idle check (a lane poisoned mid-step).
    pub fn forget_lane(&mut self, epoch: &RequestEpoch) {
        self.live.remove(epoch);
    }
}

/// Whether a lane's window touches the head on head owners (a head-free
/// lane or non-spec lane allocates/uses none).
fn lane_needs_head(l: &LanePlan) -> bool {
    l.head != HeadState::Free
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Released at once (no step in flight over the lane).
    Released,
    /// In a step: output withheld and the lane released at publish.
    Deferred { phase: StepPhase },
}

/// Fold merged evidence into the staged cost model and the lanes' policies.
/// An unpriced price spends probe budget without recording a cost.
fn apply_evidence(
    cost: &mut LaneCostModel,
    staged: &mut HashMap<RequestEpoch, StagedLane>,
    _lanes: &HashMap<RequestEpoch, RootLane>,
    p: &PendingObservation,
    m: &MergedCostEvidence,
) {
    if m.priced {
        cost.observe(p.decode_lanes, p.spill, p.verified, m.wall_us as f32);
    }
    let spill_per_window = p.spill as f64 / p.verify_windows.max(1) as f64;
    for o in &p.lanes {
        let Some(sl) = staged.get_mut(&o.epoch) else {
            continue; // released/terminal since: nothing to teach
        };
        let Some(policy) = sl.policy.as_mut() else {
            continue;
        };
        observe_lane(cost, policy, p, o, m, spill_per_window);
    }
}

/// One lane's floor observation of a finished window, priced at the cost
/// model's units (NaN when the step's units are not valid).
fn observe_lane(
    cost: &LaneCostModel,
    policy: &mut LaneSpecPolicy,
    p: &PendingObservation,
    o: &PendingLaneObs,
    m: &MergedCostEvidence,
    spill_per_window: f64,
) {
    match o.route {
        LaneRoute::Batched { k_max } if o.source == ProposalSource::Native => {
            let units = if m.units_valid {
                cost.window_units(p.decode_lanes, o.window.drafts_generated, spill_per_window)
            } else {
                f32::NAN
            };
            policy.observe_batched(&o.window, units, k_max, o.budget);
        }
        LaneRoute::ArSynced => policy.observe_ar_synced(&o.window, o.budget),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Rank side
// ---------------------------------------------------------------------------

/// The device-facing participant of one rank (EP rank / PP band). Topology
/// owners implement it over their real stores; the scripted executor in
/// [`scripted`] implements it for CPU tests. Every method is the rank's
/// replay of a root message: none chooses a route, depth, proposal,
/// acceptance or RNG draw.
pub trait LaneMeshExecutor {
    fn ownership(&self) -> RankOwnership;

    /// Mirror an admission (identity only).
    fn admit_lane(&mut self, admit: &LaneAdmit) -> Result<(), RankError>;
    fn release_lane(&mut self, epoch: &RequestEpoch) -> Result<(), RankError>;

    /// `begin_window`: validate keys/capacity/ownership, allocate/map and
    /// reserve tickets where owned, **before any collective**.
    fn begin_window(&mut self, start: &WindowStart) -> RankReady;
    /// Unwind a `begin_window` whose step the root rejected before any
    /// device work (another rank refused). Releases reservations; the rank
    /// stays usable.
    fn reject_window(&mut self, key: &MeshStepKey);

    /// Head owners only.
    fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError>;
    /// Every target owner.
    fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError>;
    /// Repair/truncate owned target and head state to the decision. Any
    /// error poisons the step.
    fn prepare_commit(&mut self, decision: &CommitDecision) -> Result<PreparedReceipt, RankError>;
    /// Consume validated tickets and mirror decision metadata. Infallible:
    /// no fallible GPU operation exists past the all-prepared boundary.
    fn publish_commit(&mut self, decision: &CommitDecision);
    /// Idempotent: wake every participant/barrier, quarantine.
    fn abort_step(&mut self, key: &MeshStepKey, phase: StepPhase, reason: &str);
    /// This rank's evidence for a finished step.
    fn evidence(&mut self, key: &MeshStepKey) -> Option<RankCostEvidence>;
}

/// The rank-side phase/ownership reducer. Executors embed it so the
/// validation of every root message (key, order, ownership, digests) is
/// written once.
pub struct RankSession {
    own: RankOwnership,
    live: HashMap<RequestEpoch, u64>,
    next_ticket: u64,
    step: Option<RankStep>,
    aborted: bool,
}

struct RankStep {
    start: WindowStart,
    phase: StepPhase,
    tickets: Vec<LaneTickets>,
    round: u32,
    next_pos: Vec<usize>,
    verify: Option<VerifyPlan>,
    decision: Option<CommitDecision>,
    head_calls: usize,
}

impl RankSession {
    pub fn new(own: RankOwnership) -> Self {
        Self {
            own,
            live: HashMap::new(),
            next_ticket: 1,
            step: None,
            aborted: false,
        }
    }

    pub fn ownership(&self) -> RankOwnership {
        self.own
    }

    pub fn phase(&self) -> StepPhase {
        if self.aborted {
            StepPhase::Aborted
        } else {
            self.step.as_ref().map_or(StepPhase::Idle, |s| s.phase)
        }
    }

    pub fn is_quarantined(&self) -> bool {
        self.aborted
    }

    pub fn admit(&mut self, a: &LaneAdmit) -> Result<(), RankError> {
        if self.aborted || self.step.is_some() {
            return Err(RankError::new(
                self.own.rank,
                "admit_lane: rank busy or quarantined",
            ));
        }
        if self.live.insert(a.epoch, a.lane_generation).is_some() {
            return Err(RankError::new(
                self.own.rank,
                "admit_lane: epoch already live",
            ));
        }
        Ok(())
    }

    pub fn release(&mut self, epoch: &RequestEpoch) -> Result<(), RankError> {
        if self.step.is_some() {
            return Err(RankError::new(
                self.own.rank,
                "release_lane: step in flight",
            ));
        }
        self.live
            .remove(epoch)
            .map(|_| ())
            .ok_or_else(|| RankError::new(self.own.rank, "release_lane: unknown epoch"))
    }

    /// Validate a window start. `capacity` is the rank's own preflight
    /// (mapping/allocation) outcome. On success reserves one ticket per
    /// owned slot of every lane.
    pub fn begin(&mut self, start: &WindowStart, capacity: Result<(), String>) -> RankReady {
        let rank = self.own.rank;
        let reject = |reason: String| RankReady {
            key: start.key.clone(),
            rank,
            start_digest: start.digest,
            status: ReadyStatus::Rejected { reason },
        };
        if self.aborted {
            return reject("rank is quarantined".into());
        }
        if self.step.is_some() {
            return reject("previous step not finished".into());
        }
        if start.key.lanes.len() != start.lanes.len()
            || start
                .lanes
                .iter()
                .zip(&start.key.lanes)
                .any(|(l, k)| l.key != *k)
        {
            return reject("plan lanes do not match the step key".into());
        }
        let digest = start_digest(&start.key, &start.lanes, &start.budgets, &start.stage);
        if digest != start.digest {
            return reject("plan digest mismatch".into());
        }
        for l in &start.lanes {
            match self.live.get(&l.key.request_epoch) {
                None => return reject(format!("stale or unknown epoch {:?}", l.key.request_epoch)),
                Some(g) if *g != l.key.lane_generation => {
                    return reject(format!(
                        "stale lane generation for {:?}",
                        l.key.request_epoch
                    ))
                }
                _ => {}
            }
            if l.max_rows() > start.budgets.verify_rows_per_lane {
                return reject("lane rows exceed the admitted verify width".into());
            }
        }
        if let Err(e) = capacity {
            return reject(e);
        }
        let mut tickets = Vec::with_capacity(start.lanes.len());
        for l in &start.lanes {
            let mut take = || {
                let t = TicketId(self.next_ticket);
                self.next_ticket += 1;
                t
            };
            tickets.push(LaneTickets {
                key: l.key,
                target: self.own.target.then(&mut take),
                head: (self.own.head && lane_needs_head(l)).then(&mut take),
            });
        }
        let ready = RankReady {
            key: start.key.clone(),
            rank,
            start_digest: start.digest,
            status: ReadyStatus::Ready {
                tickets: tickets.clone(),
            },
        };
        self.step = Some(RankStep {
            next_pos: start.lanes.iter().map(|l| l.position).collect(),
            start: start.clone(),
            phase: StepPhase::Armed,
            tickets,
            round: 0,
            verify: None,
            decision: None,
            head_calls: 0,
        });
        ready
    }

    pub fn reject(&mut self, key: &MeshStepKey) {
        if matches!(&self.step, Some(s) if s.phase == StepPhase::Armed && s.start.key == *key) {
            self.step = None;
        }
    }

    fn step_for(
        &mut self,
        key: &MeshStepKey,
        want: StepPhase,
        op: &str,
    ) -> Result<&mut RankStep, RankError> {
        let rank = self.own.rank;
        if self.aborted {
            return Err(RankError::new(rank, format!("{op}: rank is quarantined")));
        }
        match self.step.as_mut() {
            Some(s) if s.start.key == *key && s.phase == want => Ok(s),
            Some(s) if s.start.key != *key => {
                Err(RankError::new(rank, format!("{op}: stale step key")))
            }
            Some(s) => Err(RankError::new(
                rank,
                format!("{op}: phase {:?}, expected {want:?}", s.phase),
            )),
            None => Err(RankError::new(rank, format!("{op}: no step begun"))),
        }
    }

    /// Validate a head round: the rank must own the head, rounds are
    /// consecutive, every op continues a lane at the next position. Returns
    /// the ack with `proposals: None`; a sampler owner attaches its own.
    pub fn on_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
        let rank = self.own.rank;
        if !self.own.head {
            return Err(RankError::new(
                rank,
                "draft round on a rank without the head",
            ));
        }
        let s = self.step_for(&round.key, StepPhase::Armed, "draft round")?;
        if round.round != s.round {
            return Err(RankError::new(rank, "draft round out of order"));
        }
        for op in &round.ops {
            let Some(i) = s.start.lanes.iter().position(|l| l.key == op.key) else {
                return Err(RankError::new(rank, "draft op for a lane outside the plan"));
            };
            let l = &s.start.lanes[i];
            if l.source != ProposalSource::Native || op.position != s.next_pos[i] {
                return Err(RankError::new(
                    rank,
                    "draft op position/source disagrees with the plan",
                ));
            }
            if op.position == l.position && op.input != l.seed {
                return Err(RankError::new(
                    rank,
                    "first draft input is not the planned seed",
                ));
            }
            s.next_pos[i] += 1;
        }
        s.round += 1;
        s.head_calls += round.ops.len();
        Ok(RankDraftResult {
            key: round.key.clone(),
            round: round.round,
            rank,
            ops_executed: round.ops.len(),
            head_gathers: round.ops.len(),
            proposals: None,
        })
    }

    /// Validate the sealed verify against the plan and this rank's
    /// ownership; returns the state-position/ledger receipt (no sampler
    /// output; a sampler owner attaches its own).
    pub fn on_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
        let rank = self.own.rank;
        if !self.own.target {
            return Err(RankError::new(
                rank,
                "verify on a rank without target state",
            ));
        }
        let own_head = self.own.head;
        let s = self.step_for(&plan.key, StepPhase::Armed, "verify")?;
        if plan.lanes.len() != s.start.lanes.len()
            || plan.lanes.iter().zip(&s.start.lanes).any(|(v, l)| {
                v.key != l.key
                    || v.position != l.position
                    || v.rows.first() != Some(&l.seed)
                    || v.rows.len() > l.max_rows()
            })
        {
            return Err(RankError::new(
                rank,
                "verify plan departs from the window start",
            ));
        }
        if plan
            .lanes
            .iter()
            .any(|l| l.rows.is_empty() || l.rows.len() > LANE_MAX_ROWS)
        {
            return Err(RankError::new(
                rank,
                "verify lane outside the row-exact width",
            ));
        }
        if own_head && s.head_calls != plan.ledger.head_draft_calls {
            return Err(RankError::new(
                rank,
                "head call count differs from the ledger",
            ));
        }
        s.verify = Some(plan.clone());
        s.phase = StepPhase::Verified;
        Ok(RankVerifyResult {
            key: plan.key.clone(),
            rank,
            plan_digest: plan.digest,
            lane_ends: plan
                .lanes
                .iter()
                .map(|l| (l.key, l.position + l.rows.len()))
                .collect(),
            ledger: plan.ledger,
            sampler: None,
        })
    }

    /// Validate a decision and produce the receipt: this rank restored its
    /// *own* state to the decision's positions under its reserved tickets.
    pub fn on_prepare(&mut self, decision: &CommitDecision) -> Result<PreparedReceipt, RankError> {
        let rank = self.own.rank;
        let own = self.own;
        let s = self.step_for(&decision.key, StepPhase::Verified, "prepare")?;
        if decision.lanes.len() != s.start.lanes.len()
            || decision_digest(&decision.key, &decision.lanes) != decision.digest
        {
            return Err(RankError::new(rank, "decision digest mismatch"));
        }
        let verify = s.verify.as_ref().expect("verified");
        let mut lanes = Vec::with_capacity(decision.lanes.len());
        for (((d, l), v), t) in decision
            .lanes
            .iter()
            .zip(&s.start.lanes)
            .zip(&verify.lanes)
            .zip(&s.tickets)
        {
            if d.key != l.key
                || d.consumed_rows == 0
                || d.consumed_rows > v.rows.len()
                || d.target_end != l.position + d.consumed_rows
                || d.pending_hidden_row + 1 != d.consumed_rows
            {
                return Err(RankError::new(
                    rank,
                    "decision departs from the sealed rows",
                ));
            }
            lanes.push(PreparedLane {
                key: d.key,
                target_end: own.target.then_some(d.target_end),
                head_end: (own.head && lane_needs_head(l)).then_some(d.head_end),
                tickets: *t,
            });
        }
        let head_append_rows = if own.head {
            decision
                .lanes
                .iter()
                .zip(&s.start.lanes)
                .filter(|(_, l)| l.head != HeadState::Free)
                .map(|(d, _)| d.consumed_rows)
                .sum()
        } else {
            0
        };
        s.decision = Some(decision.clone());
        s.phase = StepPhase::Prepared;
        Ok(PreparedReceipt {
            key: decision.key.clone(),
            rank,
            decision_digest: decision.digest,
            lanes,
            head_append_rows,
        })
    }

    /// Consume the tickets and mirror metadata; the step ends.
    pub fn on_publish(&mut self, decision: &CommitDecision) {
        if self.aborted {
            return;
        }
        if matches!(&self.step, Some(s) if s.phase == StepPhase::Prepared && s.decision.as_ref() == Some(decision))
        {
            self.step = None;
        }
    }

    /// Idempotent abort: the rank is quarantined and drops the step.
    pub fn on_abort(&mut self) {
        self.aborted = true;
        self.step = None;
    }

    /// The expected per-rank draft-call count of the begun step (for the
    /// executor's own ledger).
    pub fn head_calls(&self) -> usize {
        self.step.as_ref().map_or(0, |s| s.head_calls)
    }
}

// ---------------------------------------------------------------------------
// Reference driver
// ---------------------------------------------------------------------------

fn abort_all(
    root: &mut MeshRoot,
    ranks: &mut [&mut dyn LaneMeshExecutor],
    reason: &str,
) -> ProtocolError {
    let key = root.step.as_ref().map(|s| s.start.key.clone());
    let phase = root.phase;
    let record = match root.abort_step(key.as_ref(), phase, reason) {
        Ok(r) => r,
        Err(e) => return e,
    };
    let abort_key = record.key.clone().or(key);
    for r in ranks.iter_mut() {
        if let Some(k) = &abort_key {
            r.abort_step(k, record.phase, &record.reason);
        }
    }
    ProtocolError::Aborted(record)
}

/// Run one complete mesh step in the protocol's message order against
/// `ranks` (dispatched serially here; topology owners dispatch each phase to
/// all ranks concurrently, but in this order with the same messages).
///
/// Any rank error or reducer-detected violation aborts the root and every
/// rank (one record, quarantine); a preflight rejection unwinds without
/// quarantine.
pub fn run_mesh_step(
    root: &mut MeshRoot,
    ranks: &mut [&mut dyn LaneMeshExecutor],
    prior_evidence: Option<&MergedCostEvidence>,
    requests: &[LaneRequest],
    budgets: &WindowBudgets,
    rule: &mut dyn AcceptRule,
) -> Result<StepOutput, ProtocolError> {
    let start = root.plan_window(prior_evidence, requests, budgets)?;
    let ready: Vec<RankReady> = ranks.iter_mut().map(|r| r.begin_window(&start)).collect();
    match root.collect_ready(&ready) {
        Ok(()) => {}
        Err(ProtocolError::Preflight(v)) => {
            for r in ranks.iter_mut() {
                r.reject_window(&start.key);
            }
            return Err(ProtocolError::Preflight(v));
        }
        Err(e) => return Err(abort_all_after(root, ranks, e)),
    }
    loop {
        let round = match root.next_draft_round() {
            Ok(Some(r)) => r,
            Ok(None) => break,
            Err(e) => return Err(abort_all_after(root, ranks, e)),
        };
        let mut results = Vec::new();
        for r in ranks.iter_mut().filter(|r| r.ownership().head) {
            match r.run_draft_round(&round) {
                Ok(x) => results.push(x),
                Err(e) => {
                    return Err(abort_all(
                        root,
                        ranks,
                        &format!("rank {}: {}", e.rank, e.reason),
                    ))
                }
            }
        }
        if let Err(e) = root.collect_draft_with(&round, &results, rule) {
            return Err(abort_all_after(root, ranks, e));
        }
    }
    let plan = match root.seal_verify() {
        Ok(p) => p,
        Err(e) => return Err(abort_all_after(root, ranks, e)),
    };
    let mut results = Vec::new();
    for r in ranks.iter_mut().filter(|r| r.ownership().target) {
        match r.run_verify(&plan) {
            Ok(x) => results.push(x),
            Err(e) => {
                return Err(abort_all(
                    root,
                    ranks,
                    &format!("rank {}: {}", e.rank, e.reason),
                ))
            }
        }
    }
    if let Err(e) = root.collect_verify(&results) {
        return Err(abort_all_after(root, ranks, e));
    }
    let decision = match root.decide_commit(rule) {
        Ok(d) => d,
        Err(e) => return Err(abort_all_after(root, ranks, e)),
    };
    let mut receipts = Vec::new();
    for r in ranks.iter_mut() {
        match r.prepare_commit(&decision) {
            Ok(x) => receipts.push(x),
            Err(e) => {
                return Err(abort_all(
                    root,
                    ranks,
                    &format!("rank {}: {}", e.rank, e.reason),
                ))
            }
        }
    }
    if let Err(e) = root.collect_prepared(&receipts) {
        return Err(abort_all_after(root, ranks, e));
    }
    let out = root.publish_commit()?;
    rule.on_publish(&out);
    for r in ranks.iter_mut() {
        r.publish_commit(&decision);
    }
    Ok(out)
}

/// The reducer already recorded the abort for rank-originated violations;
/// make the ranks follow. A pure caller-bug error is returned as is.
fn abort_all_after(
    root: &mut MeshRoot,
    ranks: &mut [&mut dyn LaneMeshExecutor],
    error: ProtocolError,
) -> ProtocolError {
    if let ProtocolError::Aborted(record) = &error {
        if let Some(k) = record.key.clone() {
            for r in ranks.iter_mut() {
                r.abort_step(&k, record.phase, &record.reason);
            }
        }
        return error;
    }
    let reason = error.to_string();
    abort_all(root, ranks, &reason)
}

// ---------------------------------------------------------------------------
// Scripted executor (CPU tests; reusable by S3/S5 tests)
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod scripted {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// What the model "does" for one lane window.
    #[derive(Clone, Debug, Default)]
    pub struct WindowScript {
        /// Head proposals in draft order (native source).
        pub drafts: Vec<u32>,
        /// Per-row target picks (`rows` long).
        pub picks: Vec<u32>,
        /// Head's `keep_going` per draft step (default `true`).
        pub keep_going: Vec<bool>,
        /// Head step `j` drafted a token that the singleton rule drops
        /// (default `false`).
        pub drop_last: Vec<bool>,
    }

    #[derive(Clone, Debug, Default)]
    pub struct Script {
        /// Keyed by `(request_tag, window_seq)`.
        pub windows: HashMap<(u64, u64), WindowScript>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum FailAt {
        Begin,
        Draft(u32),
        Verify,
        Prepare,
    }

    /// One logged call, comparable across ranks.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum Call {
        Begin(u64),
        Draft(u32, Vec<(LaneWindowKey, usize, u32)>),
        Verify(u64, Vec<Vec<u32>>, Vec<RowGroup>),
        Prepare(u64),
        Publish(u64),
        Abort(String),
    }

    pub struct ScriptedExecutor {
        pub session: RankSession,
        pub script: Rc<RefCell<Script>>,
        pub calls: Vec<Call>,
        pub fail_at: Option<FailAt>,
        /// Differing local timing per rank (µs added to every phase).
        pub skew_us: f32,
        /// A misbehaving follower attempts to return proposals/picks.
        pub rogue_sampler_output: bool,
        pub counts: HashMap<MeshStepKey, OpCounts>,
        pub aborts: Vec<String>,
        pub published: Vec<u64>,
        pub capacity_error: Option<String>,
        draft_idx: HashMap<LaneWindowKey, usize>,
        pub evidence_unpriced: bool,
    }

    impl ScriptedExecutor {
        pub fn new(own: RankOwnership, script: Rc<RefCell<Script>>) -> Self {
            Self {
                session: RankSession::new(own),
                script,
                calls: Vec::new(),
                fail_at: None,
                skew_us: 0.0,
                rogue_sampler_output: false,
                counts: HashMap::new(),
                aborts: Vec::new(),
                published: Vec::new(),
                capacity_error: None,
                draft_idx: HashMap::new(),
                evidence_unpriced: false,
            }
        }

        fn window(&self, k: &LaneWindowKey) -> WindowScript {
            self.script
                .borrow()
                .windows
                .get(&(k.request_epoch.request_tag, k.window_seq))
                .cloned()
                .unwrap_or_default()
        }

        fn rank(&self) -> usize {
            self.session.ownership().rank
        }
    }

    impl LaneMeshExecutor for ScriptedExecutor {
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
            self.calls.push(Call::Begin(start.key.step_seq));
            self.draft_idx.clear();
            let cap = if self.fail_at == Some(FailAt::Begin) {
                Err("scripted capacity failure".to_string())
            } else {
                self.capacity_error.clone().map_or(Ok(()), Err)
            };
            self.session.begin(start, cap)
        }

        fn reject_window(&mut self, key: &MeshStepKey) {
            self.session.reject(key);
        }

        fn run_draft_round(&mut self, round: &DraftRound) -> Result<RankDraftResult, RankError> {
            self.calls.push(Call::Draft(
                round.round,
                round
                    .ops
                    .iter()
                    .map(|o| (o.key, o.position, o.input))
                    .collect(),
            ));
            if self.fail_at == Some(FailAt::Draft(round.round)) {
                return Err(RankError::new(self.rank(), "scripted head failure"));
            }
            let mut res = self.session.on_draft_round(round)?;
            let own = self.ownership();
            if own.sampler || self.rogue_sampler_output {
                let mut props = Vec::new();
                for op in &round.ops {
                    let w = self.window(&op.key);
                    let j = *self
                        .draft_idx
                        .entry(op.key)
                        .and_modify(|j| *j += 1)
                        .or_insert(0);
                    let token = *w
                        .drafts
                        .get(j)
                        .ok_or_else(|| RankError::new(self.rank(), "script exhausted"))?;
                    props.push(DraftProposal {
                        key: op.key,
                        token,
                        keep_going: w.keep_going.get(j).copied().unwrap_or(true),
                        drop_last: w.drop_last.get(j).copied().unwrap_or(false),
                        read: None,
                    });
                }
                res.proposals = Some(props);
            }
            let key = round.key.clone();
            let n = round.ops.len();
            let e = self.counts.entry(key).or_default();
            e.head_draft_calls += n;
            Ok(res)
        }

        fn run_verify(&mut self, plan: &VerifyPlan) -> Result<RankVerifyResult, RankError> {
            self.calls.push(Call::Verify(
                plan.key.step_seq,
                plan.lanes.iter().map(|l| l.rows.clone()).collect(),
                plan.groups.clone(),
            ));
            if self.fail_at == Some(FailAt::Verify) {
                return Err(RankError::new(
                    self.rank(),
                    "scripted target/gather failure",
                ));
            }
            let mut res = self.session.on_verify(plan)?;
            let own = self.ownership();
            if own.sampler || self.rogue_sampler_output {
                let mut picks = Vec::new();
                for l in &plan.lanes {
                    let w = self.window(&l.key);
                    if w.picks.len() < l.rows.len() {
                        return Err(RankError::new(self.rank(), "script has too few picks"));
                    }
                    picks.push((l.key, w.picks[..l.rows.len()].to_vec()));
                }
                res.sampler = Some(SamplerOutput::picks_only(picks));
            }
            let c = self.counts.entry(plan.key.clone()).or_default();
            c.row_groups = plan.ledger.row_groups;
            c.target_exchanges = plan.ledger.target_exchanges;
            if own.kind == MeshKind::Pp {
                // PP2: one boundary; F32 streams of width 10240.
                c.edge_count = plan.ledger.microbatches;
                c.edge_bytes = plan.total_rows() as u64 * 10240 * 4;
            }
            Ok(res)
        }

        fn prepare_commit(
            &mut self,
            decision: &CommitDecision,
        ) -> Result<PreparedReceipt, RankError> {
            self.calls.push(Call::Prepare(decision.key.step_seq));
            if self.fail_at == Some(FailAt::Prepare) {
                return Err(RankError::new(self.rank(), "scripted repair failure"));
            }
            let r = self.session.on_prepare(decision)?;
            let c = self.counts.entry(decision.key.clone()).or_default();
            c.head_append_rows = r.head_append_rows;
            c.drafts_generated = decision.lanes.iter().map(|l| l.source.offered).sum();
            c.accepted = decision.lanes.iter().map(|l| l.source.accepted).sum();
            c.consumed_rows = decision.lanes.iter().map(|l| l.consumed_rows).sum();
            Ok(r)
        }

        fn publish_commit(&mut self, decision: &CommitDecision) {
            self.calls.push(Call::Publish(decision.key.step_seq));
            self.published.push(decision.key.step_seq);
            self.session.on_publish(decision);
        }

        fn abort_step(&mut self, _key: &MeshStepKey, _phase: StepPhase, reason: &str) {
            if !self.session.is_quarantined() {
                self.calls.push(Call::Abort(reason.to_string()));
            }
            self.aborts.push(reason.to_string());
            self.session.on_abort();
        }

        fn evidence(&mut self, key: &MeshStepKey) -> Option<RankCostEvidence> {
            let counts = *self.counts.get(key)?;
            let skew = self.skew_us;
            Some(RankCostEvidence {
                key: key.clone(),
                rank: self.rank(),
                completed: StepPhase::Published,
                phases: vec![PhaseTiming {
                    phase: TimedPhase::Verify,
                    us: 1000.0 + skew,
                    synchronized: true,
                }],
                nodes: Vec::new(),
                counts,
                unpriced: self.evidence_unpriced,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (DESIGN §10.1 bullets 1-4, 5 (host half), 7)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::scripted::*;
    use super::*;
    use crate::gpu_forward_lanes::row_group_count;
    use hipfire_runtime::spec_sampling::SpecRng;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    fn world(kind: MeshKind) -> Vec<RankOwnership> {
        match kind {
            MeshKind::Single => vec![RankOwnership::single()],
            MeshKind::Ep => vec![RankOwnership::ep(0), RankOwnership::ep(1)],
            // The output band (root) is rank 1 on purpose: device/layer
            // order is not identity.
            MeshKind::Pp => vec![RankOwnership::pp(0, false), RankOwnership::pp(1, true)],
        }
    }

    fn budgets(lab: LabRoutes, k: usize) -> WindowBudgets {
        WindowBudgets {
            max_lanes: 8,
            verify_rows_per_lane: k + 1,
            max_rows: 8 * (k + 1),
            copy_take_cap: 7,
            lab,
            pp_microbatch_lanes: None,
        }
    }

    fn req(tag: u64, remaining: usize) -> LaneRequest {
        LaneRequest {
            epoch: epoch(tag),
            remaining,
            context_left: 10_000,
            copy_candidates: Vec::new(),
            forced: None,
            sampling: SamplingMode::Greedy,
        }
    }

    fn interval() -> ControllerInterval {
        ControllerInterval {
            wall_us: 50_000.0,
            prefill_ran: false,
        }
    }

    struct Mesh {
        root: MeshRoot,
        ranks: Vec<ScriptedExecutor>,
        script: Rc<RefCell<Script>>,
        start: Option<WindowStart>,
        verify: Option<VerifyPlan>,
        decision: Option<CommitDecision>,
        ready: Vec<RankReady>,
        verify_results: Vec<RankVerifyResult>,
        receipts: Vec<PreparedReceipt>,
    }

    impl Mesh {
        fn new(kind: MeshKind, floor: bool) -> Self {
            let w = world(kind);
            let script = Rc::new(RefCell::new(Script::default()));
            let own = *w.iter().find(|r| r.root).unwrap();
            let root = MeshRoot::new(
                own,
                &w,
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
            let ranks = w
                .iter()
                .map(|&o| ScriptedExecutor::new(o, script.clone()))
                .collect();
            Self {
                root,
                ranks,
                script,
                start: None,
                verify: None,
                decision: None,
                ready: Vec::new(),
                verify_results: Vec::new(),
                receipts: Vec::new(),
            }
        }

        fn admit(
            &mut self,
            tag: u64,
            position: usize,
            seed: u32,
            eos: Option<u32>,
            spec_k: Option<usize>,
        ) -> RequestEpoch {
            let e = epoch(tag);
            let a = self
                .root
                .admit_lane(LaneAdmission {
                    epoch: e,
                    position,
                    seed,
                    eos,
                    rng_seed: 1000 + tag,
                    prompt: vec![1, 2, 3, tag as u32],
                    spec_k,
                    head: HeadState::Aligned,
                })
                .unwrap();
            for r in &mut self.ranks {
                r.admit_lane(&a).unwrap();
            }
            e
        }

        fn script_window(&self, tag: u64, seq: u64, drafts: &[u32], picks: &[u32]) {
            self.script.borrow_mut().windows.insert(
                (tag, seq),
                WindowScript {
                    drafts: drafts.to_vec(),
                    picks: picks.to_vec(),
                    keep_going: Vec::new(),
                    drop_last: Vec::new(),
                },
            );
        }

        fn run(
            &mut self,
            prior: Option<&MergedCostEvidence>,
            reqs: &[LaneRequest],
            b: &WindowBudgets,
            rule: &mut dyn AcceptRule,
        ) -> Result<StepOutput, ProtocolError> {
            let mut refs: Vec<&mut dyn LaneMeshExecutor> = self
                .ranks
                .iter_mut()
                .map(|r| r as &mut dyn LaneMeshExecutor)
                .collect();
            run_mesh_step(&mut self.root, &mut refs, prior, reqs, b, rule)
        }

        fn run_greedy(
            &mut self,
            reqs: &[LaneRequest],
            b: &WindowBudgets,
        ) -> Result<StepOutput, ProtocolError> {
            self.run(None, reqs, b, &mut GreedyRule)
        }

        // ---- manual stepping (success paths; tests tamper between phases)

        fn s_plan(
            &mut self,
            prior: Option<&MergedCostEvidence>,
            reqs: &[LaneRequest],
            b: &WindowBudgets,
        ) -> Result<WindowStart, ProtocolError> {
            let s = self.root.plan_window(prior, reqs, b)?;
            self.start = Some(s.clone());
            Ok(s)
        }

        fn s_ready(&mut self) -> Result<(), ProtocolError> {
            let start = self.start.clone().unwrap();
            self.ready = self
                .ranks
                .iter_mut()
                .map(|r| r.begin_window(&start))
                .collect();
            self.root.collect_ready(&self.ready.clone())
        }

        fn s_draft(&mut self) -> Result<(), ProtocolError> {
            while let Some(round) = self.root.next_draft_round()? {
                let mut res = Vec::new();
                for r in self.ranks.iter_mut().filter(|r| r.ownership().head) {
                    res.push(r.run_draft_round(&round).expect("scripted draft"));
                }
                self.root.collect_draft(&round, &res)?;
            }
            Ok(())
        }

        fn s_verify(&mut self) -> Result<VerifyPlan, ProtocolError> {
            let plan = self.root.seal_verify()?;
            self.verify_results = self
                .ranks
                .iter_mut()
                .filter(|r| r.ownership().target)
                .map(|r| r.run_verify(&plan).expect("scripted verify"))
                .collect();
            self.root.collect_verify(&self.verify_results.clone())?;
            self.verify = Some(plan.clone());
            Ok(plan)
        }

        fn s_decide(&mut self, rule: &mut dyn AcceptRule) -> Result<CommitDecision, ProtocolError> {
            let d = self.root.decide_commit(rule)?;
            self.decision = Some(d.clone());
            Ok(d)
        }

        fn s_prepare(&mut self) -> Result<(), ProtocolError> {
            let d = self.decision.clone().unwrap();
            self.receipts = self
                .ranks
                .iter_mut()
                .map(|r| r.prepare_commit(&d).expect("scripted prepare"))
                .collect();
            self.root.collect_prepared(&self.receipts.clone())
        }

        fn s_publish(&mut self) -> StepOutput {
            let out = self.root.publish_commit().unwrap();
            let d = self.decision.clone().unwrap();
            for r in &mut self.ranks {
                r.publish_commit(&d);
            }
            out
        }

        /// Everything up to and including `Prepared`.
        fn s_through_prepared(
            &mut self,
            reqs: &[LaneRequest],
            b: &WindowBudgets,
            rule: &mut dyn AcceptRule,
        ) {
            self.s_plan(None, reqs, b).unwrap();
            self.s_ready().unwrap();
            self.s_draft().unwrap();
            self.s_verify().unwrap();
            self.s_decide(rule).unwrap();
            self.s_prepare().unwrap();
        }

        fn evidence(&mut self, key: &MeshStepKey) -> Vec<RankCostEvidence> {
            self.ranks
                .iter_mut()
                .filter_map(|r| r.evidence(key))
                .collect()
        }
    }

    /// Six-window script of one lane: all drafts `10..17`, picks match the
    /// drafts up to `matched`, then diverge.
    fn script_matching(m: &Mesh, tag: u64, seq: u64, matched: usize) {
        let drafts: Vec<u32> = (10..17).collect();
        let picks: Vec<u32> = (0..8)
            .map(|i| {
                if i < matched {
                    10 + i as u32
                } else {
                    900 + i as u32
                }
            })
            .collect();
        m.script_window(tag, seq, &drafts, &picks);
    }

    fn lane_out(out: &StepOutput, tag: u64) -> &LaneOutput {
        out.lanes
            .iter()
            .find(|l| l.key.request_epoch.request_tag == tag)
            .unwrap()
    }

    /// A rule that draws from the staged request RNG (test stand-in for the
    /// S4 sampled rules): accept each draft on a 3/4 draw, one more draw for
    /// the bonus.
    struct DrawRule;

    impl AcceptRule for DrawRule {
        fn accept(
            &mut self,
            _lane: &LanePlan,
            drafts: &[u32],
            picks: &[u32],
            rng: &mut LaneRng,
        ) -> Result<GreedyAccept, String> {
            let mut committed = Vec::new();
            let mut accepted = 0;
            for &d in drafts {
                if rng.next_u64().is_multiple_of(4) {
                    break;
                }
                committed.push(d);
                accepted += 1;
            }
            let _ = rng.next_u64();
            committed.push(picks[accepted]);
            Ok(GreedyAccept {
                committed,
                accepted,
                hit_eos: false,
            })
        }
    }

    const BOTH: [MeshKind; 2] = [MeshKind::Ep, MeshKind::Pp];

    // ---- consumed_rows ----------------------------------------------------

    #[test]
    fn mesh_lane_protocol_consumed_rows_zero_partial_full_acceptance() {
        for kind in BOTH {
            let mut m = Mesh::new(kind, false);
            for tag in 1..=3 {
                m.admit(tag, 100 + tag as usize, 7, None, Some(3));
            }
            // lane 1: full (3 of 3), lane 2: partial (1), lane 3: zero.
            script_matching(&m, 1, 0, 8);
            script_matching(&m, 2, 0, 1);
            script_matching(&m, 3, 0, 0);
            let out = m
                .run_greedy(
                    &[req(1, 99), req(2, 99), req(3, 99)],
                    &budgets(LabRoutes::Batched, 3),
                )
                .unwrap();
            // Full: bonus is the 4th pick, rows consumed == r_i == 4.
            let l1 = lane_out(&out, 1);
            assert_eq!(l1.emitted, vec![10, 11, 12, 13]);
            assert_eq!(
                l1.consumed_rows, 4,
                "{kind:?} full acceptance consumes every row"
            );
            assert_eq!(l1.next_seed, *l1.emitted.last().unwrap());
            // Partial: j=1 accepted -> 1+1 rows; the correction is not consumed.
            let l2 = lane_out(&out, 2);
            assert_eq!(l2.emitted, vec![10, 901]);
            assert_eq!(l2.consumed_rows, 2);
            assert_eq!(l2.source.accepted, 1);
            // Zero: only the seed row.
            let l3 = lane_out(&out, 3);
            assert_eq!(l3.emitted, vec![900]);
            assert_eq!(l3.consumed_rows, 1);
            assert_eq!(l3.source.offered, 3);
            for (tag, consumed, next) in [(1u64, 4usize, 13u32), (2, 2, 901), (3, 1, 900)] {
                let (pos, seed) = m.root.lane_position(epoch(tag)).unwrap();
                assert_eq!(pos, 100 + tag as usize + consumed, "{kind:?} lane {tag}");
                assert_eq!(seed, next);
            }
        }
    }

    #[test]
    fn mesh_lane_protocol_consumed_rows_eos_accepted_and_bonus() {
        for kind in BOTH {
            let mut m = Mesh::new(kind, false);
            m.admit(1, 50, 7, Some(11), Some(3));
            m.admit(2, 60, 7, Some(903), Some(3));
            // lane 1: drafts [10, 11(EOS), 12] all match: the accepted EOS
            // draft stops the window and stays the pending seed.
            script_matching(&m, 1, 0, 8);
            // lane 2: all three accepted and the *bonus* is the EOS.
            script_matching(&m, 2, 0, 3);
            let out = m
                .run_greedy(&[req(1, 99), req(2, 99)], &budgets(LabRoutes::Batched, 3))
                .unwrap();
            let l1 = lane_out(&out, 1);
            assert_eq!(l1.emitted, vec![10, 11]);
            assert!(l1.terminal);
            // 1 + (accepted(2) - 1): the EOS row is not consumed.
            assert_eq!(l1.consumed_rows, 2);
            assert_eq!(l1.next_seed, 11);
            let l2 = lane_out(&out, 2);
            assert_eq!(l2.emitted, vec![10, 11, 12, 903]);
            assert!(l2.terminal);
            // Bonus EOS is predicted, not consumed: 1 + accepted.
            assert_eq!(l2.consumed_rows, 4);
            // Terminal lanes cannot be planned again.
            let again = m
                .root
                .plan_window(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3));
            assert_eq!(again.unwrap_err(), ProtocolError::LaneNotLive(epoch(1)));
        }
    }

    #[test]
    fn mesh_lane_protocol_length_termination_and_budget_clamp() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        // Two tokens left: at most one draft; the window ends the request.
        let start = m
            .root
            .plan_window(None, &[req(1, 2)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        assert_eq!(start.lanes[0].max_depth, 1);
        assert_eq!(start.lanes[0].max_rows(), 2);
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        let out = m
            .run_greedy(&[req(1, 2)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        let l = lane_out(&out, 1);
        assert_eq!(l.emitted, vec![10, 11]);
        assert!(
            l.terminal,
            "emission reached the request's remaining tokens"
        );
        // Context bound clamps drafts too.
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        let mut r = req(1, 99);
        r.context_left = 3;
        let start = m
            .root
            .plan_window(None, &[r], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        assert_eq!(start.lanes[0].max_depth, 2);
    }

    #[test]
    fn mesh_lane_protocol_forced_lane_consumes_rows_without_rng_or_policy() {
        for kind in BOTH {
            let mut m = Mesh::new(kind, false);
            m.admit(1, 20, 7, None, Some(3));
            m.admit(2, 30, 7, None, Some(3));
            script_matching(&m, 1, 0, 0);
            script_matching(&m, 2, 0, 2);
            let before = m.root.rng_probe(epoch(1)).unwrap();
            let mut forced = req(1, 99);
            forced.forced = Some(vec![41, 42, 43]);
            let out = m
                .run(
                    None,
                    &[forced, req(2, 99)],
                    &budgets(LabRoutes::Batched, 3),
                    &mut DrawRule,
                )
                .unwrap();
            let f = lane_out(&out, 1);
            assert!(f.forced);
            assert_eq!(f.emitted, vec![41, 42, 43]);
            // Rows [seed, 41, 42] consumed; 43 becomes the pending seed.
            assert_eq!(f.consumed_rows, 3);
            assert_eq!(f.next_seed, 43);
            assert_eq!(f.source.offered, 0);
            assert_eq!(m.root.lane_position(epoch(1)).unwrap(), (23, 43));
            assert_eq!(
                m.root.rng_probe(epoch(1)).unwrap(),
                before,
                "a forced token is not a proposal: no RNG draw"
            );
            // The sampled neighbour did draw.
            assert_ne!(m.root.rng_probe(epoch(2)).unwrap(), {
                let mut r = SpecRng::new(1002);
                r.next_u64()
            });
        }
    }

    // ---- ordered calls / row groups (bullet 2) ------------------------------

    #[test]
    fn mesh_lane_protocol_row_groups_match_gpu_packing_and_design_counts() {
        for rows in [
            vec![1usize; 8],
            vec![3; 8],
            vec![4; 8],
            vec![2, 8, 1, 7, 8, 1],
            vec![5, 4, 4, 5],
            vec![8],
        ] {
            assert_eq!(
                pack_row_groups(&rows, LANE_MAX_ROWS).len(),
                row_group_count(&rows),
                "{rows:?}"
            );
        }
        // C8 uniform d = 1,2,3,4,7: G = 2,4,4,8,8 (DESIGN §2.4).
        for (d, g) in [(1usize, 2usize), (2, 4), (3, 4), (4, 8), (7, 8)] {
            assert_eq!(
                pack_row_groups(&[d + 1; 8], LANE_MAX_ROWS).len(),
                g,
                "d={d}"
            );
        }
    }

    #[test]
    fn mesh_lane_protocol_c8_exchange_ledger_matches_design_totals() {
        // moe=once: 48G target + Σ head Draft invocations
        // = 104, 208, 216, 416, 440 for d = 1,2,3,4,7 (DESIGN §2.4).
        for (d, total) in [(1usize, 104usize), (2, 208), (3, 216), (4, 416), (7, 440)] {
            let mut m = Mesh::new(MeshKind::Ep, false);
            let reqs: Vec<LaneRequest> = (1..=8).map(|t| req(t, 1000)).collect();
            for t in 1..=8u64 {
                m.admit(t, 100, 7, None, Some(d));
                script_matching(&m, t, 0, 8);
            }
            let out = m
                .run_greedy(&reqs, &budgets(LabRoutes::Batched, d))
                .unwrap();
            assert_eq!(out.lanes.len(), 8);
            let verify = m.ranks[0]
                .calls
                .iter()
                .find_map(|c| match c {
                    Call::Verify(_, rows, groups) => Some((rows.clone(), groups.clone())),
                    _ => None,
                })
                .unwrap();
            assert!(verify.0.iter().all(|r| r.len() == d + 1));
            let key = out.key.clone();
            let ev = m.evidence(&key);
            let c = ev[0].counts;
            assert_eq!(c.target_exchanges + c.head_draft_calls, total, "d={d}");
            assert_eq!(c.head_draft_calls, 8 * d);
            assert_eq!(c.row_groups, verify.1.len());
        }
    }

    #[test]
    fn mesh_lane_protocol_ep_calls_identical_across_ranks_and_timings() {
        // Cycle routes (`Batched{K}`, `Batched{1}`, `ArSynced`, `Batched{2}`)
        // with EOS truncation, `keep_going` pruning, mixed lane acceptance.
        fn run(skew: f32) -> (Vec<Call>, Vec<Call>, Vec<Vec<u32>>) {
            let mut m = Mesh::new(MeshKind::Ep, false);
            m.ranks[1].skew_us = skew;
            m.ranks[0].skew_us = -skew * 0.5;
            for t in 1..=3u64 {
                m.admit(t, 40, 7, if t == 2 { Some(12) } else { None }, Some(3));
            }
            let mut emitted = Vec::new();
            for seq in 0..6u64 {
                for t in 1..=3u64 {
                    script_matching(&m, t, seq, ((seq * 3 + t) % 5) as usize);
                }
                if seq == 1 {
                    m.script
                        .borrow_mut()
                        .windows
                        .get_mut(&(3, seq))
                        .unwrap()
                        .keep_going = vec![false];
                }
                let reqs: Vec<LaneRequest> = (1..=3)
                    .filter(|t| m.root.lane_position(epoch(*t)).is_some())
                    .map(|t| req(t, 1000))
                    .collect();
                if let Ok(out) = m.run_greedy(&reqs, &budgets(LabRoutes::Cycle, 3)) {
                    for l in &out.lanes {
                        emitted.push(l.emitted.clone());
                    }
                    for l in out.lanes.iter().filter(|l| l.terminal) {
                        m.root.release_lane(l.key.request_epoch).unwrap();
                        for r in &mut m.ranks {
                            r.release_lane(&l.key.request_epoch).unwrap();
                        }
                    }
                }
            }
            (m.ranks[0].calls.clone(), m.ranks[1].calls.clone(), emitted)
        }
        let (a0, a1, ae) = run(0.0);
        assert_eq!(
            a0, a1,
            "every EP rank executes the identical ordered call list"
        );
        assert!(a0.iter().any(|c| matches!(c, Call::Draft(..))));
        // Different local timing changes nothing the ranks are asked to do.
        let (b0, b1, be) = run(7_000.0);
        assert_eq!(a0, b0);
        assert_eq!(a1, b1);
        assert_eq!(ae, be);
    }

    #[test]
    fn mesh_lane_protocol_draft_pruning_is_root_decided_and_replayed() {
        for kind in BOTH {
            let mut m = Mesh::new(kind, false);
            m.admit(1, 10, 7, None, Some(3));
            m.admit(2, 10, 7, Some(11), Some(3));
            m.admit(3, 10, 7, None, Some(3));
            for t in 1..=3 {
                script_matching(&m, t, 0, 8);
            }
            // lane 1: the head hints stop after the first draft; lane 2
            // drafts the EOS second; lane 3 runs to depth.
            m.script
                .borrow_mut()
                .windows
                .get_mut(&(1, 0))
                .unwrap()
                .keep_going = vec![false];
            let out = m
                .run_greedy(
                    &[req(1, 99), req(2, 99), req(3, 99)],
                    &budgets(LabRoutes::Batched, 3),
                )
                .unwrap();
            assert_eq!(lane_out(&out, 1).source.offered, 1);
            assert_eq!(lane_out(&out, 2).source.offered, 2);
            assert_eq!(lane_out(&out, 3).source.offered, 3);
            // Rounds shrink: [3 ops], [2 ops], [1 op].
            let head = m
                .ranks
                .iter()
                .find(|r| r.ownership().head && r.ownership().root)
                .unwrap();
            let sizes: Vec<usize> = head
                .calls
                .iter()
                .filter_map(|c| match c {
                    Call::Draft(_, ops) => Some(ops.len()),
                    _ => None,
                })
                .collect();
            assert_eq!(sizes, vec![3, 2, 1]);
            if kind == MeshKind::Pp {
                // The input band never sees head work.
                assert!(m.ranks[0]
                    .calls
                    .iter()
                    .all(|c| !matches!(c, Call::Draft(..))));
            } else {
                assert_eq!(m.ranks[0].calls, m.ranks[1].calls);
            }
        }
    }

    #[test]
    fn mesh_lane_protocol_dropped_draft_runs_its_head_step_but_is_not_kept() {
        for kind in BOTH {
            let mut m = Mesh::new(kind, false);
            m.admit(1, 10, 7, None, Some(3));
            script_matching(&m, 1, 0, 8);
            // Head step 2 drafts a token the singleton rule drops.
            m.script
                .borrow_mut()
                .windows
                .get_mut(&(1, 0))
                .unwrap()
                .drop_last = vec![false, false, true];
            let out = m
                .run_greedy(&[req(1, 99)], &budgets(LabRoutes::Batched, 3))
                .unwrap();
            let l = lane_out(&out, 1);
            assert_eq!(l.source.offered, 2, "{kind:?}: two kept drafts");
            let ev = m.evidence(&out.key);
            let head = ev.iter().find(|e| e.counts.head_draft_calls > 0).unwrap();
            assert_eq!(
                head.counts.head_draft_calls, 3,
                "the dropped step still ran"
            );
            // A drop on a lane's first draft is a protocol violation.
            let mut m = Mesh::new(kind, false);
            m.admit(1, 10, 7, None, Some(3));
            script_matching(&m, 1, 0, 8);
            m.script
                .borrow_mut()
                .windows
                .get_mut(&(1, 0))
                .unwrap()
                .drop_last = vec![true];
            let err = m
                .run_greedy(&[req(1, 99)], &budgets(LabRoutes::Batched, 3))
                .unwrap_err();
            assert!(
                matches!(&err, ProtocolError::Aborted(r) if r.reason.contains("cannot be dropped")),
                "{err:?}"
            );
        }
    }

    #[test]
    fn mesh_lane_protocol_pp_microbatches_sum_row_groups_per_microbatch() {
        let mut m = Mesh::new(MeshKind::Pp, false);
        for t in 1..=3u64 {
            m.admit(t, 10, 7, None, Some(3));
            script_matching(&m, t, 0, 8);
        }
        let mut b = budgets(LabRoutes::Batched, 3);
        b.pp_microbatch_lanes = Some(1);
        let out = m
            .run_greedy(&[req(1, 99), req(2, 99), req(3, 99)], &b)
            .unwrap();
        let plan_groups = m.ranks[1]
            .calls
            .iter()
            .find_map(|c| match c {
                Call::Verify(_, _, g) => Some(g.clone()),
                _ => None,
            })
            .unwrap();
        // Unsplit [4,4,4] would be 2 groups; three microbatches are 3.
        assert_eq!(pack_row_groups(&[4, 4, 4], LANE_MAX_ROWS).len(), 2);
        assert_eq!(plan_groups.len(), 3);
        let ev = m.evidence(&out.key);
        assert!(
            ev.iter().all(|e| e.counts.target_exchanges == 0),
            "PP has zero EP gathers"
        );
        assert!(
            ev.iter().all(|e| e.counts.edge_count == 3),
            "H edges across the one boundary"
        );
        // The head ran on the output band alone.
        assert_eq!(ev[0].counts.head_draft_calls, 0);
        assert_eq!(ev[1].counts.head_draft_calls, 9);
    }

    // ---- evidence / policy independence (bullet 1) ----------------------------

    #[test]
    fn mesh_lane_protocol_plan_independent_of_rank_local_timing() {
        fn two_steps(skew: f32, reversed: bool) -> (u64, (f64, f64, f64, f64), f64) {
            let mut m = Mesh::new(MeshKind::Ep, true);
            m.ranks[1].skew_us = skew;
            m.admit(1, 10, 7, None, Some(3));
            m.admit(2, 10, 7, None, Some(3));
            for t in 1..=2 {
                script_matching(&m, t, 0, 8);
                script_matching(&m, t, 1, 8);
            }
            let b = budgets(LabRoutes::Floor, 3);
            let out = m.run_greedy(&[req(1, 99), req(2, 99)], &b).unwrap();
            let mut ev = m.evidence(&out.key);
            if reversed {
                ev.reverse();
            }
            let merged = m.root.merge_evidence(&ev, interval()).unwrap();
            assert!(merged.priced, "{:?}", merged.unpriced_reason);
            let critical = merged.critical_path_us;
            let r = [req(1, 99), req(2, 99)];
            let next = m.s_plan(Some(&merged), &r, &b).unwrap();
            m.s_ready().unwrap();
            m.s_draft().unwrap();
            m.s_verify().unwrap();
            m.s_decide(&mut GreedyRule).unwrap();
            m.s_prepare().unwrap();
            m.s_publish();
            (next.digest, m.root.cost_fit(), critical)
        }
        let (d0, fit0, c0) = two_steps(0.0, false);
        let (d1, fit1, c1) = two_steps(9_000.0, true);
        assert_eq!(d0, d1, "one root plan whatever the ranks' local clocks");
        assert_eq!(
            fit0, fit1,
            "the fit trains on the controller interval, not rank clocks"
        );
        assert_eq!(c0, 1000.0);
        assert_eq!(
            c1, 10_000.0,
            "EP critical path is the slowest rank, not the sum"
        );
        // The merged price moved the fit away from the prior.
        let fresh = Mesh::new(MeshKind::Ep, true);
        assert_ne!(fresh.root.cost_fit(), fit0);
    }

    #[test]
    fn mesh_lane_protocol_stale_missing_incoherent_evidence_cannot_update_policy() {
        let mut m = Mesh::new(MeshKind::Ep, true);
        m.admit(1, 10, 7, None, Some(3));
        for s in 0..3 {
            script_matching(&m, 1, s, 8);
        }
        let b = budgets(LabRoutes::Floor, 3);
        let out1 = m.run_greedy(&[req(1, 99)], &b).unwrap();
        let ev1 = m.evidence(&out1.key);

        // Missing rank.
        let missing = m.root.merge_evidence(&ev1[..1], interval()).unwrap_err();
        assert!(matches!(missing, EvidenceError::Missing(_)), "{missing:?}");
        // Duplicate rank.
        let dup = vec![ev1[0].clone(), ev1[0].clone()];
        assert!(matches!(
            m.root.merge_evidence(&dup, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        // Incoherent counts: a rank crossed one extra exchange.
        let mut bad = ev1.clone();
        bad[1].counts.target_exchanges += 1;
        assert!(matches!(
            m.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        let mut bad = ev1.clone();
        bad[0].counts.row_groups += 1;
        assert!(matches!(
            m.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        // Counts that disagree with the decision.
        let mut bad = ev1.clone();
        bad[0].counts.consumed_rows += 1;
        assert!(matches!(
            m.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        // Stale key (other topology generation).
        let mut stale = ev1.clone();
        for e in &mut stale {
            e.key.topology_generation += 1;
        }
        assert!(matches!(
            m.root.merge_evidence(&stale, interval()).unwrap_err(),
            EvidenceError::Stale(_)
        ));

        // Good merge of step 1, then step 2 published: step-1 receipts are stale.
        let merged1 = m.root.merge_evidence(&ev1, interval()).unwrap();
        m.run(Some(&merged1), &[req(1, 99)], &b, &mut GreedyRule)
            .unwrap();
        assert_eq!(m.root.counters().observations_applied, 1);
        let stale_merge = m.root.merge_evidence(&ev1, interval()).unwrap_err();
        assert!(
            matches!(stale_merge, EvidenceError::Stale(_)),
            "{stale_merge:?}"
        );
        // A stale merged object is refused by plan_window and changes nothing.
        let fit = m.root.cost_fit();
        let before = m.root.counters();
        let err = m
            .root
            .plan_window(Some(&merged1), &[req(1, 99)], &b)
            .unwrap_err();
        assert!(matches!(err, ProtocolError::StaleEvidence(_)), "{err:?}");
        assert_eq!(m.root.cost_fit(), fit);
        assert_eq!(m.root.counters(), before);
        assert_eq!(
            m.root.phase(),
            StepPhase::Published,
            "refused plan leaves the root startable"
        );

        // Missing evidence: the price is dropped; the fit is untouched.
        let fit_pre = m.root.cost_fit();
        m.run(None, &[req(1, 99)], &b, &mut GreedyRule).unwrap();
        assert_eq!(m.root.counters().observations_dropped, 1);
        assert_eq!(m.root.cost_fit(), fit_pre);
    }

    #[test]
    fn mesh_lane_protocol_unpriced_evidence_spends_budget_without_training_fit() {
        let mut m = Mesh::new(MeshKind::Ep, true);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        script_matching(&m, 1, 1, 8);
        let b = budgets(LabRoutes::Floor, 3);
        let out = m.run_greedy(&[req(1, 99)], &b).unwrap();
        let fit = m.root.cost_fit();
        for (case, iv, warm) in [
            (
                "prefill",
                ControllerInterval {
                    wall_us: 5e4,
                    prefill_ran: true,
                },
                false,
            ),
            (
                "nan wall",
                ControllerInterval {
                    wall_us: f64::NAN,
                    prefill_ran: false,
                },
                false,
            ),
            ("warm bind", interval(), true),
        ] {
            let mut ev = m.evidence(&out.key);
            ev[0].unpriced = warm;
            let merged = m.root.merge_evidence(&ev, iv).unwrap();
            assert!(!merged.priced, "{case}");
            assert!(merged.unpriced_reason.is_some());
        }
        // An incomplete rank is unpriced too.
        let mut ev = m.evidence(&out.key);
        ev[1].completed = StepPhase::Verified;
        let merged = m.root.merge_evidence(&ev, interval()).unwrap();
        assert!(!merged.priced);
        let out2 = m
            .run(Some(&merged), &[req(1, 99)], &b, &mut GreedyRule)
            .unwrap();
        assert_eq!(
            m.root.cost_fit(),
            fit,
            "an unpriced window cannot train the fit"
        );
        assert_eq!(m.root.counters().observations_applied, 1);
        // Contrast: a priced merge of the next step does move the fit.
        script_matching(&m, 1, 2, 8);
        let ev2 = m.evidence(&out2.key);
        let merged2 = m.root.merge_evidence(&ev2, interval()).unwrap();
        assert!(merged2.priced);
        m.run(Some(&merged2), &[req(1, 99)], &b, &mut GreedyRule)
            .unwrap();
        assert_ne!(m.root.cost_fit(), fit);
    }

    #[test]
    fn mesh_lane_protocol_pp_wavefront_critical_path_and_edge_coherence() {
        let mut m = Mesh::new(MeshKind::Pp, false);
        for t in 1..=2u64 {
            m.admit(t, 10, 7, None, Some(3));
            script_matching(&m, t, 0, 8);
        }
        let mut b = budgets(LabRoutes::Batched, 3);
        b.pp_microbatch_lanes = Some(1);
        let out = m.run_greedy(&[req(1, 99), req(2, 99)], &b).unwrap();
        let mut ev = m.evidence(&out.key);
        // band 0 (input) and band 1 (output): fill/drain wavefront.
        let node = |band, microbatch, s, e| NodeSpan {
            band,
            microbatch,
            start_us: s,
            end_us: e,
        };
        ev[0].nodes = vec![node(0, 0, 0.0, 100.0), node(0, 1, 100.0, 200.0)];
        ev[1].nodes = vec![node(1, 0, 100.0, 200.0), node(1, 1, 200.0, 300.0)];
        ev[1].phases.push(PhaseTiming {
            phase: TimedPhase::Draft,
            us: 50.0,
            synchronized: true,
        });
        let merged = m.root.merge_evidence(&ev, interval()).unwrap();
        // Wavefront interval 300 (not max band duration 200, not the sum 400),
        // plus the root-only head phase once.
        assert_eq!(merged.critical_path_us, 350.0);
        // The sequential kill-switch arm sums non-overlapping spans.
        let mut seq = Mesh::new(MeshKind::Pp, false);
        for t in 1..=2u64 {
            seq.admit(t, 10, 7, None, Some(3));
            script_matching(&seq, t, 0, 8);
        }
        let b = budgets(LabRoutes::Batched, 3);
        let out = seq.run_greedy(&[req(1, 99), req(2, 99)], &b).unwrap();
        let mut ev = seq.evidence(&out.key);
        ev[0].nodes = vec![node(0, 0, 0.0, 100.0)];
        ev[1].nodes = vec![node(1, 0, 100.0, 200.0)];
        let merged = seq.root.merge_evidence(&ev, interval()).unwrap();
        assert_eq!(merged.critical_path_us, 200.0);
        // Incoherent PP counts: head work on the non-output band; edge payload mismatch.
        let mut bad = ev.clone();
        bad[0].counts.head_draft_calls = 1;
        assert!(matches!(
            seq.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        let mut bad = ev.clone();
        bad[1].counts.edge_bytes += 1;
        assert!(matches!(
            seq.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        let mut bad = ev.clone();
        bad[0].counts.edge_count += 1;
        bad[1].counts.edge_count += 1;
        assert!(matches!(
            seq.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
        // An EP gather on PP is incoherent too.
        let mut bad = ev.clone();
        bad[1].counts.target_exchanges = 48;
        assert!(matches!(
            seq.root.merge_evidence(&bad, interval()).unwrap_err(),
            EvidenceError::Incoherent(_)
        ));
    }

    // ---- identity, tickets, lifecycle (bullet 4) --------------------------------

    #[test]
    fn mesh_lane_protocol_stale_epoch_release_and_reuse() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        let e1 = m.admit(5, 10, 7, None, Some(3));
        assert!(
            matches!(
                m.root.admit_lane(LaneAdmission {
                    epoch: RequestEpoch {
                        request_tag: 5,
                        owner_generation: 2
                    },
                    position: 0,
                    seed: 0,
                    eos: None,
                    rng_seed: 0,
                    prompt: vec![],
                    spec_k: None,
                    head: HeadState::Free,
                }),
                Err(ProtocolError::Config(_))
            ),
            "a live request cannot hold two generations"
        );
        m.root.release_lane(e1).unwrap();
        for r in &mut m.ranks {
            r.release_lane(&e1).unwrap();
        }
        // Unknown after release.
        let err = m
            .root
            .plan_window(None, &[req(5, 9)], &budgets(LabRoutes::Batched, 3))
            .unwrap_err();
        assert_eq!(err, ProtocolError::UnknownLane(e1));
        // Re-admit the tag under a newer generation: the old epoch is stale.
        let e2 = RequestEpoch {
            request_tag: 5,
            owner_generation: 2,
        };
        let a = m
            .root
            .admit_lane(LaneAdmission {
                epoch: e2,
                position: 77,
                seed: 9,
                eos: None,
                rng_seed: 5,
                prompt: vec![8, 8],
                spec_k: Some(3),
                head: HeadState::Aligned,
            })
            .unwrap();
        for r in &mut m.ranks {
            r.admit_lane(&a).unwrap();
        }
        assert!(
            a.lane_generation > 1,
            "lane generation is monotonic across reuse"
        );
        let err = m
            .root
            .plan_window(None, &[req(5, 9)], &budgets(LabRoutes::Batched, 3))
            .unwrap_err();
        assert_eq!(err, ProtocolError::StaleEpoch(e1));
        // Duplicate lane in one plan; unadmitted epoch generation 0.
        let mut dup = req(5, 9);
        dup.epoch = e2;
        assert!(matches!(
            m.root
                .plan_window(None, &[dup.clone(), dup], &budgets(LabRoutes::Batched, 3)),
            Err(ProtocolError::Config(_))
        ));
        // A rank holding a stale lane generation rejects the whole step in preflight.
        let mut m2 = Mesh::new(MeshKind::Ep, false);
        let e = epoch(9);
        let a = m2
            .root
            .admit_lane(LaneAdmission {
                epoch: e,
                position: 5,
                seed: 1,
                eos: None,
                rng_seed: 1,
                prompt: vec![],
                spec_k: Some(3),
                head: HeadState::Aligned,
            })
            .unwrap();
        m2.ranks[0].admit_lane(&a).unwrap();
        let stale = LaneAdmit {
            lane_generation: a.lane_generation + 41,
            ..a.clone()
        };
        m2.ranks[1].admit_lane(&stale).unwrap();
        script_matching(&m2, 9, 0, 8);
        let err = m2
            .run_greedy(&[req(9, 9)], &budgets(LabRoutes::Batched, 3))
            .unwrap_err();
        match err {
            ProtocolError::Preflight(v) => {
                assert_eq!(v.len(), 1);
                assert_eq!(v[0].0, 1);
                assert!(v[0].1.contains("stale lane generation"), "{v:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mesh_lane_protocol_preflight_rejection_rejects_whole_step_before_device_work() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        m.admit(2, 10, 7, None, Some(3));
        for t in 1..=2 {
            script_matching(&m, t, 0, 8);
        }
        let before = (m.root.rng_probe(epoch(1)), m.root.lane_position(epoch(1)));
        m.ranks[1].capacity_error = Some("cannot map lane 2 arenas".to_string());
        let b = budgets(LabRoutes::Batched, 3);
        let err = m.run_greedy(&[req(1, 99), req(2, 99)], &b).unwrap_err();
        assert_eq!(
            err,
            ProtocolError::Preflight(vec![(1, "cannot map lane 2 arenas".to_string())])
        );
        // No collective ran anywhere, and nothing was quarantined or published.
        for r in &m.ranks {
            assert!(
                r.calls.iter().all(|c| matches!(c, Call::Begin(_))),
                "{:?}",
                r.calls
            );
            assert!(!r.session.is_quarantined());
        }
        assert!(m.root.quarantined().is_none());
        assert_eq!(m.root.phase(), StepPhase::Idle);
        assert_eq!(
            (m.root.rng_probe(epoch(1)), m.root.lane_position(epoch(1))),
            before
        );
        // The rank is usable again once capacity returns: same plan retried.
        m.ranks[1].capacity_error = None;
        let out = m.run_greedy(&[req(1, 99), req(2, 99)], &b).unwrap();
        assert_eq!(out.lanes.len(), 2);
    }

    #[test]
    fn mesh_lane_protocol_multiple_lane_tickets_are_distinct_and_checked() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        for t in 1..=3u64 {
            m.admit(t, 10, 7, None, Some(3));
            script_matching(&m, t, 0, 8);
        }
        let reqs = [req(1, 99), req(2, 99), req(3, 99)];
        let b = budgets(LabRoutes::Batched, 3);
        m.s_plan(None, &reqs, &b).unwrap();
        m.s_ready().unwrap();
        for ready in &m.ready {
            let ReadyStatus::Ready { tickets } = &ready.status else {
                panic!()
            };
            assert_eq!(tickets.len(), 3);
            let ids: HashSet<_> = tickets
                .iter()
                .flat_map(|t| [t.target, t.head])
                .flatten()
                .collect();
            assert_eq!(
                ids.len(),
                6,
                "one target + one head ticket per lane, all distinct"
            );
        }
        m.s_draft().unwrap();
        m.s_verify().unwrap();
        m.s_decide(&mut GreedyRule).unwrap();
        let d = m.decision.clone().unwrap();
        let mut receipts: Vec<PreparedReceipt> = m
            .ranks
            .iter_mut()
            .map(|r| r.prepare_commit(&d).unwrap())
            .collect();
        // Swapped ticket identities between two lanes of one rank: refused.
        let t0 = receipts[0].lanes[0].tickets;
        receipts[0].lanes[0].tickets = receipts[0].lanes[1].tickets;
        receipts[0].lanes[1].tickets = t0;
        let err = m.root.collect_prepared(&receipts).unwrap_err();
        match err {
            ProtocolError::Aborted(rec) => {
                assert_eq!(rec.phase, StepPhase::Decided);
                assert_eq!(rec.rank, Some(0));
                assert!(rec.reason.contains("ticket identity"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn mesh_lane_protocol_ready_ticket_ownership_and_reuse_checked() {
        // A PP input band that reserves a head ticket violates ownership.
        let mut m = Mesh::new(MeshKind::Pp, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        m.s_plan(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        let start = m.start.clone().unwrap();
        let mut ready: Vec<RankReady> =
            m.ranks.iter_mut().map(|r| r.begin_window(&start)).collect();
        let ReadyStatus::Ready { tickets } = &ready[0].status else {
            panic!()
        };
        assert!(
            tickets[0].head.is_none(),
            "the input band holds no head ticket"
        );
        assert!(tickets[0].target.is_some());
        let ReadyStatus::Ready { tickets } = &mut ready[0].status else {
            panic!()
        };
        tickets[0].head = Some(TicketId(999));
        let err = m.root.collect_ready(&ready).unwrap_err();
        assert!(
            matches!(&err, ProtocolError::Aborted(r) if r.reason.contains("ownership")),
            "{err:?}"
        );
    }

    // ---- cancellation (bullet 4) ------------------------------------------------

    #[test]
    fn mesh_lane_protocol_cancel_before_and_after_verify() {
        for cancel_at in ["idle", "drafted", "verified", "decided"] {
            let mut m = Mesh::new(MeshKind::Ep, false);
            m.admit(1, 10, 7, None, Some(3));
            m.admit(2, 10, 7, None, Some(3));
            for t in 1..=2 {
                script_matching(&m, t, 0, 8);
            }
            let reqs = [req(1, 99), req(2, 99)];
            let b = budgets(LabRoutes::Batched, 3);
            if cancel_at == "idle" {
                // Between steps the lane leaves at once and never reaches a plan.
                assert_eq!(
                    m.root.cancel_lane(epoch(1)).unwrap(),
                    CancelOutcome::Released
                );
                assert!(m.root.generated(epoch(1)).is_none());
                assert!(matches!(
                    m.root.plan_window(None, &reqs, &b),
                    Err(ProtocolError::UnknownLane(_))
                ));
                continue;
            }
            m.s_plan(None, &reqs, &b).unwrap();
            m.s_ready().unwrap();
            m.s_draft().unwrap();
            let mut cancelled = false;
            let mut cancel = |m: &mut Mesh, phase: StepPhase| {
                assert_eq!(
                    m.root.cancel_lane(epoch(1)).unwrap(),
                    CancelOutcome::Deferred { phase }
                );
                cancelled = true;
            };
            if cancel_at == "drafted" {
                cancel(&mut m, StepPhase::Drafted);
            }
            m.s_verify().unwrap();
            if cancel_at == "verified" {
                cancel(&mut m, StepPhase::Verified);
            }
            m.s_decide(&mut GreedyRule).unwrap();
            if cancel_at == "decided" {
                cancel(&mut m, StepPhase::Decided);
            }
            assert!(cancelled);
            m.s_prepare().unwrap();
            let out = m.s_publish();
            // The schedule was untouched (both lanes ran every phase on every rank)...
            assert_eq!(out.lanes.len(), 2);
            assert!(m.ranks[0]
                .calls
                .iter()
                .any(|c| matches!(c, Call::Verify(_, rows, _) if rows.len() == 2)));
            // ...but the cancelled lane streams nothing and is gone with its RNG/policy.
            let l1 = lane_out(&out, 1);
            assert!(l1.cancelled && l1.emitted.is_empty() && !l1.terminal);
            assert!(m.root.generated(epoch(1)).is_none());
            assert!(m.root.rng_probe(epoch(1)).is_none());
            // The neighbour published normally.
            let l2 = lane_out(&out, 2);
            assert!(!l2.cancelled);
            assert_eq!(m.root.generated(epoch(2)).unwrap(), l2.emitted.as_slice());
            // A cancelled lane is already gone from the root.
            assert!(m.root.release_lane(epoch(1)).is_err());
        }
    }

    // ---- failure => no publication (bullet 4) -------------------------------------

    #[test]
    fn mesh_lane_protocol_failure_before_prepare_publishes_nothing() {
        for (kind, fail, rank, phase) in [
            (MeshKind::Ep, FailAt::Draft(0), 1usize, StepPhase::Armed),
            (MeshKind::Ep, FailAt::Draft(1), 0, StepPhase::Armed),
            (MeshKind::Ep, FailAt::Verify, 1, StepPhase::Drafted),
            (MeshKind::Ep, FailAt::Prepare, 1, StepPhase::Decided),
            (MeshKind::Ep, FailAt::Prepare, 0, StepPhase::Decided),
            (MeshKind::Pp, FailAt::Verify, 0, StepPhase::Drafted),
            (MeshKind::Pp, FailAt::Prepare, 0, StepPhase::Decided),
            (MeshKind::Pp, FailAt::Draft(0), 1, StepPhase::Armed),
        ] {
            let mut m = Mesh::new(kind, false);
            for t in 1..=2 {
                m.admit(t, 30, 7, None, Some(3));
                script_matching(&m, t, 0, 8);
            }
            m.ranks[rank].fail_at = Some(fail);
            let pre: Vec<_> = (1..=2)
                .map(|t| (m.root.rng_probe(epoch(t)), m.root.lane_position(epoch(t))))
                .collect();
            let err = m
                .run(
                    None,
                    &[req(1, 99), req(2, 99)],
                    &budgets(LabRoutes::Batched, 3),
                    &mut DrawRule,
                )
                .unwrap_err();
            let ProtocolError::Aborted(rec) = err else {
                panic!("{kind:?} {fail:?}: {err:?}")
            };
            assert_eq!(rec.phase, phase, "{kind:?} {fail:?}");
            assert!(rec.reason.contains(&format!("rank {rank}")), "{rec:?}");
            // Quarantine: the root and every rank, no fallback.
            assert!(m.root.quarantined().is_some());
            assert!(
                m.ranks.iter().all(|r| r.session.is_quarantined()),
                "{kind:?} {fail:?}"
            );
            // Nothing published anywhere: no IDs, no RNG advance, no position/seed.
            assert!(m.ranks.iter().all(|r| r.published.is_empty()));
            let post: Vec<_> = (1..=2)
                .map(|t| (m.root.rng_probe(epoch(t)), m.root.lane_position(epoch(t))))
                .collect();
            assert_eq!(pre, post, "{kind:?} {fail:?}");
            assert!(m.root.generated(epoch(1)).unwrap().is_empty());
            assert_eq!(
                m.root.copy_history(epoch(1)).unwrap().tokens(),
                &[1, 2, 3, 1]
            );
            // The mesh stays down.
            let again = m
                .root
                .plan_window(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3))
                .unwrap_err();
            assert_eq!(again, ProtocolError::Aborted(rec.clone()));
            // Idempotent abort: the first record wins, any phase/reason.
            let key = rec.key.clone();
            let second = m
                .root
                .abort_step(key.as_ref(), StepPhase::Prepared, "later reason")
                .unwrap();
            assert_eq!(second, rec);
            for r in &mut m.ranks {
                let n = r.aborts.len();
                r.abort_step(key.as_ref().unwrap(), StepPhase::Prepared, "again");
                assert_eq!(r.aborts.len(), n + 1);
                assert!(r.session.is_quarantined());
            }
        }
    }

    #[test]
    fn mesh_lane_protocol_nothing_published_until_all_prepared() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 2);
        let e = epoch(1);
        let rng0 = m.root.rng_probe(e).unwrap();
        let reqs = [req(1, 99)];
        let b = budgets(LabRoutes::Batched, 3);
        m.s_plan(None, &reqs, &b).unwrap();
        let unchanged = |m: &Mesh| {
            m.root.generated(e).unwrap().is_empty()
                && m.root.lane_position(e) == Some((10, 7))
                && m.root.copy_history(e).unwrap().tokens() == &[1u32, 2, 3, 1][..]
        };
        // Calling a later phase out of order is a caller error and changes nothing.
        assert_eq!(
            m.root.publish_commit().unwrap_err(),
            ProtocolError::Phase {
                op: "publish_commit",
                actual: StepPhase::Planned
            }
        );
        m.s_ready().unwrap();
        m.s_draft().unwrap();
        m.s_verify().unwrap();
        assert!(unchanged(&m));
        let d = m.s_decide(&mut DrawRule).unwrap();
        assert!(unchanged(&m), "a decision is staged, not published");
        assert_eq!(
            m.root.rng_probe(e).unwrap(),
            rng0,
            "the RNG successor is staged"
        );
        assert!(
            m.root.publish_commit().is_err(),
            "Decided is not publishable"
        );
        // One rank's receipt missing: not prepared, still nothing published.
        let partial: Vec<PreparedReceipt> = vec![m.ranks[0].prepare_commit(&d).unwrap()];
        let err = m.root.collect_prepared(&partial).unwrap_err();
        assert!(
            matches!(&err, ProtocolError::Aborted(r) if r.reason.contains("no prepared receipt")),
            "{err:?}"
        );
        assert!(unchanged(&m));
        assert!(m.root.publish_commit().is_err());
        assert!(m.ranks.iter().all(|r| r.published.is_empty()));
    }

    #[test]
    fn mesh_lane_protocol_publish_applies_all_staged_state_once() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 2);
        let e = epoch(1);
        let rng0 = m.root.rng_probe(e).unwrap();
        m.s_through_prepared(
            &[req(1, 99)],
            &budgets(LabRoutes::Batched, 3),
            &mut DrawRule,
        );
        assert_eq!(m.root.phase(), StepPhase::Prepared);
        assert!(m.root.generated(e).unwrap().is_empty());
        let out = m.s_publish();
        let l = lane_out(&out, 1);
        assert_eq!(m.root.generated(e).unwrap(), l.emitted.as_slice());
        assert_eq!(
            m.root.copy_history(e).unwrap().tokens().len(),
            4 + l.emitted.len()
        );
        assert_eq!(
            m.root.lane_position(e).unwrap(),
            (10 + l.consumed_rows, l.next_seed)
        );
        assert_ne!(m.root.rng_probe(e).unwrap(), rng0);
        assert_eq!(m.root.counters().steps_published, 1);
        assert!(m
            .ranks
            .iter()
            .all(|r| r.published == vec![out.key.step_seq]));
        // Published twice is a phase error.
        assert!(m.root.publish_commit().is_err());
    }

    #[test]
    fn mesh_lane_protocol_abort_from_every_phase_is_idempotent_and_drops_staged_state() {
        for stop in 0..6 {
            let mut m = Mesh::new(MeshKind::Ep, false);
            m.admit(1, 10, 7, None, Some(3));
            script_matching(&m, 1, 0, 3);
            let e = epoch(1);
            let rng0 = m.root.rng_probe(e).unwrap();
            let reqs = [req(1, 99)];
            let b = budgets(LabRoutes::Batched, 3);
            let start = m.s_plan(None, &reqs, &b).unwrap();
            let mut phase = StepPhase::Planned;
            if stop >= 1 {
                m.s_ready().unwrap();
                phase = StepPhase::Armed;
            }
            if stop >= 2 {
                m.s_draft().unwrap();
                phase = StepPhase::Drafted;
            }
            if stop >= 3 {
                m.s_verify().unwrap();
                phase = StepPhase::Verified;
            }
            if stop >= 4 {
                m.s_decide(&mut DrawRule).unwrap();
                phase = StepPhase::Decided;
            }
            if stop >= 5 {
                m.s_prepare().unwrap();
                phase = StepPhase::Prepared;
            }
            assert_eq!(m.root.phase(), phase);
            let rec = m
                .root
                .abort_step(Some(&start.key), phase, "operator timeout")
                .unwrap();
            assert_eq!(rec.phase, phase);
            let rec2 = m
                .root
                .abort_step(Some(&start.key), StepPhase::Idle, "different")
                .unwrap();
            assert_eq!(rec, rec2);
            assert_eq!(m.root.phase(), StepPhase::Aborted);
            assert_eq!(m.root.rng_probe(e).unwrap(), rng0);
            assert!(m.root.generated(e).unwrap().is_empty());
            assert!(matches!(
                m.root.publish_commit(),
                Err(ProtocolError::Aborted(_))
            ));
            assert!(matches!(
                m.root.cancel_lane(e),
                Err(ProtocolError::Aborted(_))
            ));
            assert!(matches!(
                m.root.next_draft_round(),
                Err(ProtocolError::Aborted(_))
            ));
        }
        // A stale abort (other step's key) neither aborts nor quarantines.
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 3);
        let start = m
            .s_plan(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        let mut other = start.key.clone();
        other.step_seq += 5;
        assert!(m
            .root
            .abort_step(Some(&other), StepPhase::Planned, "stale")
            .is_err());
        assert!(m.root.quarantined().is_none());
        assert_eq!(m.root.phase(), StepPhase::Planned);
    }

    // ---- followers cannot decide (bullet 5, host half) --------------------------

    #[test]
    fn mesh_lane_protocol_only_the_root_can_build_a_root() {
        let cfg = RootConfig {
            topology_generation: 1,
            stage: StagePolicy {
                moe_layers: 48,
                moe_once: true,
                stream_bf16: false,
            },
            floor_enabled: false,
        };
        let ep = world(MeshKind::Ep);
        assert!(MeshRoot::new(ep[0], &ep, cfg).is_ok());
        assert!(
            matches!(
                MeshRoot::new(ep[1], &ep, cfg),
                Err(ProtocolError::Config(_))
            ),
            "EP follower"
        );
        let pp = world(MeshKind::Pp);
        assert!(
            matches!(
                MeshRoot::new(pp[0], &pp, cfg),
                Err(ProtocolError::Config(_))
            ),
            "PP non-output band"
        );
        assert!(
            MeshRoot::new(pp[1], &pp, cfg).is_ok(),
            "the PP root is the output band, rank 1 here"
        );
        // A rank claiming root that is not the world's root.
        assert!(MeshRoot::new(
            RankOwnership::ep(0),
            &[RankOwnership::ep(1), RankOwnership::ep(2)],
            cfg
        )
        .is_err());
        // Ill-formed worlds.
        let mut two_roots = ep.clone();
        two_roots[1].root = true;
        assert!(validate_world(&two_roots).is_err());
        let mut no_head = ep.clone();
        no_head[1].head = false;
        assert!(
            validate_world(&no_head).is_err(),
            "EP ranks all run the matched head calls"
        );
        let mut pp_head = pp.clone();
        pp_head[0].head = true;
        assert!(
            validate_world(&pp_head).is_err(),
            "PP head is output-band only"
        );
        assert!(validate_world(&[ep[0], pp[1]]).is_err(), "mixed topologies");
        assert!(validate_world(&[]).is_err());
        assert!(validate_world(&[ep[0], ep[0]]).is_err(), "duplicate rank");
    }

    #[test]
    fn mesh_lane_protocol_follower_proposals_or_sampler_output_abort_the_step() {
        for (kind, rogue, spec_k, expect, expect_rank) in [
            (
                MeshKind::Ep,
                1usize,
                Some(3usize),
                "follower returned proposals",
                1usize,
            ),
            (
                MeshKind::Ep,
                1,
                None,
                "non-sampler returned target output",
                1,
            ),
            (
                MeshKind::Pp,
                0,
                Some(3),
                "non-sampler returned target output",
                0,
            ),
        ] {
            let mut m = Mesh::new(kind, false);
            m.admit(1, 10, 7, None, spec_k);
            script_matching(&m, 1, 0, 8);
            m.ranks[rogue].rogue_sampler_output = true;
            let err = m
                .run_greedy(&[req(1, 99)], &budgets(LabRoutes::Batched, 3))
                .unwrap_err();
            let ProtocolError::Aborted(rec) = err else {
                panic!("{err:?}")
            };
            assert!(rec.reason.contains(expect), "{kind:?}: {rec:?}");
            assert_eq!(rec.rank, Some(expect_rank));
            // A PP input band is never dispatched head work and an AR lane has
            // no draft rounds: those rogue outputs surface at collect_verify.
            assert!(m
                .ranks
                .iter()
                .all(|r| r.published.is_empty() && r.session.is_quarantined()));
        }
    }

    #[test]
    fn mesh_lane_protocol_rank_session_refuses_head_work_off_the_output_band() {
        let mut m = Mesh::new(MeshKind::Pp, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        m.s_plan(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        m.s_ready().unwrap();
        let round = m.root.next_draft_round().unwrap().unwrap();
        // The input band (rank 0) must never be handed a head round.
        let err = m.ranks[0].session.on_draft_round(&round).unwrap_err();
        assert!(err.reason.contains("without the head"), "{err:?}");
        // Its tickets held no head slot (no head state to allocate or use).
        let ReadyStatus::Ready { tickets } = &m.ready[0].status else {
            panic!()
        };
        assert!(tickets.iter().all(|t| t.head.is_none()));
        // The output band replays it.
        let ok = m.ranks[1].session.on_draft_round(&round).unwrap();
        assert!(ok.proposals.is_none(), "RankSession itself never samples");
        // Out-of-order rounds and wrong positions are refused.
        let mut again = round.clone();
        again.round = 5;
        assert!(m.ranks[1].session.on_draft_round(&again).is_err());
    }

    // ---- RNG / copy history (bullet 7) ---------------------------------------------

    #[test]
    fn mesh_lane_protocol_root_rng_invariant_to_lane_scheduling() {
        type LaneSnapshot = (Vec<u32>, u64, (usize, u32));
        fn run(plan: &[&[u64]]) -> HashMap<u64, LaneSnapshot> {
            let mut m = Mesh::new(MeshKind::Ep, false);
            for t in 1..=3u64 {
                m.admit(t, 20 * t as usize, 7, None, Some(3));
                for seq in 0..4 {
                    script_matching(&m, t, seq, ((seq + t) % 4) as usize);
                }
            }
            let b = budgets(LabRoutes::Batched, 3);
            for step in plan {
                let reqs: Vec<LaneRequest> = step.iter().map(|&t| req(t, 1000)).collect();
                m.run(None, &reqs, &b, &mut DrawRule).unwrap();
            }
            (1..=3u64)
                .map(|t| {
                    (
                        t,
                        (
                            m.root.generated(epoch(t)).unwrap().to_vec(),
                            m.root.rng_probe(epoch(t)).unwrap(),
                            m.root.lane_position(epoch(t)).unwrap(),
                        ),
                    )
                })
                .collect()
        }
        // Four windows per lane in three different schedules.
        let a = run(&[&[1, 2, 3], &[1, 2, 3], &[1, 2, 3], &[1, 2, 3]]);
        let b = run(&[&[3, 1, 2], &[2, 3, 1], &[1, 3, 2], &[3, 2, 1]]);
        let sched: Vec<&[u64]> = vec![&[1], &[2, 3], &[1], &[2, 3], &[1], &[2, 3], &[1], &[2, 3]];
        let c = run(&sched);
        assert_eq!(
            a, b,
            "lane order inside a step cannot perturb a request's stream"
        );
        assert_eq!(a.keys().count(), 3);
        assert_eq!(
            a, c,
            "interleaving with other lanes cannot perturb a request"
        );
        // Distinct requests have distinct streams.
        assert_ne!(a[&1].1, a[&2].1);
    }

    #[test]
    fn mesh_lane_protocol_copy_history_never_crosses_epochs_and_keeps_retained_output_only() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        let e1 = m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 2);
        let out = m
            .run_greedy(&[req(1, 99)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        let emitted = lane_out(&out, 1).emitted.clone();
        // prompt ++ retained output; rejected drafts / candidates are never history.
        let mut want = vec![1, 2, 3, 1];
        want.extend(&emitted);
        assert_eq!(m.root.copy_history(e1).unwrap().tokens(), want.as_slice());
        assert!(!want.contains(&12), "a rejected draft is not history");
        assert_eq!(m.root.copy_history(e1).unwrap().epoch(), e1);
        // Release and reuse the tag at a new owner generation.
        m.root.release_lane(e1).unwrap();
        for r in &mut m.ranks {
            r.release_lane(&e1).unwrap();
        }
        let e2 = RequestEpoch {
            request_tag: 1,
            owner_generation: 2,
        };
        let a = m
            .root
            .admit_lane(LaneAdmission {
                epoch: e2,
                position: 10,
                seed: 7,
                eos: None,
                rng_seed: 77,
                prompt: vec![9, 9],
                spec_k: Some(3),
                head: HeadState::Aligned,
            })
            .unwrap();
        assert_eq!(
            m.root.copy_history(e2).unwrap().tokens(),
            &[9, 9],
            "nothing of epoch 1 survives"
        );
        assert!(m.root.copy_history(e1).is_none());
        assert_eq!(m.root.copy_history(e2).unwrap().epoch(), e2);
        assert!(a.lane_generation > out.lanes[0].key.lane_generation);
        assert_eq!(m.root.generated(e2).unwrap(), &[] as &[u32]);
        assert_eq!(m.root.rng_probe(e2).unwrap(), {
            let mut r = SpecRng::new(77);
            r.next_u64()
        });
    }

    #[test]
    fn mesh_lane_protocol_copy_window_is_a_root_source_without_head_rounds() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, Some(21), Some(3));
        // Cycle window 0 is Batched{K}, 1 Batched{1}, 2 ArSynced (copy overrides),
        // 3 Batched{2}: walk to the ArSynced window.
        let b = budgets(LabRoutes::Cycle, 3);
        for seq in 0..2 {
            script_matching(&m, 1, seq, 8);
            m.run_greedy(&[req(1, 1000)], &b).unwrap();
        }
        let mut r = req(1, 1000);
        r.copy_candidates = vec![30, 21, 99];
        let start = m.root.plan_window(None, &[r.clone()], &b).unwrap();
        let lane = &start.lanes[0];
        assert_eq!(
            lane.route,
            LaneRoute::ArSynced,
            "the native route survives for accounting"
        );
        assert_eq!(lane.source, ProposalSource::Copy);
        assert_eq!(
            lane.copy_drafts,
            vec![30, 21],
            "truncated after the first EOS"
        );
        assert_eq!(lane.max_depth, 2);
        m.root
            .abort_step(Some(&start.key), StepPhase::Planned, "probe")
            .unwrap();

        // Run it for real on a fresh mesh walked to the same point.
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, Some(21), Some(3));
        for seq in 0..2 {
            script_matching(&m, 1, seq, 8);
            m.run_greedy(&[req(1, 1000)], &b).unwrap();
        }
        m.script_window(1, 2, &[], &[30, 21, 5]);
        let calls_before = m.ranks[0].calls.len();
        let out = m.run_greedy(&[r], &b).unwrap();
        let l = lane_out(&out, 1);
        assert_eq!(l.source.source, ProposalSource::Copy);
        assert_eq!(l.source.offered, 2);
        assert_eq!(
            l.emitted,
            vec![30, 21],
            "an accepted EOS candidate ends the window"
        );
        assert!(l.terminal);
        assert_eq!(l.consumed_rows, 2, "accepted EOS stays the pending seed");
        assert!(
            m.ranks[0].calls[calls_before..]
                .iter()
                .all(|c| !matches!(c, Call::Draft(..))),
            "copy proposals are root-known: no head round"
        );
    }

    // ---- plan limits ---------------------------------------------------------------

    #[test]
    fn mesh_lane_protocol_plan_limits_are_refused_before_any_state_change() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        for t in 1..=3 {
            m.admit(t, 10, 7, None, Some(3));
        }
        let reqs = [req(1, 99), req(2, 99), req(3, 99)];
        let mut b = budgets(LabRoutes::Batched, 3);
        b.verify_rows_per_lane = LANE_MAX_ROWS + 1;
        assert!(
            matches!(
                m.root.plan_window(None, &reqs, &b),
                Err(ProtocolError::Budget(_))
            ),
            "row-exact guard"
        );
        let mut b = budgets(LabRoutes::Batched, 3);
        b.max_lanes = 2;
        assert!(matches!(
            m.root.plan_window(None, &reqs, &b),
            Err(ProtocolError::Budget(_))
        ));
        let mut b = budgets(LabRoutes::Batched, 3);
        b.max_rows = 11;
        assert!(
            matches!(
                m.root.plan_window(None, &reqs, &b),
                Err(ProtocolError::Budget(_))
            ),
            "12 rows > 11"
        );
        let b = budgets(LabRoutes::Batched, 3);
        assert!(matches!(
            m.root.plan_window(None, &[], &b),
            Err(ProtocolError::Budget(_))
        ));
        assert!(matches!(
            m.root.plan_window(None, &[req(1, 0)], &b),
            Err(ProtocolError::Budget(_))
        ));
        let mut f = req(1, 9);
        f.forced = Some(vec![1; 5]);
        assert!(matches!(
            m.root.plan_window(None, &[f], &b),
            Err(ProtocolError::Budget(_))
        ));
        // None of the refusals planned anything.
        assert_eq!(m.root.phase(), StepPhase::Idle);
        // Floor-calibrating windows of a spec lane are AR-class rows (single row).
        let ok = m.root.plan_window(None, &reqs, &b).unwrap();
        assert_eq!(ok.key.lanes.len(), 3);
        assert_eq!(ok.key.topology_generation, 7);
        // Two plans of the same history agree (deterministic), digest-for-digest.
        let mut m2 = Mesh::new(MeshKind::Ep, false);
        for t in 1..=3 {
            m2.admit(t, 10, 7, None, Some(3));
        }
        assert_eq!(m2.root.plan_window(None, &reqs, &b).unwrap(), ok);
    }

    #[test]
    fn mesh_lane_protocol_malformed_acceptance_aborts() {
        struct Bad;
        impl AcceptRule for Bad {
            fn accept(
                &mut self,
                _l: &LanePlan,
                d: &[u32],
                _p: &[u32],
                _r: &mut LaneRng,
            ) -> Result<GreedyAccept, String> {
                // Claims an accepted prefix that is not the drafts.
                Ok(GreedyAccept {
                    committed: vec![d[0] + 1, 5],
                    accepted: 1,
                    hit_eos: false,
                })
            }
        }
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        script_matching(&m, 1, 0, 8);
        let err = m
            .run(
                None,
                &[req(1, 99)],
                &budgets(LabRoutes::Batched, 3),
                &mut Bad,
            )
            .unwrap_err();
        assert!(
            matches!(&err, ProtocolError::Aborted(r) if r.phase == StepPhase::Verified && r.reason.contains("window shape")),
            "{err:?}"
        );
        assert!(m
            .ranks
            .iter()
            .all(|r| r.published.is_empty() && r.session.is_quarantined()));
    }

    // ---- rank-side validation --------------------------------------------------------

    #[test]
    fn mesh_lane_protocol_rank_session_validates_window_start_and_phase_order() {
        let mut m = Mesh::new(MeshKind::Ep, false);
        m.admit(1, 10, 7, None, Some(3));
        let start = m
            .root
            .plan_window(None, &[req(1, 99)], &budgets(LabRoutes::Batched, 3))
            .unwrap();
        let mut s = RankSession::new(RankOwnership::ep(1));
        s.admit(&LaneAdmit {
            epoch: epoch(1),
            lane_generation: 1,
            position: 10,
            spec: true,
        })
        .unwrap();
        // Tampered plan (digest) is rejected.
        let mut bad = start.clone();
        bad.lanes[0].seed += 1;
        match s.begin(&bad, Ok(())).status {
            ReadyStatus::Rejected { reason } => assert!(reason.contains("digest")),
            other => panic!("{other:?}"),
        }
        // Unknown epoch is stale.
        let mut s2 = RankSession::new(RankOwnership::ep(1));
        match s2.begin(&start, Ok(())).status {
            ReadyStatus::Rejected { reason } => assert!(reason.contains("stale or unknown epoch")),
            other => panic!("{other:?}"),
        }
        // Good begin; a second begin before the step ends is refused.
        assert!(matches!(
            s.begin(&start, Ok(())).status,
            ReadyStatus::Ready { .. }
        ));
        assert!(matches!(
            s.begin(&start, Ok(())).status,
            ReadyStatus::Rejected { .. }
        ));
        // Verify before drafting finished is a different phase than the draft round expects.
        assert_eq!(s.phase(), StepPhase::Armed);
        // Release/admit refuse while a step is in flight.
        assert!(s.release(&epoch(1)).is_err());
        // reject_window unwinds a preflight.
        s.reject(&start.key);
        assert_eq!(s.phase(), StepPhase::Idle);
        assert!(s.release(&epoch(1)).is_ok());
        // Abort is idempotent and final.
        s.on_abort();
        s.on_abort();
        assert!(s.is_quarantined());
        assert!(matches!(
            s.begin(&start, Ok(())).status,
            ReadyStatus::Rejected { .. }
        ));
    }
}
