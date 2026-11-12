// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Per-rank / per-band device census of a lane store, and the one immutable
//! [`LaneSpecBudget`] every topology admits against.
//!
//! The census is a list of [`CensusLine`]s, one per allocation family, built
//! from the allocation inventories the allocators themselves are tested
//! against ([`Qwen4State::row_capture_allocation_sizes_scoped`],
//! [`MtpGpuState::allocation_sizes`], [`MtpAppendScratch::allocation_sizes`],
//! [`Qwen4State::device_bytes_scoped`]); it never carries a formula of its
//! own for a family those helpers already size. Everything is pure host
//! arithmetic: a census, and so an admission refusal, exists before any
//! device allocation, and an overflowing extent is an `Err`, never a wrapped
//! or saturated figure.
//!
//! Ownership (DESIGN §3.1, §6):
//!
//! - every rank/band charges the **target** side of every admitted lane
//!   scoped to the layers it owns: the committed request state (live state,
//!   the snapshot mirror of an MTP lane only, committed QSA pages) and the
//!   target GDN capture ring over its own GDN layers;
//! - only a **head owner** (`owner_mask` bit; EP: every rank, PP: the output
//!   band) charges the MTP head state, the pending/lag hidden, the wide
//!   hidden capture, the draft candidate slots, the prompt-fill scratch and
//!   the sampling scratch;
//! - only a band that runs the LM head (`scope.output`) charges the step
//!   logits and top-1: a non-output PP band never holds vocabulary rows, and
//!   a head owner must be such a band.

use super::MAX_LANES;
use crate::config::Qwen4Config;
use crate::gpu_forward_lanes::LANE_MAX_ROWS;
use crate::kv_backend::Qwen4ContextCommit;
use crate::mtp_gpu::{MtpAppendScratch, MtpGpuState};
use crate::state::{scope_layer_counts, Qwen4State, Qwen4StateFormat, StateMirror};
use crate::weights::Qwen4BandScope;
use rdna_compute::Gpu;

/// The copy lane's proposal/observation limit (`n_max`); not an executable
/// verify depth (DESIGN §5.2).
pub const COPY_PROPOSAL_N_MAX: usize = 63;

/// Ranks/bands one `owner_mask` can name.
pub const MAX_OWNER_RANKS: usize = u64::BITS as usize;

/// The snapshot mirror of every lane state of an AR store: none. An AR lane is
/// only stepped and reset; it never takes a snapshot ticket (only a `Verify`
/// window does, and that needs an MTP window) and arms no capture ring.
pub(crate) const AR_LANE_MIRROR: StateMirror = StateMirror::Absent;

/// The snapshot mirror of every lane state of an MTP store, on every band
/// that owns target layers: a verify window snapshots the lane state and rolls
/// it back.
pub(crate) const SPEC_LANE_MIRROR: StateMirror = StateMirror::Snapshot;

/// Immutable capacity of one lane store, fixed at load/admission.
///
/// `verify_rows_per_lane = 1 + max(native_k, copy_take_cap)` and
/// `capture_rows_per_lane = max(verify_rows_per_lane, row_exact rows)` (the
/// `native_mtp_capture_rows` contract): both are **fixed allocations**, so a
/// scheduler that picks a shorter window refunds nothing. Total step capacity
/// is `lanes * verify_rows_per_lane`, never `K * lanes` and never one shared
/// ring. Build one with [`Self::new`], [`Self::default_copy`] or
/// [`Self::legacy`]; the derived fields are never set independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSpecBudget {
    /// Admitted lanes (request owners).
    pub lanes: usize,
    /// Deepest native (head) window, drafts per window.
    pub native_k: usize,
    /// The copy lane's proposal/observation limit; not an execution depth.
    pub proposal_n_max: usize,
    /// Copy candidates one takeover verifies at most (`0`: copy not armed).
    pub copy_take_cap: usize,
    /// Verify rows one lane holds: `1 + max(native_k, copy_take_cap)`.
    pub verify_rows_per_lane: usize,
    /// Target GDN capture ring rows one lane holds on every owning band.
    pub capture_rows_per_lane: usize,
    /// Head-lag rows a retired-with-copy lane may owe its head (hidden rows
    /// and tokens, on head owners only).
    pub head_lag_rows: usize,
    /// The QSA context every lane's state and head map.
    pub context_commit: Qwen4ContextCommit,
    /// Bit `r` set: rank/band `r` owns the MTP head and the sampler.
    pub owner_mask: u64,
    /// Device bytes of sampled-verify scratch a head owner holds once (shared
    /// across lanes); `0` for greedy-only and the host-draw naive sampler.
    pub sampling_scratch: u64,
    /// Decode rows per lane the step buffers hold (`verify_rows_per_lane`
    /// unless [`Self::with_copy_steps`] narrows it).
    step_rows_per_lane: usize,
    /// Rows the widest copy step carries (`0`: no copy steps, the per-lane
    /// rows alone size the step; see [`Self::with_copy_steps`]).
    copy_step_rows: usize,
}

impl LaneSpecBudget {
    /// A validated budget. `row_exact_rows` is the arch's row-exact verify
    /// width (`qwen4_row_exact_verify_rows`); capture is at least that wide.
    /// Refused: no lane or more than the store's maximum, `native_k` outside
    /// `1..LANE_MAX_ROWS`, a copy take cap that would run a verify forward
    /// wider than the eight-row guard, a capture ring beyond the physical
    /// verify capacity, an empty owner mask, any extent that overflows.
    pub fn new(
        lanes: usize,
        native_k: usize,
        copy_take_cap: usize,
        row_exact_rows: usize,
        context_commit: Qwen4ContextCommit,
        owner_mask: u64,
    ) -> Result<Self, String> {
        if lanes == 0 || lanes > MAX_LANES {
            return Err(format!(
                "lane budget: lanes {lanes} outside 1..={MAX_LANES}"
            ));
        }
        if native_k == 0 || native_k >= LANE_MAX_ROWS {
            return Err(format!(
                "lane budget: native K {native_k} outside 1..={}",
                LANE_MAX_ROWS - 1
            ));
        }
        if copy_take_cap >= LANE_MAX_ROWS {
            return Err(format!(
                "lane budget: copy take cap {copy_take_cap} would verify {} rows, past the {LANE_MAX_ROWS}-row execution guard",
                copy_take_cap + 1
            ));
        }
        if owner_mask == 0 {
            return Err("lane budget: no rank owns the MTP head".to_string());
        }
        let verify_rows_per_lane = 1 + native_k.max(copy_take_cap);
        let capture_rows_per_lane = verify_rows_per_lane.max(row_exact_rows);
        if capture_rows_per_lane > hipfire_config::QWEN4_VERIFY_MAX_ROWS {
            return Err(format!(
                "lane budget: capture ring of {capture_rows_per_lane} rows exceeds the {}-row physical verify capacity",
                hipfire_config::QWEN4_VERIFY_MAX_ROWS
            ));
        }
        lanes.checked_mul(verify_rows_per_lane).ok_or_else(|| {
            format!("lane budget: {lanes} lanes x {verify_rows_per_lane} verify rows overflows")
        })?;
        Ok(Self {
            lanes,
            native_k,
            proposal_n_max: native_k.max(copy_take_cap),
            copy_take_cap,
            verify_rows_per_lane,
            capture_rows_per_lane,
            head_lag_rows: 0,
            context_commit,
            owner_mask,
            sampling_scratch: 0,
            step_rows_per_lane: verify_rows_per_lane,
            copy_step_rows: 0,
        })
    }

    /// The production default: external copy armed, so every lane carries
    /// `take_cap = min(n_max=63, row_exact - 1)` candidates (seven on every
    /// proven GPU) and a native `native_k = 1` still needs the full
    /// eight-row verify/capture capacity.
    pub fn default_copy(
        lanes: usize,
        native_k: usize,
        row_exact_rows: usize,
        context_commit: Qwen4ContextCommit,
        owner_mask: u64,
    ) -> Result<Self, String> {
        let take_cap = COPY_PROPOSAL_N_MAX.min(row_exact_rows.saturating_sub(1));
        let mut budget = Self::new(
            lanes,
            native_k,
            take_cap,
            row_exact_rows,
            context_commit,
            owner_mask,
        )?;
        budget.proposal_n_max = COPY_PROPOSAL_N_MAX;
        Ok(budget)
    }

    /// The budget the existing store constructor honors today: copy not
    /// armed, a `native_k + 1`-row ring (no row-exact widening), no lag, no
    /// sampling scratch. `device_bytes_scoped` builds exactly this budget, so
    /// the legacy store answer is the census.
    pub fn legacy(
        lanes: usize,
        native_k: usize,
        context_commit: Qwen4ContextCommit,
        owner_mask: u64,
    ) -> Result<Self, String> {
        Self::new(lanes, native_k, 0, 0, context_commit, owner_mask)
    }

    /// With `rows` head-lag rows owed per lane.
    pub fn with_head_lag_rows(mut self, rows: usize) -> Self {
        self.head_lag_rows = rows;
        self
    }

    /// With `bytes` of sampled-verify scratch per head owner.
    pub fn with_sampling_scratch(mut self, bytes: u64) -> Self {
        self.sampling_scratch = bytes;
        self
    }

    /// With every lane's QSA context fully mapped to `max_seq`: the
    /// worst-case growth commitment (every lane runs to the admitted
    /// context). Compare against the load-time census to read the growth.
    pub fn at_full_context(mut self) -> Self {
        self.context_commit.tokens = self.context_commit.max_seq;
        self
    }

    /// With the step buffers of a store that runs copy windows only in steps
    /// of at most `copy_lanes` lanes (every other window is native, at most
    /// `native_k + 1` rows): the buffers hold the larger of the native step
    /// (`lanes * (native_k + 1)`) and the widest copy step (`min(copy_lanes,
    /// lanes)` lanes of `verify_rows_per_lane` rows). A copy step must fit
    /// whatever the lane count: at fewer lanes than the native rows cover
    /// (`lanes * (native_k + 1) < copy_lanes * verify_rows_per_lane`) the
    /// widest copy step sets the size, so a small store still takes its copy
    /// windows. At the default eight lanes the native step is the larger
    /// and the buffers are unchanged.
    pub fn with_copy_steps(mut self, copy_lanes: usize) -> Self {
        self.step_rows_per_lane = (self.native_k + 1).min(self.verify_rows_per_lane);
        self.copy_step_rows = copy_lanes.min(self.lanes) * self.verify_rows_per_lane;
        self
    }

    /// Decode rows the step buffers hold: `lanes * step_rows_per_lane`, and at
    /// least the widest copy step ([`Self::with_copy_steps`]);
    /// `lanes * verify_rows_per_lane` without it.
    pub fn step_rows(&self) -> usize {
        (self.lanes * self.step_rows_per_lane).max(self.copy_step_rows)
    }

    /// Whether rank/band `rank` owns the MTP head and the sampler.
    pub fn owns_head(&self, rank: usize) -> bool {
        rank < MAX_OWNER_RANKS && self.owner_mask >> rank & 1 == 1
    }
}

/// One allocation family of a rank/band's lane store.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CensusComponent {
    /// Per lane: [`Qwen4State::device_bytes_with_mirror`] (live state, the
    /// snapshot mirror of an MTP lane only, committed QSA pages) over the
    /// band's own layers.
    TargetState,
    /// Per lane: the target GDN capture ring over the band's GDN layers.
    CaptureRing,
    /// Per lane, head owners: the MTP head's QSA state and snapshot mirrors.
    HeadState,
    /// Per lane, head owners: the pending target hidden row.
    PendingHidden,
    /// Per lane, head owners: owed head-lag hidden rows (the owed tokens are
    /// host values, not device bytes).
    HeadLag,
    /// Shared, LM-head bands: step logits (`(rows + lanes) * vocab` F32).
    StepLogits,
    /// Shared, LM-head bands: step top-1 picks.
    StepTop1,
    /// Shared, head owners: the wide hidden of every decode row.
    WideHidden,
    /// Shared, head owners: one draft candidate slot per lane.
    DraftTop1,
    /// Shared, head owners: the prompt-fill append scratch.
    AppendScratch,
    /// Shared, head owners: sampled-verify scratch.
    SamplingScratch,
}

impl CensusComponent {
    /// Short stable name for logs and refusals.
    pub const fn name(self) -> &'static str {
        match self {
            Self::TargetState => "target state",
            Self::CaptureRing => "GDN capture ring",
            Self::HeadState => "head state",
            Self::PendingHidden => "pending hidden",
            Self::HeadLag => "head lag",
            Self::StepLogits => "step logits",
            Self::StepTop1 => "step top-1",
            Self::WideHidden => "wide hidden",
            Self::DraftTop1 => "draft top-1",
            Self::AppendScratch => "append scratch",
            Self::SamplingScratch => "sampling scratch",
        }
    }
}

/// `copies` identical sets of device allocations (`copies = lanes` for a
/// per-lane family, `1` for a shared one). `allocations` is ONE copy's
/// allocation sizes in the allocator's order, so the line's bytes are
/// exactly `copies * sum(allocations)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CensusLine {
    pub component: CensusComponent,
    pub copies: usize,
    pub allocations: Vec<u64>,
}

impl CensusLine {
    /// Bytes of one copy; `None` on overflow.
    pub fn copy_bytes(&self) -> Option<u64> {
        self.allocations
            .iter()
            .try_fold(0u64, |total, &bytes| total.checked_add(bytes))
    }

    /// Bytes of every copy; `None` on overflow.
    pub fn bytes(&self) -> Option<u64> {
        self.copy_bytes()?
            .checked_mul(u64::try_from(self.copies).ok()?)
    }
}

/// What the census reads of one rank/band beyond the budget.
#[derive(Clone, Copy, Debug)]
pub struct CensusInputs<'a> {
    pub config: &'a Qwen4Config,
    pub format: Qwen4StateFormat,
    /// The layers/ends this rank/band owns ([`Qwen4BandScope::single`] for a
    /// whole-model rank, EP included).
    pub scope: &'a Qwen4BandScope,
    /// Rank/band index, the bit of `owner_mask` it answers to.
    pub rank: usize,
    /// The prompt-fill append scratch a head owner holds, as `(rows,
    /// id_rows)`; `None` when none is wanted. An upper bound (the real
    /// scratch is `min(rows, forward chunk)` rows).
    pub append_scratch: Option<(usize, usize)>,
}

/// One rank/band's lane-store census.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneCensus {
    pub rank: usize,
    /// The layers the band owns (`0..layers` for a whole-model rank).
    pub band: std::ops::Range<usize>,
    pub head_owner: bool,
    pub output: bool,
    pub lines: Vec<CensusLine>,
}

impl LaneCensus {
    /// Total device bytes; `None` on overflow.
    pub fn total(&self) -> Option<u64> {
        self.lines
            .iter()
            .try_fold(0u64, |total, line| total.checked_add(line.bytes()?))
    }

    /// Bytes of `component` (`0` when the rank/band holds none).
    pub fn bytes_of(&self, component: CensusComponent) -> Option<u64> {
        self.lines
            .iter()
            .filter(|line| line.component == component)
            .try_fold(0u64, |total, line| total.checked_add(line.bytes()?))
    }

    /// Every allocation size, in line order (what a recording allocator
    /// would see per copy, repeated `copies` times).
    pub fn allocation_sizes(&self) -> Vec<u64> {
        self.lines
            .iter()
            .flat_map(|line| {
                std::iter::repeat_n(line.allocations.iter().copied(), line.copies).flatten()
            })
            .collect()
    }

    /// `component: bytes` for every non-empty line, comma-joined (MiB with
    /// two decimals), for logs and refusals.
    pub fn describe(&self) -> String {
        const MIB: f64 = (1u64 << 20) as f64;
        self.lines
            .iter()
            .filter_map(|line| Some((line.component, line.bytes()?)))
            .filter(|&(_, bytes)| bytes > 0)
            .map(|(component, bytes)| format!("{} {:.2} MiB", component.name(), bytes as f64 / MIB))
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Whether this rank/band's census fits `free` bytes after `reserve`
    /// bytes kept free. The refusal names the rank, the per-component
    /// breakdown, the need and the shortfall.
    pub fn fit(&self, free: u64, reserve: u64) -> Result<(), String> {
        let total = self
            .total()
            .ok_or_else(|| format!("rank {}: lane store census overflows", self.rank))?;
        let budget = free.saturating_sub(reserve);
        if total <= budget {
            return Ok(());
        }
        Err(format!(
            "rank {} (layers {}..{}): lane store needs {total} bytes ({}) but {free} bytes are free ({reserve} reserved; short by {})",
            self.rank,
            self.band.start,
            self.band.end,
            self.describe(),
            total - budget
        ))
    }
}

/// Every rank/band's census against its own free memory: the lane store is
/// admitted only if **each** fits ([`LaneCensus::fit`]); a larger rank's
/// spare room never lends to its peer. One refusal naming every failing
/// rank, decided before any allocation.
pub fn fit_ranks(
    censuses: &[LaneCensus],
    free_per_rank: &[u64],
    reserve: u64,
) -> Result<(), String> {
    if censuses.len() != free_per_rank.len() {
        return Err(format!(
            "lane census: {} rank censuses for {} free-memory readings",
            censuses.len(),
            free_per_rank.len()
        ));
    }
    let failures: Vec<String> = censuses
        .iter()
        .zip(free_per_rank)
        .filter_map(|(census, &free)| census.fit(free, reserve).err())
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

/// Admit and stage a lane store on every rank, all or nothing.
///
/// Every rank's census must fit its own free memory ([`fit_ranks`]) before
/// `stage` is called even once: a refusal is decided from the census alone
/// and allocates nothing. Ranks are then staged in order through `stage`; if
/// one fails, `release` frees every store already staged (newest first) and
/// that failure is returned, so no rank keeps a residual store. `Ok` holds
/// one staged store per rank.
pub fn admit_and_stage<T>(
    censuses: &[LaneCensus],
    free_per_rank: &[u64],
    reserve: u64,
    mut stage: impl FnMut(usize) -> Result<T, String>,
    mut release: impl FnMut(usize, T),
) -> Result<Vec<T>, String> {
    fit_ranks(censuses, free_per_rank, reserve)?;
    let mut staged: Vec<T> = Vec::with_capacity(censuses.len());
    for rank in 0..censuses.len() {
        match stage(rank) {
            Ok(store) => staged.push(store),
            Err(error) => {
                while let Some(store) = staged.pop() {
                    release(staged.len(), store);
                }
                return Err(error);
            }
        }
    }
    Ok(staged)
}

fn overflow(what: &str) -> String {
    format!("lane census: {what} overflows")
}

fn gdn_capture_sizes(
    config: &Qwen4Config,
    inputs: &CensusInputs<'_>,
    rows: usize,
) -> Result<Vec<u64>, String> {
    let (gdn_layers, _) = scope_layer_counts(config, inputs.scope);
    Qwen4State::row_capture_allocation_sizes_scoped(config, inputs.format.gdn, rows, gdn_layers)
        .ok_or_else(|| overflow("GDN capture ring"))
}

fn step_buffer_lines(
    config: &Qwen4Config,
    scope: &Qwen4BandScope,
    rows: usize,
    lanes: usize,
    lines: &mut Vec<CensusLine>,
) -> Result<(), String> {
    if !scope.output {
        return Ok(());
    }
    let f32_bytes = std::mem::size_of::<f32>();
    let total_rows = rows
        .checked_add(lanes)
        .ok_or_else(|| overflow("step rows"))?;
    let logits = total_rows
        .checked_mul(config.vocab_size)
        .and_then(|elements| elements.checked_mul(f32_bytes))
        .ok_or_else(|| overflow("step logits"))?;
    let top1 = total_rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| overflow("step top-1"))?;
    for (component, bytes) in [
        (CensusComponent::StepLogits, logits),
        (CensusComponent::StepTop1, top1),
    ] {
        lines.push(CensusLine {
            component,
            copies: 1,
            allocations: vec![u64::try_from(bytes).map_err(|_| overflow(component.name()))?],
        });
    }
    Ok(())
}

fn target_lines(
    config: &Qwen4Config,
    inputs: &CensusInputs<'_>,
    context: &Qwen4ContextCommit,
    lanes: usize,
    capture_rows: usize,
    mirror: StateMirror,
    lines: &mut Vec<CensusLine>,
) -> Result<(), String> {
    let state =
        Qwen4State::device_bytes_with_mirror(config, inputs.format, context, inputs.scope, mirror)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| overflow("target state"))?;
    lines.push(CensusLine {
        component: CensusComponent::TargetState,
        copies: lanes,
        allocations: vec![state],
    });
    if capture_rows > 0 {
        lines.push(CensusLine {
            component: CensusComponent::CaptureRing,
            copies: lanes,
            allocations: gdn_capture_sizes(config, inputs, capture_rows)?,
        });
    }
    Ok(())
}

fn check_rank(inputs: &CensusInputs<'_>) -> Result<(), String> {
    if inputs.rank >= MAX_OWNER_RANKS {
        return Err(format!(
            "lane census: rank {} is past the {MAX_OWNER_RANKS}-rank owner mask",
            inputs.rank
        ));
    }
    Ok(())
}

/// The census of an AR lane store on one rank/band: `lanes` target states
/// ([`AR_LANE_MIRROR`]: no snapshot mirror; no capture ring, no head) and, on
/// an LM-head band, the step logits and top-1 of `lanes` decode rows plus
/// `lanes` prompt-completing rows.
pub fn ar_lane_census(
    lanes: usize,
    context: &Qwen4ContextCommit,
    inputs: &CensusInputs<'_>,
) -> Result<LaneCensus, String> {
    check_rank(inputs)?;
    let config = inputs.config;
    let mut lines = Vec::new();
    target_lines(
        config,
        inputs,
        context,
        lanes,
        0,
        AR_LANE_MIRROR,
        &mut lines,
    )?;
    step_buffer_lines(config, inputs.scope, lanes, lanes, &mut lines)?;
    Ok(LaneCensus {
        rank: inputs.rank,
        band: inputs.scope.band.clone(),
        head_owner: false,
        output: inputs.scope.output,
        lines,
    })
}

/// The census of a speculative lane store on one rank/band under `budget`.
///
/// Every owning band charges its lanes' target state (with the
/// [`SPEC_LANE_MIRROR`] snapshot mirror) and capture ring; a head owner
/// (which must hold the LM head) charges the head-side families
/// and the wide hidden; a band that runs the LM head charges the step
/// logits/top-1 over `step_rows() + lanes` rows. A head owner that does not
/// run the LM head is refused, as is a rank outside the owner mask width.
pub fn lane_spec_census(
    budget: &LaneSpecBudget,
    inputs: &CensusInputs<'_>,
) -> Result<LaneCensus, String> {
    check_rank(inputs)?;
    let config = inputs.config;
    let head_owner = budget.owns_head(inputs.rank);
    if head_owner && !inputs.scope.output {
        return Err(format!(
            "lane census: rank {} is a head owner but its band {}..{} does not run the LM head",
            inputs.rank, inputs.scope.band.start, inputs.scope.band.end
        ));
    }
    let lanes = budget.lanes;
    let f32_bytes = std::mem::size_of::<f32>();
    let mut lines = Vec::new();
    target_lines(
        config,
        inputs,
        &budget.context_commit,
        lanes,
        budget.capture_rows_per_lane,
        SPEC_LANE_MIRROR,
        &mut lines,
    )?;
    if head_owner {
        let wide = config
            .hc_count
            .checked_mul(config.hidden_size)
            .and_then(|elements| elements.checked_mul(f32_bytes))
            .ok_or_else(|| overflow("wide hidden row"))?;
        let wide_u64 = u64::try_from(wide).map_err(|_| overflow("wide hidden row"))?;
        lines.push(CensusLine {
            component: CensusComponent::HeadState,
            copies: lanes,
            allocations: MtpGpuState::allocation_sizes(config, &budget.context_commit)
                .ok_or_else(|| overflow("head state"))?,
        });
        lines.push(CensusLine {
            component: CensusComponent::PendingHidden,
            copies: lanes,
            allocations: vec![wide_u64],
        });
        if budget.head_lag_rows > 0 {
            // Owed hidden rows only: the lane's owed tokens are host values.
            // The list is the allocator's own (`HeadLagStore::new`).
            lines.push(CensusLine {
                component: CensusComponent::HeadLag,
                copies: lanes,
                allocations: crate::mesh_rank_copy::lag_allocation_sizes(
                    config,
                    budget.head_lag_rows,
                )
                .ok_or_else(|| overflow("head-lag hidden"))?,
            });
        }
    }
    step_buffer_lines(config, inputs.scope, budget.step_rows(), lanes, &mut lines)?;
    if head_owner {
        let wide = config
            .hc_count
            .checked_mul(config.hidden_size)
            .ok_or_else(|| overflow("wide hidden row"))?;
        let hidden = budget
            .step_rows()
            .checked_mul(wide)
            .and_then(|elements| elements.checked_mul(f32_bytes))
            .ok_or_else(|| overflow("wide hidden capture"))?;
        let top1 = lanes
            .checked_mul(Gpu::TOPK8_OUT_BYTES)
            .ok_or_else(|| overflow("draft top-1"))?;
        lines.push(CensusLine {
            component: CensusComponent::WideHidden,
            copies: 1,
            allocations: vec![u64::try_from(hidden).map_err(|_| overflow("wide hidden capture"))?],
        });
        lines.push(CensusLine {
            component: CensusComponent::DraftTop1,
            copies: 1,
            allocations: vec![u64::try_from(top1).map_err(|_| overflow("draft top-1"))?],
        });
        if let Some((rows, id_rows)) = inputs.append_scratch {
            lines.push(CensusLine {
                component: CensusComponent::AppendScratch,
                copies: 1,
                allocations: MtpAppendScratch::allocation_sizes(config, rows, id_rows)
                    .ok_or_else(|| overflow("append scratch"))?,
            });
        }
        if budget.sampling_scratch > 0 {
            lines.push(CensusLine {
                component: CensusComponent::SamplingScratch,
                copies: 1,
                allocations: vec![budget.sampling_scratch],
            });
        }
    }
    Ok(LaneCensus {
        rank: inputs.rank,
        band: inputs.scope.band.clone(),
        head_owner,
        output: inputs.scope.output,
        lines,
    })
}

/// The census of a mixed store from the MTP store's `census` over
/// `lanes` lanes: only `spec_lanes` slots carry the per-lane MTP families
/// (capture ring, head state, pending hidden, owed rows) and the
/// snapshot-mirrored target state; the remaining `lanes - spec_lanes`
/// AR-only slots hold one target state of `ar_state_bytes` (no mirror) and
/// nothing else. Shared lines (step buffers, wide hidden, draft slots, append
/// scratch) are the MTP store's, unchanged. `None` when `spec_lanes` exceeds
/// `lanes` or the census was not built for `lanes` lanes.
pub(crate) fn mixed_lane_census(
    mut census: LaneCensus,
    lanes: usize,
    spec_lanes: usize,
    ar_state_bytes: u64,
) -> Option<LaneCensus> {
    if spec_lanes > lanes {
        return None;
    }
    let ar_lanes = lanes - spec_lanes;
    let mut ar_state = None;
    for line in &mut census.lines {
        match line.component {
            CensusComponent::TargetState => {
                if line.copies != lanes {
                    return None;
                }
                line.copies = spec_lanes;
                ar_state = Some(CensusLine {
                    component: CensusComponent::TargetState,
                    copies: ar_lanes,
                    allocations: vec![ar_state_bytes],
                });
            }
            CensusComponent::CaptureRing
            | CensusComponent::HeadState
            | CensusComponent::PendingHidden
            | CensusComponent::HeadLag => {
                if line.copies != lanes {
                    return None;
                }
                line.copies = spec_lanes;
            }
            _ => {}
        }
    }
    census.lines.extend(ar_state);
    Some(census)
}

/// The census of a copy store whose capture ring is wide only on its first
/// `wide_lanes` MTP slots: `census` is the widened budget's, `narrow` the
/// legacy (`k + 1`-row ring) budget's over the same lanes and band. Every
/// `CaptureRing` line (its `copies` already reduced to the MTP slots) splits
/// into `min(wide_lanes, copies)` wide copies and the rest at the narrow
/// allocation, exactly as `Qwen4LaneStore::admit` sizes each slot. `None`
/// when the narrow census has no ring line to price the narrow slots from.
pub(crate) fn split_capture_ring(
    mut census: LaneCensus,
    narrow: &LaneCensus,
    wide_lanes: usize,
) -> Option<LaneCensus> {
    let narrow_line = narrow
        .lines
        .iter()
        .find(|l| l.component == CensusComponent::CaptureRing)?;
    let mut extra = Vec::new();
    for line in &mut census.lines {
        if line.component != CensusComponent::CaptureRing || line.copies <= wide_lanes {
            continue;
        }
        extra.push(CensusLine {
            component: CensusComponent::CaptureRing,
            copies: line.copies - wide_lanes,
            allocations: narrow_line.allocations.clone(),
        });
        line.copies = wide_lanes;
    }
    census.lines.extend(extra);
    Some(census)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::kv_backend::Qwen4KvBackend;
    use crate::state::Qwen4State;
    use crate::{GdnStateFormat, QsaKvFormat};

    fn format_of(gdn: GdnStateFormat) -> Qwen4StateFormat {
        Qwen4StateFormat {
            qsa: QsaKvFormat::Fp8,
            gdn,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
            gdn_stream: crate::GdnStreamKind::None,
        }
    }

    const FORMATS: [GdnStateFormat; 3] = [
        GdnStateFormat::F32,
        GdnStateFormat::Q8,
        GdnStateFormat::Q8Ef,
    ];

    fn contexts() -> Vec<Qwen4ContextCommit> {
        vec![
            Qwen4ContextCommit::legacy(4096),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 1024, 2 << 20),
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 65536, 4096, 2 << 20),
        ]
    }

    fn partition(config: &Qwen4Config, counts: &[usize]) -> Vec<Qwen4BandScope> {
        assert_eq!(counts.iter().sum::<usize>(), config.num_hidden_layers);
        let mut start = 0;
        counts
            .iter()
            .map(|&count| {
                let band = start..start + count;
                start += count;
                let scope = Qwen4BandScope {
                    input: band.start == 0,
                    output: band.end == config.num_hidden_layers,
                    band,
                };
                scope.validate(config).expect("valid band");
                scope
            })
            .collect()
    }

    fn inputs<'a>(
        config: &'a Qwen4Config,
        gdn: GdnStateFormat,
        scope: &'a Qwen4BandScope,
        rank: usize,
    ) -> CensusInputs<'a> {
        CensusInputs {
            config,
            format: format_of(gdn),
            scope,
            rank,
            append_scratch: Some((1024, 1024)),
        }
    }

    fn sum(sizes: &[u64]) -> u64 {
        sizes.iter().sum()
    }

    /// The census is the sum of the allocation components the allocators'
    /// own inventories list, family by family, for every state format,
    /// context commit, ownership split and optional charge.
    #[test]
    fn census_equals_the_sum_of_allocation_components() {
        let config = compact_test_config();
        let wide = (config.hc_count * config.hidden_size * 4) as u64;
        let scopes = {
            let mut all = vec![vec![Qwen4BandScope::single(&config)]];
            all.push(partition(&config, &[25, 23]));
            all.push(partition(&config, &[16, 16, 16]));
            all
        };
        for gdn in FORMATS {
            for context in contexts() {
                for lanes in [1usize, 2, 8] {
                    for budget in [
                        LaneSpecBudget::legacy(lanes, 3, context, 1).unwrap(),
                        LaneSpecBudget::default_copy(lanes, 3, 8, context, 1).unwrap(),
                        LaneSpecBudget::default_copy(lanes, 1, 8, context, 1)
                            .unwrap()
                            .with_head_lag_rows(16)
                            .with_sampling_scratch(7 << 20),
                    ] {
                        for bands in &scopes {
                            let head_rank = bands.iter().position(|b| b.output).unwrap();
                            let budget = LaneSpecBudget {
                                owner_mask: 1 << head_rank,
                                ..budget
                            };
                            for (rank, scope) in bands.iter().enumerate() {
                                let inputs = inputs(&config, gdn, scope, rank);
                                let census = lane_spec_census(&budget, &inputs).unwrap();
                                let head = rank == head_rank;
                                assert_eq!(census.head_owner, head);
                                let (gdn_layers, _) = scope_layer_counts(&config, scope);
                                let n = lanes as u64;
                                let mut expected = 0u64;
                                // Target state per lane, ring per lane.
                                let state = Qwen4State::device_bytes_scoped(
                                    &config,
                                    format_of(gdn),
                                    &context,
                                    scope,
                                )
                                .unwrap() as u64;
                                assert_eq!(
                                    census.bytes_of(CensusComponent::TargetState),
                                    Some(n * state)
                                );
                                expected += n * state;
                                let ring = Qwen4State::row_capture_allocation_sizes_scoped(
                                    &config,
                                    gdn,
                                    budget.capture_rows_per_lane,
                                    gdn_layers,
                                )
                                .unwrap();
                                assert_eq!(
                                    sum(&ring),
                                    Qwen4State::row_capture_bytes_scoped(
                                        &config,
                                        gdn,
                                        budget.capture_rows_per_lane,
                                        gdn_layers
                                    )
                                    .unwrap() as u64
                                );
                                let ring_line = census
                                    .lines
                                    .iter()
                                    .find(|l| l.component == CensusComponent::CaptureRing)
                                    .unwrap();
                                assert_eq!(ring_line.allocations, ring);
                                assert_eq!(ring_line.copies, lanes);
                                expected += n * sum(&ring);
                                // Step buffers on the LM-head band only.
                                let rows = budget.step_rows() as u64 + n;
                                let logits = rows * config.vocab_size as u64 * 4;
                                let top1 = rows * 4;
                                if scope.output {
                                    assert_eq!(
                                        census.bytes_of(CensusComponent::StepLogits),
                                        Some(logits)
                                    );
                                    assert_eq!(
                                        census.bytes_of(CensusComponent::StepTop1),
                                        Some(top1)
                                    );
                                    expected += logits + top1;
                                } else {
                                    assert_eq!(
                                        census.bytes_of(CensusComponent::StepLogits),
                                        Some(0)
                                    );
                                    assert_eq!(census.bytes_of(CensusComponent::StepTop1), Some(0));
                                }
                                if head {
                                    let head_state =
                                        MtpGpuState::allocation_sizes(&config, &context).unwrap();
                                    assert_eq!(
                                        sum(&head_state),
                                        MtpGpuState::device_bytes(&config, &context).unwrap()
                                            as u64
                                    );
                                    expected += n * sum(&head_state) + n * wide;
                                    let lag = budget.head_lag_rows as u64;
                                    expected += n * lag * wide;
                                    expected += budget.step_rows() as u64 * wide;
                                    expected += n * Gpu::TOPK8_OUT_BYTES as u64;
                                    let append =
                                        MtpAppendScratch::allocation_sizes(&config, 1024, 1024)
                                            .unwrap();
                                    assert_eq!(
                                        census.bytes_of(CensusComponent::AppendScratch),
                                        Some(sum(&append))
                                    );
                                    expected += sum(&append);
                                    expected += budget.sampling_scratch;
                                }
                                assert_eq!(
                                    census.total(),
                                    Some(expected),
                                    "{gdn:?} {context:?} lanes={lanes} rank={rank}"
                                );
                                assert_eq!(sum(&census.allocation_sizes()), expected);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The in-place state a snapshot copies, from the config alone: per GDN
    /// layer the recurrent state and conv history, per QSA layer the partial
    /// K/V, the selection and the raw-index circular block, and on the input
    /// band the PLE conv history and hyper feedback. Independent of
    /// `Qwen4State`'s own accounting.
    fn mirror_bytes(config: &Qwen4Config, format: Qwen4StateFormat, scope: &Qwen4BandScope) -> u64 {
        let f32_bytes = 4usize;
        let (gdn_layers, qsa_layers) = scope_layer_counts(config, scope);
        let recurrent = format.gdn.state_bytes(
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        );
        let conv_channels = config.linear_num_key_heads * config.linear_key_head_dim * 2
            + config.linear_num_value_heads * config.linear_value_head_dim;
        let conv = conv_channels * (config.linear_conv_kernel_dim - 1) * f32_bytes;
        let ratio = config.indexer_compress_ratio;
        let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
        let full_width = config.num_key_value_heads * config.head_dim;
        let qsa = ratio * raw_width * f32_bytes
            + ratio * full_width * f32_bytes
            + config.qsa_selected_capacity() * std::mem::size_of::<i32>()
            + ratio * raw_width * format.layout().index_dtype().size();
        let per_request = if scope.input {
            (config.ple_conv_history_rows() * config.hidden_size * config.hc_count
                + config.hc_count * config.hidden_size)
                * f32_bytes
        } else {
            0
        };
        (gdn_layers * (recurrent + conv) + qsa_layers * qsa + per_request) as u64
    }

    /// An AR lane store charges every lane's live state alone: the census
    /// target term drops by exactly the mirror bytes (computed from the
    /// config's in-place parts), per lane and per band, for every state
    /// format, context commit and ownership split; no other line moves; the
    /// store's own byte query is the census; and the spec census keeps the
    /// mirror on every band.
    #[test]
    fn ar_lane_census_drops_exactly_the_snapshot_mirror_bytes() {
        use super::super::Qwen4LaneStore;
        let config = compact_test_config();
        assert_eq!(AR_LANE_MIRROR, StateMirror::Absent);
        assert_eq!(SPEC_LANE_MIRROR, StateMirror::Snapshot);
        assert_eq!(Qwen4LaneStore::state_mirror(None), AR_LANE_MIRROR);
        let spec = crate::lane::LaneSpecConfig {
            k: 3,
            end_of_turn: None,
            spec_lanes: None,
        };
        assert_eq!(Qwen4LaneStore::state_mirror(Some(spec)), SPEC_LANE_MIRROR);
        // A mixed store's AR-only slots (past `spec_lanes`) are keyed per slot
        // at `new_payload`, by whether the slot carries a spec payload.
        assert_eq!(Qwen4LaneStore::state_mirror(None), AR_LANE_MIRROR);
        let scopes = [
            vec![Qwen4BandScope::single(&config)],
            partition(&config, &[25, 23]),
            partition(&config, &[16, 16, 16]),
        ];
        for gdn in FORMATS {
            let format = format_of(gdn);
            for context in contexts() {
                for bands in &scopes {
                    for (rank, scope) in bands.iter().enumerate() {
                        let mirror = mirror_bytes(&config, format, scope);
                        assert!(mirror > 0, "{gdn:?} rank={rank}");
                        let snapshot_state =
                            Qwen4State::device_bytes_scoped(&config, format, &context, scope)
                                .unwrap() as u64;
                        for lanes in [1usize, 2, 8] {
                            let n = lanes as u64;
                            let input = inputs(&config, gdn, scope, rank);
                            let ar = ar_lane_census(lanes, &context, &input).unwrap();
                            let target = ar.bytes_of(CensusComponent::TargetState).unwrap();
                            assert_eq!(
                                target,
                                n * (snapshot_state - mirror),
                                "{gdn:?} {context:?} lanes={lanes} rank={rank}"
                            );
                            assert_eq!(n * snapshot_state - target, n * mirror);
                            // The step buffers are the only other lines (LM-head
                            // band): rows = lanes decode + lanes prompt rows.
                            let step = if scope.output {
                                2 * n * (config.vocab_size as u64 * 4 + 4)
                            } else {
                                0
                            };
                            assert_eq!(ar.total(), Some(target + step));
                            assert_eq!(sum(&ar.allocation_sizes()), target + step);
                            assert_eq!(ar.bytes_of(CensusComponent::CaptureRing), Some(0));
                            // One byte source: the store's own query is the census.
                            assert_eq!(
                                Qwen4LaneStore::device_bytes_scoped(
                                    &config, format, &context, lanes, lanes, None, scope,
                                ),
                                ar.total()
                            );
                            // The spec census keeps the mirror on every band.
                            let budget =
                                LaneSpecBudget::legacy(lanes, 3, context, 1 << (bands.len() - 1))
                                    .unwrap();
                            let spec_census = lane_spec_census(&budget, &input).unwrap();
                            assert_eq!(
                                spec_census.bytes_of(CensusComponent::TargetState),
                                Some(n * snapshot_state),
                                "{gdn:?} {context:?} lanes={lanes} rank={rank}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The Flash-Next EP2 geometry (QSA fp8, GDN q8, 16384 context, the
    /// whole model on a rank) at the 8-lane AR store: the mirror bytes the
    /// store no longer charges, exactly, against the census of the state.
    #[test]
    fn ar_lane_store_flash_next_saving_is_the_mirror_of_eight_states() {
        let config = compact_test_config();
        let single = Qwen4BandScope::single(&config);
        let context = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 4096, 2 << 20);
        for gdn in [
            GdnStateFormat::F32,
            GdnStateFormat::Q8,
            GdnStateFormat::Q8Ef,
        ] {
            let format = format_of(gdn);
            let input = inputs(&config, gdn, &single, 0);
            let ar = ar_lane_census(8, &context, &input).unwrap();
            let with_mirror =
                8 * Qwen4State::device_bytes_scoped(&config, format, &context, &single).unwrap()
                    as u64;
            let target = ar.bytes_of(CensusComponent::TargetState).unwrap();
            let saving = with_mirror - target;
            assert_eq!(
                saving,
                8 * mirror_bytes(&config, format, &single),
                "{gdn:?}"
            );
            eprintln!(
                "{gdn:?}: per-lane mirror {} bytes, 8-lane saving {saving} bytes = {:.2} MiB",
                saving / 8,
                saving as f64 / (1u64 << 20) as f64
            );
        }
    }

    /// The legacy store census is exactly what the formula
    /// `Qwen4LaneStore::device_bytes_scoped` carried before the census, with
    /// each lane state carrying the snapshot mirror only under `spec_k`:
    /// the only changes a caller can see are a non-output band's spec answer
    /// (target-only bytes instead of `None`) and an AR lane's mirror.
    #[test]
    fn legacy_store_bytes_match_the_prior_formula() {
        use super::super::Qwen4LaneStore;
        let config = compact_test_config();
        let f32_bytes = 4usize;
        for gdn in FORMATS {
            let format = format_of(gdn);
            for context in contexts() {
                for scope in [
                    Qwen4BandScope::single(&config),
                    partition(&config, &[24, 24]).pop().unwrap(),
                ] {
                    for lanes in [1usize, 2, 8] {
                        for spec_k in [None, Some(1usize), Some(3), Some(7)] {
                            let mirror = spec_k.map_or(AR_LANE_MIRROR, |_| SPEC_LANE_MIRROR);
                            let mut per_lane = Qwen4State::device_bytes_with_mirror(
                                &config, format, &context, &scope, mirror,
                            )
                            .unwrap();
                            let rows_cap = match spec_k {
                                Some(k) => lanes * (k + 1),
                                None => lanes,
                            };
                            let rows = rows_cap + lanes;
                            let mut shared = rows * config.vocab_size * f32_bytes
                                + rows * std::mem::size_of::<i32>();
                            if let Some(k) = spec_k {
                                let wide = config.hc_count * config.hidden_size;
                                let (gdn_layers, _) = scope_layer_counts(&config, &scope);
                                let ring = Qwen4State::row_capture_bytes_scoped(
                                    &config,
                                    format.gdn,
                                    k + 1,
                                    gdn_layers,
                                )
                                .unwrap();
                                per_lane += MtpGpuState::device_bytes(&config, &context).unwrap()
                                    + ring
                                    + wide * f32_bytes;
                                let append = if crate::mtp_spec::mtp_fill_scratch_requested() {
                                    MtpAppendScratch::device_bytes(
                                        &config,
                                        crate::mtp_gpu::MTP_FILL_ROWS,
                                        crate::mtp_gpu::MTP_FILL_ROWS,
                                    )
                                    .unwrap_or(0)
                                } else {
                                    0
                                };
                                shared += rows_cap * wide * f32_bytes
                                    + lanes * Gpu::TOPK8_OUT_BYTES
                                    + append;
                            }
                            let expected = (lanes * per_lane + shared) as u64;
                            let got = Qwen4LaneStore::device_bytes_scoped(
                                &config, format, &context, lanes, lanes, spec_k, &scope,
                            );
                            assert_eq!(
                                got,
                                Some(expected),
                                "{gdn:?} {context:?} {lanes} {spec_k:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A PP non-output band holds target state and capture ring for its own
    /// layers and nothing else: no head state, no logits/top-1, no wide
    /// hidden, no draft slots; a head claimed there is refused.
    #[test]
    fn pp_non_output_band_has_no_head_bytes() {
        let config = compact_test_config();
        let bands = partition(&config, &[25, 23]);
        let budget = LaneSpecBudget::default_copy(
            8,
            3,
            8,
            Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 1024, 2 << 20),
            0b10,
        )
        .unwrap()
        .with_head_lag_rows(8)
        .with_sampling_scratch(1 << 20);
        let input = inputs(&config, GdnStateFormat::Q8, &bands[0], 0);
        let first = lane_spec_census(&budget, &input).unwrap();
        assert!(!first.head_owner && !first.output);
        for component in [
            CensusComponent::HeadState,
            CensusComponent::PendingHidden,
            CensusComponent::HeadLag,
            CensusComponent::StepLogits,
            CensusComponent::StepTop1,
            CensusComponent::WideHidden,
            CensusComponent::DraftTop1,
            CensusComponent::AppendScratch,
            CensusComponent::SamplingScratch,
        ] {
            assert_eq!(first.bytes_of(component), Some(0), "{component:?}");
        }
        assert_eq!(
            first.lines.iter().map(|l| l.component).collect::<Vec<_>>(),
            [CensusComponent::TargetState, CensusComponent::CaptureRing]
        );
        let output =
            lane_spec_census(&budget, &inputs(&config, GdnStateFormat::Q8, &bands[1], 1)).unwrap();
        assert!(output.head_owner && output.output);
        assert!(output.bytes_of(CensusComponent::HeadState).unwrap() > 0);
        assert!(output.bytes_of(CensusComponent::StepLogits).unwrap() > 0);
        // The store byte query agrees on the band's target-only bytes.
        assert_eq!(
            super::super::Qwen4LaneStore::device_bytes_scoped(
                &config,
                format_of(GdnStateFormat::Q8),
                &budget.context_commit,
                8,
                8,
                Some(3),
                &bands[0]
            ),
            lane_spec_census(
                &LaneSpecBudget::legacy(8, 3, budget.context_commit, 0b10).unwrap(),
                &inputs(&config, GdnStateFormat::Q8, &bands[0], 0)
            )
            .unwrap()
            .total()
        );
        // A head owner on a band without the LM head is refused.
        let wrong = LaneSpecBudget {
            owner_mask: 0b01,
            ..budget
        };
        let err = lane_spec_census(&wrong, &inputs(&config, GdnStateFormat::Q8, &bands[0], 0))
            .unwrap_err();
        assert!(err.contains("does not run the LM head"), "{err}");
    }

    #[test]
    fn budget_derivations_and_refusals() {
        let ctx = Qwen4ContextCommit::legacy(4096);
        let default = LaneSpecBudget::default_copy(8, 1, 8, ctx, 0b11).unwrap();
        assert_eq!(
            (
                default.copy_take_cap,
                default.verify_rows_per_lane,
                default.capture_rows_per_lane,
                default.proposal_n_max,
                default.step_rows()
            ),
            (7, 8, 8, COPY_PROPOSAL_N_MAX, 64)
        );
        assert!(default.owns_head(0) && default.owns_head(1) && !default.owns_head(2));
        let narrow = LaneSpecBudget::new(8, 3, 0, 8, ctx, 1).unwrap();
        assert_eq!(
            (narrow.verify_rows_per_lane, narrow.capture_rows_per_lane),
            (4, 8)
        );
        let legacy = LaneSpecBudget::legacy(8, 3, ctx, 1).unwrap();
        assert_eq!(
            (legacy.verify_rows_per_lane, legacy.capture_rows_per_lane),
            (4, 4)
        );
        assert_eq!(
            LaneSpecBudget::new(1, 3, 0, 8, ctx, 1)
                .unwrap()
                .at_full_context()
                .context_commit
                .tokens,
            4096
        );
        for bad in [
            LaneSpecBudget::new(0, 3, 0, 8, ctx, 1),
            LaneSpecBudget::new(MAX_LANES + 1, 3, 0, 8, ctx, 1),
            LaneSpecBudget::new(8, 0, 0, 8, ctx, 1),
            LaneSpecBudget::new(8, LANE_MAX_ROWS, 0, 8, ctx, 1),
            // A 64-row lane is never admitted by lifting the eight-row guard.
            LaneSpecBudget::new(8, 3, 63, 8, ctx, 1),
            LaneSpecBudget::new(8, 3, 0, 8, ctx, 0),
            LaneSpecBudget::new(8, 3, 0, hipfire_config::QWEN4_VERIFY_MAX_ROWS + 1, ctx, 1),
        ] {
            assert!(bad.is_err(), "{bad:?}");
        }
    }

    /// Overflow is a refusal decided before any allocation: no figure wraps
    /// or saturates, in the budget, any census family or the fit.
    #[test]
    fn overflow_refuses_before_allocation() {
        let config = compact_test_config();
        let scope = Qwen4BandScope::single(&config);
        let ctx = Qwen4ContextCommit::legacy(4096);
        let input = inputs(&config, GdnStateFormat::Q8, &scope, 0);
        let budget = LaneSpecBudget::default_copy(8, 3, 8, ctx, 1).unwrap();
        for bad in [
            budget.with_head_lag_rows(usize::MAX),
            budget.with_head_lag_rows(usize::MAX / 4),
        ] {
            assert!(lane_spec_census(&bad, &input).is_err(), "{bad:?}");
        }
        let huge = LaneSpecBudget {
            context_commit: Qwen4ContextCommit::legacy(usize::MAX),
            ..budget
        };
        assert!(lane_spec_census(&huge, &input).is_err());
        assert!(ar_lane_census(usize::MAX, &ctx, &input).map_or(true, |c| c.total().is_none()));
        let wide_rank = CensusInputs {
            rank: MAX_OWNER_RANKS,
            ..input
        };
        assert!(lane_spec_census(&budget, &wide_rank).is_err());
        let census = lane_spec_census(&budget, &input).unwrap();
        let total = census.total().unwrap();
        assert!(census.fit(total, 0).is_ok());
        assert!(census.fit(total, 1).is_err());
        assert!(census.fit(total - 1, 0).is_err());
        let overflowed = LaneCensus {
            lines: vec![CensusLine {
                component: CensusComponent::TargetState,
                copies: 2,
                allocations: vec![u64::MAX],
            }],
            ..census
        };
        assert_eq!(overflowed.total(), None);
        assert!(overflowed
            .fit(u64::MAX, 0)
            .unwrap_err()
            .contains("overflows"));
    }

    /// Each rank admits against its OWN free memory; the refusal names every
    /// failing rank with its byte breakdown, and the staging callback is
    /// never reached.
    #[test]
    fn rank_admission_refuses_before_staging_and_rolls_back() {
        let config = compact_test_config();
        let scope = Qwen4BandScope::single(&config);
        let ctx = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 4096, 2 << 20);
        let budget = LaneSpecBudget::default_copy(8, 3, 8, ctx, 0b11).unwrap();
        let censuses: Vec<LaneCensus> = (0..2)
            .map(|rank| {
                lane_spec_census(&budget, &inputs(&config, GdnStateFormat::Q8, &scope, rank))
                    .unwrap()
            })
            .collect();
        let need = censuses[0].total().unwrap();
        assert_eq!(censuses[0].total(), censuses[1].total());

        // Rank 1 alone is short: a larger rank 0 does not lend to it.
        let staged = std::cell::Cell::new(0);
        let released = std::cell::RefCell::new(Vec::new());
        let err = admit_and_stage(
            &censuses,
            &[need * 10, need - 1],
            0,
            |rank| {
                staged.set(staged.get() + 1);
                Ok(rank)
            },
            |rank, _| released.borrow_mut().push(rank),
        )
        .unwrap_err();
        assert!(
            err.starts_with("rank 1 ") && !err.contains("rank 0"),
            "{err}"
        );
        assert!(
            err.contains("GDN capture ring") && err.contains("short by 1"),
            "{err}"
        );
        assert_eq!(staged.get(), 0, "a refusal allocates nothing");

        // Both short: one message names both.
        let both = fit_ranks(&censuses, &[0, 0], 0).unwrap_err();
        assert!(
            both.contains("rank 0 ") && both.contains("rank 1 "),
            "{both}"
        );
        assert!(fit_ranks(&censuses, &[0], 0).is_err());

        // Admitted, but the second rank's allocation fails: the first is
        // released and no store remains.
        let result = admit_and_stage(
            &censuses,
            &[need, need],
            0,
            |rank| {
                staged.set(staged.get() + 1);
                if rank == 1 {
                    Err("rank 1: hipMalloc failed".to_string())
                } else {
                    Ok(rank)
                }
            },
            |rank, store| {
                assert_eq!(rank, store);
                released.borrow_mut().push(rank)
            },
        );
        assert_eq!(result.unwrap_err(), "rank 1: hipMalloc failed");
        assert_eq!(staged.get(), 2);
        assert_eq!(*released.borrow(), vec![0]);
        let ok = admit_and_stage(&censuses, &[need, need], 0, Ok, |_, _: usize| {}).unwrap();
        assert_eq!(ok, vec![0, 1]);
    }

    /// DESIGN §6's scale check, from the real allocation helper: per capture row
    /// per whole-model Q8 lane (36 GDN layers) about 2.85 MiB of per-row inputs
    /// plus `slots / (36 * rows)` Q8 state sets of 27.84 MiB, where `slots` is the
    /// state's ring slots across its 36 layers: `2 * 35 + rows + 1` in the shared
    /// pool (default), `36 * (rows + 1)` per-layer with
    /// `HIPFIRE_QWEN4_LANE_RESERVE_TRIM=0`, `72 * rows` with the compact kill switch.
    #[test]
    fn capture_ring_scale_matches_the_design_check() {
        use crate::state::{capture_ring_slots, current_capture_ring_total_slots};
        let config = compact_test_config();
        let gdn_layers = config.n_linear_layers();
        assert_eq!(gdn_layers, 36);
        let set_mib = (gdn_layers * 811_008) as f64 / (1u64 << 20) as f64;
        for rows in [2usize, 4, 8] {
            assert_eq!(capture_ring_slots(false, rows), Some(2 * rows));
            let slots = current_capture_ring_total_slots(rows, gdn_layers).unwrap();
            let bytes =
                Qwen4State::row_capture_bytes_scoped(&config, GdnStateFormat::Q8, rows, gdn_layers)
                    .unwrap();
            let per_row_mib = bytes as f64 / rows as f64 / (1u64 << 20) as f64;
            let expected = 2.85 + slots as f64 / gdn_layers as f64 / rows as f64 * set_mib;
            assert!(
                (per_row_mib - expected).abs() < 0.15,
                "rows={rows} slots={slots}: {per_row_mib} MiB/row, expected {expected}"
            );
        }
    }

    /// `HIPFIRE_QWEN4_LANE_RESERVE_TRIM` at the shipped EP2 serve shape (K5 = the landed depth
    /// cap, so a 6-row verify window / capture ring per lane; 8 lanes; a 48-row all-rows lane
    /// step; symtrunk's 48 layers = 36 GDN + 12 QSA): the capture ring of a lane before (one
    /// `rows + 1`-slot ring per GDN layer) and after (one shared pool), per format, and every
    /// buffer of the trimmed layout still covers its maximum use.
    #[test]
    fn lane_reserve_trim_sizes_and_covers_the_depth_cap_five_c8_step() {
        use crate::state::{capture_ring_slot, capture_view_start, CAPTURE_POOL_STRIDE};
        let config = compact_test_config();
        let (lanes, k) = (8usize, 5usize);
        let rows = k + 1;
        let gdn_layers = config.n_linear_layers();
        assert_eq!((config.num_hidden_layers, gdn_layers), (48, 36));
        let vh = config.linear_num_value_heads;
        let qkv = 2 * config.linear_num_key_heads * config.linear_key_head_dim
            + vh * config.linear_value_head_dim;
        for gdn in FORMATS {
            let layout = |compact: bool, trim: bool| {
                Qwen4State::row_capture_allocation_sizes_layout(
                    &config, gdn, rows, gdn_layers, compact, trim,
                )
                .unwrap()
            };
            let (before, after) = (layout(true, false), layout(true, true));
            let slot = gdn.storage().state_bytes(vh, 128, 128) as u64;
            let inputs = (rows * qkv * 4) as u64;
            let recurrence = (rows * (qkv + 2 * vh) * 4) as u64;
            let output = (rows * vh * 128 * 4) as u64;
            let table = (gdn_layers * 16) as u64;
            // Kill switch (`=0`): the previous per-layer layout, allocation for allocation.
            assert_eq!(before.len(), 3 * gdn_layers + 2);
            for layer in before[..3 * gdn_layers].chunks(3) {
                assert_eq!(layer, [(rows as u64 + 1) * slot, inputs, recurrence]);
            }
            assert_eq!(before[3 * gdn_layers..], [output, table]);
            // Default: one pool, then (inputs, recurrence) per layer.
            let pool_slots = (CAPTURE_POOL_STRIDE * (gdn_layers - 1) + rows + 1) as u64;
            assert_eq!(after.len(), 2 * gdn_layers + 3);
            assert_eq!(after[0], pool_slots * slot);
            for layer in after[1..1 + 2 * gdn_layers].chunks(2) {
                assert_eq!(layer, [inputs, recurrence]);
            }
            assert_eq!(after[1 + 2 * gdn_layers..], [output, table]);
            let (before_bytes, after_bytes) = (sum(&before), sum(&after));
            // The trim removes exactly the slots no layer's window can touch.
            assert_eq!(
                before_bytes - after_bytes,
                (gdn_layers as u64 * (rows as u64 + 1) - pool_slots) * slot
            );
            if gdn.storage() == GdnStateFormat::Q8 {
                // 36 x (7 x 811008 + 245760 + 248064) + 147456 + 576 = 212.00 MiB per lane
                // before; one 77-slot pool + the inputs = 76.65 MiB after: -135.35 MiB/lane.
                assert_eq!(before_bytes, 222_299_712);
                assert_eq!(after_bytes, 80_373_312);
                assert_eq!(before_bytes - after_bytes, 175 * 811_008);
            }
            // Every view of every layer, for every block length a K5 window (2..=6 rows,
            // rollback 1..=5 kept rows) and both halves, ends inside the pool.
            let max_end = (0..gdn_layers)
                .flat_map(|layer| {
                    [0usize, rows].into_iter().flat_map(move |base| {
                        (1..=rows).map(move |n| {
                            CAPTURE_POOL_STRIDE * layer
                                + capture_view_start(true, rows, base, n)
                                + n
                        })
                    })
                })
                .max()
                .unwrap() as u64;
            assert!(after[0] >= max_end * slot, "pool covers every view");
            assert_eq!(max_end, pool_slots, "and nothing more");
            // The two write targets of a layer are the only slots touched; no two layers share
            // one.
            let mut touched = std::collections::HashSet::new();
            for layer in 0..gdn_layers {
                for logical in [0usize, rows] {
                    let slot = CAPTURE_POOL_STRIDE * layer + capture_ring_slot(true, rows, logical);
                    assert!(
                        touched.insert(slot),
                        "layer {layer} target {slot} is shared"
                    );
                }
            }
            assert_eq!(touched.len(), 2 * gdn_layers);
            // The per-use buffers keep their per-window size (a 6-row verify's inputs, the
            // recurrence, the rollback's discarded output).
            assert_eq!(after[1], inputs);
            assert_eq!(after[2], recurrence);
        }
        // The whole store at 8 lanes, K5, a 48-row step.
        let ctx = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, 16384, 4096, 2 << 20);
        let budget = LaneSpecBudget::new(lanes, k, 0, 0, ctx, 1).unwrap();
        assert_eq!(
            (budget.verify_rows_per_lane, budget.capture_rows_per_lane),
            (rows, rows)
        );
        assert_eq!(budget.step_rows(), 48);
        let gdn = GdnStateFormat::Q8;
        let census = lane_spec_census(
            &budget,
            &inputs(&config, gdn, &Qwen4BandScope::single(&config), 0),
        )
        .unwrap();
        let wide = (config.hc_count * config.hidden_size * 4) as u64;
        // Step buffers cover 48 decode rows + 8 prompt-completing rows, unchanged by the trim.
        assert_eq!(
            census.bytes_of(CensusComponent::StepLogits),
            Some((48 + 8) * config.vocab_size as u64 * 4)
        );
        assert_eq!(census.bytes_of(CensusComponent::StepTop1), Some(56 * 4));
        assert_eq!(
            census.bytes_of(CensusComponent::WideHidden),
            Some(48 * wide)
        );
        assert!(census.bytes_of(CensusComponent::DraftTop1).unwrap() >= lanes as u64 * 8);
        // The census ring line is the allocator's own inventory under the current flag state,
        // so before/after totals differ by exactly `lanes` rings.
        let ring_line = census
            .lines
            .iter()
            .find(|l| l.component == CensusComponent::CaptureRing)
            .unwrap();
        let total_with = |trim: bool| {
            let ring = sum(&Qwen4State::row_capture_allocation_sizes_layout(
                &config, gdn, rows, gdn_layers, true, trim,
            )
            .unwrap());
            census.total().unwrap() - ring_line.bytes().unwrap() + lanes as u64 * ring
        };
        assert_eq!(
            total_with(false) - total_with(true),
            lanes as u64 * 175 * 811_008
        );
        let expected = total_with(crate::state::lane_reserve_trim());
        assert_eq!(census.total().unwrap(), expected);
    }

    /// Prints what one native-MTP lane adds to an EP2 rank over an AR-only lane at the shipped
    /// shape (K5 -> 6-row ring, 8 lanes, QSA fp8 / GDN q8, 2 MiB VMM granules), before and after
    /// `HIPFIRE_QWEN4_LANE_RESERVE_TRIM`:
    /// `cargo test --release -p hipfire-arch-qwen4 --lib lane::census::tests::lane_mtp_itemisation_receipt -- --ignored --nocapture`.
    #[test]
    #[ignore = "receipt generator: prints the per-lane itemisation, asserts the sum identity"]
    fn lane_mtp_itemisation_receipt() {
        use crate::state::StateMirror;
        let config = compact_test_config();
        let gdn = GdnStateFormat::Q8;
        let format = format_of(gdn);
        let scope = Qwen4BandScope::single(&config);
        let (lanes, k) = (8usize, 5usize);
        let rows = k + 1;
        let gdn_layers = config.n_linear_layers();
        let m = |bytes: u64| mib(bytes);
        for (max_seq, chunk) in [(4096usize, 2048usize), (16384, 4096), (32768, 4096)] {
            let ctx = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, max_seq, chunk, 2 << 20);
            let ar = Qwen4State::device_bytes_with_mirror(
                &config,
                format,
                &ctx,
                &scope,
                StateMirror::Absent,
            )
            .unwrap() as u64;
            let snap = Qwen4State::device_bytes_with_mirror(
                &config,
                format,
                &ctx,
                &scope,
                StateMirror::Snapshot,
            )
            .unwrap() as u64;
            let head: Vec<u64> = MtpGpuState::allocation_sizes(&config, &ctx).unwrap();
            let wide = (config.hc_count * config.hidden_size * 4) as u64;
            let layout = |trim: bool| {
                Qwen4State::row_capture_allocation_sizes_layout(
                    &config, gdn, rows, gdn_layers, true, trim,
                )
                .unwrap()
            };
            let (before, after) = (layout(false), layout(true));
            println!("## max_seq {max_seq} chunk {chunk} (K{k} -> {rows}-row ring), per MTP lane over an AR lane");
            println!(
                "AR lane target state                 {:>14} B {:>9} MiB (not MTP)",
                ar,
                m(ar)
            );
            println!(
                "snapshot mirror of the target state  {:>14} B {:>9} MiB",
                snap - ar,
                m(snap - ar)
            );
            println!(
                "MTP head state (QSA arenas+carries)  {:>14} B {:>9} MiB  {} allocations",
                sum(&head),
                m(sum(&head)),
                head.len()
            );
            println!(
                "pending hidden                       {:>14} B {:>9} MiB",
                wide,
                m(wide)
            );
            let slot = gdn
                .storage()
                .state_bytes(config.linear_num_value_heads, 128, 128) as u64;
            for (name, ring) in [
                ("capture ring BEFORE (kill switch)", &before),
                ("capture ring AFTER  (default)    ", &after),
            ] {
                println!(
                    "{name} {:>14} B {:>9} MiB  {} allocations",
                    sum(ring),
                    m(sum(ring)),
                    ring.len()
                );
            }
            println!(
                "  state slot {slot} B x {} layers x {} slots (before) | pool {} slots (after)",
                gdn_layers,
                rows + 1,
                2 * (gdn_layers - 1) + rows + 1
            );
            let before_lane = (snap - ar) + sum(&head) + wide + sum(&before);
            let after_lane = (snap - ar) + sum(&head) + wide + sum(&after);
            println!(
                "PER-LANE MTP increment BEFORE {:>14} B {:>9} MiB",
                before_lane,
                m(before_lane)
            );
            println!(
                "PER-LANE MTP increment AFTER  {:>14} B {:>9} MiB  (-{} MiB)",
                after_lane,
                m(after_lane),
                m(before_lane - after_lane)
            );
            println!(
                "8 lanes: BEFORE {} MiB, AFTER {} MiB; 5 extra lanes (3 -> 8) BEFORE {} MiB",
                m(8 * before_lane),
                m(8 * after_lane),
                m(5 * before_lane)
            );
            let budget = LaneSpecBudget::new(lanes, k, 0, 0, ctx, 1).unwrap();
            let census = lane_spec_census(&budget, &inputs(&config, gdn, &scope, 0)).unwrap();
            for line in &census.lines {
                println!(
                    "    census {:<18} x{:<2} {:>14} B {:>10} MiB",
                    line.component.name(),
                    line.copies,
                    line.bytes().unwrap(),
                    mib(line.bytes().unwrap())
                );
            }
            println!("    census TOTAL {} MiB", mib(census.total().unwrap()));
            let mtp_lane = (snap - ar)
                + sum(&head)
                + wide
                + sum(&Qwen4State::row_capture_allocation_sizes_scoped(
                    &config, gdn, rows, gdn_layers,
                )
                .unwrap());
            let shared: u64 = census
                .lines
                .iter()
                .filter(|l| l.copies == 1)
                .map(|l| l.bytes().unwrap())
                .sum();
            assert_eq!(
                census.total().unwrap(),
                lanes as u64 * (ar + mtp_lane) + shared
            );
        }
    }

    fn mib(bytes: u64) -> String {
        format!("{:.2}", bytes as f64 / (1u64 << 20) as f64)
    }

    /// Prints the per-rank census tables of `/mgmtp/CENSUS.md`:
    /// `cargo test -p hipfire-arch-qwen4 --lib lane::census::tests::receipt -- --ignored --nocapture`.
    /// Pure host arithmetic over the compact Flash-Next fixture (the same
    /// geometry as the artifact: 36 GDN + 12 QSA layers, width 4 x 2560,
    /// vocab 248320), QSA fp8 / GDN q8 state, 2 MiB VMM granularity.
    #[test]
    #[ignore = "receipt generator: prints tables, asserts nothing"]
    fn receipt() {
        let config = compact_test_config();
        let gdn = GdnStateFormat::Q8;
        for (name, counts, chunk, head_ranks) in [
            (
                "EP2 mq4-xts (every rank whole-model, head on both)",
                &[48usize][..],
                4096usize,
                2usize,
            ),
            (
                "PP2 mq3 local split 25/23 (head on the output band)",
                &[25, 23][..],
                1024,
                1,
            ),
        ] {
            for max_seq in [16384usize, 65536] {
                let ctx = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, max_seq, chunk, 2 << 20);
                let bands = if counts.len() == 1 {
                    vec![Qwen4BandScope::single(&config); head_ranks]
                } else {
                    partition(&config, counts)
                };
                let owner_mask = if counts.len() == 1 {
                    (1u64 << head_ranks) - 1
                } else {
                    1u64 << (bands.len() - 1)
                };
                println!("## {name}, max_seq {max_seq}, chunk {chunk}");
                for (label, budget) in [
                    (
                        "C8 default-copy (8-row verify/capture, native K=3)",
                        LaneSpecBudget::default_copy(8, 3, 8, ctx, owner_mask).unwrap(),
                    ),
                    (
                        "C8 legacy ring (K=3, 4-row, copy off) = what the current constructor sizes",
                        LaneSpecBudget::legacy(8, 3, ctx, owner_mask).unwrap(),
                    ),
                ] {
                    for growth in [false, true] {
                        let budget = if growth { budget.at_full_context() } else { budget };
                        println!(
                            "### {label}{}",
                            if growth { " [every lane mapped to max_seq]" } else { " [load-time commit]" }
                        );
                        for (rank, scope) in bands.iter().enumerate() {
                            let census =
                                lane_spec_census(&budget, &inputs(&config, gdn, scope, rank)).unwrap();
                            println!(
                                "rank {rank} layers {}..{} head_owner={} output={} TOTAL {} bytes = {} MiB",
                                scope.band.start,
                                scope.band.end,
                                census.head_owner,
                                census.output,
                                census.total().unwrap(),
                                mib(census.total().unwrap())
                            );
                            for line in &census.lines {
                                println!(
                                    "    {:<18} x{:<2} {:>14} bytes  {:>10} MiB  allocations={}",
                                    line.component.name(),
                                    line.copies,
                                    line.bytes().unwrap(),
                                    mib(line.bytes().unwrap()),
                                    line.allocations.len()
                                );
                            }
                        }
                    }
                }
                for (rank, scope) in bands.iter().enumerate() {
                    let ar = ar_lane_census(8, &ctx, &inputs(&config, gdn, scope, rank)).unwrap();
                    println!(
                        "AR C8 (staged today) rank {rank}: {} bytes = {} MiB [{}]",
                        ar.total().unwrap(),
                        mib(ar.total().unwrap()),
                        ar.describe()
                    );
                }
                for lanes in 1..=8usize {
                    let budget =
                        LaneSpecBudget::default_copy(lanes, 3, 8, ctx, owner_mask).unwrap();
                    let totals: Vec<String> = bands
                        .iter()
                        .enumerate()
                        .map(|(rank, scope)| {
                            let census =
                                lane_spec_census(&budget, &inputs(&config, gdn, scope, rank))
                                    .unwrap();
                            mib(census.total().unwrap())
                        })
                        .collect();
                    println!(
                        "SWEEP default-copy lanes={lanes} per-rank MiB {}",
                        totals.join(" ")
                    );
                }
            }
        }
    }
}
