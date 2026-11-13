// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The lane page budget: admission control for the device pages a Qwen4 VMM
//! lane maps as its context grows.
//!
//! # Why
//!
//! A lane store charges every lane its fixed state plus the pages of the first
//! [`crate::ep::ep_lane_commit_rows`] context rows at load; everything a lane
//! maps beyond that (QSA context arenas of the trunk and of the MTP head, the
//! shared gathered-attention workspace) is demand-mapped from whatever device
//! memory is free. Admission checked slots and context length only, so a burst
//! of long requests could pass admission and then fail a `hipMemCreate` in the
//! middle of a step (`provision_step`: [`crate::lane_exec::PROVISION_MAP_REFUSED`]),
//! failing every lane of that step. The budget moves the decision to admission:
//! a request is admitted only when the growth it can still need fits the free
//! memory that is not already promised to other lanes, otherwise it stays
//! queued and is retried at the next step boundary.
//!
//! # The model
//!
//! Per rank, from live measurements and the lane table (nothing is cached, so
//! there is no drift and the outstanding total cannot go negative):
//!
//! * `free`: `hipMemGetInfo` now (pages a lane keeps mapped after retiring are
//!   already out of it).
//! * `earmark`: free memory the load-time lane charge still owns: for every slot
//!   whose state was never allocated its whole fresh census
//!   ([`Qwen4LaneStore::slot_alloc_bytes`]: fixed state, ring, head, pending
//!   hidden, and the pages of the charged rows); for an allocated slot the
//!   charged pages it has not mapped yet; and the gathered workspace's charged
//!   but uncommitted bytes.
//! * `outstanding`: for every live lane that reserved rows, the pages between
//!   what it has mapped (or the charge, whichever is larger) and what its
//!   reservation needs, over the trunk arenas and, while the head is not
//!   retired, the head arenas; plus the workspace growth the largest live
//!   reservation needs.
//! * `need`: the same growth for the request being admitted, in the slot it
//!   would take (the lowest free one).
//!
//! The pool is `free - earmark - margin`; a request is admitted when
//! `outstanding + need <= pool` on EVERY rank. A request whose `need` is zero
//! (it stays inside the charge) is always admitted: the load reserved it. The
//! byte figures are the lane census' own ([`Qwen4State::context_commit_bytes`]
//! is the term [`Qwen4State::device_bytes_with_mirror`] charges per QSA layer,
//! the head's is [`Qwen4ContextCommit::committed_layer_bytes`] over the F32
//! layout [`MtpGpuState::allocation_sizes`] uses), at the driver granule, so
//! what the budget plans is what `ensure_mapped_capacity` maps.
//!
//! # Agreement across EP ranks
//!
//! Ranks never decide separately: every rank reports its numbers
//! ([`RankPages`]) to the root, which evaluates [`page_verdict`] once over the
//! whole vector. The verdict is one value; admission then fans out to every
//! rank (`admit_lane`, [`Qwen4Bundle::reserve_lane_pages`]), so no rank maps
//! for a request another rank refused.

use super::{census, LaneCopyConfig, LaneStatus, Qwen4LaneStore};
use crate::bundle::{BundleError, Qwen4Bundle};
use crate::config::Qwen4Config;
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::state::{Qwen4State, Qwen4StateFormat};
use crate::weights::Qwen4BandScope;
use hipfire_runtime::slot_batch::RequestEpoch;
use rdna_compute::tensor_ops::QsaKvFormat;
use rdna_compute::Gpu;

/// Prefix of the refusal a request that can never fit gets at admission
/// (`lane_error_wire` answers it with a retryable overload, like
/// [`crate::lane_exec::PROVISION_MAP_REFUSED`]).
pub const LANE_PAGE_BUDGET_REFUSED: &str = "lane page budget: ";

/// `HIPFIRE_QWEN4_LANE_PAGE_BUDGET` (default on, `0` restores admission by
/// slots and length only).
pub fn page_budget_enabled() -> bool {
    hipfire_config::qwen4_lane_page_budget()
}

/// Free bytes per rank the budget never hands out
/// (`HIPFIRE_QWEN4_LANE_PAGE_BUDGET_MARGIN_MIB`, default 256).
pub fn page_margin_bytes() -> u64 {
    hipfire_config::qwen4_lane_page_budget_margin_bytes()
}

/// `HIPFIRE_QWEN4_LANE_PAGE_TRACE` (developer, default off): stderr lines naming every rank's
/// page numbers at each admission and the free device bytes before and after every lane
/// context map with the bytes the budget predicted for it (`[lane-pages]`).
pub fn page_trace() -> bool {
    hipfire_config::qwen4_lane_page_trace()
}

/// One `[lane-pages] rank` line: the rank's free bytes (and the pool bytes parked, which
/// `free` does not include), the budget's terms and every slot's reservation and mapped pages.
fn trace_rank(
    geometry: &PageGeometry<'_>,
    slots: &[SlotPages],
    numbers: &RankNumbers,
    parked: u64,
) {
    const MIB: f64 = (1u64 << 20) as f64;
    let pages = &numbers.pages;
    let mib = |bytes: u64| bytes as f64 / MIB;
    let detail: Vec<String> = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            format!(
                "{index}:{}{}/res={}/trunk={:.1}M@{}t/head={}",
                if slot.live { "live" } else { "idle" },
                if geometry.slot_has_head(index) {
                    "+h"
                } else {
                    ""
                },
                slot.reserved_rows,
                mib(slot.have_trunk),
                slot.mapped_tokens,
                slot.have_head
                    .map_or_else(|| "-".to_string(), |bytes| format!("{:.1}M", mib(bytes))),
            )
        })
        .collect();
    eprintln!(
        "[lane-pages] rank={} free={:.1}M parked={:.1}M earmark={:.1}M (cold={} reserve={:.0}M) outstanding={:.1}M need={:.1}M active={} slots=[{}]",
        pages.rank,
        mib(pages.free),
        mib(parked),
        mib(pages.earmark),
        !geometry.warm,
        mib(geometry.cold_reserve),
        mib(pages.outstanding),
        mib(pages.need),
        pages.active_lanes,
        detail.join(" "),
    );
}

/// What the map trace needs of a store, as plain values (so it can be held while a lane of the
/// store is borrowed mutably).
#[derive(Clone, Copy)]
pub(crate) struct MapTraceInputs {
    format: Qwen4StateFormat,
    backend: Qwen4KvBackend,
    max_seq: usize,
    head_max_seq: Option<usize>,
    granularity: usize,
    rank: usize,
}

impl MapTraceInputs {
    pub(crate) fn new(bundle: &Qwen4Bundle, store: &Qwen4LaneStore, gpu: &Gpu) -> Self {
        Self {
            format: store.format,
            backend: store.backend,
            max_seq: bundle.state.max_seq_len,
            head_max_seq: store.spec.as_ref().map(|spec| spec.head_max_seq),
            granularity: gpu.vmm_recommended_granularity().unwrap_or(1),
            rank: bundle
                .weights
                .ep_placement()
                .map_or(0, |placement| placement.rank),
        }
    }
}

/// The device's free bytes and one lane's mapped context pages before `provision_step` maps its
/// context for a step; [`Self::finish`] prints the `[lane-pages] map` line after.
pub(crate) struct LaneMapProbe {
    free: u64,
    trunk: u64,
    head: Option<u64>,
}

impl LaneMapProbe {
    fn free_of(gpu: &Gpu) -> u64 {
        gpu.device_mem_info().map_or(0, |(free, _)| free as u64)
    }

    fn pages_of(gpu: &Gpu, lane: &super::Qwen4LaneState) -> (u64, Option<u64>) {
        (
            lane.state.mapped_context_bytes(gpu).map_or(0, |b| b as u64),
            lane.spec
                .as_ref()
                .map(|spec| spec.head.mapped_context_bytes(gpu).map_or(0, |b| b as u64)),
        )
    }

    pub(crate) fn start(gpu: &Gpu, lane: &super::Qwen4LaneState) -> Self {
        let (trunk, head) = Self::pages_of(gpu, lane);
        Self {
            free: Self::free_of(gpu),
            trunk,
            head,
        }
    }

    /// `end` positions were just mapped: print the free bytes before and after, what the arenas
    /// mapped and what the budget's arithmetic ([`PageGeometry::trunk`] / [`PageGeometry::head`], the
    /// census' figure at the driver granule) planned for the same growth.
    pub(crate) fn finish(
        self,
        inputs: &MapTraceInputs,
        config: &Qwen4Config,
        gpu: &Gpu,
        lane: &super::Qwen4LaneState,
        end: usize,
    ) {
        const MIB: f64 = (1u64 << 20) as f64;
        let mib = |bytes: u64| bytes as f64 / MIB;
        let geometry = PageGeometry {
            config,
            format: inputs.format,
            backend: inputs.backend,
            max_seq: inputs.max_seq,
            granularity: inputs.granularity,
            charge_rows: 0,
            warm: true,
            cold_reserve: 0,
            spec: inputs.head_max_seq.map(|head_max_seq| PageSpec {
                k: 0,
                spec_lanes: None,
                head_max_seq,
                copy: None,
            }),
        };
        let (trunk, head) = Self::pages_of(gpu, lane);
        let free = Self::free_of(gpu);
        let planned_trunk = geometry
            .trunk(end)
            .map_or(0, |bytes| bytes.saturating_sub(self.trunk));
        let head_part = match (self.head, head) {
            (Some(before), Some(after)) => {
                let planned = geometry
                    .head(end.min(inputs.head_max_seq.unwrap_or(0)))
                    .map_or(0, |bytes| bytes.saturating_sub(before));
                format!(
                    " head +{:.2}M (planned {:.2}M)",
                    mib(after.saturating_sub(before)),
                    mib(planned)
                )
            }
            _ => String::new(),
        };
        eprintln!(
            "[lane-pages] map rank={} epoch={:?} end={end} free {:.1}M -> {:.1}M (took {:.2}M) trunk +{:.2}M (planned {:.2}M){head_part}",
            inputs.rank,
            lane.host.epoch,
            mib(self.free),
            mib(free),
            mib(self.free.saturating_sub(free)),
            mib(trunk.saturating_sub(self.trunk)),
            mib(planned_trunk),
        );
    }
}

impl Qwen4LaneStore {
    /// The values [`MapTraceInputs`] reads.
    pub(crate) fn map_trace_inputs(&self, bundle: &Qwen4Bundle, gpu: &Gpu) -> MapTraceInputs {
        MapTraceInputs::new(bundle, self, gpu)
    }
}

/// Context rows a request reserves: its prompt, its generation budget, the
/// trailer row the admission length check charges, and the draft rows a verify
/// window can map past the last budgeted position
/// ([`crate::state::map_ahead_draft_rows`]), capped at the admitted context.
pub fn reservation_rows(
    prompt_tokens: usize,
    max_tokens: usize,
    draft_rows: usize,
    max_seq: usize,
) -> usize {
    prompt_tokens
        .saturating_add(max_tokens)
        .saturating_add(1)
        .saturating_add(draft_rows)
        .min(max_seq)
}

/// One rank's page numbers for a request (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankPages {
    pub rank: usize,
    /// Free device bytes now.
    pub free: u64,
    /// Free bytes the load-time lane charge still owns.
    pub earmark: u64,
    /// Growth already promised to live lanes and not yet mapped.
    pub outstanding: u64,
    /// This request's growth beyond the load-time charge.
    pub need: u64,
    /// Live lanes on the rank's store (identical on every rank).
    pub active_lanes: usize,
}

impl RankPages {
    /// Bytes growth may still be promised from: free, minus the charge the
    /// load owns, minus the margin.
    pub fn pool(&self, margin: u64) -> u64 {
        self.free
            .saturating_sub(self.earmark)
            .saturating_sub(margin)
    }

    fn shortfall(&self, margin: u64) -> u64 {
        self.outstanding
            .saturating_add(self.need)
            .saturating_sub(self.pool(margin))
    }
}

/// Why a request does not fit now: the rank that is furthest short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageShortfall {
    pub pages: RankPages,
    pub margin: u64,
}

impl std::fmt::Display for PageShortfall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MIB: u64 = 1 << 20;
        let p = &self.pages;
        write!(
            f,
            "rank {}: the request needs {} MiB of context pages beyond the lane charge and {} MiB are already \
             promised to {} live lane(s), but only {} MiB are free ({} MiB free - {} MiB still owned by the \
             load-time lane charge - {} MiB margin)",
            p.rank,
            p.need / MIB,
            p.outstanding / MIB,
            p.active_lanes,
            p.pool(self.margin) / MIB,
            p.free / MIB,
            p.earmark / MIB,
            self.margin / MIB,
        )
    }
}

/// The admission decision of the page budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageVerdict {
    /// Admit: every rank can map what the request can still need.
    Fit,
    /// Does not fit while other lanes hold promises or slots: keep it queued
    /// and ask again at the next step boundary.
    Wait(PageShortfall),
    /// Does not fit and no lane is active, so waiting cannot change the
    /// answer: refuse it.
    Never(PageShortfall),
}

/// The one decision over every rank's numbers. Identical for any rank order
/// and independent of which rank asks: `Fit` iff every rank's outstanding
/// total plus the request's need is within its pool (or the request needs
/// nothing beyond the charge on every rank); otherwise the furthest-short rank
/// is named, and the request waits while any lane is active.
pub fn page_verdict(ranks: &[RankPages], margin: u64) -> PageVerdict {
    if ranks.iter().all(|rank| rank.need == 0) {
        return PageVerdict::Fit;
    }
    let worst = ranks
        .iter()
        .filter(|rank| rank.need != 0 && rank.shortfall(margin) != 0)
        .max_by_key(|rank| (rank.shortfall(margin), std::cmp::Reverse(rank.rank)));
    let Some(worst) = worst else {
        return PageVerdict::Fit;
    };
    let shortfall = PageShortfall {
        pages: *worst,
        margin,
    };
    if ranks.iter().all(|rank| rank.active_lanes == 0) {
        PageVerdict::Never(shortfall)
    } else {
        PageVerdict::Wait(shortfall)
    }
}

/// One free slot's growth as a rank prices it (see [`RankNumbers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotNeed {
    /// The slot carries MTP state (a head).
    pub head: bool,
    /// Growth for the request in this slot beyond the load-time charge, from
    /// the pages the slot already holds; `None` = a live slot.
    pub need: Option<u64>,
}

/// One rank's report to the root: its [`RankPages`] (with the request priced for
/// this rank's own best free slot) plus the per-slot needs the root chooses the
/// slot from, and the slot-independent growth (the shared gather workspace).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankNumbers {
    pub pages: RankPages,
    pub(crate) slots: Vec<SlotNeed>,
    pub(crate) extra: u64,
}

/// The admission decision: the verdict and the slot every rank admits the
/// request into (`None`: no choice, the lowest free slot as before).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageDecision {
    pub verdict: PageVerdict,
    pub slot: Option<usize>,
}

impl PageDecision {
    /// Today's admission: everything fits, the lowest free slot.
    pub const UNBUDGETED: Self = Self {
        verdict: PageVerdict::Fit,
        slot: None,
    };
}

/// The slot every rank admits into: among the free slots (free on every rank), prefer
/// slots that carry MTP state (low concurrency keeps landing on native lanes, as the
/// lowest-free-slot rule did), and among those the one that needs the fewest new pages
/// on its worst rank, because a slot keeps the pages it mapped for an earlier request:
/// the request then reuses them instead of mapping fresh ones. Ties take the lowest
/// slot. A pure function of the ranks' vectors, so the root decides once.
pub(crate) fn choose_slot(ranks: &[RankNumbers]) -> Option<usize> {
    let slots = ranks.first()?.slots.len();
    let free: Vec<usize> = (0..slots)
        .filter(|&slot| {
            ranks
                .iter()
                .all(|rank| rank.slots.get(slot).is_some_and(|s| s.need.is_some()))
        })
        .collect();
    let head = free.iter().any(|&slot| ranks[0].slots[slot].head);
    free.into_iter()
        .filter(|&slot| ranks[0].slots[slot].head == head)
        .min_by_key(|&slot| {
            let worst = ranks
                .iter()
                .filter_map(|rank| rank.slots[slot].need)
                .max()
                .unwrap_or(0);
            (worst, slot)
        })
}

/// The decision over every rank's report: choose the slot once
/// ([`choose_slot`]), price the request in it on every rank, and run
/// [`page_verdict`]. `enabled == false` is the kill switch: today's admission.
pub fn page_decision(enabled: bool, ranks: &[RankNumbers], margin: u64) -> PageDecision {
    if !enabled {
        return PageDecision::UNBUDGETED;
    }
    let slot = choose_slot(ranks);
    let priced: Vec<RankPages> = ranks
        .iter()
        .map(|rank| {
            let mut pages = rank.pages;
            if let Some(need) = slot.and_then(|slot| rank.slots[slot].need) {
                pages.need = need.saturating_add(rank.extra);
            }
            pages
        })
        .collect();
    PageDecision {
        verdict: page_verdict(&priced, margin),
        slot,
    }
}

/// Rows from which a prefill forward builds the lazily created F16 weight shadows (the
/// route-row floor of every ≥512-gated GEMM, `rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS`).
/// `route_rows` is the forward's `Gpu::route_rows(rows)`, so a wave running under the EP route
/// cap (511) never qualifies.
pub fn long_prefill_route_rows(route_rows: usize) -> bool {
    route_rows >= rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS
}

/// Step boundaries without a ≥512-row prefill or an elastic reclaim before the elastic reserve
/// gives the long-prefill caches back or lends its bank.
pub const ELASTIC_IDLE_STEPS: usize = 64;
/// The smallest bank worth lending (a smaller lendable figure lends nothing).
pub const ELASTIC_MIN_LEND_BYTES: u64 = 256 << 20;
/// Page margin ([`page_margin_bytes`]'s default), placement margin and emergency headroom the
/// lendable figure leaves free.
const ELASTIC_PAGE_MARGIN_BYTES: u64 = 256 << 20;
const ELASTIC_PLACEMENT_MARGIN_BYTES: u64 = 512 << 20;
const ELASTIC_EMERGENCY_BYTES: u64 = 128 << 20;

/// One rank's elastic reserve numbers ([`Qwen4Bundle::elastic_numbers_local`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ElasticNumbers {
    /// Free device bytes now (`hipMemGetInfo`; an already lent bank is out of it).
    pub free: u64,
    /// The page earmark WITHOUT the cold reserve (the cold reserve is the revocable part).
    pub earmark_noncold: u64,
    /// Growth the live lanes' reservations still need.
    pub outstanding: u64,
    /// Device bytes of the elastic bank this rank holds (0 = none).
    pub lent: u64,
    /// The lazily built ≥512-row weight shadows are resident.
    pub caches_resident: bool,
}

impl ElasticNumbers {
    /// What this rank can lend: `free - earmark_noncold - outstanding` less the page margin, the
    /// placement margin and the emergency headroom (saturating).
    pub(crate) fn lendable(&self) -> u64 {
        self.free
            .saturating_sub(self.earmark_noncold)
            .saturating_sub(self.outstanding)
            .saturating_sub(ELASTIC_PAGE_MARGIN_BYTES)
            .saturating_sub(ELASTIC_PLACEMENT_MARGIN_BYTES)
            .saturating_sub(ELASTIC_EMERGENCY_BYTES)
    }
}

/// What the root decides over every rank's [`ElasticNumbers`] at an idle step boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ElasticAction {
    Nothing,
    /// Every rank frees its long-prefill caches.
    ReleaseCaches,
    /// Every rank lends a bank of at most this many bytes.
    Lend(u64),
}

/// The elastic reserve's decision: nothing before [`ELASTIC_IDLE_STEPS`] idle steps (or with the
/// reserve off, `cap == 0`); then release the caches when any rank holds them; else, with nothing
/// lent anywhere, lend the minimum over the ranks of `min(cap, lendable)` when it reaches
/// [`ELASTIC_MIN_LEND_BYTES`].
pub(crate) fn elastic_action(
    cap: u64,
    idle_steps: usize,
    ranks: &[ElasticNumbers],
) -> ElasticAction {
    if cap == 0 || ranks.is_empty() || idle_steps < ELASTIC_IDLE_STEPS {
        return ElasticAction::Nothing;
    }
    if ranks.iter().any(|rank| rank.caches_resident) {
        return ElasticAction::ReleaseCaches;
    }
    if ranks.iter().any(|rank| rank.lent > 0) {
        return ElasticAction::Nothing;
    }
    let bytes = ranks
        .iter()
        .map(|rank| rank.lendable().min(cap))
        .min()
        .unwrap_or(0);
    if bytes >= ELASTIC_MIN_LEND_BYTES {
        ElasticAction::Lend(bytes)
    } else {
        ElasticAction::Nothing
    }
}

/// The idle step count after one more step boundary: restarted by a prefill tile of >= 512 route
/// rows (`long_prefill`) or by a reclaim since the last boundary (`reclaims != seen_reclaims`),
/// else one more.
pub(crate) fn elastic_idle_next(
    steps: usize,
    seen_reclaims: u64,
    reclaims: u64,
    long_prefill: bool,
) -> usize {
    if reclaims != seen_reclaims || long_prefill {
        0
    } else {
        steps.saturating_add(1)
    }
}

/// Whether a rank counts as warm: the long-prefill memory is already out of `free`. With the
/// elastic reserve on that is exactly "the long-prefill caches are resident" (a release makes the
/// rank cold again, so the cold reserve is held for the next long prompt); otherwise today's
/// sticky rule: some lane has run past the charge, ever.
pub(crate) fn lane_warm(
    elastic: bool,
    caches_resident: bool,
    seen_before: bool,
    slot_past_charge: bool,
) -> bool {
    if elastic {
        caches_resident
    } else {
        seen_before || slot_past_charge
    }
}

/// One rank's outcome of the lab lend (`Qwen4Bundle::elastic_lab_lend`): what the lend started and
/// what `settle` found published once every fill and publication ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticLabRank {
    pub rank: usize,
    /// Units / bytes the lend started filling (0 when it lent nothing).
    pub lent_units: usize,
    pub lent_bytes: u64,
    /// Units published after the settle, and the bank's device bytes.
    pub settled_units: usize,
    pub settled_bytes: u64,
}

/// One rank's elastic reserve under an EP load that has a D2 tail: the tail that owns the bank,
/// the per-rank cap and the log switch (all read once at attach).
pub(crate) struct ElasticRank {
    pub tail: std::sync::Arc<crate::expert_unit_tail_rank::UnitTailRank>,
    /// Cap in bytes per rank (0 = the reserve is off: every op is a no-op).
    pub cap: u64,
    pub log: bool,
    /// Reclaims on this rank that freed a bank (the idle step count restarts when it moves).
    pub reclaims: std::sync::atomic::AtomicU64,
    /// Root only: step boundaries since the last long prefill or reclaim
    /// ([`Qwen4Bundle::elastic_idle_boundary`]). It lives here, not in the serve loop: the batch
    /// loop returns whenever no lane is active and nothing is queued, so a loop-local count
    /// restarts with every request and a workload of short replies never reached
    /// [`ELASTIC_IDLE_STEPS`].
    pub idle_steps: usize,
    /// Root only: the reclaim count [`Self::idle_steps`] last saw.
    pub idle_reclaims: u64,
    /// Root only: the idle-step count of the last tick that probed the ranks, so an unchanged
    /// "nothing to do" answer is asked again only after another [`ELASTIC_IDLE_STEPS`] steps.
    pub probed_at: Option<usize>,
    /// The long-prefill note of the lane step being provisioned (a serial tile routing at >= 512
    /// rows as the forward runs it); every rank notes it, the root's step boundary consumes it.
    pub step_long_prefill: bool,
}

/// The store-level facts the byte arithmetic reads (all host values).
#[derive(Clone, Copy)]
pub(crate) struct PageGeometry<'a> {
    pub config: &'a Qwen4Config,
    pub format: Qwen4StateFormat,
    pub backend: Qwen4KvBackend,
    pub max_seq: usize,
    pub granularity: usize,
    /// Rows the load-time lane reservation charges per lane.
    pub charge_rows: usize,
    /// The lazily grown non-lane memory of a long prefill is already out of the free figure
    /// ([`lane_warm`]): without the elastic reserve, a lane has prefilled past the charged rows on
    /// this rank since the load (sticky); with it, the long-prefill caches are resident (a release
    /// makes the rank cold again).
    pub warm: bool,
    /// Free bytes the first long prefill is expected to take beyond the lanes'
    /// own pages while the rank is not `warm`.
    pub cold_reserve: u64,
    pub spec: Option<PageSpec<'a>>,
}

/// An MTP store's staging as far as the page arithmetic needs it.
#[derive(Clone, Copy)]
pub(crate) struct PageSpec<'a> {
    pub k: usize,
    /// Mixed store: only the first `spec_lanes` slots carry MTP state.
    pub spec_lanes: Option<usize>,
    /// The lane heads' context bound.
    pub head_max_seq: usize,
    pub copy: Option<&'a LaneCopyConfig>,
}

/// The shared gathered-attention workspace of a rank, when its route needs it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherPages {
    pub kv_heads: usize,
    /// Bytes the workspace has committed now.
    pub committed: u64,
    pub granule: u64,
}

/// One slot as the arithmetic sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SlotPages {
    pub live: bool,
    /// Rows reserved at admission; `0` = none (a poisoned lane drops it).
    pub reserved_rows: usize,
    /// The slot's state was allocated (a released slot keeps it).
    pub materialised: bool,
    /// Mapped bytes of the trunk context arenas.
    pub have_trunk: u64,
    /// Tokens the trunk's context arenas cover.
    pub mapped_tokens: usize,
    /// Mapped bytes of the head's context arenas (`None`: no head state).
    pub have_head: Option<u64>,
    /// The lane's head retired to head-free AR: it maps no more.
    pub head_retired: bool,
}

impl PageGeometry<'_> {
    fn context(&self, max_seq: usize, tokens: usize) -> Qwen4ContextCommit {
        Qwen4ContextCommit::new(self.backend, max_seq, tokens.min(max_seq), self.granularity)
    }

    /// Pages of the trunk's context arenas at `tokens` tokens.
    fn trunk(&self, tokens: usize) -> Result<u64, String> {
        Qwen4State::context_commit_bytes(
            self.config,
            self.format,
            &self.context(self.max_seq, tokens),
        )
        .ok_or_else(|| "lane page budget: trunk context bytes overflow".to_string())
    }

    /// Pages of one lane head's context arenas at `tokens` tokens.
    fn head(&self, tokens: usize) -> Result<u64, String> {
        let Some(spec) = self.spec else {
            return Ok(0);
        };
        self.context(spec.head_max_seq, tokens)
            .committed_layer_bytes(self.config, QsaKvFormat::F32)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| "lane page budget: head context bytes overflow".to_string())
    }

    /// Whether `slot` carries MTP state (and so a head).
    fn slot_has_head(&self, slot: usize) -> bool {
        self.spec
            .is_some_and(|spec| spec.spec_lanes.is_none_or(|lanes| slot < lanes))
    }

    /// Everything a never-allocated slot allocates at its first admission,
    /// with its context at the charged rows (the census' own figure).
    fn fresh_slot_bytes(&self, slot: usize) -> Result<u64, String> {
        let context = self.context(self.max_seq, self.charge_rows);
        let spec = self
            .spec
            .filter(|_| self.slot_has_head(slot))
            .map(|spec| (spec.k, spec.copy));
        Qwen4LaneStore::slot_alloc_bytes(self.config, self.format, &context, spec)
            .ok_or_else(|| "lane page budget: slot allocation bytes overflow".to_string())
    }
}

/// Growth of one slot from what it has mapped, or the charge if larger, to
/// `rows` rows: trunk arenas, plus the head's while it is live.
fn slot_growth(
    geometry: &PageGeometry<'_>,
    slot_index: usize,
    slot: &SlotPages,
    rows: usize,
    trunk_charged: u64,
    head_charged: u64,
) -> Result<u64, String> {
    let rows = rows.min(geometry.max_seq);
    let trunk = geometry
        .trunk(rows)?
        .saturating_sub(slot.have_trunk.max(trunk_charged));
    let head = if geometry.slot_has_head(slot_index) && !slot.head_retired {
        let head_rows = rows.min(geometry.spec.map_or(0, |spec| spec.head_max_seq));
        geometry
            .head(head_rows)?
            .saturating_sub(slot.have_head.unwrap_or(0).max(head_charged))
    } else {
        0
    };
    Ok(trunk.saturating_add(head))
}

fn gather_pages(gather: &GatherPages, tokens: usize) -> Result<u64, String> {
    if tokens == 0 {
        return Ok(0);
    }
    let bytes = rdna_compute::tensor_ops::qsa_gathered_wmma_scratch_bytes(gather.kv_heads, tokens)
        .ok_or_else(|| "lane page budget: gather workspace bytes overflow".to_string())?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| "lane page budget: gather workspace bytes overflow".to_string())?;
    let granule = gather.granule.max(1);
    Ok(bytes.div_ceil(granule).saturating_mul(granule))
}

/// [`rank_numbers`]' pages alone: the request priced in this rank's best free slot.
#[cfg(test)]
pub(crate) fn rank_pages(
    geometry: &PageGeometry<'_>,
    slots: &[SlotPages],
    gather: Option<&GatherPages>,
    rank: usize,
    free: u64,
    rows: usize,
) -> Result<RankPages, String> {
    rank_numbers(geometry, slots, gather, rank, free, rows).map(|numbers| numbers.pages)
}

/// One rank's report for a request reserving `rows` rows. `free` is the rank's
/// measured free device memory. A legacy context backend maps nothing on
/// demand: all zeros.
pub(crate) fn rank_numbers(
    geometry: &PageGeometry<'_>,
    slots: &[SlotPages],
    gather: Option<&GatherPages>,
    rank: usize,
    free: u64,
    rows: usize,
) -> Result<RankNumbers, String> {
    let active_lanes = slots.iter().filter(|slot| slot.live).count();
    let mut pages = RankPages {
        rank,
        free,
        active_lanes,
        ..RankPages::default()
    };
    let mut needs: Vec<SlotNeed> = slots
        .iter()
        .enumerate()
        .map(|(index, slot)| SlotNeed {
            head: geometry.slot_has_head(index),
            need: (!slot.live).then_some(0),
        })
        .collect();
    let mut extra = 0u64;
    if geometry.backend != Qwen4KvBackend::Vmm {
        return Ok(RankNumbers {
            pages,
            slots: needs,
            extra,
        });
    }
    if !geometry.warm {
        pages.earmark = geometry.cold_reserve;
    }
    let charge = geometry.charge_rows.min(geometry.max_seq);
    let trunk_charged = geometry.trunk(charge)?;
    let head_charged = geometry.head(charge)?;
    let mut live_rows = 0usize;
    for (index, slot) in slots.iter().enumerate() {
        pages.earmark = pages.earmark.saturating_add(if slot.materialised {
            trunk_charged
                .saturating_sub(slot.have_trunk)
                .saturating_add(
                    slot.have_head
                        .map_or(0, |have| head_charged.saturating_sub(have)),
                )
        } else {
            geometry.fresh_slot_bytes(index)?
        });
        if slot.live && slot.reserved_rows > 0 {
            pages.outstanding = pages.outstanding.saturating_add(slot_growth(
                geometry,
                index,
                slot,
                slot.reserved_rows,
                trunk_charged,
                head_charged,
            )?);
            live_rows = live_rows.max(slot.reserved_rows.min(geometry.max_seq));
        }
    }
    let rows = rows.min(geometry.max_seq);
    for (index, slot) in slots.iter().enumerate() {
        if !slot.live {
            needs[index].need = Some(slot_growth(
                geometry,
                index,
                slot,
                rows,
                trunk_charged,
                head_charged,
            )?);
        }
    }
    if let Some(gather) = gather {
        // One workspace per GPU, shared by every lane: it grows to the largest
        // context any lane reaches, so the request adds only its excess.
        let charged = gather_pages(gather, charge)?;
        let committed = gather.committed;
        let growth = |tokens: usize| -> Result<u64, String> {
            Ok(gather_pages(gather, tokens)?.saturating_sub(committed.max(charged)))
        };
        let live = growth(live_rows)?;
        let all = growth(live_rows.max(rows))?;
        pages.earmark = pages
            .earmark
            .saturating_add(charged.saturating_sub(committed));
        pages.outstanding = pages.outstanding.saturating_add(live);
        extra = all.saturating_sub(live);
    }
    let numbers = RankNumbers {
        pages,
        slots: needs,
        extra,
    };
    // This rank's own best slot, for its report line and the single-rank tests.
    let local = std::slice::from_ref(&numbers);
    let mut pages = numbers.pages;
    if let Some(need) = choose_slot(local).and_then(|slot| numbers.slots[slot].need) {
        pages.need = need.saturating_add(numbers.extra);
    } else {
        pages.need = numbers.extra;
    }
    Ok(RankNumbers { pages, ..numbers })
}

impl Qwen4LaneStore {
    /// Device bytes one never-allocated slot allocates at its first admission
    /// ([`super::new_payload`]) with its context as `context` commits it: an
    /// AR slot is a mirror-less target state; an MTP slot (`spec = Some((k,
    /// copy))`) adds the capture ring, head state, pending hidden and owed
    /// rows, summed from the very census [`Self::device_bytes_mixed`] charges
    /// (the per-lane families of a one-lane census), so planned bytes are the
    /// allocated bytes. `None` on overflow.
    pub(crate) fn slot_alloc_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        spec: Option<(usize, Option<&LaneCopyConfig>)>,
    ) -> Option<u64> {
        let scope = Qwen4BandScope::single(config);
        let Some((k, copy)) = spec else {
            return Qwen4State::device_bytes_with_mirror(
                config,
                format.lane(),
                context,
                &scope,
                Self::state_mirror(None),
            )
            .and_then(|bytes| u64::try_from(bytes).ok());
        };
        let inputs = census::CensusInputs {
            config,
            format,
            scope: &scope,
            rank: 0,
            append_scratch: None,
        };
        let budget = match copy {
            Some(copy) => crate::mesh_rank_copy::copy_budget(
                1,
                k,
                copy,
                crate::gpu_forward_lanes::LANE_MAX_ROWS,
                *context,
                1,
            )
            .ok()?,
            None => census::LaneSpecBudget::new(1, k, 0, 0, *context, 1).ok()?,
        };
        let census = census::lane_spec_census(&budget, &inputs).ok()?;
        [
            census::CensusComponent::TargetState,
            census::CensusComponent::CaptureRing,
            census::CensusComponent::HeadState,
            census::CensusComponent::PendingHidden,
            census::CensusComponent::HeadLag,
        ]
        .into_iter()
        .try_fold(0u64, |total, component| {
            total.checked_add(census.bytes_of(component)?)
        })
    }

    /// Every slot as the page arithmetic reads it: liveness, reservation and
    /// the pages the slot's arenas have mapped on `gpu`.
    pub(crate) fn page_slots(&self, gpu: &Gpu) -> Result<Vec<SlotPages>, String> {
        let vmm = self.backend == Qwen4KvBackend::Vmm;
        (0..self.table.max_lanes())
            .map(|slot| {
                let live = self.table.live[slot];
                let Some(cell) = self.table.cells[slot].as_ref() else {
                    return Ok(SlotPages {
                        live,
                        ..SlotPages::default()
                    });
                };
                let mapped = |bytes: Result<usize, String>| -> Result<u64, String> {
                    if vmm {
                        bytes.map(|bytes| bytes as u64)
                    } else {
                        Ok(0)
                    }
                };
                let have_trunk = mapped(
                    cell.state
                        .mapped_context_bytes(gpu)
                        .map_err(|e| format!("lane {slot} trunk arenas: {e}")),
                )?;
                let (have_head, head_retired) = match cell.spec.as_ref() {
                    Some(spec) => (
                        Some(mapped(
                            spec.head
                                .mapped_context_bytes(gpu)
                                .map_err(|e| format!("lane {slot} head arenas: {e}")),
                        )?),
                        spec.stats.retired,
                    ),
                    None => (None, false),
                };
                let reserved_rows = if live && cell.host.status != LaneStatus::Poisoned {
                    self.page_reserved[slot]
                } else {
                    0
                };
                Ok(SlotPages {
                    live,
                    reserved_rows,
                    materialised: true,
                    have_trunk,
                    mapped_tokens: cell.state.mapped_context_tokens(),
                    have_head,
                    head_retired,
                })
            })
            .collect()
    }

    /// Reserve `rows` context rows for `epoch`'s lane (host only). Replaces
    /// any earlier reservation; the release points are [`Self::release`] and a
    /// poisoned lane (see [`Self::page_slots`]).
    pub(crate) fn reserve_pages(
        &mut self,
        epoch: &RequestEpoch,
        rows: usize,
    ) -> Result<(), String> {
        let slot = self
            .table
            .slot_of(epoch)
            .ok_or_else(|| format!("reserve_lane_pages: unknown or stale epoch {epoch:?}"))?;
        self.page_reserved[slot] = rows;
        Ok(())
    }

    /// Draft rows a verify window of this store can map past the last budgeted
    /// position ([`crate::state::map_ahead_draft_rows`]; none for AR lanes).
    pub(crate) fn page_draft_rows(&self) -> usize {
        self.spec.as_ref().map_or(0, |spec| {
            crate::state::map_ahead_draft_rows(
                spec.config.k,
                crate::gpu_forward_lanes::LANE_MAX_ROWS,
            )
        })
    }
}

impl Qwen4Bundle {
    /// Rows of the first prefill chunk on this rank (the one expression the load
    /// reservation, [`crate::ep`]'s staging fit check and the lane tiles use).
    pub(crate) fn lane_prefill_chunk(&self, gpu: &Gpu) -> usize {
        let mode = self
            .forward_scratch_mode()
            .unwrap_or(crate::gpu_forward::Qwen4ForwardScratchMode::Full);
        crate::gpu_forward::qwen4_prefill_chunk_requested(gpu, self.state.max_seq_len, mode)
    }

    /// Rows the load-time lane reservation charges per lane on this rank: the
    /// one expression [`crate::ep`]'s staging fit check and the loader's
    /// reservation use, so the earmark equals what the load reserved.
    pub(crate) fn lane_charge_rows(&self, gpu: &Gpu) -> usize {
        crate::ep::ep_lane_commit_rows(self.lane_prefill_chunk(gpu))
    }

    /// Draft rows to add to a request's reservation ([`reservation_rows`]);
    /// `0` without a staged store.
    pub fn lane_page_draft_rows(&self) -> usize {
        self.lanes
            .as_ref()
            .map_or(0, |store| store.page_draft_rows())
    }

    /// This rank's report for a request reserving `rows` rows
    /// ([`rank_numbers`]).
    pub(crate) fn lane_page_numbers_local(
        &self,
        gpu: &Gpu,
        rows: usize,
    ) -> Result<RankNumbers, String> {
        self.lane_page_numbers_cold(gpu, rows)
            .map(|(numbers, _)| numbers)
    }

    /// [`Self::lane_page_numbers_local`] and the cold reserve its earmark holds (0 once the rank
    /// is warm).
    fn lane_page_numbers_cold(&self, gpu: &Gpu, rows: usize) -> Result<(RankNumbers, u64), String> {
        let store = self
            .lanes
            .as_ref()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        let rank = self
            .weights
            .ep_placement()
            .map_or(0, |placement| placement.rank);
        let free = gpu
            .device_mem_info()
            .map(|(free, _)| free as u64)
            .map_err(|error| format!("rank {rank}: device memory query failed: {error}"))?;
        let granularity = gpu.vmm_recommended_granularity().unwrap_or(1);
        let charge_rows = self.lane_charge_rows(gpu);
        let slots = store.page_slots(gpu)?;
        // Without the elastic reserve: a lane that mapped past the charged rows has run a long
        // prompt tile (a step provisions and forwards in one tick, so what a gate sees mapped has
        // run): the lazily grown non-lane memory of a long prefill (pooled tile scratch, GEMM
        // input buffers, kernel modules) is then out of `free`. The flag is sticky: it never
        // clears, whatever the slots later hold. With the elastic reserve the long-prefill caches
        // can be given back, so the rank is warm exactly while they are resident, and the sticky
        // flag is left alone.
        let elastic = self.elastic_active();
        let seen_before = store
            .long_prefill_seen
            .load(std::sync::atomic::Ordering::Relaxed);
        let warm = lane_warm(
            elastic,
            elastic && gpu.long_prefill_caches_resident(),
            seen_before,
            slots.iter().any(|slot| slot.mapped_tokens > charge_rows),
        );
        if !elastic {
            store
                .long_prefill_seen
                .store(warm, std::sync::atomic::Ordering::Relaxed);
        }
        let geometry = PageGeometry {
            config: &self.config,
            format: store.format,
            backend: store.backend,
            max_seq: self.state.max_seq_len,
            granularity,
            charge_rows,
            warm,
            cold_reserve: hipfire_config::qwen4_lane_page_budget_cold_bytes(),
            spec: store.spec.as_ref().map(|spec| PageSpec {
                k: spec.config.k,
                spec_lanes: spec.config.spec_lanes,
                head_max_seq: spec.head_max_seq,
                copy: store.copy.as_ref().map(|copy| &copy.config),
            }),
        };
        let gather = self
            .state
            .qsa
            .first()
            .filter(|qsa| {
                rdna_compute::tensor_ops::qsa_gather_workspace_needed(
                    gpu,
                    qsa.format,
                    qsa.f16_arena.is_some(),
                )
            })
            .map(|_| GatherPages {
                kv_heads: self.config.num_key_value_heads,
                committed: gpu.qsa_gather_scratch_bytes() as u64,
                granule: granularity as u64,
            });
        let numbers = rank_numbers(&geometry, &slots, gather.as_ref(), rank, free, rows)
            .map_err(|error| format!("rank {rank}: {error}"))?;
        if page_trace() {
            trace_rank(&geometry, &slots, &numbers, gpu.pool_parked_bytes() as u64);
        }
        let cold_held = if geometry.warm || geometry.backend != Qwen4KvBackend::Vmm {
            0
        } else {
            geometry.cold_reserve
        };
        Ok((numbers, cold_held))
    }

    /// This rank's D2 expert tail (installed by the EP row-gather hook at attach), the owner of the
    /// elastic bank. `None` without an EP load or without the tail.
    pub(crate) fn ep_unit_tail(
        &self,
    ) -> Option<&std::sync::Arc<crate::expert_unit_tail_rank::UnitTailRank>> {
        self.elastic.as_ref().map(|elastic| &elastic.tail)
    }

    /// Resolve the elastic reserve of this rank once at EP attach: the cap
    /// (`HIPFIRE_EP_ELASTIC_MIB`, a bad value is a load error) and the rank's D2 tail. Without a
    /// tail there is nothing to lend through and the reserve stays inert.
    pub(crate) fn install_elastic(
        &mut self,
        tail: Option<std::sync::Arc<crate::expert_unit_tail_rank::UnitTailRank>>,
    ) -> Result<(), String> {
        let cap = hipfire_config::memory::ep_elastic_cap_bytes()
            .map_err(|error| format!("qwen4: {error}"))?;
        self.elastic = tail.map(|tail| ElasticRank {
            tail,
            cap,
            log: hipfire_config::memory::ep_elastic_log(),
            reclaims: std::sync::atomic::AtomicU64::new(0),
            idle_steps: 0,
            idle_reclaims: 0,
            probed_at: None,
            step_long_prefill: false,
        });
        Ok(())
    }

    /// The elastic reserve is on for this bundle: an EP rank with a D2 tail and a positive cap
    /// (`HIPFIRE_EP_ELASTIC_MIB` not `0`). Every elastic op is a no-op while this is false.
    pub fn elastic_active(&self) -> bool {
        self.elastic.as_ref().is_some_and(|elastic| elastic.cap > 0)
    }

    /// One step boundary of the serve loop: the count of boundaries since the last long prefill
    /// or elastic reclaim, which restarts it. The long prefill is the note the step's provisioning
    /// left ([`Self::elastic_note_step_prefill`]: a serial tile routing at >= 512 rows as the
    /// forward runs it), consumed here, so a later boundary without a provisioned step (a mesh
    /// decode step, an idle iteration) counts on. The count spans the loop's busy periods (a
    /// request's end does not restart it). 0 without the reserve.
    pub fn elastic_idle_boundary(&mut self) -> usize {
        let Some(elastic) = self.elastic.as_mut() else {
            return 0;
        };
        let reclaims = elastic.reclaims.load(std::sync::atomic::Ordering::Relaxed);
        let long_prefill = std::mem::take(&mut elastic.step_long_prefill);
        elastic.idle_steps = elastic_idle_next(
            elastic.idle_steps,
            elastic.idle_reclaims,
            reclaims,
            long_prefill,
        );
        elastic.idle_reclaims = reclaims;
        elastic.idle_steps
    }

    fn elastic_rank(&self) -> usize {
        self.weights
            .ep_placement()
            .map_or(0, |placement| placement.rank)
    }

    /// This rank's [`ElasticNumbers`]: one `device_mem_info`, the page earmark without the cold
    /// reserve and the live lanes' outstanding growth.
    pub(crate) fn elastic_numbers_local(&self, gpu: &Gpu) -> Result<ElasticNumbers, String> {
        let elastic = self
            .elastic
            .as_ref()
            .ok_or_else(|| "the elastic reserve is not installed".to_string())?;
        let (numbers, cold_held) = self.lane_page_numbers_cold(gpu, 0)?;
        Ok(ElasticNumbers {
            free: numbers.pages.free,
            earmark_noncold: numbers.pages.earmark.saturating_sub(cold_held),
            outstanding: numbers.pages.outstanding,
            lent: elastic.tail.elastic_bytes(),
            caches_resident: gpu.long_prefill_caches_resident(),
        })
    }

    /// Unpublish and free this rank's bank (rank-local mechanics; see
    /// `UnitTailRank::elastic_reclaim`). `Ok(true)` when a bank was freed, `Ok(false)` when none
    /// was lent (cheap: no device work). `Err`: the bank is still published and valid.
    pub(crate) fn elastic_reclaim_local(&self, gpu: &mut Gpu, why: &str) -> Result<bool, String> {
        let Some(elastic) = self.elastic.as_ref().filter(|elastic| elastic.cap > 0) else {
            return Ok(false);
        };
        if elastic.tail.elastic_bytes() == 0 {
            return Ok(false);
        }
        let rank = self.elastic_rank();
        match elastic.tail.elastic_reclaim(gpu, why) {
            Ok(Some(report)) => {
                elastic
                    .reclaims
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if elastic.log {
                    eprintln!(
                        "qwen4 EP rank {rank}: [elastic] reclaim ({why}): {} units {:.0} MiB in {:.2} ms (wait {:.2}, tables {:.2}, free {:.2}, settle {:.2})",
                        report.units,
                        report.bytes as f64 / f64::from(1u32 << 20),
                        report.total_ms,
                        report.wait_ms,
                        report.tables_ms,
                        report.free_ms,
                        report.settle_ms
                    );
                }
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(error) => Err(format!("rank {rank}: elastic reclaim ({why}): {error}")),
        }
    }

    /// [`Self::elastic_reclaim_local`] for the backstops that cannot act on a failure: it is
    /// logged and the bank stays in place (the caller then just sees less free memory).
    pub(crate) fn elastic_reclaim_logged(&self, gpu: &mut Gpu, why: &str) {
        if let Err(error) = self.elastic_reclaim_local(gpu, why) {
            eprintln!("  qwen4 EP elastic: {error}");
        }
    }

    /// Backstop before a forward tile on this rank whose GEMM routing rows (`route_rows`, as the
    /// tile runs them: `Qwen4GpuForward::tile_route_rows`, so the ordinary-prefill intent's
    /// canonical chunk counts whatever the extent and a wave under the EP route cap never
    /// qualifies) reach the F16 shadow floor: it builds the long-prefill caches, so it reclaims
    /// the bank first.
    pub(crate) fn elastic_prefill_backstop(&self, gpu: &mut Gpu, route_rows: usize) {
        if self.elastic_active() && long_prefill_route_rows(route_rows) {
            self.elastic_reclaim_logged(gpu, "prefill");
        }
    }

    /// The routing rows a SERIAL lane prefill tile of `rows` rows runs at on this bundle's forward
    /// (prefill and lane scope, no wave: `Qwen4GpuForward::tile_route_rows`); `rows` without a
    /// forward. A distribution-equivalent lane tile routes at its own rows; a `serve.batch_exact`
    /// lane tile runs under the ordinary-prefill intent and routes at the canonical chunk. Host
    /// flags only, every scope restored. The elastic accounting reads this, never the bare
    /// `Gpu::route_rows` outside the tile's scopes (which misses the intent).
    pub(crate) fn lane_prefill_tile_route_rows(&mut self, gpu: &mut Gpu, rows: usize) -> usize {
        let Some(forward) = self.execution.as_mut() else {
            return rows;
        };
        let prefill = std::mem::replace(&mut forward.prefill_scope, true);
        let lane = std::mem::replace(&mut forward.lane_scope, true);
        let wave = forward.wave_route.take();
        let route_rows = forward.tile_route_rows(gpu, rows);
        forward.prefill_scope = prefill;
        forward.lane_scope = lane;
        forward.wave_route = wave;
        route_rows
    }

    /// Note whether the lane step being provisioned holds a long prefill; the step's boundary
    /// consumes it ([`Self::elastic_idle_boundary`]). A no-op without the reserve.
    pub(crate) fn elastic_note_step_prefill(&mut self, long_prefill: bool) {
        if let Some(elastic) = self.elastic.as_mut() {
            elastic.step_long_prefill = long_prefill;
        }
    }

    /// Lend a bank of at most `bytes` (rank-local; see `UnitTailRank::elastic_lend`). `Ok(None)`:
    /// nothing lent (refused or no candidate).
    pub(crate) fn elastic_lend_local(
        &self,
        gpu: &mut Gpu,
        bytes: u64,
    ) -> Result<Option<(usize, u64)>, String> {
        let Some(elastic) = self.elastic.as_ref().filter(|elastic| elastic.cap > 0) else {
            return Ok(None);
        };
        let rank = self.elastic_rank();
        match elastic.tail.elastic_lend(gpu, bytes) {
            Ok(Some(report)) => {
                if elastic.log {
                    eprintln!(
                        "qwen4 EP rank {rank}: [elastic] lend: {} units {:.0} MiB in {:.2} ms",
                        report.units,
                        report.bytes as f64 / f64::from(1u32 << 20),
                        report.ms
                    );
                }
                Ok(Some((report.units, report.bytes)))
            }
            Ok(None) => {
                if elastic.log {
                    eprintln!(
                        "qwen4 EP rank {rank}: [elastic] lend: nothing lent (asked {:.0} MiB; refused or too few candidates)",
                        bytes as f64 / f64::from(1u32 << 20)
                    );
                }
                Ok(None)
            }
            Err(error) => Err(format!("rank {rank}: elastic lend: {error}")),
        }
    }

    /// Free this rank's long-prefill caches ([`Gpu::release_long_prefill_caches`]); returns the
    /// device bytes released.
    pub(crate) fn elastic_release_caches_local(&self, gpu: &mut Gpu) -> Result<u64, String> {
        let rank = self.elastic_rank();
        let freed = gpu
            .release_long_prefill_caches()
            .map_err(|error| format!("rank {rank}: release long-prefill caches: {error}"))?;
        if self.elastic.as_ref().is_some_and(|elastic| elastic.log) {
            eprintln!(
                "qwen4 EP rank {rank}: [elastic] release caches: {:.0} MiB",
                freed as f64 / f64::from(1u32 << 20)
            );
        }
        Ok(freed)
    }

    /// Lab (`fn_mesh_lane_oracle --elastic`): lend a bank of at most `bytes` on this rank, then
    /// drive its fills and publication to completion (`UnitTailRank::elastic_settle`), so the
    /// bank is fully published before the measured runs. Ignores the cap and the idle rule; needs
    /// only the D2 tail.
    pub(crate) fn elastic_lab_lend_local(
        &self,
        gpu: &mut Gpu,
        bytes: u64,
    ) -> Result<ElasticLabRank, String> {
        let rank = self.elastic_rank();
        let tail = self
            .ep_unit_tail()
            .ok_or_else(|| format!("rank {rank}: no D2 tail to lend through"))?;
        let lent = tail
            .elastic_lend(gpu, bytes)
            .map_err(|error| format!("rank {rank}: elastic lend: {error}"))?;
        let settled_units = tail
            .elastic_settle(gpu)
            .map_err(|error| format!("rank {rank}: elastic settle: {error}"))?;
        Ok(ElasticLabRank {
            rank,
            lent_units: lent.as_ref().map_or(0, |report| report.units),
            lent_bytes: lent.as_ref().map_or(0, |report| report.bytes),
            settled_units,
            settled_bytes: tail.elastic_bytes(),
        })
    }

    /// Lab: reclaim this rank's bank whatever the cap; whether one was freed.
    pub(crate) fn elastic_lab_reclaim_local(&self, gpu: &mut Gpu) -> Result<bool, String> {
        let rank = self.elastic_rank();
        let tail = self
            .ep_unit_tail()
            .ok_or_else(|| format!("rank {rank}: no D2 tail"))?;
        tail.elastic_reclaim(gpu, "lab")
            .map(|report| report.is_some())
            .map_err(|error| format!("rank {rank}: elastic reclaim: {error}"))
    }

    /// The page budget's decision for a request reserving `rows` rows: this
    /// rank's numbers (single device) or every rank's (EP, evaluated once on the
    /// root so all ranks share the verdict and the slot). Today's admission
    /// under the kill switch and on a pipeline-parallel store, which has no
    /// per-rank lane pages.
    pub fn lane_page_decision(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<PageDecision, String> {
        // The kill switch measures nothing: today's admission, exactly.
        let enabled = page_budget_enabled();
        if !enabled || self.pp.is_some() {
            return Ok(page_decision(false, &[], 0));
        }
        let ranks = if self.ep.is_some() {
            self.ep_lane_page_numbers(gpu, rows)?
        } else {
            vec![self.lane_page_numbers_local(gpu, rows)?]
        };
        let decision = page_decision(true, &ranks, page_margin_bytes());
        if page_trace() {
            eprintln!(
                "[lane-pages] decision rows={rows} slot={:?} verdict={}",
                decision.slot,
                match decision.verdict {
                    PageVerdict::Fit => "fit",
                    PageVerdict::Wait(_) => "wait",
                    PageVerdict::Never(_) => "never",
                }
            );
        }
        Ok(decision)
    }

    /// Reserve `rows` context rows for the just-admitted lane of `epoch` on
    /// every rank (host only; a no-op under the kill switch).
    pub fn reserve_lane_pages(
        &mut self,
        epoch: &RequestEpoch,
        rows: usize,
    ) -> Result<(), BundleError> {
        if !page_budget_enabled() || self.pp.is_some() {
            return Ok(());
        }
        if self.ep.is_some() {
            return self.ep_reserve_lane_pages(epoch, rows);
        }
        self.reserve_lane_pages_local(epoch, rows)
            .map_err(BundleError::Forward)
    }

    pub(crate) fn reserve_lane_pages_local(
        &mut self,
        epoch: &RequestEpoch,
        rows: usize,
    ) -> Result<(), String> {
        self.lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?
            .reserve_pages(epoch, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::ep::{ep_lane_commit_rows_for, ep_lane_store_device_bytes_mixed};
    use crate::{GdnStateFormat, QsaKvFormat as Fmt};

    const MIB: u64 = 1 << 20;
    const GRANULE: usize = 2 << 20;
    const MAX_SEQ: usize = 16384;

    fn format() -> Qwen4StateFormat {
        Qwen4StateFormat {
            qsa: Fmt::Fp8,
            gdn: GdnStateFormat::F32,
            qsa_f16_arena: false,
            qsa_pooled_bf16: false,
            gdn_stream: crate::GdnStreamKind::None,
        }
    }

    fn geometry<'a>(config: &'a Qwen4Config, spec: Option<PageSpec<'a>>) -> PageGeometry<'a> {
        PageGeometry {
            config,
            format: format(),
            backend: Qwen4KvBackend::Vmm,
            max_seq: MAX_SEQ,
            granularity: GRANULE,
            charge_rows: ep_lane_commit_rows_for(4096, None),
            warm: true,
            cold_reserve: 0,
            spec,
        }
    }

    fn mtp<'a>(spec_lanes: Option<usize>) -> PageSpec<'a> {
        PageSpec {
            k: 5,
            spec_lanes,
            head_max_seq: MAX_SEQ,
            copy: None,
        }
    }

    /// A live lane that mapped exactly what `rows` need.
    fn mapped_lane(
        g: &PageGeometry<'_>,
        index: usize,
        reserved: usize,
        mapped: usize,
    ) -> SlotPages {
        SlotPages {
            live: true,
            reserved_rows: reserved,
            materialised: true,
            have_trunk: g.trunk(mapped).unwrap(),
            mapped_tokens: mapped,
            have_head: g.slot_has_head(index).then(|| g.head(mapped).unwrap()),
            head_retired: false,
        }
    }

    /// The growth the budget prices for rows `charge..=rows` is exactly the
    /// difference of two lane-store reservations at those rows: planned ==
    /// what the census (and so `ensure_mapped_capacity`) commits, at the
    /// symtrunk geometry (36 GDN + 12 QSA, fp8 QSA, 2 MiB granules,
    /// max_seq 16384), for an AR store, an all-MTP store and a mixed store.
    #[test]
    fn per_row_bytes_match_the_lane_reservation_arithmetic() {
        let config = compact_test_config();
        let format = format();
        let charge = ep_lane_commit_rows_for(4096, None);
        assert_eq!(charge, 2048);
        for (spec, spec_lanes, lanes) in [
            (None, None, 8usize),
            (Some(5usize), None, 8),
            (Some(5), Some(3), 8),
            (Some(2), Some(1), 3),
        ] {
            let g = geometry(
                &config,
                spec.map(|k| PageSpec {
                    k,
                    ..mtp(spec_lanes)
                }),
            );
            let store = |rows: usize| {
                ep_lane_store_device_bytes_mixed(
                    &config,
                    format,
                    Qwen4KvBackend::Vmm,
                    MAX_SEQ,
                    rows,
                    GRANULE,
                    lanes,
                    lanes,
                    spec,
                    None,
                    spec_lanes,
                )
                .unwrap()
            };
            for rows in [2049usize, 3072, 4097, 8192, 12345, MAX_SEQ] {
                let planned: u64 = (0..lanes)
                    .map(|slot| {
                        let trunk = g.trunk(rows).unwrap() - g.trunk(charge).unwrap();
                        let head = if g.slot_has_head(slot) {
                            g.head(rows).unwrap() - g.head(charge).unwrap()
                        } else {
                            0
                        };
                        trunk + head
                    })
                    .sum();
                assert_eq!(
                    store(rows) - store(charge),
                    planned,
                    "spec {spec:?} lanes {spec_lanes:?}/{lanes} rows {rows}"
                );
            }
        }
    }

    /// The fresh-slot figure is the census' per-lane families: slot kinds
    /// differ by exactly what the mixed census moves between them, and the
    /// mixed store is the shared buffers plus its slots.
    #[test]
    fn slot_alloc_bytes_are_the_census_per_lane_families() {
        let config = compact_test_config();
        let format = format();
        let lanes = 8usize;
        let context = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, MAX_SEQ, 2048, GRANULE);
        let ar = Qwen4LaneStore::slot_alloc_bytes(&config, format, &context, None).unwrap();
        assert_eq!(
            ar,
            Qwen4State::device_bytes_with_mirror(
                &config,
                format,
                &context,
                &Qwen4BandScope::single(&config),
                Qwen4LaneStore::state_mirror(None),
            )
            .unwrap() as u64
        );
        for k in [2usize, 5] {
            let mtp_slot =
                Qwen4LaneStore::slot_alloc_bytes(&config, format, &context, Some((k, None)))
                    .unwrap();
            assert!(mtp_slot > ar);
            let store = |spec_lanes: usize| {
                Qwen4LaneStore::device_bytes_mixed(
                    &config, format, &context, lanes, spec_lanes, k, None,
                )
                .unwrap()
            };
            // Each slot moved from AR-only to MTP adds `mtp_slot - ar` bytes.
            for spec_lanes in 1..=lanes {
                assert_eq!(
                    store(spec_lanes) - store(spec_lanes - 1),
                    mtp_slot - ar,
                    "K{k} spec_lanes {spec_lanes}"
                );
            }
            // The store is its shared buffers plus its slots.
            let shared = store(lanes) - lanes as u64 * mtp_slot;
            for spec_lanes in 0..=lanes {
                assert_eq!(
                    store(spec_lanes),
                    shared + spec_lanes as u64 * mtp_slot + (lanes - spec_lanes) as u64 * ar
                );
            }
        }
    }

    fn ranks_with(
        free: u64,
        earmark: u64,
        outstanding: u64,
        need: u64,
        active: usize,
    ) -> RankPages {
        RankPages {
            rank: 0,
            free,
            earmark,
            outstanding,
            need,
            active_lanes: active,
        }
    }

    #[test]
    fn a_request_inside_the_charge_always_fits_and_zero_need_is_free() {
        // Even with nothing free: the load reserved the charge.
        let none = RankPages {
            free: 0,
            earmark: 10 * MIB,
            ..RankPages::default()
        };
        assert_eq!(page_verdict(&[none, none], 256 * MIB), PageVerdict::Fit);
    }

    #[test]
    fn the_budget_refuses_then_admits_as_lanes_retire() {
        let config = compact_test_config();
        let g = geometry(&config, Some(mtp(None)));
        let margin = 256 * MIB;
        let long = reservation_rows(16_000, 128, 7, MAX_SEQ);
        // Eight lanes allocated, charged, all live and reserved for `long`.
        let mut slots: Vec<SlotPages> = (0..8).map(|i| mapped_lane(&g, i, long, 2048)).collect();
        let per_lane = g.trunk(long).unwrap() - g.trunk(2048).unwrap() + g.head(long).unwrap()
            - g.head(2048).unwrap();
        assert!(
            per_lane > 200 * MIB,
            "a 16k lane grows by ~{} MiB",
            per_lane / MIB
        );
        // 7 live lanes + one free slot: room for 7 promised lanes only.
        slots[7].live = false;
        slots[7].reserved_rows = 0;
        // Free memory = the growth of 6.5 lanes + margin: the 8th request must wait.
        let free = (per_lane * 13) / 2 + margin;
        let numbers =
            |slots: &[SlotPages], free: u64| rank_pages(&g, slots, None, 0, free, long).unwrap();
        let waiting = numbers(&slots, free);
        assert_eq!(waiting.outstanding, 7 * per_lane);
        assert_eq!(waiting.need, per_lane);
        let verdict = page_verdict(&[waiting], margin);
        let PageVerdict::Wait(short) = verdict else {
            panic!("expected wait, got {verdict:?}");
        };
        assert_eq!(short.pages.rank, 0);
        assert!(short.to_string().contains("rank 0"));
        // Two lanes retire (release: not live, reservation gone; their mapped
        // pages stay mapped and stay out of free).
        for slot in &mut slots[..2] {
            slot.live = false;
            slot.reserved_rows = 0;
        }
        let admitted = numbers(&slots, free);
        assert_eq!(admitted.outstanding, 5 * per_lane);
        // The lowest free slot is now slot 0, which already holds its 2048 rows
        // only (mapped == charge): need is the full growth.
        assert_eq!(admitted.need, per_lane);
        assert_eq!(page_verdict(&[admitted], margin), PageVerdict::Fit);
    }

    /// Mapping moves bytes from outstanding to mapped without changing what
    /// is left to promise, and outstanding never goes below zero.
    #[test]
    fn mapping_moves_outstanding_into_mapped_and_it_never_goes_negative() {
        let config = compact_test_config();
        let g = geometry(&config, None);
        let long = reservation_rows(15_000, 1000, 0, MAX_SEQ);
        let margin = 64 * MIB;
        let mut free = 4 * 1024 * MIB;
        let mut slots = vec![
            mapped_lane(&g, 0, long, 2048),
            mapped_lane(&g, 1, long, 2048),
        ];
        slots.push(SlotPages::default());
        let mut last_slack = None;
        for mapped in [2048usize, 4096, 4097, 8192, 9000, long, long] {
            let pages = rank_pages(&g, &slots, None, 0, free, long).unwrap();
            let slack = pages.pool(margin) as i128 - pages.outstanding as i128;
            if let Some(last) = last_slack {
                assert_eq!(slack, last, "mapping must not change the slack");
            }
            last_slack = Some(slack);
            // Lane 0 maps to `mapped` rows.
            let before = slots[0].have_trunk;
            slots[0] = mapped_lane(&g, 0, long, mapped.max(2048));
            free -= slots[0].have_trunk - before.min(slots[0].have_trunk);
            assert!(slots[0].have_trunk >= before);
        }
        // Past its reservation a lane promises nothing (saturating, never wraps).
        let over = rank_pages(&g, &slots, None, 0, free, long).unwrap();
        assert_eq!(
            over.outstanding,
            slot_growth(
                &g,
                1,
                &slots[1],
                long,
                g.trunk(2048).unwrap(),
                g.head(2048).unwrap()
            )
            .unwrap()
        );
    }

    /// A rank that fails the budget fails it for everyone, whatever order the
    /// ranks report in.
    #[test]
    fn ep_ranks_reach_the_same_verdict() {
        let margin = 256 * MIB;
        let roomy = RankPages {
            rank: 0,
            free: 4096 * MIB,
            outstanding: 0,
            need: 500 * MIB,
            active_lanes: 3,
            ..RankPages::default()
        };
        let tight = RankPages {
            rank: 1,
            free: 600 * MIB,
            outstanding: 100 * MIB,
            need: 500 * MIB,
            active_lanes: 3,
            ..RankPages::default()
        };
        let a = page_verdict(&[roomy, tight], margin);
        let b = page_verdict(&[tight, roomy], margin);
        assert_eq!(a, b);
        let PageVerdict::Wait(short) = a else {
            panic!("one short rank must hold every rank back: {a:?}");
        };
        assert_eq!(short.pages.rank, 1);
        // Both fit: Fit regardless of order.
        let tight_ok = RankPages {
            free: 1200 * MIB,
            ..tight
        };
        assert_eq!(page_verdict(&[roomy, tight_ok], margin), PageVerdict::Fit);
        assert_eq!(page_verdict(&[tight_ok, roomy], margin), PageVerdict::Fit);
        // With nothing active a short rank is final for every rank.
        let idle = |rank: RankPages| RankPages {
            active_lanes: 0,
            outstanding: 0,
            ..rank
        };
        assert!(matches!(
            page_verdict(&[idle(roomy), idle(tight)], margin),
            PageVerdict::Never(_)
        ));
        assert!(matches!(
            page_verdict(&[idle(tight), idle(roomy)], margin),
            PageVerdict::Never(_)
        ));
    }

    /// One rank's report for tests: `numbers` over `slots` with `free` bytes.
    fn numbers_of(
        g: &PageGeometry<'_>,
        slots: &[SlotPages],
        rank: usize,
        free: u64,
        rows: usize,
    ) -> RankNumbers {
        rank_numbers(g, slots, None, rank, free, rows).unwrap()
    }

    #[test]
    fn the_kill_switch_admits_exactly_as_before() {
        let config = compact_test_config();
        let g = geometry(&config, None);
        // Nothing free, a full long request: the budget refuses it, the kill switch does not
        // measure, queue or choose a slot: the lowest free slot, as before.
        let slots = vec![SlotPages::default(); 2];
        let hopeless = numbers_of(&g, &slots, 0, 0, MAX_SEQ);
        let on = page_decision(true, std::slice::from_ref(&hopeless), 256 * MIB);
        assert!(matches!(on.verdict, PageVerdict::Never(_)));
        let off = page_decision(false, std::slice::from_ref(&hopeless), 256 * MIB);
        assert_eq!(off, PageDecision::UNBUDGETED);
        assert_eq!(off.verdict, PageVerdict::Fit);
        assert_eq!(off.slot, None);
    }

    /// A free slot keeps the pages it mapped for an earlier request, and the request that takes it
    /// reuses them: the budget prices the request in the slot that needs the fewest new pages (the
    /// one the hoard from the failed or retired lanes sits in), not the lowest index.
    #[test]
    fn a_hoarded_free_slot_counts_as_available_to_the_request_that_takes_it() {
        let config = compact_test_config();
        let g = geometry(&config, None);
        let rows = reservation_rows(16_000, 64, 0, MAX_SEQ);
        let hoard = |mapped: usize| SlotPages {
            live: false,
            materialised: true,
            have_trunk: g.trunk(mapped).unwrap(),
            mapped_tokens: mapped,
            ..SlotPages::default()
        };
        // Slot 0 idle at the charge, slot 2 idle and hoarding 12288 rows, slot 1 live.
        let slots = vec![hoard(2048), mapped_lane(&g, 1, rows, 4096), hoard(12288)];
        let numbers = numbers_of(&g, &slots, 0, 100 * MIB, rows);
        let decision = page_decision(true, std::slice::from_ref(&numbers), 0);
        assert_eq!(decision.slot, Some(2), "the hoarded slot, not the lowest");
        let full = g.trunk(rows).unwrap() - g.trunk(2048).unwrap();
        let reused = g.trunk(rows).unwrap() - g.trunk(12288).unwrap();
        assert!(reused < full);
        assert_eq!(numbers.slots[0].need, Some(full));
        assert_eq!(numbers.slots[2].need, Some(reused));
        assert_eq!(
            numbers.slots[1].need, None,
            "a live slot is not a candidate"
        );
        // With no lane active and nothing free: the request still fits the hoard (zero new pages)
        // when its rows are inside the pages the slot already holds.
        let idle = vec![hoard(2048), hoard(16384)];
        let numbers = numbers_of(&g, &idle, 0, 0, rows);
        let decision = page_decision(true, std::slice::from_ref(&numbers), 0);
        assert_eq!(
            (decision.slot, decision.verdict),
            (Some(1), PageVerdict::Fit)
        );
    }

    /// The slot is chosen once from every rank's vector, whatever the rank order, and an MTP
    /// store keeps landing low concurrency on a slot that carries MTP state.
    #[test]
    fn the_slot_is_chosen_once_for_every_rank_and_prefers_mtp_slots() {
        let config = compact_test_config();
        let g = geometry(&config, Some(mtp(Some(2))));
        let rows = reservation_rows(16_000, 64, 7, MAX_SEQ);
        let hoard = |index: usize, mapped: usize| SlotPages {
            live: false,
            materialised: true,
            have_trunk: g.trunk(mapped).unwrap(),
            mapped_tokens: mapped,
            have_head: g.slot_has_head(index).then(|| g.head(mapped).unwrap()),
            ..SlotPages::default()
        };
        // Rank 0 hoards in MTP slot 1; rank 1 hoards in MTP slot 0; AR slot 3 hoards most on both.
        let r0 = vec![
            hoard(0, 2048),
            hoard(1, 8192),
            hoard(2, 2048),
            hoard(3, 16384),
        ];
        let r1 = vec![
            hoard(0, 8192),
            hoard(1, 2048),
            hoard(2, 2048),
            hoard(3, 16384),
        ];
        let a = numbers_of(&g, &r0, 0, 4096 * MIB, rows);
        let b = numbers_of(&g, &r1, 1, 4096 * MIB, rows);
        let forward = page_decision(true, &[a.clone(), b.clone()], 256 * MIB);
        let backward = page_decision(true, &[b, a], 256 * MIB);
        assert_eq!(forward.slot, backward.slot);
        assert_eq!(forward.verdict, backward.verdict);
        // MTP slots 0 and 1 are free: the AR slot 3 hoard is not preferred; the two MTP slots tie on
        // their worst rank (each rank hoards 8192 in one of them), so the lowest index wins.
        assert_eq!(forward.slot, Some(0));
        // Only AR slots free: the AR slot with the larger hoard.
        let ar_only = |hoards: [usize; 2]| {
            let mut v: Vec<SlotPages> = vec![
                mapped_lane(&g, 0, rows, 2048),
                mapped_lane(&g, 1, rows, 2048),
            ];
            v.push(hoard(2, hoards[0]));
            v.push(hoard(3, hoards[1]));
            v
        };
        let n = numbers_of(&g, &ar_only([2048, 12288]), 0, 4096 * MIB, rows);
        assert_eq!(page_decision(true, &[n], 0).slot, Some(3));
    }

    /// While no lane has mapped past the first chunk the first long tile's non-lane memory is still
    /// ahead: the budget holds `cold_reserve` back; afterwards it is already out of `free`. j7704:
    /// 1002 MiB free per card, two admitted lanes promised 584 MiB, and the first tile alone took
    /// more than the rest.
    #[test]
    fn the_first_long_tile_is_reserved_until_it_has_run() {
        let config = compact_test_config();
        let mut g = geometry(&config, Some(mtp(None)));
        g.warm = false;
        g.cold_reserve = 1280 * MIB;
        let rows = reservation_rows(16_303, 64, 7, MAX_SEQ);
        // The j7704 shape: all eight slots allocated at the charge, none live.
        let idle: Vec<SlotPages> = (0..8)
            .map(|i| SlotPages {
                live: false,
                materialised: true,
                have_trunk: g.trunk(2048).unwrap(),
                mapped_tokens: 2048,
                have_head: g.slot_has_head(i).then(|| g.head(2048).unwrap()),
                ..SlotPages::default()
            })
            .collect();
        let cold = numbers_of(&g, &idle, 0, 1002 * MIB, rows);
        assert_eq!(cold.pages.earmark, 1280 * MIB);
        let decision = page_decision(true, std::slice::from_ref(&cold), 256 * MIB);
        // No lane active and the 1002 MiB cannot cover the first tile plus the lane: refused up front
        // (a clean overload at admission), not admitted to fail mid-step.
        assert!(matches!(decision.verdict, PageVerdict::Never(_)));
        // The same card after one long tile ran: only the lane's own growth is promised.
        g.warm = true;
        let warm = numbers_of(&g, &idle, 0, 1002 * MIB, rows);
        assert_eq!(warm.pages.earmark, 0);
        assert_eq!(
            page_decision(true, std::slice::from_ref(&warm), 256 * MIB).verdict,
            PageVerdict::Fit
        );
        // A request inside the charge never waits for the cold reserve.
        let small = numbers_of(&g, &idle, 0, 0, 1000);
        assert_eq!(
            page_decision(true, &[small], 256 * MIB).verdict,
            PageVerdict::Fit
        );
    }

    /// Without the elastic reserve the warm flag is today's sticky rule (the caches never matter);
    /// with it the flag is exactly "the long-prefill caches are resident", so a release makes the
    /// rank cold again and the cold reserve is held for the next long prompt.
    #[test]
    fn the_warm_flag_follows_the_caches_only_with_the_elastic_reserve() {
        // Reserve off: sticky or a slot past the charge, whatever the caches hold.
        for resident in [false, true] {
            assert!(!lane_warm(false, resident, false, false));
            assert!(lane_warm(false, resident, true, false));
            assert!(lane_warm(false, resident, false, true));
        }
        // Reserve on: the caches alone, whatever the sticky history says.
        for (seen, past) in [(false, false), (true, false), (false, true), (true, true)] {
            assert!(lane_warm(true, true, seen, past));
            assert!(!lane_warm(true, false, seen, past));
        }
        // After a release (history says warm, caches gone) the earmark holds the cold reserve
        // again; the next long tile rebuilds the caches and the rank is warm.
        let config = compact_test_config();
        let mut g = geometry(&config, Some(mtp(None)));
        g.cold_reserve = 1280 * MIB;
        let slots = vec![SlotPages::default(); 8];
        g.warm = lane_warm(true, false, true, true);
        let released = numbers_of(&g, &slots, 0, 4096 * MIB, 1000);
        g.warm = lane_warm(true, true, true, true);
        let resident = numbers_of(&g, &slots, 0, 4096 * MIB, 1000);
        assert_eq!(
            released.pages.earmark - resident.pages.earmark,
            1280 * MIB,
            "the released rank holds exactly the cold reserve more"
        );
    }

    #[test]
    fn lendable_is_free_less_the_earmark_the_growth_and_the_margins() {
        let cp = ElasticNumbers {
            free: 2600 * MIB,
            earmark_noncold: 36 * MIB,
            outstanding: 0,
            ..ElasticNumbers::default()
        };
        // 256 page + 512 placement + 128 emergency = 896 MiB of margins.
        assert_eq!(cp.lendable(), (2600 - 36 - 896) * MIB);
        let grown = ElasticNumbers {
            outstanding: 900 * MIB,
            ..cp
        };
        assert_eq!(grown.lendable(), (2600 - 36 - 900 - 896) * MIB);
        // Saturating: a card with less than the margins free lends nothing.
        let tight = ElasticNumbers {
            free: 500 * MIB,
            ..cp
        };
        assert_eq!(tight.lendable(), 0);
    }

    #[test]
    fn the_elastic_action_waits_releases_then_lends_the_smallest_cap_bounded_bank() {
        let rank = |free_mib: u64| ElasticNumbers {
            free: free_mib * MIB,
            earmark_noncold: 36 * MIB,
            ..ElasticNumbers::default()
        };
        let cap = 1280 * MIB;
        let idle = ELASTIC_IDLE_STEPS;
        let both = [rank(2600), rank(2600)];
        // Not idle long enough, or the reserve off: nothing, whatever the numbers say.
        assert_eq!(elastic_action(cap, idle - 1, &both), ElasticAction::Nothing);
        assert_eq!(elastic_action(0, idle, &both), ElasticAction::Nothing);
        assert_eq!(elastic_action(cap, idle, &[]), ElasticAction::Nothing);
        // Idle with plenty of room: the cap.
        assert_eq!(elastic_action(cap, idle, &both), ElasticAction::Lend(cap));
        // The tighter rank bounds every rank's bank.
        let uneven = [rank(2600), rank(1500)];
        assert_eq!(
            elastic_action(cap, idle, &uneven),
            ElasticAction::Lend((1500 - 36 - 896) * MIB)
        );
        // Under the minimum bank on any rank nothing is lent.
        let thin = [rank(2600), rank(1000)];
        assert_eq!(elastic_action(cap, idle, &thin), ElasticAction::Nothing);
        // Resident caches on any rank are released first, lent or not.
        let mut holding = both;
        holding[1].caches_resident = true;
        assert_eq!(
            elastic_action(cap, idle, &holding),
            ElasticAction::ReleaseCaches
        );
        holding[0].lent = 1280 * MIB;
        assert_eq!(
            elastic_action(cap, idle, &holding),
            ElasticAction::ReleaseCaches
        );
        // A bank already lent on any rank is not lent again.
        let mut lent = both;
        lent[0].lent = 1280 * MIB;
        assert_eq!(elastic_action(cap, idle, &lent), ElasticAction::Nothing);
    }

    #[test]
    fn the_idle_count_spans_requests_and_restarts_on_a_long_prefill_or_a_reclaim() {
        // 64 step boundaries accumulate whatever busy period they fall in: a workload of short
        // replies (about 30 steps each, the loop returns between them) still reaches the window.
        let mut steps = 0;
        for _ in 0..ELASTIC_IDLE_STEPS {
            steps = elastic_idle_next(steps, 0, 0, false);
        }
        assert_eq!(steps, ELASTIC_IDLE_STEPS);
        // A prefill tile of >= 512 route rows restarts it.
        assert_eq!(elastic_idle_next(steps, 0, 0, true), 0);
        // So does any reclaim (the bundle's reclaim count moved), long prefill or not.
        assert_eq!(elastic_idle_next(steps, 3, 4, false), 0);
        assert_eq!(elastic_idle_next(steps, 3, 4, true), 0);
        // The next boundary after a restart counts again.
        assert_eq!(elastic_idle_next(0, 4, 4, false), 1);
        assert_eq!(elastic_idle_next(usize::MAX, 4, 4, false), usize::MAX);
    }

    #[test]
    fn only_route_rows_at_the_f16_floor_are_a_long_prefill() {
        assert!(!long_prefill_route_rows(1));
        assert!(!long_prefill_route_rows(511));
        assert!(long_prefill_route_rows(512));
        assert!(long_prefill_route_rows(4096));
    }

    #[test]
    fn the_earmark_covers_fresh_slots_and_unmapped_charge() {
        let config = compact_test_config();
        let g = geometry(&config, Some(mtp(Some(2))));
        let fresh_mtp = g.fresh_slot_bytes(0).unwrap();
        let fresh_ar = g.fresh_slot_bytes(5).unwrap();
        assert!(fresh_mtp > fresh_ar);
        // Eight never-allocated slots: the whole charge is earmarked and an
        // empty server has no outstanding growth.
        let slots = vec![SlotPages::default(); 8];
        let pages = rank_pages(&g, &slots, None, 0, 10_000 * MIB, 4096).unwrap();
        assert_eq!(pages.earmark, 2 * fresh_mtp + 6 * fresh_ar);
        assert_eq!(pages.outstanding, 0);
        assert_eq!(pages.active_lanes, 0);
        // A request of 4096 rows needs the growth over the charge on its
        // fresh MTP slot: trunk + head.
        let want = g.trunk(4096).unwrap() - g.trunk(2048).unwrap() + g.head(4096).unwrap()
            - g.head(2048).unwrap();
        assert_eq!(pages.need, want);
        // An allocated slot that mapped less than the charge still owns the rest.
        let mut slots = slots;
        slots[0] = SlotPages {
            materialised: true,
            have_trunk: g.trunk(1).unwrap(),
            have_head: Some(g.head(1).unwrap()),
            ..SlotPages::default()
        };
        let part = rank_pages(&g, &slots, None, 0, 10_000 * MIB, 1).unwrap();
        let charged_gap = g.trunk(2048).unwrap() - g.trunk(1).unwrap() + g.head(2048).unwrap()
            - g.head(1).unwrap();
        assert_eq!(
            part.earmark,
            charged_gap + fresh_mtp + 6 * fresh_ar,
            "an allocated slot owns only its unmapped charged pages"
        );
    }

    #[test]
    fn a_legacy_backend_and_a_retired_head_promise_nothing() {
        let config = compact_test_config();
        let mut g = geometry(&config, Some(mtp(None)));
        let long = MAX_SEQ;
        let slots = vec![mapped_lane(&g, 0, long, 2048), SlotPages::default()];
        let live = rank_pages(&g, &slots, None, 0, 100_000 * MIB, long).unwrap();
        let mut retired = slots.clone();
        retired[0].head_retired = true;
        let r = rank_pages(&g, &retired, None, 0, 100_000 * MIB, long).unwrap();
        assert_eq!(
            live.outstanding - r.outstanding,
            g.head(long).unwrap() - g.head(2048).unwrap()
        );
        g.backend = Qwen4KvBackend::Legacy;
        let legacy = rank_pages(&g, &slots, None, 0, 1, long).unwrap();
        assert_eq!((legacy.need, legacy.outstanding, legacy.earmark), (0, 0, 0));
        assert_eq!(page_verdict(&[legacy], 256 * MIB), PageVerdict::Fit);
    }

    #[test]
    fn the_gather_workspace_is_shared_not_summed() {
        let config = compact_test_config();
        let g = geometry(&config, None);
        let gather = GatherPages {
            kv_heads: 2,
            committed: 0,
            granule: GRANULE as u64,
        };
        let long = MAX_SEQ;
        let live = |rows| mapped_lane(&g, 0, rows, 2048);
        let slots = [live(long), live(long), SlotPages::default()];
        let with = rank_pages(&g, &slots, Some(&gather), 0, 100_000 * MIB, long).unwrap();
        let without = rank_pages(&g, &slots, None, 0, 100_000 * MIB, long).unwrap();
        let workspace = gather_pages(&gather, long).unwrap();
        let charged = gather_pages(&gather, 2048).unwrap();
        // Two live lanes at the same context share one workspace; the third
        // request adds nothing to it.
        assert_eq!(with.outstanding - without.outstanding, workspace - charged);
        assert_eq!(with.need, without.need);
        assert_eq!(with.earmark - without.earmark, charged);
    }

    #[test]
    fn reservation_rows_cap_at_the_admitted_context() {
        assert_eq!(reservation_rows(100, 50, 7, MAX_SEQ), 158);
        assert_eq!(reservation_rows(16_000, 1000, 7, MAX_SEQ), MAX_SEQ);
        assert_eq!(
            reservation_rows(usize::MAX, usize::MAX, 7, MAX_SEQ),
            MAX_SEQ
        );
    }

    #[test]
    fn spec_lane_config_is_unused_by_the_ar_slot_figure() {
        // An AR-only slot of a mixed store costs a mirror-less target state
        // whatever the store's spec.
        let config = compact_test_config();
        let context = Qwen4ContextCommit::new(Qwen4KvBackend::Vmm, MAX_SEQ, 2048, GRANULE);
        let g = geometry(&config, Some(mtp(Some(3))));
        let ar = Qwen4LaneStore::slot_alloc_bytes(&config, format(), &context, None).unwrap();
        assert_eq!(g.fresh_slot_bytes(3).unwrap(), ar);
        assert_eq!(g.fresh_slot_bytes(7).unwrap(), ar);
        assert!(g.fresh_slot_bytes(2).unwrap() > ar);
    }
}
