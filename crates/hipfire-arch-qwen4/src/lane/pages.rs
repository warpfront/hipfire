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

/// [`page_verdict`] under the kill switch: `enabled == false` is today's
/// admission (everything fits).
pub fn budget_verdict(enabled: bool, ranks: &[RankPages], margin: u64) -> PageVerdict {
    if enabled {
        page_verdict(ranks, margin)
    } else {
        PageVerdict::Fit
    }
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

/// One rank's numbers for a request reserving `rows` rows into the lowest free
/// slot. `free` is the rank's measured free device memory. A legacy context
/// backend maps nothing on demand: all zeros.
pub(crate) fn rank_pages(
    geometry: &PageGeometry<'_>,
    slots: &[SlotPages],
    gather: Option<&GatherPages>,
    rank: usize,
    free: u64,
    rows: usize,
) -> Result<RankPages, String> {
    let active_lanes = slots.iter().filter(|slot| slot.live).count();
    let mut pages = RankPages {
        rank,
        free,
        active_lanes,
        ..RankPages::default()
    };
    if geometry.backend != Qwen4KvBackend::Vmm {
        return Ok(pages);
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
    if let Some(index) = slots.iter().position(|slot| !slot.live) {
        pages.need = slot_growth(
            geometry,
            index,
            &slots[index],
            rows,
            trunk_charged,
            head_charged,
        )?;
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
        pages.need = pages.need.saturating_add(all.saturating_sub(live));
    }
    Ok(pages)
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
                format,
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
    /// Rows the load-time lane reservation charges per lane on this rank: the
    /// one expression [`crate::ep`]'s staging fit check and the loader's
    /// reservation use, so the earmark equals what the load reserved.
    pub(crate) fn lane_charge_rows(&self, gpu: &Gpu) -> usize {
        let max_seq = self.state.max_seq_len;
        let mode = self
            .forward_scratch_mode()
            .unwrap_or(crate::gpu_forward::Qwen4ForwardScratchMode::Full);
        let chunk = crate::gpu_forward::qwen4_prefill_chunk_requested(gpu, max_seq, mode);
        crate::ep::ep_lane_commit_rows(chunk)
    }

    /// Draft rows to add to a request's reservation ([`reservation_rows`]);
    /// `0` without a staged store.
    pub fn lane_page_draft_rows(&self) -> usize {
        self.lanes
            .as_ref()
            .map_or(0, |store| store.page_draft_rows())
    }

    /// This rank's page numbers for a request reserving `rows` rows
    /// ([`rank_pages`]).
    pub(crate) fn lane_page_numbers_local(
        &self,
        gpu: &Gpu,
        rows: usize,
    ) -> Result<RankPages, String> {
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
        let geometry = PageGeometry {
            config: &self.config,
            format: store.format,
            backend: store.backend,
            max_seq: self.state.max_seq_len,
            granularity,
            charge_rows: self.lane_charge_rows(gpu),
            spec: store.spec.as_ref().map(|spec| PageSpec {
                k: spec.config.k,
                spec_lanes: spec.config.spec_lanes,
                head_max_seq: spec.head_max_seq,
                copy: store.copy.as_ref().map(|copy| &copy.config),
            }),
        };
        let slots = store.page_slots(gpu)?;
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
        rank_pages(&geometry, &slots, gather.as_ref(), rank, free, rows)
            .map_err(|error| format!("rank {rank}: {error}"))
    }

    /// The page budget's verdict for a request reserving `rows` rows: this
    /// rank's numbers (single device) or every rank's (EP, evaluated once on
    /// the root so all ranks share the verdict). `Fit` under the kill switch
    /// and on a pipeline-parallel store, which has no per-rank lane pages.
    pub fn lane_page_verdict(&mut self, gpu: &mut Gpu, rows: usize) -> Result<PageVerdict, String> {
        // The kill switch measures nothing: today's admission, exactly.
        let enabled = page_budget_enabled();
        if !enabled || self.pp.is_some() {
            return Ok(budget_verdict(false, &[], 0));
        }
        let ranks = if self.ep.is_some() {
            self.ep_lane_page_numbers(gpu, rows)?
        } else {
            vec![self.lane_page_numbers_local(gpu, rows)?]
        };
        Ok(budget_verdict(enabled, &ranks, page_margin_bytes()))
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

    #[test]
    fn the_kill_switch_admits_exactly_as_before() {
        let hopeless = ranks_with(0, 4096 * MIB, 4096 * MIB, 4096 * MIB, 0);
        assert!(matches!(
            budget_verdict(true, &[hopeless], 256 * MIB),
            PageVerdict::Never(_)
        ));
        assert_eq!(
            budget_verdict(false, &[hopeless], 256 * MIB),
            PageVerdict::Fit
        );
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
