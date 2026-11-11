// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EP reserve decoupling, slice P4: the forward integration of an
//! `HIPFIRE_EP_RESERVE_DECOUPLE=1` load (`ep-reserve/DECISION.md` §4, §5, §7).
//!
//! A child module of `gpu_forward` so the monolithic forward keeps only the call
//! sites. Everything here is gated on `Qwen4GpuForward::ep_reserve` being
//! `Some`; a load without a plan never reaches any of it, and the ordinary and
//! single-device decode-share branches are untouched.
//!
//! * [`EpEpoch`] is the pure epoch/prompt state machine. The forward drives
//!   it; the CPU route-matrix tests drive the same code.
//! * [`EpReserveForward`] owns the P2 banks, the P3 loan, the fixed S2 host
//!   source table and the registered exchange root view.
//! * The `impl Qwen4GpuForward` block holds the route-safe bank switch, the
//!   all-rank acquire / restore protocol and the prompt nesting.
//!
//! # Protocol (§5)
//!
//! * Acquire (first borrowed call of an epoch): complete this rank's previous
//!   work, cross the abortable all-rank host barrier (the tile hook's), only
//!   then mark the loan dirty and zero the touched extents.
//! * A call of at most 512 rows (decode, verify, a tiny tail, the 512-row S1
//!   call) or any Full-table consumer: restore first (drain S2, restore the
//!   dirty extents, host-wait), then cross the barrier.
//! * Outermost prompt completion: the same global restore, synchronous and
//!   included in the prompt time.
//! * Clean epoch: a rank-local flag set only by a completed global restore and
//!   cleared only at an acquire. A clean call takes it and does nothing: no
//!   per-token barrier.
//! * Failure: local recovery only (no entry into an aborted barrier); the load
//!   is poisoned and every later forward refuses.

use super::ep_reserve_loan::{EpLoanDonor, EpLoanMirror, EpLoanTeardownPermit, EpReserveLoan};
use super::ep_reserve_scratch::{
    EpBankKind, EpBankRoute, EpDonorSegment, EpExchangeCapacity, EpReserveBanks, EpRootPlacement,
};
use super::*;
use crate::ep_reserve::{EpDonorSlot, EpReserveMeshReceipt, EpReservePlan};
use hipfire_runtime::ep_tile_exchange::{HOST_RESERVE_ACQUIRE, HOST_RESERVE_RESTORE};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Developer knob: per-prompt `EP_RESERVE_RECEIPT` records with the donor hashes
/// before and after the prompt (a full device read-back of the donors, a gate
/// diagnostic) and the loan's epoch log. Read once at load.
pub(crate) const EP_RESERVE_RECEIPT_ENV: &str = "HIPFIRE_EP_RESERVE_RECEIPT";

/// Developer knob (default `1`): `0` keeps the receipt records but skips the
/// donor-hash read-back, so a timed request's wall clock carries no hash cost.
pub(crate) const EP_RESERVE_RECEIPT_HASH_ENV: &str = "HIPFIRE_EP_RESERVE_RECEIPT_HASH";

/// Prefix of every receipt log line (`scripts/fn_ep_compare.py reserve harvest`).
const RECEIPT_PREFIX: &str = "EP_RESERVE_RECEIPT";

/// Sentinel of [`ep_reserve_receipt_scope`]: no oracle prompt is in scope.
const RECEIPT_SCOPE_UNSET: u64 = u64::MAX;

/// The oracle prompt index every `prompt_rank` record names (process-global:
/// the rank threads read it at the outermost prompt entry).
static RECEIPT_SCOPE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(RECEIPT_SCOPE_UNSET);

/// Developer/oracle hook: name the oracle prompt (`Some(i)`, before the prompt
/// runs) or clear it (`None`, after). While set, every outermost prompt entry
/// on any rank records `"prompt": i`; unset, a record falls back to the rank's
/// own entry sequence number. `u64::MAX` is the unset sentinel and is never a
/// prompt index. Reads and writes are `Relaxed`: the oracle sets the scope
/// before it starts the ranks' work (a mesh fan-out is a happens-before edge).
pub fn ep_reserve_receipt_scope(prompt: Option<u64>) {
    RECEIPT_SCOPE.store(
        prompt.unwrap_or(RECEIPT_SCOPE_UNSET),
        std::sync::atomic::Ordering::Relaxed,
    );
}

fn receipt_scope() -> Option<u64> {
    match RECEIPT_SCOPE.load(std::sync::atomic::Ordering::Relaxed) {
        RECEIPT_SCOPE_UNSET => None,
        prompt => Some(prompt),
    }
}

/// Which driver opened an outermost prompt entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum EpReceiptPhase {
    /// The target forward: prefill, advance, verify, decode.
    Target,
    /// The native-MTP fill of the prompt (`replica_mtp_prefill`,
    /// `mtp_prefill_ops`).
    MtpFill,
}

impl EpReceiptPhase {
    fn name(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::MtpFill => "mtp_fill",
        }
    }
}

/// Per-rank count of outermost entries within one `(prompt, phase)`.
#[derive(Debug, Default)]
pub(crate) struct EpReceiptOrdinals(BTreeMap<(u64, EpReceiptPhase), u64>);

impl EpReceiptOrdinals {
    /// The `(prompt, phase, ordinal)` key of the next outermost entry; `seq` is
    /// the rank's own entry sequence, the prompt when no scope is set.
    fn next_key(&mut self, seq: u64, mtp_fill: bool) -> (u64, EpReceiptPhase, u64) {
        let prompt = receipt_scope().unwrap_or(seq);
        let phase = if mtp_fill {
            EpReceiptPhase::MtpFill
        } else {
            EpReceiptPhase::Target
        };
        let count = self.0.entry((prompt, phase)).or_insert(0);
        let ordinal = *count;
        *count += 1;
        (prompt, phase, ordinal)
    }
}

/// Prompt-record state of an ORDINARY EP load under
/// `HIPFIRE_EP_RESERVE_RECEIPT=1` (no reserve: the off arm of the gate). Pure
/// host bookkeeping; nothing here touches the device.
#[derive(Debug, Default)]
pub(crate) struct EpOrdinaryReceipt {
    depth: u32,
    seq: u64,
    ordinals: EpReceiptOrdinals,
    prompt: Option<EpPromptRecord>,
}

// ---------------------------------------------------------------------------
// Pure epoch / prompt state machine
// ---------------------------------------------------------------------------

/// What one forward call needs before and around its bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpCallPlan {
    pub bank: EpBankKind,
    /// Restore globally (drain S2, restore donors, barrier) before this call.
    pub restore_first: bool,
    /// Open a borrowed epoch (barrier, then overwrite donors) before this call.
    pub acquire: bool,
}

/// What the end of a whole-prompt entry requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpPromptEnd {
    /// An inner entry: nothing, the outermost one finishes the prompt.
    Nested,
    /// Outermost, donors already whole (never borrowed, or restored before a tail).
    Clean,
    /// Outermost and successful with a dirty epoch: one synchronous global restore.
    Restore,
    /// Outermost, failed, dirty epoch: local blocking recovery, no barrier.
    Recover,
}

/// The rank-local epoch proof and prompt nesting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EpEpoch {
    dirty: bool,
    depth: u32,
    epochs: u64,
}

impl EpEpoch {
    /// A borrowed epoch is open (donors dirty): not globally resident.
    pub(crate) fn dirty(&self) -> bool {
        self.dirty
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> u32 {
        self.depth
    }

    /// Borrowed epochs opened so far.
    pub(crate) fn epochs(&self) -> u64 {
        self.epochs
    }

    /// The plan of a call of `rows` rows whose sealed route is `route`. At most
    /// `resident_rows`: the resident bank after any dirty epoch is restored.
    /// More: the borrowed bank, only on the tile-exchange route (a wide
    /// Full-table call is refused before anything is written).
    pub(crate) fn plan_call(
        &self,
        rows: usize,
        resident_rows: usize,
        prompt_rows: usize,
        route: EpBankRoute,
    ) -> Result<EpCallPlan, String> {
        if rows == 0 {
            return Err("an EP reserve call carries at least one row".to_string());
        }
        if rows > prompt_rows {
            return Err(format!(
                "a {rows}-row call exceeds the {prompt_rows}-row prompt capacity"
            ));
        }
        if rows <= resident_rows {
            return Ok(EpCallPlan {
                bank: EpBankKind::Resident,
                restore_first: self.dirty,
                acquire: false,
            });
        }
        if route != EpBankRoute::TileExchange {
            return Err(format!(
                "a {rows}-row call would read Full peer tables, which read the donor layers a \
                 borrowed bank overwrites (the resident bank holds {resident_rows} rows): \
                 unsupported routing on an EP reserve-sharing load"
            ));
        }
        Ok(EpCallPlan {
            bank: EpBankKind::Borrowed,
            restore_first: false,
            acquire: !self.dirty,
        })
    }

    /// The all-rank barrier passed and the loan is dirty.
    pub(crate) fn acquired(&mut self) {
        self.dirty = true;
        self.epochs += 1;
    }

    /// A global restore completed (local restore proven, barrier crossed), or a
    /// local recovery completed on a failed path.
    pub(crate) fn restored(&mut self) {
        self.dirty = false;
    }

    /// A whole-prompt entry begins; `true` when it is the outermost.
    pub(crate) fn begin_prompt(&mut self) -> bool {
        self.depth += 1;
        self.depth == 1
    }

    /// A whole-prompt entry ends (`ok`: it returned `Ok`).
    pub(crate) fn end_prompt(&mut self, ok: bool) -> EpPromptEnd {
        self.depth = self.depth.saturating_sub(1);
        if self.depth > 0 {
            return EpPromptEnd::Nested;
        }
        match (self.dirty, ok) {
            (false, _) => EpPromptEnd::Clean,
            (true, true) => EpPromptEnd::Restore,
            (true, false) => EpPromptEnd::Recover,
        }
    }
}

// ---------------------------------------------------------------------------
// Exchange capacity
// ---------------------------------------------------------------------------

/// Tiles the owner-routed exchange is registered for, from the 4096-row plan:
/// the masked-id capacity and the grouped-down root at one BF16 tile each,
/// whichever is smaller. (`Qwen4TileRoute::admit` takes the same minimum from
/// the selected bank on an ordinary load.)
pub(super) fn exchange_max_tiles(capacity: &EpExchangeCapacity, hidden: usize) -> usize {
    let tile_bytes = (16 * hidden * DType::BF16.size()) as u64;
    let by_root = (capacity.grouped_down_bytes / tile_bytes.max(1)) as usize;
    capacity.tile_ids.min(by_root)
}

// ---------------------------------------------------------------------------
// The one sealed owner-routed decision
// ---------------------------------------------------------------------------

/// Whether a call of `n` rows on `ep` takes the owner-routed tile exchange:
/// `forward_chunk_scoped` and the reserve bank switch share this one predicate
/// so the bank is chosen from the same decision the forward then executes.
/// `route_n` is the routing row count under the call's routing scopes.
pub(super) fn ep_tile_active(
    ep: Option<&Qwen4ForwardEp>,
    gpu: &Gpu,
    config: &Qwen4Config,
    n: usize,
    route_n: usize,
) -> bool {
    let Some(ep) = ep else { return false };
    let few_rows = (2..=QWEN4_EP_GATHER_MAX_ROWS).contains(&route_n) && route_n == n;
    let Some(route) = ep
        .tile
        .as_ref()
        .filter(|route| route_n > 1 && !few_rows && route.hook.is_some())
    else {
        return false;
    };
    route_n == n
        && if route.wmma {
            gpu.qwen4_moe_f16_wmma_grouped_applies(
                route_n,
                config.moe_intermediate_size,
                config.hidden_size,
            )
        } else {
            gpu.qwen4_moe_sym_iu4_applies(route_n)
        }
}

/// Whether a call of `n` rows (`route_n` routing rows) on an EP forward takes the trunk-shard
/// PARTIAL chunk route (`partial_active`: the thread's [`crate::trunk_route`] FFN route is
/// installed): a wide chunk binds this rank's COMPACT table (non-owned experts the zero
/// dummy) and the all-reduce of its FFN HC write completes the FFN. `forward_chunk_tile` and
/// the reserve's bank choice share this one predicate.
pub(super) fn ep_trunk_partial_chunk(
    ep_present: bool,
    n: usize,
    route_n: usize,
    partial_active: bool,
) -> bool {
    let few_rows = (2..=QWEN4_EP_GATHER_MAX_ROWS).contains(&route_n) && route_n == n;
    ep_present && partial_active && route_n > 1 && !few_rows
}

/// The bank route of a call wider than the resident bank. The borrowed bank overwrites donor
/// expert bytes, so it may serve only a route whose peers read no donor entry: the owner-routed
/// tile exchange (peers read the persistent grouped-down root) or the trunk-shard partial chunk
/// (this rank's own compact table, no peer expert read at all). Any other wide call is the
/// full-table route, which the epoch refuses above the resident bank.
pub(super) fn ep_wide_bank_route(tile_active: bool, trunk_partial_chunk: bool) -> EpBankRoute {
    if tile_active || trunk_partial_chunk {
        EpBankRoute::TileExchange
    } else {
        EpBankRoute::FullTable
    }
}

// ---------------------------------------------------------------------------
// Lane prefill tiles and the expert stage
// ---------------------------------------------------------------------------

/// [`QWEN4_EP_LANE_PREFILL_STAGE_MIN_ROWS_ENV`] parsed: an integer of at least 2, else `None`
/// (unset, empty, not a number, `0`, `1`: no lane threshold, the stage keeps its own).
pub(crate) fn ep_lane_stage_knob_from(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&rows| rows >= 2)
}

/// The lane stage threshold knob, read once per process (every rank thread agrees).
static EP_LANE_STAGE_KNOB: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
    ep_lane_stage_knob_from(
        hipfire_config::developer_var(QWEN4_EP_LANE_PREFILL_STAGE_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
    )
});

pub(super) fn ep_lane_stage_knob() -> Option<usize> {
    *EP_LANE_STAGE_KNOB
}

/// The fewest rows at which a lane prefill tile may stage whatever the knob says. The stage is
/// only ever used by a forward on the owner-routed compact route (`ep_tile_active`: at least
/// [`rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS`] routing rows), and on a decoupled-reserve
/// load (`resident_call_limit` = the resident bank's call limit) strictly above that limit: a
/// call of at most the limit runs on the resident bank, which a borrowed-only stage must never
/// serve (`ep_reserve_stage_refusal`). Lowering the knob below this floor is inert, never unsafe.
pub(crate) fn ep_lane_stage_floor(resident_call_limit: Option<usize>) -> usize {
    resident_call_limit
        .map_or(0, |limit| limit.saturating_add(1))
        .max(rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS)
}

/// The row threshold a forward of the expert stage compares its rows with. `stage_min_rows` is
/// the stage's own (`HIPFIRE_QWEN4_EXPERT_STAGE_MIN_ROWS`, else the arch default: 2048 on
/// gfx1201). Only an EP lane prefill tile (`ep_lane_tile`) with the knob set takes
/// `max(knob, floor)`; every other forward, and every forward with the knob unset, keeps
/// `stage_min_rows` exactly.
pub(crate) fn ep_lane_stage_min_rows(
    stage_min_rows: usize,
    knob: Option<usize>,
    ep_lane_tile: bool,
    resident_call_limit: Option<usize>,
) -> usize {
    match knob {
        Some(rows) if ep_lane_tile => rows.max(ep_lane_stage_floor(resident_call_limit)),
        _ => stage_min_rows,
    }
}

/// The MoE route of one EP lane prefill tile or wave on this rank (the `route=` of the
/// `HIPFIRE_CB_PHASES` per-tile line).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LaneTileRoute {
    /// The EP full table: every expert of both ranks read through this rank's links.
    FullTable,
    /// The owner-routed compact route with the tile exchange (`ep_tile_active`).
    Tile,
    /// The trunk shard's partial chunk route (`ep_trunk_partial_chunk`): this rank's compact
    /// table, non-owned experts run on the zero dummy, the all-reduce completes the FFN.
    Partial,
    /// [`Self::Partial`] with the partial tile mask (`HIPFIRE_QWEN4_EP_TRUNK_SHARD_PMASK`): the
    /// non-owned tiles are masked out of the grouped launches, so the rank computes only its
    /// owned experts.
    PartialMask,
}

impl LaneTileRoute {
    /// The route a chunk bound: the trunk shard's partial route takes precedence over the tile
    /// exchange (`forward_chunk_scoped`), the mask applies only on the partial route.
    pub(crate) fn of(tile_active: bool, trunk_partial: bool, partial_mask: bool) -> Self {
        match (trunk_partial, partial_mask, tile_active) {
            (true, true, _) => Self::PartialMask,
            (true, false, _) => Self::Partial,
            (false, _, true) => Self::Tile,
            (false, _, false) => Self::FullTable,
        }
    }
}

/// The route one EP lane prefill tile took on this rank, kept for the `HIPFIRE_CB_PHASES`
/// per-tile line (`[fn-lane-tile]`); it never feeds a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaneTileNote {
    /// Rows of the tile.
    pub(crate) rows: usize,
    /// The MoE route the tile bound.
    pub(crate) route: LaneTileRoute,
    /// The tile staged the host experts through the expert stage.
    pub(crate) staged: bool,
    /// The row threshold the tile compared with (`None`: this load has no expert stage).
    pub(crate) stage_min_rows: Option<usize>,
}

impl LaneTileNote {
    /// Short route label of the per-tile line.
    pub(crate) fn label(&self) -> &'static str {
        match (self.route, self.staged) {
            (LaneTileRoute::FullTable, _) => "full-table",
            (LaneTileRoute::Tile, false) => "tile",
            (LaneTileRoute::Tile, true) => "tile+stage",
            (LaneTileRoute::Partial, false) => "partial",
            (LaneTileRoute::Partial, true) => "partial+stage",
            (LaneTileRoute::PartialMask, false) => "partial-mask",
            (LaneTileRoute::PartialMask, true) => "partial-mask+stage",
        }
    }
}

// ---------------------------------------------------------------------------
// The EP prefill tile route's row floor (`HIPFIRE_EP_PREFILL_TILE_MIN_ROWS`)
// ---------------------------------------------------------------------------

/// [`QWEN4_EP_PREFILL_TILE_MIN_ROWS_ENV`] parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PrefillTileKnob {
    /// Unset or empty: the default floor (lowered on the A8 chain, else 512).
    Unset,
    /// A usable floor (the value as given; [`ep_prefill_tile_sym_floor`] clamps it to the
    /// route's own range).
    Floor(usize),
    /// Refused: the reason, logged once; the default floor applies.
    Refused(String),
}

/// Parse the knob. An integer in `LOWEST..=` anything is usable; below
/// [`rdna_compute::gemm::QWEN4_MOE_SYM_IU4_LOWEST_FLOOR`] (the few-row routes' rows: the gathered
/// decode / verify MoE must keep them) or not an integer is refused.
pub(crate) fn ep_prefill_tile_knob_from(value: Option<&str>) -> PrefillTileKnob {
    let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) else {
        return PrefillTileKnob::Unset;
    };
    let lowest = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_LOWEST_FLOOR;
    match value.parse::<usize>() {
        Err(_) => PrefillTileKnob::Refused(format!("{value:?} is not an integer")),
        Ok(rows) if rows < lowest => PrefillTileKnob::Refused(format!(
            "{rows} is below the floor of {lowest} rows: calls of at most {} rows are decode and verify and keep the gathered MoE",
            lowest - 1
        )),
        Ok(rows) => PrefillTileKnob::Floor(rows),
    }
}

/// The knob, read once per process (every rank thread agrees); a refusal is logged once.
static EP_PREFILL_TILE_KNOB: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
    match ep_prefill_tile_knob_from(
        hipfire_config::developer_var(QWEN4_EP_PREFILL_TILE_MIN_ROWS_ENV)
            .ok()
            .as_deref(),
    ) {
        PrefillTileKnob::Unset => None,
        PrefillTileKnob::Floor(rows) => Some(rows),
        PrefillTileKnob::Refused(reason) => {
            eprintln!(
                "  qwen4 EP prefill tile route: {QWEN4_EP_PREFILL_TILE_MIN_ROWS_ENV} ignored ({reason}); the default floor applies"
            );
            None
        }
    }
});

pub(super) fn ep_prefill_tile_knob() -> Option<usize> {
    *EP_PREFILL_TILE_KNOB
}

/// The symmetric-IU4 row floor ([`Gpu::qwen4_moe_sym_iu4_floor`]) a forward runs under. Lowered
/// only for a PREFILL forward (`prefill_scope`: lane tile, singleton prefill, MTP prompt-fill
/// tile; never decode, verify or the MTP head) of an EP load whose owner-routed tile exchange is
/// installed (`tile_hook`: the forward then takes the compact route with the exchange, never
/// the symmetric arm on the full table), and never raised above today's floor or lowered below
/// the few-row routes' rows. `a8_chain`: the A8 owner-routed chain is the active chain of this
/// forward (`Gpu::qwen4_moe_ep_a8_requested`; the sub-512 floor is quality-neutral only on A8,
/// the A4 chain fails the paired KLD there): an unset knob then lowers the floor to
/// [`EP_PREFILL_TILE_DEFAULT_FLOOR`], else it stays 512. An explicit knob value wins either
/// way (512 is the kill switch of the floor alone). Pure: the lane tile and the singleton of the
/// same rows read the same value, so they take the same route.
pub(crate) fn ep_prefill_tile_sym_floor(
    knob: Option<usize>,
    prefill_scope: bool,
    tile_hook: bool,
    a8_chain: bool,
) -> usize {
    let default = rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS;
    if !(prefill_scope && tile_hook) {
        return default;
    }
    match knob {
        Some(rows) => rows.clamp(rdna_compute::gemm::QWEN4_MOE_SYM_IU4_LOWEST_FLOOR, default),
        None if a8_chain => EP_PREFILL_TILE_DEFAULT_FLOOR,
        None => default,
    }
}

/// The floor of an EP prefill forward on the A8 chain when the knob is unset: the lowest one
/// (9 rows: decode and verify, at most 8 rows, keep the gathered route).
pub(crate) const EP_PREFILL_TILE_DEFAULT_FLOOR: usize =
    rdna_compute::gemm::QWEN4_MOE_SYM_IU4_LOWEST_FLOOR;

// ---------------------------------------------------------------------------
// Plan geometry (pure)
// ---------------------------------------------------------------------------

/// Donor segments and root placements of `plan`, with each donor tensor's
/// device address from `owner_of(layer, slot) -> (address, owner bytes)`. A
/// segment lends `capacity` bytes and must lie inside its owner.
pub(crate) fn ep_reserve_layout(
    plan: &EpReservePlan,
    mut owner_of: impl FnMut(usize, EpDonorSlot) -> Result<(usize, u64), Qwen4GpuForwardError>,
) -> Result<(Vec<EpDonorSegment>, Vec<EpRootPlacement>), Qwen4GpuForwardError> {
    let mut segments = Vec::with_capacity(plan.segments.len());
    for segment in &plan.segments {
        let (address, owner_bytes) = owner_of(segment.layer, segment.slot)?;
        if address == 0 || owner_bytes < segment.capacity {
            return Err(invalid(format!(
                "EP reserve donor layer {} {:?} holds {owner_bytes} bytes at {address:#x}, the plan \
                 lends {}",
                segment.layer, segment.slot, segment.capacity
            )));
        }
        segments.push(EpDonorSegment {
            address,
            bytes: segment.capacity,
        });
    }
    let mut placements = Vec::with_capacity(plan.placements.len());
    for placement in &plan.placements {
        let segment = plan
            .segments
            .get(placement.segment)
            .filter(|segment| (segment.layer, segment.slot) == (placement.layer, placement.slot))
            .ok_or_else(|| {
                invalid(format!(
                    "EP reserve root {:?} names segment {} that is not layer {} {:?}",
                    placement.field, placement.segment, placement.layer, placement.slot
                ))
            })?;
        let _ = segment;
        placements.push(EpRootPlacement {
            field: placement.field,
            segment: placement.segment,
            offset: placement.offset,
            bytes: placement.bytes,
        });
    }
    Ok((segments, placements))
}

/// The boundaries of `plan` the forward relies on: contiguous scratch donors
/// ending at the resident count, S2 donors `H-2`/`H-1` ordered by parity and
/// disjoint from every scratch donor, all inside the trunk.
pub(crate) fn check_stage_plan(plan: &EpReservePlan) -> Result<(), String> {
    if plan.explicit_donors {
        // Caller-chosen donors (unit-plan regime): any ascending set of
        // distinct trunk layers, no S2 donors (the unit stage serves tiles).
        let sorted = plan.donor_layers.windows(2).all(|pair| pair[0] < pair[1]);
        if plan.donor_layers.is_empty()
            || !sorted
            || plan
                .donor_layers
                .iter()
                .any(|&layer| layer >= plan.trunk_layers)
            || plan.donor_first_layer != plan.donor_layers[0]
        {
            return Err(format!(
                "explicit scratch donors {:?} are not an ascending set of distinct layers of the \
                 {}-layer trunk",
                plan.donor_layers, plan.trunk_layers
            ));
        }
        return Ok(());
    }
    let first = plan.donor_first_layer;
    let end = plan.resident_layers;
    if plan.donor_layers.is_empty()
        || plan.donor_layers != (first..end).collect::<Vec<_>>()
        || end >= plan.trunk_layers
        || first < 2
    {
        return Err(format!(
            "scratch donors {:?} are not a suffix [H, R) of the resident prefix {end} of a \
             {}-layer trunk with two layers below it",
            plan.donor_layers, plan.trunk_layers
        ));
    }
    let [even, odd] = plan.s2_donor_layers;
    if even % 2 != 0 || odd % 2 != 1 || [even.min(odd), even.max(odd)] != [first - 2, first - 1] {
        return Err(format!(
            "S2 donors {:?} are not layers {} and {} ordered even parity first",
            plan.s2_donor_layers,
            first - 2,
            first - 1
        ));
    }
    if plan
        .donor_layers
        .iter()
        .any(|layer| plan.s2_donor_layers.contains(layer))
    {
        return Err("a layer is both a scratch donor and an S2 donor".to_string());
    }
    Ok(())
}

/// The stage host-source resolver: per donor layer (the suffix `[H, R)`, or a unit plan's
/// explicit whole-Resident layers) the pinned CPU span of each loan mirror (gate/up, down).
/// Layers outside the donor set resolve to their original pinned owners at the stage, as on an
/// ordinary load.
pub(crate) struct EpStageHostSources {
    /// The donor layers, ascending; `spans[i]` belongs to `layers[i]`.
    layers: Vec<usize>,
    spans: Vec<[(usize, usize); 2]>,
    /// Device-visible base address of each mirror (the pinned allocation's
    /// device alias), `device[i]` for `layers[i]`, `[gate/up, down]`.
    device: Vec<[usize; 2]>,
}

impl EpStageHostSources {
    pub(crate) fn new(
        gpu: &Gpu,
        loan: &EpReserveLoan,
        plan: &EpReservePlan,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let mut spans = Vec::with_capacity(plan.donor_layers.len());
        let mut device = Vec::with_capacity(plan.donor_layers.len());
        for &layer in &plan.donor_layers {
            let span = |slot: EpDonorSlot| -> Result<(usize, usize), Qwen4GpuForwardError> {
                let mirror = loan.mirror(layer, slot).ok_or_else(|| {
                    invalid(format!(
                        "EP reserve loan has no mirror of layer {layer} {slot:?}"
                    ))
                })?;
                let host = gpu
                    .host_bytes(mirror.tensor)
                    .filter(|host| host.len() == mirror.bytes);
                host.map(|host| (host.as_ptr() as usize, host.len()))
                    .ok_or_else(|| {
                        invalid(format!(
                            "EP reserve mirror of layer {layer} {slot:?} has no pinned host bytes"
                        ))
                    })
            };
            spans.push([span(EpDonorSlot::GateUp)?, span(EpDonorSlot::Down)?]);
            let base = |slot: EpDonorSlot| -> Result<usize, Qwen4GpuForwardError> {
                loan.mirror(layer, slot)
                    .map(|mirror| mirror.tensor.buf.as_ptr() as usize)
                    .ok_or_else(|| {
                        invalid(format!(
                            "EP reserve loan has no mirror of layer {layer} {slot:?}"
                        ))
                    })
            };
            device.push([base(EpDonorSlot::GateUp)?, base(EpDonorSlot::Down)?]);
        }
        Ok(Self {
            layers: plan.donor_layers.clone(),
            spans,
            device,
        })
    }

    /// The mirrors' complete pinned bytes of `layer` (gate/up, down), `None`
    /// outside the donor layers.
    pub(crate) fn pair(&self, layer: usize) -> Option<[&[u8]; 2]> {
        let spans = self.spans.get(self.layers.binary_search(&layer).ok()?)?;
        // SAFETY: each span is the pinned host allocation of a loan mirror
        // (`EpReserveLoan` owns it until its `release`, which the forward runs
        // after the stage that reads these bytes is freed); nothing writes a
        // mirror after materialization, and the slice lives only as long as
        // this borrow of the forward-owned resolver.
        Some(
            spans.map(|(address, len)| unsafe {
                std::slice::from_raw_parts(address as *const u8, len)
            }),
        )
    }

    /// [`Self::pair`] with each mirror's device-visible base address (the
    /// pinned allocation's device alias) beside its pinned bytes, `None`
    /// outside the donor layers. The unit stage's donor regions read this
    /// (`slab_stage::donor_stage_regions`).
    pub(crate) fn pair_with_device(&self, layer: usize) -> Option<[(&[u8], usize); 2]> {
        let bytes = self.pair(layer)?;
        let device = self.device.get(self.layers.binary_search(&layer).ok()?)?;
        Some([(bytes[0], device[0]), (bytes[1], device[1])])
    }

    /// The donor layers with a loan mirror, ascending (the plan's
    /// `donor_layers`).
    pub(crate) fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// A resolver over caller-built spans, for the CPU tests of the unit
    /// stage's donor regions: `parts[i] = (layer, [(cpu address, len);2],
    /// [device base;2])`, layers ascending.
    #[cfg(test)]
    pub(crate) fn from_parts(parts: &[(usize, [(usize, usize); 2], [usize; 2])]) -> Self {
        Self {
            layers: parts.iter().map(|part| part.0).collect(),
            spans: parts.iter().map(|part| part.1).collect(),
            device: parts.iter().map(|part| part.2).collect(),
        }
    }
}

/// What the reserve admission reads of the built expert stage.
pub(crate) struct EpReserveStageFacts {
    /// The stage's first host (staged) layer.
    pub first_host: usize,
    /// The stage's S2 donor layers (the dedicated unit stage has none: `[usize::MAX; 2]`; a
    /// borrowed unit stage names its two lent layers, which must not be scratch donors).
    pub donors: [usize; 2],
    pub min_rows: usize,
    /// The stage reads every plan donor layer from its own unit-stage slot.
    pub serves_donor_layers: bool,
}

/// Why the built stage cannot serve the reserve `plan`, `None` when it can. A suffix plan
/// needs the S2 stage with exactly the plan's first host layer and S2 donors; a plan with
/// explicit donors (`plan.explicit_donors`, no S2 donors) needs the unit stage serving every
/// donor layer. Either stage must be borrowed-only: its `min_rows` above the resident bank.
pub(crate) fn ep_reserve_stage_refusal(
    plan: &EpReservePlan,
    resident_rows: usize,
    stage: &EpReserveStageFacts,
) -> Option<String> {
    let rows_bad = stage.min_rows <= resident_rows;
    if plan.explicit_donors {
        let lent_donor = plan
            .donor_layers
            .iter()
            .find(|layer| stage.donors.contains(layer));
        if let Some(layer) = lent_donor {
            return Some(format!(
                "EP reserve stage mismatch: scratch donor layer {layer} is also a lent stage layer {:?}",
                stage.donors
            ));
        }
        if rows_bad || !stage.serves_donor_layers {
            return Some(format!(
                "EP reserve stage mismatch: the unit stage serves the scratch donors {:?}: {} / \
                 min rows {} (a borrowed-only stage must not serve the {resident_rows}-row \
                 resident bank)",
                plan.donor_layers, stage.serves_donor_layers, stage.min_rows
            ));
        }
        return None;
    }
    if rows_bad
        || stage.first_host != plan.donor_first_layer
        || stage.donors != plan.s2_donor_layers
    {
        return Some(format!(
            "EP reserve stage mismatch: first host layer {} / donors {:?} / min rows {} \
             against the plan's {} / {:?} (a borrowed-only stage must not serve the \
             {resident_rows}-row resident bank)",
            stage.first_host,
            stage.donors,
            stage.min_rows,
            plan.donor_first_layer,
            plan.s2_donor_layers
        ));
    }
    None
}

/// `mirror` as a typed, host-mapped source descriptor for a borrowed runtime:
/// the loan's pinned bytes under the donor's expert dtype. It is a view of the
/// loan-owned allocation and must never reach a free path (the runtime builder
/// only reads it).
pub(crate) fn typed_mirror_source(mirror: &EpLoanMirror<'_>, dtype: DType) -> GpuTensor {
    GpuTensor {
        // SAFETY: the mirror is a live pinned host allocation of `bytes` bytes
        // owned by the loan; the descriptor marks the bytes host-mapped (they
        // are) and is dropped by the runtime builder without being freed.
        buf: unsafe {
            hip_bridge::DeviceBuffer::from_host_mapped(mirror.tensor.buf.as_ptr(), mirror.bytes)
        },
        shape: vec![mirror.bytes],
        dtype,
    }
}

// ---------------------------------------------------------------------------
// Receipts
// ---------------------------------------------------------------------------

/// One restore event of a prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EpRestoreEvent {
    /// Index of the forward the restore precedes; the forward count for the
    /// outer prompt completion.
    pub before_chunk: usize,
    pub bytes: u64,
    /// Drain of S2 plus the loan restore, host-waited.
    pub restore_ms: f64,
    /// Time in the all-rank barrier after the local restore.
    pub barrier_wait_ms: f64,
    /// Time draining the S2 stage's outstanding DMA (what the trunk did not hide).
    pub s2_dma_tail_ms: f64,
}

/// One outermost prompt entry on this rank.
#[derive(Debug)]
pub(crate) struct EpPromptRecord {
    /// The `(prompt, phase, ordinal)` key: the oracle prompt in scope (else the
    /// rank's entry sequence), the driver, and the entry's count within them.
    pub prompt: u64,
    pub phase: EpReceiptPhase,
    pub ordinal: u64,
    pub chunk_rows: Vec<usize>,
    pub banks: Vec<EpBankKind>,
    pub restores: Vec<EpRestoreEvent>,
    pub epochs_before: u64,
    started: Instant,
    /// Time spent hashing donors, excluded from the prompt time.
    excluded: Duration,
    exchange_before: Option<usize>,
    hash_before: Option<String>,
}

fn bank_name(bank: EpBankKind) -> &'static str {
    match bank {
        EpBankKind::Resident => "resident",
        EpBankKind::Borrowed => "borrowed",
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn fnv_fold(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// FNV-1a 64 of a `source_generation` string, the mesh receipt's `u64`.
pub(crate) fn generation_fingerprint(generation: &str) -> u64 {
    fnv_fold(0xcbf2_9ce4_8422_2325, generation.as_bytes())
}

/// The prompt record as one `EP_RESERVE_RECEIPT` JSON object (kind
/// `prompt_rank`: this rank's half of the schema's `prompt` record; the harvest
/// merges the two ranks' halves, per-rank scalars becoming per-rank lists).
pub(crate) fn prompt_rank_json(
    rank: usize,
    record: &EpPromptRecord,
    borrowed_epochs: u64,
    elapsed: Duration,
    exchange_after: Option<usize>,
    hash_after: Option<String>,
) -> serde_json::Value {
    let restores: Vec<_> = record
        .restores
        .iter()
        .map(|event| {
            serde_json::json!({
                "before_chunk": event.before_chunk,
                "bytes": event.bytes,
                "restore_ms": event.restore_ms,
                "barrier_wait_ms": event.barrier_wait_ms,
                "s2_dma_tail_ms": event.s2_dma_tail_ms,
            })
        })
        .collect();
    let mut json = serde_json::json!({
        "kind": "prompt_rank",
        "rank": rank,
        "prompt": record.prompt,
        "phase": record.phase.name(),
        "ordinal": record.ordinal,
        "rows": record.chunk_rows.iter().sum::<usize>(),
        "chunk_rows": record.chunk_rows,
        "banks": record.banks.iter().map(|bank| bank_name(*bank)).collect::<Vec<_>>(),
        "restores": restores,
        "borrowed_epochs": borrowed_epochs,
        "elapsed_ms_incl_restore": ms(elapsed),
    });
    if let (Some(before), Some(after)) = (record.exchange_before, exchange_after) {
        json["exchange_address_before"] = serde_json::Value::String(format!("{before:#x}"));
        json["exchange_address_after"] = serde_json::Value::String(format!("{after:#x}"));
    }
    if let (Some(before), Some(after)) = (record.hash_before.as_ref(), hash_after) {
        json["donor_hash_before"] = serde_json::Value::String(before.clone());
        json["donor_hash_after"] = serde_json::Value::String(after);
    }
    json
}

fn emit_receipt(json: &serde_json::Value) {
    eprintln!("{RECEIPT_PREFIX} {json}");
}

// ---------------------------------------------------------------------------
// The owner
// ---------------------------------------------------------------------------

/// Per-rank EP reserve state of a forward.
pub(crate) struct EpReserveForward {
    pub(super) plan: Arc<EpReservePlan>,
    /// `u64` fingerprint of the sealed source generation (mesh receipt).
    pub(super) source_generation: u64,
    pub(super) banks: EpReserveBanks,
    pub(super) loan: EpReserveLoan,
    pub(super) host_sources: EpStageHostSources,
    /// Non-owning F32 view of the whole persistent grouped-down root: what the
    /// tile exchange registers, whichever bank is selected.
    pub(super) exchange_root: GpuTensor,
    pub(super) epoch: EpEpoch,
    /// Set on any failure after the reduced reserve was in use: every later
    /// forward refuses (drain and reload).
    pub(super) poisoned: Option<String>,
    /// `HIPFIRE_EP_RESERVE_RECEIPT`: per-prompt records and donor hashes.
    pub(super) receipts: bool,
    /// `HIPFIRE_EP_RESERVE_RECEIPT_HASH` (receipts only): read the donor hashes.
    pub(super) hashes: bool,
    /// The weight shadows live in the donor segments (rebuilt at every epoch
    /// open, never ready outside one) and the resident bank serves only calls
    /// below the shadow-reading routes' 512 rows.
    pub(super) shadows: bool,
    pub(super) prompt: Option<EpPromptRecord>,
    pub(super) prompt_seq: u64,
    /// Per-rank entry counts within one `(prompt, phase)` (receipts only).
    pub(super) ordinals: EpReceiptOrdinals,
}

/// What `EpReserveForward::teardown` hands back for the forward's free order:
/// banks after the scratch, the loan (with its permit) after the stage.
pub(crate) struct EpReserveTeardown {
    pub banks: EpReserveBanks,
    pub loan: Option<(EpReserveLoan, EpLoanTeardownPermit)>,
}

impl EpReserveForward {
    /// The donor tensor `(layer, slot)` of this rank's weights.
    fn donor_tensor<'a>(
        bundle: &'a Qwen4Bundle,
        layer: usize,
        slot: EpDonorSlot,
    ) -> Result<&'a GpuTensor, Qwen4GpuForwardError> {
        let weights = bundle
            .weights
            .layer(layer)
            .ok_or_else(|| invalid(format!("EP reserve donor layer {layer} is not owned")))?
            .ffn
            .moe()
            .ok_or_else(|| invalid(format!("EP reserve donor layer {layer} has no MoE FFN")))?;
        let source = match slot {
            EpDonorSlot::GateUp => &weights.experts_gate_up,
            EpDonorSlot::Down => &weights.experts_down,
        };
        Ok(bundle.weights.resident(source)?)
    }

    /// Segments and placements of `plan` over this rank's donor tensors.
    pub(crate) fn layout(
        bundle: &Qwen4Bundle,
        plan: &EpReservePlan,
    ) -> Result<(Vec<EpDonorSegment>, Vec<EpRootPlacement>), Qwen4GpuForwardError> {
        ep_reserve_layout(plan, |layer, slot| {
            let owner = Self::donor_tensor(bundle, layer, slot)?;
            Ok((owner.buf.as_ptr() as usize, owner.buf.size() as u64))
        })
    }

    /// Register the weight shadows of the plan's `shadow_placements` as
    /// non-owning views into the donor segments and lower the resident bank's
    /// call limit below the F16 route's row threshold, so no call that reads a
    /// shadow can run outside a borrowed epoch.
    fn install_shadows(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
    ) -> Result<(), Qwen4GpuForwardError> {
        use rdna_compute::shadow_borrow::{F16ShadowSpan, ShadowBorrowPlan, SharedI8ShadowSpan};
        let plan = self.plan.clone();
        let (f16, shared_i8) = crate::decode_share_shadows::shadow_flags(
            &gpu.arch,
            gpu.qwen4_f16_shadow_capable(),
            gpu.qwen4_shared_i8_enabled(),
            plan.forward.prompt_rows,
        )
        .map_err(|reason| invalid(format!("EP reserve shadows: {reason}")))?;
        let region = crate::decode_share_shadows::plan_shadow_region(
            &bundle.weights.manifest.weights,
            bundle.config.num_hidden_layers,
            f16,
            shared_i8,
        )
        .map_err(invalid)?;
        let spans = region.spans();
        if spans.len() != plan.shadow_placements.len()
            || spans
                .iter()
                .zip(&plan.shadow_placements)
                .any(|(span, placement)| span.2 != placement.bytes)
        {
            return Err(invalid(
                "EP reserve shadows: the plan's shadow members are not the manifest's shadow region",
            ));
        }
        let base_of = |layer: usize, slot: EpDonorSlot| -> Result<usize, Qwen4GpuForwardError> {
            self.loan
                .mirror(layer, slot)
                .map(|mirror| mirror.donor_device_base)
                .ok_or_else(|| invalid(format!("EP reserve shadows: no donor {layer} {slot:?}")))
        };
        let at = |member: usize| -> Result<usize, Qwen4GpuForwardError> {
            let placement = &plan.shadow_placements[member];
            usize::try_from(placement.offset)
                .ok()
                .and_then(|offset| {
                    base_of(placement.layer, placement.slot)
                        .ok()?
                        .checked_add(offset)
                })
                .ok_or_else(|| invalid("EP reserve shadow address overflows"))
        };
        let resident = |name: &str, layer: Option<usize>| {
            bundle
                .weights
                .resident_by_name(name, layer)
                .map_err(|error| invalid(format!("EP reserve shadow source '{name}': {error}")))
        };
        let mut f16_spans = Vec::with_capacity(region.f16.len());
        for (index, member) in region.f16.iter().enumerate() {
            let tensor = resident(&member.name, member.layer)?;
            f16_spans.push(F16ShadowSpan {
                name: member.name.clone(),
                src_ptr: tensor.buf.as_ptr() as usize,
                dst_ptr: at(index)?,
                rows: member.rows,
                cols: member.cols,
                hc8: tensor.dtype == rdna_compute::DType::HC8E4M3Row,
            });
        }
        let mut i8_spans = Vec::with_capacity(region.shared_i8.len());
        for (j, member) in region.shared_i8.iter().enumerate() {
            let ptr = |proj: &str| -> Result<usize, Qwen4GpuForwardError> {
                let name = format!(
                    "model.language_model.layers.{}.mlp.shared_expert.{proj}.weight",
                    member.layer
                );
                Ok(resident(&name, Some(member.layer))?.buf.as_ptr() as usize)
            };
            i8_spans.push(SharedI8ShadowSpan {
                layer: member.layer,
                gate_ptr: ptr("gate_proj")?,
                up_ptr: ptr("up_proj")?,
                down_ptr: ptr("down_proj")?,
                intermediate: member.intermediate,
                hidden: member.hidden,
                down_rows: member.down_rows,
                gate_up_dst: at(region.f16.len() + 2 * j)?,
                down_dst: at(region.f16.len() + 2 * j + 1)?,
            });
        }
        let mut extra_regions = Vec::with_capacity(plan.segments.len());
        for segment in &plan.segments {
            extra_regions.push((
                base_of(segment.layer, segment.slot)?,
                usize::try_from(segment.capacity)
                    .map_err(|_| invalid("EP reserve segment bytes overflow"))?,
            ));
        }
        gpu.install_shadow_borrow(ShadowBorrowPlan {
            region_ptr: 0,
            region_bytes: 0,
            extra_regions,
            f16: f16_spans,
            shared_i8: i8_spans,
        })?;
        self.banks
            .set_resident_call_limit(rdna_compute::gemm::QWEN4_F16_WMMA_MIN_TOKENS - 1);
        self.shadows = true;
        eprintln!(
            "  qwen4 EP reserve: {} F16 + {} INT8 weight shadows live in the donor segments \
             (rebuilt per borrow epoch); the resident bank serves calls of at most {} rows",
            region.f16.len(),
            region.shared_i8.len(),
            self.banks.resident_call_limit()
        );
        Ok(())
    }

    /// Mirror every scratch-donor tensor, bind the touched extents and the S2
    /// host sources. On failure the banks are released (nothing else exists
    /// yet) and the original error returned.
    pub(crate) fn materialize(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        plan: Arc<EpReservePlan>,
        banks: EpReserveBanks,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let built = Self::materialize_inner(gpu, bundle, &plan, &banks);
        match built {
            Ok((source_generation, loan, host_sources, exchange_root, receipts)) => {
                let mut forward = Self {
                    plan,
                    source_generation,
                    banks,
                    loan,
                    host_sources,
                    exchange_root,
                    epoch: EpEpoch::default(),
                    poisoned: None,
                    receipts,
                    hashes: receipts
                        && hipfire_config::developer_bool(EP_RESERVE_RECEIPT_HASH_ENV, true),
                    shadows: false,
                    prompt: None,
                    prompt_seq: 0,
                    ordinals: EpReceiptOrdinals::default(),
                };
                if !forward.plan.shadow_placements.is_empty() {
                    if let Err(error) = forward.install_shadows(gpu, bundle) {
                        release_unused_loan(gpu, forward.loan);
                        let _ = forward.banks.release(gpu);
                        return Err(error);
                    }
                }
                Ok(forward)
            }
            Err(error) => {
                let _ = banks.release(gpu);
                Err(error)
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn materialize_inner(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        plan: &Arc<EpReservePlan>,
        banks: &EpReserveBanks,
    ) -> Result<(u64, EpReserveLoan, EpStageHostSources, GpuTensor, bool), Qwen4GpuForwardError>
    {
        check_stage_plan(plan).map_err(invalid)?;
        if banks.prompt_rows() != plan.forward.prompt_rows
            || banks.resident_rows() != plan.forward.resident_rows
        {
            return Err(invalid(
                "EP reserve banks disagree with the plan's row counts",
            ));
        }
        let seal = bundle.ep_seal().ok_or_else(|| {
            invalid(
                "an EP reserve load needs a sealed EP rank (bind_ep_seal before the forward \
                 attaches): its mesh receipt names the sealed source generation",
            )
        })?;
        let source_generation = generation_fingerprint(seal.seal.source_generation());
        let mut donors = Vec::with_capacity(plan.donor_layers.len() * 2);
        for &layer in &plan.donor_layers {
            for slot in [EpDonorSlot::GateUp, EpDonorSlot::Down] {
                donors.push(EpLoanDonor {
                    layer,
                    slot,
                    owner: Self::donor_tensor(bundle, layer, slot)?,
                });
            }
        }
        let receipts = hipfire_config::developer_bool(EP_RESERVE_RECEIPT_ENV, false);
        let loan = EpReserveLoan::materialize(gpu, &donors, &plan.touched, false, receipts)?;
        let sources = match EpStageHostSources::new(gpu, &loan, plan) {
            Ok(sources) => sources,
            Err(error) => {
                release_unused_loan(gpu, loan);
                return Err(error);
            }
        };
        let (address, bytes) = banks.registered_grouped_down();
        let slot = &banks.bank_slots(EpBankKind::Borrowed)[ScratchField::MoeYDownGrouped as usize];
        let exchange_root = GpuTensor {
            // SAFETY: a `Borrowed` alias of the persistent grouped-down root
            // `EpReserveBanks` owns until its `release`; the view is dropped
            // with the forward and never reaches a free path.
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(address as *mut std::ffi::c_void, bytes)
            },
            shape: vec![slot.elements],
            dtype: slot.dtype,
        };
        Ok((source_generation, loan, sources, exchange_root, receipts))
    }

    /// The loan's complete mirror of `(layer, slot)`.
    pub(crate) fn mirror(&self, layer: usize, slot: EpDonorSlot) -> Option<EpLoanMirror<'_>> {
        self.loan.mirror(layer, slot)
    }

    /// Mark the load failed; the first reason wins.
    pub(super) fn poison(&mut self, reason: impl Into<String>) {
        self.poisoned.get_or_insert_with(|| reason.into());
    }

    /// Refuse a forward on a failed or poisoned load.
    pub(super) fn check(&self) -> Result<(), Qwen4GpuForwardError> {
        if let Some(reason) = &self.poisoned {
            return Err(invalid(format!(
                "the EP reserve-sharing load failed earlier ({reason}); drain and reload: a \
                 locally successful donor restore does not make the mesh serviceable"
            )));
        }
        self.loan.check().map_err(invalid)
    }

    /// Restore every donor (synchronously) and hand back the owners in the
    /// order the forward frees them. A restore that cannot be proven complete
    /// leaks the loan (never frees it under a possibly pending copy).
    pub(crate) fn teardown(mut self, gpu: &mut Gpu) -> EpReserveTeardown {
        let torn = self.loan.teardown(gpu);
        if self.shadows {
            gpu.clear_shadow_borrow();
        }
        eprintln!(
            "  qwen4 EP reserve: {} borrowed epochs; {} bank swaps in, {} out; loan {}",
            self.epoch.epochs(),
            self.banks.stats().swaps_in,
            self.banks.stats().swaps_out,
            self.loan.summary()
        );
        let loan = match torn {
            Ok(permit) => Some((self.loan, permit)),
            Err(error) => {
                eprintln!(
                    "  qwen4 EP reserve: teardown restore not proven ({error}); leaking the donor \
                     loan rather than freeing under a possibly pending copy"
                );
                std::mem::forget(self.loan);
                None
            }
        };
        EpReserveTeardown {
            banks: self.banks,
            loan,
        }
    }

    /// One `rank` receipt record (and its header) of the admitted load.
    pub(crate) fn rank_receipt_json(
        &self,
        rank: usize,
        s1: bool,
        s2: bool,
    ) -> [serde_json::Value; 2] {
        let receipt = self.plan.mesh_receipt(self.source_generation, s1, s2);
        let (address, bytes) = self.banks.registered_grouped_down();
        [
            serde_json::json!({
                "kind": "header",
                "decouple": true,
                "prompt_chunk_rows": receipt.prompt_rows,
                "resident_rows": self.plan.forward.resident_rows,
            }),
            serde_json::json!({
                "kind": "rank",
                "rank": rank,
                "source_generation": receipt.source_generation.to_string(),
                "plan_digest": format!("{:016x}", receipt.plan_digest),
                "resident_layers": receipt.resident_layers,
                "donor_first_layer": receipt.donor_first_layer,
                "donor_count": receipt.donor_count,
                "s2_donor_layers": receipt.s2_donor_layers,
                "prompt_rows": receipt.prompt_rows,
                "exchange_rows": receipt.exchange_rows,
                "s1_admitted": receipt.s1_admitted,
                "s2_admitted": receipt.s2_admitted,
                "exchange_address": format!("{address:#x}"),
                "exchange_bytes": bytes,
                "mirror_bytes": self.plan.mirror_bytes_per_rank,
                "restore_bytes": self.plan.restore_bytes_per_rank,
            }),
        ]
    }
}

/// Free a loan that never served a forward (a construction failure): nothing
/// was ever dirty, so its teardown is a no-op that yields the permit.
fn release_unused_loan(gpu: &mut Gpu, mut loan: EpReserveLoan) {
    match loan.teardown(gpu) {
        Ok(permit) => {
            let _ = loan.release(gpu, permit);
        }
        Err(_) => std::mem::forget(loan),
    }
}

/// FNV-1a 64 over every donor tensor's device bytes (a gate diagnostic: a full
/// read-back of the scratch-donor layers).
fn donor_hash(gpu: &mut Gpu, loan: &EpReserveLoan) -> Result<String, Qwen4GpuForwardError> {
    const CHUNK: usize = 32 << 20;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut host = vec![0u8; CHUNK];
    for mirror in loan.mirrors() {
        let mut done = 0usize;
        while done < mirror.bytes {
            let n = (mirror.bytes - done).min(CHUNK);
            // SAFETY: inside the donor tensor the weights own for this
            // forward's life; the buffer is a `Borrowed` alias.
            let src = unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    (mirror.donor_device_base + done) as *mut std::ffi::c_void,
                    n,
                )
            };
            gpu.memcpy_dtoh_auto(&mut host[..n], &src)?;
            hash = fnv_fold(hash, &host[..n]);
            done += n;
        }
    }
    Ok(format!("{hash:016x}"))
}

// ---------------------------------------------------------------------------
// Forward integration
// ---------------------------------------------------------------------------

impl Qwen4GpuForward {
    /// This rank's reserve receipt for the mesh to compare at attach; `None` on
    /// an ordinary load. The S1 bit is the owner-routed route's admission, the
    /// S2 bit the borrowed-view stage's.
    pub(crate) fn ep_reserve_plan_receipt(&self) -> Option<EpReserveMeshReceipt> {
        let reserve = self.ep_reserve.as_ref()?;
        let s1 = self.ep.as_ref().is_some_and(|ep| ep.tile.is_some());
        let s2 = self.expert_stage.is_some();
        Some(reserve.plan.mesh_receipt(reserve.source_generation, s1, s2))
    }

    /// Rows of the resident scratch bank of an EP reserve-sharing load; `None`
    /// on an ordinary load. Lane decode steps run on it.
    pub(crate) fn ep_reserve_resident_rows(&self) -> Option<usize> {
        self.ep_reserve
            .as_ref()
            .map(|reserve| reserve.banks.resident_rows())
    }

    /// Whether a borrowed-bank epoch is open (donors dirty); `false` on an
    /// ordinary load.
    pub(crate) fn ep_reserve_epoch_open(&self) -> bool {
        self.ep_reserve
            .as_ref()
            .is_some_and(|reserve| reserve.epoch.dirty())
    }

    /// Entry of a lane decode step (`forward_lanes`, `rows` decode rows) on an
    /// EP reserve-sharing load; a no-op on an ordinary load. A lane step reads
    /// the Full peer tables and the donor layers' experts, so a dirty borrow
    /// epoch (a lane prefill tile on the borrowed bank ran since the last
    /// restore) is globally restored first, exactly as a resident-bank call of
    /// [`Self::ep_reserve_forward_chunk`] restores; a clean epoch passes with no
    /// barrier. The step then runs on the resident bank, which is checked to be
    /// the one in `scratch` (never swapped here).
    pub(crate) fn ep_reserve_lane_step_enter(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let Some(reserve) = self.ep_reserve.as_ref() else {
            return Ok(());
        };
        reserve.check()?;
        if rows == 0 || rows > reserve.banks.resident_rows() {
            return Err(invalid(format!(
                "a {rows}-row lane step does not fit the {}-row resident bank of an EP reserve load",
                reserve.banks.resident_rows()
            )));
        }
        self.ep_reserve_require_resident(gpu, "a lane step")?;
        let reserve = self.ep_reserve.as_mut().expect("checked above");
        if let Err(error) = reserve
            .banks
            .select(&mut self.scratch, rows, EpBankRoute::TileExchange)
        {
            reserve.poison(error.to_string());
            return Err(error);
        }
        Ok(())
    }

    /// Print this rank's header and rank receipt records (once, when the mesh
    /// is serviceable). A reserve load prints the plan's receipt; an ordinary
    /// EP load under `HIPFIRE_EP_RESERVE_RECEIPT=1` prints the real loaded
    /// residency (resident expert layers of this rank's runtimes, the prompt
    /// chunk the scratch was built for). Host-only: no device work.
    pub(crate) fn ep_receipt_log_rank(&self) {
        let Some(ep) = self.ep.as_ref() else {
            return;
        };
        if let Some(reserve) = self.ep_reserve.as_ref() {
            let s1 = ep.tile.is_some();
            for record in
                reserve.rank_receipt_json(ep.placement.rank, s1, self.expert_stage.is_some())
            {
                emit_receipt(&record);
            }
            return;
        }
        if self.ep_ordinary_receipt.is_none() {
            return;
        }
        let resident_layers = self
            .moe
            .iter()
            .flatten()
            .filter(|layer| !layer.host_mapped)
            .count();
        emit_receipt(&serde_json::json!({
            "kind": "header",
            "decouple": false,
            "prompt_chunk_rows": self.prompt_chunk_rows(),
        }));
        emit_receipt(&serde_json::json!({
            "kind": "rank",
            "rank": ep.placement.rank,
            "resident_layers": resident_layers,
        }));
    }

    /// The borrowed epochs this rank has opened (`0` without an EP reserve): the
    /// epoch every rendezvous tag carries, so ranks that disagree on it fail at
    /// the rendezvous naming both.
    pub(super) fn ep_reserve_epoch(&self) -> u32 {
        self.ep_reserve
            .as_ref()
            .map_or(0, |reserve| reserve.epoch.epochs() as u32)
    }

    /// All-rank host rendezvous of kind `what` (`HOST_RESERVE_ACQUIRE` /
    /// `HOST_RESERVE_RESTORE`): the installed tile hook's barrier (abortable,
    /// owned by the mesh), tagged with the kind, this rank's epoch count and its
    /// rendezvous count so a rank-local divergence is named instead of timing out.
    fn ep_reserve_barrier(&self, what: u8) -> Result<(), Qwen4GpuForwardError> {
        let hook = self
            .ep
            .as_ref()
            .and_then(|ep| ep.tile.as_ref())
            .and_then(|route| route.hook.as_ref())
            .ok_or_else(|| {
                invalid("the EP reserve rendezvous needs the installed tile exchange")
            })?;
        hook.host_barrier(what, self.ep_reserve_epoch())
            .map_err(|error| invalid(format!("EP reserve barrier: {error}")))
    }

    fn ep_reserve_capturing(gpu: &Gpu) -> Result<bool, Qwen4GpuForwardError> {
        Ok(match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.stream_is_capturing(stream)?,
            None => false,
        })
    }

    /// Before a call that reads donor layers or runs on the resident bank:
    /// donors globally whole. A proven clean epoch is a checked no-op (no
    /// per-token barrier). Public to the bundle (`Qwen4Bundle::
    /// ep_reserve_require_resident`): reset, state restore, snapshot.
    pub(crate) fn ep_reserve_require_resident(
        &mut self,
        gpu: &mut Gpu,
        reason: &str,
    ) -> Result<(), Qwen4GpuForwardError> {
        let Some(reserve) = self.ep_reserve.as_ref() else {
            return Ok(());
        };
        reserve.check()?;
        if !reserve.epoch.dirty() {
            return Ok(());
        }
        self.ep_reserve_restore_global(gpu, reason)
    }

    /// The global restore: drain S2, restore the dirty extents and host-wait,
    /// then cross the all-rank barrier; only then is this rank's epoch clean.
    fn ep_reserve_restore_global(
        &mut self,
        gpu: &mut Gpu,
        reason: &str,
    ) -> Result<(), Qwen4GpuForwardError> {
        let started = Instant::now();
        let outcome = (|| {
            let tail = Instant::now();
            if let Some(stage) = self.expert_stage.as_ref() {
                stage.ep_quiesce(gpu)?;
            }
            let s2_dma_tail_ms = ms(tail.elapsed());
            let reserve = self.ep_reserve.as_mut().expect("checked by the caller");
            let stats = reserve.loan.stats();
            let before = stats.async_restore_bytes + stats.blocking_restore_bytes;
            reserve.loan.require_resident(gpu).map_err(invalid)?;
            let stats = reserve.loan.stats();
            let bytes = stats.async_restore_bytes + stats.blocking_restore_bytes - before;
            let restore_ms = ms(started.elapsed());
            let wait = Instant::now();
            self.ep_reserve_barrier(HOST_RESERVE_RESTORE)?;
            Ok::<_, Qwen4GpuForwardError>((bytes, restore_ms, ms(wait.elapsed()), s2_dma_tail_ms))
        })();
        let reserve = self.ep_reserve.as_mut().expect("checked by the caller");
        match outcome {
            Ok((bytes, restore_ms, barrier_wait_ms, s2_dma_tail_ms)) => {
                reserve.epoch.restored();
                gpu.shadow_borrow_set_ready(false);
                if let Some(prompt) = reserve.prompt.as_mut() {
                    prompt.restores.push(EpRestoreEvent {
                        before_chunk: prompt.chunk_rows.len(),
                        bytes,
                        restore_ms,
                        barrier_wait_ms,
                        s2_dma_tail_ms,
                    });
                }
                Ok(())
            }
            Err(error) => {
                reserve.poison(format!("restore before {reason}: {error}"));
                Err(error)
            }
        }
    }

    /// First borrowed call of an epoch (`barrier`): complete this rank's
    /// previous work and cross the all-rank barrier BEFORE any rank overwrites
    /// donor bytes, then mark the loan dirty and zero the touched extents. Every
    /// borrowed call re-enters the loan (capture refusal, counters).
    fn ep_reserve_acquire(
        &mut self,
        gpu: &mut Gpu,
        barrier: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        if Self::ep_reserve_capturing(gpu)? {
            return Err(invalid(
                "an EP reserve borrowed tile cannot run inside a stream capture (its scratch \
                 overwrites donor experts that need a restore)",
            ));
        }
        if barrier {
            gpu.hip.device_synchronize()?;
            self.ep_reserve_barrier(HOST_RESERVE_ACQUIRE)?;
        }
        let reserve = self.ep_reserve.as_mut().expect("checked by the caller");
        reserve.loan.acquire_all(gpu).map_err(invalid)?;
        if barrier {
            reserve.epoch.acquired();
            if reserve.shadows {
                gpu.rebuild_borrowed_shadows()?;
            }
        }
        Ok(())
    }

    /// `forward_chunk_inner` on an EP reserve load: the bank switch.
    ///
    /// The sealed route of the call is taken first (under the call's own
    /// routing scopes, before any destructive write): at most 512 rows run on
    /// the resident bank after a global restore of any dirty epoch; more rows
    /// run on the one fixed borrowed bank, only on the owner-routed tile
    /// exchange. The selected bank and the borrowed runtime view are swapped
    /// back on every return path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ep_reserve_forward_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        argmax_of: Option<&GpuTensor>,
        logits: Option<&GpuTensor>,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
        output_policy: Qwen4OutputPolicy,
    ) -> Result<(), Qwen4GpuForwardError> {
        let n = tokens.len();
        let resident_rows = {
            let reserve = self.ep_reserve.as_ref().expect("an EP reserve forward");
            reserve.check()?;
            reserve.banks.resident_call_limit()
        };
        let route = if n > resident_rows {
            // The same decision `forward_chunk_scoped` takes, under the same scopes.
            let (scopes, _) = self.enter_tile_scopes(gpu, n);
            let route_n = gpu.route_rows(n);
            let tile = ep_tile_active(self.ep.as_ref(), gpu, &bundle.config, n, route_n);
            let partial = ep_trunk_partial_chunk(
                self.ep.is_some(),
                n,
                route_n,
                crate::trunk_route::ffn_partial_active(),
            );
            Self::leave_tile_scopes(gpu, scopes);
            ep_wide_bank_route(tile, partial)
        } else {
            EpBankRoute::TileExchange
        };
        // A lane PREFILL tile is the singleton chunk program on a lane state
        // (`with_lane`, `lane_scope`), so it borrows exactly as a singleton
        // prompt tile does; only a device-argmax decode or a mesh segment is
        // refused. Lane DECODE steps never reach here (`forward_lanes` enters
        // through [`Self::ep_reserve_lane_step_enter`]).
        let no_borrow = argmax_of.is_some() || self.mesh_route_rows.is_some();
        self.ep_reserve_call(gpu, n, route, no_borrow, |this, gpu| {
            this.forward_chunk_tile(
                bundle,
                gpu,
                tokens,
                argmax_of,
                logits,
                top1,
                wide_hidden_capture,
                output_policy,
            )
        })
    }

    /// The EP reserve bank switch around one forward call of `n` rows whose sealed
    /// route is `route` (the body [`Self::ep_reserve_forward_chunk`] runs for a
    /// singleton chunk, and `forward_wave` for a prefill wave): plan the call
    /// against the epoch (a wide Full-table call is refused before anything is
    /// written), restore globally first when the plan says so, acquire the borrowed
    /// epoch, select and swap the bank and the borrowed runtime view, run `run`,
    /// and swap everything back on every return path. `no_borrow`: the call may
    /// not run on the borrowed bank (a device-argmax or mesh forward).
    pub(crate) fn ep_reserve_call(
        &mut self,
        gpu: &mut Gpu,
        n: usize,
        route: EpBankRoute,
        no_borrow: bool,
        run: impl FnOnce(&mut Self, &mut Gpu) -> Result<(), Qwen4GpuForwardError>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let (resident_rows, prompt_rows) = {
            let reserve = self.ep_reserve.as_ref().expect("an EP reserve forward");
            reserve.check()?;
            (
                reserve.banks.resident_call_limit(),
                reserve.banks.prompt_rows(),
            )
        };
        let plan = self
            .ep_reserve
            .as_ref()
            .expect("an EP reserve forward")
            .epoch
            .plan_call(n, resident_rows, prompt_rows, route)
            .map_err(invalid)?;
        if plan.bank == EpBankKind::Borrowed && no_borrow {
            return Err(invalid(
                "an argmax or mesh forward cannot run on the borrowed EP reserve bank",
            ));
        }
        self.ep_reserve_hash_before(gpu, plan.bank == EpBankKind::Borrowed)?;
        if plan.restore_first {
            self.ep_reserve_restore_global(gpu, "a resident-bank call")?;
        }
        self.ep_reserve_note_call(n, plan.bank);
        if plan.bank == EpBankKind::Resident {
            // The resident bank must be the one in `scratch` (checked, never swapped here).
            let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
            if let Err(error) = reserve.banks.select(&mut self.scratch, n, route) {
                reserve.poison(error.to_string());
                return Err(error);
            }
            let result = run(self, gpu);
            self.ep_reserve_note_result(&result);
            return result;
        }
        if let Err(error) = self.ep_reserve_acquire(gpu, plan.acquire) {
            self.ep_reserve_poison(&error);
            return Err(error);
        }
        let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
        if let Err(error) = reserve.banks.select(&mut self.scratch, n, route) {
            // The loan is dirty: recover before reporting.
            let recovery = reserve.loan.after_tile(gpu, false).map_err(invalid);
            reserve.poison(error.to_string());
            return merge_share_result(Err(error), recovery);
        }
        self.borrow_runtimes_active = true;
        self.borrow_view = 0;
        let result = run(self, gpu);
        self.borrow_runtimes_active = false;
        let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
        let swapped = reserve.banks.swap_out(&mut self.scratch).map(|_| ());
        let after = reserve
            .loan
            .after_tile(gpu, result.is_ok())
            .map_err(invalid);
        let merged = merge_share_result(merge_share_result(result, swapped), after);
        self.ep_reserve_note_result(&merged);
        merged
    }

    fn ep_reserve_note_call(&mut self, rows: usize, bank: EpBankKind) {
        if let Some(prompt) = self
            .ep_reserve
            .as_mut()
            .and_then(|reserve| reserve.prompt.as_mut())
        {
            prompt.chunk_rows.push(rows);
            prompt.banks.push(bank);
        }
    }

    fn ep_reserve_note_result(&mut self, result: &Result<(), Qwen4GpuForwardError>) {
        if let (Err(error), Some(reserve)) = (result, self.ep_reserve.as_mut()) {
            reserve.poison(error.to_string());
        }
    }

    fn ep_reserve_poison(&mut self, error: &Qwen4GpuForwardError) {
        if let Some(reserve) = self.ep_reserve.as_mut() {
            reserve.poison(error.to_string());
        }
    }

    /// First forward of a recorded prompt: the donor hash before the prompt
    /// (clean epoch only; excluded from the prompt time). Only an entry that
    /// opens on the borrowed bank is hashed: a resident-bank entry (decode,
    /// verify windows, short prompts) never dirties the donors, and a full
    /// read-back per decode window would dominate a gate run.
    fn ep_reserve_hash_before(
        &mut self,
        gpu: &mut Gpu,
        borrowed: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
        let wanted = reserve.hashes
            && borrowed
            && !reserve.epoch.dirty()
            && reserve
                .prompt
                .as_ref()
                .is_some_and(|prompt| prompt.hash_before.is_none() && prompt.chunk_rows.is_empty());
        if !wanted {
            return Ok(());
        }
        let started = Instant::now();
        let hash = donor_hash(gpu, &reserve.loan)?;
        if let Some(prompt) = reserve.prompt.as_mut() {
            prompt.hash_before = Some(hash);
            prompt.excluded += started.elapsed();
        }
        Ok(())
    }

    /// A whole-prompt entry begins on this rank (no-op without an EP reserve).
    pub(super) fn ep_reserve_prompt_begin(&mut self) {
        let mtp_fill = self.receipt_mtp_fill;
        let address = self
            .ep_reserve
            .as_ref()
            .map(|reserve| reserve.banks.registered_grouped_down().0);
        let (Some(reserve), Some(address)) = (self.ep_reserve.as_mut(), address) else {
            self.ep_ordinary_prompt_begin(mtp_fill);
            return;
        };
        if reserve.epoch.begin_prompt() {
            let seq = reserve.prompt_seq;
            reserve.prompt_seq += 1;
            let (prompt, phase, ordinal) = if reserve.receipts {
                reserve.ordinals.next_key(seq, mtp_fill)
            } else {
                (seq, EpReceiptPhase::Target, 0)
            };
            reserve.prompt = Some(EpPromptRecord {
                prompt,
                phase,
                ordinal,
                chunk_rows: Vec::new(),
                banks: Vec::new(),
                restores: Vec::new(),
                epochs_before: reserve.epoch.epochs(),
                started: Instant::now(),
                excluded: Duration::ZERO,
                exchange_before: Some(address),
                hash_before: None,
            });
        }
    }

    /// An ordinary EP load's outermost entry begins (receipts only).
    fn ep_ordinary_prompt_begin(&mut self, mtp_fill: bool) {
        let address = self.ep_tile_exchange_address();
        let Some(state) = self.ep_ordinary_receipt.as_mut() else {
            return;
        };
        state.depth += 1;
        if state.depth != 1 {
            return;
        }
        let seq = state.seq;
        state.seq += 1;
        let (prompt, phase, ordinal) = state.ordinals.next_key(seq, mtp_fill);
        state.prompt = Some(EpPromptRecord {
            prompt,
            phase,
            ordinal,
            chunk_rows: Vec::new(),
            banks: Vec::new(),
            restores: Vec::new(),
            epochs_before: 0,
            started: Instant::now(),
            excluded: Duration::ZERO,
            exchange_before: address,
            hash_before: None,
        });
    }

    /// The address the owner-routed exchange is registered at on an ordinary
    /// load (the rank's grouped-down root), `None` without the tile route.
    fn ep_tile_exchange_address(&self) -> Option<usize> {
        self.ep.as_ref()?.tile.as_ref()?;
        Some(self.scratch.moe_y_down_grouped.buf.as_ptr() as usize)
    }

    /// A forward call of `rows` rows inside an ordinary load's open entry.
    pub(super) fn ep_ordinary_note_call(&mut self, rows: usize) {
        if let Some(prompt) = self
            .ep_ordinary_receipt
            .as_mut()
            .and_then(|state| state.prompt.as_mut())
        {
            prompt.chunk_rows.push(rows);
            prompt.banks.push(EpBankKind::Resident);
        }
    }

    /// An ordinary EP load's entry ends (`ok`: it returned `Ok`): the
    /// outermost one prints the prompt record. Host-only; no sync, so the
    /// elapsed time is the host wall clock of the entry.
    pub(super) fn ep_ordinary_prompt_end(&mut self, ok: bool) {
        let rank = self.ep.as_ref().map(|ep| ep.placement.rank);
        let address_after = self.ep_tile_exchange_address();
        let Some(state) = self.ep_ordinary_receipt.as_mut() else {
            return;
        };
        state.depth = state.depth.saturating_sub(1);
        if state.depth > 0 {
            return;
        }
        let Some(prompt) = state.prompt.take() else {
            return;
        };
        let Some(rank) = rank else {
            return;
        };
        if !ok || prompt.chunk_rows.is_empty() {
            return;
        }
        emit_receipt(&prompt_rank_json(
            rank,
            &prompt,
            0,
            prompt.started.elapsed(),
            address_after,
            None,
        ));
    }

    /// A whole-prompt entry ends (`ok`: it returned `Ok`). The outermost one
    /// restores a dirty epoch synchronously, before the prompt returns and
    /// inside the prompt time, unless a small tail already restored.
    pub(super) fn ep_reserve_prompt_end(
        &mut self,
        gpu: &mut Gpu,
        ok: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        let end = match self.ep_reserve.as_mut() {
            Some(reserve) => reserve.epoch.end_prompt(ok),
            None => return Ok(()),
        };
        let result = match end {
            EpPromptEnd::Nested | EpPromptEnd::Clean => Ok(()),
            EpPromptEnd::Restore => self.ep_reserve_restore_global(gpu, "prompt completion"),
            EpPromptEnd::Recover => {
                let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
                // Local recovery only: the mesh aborts the other ranks' waits
                // and the load is poisoned either way.
                let recovered = reserve.loan.finish_prompt(gpu, false).map_err(invalid);
                reserve.epoch.restored();
                gpu.shadow_borrow_set_ready(false);
                reserve.poison("a prompt ended with an error while a borrowed epoch was open");
                recovered
            }
        };
        if matches!(end, EpPromptEnd::Nested) {
            return result;
        }
        self.ep_reserve_finish_record(gpu, result)
    }

    /// Print the outermost prompt's record (when enabled) and drop it.
    fn ep_reserve_finish_record(
        &mut self,
        gpu: &mut Gpu,
        result: Result<(), Qwen4GpuForwardError>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let rank = self.ep.as_ref().map_or(0, |ep| ep.placement.rank);
        let reserve = self.ep_reserve.as_mut().expect("an EP reserve forward");
        let Some(prompt) = reserve.prompt.take() else {
            return result;
        };
        if !reserve.receipts || result.is_err() || prompt.chunk_rows.is_empty() {
            return result;
        }
        let elapsed = prompt.started.elapsed().saturating_sub(prompt.excluded);
        let hash_after = if prompt.hash_before.is_none() {
            None
        } else {
            match donor_hash(gpu, &reserve.loan) {
                Ok(hash) => Some(hash),
                Err(error) => {
                    reserve.poison(error.to_string());
                    return Err(error);
                }
            }
        };
        let epochs = reserve.epoch.epochs() - prompt.epochs_before;
        let exchange_after = Some(reserve.banks.registered_grouped_down().0);
        emit_receipt(&prompt_rank_json(
            rank,
            &prompt,
            epochs,
            elapsed,
            exchange_after,
            hash_after,
        ));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::ep_reserve_scratch::EpScratchDevice;
    use super::*;
    use crate::config::compact_test_config;
    use crate::ep_reserve::{
        plan_ep_reserve, EpRankGeometry, EpReserveCapacity, EpReserveForwardPlan,
        EP_RESERVE_MIN_RESIDENT_LAYERS,
    };
    use std::collections::BTreeMap;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;
    const RESIDENT: usize = 512;
    const PROMPT: usize = 4096;
    type Bank = EpBankKind;
    const TILE: EpBankRoute = EpBankRoute::TileExchange;
    const FULL: EpBankRoute = EpBankRoute::FullTable;

    fn plan() -> EpReservePlan {
        let config = compact_test_config();
        let forward = EpReserveForwardPlan::from_config(&config, Qwen4ForwardScratchMode::Full, 0)
            .expect("pinned forward plan");
        let geometry = EpRankGeometry::flash_next(&config, 2).expect("geometry");
        let layer = geometry.layer_bytes().unwrap();
        let (non_expert, reserve) = (6 * GIB, 6 * GIB);
        let free = non_expert + reserve + layer * 33;
        plan_ep_reserve(
            &forward,
            &geometry,
            &EpReserveCapacity {
                free_per_rank: vec![free, free],
                non_expert_bytes: non_expert,
                reserve_bytes: reserve,
                lane_bytes: 0,
                host_budget_bytes: 64 * GIB,
                min_resident_layers: EP_RESERVE_MIN_RESIDENT_LAYERS,
            },
        )
        .expect("a 33-layer plan")
    }

    /// What the forward does around one call, mirroring `ep_reserve_forward_chunk`
    /// and the prompt bracket over the pure state machine.
    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Acquire,
        Restore { before: usize },
        Call { rows: usize, bank: Bank },
        Completion,
    }

    /// Run `prompt` (the forward row sequence of ONE outer prompt) with the given
    /// route of every call; `routes` defaults to the tile exchange.
    fn drive(
        prompt: &[usize],
        route_of: impl Fn(usize) -> EpBankRoute,
    ) -> Result<Vec<Event>, String> {
        let mut epoch = EpEpoch::default();
        let mut events = Vec::new();
        epoch.begin_prompt();
        for (index, &rows) in prompt.iter().enumerate() {
            let call = epoch.plan_call(rows, RESIDENT, PROMPT, route_of(rows))?;
            if call.restore_first {
                events.push(Event::Restore { before: index });
                epoch.restored();
            }
            if call.acquire {
                events.push(Event::Acquire);
                epoch.acquired();
            }
            events.push(Event::Call {
                rows,
                bank: call.bank,
            });
        }
        if epoch.end_prompt(true) == EpPromptEnd::Restore {
            events.push(Event::Completion);
            epoch.restored();
        }
        assert!(!epoch.dirty(), "{prompt:?} left the epoch dirty");
        assert_eq!(epoch.depth(), 0);
        Ok(events)
    }

    fn banks_of(events: &[Event]) -> Vec<Bank> {
        events
            .iter()
            .filter_map(|event| match event {
                Event::Call { bank, .. } => Some(*bank),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn route_matrix_picks_the_bank_from_the_actual_rows() {
        for (rows, bank) in [
            (1, Bank::Resident),
            (8, Bank::Resident),
            (9, Bank::Resident),
            (511, Bank::Resident),
            (512, Bank::Resident),
            (513, Bank::Borrowed),
            (1024, Bank::Borrowed),
            (2047, Bank::Borrowed),
            (2048, Bank::Borrowed),
            (4095, Bank::Borrowed),
            (4096, Bank::Borrowed),
        ] {
            // A resident call of at most 512 rows takes whichever route it likes; a wide
            // call only the tile exchange.
            for route in [TILE, FULL] {
                let planned = EpEpoch::default().plan_call(rows, RESIDENT, PROMPT, route);
                if rows <= RESIDENT {
                    let call = planned.unwrap();
                    assert_eq!(
                        call,
                        EpCallPlan {
                            bank: bank,
                            restore_first: false,
                            acquire: false
                        },
                        "{rows} rows on {route:?}"
                    );
                } else if route == TILE {
                    assert_eq!(
                        planned.unwrap(),
                        EpCallPlan {
                            bank,
                            restore_first: false,
                            acquire: true
                        },
                        "{rows} rows"
                    );
                } else {
                    // Fail closed before anything is written.
                    assert!(planned.unwrap_err().contains("Full peer tables"), "{rows}");
                }
            }
            // A clean epoch never restores; no call plans a resident acquire.
            let events = drive(&[rows], |_| TILE).unwrap();
            let expect_borrowed = rows > RESIDENT;
            assert_eq!(banks_of(&events), vec![bank]);
            assert_eq!(
                events.contains(&Event::Acquire),
                expect_borrowed,
                "{rows} rows"
            );
            assert_eq!(
                events.contains(&Event::Completion),
                expect_borrowed,
                "{rows} rows"
            );
        }
        assert!(EpEpoch::default()
            .plan_call(0, RESIDENT, PROMPT, TILE)
            .is_err());
        assert!(EpEpoch::default()
            .plan_call(PROMPT + 1, RESIDENT, PROMPT, TILE)
            .is_err());
    }

    #[test]
    fn full_tile_then_full_tail_restores_before_the_tail() {
        // [4096, 4096, 12]: the 12-row tail takes Full peer tables.
        let events = drive(&[4096, 4096, 12], |_| TILE).unwrap();
        assert_eq!(
            events,
            vec![
                Event::Acquire,
                Event::Call {
                    rows: 4096,
                    bank: Bank::Borrowed
                },
                Event::Call {
                    rows: 4096,
                    bank: Bank::Borrowed
                },
                Event::Restore { before: 2 },
                Event::Call {
                    rows: 12,
                    bank: Bank::Resident
                },
            ],
            "one acquire, none between tiles, one restore before the tail, none at completion"
        );
    }

    #[test]
    fn borrowed_tile_then_512_row_s1_call_restores_before_the_resident_call() {
        let events = drive(&[4096, 512], |_| TILE).unwrap();
        assert_eq!(
            events,
            vec![
                Event::Acquire,
                Event::Call {
                    rows: 4096,
                    bank: Bank::Borrowed
                },
                Event::Restore { before: 1 },
                Event::Call {
                    rows: 512,
                    bank: Bank::Resident
                },
            ]
        );
    }

    #[test]
    fn two_full_tiles_stay_borrowed_until_completion() {
        let events = drive(&[4096, 4096], |_| TILE).unwrap();
        assert_eq!(
            events,
            vec![
                Event::Acquire,
                Event::Call {
                    rows: 4096,
                    bank: Bank::Borrowed
                },
                Event::Call {
                    rows: 4096,
                    bank: Bank::Borrowed
                },
                Event::Completion,
            ]
        );
        // A prompt that is all borrowed restores once; many tiles never restore per tile.
        let long = drive(&[4096; 16], |_| TILE).unwrap();
        assert_eq!(long.iter().filter(|e| **e == Event::Acquire).count(), 1);
        assert_eq!(
            long.iter()
                .filter(|e| matches!(e, Event::Restore { .. } | Event::Completion))
                .count(),
            1
        );
    }

    #[test]
    fn a_wide_full_table_call_is_refused_before_any_restore_or_acquire() {
        assert!(drive(&[1000], |_| FULL).is_err());
        // An invariant-prefill routing override that sends a tail to Full tables is fine
        // when it is small (resident bank) and refused when wide.
        assert!(drive(&[4096, 300], |rows| if rows == 300 { FULL } else { TILE }).is_ok());
        assert!(drive(&[4096, 600], |rows| if rows == 600 { FULL } else { TILE }).is_err());
    }

    #[test]
    fn nested_prompts_restore_once_at_the_outermost_end() {
        let mut epoch = EpEpoch::default();
        assert!(epoch.begin_prompt());
        assert!(!epoch.begin_prompt());
        assert!(!epoch.begin_prompt());
        epoch.acquired();
        assert_eq!(epoch.end_prompt(true), EpPromptEnd::Nested);
        assert_eq!(epoch.end_prompt(true), EpPromptEnd::Nested);
        assert_eq!(epoch.end_prompt(true), EpPromptEnd::Restore);
        epoch.restored();
        // Tiny tail already restored: the completion is clean.
        assert!(epoch.begin_prompt());
        assert_eq!(epoch.end_prompt(true), EpPromptEnd::Clean);
        // A failed outermost prompt with an open epoch recovers locally.
        assert!(epoch.begin_prompt());
        epoch.acquired();
        assert_eq!(epoch.end_prompt(false), EpPromptEnd::Recover);
        assert_eq!(epoch.epochs(), 2);
        // Unbalanced ends never underflow.
        let mut fresh = EpEpoch::default();
        assert_eq!(fresh.end_prompt(true), EpPromptEnd::Clean);
        assert_eq!(fresh.depth(), 0);
    }

    #[test]
    fn a_clean_epoch_takes_no_restore_for_decode_and_verify_calls() {
        // After a globally restored prompt every 1..=512-row call (decode, verify, tail) is
        // a clean fast path: no restore, no acquire, resident bank.
        let mut epoch = EpEpoch::default();
        for rows in [1, 2, 9, 64, 512, 1, 1, 1] {
            let call = epoch.plan_call(rows, RESIDENT, PROMPT, TILE).unwrap();
            assert_eq!(
                call,
                EpCallPlan {
                    bank: Bank::Resident,
                    restore_first: false,
                    acquire: false
                }
            );
        }
        epoch.acquired();
        assert!(
            epoch
                .plan_call(1, RESIDENT, PROMPT, TILE)
                .unwrap()
                .restore_first
        );
    }

    #[test]
    fn scratch_donors_and_s2_donors_are_disjoint_and_pinned_to_the_plan() {
        let plan = plan();
        check_stage_plan(&plan).unwrap();
        let (h, r) = (plan.donor_first_layer, plan.resident_layers);
        assert_eq!(r, 33);
        assert_eq!(plan.donor_layers.len(), 7);
        assert_eq!(h, 26);
        assert_eq!(plan.s2_donor_layers, [24, 25]);
        assert!(plan
            .donor_layers
            .iter()
            .all(|layer| !plan.s2_donor_layers.contains(layer)));
        // Overlap, a gap or a wrong parity order is refused.
        let mut broken = plan.clone();
        broken.s2_donor_layers = [25, 24];
        assert!(check_stage_plan(&broken).is_err());
        let mut broken = plan.clone();
        broken.s2_donor_layers = [24, 26];
        assert!(check_stage_plan(&broken).is_err());
        let mut broken = plan.clone();
        broken.donor_layers.remove(0);
        assert!(check_stage_plan(&broken).is_err());
        let mut broken = plan.clone();
        broken.resident_layers = broken.trunk_layers;
        assert!(check_stage_plan(&broken).is_err());
    }

    #[test]
    fn stage_sources_come_from_loan_mirrors_below_r_and_owners_above() {
        let plan = plan();
        // The real resolver: the loan mirrors cover exactly [H, R); every other layer resolves
        // to its original pinned owner at the stage (`pair` is `None`).
        let buffers: Vec<[Vec<u8>; 2]> = plan
            .donor_layers
            .iter()
            .map(|&layer| [vec![layer as u8; 64], vec![!(layer as u8); 32]])
            .collect();
        let resolver = EpStageHostSources {
            layers: plan.donor_layers.clone(),
            device: vec![[0, 0]; plan.donor_layers.len()],
            spans: buffers
                .iter()
                .map(|pair| {
                    [
                        (pair[0].as_ptr() as usize, pair[0].len()),
                        (pair[1].as_ptr() as usize, pair[1].len()),
                    ]
                })
                .collect(),
        };
        for layer in 0..plan.trunk_layers {
            let expected = (26..33).contains(&layer);
            let got = resolver.pair(layer);
            assert_eq!(got.is_some(), expected, "layer {layer}");
            if let Some([gate, down]) = got {
                let buffer = &buffers[layer - 26];
                assert_eq!(gate.as_ptr(), buffer[0].as_ptr());
                assert_eq!((gate.len(), down.len()), (64, 32));
                assert_eq!(down[0], !(layer as u8));
            }
        }
        // Parity chain of the stage: from each S2 donor, layers donor+2, donor+4, ... are
        // staged in turn; every host layer is reached exactly once and the last of each
        // parity is followed by the final donor restore.
        let first_host = plan.donor_first_layer;
        for donor in plan.s2_donor_layers {
            let chain: Vec<usize> = (donor + 2..plan.trunk_layers).step_by(2).collect();
            assert_eq!(
                chain[0],
                first_host + (donor % 2 != first_host % 2) as usize
            );
            assert!(chain.iter().all(|&layer| layer >= first_host));
        }
        let mut staged: Vec<usize> = plan
            .s2_donor_layers
            .iter()
            .flat_map(|&donor| (donor + 2..plan.trunk_layers).step_by(2))
            .filter(|&layer| layer >= first_host)
            .collect();
        staged.sort_unstable();
        assert_eq!(staged, (first_host..plan.trunk_layers).collect::<Vec<_>>());
        // 22 host layers per full tile (7 loan + 15 original), not 20 (§4.4).
        assert_eq!(staged.len(), 22);
    }

    #[test]
    fn borrowed_compact_entries_name_only_stage_slots_and_dummies() {
        // The borrowed compact table of a loan-mirror layer: owned experts at the mirror
        // (or, redirected, the stage donor), every other expert the dummy; the canonical
        // Full table over the device tensor is a separate pure computation and does not
        // move.
        let placement = Qwen4EpPlacement { rank: 1, ranks: 2 };
        let src = |gate, down| Qwen4PeerExpertSource {
            gate_up_base: gate,
            gate_up_bytes: 256 * 1_740_800,
            down_base: down,
            down_bytes: 256 * 870_400,
        };
        let device = src(0x10_0000_0000, 0x11_0000_0000);
        let mirror = src(0x20_0000_0000, 0x21_0000_0000);
        let dummy = Qwen4PeerExpertSource {
            gate_up_base: 0x30_0000_0000,
            gate_up_bytes: 1_740_800,
            down_base: 0x31_0000_0000,
            down_bytes: 870_400,
        };
        let peers = [src(0x40_0000_0000, 0x41_0000_0000), device];
        let full = |local| {
            qwen4_expert_table_entries(
                &Qwen4ExpertTableLayout::Full {
                    placement,
                    peers: &peers,
                },
                local,
                512,
                1_740_800,
                870_400,
            )
            .unwrap()
        };
        let canonical = full(device);
        let borrowed = qwen4_expert_table_entries(
            &Qwen4ExpertTableLayout::Ep { placement, dummy },
            mirror,
            512,
            1_740_800,
            870_400,
        )
        .unwrap();
        for (expert, entry) in borrowed.iter().enumerate() {
            if placement.owns(expert) {
                assert_eq!(entry.origin, Qwen4ExpertOrigin::Local);
                assert!(
                    (mirror.gate_up_base..mirror.gate_up_base + mirror.gate_up_bytes)
                        .contains(&entry.gate_up_ptr)
                );
                assert!((mirror.down_base..mirror.down_base + mirror.down_bytes)
                    .contains(&entry.down_ptr));
            } else {
                assert_eq!(entry.origin, Qwen4ExpertOrigin::Dummy);
                assert_eq!(entry.gate_up_ptr, dummy.gate_up_base);
                assert_eq!(entry.down_ptr, dummy.down_base);
            }
        }
        // The canonical table is untouched by the borrowed computation: owned experts still
        // name the device tensor, the peer's the peer's.
        assert_eq!(canonical, full(device));
        assert!(canonical.iter().enumerate().all(|(expert, entry)| {
            if placement.owns(expert) {
                entry.gate_up_ptr >= device.gate_up_base
                    && entry.gate_up_ptr < device.gate_up_base + device.gate_up_bytes
            } else {
                entry.gate_up_ptr >= peers[0].gate_up_base
                    && entry.gate_up_ptr < peers[0].gate_up_base + peers[0].gate_up_bytes
            }
        }));
        // The S2 redirect keeps non-owned entries and rewrites owned ones to the stage donor.
        let entries: Vec<usize> = borrowed.iter().map(|e| e.gate_up_ptr).collect();
        let donor = 0x50_0000_0000;
        let redirected = redirect_owned_entries(placement, donor, 1_740_800, &entries);
        for (expert, chunk) in redirected.chunks_exact(8).enumerate() {
            let pointer = usize::from_ne_bytes(chunk.try_into().unwrap());
            if placement.owns(expert) {
                assert_eq!(pointer, donor + (expert / 2) * 1_740_800);
            } else {
                assert_eq!(pointer, dummy.gate_up_base);
            }
        }
    }

    #[test]
    fn layout_resolves_every_segment_inside_its_owner() {
        let plan = plan();
        let address = |layer: usize, slot: EpDonorSlot| {
            0x10_0000_0000usize
                + layer * (4 << 30)
                + usize::from(slot == EpDonorSlot::Down) * (2 << 30)
        };
        let (segments, placements) = ep_reserve_layout(&plan, |layer, slot| {
            let bytes = match slot {
                EpDonorSlot::GateUp => 256 * 1_740_800,
                EpDonorSlot::Down => 256 * 870_400,
            };
            Ok((address(layer, slot), bytes))
        })
        .unwrap();
        assert_eq!(segments.len(), plan.segments.len());
        assert_eq!(placements.len(), plan.placements.len());
        for (segment, plan_segment) in segments.iter().zip(&plan.segments) {
            assert_eq!(
                segment.address,
                address(plan_segment.layer, plan_segment.slot)
            );
            assert_eq!(segment.bytes, plan_segment.capacity);
        }
        // An owner smaller than the lent capacity, or a null address, is refused.
        assert!(ep_reserve_layout(&plan, |_, _| Ok((0x1000, 1))).is_err());
        assert!(ep_reserve_layout(&plan, |_, _| Ok((0, u64::MAX))).is_err());
    }

    // -- bank switching over a fake device (no GPU) -------------------------

    #[derive(Default)]
    struct FakeDevice {
        next: usize,
        live: BTreeMap<usize, u64>,
    }

    impl EpScratchDevice for FakeDevice {
        fn alloc_zeroed(
            &mut self,
            elements: usize,
            dtype: DType,
        ) -> hip_bridge::HipResult<GpuTensor> {
            let bytes = elements * dtype.size();
            let address = self.next;
            self.next = (address + bytes + 255) / 256 * 256 + 256;
            self.live.insert(address, bytes as u64);
            Ok(GpuTensor {
                // SAFETY: a CPU-only address, never dereferenced.
                buf: unsafe {
                    hip_bridge::DeviceBuffer::from_raw(address as *mut std::ffi::c_void, bytes)
                },
                shape: vec![elements],
                dtype,
            })
        }

        fn release_tensor(&mut self, tensor: GpuTensor) -> hip_bridge::HipResult<()> {
            self.live
                .remove(&(tensor.buf.as_ptr() as usize))
                .map(|_| ())
                .ok_or_else(|| hip_bridge::HipError::new(1, "free of a non-live allocation"))
        }
    }

    fn banks(device: &mut FakeDevice) -> (Qwen4GpuForwardScratch, EpReserveBanks, EpReservePlan) {
        device.next = 0x4_0000_0000;
        let plan = plan();
        let (segments, placements) = ep_reserve_layout(&plan, |layer, slot| {
            let base = 0x20_0000_0000usize + layer * (2 << 30);
            Ok(match slot {
                EpDonorSlot::GateUp => (base, 256 * 1_740_800),
                EpDonorSlot::Down => (base + (1 << 30), 256 * 870_400),
            })
        })
        .unwrap();
        let (scratch, banks) = EpReserveBanks::build(
            device,
            &compact_test_config(),
            Qwen4ForwardScratchMode::Full,
            PROMPT,
            &segments,
            &placements,
        )
        .unwrap();
        (scratch, banks, plan)
    }

    /// `ep_reserve_forward_chunk`'s bank handling around `body`: select, run,
    /// swap back on every return path.
    fn run_tile<R>(
        banks: &mut EpReserveBanks,
        scratch: &mut Qwen4GpuForwardScratch,
        rows: usize,
        route: EpBankRoute,
        body: impl FnOnce(&Qwen4GpuForwardScratch) -> Result<R, String>,
    ) -> Result<R, String> {
        banks
            .select(scratch, rows, route)
            .map_err(|e| e.to_string())?;
        let result = body(scratch);
        banks.swap_out(scratch).map_err(|e| e.to_string())?;
        result
    }

    #[test]
    fn every_return_path_restores_the_resident_bank_and_the_registered_root() {
        let mut device = FakeDevice::default();
        let (mut scratch, mut banks, _plan) = banks(&mut device);
        let (root_address, root_bytes) = banks.registered_grouped_down();
        let ids_address = banks.registered_token_ids();
        assert_eq!(root_bytes as u64, 475 * MIB);
        assert_eq!(scratch.max_chunk, RESIDENT);
        for (rows, route, fail) in [
            (4096, TILE, false),
            (4096, TILE, true),
            (700, TILE, false),
            (512, TILE, false),
            (12, FULL, false),
            (4096, TILE, false),
        ] {
            let outcome = run_tile(&mut banks, &mut scratch, rows, route, |bank| {
                // The registered root and the token ids are the same in either bank, at the
                // full persistent extent.
                assert_eq!(bank.moe_y_down_grouped.buf.as_ptr() as usize, root_address);
                assert_eq!(bank.moe_y_down_grouped.buf.size(), root_bytes);
                assert_eq!(bank.token_ids.buf.as_ptr() as usize, ids_address);
                if rows > RESIDENT {
                    assert_eq!(bank.max_chunk, PROMPT);
                }
                if fail {
                    Err("tile failed".to_string())
                } else {
                    Ok(())
                }
            });
            assert_eq!(outcome.is_err(), fail);
            // Whatever happened, the next call boundary sees the resident bank.
            assert_eq!(scratch.max_chunk, RESIDENT, "{rows} rows, fail {fail}");
            assert_eq!(banks.active(), Bank::Resident);
            banks.check_stable(&scratch).unwrap();
        }
        // A wide Full-table call is refused and swaps nothing.
        assert!(run_tile(&mut banks, &mut scratch, 1000, FULL, |_| Ok(())).is_err());
        assert_eq!(scratch.max_chunk, RESIDENT);
        assert_eq!(banks.active(), Bank::Resident);
        assert_eq!(banks.stats().swaps_in, banks.stats().swaps_out);
    }

    #[test]
    fn exchange_capacity_comes_from_the_prompt_plan_not_the_resident_bank() {
        let mut device = FakeDevice::default();
        let (scratch, banks, _plan) = banks(&mut device);
        let capacity = banks.exchange_capacity();
        let hidden = compact_test_config().hidden_size;
        assert_eq!(capacity.rows, PROMPT);
        // G(4096) = 48640 grouped rows = 3040 tiles of 16: the largest mask capacity.
        assert_eq!(capacity.grouped_rows, 48_640);
        assert_eq!(capacity.tile_ids, 3_040);
        assert_eq!(exchange_max_tiles(&capacity, hidden), 3_040);
        // The resident bank's own ids hold far fewer; the registration must not use them.
        let resident_tiles = scratch.moe_expert_tile_ids.buf.size() / std::mem::size_of::<i32>();
        assert_eq!(resident_tiles, 12_800 / 16);
        assert!(resident_tiles < capacity.tile_ids);
        // The registered view spans the whole 475 MiB root in both banks.
        assert_eq!(banks.registered_grouped_down().1 as u64, 475 * MIB);
    }

    #[test]
    fn receipt_keys_follow_the_oracle_scope_phase_and_ordinal() {
        // The scope is process-global: this is the only test that sets it.
        let mut ordinals = EpReceiptOrdinals::default();
        ep_reserve_receipt_scope(None);
        // Unset: the rank's own entry sequence names the prompt.
        assert_eq!(ordinals.next_key(7, false), (7, EpReceiptPhase::Target, 0));
        ep_reserve_receipt_scope(Some(2));
        assert_eq!(ordinals.next_key(8, false), (2, EpReceiptPhase::Target, 0));
        assert_eq!(ordinals.next_key(9, true), (2, EpReceiptPhase::MtpFill, 0));
        assert_eq!(ordinals.next_key(10, false), (2, EpReceiptPhase::Target, 1));
        assert_eq!(ordinals.next_key(11, false), (2, EpReceiptPhase::Target, 2));
        assert_eq!(ordinals.next_key(12, true), (2, EpReceiptPhase::MtpFill, 1));
        ep_reserve_receipt_scope(Some(3));
        assert_eq!(ordinals.next_key(13, false), (3, EpReceiptPhase::Target, 0));
        ep_reserve_receipt_scope(None);
        assert_eq!(receipt_scope(), None);
        // Another rank's counters are independent.
        assert_eq!(
            EpReceiptOrdinals::default().next_key(0, false),
            (0, EpReceiptPhase::Target, 0)
        );
        assert_eq!(EpReceiptPhase::MtpFill.name(), "mtp_fill");
        assert_eq!(EpReceiptPhase::Target.name(), "target");
    }

    #[test]
    fn receipts_and_fingerprints_are_deterministic() {
        assert_eq!(generation_fingerprint("a"), generation_fingerprint("a"));
        assert_ne!(generation_fingerprint("a"), generation_fingerprint("b"));
        let record = EpPromptRecord {
            prompt: 3,
            phase: EpReceiptPhase::MtpFill,
            ordinal: 2,
            chunk_rows: vec![4096, 12],
            banks: vec![Bank::Borrowed, Bank::Resident],
            restores: vec![EpRestoreEvent {
                before_chunk: 1,
                bytes: 1 << 30,
                restore_ms: 12.5,
                barrier_wait_ms: 0.25,
                s2_dma_tail_ms: 1.0,
            }],
            epochs_before: 0,
            started: Instant::now(),
            excluded: Duration::ZERO,
            exchange_before: Some(0xabc000),
            hash_before: Some("00ff".to_string()),
        };
        let json = prompt_rank_json(
            1,
            &record,
            1,
            Duration::from_millis(40),
            Some(0xabc000),
            Some("00ff".to_string()),
        );
        assert_eq!(json["prompt"], 3);
        assert_eq!(json["phase"], "mtp_fill");
        assert_eq!(json["ordinal"], 2);
        assert_eq!(json["kind"], "prompt_rank");
        assert_eq!(json["rank"], 1);
        assert_eq!(json["rows"], 4108);
        assert_eq!(json["banks"], serde_json::json!(["borrowed", "resident"]));
        assert_eq!(json["restores"][0]["before_chunk"], 1);
        assert_eq!(json["restores"][0]["bytes"], 1u64 << 30);
        assert_eq!(
            json["exchange_address_before"],
            json["exchange_address_after"]
        );
        assert_eq!(json["donor_hash_before"], json["donor_hash_after"]);
        // Without hashes the keys are absent (the gate then rejects the record).
        let bare = prompt_rank_json(0, &record, 1, Duration::from_millis(1), Some(1), None);
        assert!(bare.get("donor_hash_after").is_none());
    }

    /// The plan with explicit donors (a unit plan's forced whole-Resident layers), the same
    /// capacity as [`plan`].
    fn explicit_plan() -> EpReservePlan {
        let config = compact_test_config();
        let forward = EpReserveForwardPlan::from_config(&config, Qwen4ForwardScratchMode::Full, 0)
            .expect("pinned forward plan");
        let geometry = EpRankGeometry::flash_next(&config, 2).expect("geometry");
        let layer = geometry.layer_bytes().unwrap();
        let (non_expert, reserve) = (6 * GIB, 6 * GIB);
        let free = non_expert + reserve + layer * 33;
        crate::ep_reserve::plan_ep_reserve_with_donors(
            &forward,
            &geometry,
            &EpReserveCapacity {
                free_per_rank: vec![free, free],
                non_expert_bytes: non_expert,
                reserve_bytes: reserve,
                lane_bytes: 0,
                host_budget_bytes: 64 * GIB,
                min_resident_layers: EP_RESERVE_MIN_RESIDENT_LAYERS,
            },
            41,
            &[40, 3, 17, 9, 22, 31, 12, 5, 44],
        )
        .expect("an explicit-donor plan")
    }

    #[test]
    fn explicit_donor_plans_are_served_by_the_unit_stage_not_the_s2_donors() {
        let plan = explicit_plan();
        assert!(plan.explicit_donors);
        check_stage_plan(&plan).unwrap();
        assert_eq!(plan.donor_layers, vec![3, 9, 12, 17, 22, 31, 40]);
        let facts = |first_host, donors, min_rows, serves| EpReserveStageFacts {
            first_host,
            donors,
            min_rows,
            serves_donor_layers: serves,
        };
        // The unit stage has no S2 donors and its first staged layer is whatever the plan stages:
        // only "every donor has a slot" and "borrowed-only" matter.
        assert_eq!(
            ep_reserve_stage_refusal(&plan, RESIDENT, &facts(0, [usize::MAX; 2], 513, true)),
            None
        );
        assert_eq!(
            ep_reserve_stage_refusal(&plan, RESIDENT, &facts(7, [usize::MAX; 2], 4096, true)),
            None
        );
        let text =
            ep_reserve_stage_refusal(&plan, RESIDENT, &facts(0, [usize::MAX; 2], 513, false))
                .expect("a stage without a donor slot is refused");
        assert!(text.contains("[3, 9, 12, 17, 22, 31, 40]"), "{text}");
        // A stage that would serve the resident bank is refused, whichever kind it is.
        assert!(ep_reserve_stage_refusal(
            &plan,
            RESIDENT,
            &facts(0, [usize::MAX; 2], RESIDENT, true)
        )
        .is_some());
    }

    #[test]
    fn a_borrowed_unit_stage_never_lends_a_scratch_donor() {
        let plan = explicit_plan();
        let facts = |donors| EpReserveStageFacts {
            first_host: 2,
            donors,
            min_rows: 4096,
            serves_donor_layers: true,
        };
        // Lent layers 0 and 1 are not scratch donors of this plan: admitted.
        assert_eq!(
            ep_reserve_stage_refusal(&plan, RESIDENT, &facts([0, 1])),
            None
        );
        // A lent layer that is also a scratch donor would be overwritten while it is scratch.
        let text = ep_reserve_stage_refusal(&plan, RESIDENT, &facts([2, 3]))
            .expect("a lent scratch donor is refused");
        assert!(text.contains("scratch donor layer 3"), "{text}");
    }

    #[test]
    fn suffix_plans_keep_the_s2_stage_admission_exactly() {
        let plan = plan();
        assert!(!plan.explicit_donors);
        let facts = |first_host, donors, min_rows| EpReserveStageFacts {
            first_host,
            donors,
            min_rows,
            // A suffix plan never asks the unit stage: this bit must not matter.
            serves_donor_layers: false,
        };
        assert_eq!(
            ep_reserve_stage_refusal(&plan, RESIDENT, &facts(26, [24, 25], 513)),
            None
        );
        for (first_host, donors, min_rows) in [
            (25, [24, 25], 513),
            (26, [25, 24], 513),
            (26, [24, 25], RESIDENT),
            (26, [24, 25], 1),
        ] {
            let text =
                ep_reserve_stage_refusal(&plan, RESIDENT, &facts(first_host, donors, min_rows))
                    .unwrap_or_else(|| {
                        panic!("{first_host} {donors:?} {min_rows} must be refused")
                    });
            assert!(
                text.starts_with("EP reserve stage mismatch: first host layer"),
                "{text}"
            );
            assert!(text.contains("against the plan's 26 / [24, 25]"), "{text}");
        }
    }

    /// The trunk-shard partial chunk route binds this rank's compact table (no peer expert
    /// read), so a wide call on it may borrow the donor bank exactly like the tile exchange;
    /// a wide call on neither route stays the refused Full-table route. Decode, a lone row
    /// and the few-row (2..=8 row) verify windows are never the partial CHUNK route.
    #[test]
    fn the_trunk_shard_partial_chunk_may_borrow_the_donor_bank() {
        assert_eq!(ep_wide_bank_route(true, false), EpBankRoute::TileExchange);
        assert_eq!(ep_wide_bank_route(false, true), EpBankRoute::TileExchange);
        assert_eq!(ep_wide_bank_route(true, true), EpBankRoute::TileExchange);
        assert_eq!(ep_wide_bank_route(false, false), EpBankRoute::FullTable);
        let epoch = EpEpoch::default();
        let wide = RESIDENT + 1;
        let borrowed = epoch
            .plan_call(wide, RESIDENT, PROMPT, ep_wide_bank_route(false, true))
            .expect("the partial chunk route borrows");
        assert_eq!(borrowed.bank, EpBankKind::Borrowed);
        assert!(borrowed.acquire && !borrowed.restore_first);
        let refused = epoch
            .plan_call(wide, RESIDENT, PROMPT, ep_wide_bank_route(false, false))
            .unwrap_err();
        assert!(refused.contains("Full peer tables"), "{refused}");
        // The predicate: needs an EP forward, the installed route, and a routing row count
        // that is not a lone row or a few-row window.
        assert!(ep_trunk_partial_chunk(true, 4096, 4096, true));
        assert!(ep_trunk_partial_chunk(true, 9, 9, true));
        assert!(
            !ep_trunk_partial_chunk(true, 4096, 4096, false),
            "route not installed"
        );
        assert!(
            !ep_trunk_partial_chunk(false, 4096, 4096, true),
            "not an EP forward"
        );
        assert!(!ep_trunk_partial_chunk(true, 1, 1, true), "decode");
        for rows in 2..=QWEN4_EP_GATHER_MAX_ROWS {
            assert!(
                !ep_trunk_partial_chunk(true, rows, rows, true),
                "{rows}-row window"
            );
        }
        // a wide launch whose routing rows are a few-row count is not a verify window (its
        // routing rows differ from its launch rows): it takes the chunk route
        assert!(ep_trunk_partial_chunk(true, 4096, 8, true));
    }

    #[test]
    fn lane_stage_knob_is_an_integer_of_at_least_two() {
        for value in [
            None,
            Some(""),
            Some("abc"),
            Some("-5"),
            Some("0"),
            Some("1"),
            Some("5.5"),
        ] {
            assert_eq!(ep_lane_stage_knob_from(value), None, "{value:?}");
        }
        assert_eq!(ep_lane_stage_knob_from(Some("2")), Some(2));
        assert_eq!(ep_lane_stage_knob_from(Some(" 513 ")), Some(513));
        assert_eq!(ep_lane_stage_knob_from(Some("2048")), Some(2048));
    }

    #[test]
    fn lane_stage_floor_is_the_owner_routed_route_and_the_resident_bank() {
        // No reserve: only the owner-routed route's 512 routing rows bound it.
        assert_eq!(ep_lane_stage_floor(None), 512);
        // A reserve load: strictly above the resident bank's call limit.
        assert_eq!(ep_lane_stage_floor(Some(RESIDENT)), 513);
        // A call limit lowered below the route floor never lowers the floor under it.
        assert_eq!(ep_lane_stage_floor(Some(RESIDENT - 1)), 512);
        assert_eq!(ep_lane_stage_floor(Some(0)), 512);
        // A larger resident bank raises it; the sum saturates.
        assert_eq!(ep_lane_stage_floor(Some(2048)), 2049);
        assert_eq!(ep_lane_stage_floor(Some(usize::MAX)), usize::MAX);
    }

    #[test]
    fn lane_stage_threshold_is_exactly_the_stage_s_without_the_knob_or_off_a_lane_tile() {
        for stage_min_rows in [2, 511, 512, 513, 768, 1536, 2048, 4096] {
            for resident in [None, Some(0), Some(RESIDENT), Some(4096)] {
                // Knob unset: today's threshold, lane tile or not.
                for lane in [false, true] {
                    assert_eq!(
                        ep_lane_stage_min_rows(stage_min_rows, None, lane, resident),
                        stage_min_rows
                    );
                }
                // Knob set, not a lane tile (singleton prefill, decode, verify): unchanged.
                for knob in [2, 256, 513, 1024, 4096] {
                    assert_eq!(
                        ep_lane_stage_min_rows(stage_min_rows, Some(knob), false, resident),
                        stage_min_rows
                    );
                }
            }
        }
    }

    /// Whether a tile of `rows` stages on the rows axis (the call site's `n >= threshold`).
    fn stages(rows: usize, stage_min: usize, knob: Option<usize>, resident: Option<usize>) -> bool {
        rows >= ep_lane_stage_min_rows(stage_min, knob, true, resident)
    }

    #[test]
    fn lane_stage_decision_table_rows_axis() {
        const STAGE_MIN: usize = 2048;
        // (resident call limit: none = ordinary / slab load, Some = decoupled reserve)
        // Knob unset: a lane tile stages from the stage's 2048 rows, exactly as a singleton.
        for resident in [None, Some(RESIDENT)] {
            assert!(!stages(2047, STAGE_MIN, None, resident));
            assert!(stages(2048, STAGE_MIN, None, resident));
            assert!(stages(4096, STAGE_MIN, None, resident));
        }
        // Ordinary / slab load, knob 256 (below the route floor): the floor is 512.
        assert!(!stages(149, STAGE_MIN, Some(256), None));
        assert!(!stages(511, STAGE_MIN, Some(256), None));
        assert!(stages(512, STAGE_MIN, Some(256), None));
        assert!(stages(2047, STAGE_MIN, Some(256), None));
        // Decoupled reserve (unit stage or S2 donors), knob 256: the floor is the bank limit + 1.
        assert!(!stages(512, STAGE_MIN, Some(256), Some(RESIDENT)));
        assert!(stages(513, STAGE_MIN, Some(256), Some(RESIDENT)));
        // Knob above the floor is authoritative, below and above the stage's own threshold.
        for resident in [None, Some(RESIDENT)] {
            assert!(!stages(1023, STAGE_MIN, Some(1024), resident));
            assert!(stages(1024, STAGE_MIN, Some(1024), resident));
            assert!(!stages(2048, STAGE_MIN, Some(4096), resident));
            assert!(stages(4096, STAGE_MIN, Some(4096), resident));
        }
        // Decode, verify and every serve-shaped lane prompt (60..150 rows, the C8 story / code
        // cells) are below every floor: the knob never reaches them.
        for rows in [1, 2, 8, 9, 60, 149, 256, 511] {
            for knob in [2, 64, 256, 512] {
                for resident in [None, Some(RESIDENT)] {
                    assert!(
                        !stages(rows, STAGE_MIN, Some(knob), resident),
                        "{rows} {knob}"
                    );
                }
            }
        }
    }

    #[test]
    fn lane_stage_threshold_never_serves_the_resident_bank_on_any_reserve_plan_kind() {
        // The two reserve plan kinds `ep_reserve_stage_refusal` admits: a suffix plan with the S2
        // donors (the release default) and an explicit-donor plan served by the unit stage (D1).
        let suffix = plan();
        let explicit = explicit_plan();
        assert!(!suffix.explicit_donors && explicit.explicit_donors);
        // The stage facts each plan kind's built stage reports.
        let facts = |plan: &EpReservePlan, min_rows| {
            if plan.explicit_donors {
                EpReserveStageFacts {
                    first_host: 0,
                    donors: [usize::MAX; 2],
                    min_rows,
                    serves_donor_layers: true,
                }
            } else {
                EpReserveStageFacts {
                    first_host: 26,
                    donors: [24, 25],
                    min_rows,
                    serves_donor_layers: false,
                }
            }
        };
        for (name, plan) in [("S2 donors", &suffix), ("unit stage", &explicit)] {
            // The stage's own threshold is admitted as built.
            assert_eq!(
                ep_reserve_stage_refusal(plan, RESIDENT, &facts(plan, 2048)),
                None,
                "{name}"
            );
            for knob in [2, 64, 256, 511, 512, 513, 1024, 2047, 2048, 4096, 1 << 20] {
                let threshold = ep_lane_stage_min_rows(2048, Some(knob), true, Some(RESIDENT));
                // A lane tile of at most the resident bank's rows never stages ...
                assert!(threshold > RESIDENT, "{name} knob {knob}: {threshold}");
                assert!(
                    !stages(RESIDENT, 2048, Some(knob), Some(RESIDENT)),
                    "{name} {knob}"
                );
                // ... so the stage the lane threshold produces is the borrowed-only stage the
                // reserve admission requires.
                assert_eq!(
                    ep_reserve_stage_refusal(plan, RESIDENT, &facts(plan, threshold)),
                    None,
                    "{name} knob {knob}"
                );
            }
        }
    }

    #[test]
    fn lane_tile_note_labels_every_route() {
        let note = |route, staged| LaneTileNote {
            rows: 149,
            route,
            staged,
            stage_min_rows: Some(2048),
        };
        use LaneTileRoute::*;
        assert_eq!(note(FullTable, false).label(), "full-table");
        assert_eq!(note(Tile, false).label(), "tile");
        assert_eq!(note(Tile, true).label(), "tile+stage");
        assert_eq!(note(Partial, false).label(), "partial");
        assert_eq!(note(Partial, true).label(), "partial+stage");
        assert_eq!(note(PartialMask, false).label(), "partial-mask");
        assert_eq!(note(PartialMask, true).label(), "partial-mask+stage");
        // The trunk shard's partial route wins over the tile exchange; the mask only applies there.
        assert_eq!(LaneTileRoute::of(true, true, true), PartialMask);
        assert_eq!(LaneTileRoute::of(false, true, false), Partial);
        assert_eq!(LaneTileRoute::of(true, false, false), Tile);
        assert_eq!(LaneTileRoute::of(false, false, false), FullTable);
    }

    #[test]
    fn prefill_tile_knob_parses_and_refuses_below_the_few_row_routes() {
        for unset in [None, Some(""), Some("   ")] {
            assert_eq!(ep_prefill_tile_knob_from(unset), PrefillTileKnob::Unset);
        }
        for bad in ["abc", "-1", "1.5", "0x20", "9rows"] {
            let PrefillTileKnob::Refused(reason) = ep_prefill_tile_knob_from(Some(bad)) else {
                panic!("{bad:?} must be refused");
            };
            assert!(reason.contains("not an integer"), "{reason}");
        }
        // The few-row routes (decode, verify: at most 8 rows) keep the gathered MoE.
        for rows in ["0", "1", "2", "8"] {
            let PrefillTileKnob::Refused(reason) = ep_prefill_tile_knob_from(Some(rows)) else {
                panic!("{rows} must be refused");
            };
            assert!(reason.contains("below the floor of 9"), "{reason}");
        }
        assert_eq!(
            ep_prefill_tile_knob_from(Some("9")),
            PrefillTileKnob::Floor(9)
        );
        assert_eq!(
            ep_prefill_tile_knob_from(Some(" 128 ")),
            PrefillTileKnob::Floor(128)
        );
        assert_eq!(
            ep_prefill_tile_knob_from(Some("512")),
            PrefillTileKnob::Floor(512)
        );
        assert_eq!(
            ep_prefill_tile_knob_from(Some("100000")),
            PrefillTileKnob::Floor(100000)
        );
    }

    #[test]
    fn prefill_tile_floor_decision_table() {
        const DEFAULT: usize = 512;
        // (knob, prefill scope, tile hook, A8 chain active) -> floor
        let table: &[(Option<usize>, bool, bool, bool, usize)] = &[
            // Unset: 9 iff the A8 chain is the active chain of an EP prefill forward with the
            // symmetric-IU4 tile exchange installed; A8 killed (A4 chain) keeps 512.
            (None, true, true, true, 9),
            (None, true, true, false, DEFAULT),
            // Not a prefill forward (decode, verify, MTP head): 512 whatever else is set.
            (None, false, true, true, DEFAULT),
            (Some(9), false, true, true, DEFAULT),
            // No tile hook (single device, gfx1100 WMMA tile route, no exchange): 512.
            (None, true, false, true, DEFAULT),
            (Some(9), true, false, true, DEFAULT),
            // An explicit knob wins over the A8 default, up and down: 512 is the kill switch of
            // the floor alone, a lower value is a lab floor (also on the A4 chain).
            (Some(512), true, true, true, DEFAULT),
            (Some(9), true, true, false, 9),
            (Some(128), true, true, true, 128),
            (Some(128), true, true, false, 128),
            (Some(511), true, true, true, 511),
            // Never raised above today's floor, never lowered below the few-row routes' rows.
            (Some(4096), true, true, true, DEFAULT),
            (Some(2), true, true, true, 9),
        ];
        for &(knob, prefill, hook, a8, want) in table {
            assert_eq!(
                ep_prefill_tile_sym_floor(knob, prefill, hook, a8),
                want,
                "knob {knob:?} prefill {prefill} hook {hook} a8 {a8}"
            );
        }
        assert_eq!(EP_PREFILL_TILE_DEFAULT_FLOOR, 9);
    }

    #[test]
    fn prefill_tile_floor_decides_the_same_for_a_lane_tile_and_the_singleton() {
        // The decision reads no row count and no lane state: a lane prefill tile, the singleton
        // prefill and the MTP prompt-fill tile are all `prefill_scope` forwards of one load
        // entering `enter_tile_scopes`, so they get the same floor and the same route for the
        // same rows. Pinned at the one call site that feeds the decision.
        // The scope that feeds it: enter_tile_scopes passes self.prefill_scope, the A8 request
        // of the scope it just set and the installed hook, nothing per lane.
        let src = include_str!("../gpu_forward.rs");
        let at = src
            .find("let sym_floor_now = ep_reserve_forward::ep_prefill_tile_sym_floor(")
            .unwrap();
        let call = &src[at..at + 260];
        assert!(call.contains("self.prefill_scope"), "{call}");
        assert!(call.contains("tile_hook"), "{call}");
        assert!(call.contains("a8_chain"), "{call}");
        assert!(!call.contains("lane"), "{call}");
    }

    #[test]
    fn prefill_tile_floor_never_reaches_a_decode_or_verify_row_count() {
        // The EP gathered row gather (2..=8 rows), the historic few-row routes (8) and the
        // gfx1201 row-exact verify window (8) are all below every floor the function returns.
        let few = QWEN4_EP_GATHER_MAX_ROWS
            .max(rdna_compute::prefill_invariant::FEW_ROW_ROUTE_ROWS)
            .max(hipfire_config::QWEN4_ROW_EXACT_VERIFY_ROWS_GFX1201);
        assert_eq!(few, 8);
        assert_eq!(rdna_compute::gemm::QWEN4_MOE_SYM_IU4_LOWEST_FLOOR, few + 1);
        for knob in (0..=700).map(Some).chain([None]) {
            for prefill in [false, true] {
                for hook in [false, true] {
                    for a8 in [false, true] {
                        let floor = ep_prefill_tile_sym_floor(knob, prefill, hook, a8);
                        assert!(floor > few, "{knob:?} {prefill} {hook} {a8}: {floor}");
                        assert!(floor <= rdna_compute::gemm::QWEN4_MOE_SYM_IU4_MIN_ROWS);
                        // Rows 8 / 9 / 511 / 512 against the default A8 floor: the few-row
                        // boundary and the historic floor, both sides.
                        if knob.is_none() && prefill && hook && a8 {
                            assert_eq!(floor, 9);
                            assert!(
                                [8usize, 9, 511, 512].map(|rows| rows >= floor)
                                    == [false, true, true, true]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn declared_route_capability_follows_the_floor_in_scope() {
        use hipfire_dispatch::families::moe::MoeRouteCapability::{
            Fp8B128Grouped, Qt44Qt53Grouped, Qt44Qt53GroupedSymmetric as Symmetric,
            SlabCodecGrouped,
        };
        // A verified-symmetric layer: the floor in scope decides.
        assert_eq!(
            declared_route_capability(false, false, true, 511, 512),
            Qt44Qt53Grouped
        );
        assert_eq!(
            declared_route_capability(false, false, true, 512, 512),
            Symmetric
        );
        assert_eq!(
            declared_route_capability(false, false, true, 8, 9),
            Qt44Qt53Grouped
        );
        assert_eq!(
            declared_route_capability(false, false, true, 9, 9),
            Symmetric
        );
        assert_eq!(
            declared_route_capability(false, false, true, 149, 9),
            Symmetric
        );
        assert_eq!(
            declared_route_capability(false, false, true, 149, 150),
            Qt44Qt53Grouped
        );
        // An unverified layer never declares it, at any floor.
        for floor in [9, 128, 512] {
            assert_eq!(
                declared_route_capability(false, false, false, 4096, floor),
                Qt44Qt53Grouped
            );
        }
        // Slab and FP8 layers are their own routes at every row count and floor.
        assert_eq!(
            declared_route_capability(true, false, true, 149, 9),
            SlabCodecGrouped
        );
        assert_eq!(
            declared_route_capability(false, true, true, 149, 9),
            Fp8B128Grouped
        );
    }
}
