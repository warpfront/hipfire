// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EP tail borrow (PrefBorrow D3): an EP2 compact load prefills in 8192-row tiles while its
//! resident scratch stays the 4096-row bank, by lending the D2 tail's slot pool to the prefill
//! as the extra scratch bank (`HIPFIRE_QWEN4_EP_TAIL_BORROW`).
//!
//! A child module of `gpu_forward`, so the monolithic forward keeps only the call sites.
//! Everything here is gated on `Qwen4GpuForward::ep_tail_borrow` being `Some`; a load without a
//! plan never reaches any of it.
//!
//! * [`resolve_ep_tail_borrow`] is the pure knob / admission resolver the loader and the
//!   bundle share; [`qwen4_ep_tail_borrow_forward_bytes`] is the forward charge of a planned
//!   load (the resident bank plus the persistent roots, never the borrowed-row scratch).
//! * [`EpTailEpoch`] is the pure epoch / prompt-bracket state machine, factored like
//!   `EpEpoch` of the EP reserve (`ep_reserve_forward`); the forward drives it and the CPU
//!   tests drive the same code.
//! * [`EpTailBorrow`] owns the persistent roots and, once the mesh attached it, the borrowed
//!   bank, and the `impl Qwen4GpuForward` block holds the bank switch, the all-rank open /
//!   close protocol and the prompt nesting.
//!
//! # Memory model
//!
//! Two roots are persistent (one allocation each, sized for the borrowed rows, owned by
//! [`EpTailBorrow`], viewed by both banks): `TokenIds` (the ids a prompt tile uploaded are read
//! through whichever bank is active) and `MoeYDownGrouped`, the registered tile-exchange root
//! (the exchange rejects a grouped-down buffer whose address or size differs from its
//! registration, and a borrowed-row call needs more tiles than the resident rows' root holds).
//! The resident bank (`resident_rows`, today's EP chunk) owns every other root of its plan; its
//! views of the two persistent roots span the whole allocation, so the registered address is
//! identical in both banks and across every swap. The borrowed bank allocates nothing: every
//! other owning root is a non-owning view at `base + offset` of the tail pool span.
//!
//! # Epoch protocol
//!
//! Open (first prefill tile wider than the resident bank): refuse under stream capture, drain
//! the device, `UnitTailRank::borrow_suspend` (canonical tables, policy suspended), cross the
//! all-rank host barrier (no rank can read any rank's pool afterwards), zero the span. Close
//! (restore) at the outermost prompt end with no continuation, on a failed prompt and at
//! teardown: drain, `borrow_restore`, optional `borrow_verify`, all-rank barrier,
//! `borrow_resume`. A failure of the close poisons the borrow: every later forward refuses.

use super::ep_reserve_scratch::{check_aliases_inside_roots, EpExchangeCapacity};
use super::*;
use crate::expert_unit_tail_rank::UnitTailRank;
use hipfire_runtime::ep_tile_exchange::{HOST_TAIL_BORROW_ACQUIRE, HOST_TAIL_BORROW_RESTORE};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Knobs and the planning resolver
// ---------------------------------------------------------------------------

/// Developer knob: unset = [`EP_TAIL_BORROW_DEFAULT_ON`], `0` off, `1` on with
/// [`EP_TAIL_BORROW_DEFAULT_ROWS`] borrowed rows, an integer N above the resident rows (a
/// multiple of 256) = N borrowed rows. Ignored (with a note) when `HIPFIRE_PREFILL_CHUNK_ROWS`
/// is set, without the D2 tail's prefill lever, without the EP tile hook, off gfx1201 or the
/// compact plan.
pub const EP_TAIL_BORROW_ENV: &str = "HIPFIRE_QWEN4_EP_TAIL_BORROW";

/// Developer knob: `1` runs `UnitTailRank::borrow_verify` (a byte compare of every occupied
/// slot against its canonical host bytes) at every epoch close and keeps its digest.
pub(crate) const EP_TAIL_BORROW_VERIFY_ENV: &str = "HIPFIRE_QWEN4_EP_TAIL_BORROW_VERIFY";

/// Developer knob: `1` prints one stderr line per closed epoch.
pub(crate) const EP_TAIL_BORROW_LOG_ENV: &str = "HIPFIRE_QWEN4_EP_TAIL_BORROW_LOG";

/// Whether the borrow is on when [`EP_TAIL_BORROW_ENV`] is unset. One constant flips the
/// shipped default once the served A/B lands.
pub const EP_TAIL_BORROW_DEFAULT_ON: bool = false;

/// Borrowed rows of `=1` and of the shipped default.
pub const EP_TAIL_BORROW_DEFAULT_ROWS: usize = 8192;

/// What [`EP_TAIL_BORROW_ENV`] asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpTailBorrowKnob {
    /// Unset (or empty): the shipped default.
    Default,
    /// `0`.
    Off,
    /// `1`: [`EP_TAIL_BORROW_DEFAULT_ROWS`] borrowed rows.
    On,
    /// An integer: that many borrowed rows (range checked by the resolver).
    Rows(usize),
}

/// Parse [`EP_TAIL_BORROW_ENV`]; `Err` names a value that is none of `0`, `1` or a row count.
pub fn parse_ep_tail_borrow_knob(raw: Option<&str>) -> Result<EpTailBorrowKnob, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(EpTailBorrowKnob::Default);
    };
    match raw {
        "0" => Ok(EpTailBorrowKnob::Off),
        "1" => Ok(EpTailBorrowKnob::On),
        other => other
            .parse::<usize>()
            .map(EpTailBorrowKnob::Rows)
            .map_err(|_| format!("{EP_TAIL_BORROW_ENV}={other}: expected 0, 1 or a row count")),
    }
}

/// The facts [`resolve_ep_tail_borrow`] judges. Every field is a plain fact the caller reads
/// (the loader from its own plan, the bundle from its weights and the device).
#[derive(Clone, Copy, Debug)]
pub struct EpTailBorrowFacts {
    /// The load is expert-parallel (a placement is bound).
    pub ep: bool,
    /// The device is exactly gfx1201.
    pub gfx1201: bool,
    /// The forward scratch plan of the load.
    pub mode: Qwen4ForwardScratchMode,
    /// The D2 expert tail is configured WITH its prefill lever over a per-expert unit plan.
    pub tail_prefill: bool,
    /// Rows of the resident bank: today's EP chunk.
    pub resident_rows: usize,
    /// The admitted context (the prompt chunk never exceeds it).
    pub max_seq: usize,
    /// `HIPFIRE_PREFILL_CHUNK_ROWS` pins the chunk.
    pub explicit_chunk: bool,
    /// A lever this one never combines with (EP reserve decoupling, the chunk auto-fit,
    /// decode-share), as its name; `None` when none is active.
    pub conflict: Option<&'static str>,
}

/// The rows of a planned borrow: the load's prompt chunk is `borrowed_rows`, the forward's
/// resident scratch bank holds `resident_rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpTailBorrowPlan {
    pub borrowed_rows: usize,
    pub resident_rows: usize,
}

/// [`resolve_ep_tail_borrow`]'s verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpTailBorrowResolution {
    /// `Some` when the borrow is planned.
    pub plan: Option<EpTailBorrowPlan>,
    /// A line to print: the operator set the knob and it is ignored (or invalid). Never set
    /// for the unset default or `0`, so the shipped default stays silent on a load it does not
    /// apply to.
    pub note: Option<String>,
}

/// The pure planning decision: the knob (`raw`, with `default_on` standing for the unset
/// value) against the load's `facts`. The loader publishes the plan before any rank sizes a
/// placement or a forward ([`qwen4_set_ep_tail_borrow_plan`]); the bundle reads it back.
pub fn resolve_ep_tail_borrow(
    raw: Option<&str>,
    default_on: bool,
    facts: &EpTailBorrowFacts,
) -> EpTailBorrowResolution {
    let none = |note: Option<String>| EpTailBorrowResolution { plan: None, note };
    let knob = match parse_ep_tail_borrow_knob(raw) {
        Ok(knob) => knob,
        Err(error) => return none(Some(format!("{error}; the EP tail borrow stays off"))),
    };
    let explicit = !matches!(knob, EpTailBorrowKnob::Default);
    let rows = match knob {
        EpTailBorrowKnob::Off => return none(None),
        EpTailBorrowKnob::Default if !default_on => return none(None),
        EpTailBorrowKnob::Default | EpTailBorrowKnob::On => EP_TAIL_BORROW_DEFAULT_ROWS,
        EpTailBorrowKnob::Rows(rows) => rows,
    };
    let refuse = |reason: String| {
        let note = explicit.then(|| format!("qwen4 EP tail borrow: ignored ({reason})"));
        none(note)
    };
    if !facts.ep {
        return refuse("needs an expert-parallel load".into());
    }
    if !facts.gfx1201 {
        return refuse("needs gfx1201".into());
    }
    if facts.mode != Qwen4ForwardScratchMode::FnSymGfx1201 {
        return refuse("needs the compact FnSymGfx1201 scratch plan".into());
    }
    if !facts.tail_prefill {
        return refuse(
            "needs the D2 expert tail with its prefill lever over a per-expert unit plan".into(),
        );
    }
    if facts.explicit_chunk {
        return refuse(format!(
            "{QWEN4_PREFILL_CHUNK_ROWS_ENV} pins the chunk"
        ));
    }
    if let Some(conflict) = facts.conflict {
        return refuse(format!("it does not combine with {conflict}"));
    }
    if rows <= facts.resident_rows || rows % 256 != 0 {
        return refuse(format!(
            "{rows} borrowed rows must be a multiple of 256 above the {} resident rows",
            facts.resident_rows
        ));
    }
    if facts.max_seq < rows {
        return refuse(format!(
            "the {}-token context is shorter than the {rows}-row borrowed tile",
            facts.max_seq
        ));
    }
    EpTailBorrowResolution {
        plan: Some(EpTailBorrowPlan {
            borrowed_rows: rows,
            resident_rows: facts.resident_rows,
        }),
        note: None,
    }
}

/// The plan the current load published (`None`: no borrow). Like
/// [`qwen4_set_ep_auto_prefill_chunk`], one expert-parallel load decides it once, before any
/// rank allocates; every Qwen4 load clears it before planning.
static EP_TAIL_BORROW_PLAN: std::sync::Mutex<Option<EpTailBorrowPlan>> =
    std::sync::Mutex::new(None);

/// Publish (`Some`) or clear (`None`) the load's tail-borrow plan.
pub fn qwen4_set_ep_tail_borrow_plan(plan: Option<EpTailBorrowPlan>) {
    *EP_TAIL_BORROW_PLAN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = plan;
}

/// The published tail-borrow plan.
pub fn qwen4_ep_tail_borrow_plan() -> Option<EpTailBorrowPlan> {
    *EP_TAIL_BORROW_PLAN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What the bundle's forward attach does with the published plan: the plan when this attach
/// can honour it, else the reason it cannot (the forward then builds as it would without the
/// knob; the loader's charge was larger, never smaller). `requested` is the chunk the attach
/// resolved (the published borrowed rows unless the PLE store, the context or a slow-link tile
/// narrowed it).
pub(crate) fn ep_tail_borrow_attach_plan(
    published: Option<EpTailBorrowPlan>,
    ep: bool,
    mode: Qwen4ForwardScratchMode,
    requested: usize,
    excluded: Option<&'static str>,
) -> Result<Option<EpTailBorrowPlan>, String> {
    let Some(plan) = published else {
        return Ok(None);
    };
    if !ep {
        return Err("the load is not expert-parallel".into());
    }
    if mode != Qwen4ForwardScratchMode::FnSymGfx1201 {
        return Err(format!("the forward scratch plan is {mode:?}"));
    }
    if let Some(what) = excluded {
        return Err(format!("the load carries {what}"));
    }
    if requested != plan.borrowed_rows {
        return Err(format!(
            "the attach resolved a {requested}-row chunk, the plan borrows {} rows",
            plan.borrowed_rows
        ));
    }
    Ok(Some(plan))
}

// ---------------------------------------------------------------------------
// Charge and exchange capacity (pure over the scratch plans)
// ---------------------------------------------------------------------------

/// The roots a tail-borrow load keeps OUT of the pool, persistent across both banks, in charge
/// order: `TokenIds` always (`uploaded_token_ids` outlives a forward) and `MoeYDownGrouped` when
/// the plan owns it as the tile exchange's grouped-down root (a peer reads it,
/// [`super::qwen4_ep_dedicated_exchange_root`]). Under the trunk shard the grouped-down rows are
/// a phase-arena view no peer reads, so they borrow with the rest of the arena.
fn persistent_fields(slots: &[ScratchSlot]) -> Vec<ScratchField> {
    let owns = |field: ScratchField| {
        slots
            .get(field as usize)
            .is_some_and(|slot| slot.owner.is_none() && slot.elements != 0)
    };
    [ScratchField::TokenIds, ScratchField::MoeYDownGrouped]
        .into_iter()
        .filter(|&field| field == ScratchField::TokenIds || owns(field))
        .collect()
}

/// `(field, bytes)` of every persistent root of `slots` ([`persistent_fields`]); `None` unless
/// `TokenIds` is an owning non-empty root.
fn persistent_root_bytes(slots: &[ScratchSlot]) -> Option<Vec<(ScratchField, u64)>> {
    persistent_fields(slots)
        .into_iter()
        .map(|field| {
            let slot = slots.get(field as usize)?;
            if slot.owner.is_some() || slot.elements == 0 {
                return None;
            }
            Some((field, u64::try_from(slot.bytes().ok()?).ok()?))
        })
        .collect()
}

/// The device bytes a planned load charges for its forward: the resident bank's whole
/// forward (`resident_total`) with its persistent roots replaced by the borrowed-row ones
/// (the persistent allocations the resident bank views instead of owning). `None` unless both
/// plans keep the same persistent roots and the borrowed ones are not smaller.
fn charged_forward_bytes(
    resident_total: u64,
    resident: &[ScratchSlot],
    borrowed: &[ScratchSlot],
) -> Option<u64> {
    let resident = persistent_root_bytes(resident)?;
    let borrowed = persistent_root_bytes(borrowed)?;
    if resident.len() != borrowed.len() {
        return None;
    }
    let mut total = resident_total;
    for (&(field_r, bytes_r), &(field_b, bytes_b)) in resident.iter().zip(&borrowed) {
        if field_r != field_b || bytes_b < bytes_r {
            return None;
        }
        total = total.checked_sub(bytes_r)?.checked_add(bytes_b)?;
    }
    Some(total)
}

/// Device bytes a load planned for `plan` charges for its forward, in place of
/// [`qwen4_forward_device_bytes`]`(config, borrowed_rows, mode)`: the resident rows' forward
/// plus the growth of the two persistent roots (token ids and the grouped-down exchange root)
/// to the borrowed rows. The borrowed bank's other roots live in the D2 tail pool and cost no
/// VRAM. `None` on overflow or a plan that does not own both roots (the unit tests pin the
/// arithmetic against the plan's own allocations).
pub fn qwen4_ep_tail_borrow_forward_bytes(
    config: &Qwen4Config,
    plan: &EpTailBorrowPlan,
    mode: Qwen4ForwardScratchMode,
) -> Option<u64> {
    let scope = Qwen4BandScope::single(config);
    let (resident, _) =
        Qwen4GpuForwardScratch::plan_scoped(config, plan.resident_rows, mode, &scope).ok()?;
    let (borrowed, _) =
        Qwen4GpuForwardScratch::plan_scoped(config, plan.borrowed_rows, mode, &scope).ok()?;
    charged_forward_bytes(
        qwen4_forward_device_bytes(config, plan.resident_rows, mode)?,
        &resident,
        &borrowed,
    )
}

/// The tile-exchange capacity of the borrowed-row plan (what the persistent root and the
/// masked tile ids are registered for), like `EpReserveBanks::exchange_capacity`.
fn exchange_capacity_of(rows: usize, slots: &[ScratchSlot]) -> Option<EpExchangeCapacity> {
    let y_down = slots.get(ScratchField::MoeYDownGrouped as usize)?;
    let tile_ids_bytes = slots.get(ScratchField::MoeExpertTileIds as usize)?.bytes().ok()?;
    let tile_ids = tile_ids_bytes / std::mem::size_of::<i32>();
    Some(EpExchangeCapacity {
        rows,
        grouped_down_elements: y_down.elements,
        grouped_down_bytes: u64::try_from(y_down.bytes().ok()?).ok()?,
        tile_ids,
        grouped_rows: tile_ids.checked_mul(MOE_GROUPED_BLOCK_M)?,
    })
}

/// Where the owning roots of a borrowed bank sit inside the tail pool span: every owning
/// non-empty root of `slots` in [`ScratchField`] order, each start aligned to 256 bytes,
/// EXCEPT the persistent roots ([`persistent_fields`]: they are never donor memory). Like
/// [`Qwen4GpuForwardScratch::borrowed_backing`] otherwise.
fn tail_backing_of(rows: usize, slots: &[ScratchSlot]) -> Result<ScratchBacking, Qwen4GpuForwardError> {
    let overflow = || invalid("borrowed scratch backing overflows");
    let align = |x: u64| -> Result<u64, Qwen4GpuForwardError> {
        x.checked_add(DECODE_SHARE_ROOT_ALIGN - 1)
            .map(|x| x & !(DECODE_SHARE_ROOT_ALIGN - 1))
            .ok_or_else(overflow)
    };
    let persistent = persistent_fields(slots);
    let mut roots = Vec::with_capacity(slots.len());
    let mut cursor = 0u64;
    for (index, slot) in slots.iter().enumerate() {
        let field = ScratchField::ALL[index];
        if persistent.contains(&field) || slot.owner.is_some() {
            continue;
        }
        let bytes = u64::try_from(slot.bytes()?).map_err(|_| overflow())?;
        if bytes == 0 {
            continue;
        }
        let offset = align(cursor)?;
        cursor = offset.checked_add(bytes).ok_or_else(overflow)?;
        roots.push(ScratchRootPlacement {
            field,
            offset,
            bytes,
        });
    }
    Ok(ScratchBacking {
        capacity_rows: rows,
        roots,
        span_bytes: align(cursor)?,
    })
}

// ---------------------------------------------------------------------------
// Pure epoch / prompt state machine
// ---------------------------------------------------------------------------

/// What one forward call needs of the borrow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpTailCallPlan {
    /// Run on the resident bank, no epoch work (canonical tables make it correct inside an
    /// open epoch too).
    Resident,
    /// Run on the borrowed bank; `open_first`: the epoch is closed, open it before the call.
    Borrow { open_first: bool },
}

/// What the end of a whole-prompt bracket requires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpTailPromptEnd {
    /// An inner bracket: the outermost one finishes the prompt.
    Nested,
    /// Outermost, nothing open: nothing to do.
    Clean,
    /// Outermost, epoch open, the prompt (lane) continues in a later step: keep it open.
    Keep,
    /// Outermost, successful, nothing continues: close (restore) the epoch.
    Close,
    /// Outermost, failed, epoch open: close (recovery), then report the failure.
    Recover,
}

/// The rank-local epoch proof, prompt nesting and poison.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct EpTailEpoch {
    open: bool,
    depth: u32,
    epochs: u64,
    poisoned: Option<String>,
}

impl EpTailEpoch {
    /// The tail pool holds borrowed-bank bytes: tables canonical, policy suspended.
    pub(crate) fn open(&self) -> bool {
        self.open
    }

    /// Epochs opened so far (the count every barrier tag carries).
    pub(crate) fn epochs(&self) -> u64 {
        self.epochs
    }

    #[cfg(test)]
    pub(crate) fn depth(&self) -> u32 {
        self.depth
    }

    #[cfg(test)]
    pub(crate) fn poisoned(&self) -> Option<&str> {
        self.poisoned.as_deref()
    }

    /// `Err` with the load-fatal message once poisoned.
    pub(crate) fn check(&self) -> Result<(), String> {
        match self.poisoned.as_deref() {
            Some(reason) => Err(format!(
                "EP tail borrow poisoned ({reason}); reload the model"
            )),
            None => Ok(()),
        }
    }

    /// The plan of a call of `rows` rows: at most `resident_rows` runs on the resident bank;
    /// more runs on the borrowed bank (`capacity_rows`), only when `borrowable` (a prefill
    /// tile: not a device-argmax forward, not a mesh segment). A wider call that cannot
    /// borrow, or exceeds the capacity, is refused before anything is written. A poisoned
    /// borrow refuses every call. A zero-row call is left to the forward's own validation.
    pub(crate) fn plan_call(
        &self,
        rows: usize,
        resident_rows: usize,
        capacity_rows: usize,
        borrowable: bool,
    ) -> Result<EpTailCallPlan, String> {
        self.check()?;
        if rows <= resident_rows {
            return Ok(EpTailCallPlan::Resident);
        }
        if rows > capacity_rows {
            return Err(format!(
                "a {rows}-row call exceeds the {capacity_rows}-row borrowed prompt capacity"
            ));
        }
        if !borrowable {
            return Err(format!(
                "a {rows}-row call exceeds the {resident_rows}-row resident scratch bank and is \
                 not a prefill tile that may run on the borrowed EP tail bank (a device-argmax \
                 or mesh forward)"
            ));
        }
        Ok(EpTailCallPlan::Borrow {
            open_first: !self.open,
        })
    }

    /// The open sequence passed its barrier: the pool is about to hold borrowed-bank bytes.
    pub(crate) fn opened(&mut self) {
        self.open = true;
        self.epochs += 1;
    }

    /// The close sequence completed.
    pub(crate) fn closed(&mut self) {
        self.open = false;
    }

    /// Nothing may ever read the pool again: every later call refuses.
    pub(crate) fn poison(&mut self, reason: impl Into<String>) {
        self.open = false;
        self.poisoned.get_or_insert_with(|| reason.into());
    }

    /// A whole-prompt bracket begins; `true` when it is the outermost.
    pub(crate) fn begin_prompt(&mut self) -> bool {
        self.depth += 1;
        self.depth == 1
    }

    /// A whole-prompt bracket ends (`ok`: it returned `Ok`; `continues`: the same prompt goes
    /// on in a later bracket, a lane prompt split over several steps).
    pub(crate) fn end_prompt(&mut self, ok: bool, continues: bool) -> EpTailPromptEnd {
        self.depth = self.depth.saturating_sub(1);
        if self.depth > 0 {
            return EpTailPromptEnd::Nested;
        }
        match (self.open, ok, continues) {
            (false, _, _) => EpTailPromptEnd::Clean,
            (true, true, true) => EpTailPromptEnd::Keep,
            (true, true, false) => EpTailPromptEnd::Close,
            (true, false, _) => EpTailPromptEnd::Recover,
        }
    }

    /// Whether an epoch is open with no bracket around it (a continuation whose prompt never
    /// came back): the owner closes it.
    pub(crate) fn idle_open(&self) -> bool {
        self.open && self.depth == 0
    }
}

// ---------------------------------------------------------------------------
// The forward's borrow state
// ---------------------------------------------------------------------------

/// Counters of the borrow (tests, receipts).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[allow(dead_code)] // `epochs` and the counters are read by tests and receipts
pub(crate) struct EpTailBorrowStats {
    /// Epochs opened.
    pub epochs: u64,
    /// Tiles run on the borrowed bank.
    pub tiles: u64,
    /// Closes that completed (restores).
    pub restores: u64,
    /// Bytes `borrow_restore` copied, over every close.
    pub restore_bytes: u64,
    /// Milliseconds of `borrow_restore` (device drain included), over every close.
    pub restore_ms: f64,
    /// Milliseconds spent waiting at the all-rank barriers (open and close).
    pub barrier_wait_ms: f64,
    /// Digest of the last verified close (`HIPFIRE_QWEN4_EP_TAIL_BORROW_VERIFY=1`).
    pub last_digest: Option<u64>,
}

/// What the mesh attached.
struct EpTailAttached {
    tail: Arc<UnitTailRank>,
    /// The borrowed bank while the resident one is in `scratch`.
    bank: Option<Qwen4GpuForwardScratch>,
    /// The resident bank while the borrowed one is in `scratch`.
    parked: Option<Qwen4GpuForwardScratch>,
    /// Device base and bytes of the span the bank views (the tail pool).
    span: (usize, usize),
}

enum EpTailState {
    /// Planned at construction; the mesh has not decided yet.
    Planned,
    Attached(Box<EpTailAttached>),
    /// The attach refused it (reason kept): the load keeps its resident rows as the prompt
    /// chunk.
    Refused(String),
}

/// The owner of a planned EP tail borrow: the persistent roots, the attach state and the
/// epoch. One field of `Qwen4GpuForward`.
pub(crate) struct EpTailBorrow {
    pub(crate) plan: EpTailBorrowPlan,
    /// Persistent token ids (borrowed rows), owned here, viewed by both banks.
    ids: GpuTensor,
    /// Persistent grouped-down exchange root (borrowed rows), owned here, viewed by both banks
    /// and registered with the tile exchange; `None` when the plan keeps the grouped-down rows
    /// in its phase arena (the trunk shard: no peer reads them), so they borrow with the arena.
    exchange_root: Option<GpuTensor>,
    /// Tile-exchange capacity of the borrowed-row plan.
    exchange: EpExchangeCapacity,
    /// The D2 tail the install handed over, until the mesh attaches the bank.
    tail: Option<Arc<UnitTailRank>>,
    state: EpTailState,
    epoch: EpTailEpoch,
    stats: EpTailBorrowStats,
    verify: bool,
    log: bool,
    /// Per-epoch counters for the log line.
    epoch_tiles: u64,
    epoch_barrier_ms: f64,
}

impl EpTailBorrow {
    /// Allocate the persistent roots ([`persistent_fields`]) and build the resident bank over
    /// them. Returns the resident bank (the forward's `scratch`), the owner and the host staging
    /// vectors sized for the borrowed rows. Nothing is allocated on error.
    pub(crate) fn build(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        plan: EpTailBorrowPlan,
    ) -> Result<(Qwen4GpuForwardScratch, Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        let (borrowed_slots, borrowed_ple) =
            Qwen4GpuForwardScratch::plan_scoped(config, plan.borrowed_rows, mode, scope)?;
        let (resident_slots, resident_ple) =
            Qwen4GpuForwardScratch::plan_scoped(config, plan.resident_rows, mode, scope)?;
        check_aliases_inside_roots(plan.borrowed_rows, &borrowed_slots)?;
        check_aliases_inside_roots(plan.resident_rows, &resident_slots)?;
        let persistent = persistent_fields(&borrowed_slots);
        if persistent != persistent_fields(&resident_slots) {
            return Err(invalid(format!(
                "the {}-row and {}-row {mode:?} plans keep different persistent roots",
                plan.resident_rows, plan.borrowed_rows
            )));
        }
        for &field in &persistent {
            let (borrowed, resident) = (
                &borrowed_slots[field as usize],
                &resident_slots[field as usize],
            );
            if borrowed.owner.is_some() || borrowed.elements == 0 || resident.owner.is_some() {
                return Err(invalid(format!(
                    "the EP tail borrow's persistent {field:?} is not an owning root of the \
                     {mode:?} plan"
                )));
            }
            if borrowed.dtype != resident.dtype || resident.elements > borrowed.elements {
                return Err(invalid(format!(
                    "the resident {field:?} ({:?} x {}) does not fit inside the persistent \
                     borrowed-row root ({:?} x {})",
                    resident.dtype, resident.elements, borrowed.dtype, borrowed.elements
                )));
            }
        }
        let exchange = exchange_capacity_of(plan.borrowed_rows, &borrowed_slots)
            .ok_or_else(|| invalid("the borrowed-row exchange capacity overflows"))?;
        let ids_slot = &borrowed_slots[ScratchField::TokenIds as usize];
        let down_slot = &borrowed_slots[ScratchField::MoeYDownGrouped as usize];
        let ids = gpu.zeros(&[ids_slot.elements], ids_slot.dtype)?;
        let exchange_root = if persistent.contains(&ScratchField::MoeYDownGrouped) {
            match gpu.zeros(&[down_slot.elements], down_slot.dtype) {
                Ok(tensor) => Some(tensor),
                Err(error) => {
                    let _ = gpu.free_tensor(ids);
                    return Err(error.into());
                }
            }
        } else {
            None
        };
        let (resident, resident_token_bytes, resident_ple_bytes) =
            match Qwen4GpuForwardScratch::new_resident_tail_borrow(
                gpu,
                config,
                plan.resident_rows,
                mode,
                scope,
                &ids,
                exchange_root.as_ref(),
            ) {
                Ok(built) => built,
                Err(error) => {
                    if let Some(root) = exchange_root {
                        let _ = gpu.free_tensor(root);
                    }
                    let _ = gpu.free_tensor(ids);
                    return Err(error);
                }
            };
        // Host staging sized for the borrowed rows (the largest bank).
        let host_token_bytes = vec![0u8; plan.borrowed_rows * std::mem::size_of::<i32>()];
        debug_assert!(host_token_bytes.len() >= resident_token_bytes.len());
        let host_ple_bytes =
            vec![0u8; borrowed_ple.max(resident_ple).max(resident_ple_bytes.len())];
        let owner = Self {
            plan,
            ids,
            exchange_root,
            exchange,
            tail: None,
            state: EpTailState::Planned,
            epoch: EpTailEpoch::default(),
            stats: EpTailBorrowStats::default(),
            verify: hipfire_config::developer_bool(EP_TAIL_BORROW_VERIFY_ENV, false),
            log: hipfire_config::developer_bool(EP_TAIL_BORROW_LOG_ENV, false),
            epoch_tiles: 0,
            epoch_barrier_ms: 0.0,
        };
        Ok((resident, owner, host_token_bytes, host_ple_bytes))
    }

    /// The tile-exchange capacity of the borrowed-row plan: what `Qwen4TileRoute::admit`
    /// registers the exchange for.
    pub(crate) fn exchange_capacity(&self) -> EpExchangeCapacity {
        self.exchange
    }

    /// The prompt chunk of this load: the planned borrowed rows while the borrow is planned or
    /// attached, `None` once the attach refused it.
    pub(crate) fn prompt_rows(&self) -> Option<usize> {
        match self.state {
            EpTailState::Refused(_) => None,
            _ => Some(self.plan.borrowed_rows),
        }
    }

    fn attached(&self) -> bool {
        matches!(self.state, EpTailState::Attached(_))
    }

    fn state_note(&self) -> String {
        match &self.state {
            EpTailState::Planned => "planned but not attached yet".to_string(),
            EpTailState::Attached(_) => "attached".to_string(),
            EpTailState::Refused(reason) => format!("not attached ({reason})"),
        }
    }

    /// Free the persistent roots (the banks own nothing else of this borrow).
    pub(crate) fn release(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in self.exchange_root.into_iter().chain(std::iter::once(self.ids)) {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

// ---------------------------------------------------------------------------
// The banks
// ---------------------------------------------------------------------------

impl Qwen4GpuForwardScratch {
    /// The resident bank of a tail-borrow load: [`Self::new_scoped`] for `rows` rows except that
    /// `TokenIds` and (when the plan owns it, `exchange_root` is `Some`) `MoeYDownGrouped` are
    /// views of the persistent roots `ids` and `exchange_root` (owned by [`EpTailBorrow`]; the
    /// bank allocates and frees neither).
    fn new_resident_tail_borrow(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        ids: &GpuTensor,
        exchange_root: Option<&GpuTensor>,
    ) -> Result<(Self, Vec<u8>, Vec<u8>), Qwen4GpuForwardError> {
        let (slots, max_ple_bytes) = Self::plan_scoped(config, rows, mode, scope)?;
        Self::check_external_ids(&slots, ids)?;
        let persistent = persistent_fields(&slots);
        let down = &slots[ScratchField::MoeYDownGrouped as usize];
        match (persistent.contains(&ScratchField::MoeYDownGrouped), exchange_root) {
            (true, Some(root)) if root.dtype == down.dtype && root.numel() >= down.elements => {}
            (false, None) => {}
            _ => {
                return Err(invalid(format!(
                    "the persistent exchange root does not match the {rows}-row bank's \
                     grouped-down root ({:?} x {}, owned by the plan: {})",
                    down.dtype,
                    down.elements,
                    persistent.contains(&ScratchField::MoeYDownGrouped)
                )))
            }
        }
        let mut tensors: Vec<Option<GpuTensor>> = (0..slots.len()).map(|_| None).collect();
        for (index, slot) in slots.iter().enumerate() {
            let field = ScratchField::ALL[index];
            if slot.owner.is_some() || slot.elements == 0 {
                continue;
            }
            let created = match field {
                ScratchField::TokenIds => Ok(ids.sub_offset(0, ids.numel())),
                ScratchField::MoeYDownGrouped if exchange_root.is_some() => Ok(exchange_root
                    .expect("matched above")
                    .sub_offset(0, slot.elements)),
                _ => gpu.zeros(&[slot.elements], slot.dtype),
            };
            match created {
                Ok(tensor) => tensors[index] = Some(tensor),
                Err(error) => {
                    // Only the bank's own allocations are freed: the persistent views are
                    // aliases of roots the caller frees.
                    for (index, tensor) in tensors.into_iter().enumerate() {
                        if let Some(tensor) = tensor {
                            if !persistent.contains(&ScratchField::ALL[index]) {
                                let _ = gpu.free_tensor(tensor);
                            }
                        }
                    }
                    return Err(error.into());
                }
            }
        }
        Self::bind_views(&slots, &mut tensors);
        Ok(Self::from_tensors(
            rows,
            mode,
            None,
            ScratchBackingKind::ExternalRoots,
            slots,
            tensors,
            max_ple_bytes,
        ))
    }

    /// The borrowed bank of a tail-borrow load for `rows` rows: no allocation. Every owning
    /// root of the plan except the persistent ones is a non-owning tensor at
    /// `base + root.offset` of the tail pool span ([`tail_backing_of`] places them; the span is
    /// `span_bytes` long); the persistent roots are views of `ids` / `exchange_root`.
    /// [`Self::free_gpu`] frees nothing of it.
    #[allow(clippy::too_many_arguments)]
    fn new_tail_borrowed(
        config: &Qwen4Config,
        rows: usize,
        mode: Qwen4ForwardScratchMode,
        scope: &Qwen4BandScope,
        base: usize,
        span_bytes: usize,
        ids: &GpuTensor,
        exchange_root: Option<&GpuTensor>,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let (slots, max_ple_bytes) = Self::plan_scoped(config, rows, mode, scope)?;
        let backing = tail_backing_of(rows, &slots)?;
        if backing.span_bytes > span_bytes as u64 {
            return Err(invalid(format!(
                "a {rows}-row borrowed scratch bank needs {} bytes but its span has {span_bytes}",
                backing.span_bytes
            )));
        }
        if base % DECODE_SHARE_ROOT_ALIGN as usize != 0 {
            return Err(invalid(format!(
                "borrowed scratch span base {base:#x} is not {DECODE_SHARE_ROOT_ALIGN}-byte aligned"
            )));
        }
        Self::check_external_ids(&slots, ids)?;
        let owns_down = persistent_fields(&slots).contains(&ScratchField::MoeYDownGrouped);
        let down = &slots[ScratchField::MoeYDownGrouped as usize];
        let exact = match exchange_root {
            Some(root) => {
                owns_down
                    && root.dtype == down.dtype
                    && root.numel() == down.elements
                    && root.buf.size() == down.bytes()?
            }
            None => !owns_down,
        };
        if !exact {
            return Err(invalid(
                "the persistent exchange root is not exactly the borrowed bank's grouped-down root",
            ));
        }
        let mut tensors: Vec<Option<GpuTensor>> = (0..slots.len()).map(|_| None).collect();
        for root in &backing.roots {
            let index = root.field as usize;
            let slot = &slots[index];
            let address = usize::try_from(root.offset)
                .ok()
                .and_then(|offset| base.checked_add(offset))
                .ok_or_else(|| invalid("borrowed scratch root address overflows"))?;
            let bytes = usize::try_from(root.bytes)
                .map_err(|_| invalid("borrowed scratch root bytes overflow"))?;
            // SAFETY: `root.offset + root.bytes <= backing.span_bytes <= span_bytes`, and the
            // caller guarantees `[base, base + span_bytes)` is live device memory (the D2 tail
            // pool, released only after the forward's teardown closed the epoch) for the
            // bank's life. The tensor is a `Borrowed` buffer: no free path ever receives it
            // (`ScratchBackingKind::Borrowed` frees nothing).
            let buf = unsafe {
                hip_bridge::DeviceBuffer::from_raw(address as *mut std::ffi::c_void, bytes)
            };
            tensors[index] = Some(GpuTensor {
                buf,
                shape: vec![slot.elements],
                dtype: slot.dtype,
            });
        }
        tensors[ScratchField::TokenIds as usize] =
            Some(ids.sub_offset(0, slots[ScratchField::TokenIds as usize].elements));
        if let Some(root) = exchange_root {
            tensors[ScratchField::MoeYDownGrouped as usize] =
                Some(root.sub_offset(0, root.numel()));
        }
        for (index, slot) in slots.iter().enumerate() {
            if slot.owner.is_none() && slot.elements != 0 && tensors[index].is_none() {
                return Err(invalid(format!(
                    "borrowed scratch backing places no root for {:?}",
                    ScratchField::ALL[index]
                )));
            }
        }
        Self::bind_views(&slots, &mut tensors);
        let (bank, _, _) = Self::from_tensors(
            rows,
            mode,
            None,
            ScratchBackingKind::Borrowed,
            slots,
            tensors,
            max_ple_bytes,
        );
        Ok(bank)
    }
}

// ---------------------------------------------------------------------------
// Forward integration
// ---------------------------------------------------------------------------

impl Qwen4GpuForward {
    /// The D2 tail the install handed over (called by the EP attach right after the tail is
    /// installed with its prefill lever); a no-op on a load without a planned borrow.
    pub(crate) fn ep_tail_borrow_take_tail(&mut self, tail: Arc<UnitTailRank>) {
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            borrow.tail = Some(tail);
        }
    }

    /// The mesh's attach of a planned borrow, once the tile exchange hook is installed (or
    /// refused): admit it on this rank and build the borrowed bank over the tail pool span.
    /// `Ok(true)`: attached; `Ok(false)`: refused (state kept, reason logged once, the load
    /// keeps its resident rows as the prompt chunk); `Err`: a failure the load must report.
    /// A no-op (`Ok(false)`) on a load without a planned borrow. Ranks must agree on the
    /// verdict ([`Self::ep_tail_borrow_refuse`] undoes an attach the others refused).
    pub(crate) fn attach_ep_tail_borrow(
        &mut self,
        config: &Qwen4Config,
        scope: &Qwen4BandScope,
    ) -> Result<bool, String> {
        let rank = self.ep.as_ref().map_or(0, |ep| ep.placement.rank);
        let has_hook = self
            .ep
            .as_ref()
            .and_then(|ep| ep.tile.as_ref())
            .is_some_and(|route| route.hook.is_some());
        let no_reserve = self.ep_reserve.is_none();
        let no_share = self.share.is_none();
        let mode = self.scratch.mode;
        let resident_rows = self.scratch.max_chunk;
        let Some(borrow) = self.ep_tail_borrow.as_mut() else {
            return Ok(false);
        };
        if !matches!(borrow.state, EpTailState::Planned) {
            return Ok(borrow.attached());
        }
        let refuse = |borrow: &mut EpTailBorrow, reason: String| -> Result<bool, String> {
            eprintln!(
                "qwen4 EP rank {rank}: tail borrow off: {reason}; prompt chunk {resident_rows} rows"
            );
            borrow.state = EpTailState::Refused(reason);
            borrow.tail = None;
            Ok(false)
        };
        let Some(tail) = borrow.tail.clone() else {
            return refuse(
                borrow,
                "no D2 tail with its prefill lever was installed on this rank".to_string(),
            );
        };
        if !has_hook {
            return refuse(
                borrow,
                "the owner-routed tile exchange is not installed (no host barrier)".to_string(),
            );
        }
        if !no_reserve || !no_share {
            return refuse(
                borrow,
                "the load carries an EP reserve or decode-share".to_string(),
            );
        }
        let (rows, plan_resident) = (borrow.plan.borrowed_rows, borrow.plan.resident_rows);
        if resident_rows != plan_resident {
            return refuse(
                borrow,
                format!(
                    "the resident bank holds {resident_rows} rows, the plan expects {plan_resident}"
                ),
            );
        }
        let Some((base, span)) = tail.pool_span() else {
            return refuse(borrow, "the D2 tail pool is released".to_string());
        };
        let (slots, _) = Qwen4GpuForwardScratch::plan_scoped(config, rows, mode, scope)
            .map_err(|error| error.to_string())?;
        let needed = tail_backing_of(rows, &slots)
            .map_err(|error| error.to_string())?
            .span_bytes;
        if (span as u64) < needed {
            return refuse(
                borrow,
                format!(
                    "the D2 tail pool holds {} MiB, a {rows}-row bank needs {} MiB",
                    span >> 20,
                    needed >> 20
                ),
            );
        }
        if base % DECODE_SHARE_ROOT_ALIGN as usize != 0 {
            return refuse(
                borrow,
                format!("the D2 tail pool base {base:#x} is not 256-byte aligned"),
            );
        }
        let bank = Qwen4GpuForwardScratch::new_tail_borrowed(
            config,
            rows,
            mode,
            scope,
            base,
            span,
            &borrow.ids,
            borrow.exchange_root.as_ref(),
        )
        .map_err(|error| error.to_string())?;
        eprintln!(
            "qwen4 EP rank {rank}: tail borrow on: {rows}-row prefill bank ({} MiB) in the {} MiB \
             D2 tail pool, resident bank {resident_rows} rows",
            needed >> 20,
            span >> 20
        );
        borrow.state = EpTailState::Attached(Box::new(EpTailAttached {
            tail,
            bank: Some(bank),
            parked: None,
            span: (base, span),
        }));
        Ok(true)
    }

    /// Undo [`Self::attach_ep_tail_borrow`] because another rank refused (ranks must agree on
    /// the verdict, or their prompt tiles diverge): the borrowed bank is dropped (it owns
    /// nothing) and the load keeps its resident rows as the prompt chunk.
    pub(crate) fn ep_tail_borrow_refuse(&mut self, reason: &str) {
        let rank = self.ep.as_ref().map_or(0, |ep| ep.placement.rank);
        let rows = self.scratch.max_chunk;
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            if !matches!(borrow.state, EpTailState::Refused(_)) {
                eprintln!("qwen4 EP rank {rank}: tail borrow off: {reason}; prompt chunk {rows} rows");
                borrow.state = EpTailState::Refused(reason.to_string());
                borrow.tail = None;
            }
        }
    }

    /// `Err` once the borrow is poisoned: every forward, on any route, refuses.
    pub(crate) fn ep_tail_borrow_check(&self) -> Result<(), Qwen4GpuForwardError> {
        match self.ep_tail_borrow.as_ref() {
            Some(borrow) => borrow.epoch.check().map_err(invalid),
            None => Ok(()),
        }
    }

    /// The borrow's counters (`None` without a planned borrow).
    #[allow(dead_code)] // tests and receipts
    pub(crate) fn ep_tail_borrow_stats(&self) -> Option<EpTailBorrowStats> {
        self.ep_tail_borrow.as_ref().map(|borrow| {
            let mut stats = borrow.stats;
            stats.epochs = borrow.epoch.epochs();
            stats
        })
    }

    /// Whether the borrowed bank is attached.
    #[allow(dead_code)] // tests and receipts
    pub(crate) fn ep_tail_borrow_attached(&self) -> bool {
        self.ep_tail_borrow
            .as_ref()
            .is_some_and(EpTailBorrow::attached)
    }

    /// Whether an epoch is open (the tail pool holds borrowed-bank bytes).
    #[allow(dead_code)] // tests and receipts
    pub(crate) fn ep_tail_borrow_epoch_open(&self) -> bool {
        self.ep_tail_borrow
            .as_ref()
            .is_some_and(|borrow| borrow.epoch.open())
    }

    /// The installed tile hook's all-rank host barrier of kind `what`, tagged with the epoch
    /// count so a rank-local divergence is named instead of timing out.
    fn ep_tail_barrier(&self, what: u8) -> Result<(), Qwen4GpuForwardError> {
        let hook = self
            .ep
            .as_ref()
            .and_then(|ep| ep.tile.as_ref())
            .and_then(|route| route.hook.as_ref())
            .ok_or_else(|| invalid("the EP tail borrow rendezvous needs the installed tile exchange"))?;
        let epoch = self
            .ep_tail_borrow
            .as_ref()
            .map_or(0, |borrow| borrow.epoch.epochs() as u32);
        hook.host_barrier(what, epoch)
            .map_err(|error| invalid(format!("EP tail borrow barrier: {error}")))
    }

    fn ep_tail_capturing(gpu: &Gpu) -> Result<bool, Qwen4GpuForwardError> {
        Ok(match gpu.active_stream.as_ref() {
            Some(stream) => gpu.hip.stream_is_capturing(stream)?,
            None => false,
        })
    }

    fn ep_tail_poison(&mut self, reason: String) -> Qwen4GpuForwardError {
        let rank = self.ep.as_ref().map_or(0, |ep| ep.placement.rank);
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            eprintln!("qwen4 EP rank {rank}: tail borrow poisoned: {reason}");
            borrow.epoch.poison(reason.clone());
        }
        invalid(format!(
            "EP tail borrow poisoned ({reason}); reload the model"
        ))
    }

    /// `forward_chunk_inner` on a load with a planned EP tail borrow: the bank switch.
    ///
    /// Rows at most the resident bank's run on it with no epoch work. More rows are a prefill
    /// tile (`prefill_scope`, not a device-argmax forward, not a mesh segment; a lane prefill
    /// tile under `lane_scope` borrows exactly as a singleton tile does) that runs on the
    /// borrowed bank inside the epoch: opened at the first such tile, kept across later tiles
    /// and closed at the outermost prompt end ([`Self::ep_tail_prompt_end`]). A wider call
    /// that cannot borrow, or any call while the borrow is unattached, fails before anything
    /// is written. The resident bank is the one in `scratch` at every call boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ep_tail_forward_chunk(
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
        let borrowable =
            self.prefill_scope && argmax_of.is_none() && self.mesh_route_rows.is_none();
        let plan = {
            let borrow = self
                .ep_tail_borrow
                .as_ref()
                .expect("an EP tail borrow forward");
            borrow.epoch.check().map_err(invalid)?;
            if borrow.attached() {
                borrow
                    .epoch
                    .plan_call(
                        n,
                        borrow.plan.resident_rows,
                        borrow.plan.borrowed_rows,
                        borrowable,
                    )
                    .map_err(invalid)?
            } else if n > borrow.plan.resident_rows {
                return Err(invalid(format!(
                    "a {n}-row call exceeds the {}-row resident scratch bank and the EP tail \
                     borrow is {}",
                    borrow.plan.resident_rows,
                    borrow.state_note()
                )));
            } else {
                EpTailCallPlan::Resident
            }
        };
        let open_first = match plan {
            EpTailCallPlan::Resident => {
                return self.forward_chunk_tile(
                    bundle,
                    gpu,
                    tokens,
                    argmax_of,
                    logits,
                    top1,
                    wide_hidden_capture,
                    output_policy,
                );
            }
            EpTailCallPlan::Borrow { open_first } => open_first,
        };
        if open_first {
            self.ep_tail_open(gpu)?;
        }
        self.ep_tail_swap_in()?;
        let result = self.forward_chunk_tile(
            bundle,
            gpu,
            tokens,
            argmax_of,
            logits,
            top1,
            wide_hidden_capture,
            output_policy,
        );
        let swapped = self.ep_tail_swap_out();
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            borrow.stats.tiles += 1;
            borrow.epoch_tiles += 1;
        }
        let merged = merge_share_result(result, swapped);
        if merged.is_err() {
            // A failed tile leaves borrowed-bank bytes in the pool: close (blocking) before
            // reporting; a close that fails poisons the borrow.
            let closed = self.ep_tail_close(gpu, "a failed borrowed tile");
            return merge_share_result(merged, closed);
        }
        merged
    }

    fn ep_tail_swap_in(&mut self) -> Result<(), Qwen4GpuForwardError> {
        let borrow = self
            .ep_tail_borrow
            .as_mut()
            .expect("an EP tail borrow forward");
        let EpTailState::Attached(attached) = &mut borrow.state else {
            return Err(invalid("the EP tail borrow is not attached"));
        };
        let bank = attached
            .bank
            .take()
            .ok_or_else(|| invalid("the borrowed scratch bank is already swapped in"))?;
        attached.parked = Some(std::mem::replace(&mut self.scratch, bank));
        Ok(())
    }

    /// Swap the resident bank back into `scratch`; a no-op when it already is.
    fn ep_tail_swap_out(&mut self) -> Result<(), Qwen4GpuForwardError> {
        let borrow = self
            .ep_tail_borrow
            .as_mut()
            .expect("an EP tail borrow forward");
        let EpTailState::Attached(attached) = &mut borrow.state else {
            return Ok(());
        };
        if let Some(resident) = attached.parked.take() {
            attached.bank = Some(std::mem::replace(&mut self.scratch, resident));
        }
        Ok(())
    }

    /// Open the epoch (acquire): refuse under capture, drain the device, suspend the tail
    /// (canonical tables on this rank), cross the all-rank barrier, then zero the span (donor
    /// bytes are arbitrary; padded rows must read 0). A failure after the suspend poisons the
    /// borrow (the pool state is unproven); a refusal before it changes nothing.
    fn ep_tail_open(&mut self, gpu: &mut Gpu) -> Result<(), Qwen4GpuForwardError> {
        if Self::ep_tail_capturing(gpu)? {
            return Err(invalid(
                "an EP tail borrowed tile cannot run inside a stream capture (its scratch \
                 overwrites the D2 tail pool, which needs a restore)",
            ));
        }
        gpu.hip.device_synchronize()?;
        let (tail, span) = {
            let borrow = self
                .ep_tail_borrow
                .as_ref()
                .expect("an EP tail borrow forward");
            match &borrow.state {
                EpTailState::Attached(attached) => (attached.tail.clone(), attached.span),
                _ => return Err(invalid("the EP tail borrow is not attached")),
            }
        };
        if let Err(error) = tail.borrow_suspend(gpu) {
            return Err(self.ep_tail_poison(format!("suspend before a borrowed tile: {error}")));
        }
        let wait = Instant::now();
        if let Err(error) = self.ep_tail_barrier(HOST_TAIL_BORROW_ACQUIRE) {
            return Err(self.ep_tail_poison(format!("acquire rendezvous: {error}")));
        }
        let waited = ms(wait.elapsed());
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            borrow.epoch.opened();
            borrow.stats.barrier_wait_ms += waited;
            borrow.epoch_tiles = 0;
            borrow.epoch_barrier_ms = waited;
        }
        // Every rank crossed: no rank reads any rank's pool until the close. Zero the span.
        // SAFETY: `span` is the live tail pool the attach validated; the epoch is open, so
        // nothing reads it.
        let zeroed = unsafe {
            let buf = hip_bridge::DeviceBuffer::from_raw(span.0 as *mut std::ffi::c_void, span.1);
            match gpu.active_stream.as_ref() {
                Some(stream) => gpu.hip.memset_async(&buf, 0, span.1, stream),
                None => gpu.hip.memset(&buf, 0, span.1),
            }
        };
        if let Err(error) = zeroed {
            let closed = self.ep_tail_close(gpu, "a failed span zero-fill");
            return merge_share_result(Err(error.into()), closed);
        }
        Ok(())
    }

    /// Close the epoch (restore): drain, re-copy every occupied slot from its canonical host
    /// bytes, optionally verify them, cross the all-rank barrier, then resume the tail. Any
    /// failure poisons the borrow (nothing may ever read the pool again, and the pool is never
    /// republished on scratch bytes). A no-op when no epoch is open.
    fn ep_tail_close(&mut self, gpu: &mut Gpu, why: &str) -> Result<(), Qwen4GpuForwardError> {
        let (tail, verify, log, epoch, tiles, open_wait) = {
            let Some(borrow) = self.ep_tail_borrow.as_ref() else {
                return Ok(());
            };
            if !borrow.epoch.open() {
                return Ok(());
            }
            let EpTailState::Attached(attached) = &borrow.state else {
                return Ok(());
            };
            (
                attached.tail.clone(),
                borrow.verify,
                borrow.log,
                borrow.epoch.epochs(),
                borrow.epoch_tiles,
                borrow.epoch_barrier_ms,
            )
        };
        let rank = self.ep.as_ref().map_or(0, |ep| ep.placement.rank);
        let started = Instant::now();
        let (slots, _) = tail.borrow_occupied();
        let outcome: Result<(u64, f64, f64, Option<u64>), Qwen4GpuForwardError> = (|| {
            gpu.hip.device_synchronize()?;
            let bytes = tail.borrow_restore(gpu).map_err(invalid)?;
            let restore_ms = ms(started.elapsed());
            let digest = if verify {
                Some(tail.borrow_verify(gpu).map_err(invalid)?)
            } else {
                None
            };
            let wait = Instant::now();
            self.ep_tail_barrier(HOST_TAIL_BORROW_RESTORE)?;
            let barrier_ms = ms(wait.elapsed());
            tail.borrow_resume(gpu).map_err(invalid)?;
            Ok((bytes, restore_ms, barrier_ms, digest))
        })();
        match outcome {
            Ok((bytes, restore_ms, barrier_ms, digest)) => {
                let borrow = self
                    .ep_tail_borrow
                    .as_mut()
                    .expect("an EP tail borrow forward");
                borrow.epoch.closed();
                borrow.stats.restores += 1;
                borrow.stats.restore_bytes += bytes;
                borrow.stats.restore_ms += restore_ms;
                borrow.stats.barrier_wait_ms += barrier_ms;
                if digest.is_some() {
                    borrow.stats.last_digest = digest;
                }
                if log {
                    eprintln!(
                        "qwen4 EP rank {rank}: tail borrow epoch {epoch}: {tiles} tiles, \
                         {slots} occupied slots, restore {bytes} B in {restore_ms:.1} ms, \
                         barrier wait {:.1} ms{} ({why})",
                        open_wait + barrier_ms,
                        digest.map_or(String::new(), |digest| format!(", digest {digest:016x}"))
                    );
                }
                Ok(())
            }
            Err(error) => Err(self.ep_tail_poison(format!("close after {why}: {error}"))),
        }
    }

    /// A whole-prompt bracket begins on this rank (no-op without a planned borrow).
    pub(super) fn ep_tail_prompt_begin(&mut self) {
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            borrow.epoch.begin_prompt();
        }
    }

    /// A whole-prompt bracket ends (`ok`: it returned `Ok`; `continues`: the same prompt goes
    /// on in a later bracket). The outermost end of a successful prompt with nothing
    /// continuing closes an open epoch (one restore per long prompt); a continuing one keeps
    /// it open; a failed one closes it (recovery) before the caller reports its failure.
    pub(super) fn ep_tail_prompt_end(
        &mut self,
        gpu: &mut Gpu,
        ok: bool,
        continues: bool,
    ) -> Result<(), Qwen4GpuForwardError> {
        let end = match self.ep_tail_borrow.as_mut() {
            Some(borrow) => borrow.epoch.end_prompt(ok, continues),
            None => return Ok(()),
        };
        match end {
            EpTailPromptEnd::Nested | EpTailPromptEnd::Clean | EpTailPromptEnd::Keep => Ok(()),
            EpTailPromptEnd::Close => self.ep_tail_close(gpu, "prompt completion"),
            EpTailPromptEnd::Recover => self.ep_tail_close(gpu, "a failed prompt"),
        }
    }

    /// Close an epoch left open outside every bracket: a lane prompt that continued but never
    /// came back (cancelled, released). Called by the lane executor on a step with no prefill
    /// tile and no lane left prefilling; ranks decide it identically (lockstep lane stores).
    pub(crate) fn ep_tail_idle_close(&mut self, gpu: &mut Gpu) -> Result<(), Qwen4GpuForwardError> {
        let idle = self
            .ep_tail_borrow
            .as_ref()
            .is_some_and(|borrow| borrow.epoch.idle_open());
        if idle {
            self.ep_tail_close(gpu, "an idle lane step")
        } else {
            Ok(())
        }
    }

    /// Teardown (before the tail is released): put the resident bank back, drain the device
    /// and restore an open epoch locally. No barrier: the ranks tear down independently, and a
    /// rank that already left would hang the rendezvous; the pool is whole after the local
    /// restore, so resuming the tail is safe whatever the peers do, and nothing reads it
    /// afterwards. The first error is returned (the teardown continues).
    pub(super) fn ep_tail_teardown(&mut self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let _ = self.ep_tail_swap_out();
        if let Some(borrow) = self.ep_tail_borrow.as_ref().filter(|borrow| borrow.log) {
            let (rank, stats) = (
                self.ep.as_ref().map_or(0, |ep| ep.placement.rank),
                borrow.stats,
            );
            eprintln!(
                "qwen4 EP rank {rank}: tail borrow: {} epochs, {} borrowed tiles, {} restores \
                 ({} B, {:.1} ms), barrier wait {:.1} ms{}",
                borrow.epoch.epochs(),
                stats.tiles,
                stats.restores,
                stats.restore_bytes,
                stats.restore_ms,
                stats.barrier_wait_ms,
                stats
                    .last_digest
                    .map_or(String::new(), |digest| format!(", last digest {digest:016x}"))
            );
        }
        let (tail, open) = {
            let borrow = self.ep_tail_borrow.as_ref()?;
            let EpTailState::Attached(attached) = &borrow.state else {
                return None;
            };
            (attached.tail.clone(), borrow.epoch.open())
        };
        if !open {
            return None;
        }
        let mut first = None;
        if let Err(error) = gpu.hip.device_synchronize() {
            first = Some(error);
        }
        if first.is_none() {
            let restored = tail
                .borrow_restore(gpu)
                .and_then(|_| tail.borrow_resume(gpu));
            if let Err(error) = restored {
                first = Some(hip_bridge::HipError::new(
                    999,
                    &format!("EP tail borrow teardown restore failed: {error}"),
                ));
            }
        }
        if let Some(borrow) = self.ep_tail_borrow.as_mut() {
            borrow.epoch.closed();
        }
        first
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;

    const RESIDENT: usize = 4096;
    const BORROWED: usize = 8192;

    fn facts() -> EpTailBorrowFacts {
        EpTailBorrowFacts {
            ep: true,
            gfx1201: true,
            mode: Qwen4ForwardScratchMode::FnSymGfx1201,
            tail_prefill: true,
            resident_rows: RESIDENT,
            max_seq: 65_536,
            explicit_chunk: false,
            conflict: None,
        }
    }

    fn plan(rows: usize) -> Option<EpTailBorrowPlan> {
        Some(EpTailBorrowPlan {
            borrowed_rows: rows,
            resident_rows: RESIDENT,
        })
    }

    #[test]
    fn the_shipped_default_is_off_and_one_constant_flips_it() {
        // Unset with the shipped constant: whatever the facts, no plan, no note.
        assert!(!EP_TAIL_BORROW_DEFAULT_ON);
        let off = resolve_ep_tail_borrow(None, EP_TAIL_BORROW_DEFAULT_ON, &facts());
        assert_eq!(off, EpTailBorrowResolution { plan: None, note: None });
        // The same resolver with the default flipped plans the 8192-row borrow.
        let on = resolve_ep_tail_borrow(None, true, &facts());
        assert_eq!(on.plan, plan(BORROWED));
        assert_eq!(on.note, None);
        // An empty value is unset.
        assert_eq!(resolve_ep_tail_borrow(Some("  "), false, &facts()).plan, None);
    }

    #[test]
    fn the_knob_values() {
        let resolve = |raw: &str| resolve_ep_tail_borrow(Some(raw), false, &facts());
        assert_eq!(resolve("0").plan, None);
        assert_eq!(resolve("0").note, None);
        assert_eq!(resolve("1").plan, plan(BORROWED));
        assert_eq!(resolve("12288").plan, plan(12288));
        // `0` beats a flipped default; `1` beats the shipped one.
        assert_eq!(resolve_ep_tail_borrow(Some("0"), true, &facts()).plan, None);
        // Rows must exceed the resident bank, be tile multiples and fit the context.
        for bad in ["4096", "256", "8200"] {
            let verdict = resolve(bad);
            assert_eq!(verdict.plan, None, "{bad}");
            assert!(verdict.note.unwrap().contains("multiple of 256"), "{bad}");
        }
        let short = resolve_ep_tail_borrow(
            Some("1"),
            false,
            &EpTailBorrowFacts {
                max_seq: 6000,
                ..facts()
            },
        );
        assert_eq!(short.plan, None);
        assert!(short.note.unwrap().contains("6000-token context"));
        // A value that is no number is named, the borrow stays off.
        let junk = resolve("yes");
        assert_eq!(junk.plan, None);
        assert!(junk.note.unwrap().contains("expected 0, 1 or a row count"));
        assert_eq!(parse_ep_tail_borrow_knob(None), Ok(EpTailBorrowKnob::Default));
        assert_eq!(parse_ep_tail_borrow_knob(Some("1")), Ok(EpTailBorrowKnob::On));
        assert_eq!(
            parse_ep_tail_borrow_knob(Some(" 8192 ")),
            Ok(EpTailBorrowKnob::Rows(8192))
        );
    }

    #[test]
    fn every_refusal_is_named_and_a_default_stays_silent() {
        let cases: [(EpTailBorrowFacts, &str); 7] = [
            (EpTailBorrowFacts { ep: false, ..facts() }, "expert-parallel"),
            (
                EpTailBorrowFacts {
                    gfx1201: false,
                    ..facts()
                },
                "gfx1201",
            ),
            (
                EpTailBorrowFacts {
                    mode: Qwen4ForwardScratchMode::Full,
                    ..facts()
                },
                "compact",
            ),
            (
                EpTailBorrowFacts {
                    tail_prefill: false,
                    ..facts()
                },
                "prefill lever",
            ),
            (
                EpTailBorrowFacts {
                    explicit_chunk: true,
                    ..facts()
                },
                "HIPFIRE_PREFILL_CHUNK_ROWS",
            ),
            (
                EpTailBorrowFacts {
                    conflict: Some("the EP reserve"),
                    ..facts()
                },
                "the EP reserve",
            ),
            (
                EpTailBorrowFacts {
                    resident_rows: BORROWED,
                    ..facts()
                },
                "resident rows",
            ),
        ];
        for (facts, needle) in cases {
            let explicit = resolve_ep_tail_borrow(Some("1"), false, &facts);
            assert_eq!(explicit.plan, None, "{needle}");
            let note = explicit.note.expect("an explicit knob names the refusal");
            assert!(note.contains(needle), "{note} lacks {needle}");
            assert!(note.starts_with("qwen4 EP tail borrow: ignored ("), "{note}");
            // The (flipped) default on a load it does not apply to is silent.
            let default = resolve_ep_tail_borrow(None, true, &facts);
            assert_eq!((default.plan, default.note), (None, None), "{needle}");
        }
    }

    #[test]
    fn the_published_plan_round_trips_and_the_attach_admits_only_what_was_planned() {
        qwen4_set_ep_tail_borrow_plan(plan(BORROWED));
        assert_eq!(qwen4_ep_tail_borrow_plan(), plan(BORROWED));
        qwen4_set_ep_tail_borrow_plan(None);
        assert_eq!(qwen4_ep_tail_borrow_plan(), None);
        let mode = Qwen4ForwardScratchMode::FnSymGfx1201;
        // Nothing published: the ordinary attach, no log.
        assert_eq!(ep_tail_borrow_attach_plan(None, true, mode, 4096, None), Ok(None));
        assert_eq!(
            ep_tail_borrow_attach_plan(plan(BORROWED), true, mode, BORROWED, None),
            Ok(plan(BORROWED))
        );
        for (ep, mode, requested, excluded, needle) in [
            (false, mode, BORROWED, None, "not expert-parallel"),
            (true, Qwen4ForwardScratchMode::Full, BORROWED, None, "Full"),
            (true, mode, 4096, None, "4096-row chunk"),
            (true, mode, BORROWED, Some("an S1 slow-link profile"), "S1"),
        ] {
            let error = ep_tail_borrow_attach_plan(plan(BORROWED), ep, mode, requested, excluded)
                .unwrap_err();
            assert!(error.contains(needle), "{error}");
        }
    }

    // ----- charge arithmetic --------------------------------------------------------------

    /// The compact plan with its own exchange root (what an EP load plans), and the Full plan.
    fn plans(
        mode: Qwen4ForwardScratchMode,
        rows: usize,
    ) -> (Vec<ScratchSlot>, usize) {
        let config = compact_test_config();
        Qwen4GpuForwardScratch::plan_scoped_root(
            &config,
            rows,
            mode,
            &Qwen4BandScope::single(&config),
            true,
        )
        .expect("plan")
    }

    fn owning_bytes(slots: &[ScratchSlot]) -> u64 {
        Qwen4GpuForwardScratch::owning_slot_bytes(slots)
            .map(|bytes| bytes.unwrap())
            .sum()
    }

    /// A plan without the dedicated exchange root (the trunk shard's compact plan: the
    /// grouped-down rows are a phase-arena view).
    fn shard_plans(rows: usize) -> Vec<ScratchSlot> {
        let config = compact_test_config();
        Qwen4GpuForwardScratch::plan_scoped_root(
            &config,
            rows,
            Qwen4ForwardScratchMode::FnSymGfx1201,
            &Qwen4BandScope::single(&config),
            false,
        )
        .expect("plan")
        .0
    }

    fn persistent_sum(slots: &[ScratchSlot]) -> u64 {
        persistent_root_bytes(slots)
            .unwrap()
            .iter()
            .map(|&(_, bytes)| bytes)
            .sum()
    }

    #[test]
    fn the_charge_is_the_resident_forward_plus_the_persistent_root_growth() {
        let config = compact_test_config();
        let cases = [
            (
                "compact, own root",
                plans(Qwen4ForwardScratchMode::FnSymGfx1201, RESIDENT).0,
                plans(Qwen4ForwardScratchMode::FnSymGfx1201, BORROWED).0,
                2usize,
            ),
            (
                "full, own root",
                plans(Qwen4ForwardScratchMode::Full, RESIDENT).0,
                plans(Qwen4ForwardScratchMode::Full, BORROWED).0,
                2,
            ),
            (
                "compact, trunk shard",
                shard_plans(RESIDENT),
                shard_plans(BORROWED),
                1,
            ),
        ];
        for (what, resident, borrowed, persistent) in cases {
            assert_eq!(persistent_fields(&borrowed).len(), persistent, "{what}");
            assert_eq!(persistent_fields(&resident), persistent_fields(&borrowed), "{what}");
            let spec = qwen4_spec_buffer_bytes(&config, RESIDENT).unwrap();
            let resident_total = owning_bytes(&resident) + spec[0] + spec[1];
            let charged = charged_forward_bytes(resident_total, &resident, &borrowed).unwrap();
            let (kept_r, kept_b) = (persistent_sum(&resident), persistent_sum(&borrowed));
            // Itemised: every owning root of the resident plan but the persistent ones, the
            // persistent roots at the borrowed rows, the speculative buffers.
            assert_eq!(
                charged,
                owning_bytes(&resident) - kept_r + kept_b + spec[0] + spec[1],
                "{what}"
            );
            // Strictly between the resident forward and the borrowed-row forward: the pool
            // holds the rest.
            assert!(charged > resident_total, "{what}");
            let borrowed_total = owning_bytes(&borrowed) + spec[0] + spec[1];
            assert!(charged < borrowed_total, "{what}");
            // The saving against a dedicated borrowed-row scratch is exactly the borrowed
            // bank's pool-backed roots, less what the resident bank no longer owns.
            let backing = tail_backing_of(BORROWED, &borrowed).unwrap();
            assert_eq!(
                borrowed_total - charged,
                (owning_bytes(&borrowed) - kept_b) - (owning_bytes(&resident) - kept_r),
                "{what}"
            );
            // The pool span holds exactly the borrowed bank's non-persistent roots (plus
            // 256-byte alignment padding).
            let pooled: u64 = backing.roots.iter().map(|root| root.bytes).sum();
            assert_eq!(pooled, owning_bytes(&borrowed) - kept_b, "{what}");
            // At most one 256-byte pad per root plus the final round-up.
            assert!(
                backing.span_bytes >= pooled
                    && backing.span_bytes - pooled <= 256 * (backing.roots.len() as u64 + 1)
            );
        }
    }

    /// The served shape: under the trunk shard only `TokenIds` stays out of the pool, the
    /// 8192-row bank fits the 1536-slot (3712 MiB) pool, and the charge over the resident
    /// 4096-row forward is the token ids' growth alone.
    #[test]
    fn under_the_trunk_shard_only_the_token_ids_persist_and_the_bank_fits_the_served_pool() {
        let (resident, borrowed) = (shard_plans(RESIDENT), shard_plans(BORROWED));
        assert_eq!(persistent_fields(&borrowed), vec![ScratchField::TokenIds]);
        let total = owning_bytes(&resident);
        let ids = |slots: &[ScratchSlot]| {
            slots[ScratchField::TokenIds as usize].bytes().unwrap() as u64
        };
        assert_eq!(
            charged_forward_bytes(total, &resident, &borrowed),
            Some(total - ids(&resident) + ids(&borrowed))
        );
        let backing = tail_backing_of(BORROWED, &borrowed).unwrap();
        assert!(
            backing.span_bytes <= 3712 << 20,
            "{} MiB",
            backing.span_bytes >> 20
        );
        assert!(backing
            .roots
            .iter()
            .all(|root| root.field != ScratchField::TokenIds));
    }

    #[test]
    fn the_charge_refuses_a_plan_that_does_not_own_both_roots_or_shrinks_them() {
        let (resident, _) = plans(Qwen4ForwardScratchMode::Full, RESIDENT);
        let (borrowed, _) = plans(Qwen4ForwardScratchMode::Full, BORROWED);
        let total = owning_bytes(&resident);
        assert!(charged_forward_bytes(total, &resident, &borrowed).is_some());
        // Swapped plans: the borrowed roots are smaller than the resident ones.
        assert_eq!(charged_forward_bytes(total, &borrowed, &resident), None);
        // A view in place of an owning root.
        let mut viewed = borrowed.clone();
        viewed[ScratchField::MoeYDownGrouped as usize].owner = Some(ScratchField::MoeGateUp);
        assert_eq!(charged_forward_bytes(total, &resident, &viewed), None);
        // Overflow of the resident total.
        assert_eq!(charged_forward_bytes(0, &resident, &borrowed), None);
        // The public form refuses a zero-row plan and an unplannable geometry.
        let config = compact_test_config();
        assert_eq!(
            qwen4_ep_tail_borrow_forward_bytes(
                &config,
                &EpTailBorrowPlan {
                    borrowed_rows: 0,
                    resident_rows: RESIDENT
                },
                Qwen4ForwardScratchMode::Full
            ),
            None
        );
    }

    #[test]
    fn the_exchange_capacity_is_the_borrowed_plans_not_the_residents() {
        let (resident, _) = plans(Qwen4ForwardScratchMode::FnSymGfx1201, RESIDENT);
        let (borrowed, _) = plans(Qwen4ForwardScratchMode::FnSymGfx1201, BORROWED);
        let small = exchange_capacity_of(RESIDENT, &resident).unwrap();
        let wide = exchange_capacity_of(BORROWED, &borrowed).unwrap();
        assert_eq!(wide.rows, BORROWED);
        assert!(wide.grouped_down_bytes > small.grouped_down_bytes);
        assert!(wide.tile_ids > small.tile_ids);
        assert_eq!(wide.grouped_rows, wide.tile_ids * MOE_GROUPED_BLOCK_M);
        let down = persistent_root_bytes(&borrowed)
            .unwrap()
            .into_iter()
            .find(|&(field, _)| field == ScratchField::MoeYDownGrouped)
            .unwrap()
            .1;
        assert_eq!(wide.grouped_down_bytes, down);
    }

    #[test]
    fn the_backing_leaves_both_persistent_roots_out_and_aligns_every_root() {
        let (borrowed, _) = plans(Qwen4ForwardScratchMode::FnSymGfx1201, BORROWED);
        let backing = tail_backing_of(BORROWED, &borrowed).unwrap();
        assert_eq!(backing.capacity_rows, BORROWED);
        let persistent = persistent_fields(&borrowed);
        assert!(backing
            .roots
            .iter()
            .all(|root| !persistent.contains(&root.field)));
        let mut end = 0u64;
        for root in &backing.roots {
            assert_eq!(root.offset % DECODE_SHARE_ROOT_ALIGN, 0, "{:?}", root.field);
            assert!(root.offset >= end, "{:?} overlaps its predecessor", root.field);
            end = root.offset + root.bytes;
        }
        assert!(backing.span_bytes >= end);
        assert_eq!(backing.span_bytes % DECODE_SHARE_ROOT_ALIGN, 0);
        // Strictly smaller than the decode-share packing, which carries the grouped-down root.
        assert!(backing.span_bytes < shared_span(&borrowed));
    }

    /// The decode-share packing (`borrowed_backing`'s layout) of `slots`, for the comparison
    /// above.
    fn shared_span(slots: &[ScratchSlot]) -> u64 {
        let mut cursor = 0u64;
        for (index, slot) in slots.iter().enumerate() {
            if ScratchField::ALL[index] == ScratchField::TokenIds || slot.owner.is_some() {
                continue;
            }
            let bytes = slot.bytes().unwrap() as u64;
            if bytes == 0 {
                continue;
            }
            cursor = (cursor + 255) & !255;
            cursor += bytes;
        }
        (cursor + 255) & !255
    }

    // ----- epoch state machine ------------------------------------------------------------

    fn borrow_call(epoch: &mut EpTailEpoch, rows: usize) -> EpTailCallPlan {
        let plan = epoch
            .plan_call(rows, RESIDENT, BORROWED, true)
            .expect("a borrowable call");
        if plan == (EpTailCallPlan::Borrow { open_first: true }) {
            epoch.opened();
        }
        plan
    }

    #[test]
    fn a_call_plans_the_resident_bank_or_the_borrowed_one() {
        let epoch = EpTailEpoch::default();
        for rows in [0, 1, 512, RESIDENT] {
            assert_eq!(
                epoch.plan_call(rows, RESIDENT, BORROWED, false),
                Ok(EpTailCallPlan::Resident),
                "{rows}"
            );
        }
        assert_eq!(
            epoch.plan_call(RESIDENT + 1, RESIDENT, BORROWED, true),
            Ok(EpTailCallPlan::Borrow { open_first: true })
        );
        assert_eq!(
            epoch.plan_call(BORROWED, RESIDENT, BORROWED, true),
            Ok(EpTailCallPlan::Borrow { open_first: true })
        );
        // Not a prefill tile (device argmax / mesh segment): refused before any write.
        let error = epoch
            .plan_call(BORROWED, RESIDENT, BORROWED, false)
            .unwrap_err();
        assert!(error.contains("device-argmax"), "{error}");
        // Above the borrowed capacity.
        let error = epoch
            .plan_call(BORROWED + 1, RESIDENT, BORROWED, true)
            .unwrap_err();
        assert!(error.contains("8192-row borrowed prompt capacity"), "{error}");
        // Planning never mutates.
        assert_eq!(epoch, EpTailEpoch::default());
    }

    #[test]
    fn prefill_then_decode_then_prefill_opens_and_closes_each_epoch() {
        let mut epoch = EpTailEpoch::default();
        // A long singleton prompt: three wide tiles and a short tail inside one bracket.
        assert!(epoch.begin_prompt());
        assert_eq!(borrow_call(&mut epoch, BORROWED), EpTailCallPlan::Borrow { open_first: true });
        assert!(epoch.open());
        assert_eq!(borrow_call(&mut epoch, BORROWED), EpTailCallPlan::Borrow { open_first: false });
        // The tail tile fits the resident bank: resident, the epoch stays open.
        assert_eq!(borrow_call(&mut epoch, 1000), EpTailCallPlan::Resident);
        assert!(epoch.open());
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Close);
        epoch.closed();
        assert!(!epoch.open());
        // Decode: resident, nothing to open or close.
        for _ in 0..5 {
            assert_eq!(borrow_call(&mut epoch, 1), EpTailCallPlan::Resident);
        }
        assert_eq!(epoch.epochs(), 1);
        // The next long prompt opens a second epoch.
        assert!(epoch.begin_prompt());
        assert_eq!(borrow_call(&mut epoch, 6000), EpTailCallPlan::Borrow { open_first: true });
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Close);
        epoch.closed();
        assert_eq!((epoch.epochs(), epoch.depth()), (2, 0));
        // A short prompt never opens one.
        assert!(epoch.begin_prompt());
        assert_eq!(borrow_call(&mut epoch, 2000), EpTailCallPlan::Resident);
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Clean);
        assert_eq!(epoch.epochs(), 2);
    }

    #[test]
    fn nested_brackets_close_only_at_the_outermost_end() {
        let mut epoch = EpTailEpoch::default();
        assert!(epoch.begin_prompt()); // forward_chunk
        assert!(!epoch.begin_prompt()); // an MTP fill / wave inside it
        assert_eq!(borrow_call(&mut epoch, BORROWED), EpTailCallPlan::Borrow { open_first: true });
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Nested);
        assert!(epoch.open());
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Close);
        // An unbalanced end saturates instead of wrapping.
        let mut stray = EpTailEpoch::default();
        assert_eq!(stray.end_prompt(true, false), EpTailPromptEnd::Clean);
        assert_eq!(stray.depth(), 0);
    }

    #[test]
    fn a_lane_prompt_spanning_three_steps_closes_once() {
        let mut epoch = EpTailEpoch::default();
        let mut closes = 0;
        // Each lane step is one bracket carrying one wide tile and, between steps, resident
        // decode calls; only the last step's bracket ends without a continuation.
        for (step, continues) in [(0, true), (1, true), (2, false)] {
            assert!(epoch.begin_prompt());
            // The step's decode rows ran before its prefill tile, on the resident bank.
            assert_eq!(borrow_call(&mut epoch, 8), EpTailCallPlan::Resident);
            let plan = borrow_call(&mut epoch, BORROWED);
            assert_eq!(plan, EpTailCallPlan::Borrow { open_first: step == 0 }, "step {step}");
            match epoch.end_prompt(true, continues) {
                EpTailPromptEnd::Keep => assert!(continues && epoch.open()),
                EpTailPromptEnd::Close => {
                    assert!(!continues);
                    epoch.closed();
                    closes += 1;
                }
                other => panic!("step {step}: {other:?}"),
            }
        }
        assert_eq!((closes, epoch.epochs(), epoch.open()), (1, 1, false));
        // A step with no bracket at all (decode only) never touches the epoch.
        assert_eq!(borrow_call(&mut epoch, 8), EpTailCallPlan::Resident);
    }

    #[test]
    fn a_continuation_that_never_comes_back_is_closed_idle() {
        let mut epoch = EpTailEpoch::default();
        assert!(!epoch.idle_open());
        epoch.begin_prompt();
        borrow_call(&mut epoch, BORROWED);
        assert_eq!(epoch.end_prompt(true, true), EpTailPromptEnd::Keep);
        assert!(epoch.idle_open());
        // Inside a bracket it is not idle.
        epoch.begin_prompt();
        assert!(!epoch.idle_open());
        assert_eq!(epoch.end_prompt(true, true), EpTailPromptEnd::Keep);
        assert!(epoch.idle_open());
        epoch.closed();
        assert!(!epoch.idle_open());
    }

    #[test]
    fn a_failed_prompt_recovers_an_open_epoch_and_a_closed_one_is_clean() {
        let mut epoch = EpTailEpoch::default();
        epoch.begin_prompt();
        borrow_call(&mut epoch, BORROWED);
        // Even a prompt that would continue recovers when it failed.
        assert_eq!(epoch.end_prompt(false, true), EpTailPromptEnd::Recover);
        epoch.closed();
        epoch.begin_prompt();
        assert_eq!(epoch.end_prompt(false, false), EpTailPromptEnd::Clean);
        // The failing nested bracket does not recover; the outermost does.
        epoch.begin_prompt();
        epoch.begin_prompt();
        borrow_call(&mut epoch, BORROWED);
        assert_eq!(epoch.end_prompt(false, false), EpTailPromptEnd::Nested);
        assert_eq!(epoch.end_prompt(false, false), EpTailPromptEnd::Recover);
    }

    #[test]
    fn a_failed_tile_closes_and_the_next_prompt_reopens() {
        let mut epoch = EpTailEpoch::default();
        epoch.begin_prompt();
        assert_eq!(borrow_call(&mut epoch, BORROWED), EpTailCallPlan::Borrow { open_first: true });
        // The forward closes right after the failed tile (before the bracket ends).
        epoch.closed();
        assert_eq!(epoch.end_prompt(false, false), EpTailPromptEnd::Clean);
        assert_eq!(epoch.check(), Ok(()));
        epoch.begin_prompt();
        assert_eq!(borrow_call(&mut epoch, BORROWED), EpTailCallPlan::Borrow { open_first: true });
        assert_eq!(epoch.epochs(), 2);
    }

    #[test]
    fn a_failed_close_poisons_and_every_later_call_refuses() {
        let mut epoch = EpTailEpoch::default();
        epoch.begin_prompt();
        borrow_call(&mut epoch, BORROWED);
        epoch.poison("close after a failed prompt: host copy failed");
        assert!(!epoch.open());
        let error = epoch.check().unwrap_err();
        assert!(
            error.starts_with("EP tail borrow poisoned (close after a failed prompt: host copy failed)")
                && error.ends_with("reload the model"),
            "{error}"
        );
        // Resident calls refuse too: nothing runs on a poisoned load.
        for rows in [1, RESIDENT, RESIDENT + 1, BORROWED] {
            assert_eq!(epoch.plan_call(rows, RESIDENT, BORROWED, true), Err(error.clone()));
        }
        // The first reason sticks; a second poison never rewrites it.
        epoch.poison("later");
        assert_eq!(epoch.poisoned(), Some("close after a failed prompt: host copy failed"));
        // The bracket still balances and reports nothing to recover.
        assert_eq!(epoch.end_prompt(true, false), EpTailPromptEnd::Clean);
    }
}
