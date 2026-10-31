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

/// The S2 host-source resolver: per layer of `[H, R)` the pinned CPU span of
/// each loan mirror (gate/up, down). Layers `[R, trunk)` resolve to their
/// original pinned owners at the stage, as on an ordinary load.
pub(crate) struct EpStageHostSources {
    first: usize,
    spans: Vec<[(usize, usize); 2]>,
}

impl EpStageHostSources {
    pub(crate) fn new(
        gpu: &Gpu,
        loan: &EpReserveLoan,
        plan: &EpReservePlan,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let mut spans = Vec::with_capacity(plan.donor_layers.len());
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
        }
        Ok(Self {
            first: plan.donor_first_layer,
            spans,
        })
    }

    /// The mirrors' complete pinned bytes of `layer` (gate/up, down), `None`
    /// outside `[H, R)`.
    pub(crate) fn pair(&self, layer: usize) -> Option<[&[u8]; 2]> {
        let spans = self.spans.get(layer.checked_sub(self.first)?)?;
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
            Ok((source_generation, loan, host_sources, exchange_root, receipts)) => Ok(Self {
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
                prompt: None,
                prompt_seq: 0,
                ordinals: EpReceiptOrdinals::default(),
            }),
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
        let (resident_rows, prompt_rows) = {
            let reserve = self.ep_reserve.as_ref().expect("an EP reserve forward");
            reserve.check()?;
            (reserve.banks.resident_rows(), reserve.banks.prompt_rows())
        };
        let route = if n > resident_rows {
            // The same decision `forward_chunk_scoped` takes, under the same scopes.
            let (scopes, _) = self.enter_tile_scopes(gpu, n);
            let route_n = gpu.route_rows(n);
            let tile = ep_tile_active(self.ep.as_ref(), gpu, &bundle.config, n, route_n);
            Self::leave_tile_scopes(gpu, scopes);
            if tile {
                EpBankRoute::TileExchange
            } else {
                EpBankRoute::FullTable
            }
        } else {
            EpBankRoute::TileExchange
        };
        let plan = self
            .ep_reserve
            .as_ref()
            .expect("an EP reserve forward")
            .epoch
            .plan_call(n, resident_rows, prompt_rows, route)
            .map_err(invalid)?;
        if plan.bank == EpBankKind::Borrowed
            && (argmax_of.is_some() || self.lane_scope || self.mesh_route_rows.is_some())
        {
            return Err(invalid(
                "an argmax, lane or mesh forward cannot run on the borrowed EP reserve bank",
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
            first: plan.donor_first_layer,
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
}
