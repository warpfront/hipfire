// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! EP2 reserve decoupling (default on; `HIPFIRE_EP_RESERVE_DECOUPLE=0` is the kill switch): pure admission,
//! immutable plan and receipt types, and the deterministic packing and capacity
//! arithmetic. Nothing here allocates on a device or reads the environment
//! after a plan is frozen.
//!
//! The experiment keeps EP's whole-layer `Full` placement and charges the
//! forward a dedicated 512-row bank plus two persistent 4096-row roots
//! (`TokenIds`, `MoeYDownGrouped`, the registered tile-exchange address), while
//! the remaining 4096-row owning roots live in the expert tensors of a fixed
//! suffix of whole-layer "donor" layers (gate/up and down tensors are the
//! segments). Design and gates: `release-0.4.3/mgfn/ep-reserve/DECISION.md`.
//!
//! One immutable [`EpReservePlan`] feeds the loader's four reserve readers, the
//! forward and the mesh receipt ([`EpReserveMeshReceipt`]).

use crate::config::Qwen4Config;
use crate::gpu_forward::{
    ep_reserve_scratch::{ep_reserve_scratch, ep_reserve_spec_bytes},
    Qwen4ForwardScratchMode, ScratchField, QWEN4_DECODE_SHARE_RESIDENT_ROWS,
};
use std::fmt;

/// Kill switch of the default-on route: exact `0` is the ordinary placement
/// byte for byte; `1` forces the route (a refused capacity plan fails the
/// load); unset or any other value is the default route.
pub const EP_RESERVE_DECOUPLE_ENV: &str = "HIPFIRE_EP_RESERVE_DECOUPLE";

/// Prompt chunk of an admitted load; it is never lowered.
pub const EP_RESERVE_PROMPT_ROWS: usize = 4096;
/// Rows of the resident (dedicated) scratch bank.
pub const EP_RESERVE_RESIDENT_ROWS: usize = QWEN4_DECODE_SHARE_RESIDENT_ROWS;
/// Alignment of every root start inside a donor segment.
pub const EP_RESERVE_ROOT_ALIGN: u64 = 256;
/// Whole-layer donors of the stage (S2) borrowed view.
pub const EP_RESERVE_S2_DONORS: usize = 2;
/// Expert-parallel rank count of the admitted topology.
pub const EP_RESERVE_RANKS: usize = 2;
/// Resident layers per rank the experiment must reach to be worth cards.
pub const EP_RESERVE_MIN_RESIDENT_LAYERS: usize = 33;

/// Bytes of one expert's gate/up tensor in the pinned QT44 recipe.
pub const EP2_GATE_UP_EXPERT_BYTES: u64 = 1_740_800;
/// Bytes of one expert's down tensor in the pinned QT53 recipe.
pub const EP2_DOWN_EXPERT_BYTES: u64 = 870_400;

const MIB: u64 = 1 << 20;

/// How [`EP_RESERVE_DECOUPLE_ENV`] asks for the decoupled reserve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpReserveRequest {
    /// `HIPFIRE_EP_RESERVE_DECOUPLE=0`: the ordinary placement and reserve.
    Off,
    /// Unset (or any value but `0`/`1`): admitted loads take the decoupled
    /// reserve; a load that cannot (admission or capacity refusal) keeps the
    /// ordinary placement with a note, exactly as before the route existed.
    Default,
    /// `=1`: as `Default`, but a capacity refusal fails the load before any
    /// allocation.
    Forced,
}

impl EpReserveRequest {
    /// Whether any decoupling attempt is made.
    pub fn attempts(self) -> bool {
        self != Self::Off
    }
}

/// The request `value` (the variable's content) makes; read it once at load.
pub fn ep_reserve_request(value: Option<&str>) -> EpReserveRequest {
    match value.map(str::trim) {
        Some("0") => EpReserveRequest::Off,
        Some("1") => EpReserveRequest::Forced,
        _ => EpReserveRequest::Default,
    }
}

/// [`ep_reserve_request`] of the process environment. Call once at load
/// admission; the frozen plan never re-reads it.
pub fn ep_reserve_request_from_env() -> EpReserveRequest {
    ep_reserve_request(std::env::var(EP_RESERVE_DECOUPLE_ENV).ok().as_deref())
}

/// Why the experiment is not admitted or a plan cannot be made. The load
/// keeps today's untouched path and logs the reason (phase one), or tears the
/// candidate down (phase two).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(pub String);

impl Refusal {
    fn new(reason: impl Into<String>) -> Self {
        Self(reason.into())
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EP reserve decoupling: {}", self.0)
    }
}

impl std::error::Error for Refusal {}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

/// Everything [`admit_ep_reserve`] judges, from the one load's frozen facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpReserveFacts<'a> {
    /// Exact arch string of rank 0's device (`gfx1201`).
    pub arch: &'a str,
    /// Unified-memory (APU) device.
    pub uma: bool,
    /// Pinned Flash-Next profile (arch/model config matches).
    pub flash_next: bool,
    /// Routed-expert recipe `(gate/up qt, down qt)`; `Some((44, 53))` admitted.
    pub expert_qt: Option<(u32, u32)>,
    pub ranks: usize,
    pub scratch_mode: Qwen4ForwardScratchMode,
    /// The prompt chunk the load would use (already resolved).
    pub prompt_rows: usize,
    /// An explicit chunk override was requested (a different chunk is ordinary
    /// placement, never a hidden reschedule).
    pub explicit_chunk_override: bool,
    /// Whole-layer placement (no hot-set tier, slab or aggregate tier).
    pub whole_layer_placement: bool,
    /// Owner-routed S1 prefill requested/admitted for the load.
    pub owner_routed_s1: bool,
    /// Stage (S2) minimum rows; `None` when S2 is off. Must exceed the
    /// resident bank so every call that can stage uses the borrowed bank.
    pub s2_min_rows: Option<usize>,
    /// Dynamic AR-lane staging requested.
    pub lane_request: bool,
    /// Retained-body / capture replay requested.
    pub capture: bool,
    /// Invariant-prefill or routing overrides that can produce a wide Full-table call.
    pub routing_override: bool,
    pub pipeline_parallel: bool,
    pub expert_tier: bool,
    pub expert_slab: bool,
    pub npu: bool,
    pub dflash: bool,
    pub single_device_slowlink: bool,
    pub continuous_batch: bool,
}

/// Pure phase-one admission. `Ok` only for discrete gfx1201 EP2, pinned
/// Flash-Next QT44/QT53, `Full` scratch, a 4096-row prompt chunk, whole-layer
/// placement and ordinary routing; AR and native MTP are both in scope.
pub fn admit_ep_reserve(facts: &EpReserveFacts<'_>) -> Result<(), Refusal> {
    let no = |reason: &str| Err(Refusal::new(reason));
    if facts.arch != "gfx1201" {
        return no("needs exact gfx1201");
    }
    if facts.uma {
        return no("a unified-memory device is not a discrete EP topology");
    }
    if !facts.flash_next {
        return no("not the pinned Flash-Next profile");
    }
    if facts.expert_qt != Some((44, 53)) {
        return no("expert recipe is not the pinned QT44/QT53 pair");
    }
    if facts.ranks != EP_RESERVE_RANKS {
        return no("needs exactly two expert-parallel ranks");
    }
    if facts.scratch_mode != Qwen4ForwardScratchMode::Full {
        return no("the forward scratch plan is not Full");
    }
    if facts.prompt_rows != EP_RESERVE_PROMPT_ROWS || facts.explicit_chunk_override {
        return no("the prompt chunk is not the default 4096 rows");
    }
    if !facts.whole_layer_placement {
        return no("placement is not whole-layer (hot-set tier or slab)");
    }
    if facts.pipeline_parallel {
        return no("a pipeline-parallel band load");
    }
    if facts.expert_tier {
        return no("a hot-set tier placement");
    }
    if facts.expert_slab {
        return no("an expert-slab placement");
    }
    if facts.npu {
        return no("the NPU trunk is selected");
    }
    if facts.dflash {
        return no("a DFlash speculator is requested");
    }
    if facts.single_device_slowlink {
        return no("a single-device slow-link prefill profile");
    }
    if facts.continuous_batch {
        return no("continuous batching stages lanes");
    }
    if facts.lane_request {
        return no("dynamic AR-lane staging is requested");
    }
    if facts.capture {
        return no("retained-body capture replay is requested");
    }
    if facts.routing_override {
        return no("a routing/invariant-prefill override can take wide Full-table calls");
    }
    if let Some(min_rows) = facts.s2_min_rows {
        if min_rows <= EP_RESERVE_RESIDENT_ROWS {
            return no("stage (S2) can run on resident-bank calls (min rows <= 512)");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Forward plan
// ---------------------------------------------------------------------------

/// One owning scratch root: its [`ScratchField`] and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpReserveRoot {
    pub field: ScratchField,
    pub bytes: u64,
}

/// What one dedicated allocation of the admitted forward is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpReserveAllocKind {
    /// An owning root of the 512-row resident bank.
    ResidentRoot(ScratchField),
    /// A persistent 4096-row root shared by both banks.
    PersistentRoot(ScratchField),
    /// Speculative logits + argmax of the prompt rows.
    SpecBuffers,
    /// Fixed tables / metadata the experiment adds.
    FixedTables,
}

/// One entry of the exact dedicated allocation list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpReserveAllocation {
    pub kind: EpReserveAllocKind,
    pub bytes: u64,
}

/// Prompt/resident rows, the persistent and remaining root specs and the
/// dedicated charge, derived once from the authoritative scratch plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpReserveForwardPlan {
    pub mode: Qwen4ForwardScratchMode,
    pub prompt_rows: usize,
    pub resident_rows: usize,
    /// Persistent roots at the prompt rows (`TokenIds`, `MoeYDownGrouped`),
    /// owned once and shared by both banks.
    pub(crate) external_roots: Vec<EpReserveRoot>,
    /// The same persistent roots at the resident rows (a prefix of the above).
    pub(crate) resident_persistent: Vec<EpReserveRoot>,
    /// Owning non-persistent roots of the resident bank.
    pub(crate) resident_roots: Vec<EpReserveRoot>,
    /// Owning non-persistent roots at the prompt rows: the donor-packing input.
    pub(crate) remaining_roots: Vec<EpReserveRoot>,
    /// Speculative logits + argmax bytes of the prompt rows.
    pub spec_bytes: u64,
    /// Extra fixed table/metadata bytes the experiment charges.
    pub fixed_table_bytes: u64,
}

fn is_persistent(field: ScratchField) -> bool {
    matches!(
        field,
        ScratchField::TokenIds | ScratchField::MoeYDownGrouped
    )
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
}

fn split_specs(
    specs: &[(ScratchField, u64, bool)],
    rows: usize,
) -> Result<(Vec<EpReserveRoot>, Vec<EpReserveRoot>), Refusal> {
    let mut seen = [false; 64];
    let (mut persistent, mut owned) = (Vec::new(), Vec::new());
    for &(field, bytes, flag) in specs {
        let index = field as usize;
        if index >= seen.len() || std::mem::replace(&mut seen[index], true) {
            return Err(Refusal::new(format!(
                "scratch plan at {rows} rows repeats root {field:?}"
            )));
        }
        if flag != is_persistent(field) {
            return Err(Refusal::new(format!(
                "root {field:?} at {rows} rows has an unexpected persistence flag"
            )));
        }
        if bytes == 0 {
            continue;
        }
        let root = EpReserveRoot { field, bytes };
        if flag {
            persistent.push(root);
        } else {
            owned.push(root);
        }
    }
    for field in [ScratchField::TokenIds, ScratchField::MoeYDownGrouped] {
        if !persistent.iter().any(|root| root.field == field) {
            return Err(Refusal::new(format!(
                "scratch plan at {rows} rows has no persistent {field:?} root"
            )));
        }
    }
    Ok((persistent, owned))
}

impl EpReserveForwardPlan {
    /// The plan of `config` in `Full` mode from the production scratch planner
    /// ([`ep_reserve_scratch`]); `fixed_table_bytes` is every additional
    /// table/metadata charge the integration owner declares.
    pub fn from_config(
        config: &Qwen4Config,
        mode: Qwen4ForwardScratchMode,
        fixed_table_bytes: u64,
    ) -> Result<Self, Refusal> {
        let specs = |rows: usize| {
            ep_reserve_scratch(config, mode, rows).ok_or_else(|| {
                Refusal::new(format!(
                    "the {rows}-row scratch plan overflows or is refused"
                ))
            })
        };
        let spec_bytes = ep_reserve_spec_bytes(config, EP_RESERVE_PROMPT_ROWS)
            .ok_or_else(|| Refusal::new("speculative buffer bytes overflow"))?;
        Self::from_specs(
            mode,
            EP_RESERVE_PROMPT_ROWS,
            EP_RESERVE_RESIDENT_ROWS,
            &specs(EP_RESERVE_PROMPT_ROWS)?,
            &specs(EP_RESERVE_RESIDENT_ROWS)?,
            spec_bytes,
            fixed_table_bytes,
        )
    }

    /// The plan from explicit root specs `(field, bytes, persistent)`.
    pub(crate) fn from_specs(
        mode: Qwen4ForwardScratchMode,
        prompt_rows: usize,
        resident_rows: usize,
        prompt_specs: &[(ScratchField, u64, bool)],
        resident_specs: &[(ScratchField, u64, bool)],
        spec_bytes: u64,
        fixed_table_bytes: u64,
    ) -> Result<Self, Refusal> {
        if mode != Qwen4ForwardScratchMode::Full {
            return Err(Refusal::new("the forward scratch plan is not Full"));
        }
        if resident_rows == 0 || resident_rows >= prompt_rows {
            return Err(Refusal::new("resident rows must be in 1..prompt rows"));
        }
        let (external_roots, remaining_roots) = split_specs(prompt_specs, prompt_rows)?;
        let (resident_persistent, resident_roots) = split_specs(resident_specs, resident_rows)?;
        for resident in &resident_persistent {
            let prompt = external_roots
                .iter()
                .find(|root| root.field == resident.field)
                .expect("split_specs checked both persistent roots");
            if prompt.bytes < resident.bytes {
                return Err(Refusal::new(format!(
                    "persistent root {:?} is smaller at the prompt rows than at the resident rows",
                    resident.field
                )));
            }
        }
        Ok(Self {
            mode,
            prompt_rows,
            resident_rows,
            external_roots,
            resident_persistent,
            resident_roots,
            remaining_roots,
            spec_bytes,
            fixed_table_bytes,
        })
    }

    fn root_bytes(roots: &[EpReserveRoot], field: ScratchField) -> u64 {
        roots
            .iter()
            .find(|root| root.field == field)
            .map_or(0, |root| root.bytes)
    }

    /// Bytes of the persistent `field` root at the prompt rows (475 MiB for
    /// `MoeYDownGrouped` on the pinned 4096-row plan).
    pub fn persistent_prompt_bytes(&self, field: ScratchField) -> u64 {
        Self::root_bytes(&self.external_roots, field)
    }

    /// Bytes of the persistent `field` root at the resident rows (125 MiB for
    /// `MoeYDownGrouped` on the pinned 512-row plan).
    pub fn persistent_resident_bytes(&self, field: ScratchField) -> u64 {
        Self::root_bytes(&self.resident_persistent, field)
    }

    /// The dedicated forward bytes, exactly
    /// `FullScratch(512) - TokenIds(512) - YDown(512) + TokenIds(4096) +
    /// YDown(4096) + SpecBuffers(4096)`.
    pub fn dedicated_bytes(&self) -> Result<u64, Refusal> {
        let overflow = || Refusal::new("dedicated forward bytes overflow");
        let full_resident = checked_sum(
            self.resident_roots
                .iter()
                .chain(&self.resident_persistent)
                .map(|root| root.bytes),
        )
        .ok_or_else(overflow)?;
        let token_ids_512 = self.persistent_resident_bytes(ScratchField::TokenIds);
        let y_down_512 = self.persistent_resident_bytes(ScratchField::MoeYDownGrouped);
        let token_ids = self.persistent_prompt_bytes(ScratchField::TokenIds);
        let y_down = self.persistent_prompt_bytes(ScratchField::MoeYDownGrouped);
        full_resident
            .checked_sub(token_ids_512)
            .and_then(|bytes| bytes.checked_sub(y_down_512))
            .and_then(|bytes| bytes.checked_add(token_ids))
            .and_then(|bytes| bytes.checked_add(y_down))
            .and_then(|bytes| bytes.checked_add(self.spec_bytes))
            .ok_or_else(overflow)
    }

    /// The forward term the reserve charges instead of
    /// `qwen4_forward_device_bytes`: [`Self::dedicated_bytes`] plus the fixed
    /// table charges, in the historical floor AND the census.
    pub fn charged_forward_bytes(&self) -> Result<u64, Refusal> {
        self.dedicated_bytes()?
            .checked_add(self.fixed_table_bytes)
            .ok_or_else(|| Refusal::new("charged forward bytes overflow"))
    }

    /// Every dedicated allocation, one entry per actual allocation; the entries
    /// sum to [`Self::charged_forward_bytes`].
    pub(crate) fn allocations(&self) -> Vec<EpReserveAllocation> {
        let mut list = Vec::new();
        for root in &self.resident_roots {
            list.push(EpReserveAllocation {
                kind: EpReserveAllocKind::ResidentRoot(root.field),
                bytes: root.bytes,
            });
        }
        for root in &self.external_roots {
            list.push(EpReserveAllocation {
                kind: EpReserveAllocKind::PersistentRoot(root.field),
                bytes: root.bytes,
            });
        }
        if self.spec_bytes > 0 {
            list.push(EpReserveAllocation {
                kind: EpReserveAllocKind::SpecBuffers,
                bytes: self.spec_bytes,
            });
        }
        if self.fixed_table_bytes > 0 {
            list.push(EpReserveAllocation {
                kind: EpReserveAllocKind::FixedTables,
                bytes: self.fixed_table_bytes,
            });
        }
        list
    }

    /// Total bytes of the roots the donors must hold.
    pub fn remaining_bytes(&self) -> Result<u64, Refusal> {
        checked_sum(self.remaining_roots.iter().map(|root| root.bytes))
            .ok_or_else(|| Refusal::new("remaining root bytes overflow"))
    }
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

fn align_up(value: u64) -> Option<u64> {
    value
        .checked_add(EP_RESERVE_ROOT_ALIGN - 1)
        .map(|value| value & !(EP_RESERVE_ROOT_ALIGN - 1))
}

/// Deterministic descending-size first-fit packing of `roots` into segments of
/// the given capacities (tried in order). Roots are placed in descending size,
/// ties by [`ScratchField`] order; each start is aligned to 256 bytes; no root
/// straddles a segment. Returns `(field, segment, offset, bytes)` sorted by
/// `(segment, offset)`, or `None` when any root does not fit or on overflow.
pub(crate) fn pack_roots(
    roots: &[(ScratchField, u64)],
    segment_caps: &[u64],
) -> Option<Vec<(ScratchField, usize, u64, u64)>> {
    let mut order: Vec<(ScratchField, u64)> = roots
        .iter()
        .copied()
        .filter(|&(_, bytes)| bytes > 0)
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then((a.0 as usize).cmp(&(b.0 as usize))));
    let mut cursor = vec![0u64; segment_caps.len()];
    let mut placed = Vec::with_capacity(order.len());
    for (field, bytes) in order {
        let mut slot = None;
        for (segment, &cap) in segment_caps.iter().enumerate() {
            let start = align_up(cursor[segment])?;
            let end = start.checked_add(bytes)?;
            // The touched extent ends at the aligned end; it must be representable too.
            align_up(end)?;
            if end <= cap {
                cursor[segment] = end;
                slot = Some((segment, start));
                break;
            }
        }
        let (segment, offset) = slot?;
        placed.push((field, segment, offset, bytes));
    }
    placed.sort_by_key(|&(_, segment, offset, _)| (segment, offset));
    Some(placed)
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

/// Which expert tensor of a donor layer a segment is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpDonorSlot {
    GateUp,
    Down,
}

/// One donor segment: a whole-layer expert tensor of one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpReserveSegment {
    pub layer: usize,
    pub slot: EpDonorSlot,
    pub capacity: u64,
}

/// Where one remaining root lives. `segment` indexes [`EpReservePlan::segments`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EpReserveRootPlacement {
    pub field: ScratchField,
    pub segment: usize,
    pub layer: usize,
    pub slot: EpDonorSlot,
    pub offset: u64,
    pub bytes: u64,
}

/// The bytes of a segment the borrowed bank dirties: `[offset, offset + len)`
/// of the donor tensor; `len` is the used bytes rounded to 256.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpReserveTouchedExtent {
    pub layer: usize,
    pub slot: EpDonorSlot,
    pub offset: u64,
    pub len: u64,
}

/// One rank's expert geometry (the largest rank of an uneven split).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpRankGeometry {
    pub ranks: usize,
    /// Trunk layers (donors and S2 donors come from these).
    pub trunk_layers: usize,
    /// Trunk plus the MTP layer, when present.
    pub moe_layers: usize,
    /// One layer's gate/up tensor bytes on the largest rank.
    pub gate_up_bytes: u64,
    /// One layer's down tensor bytes on the largest rank.
    pub down_bytes: u64,
}

impl EpRankGeometry {
    /// The pinned Flash-Next QT44/QT53 geometry of `ranks` stride-owned ranks.
    pub fn flash_next(config: &Qwen4Config, ranks: usize) -> Option<Self> {
        let local = config.num_experts.checked_add(ranks.checked_sub(1)?)? / ranks;
        Some(Self {
            ranks,
            trunk_layers: config.num_hidden_layers,
            moe_layers: crate::expert_residency::moe_layer_count(config),
            gate_up_bytes: (local as u64).checked_mul(EP2_GATE_UP_EXPERT_BYTES)?,
            down_bytes: (local as u64).checked_mul(EP2_DOWN_EXPERT_BYTES)?,
        })
    }

    /// One MoE layer's routed-expert bytes on one rank.
    pub fn layer_bytes(&self) -> Option<u64> {
        self.gate_up_bytes.checked_add(self.down_bytes)
    }
}

/// The unchanged capacity census a plan is judged against.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpReserveCapacity {
    /// Every rank's free VRAM before the load; the split takes the minimum.
    pub free_per_rank: Vec<u64>,
    pub non_expert_bytes: u64,
    /// The full reserve (the 4096-row census/historical-floor maximum computed
    /// with this plan's forward charge, plus the placement safety charges), the
    /// value `ep_vram_layers` is given.
    pub reserve_bytes: u64,
    /// One rank's AR lane store, held back as in `ep_vram_layers_with_lanes`.
    pub lane_bytes: u64,
    /// Aggregate host-memory allowance (all ranks together) for the new
    /// scratch-donor mirrors.
    pub host_budget_bytes: u64,
    /// Plans resolving to fewer resident layers per rank are refused.
    pub min_resident_layers: usize,
}

/// The frozen, physical-pointer-free plan every reader consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EpReservePlan {
    pub forward: EpReserveForwardPlan,
    /// Resident trunk layers per rank (all ranks agree).
    pub resident_layers: usize,
    /// First scratch-donor layer `H = resident_layers - donor_layers.len()`;
    /// the borrowed-view first host layer.
    pub donor_first_layer: usize,
    /// Scratch donors `[H, resident_layers)`, ascending.
    pub donor_layers: Vec<usize>,
    /// S2 donors `H-2`, `H-1`, ordered by parity (even layer first).
    pub s2_donor_layers: [usize; EP_RESERVE_S2_DONORS],
    /// Segments, ascending layer, gate/up before down.
    pub segments: Vec<EpReserveSegment>,
    pub(crate) placements: Vec<EpReserveRootPlacement>,
    /// Used prefix of each used segment, in segment order.
    pub touched: Vec<EpReserveTouchedExtent>,
    /// Pinned scratch-donor mirror bytes of one rank (full donor tensors).
    pub mirror_bytes_per_rank: u64,
    /// Mirror bytes of all ranks together (checked against the host budget).
    pub mirror_bytes_total: u64,
    /// Bytes one restore moves per rank: the touched extents.
    pub restore_bytes_per_rank: u64,
    pub trunk_layers: usize,
    pub reserve_bytes: u64,
    /// Logical digest (no pointers) of everything ranks must agree on.
    pub digest: u64,
}

fn rank_layers(
    geometry: &EpRankGeometry,
    free: u64,
    non_expert: u64,
    layer_bytes: u64,
    reserve: u64,
) -> usize {
    let all = layer_bytes.saturating_mul(geometry.moe_layers as u64);
    if crate::expert_residency::fits_fully_resident(free, non_expert, all, reserve) {
        return geometry.moe_layers;
    }
    crate::expert_residency::auto_vram_layers(
        free,
        non_expert,
        layer_bytes,
        geometry.trunk_layers,
        reserve,
    )
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn digest_of(plan: &EpReservePlan) -> u64 {
    let mut h = Fnv::new();
    h.u64(plan.forward.prompt_rows as u64);
    h.u64(plan.forward.resident_rows as u64);
    h.u64(plan.forward.dedicated_bytes().unwrap_or(u64::MAX));
    h.u64(plan.forward.fixed_table_bytes);
    h.u64(plan.forward.spec_bytes);
    for root in plan
        .forward
        .external_roots
        .iter()
        .chain(&plan.forward.resident_roots)
        .chain(&plan.forward.remaining_roots)
    {
        h.u64(root.field as usize as u64);
        h.u64(root.bytes);
    }
    h.u64(plan.resident_layers as u64);
    h.u64(plan.trunk_layers as u64);
    h.u64(plan.donor_first_layer as u64);
    for layer in plan.s2_donor_layers {
        h.u64(layer as u64);
    }
    for segment in &plan.segments {
        h.u64(segment.layer as u64);
        h.u64(segment.slot as u64);
        h.u64(segment.capacity);
    }
    for placement in &plan.placements {
        h.u64(placement.field as usize as u64);
        h.u64(placement.segment as u64);
        h.u64(placement.offset);
        h.u64(placement.bytes);
    }
    for extent in &plan.touched {
        h.u64(extent.layer as u64);
        h.u64(extent.slot as u64);
        h.u64(extent.offset);
        h.u64(extent.len);
    }
    h.u64(plan.mirror_bytes_per_rank);
    h.0
}

/// CPU-only plan: the resident layer count at the minimum-rank fit, the
/// smallest donor suffix whose whole-layer segments hold every remaining root,
/// the S2 donors, the host mirror aggregate and the logical digest. `Err`
/// refuses before any allocation; the same result feeds every reserve reader.
pub fn plan_ep_reserve(
    forward: &EpReserveForwardPlan,
    geometry: &EpRankGeometry,
    capacity: &EpReserveCapacity,
) -> Result<EpReservePlan, Refusal> {
    if geometry.ranks == 0 || capacity.free_per_rank.len() != geometry.ranks {
        return Err(Refusal::new("free memory is not given for every rank"));
    }
    if geometry.trunk_layers == 0 || geometry.moe_layers < geometry.trunk_layers {
        return Err(Refusal::new("invalid layer geometry"));
    }
    let layer_bytes = geometry
        .layer_bytes()
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| Refusal::new("layer expert bytes overflow or are zero"))?;
    // Same reserve plus lane store the ordinary split keeps free.
    let reserve = capacity
        .reserve_bytes
        .checked_add(capacity.lane_bytes)
        .ok_or_else(|| Refusal::new("reserve plus lane store overflows"))?;
    let resident = capacity
        .free_per_rank
        .iter()
        .map(|&free| {
            rank_layers(
                geometry,
                free,
                capacity.non_expert_bytes,
                layer_bytes,
                reserve,
            )
        })
        .min()
        .unwrap_or(0);
    if resident >= geometry.trunk_layers {
        return Err(Refusal::new(
            "every trunk layer is resident without sharing; nothing to decouple",
        ));
    }
    if resident < capacity.min_resident_layers {
        return Err(Refusal::new(format!(
            "only {resident} resident layers fit per rank (minimum {})",
            capacity.min_resident_layers
        )));
    }
    let roots: Vec<(ScratchField, u64)> = forward
        .remaining_roots
        .iter()
        .map(|root| (root.field, root.bytes))
        .collect();
    let needed = forward.remaining_bytes()?;
    let max_donors = resident.saturating_sub(EP_RESERVE_S2_DONORS);
    let mut found = None;
    for donors in 1..=max_donors {
        let first = resident - donors;
        let mut segments = Vec::with_capacity(donors * 2);
        for layer in first..resident {
            segments.push(EpReserveSegment {
                layer,
                slot: EpDonorSlot::GateUp,
                capacity: geometry.gate_up_bytes,
            });
            segments.push(EpReserveSegment {
                layer,
                slot: EpDonorSlot::Down,
                capacity: geometry.down_bytes,
            });
        }
        let caps: Vec<u64> = segments.iter().map(|segment| segment.capacity).collect();
        if checked_sum(caps.iter().copied()).map_or(true, |total| total < needed) {
            continue;
        }
        if let Some(placed) = pack_roots(&roots, &caps) {
            found = Some((first, segments, placed));
            break;
        }
    }
    let (first, segments, placed) = found.ok_or_else(|| {
        Refusal::new(format!(
            "the remaining roots ({} MiB) do not pack into whole-layer donors within {max_donors} layers",
            needed / MIB
        ))
    })?;
    let donors = resident - first;
    let placements: Vec<EpReserveRootPlacement> = placed
        .iter()
        .map(|&(field, segment, offset, bytes)| EpReserveRootPlacement {
            field,
            segment,
            layer: segments[segment].layer,
            slot: segments[segment].slot,
            offset,
            bytes,
        })
        .collect();
    let mut touched = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let used = placements
            .iter()
            .filter(|placement| placement.segment == index)
            .map(|placement| placement.offset + placement.bytes)
            .max();
        if let Some(used) = used {
            let len = align_up(used)
                .ok_or_else(|| Refusal::new("touched extent overflows"))?
                .min(segment.capacity);
            touched.push(EpReserveTouchedExtent {
                layer: segment.layer,
                slot: segment.slot,
                offset: 0,
                len,
            });
        }
    }
    let overflow = || Refusal::new("mirror byte accounting overflows");
    let mirror_bytes_per_rank = layer_bytes
        .checked_mul(donors as u64)
        .ok_or_else(overflow)?;
    let mirror_bytes_total = mirror_bytes_per_rank
        .checked_mul(geometry.ranks as u64)
        .ok_or_else(overflow)?;
    if mirror_bytes_total > capacity.host_budget_bytes {
        return Err(Refusal::new(format!(
            "scratch-donor mirrors need {} MiB of pinned host memory across {} ranks; {} MiB allowed",
            mirror_bytes_total / MIB,
            geometry.ranks,
            capacity.host_budget_bytes / MIB
        )));
    }
    let restore_bytes_per_rank =
        checked_sum(touched.iter().map(|extent| extent.len)).ok_or_else(overflow)?;
    let mut s2 = [first - 2, first - 1];
    if s2[0] % 2 != 0 {
        s2.swap(0, 1);
    }
    let mut plan = EpReservePlan {
        forward: forward.clone(),
        resident_layers: resident,
        donor_first_layer: first,
        donor_layers: (first..resident).collect(),
        s2_donor_layers: s2,
        segments,
        placements,
        touched,
        mirror_bytes_per_rank,
        mirror_bytes_total,
        restore_bytes_per_rank,
        trunk_layers: geometry.trunk_layers,
        reserve_bytes: capacity.reserve_bytes,
        digest: 0,
    };
    plan.digest = digest_of(&plan);
    Ok(plan)
}

impl EpReservePlan {
    /// The receipt every rank publishes at attach; `source_generation` is the
    /// weight/seal generation, `s1`/`s2` the admitted route bits.
    pub fn mesh_receipt(
        &self,
        source_generation: u64,
        s1_admitted: bool,
        s2_admitted: bool,
    ) -> EpReserveMeshReceipt {
        EpReserveMeshReceipt {
            source_generation,
            plan_digest: self.digest,
            resident_layers: self.resident_layers,
            donor_first_layer: self.donor_first_layer,
            donor_count: self.donor_layers.len(),
            s2_donor_layers: self.s2_donor_layers,
            prompt_rows: self.forward.prompt_rows,
            exchange_rows: self.forward.prompt_rows,
            s1_admitted,
            s2_admitted,
        }
    }

    /// One log line of the plan.
    pub fn describe(&self) -> String {
        format!(
            "EP reserve decoupling: {} resident layers/rank, scratch donors {}..{} ({} layers, {} segments), \
             S2 donors {:?}, forward charge {} MiB, mirrors {} MiB/rank ({} MiB total), restore {} MiB/rank, digest {:016x}",
            self.resident_layers,
            self.donor_first_layer,
            self.resident_layers,
            self.donor_layers.len(),
            self.segments.len(),
            self.s2_donor_layers,
            self.forward.charged_forward_bytes().unwrap_or(0) / MIB,
            self.mirror_bytes_per_rank / MIB,
            self.mirror_bytes_total / MIB,
            self.restore_bytes_per_rank / MIB,
            self.digest
        )
    }
}

// ---------------------------------------------------------------------------
// Mesh receipt
// ---------------------------------------------------------------------------

/// What every rank must agree on before peer publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpReserveMeshReceipt {
    pub source_generation: u64,
    pub plan_digest: u64,
    pub resident_layers: usize,
    pub donor_first_layer: usize,
    pub donor_count: usize,
    pub s2_donor_layers: [usize; EP_RESERVE_S2_DONORS],
    pub prompt_rows: usize,
    pub exchange_rows: usize,
    pub s1_admitted: bool,
    pub s2_admitted: bool,
}

impl EpReserveMeshReceipt {
    /// `Ok` when `other` is identical; otherwise a refusal naming the first
    /// differing field. No rank falls back to ordinary after reducing the reserve.
    pub fn agree(&self, other: &Self) -> Result<(), Refusal> {
        macro_rules! field {
            ($name:ident) => {
                if self.$name != other.$name {
                    return Err(Refusal::new(format!(
                        "ranks disagree on {}: {:?} vs {:?}",
                        stringify!($name),
                        self.$name,
                        other.$name
                    )));
                }
            };
        }
        field!(source_generation);
        field!(plan_digest);
        field!(resident_layers);
        field!(donor_first_layer);
        field!(donor_count);
        field!(s2_donor_layers);
        field!(prompt_rows);
        field!(exchange_rows);
        field!(s1_admitted);
        field!(s2_admitted);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::compact_test_config;
    use crate::expert_residency::{auto_vram_reserve_with_forward, state_census_with_forward};
    use ScratchField as F;

    const GIB: u64 = 1 << 30;

    fn facts() -> EpReserveFacts<'static> {
        EpReserveFacts {
            arch: "gfx1201",
            uma: false,
            flash_next: true,
            expert_qt: Some((44, 53)),
            ranks: 2,
            scratch_mode: Qwen4ForwardScratchMode::Full,
            prompt_rows: 4096,
            explicit_chunk_override: false,
            whole_layer_placement: true,
            owner_routed_s1: true,
            s2_min_rows: Some(2048),
            lane_request: false,
            capture: false,
            routing_override: false,
            pipeline_parallel: false,
            expert_tier: false,
            expert_slab: false,
            npu: false,
            dflash: false,
            single_device_slowlink: false,
            continuous_batch: false,
        }
    }

    fn forward() -> EpReserveForwardPlan {
        EpReserveForwardPlan::from_config(&compact_test_config(), Qwen4ForwardScratchMode::Full, 0)
            .expect("pinned forward plan")
    }

    fn geometry() -> EpRankGeometry {
        EpRankGeometry::flash_next(&compact_test_config(), 2).expect("geometry")
    }

    /// Capacity where `layers` resident layers fit exactly on every rank.
    fn capacity_for(layers: usize, extra: u64) -> EpReserveCapacity {
        let layer = geometry().layer_bytes().unwrap();
        let (non_expert, reserve) = (6 * GIB, 6 * GIB);
        let free = non_expert + reserve + layer * layers as u64 + extra;
        EpReserveCapacity {
            free_per_rank: vec![free, free],
            non_expert_bytes: non_expert,
            reserve_bytes: reserve,
            lane_bytes: 0,
            host_budget_bytes: 64 * GIB,
            min_resident_layers: EP_RESERVE_MIN_RESIDENT_LAYERS,
        }
    }

    #[test]
    fn admission_accepts_the_pinned_scope_and_refuses_everything_else() {
        assert_eq!(admit_ep_reserve(&facts()), Ok(()));
        let cases: Vec<(&str, Box<dyn Fn(&mut EpReserveFacts<'static>)>)> = vec![
            ("arch", Box::new(|f| f.arch = "gfx1100")),
            ("uma", Box::new(|f| f.uma = true)),
            ("profile", Box::new(|f| f.flash_next = false)),
            ("recipe", Box::new(|f| f.expert_qt = Some((44, 44)))),
            ("no recipe", Box::new(|f| f.expert_qt = None)),
            ("ranks 1", Box::new(|f| f.ranks = 1)),
            ("ranks 4", Box::new(|f| f.ranks = 4)),
            (
                "mode",
                Box::new(|f| f.scratch_mode = Qwen4ForwardScratchMode::FnSymGfx1201),
            ),
            ("chunk", Box::new(|f| f.prompt_rows = 1024)),
            (
                "explicit chunk",
                Box::new(|f| f.explicit_chunk_override = true),
            ),
            ("placement", Box::new(|f| f.whole_layer_placement = false)),
            ("pp", Box::new(|f| f.pipeline_parallel = true)),
            ("tier", Box::new(|f| f.expert_tier = true)),
            ("slab", Box::new(|f| f.expert_slab = true)),
            ("npu", Box::new(|f| f.npu = true)),
            ("dflash", Box::new(|f| f.dflash = true)),
            ("slowlink", Box::new(|f| f.single_device_slowlink = true)),
            ("cb", Box::new(|f| f.continuous_batch = true)),
            ("lanes", Box::new(|f| f.lane_request = true)),
            ("capture", Box::new(|f| f.capture = true)),
            ("routing", Box::new(|f| f.routing_override = true)),
            ("s2 rows", Box::new(|f| f.s2_min_rows = Some(512))),
        ];
        for (name, mutate) in cases {
            let mut f = facts();
            mutate(&mut f);
            assert!(admit_ep_reserve(&f).is_err(), "{name} must refuse");
        }
        // S2 off and S1 off are admitted scope choices, not refusals.
        let mut f = facts();
        f.s2_min_rows = None;
        f.owner_routed_s1 = false;
        assert_eq!(admit_ep_reserve(&f), Ok(()));
    }

    #[test]
    fn switch_is_default_on_with_a_zero_kill_switch() {
        use EpReserveRequest::{Default, Forced, Off};
        assert_eq!(ep_reserve_request(None), Default);
        assert_eq!(ep_reserve_request(Some("")), Default);
        assert_eq!(ep_reserve_request(Some("true")), Default);
        assert_eq!(ep_reserve_request(Some("0")), Off);
        assert_eq!(ep_reserve_request(Some(" 0 ")), Off);
        assert_eq!(ep_reserve_request(Some("1")), Forced);
        assert_eq!(ep_reserve_request(Some(" 1 ")), Forced);
        assert!(Default.attempts() && Forced.attempts() && !Off.attempts());
    }

    #[test]
    fn persistent_roots_are_475_and_125_mib() {
        let fwd = forward();
        assert_eq!(fwd.persistent_prompt_bytes(F::MoeYDownGrouped), 475 * MIB);
        assert_eq!(fwd.persistent_resident_bytes(F::MoeYDownGrouped), 125 * MIB);
        assert!(
            fwd.persistent_prompt_bytes(F::TokenIds) >= fwd.persistent_resident_bytes(F::TokenIds)
        );
        // Neither persistent root is a donor-packing input.
        assert!(fwd
            .remaining_roots
            .iter()
            .all(|root| !is_persistent(root.field)));
        // A tensor root cannot cross allocations: the 475 MiB root fits no
        // single donor segment (425 MiB gate/up, 212.5 MiB down).
        let g = geometry();
        assert_eq!(g.gate_up_bytes, 425 * MIB);
        assert_eq!(g.down_bytes, 425 * MIB / 2);
        assert!(475 * MIB > g.gate_up_bytes);
    }

    #[test]
    fn dedicated_formula_is_exact_and_allocations_sum_to_the_charge() {
        let fwd = forward();
        let config = compact_test_config();
        let mode = Qwen4ForwardScratchMode::Full;
        let full512: u64 = ep_reserve_scratch(&config, mode, 512)
            .unwrap()
            .iter()
            .map(|r| r.1)
            .sum();
        let at = |rows, field| {
            ep_reserve_scratch(&config, mode, rows)
                .unwrap()
                .into_iter()
                .find(|r| r.0 == field)
                .unwrap()
                .1
        };
        let expected = full512 - at(512, F::TokenIds) - at(512, F::MoeYDownGrouped)
            + at(4096, F::TokenIds)
            + at(4096, F::MoeYDownGrouped)
            + ep_reserve_spec_bytes(&config, 4096).unwrap();
        assert_eq!(fwd.dedicated_bytes().unwrap(), expected);
        let sum: u64 = fwd.allocations().iter().map(|a| a.bytes).sum();
        assert_eq!(sum, fwd.charged_forward_bytes().unwrap());
        // Fixed table charges are added once, never double-counting roots.
        let with_tables = EpReserveForwardPlan::from_config(&config, mode, 7 * MIB).unwrap();
        assert_eq!(
            with_tables.charged_forward_bytes().unwrap(),
            expected + 7 * MIB
        );
    }

    #[test]
    fn historical_floor_and_census_retain_non_forward_4096_terms() {
        let config = compact_test_config();
        let mode = Qwen4ForwardScratchMode::Full;
        let fwd = forward();
        let charged = fwd.charged_forward_bytes().unwrap();
        let baseline = crate::gpu_forward::qwen4_forward_device_bytes(&config, 4096, mode).unwrap();
        assert!(charged < baseline);
        let context = crate::Qwen4ContextCommit::legacy(16384);
        let layout = crate::Qwen4StateFormat::F32.layout();
        let ordinary =
            auto_vram_reserve_with_forward(&config, &context, 4096, mode, layout, None, None, None)
                .unwrap();
        let decoupled = auto_vram_reserve_with_forward(
            &config,
            &context,
            4096,
            mode,
            layout,
            None,
            None,
            Some(charged),
        )
        .unwrap();
        assert_eq!(ordinary - decoupled, baseline - charged);
        let census = |forward| {
            state_census_with_forward(
                &config,
                crate::Qwen4StateFormat::F32,
                &context,
                4096,
                mode,
                forward,
            )
            .unwrap()
        };
        let (a, b) = (census(None), census(Some(charged)));
        assert_eq!(a.qsa_context, b.qsa_context);
        assert_eq!(a.state, b.state);
        assert_eq!(a.forward, baseline);
        assert_eq!(b.forward, charged);
    }

    #[test]
    fn packing_never_straddles_or_overlaps_and_is_aligned() {
        let fwd = forward();
        let plan = plan_ep_reserve(&fwd, &geometry(), &capacity_for(33, 0)).unwrap();
        assert!(!plan.placements.is_empty());
        let mut by_segment: Vec<Vec<&EpReserveRootPlacement>> = vec![vec![]; plan.segments.len()];
        for placement in &plan.placements {
            by_segment[placement.segment].push(placement);
            let segment = plan.segments[placement.segment];
            assert_eq!(placement.offset % EP_RESERVE_ROOT_ALIGN, 0);
            assert!(placement.offset + placement.bytes <= segment.capacity);
            assert_eq!(
                (placement.layer, placement.slot),
                (segment.layer, segment.slot)
            );
        }
        for list in &by_segment {
            for pair in list.windows(2) {
                assert!(pair[0].offset + pair[0].bytes <= pair[1].offset);
            }
        }
        // Every remaining root is placed exactly once, nothing else is.
        let mut placed: Vec<usize> = plan.placements.iter().map(|p| p.field as usize).collect();
        let mut wanted: Vec<usize> = fwd
            .remaining_roots
            .iter()
            .map(|r| r.field as usize)
            .collect();
        placed.sort();
        wanted.sort();
        assert_eq!(placed, wanted);
        // Touched extents cover exactly the used prefixes, in segment order.
        for extent in &plan.touched {
            let end = plan
                .placements
                .iter()
                .filter(|p| (p.layer, p.slot) == (extent.layer, extent.slot))
                .map(|p| p.offset + p.bytes)
                .max()
                .unwrap();
            assert_eq!(extent.offset, 0);
            assert!(extent.len >= end && extent.len < end + EP_RESERVE_ROOT_ALIGN);
        }
        assert_eq!(
            plan.restore_bytes_per_rank,
            plan.touched.iter().map(|e| e.len).sum::<u64>()
        );
    }

    #[test]
    fn pack_roots_fails_closed() {
        // A root larger than every segment, an overflow and an empty segment set.
        assert!(pack_roots(&[(F::MoeGateUp, 10)], &[5]).is_none());
        assert!(pack_roots(&[(F::MoeGateUp, 10)], &[]).is_none());
        assert!(pack_roots(&[(F::MoeGateUp, u64::MAX)], &[u64::MAX]).is_none());
        // Zero-byte roots are skipped; alignment advances the next start.
        let placed = pack_roots(
            &[(F::MoeUp, 10), (F::MoeGate, 20), (F::MoeHidden, 0)],
            &[1024],
        )
        .unwrap();
        assert_eq!(placed, vec![(F::MoeGate, 0, 0, 20), (F::MoeUp, 0, 256, 10)]);
        // No straddling: two 600-byte roots cannot share one 1024-byte segment.
        assert!(pack_roots(&[(F::MoeGate, 600), (F::MoeUp, 600)], &[1024]).is_none());
        assert!(pack_roots(&[(F::MoeGate, 600), (F::MoeUp, 600)], &[1024, 1024]).is_some());
    }

    #[test]
    fn plan_is_deterministic_with_a_logical_digest() {
        let fwd = forward();
        let a = plan_ep_reserve(&fwd, &geometry(), &capacity_for(33, 0)).unwrap();
        let b = plan_ep_reserve(&fwd, &geometry(), &capacity_for(33, 0)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.digest, b.digest);
        assert_ne!(a.digest, 0);
        // A different resident count, charge or table budget changes it.
        let c = plan_ep_reserve(&fwd, &geometry(), &capacity_for(34, 0)).unwrap();
        assert_ne!(a.digest, c.digest);
        let tables = EpReserveForwardPlan::from_config(
            &compact_test_config(),
            Qwen4ForwardScratchMode::Full,
            MIB,
        )
        .unwrap();
        let d = plan_ep_reserve(&tables, &geometry(), &capacity_for(33, 0)).unwrap();
        assert_ne!(a.digest, d.digest);
        // The receipt carries the digest and the boundaries.
        let r = a.mesh_receipt(7, true, true);
        assert_eq!(r.plan_digest, a.digest);
        assert_eq!(r.exchange_rows, 4096);
        assert_eq!(r.resident_layers, 33);
    }

    #[test]
    fn pinned_geometry_needs_seven_donors_and_fixes_the_boundaries() {
        // Source-derived bounds: six EP2 donor layers hold 3825 MiB, less than
        // the 3853.238 MiB of remaining roots; seven pack with slack.
        let fwd = forward();
        let g = geometry();
        assert!(fwd.remaining_bytes().unwrap() > 6 * g.layer_bytes().unwrap());
        assert!(fwd.remaining_bytes().unwrap() < 7 * g.layer_bytes().unwrap());
        let plan = plan_ep_reserve(&fwd, &g, &capacity_for(33, 0)).unwrap();
        assert_eq!(plan.resident_layers, 33);
        assert_eq!(plan.donor_layers.len(), 7);
        assert_eq!(plan.donor_layers, (26..33).collect::<Vec<_>>());
        assert_eq!(plan.donor_first_layer, 26);
        assert_eq!(plan.segments.len(), 14);
        let mut s2 = plan.s2_donor_layers;
        s2.sort();
        assert_eq!(s2, [24, 25]);
        assert_eq!(plan.s2_donor_layers[0] % 2, 0);
        // S2 donors are disjoint from scratch donors.
        assert!(plan
            .s2_donor_layers
            .iter()
            .all(|l| !plan.donor_layers.contains(l)));
        assert_eq!(plan.mirror_bytes_per_rank, 7 * g.layer_bytes().unwrap());
        // Four layers cannot be proven by six donors: a smaller packing set
        // refuses (D=6 would need exactly these roots in 12 segments).
        let caps: Vec<u64> = (0..6)
            .flat_map(|_| [g.gate_up_bytes, g.down_bytes])
            .collect();
        let roots: Vec<_> = fwd
            .remaining_roots
            .iter()
            .map(|r| (r.field, r.bytes))
            .collect();
        assert!(pack_roots(&roots, &caps).is_none());
        let caps: Vec<u64> = (0..7)
            .flat_map(|_| [g.gate_up_bytes, g.down_bytes])
            .collect();
        assert!(pack_roots(&roots, &caps).is_some());
    }

    #[test]
    fn unequal_free_memory_uses_the_minimum_rank_fit() {
        let fwd = forward();
        let mut cap = capacity_for(34, 0);
        let small = capacity_for(33, 0).free_per_rank[0];
        cap.free_per_rank = vec![cap.free_per_rank[0], small];
        let plan = plan_ep_reserve(&fwd, &geometry(), &cap).unwrap();
        assert_eq!(plan.resident_layers, 33);
        cap.free_per_rank.reverse();
        assert_eq!(
            plan_ep_reserve(&fwd, &geometry(), &cap)
                .unwrap()
                .resident_layers,
            33
        );
        // Wrong rank count fails closed.
        cap.free_per_rank.pop();
        assert!(plan_ep_reserve(&fwd, &geometry(), &cap).is_err());
    }

    #[test]
    fn prediction_reaches_33_only_when_complete_charges_fit() {
        let fwd = forward();
        let g = geometry();
        let layer = g.layer_bytes().unwrap();
        // One byte short of 33 layers: fewer than the minimum, refused.
        let mut cap = capacity_for(33, 0);
        for free in &mut cap.free_per_rank {
            *free -= 1;
        }
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        // Extra reserve charges (tables, lanes) push below 33: refused.
        let mut cap = capacity_for(33, 0);
        cap.lane_bytes = 1;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        let mut cap = capacity_for(33, 0);
        cap.reserve_bytes += layer;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        // Slack below one layer keeps 33.
        assert_eq!(
            plan_ep_reserve(&fwd, &g, &capacity_for(33, layer - 1))
                .unwrap()
                .resident_layers,
            33
        );
        // Fully resident trunk: nothing to decouple.
        assert!(plan_ep_reserve(&fwd, &g, &capacity_for(60, 0)).is_err());
        // Too few layers for the donors plus two S2 donors.
        let mut cap = capacity_for(8, 0);
        cap.min_resident_layers = 1;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
    }

    #[test]
    fn host_budget_aggregates_both_ranks() {
        let fwd = forward();
        let g = geometry();
        let per_rank = 7 * g.layer_bytes().unwrap();
        let mut cap = capacity_for(33, 0);
        cap.host_budget_bytes = per_rank; // one rank's worth is not enough for two
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        cap.host_budget_bytes = 2 * per_rank - 1;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        cap.host_budget_bytes = 2 * per_rank;
        let plan = plan_ep_reserve(&fwd, &g, &cap).unwrap();
        assert_eq!(plan.mirror_bytes_total, 2 * per_rank);
    }

    #[test]
    fn overflow_and_malformed_roots_fail_closed() {
        let fwd = forward();
        let g = geometry();
        let mut cap = capacity_for(33, 0);
        cap.reserve_bytes = u64::MAX;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        let mut big = fwd.clone();
        big.remaining_roots.push(EpReserveRoot {
            field: F::MoeGateUp,
            bytes: u64::MAX,
        });
        assert!(plan_ep_reserve(&big, &g, &capacity_for(33, 0)).is_err());
        let mut huge = fwd.clone();
        huge.spec_bytes = u64::MAX;
        assert!(huge.dedicated_bytes().is_err());
        // A root larger than any segment refuses even with arbitrarily many donors.
        let mut wide = fwd.clone();
        wide.remaining_roots.push(EpReserveRoot {
            field: F::MoeGateUp,
            bytes: 475 * MIB,
        });
        assert!(plan_ep_reserve(&wide, &g, &capacity_for(40, 0)).is_err());
        // Persistence flags and duplicates are validated.
        let specs = [(F::TokenIds, 8, false), (F::MoeYDownGrouped, 8, true)];
        assert!(split_specs(&specs, 1).is_err());
        let dup = [
            (F::TokenIds, 8, true),
            (F::TokenIds, 8, true),
            (F::MoeYDownGrouped, 8, true),
        ];
        assert!(split_specs(&dup, 1).is_err());
        let missing = [(F::TokenIds, 8, true)];
        assert!(split_specs(&missing, 1).is_err());
        let full = ep_reserve_scratch(&compact_test_config(), Qwen4ForwardScratchMode::Full, 4096)
            .unwrap();
        assert!(EpReserveForwardPlan::from_specs(
            Qwen4ForwardScratchMode::FnSymGfx1201,
            4096,
            512,
            &full,
            &full,
            0,
            0
        )
        .is_err());
    }

    #[test]
    fn mesh_receipts_must_agree_field_by_field() {
        let plan = plan_ep_reserve(&forward(), &geometry(), &capacity_for(33, 0)).unwrap();
        let a = plan.mesh_receipt(1, true, true);
        assert_eq!(a.agree(&a), Ok(()));
        let mut b = a;
        b.plan_digest ^= 1;
        assert!(a.agree(&b).unwrap_err().0.contains("plan_digest"));
        let mut b = a;
        b.s2_admitted = false;
        assert!(a.agree(&b).unwrap_err().0.contains("s2_admitted"));
        let mut b = a;
        b.resident_layers = 34;
        assert!(a.agree(&b).is_err());
        let mut b = a;
        b.source_generation = 2;
        assert!(a.agree(&b).is_err());
        assert!(plan.describe().contains("33 resident layers"));
    }
}
