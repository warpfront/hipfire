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
//!
//! Under the Flash-Next trunk shard (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`) the plan is still ONE plan
//! shared by both ranks, so its digest, the mesh receipt and the census/floor charge are one
//! figure; each rank sizes its scratch from its own rank-local config, and the plan is the
//! root-wise maximum of the ranks' own plans ([`EpReserveForwardPlan::from_config_for_shard`]),
//! so [`EpReserveForwardPlan::covers_local`] holds for every rank (the banks place a smaller
//! local root at its planned offset). That is a few percent below the plan of the unsharded
//! global config, which would also bound every rank but reserve (and charge) more.

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

/// Switch (default ON, exact `0` is the kill switch): a native-MTP EP lane
/// store (`--spec mtp`, the mixed store) loads on the decoupled-reserve
/// (route B) placement together with the AR lanes' placement. Off (`0`) the
/// loader skips decoupling for a load that stages spec lanes (ordinary
/// placement and reserve, the pre-share sizing) and the EP lane staging
/// refuses a spec store on a reserve-sharing load (the daemon then stages AR
/// lanes). Developer variable; read through the process configuration.
pub const EP_MTP_RESERVE_SHARE_ENV: &str = "HIPFIRE_EP_MTP_RESERVE_SHARE";

/// [`EP_MTP_RESERVE_SHARE_ENV`] (unset or `1` on; exact `0` off).
pub fn ep_mtp_reserve_share_enabled() -> bool {
    hipfire_config::developer_bool(EP_MTP_RESERVE_SHARE_ENV, true)
}

/// Whether a spec lane store keeps the decoupled reserve off a load (the
/// load-time half): `Some(reason)` when MTP lanes will be staged
/// (`spec_k` set and at least one MTP slot of `lanes`) and the kill switch
/// (`HIPFIRE_EP_MTP_RESERVE_SHARE=0`) turned the sharing off.
pub fn spec_store_blocks_reserve(
    spec_k: Option<usize>,
    spec_slots: usize,
    share_enabled: bool,
) -> Option<&'static str> {
    (spec_k.is_some() && spec_slots > 0 && !share_enabled).then_some(
        "native-MTP EP lanes are staged (`--spec mtp`) and HIPFIRE_EP_MTP_RESERVE_SHARE=0 keeps them off the decoupled reserve (it admits AR lanes only then); pass no `--spec mtp` (or HIPFIRE_EP_MTP_LANES=0) for 32-layer AR lanes, or unset HIPFIRE_EP_MTP_RESERVE_SHARE to run MTP lanes on it",
    )
}

/// The stage-time half: whether a spec lane store (`row_budget` decode rows
/// per step) may be staged on a reserve-sharing load whose resident bank holds
/// `resident_rows`. Lane decode steps and every mesh verify window run on the
/// resident bank only, so a step wider than it is refused here, labeled,
/// instead of at the first step.
pub fn spec_store_reserve_verdict(
    reserve_active: bool,
    share_enabled: bool,
    row_budget: usize,
    resident_rows: Option<usize>,
) -> Result<(), String> {
    if !reserve_active {
        return Ok(());
    }
    if !share_enabled {
        return Err(
            "native-MTP lanes are refused on an EP reserve-sharing load while \
             HIPFIRE_EP_MTP_RESERVE_SHARE=0 (AR lanes are admitted; load without \
             `--spec mtp`, set HIPFIRE_EP_MTP_LANES=0, or unset \
             HIPFIRE_EP_MTP_RESERVE_SHARE)"
                .to_string(),
        );
    }
    match resident_rows {
        Some(rows) if row_budget <= rows => Ok(()),
        Some(rows) => Err(format!(
            "native-MTP lanes on an EP reserve-sharing load run every window on the {rows}-row \
             resident scratch bank; a {row_budget}-row step budget does not fit"
        )),
        None => Err(
            "native-MTP lanes on an EP reserve-sharing load: the resident scratch bank is not \
             attached"
                .to_string(),
        ),
    }
}

/// The mesh-window start guard: a mesh decode window (draft rounds, verify,
/// prepare) never opens or crosses a borrowed-bank epoch (prompt tiles are
/// their own steps and restore the donors before returning), so an open epoch
/// at a window start is a bracket bug and refuses the window.
pub fn mesh_window_epoch_check(epoch_open: bool) -> Result<(), String> {
    if epoch_open {
        return Err(
            "an EP reserve borrowed-bank epoch is open at a mesh lane window start (a prompt \
             tile's donors were not restored); refusing to run the native-MTP window on dirty \
             donor layers"
                .to_string(),
        );
    }
    Ok(())
}

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
/// One expert's gate/up bytes in the drop-zero planar layout (`MQ4G256V2P`, DESIGN F1).
pub const EP2_PLANAR_GATE_UP_EXPERT_BYTES: u64 = 1_689_600;
/// One expert's down bytes in the drop-zero planar layout (`MQ4G128V2P`).
pub const EP2_PLANAR_DOWN_EXPERT_BYTES: u64 = 844_800;

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
    ///
    /// Under the trunk shard (`HIPFIRE_QWEN4_EP_TRUNK_SHARD`) the plan is STILL one plan for
    /// both ranks, but each rank sizes its scratch from its own rank-local config: the plan is
    /// the root-wise maximum of every rank's own plan ([`Self::from_config_for_shard`]), the
    /// tightest single plan every rank fits in ([`Self::covers_local`]).
    pub fn from_config(
        config: &Qwen4Config,
        mode: Qwen4ForwardScratchMode,
        fixed_table_bytes: u64,
    ) -> Result<Self, Refusal> {
        Self::from_config_for_shard(
            config,
            mode,
            fixed_table_bytes,
            crate::weights::trunk_shard::trunk_shard_requested(),
        )
    }

    /// [`Self::from_config`] with the shard's verdict passed in. `shard` false is exactly the
    /// plan of `config`, byte for byte what a load without the shard makes. With `shard`, the
    /// plan is the root-wise maximum of the plans of the rank-local configs of every rank of
    /// the admitted topology (rank 0 shared width 384, rank 1 shared width 256; the heads
    /// halved identically): ONE plan, so one digest, one mesh receipt and one census charge,
    /// no larger than the plan of the global config and every rank's own plan fits inside it.
    /// A geometry the shard cannot cut, or ranks whose root sets differ, refuses the plan (the
    /// load keeps the ordinary placement with the reason).
    pub fn from_config_for_shard(
        config: &Qwen4Config,
        mode: Qwen4ForwardScratchMode,
        fixed_table_bytes: u64,
        shard: bool,
    ) -> Result<Self, Refusal> {
        if config.trunk_shard.is_some() {
            return Err(Refusal::new(
                "a trunk-shard rank-local config cannot seed the shared reserve plan: it is cut \
                 from the global config",
            ));
        }
        if !shard {
            return Self::from_config_exact(config, mode, fixed_table_bytes);
        }
        let mut union: Option<Self> = None;
        for rank in 0..EP_RESERVE_RANKS {
            let local = config
                .trunk_shard_local(rank, EP_RESERVE_RANKS)
                .map_err(|error| {
                    Refusal::new(format!("trunk shard rank {rank} local config: {error}"))
                })?;
            let own = Self::from_config_exact(&local, mode, fixed_table_bytes)
                .map_err(|refusal| Refusal::new(format!("rank {rank}: {}", refusal.0)))?;
            union = Some(match union {
                None => own,
                Some(plan) => plan
                    .union(&own)
                    .map_err(|refusal| Refusal::new(format!("rank {rank}: {}", refusal.0)))?,
            });
        }
        union.ok_or_else(|| Refusal::new("no rank to plan for"))
    }

    /// The root-wise maximum of two plans of the same rows, mode and fixed charge (the same
    /// roots in the same order), speculative bytes included. A plan covers each of its parts
    /// ([`Self::covers_local`]).
    fn union(&self, other: &Self) -> Result<Self, Refusal> {
        if (
            self.mode,
            self.prompt_rows,
            self.resident_rows,
            self.fixed_table_bytes,
        ) != (
            other.mode,
            other.prompt_rows,
            other.resident_rows,
            other.fixed_table_bytes,
        ) {
            return Err(Refusal::new(
                "the plans have other rows, mode or fixed charge",
            ));
        }
        let merge = |what: &str,
                     a: &[EpReserveRoot],
                     b: &[EpReserveRoot]|
         -> Result<Vec<EpReserveRoot>, Refusal> {
            let fields =
                |roots: &[EpReserveRoot]| roots.iter().map(|r| r.field).collect::<Vec<_>>();
            if fields(a) != fields(b) {
                return Err(Refusal::new(format!(
                    "the {what} root sets {:?} and {:?} differ",
                    fields(a),
                    fields(b)
                )));
            }
            Ok(a.iter()
                .zip(b)
                .map(|(a, b)| EpReserveRoot {
                    field: a.field,
                    bytes: a.bytes.max(b.bytes),
                })
                .collect())
        };
        Ok(Self {
            mode: self.mode,
            prompt_rows: self.prompt_rows,
            resident_rows: self.resident_rows,
            external_roots: merge(
                "persistent prompt",
                &self.external_roots,
                &other.external_roots,
            )?,
            resident_persistent: merge(
                "persistent resident",
                &self.resident_persistent,
                &other.resident_persistent,
            )?,
            resident_roots: merge("resident", &self.resident_roots, &other.resident_roots)?,
            remaining_roots: merge(
                "donor-packed",
                &self.remaining_roots,
                &other.remaining_roots,
            )?,
            spec_bytes: self.spec_bytes.max(other.spec_bytes),
            fixed_table_bytes: self.fixed_table_bytes,
        })
    }

    fn from_config_exact(
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

    /// Whether a rank built from `local` (the rank-local config of a trunk shard) fits inside
    /// this plan: the SAME rows, mode and root set (every root present in one is present in
    /// the other, persistent or not, in the same order), every root of `local`'s own plan no
    /// larger than the planned one, and no more speculative bytes. The plan then bounds what
    /// the rank allocates (resident and persistent roots, [`Self::charged_forward_bytes`]) and
    /// what it borrows from the donor segments (the placements' extents), so the one shared
    /// plan, its census charge and its digest hold for every rank while each sizes its scratch
    /// from its own config.
    pub fn covers_local(&self, local: &Qwen4Config) -> Result<(), Refusal> {
        let own = Self::from_config_exact(local, self.mode, self.fixed_table_bytes)?;
        if (own.prompt_rows, own.resident_rows) != (self.prompt_rows, self.resident_rows) {
            return Err(Refusal::new("the local plan has other row counts"));
        }
        let lists: [(&str, &[EpReserveRoot], &[EpReserveRoot]); 4] = [
            (
                "persistent prompt",
                &self.external_roots,
                &own.external_roots,
            ),
            (
                "persistent resident",
                &self.resident_persistent,
                &own.resident_persistent,
            ),
            ("resident", &self.resident_roots, &own.resident_roots),
            ("donor-packed", &self.remaining_roots, &own.remaining_roots),
        ];
        for (what, planned, wanted) in lists {
            let fields =
                |roots: &[EpReserveRoot]| roots.iter().map(|r| r.field).collect::<Vec<_>>();
            if fields(planned) != fields(wanted) {
                return Err(Refusal::new(format!(
                    "the local {what} root set {:?} is not the plan's {:?}",
                    fields(wanted),
                    fields(planned)
                )));
            }
            if let Some((planned, wanted)) = planned
                .iter()
                .zip(wanted)
                .find(|(planned, wanted)| wanted.bytes > planned.bytes)
            {
                return Err(Refusal::new(format!(
                    "the local {what} root {:?} needs {} bytes, the plan reserves {}",
                    wanted.field, wanted.bytes, planned.bytes
                )));
            }
        }
        if own.spec_bytes > self.spec_bytes {
            return Err(Refusal::new(format!(
                "the local speculative buffers need {} bytes, the plan reserves {}",
                own.spec_bytes, self.spec_bytes
            )));
        }
        Ok(())
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

/// One weight-shadow member's place inside a donor segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpReserveShadowPlacement {
    /// Index in the caller's shadow member list.
    pub index: usize,
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
        Self::flash_next_layout(config, ranks, false)
    }

    /// [`Self::flash_next`] in the shipped (`planar == false`, 2,611,200 B per expert) or the
    /// drop-zero planar (DESIGN F1, 2,534,400 B per expert) expert layout.
    pub fn flash_next_layout(config: &Qwen4Config, ranks: usize, planar: bool) -> Option<Self> {
        let local = config.num_experts.checked_add(ranks.checked_sub(1)?)? / ranks;
        let (gate_up, down) = if planar {
            (
                EP2_PLANAR_GATE_UP_EXPERT_BYTES,
                EP2_PLANAR_DOWN_EXPERT_BYTES,
            )
        } else {
            (EP2_GATE_UP_EXPERT_BYTES, EP2_DOWN_EXPERT_BYTES)
        };
        Some(Self {
            ranks,
            trunk_layers: config.num_hidden_layers,
            moe_layers: crate::expert_residency::moe_layer_count(config),
            gate_up_bytes: (local as u64).checked_mul(gate_up)?,
            down_bytes: (local as u64).checked_mul(down)?,
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
    /// The donors were chosen by the caller ([`plan_ep_reserve_with_donors`]):
    /// not a suffix of the resident prefix, no S2 donors (`s2_donor_layers`
    /// is `[0, 0]`); the unit stage serves the borrowed tiles. `false` is
    /// the suffix plan of [`plan_ep_reserve`].
    pub explicit_donors: bool,
    /// Segments, ascending layer, gate/up before down.
    pub segments: Vec<EpReserveSegment>,
    pub(crate) placements: Vec<EpReserveRootPlacement>,
    /// Used prefix of each used segment, in segment order.
    pub touched: Vec<EpReserveTouchedExtent>,
    /// Where each weight-shadow member lives in the donor segments, in the
    /// caller's member order; empty unless planned with
    /// [`plan_ep_reserve_with_donors_and_shadows`].
    pub shadow_placements: Vec<EpReserveShadowPlacement>,
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
    if plan.explicit_donors {
        // Suffix plans keep their historical digest.
        h.u64(u64::MAX);
        for layer in &plan.donor_layers {
            h.u64(*layer as u64);
        }
    }
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
    for shadow in &plan.shadow_placements {
        h.u64(shadow.index as u64);
        h.u64(shadow.segment as u64);
        h.u64(shadow.offset);
        h.u64(shadow.bytes);
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
        let layers: Vec<usize> = (resident - donors..resident).collect();
        if let Some((segments, placed)) = pack_donor_layers(&roots, needed, geometry, &layers) {
            found = Some((layers, segments, placed));
            break;
        }
    }
    let (layers, segments, placed) = found.ok_or_else(|| {
        Refusal::new(format!(
            "the remaining roots ({} MiB) do not pack into whole-layer donors within {max_donors} layers",
            needed / MIB
        ))
    })?;
    let first = layers[0];
    let mut s2 = [first - 2, first - 1];
    if s2[0] % 2 != 0 {
        s2.swap(0, 1);
    }
    assemble_plan(
        forward,
        geometry,
        capacity,
        resident,
        layers,
        s2,
        false,
        segments,
        placed,
        Vec::new(),
    )
}

/// Segments of `layers` (ascending), gate/up before down, and the packing of
/// `roots` into them, or `None` when they do not hold `needed` bytes.
#[allow(clippy::type_complexity)]
fn pack_donor_layers(
    roots: &[(ScratchField, u64)],
    needed: u64,
    geometry: &EpRankGeometry,
    layers: &[usize],
) -> Option<(Vec<EpReserveSegment>, Vec<(ScratchField, usize, u64, u64)>)> {
    let mut segments = Vec::with_capacity(layers.len() * 2);
    for &layer in layers {
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
        return None;
    }
    let placed = pack_roots(roots, &caps)?;
    Some((segments, placed))
}

/// [`plan_ep_reserve`] with the scratch donors chosen by the caller instead
/// of the suffix `[H, resident)`: the unit-plan (D1) regime, where the
/// donors are whole-resident layers wherever the planner put them.
///
/// `candidates` are trunk layers, in preference order, that are
/// whole-resident on EVERY rank (the caller's contract, checked against the
/// unit plan, not here); the plan takes the shortest prefix of them whose
/// whole-layer segments hold every remaining root and returns the donors
/// ascending in [`EpReservePlan::donor_layers`]. `resident_layers` is
/// informational (the layers the planner counts resident per rank). The
/// plan has no S2 donors ([`EpReservePlan::explicit_donors`]): the unit
/// stage serves the borrowed tiles. Capacity checks other than the host
/// mirror budget (`free_per_rank`, `reserve_bytes`, `lane_bytes`,
/// `min_resident_layers`) are the unit planner's and are not repeated.
pub fn plan_ep_reserve_with_donors(
    forward: &EpReserveForwardPlan,
    geometry: &EpRankGeometry,
    capacity: &EpReserveCapacity,
    resident_layers: usize,
    candidates: &[usize],
) -> Result<EpReservePlan, Refusal> {
    plan_ep_reserve_with_donors_and_shadows(
        forward,
        geometry,
        capacity,
        resident_layers,
        candidates,
        &[],
    )
}

/// [`plan_ep_reserve_with_donors`] that also hosts the weight shadows (F16 /
/// INT8 members, `shadow_bytes[i]` bytes each, in the caller's member order)
/// in the donor segments, beside the scratch roots: the shortest candidate
/// prefix whose segments hold the roots AND every shadow. Shadows are packed
/// after the scratch roots, descending size, 256-byte aligned, never
/// straddling a segment; the placements are in
/// [`EpReservePlan::shadow_placements`] (member order) and their extents
/// join the touched (restored) extents. An empty `shadow_bytes` is exactly
/// [`plan_ep_reserve_with_donors`].
pub fn plan_ep_reserve_with_donors_and_shadows(
    forward: &EpReserveForwardPlan,
    geometry: &EpRankGeometry,
    capacity: &EpReserveCapacity,
    resident_layers: usize,
    candidates: &[usize],
    shadow_bytes: &[u64],
) -> Result<EpReservePlan, Refusal> {
    if geometry.ranks == 0 || geometry.trunk_layers == 0 {
        return Err(Refusal::new("invalid layer geometry"));
    }
    if geometry.layer_bytes().filter(|&bytes| bytes > 0).is_none() {
        return Err(Refusal::new("layer expert bytes overflow or are zero"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for &layer in candidates {
        if layer >= geometry.trunk_layers || !seen.insert(layer) {
            return Err(Refusal::new(format!(
                "donor candidate layer {layer} is outside the {}-layer trunk or repeated",
                geometry.trunk_layers
            )));
        }
    }
    if resident_layers > geometry.trunk_layers {
        return Err(Refusal::new("resident layer count exceeds the trunk"));
    }
    let roots: Vec<(ScratchField, u64)> = forward
        .remaining_roots
        .iter()
        .map(|root| (root.field, root.bytes))
        .collect();
    let shadow_total = checked_sum(shadow_bytes.iter().copied())
        .ok_or_else(|| Refusal::new("shadow bytes overflow"))?;
    let needed = forward
        .remaining_bytes()?
        .checked_add(shadow_total)
        .ok_or_else(|| Refusal::new("remaining roots plus shadows overflow"))?;
    let mut found = None;
    for donors in 1..=candidates.len() {
        let mut layers = candidates[..donors].to_vec();
        layers.sort_unstable();
        if let Some((segments, placed)) = pack_donor_layers(&roots, needed, geometry, &layers) {
            if let Some(shadows) = pack_shadows_after(&segments, &placed, shadow_bytes) {
                found = Some((layers, segments, placed, shadows));
                break;
            }
        }
    }
    let (layers, segments, placed, shadows) = found.ok_or_else(|| {
        Refusal::new(format!(
            "the remaining roots plus shadows ({} MiB) do not pack into the {} whole-resident donor candidates",
            needed / MIB,
            candidates.len()
        ))
    })?;
    assemble_plan(
        forward,
        geometry,
        capacity,
        resident_layers,
        layers,
        [0; EP_RESERVE_S2_DONORS],
        true,
        segments,
        placed,
        shadows,
    )
}

/// Pack `shadow_bytes` into the free tails of `segments` after the scratch
/// `placed` roots: first-fit, descending size (ties by member index), starts
/// 256-aligned, no straddling. `None` when a shadow does not fit.
fn pack_shadows_after(
    segments: &[EpReserveSegment],
    placed: &[(ScratchField, usize, u64, u64)],
    shadow_bytes: &[u64],
) -> Option<Vec<EpReserveShadowPlacement>> {
    let mut cursor = vec![0u64; segments.len()];
    for &(_, segment, offset, bytes) in placed {
        cursor[segment] = cursor[segment].max(offset.checked_add(bytes)?);
    }
    let mut order: Vec<usize> = (0..shadow_bytes.len())
        .filter(|&index| shadow_bytes[index] > 0)
        .collect();
    order.sort_by(|&a, &b| shadow_bytes[b].cmp(&shadow_bytes[a]).then(a.cmp(&b)));
    let mut out = Vec::with_capacity(order.len());
    for index in order {
        let bytes = shadow_bytes[index];
        let slot = segments.iter().enumerate().find_map(|(segment, seg)| {
            let start = align_up(cursor[segment])?;
            let end = start.checked_add(bytes)?;
            (end <= seg.capacity).then_some((segment, start, end))
        });
        let (segment, offset, end) = slot?;
        cursor[segment] = end;
        out.push(EpReserveShadowPlacement {
            index,
            segment,
            layer: segments[segment].layer,
            slot: segments[segment].slot,
            offset,
            bytes,
        });
    }
    out.sort_by_key(|placement| placement.index);
    Some(out)
}

/// The plan tail shared by both donor choices: placements, touched extents,
/// host mirror accounting against the budget, digest.
#[allow(clippy::too_many_arguments)]
fn assemble_plan(
    forward: &EpReserveForwardPlan,
    geometry: &EpRankGeometry,
    capacity: &EpReserveCapacity,
    resident: usize,
    layers: Vec<usize>,
    s2: [usize; EP_RESERVE_S2_DONORS],
    explicit_donors: bool,
    segments: Vec<EpReserveSegment>,
    placed: Vec<(ScratchField, usize, u64, u64)>,
    shadow_placements: Vec<EpReserveShadowPlacement>,
) -> Result<EpReservePlan, Refusal> {
    let layer_bytes = geometry
        .layer_bytes()
        .ok_or_else(|| Refusal::new("layer expert bytes overflow"))?;
    let donors = layers.len();
    let first = layers[0];
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
            .chain(
                shadow_placements
                    .iter()
                    .filter(|shadow| shadow.segment == index)
                    .map(|shadow| shadow.offset + shadow.bytes),
            )
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
    let mut plan = EpReservePlan {
        forward: forward.clone(),
        resident_layers: resident,
        donor_first_layer: first,
        donor_layers: layers,
        s2_donor_layers: s2,
        explicit_donors,
        segments,
        placements,
        touched,
        shadow_placements,
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
            capture: false,
            routing_override: false,
            pipeline_parallel: false,
            expert_tier: false,
            expert_slab: false,
            npu: false,
            dflash: false,
            single_device_slowlink: false,
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
    fn explicit_donors_pack_like_the_suffix_and_follow_the_callers_layers() {
        let fwd = forward();
        let g = geometry();
        let suffix = plan_ep_reserve(&fwd, &g, &capacity_for(33, 0)).unwrap();
        // The suffix as an explicit list packs identically (same segments,
        // placements, touched extents, mirrors) and differs only in the flag,
        // the absent S2 donors and the digest.
        let candidates: Vec<usize> = (26..33).collect();
        let same =
            plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 33, &candidates).unwrap();
        assert!(same.explicit_donors && !suffix.explicit_donors);
        assert_eq!(same.donor_layers, suffix.donor_layers);
        assert_eq!(same.segments, suffix.segments);
        assert_eq!(same.placements, suffix.placements);
        assert_eq!(same.touched, suffix.touched);
        assert_eq!(same.mirror_bytes_per_rank, suffix.mirror_bytes_per_rank);
        assert_eq!(same.restore_bytes_per_rank, suffix.restore_bytes_per_rank);
        assert_eq!(same.s2_donor_layers, [0, 0]);
        assert_ne!(same.digest, suffix.digest);
        // Interleaved donors in preference order: the shortest prefix that
        // holds the roots (seven layers), returned ascending; surplus
        // candidates are not taken.
        let preferred = [40usize, 3, 17, 9, 22, 31, 12, 5, 44];
        let plan =
            plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 41, &preferred).unwrap();
        assert_eq!(plan.donor_layers, vec![3, 9, 12, 17, 22, 31, 40]);
        assert_eq!(plan.donor_first_layer, 3);
        assert_eq!(plan.segments.len(), 14);
        assert_eq!(plan.mirror_bytes_per_rank, 7 * g.layer_bytes().unwrap());
        assert_eq!(
            plan.segments.iter().map(|s| s.layer).collect::<Vec<_>>(),
            [3, 3, 9, 9, 12, 12, 17, 17, 22, 22, 31, 31, 40, 40]
        );
        assert!(plan.touched.len() <= plan.segments.len());
        assert_eq!(
            plan,
            plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 41, &preferred).unwrap()
        );
        assert!(crate::gpu_forward::ep_reserve_forward::check_stage_plan(&plan).is_ok());
        // Six candidates cannot hold the roots; repeats and out-of-trunk
        // layers fail closed; the host mirror budget still binds.
        assert!(
            plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 33, &preferred[..6])
                .is_err()
        );
        assert!(plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 33, &[3, 3]).is_err());
        assert!(
            plan_ep_reserve_with_donors(&fwd, &g, &capacity_for(33, 0), 33, &[g.trunk_layers])
                .is_err()
        );
        let mut cap = capacity_for(33, 0);
        cap.host_budget_bytes = 2 * 7 * g.layer_bytes().unwrap() - 1;
        assert!(plan_ep_reserve_with_donors(&fwd, &g, &cap, 33, &preferred).is_err());
        // A different donor set is a different digest (ranks must agree).
        let other = plan_ep_reserve_with_donors(
            &fwd,
            &g,
            &capacity_for(33, 0),
            41,
            &[41, 3, 17, 9, 22, 31, 12],
        )
        .unwrap();
        assert_ne!(other.digest, plan.digest);
        // A broken explicit plan is refused by the stage check.
        let mut broken = plan.clone();
        broken.donor_layers.reverse();
        assert!(crate::gpu_forward::ep_reserve_forward::check_stage_plan(&broken).is_err());
    }

    #[test]
    fn shadows_pack_into_the_donor_segments_beside_the_scratch_roots() {
        let fwd = forward();
        let g = geometry();
        let weights = crate::expert_residency::j1778_bf16_manifest();
        let region =
            crate::decode_share_shadows::plan_shadow_region(&weights, 48, true, true).unwrap();
        let sizes: Vec<u64> = region.spans().iter().map(|span| span.2).collect();
        let candidates: Vec<usize> = (20..33).collect();
        let cap = capacity_for(33, 0);
        let bare = plan_ep_reserve_with_donors(&fwd, &g, &cap, 33, &candidates).unwrap();
        assert_eq!(bare.donor_layers.len(), 7);
        assert!(bare.shadow_placements.is_empty());
        let with = plan_ep_reserve_with_donors_and_shadows(&fwd, &g, &cap, 33, &candidates, &sizes)
            .unwrap();
        // 3853 MiB of roots + 1549 MiB of shadows need nine 637.5 MiB layers.
        assert!(with.donor_layers.len() >= 9 && with.donor_layers.len() <= 10);
        assert_eq!(with.shadow_placements.len(), sizes.len());
        assert_eq!(
            with.mirror_bytes_per_rank,
            with.donor_layers.len() as u64 * g.layer_bytes().unwrap()
        );
        // Every shadow is aligned, inside its segment, and disjoint from the
        // scratch roots and from every other shadow.
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        for placement in &with.placements {
            ranges.push((placement.segment, placement.offset, placement.bytes));
        }
        for (index, shadow) in with.shadow_placements.iter().enumerate() {
            assert_eq!(shadow.index, index);
            assert_eq!(shadow.bytes, sizes[index]);
            assert_eq!(shadow.offset % EP_RESERVE_ROOT_ALIGN, 0);
            let segment = &with.segments[shadow.segment];
            assert_eq!((segment.layer, segment.slot), (shadow.layer, shadow.slot));
            assert!(shadow.offset + shadow.bytes <= segment.capacity);
            ranges.push((shadow.segment, shadow.offset, shadow.bytes));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].0 == pair[1].0 {
                assert!(pair[0].1 + pair[0].2 <= pair[1].1, "overlap {pair:?}");
            }
        }
        // The restore covers the shadow extents too.
        for extent in &with.touched {
            let used = ranges
                .iter()
                .filter(|(segment, _, _)| {
                    (with.segments[*segment].layer, with.segments[*segment].slot)
                        == (extent.layer, extent.slot)
                })
                .map(|(_, offset, bytes)| offset + bytes)
                .max()
                .unwrap();
            assert!(extent.len >= used.min(with.segments[0].capacity.max(used)) - 255);
        }
        assert!(with.restore_bytes_per_rank > bare.restore_bytes_per_rank);
        assert_ne!(with.digest, bare.digest);
        assert_eq!(
            with,
            plan_ep_reserve_with_donors_and_shadows(&fwd, &g, &cap, 33, &candidates, &sizes)
                .unwrap()
        );
        // Too few candidates, or a shadow bigger than any segment: refused.
        assert!(plan_ep_reserve_with_donors_and_shadows(
            &fwd,
            &g,
            &cap,
            33,
            &candidates[..8],
            &sizes
        )
        .is_err());
        assert!(plan_ep_reserve_with_donors_and_shadows(
            &fwd,
            &g,
            &cap,
            33,
            &candidates,
            &[g.gate_up_bytes + 1]
        )
        .is_err());
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
    fn a_lane_store_trades_whole_layers_and_the_lane_minimum_is_the_callers() {
        let fwd = forward();
        let g = geometry();
        let layer = g.layer_bytes().unwrap();
        // Two layers' worth of lane store leaves 31 where 33 fit, and 31 is
        // refused only because the default minimum is 33: a lane load passes
        // its own (strictly better than the ordinary lane placement).
        let mut cap = capacity_for(33, 0);
        cap.lane_bytes = 2 * layer;
        assert!(plan_ep_reserve(&fwd, &g, &cap).is_err());
        cap.min_resident_layers = 31;
        let plan = plan_ep_reserve(&fwd, &g, &cap).unwrap();
        assert_eq!(plan.resident_layers, 31);
        // The plan's reserve is the census reserve alone: the lane store is
        // a separate term every rank's own recompute never sees.
        assert_eq!(plan.reserve_bytes, cap.reserve_bytes);
        cap.min_resident_layers = 32;
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

    #[test]
    fn a_spec_store_keeps_the_ordinary_reserve_only_under_the_kill_switch() {
        // AR store (no spec_k) and an MTP-less slot count never block.
        assert!(spec_store_blocks_reserve(None, 3, false).is_none());
        assert!(spec_store_blocks_reserve(Some(2), 0, false).is_none());
        // The kill switch (`HIPFIRE_EP_MTP_RESERVE_SHARE=0`) blocks the decoupled
        // reserve for a spec store, naming the way out.
        let why = spec_store_blocks_reserve(Some(2), 3, false).expect("blocked");
        assert!(why.contains("--spec mtp") && why.contains("HIPFIRE_EP_MTP_RESERVE_SHARE=0"));
        // The default (sharing on) lets the spec store take the reserve.
        assert!(spec_store_blocks_reserve(Some(2), 3, true).is_none());
    }

    #[test]
    fn the_stage_verdict_refuses_a_spec_store_on_a_reserve_load_only_under_the_kill_switch() {
        // An ordinary load never reaches the reserve rules.
        assert_eq!(spec_store_reserve_verdict(false, false, 9999, None), Ok(()));
        // A reserve load under the kill switch: labeled refusal.
        let refused = spec_store_reserve_verdict(true, false, 8, Some(512)).unwrap_err();
        assert!(refused.contains("refused on an EP reserve-sharing load"));
        // By default a step that fits the resident bank is admitted...
        assert_eq!(spec_store_reserve_verdict(true, true, 8, Some(512)), Ok(()));
        assert_eq!(
            spec_store_reserve_verdict(true, true, 512, Some(512)),
            Ok(())
        );
        // ...a wider one, or an unattached bank, is refused with the numbers.
        let wide = spec_store_reserve_verdict(true, true, 513, Some(512)).unwrap_err();
        assert!(wide.contains("512-row") && wide.contains("513-row"));
        assert!(spec_store_reserve_verdict(true, true, 8, None).is_err());
    }

    #[test]
    fn a_mesh_window_refuses_to_start_on_an_open_borrow_epoch() {
        assert_eq!(mesh_window_epoch_check(false), Ok(()));
        let error = mesh_window_epoch_check(true).unwrap_err();
        assert!(error.contains("borrowed-bank epoch is open"));
        // The epoch type the guard reads: a borrowed call opens it, the
        // outermost prompt end restores it.
        let mut epoch = crate::gpu_forward::ep_reserve_forward::EpEpoch::default();
        assert!(!epoch.dirty());
        epoch.acquired();
        assert!(mesh_window_epoch_check(epoch.dirty()).is_err());
        epoch.restored();
        assert_eq!(mesh_window_epoch_check(epoch.dirty()), Ok(()));
    }

    // -- trunk shard: ONE shared plan (the root-wise maximum of the local plans) -------------

    /// The headless global config the shard loads, and the two rank-local configs cut from it
    /// (rank 0 shared width 384, rank 1 shared width 256; heads halved identically).
    fn shard_global() -> Qwen4Config {
        compact_test_config()
            .into_headless()
            .expect("headless global config")
    }

    fn shard_locals() -> [Qwen4Config; 2] {
        let global = shard_global();
        [0, 1].map(|rank| global.trunk_shard_local(rank, 2).expect("local config"))
    }

    /// The forward plan every rank shares under the shard: the root-wise maximum of the two
    /// ranks' own plans.
    fn shard_forward() -> EpReserveForwardPlan {
        EpReserveForwardPlan::from_config_for_shard(
            &shard_global(),
            Qwen4ForwardScratchMode::Full,
            0,
            true,
        )
        .expect("the union plan covers both local configs")
    }

    /// The reserve plan every rank shares under the shard.
    fn shard_plan() -> EpReservePlan {
        let fwd = shard_forward();
        let cap = capacity_for(33, 0);
        plan_ep_reserve(&fwd, &geometry(), &cap).expect("the shard plan")
    }

    /// Every root of `plan`, in a fixed order.
    fn all_roots(plan: &EpReserveForwardPlan) -> Vec<EpReserveRoot> {
        plan.external_roots
            .iter()
            .chain(&plan.resident_persistent)
            .chain(&plan.resident_roots)
            .chain(&plan.remaining_roots)
            .copied()
            .collect()
    }

    #[test]
    fn the_shard_plan_is_the_union_of_the_local_plans_inside_the_global_one() {
        let (global, locals) = (shard_global(), shard_locals());
        let mode = Qwen4ForwardScratchMode::Full;
        // The two ranks really have different local configs.
        assert_ne!(locals[0], locals[1]);
        assert_ne!(
            locals[0].shared_expert_intermediate_size,
            locals[1].shared_expert_intermediate_size
        );
        // Without the shard verdict the plan is the ordinary plan of the global config:
        // flag off is byte-identical.
        let plain = EpReserveForwardPlan::from_config_for_shard(&global, mode, 0, false).unwrap();
        assert_eq!(
            plain,
            EpReserveForwardPlan::from_config(&global, mode, 0).unwrap()
        );
        let shared = shard_forward();
        assert_ne!(shared, plain);
        let owns = locals
            .clone()
            .map(|local| EpReserveForwardPlan::from_config_exact(&local, mode, 0).unwrap());
        // The shared plan IS the root-wise maximum of the ranks' own plans, and every rank's
        // plan fits inside it, root for root.
        assert_eq!(shared, owns[0].union(&owns[1]).unwrap());
        for (local, own) in locals.iter().zip(&owns) {
            shared.covers_local(local).expect("covered");
            assert!(
                own.charged_forward_bytes().unwrap() <= shared.charged_forward_bytes().unwrap()
            );
            assert!(own.remaining_bytes().unwrap() <= shared.remaining_bytes().unwrap());
            assert!(own.spec_bytes <= shared.spec_bytes);
        }
        // The union is no larger than the plan of the global config, root for root, and
        // strictly smaller in charge and donor-packed bytes.
        let (union_roots, global_roots) = (all_roots(&shared), all_roots(&plain));
        assert_eq!(union_roots.len(), global_roots.len());
        for (union, global) in union_roots.iter().zip(&global_roots) {
            assert_eq!(union.field, global.field);
            assert!(union.bytes <= global.bytes, "{:?}", union.field);
        }
        assert!(shared.spec_bytes <= plain.spec_bytes);
        assert!(shared.charged_forward_bytes().unwrap() < plain.charged_forward_bytes().unwrap());
        assert!(shared.remaining_bytes().unwrap() < plain.remaining_bytes().unwrap());
        // One rank's plan alone is NOT enough for the other (the shared widths differ), which
        // is why the shared plan is the union; the global config's plan does not fit in a
        // local one.
        let (larger, smaller) = if owns[0].charged_forward_bytes().unwrap()
            >= owns[1].charged_forward_bytes().unwrap()
        {
            (0, 1)
        } else {
            (1, 0)
        };
        assert!(
            owns[smaller].charged_forward_bytes().unwrap()
                < owns[larger].charged_forward_bytes().unwrap()
        );
        let error = owns[smaller].covers_local(&locals[larger]).unwrap_err();
        assert!(error.0.contains("plan reserves"), "{error}");
        let error = owns[smaller].covers_local(&global).unwrap_err();
        assert!(error.0.contains("plan reserves"), "{error}");
        // A local config can never seed the shared plan.
        for shard in [false, true] {
            let error = EpReserveForwardPlan::from_config_for_shard(&locals[0], mode, 0, shard)
                .unwrap_err();
            assert!(error.0.contains("cannot seed"), "{error}");
        }
        // A fixed table charge is added once on the shared plan.
        let with_tables =
            EpReserveForwardPlan::from_config_for_shard(&global, mode, MIB, true).unwrap();
        assert_eq!(
            with_tables.charged_forward_bytes().unwrap(),
            shared.charged_forward_bytes().unwrap() + MIB
        );
        // A geometry the shard cannot cut refuses the plan (the load keeps the ordinary
        // placement) instead of failing later at attach.
        let mut odd = global.clone();
        odd.linear_num_value_heads = 47;
        let error = EpReserveForwardPlan::from_config_for_shard(&odd, mode, 0, true).unwrap_err();
        assert!(error.0.contains("local config"), "{error}");
        assert!(EpReserveForwardPlan::from_config_for_shard(&odd, mode, 0, false).is_ok());
        // Plans of other rows or a different fixed charge never merge.
        let mut other = owns[0].clone();
        other.fixed_table_bytes += 1;
        assert!(owns[0].union(&other).is_err());
    }

    /// Both ranks hold the same `Arc` of the shared plan: their mesh receipts agree whatever
    /// their local configs are, and a rank that cut a plan from ITS OWN local config alone
    /// (the per-rank-plan design) publishes a different digest whenever its roots differ from
    /// the shared ones, so the mesh refuses it.
    #[test]
    fn receipts_agree_across_two_different_local_configs_and_refuse_a_per_rank_plan() {
        let mode = Qwen4ForwardScratchMode::Full;
        let (global, locals) = (shard_global(), shard_locals());
        let plan = shard_plan();
        let generation = 0x5eed;
        // what `EpReserveForward::ep_reserve_plan_receipt` returns on each rank: the plan's
        // receipt with the rank's source generation (the seal's, rank-independent) and the
        // route bits (tile carrier admitted and the stage built on both ranks)
        let receipt = |plan: &EpReservePlan| plan.mesh_receipt(generation, true, true);
        let (rank0, rank1) = (receipt(&plan), receipt(&plan));
        assert_eq!(rank0.agree(&rank1), Ok(()));
        assert_eq!(
            crate::ep::agree_reserve_receipts(&[Some(rank0), Some(rank1)]),
            Ok(Some(rank0))
        );
        // per-rank plans: each from its own local config alone, same capacity
        let own_plan = |forward: EpReserveForwardPlan| {
            plan_ep_reserve(&forward, &geometry(), &capacity_for(33, 0)).expect("a plan")
        };
        let owns = locals.clone().map(|local| {
            own_plan(EpReserveForwardPlan::from_config_exact(&local, mode, 0).unwrap())
        });
        // the rank whose own plan is smaller than the shared one publishes another digest
        let smaller = owns
            .iter()
            .find(|own| own.digest != plan.digest)
            .expect("the ranks' own plans differ");
        let mixed = [Some(receipt(&plan)), Some(receipt(smaller))];
        let error = crate::ep::agree_reserve_receipts(&mixed).unwrap_err();
        assert!(
            error.contains("disagrees") && error.contains("rank 1"),
            "{error}"
        );
        // the capacity-derived boundaries agree and the shared plan never needs fewer donor
        // layers than a rank's own plan
        for own in &owns {
            assert!(own.donor_layers.len() <= plan.donor_layers.len());
            assert_eq!(own.resident_layers, plan.resident_layers);
            assert_eq!(own.forward.prompt_rows, plan.forward.prompt_rows);
            assert!(own.mirror_bytes_per_rank <= plan.mirror_bytes_per_rank);
        }
        // ... while the plan of the unsharded global config would take at least as many donors
        // and restore at least as many bytes per borrowed epoch as the shared one
        let global_plan =
            own_plan(EpReserveForwardPlan::from_config_for_shard(&global, mode, 0, false).unwrap());
        assert!(plan.donor_layers.len() <= global_plan.donor_layers.len());
        assert!(plan.restore_bytes_per_rank <= global_plan.restore_bytes_per_rank);
        assert!(plan.mirror_bytes_per_rank <= global_plan.mirror_bytes_per_rank);
        // every placement of the shared plan is at least as large as its local root (the
        // banks put the local root at the planned offset)
        for local in &locals {
            let own = EpReserveForwardPlan::from_config_exact(local, mode, 0).unwrap();
            for root in &own.remaining_roots {
                let placed = plan
                    .placements
                    .iter()
                    .find(|placement| placement.field == root.field)
                    .expect("the shared plan places every local donor-packed root");
                assert!(placed.bytes >= root.bytes);
            }
        }
    }

    /// The census charges the shared plan's forward (the union of the ranks' own); every
    /// rank's real scratch, state and reserve is at most what the loader charged, and a load
    /// without the shard keeps the census it had.
    #[test]
    fn shard_census_charges_are_conservative_per_rank() {
        let (global, locals) = (shard_global(), shard_locals());
        let mode = Qwen4ForwardScratchMode::Full;
        let shared = shard_forward().charged_forward_bytes().unwrap();
        let context = crate::Qwen4ContextCommit::legacy(16384);
        let layout = crate::Qwen4StateFormat::F32.layout();
        let census = |config: &Qwen4Config, forward: u64| {
            state_census_with_forward(
                config,
                crate::Qwen4StateFormat::F32,
                &context,
                4096,
                mode,
                Some(forward),
            )
            .unwrap()
        };
        let reserve = |config: &Qwen4Config, forward: u64| {
            auto_vram_reserve_with_forward(
                config,
                &context,
                4096,
                mode,
                layout,
                None,
                None,
                Some(forward),
            )
            .unwrap()
        };
        let charged = census(&global, shared);
        let floor = reserve(&global, shared);
        for local in &locals {
            let own_forward = EpReserveForwardPlan::from_config_exact(local, mode, 0)
                .unwrap()
                .charged_forward_bytes()
                .unwrap();
            assert!(own_forward <= shared);
            // the shared charge is >= what the rank's own config would charge, term by term;
            // the loader's state/QSA terms come from the GLOBAL config (>= every rank's)
            let own = census(local, own_forward);
            assert!(charged.forward >= own.forward);
            assert!(
                charged.state >= own.state,
                "the GDN state halves, never grows"
            );
            assert!(charged.qsa_context >= own.qsa_context);
            // the historical floor is a measured layout of the GLOBAL geometry plus the
            // forward's difference from that layout: a rank's need is the same floor with its
            // own (smaller) forward, which the shared plan's charge bounds
            assert!(floor >= reserve(&global, own_forward));
        }
        // flag off: the plan the loader charges is the ordinary global plan, bit for bit,
        // and the shard's shared charge is below it
        let plain = EpReserveForwardPlan::from_config_for_shard(&global, mode, 0, false)
            .unwrap()
            .charged_forward_bytes()
            .unwrap();
        assert!(shared < plain);
        assert!(floor < reserve(&global, plain));
    }

    /// The bytes the shard frees on the busiest rank reach the per-rank unit budget
    /// (`resident_split_ranks_trunk` -> `non_expert` -> `ep_unit_rank_budgets`) one for one,
    /// and become `freed / unit` more resident experts at the plan's unit size. Real
    /// Flash-Next trunk shapes at the MQ6 trunk dtype (shared expert BF16).
    #[test]
    fn freed_trunk_bytes_reach_the_unit_budget() {
        use crate::expert_residency::resident_split_ranks_trunk;
        use crate::weights::trunk_shard::{census_trunk_for, trunk_share_bytes};
        use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry};
        use rdna_compute::DType;

        let global = shard_global();
        let (dims, qsa) = census_trunk_for(true, &global, 2).expect("the census view");
        // the census cuts from the GLOBAL geometry even when handed a rank-local config
        for local in shard_locals() {
            assert_eq!(census_trunk_for(true, &local, 2).unwrap().0, dims);
        }
        let mut weights: Vec<WeightEntry> = Vec::new();
        let mut add = |layer: usize, tail: &str, shape: &[usize], dtype: DType| {
            let mut entry = WeightEntry::model(
                format!("model.language_model.layers.{layer}.{tail}"),
                shape.to_vec(),
                dtype,
                ShardPolicy::Replicate,
            );
            entry.layer = Some(layer);
            weights.push(entry);
        };
        for layer in 0..48 {
            if layer % 4 == 3 {
                add(
                    layer,
                    "self_attn.q_proj.weight",
                    &[12288, 2560],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "self_attn.k_proj.weight",
                    &[512, 2560],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "self_attn.v_proj.weight",
                    &[512, 2560],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "self_attn.o_proj.weight",
                    &[2560, 6144],
                    DType::MQ6G256V2,
                );
            } else {
                add(
                    layer,
                    "linear_attn.in_proj_qkv.weight",
                    &[10240, 2560],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "linear_attn.in_proj_z.weight",
                    &[6144, 2560],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "linear_attn.out_proj.weight",
                    &[2560, 6144],
                    DType::MQ6G256V2,
                );
                add(
                    layer,
                    "linear_attn.in_proj_a.weight",
                    &[48, 2560],
                    DType::BF16,
                );
                add(
                    layer,
                    "linear_attn.in_proj_b.weight",
                    &[48, 2560],
                    DType::BF16,
                );
                add(
                    layer,
                    "linear_attn.conv1d.weight",
                    &[10240, 1, 4],
                    DType::BF16,
                );
                add(layer, "linear_attn.A_log", &[48], DType::F32);
                add(layer, "linear_attn.dt_bias", &[48], DType::BF16);
            }
            add(
                layer,
                "mlp.shared_expert.gate_proj.weight",
                &[640, 2560],
                DType::BF16,
            );
            add(
                layer,
                "mlp.shared_expert.up_proj.weight",
                &[640, 2560],
                DType::BF16,
            );
            add(
                layer,
                "mlp.shared_expert.down_proj.weight",
                &[2560, 640],
                DType::BF16,
            );
            add(layer, "mlp.gate.weight", &[512, 2560], DType::BF16);
            add(layer, "mlp.experts.gate_up_proj", &[512, 1, 1], DType::Raw);
            add(layer, "mlp.experts.down_proj", &[512, 1, 1], DType::Raw);
        }
        let bytes_of = |entry: &WeightEntry| -> Option<u64> {
            let numel: usize = entry.logical_shape.iter().product();
            Some(match entry.dtype {
                DType::MQ6G256V2 => (numel / 256 * 200) as u64,
                DType::BF16 => (numel * 2) as u64,
                DType::F32 => (numel * 4) as u64,
                // a routed-expert tensor: unchanged by the shard (EP owner split, not TP)
                _ => numel as u64 * 1_740_800,
            })
        };
        let ep2 = crate::expert_residency::EpExpertSpans::new(512, vec![256, 256]).unwrap();
        let (whole, whole_experts) =
            resident_split_ranks_trunk(&weights, bytes_of, &ep2, None).unwrap();
        let (cut, cut_experts) =
            resident_split_ranks_trunk(&weights, bytes_of, &ep2, Some((&dims, qsa))).unwrap();
        assert_eq!(
            whole_experts, cut_experts,
            "routed-expert bytes are never cut"
        );
        let freed = whole - cut;
        // the busiest rank's freed bytes, tensor by tensor
        let expected: u64 = weights
            .iter()
            .filter(|entry| !entry.name.contains(".mlp.experts."))
            .map(|entry| {
                let full = bytes_of(entry).unwrap();
                full - trunk_share_bytes(&dims, qsa, &entry.name, &entry.logical_shape, full, 2)
                    .unwrap()
            })
            .sum();
        assert_eq!(freed, expected);
        // closed form: 36 GDN layers x 22,814,864 + 12 QSA layers x 18,432,000
        //              + 48 shared experts x 3,932,160 (the busiest rank keeps 384/640)
        assert_eq!(freed, 36 * 22_814_864 + 12 * 18_432_000 + 48 * 3_932_160);
        assert_eq!(freed, 1_231_262_784);
        // the unit budget: every rank's `free - trunk - reserve` grows by exactly the freed
        // bytes (the reserve term is the plan's, independent of the trunk)
        let (free, reserve) = (
            30 * GIB,
            6 * GIB + shard_forward().charged_forward_bytes().unwrap(),
        );
        let before = crate::expert_ep_units::ep_unit_rank_budgets(&[free, free], whole, reserve);
        let after = crate::expert_ep_units::ep_unit_rank_budgets(&[free, free], cut, reserve);
        for rank in 0..2 {
            assert_eq!(after[rank] - before[rank], freed);
        }
        let unit = EP2_GATE_UP_EXPERT_BYTES + EP2_DOWN_EXPERT_BYTES;
        assert_eq!(unit, 2_611_200);
        let gained = after[0] / unit - before[0] / unit;
        assert!(gained == freed / unit || gained == freed / unit + 1);
        assert_eq!(freed / unit, 471);
        let planar = EP2_PLANAR_GATE_UP_EXPERT_BYTES + EP2_PLANAR_DOWN_EXPERT_BYTES;
        assert_eq!(freed / planar, 485);
        // no shard: the census is the one it always was
        assert_eq!(
            resident_split_ranks_trunk(&weights, bytes_of, &ep2, None).unwrap(),
            crate::expert_residency::resident_split_ranks(&weights, bytes_of, &ep2).unwrap()
        );
    }
}
