// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Frozen per-load resource snapshot: the checked allocation plan of one admitted Qwen4 load.
//!
//! This is a source-only accounting wrapper, not a planner and not evidence. The
//! loader supplies the ACTUAL production values later (admitted capture buffers,
//! the settled [`AutoReserve`](crate::expert_residency::AutoReserve) from the
//! owning planner, the settled placement bytes and the actual host-check
//! inputs); this constructor only re-adds those scalars with checked arithmetic
//! and freezes the result as `fn-oracle.load-resource-snapshot.v1`. It never
//! recomputes static allocation math, never invents prices, never marks a fit,
//! and never claims a measured peak: `device_plan_bytes` / `host_plan_bytes`
//! are allocation-plan sums, not peak observations.
//!
//! Capture and freeze happen ONCE per load, beside the immutable
//! [`Qwen4MeasuredLoadIdentity`](crate::measured_identity::Qwen4MeasuredLoadIdentity),
//! never on the drafter/speculator hot path. A missing or unsupported input is
//! `None` with a named reason upstream, never a zero-filled fallback.
//!
//! The NEW ring capture lives inside the native MTP census term of the reserve
//! and is never added to the device plan twice. When the native MTP census
//! cannot contain the new capture under the actual source, construction fails
//! with the exact mismatch instead of relaxing the check. This type alone is
//! not a resource proof and cannot qualify row-exact-wide on its own.
//!
//! Conservative upper-bound extension (Main-adjudicated, NOT a second planner):
//! the loader additionally supplies the ACTUAL backing class (the production
//! `Gpu::is_uma()`) and the ACTUAL pooled-parked bytes (the public pool
//! accessor `Gpu::pool_parked_bytes`, over `GpuPool::pooled_bytes`) read at
//! the preallocation instant and again at the settled instant. Both parked
//! readings are charged: the preallocation and settled owners may overlap, so
//! no subtraction is taken and `device_plan_bytes` is a conservative union
//! upper bound, never exact usage and never a measured peak. On UMA-shared
//! backing the root device plan is device-backed host memory and is added to
//! the host plan exactly once; on discrete backing nothing is added. The host
//! scope stays the existing disjoint host-mapped plus explicit-host-only
//! check, never a UMA `device_free_from` capacity clamp. The only lifecycle
//! this record supports is `fresh_rank_load` (a fresh production Gpu/load,
//! never a retrospective retained-model estimate and never a counterfeit
//! preload-free baseline). This block alone earns no fit qualification.

use std::sync::Arc;

use crate::expert_residency::AutoReserve;
use crate::measured_identity::Qwen4MeasuredLoadIdentity;

/// Schema marker of the frozen resource snapshot.
pub const LOAD_RESOURCE_SNAPSHOT_SCHEMA: &str = "fn-oracle.load-resource-snapshot.v1";

/// Evidence kind carried by every snapshot: a checked allocation plan, never a
/// measured peak and never a fit claim.
pub const LOAD_RESOURCE_EVIDENCE_KIND: &str = "checked_allocation_plan_not_peak";
/// Evidence kind of the root `upper_bound` block: a conservative
/// planned-component union, never a measured peak.
pub const LOAD_RESOURCE_UPPER_BOUND_EVIDENCE_KIND: &str =
    "conservative_component_union_upper_bound_not_peak";
/// Only model lifecycle this record supports: a fresh rank load on a fresh
/// production Gpu/load. A fully retained old model without its original
/// preallocation basis is unsupported upstream (named `None`), never a
/// counterfeit preload-free backing.
pub const LOAD_RESOURCE_MODEL_LIFECYCLE_FRESH_RANK_LOAD: &str = "fresh_rank_load";
/// Parked-overlap policy: the preallocation and settled owners may overlap and
/// no subtraction is taken, so both parked readings are charged.
pub const LOAD_RESOURCE_PARKED_OVERLAP_POLICY: &str = "charge_both_unknown_intersection";
/// Scope of the existing host check feeding the upper bound: host-mapped
/// experts plus explicit host-only bytes, never a UMA device-free clamp.
pub const LOAD_RESOURCE_HOST_CHECK_SCOPE: &str = "host_mapped_and_explicit_host_only";
/// Owner-contract version pinning the no-double-charge census behind the union.
pub const LOAD_RESOURCE_OWNER_CONTRACT_VERSION: &str = "qwen4-owned-components-v1";

/// Which capture the admitted load holds. Serialized tokens are frozen as
/// `direct_initial_64` / `retained_8_to_64` (explicit, human-readable);
/// loader and FullPlanAdmission consumers must match these exact tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Qwen4RingMode {
    /// Fresh 64-row capture, no carried ring: no old allocation is charged.
    #[serde(rename = "direct_initial_64")]
    DirectInitial64,
    /// Growth from a retained 8-row ring: the old 8-row storage is still
    /// charged (even when the pool retains it).
    #[serde(rename = "retained_8_to_64")]
    Retained8To64,
}

/// Geometry of the admitted capture buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Qwen4CaptureGeometry {
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub owned_gdn_layers: usize,
}

/// The ACTUAL admitted load and capture buffers. Values come from the real
/// admitted load and capture; nothing here is derived from requested env pins.
/// Unsupported PP/EP shapes stay absent upstream instead of arriving here.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Qwen4AdmittedShape {
    pub max_seq: usize,
    pub chunk_rows: usize,
    pub capture_rows: usize,
    pub ranks: usize,
    pub gdn_state: String,
    pub kv_backend: String,
    pub scratch_mode: String,
    pub geometry: Qwen4CaptureGeometry,
    pub mode: Qwen4RingMode,
    pub old_capture_bytes: u64,
    pub new_capture_bytes: u64,
}

/// The settled plan's device side: the actual planner owners' bytes.
/// `host_mapped_bytes` excludes separately reported stage mirrors; the
/// whole-layer pools are actual resident routed-expert bytes, never
/// layers-times-largest estimates, and hotset and whole-layer pools are never
/// double-charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Qwen4PlacementBytes {
    pub non_expert_device_bytes: u64,
    pub routed_device_bytes: u64,
    pub host_mapped_bytes: u64,
    pub tables_bytes: u64,
    pub stage_bytes: u64,
    pub margin_bytes: u64,
}

/// The ACTUAL host-check inputs. Unknown budgets are `None`, never zero:
/// a producer/validator must not call this a fit proof while required evidence
/// is missing. `host_check_bytes` must exclude the stage mirrors when the
/// actual check already includes them (then `stage_mirror_bytes` is 0), so the
/// host plan never double-counts stage mirrors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Qwen4HostCheckInputs {
    pub host_check_bytes: u64,
    pub stage_mirror_bytes: u64,
    pub host_headroom_bytes: u64,
    pub ttm_pool_bytes: u64,
    pub mem_available_bytes: Option<u64>,
    pub gtt_budget_bytes: Option<u64>,
}

/// Device-memory backing class of the producing load. Wire tokens are frozen
/// as `discrete` / `uma_shared`; the loader (`Gpu::is_uma`) and E validators
/// must match these exact tokens. A `gfx1151` load reports
/// [`Qwen4BackingKind::UmaShared`]; a `gfx1201` load reports
/// [`Qwen4BackingKind::Discrete`]. Never infer GTT budgets from a nonzero pool
/// reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Qwen4BackingKind {
    /// Discrete VRAM heap: pooled-parked bytes live on the device heap.
    #[serde(rename = "discrete")]
    Discrete,
    /// UMA-shared heap: the root device plan is device-backed host memory and
    /// is charged once on the host plan too (conservative: GTT-backed mappings
    /// already intersect both sides; labelled upper bound, still must fit).
    #[serde(rename = "uma_shared")]
    UmaShared,
}
/// The ACTUAL backing inputs of one fresh rank load. `kind` comes from the
/// real production `Gpu::is_uma()`; the pooled sizes come from the public pool
/// accessor (`Gpu::pool_parked_bytes`) read at the preallocation instant and
/// again at the settled instant. Never hand-fill these; an unsupported source
/// path stays absent upstream with a named reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Qwen4LoadBacking {
    pub kind: Qwen4BackingKind,
    pub preallocation_pooled_device_bytes: u64,
    pub settled_pooled_device_bytes: u64,
}

/// Frozen, validated resource snapshot of one admitted load. Only the cached
/// JSON, its canonical string and the plan sums are retained; consumers read
/// every other term through [`Self::value`]. Inputs are consumed by the
/// constructor's checks and not stored.
#[derive(Clone, Debug)]
pub struct Qwen4LoadResourceSnapshot {
    identity: Arc<Qwen4MeasuredLoadIdentity>,
    device_plan_bytes: u64,
    host_plan_bytes: u64,
    canonical: String,
    value: serde_json::Value,
}

impl Qwen4LoadResourceSnapshot {
    /// Freeze the checked allocation plan of one admitted load.
    ///
    /// Runs accounting/shape checks only: census total consistency, native MTP
    /// containment of the new capture, row-exact 64 rows, single rank,
    /// positive geometry/context/chunk, known GDN state, direct/retained old
    /// accounting, and checked plan sums. No GPU fit is claimed.
    ///
    /// `backing` is the ACTUAL production backing of this fresh rank load:
    /// `kind` from the real `Gpu::is_uma()`, the pooled sizes from the public
    /// pool accessor (`Gpu::pool_parked_bytes`) at the preallocation instant
    /// and the settled instant. The only lifecycle this record supports is
    /// `fresh_rank_load`; anything else stays absent upstream, never faked.
    pub fn new(
        identity: Arc<Qwen4MeasuredLoadIdentity>,
        admitted: Qwen4AdmittedShape,
        reserve: AutoReserve,
        placement: Qwen4PlacementBytes,
        host: Qwen4HostCheckInputs,
        preallocation_free_device_bytes: Option<u64>,
        backing: Qwen4LoadBacking,
    ) -> Result<Self, String> {
        const HERE: &str = "qwen4 resource snapshot";
        if admitted.capture_rows != 64 {
            return Err(format!(
                "{HERE}: capture_rows must be the row-exact 64, have {}",
                admitted.capture_rows
            ));
        }
        if admitted.ranks != 1 {
            return Err(format!("{HERE}: ranks must be 1, have {}", admitted.ranks));
        }
        if admitted.max_seq == 0 {
            return Err(format!("{HERE}: max_seq must be positive"));
        }
        if admitted.chunk_rows == 0 {
            return Err(format!("{HERE}: chunk_rows must be positive"));
        }
        let geometry = admitted.geometry;
        if geometry.linear_num_key_heads == 0
            || geometry.linear_num_value_heads == 0
            || geometry.linear_key_head_dim == 0
            || geometry.linear_value_head_dim == 0
            || geometry.owned_gdn_layers == 0
        {
            return Err(format!(
                "{HERE}: capture geometry must be positive, have {geometry:?}"
            ));
        }
        match admitted.gdn_state.as_str() {
            "q8" | "fp32" | "q8_ef" => {}
            other => {
                return Err(format!(
                    "{HERE}: gdn_state must be q8|fp32|q8_ef, have {other:?}"
                ));
            }
        }
        match admitted.mode {
            Qwen4RingMode::DirectInitial64 if admitted.old_capture_bytes != 0 => {
                return Err(format!(
                    "{HERE}: direct initial-64 charges no old capture, have {} old bytes",
                    admitted.old_capture_bytes
                ));
            }
            Qwen4RingMode::Retained8To64 if admitted.old_capture_bytes == 0 => {
                return Err(format!(
                    "{HERE}: retained 8-to-64 keeps the old 8-row storage charged, have 0 old bytes"
                ));
            }
            Qwen4RingMode::DirectInitial64 | Qwen4RingMode::Retained8To64 => {}
        }
        if admitted.new_capture_bytes == 0 {
            return Err(format!("{HERE}: new_capture_bytes must be positive"));
        }
        // The reserve is the owning planner's value, reused verbatim; only the
        // checked scalar total is re-verified here.
        let census_total = reserve
            .census
            .total()
            .map_err(|err| format!("{HERE}: {err}"))?;
        if census_total != reserve.census_total {
            return Err(format!(
                "{HERE}: census total mismatch: terms sum to {census_total} bytes, \
                 reserve carries census_total {census_total_carried} bytes",
                census_total_carried = reserve.census_total
            ));
        }
        // The NEW capture is already inside the native MTP census term; it is
        // never added to the device plan again. Report the exact shortfall.
        if reserve.census.mtp < admitted.new_capture_bytes {
            return Err(format!(
                "{HERE}: native MTP census cannot contain the new capture: census mtp {} bytes < \
                 new_capture_bytes {} bytes; not relaxed",
                reserve.census.mtp, admitted.new_capture_bytes
            ));
        }
        let reserve_bytes = reserve.bytes();
        // Base device plan: the OLD settled-owner formula, verbatim.
        let mut base_device_plan_bytes = placement
            .non_expert_device_bytes
            .checked_add(placement.routed_device_bytes)
            .and_then(|sum| sum.checked_add(reserve_bytes))
            .and_then(|sum| sum.checked_add(placement.tables_bytes))
            .and_then(|sum| sum.checked_add(placement.stage_bytes))
            .and_then(|sum| sum.checked_add(placement.margin_bytes))
            .ok_or_else(|| format!("{HERE}: device plan bytes overflow"))?;
        if admitted.mode == Qwen4RingMode::Retained8To64 {
            base_device_plan_bytes = base_device_plan_bytes
                .checked_add(admitted.old_capture_bytes)
                .ok_or_else(|| format!("{HERE}: device plan bytes overflow on old capture"))?;
        }
        // Conservative union upper bound: BOTH parked readings are charged. The
        // preallocation and settled owners may overlap, so no subtraction is
        // taken (known limitation, labelled `charge_both_unknown_intersection`).
        let device_plan_bytes = base_device_plan_bytes
            .checked_add(backing.preallocation_pooled_device_bytes)
            .and_then(|sum| sum.checked_add(backing.settled_pooled_device_bytes))
            .ok_or_else(|| format!("{HERE}: device plan bytes overflow on parked pools"))?;
        // UMA-shared backing is device-backed host memory: the root device plan
        // is added to the host plan exactly once. Discrete backing adds nothing
        // here; its parked terms already sit in the root device plan above.
        let device_backed_host_bytes = match backing.kind {
            Qwen4BackingKind::UmaShared => device_plan_bytes,
            Qwen4BackingKind::Discrete => 0,
        };
        let host_plan_bytes = host
            .host_check_bytes
            .checked_add(host.stage_mirror_bytes)
            .and_then(|sum| sum.checked_add(host.host_headroom_bytes))
            .and_then(|sum| sum.checked_add(device_backed_host_bytes))
            .ok_or_else(|| format!("{HERE}: host plan bytes overflow"))?;
        let measured_load: serde_json::Value = serde_json::from_str(identity.canonical_json())
            .map_err(|err| format!("{HERE}: measured-load canonical JSON does not parse: {err}"))?;
        let workspace: Vec<serde_json::Value> = reserve
            .census
            .workspace
            .iter()
            .map(|(name, bytes)| serde_json::json!({"name": name, "bytes": bytes}))
            .collect();
        let value = serde_json::json!({
            "schema": LOAD_RESOURCE_SNAPSHOT_SCHEMA,
            "evidence_kind": LOAD_RESOURCE_EVIDENCE_KIND,
            "measured_load": measured_load,
            "admitted": {
                "max_seq": admitted.max_seq,
                "chunk_rows": admitted.chunk_rows,
                "capture_rows": admitted.capture_rows,
                "ranks": admitted.ranks,
                "gdn_state": admitted.gdn_state,
                "kv_backend": admitted.kv_backend,
                "scratch_mode": admitted.scratch_mode,
                "geometry": {
                    "linear_num_key_heads": geometry.linear_num_key_heads,
                    "linear_num_value_heads": geometry.linear_num_value_heads,
                    "linear_key_head_dim": geometry.linear_key_head_dim,
                    "linear_value_head_dim": geometry.linear_value_head_dim,
                    "owned_gdn_layers": geometry.owned_gdn_layers,
                },
                "mode": admitted.mode,
                "old_capture_bytes": admitted.old_capture_bytes,
                "new_capture_bytes": admitted.new_capture_bytes,
            },
            "reserve": {
                "historical_bytes_or_null": reserve.historical,
                "census": {
                    "qsa_context": reserve.census.qsa_context,
                    "state": reserve.census.state,
                    "forward": reserve.census.forward,
                    "f16_shadows": reserve.census.f16_shadows,
                    "shared_i8_shadows": reserve.census.shared_i8_shadows,
                    "workspace": workspace,
                    "qsa_selector": reserve.census.qsa_selector,
                    "mtp": reserve.census.mtp,
                    "gather": reserve.census.gather,
                    "runtime": reserve.census.runtime,
                },
                "census_total_bytes": reserve.census_total,
                "reserve_bytes": reserve_bytes,
                "winner": reserve.winner(),
            },
            "placement": {
                "non_expert_device_bytes": placement.non_expert_device_bytes,
                "routed_device_bytes": placement.routed_device_bytes,
                "host_mapped_bytes": placement.host_mapped_bytes,
                "tables_bytes": placement.tables_bytes,
                "stage_bytes": placement.stage_bytes,
                "margin_bytes": placement.margin_bytes,
            },
            "host_check": {
                "host_check_bytes": host.host_check_bytes,
                "stage_mirror_bytes": host.stage_mirror_bytes,
                "host_headroom_bytes": host.host_headroom_bytes,
                "ttm_pool_bytes": host.ttm_pool_bytes,
                "mem_available_bytes": host.mem_available_bytes,
                "gtt_budget_bytes": host.gtt_budget_bytes,
            },
            "preallocation_free_device_bytes": preallocation_free_device_bytes,
            "upper_bound": {
                "evidence_kind": LOAD_RESOURCE_UPPER_BOUND_EVIDENCE_KIND,
                "model_lifecycle": LOAD_RESOURCE_MODEL_LIFECYCLE_FRESH_RANK_LOAD,
                "backing_kind": backing.kind,
                "preallocation_pooled_device_bytes": backing.preallocation_pooled_device_bytes,
                "settled_pooled_device_bytes": backing.settled_pooled_device_bytes,
                "parked_overlap_policy": LOAD_RESOURCE_PARKED_OVERLAP_POLICY,
                "existing_host_check_scope": LOAD_RESOURCE_HOST_CHECK_SCOPE,
                "owner_contract_version": LOAD_RESOURCE_OWNER_CONTRACT_VERSION,
                "base_device_plan_bytes": base_device_plan_bytes,
                "device_backed_host_bytes": device_backed_host_bytes,
            },
            "device_plan_bytes": device_plan_bytes,
            "host_plan_bytes": host_plan_bytes,
        });
        let canonical = serde_json::to_string(&value)
            .map_err(|err| format!("{HERE}: snapshot JSON does not render: {err}"))?;
        Ok(Self {
            identity,
            device_plan_bytes,
            host_plan_bytes,
            canonical,
            value,
        })
    }

    /// Borrow the cached snapshot JSON. No per-window serialization.
    pub fn value(&self) -> &serde_json::Value {
        &self.value
    }

    /// Borrow the cached canonical snapshot JSON string.
    pub fn canonical_json(&self) -> &str {
        &self.canonical
    }

    /// The frozen measured-load identity this snapshot was taken against.
    pub fn identity(&self) -> &Arc<Qwen4MeasuredLoadIdentity> {
        &self.identity
    }

    /// Checked device union upper bound (base plan plus both parked readings;
    /// not a measured peak, not a fit claim).
    pub fn device_plan_bytes(&self) -> u64 {
        self.device_plan_bytes
    }
    /// Checked host union upper bound (existing host check plus the UMA-only
    /// device-backed addition; not a measured peak, not a fit claim).
    pub fn host_plan_bytes(&self) -> u64 {
        self.host_plan_bytes
    }
}
/// Evidence kind of the preload component union: a conservative planned-component
/// union upper bound, never a measured peak and never a fit claim.
pub const PRELOAD_UNION_EVIDENCE_KIND: &str = "conservative_preload_component_union_not_peak";
/// Parked-reading policy of the preload union: only the preallocation instant is
/// charged; the settled reading is still pending (no subtraction is possible and
/// none is taken).
pub const PRELOAD_UNION_PARKED_POLICY: &str = "charge_preallocation_only_settled_read_pending";

/// Finite native-MTP union extra over the census logical peak, in backing bytes.
///
/// The census charges `resident + max(build_scratch, request)`. The preload
/// union instead charges the finite backing union `backing(resident) +
/// backing(build_scratch) + backing(request)` (both phases: the retired build
/// scratch freed at attach AND the first request's allocations, never the
/// larger alone), minus the logical peak the census already holds. The returned
/// extra is added exactly once; the ring payloads stay in `census.mtp` and are
/// never re-added, so the not-yet-materialized (cold) recurrent ring is not
/// duplicated and the already-resident live recurrent state (charged once in
/// the state census) is not touched. Every sum is checked; overflow or a
/// backing below the logical peak is `Err`, never a wrap.
pub fn preload_native_union_extra(
    resident: &[u64],
    build_scratch: &[u64],
    request: &[u64],
    granule: u64,
) -> Result<u64, String> {
    const HERE: &str = "qwen4 preload native union";
    let sum = |sizes: &[u64]| {
        sizes.iter().try_fold(0u64, |total, &bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("{HERE}: logical allocation list overflows"))
        })
    };
    let backing = |sizes: &[u64]| {
        sum(sizes)?
            .checked_add(
                crate::expert_residency::granule_slack_total(sizes.iter().copied(), granule)
                    .map_err(|err| format!("{HERE}: {err}"))?,
            )
            .ok_or_else(|| format!("{HERE}: backing allocation list overflows"))
    };
    let logical_peak = sum(resident)?
        .checked_add(sum(build_scratch)?.max(sum(request)?))
        .ok_or_else(|| format!("{HERE}: logical peak overflows"))?;
    let request_backing = backing(request)?;
    let backing_union = backing(resident)?
        .checked_add(backing(build_scratch)?)
        .and_then(|total| total.checked_add(request_backing))
        .ok_or_else(|| format!("{HERE}: backing union overflows"))?;
    backing_union
        .checked_sub(logical_peak)
        .ok_or_else(|| format!("{HERE}: backing union is below its logical peak"))
}
/// Finite forward-scratch union extra over the base mode's backing, in bytes.
///
/// The census charges one scratch plan (`base_mode`). The preload union covers
/// both admissible modes (`Full` and `FnSymGfx1201`) from the existing
/// allocation inventories
/// ([`crate::gpu_forward::qwen4_forward_allocation_sizes`], owners only:
/// borrowed alias views are never allocations). Only one plan materializes, so
/// the conservative bound is the max over modes, never the sum: `extra =
/// max(0, alt_backing - base_backing)` with checked backing sums (payload +
/// [`crate::expert_residency::granule_slack_total`]) per mode. No componentwise
/// Full-dominance is claimed (the compact plan re-sizes MoE buffers and borrows
/// eight fields, so per-slot comparison is not attempted); the max covers
/// whichever mode is larger. A `None` inventory for either mode is a named
/// `Err` (required-input overflow or refusal, never a zero fallback). No
/// payload scan, no flag mutation.
pub fn preload_scratch_union_extra(
    config: &crate::Qwen4Config,
    chunk_rows: usize,
    base_mode: crate::gpu_forward::Qwen4ForwardScratchMode,
    granule: u64,
) -> Result<
    (
        crate::gpu_forward::Qwen4ForwardScratchMode,
        u64,
        u64,
        u64,
        Option<String>,
    ),
    String,
> {
    const HERE: &str = "qwen4 preload scratch union";
    use crate::gpu_forward::Qwen4ForwardScratchMode;
    let alt_mode = match base_mode {
        Qwen4ForwardScratchMode::Full => Qwen4ForwardScratchMode::FnSymGfx1201,
        Qwen4ForwardScratchMode::FnSymGfx1201 => Qwen4ForwardScratchMode::Full,
    };
    let backing = |mode: Qwen4ForwardScratchMode| {
        let sizes = crate::gpu_forward::qwen4_forward_allocation_sizes(config, chunk_rows, mode)
            .ok_or_else(|| {
                format!("{HERE}: forward allocation sizes overflow or refuse {mode:?}")
            })?;
        let payload = sizes.iter().try_fold(0u64, |total, &bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("{HERE}: forward allocation list overflows ({mode:?})"))
        })?;
        let slack = crate::expert_residency::granule_slack_total(sizes.iter().copied(), granule)
            .map_err(|err| format!("{HERE}: {err} ({mode:?})"))?;
        payload
            .checked_add(slack)
            .ok_or_else(|| format!("{HERE}: forward backing overflows ({mode:?})"))
    };
    let base_backing = backing(base_mode)?;
    // A `None` alt inventory is ALWAYS a named error: a refused or overflowing
    // plan is required-input overflow, never a silent zero fallback. The only
    // valid single-mode union is when the caller's base mode IS the admissible
    // plan (the loader resolves the real scratch mode first and passes it as
    // base; an inapplicable alt never reaches this helper as a required input).
    if crate::gpu_forward::qwen4_forward_allocation_sizes(config, chunk_rows, alt_mode).is_none() {
        return Err(format!(
            "{HERE}: forward allocation sizes overflow or refuse {alt_mode:?} at {chunk_rows} rows"
        ));
    }
    let alt_backing = backing(alt_mode)?;
    let extra = alt_backing.checked_sub(base_backing).unwrap_or(0);
    Ok((alt_mode, base_backing, alt_backing, extra, None))
}

/// The conservative preload-time component union of one planned load: metadata
/// only, never a fit claim and never a measured peak.
///
/// Recomputes the authoritative [`crate::expert_residency::placement_inputs`]
/// from `args` and refuses when the caller scalars disagree with it (stale
/// reserve or non-expert plan); reuses the owners' own allocation lists through
/// [`crate::expert_residency::base_granule_slack`] and the private native
/// inventories and the forward-scratch inventories of both admissible modes.
/// The device bound is the planned owners plus charges plus the non-native
/// granule slack (including the census-plan forward slack) plus the finite
/// native build+request backing extra ([`preload_native_union_extra`],
/// superseding `slack.native_mtp`, never added beside it) plus the finite
/// forward-mode backing extra ([`preload_scratch_union_extra`], covering only
/// `max(0, alt − base)` backing, never the sum or double-added shared state)
/// plus the preallocation
/// pooled bytes, all checked. The existing census runtime allowance stays
/// inside the reserve and is additionally reported as its own line, never added
/// twice. Initial pool/free quotes are capacity basis, not components;
/// host/TTM/GTT quotes are capacity only. `scratch_mode_label` and `s1_reason`
/// are recorded labels; the scratch union additionally reports both mode
/// backings and the admissible-mode note, not a selection. Compact-mode
/// metadata reads use only [`crate::qwen4_compact_scratch_possible`]. A missing
/// head, list or capacity overflow is a named `Err`.
#[allow(clippy::too_many_arguments)]
pub fn preload_component_union(
    args: &crate::expert_residency::PlacementInputsArgs<'_>,
    charges: crate::expert_residency::PlacementCharges,
    reserve: crate::expert_residency::AutoReserve,
    non_expert_device_bytes: u64,
    routed_device_bytes: u64,
    host_mapped_bytes: u64,
    tables_bytes: u64,
    stage_mirror_bytes: u64,
    preallocation_pooled_device_bytes: u64,
    preallocation_free_device_bytes: Option<u64>,
    mem_available_bytes: Option<u64>,
    ttm_pool_bytes: u64,
    gtt_free_bytes: Option<u64>,
    granule_bytes: u64,
    scratch_mode_label: &str,
    s1_reason: &str,
) -> Result<serde_json::Value, String> {
    const HERE: &str = "qwen4 preload component union";
    // The authoritative planned inputs, recomputed from the same source facts.
    let inputs = crate::expert_residency::placement_inputs(args)
        .map_err(|err| format!("{HERE}: authoritative placement inputs refuse: {err}"))?;
    if non_expert_device_bytes != inputs.non_expert {
        return Err(format!(
            "{HERE}: planned non-expert {non_expert_device_bytes} bytes differs from \
             placement_inputs {} bytes",
            inputs.non_expert
        ));
    }
    if reserve.historical != inputs.reserve.historical
        || reserve.census_total != inputs.reserve.census_total
    {
        return Err(format!(
            "{HERE}: caller reserve differs from the placement_inputs reserve"
        ));
    }
    let census_total = reserve
        .census
        .total()
        .map_err(|err| format!("{HERE}: {err}"))?;
    if census_total != reserve.census_total {
        return Err(format!(
            "{HERE}: census total mismatch: terms sum to {census_total} bytes, reserve carries {} bytes",
            reserve.census_total
        ));
    }
    let reserve_bytes = reserve.bytes();
    // Per-owner backing slack of the same reserve, from the owners' own
    // allocation lists (individual-allocation counterfactual: no arena plan).
    let slack =
        crate::expert_residency::base_granule_slack(&crate::expert_residency::GranuleSlackArgs {
            gpu: args.gpu,
            config: args.config,
            weights: args.weights,
            bytes_of: args.bytes_of,
            backend: args.backend,
            max_seq: args.max_seq,
            vmm_granularity: args.vmm_granularity,
            mtp_kept: args.mtp_kept,
            max_k: args.max_k,
            state_format: args.state_format,
            scratch_mode: args.scratch_mode,
            chunk_rows: args.chunk_rows,
            reserve,
            granule: granule_bytes,
            weight_arenas: None,
        })
        .map_err(|err| format!("{HERE}: base granule slack refuses: {err}"))?;
    let slack_total = slack.total().map_err(|err| format!("{HERE}: {err}"))?;
    // The native slack term is SUPERSEDED, not kept: `slack.native_mtp` equals
    // backing(resident)+backing(max phase)-logical peak and is removed entirely.
    // The finite build+request union below adds `native_extra` =
    // backing(resident)+backing(build)+backing(request)-logical peak, i.e. the
    // old term plus exactly min(backing(build), backing(request)) — the retired
    // phase the max-phase census drops. No double rounding: the native term is
    // charged exactly once, as the full-union extra. From the same private
    // allocation lists the census sizes (resident head, retired build scratch,
    // first request).
    let slack_without_native = slack_total
        .checked_sub(slack.native_mtp)
        .ok_or_else(|| format!("{HERE}: native granule slack exceeds total slack"))?;
    let sum_checked = |sizes: &Vec<u64>| {
        sizes.iter().try_fold(0u64, |total, &bytes| {
            total
                .checked_add(bytes)
                .ok_or_else(|| format!("{HERE}: native allocation list overflows"))
        })
    };
    let (native_resident, native_build, native_request, native_extra) = if args.mtp_kept {
        let head = crate::expert_residency::language_head_dtype(args.weights).ok_or_else(|| {
            format!("{HERE}: native head kept but the manifest has no language head")
        })?;
        let drafts = crate::mtp_spec::native_mtp_proposal_drafts(args.max_k, &args.gpu.arch);
        let row_capture = crate::mtp_spec::native_mtp_capture(
            args.gpu,
            args.config,
            args.state_format.gdn,
            drafts,
        );
        let census_context = crate::Qwen4ContextCommit::new(
            args.backend,
            args.max_seq,
            args.max_seq,
            args.vmm_granularity,
        );
        let (resident, build_scratch) = crate::mtp_gpu::Qwen4MtpGpu::allocation_sizes(
            args.config,
            &census_context,
            head,
            false,
        )
        .ok_or_else(|| format!("{HERE}: native MTP head allocation sizes overflow"))?;
        let request = crate::mtp_spec::native_mtp_request_allocation_sizes(
            args.config,
            census_context.max_seq,
            args.chunk_rows,
            drafts,
            row_capture,
            crate::mtp_spec::native_mtp_fill_scratch(args.gpu),
        )
        .ok_or_else(|| format!("{HERE}: native MTP request allocation sizes overflow"))?;
        let extra = preload_native_union_extra(&resident, &build_scratch, &request, granule_bytes)?;
        let (resident_sum, build_sum, request_sum) = (
            sum_checked(&resident)?,
            sum_checked(&build_scratch)?,
            sum_checked(&request)?,
        );
        let logical_peak = resident_sum
            .checked_add(build_sum.max(request_sum))
            .ok_or_else(|| format!("{HERE}: native logical peak overflows"))?;
        if logical_peak != reserve.census.mtp {
            return Err(format!(
                "{HERE}: native logical peak {logical_peak} bytes differs from census mtp {} bytes",
                reserve.census.mtp
            ));
        }
        (resident_sum, build_sum, request_sum, extra)
    } else {
        (0, 0, 0, 0)
    };
    let native_backing_union = native_resident
        .checked_add(native_build.max(native_request))
        .and_then(|logical| logical.checked_add(native_extra))
        .ok_or_else(|| format!("{HERE}: native backing union overflows"))?;
    // Finite forward-scratch union over both admissible modes (Full vs
    // FnSymGfx1201): the conservative max, never the sum; 0 when the census
    // base mode dominates (shared state is inside both backings and is never
    // double-added). Metadata-only; flags untouched. Single-mode gate: the alt
    // inventory is unioned only when the compact predicate
    // (`qwen4_compact_scratch_possible`, Gpu + config authority) says the
    // source COULD take the compact plan; a predicate-false alt validly leaves
    // `extra = 0` + note (admissible set is the base mode alone). A
    // predicate-true alt whose inventory is missing/overflowing is a named Err
    // inside `preload_scratch_union_extra`, never a zero fallback.
    let compact_possible = crate::qwen4_compact_scratch_possible(args.gpu, args.config);
    let (alt_scratch_mode, scratch_base_backing, scratch_alt_backing, scratch_extra, scratch_note) =
        if !compact_possible {
            (
                match args.scratch_mode {
                    crate::gpu_forward::Qwen4ForwardScratchMode::Full => {
                        crate::gpu_forward::Qwen4ForwardScratchMode::FnSymGfx1201
                    }
                    crate::gpu_forward::Qwen4ForwardScratchMode::FnSymGfx1201 => {
                        crate::gpu_forward::Qwen4ForwardScratchMode::Full
                    }
                },
                0,
                0,
                0,
                Some("compact plan not admissible for this source (qwen4_compact_scratch_possible is false): admissible set is the base mode alone".to_string()),
            )
        } else {
            preload_scratch_union_extra(
                args.config,
                args.chunk_rows,
                args.scratch_mode,
                granule_bytes,
            )?
        };
    // Planned owners plus charges plus once-only slack/union extras plus the
    // preallocation pooled bytes. The runtime allowance is already inside the
    // reserve and is reported separately below, never added again.
    let charged = charges
        .reserve(reserve_bytes)
        .map_err(|err| format!("{HERE}: {err}"))?;
    let device_union_bytes = non_expert_device_bytes
        .checked_add(routed_device_bytes)
        .and_then(|sum| sum.checked_add(charged))
        .and_then(|sum| sum.checked_add(tables_bytes))
        .and_then(|sum| sum.checked_add(slack_without_native))
        .and_then(|sum| sum.checked_add(native_extra))
        .and_then(|sum| sum.checked_add(scratch_extra))
        .and_then(|sum| sum.checked_add(preallocation_pooled_device_bytes))
        .ok_or_else(|| format!("{HERE}: device union bytes overflow"))?;
    // Host scope stays the existing disjoint host-mapped plus explicit-host-only
    // check; on UMA-shared backing the root device union is device-backed host
    // memory and is charged once more here, exactly as the settled record does.
    let uma_shared = args.gpu.is_uma();
    let device_backed_host_bytes = if uma_shared { device_union_bytes } else { 0 };
    let host_union_bytes = host_mapped_bytes
        .checked_add(stage_mirror_bytes)
        .and_then(|sum| sum.checked_add(device_backed_host_bytes))
        .ok_or_else(|| format!("{HERE}: host union bytes overflow"))?;
    let workspace: Vec<serde_json::Value> = reserve
        .census
        .workspace
        .iter()
        .map(|(name, bytes)| serde_json::json!({"name": name, "bytes": bytes}))
        .collect();
    Ok(serde_json::json!({
        "evidence_kind": PRELOAD_UNION_EVIDENCE_KIND,
        "owner_contract_version": LOAD_RESOURCE_OWNER_CONTRACT_VERSION,
        "parked_policy": PRELOAD_UNION_PARKED_POLICY,
        "scratch_mode_label": scratch_mode_label,
        "s1_reason": s1_reason,
        "planned": {
            "non_expert_device_bytes": non_expert_device_bytes,
            "layer_expert_bytes": inputs.layer_experts,
            "routed_device_bytes": routed_device_bytes,
            "host_mapped_bytes": host_mapped_bytes,
            "tables_bytes": tables_bytes,
            "stage_mirror_bytes": stage_mirror_bytes,
            "stage_bytes": charges.stage,
            "margin_bytes": charges.margin,
            "slowlink_bytes": charges.slowlink,
            "reserve": {
                "historical_bytes_or_null": reserve.historical,
                "census": {
                    "qsa_context": reserve.census.qsa_context,
                    "state": reserve.census.state,
                    "forward": reserve.census.forward,
                    "f16_shadows": reserve.census.f16_shadows,
                    "shared_i8_shadows": reserve.census.shared_i8_shadows,
                    "workspace": workspace,
                    "qsa_selector": reserve.census.qsa_selector,
                    "mtp": reserve.census.mtp,
                    "gather": reserve.census.gather,
                    "runtime": reserve.census.runtime,
                },
                "census_total_bytes": reserve.census_total,
                "reserve_bytes": reserve_bytes,
                "winner": reserve.winner(),
            },
        },
        "native_union": {
            "mtp_kept": args.mtp_kept,
            "granule_bytes": granule_bytes,
            "resident_logical_bytes": native_resident,
            "build_scratch_logical_bytes": native_build,
            "request_logical_bytes": native_request,
            "census_logical_peak_bytes": native_resident + native_build.max(native_request),
            "backing_union_bytes": native_backing_union,
            "union_extra_added_bytes": native_extra,
            "supersedes_granule_slack_native_term": true,
        },
        "scratch_union": {
            "base_mode": format!("{:?}", args.scratch_mode),
            "alt_mode": format!("{alt_scratch_mode:?}"),
            "compact_possible": compact_possible,
            "base_backing_bytes": scratch_base_backing,
            "alt_backing_bytes": scratch_alt_backing,
            "union_extra_added_bytes": scratch_extra,
            "admissible_note_or_null": scratch_note,
        },
        "granule_slack": {
            "terms": slack.terms().iter().map(|(name, bytes)| serde_json::json!({"name": name, "bytes": bytes})).collect::<Vec<_>>(),
            "total_bytes": slack_total,
            "native_term_superseded_bytes": slack.native_mtp,
            "charged_bytes": slack_without_native,
        },
        "runtime_allowance_bytes": reserve.census.runtime,
        "preallocation": {
            "pooled_device_bytes": preallocation_pooled_device_bytes,
            "free_device_bytes_or_null": preallocation_free_device_bytes,
        },
        "host_capacity": {
            "mem_available_bytes_or_null": mem_available_bytes,
            "ttm_pool_bytes": ttm_pool_bytes,
            "gtt_free_bytes_or_null": gtt_free_bytes,
        },
        "uma_shared_backing": uma_shared,
        "device_backed_host_bytes": device_backed_host_bytes,
        "device_union_upper_bound_bytes": device_union_bytes,
        "host_union_upper_bound_bytes": host_union_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    const WORKSPACE: [(&str, u64); crate::expert_residency::WORKSPACE_TERM_COUNT] = [
        ("synthetic-ws-0", 11),
        ("synthetic-ws-1", 22),
        ("synthetic-ws-2", 33),
        ("synthetic-ws-3", 44),
        ("synthetic-ws-4", 55),
        ("synthetic-ws-5", 66),
        ("synthetic-ws-6", 77),
        ("synthetic-ws-7", 88),
        ("synthetic-ws-8", 99),
        ("synthetic-ws-9", 110),
        ("synthetic-ws-10", 121),
    ];

    /// Clearly synthetic valid descriptor: overlay applied, hotset none.
    fn synthetic_identity() -> Arc<Qwen4MeasuredLoadIdentity> {
        let fields = BTreeMap::from([
            ("schema".to_owned(), "qwen4-measured-load.v1".to_owned()),
            ("model.sha256".to_owned(), "ab".repeat(32)),
            ("model.bytes".to_owned(), "123456".to_owned()),
            ("ple.sha256".to_owned(), "cd".repeat(32)),
            ("ple.bytes".to_owned(), "7890".to_owned()),
            ("ple.encoding".to_owned(), "q8-synthetic".to_owned()),
            ("overlay.sha256".to_owned(), "ef".repeat(32)),
            ("overlay.bytes".to_owned(), "456".to_owned()),
            ("hotset".to_owned(), "none".to_owned()),
            (
                "placement.digest".to_owned(),
                "synthetic-placement-digest-0".to_owned(),
            ),
            (
                "recipe".to_owned(),
                "synthetic-recipe-trunk-fp8-mtp-test-only".to_owned(),
            ),
            (
                "state".to_owned(),
                "SyntheticTestState(TestOnly)".to_owned(),
            ),
            ("kv_backend".to_owned(), "synthetic-test-backend".to_owned()),
            ("max_seq".to_owned(), "2048".to_owned()),
            ("capture_rows".to_owned(), "64".to_owned()),
            ("gpu.arch".to_owned(), "gfx1201-synthetic".to_owned()),
            ("gpu.uuid".to_owned(), "SYNTHETIC-GPU-UUID-0".to_owned()),
            (
                "scope".to_owned(),
                "synthetic-scope/chunk0/state/test-only".to_owned(),
            ),
            (
                "flags".to_owned(),
                "SyntheticFlags(test_only:true)".to_owned(),
            ),
        ]);
        Arc::new(
            Qwen4MeasuredLoadIdentity::from_fields(fields).expect("synthetic valid descriptor"),
        )
    }

    fn synthetic_census(mtp: u64) -> crate::expert_residency::ReserveCensus {
        crate::expert_residency::ReserveCensus {
            qsa_context: 1_000,
            state: 2_000,
            forward: 3_000,
            f16_shadows: 400,
            shared_i8_shadows: 0,
            workspace: WORKSPACE,
            qsa_selector: 500,
            mtp,
            gather: 600,
            runtime: 700,
        }
    }

    fn synthetic_admitted(mode: Qwen4RingMode, old_capture_bytes: u64) -> Qwen4AdmittedShape {
        Qwen4AdmittedShape {
            max_seq: 2048,
            chunk_rows: 256,
            capture_rows: 64,
            ranks: 1,
            gdn_state: "q8".to_owned(),
            kv_backend: "synthetic-test-backend".to_owned(),
            scratch_mode: "synthetic-test-scratch".to_owned(),
            geometry: Qwen4CaptureGeometry {
                linear_num_key_heads: 8,
                linear_num_value_heads: 8,
                linear_key_head_dim: 128,
                linear_value_head_dim: 128,
                owned_gdn_layers: 36,
            },
            mode,
            old_capture_bytes,
            new_capture_bytes: 900,
        }
    }

    fn synthetic_placement() -> Qwen4PlacementBytes {
        Qwen4PlacementBytes {
            non_expert_device_bytes: 5 << 30,
            routed_device_bytes: 40 << 30,
            host_mapped_bytes: 20 << 30,
            tables_bytes: 1 << 20,
            stage_bytes: 2 << 20,
            margin_bytes: 512 << 20,
        }
    }

    fn synthetic_host() -> Qwen4HostCheckInputs {
        Qwen4HostCheckInputs {
            host_check_bytes: 20 << 30,
            stage_mirror_bytes: 1 << 30,
            host_headroom_bytes: crate::expert_residency::HOST_RAM_HEADROOM_BYTES,
            ttm_pool_bytes: 0,
            mem_available_bytes: Some(64 << 30),
            gtt_budget_bytes: None,
        }
    }

    fn synthetic_backing() -> Qwen4LoadBacking {
        Qwen4LoadBacking {
            kind: Qwen4BackingKind::Discrete,
            preallocation_pooled_device_bytes: 3 << 20,
            settled_pooled_device_bytes: 5 << 20,
        }
    }

    fn synthetic_snapshot(
        mode: Qwen4RingMode,
        old_capture_bytes: u64,
    ) -> Qwen4LoadResourceSnapshot {
        synthetic_snapshot_backed(mode, old_capture_bytes, synthetic_backing())
    }

    fn synthetic_snapshot_backed(
        mode: Qwen4RingMode,
        old_capture_bytes: u64,
        backing: Qwen4LoadBacking,
    ) -> Qwen4LoadResourceSnapshot {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(mode, old_capture_bytes),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            Some(8 << 30),
            backing,
        )
        .expect("synthetic valid snapshot")
    }

    /// Re-adds the OLD settled-owner formula exactly (no parked terms).
    fn base_device_plan(reserve_bytes: u64, old_capture_bytes: u64) -> u64 {
        let placement = synthetic_placement();
        let mut base = placement
            .non_expert_device_bytes
            .checked_add(placement.routed_device_bytes)
            .and_then(|sum| sum.checked_add(reserve_bytes))
            .and_then(|sum| sum.checked_add(placement.tables_bytes))
            .and_then(|sum| sum.checked_add(placement.stage_bytes))
            .and_then(|sum| sum.checked_add(placement.margin_bytes))
            .expect("synthetic base fits");
        if old_capture_bytes != 0 {
            base = base
                .checked_add(old_capture_bytes)
                .expect("synthetic base fits");
        }
        base
    }

    #[test]
    fn positive_direct_snapshot_freezes_schema_and_plans() {
        let backing = synthetic_backing();
        let snapshot = synthetic_snapshot(Qwen4RingMode::DirectInitial64, 0);
        let value = snapshot.value();
        assert_eq!(value["schema"], LOAD_RESOURCE_SNAPSHOT_SCHEMA);
        assert_eq!(value["evidence_kind"], LOAD_RESOURCE_EVIDENCE_KIND);
        // No fit claim anywhere in the frozen output: a new data block is not
        // an actual-fit qualification.
        assert!(value.get("fits").is_none());
        assert!(value.get("fit").is_none());
        assert!(value["upper_bound"].get("fits").is_none());
        let census = synthetic_census(4_000);
        let total = census.total().expect("synthetic total");
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        let base = base_device_plan(reserve.bytes(), 0);
        assert_eq!(
            value["upper_bound"]["base_device_plan_bytes"],
            serde_json::json!(base)
        );
        let backing_parked =
            backing.preallocation_pooled_device_bytes + backing.settled_pooled_device_bytes;
        let expect_device = base + backing_parked;
        assert_eq!(value["device_plan_bytes"], serde_json::json!(expect_device));
        assert_eq!(snapshot.device_plan_bytes(), expect_device);
        // Fresh-rank-load lifecycle is the only one this record supports.
        assert_eq!(value["upper_bound"]["model_lifecycle"], "fresh_rank_load");
        let host = synthetic_host();
        // Discrete backing adds no device-backed host bytes.
        let expect_host =
            host.host_check_bytes + host.stage_mirror_bytes + host.host_headroom_bytes;
        assert_eq!(value["host_plan_bytes"], serde_json::json!(expect_host));
        assert_eq!(snapshot.host_plan_bytes(), expect_host);
        // REQUIRED exact upper-bound block: every field, every literal.
        assert_eq!(
            value["upper_bound"]["evidence_kind"],
            LOAD_RESOURCE_UPPER_BOUND_EVIDENCE_KIND
        );
        assert_eq!(
            value["upper_bound"]["model_lifecycle"],
            LOAD_RESOURCE_MODEL_LIFECYCLE_FRESH_RANK_LOAD
        );
        assert_eq!(value["upper_bound"]["backing_kind"], "discrete");
        assert_eq!(
            value["upper_bound"]["preallocation_pooled_device_bytes"],
            serde_json::json!(backing.preallocation_pooled_device_bytes)
        );
        assert_eq!(
            value["upper_bound"]["settled_pooled_device_bytes"],
            serde_json::json!(backing.settled_pooled_device_bytes)
        );
        assert_eq!(
            value["upper_bound"]["parked_overlap_policy"],
            LOAD_RESOURCE_PARKED_OVERLAP_POLICY
        );
        assert_eq!(
            value["upper_bound"]["existing_host_check_scope"],
            LOAD_RESOURCE_HOST_CHECK_SCOPE
        );
        assert_eq!(
            value["upper_bound"]["owner_contract_version"],
            LOAD_RESOURCE_OWNER_CONTRACT_VERSION
        );
        assert_eq!(
            value["upper_bound"]["device_backed_host_bytes"],
            serde_json::json!(0u64)
        );
        // Reserve carries the historical floor, the verbatim census and the winner.
        assert_eq!(
            value["reserve"]["historical_bytes_or_null"],
            serde_json::json!(6u64 << 30)
        );
        assert_eq!(
            value["reserve"]["census_total_bytes"],
            serde_json::json!(total)
        );
        assert_eq!(
            value["reserve"]["reserve_bytes"],
            serde_json::json!(reserve.bytes())
        );
        assert_eq!(
            value["reserve"]["winner"],
            serde_json::json!(reserve.winner())
        );
        assert_eq!(
            value["reserve"]["census"]["workspace"]
                .as_array()
                .expect("workspace array")
                .len(),
            11
        );
        assert_eq!(
            value["reserve"]["census"]["workspace"][0]["bytes"],
            serde_json::json!(11u64)
        );
    }

    #[test]
    fn positive_retained_snapshot_charges_old_capture() {
        let backing = synthetic_backing();
        let snapshot = synthetic_snapshot(Qwen4RingMode::Retained8To64, 1 << 20);
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        let expect = base_device_plan(reserve.bytes(), 1 << 20)
            + backing.preallocation_pooled_device_bytes
            + backing.settled_pooled_device_bytes;
        assert_eq!(snapshot.device_plan_bytes(), expect);
        assert_eq!(snapshot.value()["device_plan_bytes"], expect);
        assert_eq!(
            snapshot.value()["upper_bound"]["base_device_plan_bytes"],
            base_device_plan(reserve.bytes(), 1 << 20)
        );
        assert_eq!(snapshot.value()["admitted"]["mode"], "retained_8_to_64");
    }

    #[test]
    fn new_capture_is_never_added_to_device_plan_twice() {
        let backing = synthetic_backing();
        let snapshot = synthetic_snapshot(Qwen4RingMode::DirectInitial64, 0);
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        // Device plan excludes the new capture even though the MTP census
        // term contains it; only the parked readings extend the base.
        let without_new = base_device_plan(reserve.bytes(), 0)
            + backing.preallocation_pooled_device_bytes
            + backing.settled_pooled_device_bytes;
        assert_eq!(snapshot.device_plan_bytes(), without_new);
        assert!(reserve.census.mtp >= 900);
    }

    #[test]
    fn provenance_reference_matches_identity_canonical() {
        let identity = synthetic_identity();
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(None, census).expect("synthetic reserve");
        let snapshot = Qwen4LoadResourceSnapshot::new(
            Arc::clone(&identity),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect("synthetic valid snapshot");
        let parsed: serde_json::Value =
            serde_json::from_str(identity.canonical_json()).expect("identity canonical parses");
        assert_eq!(snapshot.value()["measured_load"], parsed);
        assert!(Arc::ptr_eq(snapshot.identity(), &identity));
        assert_eq!(
            snapshot.value()["preallocation_free_device_bytes"],
            serde_json::Value::Null
        );
        assert_eq!(
            snapshot.value()["host_check"]["gtt_budget_bytes"],
            serde_json::Value::Null
        );
        assert_eq!(
            snapshot.value()["reserve"]["historical_bytes_or_null"],
            serde_json::Value::Null
        );
        assert_eq!(
            snapshot.value()["reserve"]["winner"],
            "census (historical floor retired)"
        );
    }

    #[test]
    fn rejects_non_row_exact_capture_rows() {
        let mut admitted = synthetic_admitted(Qwen4RingMode::DirectInitial64, 0);
        admitted.capture_rows = 8;
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            admitted,
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("8 rows are not the row-exact 64");
        assert!(err.contains("64"), "{err}");
    }

    #[test]
    fn rejects_non_single_rank_and_empty_context() {
        let census = synthetic_census(4_000);
        for ranks in [0, 2] {
            let mut admitted = synthetic_admitted(Qwen4RingMode::DirectInitial64, 0);
            admitted.ranks = ranks;
            let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
            let err = Qwen4LoadResourceSnapshot::new(
                synthetic_identity(),
                admitted,
                reserve,
                synthetic_placement(),
                synthetic_host(),
                None,
                synthetic_backing(),
            )
            .expect_err("ranks must be 1");
            assert!(err.contains("ranks must be 1"), "{err}");
        }
        for (max_seq, chunk_rows) in [(0, 256), (2048, 0)] {
            let mut admitted = synthetic_admitted(Qwen4RingMode::DirectInitial64, 0);
            admitted.max_seq = max_seq;
            admitted.chunk_rows = chunk_rows;
            let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
            assert!(Qwen4LoadResourceSnapshot::new(
                synthetic_identity(),
                admitted,
                reserve,
                synthetic_placement(),
                synthetic_host(),
                None,
                synthetic_backing(),
            )
            .is_err());
        }
        let mut admitted = synthetic_admitted(Qwen4RingMode::DirectInitial64, 0);
        admitted.geometry.linear_key_head_dim = 0;
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            admitted,
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("zero head dim is not positive geometry");
        assert!(err.contains("geometry"), "{err}");
        let mut admitted = synthetic_admitted(Qwen4RingMode::DirectInitial64, 0);
        admitted.gdn_state = "fp16".to_owned();
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            admitted,
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("unknown GDN state");
        assert!(err.contains("gdn_state"), "{err}");
    }

    #[test]
    fn rejects_direct_with_old_bytes_and_retained_without() {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 1 << 20),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("direct charges no old capture");
        assert!(err.contains("no old capture"), "{err}");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::Retained8To64, 0),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("retained keeps the old storage charged");
        assert!(err.contains("old 8-row storage"), "{err}");
    }

    #[test]
    fn rejects_census_total_mismatch() {
        let census = synthetic_census(4_000);
        let mut reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        reserve.census_total += 1;
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("bad census total");
        assert!(err.contains("census total mismatch"), "{err}");
    }

    #[test]
    fn rejects_mtp_shortfall_with_exact_mismatch() {
        // New capture of 900 bytes cannot sit in an 899-byte MTP census term.
        let census = synthetic_census(899);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("MTP census cannot contain the new capture");
        assert!(err.contains("899"), "{err}");
        assert!(err.contains("900"), "{err}");
        assert!(err.contains("not relaxed"), "{err}");
    }

    #[test]
    fn rejects_device_plan_overflow() {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let mut placement = synthetic_placement();
        placement.non_expert_device_bytes = u64::MAX;
        placement.routed_device_bytes = 1;
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            placement,
            synthetic_host(),
            None,
            synthetic_backing(),
        )
        .expect_err("device plan overflows");
        assert!(err.contains("overflow"), "{err}");
        let mut host = synthetic_host();
        host.host_check_bytes = u64::MAX;
        host.stage_mirror_bytes = 1;
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            host,
            None,
            synthetic_backing(),
        )
        .expect_err("host plan overflows");
        assert!(err.contains("overflow"), "{err}");
    }

    #[test]
    fn positive_uma_snapshot_adds_device_plan_to_host_once() {
        let backing = Qwen4LoadBacking {
            kind: Qwen4BackingKind::UmaShared,
            preallocation_pooled_device_bytes: 3 << 20,
            settled_pooled_device_bytes: 5 << 20,
        };
        let snapshot = synthetic_snapshot_backed(Qwen4RingMode::DirectInitial64, 0, backing);
        let value = snapshot.value();
        assert_eq!(value["upper_bound"]["backing_kind"], "uma_shared");
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        let base = base_device_plan(reserve.bytes(), 0);
        let expect_device = base + (3 << 20) + (5 << 20);
        assert_eq!(snapshot.device_plan_bytes(), expect_device);
        assert_eq!(
            value["upper_bound"]["device_backed_host_bytes"],
            serde_json::json!(expect_device)
        );
        // Backing-independent device side: the UMA root equals the discrete root.
        let discrete = synthetic_snapshot(Qwen4RingMode::DirectInitial64, 0);
        assert_eq!(discrete.device_plan_bytes(), expect_device);
        let host = synthetic_host();
        let old_host = host.host_check_bytes + host.stage_mirror_bytes + host.host_headroom_bytes;
        assert_eq!(snapshot.host_plan_bytes(), old_host + expect_device);
        assert_eq!(
            value["host_plan_bytes"],
            serde_json::json!(old_host + expect_device)
        );
    }

    #[test]
    fn nonzero_parked_union_always_exceeds_base() {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        let base = base_device_plan(reserve.bytes(), 0);
        let snapshot = synthetic_snapshot(Qwen4RingMode::DirectInitial64, 0);
        // Synthetic backing parks 3 MiB + 5 MiB: the union strictly exceeds base.
        assert_eq!(snapshot.device_plan_bytes(), base + (8 << 20));
        assert!(snapshot.device_plan_bytes() > base);
        assert_eq!(
            snapshot.value()["upper_bound"]["base_device_plan_bytes"],
            serde_json::json!(base)
        );
    }

    #[test]
    fn alias_contract_tokens_and_overlap_policy_are_exact() {
        let backing = synthetic_backing();
        assert_eq!(
            serde_json::to_value(backing.kind).expect("kind serializes"),
            "discrete"
        );
        assert_eq!(
            serde_json::to_value(Qwen4BackingKind::UmaShared).expect("kind serializes"),
            "uma_shared"
        );
        let snapshot = synthetic_snapshot(Qwen4RingMode::DirectInitial64, 0);
        let bound = &snapshot.value()["upper_bound"];
        // The known alias limitation is labelled, never subtracted away.
        assert_eq!(
            bound["parked_overlap_policy"],
            "charge_both_unknown_intersection"
        );
        assert_eq!(
            bound["existing_host_check_scope"],
            "host_mapped_and_explicit_host_only"
        );
        assert_eq!(bound["owner_contract_version"], "qwen4-owned-components-v1");
        // Both parked readings are charged even when equal (unknown intersection).
        let aliased = Qwen4LoadBacking {
            kind: Qwen4BackingKind::Discrete,
            preallocation_pooled_device_bytes: 5 << 20,
            settled_pooled_device_bytes: 5 << 20,
        };
        let aliased_snapshot =
            synthetic_snapshot_backed(Qwen4RingMode::DirectInitial64, 0, aliased);
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(6 << 30), census).expect("synthetic reserve");
        assert_eq!(
            aliased_snapshot.device_plan_bytes(),
            base_device_plan(reserve.bytes(), 0) + (10 << 20)
        );
    }

    #[test]
    fn rejects_parked_pool_overflow() {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let backing = Qwen4LoadBacking {
            kind: Qwen4BackingKind::Discrete,
            preallocation_pooled_device_bytes: u64::MAX,
            settled_pooled_device_bytes: 1,
        };
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            synthetic_host(),
            None,
            backing,
        )
        .expect_err("parked pools overflow");
        assert!(err.contains("overflow"), "{err}");
    }

    #[test]
    fn rejects_uma_host_addition_overflow() {
        let census = synthetic_census(4_000);
        let reserve = AutoReserve::new(Some(1), census).expect("synthetic reserve");
        let mut host = synthetic_host();
        host.host_check_bytes = u64::MAX;
        host.stage_mirror_bytes = 0;
        host.host_headroom_bytes = 0;
        // A maxed host check overflows the shared checked chain even with zero
        // parked bytes; UMA would only add more on top.
        let err = Qwen4LoadResourceSnapshot::new(
            synthetic_identity(),
            synthetic_admitted(Qwen4RingMode::DirectInitial64, 0),
            reserve,
            synthetic_placement(),
            host,
            None,
            Qwen4LoadBacking {
                kind: Qwen4BackingKind::UmaShared,
                preallocation_pooled_device_bytes: 0,
                settled_pooled_device_bytes: 0,
            },
        )
        .expect_err("UMA host addition overflows");
        assert!(err.contains("overflow"), "{err}");
    }
    // ---- preload_component_union (A): conservative finite union helpers ----

    #[test]
    fn preload_native_union_extra_charges_both_phases_over_logical_peak() {
        // Resident 1000 + build 400 + request 700: logical peak 1700, backing
        // union at granule 256 is 1024+512+768 = 2304, so extra = 604, added
        // once. Both phases are charged, not the larger alone (the max-phase
        // slack would be only 24+112 = 136 on top of resident slack 24).
        assert_eq!(
            preload_native_union_extra(&[1000], &[400], &[700], 256),
            Ok(604)
        );
    }

    #[test]
    fn preload_native_union_extra_rejects_overflow_and_short_backing() {
        // A backing union is never below its payload: zero granule keeps every
        // backing equal to its payload, so the union extra is exactly the kept
        // smaller phase (both finite phases, no duplicate cold charge).
        // Overflow of any list sum refuses with a named error.
        assert!(preload_native_union_extra(&[u64::MAX], &[1], &[0], 0).is_err());
        assert!(preload_native_union_extra(&[0], &[u64::MAX], &[1], 0).is_err());
        assert!(preload_native_union_extra(&[0], &[0], &[u64::MAX, 1], 0).is_err());
    }

    #[test]
    fn preload_native_union_extra_agrees_with_native_mtp_granule_slack_on_max() {
        // Backing(resident)+backing(build)+backing(request) - logical peak
        // equals native_mtp_granule_slack + min(backing of the two phases):
        // the finite union keeps BOTH phases; the phase slack keeps the max.
        let (resident, build, request) = (vec![1000u64], vec![400u64], vec![700u64]);
        let extra = preload_native_union_extra(&resident, &build, &request, 256).unwrap();
        let phase =
            crate::expert_residency::native_mtp_granule_slack(&resident, &build, &request, 256)
                .unwrap();
        let backing = |sizes: &Vec<u64>| {
            sizes.iter().sum::<u64>()
                + crate::expert_residency::granule_slack_total(sizes.iter().copied(), 256).unwrap()
        };
        assert_eq!(extra, phase + backing(&build).min(backing(&request)));
        assert!(extra >= phase);
    }

    #[test]
    fn preload_native_union_extra_has_no_duplicate_cold_recurrent_charge() {
        // A retained ring built twice (old + new) still charges the finite
        // union extra only once at preload: the extra is a pure function of
        // the allocation lists, so identical lists give identical extras and
        // the caller cannot add a second copy without a second list.
        let (resident, build, request) = (vec![2048u64], vec![512u64], vec![4096u64]);
        let once = preload_native_union_extra(&resident, &build, &request, 1 << 21).unwrap();
        let twice = preload_native_union_extra(&resident, &build, &request, 1 << 21).unwrap();
        assert_eq!(once, twice);
        // Zero granule (no rounding) still separates identical phases: build
        // and request are both kept even when equal.
        let equal = preload_native_union_extra(&[100u64], &[50u64], &[50u64], 0).unwrap();
        assert_eq!(equal, 50);
    }
    #[test]
    fn preload_scratch_union_extra_covers_both_modes_without_summing() {
        let config = crate::config::compact_test_config();
        for mode in [
            crate::gpu_forward::Qwen4ForwardScratchMode::Full,
            crate::gpu_forward::Qwen4ForwardScratchMode::FnSymGfx1201,
        ] {
            let (alt, base, alt_backing, extra, note) =
                preload_scratch_union_extra(&config, 1536, mode, 1 << 21).unwrap();
            assert_ne!(alt, mode);
            // Conservative max: base + extra equals the larger backing, never
            // the sum of both modes.
            assert_eq!(base + extra, base.max(alt_backing));
            assert!(note.is_none());
            // Reversed base agrees: the same larger backing wins either way.
            let (back_alt, back_base, back_alt_backing, back_extra, _) =
                preload_scratch_union_extra(&config, 1536, alt, 1 << 21).unwrap();
            assert_eq!(back_alt, mode);
            assert_eq!(back_base, alt_backing);
            assert_eq!(back_alt_backing, base);
            assert_eq!(back_extra, base.saturating_sub(alt_backing));
        }
    }

    #[test]
    fn preload_scratch_union_extra_refuses_overflow_like_the_inventory() {
        let mut config = crate::config::compact_test_config();
        config.vocab_size = usize::MAX;
        assert!(preload_scratch_union_extra(
            &config,
            64,
            crate::gpu_forward::Qwen4ForwardScratchMode::Full,
            1 << 21
        )
        .is_err());
    }
}
